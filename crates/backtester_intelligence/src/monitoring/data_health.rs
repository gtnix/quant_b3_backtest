//! Data Health Checks for Monitoring.
//!
//! Implements 8 core checks:
//! 1. Freshness - last OHLCV within X business days
//! 2. Coverage - % symbols with sufficient data
//! 3. Watermark - no regression in dates
//! 4. Nulls - critical fields not null
//! 5. Outliers - values within N sigma
//! 6. Schema - types and constraints valid
//! 7. Dividends - coverage and recency
//! 8. Interest Rates - recency by region

use chrono::{DateTime, NaiveDate, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;

use crate::filters::Market;
use super::config::{DataHealthConfig, ThresholdEvaluator};
use super::types::{CheckCategory, CheckResult, Evidence, Severity};

/// Context for data health checks.
#[derive(Debug, Clone, Default)]
pub struct DataContext {
    /// Last OHLCV date per market
    pub last_ohlcv_date: HashMap<Market, NaiveDate>,
    /// Symbol count per market
    pub symbol_count: HashMap<Market, usize>,
    /// Symbols with sufficient data per market
    pub symbols_with_data: HashMap<Market, usize>,
    /// Previous watermark dates
    pub previous_watermarks: HashMap<Market, NaiveDate>,
    /// Current watermark dates
    pub current_watermarks: HashMap<Market, NaiveDate>,
    /// Null counts by field
    pub null_counts: HashMap<String, usize>,
    /// Total rows checked for nulls
    pub total_rows: usize,
    /// Outlier counts by field
    pub outlier_counts: HashMap<String, usize>,
    /// Dividend count in last 30 days
    pub dividends_30d: u32,
    /// Dividend types found
    pub dividend_types: Vec<String>,
    /// Last interest rate date per region
    pub last_interest_rate: HashMap<Market, NaiveDate>,
    /// Interest rate count per region
    pub interest_rate_count: HashMap<Market, usize>,
    /// Schema validation passed
    pub schema_valid: bool,
    /// Schema errors if any
    pub schema_errors: Vec<String>,
    /// Reference date for checks
    pub as_of: NaiveDate,
}

impl DataContext {
    pub fn new(as_of: NaiveDate) -> Self {
        Self {
            as_of,
            ..Default::default()
        }
    }

    /// Calculate coverage percentage for a market.
    pub fn coverage_pct(&self, market: Market) -> Decimal {
        let total = *self.symbol_count.get(&market).unwrap_or(&0);
        let with_data = *self.symbols_with_data.get(&market).unwrap_or(&0);
        
        if total == 0 {
            return dec!(0);
        }
        
        Decimal::from(with_data * 100) / Decimal::from(total)
    }

    /// Get days since last OHLCV for a market.
    pub fn days_since_ohlcv(&self, market: Market) -> Option<i64> {
        self.last_ohlcv_date.get(&market).map(|d| {
            (self.as_of - *d).num_days()
        })
    }
}

/// Trait for data health checks.
pub trait DataHealthCheck: Send + Sync {
    /// Check name for logging.
    fn name(&self) -> &str;
    /// Run the check and return result.
    fn run(&self, ctx: &DataContext, config: &DataHealthConfig) -> CheckResult;
}

/// Freshness check - last OHLCV within threshold.
#[derive(Debug, Clone, Default)]
pub struct FreshnessCheck {
    pub market: Market,
}

impl FreshnessCheck {
    pub fn new(market: Market) -> Self {
        Self { market }
    }
}

impl DataHealthCheck for FreshnessCheck {
    fn name(&self) -> &str {
        "Freshness"
    }

    fn run(&self, ctx: &DataContext, config: &DataHealthConfig) -> CheckResult {
        let days = ctx.days_since_ohlcv(self.market).unwrap_or(999);
        let warn_threshold = config.freshness_warn(self.market);
        let crit_threshold = config.freshness_crit(self.market);

        let severity = ThresholdEvaluator::freshness_severity(
            days as u32, warn_threshold, crit_threshold
        );

        let passed = severity == Severity::Info;
        let msg = format!(
            "Last OHLCV for {:?}: {} days ago (WARN: {}, CRIT: {})",
            self.market, days, warn_threshold, crit_threshold
        );

        let evidence = Evidence::new("ohlcv.max_date")
            .with_current(Decimal::from(days))
            .with_context(format!("Market: {:?}", self.market));

        let mut result = if passed {
            CheckResult::pass(format!("Freshness_{:?}", self.market), CheckCategory::DataHealth)
        } else if severity == Severity::Crit {
            CheckResult::crit(format!("Freshness_{:?}", self.market), CheckCategory::DataHealth, &msg)
        } else {
            CheckResult::warn(format!("Freshness_{:?}", self.market), CheckCategory::DataHealth, &msg)
        };

        result.value = Decimal::from(days);
        result.threshold = Decimal::from(crit_threshold);
        result.message = msg;
        result.evidence = evidence;
        result.market = Some(self.market);
        result
    }
}

/// Coverage check - % symbols with sufficient data.
#[derive(Debug, Clone, Default)]
pub struct CoverageCheck {
    pub market: Market,
}

impl CoverageCheck {
    pub fn new(market: Market) -> Self {
        Self { market }
    }
}

impl DataHealthCheck for CoverageCheck {
    fn name(&self) -> &str {
        "Coverage"
    }

    fn run(&self, ctx: &DataContext, config: &DataHealthConfig) -> CheckResult {
        let coverage = ctx.coverage_pct(self.market);
        let severity = ThresholdEvaluator::coverage_severity(
            coverage, config.coverage_warn_pct, config.coverage_min_pct
        );

        let passed = severity == Severity::Info;
        let total = *ctx.symbol_count.get(&self.market).unwrap_or(&0);
        let with_data = *ctx.symbols_with_data.get(&self.market).unwrap_or(&0);

        let msg = format!(
            "Coverage {:?}: {:.1}% ({}/{}) - WARN: {}%, CRIT: {}%",
            self.market, coverage, with_data, total,
            config.coverage_warn_pct, config.coverage_min_pct
        );

        let evidence = Evidence::new("symbols.coverage")
            .with_current(coverage)
            .with_sample(vec![
                format!("total_symbols: {}", total),
                format!("with_data: {}", with_data),
            ]);

        let mut result = match severity {
            Severity::Info => CheckResult::pass(format!("Coverage_{:?}", self.market), CheckCategory::DataHealth),
            Severity::Crit => CheckResult::crit(format!("Coverage_{:?}", self.market), CheckCategory::DataHealth, &msg),
            _ => CheckResult::warn(format!("Coverage_{:?}", self.market), CheckCategory::DataHealth, &msg),
        };

        result.value = coverage;
        result.threshold = config.coverage_min_pct;
        result.message = msg;
        result.evidence = evidence;
        result.market = Some(self.market);
        result
    }
}

/// Watermark check - no regression in dates.
#[derive(Debug, Clone, Default)]
pub struct WatermarkCheck {
    pub market: Market,
}

impl WatermarkCheck {
    pub fn new(market: Market) -> Self {
        Self { market }
    }
}

impl DataHealthCheck for WatermarkCheck {
    fn name(&self) -> &str {
        "Watermark"
    }

    fn run(&self, ctx: &DataContext, config: &DataHealthConfig) -> CheckResult {
        if !config.check_watermark_regression {
            return CheckResult::pass(format!("Watermark_{:?}", self.market), CheckCategory::DataHealth);
        }

        let prev = ctx.previous_watermarks.get(&self.market);
        let curr = ctx.current_watermarks.get(&self.market);

        match (prev, curr) {
            (Some(p), Some(c)) if c < p => {
                let msg = format!(
                    "Watermark regression {:?}: {} -> {} (went back {} days)",
                    self.market, p, c, (*p - *c).num_days()
                );
                CheckResult::crit(format!("Watermark_{:?}", self.market), CheckCategory::DataHealth, &msg)
                    .with_market(self.market)
                    .with_evidence(Evidence::new("watermark.check")
                        .with_sample(vec![
                            format!("previous: {}", p),
                            format!("current: {}", c),
                        ]))
            }
            _ => CheckResult::pass(format!("Watermark_{:?}", self.market), CheckCategory::DataHealth)
                .with_market(self.market)
        }
    }
}

/// Null check - critical fields not null.
#[derive(Debug, Clone, Default)]
pub struct NullCheck;

impl DataHealthCheck for NullCheck {
    fn name(&self) -> &str {
        "Nulls"
    }

    fn run(&self, ctx: &DataContext, config: &DataHealthConfig) -> CheckResult {
        if !config.check_nulls {
            return CheckResult::pass("Nulls", CheckCategory::DataHealth);
        }

        let critical_fields = ["close", "volume", "shares_outstanding"];
        let mut critical_nulls = 0;
        let mut null_details = Vec::new();

        for (field, count) in &ctx.null_counts {
            if critical_fields.contains(&field.as_str()) {
                critical_nulls += count;
            }
            if *count > 0 {
                null_details.push(format!("{}: {}", field, count));
            }
        }

        if critical_nulls > 0 {
            let msg = format!("Critical nulls found: {}", critical_nulls);
            CheckResult::crit("Nulls", CheckCategory::DataHealth, &msg)
                .with_value(Decimal::from(critical_nulls))
                .with_evidence(Evidence::new("null_check")
                    .with_sample(null_details))
        } else if !null_details.is_empty() {
            let msg = format!("Non-critical nulls: {:?}", null_details);
            CheckResult::warn("Nulls", CheckCategory::DataHealth, &msg)
                .with_evidence(Evidence::new("null_check")
                    .with_sample(null_details))
        } else {
            CheckResult::pass("Nulls", CheckCategory::DataHealth)
        }
    }
}

/// Outlier check - values within N sigma.
#[derive(Debug, Clone, Default)]
pub struct OutlierCheck;

impl DataHealthCheck for OutlierCheck {
    fn name(&self) -> &str {
        "Outliers"
    }

    fn run(&self, ctx: &DataContext, config: &DataHealthConfig) -> CheckResult {
        let total_outliers: usize = ctx.outlier_counts.values().sum();
        let outlier_rate = if ctx.total_rows > 0 {
            Decimal::from(total_outliers * 100) / Decimal::from(ctx.total_rows)
        } else {
            dec!(0)
        };

        // Check for price <= 0 (critical)
        let price_outliers = *ctx.outlier_counts.get("price_zero_negative").unwrap_or(&0);
        
        let details: Vec<String> = ctx.outlier_counts.iter()
            .filter(|(_, &c)| c > 0)
            .map(|(f, c)| format!("{}: {}", f, c))
            .collect();

        if price_outliers > 0 {
            let msg = format!("Critical: {} records with price <= 0", price_outliers);
            CheckResult::crit("Outliers", CheckCategory::DataHealth, &msg)
                .with_value(Decimal::from(price_outliers))
                .with_threshold(dec!(0))
                .with_evidence(Evidence::new("outlier_check")
                    .with_sample(details))
        } else if outlier_rate > dec!(1) {
            let msg = format!("Outlier rate: {:.2}% ({} outliers)", outlier_rate, total_outliers);
            CheckResult::warn("Outliers", CheckCategory::DataHealth, &msg)
                .with_value(outlier_rate)
                .with_threshold(dec!(1))
                .with_evidence(Evidence::new("outlier_check")
                    .with_sample(details))
        } else {
            CheckResult::pass("Outliers", CheckCategory::DataHealth)
                .with_value(outlier_rate)
        }
    }
}

/// Schema check - types and constraints valid.
#[derive(Debug, Clone, Default)]
pub struct SchemaCheck;

impl DataHealthCheck for SchemaCheck {
    fn name(&self) -> &str {
        "Schema"
    }

    fn run(&self, ctx: &DataContext, config: &DataHealthConfig) -> CheckResult {
        if !config.check_schema {
            return CheckResult::pass("Schema", CheckCategory::DataHealth);
        }

        if ctx.schema_valid {
            CheckResult::pass("Schema", CheckCategory::DataHealth)
        } else {
            let msg = format!("Schema errors: {:?}", ctx.schema_errors);
            CheckResult::crit("Schema", CheckCategory::DataHealth, &msg)
                .with_evidence(Evidence::new("schema_validation")
                    .with_sample(ctx.schema_errors.clone()))
        }
    }
}

/// Dividends check - coverage and recency.
#[derive(Debug, Clone, Default)]
pub struct DividendsCheck {
    pub market: Market,
}

impl DividendsCheck {
    pub fn new(market: Market) -> Self {
        Self { market }
    }
}

impl DataHealthCheck for DividendsCheck {
    fn name(&self) -> &str {
        "Dividends"
    }

    fn run(&self, ctx: &DataContext, config: &DataHealthConfig) -> CheckResult {
        let count = ctx.dividends_30d;
        let min = config.dividends_min_30d;

        let evidence = Evidence::new("dividends.count_30d")
            .with_current(Decimal::from(count))
            .with_sample(ctx.dividend_types.clone());

        if count == 0 {
            let msg = format!("No dividends in last 30 days (expected >= {})", min);
            CheckResult::crit(format!("Dividends_{:?}", self.market), CheckCategory::DataHealth, &msg)
                .with_value(Decimal::from(count))
                .with_threshold(Decimal::from(min))
                .with_evidence(evidence)
                .with_market(self.market)
        } else if count < min {
            let msg = format!("Low dividend count: {} (expected >= {})", count, min);
            CheckResult::warn(format!("Dividends_{:?}", self.market), CheckCategory::DataHealth, &msg)
                .with_value(Decimal::from(count))
                .with_threshold(Decimal::from(min))
                .with_evidence(evidence)
                .with_market(self.market)
        } else {
            CheckResult::pass(format!("Dividends_{:?}", self.market), CheckCategory::DataHealth)
                .with_value(Decimal::from(count))
                .with_evidence(evidence)
                .with_market(self.market)
        }
    }
}

/// Interest rates check - recency by region.
#[derive(Debug, Clone, Default)]
pub struct InterestRatesCheck {
    pub market: Market,
}

impl InterestRatesCheck {
    pub fn new(market: Market) -> Self {
        Self { market }
    }
}

impl DataHealthCheck for InterestRatesCheck {
    fn name(&self) -> &str {
        "InterestRates"
    }

    fn run(&self, ctx: &DataContext, config: &DataHealthConfig) -> CheckResult {
        let last_date = ctx.last_interest_rate.get(&self.market);
        let count = *ctx.interest_rate_count.get(&self.market).unwrap_or(&0);
        let max_days = config.interest_rates_max_days;

        let evidence = Evidence::new("interest_rates")
            .with_sample(vec![
                format!("count: {}", count),
                format!("last_date: {:?}", last_date),
            ]);

        match last_date {
            Some(d) => {
                let days = (ctx.as_of - *d).num_days();
                if days > max_days as i64 {
                    let msg = format!("Interest rates {:?} stale: {} days (max: {})", self.market, days, max_days);
                    CheckResult::warn(format!("InterestRates_{:?}", self.market), CheckCategory::DataHealth, &msg)
                        .with_value(Decimal::from(days))
                        .with_threshold(Decimal::from(max_days))
                        .with_evidence(evidence)
                        .with_market(self.market)
                } else {
                    CheckResult::pass(format!("InterestRates_{:?}", self.market), CheckCategory::DataHealth)
                        .with_value(Decimal::from(days))
                        .with_evidence(evidence)
                        .with_market(self.market)
                }
            }
            None => {
                let msg = format!("No interest rates for {:?}", self.market);
                CheckResult::warn(format!("InterestRates_{:?}", self.market), CheckCategory::DataHealth, &msg)
                    .with_evidence(evidence)
                    .with_market(self.market)
            }
        }
    }
}

/// Data health engine that runs all checks.
pub struct DataHealthEngine {
    checks: Vec<Box<dyn DataHealthCheck>>,
}

impl DataHealthEngine {
    pub fn new(markets: &[Market]) -> Self {
        let mut checks: Vec<Box<dyn DataHealthCheck>> = Vec::new();
        
        for &market in markets {
            checks.push(Box::new(FreshnessCheck::new(market)));
            checks.push(Box::new(CoverageCheck::new(market)));
            checks.push(Box::new(WatermarkCheck::new(market)));
            checks.push(Box::new(DividendsCheck::new(market)));
            checks.push(Box::new(InterestRatesCheck::new(market)));
        }
        
        // Global checks
        checks.push(Box::new(NullCheck));
        checks.push(Box::new(OutlierCheck));
        checks.push(Box::new(SchemaCheck));

        Self { checks }
    }

    /// Run all data health checks.
    pub fn run_all(&self, ctx: &DataContext, config: &DataHealthConfig) -> Vec<CheckResult> {
        self.checks.iter()
            .map(|check| check.run(ctx, config))
            .collect()
    }

    /// Get check names.
    pub fn check_names(&self) -> Vec<&str> {
        self.checks.iter().map(|c| c.name()).collect()
    }
}

impl Default for DataHealthEngine {
    fn default() -> Self {
        Self::new(&[Market::BR, Market::US])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    #[test]
    fn test_freshness_pass() {
        let mut ctx = DataContext::new(date(2024, 1, 10));
        ctx.last_ohlcv_date.insert(Market::BR, date(2024, 1, 9));

        let check = FreshnessCheck::new(Market::BR);
        let config = DataHealthConfig::default();
        let result = check.run(&ctx, &config);

        assert!(result.passed);
        assert_eq!(result.severity, Severity::Info);
    }

    #[test]
    fn test_freshness_warn() {
        let mut ctx = DataContext::new(date(2024, 1, 10));
        ctx.last_ohlcv_date.insert(Market::BR, date(2024, 1, 6)); // 4 days ago

        let check = FreshnessCheck::new(Market::BR);
        let config = DataHealthConfig::default(); // warn at 2, crit at 5
        let result = check.run(&ctx, &config);

        assert!(!result.passed);
        assert_eq!(result.severity, Severity::Warn);
    }

    #[test]
    fn test_freshness_crit() {
        let mut ctx = DataContext::new(date(2024, 1, 15));
        ctx.last_ohlcv_date.insert(Market::BR, date(2024, 1, 1)); // 14 days ago

        let check = FreshnessCheck::new(Market::BR);
        let config = DataHealthConfig::default();
        let result = check.run(&ctx, &config);

        assert!(!result.passed);
        assert_eq!(result.severity, Severity::Crit);
    }

    #[test]
    fn test_coverage_pass() {
        let mut ctx = DataContext::new(date(2024, 1, 10));
        ctx.symbol_count.insert(Market::BR, 100);
        ctx.symbols_with_data.insert(Market::BR, 90);

        let check = CoverageCheck::new(Market::BR);
        let config = DataHealthConfig::default();
        let result = check.run(&ctx, &config);

        assert!(result.passed);
        assert_eq!(result.value, dec!(90));
    }

    #[test]
    fn test_coverage_warn() {
        let mut ctx = DataContext::new(date(2024, 1, 10));
        ctx.symbol_count.insert(Market::BR, 100);
        ctx.symbols_with_data.insert(Market::BR, 70);

        let check = CoverageCheck::new(Market::BR);
        let config = DataHealthConfig::default();
        let result = check.run(&ctx, &config);

        assert!(!result.passed);
        assert_eq!(result.severity, Severity::Warn);
    }

    #[test]
    fn test_watermark_regression_crit() {
        let mut ctx = DataContext::new(date(2024, 1, 10));
        ctx.previous_watermarks.insert(Market::BR, date(2024, 1, 8));
        ctx.current_watermarks.insert(Market::BR, date(2024, 1, 5)); // went back!

        let check = WatermarkCheck::new(Market::BR);
        let config = DataHealthConfig::default();
        let result = check.run(&ctx, &config);

        assert!(!result.passed);
        assert_eq!(result.severity, Severity::Crit);
    }

    #[test]
    fn test_null_check_crit() {
        let mut ctx = DataContext::new(date(2024, 1, 10));
        ctx.null_counts.insert("close".to_string(), 5);
        ctx.total_rows = 1000;

        let check = NullCheck;
        let config = DataHealthConfig::default();
        let result = check.run(&ctx, &config);

        assert!(!result.passed);
        assert_eq!(result.severity, Severity::Crit);
    }

    #[test]
    fn test_outlier_price_zero_crit() {
        let mut ctx = DataContext::new(date(2024, 1, 10));
        ctx.outlier_counts.insert("price_zero_negative".to_string(), 10);
        ctx.total_rows = 1000;

        let check = OutlierCheck;
        let config = DataHealthConfig::default();
        let result = check.run(&ctx, &config);

        assert!(!result.passed);
        assert_eq!(result.severity, Severity::Crit);
    }

    #[test]
    fn test_engine_runs_all_checks() {
        let ctx = DataContext::new(date(2024, 1, 10));
        let engine = DataHealthEngine::new(&[Market::BR]);
        let config = DataHealthConfig::default();

        let results = engine.run_all(&ctx, &config);
        
        // Should have checks for: Freshness, Coverage, Watermark, Dividends, InterestRates (BR)
        // Plus global: Nulls, Outliers, Schema
        assert!(results.len() >= 8);
    }
}


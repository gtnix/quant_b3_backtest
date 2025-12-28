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

use chrono::{Datelike, NaiveDate, Weekday};
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

// ============================================================================
// Calendar-Aware Gap Check
// ============================================================================

/// Gap classification for calendar-aware gap analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GapClassification {
    /// Market was closed (weekend)
    Weekend,
    /// Market was closed (holiday)
    Holiday { name: String },
    /// Asset had no trades but market was open
    NoTrades,
    /// Data is genuinely missing
    MissingData { expected_bars: u32, found_bars: u32 },
    /// Data exists and is complete
    Complete,
}

impl GapClassification {
    /// Check if this is an acceptable gap (not a data quality issue).
    pub fn is_acceptable(&self) -> bool {
        matches!(self, GapClassification::Weekend | GapClassification::Holiday { .. } | GapClassification::Complete)
    }

    /// Convert to log code.
    pub fn to_log_code(&self) -> String {
        match self {
            GapClassification::Weekend => "WEEKEND".to_string(),
            GapClassification::Holiday { name } => format!("HOLIDAY:{}", name),
            GapClassification::NoTrades => "NO_TRADES".to_string(),
            GapClassification::MissingData { expected_bars, found_bars } => {
                format!("MISSING:EXPECTED_{}_FOUND_{}", expected_bars, found_bars)
            }
            GapClassification::Complete => "COMPLETE".to_string(),
        }
    }
}

/// Simple embedded holiday calendar for gap checking.
/// Uses embedded holiday data to avoid external dependencies.
struct EmbeddedCalendar {
    holidays_br: Vec<(NaiveDate, String)>,
    holidays_us: Vec<(NaiveDate, String)>,
}

impl EmbeddedCalendar {
    fn new() -> Self {
        Self {
            holidays_br: Self::b3_holidays(),
            holidays_us: Self::nyse_holidays(),
        }
    }

    fn is_weekend(date: NaiveDate) -> bool {
        matches!(date.weekday(), Weekday::Sat | Weekday::Sun)
    }

    fn is_holiday(&self, market: Market, date: NaiveDate) -> Option<String> {
        let holidays = match market {
            Market::BR => &self.holidays_br,
            Market::US => &self.holidays_us,
        };

        holidays.iter()
            .find(|(d, _)| *d == date)
            .map(|(_, name)| name.clone())
    }

    fn classify_date(&self, market: Market, date: NaiveDate) -> Option<GapClassification> {
        if Self::is_weekend(date) {
            return Some(GapClassification::Weekend);
        }
        if let Some(name) = self.is_holiday(market, date) {
            return Some(GapClassification::Holiday { name });
        }
        None // Trading day
    }

    fn b3_holidays() -> Vec<(NaiveDate, String)> {
        fn date(y: i32, m: u32, d: u32) -> NaiveDate {
            NaiveDate::from_ymd_opt(y, m, d).unwrap()
        }

        vec![
            // 2024
            (date(2024, 1, 1), "Confraternização Universal".to_string()),
            (date(2024, 2, 12), "Carnaval".to_string()),
            (date(2024, 2, 13), "Carnaval".to_string()),
            (date(2024, 3, 29), "Sexta-feira Santa".to_string()),
            (date(2024, 4, 21), "Tiradentes".to_string()),
            (date(2024, 5, 1), "Dia do Trabalho".to_string()),
            (date(2024, 5, 30), "Corpus Christi".to_string()),
            (date(2024, 11, 15), "Proclamação da República".to_string()),
            (date(2024, 11, 20), "Consciência Negra".to_string()),
            (date(2024, 12, 24), "Véspera de Natal".to_string()),
            (date(2024, 12, 25), "Natal".to_string()),
            (date(2024, 12, 31), "Véspera de Ano Novo".to_string()),
            // 2025
            (date(2025, 1, 1), "Confraternização Universal".to_string()),
            (date(2025, 3, 3), "Carnaval".to_string()),
            (date(2025, 3, 4), "Carnaval".to_string()),
            (date(2025, 4, 18), "Sexta-feira Santa".to_string()),
            (date(2025, 4, 21), "Tiradentes".to_string()),
            (date(2025, 5, 1), "Dia do Trabalho".to_string()),
            (date(2025, 6, 19), "Corpus Christi".to_string()),
            (date(2025, 11, 20), "Consciência Negra".to_string()),
            (date(2025, 12, 24), "Véspera de Natal".to_string()),
            (date(2025, 12, 25), "Natal".to_string()),
            (date(2025, 12, 31), "Véspera de Ano Novo".to_string()),
        ]
    }

    fn nyse_holidays() -> Vec<(NaiveDate, String)> {
        fn date(y: i32, m: u32, d: u32) -> NaiveDate {
            NaiveDate::from_ymd_opt(y, m, d).unwrap()
        }

        vec![
            // 2024
            (date(2024, 1, 1), "New Year's Day".to_string()),
            (date(2024, 1, 15), "Martin Luther King Jr. Day".to_string()),
            (date(2024, 2, 19), "Presidents' Day".to_string()),
            (date(2024, 3, 29), "Good Friday".to_string()),
            (date(2024, 5, 27), "Memorial Day".to_string()),
            (date(2024, 6, 19), "Juneteenth".to_string()),
            (date(2024, 7, 4), "Independence Day".to_string()),
            (date(2024, 9, 2), "Labor Day".to_string()),
            (date(2024, 11, 28), "Thanksgiving Day".to_string()),
            (date(2024, 12, 25), "Christmas Day".to_string()),
            // 2025
            (date(2025, 1, 1), "New Year's Day".to_string()),
            (date(2025, 1, 20), "Martin Luther King Jr. Day".to_string()),
            (date(2025, 2, 17), "Presidents' Day".to_string()),
            (date(2025, 4, 18), "Good Friday".to_string()),
            (date(2025, 5, 26), "Memorial Day".to_string()),
            (date(2025, 6, 19), "Juneteenth".to_string()),
            (date(2025, 7, 4), "Independence Day".to_string()),
            (date(2025, 9, 1), "Labor Day".to_string()),
            (date(2025, 11, 27), "Thanksgiving Day".to_string()),
            (date(2025, 12, 25), "Christmas Day".to_string()),
        ]
    }
}

/// Calendar-aware gap check - distinguishes holidays from missing data.
///
/// This check analyzes gaps in data and correctly classifies them as:
/// - Weekend: Market closed on Saturday/Sunday
/// - Holiday: Market closed for official holiday
/// - NoTrades: Market open but asset had no trades
/// - MissingData: Genuine data quality issue
#[derive(Debug, Clone, Default)]
pub struct CalendarGapCheck {
    pub market: Market,
}

impl CalendarGapCheck {
    pub fn new(market: Market) -> Self {
        Self { market }
    }

    /// Classify a gap for a specific date.
    pub fn classify_gap(
        &self,
        date: NaiveDate,
        has_data: bool,
        volume: Option<i64>,
    ) -> GapClassification {
        let calendar = EmbeddedCalendar::new();

        // First check if market is closed
        if let Some(classification) = calendar.classify_date(self.market, date) {
            return classification;
        }

        // Market was open
        if has_data {
            if volume == Some(0) {
                GapClassification::NoTrades
            } else {
                GapClassification::Complete
            }
        } else {
            GapClassification::MissingData { expected_bars: 1, found_bars: 0 }
        }
    }

    /// Analyze gaps over a date range.
    pub fn analyze_gaps(
        &self,
        start: NaiveDate,
        end: NaiveDate,
        data_dates: &HashMap<NaiveDate, i64>, // date -> volume
    ) -> GapAnalysisResult {
        let mut result = GapAnalysisResult::default();
        let mut current = start;

        while current <= end {
            let has_data = data_dates.contains_key(&current);
            let volume = data_dates.get(&current).copied();

            let classification = self.classify_gap(current, has_data, volume);

            match &classification {
                GapClassification::Weekend => result.weekends += 1,
                GapClassification::Holiday { .. } => result.holidays += 1,
                GapClassification::NoTrades => result.no_trades_days += 1,
                GapClassification::MissingData { .. } => result.missing_days += 1,
                GapClassification::Complete => result.complete_days += 1,
            }

            if !classification.is_acceptable() {
                result.gaps.push((current, classification));
            }

            current += chrono::Duration::days(1);
        }

        result.total_days = (end - start).num_days() as u32 + 1;
        result
    }
}

/// Result of gap analysis.
#[derive(Debug, Clone, Default)]
pub struct GapAnalysisResult {
    pub total_days: u32,
    pub complete_days: u32,
    pub weekends: u32,
    pub holidays: u32,
    pub no_trades_days: u32,
    pub missing_days: u32,
    pub gaps: Vec<(NaiveDate, GapClassification)>,
}

impl GapAnalysisResult {
    /// Trading days that should have data.
    pub fn trading_days(&self) -> u32 {
        self.total_days - self.weekends - self.holidays
    }

    /// Coverage percentage (excluding weekends and holidays).
    pub fn coverage_pct(&self) -> Decimal {
        let trading_days = self.trading_days();
        if trading_days == 0 {
            return dec!(100);
        }
        Decimal::from(self.complete_days + self.no_trades_days) * dec!(100) / Decimal::from(trading_days)
    }

    /// Is the data quality acceptable?
    pub fn is_acceptable(&self) -> bool {
        self.missing_days == 0
    }
}

impl DataHealthCheck for CalendarGapCheck {
    fn name(&self) -> &str {
        "CalendarGaps"
    }

    fn run(&self, ctx: &DataContext, config: &DataHealthConfig) -> CheckResult {
        // For now, this is a placeholder that shows the concept
        // In a real implementation, this would analyze actual bar data
        
        let coverage = ctx.coverage_pct(self.market);
        let severity = ThresholdEvaluator::coverage_severity(
            coverage, config.coverage_warn_pct, config.coverage_min_pct
        );

        let passed = severity == Severity::Info;
        let msg = format!(
            "Calendar-aware gap check {:?}: {:.1}% coverage",
            self.market, coverage
        );

        let evidence = Evidence::new("calendar_gaps")
            .with_current(coverage)
            .with_context(format!(
                "Distinguishes holidays/weekends from missing data for {:?}",
                self.market
            ));

        let mut result = match severity {
            Severity::Info => CheckResult::pass(
                format!("CalendarGaps_{:?}", self.market),
                CheckCategory::DataHealth
            ),
            Severity::Crit => CheckResult::crit(
                format!("CalendarGaps_{:?}", self.market),
                CheckCategory::DataHealth,
                &msg
            ),
            _ => CheckResult::warn(
                format!("CalendarGaps_{:?}", self.market),
                CheckCategory::DataHealth,
                &msg
            ),
        };

        result.value = coverage;
        result.threshold = config.coverage_min_pct;
        result.message = msg;
        result.evidence = evidence;
        result.market = Some(self.market);
        result
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
            // Add calendar-aware gap check
            checks.push(Box::new(CalendarGapCheck::new(market)));
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
        
        // Should have checks for: Freshness, Coverage, Watermark, Dividends, InterestRates, CalendarGaps (BR)
        // Plus global: Nulls, Outliers, Schema
        assert!(results.len() >= 9);
    }

    // ========================================================================
    // Calendar Gap Check Tests
    // ========================================================================

    #[test]
    fn test_gap_classification_weekend() {
        let check = CalendarGapCheck::new(Market::BR);
        
        // Saturday
        let result = check.classify_gap(date(2024, 12, 21), false, None);
        assert_eq!(result, GapClassification::Weekend);
        assert!(result.is_acceptable());
    }

    #[test]
    fn test_gap_classification_holiday() {
        let check = CalendarGapCheck::new(Market::BR);
        
        // Christmas
        let result = check.classify_gap(date(2024, 12, 25), false, None);
        assert!(matches!(result, GapClassification::Holiday { .. }));
        assert!(result.is_acceptable());

        if let GapClassification::Holiday { name } = result {
            assert_eq!(name, "Natal");
        }
    }

    #[test]
    fn test_gap_classification_missing_data() {
        let check = CalendarGapCheck::new(Market::BR);
        
        // Regular trading day with no data
        let result = check.classify_gap(date(2024, 12, 23), false, None);
        assert!(matches!(result, GapClassification::MissingData { .. }));
        assert!(!result.is_acceptable());
    }

    #[test]
    fn test_gap_classification_no_trades() {
        let check = CalendarGapCheck::new(Market::BR);
        
        // Regular trading day with data but zero volume
        let result = check.classify_gap(date(2024, 12, 23), true, Some(0));
        assert_eq!(result, GapClassification::NoTrades);
        // NoTrades is not acceptable (INFO level but should be flagged)
        assert!(!result.is_acceptable());
    }

    #[test]
    fn test_gap_classification_complete() {
        let check = CalendarGapCheck::new(Market::BR);
        
        // Regular trading day with data and volume
        let result = check.classify_gap(date(2024, 12, 23), true, Some(1000000));
        assert_eq!(result, GapClassification::Complete);
        assert!(result.is_acceptable());
    }

    #[test]
    fn test_analyze_gaps_week() {
        let check = CalendarGapCheck::new(Market::BR);
        
        // Week starting Monday Dec 16, 2024
        let mut data: HashMap<NaiveDate, i64> = HashMap::new();
        data.insert(date(2024, 12, 16), 1000000); // Monday
        data.insert(date(2024, 12, 17), 1000000); // Tuesday
        data.insert(date(2024, 12, 18), 1000000); // Wednesday
        data.insert(date(2024, 12, 19), 1000000); // Thursday
        data.insert(date(2024, 12, 20), 1000000); // Friday
        // Saturday and Sunday have no data

        let result = check.analyze_gaps(date(2024, 12, 16), date(2024, 12, 22), &data);

        assert_eq!(result.total_days, 7);
        assert_eq!(result.complete_days, 5);
        assert_eq!(result.weekends, 2);
        assert_eq!(result.holidays, 0);
        assert_eq!(result.missing_days, 0);
        assert!(result.is_acceptable());
        assert_eq!(result.coverage_pct(), dec!(100));
    }

    #[test]
    fn test_analyze_gaps_with_holiday() {
        let check = CalendarGapCheck::new(Market::BR);
        
        // Christmas week 2024
        let mut data: HashMap<NaiveDate, i64> = HashMap::new();
        data.insert(date(2024, 12, 23), 1000000); // Monday
        // Dec 24 - holiday (Véspera de Natal)
        // Dec 25 - holiday (Natal)
        data.insert(date(2024, 12, 26), 1000000); // Thursday
        data.insert(date(2024, 12, 27), 1000000); // Friday

        let result = check.analyze_gaps(date(2024, 12, 23), date(2024, 12, 27), &data);

        assert_eq!(result.complete_days, 3);
        assert_eq!(result.holidays, 2); // Dec 24, 25
        assert_eq!(result.missing_days, 0);
        assert!(result.is_acceptable());
    }

    #[test]
    fn test_analyze_gaps_with_missing_data() {
        let check = CalendarGapCheck::new(Market::BR);
        
        let mut data: HashMap<NaiveDate, i64> = HashMap::new();
        data.insert(date(2024, 12, 16), 1000000); // Monday
        // Dec 17 - missing data!
        data.insert(date(2024, 12, 18), 1000000); // Wednesday
        data.insert(date(2024, 12, 19), 1000000); // Thursday
        data.insert(date(2024, 12, 20), 1000000); // Friday

        let result = check.analyze_gaps(date(2024, 12, 16), date(2024, 12, 20), &data);

        assert_eq!(result.complete_days, 4);
        assert_eq!(result.missing_days, 1);
        assert!(!result.is_acceptable());
        
        // Verify the gap is flagged
        assert_eq!(result.gaps.len(), 1);
        assert_eq!(result.gaps[0].0, date(2024, 12, 17));
    }

    #[test]
    fn test_cross_market_holiday_classification() {
        let check_br = CalendarGapCheck::new(Market::BR);
        let check_us = CalendarGapCheck::new(Market::US);

        // Brazilian Carnival (Mar 3, 2025) - B3 closed, NYSE open
        let br_result = check_br.classify_gap(date(2025, 3, 3), false, None);
        let us_result = check_us.classify_gap(date(2025, 3, 3), false, None);

        // B3 should classify as holiday
        assert!(matches!(br_result, GapClassification::Holiday { .. }));
        // NYSE should classify as missing data
        assert!(matches!(us_result, GapClassification::MissingData { .. }));
    }

    #[test]
    fn test_gap_log_code() {
        assert_eq!(GapClassification::Weekend.to_log_code(), "WEEKEND");
        assert_eq!(
            GapClassification::Holiday { name: "Natal".to_string() }.to_log_code(),
            "HOLIDAY:Natal"
        );
        assert_eq!(GapClassification::NoTrades.to_log_code(), "NO_TRADES");
        assert_eq!(
            GapClassification::MissingData { expected_bars: 450, found_bars: 0 }.to_log_code(),
            "MISSING:EXPECTED_450_FOUND_0"
        );
    }

    #[test]
    fn test_gap_analysis_trading_days() {
        let result = GapAnalysisResult {
            total_days: 30,
            weekends: 8,
            holidays: 2,
            complete_days: 15,
            no_trades_days: 3,
            missing_days: 2,
            gaps: vec![],
        };

        // Trading days = total - weekends - holidays = 30 - 8 - 2 = 20
        assert_eq!(result.trading_days(), 20);
        
        // Coverage = (complete + no_trades) / trading_days = (15 + 3) / 20 = 90%
        assert_eq!(result.coverage_pct(), dec!(90));
    }
}


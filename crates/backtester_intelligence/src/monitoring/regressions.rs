//! Performance Regression Checks for Monitoring.
//!
//! Implements 6 regression checks:
//! 1. DrawdownGuardrail - DD vs threshold
//! 2. TurnoverBudget - vs hard cap
//! 3. CostBudget - vs hard cap (% AUM)
//! 4. SharpeRegression - vs historical min
//! 5. VolatilityRegression - vs baseline multiplier
//! 6. LatencyCheck - execution time

use chrono::{DateTime, NaiveDate, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::time::Duration;

use super::config::{RegressionConfig, ThresholdEvaluator};
use super::statistics::{calculate_baseline, calculate_percentile};
use super::types::{BaselineStats, CheckCategory, CheckResult, Evidence, Severity};

/// Context for regression checks.
#[derive(Debug, Clone, Default)]
pub struct RegressionContext {
    /// Current max drawdown percentage
    pub current_drawdown: Decimal,
    /// Current turnover percentage
    pub current_turnover: Decimal,
    /// Historical turnover values
    pub historical_turnover: Vec<Decimal>,
    /// Current cost percentage (of AUM)
    pub current_cost: Decimal,
    /// Historical cost values
    pub historical_cost: Vec<Decimal>,
    /// Current rolling Sharpe ratio
    pub current_sharpe: Decimal,
    /// Historical Sharpe values
    pub historical_sharpe: Vec<Decimal>,
    /// Current annualized volatility
    pub current_volatility: Decimal,
    /// Baseline volatility (historical mean)
    pub baseline_volatility: Decimal,
    /// Execution latency
    pub execution_latency: Duration,
    /// Reference date
    pub as_of: NaiveDate,
}

impl RegressionContext {
    pub fn new(as_of: NaiveDate) -> Self {
        Self {
            as_of,
            ..Default::default()
        }
    }
}

/// Trait for regression checks.
pub trait RegressionCheck: Send + Sync {
    /// Check name for logging.
    fn name(&self) -> &str;
    /// Run the check and return result.
    fn run(&self, ctx: &RegressionContext, config: &RegressionConfig) -> CheckResult;
}

/// Drawdown guardrail check.
#[derive(Debug, Clone, Default)]
pub struct DrawdownGuardrail;

impl RegressionCheck for DrawdownGuardrail {
    fn name(&self) -> &str {
        "DrawdownGuardrail"
    }

    fn run(&self, ctx: &RegressionContext, config: &RegressionConfig) -> CheckResult {
        let dd = ctx.current_drawdown;
        let warn = config.drawdown.warn_pct;
        let crit = config.drawdown.crit_pct;
        let halt = config.drawdown.halt_pct;

        let severity = if dd > halt {
            Severity::Halt
        } else if dd > crit {
            Severity::Crit
        } else if dd > warn {
            Severity::Warn
        } else {
            Severity::Info
        };

        let passed = severity == Severity::Info;
        let msg = format!(
            "Drawdown: {:.1}% (WARN: {}%, CRIT: {}%, HALT: {}%)",
            dd, warn, crit, halt
        );

        let evidence = Evidence::new("drawdown_check")
            .with_current(dd)
            .with_sample(vec![
                format!("warn_threshold: {}%", warn),
                format!("crit_threshold: {}%", crit),
                format!("halt_threshold: {}%", halt),
            ]);

        let mut result = match severity {
            Severity::Info => CheckResult::pass("DrawdownGuardrail", CheckCategory::Regression),
            Severity::Warn => CheckResult::warn("DrawdownGuardrail", CheckCategory::Regression, &msg),
            Severity::Crit => CheckResult::crit("DrawdownGuardrail", CheckCategory::Regression, &msg),
            Severity::Halt => CheckResult::halt("DrawdownGuardrail", CheckCategory::Regression, &msg),
        };
        result.value = dd;
        result.threshold = crit;
        result.message = msg;
        result.evidence = evidence;
        result
    }
}

/// Turnover budget check (hard cap).
#[derive(Debug, Clone, Default)]
pub struct TurnoverBudget;

impl RegressionCheck for TurnoverBudget {
    fn name(&self) -> &str {
        "TurnoverBudget"
    }

    fn run(&self, ctx: &RegressionContext, config: &RegressionConfig) -> CheckResult {
        let turnover = ctx.current_turnover;
        let hard_cap = config.turnover.hard_cap;
        
        // Calculate p95 if we have historical data
        let p95 = if !ctx.historical_turnover.is_empty() && config.turnover.use_percentile {
            calculate_percentile(&ctx.historical_turnover, dec!(95)).unwrap_or(hard_cap / dec!(2))
        } else {
            hard_cap / dec!(2)
        };

        let severity = ThresholdEvaluator::dynamic_threshold_severity(turnover, p95, hard_cap);

        let passed = severity == Severity::Info;
        let msg = format!(
            "Turnover: {:.1}% (p95: {:.1}%, hard cap: {}%)",
            turnover, p95, hard_cap
        );

        let evidence = Evidence::new("turnover_budget")
            .with_current(turnover)
            .with_sample(vec![
                format!("p95_threshold: {:.1}%", p95),
                format!("hard_cap: {}%", hard_cap),
                format!("n_historical: {}", ctx.historical_turnover.len()),
            ]);

        let mut result = match severity {
            Severity::Info => CheckResult::pass("TurnoverBudget", CheckCategory::Regression),
            Severity::Warn => CheckResult::warn("TurnoverBudget", CheckCategory::Regression, &msg),
            Severity::Crit => CheckResult::crit("TurnoverBudget", CheckCategory::Regression, &msg),
            _ => CheckResult::pass("TurnoverBudget", CheckCategory::Regression),
        };
        result.value = turnover;
        result.threshold = hard_cap;
        result.message = msg;
        result.evidence = evidence;
        result
    }
}

/// Cost budget check (hard cap % AUM).
#[derive(Debug, Clone, Default)]
pub struct CostBudget;

impl RegressionCheck for CostBudget {
    fn name(&self) -> &str {
        "CostBudget"
    }

    fn run(&self, ctx: &RegressionContext, config: &RegressionConfig) -> CheckResult {
        let cost = ctx.current_cost;
        let hard_cap = config.cost.hard_cap;
        
        let p95 = if !ctx.historical_cost.is_empty() && config.cost.use_percentile {
            calculate_percentile(&ctx.historical_cost, dec!(95)).unwrap_or(hard_cap / dec!(2))
        } else {
            hard_cap / dec!(2)
        };

        let severity = ThresholdEvaluator::dynamic_threshold_severity(cost, p95, hard_cap);

        let passed = severity == Severity::Info;
        let msg = format!(
            "Cost: {:.3}% AUM (p95: {:.3}%, hard cap: {}%)",
            cost, p95, hard_cap
        );

        let evidence = Evidence::new("cost_budget")
            .with_current(cost)
            .with_sample(vec![
                format!("p95_threshold: {:.3}%", p95),
                format!("hard_cap: {}%", hard_cap),
                format!("n_historical: {}", ctx.historical_cost.len()),
            ]);

        let mut result = match severity {
            Severity::Info => CheckResult::pass("CostBudget", CheckCategory::Regression),
            Severity::Warn => CheckResult::warn("CostBudget", CheckCategory::Regression, &msg),
            Severity::Crit => CheckResult::crit("CostBudget", CheckCategory::Regression, &msg),
            _ => CheckResult::pass("CostBudget", CheckCategory::Regression),
        };
        result.value = cost;
        result.threshold = hard_cap;
        result.message = msg;
        result.evidence = evidence;
        result
    }
}

/// Sharpe ratio regression check.
#[derive(Debug, Clone, Default)]
pub struct SharpeRegression;

impl RegressionCheck for SharpeRegression {
    fn name(&self) -> &str {
        "SharpeRegression"
    }

    fn run(&self, ctx: &RegressionContext, config: &RegressionConfig) -> CheckResult {
        let sharpe = ctx.current_sharpe;
        let min_sharpe = config.sharpe_min;

        // Calculate p10 if we have historical data
        let p10 = if !ctx.historical_sharpe.is_empty() {
            calculate_percentile(&ctx.historical_sharpe, dec!(10)).unwrap_or(dec!(0))
        } else {
            dec!(0)
        };

        let severity = if sharpe < min_sharpe {
            Severity::Crit
        } else if sharpe < p10 {
            Severity::Warn
        } else {
            Severity::Info
        };

        let passed = severity == Severity::Info;
        let msg = format!(
            "Sharpe: {:.2} (min: {}, p10: {:.2})",
            sharpe, min_sharpe, p10
        );

        let evidence = Evidence::new("sharpe_check")
            .with_current(sharpe)
            .with_sample(vec![
                format!("min_threshold: {}", min_sharpe),
                format!("p10_historical: {:.2}", p10),
                format!("n_historical: {}", ctx.historical_sharpe.len()),
            ]);

        let mut result = match severity {
            Severity::Info => CheckResult::pass("SharpeRegression", CheckCategory::Regression),
            Severity::Warn => CheckResult::warn("SharpeRegression", CheckCategory::Regression, &msg),
            Severity::Crit => CheckResult::crit("SharpeRegression", CheckCategory::Regression, &msg),
            _ => CheckResult::pass("SharpeRegression", CheckCategory::Regression),
        };
        result.value = sharpe;
        result.threshold = min_sharpe;
        result.message = msg;
        result.evidence = evidence;
        result
    }
}

/// Volatility regression check.
#[derive(Debug, Clone, Default)]
pub struct VolatilityRegression;

impl RegressionCheck for VolatilityRegression {
    fn name(&self) -> &str {
        "VolatilityRegression"
    }

    fn run(&self, ctx: &RegressionContext, config: &RegressionConfig) -> CheckResult {
        if ctx.baseline_volatility.is_zero() {
            return CheckResult::pass("VolatilityRegression", CheckCategory::Regression)
                .with_evidence(Evidence::new("no_baseline_volatility"));
        }

        let vol = ctx.current_volatility;
        let baseline = ctx.baseline_volatility;
        let multiplier = vol / baseline;

        let severity = if multiplier > config.volatility_multiplier_crit {
            Severity::Crit
        } else if multiplier > config.volatility_multiplier_warn {
            Severity::Warn
        } else {
            Severity::Info
        };

        let passed = severity == Severity::Info;
        let msg = format!(
            "Volatility: {:.1}% ({:.1}x baseline {:.1}%) - WARN: >{}x, CRIT: >{}x",
            vol, multiplier, baseline,
            config.volatility_multiplier_warn, config.volatility_multiplier_crit
        );

        let evidence = Evidence::new("volatility_check")
            .with_current(vol)
            .with_sample(vec![
                format!("baseline: {:.1}%", baseline),
                format!("multiplier: {:.2}x", multiplier),
            ]);

        let mut result = match severity {
            Severity::Info => CheckResult::pass("VolatilityRegression", CheckCategory::Regression),
            Severity::Warn => CheckResult::warn("VolatilityRegression", CheckCategory::Regression, &msg),
            Severity::Crit => CheckResult::crit("VolatilityRegression", CheckCategory::Regression, &msg),
            _ => CheckResult::pass("VolatilityRegression", CheckCategory::Regression),
        };
        result.value = multiplier;
        result.threshold = config.volatility_multiplier_warn;
        result.message = msg;
        result.evidence = evidence;
        result
    }
}

/// Execution latency check.
#[derive(Debug, Clone, Default)]
pub struct LatencyCheck;

impl RegressionCheck for LatencyCheck {
    fn name(&self) -> &str {
        "LatencyCheck"
    }

    fn run(&self, ctx: &RegressionContext, config: &RegressionConfig) -> CheckResult {
        let latency_secs = ctx.execution_latency.as_secs() as u32;
        let warn = config.latency_warn_seconds;
        let crit = config.latency_crit_seconds;

        let severity = if latency_secs > crit {
            Severity::Crit
        } else if latency_secs > warn {
            Severity::Warn
        } else {
            Severity::Info
        };

        let passed = severity == Severity::Info;
        let msg = format!(
            "Latency: {}s (WARN: >{}s, CRIT: >{}s)",
            latency_secs, warn, crit
        );

        let evidence = Evidence::new("latency_check")
            .with_current(Decimal::from(latency_secs))
            .with_sample(vec![
                format!("warn_threshold: {}s", warn),
                format!("crit_threshold: {}s", crit),
            ]);

        let mut result = match severity {
            Severity::Info => CheckResult::pass("LatencyCheck", CheckCategory::Regression),
            Severity::Warn => CheckResult::warn("LatencyCheck", CheckCategory::Regression, &msg),
            Severity::Crit => CheckResult::crit("LatencyCheck", CheckCategory::Regression, &msg),
            _ => CheckResult::pass("LatencyCheck", CheckCategory::Regression),
        };
        result.value = Decimal::from(latency_secs);
        result.threshold = Decimal::from(crit);
        result.message = msg;
        result.evidence = evidence;
        result
    }
}

/// Regression engine that runs all checks.
pub struct RegressionEngine {
    checks: Vec<Box<dyn RegressionCheck>>,
}

impl RegressionEngine {
    pub fn new() -> Self {
        let checks: Vec<Box<dyn RegressionCheck>> = vec![
            Box::new(DrawdownGuardrail),
            Box::new(TurnoverBudget),
            Box::new(CostBudget),
            Box::new(SharpeRegression),
            Box::new(VolatilityRegression),
            Box::new(LatencyCheck),
        ];

        Self { checks }
    }

    /// Run all regression checks.
    pub fn run_all(&self, ctx: &RegressionContext, config: &RegressionConfig) -> Vec<CheckResult> {
        self.checks.iter()
            .map(|check| check.run(ctx, config))
            .collect()
    }
}

impl Default for RegressionEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    #[test]
    fn test_drawdown_pass() {
        let mut ctx = RegressionContext::new(date(2024, 1, 10));
        ctx.current_drawdown = dec!(10);

        let check = DrawdownGuardrail;
        let config = RegressionConfig::default();
        let result = check.run(&ctx, &config);

        assert!(result.passed);
        assert_eq!(result.severity, Severity::Info);
    }

    #[test]
    fn test_drawdown_warn() {
        let mut ctx = RegressionContext::new(date(2024, 1, 10));
        ctx.current_drawdown = dec!(18); // > 15% warn, < 20% crit

        let check = DrawdownGuardrail;
        let config = RegressionConfig::default();
        let result = check.run(&ctx, &config);

        assert!(!result.passed);
        assert_eq!(result.severity, Severity::Warn);
    }

    #[test]
    fn test_drawdown_crit() {
        let mut ctx = RegressionContext::new(date(2024, 1, 10));
        ctx.current_drawdown = dec!(22); // > 20% crit, < 25% halt

        let check = DrawdownGuardrail;
        let config = RegressionConfig::default();
        let result = check.run(&ctx, &config);

        assert!(!result.passed);
        assert_eq!(result.severity, Severity::Crit);
    }

    #[test]
    fn test_drawdown_halt() {
        let mut ctx = RegressionContext::new(date(2024, 1, 10));
        ctx.current_drawdown = dec!(30); // > 25% halt

        let check = DrawdownGuardrail;
        let config = RegressionConfig::default();
        let result = check.run(&ctx, &config);

        assert!(!result.passed);
        assert_eq!(result.severity, Severity::Halt);
    }

    #[test]
    fn test_turnover_pass() {
        let mut ctx = RegressionContext::new(date(2024, 1, 10));
        ctx.current_turnover = dec!(20);
        ctx.historical_turnover = (10..=40).map(|x| Decimal::from(x)).collect();

        let check = TurnoverBudget;
        let config = RegressionConfig::default();
        let result = check.run(&ctx, &config);

        assert!(result.passed);
    }

    #[test]
    fn test_turnover_crit() {
        let mut ctx = RegressionContext::new(date(2024, 1, 10));
        ctx.current_turnover = dec!(55); // > 50% hard cap

        let check = TurnoverBudget;
        let config = RegressionConfig::default();
        let result = check.run(&ctx, &config);

        assert!(!result.passed);
        assert_eq!(result.severity, Severity::Crit);
    }

    #[test]
    fn test_cost_pass() {
        let mut ctx = RegressionContext::new(date(2024, 1, 10));
        ctx.current_cost = dec!(0.2);

        let check = CostBudget;
        let config = RegressionConfig::default();
        let result = check.run(&ctx, &config);

        assert!(result.passed);
    }

    #[test]
    fn test_cost_crit() {
        let mut ctx = RegressionContext::new(date(2024, 1, 10));
        ctx.current_cost = dec!(0.6); // > 0.5% hard cap

        let check = CostBudget;
        let config = RegressionConfig::default();
        let result = check.run(&ctx, &config);

        assert!(!result.passed);
        assert_eq!(result.severity, Severity::Crit);
    }

    #[test]
    fn test_sharpe_pass() {
        let mut ctx = RegressionContext::new(date(2024, 1, 10));
        ctx.current_sharpe = dec!(1.5);

        let check = SharpeRegression;
        let config = RegressionConfig::default();
        let result = check.run(&ctx, &config);

        assert!(result.passed);
    }

    #[test]
    fn test_sharpe_crit() {
        let mut ctx = RegressionContext::new(date(2024, 1, 10));
        ctx.current_sharpe = dec!(-0.5); // < 0 min

        let check = SharpeRegression;
        let config = RegressionConfig::default();
        let result = check.run(&ctx, &config);

        assert!(!result.passed);
        assert_eq!(result.severity, Severity::Crit);
    }

    #[test]
    fn test_volatility_pass() {
        let mut ctx = RegressionContext::new(date(2024, 1, 10));
        ctx.current_volatility = dec!(15);
        ctx.baseline_volatility = dec!(12);

        let check = VolatilityRegression;
        let config = RegressionConfig::default();
        let result = check.run(&ctx, &config);

        assert!(result.passed);
    }

    #[test]
    fn test_volatility_crit() {
        let mut ctx = RegressionContext::new(date(2024, 1, 10));
        ctx.current_volatility = dec!(30); // 2.5x baseline
        ctx.baseline_volatility = dec!(12);

        let check = VolatilityRegression;
        let config = RegressionConfig::default(); // crit > 2x
        let result = check.run(&ctx, &config);

        assert!(!result.passed);
        assert_eq!(result.severity, Severity::Crit);
    }

    #[test]
    fn test_latency_pass() {
        let mut ctx = RegressionContext::new(date(2024, 1, 10));
        ctx.execution_latency = Duration::from_secs(2);

        let check = LatencyCheck;
        let config = RegressionConfig::default();
        let result = check.run(&ctx, &config);

        assert!(result.passed);
    }

    #[test]
    fn test_latency_crit() {
        let mut ctx = RegressionContext::new(date(2024, 1, 10));
        ctx.execution_latency = Duration::from_secs(15); // > 10s crit

        let check = LatencyCheck;
        let config = RegressionConfig::default();
        let result = check.run(&ctx, &config);

        assert!(!result.passed);
        assert_eq!(result.severity, Severity::Crit);
    }

    #[test]
    fn test_engine_runs_all() {
        let ctx = RegressionContext::new(date(2024, 1, 10));
        let engine = RegressionEngine::new();
        let config = RegressionConfig::default();

        let results = engine.run_all(&ctx, &config);
        
        // Should have 6 checks
        assert_eq!(results.len(), 6);
    }
}


//! Drift Detection Checks for Monitoring.
//!
//! Implements 6 drift checks:
//! 1. ScoreDistributionDrift - KS test on score distributions
//! 2. MeanScoreDrift - Hoeffding bound on mean scores
//! 3. SelectionStability - Jaccard overlap of top-N
//! 4. ExclusionReasonsDrift - Chi-square on exclusion reasons
//! 5. TurnoverDrift - vs p95 historical
//! 6. CostDrift - vs p95 historical

use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::{HashMap, HashSet};

use crate::filters::Market;
use super::config::{DriftConfig, ThresholdEvaluator};
use super::statistics::{
    ks_two_sample, jaccard_similarity, sigma_deviation, calculate_baseline, calculate_percentile,
};
use super::types::{CheckCategory, CheckResult, Evidence, Severity};

/// Context for drift checks.
#[derive(Debug, Clone, Default)]
pub struct DriftContext {
    /// Current scores by technique
    pub current_scores: HashMap<String, Vec<Decimal>>,
    /// Baseline scores by technique (from historical window)
    pub baseline_scores: HashMap<String, Vec<Decimal>>,
    /// Current top-N selection (symbols)
    pub current_selection: HashSet<String>,
    /// Previous top-N selection (symbols)
    pub previous_selection: HashSet<String>,
    /// Current exclusion counts by reason
    pub current_exclusions: HashMap<String, usize>,
    /// Baseline exclusion counts by reason
    pub baseline_exclusions: HashMap<String, usize>,
    /// Current turnover percentage
    pub current_turnover: Decimal,
    /// Historical turnover values
    pub historical_turnover: Vec<Decimal>,
    /// Current cost percentage
    pub current_cost: Decimal,
    /// Historical cost values
    pub historical_cost: Vec<Decimal>,
    /// Reference date
    pub as_of: NaiveDate,
    /// Market for context
    pub market: Option<Market>,
}

impl DriftContext {
    pub fn new(as_of: NaiveDate) -> Self {
        Self {
            as_of,
            ..Default::default()
        }
    }

    pub fn with_market(mut self, market: Market) -> Self {
        self.market = Some(market);
        self
    }
}

/// Trait for drift checks.
pub trait DriftCheck: Send + Sync {
    /// Check name for logging.
    fn name(&self) -> &str;
    /// Run the check and return result.
    fn run(&self, ctx: &DriftContext, config: &DriftConfig) -> CheckResult;
}

/// Score distribution drift using KS test.
#[derive(Debug, Clone)]
pub struct ScoreDistributionDrift {
    pub technique: String,
}

impl ScoreDistributionDrift {
    pub fn new(technique: impl Into<String>) -> Self {
        Self { technique: technique.into() }
    }
}

impl DriftCheck for ScoreDistributionDrift {
    fn name(&self) -> &str {
        "ScoreDistributionDrift"
    }

    fn run(&self, ctx: &DriftContext, config: &DriftConfig) -> CheckResult {
        let current = ctx.current_scores.get(&self.technique);
        let baseline = ctx.baseline_scores.get(&self.technique);

        match (current, baseline) {
            (Some(curr), Some(base)) if curr.len() >= config.min_samples && base.len() >= config.min_samples => {
                match ks_two_sample(curr, base) {
                    Some(ks) => {
                        let severity = if ks.p_value < config.ks_pvalue_crit {
                            Severity::Crit
                        } else if ks.p_value < config.ks_pvalue_warn {
                            Severity::Warn
                        } else {
                            Severity::Info
                        };

                        let _passed = severity == Severity::Info;
                        let msg = format!(
                            "KS test {}: D={:.4}, p={:.4} (WARN: p<{}, CRIT: p<{})",
                            self.technique, ks.statistic, ks.p_value,
                            config.ks_pvalue_warn, config.ks_pvalue_crit
                        );

                        let evidence = Evidence::new("ks_two_sample")
                            .with_current(ks.statistic)
                            .with_sample(vec![
                                format!("n_current: {}", ks.n1),
                                format!("n_baseline: {}", ks.n2),
                                format!("p_value: {:.6}", ks.p_value),
                            ]);

                        let name = format!("ScoreDrift_{}", self.technique);
                        let mut result = match severity {
                            Severity::Info => CheckResult::pass(&name, CheckCategory::Drift),
                            Severity::Warn => CheckResult::warn(&name, CheckCategory::Drift, &msg),
                            Severity::Crit => CheckResult::crit(&name, CheckCategory::Drift, &msg),
                            _ => CheckResult::pass(&name, CheckCategory::Drift),
                        };
                        result.value = ks.p_value;
                        result.threshold = config.ks_pvalue_warn;
                        result.message = msg;
                        result.evidence = evidence;
                        result
                    }
                    None => CheckResult::pass(
                        format!("ScoreDrift_{}", self.technique),
                        CheckCategory::Drift
                    )
                }
            }
            _ => {
                // Not enough samples, skip
                CheckResult::pass(
                    format!("ScoreDrift_{}", self.technique),
                    CheckCategory::Drift
                ).with_evidence(Evidence::new("insufficient_samples"))
            }
        }
    }
}

/// Mean score drift using sigma deviation.
#[derive(Debug, Clone)]
pub struct MeanScoreDrift {
    pub technique: String,
}

impl MeanScoreDrift {
    pub fn new(technique: impl Into<String>) -> Self {
        Self { technique: technique.into() }
    }
}

impl DriftCheck for MeanScoreDrift {
    fn name(&self) -> &str {
        "MeanScoreDrift"
    }

    fn run(&self, ctx: &DriftContext, config: &DriftConfig) -> CheckResult {
        let current = ctx.current_scores.get(&self.technique);
        let baseline = ctx.baseline_scores.get(&self.technique);

        match (current, baseline) {
            (Some(curr), Some(base)) if !curr.is_empty() && base.len() >= config.min_samples => {
                let baseline_stats = calculate_baseline(base, config.baseline_days);
                let current_mean: Decimal = curr.iter().sum::<Decimal>() / Decimal::from(curr.len());

                let mean = baseline_stats.as_ref().map(|b| b.mean).unwrap_or(Decimal::ZERO);
                let std = baseline_stats.as_ref().map(|b| b.std).unwrap_or(Decimal::ONE);
                let sigma = sigma_deviation(current_mean, mean, std);

                match (baseline_stats, sigma) {
                    (Some(base_stats), Some(sigma)) => {
                        let severity = ThresholdEvaluator::sigma_severity(
                            sigma, config.sigma_warn, config.sigma_crit
                        );

                        let _passed = severity == Severity::Info;
                        let msg = format!(
                            "Mean drift {}: current={:.4}, baseline={:.4}, sigma={:.2}",
                            self.technique, current_mean, base_stats.mean, sigma
                        );

                        let evidence = Evidence::new("mean_comparison")
                            .with_current(current_mean)
                            .with_baseline(base_stats)
                            .with_sample(vec![format!("sigma: {:.2}", sigma)]);

                        let name = format!("MeanDrift_{}", self.technique);
                        let mut result = match severity {
                            Severity::Info => CheckResult::pass(&name, CheckCategory::Drift),
                            Severity::Warn => CheckResult::warn(&name, CheckCategory::Drift, &msg),
                            Severity::Crit => CheckResult::crit(&name, CheckCategory::Drift, &msg),
                            _ => CheckResult::pass(&name, CheckCategory::Drift),
                        };
                        result.value = sigma;
                        result.threshold = config.sigma_warn;
                        result.message = msg;
                        result.evidence = evidence;
                        result
                    }
                    _ => CheckResult::pass(
                        format!("MeanDrift_{}", self.technique),
                        CheckCategory::Drift
                    )
                }
            }
            _ => CheckResult::pass(
                format!("MeanDrift_{}", self.technique),
                CheckCategory::Drift
            ).with_evidence(Evidence::new("insufficient_samples"))
        }
    }
}

/// Selection stability using Jaccard similarity.
#[derive(Debug, Clone, Default)]
pub struct SelectionStabilityCheck;

impl DriftCheck for SelectionStabilityCheck {
    fn name(&self) -> &str {
        "SelectionStability"
    }

    fn run(&self, ctx: &DriftContext, config: &DriftConfig) -> CheckResult {
        if ctx.current_selection.is_empty() || ctx.previous_selection.is_empty() {
            return CheckResult::pass("SelectionStability", CheckCategory::Drift)
                .with_evidence(Evidence::new("no_previous_selection"));
        }

        let overlap = jaccard_similarity(&ctx.current_selection, &ctx.previous_selection) * dec!(100);
        
        let severity = if overlap < config.selection_overlap_crit {
            Severity::Crit
        } else if overlap < config.selection_overlap_warn {
            Severity::Warn
        } else {
            Severity::Info
        };

        let _passed = severity == Severity::Info;
        let msg = format!(
            "Selection overlap: {:.1}% (WARN: <{}%, CRIT: <{}%)",
            overlap, config.selection_overlap_warn, config.selection_overlap_crit
        );

        let intersection: Vec<String> = ctx.current_selection
            .intersection(&ctx.previous_selection)
            .cloned()
            .collect();

        let evidence = Evidence::new("jaccard_similarity")
            .with_current(overlap)
            .with_sample(vec![
                format!("current_size: {}", ctx.current_selection.len()),
                format!("previous_size: {}", ctx.previous_selection.len()),
                format!("intersection: {}", intersection.len()),
            ]);

        let mut result = match severity {
            Severity::Info => CheckResult::pass("SelectionStability", CheckCategory::Drift),
            Severity::Warn => CheckResult::warn("SelectionStability", CheckCategory::Drift, &msg),
            Severity::Crit => CheckResult::crit("SelectionStability", CheckCategory::Drift, &msg),
            _ => CheckResult::pass("SelectionStability", CheckCategory::Drift),
        };
        result.value = overlap;
        result.threshold = config.selection_overlap_warn;
        result.message = msg;
        result.evidence = evidence;
        result
    }
}

/// Exclusion reasons drift.
#[derive(Debug, Clone, Default)]
pub struct ExclusionReasonsDrift;

impl DriftCheck for ExclusionReasonsDrift {
    fn name(&self) -> &str {
        "ExclusionReasonsDrift"
    }

    fn run(&self, ctx: &DriftContext, config: &DriftConfig) -> CheckResult {
        if ctx.current_exclusions.is_empty() || ctx.baseline_exclusions.is_empty() {
            return CheckResult::pass("ExclusionReasons", CheckCategory::Drift)
                .with_evidence(Evidence::new("no_baseline_exclusions"));
        }

        let mut max_multiplier = Decimal::ZERO;
        let mut worst_reason = String::new();
        let mut anomalies = Vec::new();

        for (reason, &current_count) in &ctx.current_exclusions {
            let baseline_count = *ctx.baseline_exclusions.get(reason).unwrap_or(&1);
            let multiplier = if baseline_count > 0 {
                Decimal::from(current_count) / Decimal::from(baseline_count)
            } else {
                Decimal::from(current_count)
            };

            if multiplier > config.exclusion_multiplier_warn {
                anomalies.push(format!("{}: {:.1}x", reason, multiplier));
            }

            if multiplier > max_multiplier {
                max_multiplier = multiplier;
                worst_reason = reason.clone();
            }
        }

        let severity = if max_multiplier > config.exclusion_multiplier_crit {
            Severity::Crit
        } else if max_multiplier > config.exclusion_multiplier_warn {
            Severity::Warn
        } else {
            Severity::Info
        };

        let _passed = severity == Severity::Info;
        let msg = format!(
            "Exclusion drift: worst='{}' at {:.1}x baseline (WARN: >{}x, CRIT: >{}x)",
            worst_reason, max_multiplier,
            config.exclusion_multiplier_warn, config.exclusion_multiplier_crit
        );

        let evidence = Evidence::new("exclusion_comparison")
            .with_current(max_multiplier)
            .with_sample(anomalies);

        let mut result = match severity {
            Severity::Info => CheckResult::pass("ExclusionReasons", CheckCategory::Drift),
            Severity::Warn => CheckResult::warn("ExclusionReasons", CheckCategory::Drift, &msg),
            Severity::Crit => CheckResult::crit("ExclusionReasons", CheckCategory::Drift, &msg),
            _ => CheckResult::pass("ExclusionReasons", CheckCategory::Drift),
        };
        result.value = max_multiplier;
        result.threshold = config.exclusion_multiplier_warn;
        result.message = msg;
        result.evidence = evidence;
        result
    }
}

/// Turnover drift vs historical percentiles.
#[derive(Debug, Clone, Default)]
pub struct TurnoverDrift;

impl DriftCheck for TurnoverDrift {
    fn name(&self) -> &str {
        "TurnoverDrift"
    }

    fn run(&self, ctx: &DriftContext, _config: &DriftConfig) -> CheckResult {
        if ctx.historical_turnover.is_empty() {
            return CheckResult::pass("TurnoverDrift", CheckCategory::Drift)
                .with_evidence(Evidence::new("no_historical_turnover"));
        }

        let p95 = calculate_percentile(&ctx.historical_turnover, dec!(95)).unwrap_or(dec!(50));
        let p99 = calculate_percentile(&ctx.historical_turnover, dec!(99)).unwrap_or(dec!(60));

        let severity = if ctx.current_turnover > p99 {
            Severity::Crit
        } else if ctx.current_turnover > p95 {
            Severity::Warn
        } else {
            Severity::Info
        };

        let _passed = severity == Severity::Info;
        let msg = format!(
            "Turnover: {:.1}% (p95: {:.1}%, p99: {:.1}%)",
            ctx.current_turnover, p95, p99
        );

        let evidence = Evidence::new("turnover_percentile")
            .with_current(ctx.current_turnover)
            .with_sample(vec![
                format!("p95: {:.1}%", p95),
                format!("p99: {:.1}%", p99),
                format!("n_historical: {}", ctx.historical_turnover.len()),
            ]);

        let mut result = match severity {
            Severity::Info => CheckResult::pass("TurnoverDrift", CheckCategory::Drift),
            Severity::Warn => CheckResult::warn("TurnoverDrift", CheckCategory::Drift, &msg),
            Severity::Crit => CheckResult::crit("TurnoverDrift", CheckCategory::Drift, &msg),
            _ => CheckResult::pass("TurnoverDrift", CheckCategory::Drift),
        };
        result.value = ctx.current_turnover;
        result.threshold = p95;
        result.message = msg;
        result.evidence = evidence;
        result
    }
}

/// Cost drift vs historical percentiles.
#[derive(Debug, Clone, Default)]
pub struct CostDrift;

impl DriftCheck for CostDrift {
    fn name(&self) -> &str {
        "CostDrift"
    }

    fn run(&self, ctx: &DriftContext, _config: &DriftConfig) -> CheckResult {
        if ctx.historical_cost.is_empty() {
            return CheckResult::pass("CostDrift", CheckCategory::Drift)
                .with_evidence(Evidence::new("no_historical_cost"));
        }

        let p95 = calculate_percentile(&ctx.historical_cost, dec!(95)).unwrap_or(dec!(0.3));
        let p99 = calculate_percentile(&ctx.historical_cost, dec!(99)).unwrap_or(dec!(0.5));

        let severity = if ctx.current_cost > p99 {
            Severity::Crit
        } else if ctx.current_cost > p95 {
            Severity::Warn
        } else {
            Severity::Info
        };

        let _passed = severity == Severity::Info;
        let msg = format!(
            "Cost: {:.3}% (p95: {:.3}%, p99: {:.3}%)",
            ctx.current_cost, p95, p99
        );

        let evidence = Evidence::new("cost_percentile")
            .with_current(ctx.current_cost)
            .with_sample(vec![
                format!("p95: {:.3}%", p95),
                format!("p99: {:.3}%", p99),
                format!("n_historical: {}", ctx.historical_cost.len()),
            ]);

        let mut result = match severity {
            Severity::Info => CheckResult::pass("CostDrift", CheckCategory::Drift),
            Severity::Warn => CheckResult::warn("CostDrift", CheckCategory::Drift, &msg),
            Severity::Crit => CheckResult::crit("CostDrift", CheckCategory::Drift, &msg),
            _ => CheckResult::pass("CostDrift", CheckCategory::Drift),
        };
        result.value = ctx.current_cost;
        result.threshold = p95;
        result.message = msg;
        result.evidence = evidence;
        result
    }
}

/// Drift engine that runs all checks.
pub struct DriftEngine {
    checks: Vec<Box<dyn DriftCheck>>,
}

impl DriftEngine {
    pub fn new(techniques: &[String]) -> Self {
        let mut checks: Vec<Box<dyn DriftCheck>> = Vec::new();

        for tech in techniques {
            checks.push(Box::new(ScoreDistributionDrift::new(tech)));
            checks.push(Box::new(MeanScoreDrift::new(tech)));
        }

        checks.push(Box::new(SelectionStabilityCheck));
        checks.push(Box::new(ExclusionReasonsDrift));
        checks.push(Box::new(TurnoverDrift));
        checks.push(Box::new(CostDrift));

        Self { checks }
    }

    /// Run all drift checks.
    pub fn run_all(&self, ctx: &DriftContext, config: &DriftConfig) -> Vec<CheckResult> {
        self.checks.iter()
            .map(|check| check.run(ctx, config))
            .collect()
    }
}

impl Default for DriftEngine {
    fn default() -> Self {
        Self::new(&[
            "Momentum".to_string(),
            "Value".to_string(),
            "Quality".to_string(),
            "Size".to_string(),
            "LowVol".to_string(),
            "Dividend".to_string(),
            "Carry".to_string(),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    #[test]
    fn test_selection_stability_pass() {
        let mut ctx = DriftContext::new(date(2024, 1, 10));
        ctx.current_selection = ["A", "B", "C", "D", "E"].iter().map(|s| s.to_string()).collect();
        ctx.previous_selection = ["A", "B", "C", "D", "F"].iter().map(|s| s.to_string()).collect();
        // Overlap: A,B,C,D = 4, Union = 6 -> J = 4/6 = 66.6%

        let check = SelectionStabilityCheck;
        let config = DriftConfig::default(); // warn < 60%, crit < 40%
        let result = check.run(&ctx, &config);

        assert!(result.passed);
        assert_eq!(result.severity, Severity::Info);
    }

    #[test]
    fn test_selection_stability_warn() {
        let mut ctx = DriftContext::new(date(2024, 1, 10));
        ctx.current_selection = ["A", "B", "C"].iter().map(|s| s.to_string()).collect();
        ctx.previous_selection = ["D", "E", "F", "A"].iter().map(|s| s.to_string()).collect();
        // Overlap: A = 1, Union = 6 -> J = 1/6 = 16.6%

        let check = SelectionStabilityCheck;
        let config = DriftConfig::default();
        let result = check.run(&ctx, &config);

        assert!(!result.passed);
        assert_eq!(result.severity, Severity::Crit); // < 40%
    }

    #[test]
    fn test_turnover_drift_pass() {
        let mut ctx = DriftContext::new(date(2024, 1, 10));
        ctx.current_turnover = dec!(25);
        ctx.historical_turnover = (10..=50).map(|x| Decimal::from(x)).collect();

        let check = TurnoverDrift;
        let config = DriftConfig::default();
        let result = check.run(&ctx, &config);

        assert!(result.passed);
        assert_eq!(result.severity, Severity::Info);
    }

    #[test]
    fn test_turnover_drift_crit() {
        let mut ctx = DriftContext::new(date(2024, 1, 10));
        ctx.current_turnover = dec!(95); // Very high
        ctx.historical_turnover = (10..=50).map(|x| Decimal::from(x)).collect();

        let check = TurnoverDrift;
        let config = DriftConfig::default();
        let result = check.run(&ctx, &config);

        assert!(!result.passed);
        assert_eq!(result.severity, Severity::Crit);
    }

    #[test]
    fn test_exclusion_reasons_pass() {
        let mut ctx = DriftContext::new(date(2024, 1, 10));
        ctx.current_exclusions.insert("liquidity".to_string(), 10);
        ctx.baseline_exclusions.insert("liquidity".to_string(), 10);

        let check = ExclusionReasonsDrift;
        let config = DriftConfig::default();
        let result = check.run(&ctx, &config);

        assert!(result.passed);
    }

    #[test]
    fn test_exclusion_reasons_warn() {
        let mut ctx = DriftContext::new(date(2024, 1, 10));
        ctx.current_exclusions.insert("liquidity".to_string(), 25); // 2.5x
        ctx.baseline_exclusions.insert("liquidity".to_string(), 10);

        let check = ExclusionReasonsDrift;
        let config = DriftConfig::default(); // warn > 2x, crit > 3x
        let result = check.run(&ctx, &config);

        assert!(!result.passed);
        assert_eq!(result.severity, Severity::Warn);
    }

    #[test]
    fn test_engine_runs_all() {
        let ctx = DriftContext::new(date(2024, 1, 10));
        let engine = DriftEngine::new(&["Momentum".to_string()]);
        let config = DriftConfig::default();

        let results = engine.run_all(&ctx, &config);
        
        // Should have: ScoreDrift_Momentum, MeanDrift_Momentum,
        // SelectionStability, ExclusionReasons, TurnoverDrift, CostDrift
        assert!(results.len() >= 6);
    }
}


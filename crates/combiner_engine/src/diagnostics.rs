//! Diagnostics module for Stage A vs Stage B analysis.
//!
//! Provides tools for comparing Stage A and Stage B metrics to identify
//! potential issues with validation thresholds, cost assumptions, or overfitting.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use combiner_core::StrategyIdentity;
use crate::engine::FailedCandidate;

/// Comparison of Stage A and Stage B metrics for a single strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageComparison {
    /// Strategy identifier
    pub strategy_id: String,
    /// Full strategy identity
    pub identity: StrategyIdentity,
    
    // Stage A metrics (in-sample fitness evaluation)
    /// Stage A Sharpe ratio
    pub stage_a_sharpe: f64,
    /// Stage A CAGR
    pub stage_a_cagr: f64,
    /// Stage A max drawdown
    pub stage_a_max_dd: f64,
    /// Stage A trade count
    pub stage_a_trades: u32,
    
    // Stage B metrics (out-of-sample validation)
    /// Stage B Sharpe ratio
    pub stage_b_sharpe: f64,
    /// Stage B CAGR
    pub stage_b_cagr: f64,
    /// Stage B max drawdown
    pub stage_b_max_dd: f64,
    /// Stage B trade count
    pub stage_b_trades: u32,
    
    // Deltas
    /// Absolute delta: Stage B - Stage A
    pub delta_sharpe: f64,
    /// Percentage delta: (Stage B - Stage A) / Stage A * 100
    pub delta_sharpe_pct: f64,
    /// Drawdown delta
    pub delta_dd: f64,
    
    // Validation status
    /// Whether the strategy passed Stage B
    pub passed: bool,
    /// List of failure reasons (empty if passed)
    pub failure_reasons: Vec<String>,
}

impl StageComparison {
    /// Create from a FailedCandidate.
    pub fn from_failed(failed: &FailedCandidate) -> Self {
        let stage_a_sharpe = failed.stage_a_sharpe;
        let stage_b_sharpe = failed.stage_b_sharpe;
        
        let delta_sharpe = stage_b_sharpe - stage_a_sharpe;
        let delta_sharpe_pct = if stage_a_sharpe.abs() > 0.001 {
            delta_sharpe / stage_a_sharpe * 100.0
        } else {
            0.0
        };
        
        // Get Stage A fitness data
        let (stage_a_cagr, stage_a_max_dd, stage_a_trades) = failed.genome.fitness.as_ref()
            .map(|f| (f.cagr, f.max_drawdown, f.total_trades))
            .unwrap_or((0.0, 0.0, 0));
        
        Self {
            strategy_id: failed.identity.strategy_id.clone(),
            identity: failed.identity.clone(),
            stage_a_sharpe,
            stage_a_cagr,
            stage_a_max_dd,
            stage_a_trades,
            stage_b_sharpe,
            stage_b_cagr: 0.0, // Not tracked in FailedCandidate
            stage_b_max_dd: failed.stage_b_max_dd,
            stage_b_trades: 0, // Not tracked in FailedCandidate
            delta_sharpe,
            delta_sharpe_pct,
            delta_dd: failed.stage_b_max_dd - stage_a_max_dd,
            passed: false,
            failure_reasons: failed.failure_reasons.clone(),
        }
    }
}

/// Diagnosis of why there's a gap between Stage A and Stage B performance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GapDiagnosis {
    /// BR thresholds are stricter than US thresholds
    ThresholdsTooStrict,
    /// Different market regime/volatility between periods
    RegimeVolatility,
    /// Cost assumptions differ between Stage A and Stage B
    CostMismatch,
    /// Data quality issues (survivorship bias, gaps, etc.)
    DataInconsistency,
    /// Stage A is overly optimistic (overfitting)
    OverfittingStageA,
    /// Multiple factors contribute
    MultipleCauses,
    /// Unable to determine cause
    Unknown,
}

impl std::fmt::Display for GapDiagnosis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GapDiagnosis::ThresholdsTooStrict => write!(f, "Thresholds too strict"),
            GapDiagnosis::RegimeVolatility => write!(f, "Regime/volatility difference"),
            GapDiagnosis::CostMismatch => write!(f, "Cost assumption mismatch"),
            GapDiagnosis::DataInconsistency => write!(f, "Data inconsistency"),
            GapDiagnosis::OverfittingStageA => write!(f, "Stage A overfitting"),
            GapDiagnosis::MultipleCauses => write!(f, "Multiple causes"),
            GapDiagnosis::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Simple histogram for distribution analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Histogram {
    /// Bin edges
    pub bins: Vec<f64>,
    /// Counts per bin
    pub counts: Vec<usize>,
    /// Total count
    pub total: usize,
    /// Min value
    pub min: f64,
    /// Max value
    pub max: f64,
    /// Mean value
    pub mean: f64,
    /// Median value
    pub median: f64,
}

impl Histogram {
    /// Create from a vector of values.
    pub fn from_values(values: &[f64], num_bins: usize) -> Self {
        if values.is_empty() {
            return Self {
                bins: vec![],
                counts: vec![],
                total: 0,
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                median: 0.0,
            };
        }
        
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;
        let bin_width = if range > 0.0 { range / num_bins as f64 } else { 1.0 };
        
        let mut bins: Vec<f64> = (0..=num_bins)
            .map(|i| min + i as f64 * bin_width)
            .collect();
        
        // Ensure last bin includes max
        if let Some(last) = bins.last_mut() {
            *last = max + 0.001;
        }
        
        let mut counts = vec![0usize; num_bins];
        for &v in values {
            let idx = ((v - min) / bin_width).floor() as usize;
            let idx = idx.min(num_bins - 1);
            counts[idx] += 1;
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };
        
        Self {
            bins,
            counts,
            total: values.len(),
            min,
            max,
            mean,
            median,
        }
    }
}

/// Aggregated diagnostic report for a market (typically BR).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDiagnosticReport {
    /// Market identifier ("BR" or "US")
    pub market: String,
    /// Total strategies analyzed
    pub total_strategies: usize,
    /// Strategies that passed Stage B
    pub passed_count: usize,
    /// Strategies that failed Stage B
    pub failed_count: usize,
    /// Pass rate percentage
    pub pass_rate: f64,
    
    // Top 5 failure reasons
    /// Reason, count, and percentage
    pub top_failure_reasons: Vec<(String, usize, f64)>,
    
    // Distributions
    /// Stage A Sharpe distribution
    pub stage_a_sharpe_dist: Histogram,
    /// Stage B Sharpe distribution
    pub stage_b_sharpe_dist: Histogram,
    /// Sharpe degradation distribution
    pub degradation_dist: Histogram,
    
    // Near-miss strategies (closest to passing)
    /// Top 10 strategies that almost passed
    pub near_miss_strategies: Vec<StageComparison>,
    
    // Diagnosis
    /// Overall gap diagnosis
    pub gap_diagnosis: GapDiagnosis,
    /// Detailed diagnosis notes
    pub diagnosis_notes: Vec<String>,
}

impl MarketDiagnosticReport {
    /// Generate report from failed candidates.
    pub fn from_failed_candidates(market: &str, failed: &[FailedCandidate]) -> Self {
        let comparisons: Vec<StageComparison> = failed.iter()
            .map(StageComparison::from_failed)
            .collect();
        
        let total_strategies = comparisons.len();
        let passed_count = 0; // All are failed candidates
        let failed_count = total_strategies;
        let pass_rate = 0.0;
        
        // Count failure reasons
        let mut reason_counts: HashMap<String, usize> = HashMap::new();
        for comp in &comparisons {
            for reason in &comp.failure_reasons {
                // Normalize reason to category
                let category = Self::categorize_failure(reason);
                *reason_counts.entry(category).or_insert(0) += 1;
            }
        }
        
        let mut top_failure_reasons: Vec<_> = reason_counts.into_iter()
            .map(|(reason, count)| {
                let pct = if total_strategies > 0 {
                    count as f64 / total_strategies as f64 * 100.0
                } else {
                    0.0
                };
                (reason, count, pct)
            })
            .collect();
        top_failure_reasons.sort_by(|a, b| b.1.cmp(&a.1));
        top_failure_reasons.truncate(5);
        
        // Build distributions
        let stage_a_sharpes: Vec<f64> = comparisons.iter()
            .map(|c| c.stage_a_sharpe)
            .collect();
        let stage_b_sharpes: Vec<f64> = comparisons.iter()
            .map(|c| c.stage_b_sharpe)
            .collect();
        let degradations: Vec<f64> = comparisons.iter()
            .map(|c| c.delta_sharpe_pct)
            .collect();
        
        let stage_a_sharpe_dist = Histogram::from_values(&stage_a_sharpes, 10);
        let stage_b_sharpe_dist = Histogram::from_values(&stage_b_sharpes, 10);
        let degradation_dist = Histogram::from_values(&degradations, 10);
        
        // Find near-miss strategies (sorted by how close to passing on sharpe)
        let mut near_miss: Vec<StageComparison> = comparisons.clone();
        near_miss.sort_by(|a, b| {
            b.stage_b_sharpe.partial_cmp(&a.stage_b_sharpe).unwrap_or(std::cmp::Ordering::Equal)
        });
        near_miss.truncate(10);
        
        // Diagnose the gap
        let (gap_diagnosis, diagnosis_notes) = Self::diagnose_gap(&comparisons, &top_failure_reasons);
        
        Self {
            market: market.to_string(),
            total_strategies,
            passed_count,
            failed_count,
            pass_rate,
            top_failure_reasons,
            stage_a_sharpe_dist,
            stage_b_sharpe_dist,
            degradation_dist,
            near_miss_strategies: near_miss,
            gap_diagnosis,
            diagnosis_notes,
        }
    }
    
    /// Categorize a failure reason to a standard category.
    fn categorize_failure(reason: &str) -> String {
        if reason.contains("sharpe") {
            "Sharpe too low".to_string()
        } else if reason.contains("pbo") {
            "PBO too high".to_string()
        } else if reason.contains("degrad") {
            "Degradation too high".to_string()
        } else if reason.contains("dd") {
            "Drawdown too deep".to_string()
        } else {
            reason.to_string()
        }
    }
    
    /// Diagnose the gap between Stage A and Stage B.
    fn diagnose_gap(
        comparisons: &[StageComparison],
        top_reasons: &[(String, usize, f64)],
    ) -> (GapDiagnosis, Vec<String>) {
        let mut notes = Vec::new();
        
        if comparisons.is_empty() {
            return (GapDiagnosis::Unknown, vec!["No data to analyze".to_string()]);
        }
        
        // Analyze average degradation
        let avg_degradation: f64 = comparisons.iter()
            .map(|c| c.delta_sharpe_pct)
            .sum::<f64>() / comparisons.len() as f64;
        
        notes.push(format!("Average Sharpe degradation: {:.1}%", avg_degradation));
        
        // Check primary failure reason
        let primary_reason = top_reasons.first().map(|(r, _, _)| r.as_str()).unwrap_or("");
        
        let diagnosis = if avg_degradation < -50.0 {
            notes.push("Severe degradation suggests Stage A overfitting".to_string());
            GapDiagnosis::OverfittingStageA
        } else if primary_reason.contains("Sharpe") && avg_degradation > -30.0 {
            notes.push("Moderate degradation with Sharpe failures suggests thresholds may be too strict".to_string());
            GapDiagnosis::ThresholdsTooStrict
        } else if primary_reason.contains("PBO") {
            notes.push("High PBO failures indicate strategies are overfitting to specific data patterns".to_string());
            GapDiagnosis::OverfittingStageA
        } else if primary_reason.contains("Drawdown") {
            notes.push("Drawdown failures may indicate regime/volatility differences in test periods".to_string());
            GapDiagnosis::RegimeVolatility
        } else if top_reasons.len() > 2 && top_reasons[0].2 < 40.0 {
            notes.push("Multiple failure types with no dominant cause".to_string());
            GapDiagnosis::MultipleCauses
        } else {
            GapDiagnosis::Unknown
        };
        
        (diagnosis, notes)
    }
}

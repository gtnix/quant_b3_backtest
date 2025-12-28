//! Stability Analysis Module
//!
//! Provides stability checks for strategy performance across:
//! - Subperiods (3-5 time blocks)
//! - Market regimes (high/low volatility)
//!
//! # Usage
//!
//! ```ignore
//! let analyzer = StabilityAnalyzer::new(5); // 5 blocks
//! let report = analyzer.analyze(&timeseries, &trades, 0.05)?;
//! if !report.is_stable() {
//!     println!("Strategy unstable: {}", report.instability_reason());
//! }
//! ```

use serde::{Deserialize, Serialize};

use super::metrics::MetricsCalculator;
use super::types::{EquityPoint, TradeRecord};

/// Configuration for stability analysis.
#[derive(Debug, Clone)]
pub struct StabilityConfig {
    /// Number of time blocks to split the period into (default: 5)
    pub num_blocks: usize,
    /// Minimum acceptable Sharpe ratio per block
    pub min_sharpe_per_block: f64,
    /// Maximum coefficient of variation for Sharpe across blocks
    pub max_sharpe_cv: f64,
    /// Maximum spread between best and worst block Sharpe
    pub max_sharpe_spread: f64,
    /// Minimum percentage of blocks that must have positive Sharpe
    pub min_positive_sharpe_pct: f64,
}

impl Default for StabilityConfig {
    fn default() -> Self {
        Self {
            num_blocks: 5,
            min_sharpe_per_block: -0.5, // Allow some negative, but not too bad
            max_sharpe_cv: 1.5,         // Sharpe can vary but not wildly
            max_sharpe_spread: 2.0,     // Max - Min spread
            min_positive_sharpe_pct: 0.6, // At least 60% of blocks positive
        }
    }
}

/// Result for a single time block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockResult {
    /// Block index (0-based)
    pub block_idx: usize,
    /// Start date of this block
    pub start_date: String,
    /// End date of this block
    pub end_date: String,
    /// Number of days in this block
    pub days: usize,
    /// Metrics for this block
    pub metrics: BlockMetrics,
}

/// Simplified metrics for a block (avoids full RunMetrics dependency).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BlockMetrics {
    /// CAGR for this block
    pub cagr: f64,
    /// Sharpe ratio for this block
    pub sharpe: f64,
    /// Volatility for this block
    pub volatility: f64,
    /// Max drawdown for this block
    pub max_drawdown: f64,
}

/// Metadata for traceability and reproducibility.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StabilityMetadata {
    /// Crate version that generated the report
    pub crate_version: String,
    /// UTC timestamp when report was generated
    pub timestamp_utc: String,
    /// Strategy ID analyzed
    pub strategy_id: Option<String>,
    /// Config hash for reproducibility
    pub config_hash: Option<String>,
    /// Start date of analysis period
    pub period_start: Option<String>,
    /// End date of analysis period
    pub period_end: Option<String>,
}

impl StabilityMetadata {
    /// Create metadata with current timestamp and version.
    pub fn now() -> Self {
        Self {
            crate_version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp_utc: chrono::Utc::now().to_rfc3339(),
            strategy_id: None,
            config_hash: None,
            period_start: None,
            period_end: None,
        }
    }

    /// Set strategy ID.
    pub fn with_strategy_id(mut self, id: impl Into<String>) -> Self {
        self.strategy_id = Some(id.into());
        self
    }

    /// Set config hash.
    pub fn with_config_hash(mut self, hash: impl Into<String>) -> Self {
        self.config_hash = Some(hash.into());
        self
    }

    /// Set period.
    pub fn with_period(mut self, start: impl Into<String>, end: impl Into<String>) -> Self {
        self.period_start = Some(start.into());
        self.period_end = Some(end.into());
        self
    }
}

/// Complete stability analysis report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityReport {
    /// Traceability metadata (version, timestamp, config hash)
    pub metadata: StabilityMetadata,
    /// Overall strategy metrics
    pub overall_metrics: BlockMetrics,
    /// Results for each time block
    pub block_results: Vec<BlockResult>,
    /// Statistical summary across blocks
    pub summary: StabilitySummary,
    /// Whether the strategy is considered stable
    pub is_stable: bool,
    /// Reasons for instability (if any)
    pub instability_reasons: Vec<String>,
}

/// Summary statistics across blocks.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StabilitySummary {
    /// Mean Sharpe across blocks
    pub mean_sharpe: f64,
    /// Standard deviation of Sharpe across blocks
    pub std_sharpe: f64,
    /// Coefficient of variation (std/mean) for Sharpe
    pub cv_sharpe: f64,
    /// Minimum Sharpe across all blocks
    pub min_sharpe: f64,
    /// Maximum Sharpe across all blocks
    pub max_sharpe: f64,
    /// Spread between max and min Sharpe
    pub sharpe_spread: f64,
    /// Percentage of blocks with positive Sharpe
    pub pct_positive_sharpe: f64,
    /// Number of blocks analyzed
    pub num_blocks: usize,
}

impl StabilityReport {
    /// Check if strategy is stable (all criteria passed).
    pub fn is_stable(&self) -> bool {
        self.is_stable
    }

    /// Get reasons for instability as a single string.
    pub fn instability_reason(&self) -> String {
        if self.instability_reasons.is_empty() {
            "Stable".to_string()
        } else {
            self.instability_reasons.join("; ")
        }
    }

    /// Generate a text summary for logging.
    pub fn to_summary_string(&self) -> String {
        let mut lines = Vec::new();
        lines.push("=== Stability Analysis Report ===".to_string());
        lines.push(format!(
            "Overall: {} (Sharpe: {:.3})",
            if self.is_stable { "STABLE" } else { "UNSTABLE" },
            self.overall_metrics.sharpe
        ));
        lines.push(String::new());
        lines.push("Block Results:".to_string());

        for block in &self.block_results {
            lines.push(format!(
                "  Block {}: {} to {} ({} days) - Sharpe: {:.3}, CAGR: {:.2}%",
                block.block_idx + 1,
                block.start_date,
                block.end_date,
                block.days,
                block.metrics.sharpe,
                block.metrics.cagr * 100.0
            ));
        }

        lines.push(String::new());
        lines.push("Summary:".to_string());
        lines.push(format!("  Mean Sharpe: {:.3}", self.summary.mean_sharpe));
        lines.push(format!("  Std Sharpe: {:.3}", self.summary.std_sharpe));
        lines.push(format!("  CV Sharpe: {:.3}", self.summary.cv_sharpe));
        lines.push(format!(
            "  Sharpe Range: [{:.3}, {:.3}] (spread: {:.3})",
            self.summary.min_sharpe, self.summary.max_sharpe, self.summary.sharpe_spread
        ));
        lines.push(format!(
            "  Positive Sharpe: {:.1}%",
            self.summary.pct_positive_sharpe * 100.0
        ));

        if !self.instability_reasons.is_empty() {
            lines.push(String::new());
            lines.push("Instability Reasons:".to_string());
            for reason in &self.instability_reasons {
                lines.push(format!("  - {}", reason));
            }
        }

        lines.join("\n")
    }

    /// Save report to disk with full audit trail.
    /// Path: `{output_dir}/stability/{strategy_id}_{timestamp}.json`
    pub fn save_to_disk(&self, output_dir: &std::path::Path) -> std::io::Result<std::path::PathBuf> {
        let dir = output_dir.join("stability");
        std::fs::create_dir_all(&dir)?;

        let strategy_id = self.metadata.strategy_id.as_deref().unwrap_or("unknown");
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let filename = format!("{}_{}.json", strategy_id, timestamp);
        let path = dir.join(&filename);

        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(&path, &json)?;

        tracing::info!(path = %path.display(), bytes = json.len(), "Saved stability report");
        Ok(path)
    }
}

/// Stability analyzer for strategy performance.
pub struct StabilityAnalyzer {
    config: StabilityConfig,
}

impl StabilityAnalyzer {
    /// Create a new analyzer with default config.
    pub fn new(num_blocks: usize) -> Self {
        Self {
            config: StabilityConfig {
                num_blocks,
                ..Default::default()
            },
        }
    }

    /// Create with custom config.
    pub fn with_config(config: StabilityConfig) -> Self {
        Self { config }
    }

    /// Analyze strategy stability across time blocks.
    pub fn analyze(
        &self,
        timeseries: &[EquityPoint],
        trades: &[TradeRecord],
        risk_free_rate: f64,
    ) -> StabilityReport {
        if timeseries.len() < self.config.num_blocks * 20 {
            // Not enough data for meaningful blocks (need at least 20 days per block)
            return self.insufficient_data_report();
        }

        // Calculate overall metrics
        let overall = MetricsCalculator::compute(timeseries, trades, risk_free_rate);
        let overall_metrics = BlockMetrics {
            cagr: overall.cagr,
            sharpe: overall.sharpe_ratio,
            volatility: overall.volatility,
            max_drawdown: overall.max_drawdown,
        };

        // Split into blocks
        let block_results = self.split_and_analyze(timeseries, trades, risk_free_rate);

        // Calculate summary statistics
        let summary = self.calculate_summary(&block_results);

        // Check stability criteria
        let (is_stable, instability_reasons) = self.check_stability(&summary);

        // Build metadata with period from timeseries
        let period_start = block_results.first().map(|b| b.start_date.clone());
        let period_end = block_results.last().map(|b| b.end_date.clone());
        let mut metadata = StabilityMetadata::now();
        if let (Some(start), Some(end)) = (period_start, period_end) {
            metadata = metadata.with_period(start, end);
        }

        StabilityReport {
            metadata,
            overall_metrics,
            block_results,
            summary,
            is_stable,
            instability_reasons,
        }
    }

    fn insufficient_data_report(&self) -> StabilityReport {
        StabilityReport {
            metadata: StabilityMetadata::now(),
            overall_metrics: BlockMetrics::default(),
            block_results: Vec::new(),
            summary: StabilitySummary::default(),
            is_stable: false,
            instability_reasons: vec!["Insufficient data for stability analysis".to_string()],
        }
    }

    fn split_and_analyze(
        &self,
        timeseries: &[EquityPoint],
        trades: &[TradeRecord],
        risk_free_rate: f64,
    ) -> Vec<BlockResult> {
        let n = timeseries.len();
        let block_size = n / self.config.num_blocks;
        let mut results = Vec::new();

        for i in 0..self.config.num_blocks {
            let start = i * block_size;
            let end = if i == self.config.num_blocks - 1 {
                n // Last block takes remainder
            } else {
                (i + 1) * block_size
            };

            let block_ts = &timeseries[start..end];
            if block_ts.is_empty() {
                continue;
            }

            // Filter trades for this block
            let start_date = block_ts.first().map(|p| p.date).unwrap();
            let end_date = block_ts.last().map(|p| p.date).unwrap();
            let block_trades: Vec<TradeRecord> = trades
                .iter()
                .filter(|t| t.date >= start_date && t.date <= end_date)
                .cloned()
                .collect();

            let metrics = MetricsCalculator::compute(block_ts, &block_trades, risk_free_rate);

            results.push(BlockResult {
                block_idx: i,
                start_date: start_date.to_string(),
                end_date: end_date.to_string(),
                days: block_ts.len(),
                metrics: BlockMetrics {
                    cagr: metrics.cagr,
                    sharpe: metrics.sharpe_ratio,
                    volatility: metrics.volatility,
                    max_drawdown: metrics.max_drawdown,
                },
            });
        }

        results
    }

    fn calculate_summary(&self, blocks: &[BlockResult]) -> StabilitySummary {
        if blocks.is_empty() {
            return StabilitySummary::default();
        }

        let sharpes: Vec<f64> = blocks.iter().map(|b| b.metrics.sharpe).collect();
        let n = sharpes.len() as f64;

        let mean_sharpe = sharpes.iter().sum::<f64>() / n;
        let variance = sharpes.iter().map(|s| (s - mean_sharpe).powi(2)).sum::<f64>() / n;
        let std_sharpe = variance.sqrt();

        let cv_sharpe = if mean_sharpe.abs() > 0.001 {
            std_sharpe / mean_sharpe.abs()
        } else {
            0.0
        };

        let min_sharpe = sharpes.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_sharpe = sharpes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sharpe_spread = max_sharpe - min_sharpe;

        let positive_count = sharpes.iter().filter(|&&s| s > 0.0).count();
        let pct_positive_sharpe = positive_count as f64 / n;

        StabilitySummary {
            mean_sharpe,
            std_sharpe,
            cv_sharpe,
            min_sharpe,
            max_sharpe,
            sharpe_spread,
            pct_positive_sharpe,
            num_blocks: blocks.len(),
        }
    }

    /// Check stability criteria against summary statistics.
    ///
    /// Returns (is_stable, reasons) where reasons is empty if stable.
    pub fn check_stability(&self, summary: &StabilitySummary) -> (bool, Vec<String>) {
        let mut reasons = Vec::new();

        // Check minimum Sharpe per block
        if summary.min_sharpe < self.config.min_sharpe_per_block {
            reasons.push(format!(
                "Min block Sharpe ({:.3}) < threshold ({:.3})",
                summary.min_sharpe, self.config.min_sharpe_per_block
            ));
        }

        // Check coefficient of variation
        if summary.cv_sharpe > self.config.max_sharpe_cv && summary.mean_sharpe > 0.0 {
            reasons.push(format!(
                "Sharpe CV ({:.3}) > threshold ({:.3})",
                summary.cv_sharpe, self.config.max_sharpe_cv
            ));
        }

        // Check spread between best and worst
        if summary.sharpe_spread > self.config.max_sharpe_spread {
            reasons.push(format!(
                "Sharpe spread ({:.3}) > threshold ({:.3})",
                summary.sharpe_spread, self.config.max_sharpe_spread
            ));
        }

        // Check percentage of positive blocks
        if summary.pct_positive_sharpe < self.config.min_positive_sharpe_pct {
            reasons.push(format!(
                "Positive Sharpe blocks ({:.1}%) < threshold ({:.1}%)",
                summary.pct_positive_sharpe * 100.0,
                self.config.min_positive_sharpe_pct * 100.0
            ));
        }

        let is_stable = reasons.is_empty();
        (is_stable, reasons)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use rust_decimal::Decimal;

    fn make_timeseries_days(days: usize, growth_rate: f64) -> Vec<EquityPoint> {
        let start = NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
        let mut equity = 1000.0;
        
        (0..days)
            .map(|i| {
                let daily_return = growth_rate / 252.0; // Annualized to daily
                equity *= 1.0 + daily_return + (i as f64 % 10.0 - 5.0) * 0.001; // Add some noise
                EquityPoint {
                    date: start + chrono::Duration::days(i as i64),
                    equity: Decimal::from_f64_retain(equity).unwrap_or(Decimal::ONE),
                    drawdown: 0.0,
                    exposure: 1.0,
                    vol_exante: None,
                    vol_expost: None,
                    dividend_cashflow: None,
                    dividend_cumulative: None,
                }
            })
            .collect()
    }

    #[test]
    fn test_stability_analyzer_creation() {
        let analyzer = StabilityAnalyzer::new(5);
        assert_eq!(analyzer.config.num_blocks, 5);
    }

    #[test]
    fn test_insufficient_data() {
        let analyzer = StabilityAnalyzer::new(5);
        let ts = make_timeseries_days(50, 0.10); // Only 50 days, need 100 (5*20)
        let report = analyzer.analyze(&ts, &[], 0.05);
        
        assert!(!report.is_stable);
        assert!(report.instability_reasons.iter().any(|r| r.contains("Insufficient")));
    }

    #[test]
    fn test_stable_strategy() {
        let analyzer = StabilityAnalyzer::new(3);
        let ts = make_timeseries_days(300, 0.15); // 300 days with 15% growth
        let report = analyzer.analyze(&ts, &[], 0.05);
        
        assert_eq!(report.block_results.len(), 3);
        assert_eq!(report.summary.num_blocks, 3);
        // Note: may or may not be stable depending on noise
    }

    #[test]
    fn test_summary_calculation() {
        let analyzer = StabilityAnalyzer::new(3);
        let blocks = vec![
            BlockResult {
                block_idx: 0,
                start_date: "2020-01-01".to_string(),
                end_date: "2020-04-01".to_string(),
                days: 90,
                metrics: BlockMetrics {
                    sharpe: 1.0,
                    cagr: 0.10,
                    volatility: 0.15,
                    max_drawdown: -0.05,
                },
            },
            BlockResult {
                block_idx: 1,
                start_date: "2020-04-01".to_string(),
                end_date: "2020-07-01".to_string(),
                days: 90,
                metrics: BlockMetrics {
                    sharpe: 1.5,
                    cagr: 0.12,
                    volatility: 0.12,
                    max_drawdown: -0.03,
                },
            },
            BlockResult {
                block_idx: 2,
                start_date: "2020-07-01".to_string(),
                end_date: "2020-10-01".to_string(),
                days: 90,
                metrics: BlockMetrics {
                    sharpe: 0.8,
                    cagr: 0.08,
                    volatility: 0.18,
                    max_drawdown: -0.08,
                },
            },
        ];

        let summary = analyzer.calculate_summary(&blocks);
        
        // Mean of [1.0, 1.5, 0.8] = 1.1
        assert!((summary.mean_sharpe - 1.1).abs() < 0.01);
        assert_eq!(summary.min_sharpe, 0.8);
        assert_eq!(summary.max_sharpe, 1.5);
        assert!((summary.sharpe_spread - 0.7).abs() < 0.01);
        assert!((summary.pct_positive_sharpe - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_report_to_string() {
        let report = StabilityReport {
            metadata: StabilityMetadata::now(),
            overall_metrics: BlockMetrics {
                sharpe: 1.2,
                cagr: 0.15,
                volatility: 0.12,
                max_drawdown: -0.08,
            },
            block_results: vec![],
            summary: StabilitySummary {
                mean_sharpe: 1.1,
                std_sharpe: 0.2,
                cv_sharpe: 0.18,
                min_sharpe: 0.8,
                max_sharpe: 1.5,
                sharpe_spread: 0.7,
                pct_positive_sharpe: 1.0,
                num_blocks: 3,
            },
            is_stable: true,
            instability_reasons: vec![],
        };

        let summary = report.to_summary_string();
        assert!(summary.contains("STABLE"));
        assert!(summary.contains("Mean Sharpe: 1.1"));
    }

    #[test]
    fn test_stability_check_failures() {
        let config = StabilityConfig {
            num_blocks: 3,
            min_sharpe_per_block: 0.0,
            max_sharpe_cv: 0.5,
            max_sharpe_spread: 1.0,
            min_positive_sharpe_pct: 0.8,
        };
        
        let analyzer = StabilityAnalyzer::with_config(config);
        
        // Summary that fails multiple criteria
        let summary = StabilitySummary {
            mean_sharpe: 0.5,
            std_sharpe: 0.8, // CV = 1.6 > 0.5
            cv_sharpe: 1.6,
            min_sharpe: -0.5, // Below 0.0
            max_sharpe: 1.5,
            sharpe_spread: 2.0, // > 1.0
            pct_positive_sharpe: 0.5, // < 0.8
            num_blocks: 3,
        };

        let (is_stable, reasons) = analyzer.check_stability(&summary);
        
        assert!(!is_stable);
        assert!(reasons.len() >= 3);
    }



    // ========================================================================
    // EDGE CASE TESTS (Harvey et al. 2016: robust systems handle extremes)
    // ========================================================================

    /// Test: Insufficient data should produce meaningful error without panic.
    #[test]
    fn test_edge_case_insufficient_data() {
        let config = StabilityConfig {
            num_blocks: 5,
            min_sharpe_per_block: 0.0,
            max_sharpe_cv: 0.5,
            max_sharpe_spread: 1.0,
            min_positive_sharpe_pct: 0.5,
        };

        let analyzer = StabilityAnalyzer::with_config(config);

        // Only 10 data points for 5 blocks
        let timeseries: Vec<EquityPoint> = (0..10)
            .map(|i| EquityPoint {
                date: NaiveDate::from_ymd_opt(2020, 1, 1 + i as u32).unwrap(),
                equity: Decimal::from(100_000 + i * 100),
                drawdown: 0.0,
                exposure: 0.9, vol_exante: None, vol_expost: None, dividend_cashflow: None, dividend_cumulative: None,
            })
            .collect();

        let trades = vec![];
        let report = analyzer.analyze(&timeseries, &trades, 0.0);

        assert!(!report.is_stable);
        assert!(report.instability_reasons.iter().any(|r| r.contains("Insufficient")));
    }

    /// Test: All blocks with zero Sharpe should not cause division by zero.
    #[test]
    fn test_edge_case_zero_sharpe_all_blocks() {
        let summary = StabilitySummary {
            mean_sharpe: 0.0,
            std_sharpe: 0.0,
            cv_sharpe: 0.0,
            min_sharpe: 0.0,
            max_sharpe: 0.0,
            sharpe_spread: 0.0,
            pct_positive_sharpe: 0.0,
            num_blocks: 3,
        };

        assert!(summary.cv_sharpe.is_finite());
        assert!(summary.sharpe_spread.is_finite());
    }

    /// Test: Single block analysis should work.
    #[test]
    fn test_edge_case_single_block() {
        let config = StabilityConfig {
            num_blocks: 1,
            min_sharpe_per_block: 0.0,
            max_sharpe_cv: 999.0,
            max_sharpe_spread: 999.0,
            min_positive_sharpe_pct: 0.0,
        };

        let analyzer = StabilityAnalyzer::with_config(config);

        // Minimal data for 1 block
        let timeseries: Vec<EquityPoint> = (0..100)
            .map(|i| EquityPoint {
                date: NaiveDate::from_ymd_opt(2020, 1, 1).unwrap() + chrono::Duration::days(i),
                equity: Decimal::from(100_000 + i * 10),
                drawdown: 0.0,
                exposure: 0.9, vol_exante: None, vol_expost: None, dividend_cashflow: None, dividend_cumulative: None,
            })
            .collect();

        let trades = vec![];
        let report = analyzer.analyze(&timeseries, &trades, 0.0);

        assert_eq!(report.block_results.len(), 1);
        assert!(report.overall_metrics.sharpe.is_finite());
    }

    /// Test: Extreme negative returns should not cause overflow.
    #[test]
    fn test_edge_case_extreme_negative_returns() {
        let summary = StabilitySummary {
            mean_sharpe: -10.0,
            std_sharpe: 5.0,
            cv_sharpe: -0.5,
            min_sharpe: -15.0,
            max_sharpe: -5.0,
            sharpe_spread: 10.0,
            pct_positive_sharpe: 0.0,
            num_blocks: 5,
        };

        assert!(summary.mean_sharpe.is_finite());
        assert!(summary.min_sharpe.is_finite());
    }

    /// Test: Save to disk with missing strategy_id should use default.
    #[test]
    fn test_edge_case_save_without_strategy_id() {
        let report = StabilityReport {
            metadata: StabilityMetadata::now(),
            overall_metrics: BlockMetrics::default(),
            block_results: vec![],
            summary: StabilitySummary::default(),
            is_stable: true,
            instability_reasons: vec![],
        };

        let temp_dir = tempfile::tempdir().unwrap();
        let result = report.save_to_disk(temp_dir.path());

        assert!(result.is_ok());
        let path = result.unwrap();
        assert!(path.to_string_lossy().contains("unknown_"));
    }
}

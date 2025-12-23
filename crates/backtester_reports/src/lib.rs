//! # Backtester Reports
//!
//! Report generation and result formatting.
//!
//! Responsibilities:
//! - Aggregate final metrics
//! - Generate output files (CSV, JSON)
//! - Create run manifests for audit trail
//! - Hash results for determinism verification
//!
//! Note: This module runs AFTER the simulation loop and is NOT in the hot path.

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

pub use backtester_core::Timestamp;
pub use backtester_portfolio::Portfolio;

/// Summary report of a backtest run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestReport {
    /// Start timestamp of the backtest
    pub start_time: Timestamp,
    /// End timestamp of the backtest
    pub end_time: Timestamp,
    /// Initial capital
    pub initial_capital: f64,
    /// Final NAV
    pub final_nav: f64,
    /// Total return (percentage)
    pub total_return: f64,
    /// Maximum drawdown (percentage)
    pub max_drawdown: f64,
    /// Total number of trades
    pub total_trades: u64,
    /// Total realized PnL
    pub realized_pnl: f64,
    /// Total costs
    pub total_costs: f64,
}

impl BacktestReport {
    /// Generate a report from portfolio state.
    #[must_use]
    pub fn from_portfolio(
        portfolio: &Portfolio,
        start_time: Timestamp,
        end_time: Timestamp,
        total_trades: u64,
    ) -> Self {
        let final_nav = portfolio.nav();
        let total_return = (final_nav - portfolio.initial_capital) / portfolio.initial_capital;

        Self {
            start_time,
            end_time,
            initial_capital: portfolio.initial_capital,
            final_nav,
            total_return,
            max_drawdown: portfolio.max_drawdown,
            total_trades,
            realized_pnl: portfolio.total_realized_pnl(),
            total_costs: portfolio.total_costs,
        }
    }

    /// Convert to JSON string.
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Calculate SHA256 hash of the report for determinism verification.
    #[must_use]
    pub fn hash(&self) -> String {
        let mut hasher = Sha256::new();
        // Use canonical representation for deterministic hashing
        let canonical = format!(
            "nav:{:.8},dd:{:.8},ret:{:.8},trades:{},pnl:{:.8},costs:{:.8}",
            self.final_nav,
            self.max_drawdown,
            self.total_return,
            self.total_trades,
            self.realized_pnl,
            self.total_costs,
        );
        hasher.update(canonical.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

/// Run manifest for audit trail (Module 11).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifest {
    /// Unique run ID
    pub run_id: String,
    /// Run type (build_run, determinism_proof, perf_benchmark, etc.)
    pub run_type: String,
    /// Creation timestamp (ISO 8601)
    pub created_at_utc: String,
    /// Git commit hash (if available)
    pub git_commit: Option<String>,
    /// Build profile (release/debug)
    pub build_profile: String,
    /// Dataset file hash
    pub dataset_signature: String,
    /// Config file hash
    pub config_signature: String,
    /// Strategy identifier
    pub strategy_id: String,
    /// Machine fingerprint
    pub machine_fingerprint: String,
}

impl RunManifest {
    /// Create a new run manifest.
    pub fn new(strategy_name: &str, config_path: &Path, data_path: &Path) -> Self {
        let now: DateTime<Utc> = Utc::now();
        let run_id = format!(
            "{}_build_run_{}_{}",
            now.format("%Y%m%d-%H%M%S"),
            strategy_name,
            &Self::file_hash(config_path)[..8]
        );

        Self {
            run_id,
            run_type: "build_run".to_string(),
            created_at_utc: now.to_rfc3339(),
            git_commit: Self::get_git_commit(),
            build_profile: if cfg!(debug_assertions) {
                "debug".to_string()
            } else {
                "release".to_string()
            },
            dataset_signature: Self::file_hash(data_path),
            config_signature: Self::file_hash(config_path),
            strategy_id: strategy_name.to_string(),
            machine_fingerprint: Self::get_machine_fingerprint(),
        }
    }

    /// Calculate SHA256 hash of a file.
    fn file_hash(path: &Path) -> String {
        match fs::read(path) {
            Ok(content) => {
                let mut hasher = Sha256::new();
                hasher.update(&content);
                format!("{:x}", hasher.finalize())
            }
            Err(_) => "file_not_found".to_string(),
        }
    }

    /// Try to get current git commit hash.
    fn get_git_commit() -> Option<String> {
        std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .ok()
            .and_then(|output| {
                if output.status.success() {
                    String::from_utf8(output.stdout)
                        .ok()
                        .map(|s| s.trim().to_string())
                } else {
                    None
                }
            })
    }

    /// Get machine fingerprint.
    fn get_machine_fingerprint() -> String {
        let os = std::env::consts::OS;
        let arch = std::env::consts::ARCH;
        format!("{}-{}", os, arch)
    }

    /// Calculate result hash for determinism verification.
    #[must_use]
    pub fn calculate_result_hash(&self, result_json: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(result_json.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Save manifest to JSON file.
    pub fn save(&self, path: &Path) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        fs::write(path, json)
    }

    /// Load manifest from JSON file.
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let json = fs::read_to_string(path)?;
        let manifest = serde_json::from_str(&json)?;
        Ok(manifest)
    }
}

/// Benchmark results for performance tracking (Module 06).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Wall clock time in seconds
    pub wall_clock_seconds: f64,
    /// Events processed per second
    pub events_per_second: f64,
    /// Allocations in hot path (should be 0)
    pub hot_path_allocations: u64,
    /// P99 latency in nanoseconds
    pub p99_latency_ns: u64,
}

impl BenchmarkResult {
    /// Create benchmark result from timing data.
    #[must_use]
    pub fn new(
        wall_clock_seconds: f64,
        events_processed: u64,
        hot_path_allocations: u64,
        p99_latency_ns: u64,
    ) -> Self {
        #[allow(clippy::cast_precision_loss)]
        let events_per_second = events_processed as f64 / wall_clock_seconds;
        Self {
            wall_clock_seconds,
            events_per_second,
            hot_path_allocations,
            p99_latency_ns,
        }
    }

    /// Check if benchmark passes performance gates.
    #[must_use]
    pub fn passes_gates(&self) -> bool {
        self.hot_path_allocations == 0
    }
}

/// Chicago-standard performance metrics.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceMetrics {
    /// Sharpe Ratio = (return - risk_free) / volatility (annualized)
    pub sharpe_ratio: f64,
    /// Sortino Ratio = return / downside_volatility
    pub sortino_ratio: f64,
    /// Calmar Ratio = annualized_return / max_drawdown
    pub calmar_ratio: f64,
    /// Win Rate = winning_trades / total_trades
    pub win_rate: f64,
    /// Profit Factor = gross_profit / gross_loss
    pub profit_factor: f64,
    /// Total number of trades
    pub total_trades: u64,
    /// Number of winning trades
    pub winning_trades: u64,
    /// Number of losing trades
    pub losing_trades: u64,
    /// Gross profit
    pub gross_profit: f64,
    /// Gross loss (absolute value)
    pub gross_loss: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Annualized volatility
    pub annualized_volatility: f64,
}

impl PerformanceMetrics {
    /// Calculate metrics from NAV time series and trade data.
    ///
    /// # Arguments
    /// * `nav_series` - Daily NAV values
    /// * `trade_pnls` - PnL for each trade (positive = profit, negative = loss)
    /// * `max_drawdown` - Maximum drawdown as fraction (0.10 = 10%)
    /// * `risk_free_rate` - Annual risk-free rate (0.05 = 5%)
    /// * `trading_days_per_year` - Typically 252
    #[must_use]
    pub fn calculate(
        nav_series: &[f64],
        trade_pnls: &[f64],
        max_drawdown: f64,
        risk_free_rate: f64,
        trading_days_per_year: usize,
    ) -> Self {
        // Calculate returns from NAV series
        let returns: Vec<f64> = nav_series
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        if returns.is_empty() {
            return Self::default();
        }

        // Mean daily return
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;

        // Daily volatility (standard deviation)
        let variance = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / returns.len() as f64;
        let daily_volatility = variance.sqrt();

        // Downside volatility (only negative returns)
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();
        let downside_volatility = if downside_returns.is_empty() {
            0.0
        } else {
            let down_var = downside_returns.iter().map(|r| r.powi(2)).sum::<f64>()
                / downside_returns.len() as f64;
            down_var.sqrt()
        };

        // Annualization
        let annualization_factor = (trading_days_per_year as f64).sqrt();
        let annualized_return = mean_return * trading_days_per_year as f64;
        let annualized_volatility = daily_volatility * annualization_factor;
        let annualized_downside_vol = downside_volatility * annualization_factor;

        // Sharpe Ratio
        let sharpe_ratio = if annualized_volatility > 0.0 {
            (annualized_return - risk_free_rate) / annualized_volatility
        } else {
            0.0
        };

        // Sortino Ratio
        let sortino_ratio = if annualized_downside_vol > 0.0 {
            (annualized_return - risk_free_rate) / annualized_downside_vol
        } else {
            0.0
        };

        // Calmar Ratio
        let calmar_ratio = if max_drawdown > 0.0 {
            annualized_return / max_drawdown
        } else {
            0.0
        };

        // Trade statistics
        let total_trades = trade_pnls.len() as u64;
        let winning_trades = trade_pnls.iter().filter(|&&p| p > 0.0).count() as u64;
        let losing_trades = trade_pnls.iter().filter(|&&p| p < 0.0).count() as u64;
        let gross_profit: f64 = trade_pnls.iter().filter(|&&p| p > 0.0).sum();
        let gross_loss: f64 = trade_pnls.iter().filter(|&&p| p < 0.0).map(|p| p.abs()).sum();

        let win_rate = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64
        } else {
            0.0
        };

        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        Self {
            sharpe_ratio,
            sortino_ratio,
            calmar_ratio,
            win_rate,
            profit_factor,
            total_trades,
            winning_trades,
            losing_trades,
            gross_profit,
            gross_loss,
            annualized_return,
            annualized_volatility,
        }
    }

    /// Create metrics from simple summary data (when full series not available).
    #[must_use]
    pub fn from_summary(
        total_return: f64,
        max_drawdown: f64,
        num_trading_days: usize,
        winning_trades: u64,
        losing_trades: u64,
        gross_profit: f64,
        gross_loss: f64,
    ) -> Self {
        let total_trades = winning_trades + losing_trades;
        let trading_days_per_year = 252;

        // Approximate annualized return
        let years = num_trading_days as f64 / trading_days_per_year as f64;
        let annualized_return = if years > 0.0 {
            ((1.0 + total_return).powf(1.0 / years)) - 1.0
        } else {
            0.0
        };

        let win_rate = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64
        } else {
            0.0
        };

        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let calmar_ratio = if max_drawdown > 0.0 {
            annualized_return / max_drawdown
        } else {
            0.0
        };

        Self {
            sharpe_ratio: 0.0, // Needs volatility
            sortino_ratio: 0.0,
            calmar_ratio,
            win_rate,
            profit_factor,
            total_trades,
            winning_trades,
            losing_trades,
            gross_profit,
            gross_loss,
            annualized_return,
            annualized_volatility: 0.0,
        }
    }
}

/// NAV time series for analytics.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NavHistory {
    /// Timestamps for each NAV observation
    pub timestamps: Vec<Timestamp>,
    /// NAV values
    pub nav_values: Vec<f64>,
    /// Drawdown at each point (as fraction)
    pub drawdowns: Vec<f64>,
    /// Peak NAV at each point
    pub peaks: Vec<f64>,
}

impl NavHistory {
    /// Create empty NAV history with capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            timestamps: Vec::with_capacity(capacity),
            nav_values: Vec::with_capacity(capacity),
            drawdowns: Vec::with_capacity(capacity),
            peaks: Vec::with_capacity(capacity),
        }
    }

    /// Record a NAV observation.
    pub fn record(&mut self, timestamp: Timestamp, nav: f64) {
        let peak = self.peaks.last().copied().unwrap_or(nav).max(nav);
        let drawdown = if peak > 0.0 { (peak - nav) / peak } else { 0.0 };

        self.timestamps.push(timestamp);
        self.nav_values.push(nav);
        self.peaks.push(peak);
        self.drawdowns.push(drawdown);
    }

    /// Get maximum drawdown.
    #[must_use]
    pub fn max_drawdown(&self) -> f64 {
        self.drawdowns.iter().copied().fold(0.0, f64::max)
    }

    /// Get final NAV.
    #[must_use]
    pub fn final_nav(&self) -> f64 {
        self.nav_values.last().copied().unwrap_or(0.0)
    }

    /// Calculate total return.
    #[must_use]
    pub fn total_return(&self) -> f64 {
        if self.nav_values.len() < 2 {
            return 0.0;
        }
        let first = self.nav_values[0];
        let last = self.nav_values.last().copied().unwrap_or(first);
        if first > 0.0 {
            (last - first) / first
        } else {
            0.0
        }
    }

    /// Get length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nav_values.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nav_values.is_empty()
    }

    /// Calculate performance metrics.
    #[must_use]
    pub fn calculate_metrics(&self, trade_pnls: &[f64], risk_free_rate: f64) -> PerformanceMetrics {
        PerformanceMetrics::calculate(
            &self.nav_values,
            trade_pnls,
            self.max_drawdown(),
            risk_free_rate,
            252,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_generation() {
        let portfolio = Portfolio::new(100_000.0, 10);
        let report = BacktestReport::from_portfolio(&portfolio, 0, 1_000_000, 0);
        assert!((report.total_return).abs() < f64::EPSILON);
        assert!((report.initial_capital - 100_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn report_hash_is_deterministic() {
        let portfolio = Portfolio::new(100_000.0, 10);
        let report1 = BacktestReport::from_portfolio(&portfolio, 0, 1_000_000, 5);
        let report2 = BacktestReport::from_portfolio(&portfolio, 0, 1_000_000, 5);
        
        assert_eq!(report1.hash(), report2.hash());
    }

    #[test]
    fn benchmark_result_gates() {
        let good = BenchmarkResult::new(1.0, 1000, 0, 1000);
        assert!(good.passes_gates());

        let bad = BenchmarkResult::new(1.0, 1000, 5, 1000);
        assert!(!bad.passes_gates());
    }

    #[test]
    fn nav_history_tracks_drawdown() {
        let mut nav = NavHistory::with_capacity(10);
        nav.record(0, 100_000.0);
        nav.record(1, 110_000.0); // +10%
        nav.record(2, 99_000.0);  // -10% from peak
        nav.record(3, 105_000.0);

        assert!((nav.max_drawdown() - 0.1).abs() < 0.001);
        assert!((nav.final_nav() - 105_000.0).abs() < f64::EPSILON);
        assert_eq!(nav.len(), 4);
    }

    #[test]
    fn nav_history_total_return() {
        let mut nav = NavHistory::with_capacity(3);
        nav.record(0, 100_000.0);
        nav.record(1, 120_000.0);
        
        assert!((nav.total_return() - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn performance_metrics_win_rate() {
        let trade_pnls = vec![100.0, -50.0, 200.0, -30.0, 150.0];
        let nav_series = vec![100_000.0, 100_100.0, 100_050.0, 100_250.0, 100_220.0, 100_370.0];
        
        let metrics = PerformanceMetrics::calculate(&nav_series, &trade_pnls, 0.01, 0.0, 252);
        
        assert_eq!(metrics.total_trades, 5);
        assert_eq!(metrics.winning_trades, 3);
        assert_eq!(metrics.losing_trades, 2);
        assert!((metrics.win_rate - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn performance_metrics_profit_factor() {
        let trade_pnls = vec![100.0, -50.0, 200.0];
        let nav_series = vec![100_000.0, 100_100.0, 100_050.0, 100_250.0];
        
        let metrics = PerformanceMetrics::calculate(&nav_series, &trade_pnls, 0.01, 0.0, 252);
        
        // Profit factor = 300 / 50 = 6.0
        assert!((metrics.profit_factor - 6.0).abs() < f64::EPSILON);
        assert!((metrics.gross_profit - 300.0).abs() < f64::EPSILON);
        assert!((metrics.gross_loss - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn performance_metrics_calmar() {
        let metrics = PerformanceMetrics::from_summary(
            0.25,  // 25% total return
            0.10,  // 10% max drawdown
            252,   // 1 year
            10,
            5,
            1000.0,
            500.0,
        );
        
        // Calmar = annualized_return / max_drawdown
        // For 1 year, annualized = 25%, calmar = 0.25 / 0.10 = 2.5
        assert!((metrics.calmar_ratio - 2.5).abs() < 0.001);
    }
}

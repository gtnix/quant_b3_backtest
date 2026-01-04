//! # Backtester Reports
//!
//! Report generation and result formatting with Chicago-standard metrics.
//!
//! Responsibilities:
//! - Calculate performance metrics (Sharpe, Sortino, Calmar, etc.)
//! - Track NAV history and drawdowns
//! - Generate output files (CSV, JSON)
//! - Create run manifests for audit trail
//!
//! Note: This module runs AFTER the simulation loop and is NOT in the hot path.
//! Performance: Uses SIMD-optimized calculations from `backtester_core::simd`.

#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

use chrono::{DateTime, Utc};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

use backtester_core::simd;
pub use backtester_portfolio::{Portfolio, Trade};

// =============================================================================
// BACKTEST RESULT (Complete)
// =============================================================================

/// Critical warning in a backtest result.
/// These indicate issues that make the results unreliable.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BacktestWarning {
    /// Strategy executed zero trades. Metrics are artificial and unreliable.
    /// This typically indicates:
    /// - Filter thresholds too restrictive
    /// - Empty universe after gating
    /// - Entry signals never triggered
    ZeroTrades,
    
    /// Very few trades executed (less than minimum for statistical significance).
    /// Metrics may be unreliable due to small sample size.
    LowTradeCount {
        /// Actual trade count
        actual: u32,
        /// Recommended minimum
        recommended_min: u32,
    },
    
    /// Suspiciously high Sharpe ratio that may indicate data issues.
    /// Common causes: survivorship bias, look-ahead bias, overfitting.
    UnrealisticSharpe {
        /// Reported Sharpe ratio
        sharpe: f64,
    },
    
    /// Suspiciously high returns with zero drawdown.
    /// This pattern typically indicates data errors or backtest bugs.
    PerfectEquityCurve,
    
    /// Empty universe was encountered during the backtest.
    EmptyUniverseEncountered {
        /// Number of times empty universe occurred
        occurrences: u32,
    },
}

impl std::fmt::Display for BacktestWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroTrades => write!(f, "CRITICAL: Strategy executed 0 trades. Metrics are artificial and unreliable."),
            Self::LowTradeCount { actual, recommended_min } => 
                write!(f, "WARNING: Only {} trades (recommended min: {}). Results may be unreliable.", actual, recommended_min),
            Self::UnrealisticSharpe { sharpe } => 
                write!(f, "WARNING: Sharpe ratio {:.2} is suspiciously high. Check for bias.", sharpe),
            Self::PerfectEquityCurve => 
                write!(f, "WARNING: Perfect equity curve with 0 drawdown. Likely data error."),
            Self::EmptyUniverseEncountered { occurrences } => 
                write!(f, "WARNING: Empty universe encountered {} times during backtest.", occurrences),
        }
    }
}

/// Complete backtest result with all metrics.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BacktestResult {
    // Basic metrics
    /// Total return (as fraction, e.g., 0.25 = 25%).
    pub total_return: f64,
    /// Annualized return.
    pub annual_return: f64,
    /// Annualized volatility.
    pub annual_volatility: f64,

    // Risk-adjusted metrics
    /// Sharpe Ratio = (return - risk_free) / volatility.
    pub sharpe_ratio: f64,
    /// Sortino Ratio = (return - risk_free) / downside_volatility.
    pub sortino_ratio: f64,
    /// Calmar Ratio = annual_return / max_drawdown.
    pub calmar_ratio: f64,

    // Drawdown metrics
    /// Maximum drawdown (as fraction).
    pub max_drawdown: f64,
    /// Maximum drawdown duration (in bars/days).
    pub max_drawdown_duration: u32,
    /// Current drawdown.
    pub current_drawdown: f64,

    // Trade metrics
    /// Win rate = winning_trades / total_trades.
    pub win_rate: f64,
    /// Profit factor = gross_profit / gross_loss.
    pub profit_factor: f64,
    /// Total number of trades.
    pub num_trades: u32,
    /// Number of winning trades.
    pub num_winning_trades: u32,
    /// Number of losing trades.
    pub num_losing_trades: u32,

    // Average trade metrics
    /// Average trade return (as fraction).
    pub avg_trade_return: f64,
    /// Average winning trade return.
    pub avg_winning_trade: f64,
    /// Average losing trade return.
    pub avg_losing_trade: f64,
    /// Maximum consecutive wins.
    pub max_consecutive_wins: u32,
    /// Maximum consecutive losses.
    pub max_consecutive_losses: u32,

    // PnL
    /// Gross profit.
    pub gross_profit: f64,
    /// Gross loss (as positive value).
    pub gross_loss: f64,
    /// Net profit/loss.
    pub net_pnl: f64,
    /// Total costs (commissions, fees).
    pub total_costs: f64,

    // Portfolio state
    /// Final NAV.
    pub final_nav: f64,
    /// Initial capital.
    pub initial_capital: f64,

    // Events
    /// Total events processed.
    pub events_processed: u64,
    /// Total fills executed.
    pub fills_executed: u64,

    // Validation
    /// Whether this result is considered valid for analysis.
    /// False if critical warnings are present (e.g., zero trades).
    #[serde(default = "default_true")]
    pub is_valid: bool,
    /// Warnings about potential issues with the backtest.
    #[serde(default)]
    pub warnings: Vec<BacktestWarning>,
}

fn default_true() -> bool {
    true
}

impl BacktestResult {
    /// Create result from NAV history and trades.
    #[must_use]
    pub fn calculate(
        nav_history: &NavHistory,
        trades: &[Trade],
        initial_capital: f64,
        total_costs: f64,
        events_processed: u64,
        fills_executed: u64,
        risk_free_rate: f64,
    ) -> Self {
        let mut result = Self::default();
        result.initial_capital = initial_capital;
        result.final_nav = nav_history.final_nav();
        result.total_costs = total_costs;
        result.events_processed = events_processed;
        result.fills_executed = fills_executed;

        // Basic returns
        result.total_return = nav_history.total_return();
        result.max_drawdown = nav_history.max_drawdown();
        result.current_drawdown = nav_history.drawdowns.last().copied().unwrap_or(0.0);
        result.max_drawdown_duration = nav_history.max_drawdown_duration();

        // Calculate volatility and risk-adjusted metrics
        let returns = nav_history.calculate_returns();
        if !returns.is_empty() {
            let (mean, std_dev) = mean_and_std(&returns);
            let annualization_factor = 252.0_f64.sqrt();

            result.annual_return = mean * 252.0;
            result.annual_volatility = std_dev * annualization_factor;

            // Sharpe Ratio (clamped to [-10, 10] to prevent unrealistic values)
            if result.annual_volatility > 0.0 {
                let sharpe_raw = (result.annual_return - risk_free_rate) / result.annual_volatility;
                result.sharpe_ratio = sharpe_raw.clamp(-10.0, 10.0);
            }

            // Sortino Ratio (downside volatility, clamped to [-20, 20])
            let downside_returns: Vec<f64> =
                returns.iter().filter(|&&r| r < 0.0).copied().collect();
            if !downside_returns.is_empty() {
                let downside_vol = std_dev_of(&downside_returns) * annualization_factor;
                if downside_vol > 0.0 {
                    let sortino_raw = (result.annual_return - risk_free_rate) / downside_vol;
                    result.sortino_ratio = sortino_raw.clamp(-20.0, 20.0);
                }
            }

            // Calmar Ratio (clamped to [-20, 20])
            if result.max_drawdown > 0.0 {
                let calmar_raw = result.annual_return / result.max_drawdown;
                result.calmar_ratio = calmar_raw.clamp(-20.0, 20.0);
            }
        }

        // Trade metrics (uses Rayon for parallel computation on large sets)
        result.num_trades = trades.len() as u32;
        if !trades.is_empty() {
            // Parallel partitioning for large trade sets
            let winning: Vec<_>;
            let losing: Vec<_>;

            if trades.len() > 1000 {
                // Use parallel iteration for large trade sets
                let wins: Vec<_> = trades.par_iter().filter(|t| t.net_pnl > 0.0).collect();
                let losses: Vec<_> = trades.par_iter().filter(|t| t.net_pnl <= 0.0).collect();
                winning = wins;
                losing = losses;

                result.gross_profit = trades
                    .par_iter()
                    .filter(|t| t.net_pnl > 0.0)
                    .map(|t| t.net_pnl)
                    .sum();
                result.gross_loss = trades
                    .par_iter()
                    .filter(|t| t.net_pnl <= 0.0)
                    .map(|t| t.net_pnl.abs())
                    .sum();
            } else {
                (winning, losing) = trades.iter().partition(|t| t.net_pnl > 0.0);
                result.gross_profit = winning.iter().map(|t| t.net_pnl).sum();
                result.gross_loss = losing.iter().map(|t| t.net_pnl.abs()).sum();
            }

            result.num_winning_trades = winning.len() as u32;
            result.num_losing_trades = losing.len() as u32;
            result.net_pnl = result.gross_profit - result.gross_loss;

            // Win rate
            result.win_rate = result.num_winning_trades as f64 / result.num_trades as f64;

            // Profit factor
            if result.gross_loss > 0.0 {
                result.profit_factor = result.gross_profit / result.gross_loss;
            } else if result.gross_profit > 0.0 {
                result.profit_factor = f64::INFINITY;
            }

            // Average trade returns (parallel for large sets)
            let trade_returns: Vec<f64> = if trades.len() > 1000 {
                trades.par_iter().map(|t| t.return_pct()).collect()
            } else {
                trades.iter().map(|t| t.return_pct()).collect()
            };
            result.avg_trade_return = simd::simd_mean(&trade_returns);

            if !winning.is_empty() {
                let win_returns: Vec<f64> = winning.iter().map(|t| t.return_pct()).collect();
                result.avg_winning_trade = simd::simd_mean(&win_returns);
            }
            if !losing.is_empty() {
                let loss_returns: Vec<f64> = losing.iter().map(|t| t.return_pct()).collect();
                result.avg_losing_trade = simd::simd_mean(&loss_returns);
            }

            // Consecutive wins/losses (sequential - order matters)
            let (max_wins, max_losses) = max_consecutive_wins_losses(trades);
            result.max_consecutive_wins = max_wins;
            result.max_consecutive_losses = max_losses;
        }

        // Validation and warnings
        result.validate();

        result
    }

    /// Validate the result and add warnings for suspicious patterns.
    fn validate(&mut self) {
        self.is_valid = true;
        self.warnings.clear();

        // CRITICAL: Zero trades makes all metrics meaningless
        if self.num_trades == 0 {
            self.warnings.push(BacktestWarning::ZeroTrades);
            self.is_valid = false;
        } else if self.num_trades < 30 {
            // Statistical significance requires at least 30 trades
            self.warnings.push(BacktestWarning::LowTradeCount {
                actual: self.num_trades,
                recommended_min: 30,
            });
        }

        // Suspiciously high Sharpe (> 3.0 is extremely rare in practice)
        if self.sharpe_ratio > 3.0 && self.num_trades > 0 {
            self.warnings.push(BacktestWarning::UnrealisticSharpe {
                sharpe: self.sharpe_ratio,
            });
        }

        // Perfect equity curve with positive returns and zero drawdown
        if self.total_return > 0.05 && self.max_drawdown < 0.001 && self.num_trades > 10 {
            self.warnings.push(BacktestWarning::PerfectEquityCurve);
            self.is_valid = false;
        }
    }

    /// Convert to JSON string.
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Calculate deterministic hash for verification.
    #[must_use]
    pub fn hash(&self) -> String {
        let mut hasher = Sha256::new();
        let canonical = format!(
            "nav:{:.8},dd:{:.8},ret:{:.8},trades:{},sharpe:{:.8}",
            self.final_nav,
            self.max_drawdown,
            self.total_return,
            self.num_trades,
            self.sharpe_ratio,
        );
        hasher.update(canonical.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

/// Calculate mean and standard deviation using SIMD.
fn mean_and_std(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mean = simd::simd_mean(values);
    let variance = simd::simd_variance(values, mean);
    (mean, variance.sqrt())
}

/// Calculate standard deviation using SIMD.
fn std_dev_of(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = simd::simd_mean(values);
    simd::simd_variance(values, mean).sqrt()
}

/// Calculate max consecutive wins and losses.
fn max_consecutive_wins_losses(trades: &[Trade]) -> (u32, u32) {
    let mut max_wins = 0u32;
    let mut max_losses = 0u32;
    let mut current_wins = 0u32;
    let mut current_losses = 0u32;

    for trade in trades {
        if trade.net_pnl > 0.0 {
            current_wins += 1;
            current_losses = 0;
            max_wins = max_wins.max(current_wins);
        } else if trade.net_pnl < 0.0 {
            current_losses += 1;
            current_wins = 0;
            max_losses = max_losses.max(current_losses);
        }
    }

    (max_wins, max_losses)
}

// =============================================================================
// BACKTEST REPORT (Summary)
// =============================================================================

/// Summary report of a backtest run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestReport {
    /// Start timestamp.
    pub start_time: i64,
    /// End timestamp.
    pub end_time: i64,
    /// Initial capital.
    pub initial_capital: f64,
    /// Final NAV.
    pub final_nav: f64,
    /// Total return.
    pub total_return: f64,
    /// Maximum drawdown.
    pub max_drawdown: f64,
    /// Total trades.
    pub total_trades: u64,
    /// Realized PnL.
    pub realized_pnl: f64,
    /// Total costs.
    pub total_costs: f64,
}

impl BacktestReport {
    /// Generate report from portfolio.
    #[must_use]
    pub fn from_portfolio(
        portfolio: &Portfolio,
        start_time: i64,
        end_time: i64,
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
            max_drawdown: portfolio.max_drawdown(),
            total_trades,
            realized_pnl: portfolio.total_realized_pnl(),
            total_costs: portfolio.total_costs(),
        }
    }

    /// Convert to JSON.
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Calculate hash.
    #[must_use]
    pub fn hash(&self) -> String {
        let mut hasher = Sha256::new();
        let canonical = format!(
            "nav:{:.8},dd:{:.8},ret:{:.8},trades:{},pnl:{:.8}",
            self.final_nav,
            self.max_drawdown,
            self.total_return,
            self.total_trades,
            self.realized_pnl,
        );
        hasher.update(canonical.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

// =============================================================================
// NAV HISTORY
// =============================================================================

/// NAV time series for analytics.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NavHistory {
    /// Timestamps.
    pub timestamps: Vec<i64>,
    /// NAV values.
    pub nav_values: Vec<f64>,
    /// Drawdown at each point.
    pub drawdowns: Vec<f64>,
    /// Peak NAV at each point.
    pub peaks: Vec<f64>,
}

impl NavHistory {
    /// Create with capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            timestamps: Vec::with_capacity(capacity),
            nav_values: Vec::with_capacity(capacity),
            drawdowns: Vec::with_capacity(capacity),
            peaks: Vec::with_capacity(capacity),
        }
    }

    /// Record NAV observation.
    pub fn record(&mut self, timestamp: i64, nav: f64) {
        let peak = self.peaks.last().copied().unwrap_or(nav).max(nav);
        let drawdown = if peak > 0.0 { (peak - nav) / peak } else { 0.0 };

        self.timestamps.push(timestamp);
        self.nav_values.push(nav);
        self.peaks.push(peak);
        self.drawdowns.push(drawdown);
    }

    /// Get maximum drawdown using SIMD when data is large.
    #[must_use]
    pub fn max_drawdown(&self) -> f64 {
        if self.nav_values.len() > 100 {
            // Use SIMD for large datasets
            let (dd, _) = simd::simd_drawdown(&self.nav_values);
            dd
        } else {
            self.drawdowns.iter().copied().fold(0.0, f64::max)
        }
    }

    /// Get maximum drawdown duration (in bars).
    #[must_use]
    pub fn max_drawdown_duration(&self) -> u32 {
        let mut max_duration = 0u32;
        let mut current_duration = 0u32;

        for &dd in &self.drawdowns {
            if dd > 0.0 {
                current_duration += 1;
                max_duration = max_duration.max(current_duration);
            } else {
                current_duration = 0;
            }
        }

        max_duration
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

    /// Calculate returns series using SIMD.
    #[must_use]
    pub fn calculate_returns(&self) -> Vec<f64> {
        if self.nav_values.len() < 2 {
            return Vec::new();
        }
        simd::simd_returns(&self.nav_values)
    }

    /// Calculate Sharpe ratio using SIMD.
    #[must_use]
    pub fn calculate_sharpe(&self, risk_free_rate: f64) -> f64 {
        let returns = self.calculate_returns();
        if returns.is_empty() {
            return 0.0;
        }
        simd::simd_sharpe(&returns, risk_free_rate)
    }

    /// Calculate Sortino ratio using SIMD.
    #[must_use]
    pub fn calculate_sortino(&self, risk_free_rate: f64) -> f64 {
        let returns = self.calculate_returns();
        if returns.is_empty() {
            return 0.0;
        }
        simd::simd_sortino(&returns, risk_free_rate)
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
}

// =============================================================================
// RUN MANIFEST
// =============================================================================

/// Run manifest for audit trail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifest {
    /// Unique run ID.
    pub run_id: String,
    /// Run type.
    pub run_type: String,
    /// Creation timestamp (ISO 8601).
    pub created_at_utc: String,
    /// Git commit hash.
    pub git_commit: Option<String>,
    /// Build profile.
    pub build_profile: String,
    /// Dataset signature.
    pub dataset_signature: String,
    /// Config signature.
    pub config_signature: String,
    /// Strategy ID.
    pub strategy_id: String,
    /// Machine fingerprint.
    pub machine_fingerprint: String,
}

impl RunManifest {
    /// Create new manifest.
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
                "debug"
            } else {
                "release"
            }
            .to_string(),
            dataset_signature: Self::file_hash(data_path),
            config_signature: Self::file_hash(config_path),
            strategy_id: strategy_name.to_string(),
            machine_fingerprint: Self::get_machine_fingerprint(),
        }
    }

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

    fn get_machine_fingerprint() -> String {
        format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
    }

    /// Calculate result hash.
    #[must_use]
    pub fn calculate_result_hash(&self, result_json: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(result_json.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Save to file.
    pub fn save(&self, path: &Path) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        fs::write(path, json)
    }

    /// Load from file.
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let json = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }
}

// =============================================================================
// RESULTS CALCULATOR
// =============================================================================

/// Calculates complete backtest results from portfolio and NAV history.
pub struct ResultsCalculator;

impl ResultsCalculator {
    /// Calculate complete results.
    #[must_use]
    pub fn calculate(
        portfolio: &Portfolio,
        nav_history: &NavHistory,
        events_processed: u64,
        fills_executed: u64,
        risk_free_rate: f64,
    ) -> BacktestResult {
        BacktestResult::calculate(
            nav_history,
            portfolio.get_closed_trades(),
            portfolio.initial_capital,
            portfolio.total_costs(),
            events_processed,
            fills_executed,
            risk_free_rate,
        )
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use backtester_core::{AssetId, OrderDirection};

    fn make_trade(pnl: f64) -> Trade {
        Trade::new(
            AssetId::new(0),
            OrderDirection::Buy,
            100.0,
            100.0 + pnl / 100.0,
            100,
            0,
            1000,
            1.0,
        )
    }

    #[test]
    fn nav_history_tracks_drawdown() {
        let mut nav = NavHistory::with_capacity(10);
        nav.record(0, 100_000.0);
        nav.record(1, 110_000.0);
        nav.record(2, 99_000.0);
        nav.record(3, 105_000.0);

        assert!((nav.max_drawdown() - 0.1).abs() < 0.001);
        assert!((nav.final_nav() - 105_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn nav_history_calculates_sharpe() {
        let mut nav = NavHistory::with_capacity(10);
        for i in 0..100 {
            nav.record(i, 100_000.0 + (i as f64) * 100.0);
        }
        let sharpe = nav.calculate_sharpe(0.0);
        assert!(sharpe > 0.0);
    }

    #[test]
    fn backtest_result_win_rate() {
        let mut nav = NavHistory::with_capacity(10);
        nav.record(0, 100_000.0);
        nav.record(1, 101_000.0);

        let trades = vec![make_trade(100.0), make_trade(-50.0), make_trade(200.0)];

        let result = BacktestResult::calculate(&nav, &trades, 100_000.0, 10.0, 100, 3, 0.0);
        assert_eq!(result.num_trades, 3);
        assert_eq!(result.num_winning_trades, 2);
        assert!((result.win_rate - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn consecutive_wins_losses() {
        let trades = vec![
            make_trade(100.0),
            make_trade(100.0),
            make_trade(100.0),
            make_trade(-50.0),
            make_trade(-50.0),
            make_trade(100.0),
        ];
        let (wins, losses) = max_consecutive_wins_losses(&trades);
        assert_eq!(wins, 3);
        assert_eq!(losses, 2);
    }

    #[test]
    fn drawdown_duration() {
        let mut nav = NavHistory::with_capacity(10);
        nav.record(0, 100_000.0);
        nav.record(1, 100_000.0);
        nav.record(2, 90_000.0); // DD starts
        nav.record(3, 85_000.0);
        nav.record(4, 88_000.0);
        nav.record(5, 100_000.0); // DD ends
        nav.record(6, 100_000.0);

        assert_eq!(nav.max_drawdown_duration(), 3);
    }
}

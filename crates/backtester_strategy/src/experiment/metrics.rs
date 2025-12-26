//! Metrics calculator for experiment performance analysis.
//!
//! Computes key portfolio metrics: CAGR, Sharpe, max drawdown, etc.
//!
//! # Conventions
//!
//! - **Trading days per year**: 252 (used for annualization)
//! - **Return type**: Simple returns `(P_t - P_{t-1}) / P_{t-1}` (not log returns)
//! - **Risk-free rate**: Expected as annualized rate (e.g., 0.05 for 5%)
//! - **Volatility**: Population standard deviation (not sample)
//! - **Drawdown**: Peak-to-trough from high-water mark (HWM)
//!
//! # Infinity Handling
//!
//! Metrics that would return infinity (e.g., Sortino with no negative returns,
//! profit factor with no losses) return a capped maximum value to avoid
//! serialization and comparison issues.

use rust_decimal::Decimal;
use rust_decimal::prelude::*;

use super::types::{EquityPoint, RunMetrics, TradeRecord, TradeSide};

// ============================================================================
// Constants for metric calculations
// ============================================================================

/// Trading days per year (used for annualization).
pub const TRADING_DAYS_PER_YEAR: f64 = 252.0;

/// Default risk-free rate (annualized) if not specified.
pub const DEFAULT_RISK_FREE_RATE: f64 = 0.05;

/// Tolerance for weight sum validation (weights should sum to ~1.0).
pub const WEIGHT_SUM_TOLERANCE: f64 = 0.001;

/// Minimum volatility threshold to avoid division by zero.
pub const MIN_VOLATILITY_THRESHOLD: f64 = 0.0001;

/// Maximum value for ratios that would otherwise be infinity.
/// Used when denominator is zero (e.g., no losses for profit factor).
pub const MAX_RATIO_VALUE: f64 = 999.99;

/// Square root of trading days (precomputed for performance).
const SQRT_TRADING_DAYS: f64 = 15.874507866387544; // sqrt(252)

// ============================================================================
// MetricsCalculator
// ============================================================================

/// Calculator for portfolio performance metrics.
pub struct MetricsCalculator;

impl MetricsCalculator {
    /// Compute all metrics from timeseries and trades.
    pub fn compute(
        timeseries: &[EquityPoint],
        trades: &[TradeRecord],
        risk_free_rate: f64,
    ) -> RunMetrics {
        if timeseries.is_empty() {
            return RunMetrics::default();
        }

        let returns = Self::compute_returns(timeseries);
        let (max_dd, max_dd_days) = Self::max_drawdown(timeseries);

        let start_equity = timeseries.first().map(|p| p.equity).unwrap_or(Decimal::ONE);
        let end_equity = timeseries.last().map(|p| p.equity).unwrap_or(Decimal::ONE);
        let total_days = timeseries.len() as u32;
        let years = total_days as f64 / TRADING_DAYS_PER_YEAR;

        let cagr = Self::cagr(start_equity, end_equity, years);
        let volatility = Self::volatility(&returns);
        let sharpe = Self::sharpe(&returns, risk_free_rate);
        let sortino = Self::sortino(&returns, risk_free_rate);
        let calmar = if max_dd.abs() > 0.0001 {
            cagr / max_dd.abs()
        } else {
            0.0
        };

        let (hit_rate, avg_win, avg_loss, profit_factor, win_loss_ratio) =
            Self::trade_stats(trades);
        let turnover = Self::compute_turnover(trades, timeseries);

        RunMetrics {
            cagr,
            volatility,
            sharpe_ratio: sharpe,
            max_drawdown: max_dd,
            max_drawdown_duration_days: max_dd_days,
            turnover_annual: turnover,
            hit_rate,
            profit_factor,
            total_trades: trades.len() as u32,
            total_days,
            sortino_ratio: sortino,
            calmar_ratio: calmar,
            avg_win,
            avg_loss,
            win_loss_ratio,
        }
    }

    /// Compute daily returns from equity curve.
    pub fn compute_returns(timeseries: &[EquityPoint]) -> Vec<f64> {
        if timeseries.len() < 2 {
            return Vec::new();
        }

        timeseries
            .windows(2)
            .map(|w| {
                let prev = w[0].equity.to_f64().unwrap_or(1.0);
                let curr = w[1].equity.to_f64().unwrap_or(1.0);
                if prev > 0.0 {
                    (curr - prev) / prev
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Compound Annual Growth Rate.
    /// Formula: (end/start)^(1/years) - 1
    pub fn cagr(start: Decimal, end: Decimal, years: f64) -> f64 {
        if start.is_zero() || years <= 0.0 {
            return 0.0;
        }

        let start_f = start.to_f64().unwrap_or(1.0);
        let end_f = end.to_f64().unwrap_or(1.0);

        if start_f <= 0.0 {
            return 0.0;
        }

        let ratio = end_f / start_f;
        if ratio <= 0.0 {
            return -1.0;
        }

        ratio.powf(1.0 / years) - 1.0
    }

    /// Annualized volatility (standard deviation of returns * sqrt(252)).
    ///
    /// Uses population standard deviation (N divisor, not N-1).
    /// Returns 0.0 for empty or single-element series.
    pub fn volatility(returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let n = returns.len() as f64;
        let mean = returns.iter().sum::<f64>() / n;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let daily_vol = variance.sqrt();

        // Annualize using precomputed sqrt(252)
        daily_vol * SQRT_TRADING_DAYS
    }

    /// Sharpe ratio: (annualized return - risk_free) / annualized volatility.
    ///
    /// - `risk_free_rate`: Expected as annualized (e.g., 0.05 for 5%)
    /// - Returns 0.0 if volatility is below threshold to avoid division issues
    pub fn sharpe(returns: &[f64], risk_free_rate: f64) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mean_daily = returns.iter().sum::<f64>() / returns.len() as f64;
        let annualized_return = mean_daily * TRADING_DAYS_PER_YEAR;
        let vol = Self::volatility(returns);

        if vol < MIN_VOLATILITY_THRESHOLD {
            return 0.0;
        }

        (annualized_return - risk_free_rate) / vol
    }

    /// Sortino ratio: (annualized return - risk_free) / downside deviation.
    ///
    /// Downside deviation uses only negative returns for volatility calculation.
    /// Returns `MAX_RATIO_VALUE` if no negative returns (perfect strategy).
    pub fn sortino(returns: &[f64], risk_free_rate: f64) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mean_daily = returns.iter().sum::<f64>() / returns.len() as f64;
        let annualized_return = mean_daily * TRADING_DAYS_PER_YEAR;

        // Downside deviation: std dev of negative returns only
        let negative_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();

        if negative_returns.is_empty() {
            // No negative returns - return capped max value instead of infinity
            return MAX_RATIO_VALUE;
        }

        let n = negative_returns.len() as f64;
        let mean_neg = negative_returns.iter().sum::<f64>() / n;
        let variance = negative_returns
            .iter()
            .map(|r| (r - mean_neg).powi(2))
            .sum::<f64>()
            / n;
        let downside_vol = variance.sqrt() * SQRT_TRADING_DAYS;

        if downside_vol < MIN_VOLATILITY_THRESHOLD {
            return MAX_RATIO_VALUE;
        }

        let ratio = (annualized_return - risk_free_rate) / downside_vol;
        ratio.min(MAX_RATIO_VALUE) // Cap to avoid serialization issues
    }

    /// Maximum drawdown and duration.
    /// Returns (max_drawdown_pct, duration_in_days).
    pub fn max_drawdown(timeseries: &[EquityPoint]) -> (f64, u32) {
        if timeseries.is_empty() {
            return (0.0, 0);
        }

        let mut peak = timeseries[0].equity.to_f64().unwrap_or(1.0);
        let mut max_dd = 0.0;
        let mut max_dd_duration = 0u32;
        let mut current_dd_start = 0usize;
        let mut in_drawdown = false;

        for (i, point) in timeseries.iter().enumerate() {
            let equity = point.equity.to_f64().unwrap_or(1.0);

            if equity > peak {
                peak = equity;
                if in_drawdown {
                    let duration = (i - current_dd_start) as u32;
                    if duration > max_dd_duration {
                        max_dd_duration = duration;
                    }
                    in_drawdown = false;
                }
            } else if peak > 0.0 {
                let dd = (equity - peak) / peak;
                if dd < max_dd {
                    max_dd = dd;
                }
                if !in_drawdown {
                    in_drawdown = true;
                    current_dd_start = i;
                }
            }
        }

        // If still in drawdown at end
        if in_drawdown {
            let duration = (timeseries.len() - current_dd_start) as u32;
            if duration > max_dd_duration {
                max_dd_duration = duration;
            }
        }

        (max_dd, max_dd_duration)
    }

    /// Trade statistics: hit_rate, avg_win, avg_loss, profit_factor, win_loss_ratio.
    ///
    /// - `hit_rate`: Proportion of winning trades (PnL > 0)
    /// - `avg_win`: Average profit on winning trades
    /// - `avg_loss`: Average loss on losing trades (positive value)
    /// - `profit_factor`: Gross profit / gross loss (capped at MAX_RATIO_VALUE)
    /// - `win_loss_ratio`: avg_win / avg_loss (capped at MAX_RATIO_VALUE)
    pub fn trade_stats(trades: &[TradeRecord]) -> (f64, f64, f64, f64, f64) {
        if trades.is_empty() {
            return (0.0, 0.0, 0.0, 0.0, 0.0);
        }

        let mut wins = 0;
        let mut total_win = 0.0;
        let mut total_loss = 0.0;
        let mut win_count = 0;
        let mut loss_count = 0;

        for trade in trades {
            if let Some(pnl) = trade.pnl {
                let pnl_f = pnl.to_f64().unwrap_or(0.0);
                if pnl_f > 0.0 {
                    wins += 1;
                    total_win += pnl_f;
                    win_count += 1;
                } else if pnl_f < 0.0 {
                    total_loss += pnl_f.abs();
                    loss_count += 1;
                }
            }
        }

        let hit_rate = wins as f64 / trades.len() as f64;

        let avg_win = if win_count > 0 {
            total_win / win_count as f64
        } else {
            0.0
        };

        let avg_loss = if loss_count > 0 {
            total_loss / loss_count as f64
        } else {
            0.0
        };

        // Cap ratios at MAX_RATIO_VALUE to avoid infinity
        let profit_factor = if total_loss > 0.0 {
            (total_win / total_loss).min(MAX_RATIO_VALUE)
        } else if total_win > 0.0 {
            MAX_RATIO_VALUE // No losses, cap instead of infinity
        } else {
            0.0
        };

        let win_loss_ratio = if avg_loss > 0.0 {
            (avg_win / avg_loss).min(MAX_RATIO_VALUE)
        } else if avg_win > 0.0 {
            MAX_RATIO_VALUE // No losses, cap instead of infinity
        } else {
            0.0
        };

        (hit_rate, avg_win, avg_loss, profit_factor, win_loss_ratio)
    }

    /// Compute annualized turnover.
    ///
    /// Turnover = (total traded value) / (average portfolio value) / years
    /// Represents how many times the portfolio is "turned over" per year.
    pub fn compute_turnover(trades: &[TradeRecord], timeseries: &[EquityPoint]) -> f64 {
        if trades.is_empty() || timeseries.is_empty() {
            return 0.0;
        }

        // Total traded value (buys + sells)
        let total_traded: f64 = trades
            .iter()
            .map(|t| t.value.to_f64().unwrap_or(0.0).abs())
            .sum();

        // Average portfolio value
        let avg_equity: f64 = timeseries
            .iter()
            .map(|p| p.equity.to_f64().unwrap_or(0.0))
            .sum::<f64>()
            / timeseries.len() as f64;

        if avg_equity < MIN_VOLATILITY_THRESHOLD {
            return 0.0;
        }

        // Turnover = total traded / avg equity
        let raw_turnover = total_traded / avg_equity;

        // Annualize based on period length
        let days = timeseries.len() as f64;
        let years = days / TRADING_DAYS_PER_YEAR;

        if years > 0.0 {
            raw_turnover / years
        } else {
            raw_turnover
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use rust_decimal_macros::dec;

    fn make_timeseries(values: &[f64]) -> Vec<EquityPoint> {
        let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        values
            .iter()
            .enumerate()
            .map(|(i, &v)| EquityPoint {
                date: start + chrono::Duration::days(i as i64),
                equity: Decimal::from_f64(v).unwrap_or(Decimal::ONE),
                drawdown: 0.0,
                exposure: 1.0,
                vol_exante: None,
                vol_expost: None,
            })
            .collect()
    }

    #[test]
    fn test_cagr_positive() {
        // 100 -> 150 over 2 years = 22.47% CAGR
        let cagr = MetricsCalculator::cagr(dec!(100), dec!(150), 2.0);
        assert!((cagr - 0.2247).abs() < 0.01);
    }

    #[test]
    fn test_cagr_negative() {
        // 100 -> 80 over 1 year = -20% CAGR
        let cagr = MetricsCalculator::cagr(dec!(100), dec!(80), 1.0);
        assert!((cagr - (-0.20)).abs() < 0.01);
    }

    #[test]
    fn test_volatility_constant() {
        // Constant returns should have zero volatility
        let returns = vec![0.01, 0.01, 0.01, 0.01, 0.01];
        let vol = MetricsCalculator::volatility(&returns);
        assert!(vol.abs() < 0.001);
    }

    #[test]
    fn test_volatility_variable() {
        // Variable returns should have positive volatility
        let returns = vec![0.02, -0.01, 0.03, -0.02, 0.01];
        let vol = MetricsCalculator::volatility(&returns);
        assert!(vol > 0.0);
    }

    #[test]
    fn test_max_drawdown() {
        // Equity: 100 -> 120 -> 90 -> 110
        // Max DD should be (90-120)/120 = -25%
        let ts = make_timeseries(&[100.0, 120.0, 90.0, 110.0]);
        let (dd, _) = MetricsCalculator::max_drawdown(&ts);
        assert!((dd - (-0.25)).abs() < 0.01);
    }

    #[test]
    fn test_sharpe_positive() {
        // Positive returns with low vol should have positive Sharpe
        let returns = vec![0.001, 0.002, 0.001, 0.002, 0.001]; // ~0.14% daily, ~35% annual
        let sharpe = MetricsCalculator::sharpe(&returns, 0.05);
        assert!(sharpe > 0.0);
    }

    #[test]
    fn test_trade_stats() {
        let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let trades = vec![
            TradeRecord {
                date: start,
                symbol: "A".into(),
                side: TradeSide::Sell,
                quantity: 100,
                price: dec!(10),
                value: dec!(1000),
                pnl: Some(dec!(100)),
            },
            TradeRecord {
                date: start,
                symbol: "B".into(),
                side: TradeSide::Sell,
                quantity: 100,
                price: dec!(10),
                value: dec!(1000),
                pnl: Some(dec!(-50)),
            },
        ];

        let (hit_rate, avg_win, avg_loss, profit_factor, _) =
            MetricsCalculator::trade_stats(&trades);

        assert!((hit_rate - 0.5).abs() < 0.01);
        assert!((avg_win - 100.0).abs() < 0.01);
        assert!((avg_loss - 50.0).abs() < 0.01);
        assert!((profit_factor - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_full_metrics_computation() {
        let ts = make_timeseries(&[100.0, 102.0, 105.0, 103.0, 108.0]);
        let trades = vec![];
        
        let metrics = MetricsCalculator::compute(&ts, &trades, 0.05);
        
        assert!(metrics.cagr != 0.0 || metrics.total_days > 0);
        assert_eq!(metrics.total_days, 5);
    }
}


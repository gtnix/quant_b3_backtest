//! Kelly Criterion Calculator for Dynamic Position Sizing.
//!
//! Implements the Kelly Criterion for optimal bet sizing:
//! f* = (p*W - q*L) / (W*L)
//!
//! References:
//! - Ziemba & MacLean (2011): Kelly Criterion for Investing
//! - Thorp (2006): Kelly Criterion in Stock Market
//! - Vince (1992): Mathematics of Money Management

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// Configuration for Kelly Calculator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KellyConfig {
    /// Minimum number of trades required for calculation.
    #[serde(default = "default_min_trades")]
    pub min_trades: usize,
    
    /// Fractional Kelly multiplier (0.25 = Quarter-Kelly, 0.5 = Half-Kelly).
    /// Academic maximum for practical use is 0.5.
    #[serde(default = "default_kelly_fraction")]
    pub kelly_fraction: f64,
    
    /// Maximum position size cap (as percentage of capital).
    /// Acts as a hard cap regardless of Kelly calculation.
    #[serde(default = "default_max_position_pct")]
    pub max_position_pct: f64,
    
    /// Minimum position size (below this, don't trade).
    #[serde(default = "default_min_position_pct")]
    pub min_position_pct: f64,
    
    /// Use geometric vs arithmetic mean for payoff calculation.
    #[serde(default)]
    pub use_geometric_mean: bool,
}

fn default_min_trades() -> usize { 30 }
fn default_kelly_fraction() -> f64 { 0.25 } // Quarter-Kelly (conservative)
fn default_max_position_pct() -> f64 { 0.02 } // 2% max
fn default_min_position_pct() -> f64 { 0.005 } // 0.5% min

impl Default for KellyConfig {
    fn default() -> Self {
        Self {
            min_trades: default_min_trades(),
            kelly_fraction: default_kelly_fraction(),
            max_position_pct: default_max_position_pct(),
            min_position_pct: default_min_position_pct(),
            use_geometric_mean: false,
        }
    }
}

/// Result of Kelly calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KellyResult {
    /// Raw Kelly fraction (full f*).
    pub full_kelly: f64,
    /// Fractional Kelly (f* * kelly_fraction).
    pub fractional_kelly: f64,
    /// Final position size after caps.
    pub final_position_pct: f64,
    /// Win rate (probability of winning).
    pub win_rate: f64,
    /// Average win (as return percentage).
    pub avg_win: f64,
    /// Average loss (as return percentage, positive value).
    pub avg_loss: f64,
    /// Payoff ratio (avg_win / avg_loss).
    pub payoff_ratio: f64,
    /// Number of trades used in calculation.
    pub n_trades: usize,
    /// Whether result is valid (enough trades, positive expectancy).
    pub is_valid: bool,
    /// Reason if invalid.
    pub invalid_reason: Option<String>,
}

impl Default for KellyResult {
    fn default() -> Self {
        Self {
            full_kelly: 0.0,
            fractional_kelly: 0.0,
            final_position_pct: 0.0,
            win_rate: 0.0,
            avg_win: 0.0,
            avg_loss: 0.0,
            payoff_ratio: 0.0,
            n_trades: 0,
            is_valid: false,
            invalid_reason: Some("No trades".to_string()),
        }
    }
}

/// Trade result for Kelly calculation.
#[derive(Debug, Clone)]
pub struct TradeForKelly {
    /// Return as percentage (e.g., 0.05 for 5% gain, -0.03 for 3% loss).
    pub return_pct: f64,
}

impl TradeForKelly {
    pub fn new(return_pct: f64) -> Self {
        Self { return_pct }
    }
    
    /// Create from entry/exit prices.
    pub fn from_prices(entry: Decimal, exit: Decimal) -> Self {
        let entry_f: f64 = entry.try_into().unwrap_or(1.0);
        let exit_f: f64 = exit.try_into().unwrap_or(1.0);
        let return_pct = if entry_f > 0.0 {
            (exit_f - entry_f) / entry_f
        } else {
            0.0
        };
        Self { return_pct }
    }
    
    pub fn is_win(&self) -> bool {
        self.return_pct > 0.0
    }
}

/// Kelly Criterion Calculator.
///
/// Calculates optimal position sizing based on historical trade performance.
/// Uses Fractional Kelly to reduce variance while maintaining growth.
#[derive(Debug, Clone)]
pub struct KellyCalculator {
    config: KellyConfig,
}

impl KellyCalculator {
    pub fn new(config: KellyConfig) -> Self {
        Self { config }
    }
    
    /// Calculate Kelly fraction from a series of trades.
    ///
    /// Formula: f* = (p*W - q*L) / (W*L)
    /// Where:
    /// - p = probability of winning
    /// - q = probability of losing (1 - p)
    /// - W = average win (as percentage)
    /// - L = average loss (as percentage, positive)
    pub fn calculate(&self, trades: &[TradeForKelly]) -> KellyResult {
        if trades.len() < self.config.min_trades {
            return KellyResult {
                n_trades: trades.len(),
                is_valid: false,
                invalid_reason: Some(format!(
                    "Insufficient trades: {} < {}", 
                    trades.len(), 
                    self.config.min_trades
                )),
                ..Default::default()
            };
        }

        // Separate wins and losses
        let (wins, losses): (Vec<_>, Vec<_>) = trades
            .iter()
            .partition(|t| t.is_win());

        let n_wins = wins.len();
        let n_losses = losses.len();
        let n_total = trades.len();

        if n_wins == 0 {
            return KellyResult {
                n_trades: n_total,
                win_rate: 0.0,
                is_valid: false,
                invalid_reason: Some("No winning trades".to_string()),
                ..Default::default()
            };
        }

        if n_losses == 0 {
            // All wins - can't calculate payoff ratio
            return KellyResult {
                n_trades: n_total,
                win_rate: 1.0,
                is_valid: false,
                invalid_reason: Some("No losing trades (can't calculate payoff)".to_string()),
                ..Default::default()
            };
        }

        // Calculate statistics
        let win_rate = n_wins as f64 / n_total as f64;
        let loss_rate = 1.0 - win_rate;

        let avg_win = wins.iter().map(|t| t.return_pct).sum::<f64>() / n_wins as f64;
        let avg_loss = losses.iter().map(|t| t.return_pct.abs()).sum::<f64>() / n_losses as f64;

        if avg_loss < 1e-10 {
            return KellyResult {
                n_trades: n_total,
                win_rate,
                avg_win,
                avg_loss: 0.0,
                is_valid: false,
                invalid_reason: Some("Average loss is zero".to_string()),
                ..Default::default()
            };
        }

        let payoff_ratio = avg_win / avg_loss;

        // Kelly formula: f* = (p*W - q*L) / (W*L)
        // Simplified: f* = p - q/B where B = W/L (payoff ratio)
        let full_kelly = win_rate - (loss_rate / payoff_ratio);

        if full_kelly <= 0.0 {
            return KellyResult {
                full_kelly,
                fractional_kelly: 0.0,
                final_position_pct: 0.0,
                win_rate,
                avg_win,
                avg_loss,
                payoff_ratio,
                n_trades: n_total,
                is_valid: false,
                invalid_reason: Some("Negative expectancy (Kelly <= 0)".to_string()),
            };
        }

        // Apply fractional Kelly
        let fractional_kelly = full_kelly * self.config.kelly_fraction;

        // Apply caps
        let final_position_pct = fractional_kelly
            .max(self.config.min_position_pct)
            .min(self.config.max_position_pct);

        KellyResult {
            full_kelly,
            fractional_kelly,
            final_position_pct,
            win_rate,
            avg_win,
            avg_loss,
            payoff_ratio,
            n_trades: n_total,
            is_valid: true,
            invalid_reason: None,
        }
    }
    
    /// Calculate from return percentages directly.
    pub fn calculate_from_returns(&self, returns: &[f64]) -> KellyResult {
        let trades: Vec<TradeForKelly> = returns
            .iter()
            .map(|&r| TradeForKelly::new(r))
            .collect();
        self.calculate(&trades)
    }
    
    /// Get the configured kelly fraction multiplier.
    pub fn kelly_fraction(&self) -> f64 {
        self.config.kelly_fraction
    }
    
    /// Get the max position cap.
    pub fn max_position_pct(&self) -> f64 {
        self.config.max_position_pct
    }
}

impl Default for KellyCalculator {
    fn default() -> Self {
        Self::new(KellyConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kelly_basic() {
        let calc = KellyCalculator::new(KellyConfig {
            min_trades: 10,
            kelly_fraction: 0.5, // Half-Kelly
            max_position_pct: 0.10,
            min_position_pct: 0.01,
            ..Default::default()
        });

        // 60% win rate, 2:1 payoff (avg win 10%, avg loss 5%)
        let trades: Vec<TradeForKelly> = vec![
            TradeForKelly::new(0.10),  // win
            TradeForKelly::new(0.12),  // win
            TradeForKelly::new(-0.05), // loss
            TradeForKelly::new(0.08),  // win
            TradeForKelly::new(0.10),  // win
            TradeForKelly::new(-0.04), // loss
            TradeForKelly::new(0.10),  // win
            TradeForKelly::new(-0.06), // loss
            TradeForKelly::new(0.10),  // win
            TradeForKelly::new(-0.05), // loss
        ];

        let result = calc.calculate(&trades);

        assert!(result.is_valid, "Should be valid: {:?}", result.invalid_reason);
        assert!(result.win_rate > 0.5 && result.win_rate < 0.7);
        assert!(result.full_kelly > 0.0);
        assert!(result.fractional_kelly < result.full_kelly);
        assert!(result.final_position_pct >= 0.01);
        assert!(result.final_position_pct <= 0.10);
    }

    #[test]
    fn test_kelly_insufficient_trades() {
        let calc = KellyCalculator::new(KellyConfig {
            min_trades: 30,
            ..Default::default()
        });

        let trades: Vec<TradeForKelly> = vec![
            TradeForKelly::new(0.05),
            TradeForKelly::new(-0.03),
        ];

        let result = calc.calculate(&trades);

        assert!(!result.is_valid);
        assert!(result.invalid_reason.as_ref().unwrap().contains("Insufficient"));
    }

    #[test]
    fn test_kelly_negative_expectancy() {
        let calc = KellyCalculator::new(KellyConfig {
            min_trades: 5,
            ..Default::default()
        });

        // 30% win rate, 1:2 payoff (bad system)
        let trades: Vec<TradeForKelly> = vec![
            TradeForKelly::new(0.02),  // win
            TradeForKelly::new(-0.05), // loss
            TradeForKelly::new(-0.04), // loss
            TradeForKelly::new(-0.05), // loss
            TradeForKelly::new(-0.06), // loss
        ];

        let result = calc.calculate(&trades);

        assert!(!result.is_valid);
        assert!(result.full_kelly <= 0.0);
    }

    #[test]
    fn test_kelly_formula_verification() {
        // Known values: p=0.6, W=0.10, L=0.05
        // f* = p - q/B = 0.6 - 0.4/(0.10/0.05) = 0.6 - 0.4/2 = 0.6 - 0.2 = 0.4
        let calc = KellyCalculator::new(KellyConfig {
            min_trades: 10,
            kelly_fraction: 1.0, // Full Kelly for testing
            max_position_pct: 1.0,
            min_position_pct: 0.0,
            ..Default::default()
        });

        // Create exactly 60% wins with 10% avg win and 5% avg loss
        let trades: Vec<TradeForKelly> = vec![
            TradeForKelly::new(0.10),  // win
            TradeForKelly::new(0.10),  // win
            TradeForKelly::new(0.10),  // win
            TradeForKelly::new(0.10),  // win
            TradeForKelly::new(0.10),  // win
            TradeForKelly::new(0.10),  // win
            TradeForKelly::new(-0.05), // loss
            TradeForKelly::new(-0.05), // loss
            TradeForKelly::new(-0.05), // loss
            TradeForKelly::new(-0.05), // loss
        ];

        let result = calc.calculate(&trades);

        assert!(result.is_valid);
        assert!((result.win_rate - 0.6).abs() < 0.01, "Win rate: {}", result.win_rate);
        assert!((result.avg_win - 0.10).abs() < 0.001, "Avg win: {}", result.avg_win);
        assert!((result.avg_loss - 0.05).abs() < 0.001, "Avg loss: {}", result.avg_loss);
        assert!((result.payoff_ratio - 2.0).abs() < 0.01, "Payoff: {}", result.payoff_ratio);
        assert!((result.full_kelly - 0.4).abs() < 0.01, "Full Kelly: {}", result.full_kelly);
    }

    #[test]
    fn test_kelly_caps() {
        let calc = KellyCalculator::new(KellyConfig {
            min_trades: 4,
            kelly_fraction: 0.5,
            max_position_pct: 0.02, // 2% cap
            min_position_pct: 0.005,
            ..Default::default()
        });

        // High Kelly scenario (70% win, 3:1 payoff)
        let trades: Vec<TradeForKelly> = vec![
            TradeForKelly::new(0.15),
            TradeForKelly::new(0.15),
            TradeForKelly::new(0.15),
            TradeForKelly::new(-0.05),
        ];

        let result = calc.calculate(&trades);

        assert!(result.is_valid);
        assert_eq!(result.final_position_pct, 0.02, "Should be capped at 2%");
    }

    #[test]
    fn test_kelly_from_returns() {
        let calc = KellyCalculator::new(KellyConfig {
            min_trades: 5,
            kelly_fraction: 0.25,
            ..Default::default()
        });

        let returns = vec![0.05, -0.02, 0.03, 0.04, -0.01];
        let result = calc.calculate_from_returns(&returns);

        assert!(result.is_valid);
        assert!(result.n_trades == 5);
    }
}


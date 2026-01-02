//! Asset attribution: calculate PnL by asset.
//!
//! Generates asset_attribution.csv with:
//! - PnL per asset (net and gross)
//! - Number of trades
//! - Win rate
//! - Contribution percentage

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use crate::{ValidationError, ValidationWarning, Verdict};

/// Configuration for attribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributionConfig {
    /// Concentration threshold for warning (single asset > X% of PnL).
    pub concentration_warn_threshold: f64,
    /// Number of top/bottom assets to include in report.
    pub top_n: usize,
}

impl Default for AttributionConfig {
    fn default() -> Self {
        Self {
            concentration_warn_threshold: 0.8, // 80%
            top_n: 10,
        }
    }
}

/// Attribution for a single asset.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AssetAttribution {
    /// Asset symbol.
    pub symbol: String,
    /// Net PnL (after costs).
    pub net_pnl: f64,
    /// Gross PnL (before costs).
    pub gross_pnl: f64,
    /// Total costs for this asset.
    pub total_costs: f64,
    /// Number of trades.
    pub num_trades: u32,
    /// Winning trades.
    pub winning_trades: u32,
    /// Losing trades.
    pub losing_trades: u32,
    /// Win rate.
    pub win_rate: f64,
    /// Average trade PnL.
    pub avg_trade_pnl: f64,
    /// Contribution to total PnL (as fraction).
    pub contribution_pct: f64,
}

/// Result of attribution calculation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AttributionResult {
    /// All asset attributions.
    pub attributions: Vec<AssetAttribution>,
    /// Total net PnL.
    pub total_net_pnl: f64,
    /// Total gross PnL.
    pub total_gross_pnl: f64,
    /// Total trades.
    pub total_trades: u32,
    /// Concentration metrics.
    pub concentration: ConcentrationMetrics,
    /// Warnings generated.
    pub warnings: Vec<ValidationWarning>,
    /// Verdict.
    pub verdict: Verdict,
}

/// Concentration metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConcentrationMetrics {
    /// Top 1 asset contribution.
    pub top_1_pct: f64,
    /// Top 5 assets contribution.
    pub top_5_pct: f64,
    /// Top 10 assets contribution.
    pub top_10_pct: f64,
    /// Herfindahl-Hirschman Index.
    pub hhi: f64,
}

/// A trade record for attribution.
#[derive(Debug, Clone)]
pub struct TradeRecord {
    /// Asset symbol.
    pub symbol: String,
    /// Trade PnL (net).
    pub net_pnl: f64,
    /// Trade PnL (gross).
    pub gross_pnl: f64,
    /// Trade cost.
    pub cost: f64,
}

/// Calculator for asset attribution.
pub struct AttributionCalculator {
    config: AttributionConfig,
}

impl Default for AttributionCalculator {
    fn default() -> Self {
        Self::new(AttributionConfig::default())
    }
}

impl AttributionCalculator {
    /// Create a new calculator.
    pub fn new(config: AttributionConfig) -> Self {
        Self { config }
    }

    /// Load trades from CSV.
    pub fn load_trades(&self, path: &Path) -> Result<Vec<TradeRecord>, ValidationError> {
        if !path.exists() {
            return Ok(Vec::new());
        }

        let mut reader = csv::Reader::from_path(path)?;
        let headers = reader.headers()?.clone();
        
        // Find column indices
        let symbol_idx = self.find_column(&headers, &["symbol", "ticker", "asset"]);
        let pnl_idx = self.find_column(&headers, &["net_pnl", "pnl", "realized_pnl"]);
        let gross_idx = self.find_column(&headers, &["gross_pnl"]);
        let cost_idx = self.find_column(&headers, &["cost", "costs", "commission"]);

        let mut trades = Vec::new();

        for result in reader.records() {
            let record = result?;
            
            let symbol = symbol_idx
                .and_then(|i| record.get(i))
                .unwrap_or("UNKNOWN")
                .to_string();

            let net_pnl = pnl_idx
                .and_then(|i| record.get(i))
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(0.0);

            let gross_pnl = gross_idx
                .and_then(|i| record.get(i))
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(net_pnl);

            let cost = cost_idx
                .and_then(|i| record.get(i))
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(0.0);

            trades.push(TradeRecord {
                symbol,
                net_pnl,
                gross_pnl,
                cost,
            });
        }

        Ok(trades)
    }

    /// Find column index by possible names.
    fn find_column(&self, headers: &csv::StringRecord, names: &[&str]) -> Option<usize> {
        for (i, header) in headers.iter().enumerate() {
            let lower = header.to_lowercase();
            for name in names {
                if lower.contains(name) {
                    return Some(i);
                }
            }
        }
        None
    }

    /// Calculate attribution from trades.
    pub fn calculate(&self, trades: &[TradeRecord]) -> AttributionResult {
        if trades.is_empty() {
            return AttributionResult {
                verdict: Verdict::Pass,
                ..Default::default()
            };
        }

        // Aggregate by symbol
        let mut by_symbol: HashMap<String, AssetAttribution> = HashMap::new();

        for trade in trades {
            let entry = by_symbol.entry(trade.symbol.clone()).or_insert_with(|| {
                AssetAttribution {
                    symbol: trade.symbol.clone(),
                    ..Default::default()
                }
            });

            entry.net_pnl += trade.net_pnl;
            entry.gross_pnl += trade.gross_pnl;
            entry.total_costs += trade.cost;
            entry.num_trades += 1;

            if trade.net_pnl > 0.0 {
                entry.winning_trades += 1;
            } else if trade.net_pnl < 0.0 {
                entry.losing_trades += 1;
            }
        }

        // Calculate derived metrics
        let total_net_pnl: f64 = by_symbol.values().map(|a| a.net_pnl).sum();
        let total_gross_pnl: f64 = by_symbol.values().map(|a| a.gross_pnl).sum();
        let total_trades: u32 = by_symbol.values().map(|a| a.num_trades).sum();

        for attr in by_symbol.values_mut() {
            attr.win_rate = if attr.num_trades > 0 {
                attr.winning_trades as f64 / attr.num_trades as f64
            } else {
                0.0
            };

            attr.avg_trade_pnl = if attr.num_trades > 0 {
                attr.net_pnl / attr.num_trades as f64
            } else {
                0.0
            };

            attr.contribution_pct = if total_net_pnl.abs() > 1e-10 {
                attr.net_pnl / total_net_pnl
            } else {
                0.0
            };
        }

        // Sort by net_pnl descending
        let mut attributions: Vec<AssetAttribution> = by_symbol.into_values().collect();
        attributions.sort_by(|a, b| b.net_pnl.partial_cmp(&a.net_pnl).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate concentration
        let concentration = self.calculate_concentration(&attributions, total_net_pnl);

        // Generate warnings
        let mut warnings = Vec::new();
        if concentration.top_1_pct > self.config.concentration_warn_threshold {
            warnings.push(ValidationWarning::new(
                "HIGH_CONCENTRATION",
                format!(
                    "Single asset accounts for {:.1}% of PnL (threshold: {:.0}%)",
                    concentration.top_1_pct * 100.0,
                    self.config.concentration_warn_threshold * 100.0
                ),
            ));
        }

        let verdict = if warnings.is_empty() { Verdict::Pass } else { Verdict::Warn };

        AttributionResult {
            attributions,
            total_net_pnl,
            total_gross_pnl,
            total_trades,
            concentration,
            warnings,
            verdict,
        }
    }

    /// Calculate concentration metrics.
    fn calculate_concentration(&self, attributions: &[AssetAttribution], total_pnl: f64) -> ConcentrationMetrics {
        if attributions.is_empty() || total_pnl.abs() < 1e-10 {
            return ConcentrationMetrics::default();
        }

        let top_1_pct = attributions.first().map(|a| a.net_pnl.abs() / total_pnl.abs()).unwrap_or(0.0);
        
        let top_5_sum: f64 = attributions.iter().take(5).map(|a| a.net_pnl.abs()).sum();
        let top_5_pct = top_5_sum / total_pnl.abs();

        let top_10_sum: f64 = attributions.iter().take(10).map(|a| a.net_pnl.abs()).sum();
        let top_10_pct = top_10_sum / total_pnl.abs();

        // HHI: sum of squared market shares
        let hhi: f64 = attributions
            .iter()
            .map(|a| (a.net_pnl.abs() / total_pnl.abs()).powi(2))
            .sum();

        ConcentrationMetrics {
            top_1_pct,
            top_5_pct,
            top_10_pct,
            hhi,
        }
    }

    /// Write attribution to CSV.
    pub fn write_csv(&self, result: &AttributionResult, path: &Path) -> Result<(), ValidationError> {
        let mut file = std::fs::File::create(path)?;
        
        // Header
        writeln!(
            file,
            "symbol,net_pnl,gross_pnl,total_costs,num_trades,winning_trades,losing_trades,win_rate,avg_trade_pnl,contribution_pct"
        )?;

        // Data rows
        for attr in &result.attributions {
            writeln!(
                file,
                "{},{:.2},{:.2},{:.2},{},{},{},{:.4},{:.2},{:.4}",
                attr.symbol,
                attr.net_pnl,
                attr.gross_pnl,
                attr.total_costs,
                attr.num_trades,
                attr.winning_trades,
                attr.losing_trades,
                attr.win_rate,
                attr.avg_trade_pnl,
                attr.contribution_pct
            )?;
        }

        Ok(())
    }

    /// Get top N winners.
    pub fn top_winners<'a>(&self, result: &'a AttributionResult) -> Vec<&'a AssetAttribution> {
        result
            .attributions
            .iter()
            .filter(|a| a.net_pnl > 0.0)
            .take(self.config.top_n)
            .collect()
    }

    /// Get top N losers.
    pub fn top_losers<'a>(&self, result: &'a AttributionResult) -> Vec<&'a AssetAttribution> {
        result
            .attributions
            .iter()
            .filter(|a| a.net_pnl < 0.0)
            .rev()
            .take(self.config.top_n)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attribution_basic() {
        let calc = AttributionCalculator::default();
        
        let trades = vec![
            TradeRecord { symbol: "PETR4".into(), net_pnl: 100.0, gross_pnl: 105.0, cost: 5.0 },
            TradeRecord { symbol: "PETR4".into(), net_pnl: 50.0, gross_pnl: 55.0, cost: 5.0 },
            TradeRecord { symbol: "VALE3".into(), net_pnl: -30.0, gross_pnl: -25.0, cost: 5.0 },
        ];

        let result = calc.calculate(&trades);

        assert_eq!(result.attributions.len(), 2);
        assert_eq!(result.total_trades, 3);
        
        // PETR4 should be first (highest PnL)
        assert_eq!(result.attributions[0].symbol, "PETR4");
        assert_eq!(result.attributions[0].net_pnl, 150.0);
        assert_eq!(result.attributions[0].num_trades, 2);
    }

    #[test]
    fn test_concentration_warning() {
        let calc = AttributionCalculator::new(AttributionConfig {
            concentration_warn_threshold: 0.5,
            ..Default::default()
        });

        // One asset dominates
        let trades = vec![
            TradeRecord { symbol: "PETR4".into(), net_pnl: 900.0, gross_pnl: 900.0, cost: 0.0 },
            TradeRecord { symbol: "VALE3".into(), net_pnl: 100.0, gross_pnl: 100.0, cost: 0.0 },
        ];

        let result = calc.calculate(&trades);

        assert!(!result.warnings.is_empty());
        assert!(result.concentration.top_1_pct > 0.8);
    }

    #[test]
    fn test_win_rate() {
        let calc = AttributionCalculator::default();
        
        let trades = vec![
            TradeRecord { symbol: "PETR4".into(), net_pnl: 100.0, gross_pnl: 100.0, cost: 0.0 },
            TradeRecord { symbol: "PETR4".into(), net_pnl: -50.0, gross_pnl: -50.0, cost: 0.0 },
            TradeRecord { symbol: "PETR4".into(), net_pnl: 75.0, gross_pnl: 75.0, cost: 0.0 },
        ];

        let result = calc.calculate(&trades);

        // 2 wins out of 3 trades = 66.67%
        let petr = &result.attributions[0];
        assert!((petr.win_rate - 0.6667).abs() < 0.01);
    }
}


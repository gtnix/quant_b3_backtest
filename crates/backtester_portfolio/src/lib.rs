//! # Backtester Portfolio
//!
//! Portfolio state management, `PnL` calculation, and drawdown tracking.
//!
//! Responsibilities:
//! - Track positions per asset (SoA layout)
//! - Maintain cash balance
//! - Calculate realized/unrealized `PnL`
//! - Track drawdown
//!
//! All operations are O(1) per asset for hot path performance.

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]

pub use backtester_core::{AssetId, FillEvent, Timestamp};

/// Portfolio state with SoA layout for cache efficiency.
#[derive(Debug, Clone)]
pub struct Portfolio {
    // Core state (SoA layout)
    /// Positions by AssetId (positive = long, negative = short)
    positions: Vec<i64>,
    /// Average cost per asset
    avg_costs: Vec<f64>,
    /// Realized PnL per asset
    realized_pnl: Vec<f64>,
    /// Unrealized PnL per asset
    unrealized_pnl: Vec<f64>,
    /// Accumulated costs (fees) per asset
    costs_per_asset: Vec<f64>,
    /// Last known prices for mark-to-market
    last_prices: Vec<f64>,

    // Global state
    /// Available cash
    pub cash: f64,
    /// Initial capital
    pub initial_capital: f64,
    /// Peak NAV (for drawdown calculation)
    pub peak_nav: f64,
    /// Current drawdown percentage
    pub drawdown: f64,
    /// Maximum drawdown observed
    pub max_drawdown: f64,
    /// Total costs accumulated
    pub total_costs: f64,

    /// Number of assets
    num_assets: usize,
}

impl Portfolio {
    /// Create a new portfolio with initial capital.
    #[must_use]
    pub fn new(initial_capital: f64, num_assets: usize) -> Self {
        Self {
            positions: vec![0; num_assets],
            avg_costs: vec![0.0; num_assets],
            realized_pnl: vec![0.0; num_assets],
            unrealized_pnl: vec![0.0; num_assets],
            costs_per_asset: vec![0.0; num_assets],
            last_prices: vec![0.0; num_assets],
            cash: initial_capital,
            initial_capital,
            peak_nav: initial_capital,
            drawdown: 0.0,
            max_drawdown: 0.0,
            total_costs: 0.0,
            num_assets,
        }
    }

    /// Get current position for an asset.
    #[must_use]
    pub fn get_position(&self, asset_id: AssetId) -> i64 {
        self.positions.get(asset_id as usize).copied().unwrap_or(0)
    }

    /// Get average cost for an asset.
    #[must_use]
    pub fn get_avg_cost(&self, asset_id: AssetId) -> f64 {
        self.avg_costs.get(asset_id as usize).copied().unwrap_or(0.0)
    }

    /// Get realized PnL for an asset.
    #[must_use]
    pub fn get_realized_pnl(&self, asset_id: AssetId) -> f64 {
        self.realized_pnl.get(asset_id as usize).copied().unwrap_or(0.0)
    }

    /// Get unrealized PnL for an asset.
    #[must_use]
    pub fn get_unrealized_pnl(&self, asset_id: AssetId) -> f64 {
        self.unrealized_pnl.get(asset_id as usize).copied().unwrap_or(0.0)
    }

    /// Get total realized PnL across all assets.
    #[must_use]
    pub fn total_realized_pnl(&self) -> f64 {
        self.realized_pnl.iter().sum()
    }

    /// Get total unrealized PnL across all assets.
    #[must_use]
    pub fn total_unrealized_pnl(&self) -> f64 {
        self.unrealized_pnl.iter().sum()
    }

    /// Get positions value at current market prices.
    #[must_use]
    pub fn positions_value(&self) -> f64 {
        self.positions
            .iter()
            .zip(self.last_prices.iter())
            .map(|(&pos, &price)| {
                #[allow(clippy::cast_precision_loss)]
                let value = pos as f64 * price;
                value
            })
            .sum()
    }

    /// Get current NAV (cash + positions value).
    #[must_use]
    pub fn nav(&self) -> f64 {
        self.cash + self.positions_value()
    }

    /// Get gross exposure (sum of absolute position values).
    #[must_use]
    pub fn gross_exposure(&self) -> f64 {
        self.positions
            .iter()
            .zip(self.last_prices.iter())
            .map(|(&pos, &price)| {
                #[allow(clippy::cast_precision_loss)]
                let value = (pos as f64 * price).abs();
                value
            })
            .sum()
    }

    /// Get net exposure (sum of position values with sign).
    #[must_use]
    pub fn net_exposure(&self) -> f64 {
        self.positions_value()
    }

    /// Process a fill event and update portfolio state. O(1) operation.
    pub fn process_fill(&mut self, fill: &FillEvent) {
        let id = fill.asset_id as usize;
        if id >= self.num_assets {
            return;
        }

        let old_position = self.positions[id];
        let fill_qty = fill.quantity;
        let fill_price = fill.price;

        #[allow(clippy::cast_precision_loss)]
        let fill_qty_f64 = fill_qty as f64;

        // Update costs
        self.costs_per_asset[id] += fill.cost;
        self.total_costs += fill.cost;
        self.cash -= fill.cost;

        // Handle position update and PnL calculation
        if old_position == 0 {
            // Opening new position
            self.positions[id] = fill_qty;
            self.avg_costs[id] = fill_price;
            self.cash -= fill_qty_f64 * fill_price;
        } else if (old_position > 0 && fill_qty > 0) || (old_position < 0 && fill_qty < 0) {
            // Adding to existing position - update average cost
            #[allow(clippy::cast_precision_loss)]
            let old_pos_f64 = old_position as f64;
            let new_position = old_position + fill_qty;
            #[allow(clippy::cast_precision_loss)]
            let new_pos_f64 = new_position as f64;
            
            let old_cost_total = old_pos_f64 * self.avg_costs[id];
            let new_cost_total = fill_qty_f64 * fill_price;
            self.avg_costs[id] = (old_cost_total + new_cost_total) / new_pos_f64;
            self.positions[id] = new_position;
            self.cash -= fill_qty_f64 * fill_price;
        } else {
            // Reducing or reversing position
            let close_qty = fill_qty.abs().min(old_position.abs());
            #[allow(clippy::cast_precision_loss)]
            let close_qty_f64 = close_qty as f64;

            // Realize PnL on closed portion
            let pnl = if old_position > 0 {
                // Closing long: sell price - avg cost
                close_qty_f64 * (fill_price - self.avg_costs[id])
            } else {
                // Closing short: avg cost - buy price
                close_qty_f64 * (self.avg_costs[id] - fill_price)
            };
            self.realized_pnl[id] += pnl;

            let new_position = old_position + fill_qty;
            self.positions[id] = new_position;

            // Update cash (buy is negative cash, sell is positive)
            self.cash -= fill_qty_f64 * fill_price;

            // If position reversed, set new average cost
            if (new_position > 0 && old_position < 0) || (new_position < 0 && old_position > 0) {
                let excess_qty = fill_qty.abs() - close_qty;
                if excess_qty > 0 {
                    self.avg_costs[id] = fill_price;
                }
            } else if new_position == 0 {
                self.avg_costs[id] = 0.0;
            }
        }

        // Update unrealized PnL
        self.update_unrealized_pnl(fill.asset_id, fill_price);
        
        // Update last price
        self.last_prices[id] = fill_price;
    }

    /// Update unrealized PnL for a single asset.
    fn update_unrealized_pnl(&mut self, asset_id: AssetId, current_price: f64) {
        let id = asset_id as usize;
        if id >= self.num_assets {
            return;
        }
        let position = self.positions[id];
        if position == 0 {
            self.unrealized_pnl[id] = 0.0;
            return;
        }
        #[allow(clippy::cast_precision_loss)]
        let pos_f64 = position as f64;
        self.unrealized_pnl[id] = pos_f64 * (current_price - self.avg_costs[id]);
    }

    /// Mark-to-market all positions with current prices. O(N) where N = num_assets.
    pub fn mark_to_market(&mut self, prices: &[f64]) {
        for (id, &price) in prices.iter().enumerate() {
            if id >= self.num_assets {
                break;
            }
            if price > 0.0 {
                self.last_prices[id] = price;
                #[allow(clippy::cast_possible_truncation)]
                self.update_unrealized_pnl(id as AssetId, price);
            }
        }
        self.update_drawdown();
    }

    /// Update drawdown calculation.
    pub fn update_drawdown(&mut self) {
        let nav = self.nav();
        if nav > self.peak_nav {
            self.peak_nav = nav;
        }
        if self.peak_nav > 0.0 {
            self.drawdown = (self.peak_nav - nav) / self.peak_nav;
            if self.drawdown > self.max_drawdown {
                self.max_drawdown = self.drawdown;
            }
        }
    }

    /// Check if all positions are zero (net zero / flat).
    #[must_use]
    pub fn is_flat(&self) -> bool {
        self.positions.iter().all(|&p| p == 0)
    }

    /// Get all non-zero positions as (AssetId, quantity) pairs.
    #[must_use]
    pub fn open_positions(&self) -> Vec<(AssetId, i64)> {
        self.positions
            .iter()
            .enumerate()
            .filter(|(_, &pos)| pos != 0)
            .map(|(id, &pos)| {
                #[allow(clippy::cast_possible_truncation)]
                (id as AssetId, pos)
            })
            .collect()
    }

    /// Get number of assets.
    #[must_use]
    pub fn num_assets(&self) -> usize {
        self.num_assets
    }
}

/// Portfolio view for strategies (read-only access).
#[derive(Debug)]
pub struct PortfolioView<'a> {
    portfolio: &'a Portfolio,
}

impl<'a> PortfolioView<'a> {
    /// Create a view from a portfolio.
    #[must_use]
    pub fn new(portfolio: &'a Portfolio) -> Self {
        Self { portfolio }
    }

    /// Get position for an asset.
    #[must_use]
    pub fn get_position(&self, asset_id: AssetId) -> i64 {
        self.portfolio.get_position(asset_id)
    }

    /// Get current NAV.
    #[must_use]
    pub fn nav(&self) -> f64 {
        self.portfolio.nav()
    }

    /// Get available cash.
    #[must_use]
    pub fn cash(&self) -> f64 {
        self.portfolio.cash
    }

    /// Check if portfolio is flat.
    #[must_use]
    pub fn is_flat(&self) -> bool {
        self.portfolio.is_flat()
    }

    /// Get gross exposure.
    #[must_use]
    pub fn gross_exposure(&self) -> f64 {
        self.portfolio.gross_exposure()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portfolio_creation() {
        let portfolio = Portfolio::new(100_000.0, 10);
        assert!((portfolio.cash - 100_000.0).abs() < f64::EPSILON);
        assert!((portfolio.drawdown).abs() < f64::EPSILON);
        assert!(portfolio.is_flat());
    }

    #[test]
    fn process_fill_opens_position() {
        let mut portfolio = Portfolio::new(100_000.0, 10);
        let fill = FillEvent {
            timestamp: 0,
            asset_id: 0,
            quantity: 100,
            price: 50.0,
            cost: 10.0,
        };
        portfolio.process_fill(&fill);
        
        assert_eq!(portfolio.get_position(0), 100);
        assert!((portfolio.get_avg_cost(0) - 50.0).abs() < f64::EPSILON);
        // Cash = 100_000 - 100*50 - 10 = 94_990
        assert!((portfolio.cash - 94_990.0).abs() < f64::EPSILON);
    }

    #[test]
    fn process_fill_closes_position_with_profit() {
        let mut portfolio = Portfolio::new(100_000.0, 10);
        
        // Buy 100 at 50
        portfolio.process_fill(&FillEvent {
            timestamp: 0,
            asset_id: 0,
            quantity: 100,
            price: 50.0,
            cost: 5.0,
        });

        // Sell 100 at 55 (profit of 500)
        portfolio.process_fill(&FillEvent {
            timestamp: 1,
            asset_id: 0,
            quantity: -100,
            price: 55.0,
            cost: 5.0,
        });

        assert_eq!(portfolio.get_position(0), 0);
        assert!((portfolio.get_realized_pnl(0) - 500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn drawdown_calculation() {
        let mut portfolio = Portfolio::new(100_000.0, 10);
        portfolio.update_drawdown();
        assert!((portfolio.drawdown).abs() < f64::EPSILON);

        // Simulate loss
        portfolio.cash = 90_000.0;
        portfolio.update_drawdown();
        assert!((portfolio.drawdown - 0.1).abs() < 0.001);
    }

    #[test]
    fn mark_to_market_updates_unrealized() {
        let mut portfolio = Portfolio::new(100_000.0, 10);
        
        // Buy 100 at 50
        portfolio.process_fill(&FillEvent {
            timestamp: 0,
            asset_id: 0,
            quantity: 100,
            price: 50.0,
            cost: 0.0,
        });

        // Price moves to 55
        let prices = vec![55.0; 10];
        portfolio.mark_to_market(&prices);

        // Unrealized PnL = 100 * (55 - 50) = 500
        assert!((portfolio.get_unrealized_pnl(0) - 500.0).abs() < f64::EPSILON);
    }
}

//! # Backtester Portfolio
//!
//! Portfolio state management, PnL calculation, and drawdown tracking.
//!
//! Responsibilities:
//! - Track positions per asset (SoA layout for hot path)
//! - Maintain cash balance
//! - Calculate realized/unrealized PnL
//! - Track drawdown
//! - Record closed trades for metrics
//!
//! All operations are O(1) per asset for hot path performance.

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub use backtester_core::{AssetId, FillEvent, OrderDirection};

// =============================================================================
// POSITION
// =============================================================================

/// Position in a single asset with full tracking.
#[derive(Debug, Clone, PartialEq)]
pub struct Position {
    /// Asset identifier.
    pub asset_id: AssetId,
    /// Quantity (positive = long, negative = short).
    pub quantity: i64,
    /// Average entry price.
    pub entry_price: f64,
    /// Current market price.
    pub current_price: f64,
    /// Entry timestamp (first fill).
    pub entry_timestamp: i64,
    /// Last update timestamp.
    pub last_update: i64,
    /// Unrealized PnL.
    pub unrealized_pnl: f64,
    /// Realized PnL (from partial closes).
    pub realized_pnl: f64,
}

impl Position {
    /// Create a new position from a fill.
    #[must_use]
    pub fn new(asset_id: AssetId, quantity: i64, entry_price: f64, timestamp: i64) -> Self {
        Self {
            asset_id,
            quantity,
            entry_price,
            current_price: entry_price,
            entry_timestamp: timestamp,
            last_update: timestamp,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
        }
    }

    /// Update position with current market price.
    pub fn mark_to_market(&mut self, current_price: f64) {
        self.current_price = current_price;
        #[allow(clippy::cast_precision_loss)]
        let pos_f64 = self.quantity as f64;
        self.unrealized_pnl = pos_f64 * (current_price - self.entry_price);
    }

    /// Get notional value (quantity * current_price).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn notional_value(&self) -> f64 {
        self.quantity as f64 * self.current_price
    }

    /// Get absolute notional (for gross exposure).
    #[must_use]
    pub fn abs_notional(&self) -> f64 {
        self.notional_value().abs()
    }

    /// Check if position is long.
    #[must_use]
    pub fn is_long(&self) -> bool {
        self.quantity > 0
    }

    /// Check if position is short.
    #[must_use]
    pub fn is_short(&self) -> bool {
        self.quantity < 0
    }

    /// Check if position is flat (zero).
    #[must_use]
    pub fn is_flat(&self) -> bool {
        self.quantity == 0
    }
}

impl Default for Position {
    fn default() -> Self {
        Self {
            asset_id: AssetId::new(0),
            quantity: 0,
            entry_price: 0.0,
            current_price: 0.0,
            entry_timestamp: 0,
            last_update: 0,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
        }
    }
}

// =============================================================================
// TRADE (Closed)
// =============================================================================

/// Closed trade record for metrics and audit.
#[derive(Debug, Clone, PartialEq)]
pub struct Trade {
    /// Asset identifier.
    pub asset_id: AssetId,
    /// Trade direction (direction of entry).
    pub direction: OrderDirection,
    /// Entry price.
    pub entry_price: f64,
    /// Exit price.
    pub exit_price: f64,
    /// Quantity traded.
    pub quantity: i64,
    /// Entry timestamp.
    pub entry_timestamp: i64,
    /// Exit timestamp.
    pub exit_timestamp: i64,
    /// Gross PnL (before costs).
    pub gross_pnl: f64,
    /// Net PnL (after costs).
    pub net_pnl: f64,
    /// Total commission paid.
    pub commission: f64,
    /// Holding period in nanoseconds.
    pub holding_period: i64,
}

impl Trade {
    /// Create a new closed trade.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn new(
        asset_id: AssetId,
        direction: OrderDirection,
        entry_price: f64,
        exit_price: f64,
        quantity: i64,
        entry_timestamp: i64,
        exit_timestamp: i64,
        commission: f64,
    ) -> Self {
        let qty_f64 = quantity.unsigned_abs() as f64;
        let gross_pnl = match direction {
            OrderDirection::Buy => qty_f64 * (exit_price - entry_price),
            OrderDirection::Sell => qty_f64 * (entry_price - exit_price),
        };
        let net_pnl = gross_pnl - commission;
        let holding_period = exit_timestamp - entry_timestamp;

        Self {
            asset_id,
            direction,
            entry_price,
            exit_price,
            quantity,
            entry_timestamp,
            exit_timestamp,
            gross_pnl,
            net_pnl,
            commission,
            holding_period,
        }
    }

    /// Check if trade was profitable.
    #[must_use]
    pub fn is_winner(&self) -> bool {
        self.net_pnl > 0.0
    }

    /// Get return percentage.
    #[must_use]
    pub fn return_pct(&self) -> f64 {
        if self.entry_price > 0.0 {
            match self.direction {
                OrderDirection::Buy => (self.exit_price - self.entry_price) / self.entry_price,
                OrderDirection::Sell => (self.entry_price - self.exit_price) / self.entry_price,
            }
        } else {
            0.0
        }
    }
}

// =============================================================================
// CACHE-ALIGNED HOT DATA
// =============================================================================

/// Hot data accessed on every tick - cache-line aligned.
/// Separated from cold data to maximize L1 cache hits.
#[repr(C, align(64))]
#[derive(Debug, Clone)]
pub struct HotData {
    /// Available cash (accessed on every fill).
    pub cash: f64,
    /// Peak NAV (accessed on every tick for drawdown).
    pub peak_nav: f64,
    /// Current drawdown.
    pub drawdown: f64,
    /// Maximum drawdown observed.
    pub max_drawdown: f64,
    /// Total costs accumulated.
    pub total_costs: f64,
    /// Padding to fill cache line.
    _padding: [f64; 3],
}

impl Default for HotData {
    fn default() -> Self {
        Self {
            cash: 0.0,
            peak_nav: 0.0,
            drawdown: 0.0,
            max_drawdown: 0.0,
            total_costs: 0.0,
            _padding: [0.0; 3],
        }
    }
}

// =============================================================================
// PORTFOLIO
// =============================================================================

/// Portfolio state with SoA layout for cache efficiency.
/// Hot data is separated and cache-line aligned for maximum L1 cache hits.
///
/// # Deprecation Notice
///
/// For new code, prefer [`backtester_intelligence::PortfolioState`] which provides:
/// - `rust_decimal::Decimal` precision for financial calculations
/// - Dividend cashflow support via `add_cash()`
/// - Integration with Entry/Exit engines
///
/// This struct remains useful for high-frequency hot path scenarios where
/// f64 performance is critical and dividend tracking is not needed.
#[derive(Debug, Clone)]
pub struct Portfolio {
    // Hot data - cache-line aligned
    /// Hot data accessed every tick.
    hot: HotData,

    // Core state (SoA layout for hot path)
    /// Positions by AssetId index.
    positions: Vec<i64>,
    /// Average cost per asset.
    avg_costs: Vec<f64>,
    /// Entry timestamps per asset.
    entry_timestamps: Vec<i64>,
    /// Realized PnL per asset.
    realized_pnl: Vec<f64>,
    /// Unrealized PnL per asset.
    unrealized_pnl: Vec<f64>,
    /// Accumulated costs per asset.
    costs_per_asset: Vec<f64>,
    /// Last known prices.
    last_prices: Vec<f64>,

    // Cold data (accessed less frequently)
    /// Initial capital.
    pub initial_capital: f64,
    /// Closed trades for metrics.
    closed_trades: Vec<Trade>,
    /// Number of assets.
    num_assets: usize,
}

// Public accessors for hot data fields
impl Portfolio {
    /// Get available cash.
    #[inline(always)]
    #[must_use]
    pub fn cash(&self) -> f64 {
        self.hot.cash
    }

    /// Get peak NAV.
    #[inline(always)]
    #[must_use]
    pub fn peak_nav(&self) -> f64 {
        self.hot.peak_nav
    }

    /// Get current drawdown.
    #[inline(always)]
    #[must_use]
    pub fn drawdown(&self) -> f64 {
        self.hot.drawdown
    }

    /// Get maximum drawdown.
    #[inline(always)]
    #[must_use]
    pub fn max_drawdown(&self) -> f64 {
        self.hot.max_drawdown
    }

    /// Get total costs.
    #[inline(always)]
    #[must_use]
    pub fn total_costs(&self) -> f64 {
        self.hot.total_costs
    }
}

// Legacy field access for backwards compatibility
impl Portfolio {
    /// Legacy cash field accessor.
    #[doc(hidden)]
    pub fn set_cash(&mut self, value: f64) {
        self.hot.cash = value;
    }
}

impl Portfolio {
    /// Create a new portfolio with initial capital.
    #[must_use]
    pub fn new(initial_capital: f64, num_assets: usize) -> Self {
        Self {
            hot: HotData {
                cash: initial_capital,
                peak_nav: initial_capital,
                drawdown: 0.0,
                max_drawdown: 0.0,
                total_costs: 0.0,
                _padding: [0.0; 3],
            },
            positions: vec![0; num_assets],
            avg_costs: vec![0.0; num_assets],
            entry_timestamps: vec![0; num_assets],
            realized_pnl: vec![0.0; num_assets],
            unrealized_pnl: vec![0.0; num_assets],
            costs_per_asset: vec![0.0; num_assets],
            last_prices: vec![0.0; num_assets],
            initial_capital,
            closed_trades: Vec::with_capacity(100),
            num_assets,
        }
    }

    /// Get current position for an asset.
    #[must_use]
    pub fn get_position(&self, asset_id: AssetId) -> i64 {
        self.positions
            .get(asset_id.as_usize())
            .copied()
            .unwrap_or(0)
    }

    /// Get average cost for an asset.
    #[must_use]
    pub fn get_avg_cost(&self, asset_id: AssetId) -> f64 {
        self.avg_costs
            .get(asset_id.as_usize())
            .copied()
            .unwrap_or(0.0)
    }

    /// Get realized PnL for an asset.
    #[must_use]
    pub fn get_realized_pnl(&self, asset_id: AssetId) -> f64 {
        self.realized_pnl
            .get(asset_id.as_usize())
            .copied()
            .unwrap_or(0.0)
    }

    /// Get unrealized PnL for an asset.
    #[must_use]
    pub fn get_unrealized_pnl(&self, asset_id: AssetId) -> f64 {
        self.unrealized_pnl
            .get(asset_id.as_usize())
            .copied()
            .unwrap_or(0.0)
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
    #[inline(always)]
    #[must_use]
    pub fn nav(&self) -> f64 {
        self.hot.cash + self.positions_value()
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

    /// Get net exposure (signed sum of position values).
    #[must_use]
    pub fn net_exposure(&self) -> f64 {
        self.positions_value()
    }

    /// Get long exposure.
    #[must_use]
    pub fn long_exposure(&self) -> f64 {
        self.positions
            .iter()
            .zip(self.last_prices.iter())
            .filter(|(&pos, _)| pos > 0)
            .map(|(&pos, &price)| {
                #[allow(clippy::cast_precision_loss)]
                (pos as f64 * price)
            })
            .sum()
    }

    /// Get short exposure (as positive value).
    #[must_use]
    pub fn short_exposure(&self) -> f64 {
        self.positions
            .iter()
            .zip(self.last_prices.iter())
            .filter(|(&pos, _)| pos < 0)
            .map(|(&pos, &price)| {
                #[allow(clippy::cast_precision_loss)]
                (pos as f64 * price).abs()
            })
            .sum()
    }

    /// Get Position struct for an asset.
    #[must_use]
    pub fn get_position_details(&self, asset_id: AssetId) -> Option<Position> {
        let idx = asset_id.as_usize();
        if idx >= self.num_assets || self.positions[idx] == 0 {
            return None;
        }

        Some(Position {
            asset_id,
            quantity: self.positions[idx],
            entry_price: self.avg_costs[idx],
            current_price: self.last_prices[idx],
            entry_timestamp: self.entry_timestamps[idx],
            last_update: 0, // Would need timestamp tracking
            unrealized_pnl: self.unrealized_pnl[idx],
            realized_pnl: self.realized_pnl[idx],
        })
    }

    /// Get all open positions as Position structs.
    #[must_use]
    pub fn get_all_positions(&self) -> Vec<Position> {
        self.positions
            .iter()
            .enumerate()
            .filter(|(_, &pos)| pos != 0)
            .map(|(idx, _)| self.get_position_details(AssetId::new(idx as u16)).unwrap())
            .collect()
    }

    /// Get all closed trades.
    #[must_use]
    pub fn get_closed_trades(&self) -> &[Trade] {
        &self.closed_trades
    }

    /// Process a fill event and update portfolio state. O(1) operation.
    pub fn process_fill(&mut self, fill: &FillEvent) {
        let idx = fill.asset_id.as_usize();
        if idx >= self.num_assets {
            return;
        }

        let old_position = self.positions[idx];
        let fill_qty = fill.signed_quantity();
        let fill_price = fill.price;

        #[allow(clippy::cast_precision_loss)]
        let fill_qty_f64 = fill_qty as f64;

        // Update costs
        let cost = fill.total_cost;
        self.costs_per_asset[idx] += cost;
        self.hot.total_costs += cost;
        self.hot.cash -= cost;

        // Handle position update and PnL calculation
        if old_position == 0 {
            // Opening new position
            self.positions[idx] = fill_qty;
            self.avg_costs[idx] = fill_price;
            self.entry_timestamps[idx] = fill.timestamp;
            self.hot.cash -= fill_qty_f64 * fill_price;
        } else if (old_position > 0 && fill_qty > 0) || (old_position < 0 && fill_qty < 0) {
            // Adding to existing position - update average cost
            #[allow(clippy::cast_precision_loss)]
            let old_pos_f64 = old_position as f64;
            let new_position = old_position + fill_qty;
            #[allow(clippy::cast_precision_loss)]
            let new_pos_f64 = new_position as f64;

            let old_cost_total = old_pos_f64 * self.avg_costs[idx];
            let new_cost_total = fill_qty_f64 * fill_price;
            self.avg_costs[idx] = (old_cost_total + new_cost_total) / new_pos_f64;
            self.positions[idx] = new_position;
            self.hot.cash -= fill_qty_f64 * fill_price;
        } else {
            // Reducing or reversing position
            let close_qty = fill_qty.abs().min(old_position.abs());
            #[allow(clippy::cast_precision_loss)]
            let close_qty_f64 = close_qty as f64;

            // Determine direction of original position
            let entry_direction = if old_position > 0 {
                OrderDirection::Buy
            } else {
                OrderDirection::Sell
            };

            // Realize PnL on closed portion
            let pnl = if old_position > 0 {
                close_qty_f64 * (fill_price - self.avg_costs[idx])
            } else {
                close_qty_f64 * (self.avg_costs[idx] - fill_price)
            };
            self.realized_pnl[idx] += pnl;

            // Record closed trade
            let trade = Trade::new(
                fill.asset_id,
                entry_direction,
                self.avg_costs[idx],
                fill_price,
                close_qty,
                self.entry_timestamps[idx],
                fill.timestamp,
                cost,
            );
            self.closed_trades.push(trade);

            let new_position = old_position + fill_qty;
            self.positions[idx] = new_position;
            self.hot.cash -= fill_qty_f64 * fill_price;

            // If position reversed, set new average cost
            if (new_position > 0 && old_position < 0) || (new_position < 0 && old_position > 0) {
                let excess_qty = fill_qty.abs() - close_qty;
                if excess_qty > 0 {
                    self.avg_costs[idx] = fill_price;
                    self.entry_timestamps[idx] = fill.timestamp;
                }
            } else if new_position == 0 {
                self.avg_costs[idx] = 0.0;
                self.entry_timestamps[idx] = 0;
            }
        }

        // Update unrealized PnL and last price
        self.last_prices[idx] = fill_price;
        self.update_unrealized_pnl(fill.asset_id, fill_price);
    }

    /// Update unrealized PnL for a single asset.
    fn update_unrealized_pnl(&mut self, asset_id: AssetId, current_price: f64) {
        let idx = asset_id.as_usize();
        if idx >= self.num_assets {
            return;
        }
        let position = self.positions[idx];
        if position == 0 {
            self.unrealized_pnl[idx] = 0.0;
            return;
        }
        #[allow(clippy::cast_precision_loss)]
        let pos_f64 = position as f64;
        self.unrealized_pnl[idx] = pos_f64 * (current_price - self.avg_costs[idx]);
    }

    /// Mark-to-market all positions with current prices. O(N).
    pub fn mark_to_market(&mut self, prices: &[f64]) {
        for (idx, &price) in prices.iter().enumerate() {
            if idx >= self.num_assets {
                break;
            }
            if price > 0.0 {
                self.last_prices[idx] = price;
                self.update_unrealized_pnl(AssetId::new(idx as u16), price);
            }
        }
        self.update_drawdown();
    }

    /// Update drawdown calculation.
    #[inline(always)]
    pub fn update_drawdown(&mut self) {
        let nav = self.nav();
        if nav > self.hot.peak_nav {
            self.hot.peak_nav = nav;
        }
        if self.hot.peak_nav > 0.0 {
            self.hot.drawdown = (self.hot.peak_nav - nav) / self.hot.peak_nav;
            if self.hot.drawdown > self.hot.max_drawdown {
                self.hot.max_drawdown = self.hot.drawdown;
            }
        }
    }

    /// Check if all positions are zero (flat).
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
            .map(|(idx, &pos)| (AssetId::new(idx as u16), pos))
            .collect()
    }

    /// Get number of open positions.
    #[must_use]
    pub fn num_open_positions(&self) -> usize {
        self.positions.iter().filter(|&&p| p != 0).count()
    }

    /// Get number of closed trades.
    #[must_use]
    pub fn num_closed_trades(&self) -> usize {
        self.closed_trades.len()
    }

    /// Get number of assets.
    #[must_use]
    pub fn num_assets(&self) -> usize {
        self.num_assets
    }
}

// =============================================================================
// PORTFOLIO VIEW (Read-only)
// =============================================================================

/// Portfolio view for strategies (read-only access).
#[derive(Debug)]
pub struct PortfolioView<'a> {
    portfolio: &'a Portfolio,
}

impl<'a> PortfolioView<'a> {
    /// Create a view from a portfolio.
    #[must_use]
    pub const fn new(portfolio: &'a Portfolio) -> Self {
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
        self.portfolio.cash()
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

    /// Get net exposure.
    #[must_use]
    pub fn net_exposure(&self) -> f64 {
        self.portfolio.net_exposure()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use backtester_core::{FillId, OrderId};

    fn make_fill(asset_id: u16, quantity: i64, price: f64, cost: f64) -> FillEvent {
        let direction = if quantity >= 0 {
            OrderDirection::Buy
        } else {
            OrderDirection::Sell
        };
        FillEvent::new(
            FillId::new(1),
            OrderId::new(1),
            0,
            AssetId::new(asset_id),
            direction,
            quantity.abs(),
            price,
            cost,
            0.0,
        )
    }

    #[test]
    fn portfolio_creation() {
        let portfolio = Portfolio::new(100_000.0, 10);
        assert!((portfolio.cash() - 100_000.0).abs() < f64::EPSILON);
        assert!((portfolio.drawdown()).abs() < f64::EPSILON);
        assert!(portfolio.is_flat());
    }

    #[test]
    fn process_fill_opens_position() {
        let mut portfolio = Portfolio::new(100_000.0, 10);
        let fill = make_fill(0, 100, 50.0, 10.0);
        portfolio.process_fill(&fill);

        assert_eq!(portfolio.get_position(AssetId::new(0)), 100);
        assert!((portfolio.get_avg_cost(AssetId::new(0)) - 50.0).abs() < f64::EPSILON);
        // Cash = 100_000 - 100*50 - 10 = 94_990
        assert!((portfolio.cash() - 94_990.0).abs() < f64::EPSILON);
    }

    #[test]
    fn process_fill_closes_position_with_profit() {
        let mut portfolio = Portfolio::new(100_000.0, 10);

        // Buy 100 at 50
        portfolio.process_fill(&make_fill(0, 100, 50.0, 5.0));

        // Sell 100 at 55 (profit of 500)
        portfolio.process_fill(&make_fill(0, -100, 55.0, 5.0));

        assert_eq!(portfolio.get_position(AssetId::new(0)), 0);
        assert!((portfolio.get_realized_pnl(AssetId::new(0)) - 500.0).abs() < f64::EPSILON);
        assert_eq!(portfolio.num_closed_trades(), 1);

        let trade = &portfolio.get_closed_trades()[0];
        assert!(trade.is_winner());
        assert!((trade.gross_pnl - 500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn position_struct() {
        let mut pos = Position::new(AssetId::new(0), 100, 50.0, 1000);
        assert!(pos.is_long());
        assert!(!pos.is_short());

        pos.mark_to_market(55.0);
        assert!((pos.unrealized_pnl - 500.0).abs() < f64::EPSILON);
        assert!((pos.notional_value() - 5500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn trade_struct() {
        let trade = Trade::new(
            AssetId::new(0),
            OrderDirection::Buy,
            50.0,
            55.0,
            100,
            1000,
            2000,
            10.0,
        );

        assert!(trade.is_winner());
        assert!((trade.gross_pnl - 500.0).abs() < f64::EPSILON);
        assert!((trade.net_pnl - 490.0).abs() < f64::EPSILON);
        assert!((trade.return_pct() - 0.1).abs() < 0.001);
    }

    #[test]
    fn drawdown_calculation() {
        let mut portfolio = Portfolio::new(100_000.0, 10);
        portfolio.update_drawdown();
        assert!((portfolio.drawdown()).abs() < f64::EPSILON);

        // Simulate loss
        portfolio.set_cash(90_000.0);
        portfolio.update_drawdown();
        assert!((portfolio.drawdown() - 0.1).abs() < 0.001);
    }

    #[test]
    fn mark_to_market_updates_unrealized() {
        let mut portfolio = Portfolio::new(100_000.0, 10);
        portfolio.process_fill(&make_fill(0, 100, 50.0, 0.0));

        // Price moves to 55
        let prices = vec![55.0; 10];
        portfolio.mark_to_market(&prices);

        assert!((portfolio.get_unrealized_pnl(AssetId::new(0)) - 500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn exposure_calculations() {
        let mut portfolio = Portfolio::new(100_000.0, 10);

        // Long 100 shares at 50 = 5000 exposure
        portfolio.process_fill(&make_fill(0, 100, 50.0, 0.0));

        // Short 100 shares at 40 = 4000 exposure
        portfolio.process_fill(&make_fill(1, -100, 40.0, 0.0));

        assert!((portfolio.long_exposure() - 5000.0).abs() < f64::EPSILON);
        assert!((portfolio.short_exposure() - 4000.0).abs() < f64::EPSILON);
        assert!((portfolio.gross_exposure() - 9000.0).abs() < f64::EPSILON);
        assert!((portfolio.net_exposure() - 1000.0).abs() < f64::EPSILON);
    }
}

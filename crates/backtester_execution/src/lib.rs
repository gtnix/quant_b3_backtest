//! # Backtester Execution
//!
//! Order execution simulation with slippage, costs, and latency modeling.
//!
//! Responsibilities:
//! - Transform `OrderEvent`s into `FillEvent`s
//! - Apply slippage models (constant, volume-based, volatility-based)
//! - Calculate execution costs (brokerage, fees)
//! - Handle limit orders with OHLC verification
//! - Support partial fills based on volume participation

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]

pub use backtester_core::{Bar, ExecutionModel, FillEvent, OrderEvent};

/// Slippage model types.
#[derive(Debug, Clone, Copy)]
pub enum SlippageModel {
    /// No slippage
    None,
    /// Fixed basis points slippage
    Constant {
        /// Slippage in basis points (e.g., 10 = 0.1%)
        bps: f64,
    },
    /// Slippage proportional to order size relative to bar volume
    VolumeLinear {
        /// Coefficient for volume impact
        coefficient: f64,
    },
    /// Slippage based on bar volatility (high - low range)
    Volatility {
        /// Coefficient for volatility impact
        coefficient: f64,
    },
}

impl SlippageModel {
    /// Calculate slippage for an order.
    /// Returns the price adjustment (positive = worse for buyer, negative = worse for seller).
    #[must_use]
    pub fn calculate(&self, order: &OrderEvent, bar: &Bar) -> f64 {
        let base_price = bar.close;
        let is_buy = order.quantity > 0;

        let slippage_pct = match self {
            Self::None => 0.0,
            Self::Constant { bps } => bps / 10_000.0,
            Self::VolumeLinear { coefficient } => {
                if bar.volume > 0.0 {
                    #[allow(clippy::cast_precision_loss)]
                    let order_ratio = order.quantity.abs() as f64 / bar.volume;
                    coefficient * order_ratio
                } else {
                    0.0
                }
            }
            Self::Volatility { coefficient } => {
                let range = bar.high - bar.low;
                if base_price > 0.0 {
                    coefficient * (range / base_price)
                } else {
                    0.0
                }
            }
        };

        // Slippage always works against the trader
        let adjustment = base_price * slippage_pct;
        if is_buy {
            adjustment // Buyer pays more
        } else {
            -adjustment // Seller receives less
        }
    }
}

impl Default for SlippageModel {
    fn default() -> Self {
        Self::Constant { bps: 5.0 }
    }
}

/// Cost model for execution fees.
#[derive(Debug, Clone, Copy)]
pub struct CostModel {
    /// Fixed cost per trade
    pub fixed_cost: f64,
    /// Proportional cost (as fraction, e.g., 0.001 = 0.1%)
    pub proportional_bps: f64,
    /// Per-share/unit cost
    pub per_unit_cost: f64,
}

impl CostModel {
    /// Create a new cost model.
    #[must_use]
    pub fn new(fixed_cost: f64, proportional_bps: f64, per_unit_cost: f64) -> Self {
        Self {
            fixed_cost,
            proportional_bps,
            per_unit_cost,
        }
    }

    /// Calculate total cost for an order.
    #[must_use]
    pub fn calculate(&self, notional: f64, quantity: i64) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let qty_f64 = quantity.abs() as f64;
        self.fixed_cost + (notional * self.proportional_bps / 10_000.0) + (qty_f64 * self.per_unit_cost)
    }
}

impl Default for CostModel {
    fn default() -> Self {
        Self::new(10.0, 10.0, 0.0) // R$10 fixed + 10 bps
    }
}

/// Liquidity model for partial fills.
#[derive(Debug, Clone, Copy)]
pub struct LiquidityModel {
    /// Maximum participation rate in bar volume (e.g., 0.1 = 10%)
    pub max_participation: f64,
    /// Whether to allow partial fills
    pub allow_partial: bool,
}

impl LiquidityModel {
    /// Create a new liquidity model.
    #[must_use]
    pub fn new(max_participation: f64, allow_partial: bool) -> Self {
        Self {
            max_participation: max_participation.clamp(0.01, 1.0),
            allow_partial,
        }
    }

    /// Calculate maximum fillable quantity based on bar volume.
    #[must_use]
    pub fn max_fill_quantity(&self, order_qty: i64, bar_volume: f64) -> i64 {
        let max_qty = (bar_volume * self.max_participation) as i64;
        let max_qty = (max_qty / 100) * 100; // B3 round-lot
        
        if self.allow_partial {
            order_qty.abs().min(max_qty.max(100)) * order_qty.signum()
        } else if order_qty.abs() <= max_qty || max_qty == 0 {
            order_qty
        } else {
            0 // Order too large, no fill
        }
    }
}

impl Default for LiquidityModel {
    fn default() -> Self {
        Self::new(0.1, true) // 10% max participation, allow partial
    }
}

/// Configuration for the execution model.
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Slippage model
    pub slippage: SlippageModel,
    /// Cost model
    pub costs: CostModel,
    /// Liquidity model
    pub liquidity: LiquidityModel,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            slippage: SlippageModel::default(),
            costs: CostModel::default(),
            liquidity: LiquidityModel::default(),
        }
    }
}

/// Advanced execution model with configurable slippage, costs, and liquidity.
#[derive(Debug, Clone)]
pub struct AdvancedExecutionModel {
    config: ExecutionConfig,
}

impl AdvancedExecutionModel {
    /// Create a new advanced execution model.
    #[must_use]
    pub fn new(config: ExecutionConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(ExecutionConfig::default())
    }

    /// Check if a limit order can be filled based on bar range.
    fn check_limit_order(&self, order: &OrderEvent, bar: &Bar) -> bool {
        if let Some(limit_price) = order.limit_price {
            if order.quantity > 0 {
                // Buy limit: fills if bar low <= limit price
                bar.low <= limit_price
            } else {
                // Sell limit: fills if bar high >= limit price
                bar.high >= limit_price
            }
        } else {
            true // Market order always attempts to fill
        }
    }

    /// Get execution price for limit orders (pessimistic).
    fn get_limit_fill_price(&self, order: &OrderEvent) -> Option<f64> {
        order.limit_price
    }
}

impl ExecutionModel for AdvancedExecutionModel {
    fn execute(&self, order: &OrderEvent, current_bar: &Bar) -> Option<FillEvent> {
        // Check limit order conditions
        if !self.check_limit_order(order, current_bar) {
            return None;
        }

        // Calculate fillable quantity based on liquidity
        let fill_qty = self.config.liquidity.max_fill_quantity(order.quantity, current_bar.volume);
        if fill_qty == 0 {
            return None;
        }

        // Determine base price
        let base_price = if order.limit_price.is_some() {
            // Limit order: use limit price (pessimistic)
            self.get_limit_fill_price(order)?
        } else {
            // Market order: use close price
            current_bar.close
        };

        // Apply slippage (only for market orders)
        let slippage = if order.limit_price.is_none() {
            self.config.slippage.calculate(order, current_bar)
        } else {
            0.0
        };
        let fill_price = base_price + slippage;

        // Calculate costs
        #[allow(clippy::cast_precision_loss)]
        let notional = fill_qty.abs() as f64 * fill_price;
        let cost = self.config.costs.calculate(notional, fill_qty);

        Some(FillEvent {
            timestamp: order.timestamp,
            asset_id: order.asset_id,
            quantity: fill_qty,
            price: fill_price,
            cost,
        })
    }
}

/// Simple execution model with fixed costs and no slippage (legacy/simple use).
#[derive(Debug, Clone)]
pub struct SimpleExecutionModel {
    /// Fixed cost per trade
    pub fixed_cost: f64,
    /// Proportional cost (e.g., 0.001 = 0.1%)
    pub proportional_cost: f64,
}

impl SimpleExecutionModel {
    /// Create a new simple execution model.
    #[must_use]
    pub fn new(fixed_cost: f64, proportional_cost: f64) -> Self {
        Self {
            fixed_cost,
            proportional_cost,
        }
    }
}

impl ExecutionModel for SimpleExecutionModel {
    fn execute(&self, order: &OrderEvent, current_bar: &Bar) -> Option<FillEvent> {
        let price = current_bar.close;
        let quantity = order.quantity;
        #[allow(clippy::cast_precision_loss)]
        let notional = quantity.abs() as f64 * price;
        let cost = self.fixed_cost + notional * self.proportional_cost;

        Some(FillEvent {
            timestamp: order.timestamp,
            asset_id: order.asset_id,
            quantity,
            price,
            cost,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bar(price: f64, volume: f64) -> Bar {
        Bar {
            timestamp: 0,
            open: price,
            high: price + 1.0,
            low: price - 1.0,
            close: price,
            volume,
        }
    }

    fn make_order(quantity: i64, limit: Option<f64>) -> OrderEvent {
        OrderEvent {
            timestamp: 0,
            asset_id: 0,
            quantity,
            limit_price: limit,
        }
    }

    #[test]
    fn slippage_constant() {
        let model = SlippageModel::Constant { bps: 10.0 };
        let bar = make_bar(100.0, 1000.0);
        
        let buy_order = make_order(100, None);
        let slip = model.calculate(&buy_order, &bar);
        assert!((slip - 0.1).abs() < f64::EPSILON); // 10 bps of 100

        let sell_order = make_order(-100, None);
        let slip = model.calculate(&sell_order, &bar);
        assert!((slip - (-0.1)).abs() < f64::EPSILON);
    }

    #[test]
    fn slippage_volume_linear() {
        let model = SlippageModel::VolumeLinear { coefficient: 1.0 };
        let bar = make_bar(100.0, 1000.0);
        let order = make_order(100, None); // 10% of volume
        
        let slip = model.calculate(&order, &bar);
        // coefficient * (100/1000) * 100 = 1.0 * 0.1 * 100 = 10.0
        assert!((slip - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn slippage_volatility() {
        let model = SlippageModel::Volatility { coefficient: 0.5 };
        let bar = Bar {
            timestamp: 0,
            open: 100.0,
            high: 102.0, // range = 4
            low: 98.0,
            close: 100.0,
            volume: 1000.0,
        };
        let order = make_order(100, None);
        
        let slip = model.calculate(&order, &bar);
        // 0.5 * (4/100) * 100 = 2.0
        assert!((slip - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn cost_model_calculation() {
        let model = CostModel::new(10.0, 10.0, 0.01);
        let cost = model.calculate(10_000.0, 100);
        // 10 + (10000 * 10/10000) + (100 * 0.01) = 10 + 10 + 1 = 21
        assert!((cost - 21.0).abs() < f64::EPSILON);
    }

    #[test]
    fn liquidity_model_limits_fill() {
        let model = LiquidityModel::new(0.1, true);
        
        // Order 500, volume 1000, max = 100
        let fill_qty = model.max_fill_quantity(500, 1000.0);
        assert_eq!(fill_qty, 100);
        
        // Order 50, volume 1000, fills fully
        let fill_qty = model.max_fill_quantity(100, 1000.0);
        assert_eq!(fill_qty, 100);
    }

    #[test]
    fn limit_order_fills_when_in_range() {
        let config = ExecutionConfig {
            slippage: SlippageModel::None,
            costs: CostModel::new(0.0, 0.0, 0.0),
            liquidity: LiquidityModel::new(1.0, true),
        };
        let model = AdvancedExecutionModel::new(config);
        
        let bar = Bar {
            timestamp: 0,
            open: 100.0,
            high: 102.0,
            low: 98.0,
            close: 100.0,
            volume: 1000.0,
        };

        // Buy limit at 99 should fill (low = 98)
        let buy_limit = make_order(100, Some(99.0));
        let fill = model.execute(&buy_limit, &bar);
        assert!(fill.is_some());
        assert!((fill.unwrap().price - 99.0).abs() < f64::EPSILON);

        // Buy limit at 97 should NOT fill (low = 98)
        let buy_limit_miss = make_order(100, Some(97.0));
        let fill = model.execute(&buy_limit_miss, &bar);
        assert!(fill.is_none());
    }

    #[test]
    fn simple_execution_calculates_costs() {
        let model = SimpleExecutionModel::new(10.0, 0.001);
        let order = make_order(100, None);
        let bar = make_bar(50.0, 1000.0);

        let fill = model.execute(&order, &bar).unwrap();
        // Cost = 10 + (100 * 50 * 0.001) = 10 + 5 = 15
        assert!((fill.cost - 15.0).abs() < f64::EPSILON);
        assert_eq!(fill.quantity, 100);
        assert!((fill.price - 50.0).abs() < f64::EPSILON);
    }
}

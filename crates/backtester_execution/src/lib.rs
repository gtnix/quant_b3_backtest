//! # Backtester Execution
//!
//! Order execution simulation with slippage, costs, and latency modeling.
//!
//! Responsibilities:
//! - Transform `OrderEvent`s into `FillEvent`s
//! - Apply slippage models (constant, volume-based, volatility-based)
//! - Calculate execution costs (brokerage, fees, emoluments)
//! - Handle limit orders with OHLC verification
//! - Support partial fills based on volume participation

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod config;
pub mod cost_report;
pub mod gates;
pub mod stress;

pub use backtester_core::{
    AssetId, Bar, ExecutionModel, FillEvent, FillId, OrderDirection, OrderEvent, OrderId, OrderType,
};

pub use config::{
    ExecutionModelConfig, SlippageModelConfig, FeeModelConfig, FeeTier,
    FillPolicyConfig, GapPolicy, RejectPolicy, InstitutionalGatesConfig, ConfigError,
};

pub use gates::{GateChecker, GateResult, GateCheck, GateStatus};

pub use stress::{
    StressSuite, StressScenario, StressTransformType, AcceptanceCriteria,
    StressResult, StressSuiteResult,
};

// =============================================================================
// SLIPPAGE MODEL
// =============================================================================

/// Slippage model types.
#[derive(Debug, Clone, Copy)]
pub enum SlippageModel {
    /// No slippage.
    None,
    /// Fixed basis points slippage.
    Constant {
        /// Slippage in basis points (e.g., 10 = 0.1%).
        bps: f64,
    },
    /// Slippage proportional to order size relative to bar volume.
    VolumeLinear {
        /// Base slippage in bps.
        base_bps: f64,
        /// Coefficient for volume impact.
        volume_factor: f64,
    },
    /// Slippage based on bar volatility (high - low range).
    Volatility {
        /// Base slippage in bps.
        base_bps: f64,
        /// Coefficient for volatility impact.
        vol_factor: f64,
    },
}

impl SlippageModel {
    /// Calculate slippage for an order.
    /// Returns the price adjustment (positive = worse for buyer).
    #[must_use]
    pub fn calculate(&self, order: &OrderEvent, bar: &Bar) -> f64 {
        let base_price = bar.close;
        let is_buy = order.direction == OrderDirection::Buy;

        let slippage_pct = match self {
            Self::None => 0.0,
            Self::Constant { bps } => bps / 10_000.0,
            Self::VolumeLinear {
                base_bps,
                volume_factor,
            } => {
                let base = base_bps / 10_000.0;
                if bar.volume > 0.0 {
                    #[allow(clippy::cast_precision_loss)]
                    let order_ratio = order.quantity as f64 / bar.volume;
                    base + (*volume_factor * order_ratio)
                } else {
                    base
                }
            }
            Self::Volatility {
                base_bps,
                vol_factor,
            } => {
                let base = base_bps / 10_000.0;
                let range = bar.high - bar.low;
                if base_price > 0.0 {
                    base + (*vol_factor * (range / base_price))
                } else {
                    base
                }
            }
        };

        // Slippage always works against the trader
        let adjustment = base_price * slippage_pct;
        if is_buy {
            adjustment.abs()
        } else {
            -adjustment.abs()
        }
    }

    /// Apply slippage to get the fill price.
    #[must_use]
    pub fn apply(&self, base_price: f64, order: &OrderEvent, bar: &Bar) -> f64 {
        base_price + self.calculate(order, bar)
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: &SlippageModelConfig) -> Self {
        match config {
            SlippageModelConfig::None => Self::None,
            SlippageModelConfig::Constant { bps } => Self::Constant { bps: *bps },
            SlippageModelConfig::VolumeImpact {
                base_bps,
                volume_factor,
                ..
            } => Self::VolumeLinear {
                base_bps: *base_bps,
                volume_factor: *volume_factor,
            },
            SlippageModelConfig::VolatilityAdaptive {
                base_bps,
                vol_factor,
                ..
            } => Self::Volatility {
                base_bps: *base_bps,
                vol_factor: *vol_factor,
            },
            SlippageModelConfig::SpreadProxy { base_bps, .. } => {
                // Fallback to constant for spread proxy
                Self::Constant { bps: *base_bps }
            }
        }
    }
}

impl Default for SlippageModel {
    fn default() -> Self {
        Self::Constant { bps: 5.0 }
    }
}

// =============================================================================
// COST MODEL
// =============================================================================

/// Cost model for execution fees (B3-compatible).
#[derive(Debug, Clone, Copy)]
pub struct CostModel {
    /// Fixed cost per trade (e.g., R$ 10).
    pub fixed_cost: f64,
    /// Commission rate as fraction (e.g., 0.001 = 0.1%).
    pub commission_rate: f64,
    /// Per-share/unit cost (e.g., R$ 0.01).
    pub per_unit_cost: f64,
    /// B3 emolument rate (e.g., 0.00035 = 0.035%).
    pub emolument_rate: f64,
}

impl CostModel {
    /// Create a new cost model.
    #[must_use]
    pub const fn new(
        fixed_cost: f64,
        commission_rate: f64,
        per_unit_cost: f64,
        emolument_rate: f64,
    ) -> Self {
        Self {
            fixed_cost,
            commission_rate,
            per_unit_cost,
            emolument_rate,
        }
    }

    /// Create B3 default cost model.
    #[must_use]
    pub const fn b3_default() -> Self {
        Self::new(10.0, 0.001, 0.01, 0.000_35)
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: &FeeModelConfig) -> Self {
        Self {
            fixed_cost: config.fixed_per_trade,
            commission_rate: config.commission_rate,
            per_unit_cost: config.per_unit_cost,
            emolument_rate: config.emolument_rate,
        }
    }

    /// Calculate total cost for an order.
    #[must_use]
    pub fn calculate(&self, notional: f64, quantity: i64) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let qty = quantity.unsigned_abs() as f64;

        let commission = notional * self.commission_rate;
        let emolument = notional * self.emolument_rate;
        let unit_cost = qty * self.per_unit_cost;

        self.fixed_cost + commission + emolument + unit_cost
    }

    /// Calculate total cost breakdown.
    #[must_use]
    pub fn calculate_breakdown(&self, notional: f64, quantity: i64) -> CostBreakdown {
        #[allow(clippy::cast_precision_loss)]
        let qty = quantity.unsigned_abs() as f64;

        CostBreakdown {
            fixed: self.fixed_cost,
            commission: notional * self.commission_rate,
            emolument: notional * self.emolument_rate,
            per_unit: qty * self.per_unit_cost,
        }
    }
}

impl Default for CostModel {
    fn default() -> Self {
        Self::b3_default()
    }
}

/// Cost breakdown for detailed reporting.
#[derive(Debug, Clone, Copy, Default)]
pub struct CostBreakdown {
    /// Fixed cost.
    pub fixed: f64,
    /// Commission.
    pub commission: f64,
    /// B3 emolument.
    pub emolument: f64,
    /// Per-unit cost.
    pub per_unit: f64,
}

impl CostBreakdown {
    /// Total cost.
    #[must_use]
    pub fn total(&self) -> f64 {
        self.fixed + self.commission + self.emolument + self.per_unit
    }
}

// =============================================================================
// LIQUIDITY MODEL
// =============================================================================

/// Liquidity model for partial fills and volume constraints.
#[derive(Debug, Clone, Copy)]
pub struct LiquidityModel {
    /// Maximum participation rate in bar volume (e.g., 0.1 = 10%).
    pub max_participation: f64,
    /// Whether to allow partial fills.
    pub allow_partial_fills: bool,
}

impl LiquidityModel {
    /// Create a new liquidity model.
    #[must_use]
    pub fn new(max_participation: f64, allow_partial_fills: bool) -> Self {
        Self {
            max_participation: max_participation.clamp(0.01, 1.0),
            allow_partial_fills,
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: &FillPolicyConfig) -> Self {
        Self::new(config.max_participation, config.allow_partial)
    }

    /// Check if order can be filled given volume constraints.
    #[must_use]
    pub fn can_fill(&self, order_qty: i64, bar_volume: f64) -> bool {
        let max_qty = self.max_fillable_quantity(bar_volume);
        order_qty.unsigned_abs() as i64 <= max_qty || self.allow_partial_fills
    }

    /// Calculate maximum fillable quantity based on bar volume.
    #[must_use]
    pub fn max_fillable_quantity(&self, bar_volume: f64) -> i64 {
        let max_qty = (bar_volume * self.max_participation) as i64;
        // Round down to B3 lot size
        (max_qty / 100) * 100
    }

    /// Get actual fill quantity considering liquidity constraints.
    #[must_use]
    pub fn get_fill_quantity(&self, requested_qty: i64, bar_volume: f64) -> i64 {
        let max_qty = self.max_fillable_quantity(bar_volume).max(100);
        let abs_requested = requested_qty.unsigned_abs() as i64;

        if self.allow_partial_fills {
            let fill_qty = abs_requested.min(max_qty);
            // Round to B3 lot size
            let rounded = (fill_qty / 100) * 100;
            if requested_qty >= 0 {
                rounded
            } else {
                -rounded
            }
        } else if abs_requested <= max_qty {
            requested_qty
        } else {
            0 // Can't fill
        }
    }
}

impl Default for LiquidityModel {
    fn default() -> Self {
        Self::new(0.1, true)
    }
}

// =============================================================================
// EXECUTION CONFIG
// =============================================================================

/// Configuration for the execution model.
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Slippage model.
    pub slippage: SlippageModel,
    /// Cost model.
    pub costs: CostModel,
    /// Liquidity model.
    pub liquidity: LiquidityModel,
    /// Delay in bars before execution.
    pub delay_bars: u8,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            slippage: SlippageModel::default(),
            costs: CostModel::default(),
            liquidity: LiquidityModel::default(),
            delay_bars: 1,
        }
    }
}

impl ExecutionConfig {
    /// Create with B3 defaults.
    #[must_use]
    pub fn b3_default() -> Self {
        Self {
            slippage: SlippageModel::Constant { bps: 5.0 },
            costs: CostModel::b3_default(),
            liquidity: LiquidityModel::new(0.1, true),
            delay_bars: 1,
        }
    }

    /// Create with no slippage/costs (for testing).
    #[must_use]
    pub fn zero_cost() -> Self {
        Self {
            slippage: SlippageModel::None,
            costs: CostModel::new(0.0, 0.0, 0.0, 0.0),
            liquidity: LiquidityModel::new(1.0, true),
            delay_bars: 0,
        }
    }

    /// Create from serializable config.
    #[must_use]
    pub fn from_model_config(config: &ExecutionModelConfig) -> Self {
        if config.bypass_for_debug {
            return Self::zero_cost();
        }
        Self {
            slippage: SlippageModel::from_config(&config.slippage),
            costs: CostModel::from_config(&config.fees),
            liquidity: LiquidityModel::from_config(&config.fill_policy),
            delay_bars: config.delay_bars,
        }
    }
}

// =============================================================================
// ADVANCED EXECUTION MODEL
// =============================================================================

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

    /// Create with B3 defaults.
    #[must_use]
    pub fn b3_default() -> Self {
        Self::new(ExecutionConfig::b3_default())
    }

    /// Create from serializable config.
    #[must_use]
    pub fn from_model_config(config: &ExecutionModelConfig) -> Self {
        Self::new(ExecutionConfig::from_model_config(config))
    }

    /// Get the delay in bars.
    #[must_use]
    pub fn delay_bars(&self) -> u8 {
        self.config.delay_bars
    }

    /// Check if a limit order can be filled based on bar range.
    fn check_limit_order(&self, order: &OrderEvent, bar: &Bar) -> bool {
        match (order.order_type, order.limit_price) {
            (OrderType::Limit, Some(limit_price)) => match order.direction {
                OrderDirection::Buy => bar.low <= limit_price,
                OrderDirection::Sell => bar.high >= limit_price,
            },
            _ => true, // Market orders always attempt to fill
        }
    }

    /// Get execution price for an order.
    fn get_fill_price(&self, order: &OrderEvent, bar: &Bar) -> f64 {
        match (order.order_type, order.limit_price) {
            (OrderType::Limit, Some(limit_price)) => limit_price,
            _ => {
                // Apply slippage for market orders
                self.config.slippage.apply(bar.close, order, bar)
            }
        }
    }
}

impl ExecutionModel for AdvancedExecutionModel {
    fn execute(
        &self,
        order: &OrderEvent,
        current_bar: &Bar,
        next_id: &mut u64,
    ) -> Option<FillEvent> {
        // Check limit order conditions
        if !self.check_limit_order(order, current_bar) {
            return None;
        }

        // Check liquidity
        let fill_qty = self
            .config
            .liquidity
            .get_fill_quantity(order.signed_quantity(), current_bar.volume);
        if fill_qty == 0 {
            return None;
        }

        // Get fill price
        let fill_price = self.get_fill_price(order, current_bar);

        // Calculate slippage amount
        let slippage = if order.order_type == OrderType::Market {
            self.config.slippage.calculate(order, current_bar)
        } else {
            0.0
        };

        // Calculate costs
        #[allow(clippy::cast_precision_loss)]
        let notional = fill_qty.unsigned_abs() as f64 * fill_price;
        let commission = self.config.costs.calculate(notional, fill_qty);

        let fill_id = FillId::new(*next_id);
        *next_id += 1;

        Some(FillEvent::new(
            fill_id,
            order.order_id,
            order.timestamp,
            order.asset_id,
            order.direction,
            fill_qty.abs(),
            fill_price,
            commission,
            slippage,
        ))
    }
}

// =============================================================================
// SIMPLE EXECUTION MODEL (Legacy/Testing)
// =============================================================================

/// Simple execution model with fixed costs and optional slippage.
#[derive(Debug, Clone)]
pub struct SimpleExecutionModel {
    /// Fixed cost per trade.
    pub fixed_cost: f64,
    /// Proportional cost (e.g., 0.001 = 0.1%).
    pub proportional_cost: f64,
}

impl SimpleExecutionModel {
    /// Create a new simple execution model.
    #[must_use]
    pub const fn new(fixed_cost: f64, proportional_cost: f64) -> Self {
        Self {
            fixed_cost,
            proportional_cost,
        }
    }

    /// Create with zero costs (for testing).
    #[must_use]
    pub fn zero_cost() -> Self {
        Self::new(0.0, 0.0)
    }
}

impl Default for SimpleExecutionModel {
    fn default() -> Self {
        Self::new(10.0, 0.001)
    }
}

impl ExecutionModel for SimpleExecutionModel {
    fn execute(
        &self,
        order: &OrderEvent,
        current_bar: &Bar,
        next_id: &mut u64,
    ) -> Option<FillEvent> {
        let price = current_bar.close;
        let quantity = order.quantity;

        #[allow(clippy::cast_precision_loss)]
        let notional = quantity as f64 * price;
        let commission = self.fixed_cost + notional * self.proportional_cost;

        let fill_id = FillId::new(*next_id);
        *next_id += 1;

        Some(FillEvent::new(
            fill_id,
            order.order_id,
            order.timestamp,
            order.asset_id,
            order.direction,
            quantity,
            price,
            commission,
            0.0, // No slippage in simple model
        ))
    }
}

// =============================================================================
// TESTS
// =============================================================================

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
        let direction = if quantity >= 0 {
            OrderDirection::Buy
        } else {
            OrderDirection::Sell
        };
        let order_type = if limit.is_some() {
            OrderType::Limit
        } else {
            OrderType::Market
        };

        OrderEvent {
            order_id: OrderId::new(1),
            timestamp: 0,
            asset_id: AssetId::new(0),
            direction,
            quantity: quantity.abs(),
            order_type,
            limit_price: limit,
            stop_price: None,
            time_in_force: backtester_core::TimeInForce::Day,
        }
    }

    #[test]
    fn slippage_constant() {
        let model = SlippageModel::Constant { bps: 10.0 };
        let bar = make_bar(100.0, 1000.0);

        let buy_order = make_order(100, None);
        let slip = model.calculate(&buy_order, &bar);
        assert!((slip - 0.1).abs() < 0.001);

        let sell_order = make_order(-100, None);
        let slip = model.calculate(&sell_order, &bar);
        assert!((slip - (-0.1)).abs() < 0.001);
    }

    #[test]
    fn slippage_volume_linear() {
        let model = SlippageModel::VolumeLinear {
            base_bps: 0.0,
            volume_factor: 1.0,
        };
        let bar = make_bar(100.0, 1000.0);
        let order = make_order(100, None);

        let slip = model.calculate(&order, &bar);
        // volume_factor * (100/1000) * 100 = 1.0 * 0.1 * 100 = 10.0
        assert!((slip - 10.0).abs() < 0.01);
    }

    #[test]
    fn cost_model_calculation() {
        let model = CostModel::new(10.0, 0.001, 0.01, 0.00035);
        let cost = model.calculate(10_000.0, 100);
        // 10 + (10000 * 0.001) + (10000 * 0.00035) + (100 * 0.01)
        // = 10 + 10 + 3.5 + 1 = 24.5
        assert!((cost - 24.5).abs() < 0.01);
    }

    #[test]
    fn cost_breakdown() {
        let model = CostModel::new(10.0, 0.001, 0.01, 0.00035);
        let breakdown = model.calculate_breakdown(10_000.0, 100);

        assert!((breakdown.fixed - 10.0).abs() < 0.01);
        assert!((breakdown.commission - 10.0).abs() < 0.01);
        assert!((breakdown.emolument - 3.5).abs() < 0.01);
        assert!((breakdown.per_unit - 1.0).abs() < 0.01);
        assert!((breakdown.total() - 24.5).abs() < 0.01);
    }

    #[test]
    fn liquidity_model_limits_fill() {
        let model = LiquidityModel::new(0.1, true);

        // Order 500, volume 1000, max = 100
        let fill_qty = model.get_fill_quantity(500, 1000.0);
        assert_eq!(fill_qty, 100);

        // Order 100, volume 1000, fills fully
        let fill_qty = model.get_fill_quantity(100, 1000.0);
        assert_eq!(fill_qty, 100);
    }

    #[test]
    fn liquidity_no_partial_rejects_large() {
        let model = LiquidityModel::new(0.1, false);

        // Order 500, volume 1000, max = 100, no partial -> reject
        let fill_qty = model.get_fill_quantity(500, 1000.0);
        assert_eq!(fill_qty, 0);
    }

    #[test]
    fn limit_order_fills_when_in_range() {
        let config = ExecutionConfig::zero_cost();
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
        let mut next_id = 1u64;
        let fill = model.execute(&buy_limit, &bar, &mut next_id);
        assert!(fill.is_some());
        assert!((fill.unwrap().price - 99.0).abs() < f64::EPSILON);

        // Buy limit at 97 should NOT fill (low = 98)
        let buy_limit_miss = make_order(100, Some(97.0));
        let fill = model.execute(&buy_limit_miss, &bar, &mut next_id);
        assert!(fill.is_none());
    }

    #[test]
    fn simple_execution_calculates_costs() {
        let model = SimpleExecutionModel::new(10.0, 0.001);
        let order = make_order(100, None);
        let bar = make_bar(50.0, 1000.0);
        let mut next_id = 1u64;

        let fill = model.execute(&order, &bar, &mut next_id).unwrap();
        // Cost = 10 + (100 * 50 * 0.001) = 10 + 5 = 15
        assert!((fill.commission - 15.0).abs() < f64::EPSILON);
        assert_eq!(fill.quantity, 100);
        assert!((fill.price - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn execution_config_presets() {
        let b3 = ExecutionConfig::b3_default();
        assert!(matches!(b3.slippage, SlippageModel::Constant { .. }));

        let zero = ExecutionConfig::zero_cost();
        assert!(matches!(zero.slippage, SlippageModel::None));
    }

    #[test]
    fn from_model_config() {
        let model_config = ExecutionModelConfig::mvp();
        let exec_config = ExecutionConfig::from_model_config(&model_config);
        assert_eq!(exec_config.delay_bars, 1);
        assert!(matches!(exec_config.slippage, SlippageModel::Constant { .. }));
    }

    // =========================================================================
    // Phase 2.1: Comprehensive Slippage Model Validation
    // =========================================================================

    #[test]
    fn test_slippage_none_zero_impact() {
        let model = SlippageModel::None;
        let bar = make_bar(100.0, 1000.0);
        
        let buy_order = make_order(100, None);
        let slip = model.calculate(&buy_order, &bar);
        assert_eq!(slip, 0.0, "None slippage should be zero");
        
        let sell_order = make_order(-500, None);
        let slip = model.calculate(&sell_order, &bar);
        assert_eq!(slip, 0.0, "None slippage should be zero for sells too");
    }

    #[test]
    fn test_slippage_constant_symmetric() {
        let model = SlippageModel::Constant { bps: 20.0 };
        let bar = make_bar(100.0, 1000.0);
        
        let buy_order = make_order(100, None);
        let sell_order = make_order(-100, None);
        
        let buy_slip = model.calculate(&buy_order, &bar);
        let sell_slip = model.calculate(&sell_order, &bar);
        
        // Buy slippage should be positive (worse price)
        assert!(buy_slip > 0.0, "Buy slippage should be positive: {}", buy_slip);
        // Sell slippage should be negative (worse price for seller)
        assert!(sell_slip < 0.0, "Sell slippage should be negative: {}", sell_slip);
        // Absolute values should be equal (symmetric)
        assert!((buy_slip.abs() - sell_slip.abs()).abs() < 0.001,
            "Slippage should be symmetric: {} vs {}", buy_slip, sell_slip);
    }

    #[test]
    fn test_slippage_constant_scales_with_price() {
        let model = SlippageModel::Constant { bps: 10.0 }; // 0.1%
        let order = make_order(100, None);
        
        let bar_100 = make_bar(100.0, 1000.0);
        let bar_200 = make_bar(200.0, 1000.0);
        
        let slip_100 = model.calculate(&order, &bar_100);
        let slip_200 = model.calculate(&order, &bar_200);
        
        // Slippage should be proportional to price
        assert!((slip_200 / slip_100 - 2.0).abs() < 0.01,
            "Slippage should scale with price: {} / {} = {}", slip_200, slip_100, slip_200 / slip_100);
    }

    #[test]
    fn test_slippage_volume_monotonic_in_order_size() {
        let model = SlippageModel::VolumeLinear {
            base_bps: 5.0,
            volume_factor: 0.5,
        };
        let bar = make_bar(100.0, 10000.0);
        
        let order_100 = make_order(100, None);
        let order_500 = make_order(500, None);
        let order_1000 = make_order(1000, None);
        
        let slip_100 = model.calculate(&order_100, &bar);
        let slip_500 = model.calculate(&order_500, &bar);
        let slip_1000 = model.calculate(&order_1000, &bar);
        
        assert!(slip_500 > slip_100, "Larger order should have more slippage: {} > {}", slip_500, slip_100);
        assert!(slip_1000 > slip_500, "Larger order should have more slippage: {} > {}", slip_1000, slip_500);
    }

    #[test]
    fn test_slippage_volume_order_exceeds_bar_volume() {
        let model = SlippageModel::VolumeLinear {
            base_bps: 0.0,
            volume_factor: 1.0,
        };
        let bar = make_bar(100.0, 100.0); // Low volume
        
        // Order size > bar volume
        let large_order = make_order(500, None);
        let slip = model.calculate(&large_order, &bar);
        
        // Should handle gracefully (high slippage but not infinite)
        assert!(slip.is_finite(), "Slippage should be finite even for large orders");
        assert!(slip > 0.0, "Should have positive slippage");
    }

    #[test]
    fn test_slippage_volume_zero_volume_bar() {
        let model = SlippageModel::VolumeLinear {
            base_bps: 5.0,
            volume_factor: 0.5,
        };
        let bar = Bar {
            timestamp: 0,
            open: 100.0,
            high: 102.0,
            low: 98.0,
            close: 100.0,
            volume: 0.0, // Zero volume
        };
        
        let order = make_order(100, None);
        let slip = model.calculate(&order, &bar);
        
        // Should fall back to base slippage
        assert!(slip.is_finite(), "Slippage should be finite for zero volume");
        assert!((slip - 0.05).abs() < 0.01, "Should use base slippage only: {}", slip);
    }

    #[test]
    fn test_slippage_volatility_proportional_to_range() {
        let model = SlippageModel::Volatility {
            base_bps: 0.0,
            vol_factor: 1.0,
        };
        
        let order = make_order(100, None);
        
        // Low volatility bar: H-L = 2% of price
        let low_vol_bar = Bar {
            timestamp: 0,
            open: 100.0,
            high: 101.0,
            low: 99.0,
            close: 100.0,
            volume: 1000.0,
        };
        
        // High volatility bar: H-L = 10% of price
        let high_vol_bar = Bar {
            timestamp: 0,
            open: 100.0,
            high: 105.0,
            low: 95.0,
            close: 100.0,
            volume: 1000.0,
        };
        
        let slip_low = model.calculate(&order, &low_vol_bar);
        let slip_high = model.calculate(&order, &high_vol_bar);
        
        assert!(slip_high > slip_low, 
            "High vol bar should have more slippage: {} > {}", slip_high, slip_low);
    }

    #[test]
    fn test_slippage_volatility_zero_range() {
        let model = SlippageModel::Volatility {
            base_bps: 10.0,
            vol_factor: 0.5,
        };
        
        // Zero range bar (H = L)
        let flat_bar = Bar {
            timestamp: 0,
            open: 100.0,
            high: 100.0,
            low: 100.0,
            close: 100.0,
            volume: 1000.0,
        };
        
        let order = make_order(100, None);
        let slip = model.calculate(&order, &flat_bar);
        
        // Should fall back to base slippage
        assert!(slip.is_finite(), "Slippage should be finite for zero range");
        assert!((slip - 0.10).abs() < 0.01, "Should use base slippage: {}", slip);
    }

    #[test]
    fn test_slippage_order_size_zero() {
        let model = SlippageModel::Constant { bps: 10.0 };
        let bar = make_bar(100.0, 1000.0);
        
        // Zero quantity order
        let zero_order = OrderEvent {
            timestamp: 0,
            order_id: OrderId::new(1),
            asset_id: AssetId::new(0),
            direction: OrderDirection::Buy,
            quantity: 0,
            order_type: OrderType::Market,
            limit_price: None,
            stop_price: None,
            time_in_force: backtester_core::TimeInForce::Day,
        };
        
        let slip = model.calculate(&zero_order, &bar);
        // Zero quantity should still calculate slippage based on price
        assert!(slip.is_finite(), "Zero quantity should give finite slippage");
    }

    #[test]
    fn test_cost_model_zero_inputs() {
        let model = CostModel::new(0.0, 0.0, 0.0, 0.0);
        let cost = model.calculate(10_000.0, 100);
        assert_eq!(cost, 0.0, "Zero cost model should give zero cost");
    }

    #[test]
    fn test_cost_model_components_additive() {
        let model = CostModel::new(10.0, 0.001, 0.01, 0.0005);
        let breakdown = model.calculate_breakdown(10_000.0, 100);
        
        let component_sum = breakdown.fixed + breakdown.commission + 
                           breakdown.emolument + breakdown.per_unit;
        
        assert!((breakdown.total() - component_sum).abs() < 0.001,
            "Total should equal sum of components: {} vs {}", 
            breakdown.total(), component_sum);
    }

    #[test]
    fn test_liquidity_model_edge_cases() {
        // Very high participation rate
        let high_rate = LiquidityModel::new(0.9, true);
        let fill = high_rate.get_fill_quantity(1000, 1000.0);
        assert_eq!(fill, 900, "90% participation should fill 900");
        
        // Small participation rate - minimum fill is 100 (B3 round lot)
        let small_rate = LiquidityModel::new(0.01, true);
        let fill = small_rate.get_fill_quantity(100, 10000.0);
        // min(100, max(0.01 * 10000, 100)) = min(100, 100) = 100
        assert_eq!(fill, 100, "Should respect B3 round lot minimum");
        
        // Large order with small participation - capped by participation
        let fill = high_rate.get_fill_quantity(2000, 1000.0);
        // max fill = max(0.9 * 1000, 100) = 900
        assert_eq!(fill, 900, "Should cap at 90% of volume");
    }
}

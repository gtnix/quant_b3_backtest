//! # Backtester Core
//!
//! Fundamental types, traits, and event definitions for the backtesting engine.
//! This crate is the central dependency for all other crates in the workspace.

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod simd;

use std::fmt;

// =============================================================================
// IDENTIFIERS
// =============================================================================

/// Asset identifier (0-indexed, cache-friendly).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct AssetId(pub u16);

impl AssetId {
    /// Create new asset ID.
    #[must_use]
    pub const fn new(id: u16) -> Self {
        AssetId(id)
    }

    /// Get as usize for array indexing.
    #[must_use]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }

    /// Get raw value.
    #[must_use]
    pub const fn raw(self) -> u16 {
        self.0
    }
}

impl From<u16> for AssetId {
    fn from(id: u16) -> Self {
        AssetId(id)
    }
}

impl From<u32> for AssetId {
    fn from(id: u32) -> Self {
        AssetId(id as u16)
    }
}

impl From<usize> for AssetId {
    fn from(id: usize) -> Self {
        AssetId(id as u16)
    }
}

/// Order identifier for tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct OrderId(pub u64);

impl OrderId {
    /// Create new order ID.
    #[must_use]
    pub const fn new(id: u64) -> Self {
        OrderId(id)
    }

    /// Get raw value.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Fill identifier for tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct FillId(pub u64);

impl FillId {
    /// Create new fill ID.
    #[must_use]
    pub const fn new(id: u64) -> Self {
        FillId(id)
    }
}

/// Timestamp in nanoseconds since UNIX epoch (UTC).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Timestamp(pub i64);

impl Timestamp {
    /// Create from nanoseconds since epoch.
    #[must_use]
    pub const fn from_nanos(nanos: i64) -> Self {
        Timestamp(nanos)
    }

    /// Create from seconds since epoch.
    #[must_use]
    pub const fn from_secs(secs: i64) -> Self {
        Timestamp(secs * 1_000_000_000)
    }

    /// Create from milliseconds since epoch.
    #[must_use]
    pub const fn from_millis(millis: i64) -> Self {
        Timestamp(millis * 1_000_000)
    }

    /// Get as nanoseconds.
    #[must_use]
    pub const fn as_nanos(self) -> i64 {
        self.0
    }

    /// Get as seconds.
    #[must_use]
    pub const fn as_secs(self) -> i64 {
        self.0 / 1_000_000_000
    }

    /// Get as milliseconds.
    #[must_use]
    pub const fn as_millis(self) -> i64 {
        self.0 / 1_000_000
    }

    /// Zero timestamp.
    pub const ZERO: Timestamp = Timestamp(0);

    /// Maximum timestamp.
    pub const MAX: Timestamp = Timestamp(i64::MAX);
}

impl From<i64> for Timestamp {
    fn from(nanos: i64) -> Self {
        Timestamp(nanos)
    }
}

// =============================================================================
// ERRORS
// =============================================================================

/// Backtest error types.
#[derive(Debug, Clone, PartialEq)]
pub enum BacktestError {
    /// Invalid bar data (e.g., high < low).
    InvalidBar(String),
    /// Invalid order (e.g., zero quantity).
    InvalidOrder(String),
    /// Insufficient cash for operation.
    InsufficientCash(String),
    /// Position not found.
    PositionNotFound(String),
    /// Data loading/parsing error.
    DataError(String),
    /// Execution error (e.g., liquidity).
    ExecutionError(String),
    /// Calculation error.
    CalculationError(String),
    /// Configuration error.
    ConfigError(String),
}

impl fmt::Display for BacktestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BacktestError::InvalidBar(msg) => write!(f, "Invalid Bar: {msg}"),
            BacktestError::InvalidOrder(msg) => write!(f, "Invalid Order: {msg}"),
            BacktestError::InsufficientCash(msg) => write!(f, "Insufficient Cash: {msg}"),
            BacktestError::PositionNotFound(msg) => write!(f, "Position Not Found: {msg}"),
            BacktestError::DataError(msg) => write!(f, "Data Error: {msg}"),
            BacktestError::ExecutionError(msg) => write!(f, "Execution Error: {msg}"),
            BacktestError::CalculationError(msg) => write!(f, "Calculation Error: {msg}"),
            BacktestError::ConfigError(msg) => write!(f, "Config Error: {msg}"),
        }
    }
}

impl std::error::Error for BacktestError {}

/// Result type for backtest operations.
pub type Result<T> = std::result::Result<T, BacktestError>;

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Backtest configuration.
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital.
    pub initial_capital: f64,
    /// Start date (timestamp).
    pub start_date: Timestamp,
    /// End date (timestamp).
    pub end_date: Timestamp,
    /// Maximum leverage allowed (1.0 = no leverage).
    pub max_leverage: f64,
    /// Risk-free rate for Sharpe calculation (annualized).
    pub risk_free_rate: f64,
    /// Benchmark asset ID (optional).
    pub benchmark_id: Option<AssetId>,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        BacktestConfig {
            initial_capital: 100_000.0,
            start_date: Timestamp::ZERO,
            end_date: Timestamp::MAX,
            max_leverage: 1.0,
            risk_free_rate: 0.05,
            benchmark_id: None,
        }
    }
}

/// Execution configuration.
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Fixed cost per order (e.g., R$ 10).
    pub fixed_cost: f64,
    /// Commission rate (e.g., 0.001 = 0.1%).
    pub commission_rate: f64,
    /// Cost per share/unit (e.g., R$ 0.01).
    pub cost_per_unit: f64,
    /// B3 emolument rate (e.g., 0.00035 = 0.035%).
    pub emolument_rate: f64,
    /// Slippage in basis points (e.g., 10 = 0.1%).
    pub slippage_bps: f64,
    /// Maximum participation rate (% of bar volume).
    pub max_participation: f64,
    /// Allow partial fills.
    pub allow_partial_fills: bool,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        ExecutionConfig {
            fixed_cost: 10.0,
            commission_rate: 0.001,
            cost_per_unit: 0.01,
            emolument_rate: 0.000_35,
            slippage_bps: 10.0,
            max_participation: 0.1,
            allow_partial_fills: true,
        }
    }
}

// =============================================================================
// MARKET DATA
// =============================================================================

/// OHLCV bar data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bar {
    /// Timestamp in nanoseconds (UTC).
    pub timestamp: i64,
    /// Opening price.
    pub open: f64,
    /// Highest price.
    pub high: f64,
    /// Lowest price.
    pub low: f64,
    /// Closing price.
    pub close: f64,
    /// Volume.
    pub volume: f64,
}

impl Bar {
    /// Create a new bar with validation.
    pub fn new(
        timestamp: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Result<Self> {
        if high < low {
            return Err(BacktestError::InvalidBar("high < low".to_string()));
        }
        if open < low || open > high {
            return Err(BacktestError::InvalidBar(
                "open outside [low, high]".to_string(),
            ));
        }
        if close < low || close > high {
            return Err(BacktestError::InvalidBar(
                "close outside [low, high]".to_string(),
            ));
        }
        Ok(Bar {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        })
    }

    /// Calculate typical price (H+L+C)/3.
    #[must_use]
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate bar range (high - low).
    #[must_use]
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate VWAP approximation.
    #[must_use]
    pub fn vwap_approx(&self) -> f64 {
        self.typical_price()
    }
}

/// Market data event (bar update for an asset).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarketEvent {
    /// Asset identifier.
    pub asset_id: AssetId,
    /// Bar data.
    pub bar: Bar,
}

impl MarketEvent {
    /// Create new market event.
    #[must_use]
    pub const fn new(asset_id: AssetId, bar: Bar) -> Self {
        MarketEvent { asset_id, bar }
    }
}

// =============================================================================
// SIGNALS
// =============================================================================

/// Signal type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalType {
    /// Buy signal.
    Buy,
    /// Sell signal.
    Sell,
    /// Hold / no action.
    Hold,
    /// Close existing position.
    Close,
}

/// Signal direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalDirection {
    /// Buy signal.
    Buy,
    /// Sell signal.
    Sell,
    /// Neutral / no signal.
    Neutral,
}

/// Signal generated by a strategy.
#[derive(Debug, Clone, PartialEq)]
pub struct SignalEvent {
    /// Timestamp when signal was generated.
    pub timestamp: i64,
    /// Target asset.
    pub asset_id: AssetId,
    /// Signal strength (-1.0 to 1.0).
    pub strength: f64,
    /// Signal type.
    pub signal_type: SignalType,
    /// Optional metadata.
    pub metadata: Option<String>,
}

impl SignalEvent {
    /// Create a buy signal.
    #[must_use]
    pub fn buy(timestamp: i64, asset_id: AssetId, strength: f64) -> Self {
        SignalEvent {
            timestamp,
            asset_id,
            strength: strength.clamp(0.0, 1.0),
            signal_type: SignalType::Buy,
            metadata: None,
        }
    }

    /// Create a sell signal.
    #[must_use]
    pub fn sell(timestamp: i64, asset_id: AssetId, strength: f64) -> Self {
        SignalEvent {
            timestamp,
            asset_id,
            strength: strength.clamp(-1.0, 0.0).abs(),
            signal_type: SignalType::Sell,
            metadata: None,
        }
    }

    /// Create a close signal.
    #[must_use]
    pub fn close(timestamp: i64, asset_id: AssetId) -> Self {
        SignalEvent {
            timestamp,
            asset_id,
            strength: 1.0,
            signal_type: SignalType::Close,
            metadata: None,
        }
    }

    /// Create a hold signal.
    #[must_use]
    pub fn hold(timestamp: i64, asset_id: AssetId) -> Self {
        SignalEvent {
            timestamp,
            asset_id,
            strength: 0.0,
            signal_type: SignalType::Hold,
            metadata: None,
        }
    }

    /// Get signal direction from strength.
    #[must_use]
    pub fn direction(&self) -> SignalDirection {
        match self.signal_type {
            SignalType::Buy => SignalDirection::Buy,
            SignalType::Sell => SignalDirection::Sell,
            SignalType::Close | SignalType::Hold => {
                if self.strength > 0.0 {
                    SignalDirection::Buy
                } else if self.strength < 0.0 {
                    SignalDirection::Sell
                } else {
                    SignalDirection::Neutral
                }
            }
        }
    }

    /// Add metadata to signal.
    #[must_use]
    pub fn with_metadata(mut self, metadata: String) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

// =============================================================================
// ORDERS
// =============================================================================

/// Order direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderDirection {
    /// Buy order.
    Buy,
    /// Sell order.
    Sell,
}

/// Order side (alias for clarity).
pub type Side = OrderDirection;

/// Order type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    /// Market order - execute at best available price.
    Market,
    /// Limit order - execute at specified price or better.
    Limit,
    /// Stop order - trigger market order at stop price.
    Stop,
    /// Stop-limit order - trigger limit order at stop price.
    StopLimit,
}

/// Time in force for order validity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeInForce {
    /// Valid until end of day.
    Day,
    /// Good till cancelled.
    GTC,
    /// Immediate or cancel.
    IOC,
    /// Fill or kill.
    FOK,
}

impl Default for TimeInForce {
    fn default() -> Self {
        TimeInForce::Day
    }
}

/// Order status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderStatus {
    /// Order created and pending.
    Pending,
    /// Order fully filled.
    Filled,
    /// Order partially filled.
    PartialFill,
    /// Order cancelled.
    Cancelled,
    /// Order rejected.
    Rejected,
}

/// Order to be executed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrderEvent {
    /// Order ID.
    pub order_id: OrderId,
    /// Timestamp when order was created.
    pub timestamp: i64,
    /// Target asset.
    pub asset_id: AssetId,
    /// Order direction.
    pub direction: OrderDirection,
    /// Quantity (absolute value, always positive).
    pub quantity: i64,
    /// Order type.
    pub order_type: OrderType,
    /// Limit price (for limit/stop-limit orders).
    pub limit_price: Option<f64>,
    /// Stop price (for stop/stop-limit orders).
    pub stop_price: Option<f64>,
    /// Time in force.
    pub time_in_force: TimeInForce,
}

impl OrderEvent {
    /// Create a market buy order.
    #[must_use]
    pub fn market_buy(order_id: OrderId, timestamp: i64, asset_id: AssetId, quantity: i64) -> Self {
        OrderEvent {
            order_id,
            timestamp,
            asset_id,
            direction: OrderDirection::Buy,
            quantity: quantity.abs(),
            order_type: OrderType::Market,
            limit_price: None,
            stop_price: None,
            time_in_force: TimeInForce::Day,
        }
    }

    /// Create a market sell order.
    #[must_use]
    pub fn market_sell(
        order_id: OrderId,
        timestamp: i64,
        asset_id: AssetId,
        quantity: i64,
    ) -> Self {
        OrderEvent {
            order_id,
            timestamp,
            asset_id,
            direction: OrderDirection::Sell,
            quantity: quantity.abs(),
            order_type: OrderType::Market,
            limit_price: None,
            stop_price: None,
            time_in_force: TimeInForce::Day,
        }
    }

    /// Create a limit buy order.
    #[must_use]
    pub fn limit_buy(
        order_id: OrderId,
        timestamp: i64,
        asset_id: AssetId,
        quantity: i64,
        limit_price: f64,
    ) -> Self {
        OrderEvent {
            order_id,
            timestamp,
            asset_id,
            direction: OrderDirection::Buy,
            quantity: quantity.abs(),
            order_type: OrderType::Limit,
            limit_price: Some(limit_price),
            stop_price: None,
            time_in_force: TimeInForce::GTC,
        }
    }

    /// Create a limit sell order.
    #[must_use]
    pub fn limit_sell(
        order_id: OrderId,
        timestamp: i64,
        asset_id: AssetId,
        quantity: i64,
        limit_price: f64,
    ) -> Self {
        OrderEvent {
            order_id,
            timestamp,
            asset_id,
            direction: OrderDirection::Sell,
            quantity: quantity.abs(),
            order_type: OrderType::Limit,
            limit_price: Some(limit_price),
            stop_price: None,
            time_in_force: TimeInForce::GTC,
        }
    }

    /// Get signed quantity (positive for buy, negative for sell).
    #[must_use]
    pub fn signed_quantity(&self) -> i64 {
        match self.direction {
            OrderDirection::Buy => self.quantity,
            OrderDirection::Sell => -self.quantity,
        }
    }

    /// Validate B3 round-lot requirement (multiple of 100).
    #[must_use]
    pub fn is_valid_b3_lot(&self) -> bool {
        self.quantity > 0 && self.quantity % 100 == 0
    }

    /// Get order side.
    #[must_use]
    pub fn side(&self) -> Side {
        self.direction
    }
}

// =============================================================================
// FILLS
// =============================================================================

/// Executed fill.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FillEvent {
    /// Fill ID.
    pub fill_id: FillId,
    /// Order ID that generated this fill.
    pub order_id: OrderId,
    /// Timestamp when fill occurred.
    pub timestamp: i64,
    /// Asset that was filled.
    pub asset_id: AssetId,
    /// Fill direction.
    pub direction: OrderDirection,
    /// Filled quantity (absolute).
    pub quantity: i64,
    /// Execution price.
    pub price: f64,
    /// Commission paid.
    pub commission: f64,
    /// Slippage incurred.
    pub slippage: f64,
    /// Total cost (commission + slippage).
    pub total_cost: f64,
}

impl FillEvent {
    /// Create a new fill.
    #[must_use]
    pub fn new(
        fill_id: FillId,
        order_id: OrderId,
        timestamp: i64,
        asset_id: AssetId,
        direction: OrderDirection,
        quantity: i64,
        price: f64,
        commission: f64,
        slippage: f64,
    ) -> Self {
        FillEvent {
            fill_id,
            order_id,
            timestamp,
            asset_id,
            direction,
            quantity: quantity.abs(),
            price,
            commission,
            slippage,
            total_cost: commission + slippage.abs(),
        }
    }

    /// Calculate notional value of fill.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn notional(&self) -> f64 {
        self.quantity as f64 * self.price
    }

    /// Get signed quantity.
    #[must_use]
    pub fn signed_quantity(&self) -> i64 {
        match self.direction {
            OrderDirection::Buy => self.quantity,
            OrderDirection::Sell => -self.quantity,
        }
    }
}

// =============================================================================
// EVENTS
// =============================================================================

/// Session close event (end of trading day).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionCloseEvent {
    /// Timestamp of session close.
    pub timestamp: i64,
}

/// Event types processed by the simulation engine.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// Market data update.
    Market(MarketEvent),
    /// Trading signal from strategy.
    Signal(SignalEvent),
    /// Order to execute.
    Order(OrderEvent),
    /// Executed fill.
    Fill(FillEvent),
    /// Session close event.
    SessionClose(SessionCloseEvent),
}

// =============================================================================
// STRATEGY TRAIT
// =============================================================================

/// Strategy configuration passed during initialization.
#[derive(Debug, Clone, Default)]
pub struct StrategyConfig {
    /// Strategy parameters as key-value pairs.
    pub params: Vec<(String, f64)>,
}

impl StrategyConfig {
    /// Create empty config.
    #[must_use]
    pub fn new() -> Self {
        StrategyConfig { params: Vec::new() }
    }

    /// Add a parameter.
    pub fn with_param(mut self, key: &str, value: f64) -> Self {
        self.params.push((key.to_string(), value));
        self
    }

    /// Get a parameter value.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<f64> {
        self.params.iter().find(|(k, _)| k == key).map(|(_, v)| *v)
    }

    /// Get a parameter with default value.
    #[must_use]
    pub fn get_or(&self, key: &str, default: f64) -> f64 {
        self.get(key).unwrap_or(default)
    }
}

/// Strategy trait - must be implemented by user strategies.
///
/// # Lifecycle
/// 1. `on_init` - Called once at backtest start
/// 2. `on_market` - Called for each market event (hot path)
/// 3. `on_session_close` - Called at end of each trading session
/// 4. `on_backtest_end` - Called once at backtest end
///
/// # Contracts
/// - Hot path methods (`on_market`) must be fast
/// - Must be deterministic (no unseeded RNG, no system time access)
pub trait Strategy {
    /// Called once at the start of the backtest.
    fn on_init(&mut self, _config: &StrategyConfig, _num_assets: usize) {}

    /// Called for each market event. Returns optional signal.
    fn on_market(&mut self, event: &MarketEvent) -> Option<SignalEvent>;

    /// Called at the end of each trading session.
    fn on_session_close(&mut self, _timestamp: i64, _asset_id: AssetId) -> Option<SignalEvent> {
        None
    }

    /// Called once at the end of the backtest.
    fn on_backtest_end(&mut self) {}

    /// Get strategy name.
    fn name(&self) -> &str {
        "UnnamedStrategy"
    }
}

/// Execution model trait - simulates order fills.
pub trait ExecutionModel {
    /// Simulate execution of an order, returning the fill.
    fn execute(
        &self,
        order: &OrderEvent,
        current_bar: &Bar,
        next_order_id: &mut u64,
    ) -> Option<FillEvent>;
}

// =============================================================================
// LEGACY COMPATIBILITY (type aliases)
// =============================================================================

/// Legacy timestamp alias.
pub type LegacyTimestamp = i64;

/// Legacy asset ID alias.
pub type LegacyAssetId = u32;

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bar_creation() {
        let bar = Bar {
            timestamp: 1_700_000_000_000_000_000,
            open: 100.0,
            high: 105.0,
            low: 99.0,
            close: 104.0,
            volume: 10000.0,
        };
        assert!((bar.close - 104.0).abs() < f64::EPSILON);
        assert!((bar.range() - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn bar_validation() {
        let valid = Bar::new(0, 100.0, 105.0, 99.0, 104.0, 1000.0);
        assert!(valid.is_ok());

        let invalid = Bar::new(0, 100.0, 98.0, 99.0, 104.0, 1000.0);
        assert!(invalid.is_err());
    }

    #[test]
    fn event_enum_size() {
        assert!(std::mem::size_of::<Event>() <= 128);
    }

    #[test]
    fn order_b3_lot_validation() {
        let valid = OrderEvent::market_buy(OrderId::new(1), 0, AssetId::new(0), 100);
        assert!(valid.is_valid_b3_lot());

        let invalid = OrderEvent::market_buy(OrderId::new(2), 0, AssetId::new(0), 50);
        assert!(!invalid.is_valid_b3_lot());
    }

    #[test]
    fn signal_direction() {
        let buy = SignalEvent::buy(0, AssetId::new(0), 0.5);
        assert_eq!(buy.direction(), SignalDirection::Buy);

        let sell = SignalEvent::sell(0, AssetId::new(0), 0.5);
        assert_eq!(sell.direction(), SignalDirection::Sell);
    }

    #[test]
    fn timestamp_conversions() {
        let ts = Timestamp::from_secs(1_000_000);
        assert_eq!(ts.as_secs(), 1_000_000);
        assert_eq!(ts.as_nanos(), 1_000_000_000_000_000);
    }

    #[test]
    fn asset_id_conversions() {
        let id = AssetId::new(42);
        assert_eq!(id.as_usize(), 42);
        assert_eq!(id.raw(), 42);

        let from_u32: AssetId = 100u32.into();
        assert_eq!(from_u32.raw(), 100);
    }

    #[test]
    fn order_signed_quantity() {
        let buy = OrderEvent::market_buy(OrderId::new(1), 0, AssetId::new(0), 100);
        assert_eq!(buy.signed_quantity(), 100);

        let sell = OrderEvent::market_sell(OrderId::new(2), 0, AssetId::new(0), 100);
        assert_eq!(sell.signed_quantity(), -100);
    }

    #[test]
    fn fill_notional() {
        let fill = FillEvent::new(
            FillId::new(1),
            OrderId::new(1),
            0,
            AssetId::new(0),
            OrderDirection::Buy,
            100,
            50.0,
            1.0,
            0.5,
        );
        assert!((fill.notional() - 5000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn strategy_config() {
        let config = StrategyConfig::new()
            .with_param("threshold", 0.5)
            .with_param("period", 20.0);

        assert_eq!(config.get("threshold"), Some(0.5));
        assert_eq!(config.get("missing"), None);
        assert_eq!(config.get_or("missing", 1.0), 1.0);
    }
}

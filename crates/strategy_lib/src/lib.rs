//! # Strategy Library
//!
//! User-defined trading strategies.
//!
//! This crate depends ONLY on `backtester_core` to enforce separation of concerns.
//! Strategies cannot access I/O, execution details, or portfolio internals directly.
//!
//! # Example Strategies
//! - `DailyTrendStrategy`: Moving average crossover for swing trading
//! - `MeanReversionStrategy`: VWAP-based intraday mean reversion (net-zero)
//! - `BuyAndHoldStrategy`: Simple buy-once strategy
//! - `NoOpStrategy`: Never signals (for testing)

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]

pub use backtester_core::{
    AssetId, Bar, MarketEvent, SignalEvent, Strategy, StrategyConfig, Timestamp,
};

/// Rolling window buffer for price history (fixed-size, allocation-free updates).
#[derive(Debug, Clone)]
pub struct RollingBuffer {
    buffer: Vec<f64>,
    index: usize,
    len: usize,
    capacity: usize,
}

impl RollingBuffer {
    /// Create a new rolling buffer with given capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0.0; capacity],
            index: 0,
            len: 0,
            capacity,
        }
    }

    /// Push a value into the buffer (overwrites oldest if full).
    pub fn push(&mut self, value: f64) {
        self.buffer[self.index] = value;
        self.index = (self.index + 1) % self.capacity;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    /// Check if buffer is full.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    /// Calculate simple moving average.
    #[must_use]
    pub fn sma(&self) -> f64 {
        if self.len == 0 {
            return 0.0;
        }
        self.buffer[..self.len].iter().sum::<f64>() / self.len as f64
    }

    /// Get the most recent value.
    #[must_use]
    pub fn last(&self) -> Option<f64> {
        if self.len == 0 {
            None
        } else {
            let idx = if self.index == 0 {
                self.capacity - 1
            } else {
                self.index - 1
            };
            Some(self.buffer[idx])
        }
    }

    /// Get current length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Daily trend-following strategy using moving average crossover.
///
/// # Logic
/// - Buy signal when short MA crosses above long MA
/// - Sell signal when short MA crosses below long MA
/// - Suitable for swing trading (holds positions overnight)
#[derive(Debug)]
pub struct DailyTrendStrategy {
    short_period: usize,
    long_period: usize,
    /// Price buffers per asset (for SMA calculation)
    price_buffers: Vec<RollingBuffer>,
    /// Previous short MA per asset
    prev_short_ma: Vec<f64>,
    /// Previous long MA per asset
    prev_long_ma: Vec<f64>,
    /// Whether we've signaled per asset
    has_position: Vec<bool>,
    initialized: bool,
}

impl DailyTrendStrategy {
    /// Create a new trend strategy with given MA periods.
    #[must_use]
    pub fn new(short_period: usize, long_period: usize) -> Self {
        Self {
            short_period: short_period.max(1),
            long_period: long_period.max(short_period + 1),
            price_buffers: Vec::new(),
            prev_short_ma: Vec::new(),
            prev_long_ma: Vec::new(),
            has_position: Vec::new(),
            initialized: false,
        }
    }

    /// Default configuration: 20/50 day crossover.
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(20, 50)
    }

    fn calculate_short_ma(&self, asset_id: AssetId) -> f64 {
        let id = asset_id as usize;
        if id >= self.price_buffers.len() {
            return 0.0;
        }
        let buffer = &self.price_buffers[id];
        if buffer.len() < self.short_period {
            return 0.0;
        }
        // Calculate SMA of last short_period values
        let start = if buffer.len() >= self.short_period {
            buffer.len() - self.short_period
        } else {
            0
        };
        let sum: f64 = buffer.buffer[..buffer.len()]
            .iter()
            .skip(start)
            .sum();
        sum / self.short_period as f64
    }

    fn calculate_long_ma(&self, asset_id: AssetId) -> f64 {
        let id = asset_id as usize;
        if id >= self.price_buffers.len() {
            return 0.0;
        }
        self.price_buffers[id].sma()
    }
}

impl Strategy for DailyTrendStrategy {
    fn on_init(&mut self, _config: &StrategyConfig, num_assets: usize) {
        self.price_buffers = (0..num_assets)
            .map(|_| RollingBuffer::new(self.long_period))
            .collect();
        self.prev_short_ma = vec![0.0; num_assets];
        self.prev_long_ma = vec![0.0; num_assets];
        self.has_position = vec![false; num_assets];
        self.initialized = true;
    }

    fn on_market(&mut self, event: &MarketEvent) -> Option<SignalEvent> {
        let id = event.asset_id as usize;
        
        // Auto-initialize if needed
        if !self.initialized || id >= self.price_buffers.len() {
            let new_size = (id + 1).max(10);
            self.on_init(&StrategyConfig::default(), new_size);
        }

        // Update price buffer
        self.price_buffers[id].push(event.bar.close);

        // Need enough data for long MA
        if !self.price_buffers[id].is_full() {
            return None;
        }

        let short_ma = self.calculate_short_ma(event.asset_id);
        let long_ma = self.calculate_long_ma(event.asset_id);
        let prev_short = self.prev_short_ma[id];
        let prev_long = self.prev_long_ma[id];

        // Update for next iteration
        self.prev_short_ma[id] = short_ma;
        self.prev_long_ma[id] = long_ma;

        // Skip if no previous data
        if prev_short == 0.0 || prev_long == 0.0 {
            return None;
        }

        // Detect crossover
        let was_below = prev_short < prev_long;
        let is_above = short_ma > long_ma;
        let was_above = prev_short > prev_long;
        let is_below = short_ma < long_ma;

        if was_below && is_above && !self.has_position[id] {
            // Golden cross - buy signal
            self.has_position[id] = true;
            return Some(SignalEvent {
                timestamp: event.bar.timestamp,
                asset_id: event.asset_id,
                strength: 1.0,
            });
        }

        if was_above && is_below && self.has_position[id] {
            // Death cross - sell signal
            self.has_position[id] = false;
            return Some(SignalEvent {
                timestamp: event.bar.timestamp,
                asset_id: event.asset_id,
                strength: -1.0,
            });
        }

        None
    }

    fn name(&self) -> &str {
        "DailyTrendStrategy"
    }
}

/// Intraday mean reversion strategy using VWAP.
///
/// # Logic
/// - Buy when price is significantly below VWAP
/// - Sell when price is significantly above VWAP
/// - Closes all positions at session end (net-zero)
#[derive(Debug)]
pub struct MeanReversionStrategy {
    /// Deviation threshold from VWAP to trigger signal (e.g., 0.02 = 2%)
    threshold: f64,
    /// Cumulative price*volume per asset (for VWAP)
    cum_pv: Vec<f64>,
    /// Cumulative volume per asset
    cum_vol: Vec<f64>,
    /// Current position direction per asset
    position: Vec<i8>,
    /// Number of trades today per asset
    trades_today: Vec<u32>,
    /// Maximum trades per day
    max_trades_per_day: u32,
    initialized: bool,
}

impl MeanReversionStrategy {
    /// Create a new mean reversion strategy.
    #[must_use]
    pub fn new(threshold: f64, max_trades_per_day: u32) -> Self {
        Self {
            threshold: threshold.abs().max(0.001),
            cum_pv: Vec::new(),
            cum_vol: Vec::new(),
            position: Vec::new(),
            trades_today: Vec::new(),
            max_trades_per_day,
            initialized: false,
        }
    }

    /// Default configuration: 1% threshold, max 10 trades/day.
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(0.01, 10)
    }

    fn vwap(&self, asset_id: AssetId) -> f64 {
        let id = asset_id as usize;
        if id >= self.cum_vol.len() || self.cum_vol[id] == 0.0 {
            return 0.0;
        }
        self.cum_pv[id] / self.cum_vol[id]
    }
}

impl Strategy for MeanReversionStrategy {
    fn on_init(&mut self, _config: &StrategyConfig, num_assets: usize) {
        self.cum_pv = vec![0.0; num_assets];
        self.cum_vol = vec![0.0; num_assets];
        self.position = vec![0; num_assets];
        self.trades_today = vec![0; num_assets];
        self.initialized = true;
    }

    fn on_market(&mut self, event: &MarketEvent) -> Option<SignalEvent> {
        let id = event.asset_id as usize;
        
        // Auto-initialize if needed
        if !self.initialized || id >= self.cum_vol.len() {
            let new_size = (id + 1).max(10);
            self.on_init(&StrategyConfig::default(), new_size);
        }

        // Update VWAP components
        let typical = event.bar.typical_price();
        self.cum_pv[id] += typical * event.bar.volume;
        self.cum_vol[id] += event.bar.volume;

        let vwap = self.vwap(event.asset_id);
        if vwap == 0.0 {
            return None;
        }

        // Check trade limit
        if self.trades_today[id] >= self.max_trades_per_day {
            return None;
        }

        let deviation = (event.bar.close - vwap) / vwap;

        // Entry logic
        if self.position[id] == 0 {
            if deviation < -self.threshold {
                // Price below VWAP - buy
                self.position[id] = 1;
                self.trades_today[id] += 1;
                return Some(SignalEvent {
                    timestamp: event.bar.timestamp,
                    asset_id: event.asset_id,
                    strength: 1.0,
                });
            } else if deviation > self.threshold {
                // Price above VWAP - sell (short)
                self.position[id] = -1;
                self.trades_today[id] += 1;
                return Some(SignalEvent {
                    timestamp: event.bar.timestamp,
                    asset_id: event.asset_id,
                    strength: -1.0,
                });
            }
        }

        // Exit logic - mean reversion to VWAP
        if self.position[id] == 1 && deviation >= 0.0 {
            // Long position, price back to VWAP - exit
            self.position[id] = 0;
            self.trades_today[id] += 1;
            return Some(SignalEvent {
                timestamp: event.bar.timestamp,
                asset_id: event.asset_id,
                strength: -1.0,
            });
        }

        if self.position[id] == -1 && deviation <= 0.0 {
            // Short position, price back to VWAP - exit
            self.position[id] = 0;
            self.trades_today[id] += 1;
            return Some(SignalEvent {
                timestamp: event.bar.timestamp,
                asset_id: event.asset_id,
                strength: 1.0,
            });
        }

        None
    }

    fn on_session_close(&mut self, timestamp: Timestamp, asset_id: AssetId) -> Option<SignalEvent> {
        let id = asset_id as usize;
        if id >= self.position.len() {
            return None;
        }

        // Reset daily counters
        self.cum_pv[id] = 0.0;
        self.cum_vol[id] = 0.0;
        self.trades_today[id] = 0;

        // Close any open position (net-zero requirement)
        let pos = self.position[id];
        if pos != 0 {
            self.position[id] = 0;
            return Some(SignalEvent {
                timestamp,
                asset_id,
                strength: if pos > 0 { -1.0 } else { 1.0 },
            });
        }

        None
    }

    fn name(&self) -> &str {
        "MeanReversionStrategy"
    }
}

/// Simple buy-and-hold strategy (signals buy once per asset).
#[derive(Debug, Default)]
pub struct BuyAndHoldStrategy {
    signaled: Vec<bool>,
}

impl BuyAndHoldStrategy {
    /// Create a new buy-and-hold strategy.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl Strategy for BuyAndHoldStrategy {
    fn on_init(&mut self, _config: &StrategyConfig, num_assets: usize) {
        self.signaled = vec![false; num_assets];
    }

    fn on_market(&mut self, event: &MarketEvent) -> Option<SignalEvent> {
        let id = event.asset_id as usize;
        
        // Ensure capacity
        if id >= self.signaled.len() {
            self.signaled.resize(id + 1, false);
        }

        if self.signaled[id] {
            return None;
        }
        
        self.signaled[id] = true;
        Some(SignalEvent {
            timestamp: event.bar.timestamp,
            asset_id: event.asset_id,
            strength: 1.0,
        })
    }

    fn name(&self) -> &str {
        "BuyAndHoldStrategy"
    }
}

/// No-op strategy that never signals (for testing/benchmarking).
#[derive(Debug, Default)]
pub struct NoOpStrategy;

impl Strategy for NoOpStrategy {
    fn on_market(&mut self, _event: &MarketEvent) -> Option<SignalEvent> {
        None
    }

    fn name(&self) -> &str {
        "NoOpStrategy"
    }
}

/// Pairs/Spread Intraday Strategy (Net Zero).
///
/// Trades the spread between two correlated assets.
/// - Entry: spread deviates > threshold * std_dev from mean
/// - Exit: spread returns to mean or session close
#[derive(Debug)]
pub struct PairsSpreadStrategy {
    /// Asset ID for leg A
    asset_a: AssetId,
    /// Asset ID for leg B
    asset_b: AssetId,
    /// Entry threshold in standard deviations
    entry_threshold: f64,
    /// Rolling buffer for spread values
    spread_buffer: RollingBuffer,
    /// Current prices
    price_a: f64,
    price_b: f64,
    /// Position state: 0 = flat, 1 = long spread, -1 = short spread
    position: i8,
    /// Trades today
    trades_today: u32,
    /// Max trades per day
    max_trades: u32,
    initialized: bool,
}

impl PairsSpreadStrategy {
    /// Create a new pairs strategy.
    ///
    /// # Arguments
    /// * `asset_a` - First asset in the pair
    /// * `asset_b` - Second asset in the pair
    /// * `entry_threshold` - Entry threshold in std devs (e.g., 2.0)
    /// * `lookback` - Lookback period for spread statistics
    #[must_use]
    pub fn new(asset_a: AssetId, asset_b: AssetId, entry_threshold: f64, lookback: usize) -> Self {
        Self {
            asset_a,
            asset_b,
            entry_threshold: entry_threshold.abs().max(0.5),
            spread_buffer: RollingBuffer::new(lookback.max(10)),
            price_a: 0.0,
            price_b: 0.0,
            position: 0,
            trades_today: 0,
            max_trades: 20,
            initialized: false,
        }
    }

    /// Default: PETR4/VALE3 pair, 2 std dev threshold, 20 bar lookback.
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(0, 1, 2.0, 20)
    }

    fn spread(&self) -> f64 {
        self.price_a - self.price_b
    }

    fn spread_mean(&self) -> f64 {
        self.spread_buffer.sma()
    }

    fn spread_std(&self) -> f64 {
        if self.spread_buffer.len() < 2 {
            return 0.0;
        }
        let mean = self.spread_mean();
        let variance: f64 = self.spread_buffer.buffer[..self.spread_buffer.len()]
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / self.spread_buffer.len() as f64;
        variance.sqrt()
    }

    fn z_score(&self) -> f64 {
        let std = self.spread_std();
        if std < f64::EPSILON {
            return 0.0;
        }
        (self.spread() - self.spread_mean()) / std
    }
}

impl Strategy for PairsSpreadStrategy {
    fn on_init(&mut self, _config: &StrategyConfig, _num_assets: usize) {
        self.initialized = true;
    }

    fn on_market(&mut self, event: &MarketEvent) -> Option<SignalEvent> {
        // Update prices for the relevant asset
        if event.asset_id == self.asset_a {
            self.price_a = event.bar.close;
        } else if event.asset_id == self.asset_b {
            self.price_b = event.bar.close;
        } else {
            return None; // Ignore other assets
        }

        // Need both prices to calculate spread
        if self.price_a <= 0.0 || self.price_b <= 0.0 {
            return None;
        }

        // Update spread buffer
        let current_spread = self.spread();
        self.spread_buffer.push(current_spread);

        // Need full buffer for statistics
        if !self.spread_buffer.is_full() {
            return None;
        }

        // Check trade limit
        if self.trades_today >= self.max_trades {
            return None;
        }

        let z = self.z_score();

        // Entry logic (only signal on asset_a to avoid double signals)
        if event.asset_id != self.asset_a {
            return None;
        }

        if self.position == 0 {
            if z > self.entry_threshold {
                // Spread too high: sell A, buy B → short spread
                self.position = -1;
                self.trades_today += 1;
                return Some(SignalEvent {
                    timestamp: event.bar.timestamp,
                    asset_id: self.asset_a,
                    strength: -1.0, // Sell A
                });
            } else if z < -self.entry_threshold {
                // Spread too low: buy A, sell B → long spread
                self.position = 1;
                self.trades_today += 1;
                return Some(SignalEvent {
                    timestamp: event.bar.timestamp,
                    asset_id: self.asset_a,
                    strength: 1.0, // Buy A
                });
            }
        } else {
            // Exit when spread normalizes
            if (self.position == 1 && z >= 0.0) || (self.position == -1 && z <= 0.0) {
                let exit_strength = if self.position == 1 { -1.0 } else { 1.0 };
                self.position = 0;
                self.trades_today += 1;
                return Some(SignalEvent {
                    timestamp: event.bar.timestamp,
                    asset_id: self.asset_a,
                    strength: exit_strength,
                });
            }
        }

        None
    }

    fn on_session_close(&mut self, timestamp: Timestamp, asset_id: AssetId) -> Option<SignalEvent> {
        // Only process for asset_a
        if asset_id != self.asset_a {
            return None;
        }

        // Reset daily state
        self.trades_today = 0;

        // Close any open position (net-zero)
        if self.position != 0 {
            let exit_strength = if self.position == 1 { -1.0 } else { 1.0 };
            self.position = 0;
            return Some(SignalEvent {
                timestamp,
                asset_id: self.asset_a,
                strength: exit_strength,
            });
        }

        None
    }

    fn name(&self) -> &str {
        "PairsSpreadStrategy"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(asset_id: u32, price: f64, ts: Timestamp) -> MarketEvent {
        MarketEvent {
            asset_id,
            bar: Bar {
                timestamp: ts,
                open: price,
                high: price + 0.5,
                low: price - 0.5,
                close: price,
                volume: 1000.0,
            },
        }
    }

    #[test]
    fn rolling_buffer_sma() {
        let mut buf = RollingBuffer::new(3);
        buf.push(1.0);
        buf.push(2.0);
        buf.push(3.0);
        assert!((buf.sma() - 2.0).abs() < f64::EPSILON);
        
        buf.push(4.0); // Overwrites 1.0
        assert!((buf.sma() - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn buy_and_hold_signals_once_per_asset() {
        let mut strategy = BuyAndHoldStrategy::new();
        
        let event1 = make_event(0, 100.0, 1000);
        let event2 = make_event(0, 101.0, 2000);
        let event3 = make_event(1, 50.0, 3000);

        assert!(strategy.on_market(&event1).is_some());
        assert!(strategy.on_market(&event2).is_none()); // Same asset, already signaled
        assert!(strategy.on_market(&event3).is_some()); // Different asset
    }

    #[test]
    fn noop_never_signals() {
        let mut strategy = NoOpStrategy;
        let event = make_event(0, 100.0, 1000);
        assert!(strategy.on_market(&event).is_none());
    }

    #[test]
    fn mean_reversion_closes_on_session() {
        let mut strategy = MeanReversionStrategy::new(0.01, 10);
        strategy.on_init(&StrategyConfig::default(), 2);

        // Simulate position
        strategy.position[0] = 1;

        // Session close should generate exit signal
        let signal = strategy.on_session_close(100000, 0);
        assert!(signal.is_some());
        assert!(signal.unwrap().strength < 0.0); // Sell to close long
    }

    #[test]
    fn trend_strategy_needs_warmup() {
        let mut strategy = DailyTrendStrategy::new(2, 5);
        strategy.on_init(&StrategyConfig::default(), 2);

        // Not enough data - no signal
        for i in 0..4 {
            let event = make_event(0, 100.0 + i as f64, i * 1000);
            assert!(strategy.on_market(&event).is_none());
        }
    }

    #[test]
    fn pairs_strategy_signals_on_spread_deviation() {
        let mut strategy = PairsSpreadStrategy::new(0, 1, 2.0, 5);
        strategy.on_init(&StrategyConfig::default(), 2);

        // Build spread history with stable spread
        for i in 0..5 {
            let ts = i as i64 * 1000;
            strategy.on_market(&make_event(0, 100.0, ts));
            strategy.on_market(&make_event(1, 50.0, ts));
        }

        // Now introduce a large spread deviation
        let ts = 5000;
        strategy.on_market(&make_event(1, 50.0, ts)); // B stays same
        let signal = strategy.on_market(&make_event(0, 120.0, ts)); // A spikes up

        // Should signal to short A (sell)
        assert!(signal.is_some());
        assert!(signal.unwrap().strength < 0.0);
    }

    #[test]
    fn pairs_strategy_closes_on_session() {
        let mut strategy = PairsSpreadStrategy::new(0, 1, 2.0, 5);
        strategy.on_init(&StrategyConfig::default(), 2);
        strategy.position = 1; // Simulate open position

        let signal = strategy.on_session_close(10000, 0);
        assert!(signal.is_some());
        assert_eq!(strategy.position, 0);
    }
}

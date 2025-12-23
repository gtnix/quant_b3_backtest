//! # Backtester Engine
//!
//! Simulation motor and order routing.
//!
//! Responsibilities:
//! - Process events in strict chronological order
//! - Maintain market state (SoA layout for performance)
//! - Invoke strategy on market events
//! - Route orders to execution model
//! - Enforce anti-look-ahead barriers

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]

pub use backtester_core::{
    AssetId, Bar, Event, ExecutionModel, FillEvent, MarketEvent, OrderEvent, SignalEvent, Strategy,
    Timestamp,
};
use backtester_execution::SimpleExecutionModel;
use backtester_portfolio::Portfolio;
pub use backtester_reports::NavHistory;

/// Market state with SoA layout for cache-efficient access.
/// Stores current prices and bar data indexed by AssetId.
#[derive(Debug, Clone)]
pub struct MarketState {
    /// Current close prices by AssetId
    pub close_prices: Vec<f64>,
    /// Current high prices by AssetId
    pub high_prices: Vec<f64>,
    /// Current low prices by AssetId
    pub low_prices: Vec<f64>,
    /// Current open prices by AssetId
    pub open_prices: Vec<f64>,
    /// Current volumes by AssetId
    pub volumes: Vec<f64>,
    /// Current timestamps by AssetId
    pub timestamps: Vec<Timestamp>,
    /// Last full bar by AssetId
    pub bars: Vec<Option<Bar>>,
    /// Current simulation time
    pub current_time: Timestamp,
    /// Number of assets
    num_assets: usize,
}

impl MarketState {
    /// Create a new market state for the given number of assets.
    #[must_use]
    pub fn new(num_assets: usize) -> Self {
        Self {
            close_prices: vec![0.0; num_assets],
            high_prices: vec![0.0; num_assets],
            low_prices: vec![0.0; num_assets],
            open_prices: vec![0.0; num_assets],
            volumes: vec![0.0; num_assets],
            timestamps: vec![0; num_assets],
            bars: vec![None; num_assets],
            current_time: 0,
            num_assets,
        }
    }

    /// Update state from a market event. O(1) operation.
    pub fn update(&mut self, event: &MarketEvent) {
        let id = event.asset_id as usize;
        if id >= self.num_assets {
            return;
        }
        let bar = &event.bar;
        self.close_prices[id] = bar.close;
        self.high_prices[id] = bar.high;
        self.low_prices[id] = bar.low;
        self.open_prices[id] = bar.open;
        self.volumes[id] = bar.volume;
        self.timestamps[id] = bar.timestamp;
        self.bars[id] = Some(*bar);
        self.current_time = bar.timestamp;
    }

    /// Get current price for an asset.
    #[must_use]
    pub fn get_price(&self, asset_id: AssetId) -> f64 {
        self.close_prices.get(asset_id as usize).copied().unwrap_or(0.0)
    }

    /// Get current bar for an asset.
    #[must_use]
    pub fn get_bar(&self, asset_id: AssetId) -> Option<&Bar> {
        self.bars.get(asset_id as usize).and_then(|b| b.as_ref())
    }

    /// Get number of assets.
    #[must_use]
    pub fn num_assets(&self) -> usize {
        self.num_assets
    }
}

/// Order router: converts signals to orders with sizing.
pub struct OrderRouter {
    /// Default order quantity (must be multiple of 100 for B3)
    pub default_quantity: i64,
    /// Maximum position size per asset
    pub max_position: i64,
}

impl OrderRouter {
    /// Create a new order router.
    #[must_use]
    pub fn new(default_quantity: i64) -> Self {
        // Ensure B3 round-lot compliance
        let qty = (default_quantity / 100) * 100;
        Self {
            default_quantity: qty.max(100),
            max_position: qty * 10,
        }
    }

    /// Convert a signal to an order, considering current portfolio state.
    pub fn route(
        &self,
        signal: &SignalEvent,
        portfolio: &Portfolio,
        market_state: &MarketState,
    ) -> Option<OrderEvent> {
        // Check if we have a valid price
        let price = market_state.get_price(signal.asset_id);
        if price <= 0.0 {
            return None;
        }

        // Determine order size based on signal strength and current position
        let current_position = portfolio.get_position(signal.asset_id);
        let target_quantity = if signal.strength > 0.0 {
            // Buy signal
            if current_position >= self.max_position {
                return None; // Already at max long
            }
            self.default_quantity
        } else if signal.strength < 0.0 {
            // Sell signal
            if current_position <= -self.max_position {
                return None; // Already at max short
            }
            -self.default_quantity
        } else {
            return None; // No signal
        };

        // Round to B3 lot size
        let quantity = (target_quantity / 100) * 100;
        if quantity == 0 {
            return None;
        }

        Some(OrderEvent {
            timestamp: signal.timestamp,
            asset_id: signal.asset_id,
            quantity,
            limit_price: None,
        })
    }

    /// Generate an order to close a position.
    pub fn close_position(
        &self,
        asset_id: AssetId,
        timestamp: Timestamp,
        portfolio: &Portfolio,
    ) -> Option<OrderEvent> {
        let position = portfolio.get_position(asset_id);
        if position == 0 {
            return None;
        }
        Some(OrderEvent {
            timestamp,
            asset_id,
            quantity: -position, // Opposite to close
            limit_price: None,
        })
    }
}

impl Default for OrderRouter {
    fn default() -> Self {
        Self::new(100)
    }
}

/// Backtest result summary.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Total events processed
    pub events_processed: u64,
    /// Total trades executed
    pub trades_executed: u64,
    /// Final NAV
    pub final_nav: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Total realized PnL
    pub total_realized_pnl: f64,
    /// Total costs
    pub total_costs: f64,
    /// NAV history (if tracking enabled)
    pub nav_history: Option<NavHistory>,
    /// Trade PnLs for metrics calculation
    pub trade_pnls: Vec<f64>,
}

/// Simulation engine that orchestrates the backtest.
pub struct SimulationEngine<S: Strategy, E: ExecutionModel> {
    strategy: S,
    execution_model: E,
    portfolio: Portfolio,
    market_state: MarketState,
    order_router: OrderRouter,
    events_processed: u64,
    trades_executed: u64,
    /// NAV history tracking
    nav_history: NavHistory,
    /// Track NAV at each event
    track_nav: bool,
    /// Trade PnLs for performance metrics
    trade_pnls: Vec<f64>,
}

impl<S: Strategy> SimulationEngine<S, SimpleExecutionModel> {
    /// Create engine with default execution model.
    #[must_use]
    pub fn with_defaults(strategy: S, initial_capital: f64, num_assets: usize) -> Self {
        Self::new(
            strategy,
            SimpleExecutionModel::new(10.0, 0.001),
            initial_capital,
            num_assets,
        )
    }
}

impl<S: Strategy, E: ExecutionModel> SimulationEngine<S, E> {
    /// Create a new simulation engine.
    #[must_use]
    pub fn new(
        strategy: S,
        execution_model: E,
        initial_capital: f64,
        num_assets: usize,
    ) -> Self {
        Self {
            strategy,
            execution_model,
            portfolio: Portfolio::new(initial_capital, num_assets),
            market_state: MarketState::new(num_assets),
            order_router: OrderRouter::default(),
            events_processed: 0,
            trades_executed: 0,
            nav_history: NavHistory::with_capacity(1000),
            track_nav: true,
            trade_pnls: Vec::with_capacity(100),
        }
    }

    /// Set custom order router.
    pub fn with_order_router(mut self, router: OrderRouter) -> Self {
        self.order_router = router;
        self
    }

    /// Enable or disable NAV history tracking.
    pub fn with_nav_tracking(mut self, enabled: bool) -> Self {
        self.track_nav = enabled;
        self
    }

    /// Get reference to current market state.
    #[must_use]
    pub fn market_state(&self) -> &MarketState {
        &self.market_state
    }

    /// Get reference to portfolio.
    #[must_use]
    pub fn portfolio(&self) -> &Portfolio {
        &self.portfolio
    }

    /// Process a single market event through the full pipeline.
    /// Returns the fill event if an order was executed.
    pub fn process_event(&mut self, event: &MarketEvent) -> Option<FillEvent> {
        self.events_processed += 1;

        // Step 1: Update market state
        self.market_state.update(event);

        // Step 2: Update portfolio mark-to-market
        self.portfolio.mark_to_market(&self.market_state.close_prices);

        // Step 3: Track NAV history (if enabled)
        if self.track_nav {
            self.nav_history.record(event.bar.timestamp, self.portfolio.nav());
        }

        // Step 4: Invoke strategy
        let signal = self.strategy.on_market(event)?;

        // Step 5: Route signal to order
        let order = self.order_router.route(&signal, &self.portfolio, &self.market_state)?;

        // Step 6: Execute order
        let bar = self.market_state.get_bar(order.asset_id)?;
        let fill = self.execution_model.execute(&order, bar)?;

        // Step 7: Track realized PnL before fill for trade PnL calculation
        let pnl_before = self.portfolio.total_realized_pnl();

        // Step 8: Update portfolio with fill
        self.portfolio.process_fill(&fill);
        self.trades_executed += 1;

        // Step 9: Calculate trade PnL (change in realized PnL)
        let pnl_after = self.portfolio.total_realized_pnl();
        let trade_pnl = pnl_after - pnl_before;
        if trade_pnl.abs() > f64::EPSILON {
            self.trade_pnls.push(trade_pnl);
        }

        Some(fill)
    }

    /// Run simulation over a stream of events.
    pub fn run<I: IntoIterator<Item = MarketEvent>>(&mut self, events: I) -> BacktestResult {
        for event in events {
            self.process_event(&event);
        }
        self.get_result()
    }

    /// Get current backtest result.
    #[must_use]
    pub fn get_result(&self) -> BacktestResult {
        BacktestResult {
            events_processed: self.events_processed,
            trades_executed: self.trades_executed,
            final_nav: self.portfolio.nav(),
            max_drawdown: self.portfolio.max_drawdown,
            total_realized_pnl: self.portfolio.total_realized_pnl(),
            total_costs: self.portfolio.total_costs,
            nav_history: if self.track_nav {
                Some(self.nav_history.clone())
            } else {
                None
            },
            trade_pnls: self.trade_pnls.clone(),
        }
    }

    /// Get reference to NAV history.
    #[must_use]
    pub fn nav_history(&self) -> &NavHistory {
        &self.nav_history
    }

    /// Get trade PnLs.
    #[must_use]
    pub fn trade_pnls(&self) -> &[f64] {
        &self.trade_pnls
    }

    /// Get number of events processed.
    #[must_use]
    pub fn events_processed(&self) -> u64 {
        self.events_processed
    }
}

// Legacy Engine for backwards compatibility
/// Simple engine wrapper for basic use cases.
pub struct Engine<S: Strategy> {
    strategy: S,
    events_processed: u64,
}

impl<S: Strategy> Engine<S> {
    /// Create a new engine with the given strategy.
    #[must_use]
    pub fn new(strategy: S) -> Self {
        Self {
            strategy,
            events_processed: 0,
        }
    }

    /// Process a single market event.
    pub fn process_market_event(&mut self, event: &MarketEvent) -> Option<SignalEvent> {
        self.events_processed += 1;
        self.strategy.on_market(event)
    }

    /// Get the number of events processed.
    #[must_use]
    pub fn events_processed(&self) -> u64 {
        self.events_processed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestStrategy {
        signal_on_first: bool,
        signaled: bool,
    }

    impl TestStrategy {
        fn new(signal_on_first: bool) -> Self {
            Self {
                signal_on_first,
                signaled: false,
            }
        }
    }

    impl Strategy for TestStrategy {
        fn on_market(&mut self, event: &MarketEvent) -> Option<SignalEvent> {
            if self.signal_on_first && !self.signaled {
                self.signaled = true;
                return Some(SignalEvent {
                    timestamp: event.bar.timestamp,
                    asset_id: event.asset_id,
                    strength: 1.0,
                });
            }
            None
        }
    }

    fn make_bar(ts: Timestamp, price: f64) -> Bar {
        Bar {
            timestamp: ts,
            open: price,
            high: price + 1.0,
            low: price - 1.0,
            close: price,
            volume: 1000.0,
        }
    }

    #[test]
    fn market_state_updates() {
        let mut state = MarketState::new(2);
        let event = MarketEvent {
            asset_id: 0,
            bar: make_bar(1000, 100.0),
        };
        state.update(&event);
        assert!((state.get_price(0) - 100.0).abs() < f64::EPSILON);
        assert_eq!(state.current_time, 1000);
    }

    #[test]
    fn order_router_enforces_b3_lots() {
        let router = OrderRouter::new(150);
        assert_eq!(router.default_quantity, 100); // Rounded down
    }

    #[test]
    fn simulation_engine_processes_events() {
        let strategy = TestStrategy::new(false);
        let mut engine = SimulationEngine::with_defaults(strategy, 100_000.0, 2);

        let event = MarketEvent {
            asset_id: 0,
            bar: make_bar(1000, 100.0),
        };
        engine.process_event(&event);

        assert_eq!(engine.events_processed(), 1);
    }

    #[test]
    fn simulation_engine_executes_trades() {
        let strategy = TestStrategy::new(true);
        let mut engine = SimulationEngine::with_defaults(strategy, 100_000.0, 2);

        let event = MarketEvent {
            asset_id: 0,
            bar: make_bar(1000, 100.0),
        };
        let fill = engine.process_event(&event);

        assert!(fill.is_some());
        let result = engine.get_result();
        assert_eq!(result.trades_executed, 1);
    }

    #[test]
    fn legacy_engine_works() {
        struct NoOpStrategy;
        impl Strategy for NoOpStrategy {
            fn on_market(&mut self, _: &MarketEvent) -> Option<SignalEvent> {
                None
            }
        }

        let mut engine = Engine::new(NoOpStrategy);
        let event = MarketEvent {
            asset_id: 1,
            bar: make_bar(0, 100.0),
        };
        engine.process_market_event(&event);
        assert_eq!(engine.events_processed(), 1);
    }
}

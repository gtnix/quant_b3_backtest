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
//! - Apply asset selection intelligence filters

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod parallel;
pub mod rebalancer;
pub mod unified;

pub use unified::{
    DayResult, DividendApplication, DividendEvent, DividendIndex, DualPriceBar, 
    PolicyViolation, PriceType, TraceEvent, UnifiedBacktestResult, UnifiedEngine, 
    UnifiedEngineConfig,
};

// Re-export intelligence types for convenience
pub use backtester_intelligence::{
    AssetData, AssetFilter, AssetScorer, FilterMode, FilterResult, IntelligenceConfig, ScoredAsset,
};

pub use backtester_core::{
    AssetId, BacktestConfig, Bar, Event, ExecutionModel, FillEvent, MarketEvent, OrderDirection,
    OrderEvent, OrderId, OrderType, Result, SignalEvent, SignalType, Strategy, TimeInForce,
};
use backtester_execution::SimpleExecutionModel;
use backtester_portfolio::Portfolio;
pub use backtester_reports::NavHistory;

// =============================================================================
// MARKET STATE
// =============================================================================

/// Market state with SoA layout for cache-efficient access.
#[derive(Debug, Clone)]
pub struct MarketState {
    /// Current close prices by AssetId index.
    pub close_prices: Vec<f64>,
    /// Current high prices.
    pub high_prices: Vec<f64>,
    /// Current low prices.
    pub low_prices: Vec<f64>,
    /// Current open prices.
    pub open_prices: Vec<f64>,
    /// Current volumes.
    pub volumes: Vec<f64>,
    /// Current timestamps.
    pub timestamps: Vec<i64>,
    /// Last full bar by AssetId.
    pub bars: Vec<Option<Bar>>,
    /// Current simulation time.
    pub current_time: i64,
    /// Number of assets.
    num_assets: usize,
}

impl MarketState {
    /// Create a new market state.
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

    /// Update state from market event. O(1).
    pub fn update(&mut self, event: &MarketEvent) {
        let idx = event.asset_id.as_usize();
        if idx >= self.num_assets {
            return;
        }
        let bar = &event.bar;
        self.close_prices[idx] = bar.close;
        self.high_prices[idx] = bar.high;
        self.low_prices[idx] = bar.low;
        self.open_prices[idx] = bar.open;
        self.volumes[idx] = bar.volume;
        self.timestamps[idx] = bar.timestamp;
        self.bars[idx] = Some(*bar);
        self.current_time = bar.timestamp;
    }

    /// Get current price for an asset.
    #[must_use]
    pub fn get_price(&self, asset_id: AssetId) -> f64 {
        self.close_prices
            .get(asset_id.as_usize())
            .copied()
            .unwrap_or(0.0)
    }

    /// Get current bar for an asset.
    #[must_use]
    pub fn get_bar(&self, asset_id: AssetId) -> Option<&Bar> {
        self.bars.get(asset_id.as_usize()).and_then(|b| b.as_ref())
    }

    /// Get number of assets.
    #[must_use]
    pub fn num_assets(&self) -> usize {
        self.num_assets
    }
}

// =============================================================================
// ORDER ROUTER
// =============================================================================

/// Order router: converts signals to orders with sizing.
pub struct OrderRouter {
    /// Default order quantity (multiple of 100 for B3).
    pub default_quantity: i64,
    /// Maximum position size per asset.
    pub max_position: i64,
    /// Next order ID.
    next_order_id: u64,
}

impl OrderRouter {
    /// Create a new order router.
    #[must_use]
    pub fn new(default_quantity: i64) -> Self {
        let qty = (default_quantity / 100) * 100;
        Self {
            default_quantity: qty.max(100),
            max_position: qty * 10,
            next_order_id: 1,
        }
    }

    /// Convert a signal to an order.
    pub fn route(
        &mut self,
        signal: &SignalEvent,
        portfolio: &Portfolio,
        market_state: &MarketState,
    ) -> Option<OrderEvent> {
        let price = market_state.get_price(signal.asset_id);
        if price <= 0.0 {
            return None;
        }

        let current_position = portfolio.get_position(signal.asset_id);

        let (direction, quantity) = match signal.signal_type {
            SignalType::Buy => {
                if current_position >= self.max_position {
                    return None;
                }
                (OrderDirection::Buy, self.default_quantity)
            }
            SignalType::Sell => {
                if current_position <= -self.max_position {
                    return None;
                }
                (OrderDirection::Sell, self.default_quantity)
            }
            SignalType::Close => {
                if current_position == 0 {
                    return None;
                }
                let dir = if current_position > 0 {
                    OrderDirection::Sell
                } else {
                    OrderDirection::Buy
                };
                (dir, current_position.abs())
            }
            SignalType::Hold => return None,
        };

        // Round to B3 lot size
        let rounded_qty = (quantity / 100) * 100;
        if rounded_qty == 0 {
            return None;
        }

        let order_id = OrderId::new(self.next_order_id);
        self.next_order_id += 1;

        Some(OrderEvent {
            order_id,
            timestamp: signal.timestamp,
            asset_id: signal.asset_id,
            direction,
            quantity: rounded_qty,
            order_type: OrderType::Market,
            limit_price: None,
            stop_price: None,
            time_in_force: TimeInForce::Day,
        })
    }

    /// Generate an order to close a position.
    pub fn close_position(
        &mut self,
        asset_id: AssetId,
        timestamp: i64,
        portfolio: &Portfolio,
    ) -> Option<OrderEvent> {
        let position = portfolio.get_position(asset_id);
        if position == 0 {
            return None;
        }

        let direction = if position > 0 {
            OrderDirection::Sell
        } else {
            OrderDirection::Buy
        };
        let order_id = OrderId::new(self.next_order_id);
        self.next_order_id += 1;

        Some(OrderEvent {
            order_id,
            timestamp,
            asset_id,
            direction,
            quantity: position.abs(),
            order_type: OrderType::Market,
            limit_price: None,
            stop_price: None,
            time_in_force: TimeInForce::Day,
        })
    }
}

impl Default for OrderRouter {
    fn default() -> Self {
        Self::new(100)
    }
}

// =============================================================================
// BACKTEST RESULT
// =============================================================================

/// Backtest result summary.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Total events processed.
    pub events_processed: u64,
    /// Total trades executed.
    pub trades_executed: u64,
    /// Final NAV.
    pub final_nav: f64,
    /// Maximum drawdown.
    pub max_drawdown: f64,
    /// Total realized PnL.
    pub total_realized_pnl: f64,
    /// Total costs.
    pub total_costs: f64,
    /// NAV history (if tracking enabled).
    pub nav_history: Option<NavHistory>,
    /// Trade PnLs for metrics.
    pub trade_pnls: Vec<f64>,
}

// =============================================================================
// SIMULATION ENGINE
// =============================================================================

/// Simulation engine that orchestrates the backtest.
/// 
/// # Deprecation Notice
/// 
/// This engine is deprecated in favor of [`UnifiedEngine`] which provides:
/// - Decimal precision for financial calculations
/// - Dividend cashflow support with anti-double-count policy
/// - Integration with `backtester_intelligence` modules
/// 
/// See `docs/policies/corporate_actions_pnl.md` for the dividend policy.
#[deprecated(since = "0.2.0", note = "Use UnifiedEngine instead for dividend support and Decimal precision")]
pub struct SimulationEngine<S: Strategy, E: ExecutionModel> {
    strategy: S,
    execution_model: E,
    portfolio: Portfolio,
    market_state: MarketState,
    order_router: OrderRouter,
    events_processed: u64,
    trades_executed: u64,
    nav_history: NavHistory,
    track_nav: bool,
    trade_pnls: Vec<f64>,
    next_fill_id: u64,
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
    pub fn new(strategy: S, execution_model: E, initial_capital: f64, num_assets: usize) -> Self {
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
            next_fill_id: 1,
        }
    }

    /// Set custom order router.
    #[must_use]
    pub fn with_order_router(mut self, router: OrderRouter) -> Self {
        self.order_router = router;
        self
    }

    /// Enable or disable NAV history tracking.
    #[must_use]
    pub fn with_nav_tracking(mut self, enabled: bool) -> Self {
        self.track_nav = enabled;
        self
    }

    /// Get reference to market state.
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
    pub fn process_event(&mut self, event: &MarketEvent) -> Option<FillEvent> {
        self.events_processed += 1;

        // Step 1: Update market state
        self.market_state.update(event);

        // Step 2: Mark-to-market portfolio
        self.portfolio
            .mark_to_market(&self.market_state.close_prices);

        // Step 3: Track NAV
        if self.track_nav {
            self.nav_history
                .record(event.bar.timestamp, self.portfolio.nav());
        }

        // Step 4: Invoke strategy
        let signal = self.strategy.on_market(event)?;

        // Step 5: Route signal to order
        let order = self
            .order_router
            .route(&signal, &self.portfolio, &self.market_state)?;

        // Step 6: Execute order
        let bar = self.market_state.get_bar(order.asset_id)?;
        let fill = self
            .execution_model
            .execute(&order, bar, &mut self.next_fill_id)?;

        // Step 7: Track PnL before fill
        let pnl_before = self.portfolio.total_realized_pnl();

        // Step 8: Update portfolio
        self.portfolio.process_fill(&fill);
        self.trades_executed += 1;

        // Step 9: Calculate trade PnL
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
            max_drawdown: self.portfolio.max_drawdown(),
            total_realized_pnl: self.portfolio.total_realized_pnl(),
            total_costs: self.portfolio.total_costs(),
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

    /// Get events processed count.
    #[must_use]
    pub fn events_processed(&self) -> u64 {
        self.events_processed
    }
}

// =============================================================================
// LEGACY ENGINE
// =============================================================================

/// Simple engine wrapper for basic use cases.
pub struct Engine<S: Strategy> {
    strategy: S,
    events_processed: u64,
}

impl<S: Strategy> Engine<S> {
    /// Create a new engine.
    #[must_use]
    pub fn new(strategy: S) -> Self {
        Self {
            strategy,
            events_processed: 0,
        }
    }

    /// Process a market event.
    pub fn process_market_event(&mut self, event: &MarketEvent) -> Option<SignalEvent> {
        self.events_processed += 1;
        self.strategy.on_market(event)
    }

    /// Get events processed count.
    #[must_use]
    pub fn events_processed(&self) -> u64 {
        self.events_processed
    }
}

// =============================================================================
// INTELLIGENT ENGINE
// =============================================================================

/// Engine builder with intelligence configuration.
pub struct IntelligentEngineBuilder<S: Strategy> {
    strategy: S,
    initial_capital: f64,
    num_assets: usize,
    intelligence_config: Option<IntelligenceConfig>,
    asset_data: Vec<AssetData>,
}

impl<S: Strategy> IntelligentEngineBuilder<S> {
    /// Create a new builder.
    #[must_use]
    pub fn new(strategy: S, initial_capital: f64, num_assets: usize) -> Self {
        Self {
            strategy,
            initial_capital,
            num_assets,
            intelligence_config: None,
            asset_data: Vec::new(),
        }
    }

    /// Set intelligence configuration.
    #[must_use]
    pub fn with_intelligence(mut self, config: IntelligenceConfig) -> Self {
        self.intelligence_config = Some(config);
        self
    }

    /// Set asset data for filtering.
    #[must_use]
    pub fn with_asset_data(mut self, data: Vec<AssetData>) -> Self {
        self.asset_data = data;
        self
    }

    /// Set asset data with market inferred from symbol patterns.
    /// This ensures CarryFilter uses the correct risk-free rate per market.
    #[must_use]
    pub fn with_asset_data_inferred(mut self, data: Vec<AssetData>) -> Self {
        self.asset_data = data
            .into_iter()
            .map(|mut d| {
                if d.market.is_none() {
                    d.infer_market();
                }
                d
            })
            .collect();
        self
    }

    /// Infer markets for all assets that don't have one set.
    #[must_use]
    pub fn infer_markets(mut self) -> Self {
        for data in &mut self.asset_data {
            if data.market.is_none() {
                data.infer_market();
            }
        }
        self
    }

    /// Build the simulation engine.
    #[must_use]
    pub fn build(self) -> IntelligentEngine<S> {
        let scorer = self
            .intelligence_config
            .as_ref()
            .map(AssetScorer::from_config);

        // Score and filter assets
        let selected_assets = if let (Some(scorer), false) = (&scorer, self.asset_data.is_empty()) {
            if !scorer.is_empty() {
                let scored = scorer.score_and_rank(&self.asset_data);
                scored.into_iter().map(|s| s.symbol).collect()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        IntelligentEngine {
            engine: SimulationEngine::with_defaults(
                self.strategy,
                self.initial_capital,
                self.num_assets,
            ),
            scorer,
            selected_assets,
        }
    }
}

/// Simulation engine with intelligence-based asset selection.
pub struct IntelligentEngine<S: Strategy> {
    engine: SimulationEngine<S, SimpleExecutionModel>,
    scorer: Option<AssetScorer>,
    selected_assets: Vec<String>,
}

impl<S: Strategy> IntelligentEngine<S> {
    /// Create a new intelligent engine.
    #[must_use]
    pub fn new(
        strategy: S,
        initial_capital: f64,
        num_assets: usize,
    ) -> IntelligentEngineBuilder<S> {
        IntelligentEngineBuilder::new(strategy, initial_capital, num_assets)
    }

    /// Check if an asset is selected by the intelligence filters.
    #[must_use]
    pub fn is_asset_selected(&self, symbol: &str) -> bool {
        if self.selected_assets.is_empty() {
            true // No filter = all selected
        } else {
            self.selected_assets.contains(&symbol.to_string())
        }
    }

    /// Get selected asset symbols.
    #[must_use]
    pub fn selected_assets(&self) -> &[String] {
        &self.selected_assets
    }

    /// Get the underlying engine.
    #[must_use]
    pub fn inner(&self) -> &SimulationEngine<S, SimpleExecutionModel> {
        &self.engine
    }

    /// Get mutable reference to underlying engine.
    pub fn inner_mut(&mut self) -> &mut SimulationEngine<S, SimpleExecutionModel> {
        &mut self.engine
    }

    /// Process an event (delegates to inner engine).
    pub fn process_event(&mut self, event: &MarketEvent) -> Option<FillEvent> {
        self.engine.process_event(event)
    }

    /// Run simulation (delegates to inner engine).
    pub fn run<I: IntoIterator<Item = MarketEvent>>(&mut self, events: I) -> BacktestResult {
        self.engine.run(events)
    }

    /// Get result (delegates to inner engine).
    #[must_use]
    pub fn get_result(&self) -> BacktestResult {
        self.engine.get_result()
    }

    /// Score assets using the configured intelligence.
    pub fn score_assets(&self, assets: &[AssetData]) -> Vec<ScoredAsset> {
        match &self.scorer {
            Some(scorer) => scorer.score_and_rank(assets),
            None => assets
                .iter()
                .map(|a| ScoredAsset {
                    symbol: a.symbol.clone(),
                    total_score: 1.0,
                    passed_all: true,
                    filter_results: Vec::new(),
                    rank: None,
                })
                .collect(),
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

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
                return Some(SignalEvent::buy(event.bar.timestamp, event.asset_id, 1.0));
            }
            None
        }
    }

    fn make_bar(ts: i64, price: f64) -> Bar {
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
            asset_id: AssetId::new(0),
            bar: make_bar(1000, 100.0),
        };
        state.update(&event);
        assert!((state.get_price(AssetId::new(0)) - 100.0).abs() < f64::EPSILON);
        assert_eq!(state.current_time, 1000);
    }

    #[test]
    fn order_router_enforces_b3_lots() {
        let router = OrderRouter::new(150);
        assert_eq!(router.default_quantity, 100);
    }

    #[test]
    fn simulation_engine_processes_events() {
        let strategy = TestStrategy::new(false);
        let mut engine = SimulationEngine::with_defaults(strategy, 100_000.0, 2);

        let event = MarketEvent {
            asset_id: AssetId::new(0),
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
            asset_id: AssetId::new(0),
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
            asset_id: AssetId::new(1),
            bar: make_bar(0, 100.0),
        };
        engine.process_market_event(&event);
        assert_eq!(engine.events_processed(), 1);
    }
}

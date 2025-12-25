//! # Parallel Engine
//!
//! Multi-threaded backtest engine using Rayon for parallel processing.
//! Processes multiple assets in parallel for maximum throughput.

use rayon::prelude::*;
use std::collections::HashMap;

use backtester_core::{
    AssetId, Bar, ExecutionModel, FillEvent, MarketEvent, SignalEvent, Strategy,
};
use backtester_execution::SimpleExecutionModel;
use backtester_portfolio::Portfolio;
use backtester_reports::NavHistory;

use crate::{BacktestResult, MarketState, OrderRouter};

/// Thread-safe market state snapshot for parallel processing.
#[derive(Debug, Clone)]
pub struct MarketSnapshot {
    /// Close prices by AssetId index.
    pub prices: Vec<f64>,
    /// Volumes by AssetId.
    pub volumes: Vec<f64>,
    /// Current timestamp.
    pub timestamp: i64,
}

impl MarketSnapshot {
    /// Create from MarketState.
    #[must_use]
    pub fn from_state(state: &MarketState) -> Self {
        Self {
            prices: state.close_prices.clone(),
            volumes: state.volumes.clone(),
            timestamp: state.current_time,
        }
    }

    /// Get price for asset.
    #[must_use]
    pub fn get_price(&self, asset_id: AssetId) -> f64 {
        self.prices.get(asset_id.as_usize()).copied().unwrap_or(0.0)
    }
}

/// Day batch of market events for parallel processing.
#[derive(Debug, Clone)]
pub struct DayBatch {
    /// Timestamp (day start).
    pub timestamp: i64,
    /// Events for this day, grouped by asset.
    pub events: Vec<MarketEvent>,
}

impl DayBatch {
    /// Create a new day batch.
    #[must_use]
    pub fn new(timestamp: i64) -> Self {
        Self {
            timestamp,
            events: Vec::new(),
        }
    }

    /// Add event to batch.
    pub fn push(&mut self, event: MarketEvent) {
        self.events.push(event);
    }
}

/// Group events into day batches for parallel processing.
pub fn group_by_day(events: &[MarketEvent]) -> Vec<DayBatch> {
    const NANOS_PER_DAY: i64 = 86_400_000_000_000;

    let mut batches: HashMap<i64, DayBatch> = HashMap::new();

    for event in events {
        let day = (event.bar.timestamp / NANOS_PER_DAY) * NANOS_PER_DAY;
        batches
            .entry(day)
            .or_insert_with(|| DayBatch::new(day))
            .push(event.clone());
    }

    let mut result: Vec<DayBatch> = batches.into_values().collect();
    result.sort_by_key(|b| b.timestamp);
    result
}

/// Parallel backtest result accumulator.
#[derive(Debug, Clone, Default)]
pub struct ParallelResult {
    /// Fills generated per asset.
    pub fills: Vec<FillEvent>,
    /// Signals generated.
    pub signals: Vec<SignalEvent>,
}

/// Parallel simulation engine.
/// Processes multiple assets in parallel using Rayon.
pub struct ParallelEngine<S: Strategy + Clone + Send + Sync> {
    /// Strategy (cloned per thread).
    strategy: S,
    /// Execution model.
    execution_model: SimpleExecutionModel,
    /// Portfolio (single-threaded, updated after parallel phase).
    portfolio: Portfolio,
    /// Market state.
    market_state: MarketState,
    /// Order router.
    order_router: OrderRouter,
    /// NAV history.
    nav_history: NavHistory,
    /// Events processed.
    events_processed: u64,
    /// Trades executed.
    trades_executed: u64,
    /// Trade PnLs.
    trade_pnls: Vec<f64>,
    /// Next fill ID.
    next_fill_id: u64,
    /// Track NAV.
    track_nav: bool,
}

impl<S: Strategy + Clone + Send + Sync> ParallelEngine<S> {
    /// Create a new parallel engine.
    #[must_use]
    pub fn new(strategy: S, initial_capital: f64, num_assets: usize) -> Self {
        Self {
            strategy,
            execution_model: SimpleExecutionModel::new(10.0, 0.001),
            portfolio: Portfolio::new(initial_capital, num_assets),
            market_state: MarketState::new(num_assets),
            order_router: OrderRouter::default(),
            nav_history: NavHistory::with_capacity(1000),
            events_processed: 0,
            trades_executed: 0,
            trade_pnls: Vec::with_capacity(100),
            next_fill_id: 1,
            track_nav: true,
        }
    }

    /// Set custom order router.
    #[must_use]
    pub fn with_order_router(mut self, router: OrderRouter) -> Self {
        self.order_router = router;
        self
    }

    /// Process a day's events in parallel.
    pub fn process_day(&mut self, batch: &DayBatch) {
        // Phase 1: Generate signals in parallel
        let signals: Vec<Option<SignalEvent>> = batch
            .events
            .par_iter()
            .map(|event| {
                let mut strategy = self.strategy.clone();
                strategy.on_market(event)
            })
            .collect();

        // Phase 2: Sequential update (portfolio must be single-threaded)
        for (i, event) in batch.events.iter().enumerate() {
            self.events_processed += 1;

            // Update market state
            self.market_state.update(event);

            // Mark-to-market
            self.portfolio
                .mark_to_market(&self.market_state.close_prices);

            // Track NAV
            if self.track_nav {
                self.nav_history
                    .record(event.bar.timestamp, self.portfolio.nav());
            }

            // Process signal if generated
            if let Some(ref signal) = signals[i] {
                if let Some(order) =
                    self.order_router
                        .route(signal, &self.portfolio, &self.market_state)
                {
                    if let Some(bar) = self.market_state.get_bar(order.asset_id) {
                        if let Some(fill) =
                            self.execution_model
                                .execute(&order, bar, &mut self.next_fill_id)
                        {
                            let pnl_before = self.portfolio.total_realized_pnl();
                            self.portfolio.process_fill(&fill);
                            self.trades_executed += 1;
                            let pnl_after = self.portfolio.total_realized_pnl();
                            let trade_pnl = pnl_after - pnl_before;
                            if trade_pnl.abs() > f64::EPSILON {
                                self.trade_pnls.push(trade_pnl);
                            }
                        }
                    }
                }
            }
        }

        // Update main strategy state
        for event in &batch.events {
            self.strategy.on_market(event);
        }
    }

    /// Run simulation over day batches.
    pub fn run(&mut self, batches: &[DayBatch]) -> BacktestResult {
        for batch in batches {
            self.process_day(batch);
        }
        self.get_result()
    }

    /// Run simulation over raw events (groups them internally).
    pub fn run_events(&mut self, events: &[MarketEvent]) -> BacktestResult {
        let batches = group_by_day(events);
        self.run(&batches)
    }

    /// Get current result.
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

    /// Get reference to portfolio.
    #[must_use]
    pub fn portfolio(&self) -> &Portfolio {
        &self.portfolio
    }

    /// Get reference to market state.
    #[must_use]
    pub fn market_state(&self) -> &MarketState {
        &self.market_state
    }
}

/// Batch process multiple independent backtests in parallel.
/// Each backtest runs on a separate asset with the same strategy.
pub fn parallel_multi_backtest<S, F>(
    strategy_factory: F,
    events_per_asset: &[Vec<MarketEvent>],
    initial_capital: f64,
) -> Vec<BacktestResult>
where
    S: Strategy + Clone + Send + Sync,
    F: Fn() -> S + Send + Sync,
{
    events_per_asset
        .par_iter()
        .map(|events| {
            let strategy = strategy_factory();
            let mut engine = crate::SimulationEngine::with_defaults(strategy, initial_capital, 1);
            engine.run(events.iter().cloned())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct TestStrategy;

    impl Strategy for TestStrategy {
        fn on_market(&mut self, _event: &MarketEvent) -> Option<SignalEvent> {
            None
        }
    }

    fn make_event(ts: i64, asset_id: u16, price: f64) -> MarketEvent {
        MarketEvent {
            asset_id: AssetId::new(asset_id),
            bar: Bar {
                timestamp: ts,
                open: price,
                high: price + 1.0,
                low: price - 1.0,
                close: price,
                volume: 1000.0,
            },
        }
    }

    #[test]
    fn group_by_day_works() {
        const NANOS_PER_DAY: i64 = 86_400_000_000_000;
        let events = vec![
            make_event(NANOS_PER_DAY, 0, 100.0),
            make_event(NANOS_PER_DAY + 1000, 1, 50.0),
            make_event(NANOS_PER_DAY * 2, 0, 101.0),
        ];

        let batches = group_by_day(&events);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].events.len(), 2);
        assert_eq!(batches[1].events.len(), 1);
    }

    #[test]
    fn parallel_engine_processes_events() {
        let strategy = TestStrategy;
        let mut engine = ParallelEngine::new(strategy, 100_000.0, 2);

        let events = vec![make_event(1000, 0, 100.0), make_event(2000, 1, 50.0)];

        let batches = group_by_day(&events);
        let result = engine.run(&batches);

        assert_eq!(result.events_processed, 2);
    }
}

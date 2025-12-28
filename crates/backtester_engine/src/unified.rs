//! Unified Backtest Engine
//!
//! This module provides `UnifiedEngine` - the canonical backtest engine that
//! unifies the simulation capabilities from `backtester_engine` with the
//! institutional-grade accounting from `backtester_intelligence`.
//!
//! # Design Principles
//!
//! 1. **Single Source of Truth**: All backtest logic flows through this engine
//! 2. **Decimal Precision**: Uses `rust_decimal` for all financial calculations
//! 3. **Dividend Support**: Handles corporate actions with anti-double-count policy
//! 4. **Price Separation**: Distinguishes between signals (adjusted) and valuation (raw)
//! 5. **Determinism**: Same inputs always produce same outputs

use std::collections::HashMap;

use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

use backtester_intelligence::{
    accounting::PortfolioState,
    entry::{AssetCandidate, EntryEngineConfig, Order},
    exit::{ExitEngineConfig, Position},
    filters::Market,
    orchestrator::{OrchestratorConfig, RebalanceOrchestrator, RebalanceStepResult},
    performance::{PerformanceConfig, PerformanceEngine},
};

// =============================================================================
// PRICE SEPARATION POLICY
// =============================================================================

/// Price type for anti-double-count policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PriceType {
    /// Adjusted prices for signals/indicators (includes dividend adjustments)
    Signals,
    /// Raw prices for valuation/mark-to-market (dividends enter via cashflow)
    Valuation,
}

/// Dual-price market data for a single asset.
#[derive(Debug, Clone)]
pub struct DualPriceBar {
    pub symbol: String,
    pub date: NaiveDate,
    /// Adjusted close price (for signals)
    pub adjusted_close: Decimal,
    /// Raw close price (for valuation)
    pub raw_close: Decimal,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub volume: Decimal,
}

impl DualPriceBar {
    /// Get price based on intended use (anti-double-count policy).
    #[must_use]
    pub fn get_price(&self, price_type: PriceType) -> Decimal {
        match price_type {
            PriceType::Signals => self.adjusted_close,
            PriceType::Valuation => self.raw_close,
        }
    }
}

// =============================================================================
// DIVIDEND EVENT
// =============================================================================

/// A dividend event to be applied to the portfolio.
#[derive(Debug, Clone)]
pub struct DividendEvent {
    pub symbol: String,
    pub ex_date: NaiveDate,
    pub payment_date: Option<NaiveDate>,
    pub rate: Decimal,
    pub dividend_type: String,
}

impl DividendEvent {
    pub fn new(symbol: impl Into<String>, ex_date: NaiveDate, rate: Decimal) -> Self {
        Self {
            symbol: symbol.into(),
            ex_date,
            payment_date: None,
            rate,
            dividend_type: "CASH".to_string(),
        }
    }
}

/// Result of applying a dividend.
#[derive(Debug, Clone)]
pub struct DividendApplication {
    pub symbol: String,
    pub date: NaiveDate,
    pub rate: Decimal,
    pub shares: i64,
    pub cashflow: Decimal,
}

// =============================================================================
// DIVIDEND INDEX
// =============================================================================

/// Efficient index for dividend lookup by (symbol, ex_date).
/// O(1) lookup per day.
#[derive(Debug, Clone, Default)]
pub struct DividendIndex {
    /// Map from date -> (symbol -> dividend)
    by_date: HashMap<NaiveDate, HashMap<String, DividendEvent>>,
    /// Total dividends indexed
    count: usize,
}

impl DividendIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build index from a list of dividend events.
    pub fn from_events(events: Vec<DividendEvent>) -> Self {
        let mut index = Self::new();
        for event in events {
            index.add(event);
        }
        index
    }

    /// Add a dividend event to the index.
    pub fn add(&mut self, event: DividendEvent) {
        self.by_date
            .entry(event.ex_date)
            .or_default()
            .insert(event.symbol.clone(), event);
        self.count += 1;
    }

    /// Get all dividends for a specific date. O(1).
    pub fn get_by_date(&self, date: NaiveDate) -> impl Iterator<Item = &DividendEvent> {
        self.by_date
            .get(&date)
            .into_iter()
            .flat_map(|m: &HashMap<String, DividendEvent>| m.values())
    }

    /// Get dividend for a specific symbol on a date. O(1).
    pub fn get(&self, date: NaiveDate, symbol: &str) -> Option<&DividendEvent> {
        self.by_date.get(&date)?.get(symbol)
    }

    /// Check if there are any dividends on a date.
    pub fn has_dividends(&self, date: NaiveDate) -> bool {
        self.by_date.get(&date).is_some_and(|m: &HashMap<String, DividendEvent>| !m.is_empty())
    }

    /// Total number of dividend events.
    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

// =============================================================================
// TRACE EVENTS
// =============================================================================

/// Trace event for audit trail.
#[derive(Debug, Clone)]
pub enum TraceEvent {
    DividendCredited {
        date: NaiveDate,
        symbol: String,
        rate: Decimal,
        shares: i64,
        cashflow: Decimal,
    },
    OrderExecuted {
        date: NaiveDate,
        symbol: String,
        side: String,
        shares: i64,
        price: Decimal,
        cost: Decimal,
    },
    DayProcessed {
        date: NaiveDate,
        equity: Decimal,
        cash: Decimal,
        positions: usize,
        dividend_cashflow: Decimal,
    },
}

// =============================================================================
// UNIFIED ENGINE CONFIG
// =============================================================================

/// Configuration for the unified engine.
#[derive(Debug, Clone)]
pub struct UnifiedEngineConfig {
    /// Initial capital
    pub initial_capital: Decimal,
    /// Market for default operations
    pub default_market: Market,
    /// Enable dividend processing
    pub enable_dividends: bool,
    /// Price type for valuation (Raw is correct when dividends enabled)
    pub valuation_price_type: PriceType,
    /// Entry engine configuration
    pub entry_config: EntryEngineConfig,
    /// Exit engine configuration
    pub exit_config: ExitEngineConfig,
    /// Performance engine configuration  
    pub performance_config: PerformanceConfig,
    /// Transaction cost in basis points
    pub cost_bps: Decimal,
}

impl Default for UnifiedEngineConfig {
    fn default() -> Self {
        Self {
            initial_capital: dec!(1_000_000),
            default_market: Market::BR,
            enable_dividends: true,
            valuation_price_type: PriceType::Valuation, // Raw prices for valuation
            entry_config: EntryEngineConfig::default(),
            exit_config: ExitEngineConfig::default(),
            performance_config: PerformanceConfig::default(),
            cost_bps: dec!(10), // 10 bps
        }
    }
}

/// Error when anti-double-count policy is violated.
#[derive(Debug, Clone)]
pub struct PolicyViolation {
    pub message: String,
}

impl std::fmt::Display for PolicyViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Policy violation: {}", self.message)
    }
}

impl std::error::Error for PolicyViolation {}

// =============================================================================
// UNIFIED ENGINE
// =============================================================================

/// Unified backtest engine that combines simulation with institutional accounting.
///
/// # Anti-Double-Count Policy
///
/// - Signals/indicators use **adjusted** prices (dividend-adjusted)
/// - Mark-to-market/equity curve uses **raw** prices
/// - Dividends enter portfolio as **cashflow** on ex-date
///
/// This ensures dividends are counted exactly once.
pub struct UnifiedEngine {
    /// Portfolio state with Decimal precision
    portfolio: PortfolioState,
    /// Rebalance orchestrator (entry + exit coordination)
    orchestrator: RebalanceOrchestrator,
    /// Performance tracking engine
    performance: PerformanceEngine,
    /// Dividend events index
    dividend_index: DividendIndex,
    /// Current prices (dual: adjusted + raw)
    current_prices: HashMap<String, DualPriceBar>,
    /// Trace events for audit
    trace: Vec<TraceEvent>,
    /// Daily dividend cashflows for timeseries
    daily_dividend_cashflow: Vec<(NaiveDate, Decimal)>,
    /// Cumulative dividend cashflow
    cumulative_dividend: Decimal,
    /// Configuration
    config: UnifiedEngineConfig,
    /// Current simulation date
    current_date: Option<NaiveDate>,
    /// Days processed
    days_processed: u64,
}

impl UnifiedEngine {
    /// Create a new unified engine with default configuration.
    pub fn new(initial_capital: Decimal) -> Self {
        Self::with_config(UnifiedEngineConfig {
            initial_capital,
            ..Default::default()
        })
    }

    /// Create engine with custom configuration.
    pub fn with_config(config: UnifiedEngineConfig) -> Self {
        let orchestrator_config = OrchestratorConfig {
            entry: config.entry_config.clone(),
            exit: config.exit_config.clone(),
            br_cost_bps: config.cost_bps.try_into().unwrap_or(10.0),
            us_cost_bps: config.cost_bps.try_into().unwrap_or(5.0),
        };

        Self {
            portfolio: PortfolioState::new(config.initial_capital),
            orchestrator: RebalanceOrchestrator::new(orchestrator_config),
            performance: PerformanceEngine::new(
                config.performance_config.clone(),
                config.initial_capital,
            ),
            dividend_index: DividendIndex::new(),
            current_prices: HashMap::new(),
            trace: Vec::new(),
            daily_dividend_cashflow: Vec::new(),
            cumulative_dividend: Decimal::ZERO,
            config,
            current_date: None,
            days_processed: 0,
        }
    }

    /// Load dividends for the simulation period.
    pub fn load_dividends(&mut self, dividends: Vec<DividendEvent>) {
        self.dividend_index = DividendIndex::from_events(dividends);
    }

    /// Validate that the configuration does not cause dividend double-counting.
    ///
    /// # Anti-Double-Count Policy
    ///
    /// - If dividends are enabled, valuation MUST use raw prices
    /// - Using adjusted prices for valuation WITH dividend cashflow = double count
    ///
    /// # Errors
    ///
    /// Returns `PolicyViolation` if configuration would cause double-counting.
    pub fn validate_anti_double_count(&self) -> Result<(), PolicyViolation> {
        if self.config.enable_dividends && self.config.valuation_price_type == PriceType::Signals {
            return Err(PolicyViolation {
                message: "Cannot use adjusted prices (PriceType::Signals) for valuation \
                         when dividends are enabled. This would double-count dividends. \
                         Either disable dividends or use PriceType::Valuation (raw prices)."
                    .to_string(),
            });
        }
        Ok(())
    }

    /// Get the dividend policy info for metadata tracking.
    pub fn get_policy_info(&self) -> (bool, PriceType) {
        (self.config.enable_dividends, self.config.valuation_price_type)
    }

    /// Process a single day of the backtest.
    ///
    /// Order of operations (critical for correctness):
    /// 1. Update market prices (both adjusted and raw)
    /// 2. Apply dividends BEFORE mark-to-market (cashflow on ex_date)
    /// 3. Mark-to-market with RAW prices (anti-double-count)
    /// 4. Evaluate exits
    /// 5. Evaluate entries
    /// 6. Record trace and metrics
    pub fn process_day(
        &mut self,
        date: NaiveDate,
        bars: &[DualPriceBar],
        candidates: Vec<AssetCandidate>,
    ) -> DayResult {
        self.current_date = Some(date);
        self.days_processed += 1;

        // Step 1: Update current prices
        for bar in bars {
            self.current_prices.insert(bar.symbol.clone(), bar.clone());
        }

        // Step 2: Apply dividends (BEFORE mark-to-market)
        let dividend_applications = if self.config.enable_dividends {
            self.apply_dividends(date)
        } else {
            Vec::new()
        };

        let day_dividend_cashflow: Decimal = dividend_applications
            .iter()
            .map(|d| d.cashflow)
            .sum();
        
        if day_dividend_cashflow > Decimal::ZERO {
            self.daily_dividend_cashflow.push((date, day_dividend_cashflow));
            self.cumulative_dividend += day_dividend_cashflow;
        }

        // Step 3: Mark-to-market with RAW prices (anti-double-count policy)
        let raw_prices: HashMap<String, Decimal> = self.current_prices
            .iter()
            .map(|(s, b)| (s.clone(), b.raw_close))
            .collect();
        self.portfolio.update_prices(&raw_prices);

        // Step 4-5: Execute rebalance (exits then entries)
        let positions: Vec<Position> = self.portfolio.positions.values().cloned().collect();
        let (rebalance_result, _audit) = self.orchestrator.execute_rebalance(
            date,
            self.config.default_market,
            &positions,
            candidates,
            self.portfolio.cash,
            self.portfolio.equity,
            self.portfolio.peak_equity,
        );

        // Apply rebalance orders to portfolio
        let orders_applied = self.apply_orders(date, &rebalance_result);

        // Step 6: Record trace
        self.trace.push(TraceEvent::DayProcessed {
            date,
            equity: self.portfolio.equity,
            cash: self.portfolio.cash,
            positions: self.portfolio.positions.len(),
            dividend_cashflow: day_dividend_cashflow,
        });

        DayResult {
            date,
            equity: self.portfolio.equity,
            cash: self.portfolio.cash,
            drawdown: self.portfolio.drawdown_decimal(),
            dividend_cashflow: day_dividend_cashflow,
            dividends_applied: dividend_applications,
            orders_executed: orders_applied,
            positions: self.portfolio.positions.len(),
        }
    }

    /// Apply dividends for positions held on ex_date.
    ///
    /// Policy: Dividends are credited on the EX_DATE (not payment_date).
    /// Shares held = position at END of previous day (T-1 close).
    fn apply_dividends(&mut self, date: NaiveDate) -> Vec<DividendApplication> {
        if !self.dividend_index.has_dividends(date) {
            return Vec::new();
        }

        let mut applications = Vec::new();

        for div in self.dividend_index.get_by_date(date) {
            if let Some(position) = self.portfolio.get_position(&div.symbol) {
                let shares = position.shares;
                if shares > 0 {
                    let cashflow = div.rate * Decimal::from(shares);
                    
                    // Credit dividend as cashflow
                    self.portfolio.add_cash(cashflow);
                    
                    // Record trace
                    self.trace.push(TraceEvent::DividendCredited {
                        date,
                        symbol: div.symbol.clone(),
                        rate: div.rate,
                        shares,
                        cashflow,
                    });

                    applications.push(DividendApplication {
                        symbol: div.symbol.clone(),
                        date,
                        rate: div.rate,
                        shares,
                        cashflow,
                    });
                }
            }
        }

        applications
    }

    /// Apply rebalance orders to portfolio.
    fn apply_orders(&mut self, date: NaiveDate, result: &RebalanceStepResult) -> Vec<Order> {
        let mut applied = Vec::new();
        let market = result.market;

        for order in &result.net_orders {
            let order_result = match order.side {
                backtester_intelligence::entry::OrderSide::Buy => {
                    self.portfolio.apply_buy(
                        &order.symbol,
                        order.shares,
                        order.price,
                        order.estimated_cost,
                        market,
                        date,
                    )
                }
                backtester_intelligence::entry::OrderSide::Sell => {
                    self.portfolio.apply_sell(
                        &order.symbol,
                        order.shares,
                        order.price,
                        order.estimated_cost,
                    ).map(|_| ())
                }
            };

            if order_result.is_ok() {
                self.trace.push(TraceEvent::OrderExecuted {
                    date,
                    symbol: order.symbol.clone(),
                    side: format!("{:?}", order.side),
                    shares: order.shares,
                    price: order.price,
                    cost: order.estimated_cost,
                });
                applied.push(order.clone());
            }
        }

        applied
    }

    /// Get final backtest result.
    pub fn get_result(&self) -> UnifiedBacktestResult {
        UnifiedBacktestResult {
            days_processed: self.days_processed,
            final_equity: self.portfolio.equity,
            final_cash: self.portfolio.cash,
            total_return: self.portfolio.total_return(),
            max_drawdown: self.portfolio.drawdown(),
            total_dividend_cashflow: self.cumulative_dividend,
            positions: self.portfolio.positions.len(),
            trace: self.trace.clone(),
            daily_dividends: self.daily_dividend_cashflow.clone(),
        }
    }

    /// Get trace events for audit.
    pub fn trace(&self) -> &[TraceEvent] {
        &self.trace
    }

    /// Get portfolio state.
    pub fn portfolio(&self) -> &PortfolioState {
        &self.portfolio
    }

    /// Get dividend index.
    pub fn dividend_index(&self) -> &DividendIndex {
        &self.dividend_index
    }

    /// Get daily dividend cashflows for timeseries.
    pub fn daily_dividend_cashflows(&self) -> &[(NaiveDate, Decimal)] {
        &self.daily_dividend_cashflow
    }

    /// Get cumulative dividend cashflow.
    pub fn cumulative_dividend(&self) -> Decimal {
        self.cumulative_dividend
    }
}

// =============================================================================
// RESULT TYPES
// =============================================================================

/// Result of processing a single day.
#[derive(Debug, Clone)]
pub struct DayResult {
    pub date: NaiveDate,
    pub equity: Decimal,
    pub cash: Decimal,
    pub drawdown: Decimal,
    pub dividend_cashflow: Decimal,
    pub dividends_applied: Vec<DividendApplication>,
    pub orders_executed: Vec<Order>,
    pub positions: usize,
}

/// Final backtest result from the unified engine.
#[derive(Debug, Clone)]
pub struct UnifiedBacktestResult {
    pub days_processed: u64,
    pub final_equity: Decimal,
    pub final_cash: Decimal,
    pub total_return: f64,
    pub max_drawdown: f64,
    pub total_dividend_cashflow: Decimal,
    pub positions: usize,
    pub trace: Vec<TraceEvent>,
    pub daily_dividends: Vec<(NaiveDate, Decimal)>,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dual_bar(symbol: &str, date: NaiveDate, adjusted: Decimal, raw: Decimal) -> DualPriceBar {
        DualPriceBar {
            symbol: symbol.to_string(),
            date,
            adjusted_close: adjusted,
            raw_close: raw,
            open: raw,
            high: raw,
            low: raw,
            volume: dec!(1000),
        }
    }

    #[test]
    fn test_dividend_index_lookup() {
        let mut index = DividendIndex::new();
        let date = NaiveDate::from_ymd_opt(2025, 3, 15).unwrap();
        
        index.add(DividendEvent::new("TAEE11", date, dec!(0.45)));
        index.add(DividendEvent::new("BBSE3", date, dec!(0.30)));
        
        assert!(index.has_dividends(date));
        assert_eq!(index.len(), 2);
        
        let div = index.get(date, "TAEE11").unwrap();
        assert_eq!(div.rate, dec!(0.45));
    }

    #[test]
    fn test_dividend_application() {
        let mut engine = UnifiedEngine::new(dec!(100_000));
        let date = NaiveDate::from_ymd_opt(2025, 3, 15).unwrap();
        let ex_date = NaiveDate::from_ymd_opt(2025, 3, 16).unwrap();
        
        // Load dividend
        engine.load_dividends(vec![
            DividendEvent::new("TAEE11", ex_date, dec!(0.50)),
        ]);

        // Establish position on day before ex-date
        let bar = make_dual_bar("TAEE11", date, dec!(10), dec!(10));
        engine.current_prices.insert("TAEE11".to_string(), bar);
        
        let pos = Position::new("TAEE11", Market::BR, 1000, dec!(10), date, dec!(10));
        engine.portfolio.set_position(pos);
        
        // Process ex-date
        let result = engine.process_day(
            ex_date,
            &[make_dual_bar("TAEE11", ex_date, dec!(9.50), dec!(9.50))],
            vec![],
        );

        // Should have received dividend: 0.50 * 1000 = 500
        assert_eq!(result.dividend_cashflow, dec!(500));
        assert_eq!(engine.cumulative_dividend, dec!(500));
    }

    #[test]
    fn test_anti_double_count_uses_raw_prices() {
        let mut engine = UnifiedEngine::new(dec!(100_000));
        let date = NaiveDate::from_ymd_opt(2025, 3, 15).unwrap();
        
        // Buy position
        let buy_result = engine.portfolio.apply_buy(
            "PETR4", 
            100, 
            dec!(40), 
            dec!(0), 
            Market::BR, 
            date
        );
        assert!(buy_result.is_ok());
        
        // Price update with adjusted vs raw difference
        // Adjusted = 42 (includes hypothetical dividend adjustment)
        // Raw = 41 (actual market price)
        let bar = make_dual_bar("PETR4", date, dec!(42), dec!(41));
        
        // Mark-to-market should use RAW price
        let mut prices = HashMap::new();
        prices.insert("PETR4".to_string(), bar.raw_close);
        engine.portfolio.update_prices(&prices);
        
        // Equity = 96,000 (cash after buy) + 100 * 41 = 100,100
        assert_eq!(engine.portfolio.equity, dec!(100_100));
    }

    #[test]
    fn test_no_dividend_when_no_position() {
        let mut engine = UnifiedEngine::new(dec!(100_000));
        let ex_date = NaiveDate::from_ymd_opt(2025, 3, 16).unwrap();
        
        // Load dividend but NO position
        engine.load_dividends(vec![
            DividendEvent::new("TAEE11", ex_date, dec!(0.50)),
        ]);

        // Process ex-date with no position
        let result = engine.process_day(
            ex_date,
            &[make_dual_bar("TAEE11", ex_date, dec!(10), dec!(10))],
            vec![],
        );

        // No dividend should be applied
        assert_eq!(result.dividend_cashflow, Decimal::ZERO);
    }

    #[test]
    fn test_determinism() {
        // Run same scenario twice, results must match
        fn run_scenario() -> UnifiedBacktestResult {
            let mut engine = UnifiedEngine::new(dec!(100_000));
            let d1 = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
            let d2 = NaiveDate::from_ymd_opt(2025, 1, 2).unwrap();
            
            engine.load_dividends(vec![
                DividendEvent::new("VALE3", d2, dec!(1.00)),
            ]);

            // Day 1: buy position
            let pos = Position::new("VALE3", Market::BR, 100, dec!(50), d1, dec!(50));
            engine.portfolio.set_position(pos);
            engine.portfolio.cash -= dec!(5000); // simulate buy
            
            engine.process_day(d1, &[make_dual_bar("VALE3", d1, dec!(50), dec!(50))], vec![]);
            engine.process_day(d2, &[make_dual_bar("VALE3", d2, dec!(49), dec!(49))], vec![]);
            
            engine.get_result()
        }

        let r1 = run_scenario();
        let r2 = run_scenario();

        assert_eq!(r1.final_equity, r2.final_equity);
        assert_eq!(r1.total_dividend_cashflow, r2.total_dividend_cashflow);
    }
}


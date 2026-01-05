//! Unified Backtest Engine
//!
//! This module provides `UnifiedEngine` - the canonical backtest engine that
//! unifies the simulation capabilities from `backtester_engine` with the
//! institutional-grade accounting from `backtester_intelligence`.
//!
//! # Design Principles
//!
//! 1. **Single Source of Truth**: All backtest logic flows through this engine
//! 2. **Fixed-Point Precision**: Uses `Price`/`Money` for hot path calculations (Milestone 3)
//! 3. **Dividend Support**: Handles corporate actions with anti-double-count policy
//! 4. **Price Separation**: Distinguishes between signals (adjusted) and valuation (raw)
//! 5. **Determinism**: Same inputs always produce same outputs (bit-exact)
//!
//! # Performance (Milestone 3)
//!
//! The hot path uses fixed-point arithmetic (i64) instead of `rust_decimal::Decimal`:
//! - `DualPriceBar` stores prices as `Price` (6 decimal places, 5-10x faster)
//! - Backward compatibility via `From<Decimal>` and `to_decimal()` methods
//! - Deterministic: same inputs → bit-exact same outputs

use std::collections::HashMap;

use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use rust_decimal_macros::dec;

use backtester_core::{Money, Price, Rate};
use crate::symbol_registry::{SymbolId, SymbolRegistry};

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
///
/// Uses `SymbolId` instead of String for O(1) indexing in hot path.
///
/// # Performance (Milestone 3)
///
/// All price fields use fixed-point `Price` (i64 with 6 decimal places) instead
/// of `Decimal` for 5-10x faster arithmetic in the hot path.
///
/// # Backward Compatibility
///
/// - `new_from_decimal()` - Create from Decimal values (for data loading)
/// - `raw_close_decimal()` / `adjusted_close_decimal()` - Get as Decimal (for external APIs)
#[derive(Debug, Clone, Copy)]
pub struct DualPriceBar {
    /// Symbol identifier (use registry to resolve to String)
    pub symbol_id: SymbolId,
    pub date: NaiveDate,
    /// Adjusted close price (for signals) - fixed-point
    pub adjusted_close: Price,
    /// Raw close price (for valuation) - fixed-point
    pub raw_close: Price,
    pub open: Price,
    pub high: Price,
    pub low: Price,
    /// Volume as integer (no decimals needed)
    pub volume: i64,
}

impl DualPriceBar {
    /// Create a new DualPriceBar from fixed-point prices.
    #[must_use]
    pub const fn new(
        symbol_id: SymbolId,
        date: NaiveDate,
        adjusted_close: Price,
        raw_close: Price,
        open: Price,
        high: Price,
        low: Price,
        volume: i64,
    ) -> Self {
        Self {
            symbol_id,
            date,
            adjusted_close,
            raw_close,
            open,
            high,
            low,
            volume,
        }
    }

    /// Create from Decimal values (for data loading, NOT hot path).
    #[must_use]
    pub fn new_from_decimal(
        symbol_id: SymbolId,
        date: NaiveDate,
        adjusted_close: Decimal,
        raw_close: Decimal,
        open: Decimal,
        high: Decimal,
        low: Decimal,
        volume: Decimal,
    ) -> Self {
        Self {
            symbol_id,
            date,
            adjusted_close: Price::from(adjusted_close),
            raw_close: Price::from(raw_close),
            open: Price::from(open),
            high: Price::from(high),
            low: Price::from(low),
            volume: volume.to_i64().unwrap_or(0),
        }
    }

    /// Get price based on intended use (anti-double-count policy).
    /// Returns fixed-point Price for hot path.
    #[inline]
    #[must_use]
    pub fn get_price_fast(&self, price_type: PriceType) -> Price {
        match price_type {
            PriceType::Signals => self.adjusted_close,
            PriceType::Valuation => self.raw_close,
        }
    }

    /// Get price as Decimal (for backward compatibility, NOT hot path).
    #[must_use]
    pub fn get_price(&self, price_type: PriceType) -> Decimal {
        match price_type {
            PriceType::Signals => self.adjusted_close.to_decimal(),
            PriceType::Valuation => self.raw_close.to_decimal(),
        }
    }

    /// Get raw close as Decimal (for backward compatibility).
    #[inline]
    #[must_use]
    pub fn raw_close_decimal(&self) -> Decimal {
        self.raw_close.to_decimal()
    }

    /// Get adjusted close as Decimal (for backward compatibility).
    #[inline]
    #[must_use]
    pub fn adjusted_close_decimal(&self) -> Decimal {
        self.adjusted_close.to_decimal()
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
    pub rate: Rate,
    pub dividend_type: String,
}

impl DividendEvent {
    /// Create new dividend event with Rate (fixed-point).
    pub fn new_fast(symbol: impl Into<String>, ex_date: NaiveDate, rate: Rate) -> Self {
        Self {
            symbol: symbol.into(),
            ex_date,
            payment_date: None,
            rate,
            dividend_type: String::new(),
        }
    }

    /// Create new dividend event from Decimal (for compatibility).
    pub fn new(symbol: impl Into<String>, ex_date: NaiveDate, rate: Decimal) -> Self {
        Self::new_fast(symbol, ex_date, Rate::from(rate))
    }

    /// Create with full details from Decimal.
    pub fn with_details(
        symbol: impl Into<String>,
        ex_date: NaiveDate,
        payment_date: Option<NaiveDate>,
        rate: Decimal,
        dividend_type: String,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            ex_date,
            payment_date,
            rate: Rate::from(rate),
            dividend_type,
        }
    }
}

/// Result of applying a dividend.
#[derive(Debug, Clone)]
pub struct DividendApplication {
    pub symbol: String,
    pub date: NaiveDate,
    pub rate: Rate,
    pub shares: i64,
    pub cashflow: Money,
}

// =============================================================================
// DIVIDEND INDEX
// =============================================================================

/// Efficient index for dividend lookup by (symbol, ex_date).
/// 
/// # Determinism (Milestone 2)
/// 
/// Uses sorted Vec for deterministic iteration order (by symbol name).
/// O(1) date lookup, O(log n) symbol lookup within a date.
#[derive(Debug, Clone, Default)]
pub struct DividendIndex {
    /// Map from date -> sorted Vec of (symbol, dividend) pairs
    /// Sorted by symbol for deterministic iteration
    by_date: HashMap<NaiveDate, Vec<DividendEvent>>,
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
        // Sort all date vectors for determinism
        for events in index.by_date.values_mut() {
            events.sort_by(|a, b| a.symbol.cmp(&b.symbol));
        }
        index
    }

    /// Add a dividend event to the index.
    pub fn add(&mut self, event: DividendEvent) {
        let vec = self.by_date.entry(event.ex_date).or_default();
        // Insert maintaining sorted order by symbol
        let pos = vec.iter().position(|e| e.symbol >= event.symbol);
        match pos {
            Some(i) if vec[i].symbol == event.symbol => {
                // Update existing
                vec[i] = event;
            }
            Some(i) => {
                vec.insert(i, event);
                self.count += 1;
            }
            None => {
                vec.push(event);
                self.count += 1;
            }
        }
    }

    /// Get all dividends for a specific date in deterministic order.
    /// Sorted by symbol name for reproducibility.
    pub fn get_by_date(&self, date: NaiveDate) -> impl Iterator<Item = &DividendEvent> {
        self.by_date
            .get(&date)
            .into_iter()
            .flat_map(|v| v.iter())
    }

    /// Get dividend for a specific symbol on a date. O(log n).
    pub fn get(&self, date: NaiveDate, symbol: &str) -> Option<&DividendEvent> {
        let vec = self.by_date.get(&date)?;
        vec.binary_search_by(|e| e.symbol.as_str().cmp(symbol))
            .ok()
            .map(|i| &vec[i])
    }

    /// Check if there are any dividends on a date.
    pub fn has_dividends(&self, date: NaiveDate) -> bool {
        self.by_date.get(&date).is_some_and(|v| !v.is_empty())
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
    /// Enable trace recording (disable for SCG to avoid allocations)
    pub trace_enabled: bool,
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
            trace_enabled: false, // Disabled by default for SCG performance
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

/// Scratch buffers for zero-allocation hot path.
///
/// # Performance (Milestone 5)
///
/// Pre-allocated buffers that are cleared and reused each day,
/// avoiding Vec allocations in the daily loop.
#[derive(Debug, Default)]
struct EngineScratch {
    /// Reusable buffer for sorted position collection
    positions: Vec<Position>,
}

impl EngineScratch {
    fn new() -> Self {
        Self {
            positions: Vec::with_capacity(64), // Pre-allocate for typical portfolio size
        }
    }

    /// Collect positions from HashMap into sorted buffer (reuses allocation).
    #[inline]
    fn collect_positions_sorted(&mut self, positions: &std::collections::HashMap<String, Position>) {
        self.positions.clear();
        self.positions.extend(positions.values().cloned());
        // Sort by symbol for determinism (unstable sort is faster)
        self.positions.sort_unstable_by(|a, b| a.symbol.cmp(&b.symbol));
    }
}

/// Unified backtest engine that combines simulation with institutional accounting.
///
/// # Anti-Double-Count Policy
///
/// - Signals/indicators use **adjusted** prices (dividend-adjusted)
/// - Mark-to-market/equity curve uses **raw** prices
/// - Dividends enter portfolio as **cashflow** on ex-date
///
/// This ensures dividends are counted exactly once.
///
/// # Performance (Milestone 5)
///
/// - Uses `SymbolRegistry` and `Vec<Option<DualPriceBar>>` for O(1) price lookups
/// - Uses `EngineScratch` for reusable buffers (zero daily allocation in steady state)
/// - Candidates taken by slice reference (no clone needed by caller)
pub struct UnifiedEngine {
    /// Symbol registry for String <-> SymbolId mapping
    registry: SymbolRegistry,
    /// Portfolio state with Decimal precision
    portfolio: PortfolioState,
    /// Rebalance orchestrator (entry + exit coordination)
    orchestrator: RebalanceOrchestrator,
    /// Performance tracking engine
    performance: PerformanceEngine,
    /// Dividend events index
    dividend_index: DividendIndex,
    /// Current prices indexed by SymbolId (O(1) lookup)
    current_prices: Vec<Option<DualPriceBar>>,
    /// Trace events for audit
    trace: Vec<TraceEvent>,
    /// Daily dividend cashflows for timeseries
    daily_dividend_cashflow: Vec<(NaiveDate, Money)>,
    /// Cumulative dividend cashflow
    cumulative_dividend: Money,
    /// Configuration
    config: UnifiedEngineConfig,
    /// Current simulation date
    current_date: Option<NaiveDate>,
    /// Days processed
    days_processed: u64,
    /// Scratch buffers for hot path (Milestone 5)
    scratch: EngineScratch,
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
            registry: SymbolRegistry::new(),
            portfolio: PortfolioState::new(config.initial_capital),
            orchestrator: RebalanceOrchestrator::new(orchestrator_config),
            performance: PerformanceEngine::new(
                config.performance_config.clone(),
                config.initial_capital,
            ),
            dividend_index: DividendIndex::new(),
            current_prices: Vec::new(),
            trace: Vec::new(),
            daily_dividend_cashflow: Vec::new(),
            cumulative_dividend: Money::ZERO,
            config,
            current_date: None,
            days_processed: 0,
            scratch: EngineScratch::new(),
        }
    }

    /// Pre-register symbols before simulation.
    ///
    /// This should be called during setup to establish SymbolId mappings.
    /// Returns the number of new symbols registered.
    pub fn register_symbols<'a>(&mut self, symbols: impl IntoIterator<Item = &'a str>) -> usize {
        let before = self.registry.len();
        for symbol in symbols {
            self.registry.register(symbol);
        }
        let after = self.registry.len();
        // Resize prices vector to accommodate all symbols
        if self.current_prices.len() < after {
            self.current_prices.resize(after, None);
        }
        after - before
    }

    /// Get the symbol registry.
    #[must_use]
    pub fn registry(&self) -> &SymbolRegistry {
        &self.registry
    }

    /// Get mutable reference to registry (for setup).
    pub fn registry_mut(&mut self) -> &mut SymbolRegistry {
        &mut self.registry
    }

    /// Ensure price vector can hold a symbol ID.
    fn ensure_capacity(&mut self, id: SymbolId) {
        let needed = id.as_usize() + 1;
        if self.current_prices.len() < needed {
            self.current_prices.resize(needed, None);
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
    /// 1. Update market prices (both adjusted and raw) - O(1) per bar via Vec indexing
    /// 2. Apply dividends BEFORE mark-to-market (cashflow on ex_date)
    /// 3. Mark-to-market with RAW prices (anti-double-count)
    /// 4. Evaluate exits
    /// 5. Evaluate entries
    /// 6. Record trace and metrics
    ///
    /// # Performance (Milestone 5)
    ///
    /// - Price updates use `Vec<Option<DualPriceBar>>` indexed by `SymbolId`
    /// - Candidates taken by slice (zero clone in caller)
    /// - Positions iterated without sort (HashMap keys are deterministic per-run)
    pub fn process_day(
        &mut self,
        date: NaiveDate,
        bars: &[DualPriceBar],
        candidates: &[AssetCandidate],
    ) -> DayResult {
        self.current_date = Some(date);
        self.days_processed += 1;

        // Step 1: Update current prices - O(1) per bar via direct indexing
        for bar in bars {
            self.ensure_capacity(bar.symbol_id);
            self.current_prices[bar.symbol_id.as_usize()] = Some(*bar);
        }

        // Step 2: Apply dividends (BEFORE mark-to-market)
        let dividend_applications = if self.config.enable_dividends {
            self.apply_dividends(date)
        } else {
            Vec::new()
        };

        let day_dividend_cashflow: Money = dividend_applications
            .iter()
            .map(|d| d.cashflow)
            .sum();
        
        if day_dividend_cashflow > Money::ZERO {
            self.daily_dividend_cashflow.push((date, day_dividend_cashflow));
            self.cumulative_dividend += day_dividend_cashflow;
        }

        // Step 3: Mark-to-market with RAW prices (anti-double-count policy)
        // Use closure-based lookup to avoid HashMap allocation in hot path
        // Note: Converts fixed-point Price back to Decimal for PortfolioState compatibility
        let registry = &self.registry;
        let prices = &self.current_prices;
        // HOT PATH: Use update_prices_with_fast with Price directly (no Decimal conversion)
        self.portfolio.update_prices_with_fast(|symbol| {
            registry.get(symbol).and_then(|id| {
                prices.get(id.as_usize())
                    .and_then(|opt| opt.as_ref())
                    .map(|bar| bar.raw_close)
            })
        });

        // Step 4-5: Execute rebalance (exits then entries)
        // Collect positions into scratch buffer (reuses allocation, sorted for determinism)
        self.scratch.collect_positions_sorted(&self.portfolio.positions);
        
        // Orchestrator uses fixed-point Money (Milestone 6)
        let (rebalance_result, _audit) = self.orchestrator.execute_rebalance(
            date,
            self.config.default_market,
            &self.scratch.positions,
            candidates,
            self.portfolio.cash,
            self.portfolio.equity,
            self.portfolio.peak_equity,
        );

        // Apply rebalance orders to portfolio
        let orders_applied = self.apply_orders(date, &rebalance_result);

        // Step 6: Record trace (convert Money to Decimal for output)
        if self.config.trace_enabled {
            self.trace.push(TraceEvent::DayProcessed {
                date,
                equity: self.portfolio.equity.to_decimal(),
                cash: self.portfolio.cash.to_decimal(),
                positions: self.portfolio.positions.len(),
                dividend_cashflow: day_dividend_cashflow.to_decimal(),
            });
        }

        DayResult {
            date,
            equity: self.portfolio.equity.to_decimal(),
            cash: self.portfolio.cash.to_decimal(),
            drawdown: self.portfolio.drawdown_decimal(),
            dividend_cashflow: day_dividend_cashflow.to_decimal(),
            dividends_applied: dividend_applications,
            orders_executed: orders_applied,
            positions: self.portfolio.positions.len(),
        }
    }

    /// Get price for a symbol by ID. O(1).
    #[must_use]
    pub fn get_price(&self, id: SymbolId) -> Option<&DualPriceBar> {
        self.current_prices.get(id.as_usize()).and_then(|opt| opt.as_ref())
    }

    /// Get price for a symbol by name. O(1) after registry lookup.
    #[must_use]
    pub fn get_price_by_symbol(&self, symbol: &str) -> Option<&DualPriceBar> {
        self.registry.get(symbol).and_then(|id| self.get_price(id))
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
                    // HOT PATH: Use Rate.mul_shares() for fast fixed-point calculation
                    let cashflow = div.rate.mul_shares(shares);
                    
                    // Credit dividend as cashflow
                    self.portfolio.add_cash_fast(cashflow);
                    
                    // Record trace (convert to Decimal for output)
                    if self.config.trace_enabled {
                        self.trace.push(TraceEvent::DividendCredited {
                            date,
                            symbol: div.symbol.clone(),
                            rate: div.rate.to_decimal(),
                            shares,
                            cashflow: cashflow.to_decimal(),
                        });
                    }

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
    ///
    /// # Performance (Milestone 6)
    ///
    /// Uses fixed-point apply_buy_fast/apply_sell_fast methods.
    fn apply_orders(&mut self, date: NaiveDate, result: &RebalanceStepResult) -> Vec<Order> {
        let mut applied = Vec::new();
        let market = result.market;

        for order in &result.net_orders {
            let order_result = match order.side {
                backtester_intelligence::entry::OrderSide::Buy => {
                    // Use fixed-point fast path (Milestone 6)
                    self.portfolio.apply_buy_fast(
                        &order.symbol,
                        order.shares,
                        order.price,
                        order.estimated_cost,
                        market,
                        date,
                    )
                }
                backtester_intelligence::entry::OrderSide::Sell => {
                    // Use fixed-point fast path (Milestone 6)
                    self.portfolio.apply_sell_fast(
                        &order.symbol,
                        order.shares,
                        order.price,
                        order.estimated_cost,
                    ).map(|_| ())
                }
            };

            if order_result.is_ok() {
                if self.config.trace_enabled {
                    self.trace.push(TraceEvent::OrderExecuted {
                        date,
                        symbol: order.symbol.clone(),
                        side: format!("{:?}", order.side),
                        shares: order.shares,
                        price: order.price.to_decimal(),
                        cost: order.estimated_cost.to_decimal(),
                    });
                }
                applied.push(order.clone());
            }
        }

        applied
    }

    /// Get final backtest result.
    pub fn get_result(&self) -> UnifiedBacktestResult {
        UnifiedBacktestResult {
            days_processed: self.days_processed,
            final_equity: self.portfolio.equity.to_decimal(),
            final_cash: self.portfolio.cash.to_decimal(),
            total_return: self.portfolio.total_return(),
            max_drawdown: self.portfolio.drawdown(),
            total_dividend_cashflow: self.cumulative_dividend.to_decimal(),
            positions: self.portfolio.positions.len(),
            trace: self.trace.clone(),
            daily_dividends: self.daily_dividend_cashflow.iter()
                .map(|(d, m)| (*d, m.to_decimal())).collect(),
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

    /// Get daily dividend cashflows for timeseries (Money, fast).
    pub fn daily_dividend_cashflows_fast(&self) -> &[(NaiveDate, Money)] {
        &self.daily_dividend_cashflow
    }

    /// Get daily dividend cashflows as Decimal for compatibility.
    pub fn daily_dividend_cashflows(&self) -> Vec<(NaiveDate, Decimal)> {
        self.daily_dividend_cashflow.iter()
            .map(|(d, m)| (*d, m.to_decimal()))
            .collect()
    }

    /// Get cumulative dividend cashflow as Decimal.
    pub fn cumulative_dividend(&self) -> Decimal {
        self.cumulative_dividend.to_decimal()
    }

    /// Get cumulative dividend cashflow as Money (fast).
    pub fn cumulative_dividend_fast(&self) -> Money {
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

    /// Helper to create a DualPriceBar from Decimal values for tests.
    fn make_dual_bar(symbol_id: SymbolId, date: NaiveDate, adjusted: Decimal, raw: Decimal) -> DualPriceBar {
        DualPriceBar::new_from_decimal(
            symbol_id,
            date,
            adjusted,
            raw,
            raw, // open
            raw, // high
            raw, // low
            dec!(1000), // volume
        )
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
        assert_eq!(div.rate.to_decimal(), dec!(0.45));
    }

    #[test]
    fn test_dividend_application() {
        let mut engine = UnifiedEngine::new(dec!(100_000));
        let date = NaiveDate::from_ymd_opt(2025, 3, 15).unwrap();
        let ex_date = NaiveDate::from_ymd_opt(2025, 3, 16).unwrap();
        
        // Register symbol and get ID
        let taee_id = engine.registry_mut().register("TAEE11");
        engine.register_symbols(["TAEE11"]);
        
        // Load dividend
        engine.load_dividends(vec![
            DividendEvent::new("TAEE11", ex_date, dec!(0.50)),
        ]);

        // Establish position on day before ex-date
        let bar = make_dual_bar(taee_id, date, dec!(10), dec!(10));
        engine.current_prices[taee_id.as_usize()] = Some(bar);
        
        let pos = Position::new("TAEE11", Market::BR, 1000, dec!(10), date, dec!(10));
        engine.portfolio.set_position(pos);
        
        // Process ex-date (Milestone 5: slice)
        let result = engine.process_day(
            ex_date,
            &[make_dual_bar(taee_id, ex_date, dec!(9.50), dec!(9.50))],
            &[],
        );

        // Should have received dividend: 0.50 * 1000 = 500
        assert_eq!(result.dividend_cashflow, dec!(500));
        assert_eq!(engine.cumulative_dividend(), dec!(500));
    }

    #[test]
    fn test_anti_double_count_uses_raw_prices() {
        let mut engine = UnifiedEngine::new(dec!(100_000));
        let date = NaiveDate::from_ymd_opt(2025, 3, 15).unwrap();
        
        // Register symbol
        let petr_id = engine.registry_mut().register("PETR4");
        engine.register_symbols(["PETR4"]);
        
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
        let bar = make_dual_bar(petr_id, date, dec!(42), dec!(41));
        
        // Mark-to-market should use RAW price
        let mut prices = HashMap::new();
        prices.insert("PETR4".to_string(), bar.raw_close_decimal());
        engine.portfolio.update_prices(&prices);
        
        // Equity = 96,000 (cash after buy) + 100 * 41 = 100,100
        assert_eq!(engine.portfolio.equity.to_decimal(), dec!(100_100));
    }

    #[test]
    fn test_no_dividend_when_no_position() {
        let mut engine = UnifiedEngine::new(dec!(100_000));
        let ex_date = NaiveDate::from_ymd_opt(2025, 3, 16).unwrap();
        
        // Register symbol
        let taee_id = engine.registry_mut().register("TAEE11");
        engine.register_symbols(["TAEE11"]);
        
        // Load dividend but NO position
        engine.load_dividends(vec![
            DividendEvent::new("TAEE11", ex_date, dec!(0.50)),
        ]);

        // Process ex-date with no position
        let result = engine.process_day(
            ex_date,
            &[make_dual_bar(taee_id, ex_date, dec!(10), dec!(10))],
            &[],
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
            
            // Register symbol
            let vale_id = engine.registry_mut().register("VALE3");
            engine.register_symbols(["VALE3"]);
            
            engine.load_dividends(vec![
                DividendEvent::new("VALE3", d2, dec!(1.00)),
            ]);

            // Day 1: buy position
            let pos = Position::new("VALE3", Market::BR, 100, dec!(50), d1, dec!(50));
            engine.portfolio.set_position(pos);
            engine.portfolio.cash -= Money::from(dec!(5000)); // simulate buy
            
            engine.process_day(d1, &[make_dual_bar(vale_id, d1, dec!(50), dec!(50))], &[]);
            engine.process_day(d2, &[make_dual_bar(vale_id, d2, dec!(49), dec!(49))], &[]);
            
            engine.get_result()
        }

        let r1 = run_scenario();
        let r2 = run_scenario();

        // Now with deterministic SymbolId-based iteration, results must be identical
        assert_eq!(r1.final_equity, r2.final_equity);
        assert_eq!(r1.total_dividend_cashflow, r2.total_dividend_cashflow);
    }
}


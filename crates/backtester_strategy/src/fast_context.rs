//! Fast Context - SoA Layout for High-Performance Strategy Execution
//!
//! This module provides a Structure-of-Arrays (SoA) layout for candidate data,
//! optimized for cache efficiency and vectorization in the hot path.
//!
//! # Performance Benefits
//!
//! - **Cache locality**: All prices in one contiguous array, all vols in another
//! - **SIMD-friendly**: Contiguous data enables auto-vectorization
//! - **Zero-alloc iteration**: No per-candidate allocation during pipeline
//! - **Preallocated buffers**: Reused across rebalance cycles

use crate::compiled::SymbolTable;
use backtester_intelligence::filters::Market;
use chrono::NaiveDate;
use rust_decimal::Decimal;

// =============================================================================
// SOA CANDIDATE DATA
// =============================================================================

/// Structure-of-Arrays layout for candidate data.
/// All fields are parallel arrays indexed by symbol ID.
#[derive(Debug, Clone)]
pub struct CandidatesSoA {
    /// Number of candidates
    len: usize,
    /// Current prices (indexed by symbol_id)
    pub prices: Vec<f64>,
    /// Annualized volatility
    pub volatilities: Vec<f64>,
    /// Momentum returns
    pub momentum_returns: Vec<f64>,
    /// Price-to-earnings ratio
    pub price_earnings: Vec<f64>,
    /// Price-to-book ratio
    pub price_to_book: Vec<f64>,
    /// Return on equity
    pub return_on_equity: Vec<f64>,
    /// Debt-to-equity ratio
    pub debt_to_equity: Vec<f64>,
    /// Profit margins
    pub profit_margins: Vec<f64>,
    /// Dividend yield
    pub dividend_yield: Vec<f64>,
    /// Market cap (in currency units)
    pub market_cap: Vec<i64>,
    /// Scores (computed by selection blocks)
    pub scores: Vec<f64>,
    /// Valid flag (has data)
    pub valid: Vec<bool>,
}

impl CandidatesSoA {
    /// Create new SoA with given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            len: 0,
            prices: vec![0.0; capacity],
            volatilities: vec![0.0; capacity],
            momentum_returns: vec![0.0; capacity],
            price_earnings: vec![0.0; capacity],
            price_to_book: vec![0.0; capacity],
            return_on_equity: vec![0.0; capacity],
            debt_to_equity: vec![0.0; capacity],
            profit_margins: vec![0.0; capacity],
            dividend_yield: vec![0.0; capacity],
            market_cap: vec![0; capacity],
            scores: vec![0.0; capacity],
            valid: vec![false; capacity],
        }
    }

    /// Set data for a symbol ID.
    #[inline]
    pub fn set(
        &mut self,
        id: u16,
        price: f64,
        volatility: f64,
        momentum_return: f64,
    ) {
        let idx = id as usize;
        if idx < self.prices.len() {
            self.prices[idx] = price;
            self.volatilities[idx] = volatility;
            self.momentum_returns[idx] = momentum_return;
            self.valid[idx] = true;
            if idx >= self.len {
                self.len = idx + 1;
            }
        }
    }

    /// Set full fundamental data for a symbol ID.
    #[inline]
    pub fn set_fundamentals(
        &mut self,
        id: u16,
        pe: f64,
        pb: f64,
        roe: f64,
        de: f64,
        margins: f64,
        div_yield: f64,
        mcap: i64,
    ) {
        let idx = id as usize;
        if idx < self.prices.len() {
            self.price_earnings[idx] = pe;
            self.price_to_book[idx] = pb;
            self.return_on_equity[idx] = roe;
            self.debt_to_equity[idx] = de;
            self.profit_margins[idx] = margins;
            self.dividend_yield[idx] = div_yield;
            self.market_cap[idx] = mcap;
        }
    }

    /// Get price for symbol ID.
    #[inline]
    pub fn price(&self, id: u16) -> f64 {
        self.prices.get(id as usize).copied().unwrap_or(0.0)
    }

    /// Get volatility for symbol ID.
    #[inline]
    pub fn volatility(&self, id: u16) -> f64 {
        self.volatilities.get(id as usize).copied().unwrap_or(0.0)
    }

    /// Get momentum return for symbol ID.
    #[inline]
    pub fn momentum(&self, id: u16) -> f64 {
        self.momentum_returns.get(id as usize).copied().unwrap_or(0.0)
    }

    /// Get score for symbol ID.
    #[inline]
    pub fn score(&self, id: u16) -> f64 {
        self.scores.get(id as usize).copied().unwrap_or(0.0)
    }

    /// Set score for symbol ID.
    #[inline]
    pub fn set_score(&mut self, id: u16, score: f64) {
        if let Some(s) = self.scores.get_mut(id as usize) {
            *s = score;
        }
    }

    /// Check if symbol has valid data.
    #[inline]
    pub fn is_valid(&self, id: u16) -> bool {
        self.valid.get(id as usize).copied().unwrap_or(false)
    }

    /// Number of candidates.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Clear all data (for reuse).
    pub fn clear(&mut self) {
        self.len = 0;
        self.prices.iter_mut().for_each(|v| *v = 0.0);
        self.volatilities.iter_mut().for_each(|v| *v = 0.0);
        self.momentum_returns.iter_mut().for_each(|v| *v = 0.0);
        self.scores.iter_mut().for_each(|v| *v = 0.0);
        self.valid.iter_mut().for_each(|v| *v = false);
    }

    /// Resize to accommodate more symbols.
    pub fn resize(&mut self, new_len: usize) {
        if new_len > self.prices.len() {
            self.prices.resize(new_len, 0.0);
            self.volatilities.resize(new_len, 0.0);
            self.momentum_returns.resize(new_len, 0.0);
            self.price_earnings.resize(new_len, 0.0);
            self.price_to_book.resize(new_len, 0.0);
            self.return_on_equity.resize(new_len, 0.0);
            self.debt_to_equity.resize(new_len, 0.0);
            self.profit_margins.resize(new_len, 0.0);
            self.dividend_yield.resize(new_len, 0.0);
            self.market_cap.resize(new_len, 0);
            self.scores.resize(new_len, 0.0);
            self.valid.resize(new_len, false);
        }
    }

    /// Iterator over valid symbol IDs.
    pub fn valid_ids(&self) -> impl Iterator<Item = u16> + '_ {
        self.valid
            .iter()
            .enumerate()
            .filter(|(_, &v)| v)
            .map(|(i, _)| i as u16)
    }
}

// =============================================================================
// FAST CONTEXT
// =============================================================================

/// High-performance strategy context with SoA layout.
#[derive(Debug, Clone)]
pub struct FastContext {
    /// Current evaluation date
    pub date: NaiveDate,
    /// Market
    pub market: Market,
    /// Symbol table for ID lookups
    pub symbols: SymbolTable,
    /// SoA candidate data
    pub candidates: CandidatesSoA,
    /// Selected symbol IDs (result of selection blocks)
    pub selected_ids: Vec<u16>,
    /// Weights by symbol ID (dense array, 0.0 for unweighted)
    pub weights: Vec<f64>,
    /// Signal directions by symbol ID
    pub signals: Vec<SignalState>,
    /// Available cash
    pub cash: Decimal,
    /// Total equity
    pub equity: Decimal,
    /// Peak equity (for drawdown)
    pub peak_equity: Decimal,
}

/// Signal state for a symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SignalState {
    #[default]
    None,
    Long,
    Short,
    Exit,
}

impl FastContext {
    /// Create new fast context.
    pub fn new(date: NaiveDate, market: Market, cash: Decimal, capacity: usize) -> Self {
        Self {
            date,
            market,
            symbols: SymbolTable::with_capacity(capacity),
            candidates: CandidatesSoA::with_capacity(capacity),
            selected_ids: Vec::with_capacity(capacity),
            weights: vec![0.0; capacity],
            signals: vec![SignalState::None; capacity],
            cash,
            equity: cash,
            peak_equity: cash,
        }
    }

    /// Add a symbol to the context, returning its ID.
    pub fn add_symbol(&mut self, symbol: impl Into<String>) -> u16 {
        let id = self.symbols.intern(symbol);
        let len = self.symbols.len();
        self.candidates.resize(len);
        if self.weights.len() < len {
            self.weights.resize(len, 0.0);
            self.signals.resize(len, SignalState::None);
        }
        id
    }

    /// Set candidate data by symbol ID.
    #[inline]
    pub fn set_candidate(
        &mut self,
        id: u16,
        price: f64,
        volatility: f64,
        momentum_return: f64,
    ) {
        self.candidates.set(id, price, volatility, momentum_return);
    }

    /// Get price for symbol.
    #[inline]
    pub fn price(&self, id: u16) -> f64 {
        self.candidates.price(id)
    }

    /// Get weight for symbol.
    #[inline]
    pub fn weight(&self, id: u16) -> f64 {
        self.weights.get(id as usize).copied().unwrap_or(0.0)
    }

    /// Set weight for symbol.
    #[inline]
    pub fn set_weight(&mut self, id: u16, weight: f64) {
        if let Some(w) = self.weights.get_mut(id as usize) {
            *w = weight;
        }
    }

    /// Get signal for symbol.
    #[inline]
    pub fn signal(&self, id: u16) -> SignalState {
        self.signals.get(id as usize).copied().unwrap_or(SignalState::None)
    }

    /// Set signal for symbol.
    #[inline]
    pub fn set_signal(&mut self, id: u16, signal: SignalState) {
        if let Some(s) = self.signals.get_mut(id as usize) {
            *s = signal;
        }
    }

    /// Clear selection, weights, and signals (for new rebalance cycle).
    pub fn clear_results(&mut self) {
        self.selected_ids.clear();
        self.weights.iter_mut().for_each(|w| *w = 0.0);
        self.signals.iter_mut().for_each(|s| *s = SignalState::None);
    }

    /// Get total weight sum.
    pub fn total_weight(&self) -> f64 {
        self.weights.iter().sum()
    }

    /// Get selected symbols as strings.
    pub fn selected_symbols(&self) -> Vec<&str> {
        self.selected_ids
            .iter()
            .filter_map(|&id| self.symbols.get_symbol(id))
            .collect()
    }

    /// Get weighted symbols as (symbol, weight) pairs.
    pub fn weighted_symbols(&self) -> Vec<(&str, f64)> {
        self.weights
            .iter()
            .enumerate()
            .filter(|(_, &w)| w > 0.0)
            .filter_map(|(id, &w)| {
                self.symbols.get_symbol(id as u16).map(|s| (s, w))
            })
            .collect()
    }
}

// =============================================================================
// PREALLOCATED BUFFERS
// =============================================================================

/// Preallocated buffers for zero-alloc operations.
#[derive(Debug, Clone)]
pub struct PreallocBuffers {
    /// Temporary scores for sorting
    pub scores: Vec<(u16, f64)>,
    /// Temporary selected IDs
    pub selected: Vec<u16>,
    /// Temporary weights
    pub weights: Vec<(u16, f64)>,
    /// Capacity
    capacity: usize,
}

impl PreallocBuffers {
    /// Create with given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            scores: Vec::with_capacity(capacity),
            selected: Vec::with_capacity(capacity),
            weights: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Clear all buffers for reuse.
    #[inline]
    pub fn clear(&mut self) {
        self.scores.clear();
        self.selected.clear();
        self.weights.clear();
    }

    /// Get capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// =============================================================================
// FAST SELECTION (SoA-optimized)
// =============================================================================

/// Fast momentum selection on SoA data.
/// Returns number of selected assets.
#[inline]
pub fn fast_momentum_select<'a>(
    candidates: &CandidatesSoA,
    top_pct: f64,
    buffers: &'a mut PreallocBuffers,
) -> &'a [u16] {
    buffers.clear();

    // Collect valid candidates with momentum
    for id in candidates.valid_ids() {
        let momentum = candidates.momentum(id);
        if momentum.is_finite() {
            buffers.scores.push((id, momentum));
        }
    }

    // Sort by momentum descending
    buffers.scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Select top N%
    let select_count = ((buffers.scores.len() as f64) * top_pct).ceil() as usize;
    let select_count = select_count.max(1).min(buffers.scores.len());

    buffers.selected.clear();
    for (id, _) in buffers.scores.iter().take(select_count) {
        buffers.selected.push(*id);
    }

    &buffers.selected
}

/// Fast low volatility selection on SoA data.
#[inline]
pub fn fast_low_vol_select<'a>(
    candidates: &CandidatesSoA,
    max_vol: f64,
    top_pct: f64,
    buffers: &'a mut PreallocBuffers,
) -> &'a [u16] {
    buffers.clear();

    // Collect valid candidates with vol below threshold
    for id in candidates.valid_ids() {
        let vol = candidates.volatility(id);
        if vol.is_finite() && vol < max_vol {
            // Score inversely by volatility (lower = better)
            buffers.scores.push((id, -vol));
        }
    }

    // Sort by score descending (which means ascending vol)
    buffers.scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Select top N%
    let select_count = ((buffers.scores.len() as f64) * top_pct).ceil() as usize;
    let select_count = select_count.max(1).min(buffers.scores.len());

    buffers.selected.clear();
    for (id, _) in buffers.scores.iter().take(select_count) {
        buffers.selected.push(*id);
    }

    &buffers.selected
}

/// Fast equal weight sizing on SoA data.
#[inline]
pub fn fast_equal_weight(
    selected_ids: &[u16],
    max_weight: f64,
    max_positions: usize,
    weights: &mut Vec<f64>,
) -> f64 {
    // Reset weights
    weights.iter_mut().for_each(|w| *w = 0.0);

    let n = selected_ids.len().min(max_positions);
    if n == 0 {
        return 0.0;
    }

    let raw_weight = 1.0 / n as f64;
    let capped = raw_weight.min(max_weight);

    // Apply weights
    for &id in selected_ids.iter().take(n) {
        if let Some(w) = weights.get_mut(id as usize) {
            *w = capped;
        }
    }

    // Normalize
    let sum: f64 = weights.iter().sum();
    if sum > 0.0 {
        for &id in selected_ids.iter().take(n) {
            if let Some(w) = weights.get_mut(id as usize) {
                *w /= sum;
            }
        }
    }

    weights.iter().sum()
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_candidates_soa() {
        let mut candidates = CandidatesSoA::with_capacity(10);
        
        candidates.set(0, 100.0, 0.25, 0.10);
        candidates.set(1, 50.0, 0.30, 0.05);
        candidates.set(2, 75.0, 0.20, 0.15);

        assert_eq!(candidates.len(), 3);
        assert!(candidates.is_valid(0));
        assert!(candidates.is_valid(1));
        assert!(candidates.is_valid(2));
        assert!(!candidates.is_valid(3));

        assert_eq!(candidates.price(0), 100.0);
        assert_eq!(candidates.volatility(1), 0.30);
        assert_eq!(candidates.momentum(2), 0.15);
    }

    #[test]
    fn test_fast_context() {
        let mut ctx = FastContext::new(
            NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            Market::BR,
            dec!(100_000),
            10,
        );

        let id0 = ctx.add_symbol("PETR4");
        let id1 = ctx.add_symbol("VALE3");
        
        ctx.set_candidate(id0, 38.0, 0.25, 0.10);
        ctx.set_candidate(id1, 62.0, 0.30, 0.05);

        assert_eq!(ctx.price(id0), 38.0);
        assert_eq!(ctx.price(id1), 62.0);

        ctx.set_weight(id0, 0.6);
        ctx.set_weight(id1, 0.4);

        assert!((ctx.total_weight() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_fast_momentum_select() {
        let mut candidates = CandidatesSoA::with_capacity(10);
        
        // Set up 5 candidates with varying momentum
        candidates.set(0, 100.0, 0.25, 0.05);
        candidates.set(1, 100.0, 0.25, 0.15); // highest
        candidates.set(2, 100.0, 0.25, 0.10);
        candidates.set(3, 100.0, 0.25, 0.08);
        candidates.set(4, 100.0, 0.25, 0.12);

        let mut buffers = PreallocBuffers::with_capacity(10);
        let selected = fast_momentum_select(&candidates, 0.40, &mut buffers);

        // Top 40% of 5 = 2 candidates
        assert_eq!(selected.len(), 2);
        // Should be ids 1 and 4 (highest momentum)
        assert!(selected.contains(&1));
        assert!(selected.contains(&4));
    }

    #[test]
    fn test_fast_equal_weight() {
        let selected = vec![0u16, 1, 2, 3];
        let mut weights = vec![0.0; 10];

        let sum = fast_equal_weight(&selected, 0.30, 10, &mut weights);

        // Each should be ~0.25 (capped at 0.30, so 0.25 is fine)
        assert!((sum - 1.0).abs() < 0.001);
        assert!((weights[0] - 0.25).abs() < 0.001);
        assert!((weights[1] - 0.25).abs() < 0.001);
    }
}


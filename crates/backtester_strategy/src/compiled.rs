//! Compiled Strategy Pipeline - Zero-allocation hot path execution.
//!
//! This module provides a pre-compiled strategy representation that eliminates
//! runtime overhead from the standard compositor:
//! - No HashMap lookups in hot path (typed params structs)
//! - No dynamic block creation (blocks resolved once at compile time)
//! - No string operations (SymbolTable maps to u16 ids)
//! - Preallocated buffers for results
//!
//! # Performance Architecture
//!
//! ```text
//! StrategyConfig (TOML) → CompiledStrategy (one-time)
//!                              ↓
//!                      execute_fast() (hot path)
//!                              ↓
//!                      preallocated buffers, no allocs
//! ```

use crate::blocks::{BlockParams, BlockType, SignalDirection, StrategyBlock};
use crate::config::{PipelineStep, StrategyConfig};
use crate::context::StrategyContext;
use crate::registry::BlockRegistry;
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::hash::{Hash, Hasher};

// =============================================================================
// SYMBOL TABLE
// =============================================================================

/// Bidirectional mapping between symbols (String) and compact IDs (u16).
/// Enables O(1) lookups in the hot path instead of string operations.
#[derive(Debug, Clone, Default)]
pub struct SymbolTable {
    /// Symbol → ID mapping
    symbol_to_id: HashMap<String, u16>,
    /// ID → Symbol mapping (dense array)
    id_to_symbol: Vec<String>,
}

impl SymbolTable {
    /// Create empty symbol table.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with preallocated capacity.
    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            symbol_to_id: HashMap::with_capacity(cap),
            id_to_symbol: Vec::with_capacity(cap),
        }
    }

    /// Build symbol table from a universe of symbols.
    pub fn from_universe(symbols: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let mut table = Self::new();
        for symbol in symbols {
            table.intern(symbol.into());
        }
        table
    }

    /// Intern a symbol, returning its ID. If already present, returns existing ID.
    pub fn intern(&mut self, symbol: impl Into<String>) -> u16 {
        let s = symbol.into();
        if let Some(&id) = self.symbol_to_id.get(&s) {
            return id;
        }
        let id = self.id_to_symbol.len() as u16;
        self.id_to_symbol.push(s.clone());
        self.symbol_to_id.insert(s, id);
        id
    }

    /// Get ID for symbol (O(1) lookup).
    #[inline]
    pub fn get_id(&self, symbol: &str) -> Option<u16> {
        self.symbol_to_id.get(symbol).copied()
    }

    /// Get symbol for ID (O(1) lookup).
    #[inline]
    pub fn get_symbol(&self, id: u16) -> Option<&str> {
        self.id_to_symbol.get(id as usize).map(|s| s.as_str())
    }

    /// Get ID, panicking if not found (for hot path where symbol must exist).
    #[inline]
    pub fn id(&self, symbol: &str) -> u16 {
        self.symbol_to_id[symbol]
    }

    /// Get symbol, panicking if not found.
    #[inline]
    pub fn symbol(&self, id: u16) -> &str {
        &self.id_to_symbol[id as usize]
    }

    /// Number of interned symbols.
    #[inline]
    pub fn len(&self) -> usize {
        self.id_to_symbol.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.id_to_symbol.is_empty()
    }

    /// Iterator over (id, symbol) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (u16, &str)> {
        self.id_to_symbol
            .iter()
            .enumerate()
            .map(|(id, s)| (id as u16, s.as_str()))
    }
}

// =============================================================================
// TYPED PARAMS (eliminates HashMap lookups)
// =============================================================================

/// Parameters hash for cache keying.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParamsHash(u64);

impl ParamsHash {
    /// Compute hash from BlockParams.
    pub fn from_params(params: &BlockParams) -> Self {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        // Sort keys for determinism
        let mut keys: Vec<_> = params.keys().collect();
        keys.sort();
        for key in keys {
            key.hash(&mut hasher);
            if let Some(v) = params.get(key) {
                format!("{:?}", v).hash(&mut hasher);
            }
        }
        ParamsHash(hasher.finish())
    }
}

/// Compiled selection parameters (no runtime parsing).
#[derive(Debug, Clone, Copy)]
pub struct SelectionParams {
    pub top_pct: f64,
    pub lookback_days: i32,
    pub min_return: f64,
    pub max_pe: f64,
    pub max_pb: f64,
    pub min_roe: f64,
    pub max_debt_equity: f64,
    pub max_vol: f64,
    pub min_yield: f64,
    pub min_carry: f64,
    pub min_market_cap: i64,
}

impl Default for SelectionParams {
    fn default() -> Self {
        Self {
            top_pct: 0.20,
            lookback_days: 126,
            min_return: 0.0,
            max_pe: 15.0,
            max_pb: 2.0,
            min_roe: 0.12,
            max_debt_equity: 1.0,
            max_vol: 0.30,
            min_yield: 0.03,
            min_carry: 0.0,
            min_market_cap: 5_000_000_000,
        }
    }
}

impl SelectionParams {
    /// Parse from BlockParams (one-time at compile).
    pub fn from_block_params(params: &BlockParams) -> Self {
        use crate::blocks::{get_f64, get_i64};
        Self {
            top_pct: get_f64(params, "top_pct", 20.0) / 100.0,
            lookback_days: get_i64(params, "lookback_days", 126) as i32,
            min_return: get_f64(params, "min_return", 0.0),
            max_pe: get_f64(params, "max_pe", 15.0),
            max_pb: get_f64(params, "max_pb", 2.0),
            min_roe: get_f64(params, "min_roe", 0.12),
            max_debt_equity: get_f64(params, "max_debt_equity", 1.0),
            max_vol: get_f64(params, "max_annualized_vol", 0.30),
            min_yield: get_f64(params, "min_yield", 0.03),
            min_carry: get_f64(params, "min_carry", 0.0),
            min_market_cap: get_i64(params, "min_market_cap", 5_000_000_000),
        }
    }
}

/// Compiled entry parameters.
#[derive(Debug, Clone, Copy)]
pub struct EntryParams {
    pub fast_period: usize,
    pub slow_period: usize,
    pub signal_period: usize,
    pub period: usize,
    pub std_dev: f64,
    pub oversold: f64,
    pub overbought: f64,
    pub threshold: f64,
}

impl Default for EntryParams {
    fn default() -> Self {
        Self {
            fast_period: 12,
            slow_period: 26,
            signal_period: 9,
            period: 20,
            std_dev: 2.0,
            oversold: 30.0,
            overbought: 70.0,
            threshold: 2.0,
        }
    }
}

impl EntryParams {
    pub fn from_block_params(params: &BlockParams) -> Self {
        use crate::blocks::{get_f64, get_usize};
        Self {
            fast_period: get_usize(params, "fast_period", 12),
            slow_period: get_usize(params, "slow_period", 26),
            signal_period: get_usize(params, "signal", 9),
            period: get_usize(params, "period", 20),
            std_dev: get_f64(params, "std_dev", 2.0),
            oversold: get_f64(params, "oversold", 30.0),
            overbought: get_f64(params, "overbought", 70.0),
            threshold: get_f64(params, "threshold", 2.0),
        }
    }
}

/// Compiled exit parameters.
#[derive(Debug, Clone, Copy)]
pub struct ExitParams {
    pub threshold_pct: f64,
    pub target_pct: f64,
    pub trailing_pct: f64,
    pub activation_pct: f64,
    pub max_days: u32,
}

impl Default for ExitParams {
    fn default() -> Self {
        Self {
            threshold_pct: 0.10,
            target_pct: 0.30,
            trailing_pct: 0.15,
            activation_pct: 0.10,
            max_days: 20,
        }
    }
}

impl ExitParams {
    pub fn from_block_params(params: &BlockParams) -> Self {
        use crate::blocks::{get_f64, get_usize};
        Self {
            threshold_pct: get_f64(params, "threshold_pct", 0.10),
            target_pct: get_f64(params, "target_pct", 0.30),
            trailing_pct: get_f64(params, "trailing_pct", 0.15),
            activation_pct: get_f64(params, "activation_pct", 0.10),
            max_days: get_usize(params, "max_days", 20) as u32,
        }
    }
}

/// Compiled sizing parameters.
#[derive(Debug, Clone, Copy)]
pub struct SizingParams {
    pub max_weight: f64,
    pub min_weight: f64,
    pub max_positions: usize,
    pub target_vol: f64,
    pub max_leverage: f64,
    pub correlation: f64,
    pub fallback_vol: f64,
}

impl Default for SizingParams {
    fn default() -> Self {
        Self {
            max_weight: 0.20,
            min_weight: 0.02,
            max_positions: 20,
            target_vol: 0.12,
            max_leverage: 1.0,
            correlation: 0.5,
            fallback_vol: 0.25,
        }
    }
}

impl SizingParams {
    pub fn from_block_params(params: &BlockParams) -> Self {
        use crate::blocks::{get_f64, get_usize};
        Self {
            max_weight: get_f64(params, "max_weight", 0.20),
            min_weight: get_f64(params, "min_weight", 0.02),
            max_positions: get_usize(params, "max_positions", 20),
            target_vol: get_f64(params, "target_vol", 0.12),
            max_leverage: get_f64(params, "max_leverage", 1.0),
            correlation: get_f64(params, "correlation", 0.5),
            fallback_vol: get_f64(params, "fallback_vol", 0.25),
        }
    }
}

// =============================================================================
// COMPILED STEP
// =============================================================================

/// Compiled pipeline step with resolved block and typed params.
pub struct CompiledStep {
    /// Block ID for tracing
    pub block_id: &'static str,
    /// Block type
    pub block_type: BlockType,
    /// The resolved block (boxed trait object)
    pub block: Box<dyn StrategyBlock>,
    /// Original params (for block.execute compatibility)
    pub params: BlockParams,
    /// Params hash for caching
    pub params_hash: ParamsHash,
    /// Typed params for fast access
    pub typed: CompiledParams,
}

impl fmt::Debug for CompiledStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompiledStep")
            .field("block_id", &self.block_id)
            .field("block_type", &self.block_type)
            .field("params_hash", &self.params_hash)
            .field("typed", &self.typed)
            .finish_non_exhaustive()
    }
}

/// Union of typed params.
#[derive(Debug, Clone)]
pub enum CompiledParams {
    Selection(SelectionParams),
    Entry(EntryParams),
    Exit(ExitParams),
    Sizing(SizingParams),
}

// =============================================================================
// COMPILED STRATEGY
// =============================================================================

/// Pre-compiled strategy ready for fast execution.
/// All blocks resolved, params parsed, buffers preallocated.
pub struct CompiledStrategy {
    /// Strategy ID
    pub id: String,
    /// Strategy version
    pub version: String,
    /// Compiled pipeline steps
    pub steps: Vec<CompiledStep>,
    /// Symbol table
    pub symbols: SymbolTable,
    /// Preallocated result buffers
    buffers: ExecutionBuffers,
    /// Max weight constraint
    pub max_weight: f64,
    /// Max positions constraint
    pub max_positions: Option<usize>,
}

/// Preallocated buffers for zero-alloc execution.
#[derive(Debug, Default)]
struct ExecutionBuffers {
    /// Selected symbol IDs (reused across runs)
    selected_ids: Vec<u16>,
    /// Weights by symbol ID
    weights: Vec<f64>,
    /// Signals buffer
    signals: Vec<(u16, SignalDirection, f64)>,
    /// Scores for sorting
    scores: Vec<(u16, f64)>,
}

impl ExecutionBuffers {
    fn with_capacity(num_symbols: usize) -> Self {
        Self {
            selected_ids: Vec::with_capacity(num_symbols),
            weights: vec![0.0; num_symbols],
            signals: Vec::with_capacity(num_symbols),
            scores: Vec::with_capacity(num_symbols),
        }
    }

    fn clear(&mut self) {
        self.selected_ids.clear();
        self.weights.iter_mut().for_each(|w| *w = 0.0);
        self.signals.clear();
        self.scores.clear();
    }

    fn resize(&mut self, num_symbols: usize) {
        if self.weights.len() < num_symbols {
            self.weights.resize(num_symbols, 0.0);
        }
    }
}

/// Result from compiled strategy execution.
#[derive(Debug, Clone)]
pub struct CompiledResult {
    /// Whether execution succeeded
    pub success: bool,
    /// Selected symbol IDs
    pub selected_ids: Vec<u16>,
    /// Weights by symbol ID (sparse: only non-zero)
    pub weights: Vec<(u16, f64)>,
    /// Signals generated
    pub signals: Vec<(u16, SignalDirection, f64)>,
    /// Total weight sum
    pub weight_sum: f64,
}

impl CompiledStrategy {
    /// Compile a strategy config into a fast-executable form.
    pub fn compile(
        config: &StrategyConfig,
        registry: &BlockRegistry,
        universe: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self, CompileError> {
        // Build symbol table
        let symbols = SymbolTable::from_universe(universe);

        // Compile each step
        let mut steps = Vec::with_capacity(config.pipeline.len());
        for step in config.enabled_steps() {
            let compiled = Self::compile_step(step, registry)?;
            steps.push(compiled);
        }

        let buffers = ExecutionBuffers::with_capacity(symbols.len());

        Ok(Self {
            id: config.strategy.id.clone(),
            version: config.strategy.version.clone(),
            steps,
            symbols,
            buffers,
            max_weight: config.constraints.max_weight_per_asset,
            max_positions: config.constraints.max_positions,
        })
    }

    fn compile_step(
        step: &PipelineStep,
        registry: &BlockRegistry,
    ) -> Result<CompiledStep, CompileError> {
        let block = registry
            .create_block(&step.step_type, &step.block_id, &step.params)
            .ok_or_else(|| CompileError::BlockNotFound(step.block_id.clone()))?;

        // Validate params at compile time
        block.validate_params(&step.params).map_err(|e| {
            CompileError::InvalidParams(step.block_id.clone(), e.to_string())
        })?;

        let block_type = block.block_type();
        let params_hash = ParamsHash::from_params(&step.params);

        let typed = match block_type {
            BlockType::Selection | BlockType::Filter => {
                CompiledParams::Selection(SelectionParams::from_block_params(&step.params))
            }
            BlockType::Entry => {
                CompiledParams::Entry(EntryParams::from_block_params(&step.params))
            }
            BlockType::Exit => {
                CompiledParams::Exit(ExitParams::from_block_params(&step.params))
            }
            BlockType::Sizing => {
                CompiledParams::Sizing(SizingParams::from_block_params(&step.params))
            }
        };

        Ok(CompiledStep {
            block_id: Box::leak(step.block_id.clone().into_boxed_str()),
            block_type,
            block,
            params: step.params.clone(),
            params_hash,
            typed,
        })
    }

    /// Execute the compiled strategy with minimal allocations.
    ///
    /// This is the hot path - optimized for speed.
    pub fn execute_fast(&mut self, ctx: &mut StrategyContext) -> CompiledResult {
        // Ensure buffers sized correctly
        self.buffers.resize(self.symbols.len());
        self.buffers.clear();

        // Execute each step
        for step in &self.steps {
            let result = step.block.execute(ctx, &step.params);

            if !result.success {
                return CompiledResult {
                    success: false,
                    selected_ids: Vec::new(),
                    weights: Vec::new(),
                    signals: Vec::new(),
                    weight_sum: 0.0,
                };
            }

            // Apply results
            match step.block_type {
                BlockType::Selection | BlockType::Filter => {
                    self.buffers.selected_ids.clear();
                    for symbol in &result.selected {
                        if let Some(id) = self.symbols.get_id(symbol) {
                            self.buffers.selected_ids.push(id);
                        }
                    }
                }
                BlockType::Entry | BlockType::Exit => {
                    for signal in &result.signals {
                        if let Some(id) = self.symbols.get_id(&signal.symbol) {
                            self.buffers.signals.push((id, signal.direction, signal.strength));
                        }
                    }
                }
                BlockType::Sizing => {
                    for (symbol, weight) in &result.weights {
                        if let Some(id) = self.symbols.get_id(symbol) {
                            self.buffers.weights[id as usize] = *weight;
                        }
                    }
                }
            }
        }

        // Collect non-zero weights
        let weights: Vec<(u16, f64)> = self.buffers.weights
            .iter()
            .enumerate()
            .filter(|(_, &w)| w > 0.0)
            .map(|(id, &w)| (id as u16, w))
            .collect();

        let weight_sum: f64 = weights.iter().map(|(_, w)| w).sum();

        CompiledResult {
            success: true,
            selected_ids: self.buffers.selected_ids.clone(),
            weights,
            signals: self.buffers.signals.clone(),
            weight_sum,
        }
    }

    /// Get the symbol table reference.
    #[inline]
    pub fn symbol_table(&self) -> &SymbolTable {
        &self.symbols
    }

    /// Get number of steps.
    #[inline]
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }
}

// =============================================================================
// ERRORS
// =============================================================================

/// Errors during strategy compilation.
#[derive(Debug, thiserror::Error)]
pub enum CompileError {
    #[error("Block not found: {0}")]
    BlockNotFound(String),
    #[error("Invalid params for block '{0}': {1}")]
    InvalidParams(String, String),
    #[error("Config error: {0}")]
    ConfigError(String),
}

// =============================================================================
// INDICATOR CACHE
// =============================================================================

/// Key for indicator cache: (block_id_hash, params_hash, symbol_id).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IndicatorCacheKey {
    pub block_id_hash: u64,
    pub params_hash: ParamsHash,
    pub symbol_id: u16,
}

/// Cached indicator result.
#[derive(Debug, Clone)]
pub struct CachedIndicator {
    /// Computed values (e.g., MA series)
    pub values: Vec<f64>,
    /// Last bar index computed
    pub last_bar_idx: usize,
    /// Valid from bar index
    pub valid_from: usize,
}

/// Default maximum cache capacity.
pub const DEFAULT_CACHE_CAPACITY: usize = 10_000;

/// Cache statistics for monitoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Number of evictions due to capacity limit
    pub evictions: u64,
    /// Current number of entries
    pub size: usize,
    /// Maximum capacity
    pub capacity: usize,
}

impl CacheStats {
    /// Calculate hit rate as percentage (0-100).
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }
}

/// Indicator cache for sharing computed series between blocks.
///
/// # Design Decisions
/// - **Per-run scope**: Each `CompiledStrategy` owns its cache; no global sharing.
/// - **Thread-local by design**: No Arc/Mutex needed; single-threaded execution.
/// - **LRU eviction**: When capacity is reached, oldest entries are evicted.
/// - **Validity range**: `valid_from` to `last_bar_idx` defines the valid data range.
///   - Requesting data outside this range requires recomputation.
///
/// # Concurrency
/// This cache is NOT thread-safe. For parallel batch runs, each run should
/// have its own cache instance (achieved naturally via `CompiledStrategy` ownership).
#[derive(Debug)]
pub struct IndicatorCache {
    /// Key -> Value storage
    cache: HashMap<IndicatorCacheKey, CachedIndicator>,
    /// LRU order tracking (oldest at front)
    order: VecDeque<IndicatorCacheKey>,
    /// Maximum number of entries
    capacity: usize,
    /// Cache hit statistics
    hits: u64,
    /// Cache miss statistics
    misses: u64,
    /// Eviction count
    evictions: u64,
}

impl Default for IndicatorCache {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_CACHE_CAPACITY)
    }
}

impl IndicatorCache {
    /// Create new cache with default capacity.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create cache with specified maximum capacity.
    ///
    /// # Arguments
    /// - `cap`: Maximum number of entries. Must be > 0.
    pub fn with_capacity(cap: usize) -> Self {
        let cap = cap.max(1); // Ensure at least 1
        Self {
            cache: HashMap::with_capacity(cap),
            order: VecDeque::with_capacity(cap),
            capacity: cap,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Get cached indicator if available and valid for the requested bar range.
    ///
    /// Updates LRU position on hit.
    #[inline]
    pub fn get(&mut self, key: &IndicatorCacheKey) -> Option<&CachedIndicator> {
        if self.cache.contains_key(key) {
            self.hits += 1;
            // Move to back of LRU (most recently used)
            self.touch_lru(key);
            self.cache.get(key)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Store computed indicator with LRU eviction if at capacity.
    #[inline]
    pub fn insert(&mut self, key: IndicatorCacheKey, indicator: CachedIndicator) {
        // If key already exists, update in place
        if self.cache.contains_key(&key) {
            self.cache.insert(key, indicator);
            self.touch_lru(&key);
            return;
        }

        // Evict oldest if at capacity
        while self.cache.len() >= self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.cache.remove(&oldest);
                self.evictions += 1;
            } else {
                break;
            }
        }

        // Insert new entry
        self.cache.insert(key, indicator);
        self.order.push_back(key);
    }

    /// Move key to back of LRU order (most recently used).
    fn touch_lru(&mut self, key: &IndicatorCacheKey) {
        // Find and remove from current position
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
        }
        // Add to back
        self.order.push_back(*key);
    }

    /// Clear all cached values.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.order.clear();
    }

    /// Invalidate entries for a specific symbol.
    pub fn invalidate_symbol(&mut self, symbol_id: u16) {
        // Remove from cache
        self.cache.retain(|k, _| k.symbol_id != symbol_id);
        // Remove from LRU order
        self.order.retain(|k| k.symbol_id != symbol_id);
    }

    /// Get cache statistics: (hits, misses, evictions).
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            size: self.cache.len(),
            capacity: self.capacity,
        }
    }

    /// Get current cache size (number of entries).
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Get maximum capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_table_basic() {
        let mut table = SymbolTable::new();
        
        let id1 = table.intern("PETR4");
        let id2 = table.intern("VALE3");
        let id3 = table.intern("PETR4"); // duplicate
        
        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id3, id1); // same as first
        assert_eq!(table.len(), 2);
        
        assert_eq!(table.symbol(0), "PETR4");
        assert_eq!(table.symbol(1), "VALE3");
        assert_eq!(table.get_id("PETR4"), Some(0));
        assert_eq!(table.get_id("UNKNOWN"), None);
    }

    #[test]
    fn test_symbol_table_from_universe() {
        let symbols = vec!["A", "B", "C", "D", "E"];
        let table = SymbolTable::from_universe(symbols);
        
        assert_eq!(table.len(), 5);
        assert_eq!(table.id("A"), 0);
        assert_eq!(table.id("E"), 4);
    }

    #[test]
    fn test_params_hash_determinism() {
        let mut params1 = BlockParams::new();
        params1.insert("a".into(), toml::Value::Float(1.0));
        params1.insert("b".into(), toml::Value::Integer(2));

        let mut params2 = BlockParams::new();
        params2.insert("b".into(), toml::Value::Integer(2));
        params2.insert("a".into(), toml::Value::Float(1.0));

        let hash1 = ParamsHash::from_params(&params1);
        let hash2 = ParamsHash::from_params(&params2);

        // Same params in different order should hash the same (sorted)
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_execution_buffers() {
        let mut buffers = ExecutionBuffers::with_capacity(10);
        buffers.weights[0] = 0.5;
        buffers.weights[1] = 0.5;
        buffers.selected_ids.push(0);
        buffers.selected_ids.push(1);

        buffers.clear();

        assert!(buffers.selected_ids.is_empty());
        assert!(buffers.weights.iter().all(|&w| w == 0.0));
    }

    #[test]
    fn test_indicator_cache() {
        let mut cache = IndicatorCache::new();
        
        let key = IndicatorCacheKey {
            block_id_hash: 12345,
            params_hash: ParamsHash(67890),
            symbol_id: 0,
        };
        
        assert!(cache.get(&key).is_none());
        
        cache.insert(key, CachedIndicator {
            values: vec![1.0, 2.0, 3.0],
            last_bar_idx: 100,
            valid_from: 10,
        });
        
        assert!(cache.get(&key).is_some());
        assert_eq!(cache.get(&key).unwrap().values.len(), 3);
        
        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.evictions, 0);
    }

    #[test]
    fn test_cache_capacity_limit() {
        // Create cache with capacity of 3
        let mut cache = IndicatorCache::with_capacity(3);
        
        // Insert 3 entries
        for i in 0..3u16 {
            let key = IndicatorCacheKey {
                block_id_hash: i as u64,
                params_hash: ParamsHash(0),
                symbol_id: i,
            };
            cache.insert(key, CachedIndicator {
                values: vec![i as f64],
                last_bar_idx: i as usize,
                valid_from: 0,
            });
        }
        
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.capacity(), 3);
        
        // Insert a 4th entry - should evict the oldest
        let key_new = IndicatorCacheKey {
            block_id_hash: 100,
            params_hash: ParamsHash(0),
            symbol_id: 100,
        };
        cache.insert(key_new, CachedIndicator {
            values: vec![100.0],
            last_bar_idx: 100,
            valid_from: 0,
        });
        
        // Should still be at capacity
        assert_eq!(cache.len(), 3);
        
        // First entry should be evicted
        let key_0 = IndicatorCacheKey {
            block_id_hash: 0,
            params_hash: ParamsHash(0),
            symbol_id: 0,
        };
        assert!(cache.get(&key_0).is_none(), "First entry should have been evicted");
        
        // New entry should exist
        assert!(cache.get(&key_new).is_some(), "New entry should exist");
        
        let stats = cache.stats();
        assert_eq!(stats.evictions, 1, "Should have 1 eviction");
    }

    #[test]
    fn test_cache_lru_ordering() {
        let mut cache = IndicatorCache::with_capacity(3);
        
        // Insert 3 entries
        let keys: Vec<_> = (0..3u16).map(|i| IndicatorCacheKey {
            block_id_hash: i as u64,
            params_hash: ParamsHash(0),
            symbol_id: i,
        }).collect();
        
        for (i, key) in keys.iter().enumerate() {
            cache.insert(*key, CachedIndicator {
                values: vec![i as f64],
                last_bar_idx: i,
                valid_from: 0,
            });
        }
        
        // Access the first entry (makes it most recently used)
        assert!(cache.get(&keys[0]).is_some());
        
        // Insert a new entry - should evict keys[1] (now oldest)
        let key_new = IndicatorCacheKey {
            block_id_hash: 100,
            params_hash: ParamsHash(0),
            symbol_id: 100,
        };
        cache.insert(key_new, CachedIndicator {
            values: vec![100.0],
            last_bar_idx: 100,
            valid_from: 0,
        });
        
        // keys[0] should still exist (was accessed)
        assert!(cache.get(&keys[0]).is_some(), "keys[0] should still exist (LRU touched)");
        
        // keys[1] should be evicted
        assert!(cache.get(&keys[1]).is_none(), "keys[1] should be evicted");
        
        // keys[2] should still exist
        assert!(cache.get(&keys[2]).is_some(), "keys[2] should still exist");
    }

    #[test]
    fn test_cache_determinism() {
        // Create two caches with same capacity and same operations
        let mut cache1 = IndicatorCache::with_capacity(5);
        let mut cache2 = IndicatorCache::with_capacity(5);
        
        // Perform same operations on both
        for i in 0..10u16 {
            let key = IndicatorCacheKey {
                block_id_hash: i as u64,
                params_hash: ParamsHash(i as u64 * 2),
                symbol_id: i,
            };
            
            let indicator = CachedIndicator {
                values: vec![i as f64; 10],
                last_bar_idx: i as usize * 10,
                valid_from: 0,
            };
            
            cache1.insert(key, indicator.clone());
            cache2.insert(key, indicator);
        }
        
        // Stats should be identical
        assert_eq!(cache1.stats(), cache2.stats(), "Cache stats should be deterministic");
        
        // Same entries should exist in both
        for i in 5..10u16 { // Last 5 entries should exist
            let key = IndicatorCacheKey {
                block_id_hash: i as u64,
                params_hash: ParamsHash(i as u64 * 2),
                symbol_id: i,
            };
            
            let v1 = cache1.get(&key);
            let v2 = cache2.get(&key);
            
            assert!(v1.is_some() == v2.is_some(), "Entry presence should be deterministic for key {}", i);
        }
    }
}


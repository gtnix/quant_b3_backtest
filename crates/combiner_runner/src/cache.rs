//! Ultra-fast genome result caching with lock-free access.
//!
//! This module provides high-performance caching for:
//! - Stage A fitness results (genome hash -> fitness)
//! - Stage B split results (genome_hash + split_index -> split metrics)
//! - Full validation results (genome hash -> validation report)
//!
//! Design principles:
//! - Lock-free reads via DashMap
//! - Fast hashing via ahash/rustc-hash
//! - 128-bit keys for collision resistance in split cache
//! - Atomic counters for observability
//! - TTL-based pruning for memory management

use combiner_core::MultiObjectiveFitness;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::hash::{BuildHasher, Hasher};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

// ============================================================================
// Fast Hasher (using rustc-hash internally via DashMap's default)
// ============================================================================

/// Build a 128-bit cache key from genome hash and split index.
/// Uses bit-interleaving for better distribution.
#[inline(always)]
pub fn make_split_key(genome_hash: u64, split_index: u16) -> u128 {
    // Put genome hash in upper bits, split index in lower
    // This ensures good distribution across DashMap shards
    ((genome_hash as u128) << 64) | (split_index as u128)
}

/// Extract genome hash from split key
#[inline(always)]
pub fn genome_hash_from_key(key: u128) -> u64 {
    (key >> 64) as u64
}

/// Extract split index from split key
#[inline(always)]
pub fn split_index_from_key(key: u128) -> u16 {
    (key & 0xFFFF) as u16
}

// ============================================================================
// Cache Statistics (atomic, lock-free)
// ============================================================================

/// Atomic statistics for cache observability
#[derive(Debug, Default)]
pub struct CacheStats {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub inserts: AtomicU64,
    pub evictions: AtomicU64,
    pub pruned: AtomicU64,
}

impl CacheStats {
    /// Record a cache hit
    #[inline(always)]
    pub fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache miss
    #[inline(always)]
    pub fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an insert
    #[inline(always)]
    pub fn record_insert(&self) {
        self.inserts.fetch_add(1, Ordering::Relaxed);
    }

    /// Get hit rate as fraction (0.0 - 1.0)
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed) as f64;
        let misses = self.misses.load(Ordering::Relaxed) as f64;
        let total = hits + misses;
        if total > 0.0 { hits / total } else { 0.0 }
    }

    /// Reset all counters
    pub fn reset(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.inserts.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
        self.pruned.store(0, Ordering::Relaxed);
    }

    /// Get a snapshot of current values
    pub fn snapshot(&self) -> CacheStatsSnapshot {
        CacheStatsSnapshot {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            inserts: self.inserts.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            pruned: self.pruned.load(Ordering::Relaxed),
        }
    }
}

/// Immutable snapshot of cache statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStatsSnapshot {
    pub hits: u64,
    pub misses: u64,
    pub inserts: u64,
    pub evictions: u64,
    pub pruned: u64,
}

impl CacheStatsSnapshot {
    /// Get hit rate as fraction (0.0 - 1.0)
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total > 0 { self.hits as f64 / total as f64 } else { 0.0 }
    }
}

// ============================================================================
// Stage A Cache (Genome Hash -> Fitness)
// ============================================================================

/// Cache entry for Stage A fitness results
#[derive(Debug, Clone)]
pub struct FitnessCacheEntry {
    pub fitness: MultiObjectiveFitness,
    pub generation_added: u32,
    pub timestamp: Instant,
}

/// Thread-safe cache for Stage A genome fitness results.
///
/// Uses 64-bit genome hash as key. Lock-free reads via DashMap.
#[derive(Debug)]
pub struct GenomeCache {
    cache: DashMap<u64, FitnessCacheEntry>,
    stats: CacheStats,
    max_size: AtomicUsize,
    ttl_seconds: u64,
}

impl Default for GenomeCache {
    fn default() -> Self {
        Self::new()
    }
}

impl GenomeCache {
    /// Create a new empty cache with default settings
    pub fn new() -> Self {
        Self::with_capacity(10_000)
    }

    /// Create a cache with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            cache: DashMap::with_capacity(capacity),
            stats: CacheStats::default(),
            max_size: AtomicUsize::new(capacity * 2), // Allow 2x before pruning
            ttl_seconds: 3600, // 1 hour default TTL
        }
    }

    /// Create a cache with custom TTL
    pub fn with_ttl(capacity: usize, ttl_seconds: u64) -> Self {
        Self {
            cache: DashMap::with_capacity(capacity),
            stats: CacheStats::default(),
            max_size: AtomicUsize::new(capacity * 2),
            ttl_seconds,
        }
    }

    /// Get a cached fitness by genome hash.
    /// Returns None if not found or expired.
    #[inline]
    pub fn get(&self, hash: u64) -> Option<MultiObjectiveFitness> {
        if let Some(entry) = self.cache.get(&hash) {
            // Check TTL
            if entry.timestamp.elapsed().as_secs() > self.ttl_seconds {
                self.stats.record_miss();
                return None;
            }
            self.stats.record_hit();
            Some(entry.fitness.clone())
        } else {
            self.stats.record_miss();
            None
        }
    }

    /// Get entry without TTL check (for internal use)
    #[inline]
    pub fn get_unchecked(&self, hash: u64) -> Option<MultiObjectiveFitness> {
        if let Some(entry) = self.cache.get(&hash) {
            self.stats.record_hit();
            Some(entry.fitness.clone())
        } else {
            self.stats.record_miss();
            None
        }
    }

    /// Store a fitness in the cache.
    #[inline]
    pub fn insert(&self, hash: u64, fitness: MultiObjectiveFitness, generation: u32) {
        self.cache.insert(
            hash,
            FitnessCacheEntry {
                fitness,
                generation_added: generation,
                timestamp: Instant::now(),
            },
        );
        self.stats.record_insert();

        // Check if we need to prune (non-blocking check)
        if self.cache.len() > self.max_size.load(Ordering::Relaxed) {
            self.prune_expired();
        }
    }

    /// Check if a hash is in the cache (without recording stats)
    #[inline]
    pub fn contains(&self, hash: u64) -> bool {
        self.cache.contains_key(&hash)
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Get cache size
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Get hit rate as percentage
    pub fn hit_rate(&self) -> f64 {
        self.stats.hit_rate() * 100.0
    }

    /// Clear the cache
    pub fn clear(&self) {
        self.cache.clear();
        self.stats.reset();
    }

    /// Prune entries older than the given generation
    pub fn prune_older_than(&self, generation: u32) {
        let before = self.cache.len();
        self.cache.retain(|_, v| v.generation_added >= generation);
        let pruned = before - self.cache.len();
        self.stats.pruned.fetch_add(pruned as u64, Ordering::Relaxed);
    }

    /// Prune expired entries (based on TTL)
    pub fn prune_expired(&self) {
        let ttl = Duration::from_secs(self.ttl_seconds);
        let before = self.cache.len();
        self.cache.retain(|_, v| v.timestamp.elapsed() < ttl);
        let pruned = before - self.cache.len();
        self.stats.pruned.fetch_add(pruned as u64, Ordering::Relaxed);
    }
}

// ============================================================================
// Stage B Cache (Split-level caching)
// ============================================================================

/// Metrics from a single validation split
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SplitMetrics {
    pub split_index: u16,
    pub is_sharpe: f64,
    pub oos_sharpe: f64,
    pub is_cagr: f64,
    pub oos_cagr: f64,
    pub is_max_dd: f64,
    pub oos_max_dd: f64,
    pub oos_trades: u32,
    pub passed: bool,
}

/// Cache entry for split validation results
#[derive(Debug, Clone)]
struct SplitCacheEntry {
    pub metrics: SplitMetrics,
    pub timestamp: Instant,
}

/// Thread-safe cache for Stage B split validation results.
///
/// Uses 128-bit key: (genome_hash << 64) | split_index
/// Lock-free reads via DashMap with ahash for fast hashing.
#[derive(Debug)]
pub struct SplitCache {
    cache: DashMap<u128, SplitCacheEntry>,
    stats: CacheStats,
    ttl_seconds: u64,
}

impl Default for SplitCache {
    fn default() -> Self {
        Self::new()
    }
}

impl SplitCache {
    /// Create a new split cache
    pub fn new() -> Self {
        Self::with_capacity(50_000) // Larger capacity for split-level caching
    }

    /// Create with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            cache: DashMap::with_capacity(capacity),
            stats: CacheStats::default(),
            ttl_seconds: 7200, // 2 hour TTL for validation results
        }
    }

    /// Get cached split metrics
    #[inline]
    pub fn get(&self, genome_hash: u64, split_index: u16) -> Option<SplitMetrics> {
        let key = make_split_key(genome_hash, split_index);
        if let Some(entry) = self.cache.get(&key) {
            if entry.timestamp.elapsed().as_secs() > self.ttl_seconds {
                self.stats.record_miss();
                return None;
            }
            self.stats.record_hit();
            Some(entry.metrics.clone())
        } else {
            self.stats.record_miss();
            None
        }
    }

    /// Insert split metrics
    #[inline]
    pub fn insert(&self, genome_hash: u64, split_index: u16, metrics: SplitMetrics) {
        let key = make_split_key(genome_hash, split_index);
        self.cache.insert(key, SplitCacheEntry {
            metrics,
            timestamp: Instant::now(),
        });
        self.stats.record_insert();
    }

    /// Check if split result is cached
    #[inline]
    pub fn contains(&self, genome_hash: u64, split_index: u16) -> bool {
        let key = make_split_key(genome_hash, split_index);
        self.cache.contains_key(&key)
    }

    /// Get all cached splits for a genome
    pub fn get_all_splits(&self, genome_hash: u64, num_splits: u16) -> Vec<Option<SplitMetrics>> {
        (0..num_splits)
            .map(|i| self.get(genome_hash, i))
            .collect()
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Get cache size
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Clear cache
    pub fn clear(&self) {
        self.cache.clear();
        self.stats.reset();
    }

    /// Prune entries for a specific genome
    pub fn prune_genome(&self, genome_hash: u64) {
        // This is O(n) but rarely called
        self.cache.retain(|key, _| genome_hash_from_key(*key) != genome_hash);
    }
}

// ============================================================================
// Unified Validation Cache
// ============================================================================

/// Complete validation result (aggregated from all splits)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidationCacheEntry {
    pub genome_hash: u64,
    pub oos_sharpe_median: f64,
    pub oos_sharpe_mean: f64,
    pub oos_sharpe_std: f64,
    pub oos_cagr_median: f64,
    pub oos_max_dd_worst: f64,
    pub degradation_pct: f64,
    pub pbo: f64,
    pub dsr: f64,
    pub splits_evaluated: u16,
    pub splits_passed: u16,
    pub passed: bool,
    pub discard_reason: Option<String>,
}

/// Unified validation cache combining genome and split caches
#[derive(Debug)]
pub struct ValidationCache {
    /// Stage A cache: genome hash -> fitness
    pub fitness: GenomeCache,
    /// Stage B cache: (genome_hash, split_index) -> split metrics
    pub splits: SplitCache,
    /// Full validation cache: genome hash -> validation result
    pub validations: DashMap<u64, ValidationCacheEntry>,
    /// Combined statistics
    combined_stats: CacheStats,
}

impl Default for ValidationCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ValidationCache {
    /// Create a new unified cache
    pub fn new() -> Self {
        Self::with_capacity(10_000, 50_000)
    }

    /// Create with specified capacities
    pub fn with_capacity(fitness_capacity: usize, split_capacity: usize) -> Self {
        Self {
            fitness: GenomeCache::with_capacity(fitness_capacity),
            splits: SplitCache::with_capacity(split_capacity),
            validations: DashMap::with_capacity(fitness_capacity),
            combined_stats: CacheStats::default(),
        }
    }

    /// Get Stage A fitness
    #[inline]
    pub fn get_fitness(&self, genome_hash: u64) -> Option<MultiObjectiveFitness> {
        self.fitness.get(genome_hash)
    }

    /// Insert Stage A fitness
    #[inline]
    pub fn insert_fitness(&self, genome_hash: u64, fitness: MultiObjectiveFitness, gen: u32) {
        self.fitness.insert(genome_hash, fitness, gen);
    }

    /// Get Stage B split metrics
    #[inline]
    pub fn get_split(&self, genome_hash: u64, split_index: u16) -> Option<SplitMetrics> {
        self.splits.get(genome_hash, split_index)
    }

    /// Insert Stage B split metrics
    #[inline]
    pub fn insert_split(&self, genome_hash: u64, split_index: u16, metrics: SplitMetrics) {
        self.splits.insert(genome_hash, split_index, metrics);
    }

    /// Get full validation result
    #[inline]
    pub fn get_validation(&self, genome_hash: u64) -> Option<ValidationCacheEntry> {
        if let Some(entry) = self.validations.get(&genome_hash) {
            self.combined_stats.record_hit();
            Some(entry.clone())
        } else {
            self.combined_stats.record_miss();
            None
        }
    }

    /// Insert full validation result
    #[inline]
    pub fn insert_validation(&self, entry: ValidationCacheEntry) {
        self.validations.insert(entry.genome_hash, entry);
        self.combined_stats.record_insert();
    }

    /// Check if genome has full validation
    #[inline]
    pub fn has_validation(&self, genome_hash: u64) -> bool {
        self.validations.contains_key(&genome_hash)
    }

    /// Get combined statistics snapshot
    pub fn stats_snapshot(&self) -> CombinedCacheStats {
        CombinedCacheStats {
            fitness: self.fitness.stats().snapshot(),
            splits: self.splits.stats().snapshot(),
            validations: self.combined_stats.snapshot(),
            fitness_size: self.fitness.len(),
            splits_size: self.splits.len(),
            validations_size: self.validations.len(),
        }
    }

    /// Clear all caches
    pub fn clear(&self) {
        self.fitness.clear();
        self.splits.clear();
        self.validations.clear();
        self.combined_stats.reset();
    }

    /// Prune old entries
    pub fn prune(&self, min_generation: u32) {
        self.fitness.prune_older_than(min_generation);
        // Split cache uses TTL-based pruning
    }
}

/// Combined cache statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CombinedCacheStats {
    pub fitness: CacheStatsSnapshot,
    pub splits: CacheStatsSnapshot,
    pub validations: CacheStatsSnapshot,
    pub fitness_size: usize,
    pub splits_size: usize,
    pub validations_size: usize,
}

impl CombinedCacheStats {
    /// Overall hit rate
    pub fn overall_hit_rate(&self) -> f64 {
        let total_hits = self.fitness.hits + self.splits.hits + self.validations.hits;
        let total_misses = self.fitness.misses + self.splits.misses + self.validations.misses;
        let total = total_hits + total_misses;
        if total > 0 { total_hits as f64 / total as f64 } else { 0.0 }
    }

    /// Total memory footprint estimate (rough)
    pub fn estimated_memory_bytes(&self) -> usize {
        // Rough estimates:
        // - Fitness entry: ~200 bytes
        // - Split entry: ~100 bytes  
        // - Validation entry: ~150 bytes
        self.fitness_size * 200 + self.splits_size * 100 + self.validations_size * 150
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use combiner_core::FitnessConfig;

    fn create_test_fitness(sharpe: f64) -> MultiObjectiveFitness {
        let config = FitnessConfig::default();
        MultiObjectiveFitness::from_metrics(
            0.1, sharpe, -0.1, 1.0, 1.0, 1.5, 100, 0.12, 2.5, &config,
        )
    }

    #[test]
    fn test_genome_cache_basic() {
        let cache = GenomeCache::new();
        let fitness = create_test_fitness(1.0);

        cache.insert(12345, fitness.clone(), 0);

        let result = cache.get(12345);
        assert!(result.is_some());
        assert!((result.unwrap().sharpe_ratio - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_genome_cache_miss() {
        let cache = GenomeCache::new();

        let result = cache.get(99999);
        assert!(result.is_none());
        assert_eq!(cache.stats().misses.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_split_key() {
        let genome_hash: u64 = 0xDEADBEEF12345678;
        let split_index: u16 = 5;

        let key = make_split_key(genome_hash, split_index);
        
        assert_eq!(genome_hash_from_key(key), genome_hash);
        assert_eq!(split_index_from_key(key), split_index);
    }

    #[test]
    fn test_split_cache() {
        let cache = SplitCache::new();
        
        let metrics = SplitMetrics {
            split_index: 0,
            oos_sharpe: 0.8,
            is_sharpe: 1.2,
            oos_cagr: 0.10,
            is_cagr: 0.15,
            oos_max_dd: -0.15,
            is_max_dd: -0.10,
            oos_trades: 50,
            passed: true,
        };

        cache.insert(12345, 0, metrics.clone());

        let result = cache.get(12345, 0);
        assert!(result.is_some());
        assert!((result.unwrap().oos_sharpe - 0.8).abs() < 0.01);

        // Different split should not be found
        let result2 = cache.get(12345, 1);
        assert!(result2.is_none());
    }

    #[test]
    fn test_unified_cache() {
        let cache = ValidationCache::new();
        let fitness = create_test_fitness(1.5);

        // Insert fitness
        cache.insert_fitness(111, fitness, 0);
        assert!(cache.get_fitness(111).is_some());

        // Insert split
        let split = SplitMetrics {
            split_index: 0,
            oos_sharpe: 0.9,
            ..Default::default()
        };
        cache.insert_split(111, 0, split);
        assert!(cache.get_split(111, 0).is_some());

        // Insert validation
        let validation = ValidationCacheEntry {
            genome_hash: 111,
            oos_sharpe_median: 0.85,
            passed: true,
            ..Default::default()
        };
        cache.insert_validation(validation);
        assert!(cache.has_validation(111));
    }

    #[test]
    fn test_cache_stats() {
        let cache = GenomeCache::new();
        let fitness = create_test_fitness(1.0);

        cache.insert(1, fitness, 0);

        // 2 hits, 1 miss
        cache.get(1);
        cache.get(1);
        cache.get(2);

        let stats = cache.stats().snapshot();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate() - 0.6666).abs() < 0.01);
    }

    #[test]
    fn test_prune() {
        let cache = GenomeCache::new();
        let fitness = create_test_fitness(1.0);

        cache.insert(1, fitness.clone(), 0);
        cache.insert(2, fitness.clone(), 5);
        cache.insert(3, fitness, 10);

        cache.prune_older_than(5);

        assert!(!cache.contains(1));
        assert!(cache.contains(2));
        assert!(cache.contains(3));
    }
}

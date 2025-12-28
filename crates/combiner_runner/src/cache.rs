//! Genome result caching for deduplication.

use combiner_core::MultiObjectiveFitness;
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Thread-safe cache for genome fitness results.
#[derive(Debug, Default)]
pub struct GenomeCache {
    cache: DashMap<u64, CacheEntry>,
    hits: AtomicU64,
    misses: AtomicU64,
}

/// Cache entry with fitness and metadata.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub fitness: MultiObjectiveFitness,
    pub generation_added: u32,
}

impl GenomeCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a cached fitness by genome hash.
    pub fn get(&self, hash: u64) -> Option<MultiObjectiveFitness> {
        if let Some(entry) = self.cache.get(&hash) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(entry.fitness.clone())
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Store a fitness in the cache.
    pub fn insert(&self, hash: u64, fitness: MultiObjectiveFitness, generation: u32) {
        self.cache.insert(
            hash,
            CacheEntry {
                fitness,
                generation_added: generation,
            },
        );
    }

    /// Check if a hash is in the cache.
    pub fn contains(&self, hash: u64) -> bool {
        self.cache.contains_key(&hash)
    }

    /// Get the number of cache hits.
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Get the number of cache misses.
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    /// Get the cache size.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Get hit rate as a percentage.
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits() as f64;
        let total = hits + self.misses() as f64;
        if total > 0.0 {
            hits / total * 100.0
        } else {
            0.0
        }
    }

    /// Clear the cache.
    pub fn clear(&self) {
        self.cache.clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    /// Prune entries older than the given generation.
    pub fn prune_older_than(&self, generation: u32) {
        self.cache
            .retain(|_, v| v.generation_added >= generation);
    }
}

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
    fn test_cache_insert_get() {
        let cache = GenomeCache::new();
        let fitness = create_test_fitness(1.0);

        cache.insert(12345, fitness.clone(), 0);

        let result = cache.get(12345);
        assert!(result.is_some());
        assert!((result.unwrap().sharpe_ratio - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cache_miss() {
        let cache = GenomeCache::new();

        let result = cache.get(99999);
        assert!(result.is_none());
        assert_eq!(cache.misses(), 1);
    }

    #[test]
    fn test_cache_hit_rate() {
        let cache = GenomeCache::new();
        let fitness = create_test_fitness(1.0);

        cache.insert(1, fitness, 0);

        // 2 hits, 1 miss
        cache.get(1);
        cache.get(1);
        cache.get(2);

        assert!((cache.hit_rate() - 66.67).abs() < 1.0);
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


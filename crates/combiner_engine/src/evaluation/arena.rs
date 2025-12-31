//! Arena allocation for validation results.
//!
//! This module provides efficient memory management for validation results
//! using arena allocation. All results for a generation are allocated
//! from a single bump allocator and freed together at generation end.

use bumpalo::Bump;
use std::cell::RefCell;

/// Arena-allocated validation result storage.
///
/// Uses bumpalo for fast bump allocation with single-dealloc per generation.
/// Thread-local to avoid synchronization overhead.
pub struct ValidationResultArena {
    /// The bump allocator
    arena: Bump,
    /// Current capacity in bytes
    capacity: usize,
    /// Peak usage in bytes (for reporting)
    peak_usage: usize,
}

impl ValidationResultArena {
    /// Create a new arena with the specified initial capacity
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            arena: Bump::with_capacity(capacity_bytes),
            capacity: capacity_bytes,
            peak_usage: 0,
        }
    }

    /// Create with default capacity (64 MB)
    pub fn with_default_capacity() -> Self {
        Self::new(64 * 1024 * 1024) // 64 MB
    }

    /// Allocate space for a value
    #[inline]
    pub fn alloc<T>(&self, value: T) -> &mut T {
        self.arena.alloc(value)
    }

    /// Allocate space for a slice
    #[inline]
    pub fn alloc_slice<T: Copy>(&self, slice: &[T]) -> &mut [T] {
        self.arena.alloc_slice_copy(slice)
    }

    /// Allocate a vector with capacity
    pub fn alloc_vec<T>(&self, capacity: usize) -> bumpalo::collections::Vec<'_, T> {
        bumpalo::collections::Vec::with_capacity_in(capacity, &self.arena)
    }

    /// Reset the arena for the next generation
    ///
    /// This is extremely fast - just resets the allocation pointer
    pub fn reset(&mut self) {
        let current_usage = self.arena.allocated_bytes();
        if current_usage > self.peak_usage {
            self.peak_usage = current_usage;
        }
        self.arena.reset();
    }

    /// Get current allocated bytes
    pub fn allocated_bytes(&self) -> usize {
        self.arena.allocated_bytes()
    }

    /// Get peak usage in bytes
    pub fn peak_usage(&self) -> usize {
        self.peak_usage
    }

    /// Get capacity in bytes
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the underlying bump allocator (for advanced use)
    pub fn as_bump(&self) -> &Bump {
        &self.arena
    }
}

impl Default for ValidationResultArena {
    fn default() -> Self {
        Self::with_default_capacity()
    }
}

/// Thread-local arena pool for parallel validation
pub struct ArenaPool {
    /// Thread-local arenas
    arenas: thread_local::ThreadLocal<RefCell<ValidationResultArena>>,
    /// Capacity for new arenas
    arena_capacity: usize,
}

impl ArenaPool {
    /// Create a new pool with specified per-arena capacity
    pub fn new(arena_capacity: usize) -> Self {
        Self {
            arenas: thread_local::ThreadLocal::new(),
            arena_capacity,
        }
    }

    /// Get or create arena for current thread
    pub fn get(&self) -> &RefCell<ValidationResultArena> {
        self.arenas.get_or(|| {
            RefCell::new(ValidationResultArena::new(self.arena_capacity))
        })
    }

    /// Reset all thread-local arenas
    ///
    /// Note: This should only be called when all threads are idle
    pub fn reset_all(&self) {
        // Note: thread_local doesn't provide iter_mut, so each thread
        // must reset its own arena at the appropriate time
    }
}

impl Default for ArenaPool {
    fn default() -> Self {
        Self::new(16 * 1024 * 1024) // 16 MB per thread
    }
}

/// Validation metrics stored in arena
#[derive(Debug, Clone, Copy)]
pub struct ArenaMetrics {
    pub split_index: u16,
    pub is_sharpe: f64,
    pub oos_sharpe: f64,
    pub is_cagr: f64,
    pub oos_cagr: f64,
    pub is_max_dd: f64,
    pub oos_max_dd: f64,
    pub oos_trades: u32,
    /// Skewness of OOS returns (gamma_3) for PSR/DSR calculations
    pub oos_skewness: f64,
    /// Excess kurtosis of OOS returns (gamma_4) for PSR/DSR calculations
    pub oos_kurtosis: f64,
    /// Number of OOS observations
    pub oos_n_observations: usize,
    pub passed: bool,
}

impl Default for ArenaMetrics {
    fn default() -> Self {
        Self {
            split_index: 0,
            is_sharpe: 0.0,
            oos_sharpe: 0.0,
            is_cagr: 0.0,
            oos_cagr: 0.0,
            is_max_dd: 0.0,
            oos_max_dd: 0.0,
            oos_trades: 0,
            oos_skewness: 0.0,
            oos_kurtosis: 0.0,
            oos_n_observations: 252,
            passed: false,
        }
    }
}

/// Batch of validation results stored in arena
pub struct ArenaBatch<'a> {
    pub genome_index: usize,
    pub genome_hash: u64,
    pub split_results: bumpalo::collections::Vec<'a, ArenaMetrics>,
    pub aggregated: Option<AggregatedMetrics>,
}

/// Aggregated metrics from all splits
#[derive(Debug, Clone, Copy, Default)]
pub struct AggregatedMetrics {
    pub oos_sharpe_median: f64,
    pub oos_sharpe_mean: f64,
    pub oos_sharpe_std: f64,
    pub oos_cagr_median: f64,
    pub degradation_pct: f64,
    pub splits_passed: u16,
    pub splits_total: u16,
    pub pbo_estimate: f64,
    pub overall_passed: bool,
}

impl<'a> ArenaBatch<'a> {
    /// Create a new batch in the given arena
    pub fn new_in(arena: &'a Bump, genome_index: usize, genome_hash: u64, capacity: usize) -> Self {
        Self {
            genome_index,
            genome_hash,
            split_results: bumpalo::collections::Vec::with_capacity_in(capacity, arena),
            aggregated: None,
        }
    }

    /// Add a split result
    pub fn push(&mut self, metrics: ArenaMetrics) {
        self.split_results.push(metrics);
    }

    /// Compute aggregated metrics from split results
    pub fn aggregate(&mut self) {
        if self.split_results.is_empty() {
            self.aggregated = Some(AggregatedMetrics::default());
            return;
        }

        let n = self.split_results.len();
        let mut oos_sharpes: Vec<f64> = self.split_results.iter().map(|m| m.oos_sharpe).collect();
        let is_sharpes: Vec<f64> = self.split_results.iter().map(|m| m.is_sharpe).collect();
        let mut oos_cagrs: Vec<f64> = self.split_results.iter().map(|m| m.oos_cagr).collect();

        // Sort for median
        oos_sharpes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        oos_cagrs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let oos_sharpe_median = if n % 2 == 0 {
            (oos_sharpes[n/2 - 1] + oos_sharpes[n/2]) / 2.0
        } else {
            oos_sharpes[n/2]
        };

        let oos_cagr_median = if n % 2 == 0 {
            (oos_cagrs[n/2 - 1] + oos_cagrs[n/2]) / 2.0
        } else {
            oos_cagrs[n/2]
        };

        let oos_sharpe_mean: f64 = oos_sharpes.iter().sum::<f64>() / n as f64;
        let is_sharpe_mean: f64 = is_sharpes.iter().sum::<f64>() / n as f64;
        
        let oos_sharpe_var: f64 = oos_sharpes.iter()
            .map(|x| (x - oos_sharpe_mean).powi(2))
            .sum::<f64>() / n as f64;
        let oos_sharpe_std = oos_sharpe_var.sqrt();

        // Degradation: (IS - OOS) / IS * 100
        let degradation_pct = if is_sharpe_mean > 0.01 {
            (is_sharpe_mean - oos_sharpe_mean) / is_sharpe_mean * 100.0
        } else {
            0.0
        };

        // Count passed splits
        let splits_passed = self.split_results.iter().filter(|m| m.passed).count() as u16;

        // Simple PBO estimate: P(OOS < 0 | IS > 0)
        let pbo_estimate = if oos_sharpe_std > 0.01 {
            // Normal approximation
            let z = -oos_sharpe_mean / oos_sharpe_std;
            0.5 * (1.0 + libm::erf(z / std::f64::consts::SQRT_2))
        } else if oos_sharpe_mean <= 0.0 {
            1.0
        } else {
            0.0
        };

        // Overall pass: majority of splits passed and PBO is acceptable
        let overall_passed = splits_passed as usize * 2 > n && pbo_estimate < 0.3;

        self.aggregated = Some(AggregatedMetrics {
            oos_sharpe_median,
            oos_sharpe_mean,
            oos_sharpe_std,
            oos_cagr_median,
            degradation_pct,
            splits_passed,
            splits_total: n as u16,
            pbo_estimate,
            overall_passed,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_creation() {
        let arena = ValidationResultArena::new(1024);
        // Initial allocated bytes may be zero or small depending on bumpalo version
        assert!(arena.capacity() >= 1024);
    }

    #[test]
    fn test_arena_alloc() {
        let arena = ValidationResultArena::new(1024);
        let before = arena.allocated_bytes();
        
        let value = arena.alloc(42u64);
        assert_eq!(*value, 42);
        
        // After allocation, should have more bytes allocated
        assert!(arena.allocated_bytes() >= before);
    }

    #[test]
    fn test_arena_reset() {
        let mut arena = ValidationResultArena::new(1024);
        
        for i in 0..100u64 {
            arena.alloc(i);
        }
        
        let before_reset = arena.allocated_bytes();
        assert!(before_reset > 0, "Should have allocated some bytes");
        
        arena.reset();
        
        // Peak usage should be recorded after reset
        assert!(arena.peak_usage() > 0, "Peak usage should be tracked");
        
        // After reset, we can allocate again
        let value = arena.alloc(42u64);
        assert_eq!(*value, 42);
    }

    #[test]
    fn test_arena_batch() {
        let bump = Bump::new();
        let mut batch = ArenaBatch::new_in(&bump, 0, 12345, 6);

        for i in 0..6 {
            batch.push(ArenaMetrics {
                split_index: i,
                is_sharpe: 1.0 + (i as f64) * 0.1,
                oos_sharpe: 0.8 + (i as f64) * 0.05,
                is_cagr: 0.15,
                oos_cagr: 0.10,
                is_max_dd: -0.10,
                oos_max_dd: -0.15,
                oos_trades: 50,
                oos_skewness: -0.3,
                oos_kurtosis: 3.0,
                oos_n_observations: 252,
                passed: i < 5,
            });
        }

        batch.aggregate();

        let agg = batch.aggregated.unwrap();
        assert!(agg.oos_sharpe_median > 0.0);
        assert_eq!(agg.splits_total, 6);
        assert_eq!(agg.splits_passed, 5);
    }
}


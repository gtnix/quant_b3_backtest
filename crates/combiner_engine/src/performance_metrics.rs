//! Performance metrics with atomic counters and generation snapshots.
//!
//! This module provides comprehensive observability for the evolution process,
//! tracking timing, cache performance, and throughput metrics.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use arc_swap::ArcSwap;

/// Atomic timing metric (stores nanoseconds)
#[derive(Debug, Default)]
pub struct AtomicDuration {
    nanos: AtomicU64,
}

impl AtomicDuration {
    pub fn new() -> Self {
        Self { nanos: AtomicU64::new(0) }
    }

    #[inline]
    pub fn add(&self, duration: Duration) {
        self.nanos.fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
    }

    #[inline]
    pub fn get(&self) -> Duration {
        Duration::from_nanos(self.nanos.load(Ordering::Relaxed))
    }

    #[inline]
    pub fn reset(&self) {
        self.nanos.store(0, Ordering::Relaxed);
    }
}

/// System integrity status for observability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityStatus {
    /// Number of mock data usages (should be 0 in production)
    pub mock_data_used: usize,
    /// Number of path-not-found failures
    pub path_not_found_failures: usize,
    /// Repair rate (fraction of genomes that needed repair)
    pub repair_rate: f64,
    /// Overall health status
    pub is_healthy: bool,
}

/// Performance counters for a single generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationSnapshot {
    /// Generation number
    pub generation: u32,
    /// Number of genomes evaluated
    pub genomes_evaluated: usize,
    /// Number of cache hits (Stage A)
    pub stage_a_cache_hits: usize,
    /// Number of cache misses (Stage A)
    pub stage_a_cache_misses: usize,
    /// Stage A evaluation time
    pub stage_a_time_ms: f64,
    /// Number of genomes passed to Stage B
    pub stage_b_candidates: usize,
    /// Stage B validation time
    pub stage_b_time_ms: f64,
    /// Number of split evaluations
    pub splits_evaluated: usize,
    /// Number of splits skipped via cache
    pub splits_cached: usize,
    /// Number of genomes passed validation
    pub genomes_validated: usize,
    /// Number of early exits in Stage B
    pub early_exits: usize,
    /// Total generation time
    pub total_time_ms: f64,
    /// Throughput: genomes evaluated per second
    pub throughput_genomes_per_sec: f64,
    /// Throughput: splits evaluated per second
    pub throughput_splits_per_sec: f64,
    /// Peak memory usage (bytes)
    pub peak_memory_bytes: usize,
    /// Pareto ranking time
    pub pareto_time_ms: f64,
    /// Hall of Fame update time
    pub hof_time_ms: f64,
    /// Timestamp (ms since experiment start)
    pub timestamp_ms: u64,
}

impl Default for GenerationSnapshot {
    fn default() -> Self {
        Self {
            generation: 0,
            genomes_evaluated: 0,
            stage_a_cache_hits: 0,
            stage_a_cache_misses: 0,
            stage_a_time_ms: 0.0,
            stage_b_candidates: 0,
            stage_b_time_ms: 0.0,
            splits_evaluated: 0,
            splits_cached: 0,
            genomes_validated: 0,
            early_exits: 0,
            total_time_ms: 0.0,
            throughput_genomes_per_sec: 0.0,
            throughput_splits_per_sec: 0.0,
            peak_memory_bytes: 0,
            pareto_time_ms: 0.0,
            hof_time_ms: 0.0,
            timestamp_ms: 0,
        }
    }
}

/// Thread-safe performance metrics collector
#[derive(Debug)]
pub struct PerformanceMetrics {
    /// Experiment start time
    start_time: Instant,

    // === Atomic counters (lock-free) ===
    /// Total genomes evaluated
    total_genomes_evaluated: AtomicUsize,
    /// Total Stage A cache hits
    total_stage_a_hits: AtomicUsize,
    /// Total Stage A cache misses
    total_stage_a_misses: AtomicUsize,
    /// Total splits evaluated
    total_splits_evaluated: AtomicUsize,
    /// Total splits cached
    total_splits_cached: AtomicUsize,
    /// Total genomes validated (passed Stage B)
    total_genomes_validated: AtomicUsize,
    /// Total early exits
    total_early_exits: AtomicUsize,
    /// Total generations completed
    total_generations: AtomicUsize,
    
    // === Observability metrics (integrity monitoring) ===
    /// Total backtest failures
    total_backtest_failures: AtomicUsize,
    /// Backtest failures by reason: path not found
    backtest_fail_path_not_found: AtomicUsize,
    /// Backtest failures by reason: invalid genome
    backtest_fail_invalid_genome: AtomicUsize,
    /// Backtest failures by reason: execution error
    backtest_fail_execution_error: AtomicUsize,
    /// Mock data used (should be 0 in production)
    total_mock_data_used: AtomicUsize,
    /// Genomes repaired
    total_genomes_repaired: AtomicUsize,
    /// Weight clamps applied during repair
    total_weight_clamps: AtomicUsize,
    /// Position adjustments during repair
    total_position_adjustments: AtomicUsize,

    // === Timing accumulators ===
    /// Total Stage A time
    total_stage_a_time: AtomicDuration,
    /// Total Stage B time
    total_stage_b_time: AtomicDuration,
    /// Total Pareto ranking time
    total_pareto_time: AtomicDuration,
    /// Total HoF time
    total_hof_time: AtomicDuration,
    /// Total evolution time
    total_evolution_time: AtomicDuration,

    // === Snapshots (lock-free RCU via ArcSwap) ===
    /// Per-generation snapshots (lock-free reads via ArcSwap)
    snapshots: ArcSwap<Vec<GenerationSnapshot>>,
    /// Current generation being recorded
    current_generation: AtomicUsize,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceMetrics {
    /// Create new performance metrics collector
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            total_genomes_evaluated: AtomicUsize::new(0),
            total_stage_a_hits: AtomicUsize::new(0),
            total_stage_a_misses: AtomicUsize::new(0),
            total_splits_evaluated: AtomicUsize::new(0),
            total_splits_cached: AtomicUsize::new(0),
            total_genomes_validated: AtomicUsize::new(0),
            total_early_exits: AtomicUsize::new(0),
            total_generations: AtomicUsize::new(0),
            total_stage_a_time: AtomicDuration::new(),
            total_stage_b_time: AtomicDuration::new(),
            total_pareto_time: AtomicDuration::new(),
            total_hof_time: AtomicDuration::new(),
            total_evolution_time: AtomicDuration::new(),
            snapshots: ArcSwap::from_pointee(Vec::with_capacity(1000)),
            current_generation: AtomicUsize::new(0),
            total_backtest_failures: AtomicUsize::new(0),
            backtest_fail_path_not_found: AtomicUsize::new(0),
            backtest_fail_invalid_genome: AtomicUsize::new(0),
            backtest_fail_execution_error: AtomicUsize::new(0),
            total_mock_data_used: AtomicUsize::new(0),
            total_genomes_repaired: AtomicUsize::new(0),
            total_weight_clamps: AtomicUsize::new(0),
            total_position_adjustments: AtomicUsize::new(0),
        }
    }
    
    // === Observability metric recording ===
    
    /// Record a backtest failure
    #[inline]
    pub fn record_backtest_failure(&self, reason: &str) {
        self.total_backtest_failures.fetch_add(1, Ordering::Relaxed);
        if reason.contains("not found") || reason.contains("NotFound") {
            self.backtest_fail_path_not_found.fetch_add(1, Ordering::Relaxed);
        } else if reason.contains("Invalid") || reason.contains("Weight") {
            self.backtest_fail_invalid_genome.fetch_add(1, Ordering::Relaxed);
        } else {
            self.backtest_fail_execution_error.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    /// Record mock data usage
    #[inline]
    pub fn record_mock_data_used(&self) {
        self.total_mock_data_used.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record genome repair
    #[inline]
    pub fn record_genome_repair(&self, weight_clamps: u32, position_adjustments: u32) {
        self.total_genomes_repaired.fetch_add(1, Ordering::Relaxed);
        self.total_weight_clamps.fetch_add(weight_clamps as usize, Ordering::Relaxed);
        self.total_position_adjustments.fetch_add(position_adjustments as usize, Ordering::Relaxed);
    }
    
    /// Get backtest failure count
    pub fn backtest_failures(&self) -> usize {
        self.total_backtest_failures.load(Ordering::Relaxed)
    }
    
    /// Get mock data usage count (should be 0 in production)
    pub fn mock_data_used(&self) -> usize {
        self.total_mock_data_used.load(Ordering::Relaxed)
    }
    
    /// Get genomes repaired count
    pub fn genomes_repaired(&self) -> usize {
        self.total_genomes_repaired.load(Ordering::Relaxed)
    }
    
    /// Get repair rate (repaired / evaluated)
    pub fn repair_rate(&self) -> f64 {
        let evaluated = self.total_genomes_evaluated.load(Ordering::Relaxed);
        if evaluated == 0 {
            return 0.0;
        }
        self.total_genomes_repaired.load(Ordering::Relaxed) as f64 / evaluated as f64
    }
    
    /// Check if system integrity is compromised
    pub fn integrity_check(&self) -> IntegrityStatus {
        let mock_used = self.mock_data_used();
        let path_failures = self.backtest_fail_path_not_found.load(Ordering::Relaxed);
        let repair_rate = self.repair_rate();
        
        IntegrityStatus {
            mock_data_used: mock_used,
            path_not_found_failures: path_failures,
            repair_rate,
            is_healthy: mock_used == 0 && path_failures == 0 && repair_rate < 0.5,
        }
    }

    /// Reset all counters
    pub fn reset(&self) {
        self.total_genomes_evaluated.store(0, Ordering::Relaxed);
        self.total_stage_a_hits.store(0, Ordering::Relaxed);
        self.total_backtest_failures.store(0, Ordering::Relaxed);
        self.backtest_fail_path_not_found.store(0, Ordering::Relaxed);
        self.backtest_fail_invalid_genome.store(0, Ordering::Relaxed);
        self.backtest_fail_execution_error.store(0, Ordering::Relaxed);
        self.total_mock_data_used.store(0, Ordering::Relaxed);
        self.total_genomes_repaired.store(0, Ordering::Relaxed);
        self.total_weight_clamps.store(0, Ordering::Relaxed);
        self.total_position_adjustments.store(0, Ordering::Relaxed);
        self.total_stage_a_misses.store(0, Ordering::Relaxed);
        self.total_splits_evaluated.store(0, Ordering::Relaxed);
        self.total_splits_cached.store(0, Ordering::Relaxed);
        self.total_genomes_validated.store(0, Ordering::Relaxed);
        self.total_early_exits.store(0, Ordering::Relaxed);
        self.total_generations.store(0, Ordering::Relaxed);
        self.total_stage_a_time.reset();
        self.total_stage_b_time.reset();
        self.total_pareto_time.reset();
        self.total_hof_time.reset();
        self.total_evolution_time.reset();
        self.snapshots.store(Arc::new(Vec::with_capacity(1000)));
        self.current_generation.store(0, Ordering::Relaxed);
    }

    // === Increment methods (lock-free) ===

    #[inline]
    pub fn add_genomes_evaluated(&self, count: usize) {
        self.total_genomes_evaluated.fetch_add(count, Ordering::Relaxed);
    }

    #[inline]
    pub fn add_stage_a_hit(&self) {
        self.total_stage_a_hits.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn add_stage_a_hits(&self, count: usize) {
        self.total_stage_a_hits.fetch_add(count, Ordering::Relaxed);
    }

    #[inline]
    pub fn add_stage_a_miss(&self) {
        self.total_stage_a_misses.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn add_stage_a_misses(&self, count: usize) {
        self.total_stage_a_misses.fetch_add(count, Ordering::Relaxed);
    }

    #[inline]
    pub fn add_splits_evaluated(&self, count: usize) {
        self.total_splits_evaluated.fetch_add(count, Ordering::Relaxed);
    }

    #[inline]
    pub fn add_splits_cached(&self, count: usize) {
        self.total_splits_cached.fetch_add(count, Ordering::Relaxed);
    }

    #[inline]
    pub fn add_genomes_validated(&self, count: usize) {
        self.total_genomes_validated.fetch_add(count, Ordering::Relaxed);
    }

    #[inline]
    pub fn add_early_exit(&self) {
        self.total_early_exits.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn add_stage_a_time(&self, duration: Duration) {
        self.total_stage_a_time.add(duration);
    }

    #[inline]
    pub fn add_stage_b_time(&self, duration: Duration) {
        self.total_stage_b_time.add(duration);
    }

    #[inline]
    pub fn add_pareto_time(&self, duration: Duration) {
        self.total_pareto_time.add(duration);
    }

    #[inline]
    pub fn add_hof_time(&self, duration: Duration) {
        self.total_hof_time.add(duration);
    }

    // === Snapshot recording ===

    /// Record a generation snapshot (RCU pattern: clone, modify, swap)
    pub fn record_generation(&self, snapshot: GenerationSnapshot) {
        self.total_generations.fetch_add(1, Ordering::Relaxed);
        self.current_generation.store(snapshot.generation as usize, Ordering::Relaxed);
        // RCU: load current, clone, append, swap atomically
        self.snapshots.rcu(|current| {
            let mut new = (**current).clone();
            new.push(snapshot.clone());
            new
        });
    }

    /// Get current generation number
    pub fn current_generation(&self) -> u32 {
        self.current_generation.load(Ordering::Relaxed) as u32
    }

    /// Get all snapshots (lock-free read via ArcSwap)
    pub fn snapshots(&self) -> Vec<GenerationSnapshot> {
        self.snapshots.load().as_ref().clone()
    }

    /// Get latest snapshot (lock-free read via ArcSwap)
    pub fn latest_snapshot(&self) -> Option<GenerationSnapshot> {
        self.snapshots.load().last().cloned()
    }

    // === Aggregate statistics ===

    /// Get overall statistics summary
    pub fn summary(&self) -> PerformanceMetricsSummary {
        let elapsed = self.start_time.elapsed();
        let elapsed_secs = elapsed.as_secs_f64();

        let total_genomes = self.total_genomes_evaluated.load(Ordering::Relaxed);
        let stage_a_hits = self.total_stage_a_hits.load(Ordering::Relaxed);
        let stage_a_misses = self.total_stage_a_misses.load(Ordering::Relaxed);
        let total_splits = self.total_splits_evaluated.load(Ordering::Relaxed);
        let splits_cached = self.total_splits_cached.load(Ordering::Relaxed);
        let genomes_validated = self.total_genomes_validated.load(Ordering::Relaxed);
        let early_exits = self.total_early_exits.load(Ordering::Relaxed);
        let generations = self.total_generations.load(Ordering::Relaxed);

        let stage_a_total = stage_a_hits + stage_a_misses;
        let stage_a_hit_rate = if stage_a_total > 0 {
            stage_a_hits as f64 / stage_a_total as f64 * 100.0
        } else {
            0.0
        };

        let split_total = total_splits + splits_cached;
        let split_cache_hit_rate = if split_total > 0 {
            splits_cached as f64 / split_total as f64 * 100.0
        } else {
            0.0
        };

        let avg_generation_time_ms = if generations > 0 {
            elapsed_secs * 1000.0 / generations as f64
        } else {
            0.0
        };

        PerformanceMetricsSummary {
            total_genomes_evaluated: total_genomes,
            total_generations: generations,
            total_time_secs: elapsed_secs,
            avg_generation_time_ms,
            throughput_genomes_per_sec: if elapsed_secs > 0.0 {
                total_genomes as f64 / elapsed_secs
            } else {
                0.0
            },
            throughput_generations_per_min: if elapsed_secs > 0.0 {
                generations as f64 / elapsed_secs * 60.0
            } else {
                0.0
            },
            stage_a_cache_hit_rate: stage_a_hit_rate,
            split_cache_hit_rate,
            total_splits_evaluated: total_splits,
            total_splits_cached: splits_cached,
            total_genomes_validated: genomes_validated,
            validation_rate: if total_genomes > 0 {
                genomes_validated as f64 / total_genomes as f64 * 100.0
            } else {
                0.0
            },
            total_early_exits: early_exits,
            early_exit_rate: if total_genomes > 0 {
                early_exits as f64 / total_genomes as f64 * 100.0
            } else {
                0.0
            },
            stage_a_time_pct: self.time_percentage(&self.total_stage_a_time, elapsed_secs),
            stage_b_time_pct: self.time_percentage(&self.total_stage_b_time, elapsed_secs),
            pareto_time_pct: self.time_percentage(&self.total_pareto_time, elapsed_secs),
            hof_time_pct: self.time_percentage(&self.total_hof_time, elapsed_secs),
        }
    }

    #[inline]
    fn time_percentage(&self, duration: &AtomicDuration, total_secs: f64) -> f64 {
        if total_secs > 0.0 {
            duration.get().as_secs_f64() / total_secs * 100.0
        } else {
            0.0
        }
    }

    /// Get time breakdown for latest generation
    pub fn latest_time_breakdown(&self) -> Option<TimeBreakdown> {
        let snapshot = self.latest_snapshot()?;
        let total = snapshot.total_time_ms;

        Some(TimeBreakdown {
            stage_a_pct: if total > 0.0 { snapshot.stage_a_time_ms / total * 100.0 } else { 0.0 },
            stage_b_pct: if total > 0.0 { snapshot.stage_b_time_ms / total * 100.0 } else { 0.0 },
            pareto_pct: if total > 0.0 { snapshot.pareto_time_ms / total * 100.0 } else { 0.0 },
            hof_pct: if total > 0.0 { snapshot.hof_time_ms / total * 100.0 } else { 0.0 },
            other_pct: if total > 0.0 {
                100.0 - (snapshot.stage_a_time_ms + snapshot.stage_b_time_ms + snapshot.pareto_time_ms + snapshot.hof_time_ms) / total * 100.0
            } else { 0.0 },
        })
    }

    /// Get elapsed time since start
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get timestamp in ms since start
    pub fn timestamp_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }
}

/// Summary of performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetricsSummary {
    pub total_genomes_evaluated: usize,
    pub total_generations: usize,
    pub total_time_secs: f64,
    pub avg_generation_time_ms: f64,
    pub throughput_genomes_per_sec: f64,
    pub throughput_generations_per_min: f64,
    pub stage_a_cache_hit_rate: f64,
    pub split_cache_hit_rate: f64,
    pub total_splits_evaluated: usize,
    pub total_splits_cached: usize,
    pub total_genomes_validated: usize,
    pub validation_rate: f64,
    pub total_early_exits: usize,
    pub early_exit_rate: f64,
    pub stage_a_time_pct: f64,
    pub stage_b_time_pct: f64,
    pub pareto_time_pct: f64,
    pub hof_time_pct: f64,
}

/// Time breakdown by phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeBreakdown {
    pub stage_a_pct: f64,
    pub stage_b_pct: f64,
    pub pareto_pct: f64,
    pub hof_pct: f64,
    pub other_pct: f64,
}

/// Scoped timer for automatic duration recording
pub struct ScopedTimer<'a, F: Fn(Duration)> {
    start: Instant,
    callback: &'a F,
}

impl<'a, F: Fn(Duration)> ScopedTimer<'a, F> {
    pub fn new(callback: &'a F) -> Self {
        Self {
            start: Instant::now(),
            callback,
        }
    }
}

impl<'a, F: Fn(Duration)> Drop for ScopedTimer<'a, F> {
    fn drop(&mut self) {
        (self.callback)(self.start.elapsed());
    }
}

/// Macro for timed execution
#[macro_export]
macro_rules! timed {
    ($metrics:expr, $method:ident, $body:expr) => {{
        let start = std::time::Instant::now();
        let result = $body;
        $metrics.$method(start.elapsed());
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_atomic_counters() {
        let metrics = PerformanceMetrics::new();

        metrics.add_genomes_evaluated(100);
        metrics.add_genomes_evaluated(50);

        let summary = metrics.summary();
        assert_eq!(summary.total_genomes_evaluated, 150);
    }

    #[test]
    fn test_concurrent_updates() {
        let metrics = std::sync::Arc::new(PerformanceMetrics::new());

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let m = metrics.clone();
                thread::spawn(move || {
                    for _ in 0..100 {
                        m.add_genomes_evaluated(1);
                        m.add_stage_a_hit();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let summary = metrics.summary();
        assert_eq!(summary.total_genomes_evaluated, 1000);
    }

    #[test]
    fn test_generation_snapshot() {
        let metrics = PerformanceMetrics::new();

        let snapshot = GenerationSnapshot {
            generation: 1,
            genomes_evaluated: 100,
            stage_a_time_ms: 50.0,
            stage_b_time_ms: 200.0,
            total_time_ms: 300.0,
            ..Default::default()
        };

        metrics.record_generation(snapshot.clone());

        assert_eq!(metrics.current_generation(), 1);
        assert_eq!(metrics.snapshots().len(), 1);

        let latest = metrics.latest_snapshot().unwrap();
        assert_eq!(latest.generation, 1);
        assert_eq!(latest.genomes_evaluated, 100);
    }

    #[test]
    fn test_time_breakdown() {
        let metrics = PerformanceMetrics::new();

        let snapshot = GenerationSnapshot {
            generation: 1,
            stage_a_time_ms: 50.0,
            stage_b_time_ms: 150.0,
            pareto_time_ms: 25.0,
            hof_time_ms: 25.0,
            total_time_ms: 250.0,
            ..Default::default()
        };

        metrics.record_generation(snapshot);

        let breakdown = metrics.latest_time_breakdown().unwrap();
        assert_eq!(breakdown.stage_a_pct, 20.0);
        assert_eq!(breakdown.stage_b_pct, 60.0);
        assert_eq!(breakdown.pareto_pct, 10.0);
        assert_eq!(breakdown.hof_pct, 10.0);
    }

    #[test]
    fn test_cache_hit_rate() {
        let metrics = PerformanceMetrics::new();

        metrics.add_stage_a_hits(80);
        metrics.add_stage_a_misses(20);

        let summary = metrics.summary();
        assert!((summary.stage_a_cache_hit_rate - 80.0).abs() < 0.01);
    }
}


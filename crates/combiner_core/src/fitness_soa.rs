//! Struct-of-Arrays (SoA) layout for population fitness.
//!
//! This module provides cache-efficient data structures for batch processing
//! of fitness values during evolution. The SoA layout maximizes cache hits
//! during Pareto sorting and SIMD operations.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

/// Struct-of-Arrays layout for population fitness.
/// 
/// This representation is optimized for:
/// - SIMD operations on objective values
/// - Cache-friendly iteration during Pareto ranking
/// - Atomic updates during parallel evaluation
#[repr(C)]
pub struct PopulationFitnessSoA {
    // ===== Primary objectives (aligned for SIMD) =====
    /// Sharpe ratios for all genomes (maximize)
    pub sharpe_ratios: AlignedVec<f64>,
    /// CAGR values for all genomes (maximize)
    pub cagrs: AlignedVec<f64>,
    /// Max drawdowns for all genomes (maximize - less negative is better)
    pub max_drawdowns: AlignedVec<f64>,
    /// Calmar ratios for all genomes (maximize)
    pub calmar_ratios: AlignedVec<f64>,
    /// Sortino ratios for all genomes (maximize)
    pub sortino_ratios: AlignedVec<f64>,
    /// Profit factors for all genomes (maximize)
    pub profit_factors: AlignedVec<f64>,

    // ===== Secondary metrics =====
    /// Total trades count
    pub total_trades: Vec<u32>,
    /// Annualized volatility
    pub volatilities: Vec<f64>,
    /// Annual turnover
    pub turnovers: Vec<f64>,

    // ===== Pareto ranking (computed by NSGA-II) =====
    /// Pareto rank (0 = non-dominated, 1 = dominated by rank 0, etc.)
    pub pareto_ranks: Vec<u32>,
    /// Crowding distance for diversity preservation
    pub crowding_distances: Vec<f64>,

    // ===== Validation results (Stage B) =====
    /// OOS Sharpe medians (from validation)
    pub oos_sharpe_medians: Vec<f64>,
    /// Probability of Backtest Overfitting
    pub pbos: Vec<f64>,
    /// Deflated Sharpe Ratios
    pub dsrs: Vec<f64>,
    /// Whether genome has been validated (Stage B complete)
    pub is_validated: Vec<bool>,
    /// Whether genome is valid (passed all checks)
    pub is_valid: Vec<bool>,

    // ===== Metadata =====
    /// Current population size
    len: AtomicUsize,
    /// Capacity
    capacity: usize,
}

/// SIMD-aligned vector wrapper (32-byte alignment for AVX2)
#[repr(C, align(32))]
pub struct AlignedVec<T> {
    data: Vec<T>,
}

impl<T: Clone + Default> AlignedVec<T> {
    /// Create with capacity, pre-filled with defaults
    pub fn with_capacity(cap: usize) -> Self {
        let mut data = Vec::with_capacity(cap);
        data.resize(cap, T::default());
        Self { data }
    }

    /// Create empty
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Get slice
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Get mutable slice
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Get length
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Resize with default value
    pub fn resize(&mut self, new_len: usize) {
        self.data.resize(new_len, T::default());
    }

    /// Get value at index
    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }

    /// Set value at index
    #[inline(always)]
    pub fn set(&mut self, index: usize, value: T) {
        if index < self.data.len() {
            self.data[index] = value;
        }
    }
}

impl<T: Clone + Default> Default for AlignedVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> std::ops::Index<usize> for AlignedVec<T> {
    type Output = T;
    
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> std::ops::IndexMut<usize> for AlignedVec<T> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl PopulationFitnessSoA {
    /// Create a new SoA with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            sharpe_ratios: AlignedVec::with_capacity(capacity),
            cagrs: AlignedVec::with_capacity(capacity),
            max_drawdowns: AlignedVec::with_capacity(capacity),
            calmar_ratios: AlignedVec::with_capacity(capacity),
            sortino_ratios: AlignedVec::with_capacity(capacity),
            profit_factors: AlignedVec::with_capacity(capacity),
            total_trades: vec![0; capacity],
            volatilities: vec![0.0; capacity],
            turnovers: vec![0.0; capacity],
            pareto_ranks: vec![u32::MAX; capacity],
            crowding_distances: vec![0.0; capacity],
            oos_sharpe_medians: vec![f64::NEG_INFINITY; capacity],
            pbos: vec![1.0; capacity],
            dsrs: vec![0.0; capacity],
            is_validated: vec![false; capacity],
            is_valid: vec![false; capacity],
            len: AtomicUsize::new(0),
            capacity,
        }
    }

    /// Get current length
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    /// Check if empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get capacity
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Set fitness for a genome at index (thread-safe via atomic index claim)
    pub fn set_fitness(
        &mut self,
        index: usize,
        sharpe: f64,
        cagr: f64,
        max_dd: f64,
        calmar: f64,
        sortino: f64,
        profit_factor: f64,
        trades: u32,
        volatility: f64,
        turnover: f64,
        is_valid: bool,
    ) {
        if index >= self.capacity {
            return;
        }

        self.sharpe_ratios[index] = sharpe;
        self.cagrs[index] = cagr;
        self.max_drawdowns[index] = max_dd;
        self.calmar_ratios[index] = calmar;
        self.sortino_ratios[index] = sortino;
        self.profit_factors[index] = profit_factor;
        self.total_trades[index] = trades;
        self.volatilities[index] = volatility;
        self.turnovers[index] = turnover;
        self.is_valid[index] = is_valid;

        // Update length if this is a new entry
        let current_len = self.len.load(Ordering::Relaxed);
        if index >= current_len {
            self.len.store(index + 1, Ordering::Relaxed);
        }
    }

    /// Set validation results for a genome
    pub fn set_validation(
        &mut self,
        index: usize,
        oos_sharpe_median: f64,
        pbo: f64,
        dsr: f64,
    ) {
        if index >= self.capacity {
            return;
        }

        self.oos_sharpe_medians[index] = oos_sharpe_median;
        self.pbos[index] = pbo;
        self.dsrs[index] = dsr;
        self.is_validated[index] = true;
    }

    /// Set Pareto rank for a genome
    #[inline(always)]
    pub fn set_pareto_rank(&mut self, index: usize, rank: u32) {
        if index < self.capacity {
            self.pareto_ranks[index] = rank;
        }
    }

    /// Set crowding distance for a genome
    #[inline(always)]
    pub fn set_crowding_distance(&mut self, index: usize, distance: f64) {
        if index < self.capacity {
            self.crowding_distances[index] = distance;
        }
    }

    /// Reset for next generation (keeps capacity, clears data)
    pub fn reset(&mut self) {
        let cap = self.capacity;
        
        // Reset all vectors to default values
        for i in 0..cap {
            self.sharpe_ratios[i] = 0.0;
            self.cagrs[i] = 0.0;
            self.max_drawdowns[i] = 0.0;
            self.calmar_ratios[i] = 0.0;
            self.sortino_ratios[i] = 0.0;
            self.profit_factors[i] = 0.0;
            self.total_trades[i] = 0;
            self.volatilities[i] = 0.0;
            self.turnovers[i] = 0.0;
            self.pareto_ranks[i] = u32::MAX;
            self.crowding_distances[i] = 0.0;
            self.oos_sharpe_medians[i] = f64::NEG_INFINITY;
            self.pbos[i] = 1.0;
            self.dsrs[i] = 0.0;
            self.is_validated[i] = false;
            self.is_valid[i] = false;
        }
        
        self.len.store(0, Ordering::Relaxed);
    }

    /// Compute scalar fitness for simple comparisons
    /// Uses weighted combination of objectives with penalties
    #[inline]
    pub fn scalar_fitness(&self, index: usize) -> f64 {
        if index >= self.len() || !self.is_valid[index] {
            return f64::NEG_INFINITY;
        }

        let sharpe = self.sharpe_ratios[index];
        let cagr = self.cagrs[index];
        let max_dd = self.max_drawdowns[index];

        // Base score
        let mut score = sharpe;

        // Penalty for low trades
        if self.total_trades[index] < 30 {
            score *= 0.5;
        }

        // Penalty for high drawdown
        if max_dd < -0.25 {
            score *= 0.8;
        }

        // Bonus for validation
        if self.is_validated[index] {
            let validation_factor = 1.0 - self.pbos[index].min(0.5);
            score *= validation_factor;
        }

        score
    }

    /// Get indices sorted by scalar fitness (descending)
    pub fn sorted_indices(&self) -> Vec<usize> {
        let n = self.len();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            let fa = self.scalar_fitness(a);
            let fb = self.scalar_fitness(b);
            fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal)
        });
        indices
    }

    /// Get indices of Pareto-optimal solutions (rank 0)
    pub fn pareto_optimal_indices(&self) -> Vec<usize> {
        let n = self.len();
        (0..n)
            .filter(|&i| self.is_valid[i] && self.pareto_ranks[i] == 0)
            .collect()
    }

    /// Get indices of validated solutions that pass criteria
    pub fn validated_passing_indices(&self, min_oos_sharpe: f64, max_pbo: f64) -> Vec<usize> {
        let n = self.len();
        (0..n)
            .filter(|&i| {
                self.is_valid[i]
                    && self.is_validated[i]
                    && self.oos_sharpe_medians[i] >= min_oos_sharpe
                    && self.pbos[i] <= max_pbo
            })
            .collect()
    }
}

/// Compact fitness data for a single genome (for AoS interop)
#[derive(Debug, Clone, Default)]
pub struct FitnessData {
    pub sharpe_ratio: f64,
    pub cagr: f64,
    pub max_drawdown: f64,
    pub calmar_ratio: f64,
    pub sortino_ratio: f64,
    pub profit_factor: f64,
    pub total_trades: u32,
    pub volatility: f64,
    pub turnover: f64,
    pub pareto_rank: u32,
    pub crowding_distance: f64,
    pub oos_sharpe_median: f64,
    pub pbo: f64,
    pub dsr: f64,
    pub is_validated: bool,
    pub is_valid: bool,
}

impl PopulationFitnessSoA {
    /// Extract fitness data for a single genome (SoA -> AoS conversion)
    pub fn get_fitness(&self, index: usize) -> Option<FitnessData> {
        if index >= self.len() {
            return None;
        }

        Some(FitnessData {
            sharpe_ratio: self.sharpe_ratios[index],
            cagr: self.cagrs[index],
            max_drawdown: self.max_drawdowns[index],
            calmar_ratio: self.calmar_ratios[index],
            sortino_ratio: self.sortino_ratios[index],
            profit_factor: self.profit_factors[index],
            total_trades: self.total_trades[index],
            volatility: self.volatilities[index],
            turnover: self.turnovers[index],
            pareto_rank: self.pareto_ranks[index],
            crowding_distance: self.crowding_distances[index],
            oos_sharpe_median: self.oos_sharpe_medians[index],
            pbo: self.pbos[index],
            dsr: self.dsrs[index],
            is_validated: self.is_validated[index],
            is_valid: self.is_valid[index],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soa_creation() {
        let soa = PopulationFitnessSoA::with_capacity(100);
        assert_eq!(soa.capacity(), 100);
        assert_eq!(soa.len(), 0);
    }

    #[test]
    fn test_set_fitness() {
        let mut soa = PopulationFitnessSoA::with_capacity(10);
        
        soa.set_fitness(0, 1.5, 0.15, -0.10, 1.5, 1.2, 1.8, 100, 0.12, 2.5, true);
        soa.set_fitness(1, 0.8, 0.10, -0.20, 0.5, 0.6, 1.2, 50, 0.18, 3.0, true);

        assert_eq!(soa.len(), 2);
        assert_eq!(soa.sharpe_ratios[0], 1.5);
        assert_eq!(soa.sharpe_ratios[1], 0.8);
        assert!(soa.is_valid[0]);
        assert!(soa.is_valid[1]);
    }

    #[test]
    fn test_scalar_fitness() {
        let mut soa = PopulationFitnessSoA::with_capacity(10);
        
        soa.set_fitness(0, 1.5, 0.15, -0.10, 1.5, 1.2, 1.8, 100, 0.12, 2.5, true);
        soa.set_fitness(1, 0.8, 0.10, -0.20, 0.5, 0.6, 1.2, 50, 0.18, 3.0, true);

        let f0 = soa.scalar_fitness(0);
        let f1 = soa.scalar_fitness(1);

        assert!(f0 > f1); // Higher sharpe should have higher fitness
    }

    #[test]
    fn test_sorted_indices() {
        let mut soa = PopulationFitnessSoA::with_capacity(10);
        
        soa.set_fitness(0, 0.5, 0.10, -0.10, 0.5, 0.5, 1.0, 100, 0.12, 2.0, true);
        soa.set_fitness(1, 1.5, 0.15, -0.10, 1.5, 1.2, 1.8, 100, 0.12, 2.5, true);
        soa.set_fitness(2, 1.0, 0.12, -0.12, 1.0, 0.9, 1.5, 100, 0.14, 2.2, true);

        let sorted = soa.sorted_indices();
        
        // Should be sorted by scalar fitness descending
        assert_eq!(sorted[0], 1); // Highest sharpe
        assert_eq!(sorted[1], 2);
        assert_eq!(sorted[2], 0); // Lowest sharpe
    }

    #[test]
    fn test_validation() {
        let mut soa = PopulationFitnessSoA::with_capacity(10);
        
        soa.set_fitness(0, 1.5, 0.15, -0.10, 1.5, 1.2, 1.8, 100, 0.12, 2.5, true);
        soa.set_validation(0, 0.8, 0.10, 0.65);

        assert!(soa.is_validated[0]);
        assert_eq!(soa.oos_sharpe_medians[0], 0.8);
        assert_eq!(soa.pbos[0], 0.10);
        assert_eq!(soa.dsrs[0], 0.65);
    }

    #[test]
    fn test_reset() {
        let mut soa = PopulationFitnessSoA::with_capacity(10);
        
        soa.set_fitness(0, 1.5, 0.15, -0.10, 1.5, 1.2, 1.8, 100, 0.12, 2.5, true);
        assert_eq!(soa.len(), 1);

        soa.reset();
        
        assert_eq!(soa.len(), 0);
        assert_eq!(soa.sharpe_ratios[0], 0.0);
        assert!(!soa.is_valid[0]);
    }

    #[test]
    fn test_aligned_vec() {
        let v: AlignedVec<f64> = AlignedVec::with_capacity(8);
        
        // Check alignment (32 bytes for AVX2)
        let ptr = v.as_slice().as_ptr();
        assert_eq!(ptr as usize % 32, 0, "Vector should be 32-byte aligned");
    }
}


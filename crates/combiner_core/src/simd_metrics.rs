//! SIMD-accelerated metrics calculations.
//!
//! This module provides high-performance implementations of financial metrics
//! using SIMD instructions. Falls back to scalar implementations for small inputs.
//!
//! Supported operations:
//! - Sharpe ratio (annualized)
//! - Maximum drawdown
//! - Volatility (annualized standard deviation)
//! - Sortino ratio
//! - Calmar ratio
//! - CAGR
//!
//! All functions are designed for zero-allocation hot paths.

use wide::f64x4;

/// Annualization factor for daily returns
const TRADING_DAYS_PER_YEAR: f64 = 252.0;
const SQRT_TRADING_DAYS: f64 = 15.874507866387544; // sqrt(252)

// ============================================================================
// SIMD Sharpe Ratio
// ============================================================================

/// Calculate Sharpe ratio using SIMD (4-wide f64 vectors).
///
/// Formula: (mean_excess_return / std_dev) * sqrt(252)
///
/// # Arguments
/// * `returns` - Daily returns (not annualized)
/// * `rf_rate` - Daily risk-free rate (e.g., 0.0001 for ~2.5% annual)
///
/// # Returns
/// Annualized Sharpe ratio
#[inline]
pub fn sharpe_simd(returns: &[f64], rf_rate: f64) -> f64 {
    let n = returns.len();
    if n < 2 {
        return 0.0;
    }
    
    // Fall back to scalar for small inputs (SIMD overhead not worth it)
    if n < 8 {
        return sharpe_scalar(returns, rf_rate);
    }

    // SIMD calculation
    let rf_vec = f64x4::splat(rf_rate);
    let mut sum_vec = f64x4::splat(0.0);
    let mut sum_sq_vec = f64x4::splat(0.0);

    let chunks = returns.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let r = f64x4::new([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let excess = r - rf_vec;
        sum_vec += excess;
        sum_sq_vec += excess * excess;
    }

    // Reduce SIMD vectors to scalars
    let sum_arr: [f64; 4] = sum_vec.into();
    let sq_arr: [f64; 4] = sum_sq_vec.into();
    
    let mut total_sum: f64 = sum_arr.iter().sum();
    let mut total_sq: f64 = sq_arr.iter().sum();

    // Handle remainder
    for &r in remainder {
        let excess = r - rf_rate;
        total_sum += excess;
        total_sq += excess * excess;
    }

    let n_f64 = n as f64;
    let mean = total_sum / n_f64;
    let variance = (total_sq / n_f64) - (mean * mean);
    
    if variance <= 1e-20 {
        return 0.0;
    }
    
    let std_dev = variance.sqrt();
    (mean / std_dev) * SQRT_TRADING_DAYS
}

/// Scalar Sharpe ratio calculation (fallback for small inputs)
#[inline]
pub fn sharpe_scalar(returns: &[f64], rf_rate: f64) -> f64 {
    let n = returns.len();
    if n < 2 {
        return 0.0;
    }

    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for &r in returns {
        let excess = r - rf_rate;
        sum += excess;
        sum_sq += excess * excess;
    }

    let n_f64 = n as f64;
    let mean = sum / n_f64;
    let variance = (sum_sq / n_f64) - (mean * mean);
    
    if variance <= 1e-20 {
        return 0.0;
    }
    
    let std_dev = variance.sqrt();
    (mean / std_dev) * SQRT_TRADING_DAYS
}

// ============================================================================
// SIMD Maximum Drawdown
// ============================================================================

/// Calculate maximum drawdown using SIMD for cumulative sum.
///
/// Returns the maximum peak-to-trough decline as a negative fraction.
/// E.g., -0.20 means 20% maximum drawdown.
///
/// # Arguments
/// * `returns` - Daily returns (not cumulative)
///
/// # Returns
/// Maximum drawdown as negative fraction (e.g., -0.25 for 25% drawdown)
#[inline]
pub fn max_drawdown_simd(returns: &[f64]) -> f64 {
    let n = returns.len();
    if n == 0 {
        return 0.0;
    }
    
    // For drawdown, we need running max and running NAV
    // SIMD helps with cumulative sum but not running max
    // Use hybrid approach: SIMD for prefix sums, scalar for running max
    
    // Compute cumulative log returns (approximation for small returns)
    // For accuracy, use NAV approach
    let mut nav = 1.0;
    let mut peak = 1.0;
    let mut max_dd = 0.0;

    // Process in chunks for better cache locality
    for chunk in returns.chunks(64) {
        for &r in chunk {
            nav *= 1.0 + r;
            if nav > peak {
                peak = nav;
            }
            let dd = (nav - peak) / peak;
            if dd < max_dd {
                max_dd = dd;
            }
        }
    }

    max_dd
}

/// Scalar maximum drawdown calculation
#[inline]
pub fn max_drawdown_scalar(returns: &[f64]) -> f64 {
    max_drawdown_simd(returns) // Same implementation, kept for API consistency
}

// ============================================================================
// SIMD Volatility (Standard Deviation)
// ============================================================================

/// Calculate annualized volatility using SIMD.
///
/// # Arguments
/// * `returns` - Daily returns
///
/// # Returns
/// Annualized volatility (standard deviation * sqrt(252))
#[inline]
pub fn volatility_simd(returns: &[f64]) -> f64 {
    let n = returns.len();
    if n < 2 {
        return 0.0;
    }
    
    if n < 8 {
        return volatility_scalar(returns);
    }

    let mut sum_vec = f64x4::splat(0.0);
    let mut sum_sq_vec = f64x4::splat(0.0);

    let chunks = returns.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let r = f64x4::new([chunk[0], chunk[1], chunk[2], chunk[3]]);
        sum_vec += r;
        sum_sq_vec += r * r;
    }

    let sum_arr: [f64; 4] = sum_vec.into();
    let sq_arr: [f64; 4] = sum_sq_vec.into();
    
    let mut total_sum: f64 = sum_arr.iter().sum();
    let mut total_sq: f64 = sq_arr.iter().sum();

    for &r in remainder {
        total_sum += r;
        total_sq += r * r;
    }

    let n_f64 = n as f64;
    let mean = total_sum / n_f64;
    let variance = (total_sq / n_f64) - (mean * mean);
    
    if variance <= 0.0 {
        return 0.0;
    }
    
    variance.sqrt() * SQRT_TRADING_DAYS
}

/// Scalar volatility calculation
#[inline]
pub fn volatility_scalar(returns: &[f64]) -> f64 {
    let n = returns.len();
    if n < 2 {
        return 0.0;
    }

    let mean: f64 = returns.iter().sum::<f64>() / n as f64;
    let variance: f64 = returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / n as f64;
    
    if variance <= 0.0 {
        return 0.0;
    }
    
    variance.sqrt() * SQRT_TRADING_DAYS
}

// ============================================================================
// SIMD Sortino Ratio
// ============================================================================

/// Calculate Sortino ratio using SIMD.
///
/// Like Sharpe but only penalizes downside volatility.
///
/// # Arguments
/// * `returns` - Daily returns
/// * `rf_rate` - Daily risk-free rate
/// * `target` - Target return (usually same as rf_rate)
///
/// # Returns
/// Annualized Sortino ratio
#[inline]
pub fn sortino_simd(returns: &[f64], rf_rate: f64, target: f64) -> f64 {
    let n = returns.len();
    if n < 2 {
        return 0.0;
    }
    
    if n < 8 {
        return sortino_scalar(returns, rf_rate, target);
    }

    let rf_vec = f64x4::splat(rf_rate);
    let target_vec = f64x4::splat(target);
    let zero_vec = f64x4::splat(0.0);
    
    let mut sum_vec = f64x4::splat(0.0);
    let mut downside_sq_vec = f64x4::splat(0.0);

    let chunks = returns.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let r = f64x4::new([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let excess = r - rf_vec;
        sum_vec += excess;
        
        // Downside deviation: only count returns below target
        let below_target = r - target_vec;
        // Manual min with zero (SIMD comparison)
        let downside = below_target.min(zero_vec);
        downside_sq_vec += downside * downside;
    }

    let sum_arr: [f64; 4] = sum_vec.into();
    let down_arr: [f64; 4] = downside_sq_vec.into();
    
    let mut total_sum: f64 = sum_arr.iter().sum();
    let mut total_downside_sq: f64 = down_arr.iter().sum();

    for &r in remainder {
        let excess = r - rf_rate;
        total_sum += excess;
        
        let below_target = r - target;
        if below_target < 0.0 {
            total_downside_sq += below_target * below_target;
        }
    }

    let n_f64 = n as f64;
    let mean = total_sum / n_f64;
    let downside_variance = total_downside_sq / n_f64;
    
    if downside_variance <= 1e-20 {
        // No downside deviation - return high value if mean is positive
        return if mean > 0.0 { 10.0 } else { 0.0 };
    }
    
    let downside_dev = downside_variance.sqrt();
    (mean / downside_dev) * SQRT_TRADING_DAYS
}

/// Scalar Sortino ratio calculation
#[inline]
pub fn sortino_scalar(returns: &[f64], rf_rate: f64, target: f64) -> f64 {
    let n = returns.len();
    if n < 2 {
        return 0.0;
    }

    let mut sum = 0.0;
    let mut downside_sq = 0.0;

    for &r in returns {
        let excess = r - rf_rate;
        sum += excess;
        
        let below_target = r - target;
        if below_target < 0.0 {
            downside_sq += below_target * below_target;
        }
    }

    let n_f64 = n as f64;
    let mean = sum / n_f64;
    let downside_variance = downside_sq / n_f64;
    
    if downside_variance <= 1e-20 {
        return if mean > 0.0 { 10.0 } else { 0.0 };
    }
    
    let downside_dev = downside_variance.sqrt();
    (mean / downside_dev) * SQRT_TRADING_DAYS
}

// ============================================================================
// CAGR and Calmar Ratio
// ============================================================================

/// Calculate CAGR (Compound Annual Growth Rate).
///
/// # Arguments
/// * `returns` - Daily returns
///
/// # Returns
/// Annualized compound growth rate
#[inline]
pub fn cagr(returns: &[f64]) -> f64 {
    let n = returns.len();
    if n == 0 {
        return 0.0;
    }

    // Compute total return
    let mut total_return = 1.0;
    for &r in returns {
        total_return *= 1.0 + r;
    }

    // Annualize
    let years = n as f64 / TRADING_DAYS_PER_YEAR;
    if years < 0.01 {
        return 0.0;
    }

    total_return.powf(1.0 / years) - 1.0
}

/// Calculate Calmar ratio (CAGR / |Max Drawdown|).
///
/// # Arguments
/// * `returns` - Daily returns
///
/// # Returns
/// Calmar ratio (higher is better)
#[inline]
pub fn calmar_ratio(returns: &[f64]) -> f64 {
    let cagr_val = cagr(returns);
    let max_dd = max_drawdown_simd(returns);
    
    if max_dd >= -0.001 {
        // No significant drawdown
        return if cagr_val > 0.0 { 10.0 } else { 0.0 };
    }

    cagr_val / max_dd.abs()
}

/// Calculate profit factor (gross profit / gross loss).
///
/// # Arguments
/// * `returns` - Daily returns
///
/// # Returns
/// Profit factor (> 1.0 means profitable)
#[inline]
pub fn profit_factor(returns: &[f64]) -> f64 {
    let mut gross_profit = 0.0;
    let mut gross_loss = 0.0;

    for &r in returns {
        if r > 0.0 {
            gross_profit += r;
        } else {
            gross_loss += r.abs();
        }
    }

    if gross_loss < 1e-10 {
        return if gross_profit > 0.0 { 100.0 } else { 0.0 };
    }

    gross_profit / gross_loss
}

// ============================================================================
// Batch Calculations (compute multiple metrics at once)
// ============================================================================

/// Result of batch metrics calculation
#[derive(Debug, Clone, Default)]
pub struct MetricsBatch {
    pub sharpe_ratio: f64,
    pub volatility: f64,
    pub max_drawdown: f64,
    pub cagr: f64,
    pub sortino_ratio: f64,
    pub calmar_ratio: f64,
    pub profit_factor: f64,
    pub total_return: f64,
}

/// Calculate all metrics in a single pass (optimized for cache efficiency).
///
/// This is more efficient than calling individual functions when you need
/// multiple metrics, as it shares intermediate calculations.
#[inline]
pub fn calculate_all_metrics(returns: &[f64], rf_rate: f64) -> MetricsBatch {
    let n = returns.len();
    if n < 2 {
        return MetricsBatch::default();
    }

    // Single pass for mean, variance, downside variance, NAV, drawdown
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut downside_sq = 0.0;
    let mut gross_profit = 0.0;
    let mut gross_loss = 0.0;
    let mut nav = 1.0;
    let mut peak = 1.0;
    let mut max_dd = 0.0;

    for &r in returns {
        // Mean/variance
        let excess = r - rf_rate;
        sum += excess;
        sum_sq += excess * excess;
        
        // Downside deviation
        if excess < 0.0 {
            downside_sq += excess * excess;
        }
        
        // Profit factor
        if r > 0.0 {
            gross_profit += r;
        } else {
            gross_loss += r.abs();
        }
        
        // NAV and drawdown
        nav *= 1.0 + r;
        if nav > peak {
            peak = nav;
        }
        let dd = (nav - peak) / peak;
        if dd < max_dd {
            max_dd = dd;
        }
    }

    let n_f64 = n as f64;
    let mean = sum / n_f64;
    let variance = (sum_sq / n_f64) - (mean * mean);
    let downside_variance = downside_sq / n_f64;
    
    let std_dev = if variance > 1e-20 { variance.sqrt() } else { 0.0 };
    let downside_dev = if downside_variance > 1e-20 { downside_variance.sqrt() } else { 0.0 };

    let sharpe = if std_dev > 1e-20 { (mean / std_dev) * SQRT_TRADING_DAYS } else { 0.0 };
    let volatility = std_dev * SQRT_TRADING_DAYS;
    
    let sortino = if downside_dev > 1e-20 {
        (mean / downside_dev) * SQRT_TRADING_DAYS
    } else if mean > 0.0 {
        10.0
    } else {
        0.0
    };

    let total_return = nav - 1.0;
    let years = n_f64 / TRADING_DAYS_PER_YEAR;
    let cagr_val = if years > 0.01 { nav.powf(1.0 / years) - 1.0 } else { 0.0 };

    let calmar = if max_dd < -0.001 {
        cagr_val / max_dd.abs()
    } else if cagr_val > 0.0 {
        10.0
    } else {
        0.0
    };

    let pf = if gross_loss > 1e-10 {
        gross_profit / gross_loss
    } else if gross_profit > 0.0 {
        100.0
    } else {
        0.0
    };

    MetricsBatch {
        sharpe_ratio: sharpe,
        volatility,
        max_drawdown: max_dd,
        cagr: cagr_val,
        sortino_ratio: sortino,
        calmar_ratio: calmar,
        profit_factor: pf,
        total_return,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_returns(n: usize, mean: f64, std: f64) -> Vec<f64> {
        // Simple deterministic pseudo-random for tests
        (0..n)
            .map(|i| {
                let x = ((i * 1234567 + 89) % 1000) as f64 / 1000.0 - 0.5;
                mean + x * std * 2.0
            })
            .collect()
    }

    #[test]
    fn test_sharpe_basic() {
        let returns = vec![0.01, 0.02, -0.01, 0.015, 0.005, -0.005, 0.01, 0.02];
        
        let sharpe_s = sharpe_scalar(&returns, 0.0);
        let sharpe_v = sharpe_simd(&returns, 0.0);
        
        assert!((sharpe_s - sharpe_v).abs() < 1e-10, "SIMD and scalar should match");
        assert!(sharpe_s > 0.0, "Positive returns should have positive Sharpe");
    }

    #[test]
    fn test_sharpe_large() {
        let returns = generate_returns(10000, 0.0005, 0.01);
        
        let sharpe_s = sharpe_scalar(&returns, 0.0);
        let sharpe_v = sharpe_simd(&returns, 0.0);
        
        assert!((sharpe_s - sharpe_v).abs() < 1e-8, "SIMD and scalar should match for large inputs");
    }

    #[test]
    fn test_max_drawdown() {
        // Simulating a 10% drawdown
        let returns = vec![0.05, 0.03, -0.05, -0.08, 0.02, 0.01];
        
        let dd = max_drawdown_simd(&returns);
        
        assert!(dd < 0.0, "Drawdown should be negative");
        assert!(dd > -0.20, "Drawdown should not exceed actual decline");
    }

    #[test]
    fn test_volatility() {
        let returns = generate_returns(1000, 0.0, 0.01);
        
        let vol_s = volatility_scalar(&returns);
        let vol_v = volatility_simd(&returns);
        
        assert!((vol_s - vol_v).abs() < 1e-10, "SIMD and scalar should match");
        assert!(vol_v > 0.0, "Volatility should be positive");
    }

    #[test]
    fn test_sortino() {
        let returns = vec![0.01, 0.02, -0.01, 0.015, 0.005, -0.005, 0.01, 0.02];
        
        let sortino_s = sortino_scalar(&returns, 0.0, 0.0);
        let sortino_v = sortino_simd(&returns, 0.0, 0.0);
        
        assert!((sortino_s - sortino_v).abs() < 1e-10, "SIMD and scalar should match");
        assert!(sortino_v > 0.0, "Positive excess returns should have positive Sortino");
    }

    #[test]
    fn test_cagr() {
        // 10% total return over 252 days = ~10% CAGR
        let returns: Vec<f64> = (0..252).map(|_| 0.0004).collect(); // ~10% annual
        
        let c = cagr(&returns);
        
        assert!(c > 0.05 && c < 0.15, "CAGR should be around 10%: {}", c);
    }

    #[test]
    fn test_calmar() {
        let returns = generate_returns(252, 0.0005, 0.01);
        
        let calmar = calmar_ratio(&returns);
        
        // Should be finite and reasonable
        assert!(calmar.is_finite());
    }

    #[test]
    fn test_profit_factor() {
        let returns = vec![0.02, -0.01, 0.03, -0.01, 0.02];
        
        let pf = profit_factor(&returns);
        
        // Gross profit = 0.07, Gross loss = 0.02 -> PF = 3.5
        assert!((pf - 3.5).abs() < 0.01, "Profit factor should be 3.5: {}", pf);
    }

    #[test]
    fn test_batch_metrics() {
        let returns = generate_returns(1000, 0.0003, 0.015);
        
        let batch = calculate_all_metrics(&returns, 0.0);
        
        // Compare with individual calculations
        let sharpe = sharpe_simd(&returns, 0.0);
        let vol = volatility_simd(&returns);
        let dd = max_drawdown_simd(&returns);
        
        assert!((batch.sharpe_ratio - sharpe).abs() < 1e-8);
        assert!((batch.volatility - vol).abs() < 1e-8);
        assert!((batch.max_drawdown - dd).abs() < 1e-8);
    }

    #[test]
    fn test_empty_inputs() {
        let empty: Vec<f64> = vec![];
        
        assert_eq!(sharpe_simd(&empty, 0.0), 0.0);
        assert_eq!(max_drawdown_simd(&empty), 0.0);
        assert_eq!(volatility_simd(&empty), 0.0);
        assert_eq!(cagr(&empty), 0.0);
    }

    #[test]
    fn test_single_value() {
        let single = vec![0.01];
        
        assert_eq!(sharpe_simd(&single, 0.0), 0.0);
        assert_eq!(volatility_simd(&single), 0.0);
    }
}


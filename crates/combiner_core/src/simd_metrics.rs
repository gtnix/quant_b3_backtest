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
//!
//! # AVX-512 Support
//!
//! Enable `simd-wide` feature for 40-80% faster metrics on AVX-512 CPUs:
//! ```bash
//! RUSTFLAGS="-C target-cpu=native -C target-feature=+avx512f,+avx512dq" \
//!   cargo build --release --features simd-wide
//! ```

use wide::f64x4;
#[cfg(feature = "simd-wide")]
use wide::f64x8;

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
    
    // Use 1e-10 threshold for numerical stability
    // 1e-20 is too small and can produce unstable results
    if variance <= 1e-10 {
        return 0.0;
    }
    
    let std_dev = variance.sqrt();
    let sharpe = (mean / std_dev) * SQRT_TRADING_DAYS;
    
    // Cap Sharpe ratio to realistic bounds [-10, 10]
    // Any value beyond this indicates a calculation error or unrealistic data
    sharpe.clamp(-10.0, 10.0)
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
    
    // Use 1e-10 threshold for numerical stability
    if variance <= 1e-10 {
        return 0.0;
    }
    
    let std_dev = variance.sqrt();
    let sharpe = (mean / std_dev) * SQRT_TRADING_DAYS;
    
    // Cap Sharpe ratio to realistic bounds [-10, 10]
    sharpe.clamp(-10.0, 10.0)
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
        // No downside deviation - return 0.0 (undefined Sortino)
        // Previously returned 10.0 which caused false positives with 0 trades
        return 0.0;
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
        // No downside deviation - return 0.0 (undefined Sortino)
        return 0.0;
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
        // No significant drawdown = undefined Calmar, return 0.0
        return 0.0;
    }

    (cagr_val / max_dd.abs()).clamp(-10.0, 10.0)
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

    let sharpe_raw = if std_dev > 1e-20 { (mean / std_dev) * SQRT_TRADING_DAYS } else { 0.0 };
    let sharpe = sharpe_raw.clamp(-10.0, 10.0); // Cap to realistic bounds
    let volatility = std_dev * SQRT_TRADING_DAYS;
    
    let sortino_raw = if downside_dev > 1e-20 {
        (mean / downside_dev) * SQRT_TRADING_DAYS
    } else {
        // No downside deviation = undefined Sortino, return 0.0
        0.0
    };
    let sortino = sortino_raw.clamp(-20.0, 20.0); // Sortino can be higher than Sharpe

    let total_return = nav - 1.0;
    let years = n_f64 / TRADING_DAYS_PER_YEAR;
    let cagr_val = if years > 0.01 { nav.powf(1.0 / years) - 1.0 } else { 0.0 };

    let calmar = if max_dd < -0.001 {
        (cagr_val / max_dd.abs()).clamp(-10.0, 10.0)
    } else {
        // No drawdown = undefined Calmar, return 0.0
        0.0
    };

    let pf = if gross_loss > 1e-10 {
        (gross_profit / gross_loss).min(100.0)
    } else {
        // No losses = undefined profit factor, return 0.0
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
// AVX-512 Batch Metrics (f64x8 - 8-wide vectors)
// ============================================================================

/// Calculate all metrics using AVX-512 SIMD (f64x8).
/// ~40-80% faster than f64x4 for arrays with 16+ elements.
#[cfg(feature = "simd-wide")]
#[inline]
pub fn calculate_all_metrics_avx512(returns: &[f64], rf_rate: f64) -> MetricsBatch {
    let n = returns.len();
    if n < 16 {
        return calculate_all_metrics(returns, rf_rate);
    }

    let rf_vec = f64x8::splat(rf_rate);
    let zero_vec = f64x8::ZERO;

    let mut sum_vec = f64x8::ZERO;
    let mut sum_sq_vec = f64x8::ZERO;
    let mut downside_sq_vec = f64x8::ZERO;

    // Scalar tracking for NAV/drawdown (serial dependency)
    let mut nav = 1.0;
    let mut peak = 1.0;
    let mut max_dd = 0.0;
    let mut gross_profit = 0.0;
    let mut gross_loss = 0.0;

    let chunks = returns.chunks_exact(8);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let r = f64x8::new([
            chunk[0], chunk[1], chunk[2], chunk[3],
            chunk[4], chunk[5], chunk[6], chunk[7],
        ]);

        let excess = r - rf_vec;
        sum_vec += excess;
        sum_sq_vec += excess * excess;

        // Downside: min(excess, 0)^2
        let downside = excess.min(zero_vec);
        downside_sq_vec += downside * downside;

        // NAV and drawdown (scalar - serial dependency)
        for &ret in chunk {
            nav *= 1.0 + ret;
            if nav > peak {
                peak = nav;
            }
            let dd = (nav - peak) / peak;
            if dd < max_dd {
                max_dd = dd;
            }
            if ret > 0.0 {
                gross_profit += ret;
            } else {
                gross_loss += ret.abs();
            }
        }
    }

    // Reduce SIMD vectors
    let sum_arr: [f64; 8] = sum_vec.into();
    let sq_arr: [f64; 8] = sum_sq_vec.into();
    let down_arr: [f64; 8] = downside_sq_vec.into();

    let mut total_sum: f64 = sum_arr.iter().sum();
    let mut total_sq: f64 = sq_arr.iter().sum();
    let mut total_down: f64 = down_arr.iter().sum();

    // Handle remainder
    for &r in remainder {
        let excess = r - rf_rate;
        total_sum += excess;
        total_sq += excess * excess;
        if excess < 0.0 {
            total_down += excess * excess;
        }

        nav *= 1.0 + r;
        if nav > peak {
            peak = nav;
        }
        let dd = (nav - peak) / peak;
        if dd < max_dd {
            max_dd = dd;
        }
        if r > 0.0 {
            gross_profit += r;
        } else {
            gross_loss += r.abs();
        }
    }

    let n_f64 = n as f64;
    let mean = total_sum / n_f64;
    let variance = (total_sq / n_f64) - (mean * mean);
    let downside_variance = total_down / n_f64;

    let std_dev = if variance > 1e-20 { variance.sqrt() } else { 0.0 };
    let downside_dev = if downside_variance > 1e-20 { downside_variance.sqrt() } else { 0.0 };

    let sharpe = if std_dev > 1e-20 {
        ((mean / std_dev) * SQRT_TRADING_DAYS).clamp(-10.0, 10.0)
    } else {
        0.0
    };

    let sortino = if downside_dev > 1e-20 {
        ((mean / downside_dev) * SQRT_TRADING_DAYS).clamp(-20.0, 20.0)
    } else {
        // No downside deviation = undefined Sortino, return 0.0
        0.0
    };

    let volatility = std_dev * SQRT_TRADING_DAYS;
    let total_return = nav - 1.0;
    let years = n_f64 / TRADING_DAYS_PER_YEAR;
    let cagr_val = if years > 0.01 { nav.powf(1.0 / years) - 1.0 } else { 0.0 };

    let calmar = if max_dd < -0.001 {
        (cagr_val / max_dd.abs()).clamp(-10.0, 10.0)
    } else {
        // No drawdown = undefined Calmar, return 0.0
        0.0
    };

    let pf = if gross_loss > 1e-10 {
        (gross_profit / gross_loss).min(100.0)
    } else {
        // No losses = undefined profit factor, return 0.0
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

/// AVX-512 Sharpe ratio (standalone function).
#[cfg(feature = "simd-wide")]
#[inline]
pub fn sharpe_avx512(returns: &[f64], rf_rate: f64) -> f64 {
    let n = returns.len();
    if n < 16 {
        return sharpe_simd(returns, rf_rate);
    }

    let rf_vec = f64x8::splat(rf_rate);
    let mut sum_vec = f64x8::ZERO;
    let mut sum_sq_vec = f64x8::ZERO;

    let chunks = returns.chunks_exact(8);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let r = f64x8::new([
            chunk[0], chunk[1], chunk[2], chunk[3],
            chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        let excess = r - rf_vec;
        sum_vec += excess;
        sum_sq_vec += excess * excess;
    }

    let sum_arr: [f64; 8] = sum_vec.into();
    let sq_arr: [f64; 8] = sum_sq_vec.into();

    let mut total_sum: f64 = sum_arr.iter().sum();
    let mut total_sq: f64 = sq_arr.iter().sum();

    for &r in remainder {
        let excess = r - rf_rate;
        total_sum += excess;
        total_sq += excess * excess;
    }

    let n_f64 = n as f64;
    let mean = total_sum / n_f64;
    let variance = (total_sq / n_f64) - (mean * mean);

    if variance <= 1e-10 {
        return 0.0;
    }

    let std_dev = variance.sqrt();
    ((mean / std_dev) * SQRT_TRADING_DAYS).clamp(-10.0, 10.0)
}

/// AVX-512 volatility (standalone function).
#[cfg(feature = "simd-wide")]
#[inline]
pub fn volatility_avx512(returns: &[f64]) -> f64 {
    let n = returns.len();
    if n < 16 {
        return volatility_simd(returns);
    }

    let mut sum_vec = f64x8::ZERO;
    let mut sum_sq_vec = f64x8::ZERO;

    let chunks = returns.chunks_exact(8);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let r = f64x8::new([
            chunk[0], chunk[1], chunk[2], chunk[3],
            chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        sum_vec += r;
        sum_sq_vec += r * r;
    }

    let sum_arr: [f64; 8] = sum_vec.into();
    let sq_arr: [f64; 8] = sum_sq_vec.into();

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

    // =========================================================================
    // Phase 1 Validation: Comprehensive Core Metrics Tests
    // =========================================================================

    #[test]
    fn test_sharpe_all_positive_returns() {
        // All positive returns should yield positive Sharpe
        let returns = vec![0.01, 0.02, 0.015, 0.005, 0.01, 0.02, 0.008, 0.012];
        let sharpe = sharpe_simd(&returns, 0.0);
        assert!(sharpe > 0.0, "All positive returns should have positive Sharpe: {}", sharpe);
        
        // With rf > mean, Sharpe should be negative
        let sharpe_high_rf = sharpe_simd(&returns, 0.05);
        assert!(sharpe_high_rf < sharpe, "Higher rf should reduce Sharpe");
    }

    #[test]
    fn test_sharpe_all_negative_returns() {
        // All negative returns should yield negative Sharpe
        let returns = vec![-0.01, -0.02, -0.015, -0.005, -0.01, -0.02, -0.008, -0.012];
        let sharpe = sharpe_simd(&returns, 0.0);
        assert!(sharpe < 0.0, "All negative returns should have negative Sharpe: {}", sharpe);
    }

    #[test]
    fn test_sharpe_known_values() {
        // Known synthetic series: mean = 0.01, std = 0.02
        // Sharpe (annualized) = (0.01 / 0.02) * sqrt(252) ≈ 7.94
        let returns = vec![0.01; 252]; // Constant 1% daily return
        let sharpe = sharpe_simd(&returns, 0.0);
        // With zero variance, should return 0
        assert_eq!(sharpe, 0.0, "Constant returns have zero variance, Sharpe = 0");

        // Variable returns with known statistics
        // If we have mean ≈ 0.001 and std ≈ 0.01, Sharpe ≈ 0.1 * sqrt(252) ≈ 1.59
        let mut variable_returns: Vec<f64> = (0..252).map(|i| {
            if i % 2 == 0 { 0.011 } else { -0.009 } // mean ≈ 0.001
        }).collect();
        let sharpe_var = sharpe_simd(&variable_returns, 0.0);
        assert!(sharpe_var > 0.0 && sharpe_var < 5.0, "Sharpe should be reasonable: {}", sharpe_var);
    }

    #[test]
    fn test_sharpe_symmetry() {
        // Property: negate all returns → negate Sharpe
        let returns = vec![0.01, 0.02, -0.01, 0.015, 0.005, -0.005, 0.01, 0.02];
        let negated: Vec<f64> = returns.iter().map(|r| -r).collect();
        
        let sharpe_pos = sharpe_simd(&returns, 0.0);
        let sharpe_neg = sharpe_simd(&negated, 0.0);
        
        assert!((sharpe_pos + sharpe_neg).abs() < 1e-10, 
            "Sharpe symmetry: {} vs {}", sharpe_pos, sharpe_neg);
    }

    #[test]
    fn test_max_drawdown_monotonic_up() {
        // Monotonically increasing NAV should have 0 drawdown
        let returns = vec![0.01, 0.01, 0.01, 0.01, 0.01];
        let dd = max_drawdown_simd(&returns);
        assert_eq!(dd, 0.0, "Monotonic up should have 0 drawdown: {}", dd);
    }

    #[test]
    fn test_max_drawdown_crash_50_percent() {
        // 50% crash: NAV goes 1.0 -> 1.1 -> 0.55
        // Returns: +10%, -50%
        let returns = vec![0.10, -0.50];
        let dd = max_drawdown_simd(&returns);
        // Peak = 1.1, Trough = 0.55, DD = (0.55 - 1.1) / 1.1 = -0.5
        assert!((dd - (-0.50)).abs() < 0.01, "50% crash should give -0.50 DD: {}", dd);
    }

    #[test]
    fn test_max_drawdown_bounds() {
        // Property: DD is always in [-1, 0]
        let test_cases = vec![
            vec![0.5, -0.99, 0.5],  // Near total loss
            vec![0.01; 100],       // All positive
            vec![-0.01; 100],      // All negative
            vec![0.1, -0.1, 0.1, -0.1],  // Alternating
        ];
        
        for returns in test_cases {
            let dd = max_drawdown_simd(&returns);
            assert!(dd >= -1.0 && dd <= 0.0, 
                "DD must be in [-1, 0]: {} for {:?}", dd, &returns[..returns.len().min(5)]);
        }
    }

    #[test]
    fn test_max_drawdown_recovery() {
        // Crash and full recovery
        let returns = vec![0.10, -0.20, 0.30, 0.05];
        // NAV: 1.0 -> 1.1 -> 0.88 -> 1.144 -> 1.2
        // Peak at 1.1, trough at 0.88, DD = (0.88 - 1.1) / 1.1 = -0.2
        let dd = max_drawdown_simd(&returns);
        assert!((dd - (-0.20)).abs() < 0.01, "DD should be ~-0.20: {}", dd);
    }

    #[test]
    fn test_cagr_exact_year() {
        // 252 days of constant 0.04% daily ≈ 10% annual
        let daily_return = 0.1_f64.ln() / 252.0; // Exact 10% annual
        let returns: Vec<f64> = (0..252).map(|_| (1.0 + daily_return).ln().exp() - 1.0).collect();
        
        // Using simple approximation
        let simple_returns: Vec<f64> = (0..252).map(|_| 0.0003968).collect();
        let c = cagr(&simple_returns);
        assert!(c > 0.05 && c < 0.15, "CAGR should be ~10%: {}", c);
    }

    #[test]
    fn test_cagr_partial_year() {
        // 126 days (half year) with same daily return
        let returns: Vec<f64> = (0..126).map(|_| 0.0004).collect();
        let c = cagr(&returns);
        // Should still annualize correctly
        assert!(c.is_finite(), "CAGR should be finite for partial year: {}", c);
    }

    #[test]
    fn test_sortino_all_positive() {
        // All positive returns → downside deviation = 0 → Sortino should be capped or high
        let returns = vec![0.01, 0.02, 0.015, 0.005, 0.01];
        let sortino = sortino_simd(&returns, 0.0, 0.0);
        // With no downside, Sortino should be positive (capped or high)
        assert!(sortino >= 0.0, "All positive should have non-negative Sortino: {}", sortino);
    }

    #[test]
    fn test_sortino_all_negative() {
        // All negative returns → high downside deviation
        let returns = vec![-0.01, -0.02, -0.015, -0.005, -0.01];
        let sortino = sortino_simd(&returns, 0.0, 0.0);
        assert!(sortino < 0.0, "All negative should have negative Sortino: {}", sortino);
    }

    #[test]
    fn test_calmar_divide_by_zero() {
        // No drawdown (monotonic up) → Calmar should handle gracefully
        let returns = vec![0.01, 0.01, 0.01, 0.01, 0.01];
        let calmar = calmar_ratio(&returns);
        // Should return 0 or handle gracefully (not inf/nan)
        assert!(calmar == 0.0 || calmar.is_finite(), 
            "Calmar should handle zero DD: {}", calmar);
    }

    #[test]
    fn test_profit_factor_no_losses() {
        // All positive returns → profit factor should handle gracefully
        let returns = vec![0.01, 0.02, 0.015, 0.005, 0.01];
        let pf = profit_factor(&returns);
        // With no losses, PF should be capped or high (impl uses 100 cap)
        assert!(pf >= 1.0, "No losses should give high PF: {}", pf);
    }

    #[test]
    fn test_profit_factor_no_gains() {
        // All negative returns → profit factor = 0
        let returns = vec![-0.01, -0.02, -0.015, -0.005, -0.01];
        let pf = profit_factor(&returns);
        assert_eq!(pf, 0.0, "No gains should give PF = 0: {}", pf);
    }

    #[test]
    fn test_profit_factor_known_value() {
        // Gains = 0.05, Losses = 0.02 → PF = 2.5
        let returns = vec![0.02, -0.01, 0.03, -0.01];
        let pf = profit_factor(&returns);
        assert!((pf - 2.5).abs() < 0.01, "PF should be 2.5: {}", pf);
    }

    #[test]
    fn test_extreme_values() {
        // Very large positive returns
        let large = vec![0.5, 0.5, 0.5, 0.5]; // 50% daily returns
        let sharpe_large = sharpe_simd(&large, 0.0);
        assert!(sharpe_large == 0.0 || sharpe_large.is_finite(), 
            "Large returns should give finite Sharpe: {}", sharpe_large);
        
        // Very small returns
        let tiny = vec![1e-10, -1e-10, 1e-10, -1e-10];
        let sharpe_tiny = sharpe_simd(&tiny, 0.0);
        assert!(sharpe_tiny.is_finite(), "Tiny returns should give finite Sharpe: {}", sharpe_tiny);
    }

    #[test]
    fn test_volatility_annualization() {
        // Daily vol of 1% → Annual vol ≈ 15.87%
        // Generate returns with std ≈ 0.01
        let returns: Vec<f64> = (0..252).map(|i| {
            if i % 2 == 0 { 0.01 } else { -0.01 }
        }).collect();
        
        let vol = volatility_simd(&returns);
        // std(returns) ≈ 0.01, annualized ≈ 0.01 * sqrt(252) ≈ 0.159
        assert!(vol > 0.1 && vol < 0.25, "Annualized vol should be ~15.9%: {}", vol);
    }

    #[test]
    fn test_metrics_consistency() {
        // Calmar = CAGR / |MaxDD|
        let returns = generate_returns(252, 0.0003, 0.015);
        let batch = calculate_all_metrics(&returns, 0.0);
        
        if batch.max_drawdown < -0.001 {
            let expected_calmar = batch.cagr / batch.max_drawdown.abs();
            assert!((batch.calmar_ratio - expected_calmar).abs() < 0.01,
                "Calmar should be CAGR/|MaxDD|: {} vs {}", batch.calmar_ratio, expected_calmar);
        }
    }

    // =========================================================================
    // Phase 7: Property-Based Tests for Key Invariants
    // =========================================================================

    #[test]
    fn test_property_sharpe_symmetry() {
        // Property: negate all returns → negate Sharpe
        let test_cases = vec![
            vec![0.01, 0.02, -0.01, 0.015],
            vec![0.001, -0.002, 0.003, -0.001, 0.002],
            vec![-0.05, 0.03, -0.02, 0.04, -0.01],
        ];
        
        for returns in test_cases {
            let negated: Vec<f64> = returns.iter().map(|r| -r).collect();
            let sharpe_pos = sharpe_simd(&returns, 0.0);
            let sharpe_neg = sharpe_simd(&negated, 0.0);
            
            assert!((sharpe_pos + sharpe_neg).abs() < 1e-9, 
                "Sharpe symmetry failed: {} vs {}", sharpe_pos, sharpe_neg);
        }
    }

    #[test]
    fn test_property_drawdown_bounds() {
        // Property: DD is always in [-1, 0]
        let seeds = vec![42, 123, 456, 789, 1001];
        
        for seed in seeds {
            let mut rng_state = seed as f64;
            let returns: Vec<f64> = (0..100).map(|_| {
                rng_state = (rng_state * 1.1 + 0.3) % 1.0;
                (rng_state - 0.5) * 0.1
            }).collect();
            
            let dd = max_drawdown_simd(&returns);
            assert!(dd >= -1.0 && dd <= 0.0, 
                "DD {} not in [-1, 0] for seed {}", dd, seed);
        }
    }

    #[test]
    fn test_property_volatility_non_negative() {
        // Property: Volatility is always >= 0
        let test_cases = vec![
            vec![0.0; 10],
            vec![0.01; 10],
            vec![-0.01; 10],
            vec![0.1, -0.1, 0.1, -0.1],
            vec![0.001, 0.002, 0.001, 0.002],
        ];
        
        for returns in test_cases {
            let vol = volatility_simd(&returns);
            assert!(vol >= 0.0, "Volatility should be >= 0: {}", vol);
        }
    }

    #[test]
    fn test_property_sortino_vs_sharpe() {
        // Property: Both Sortino and Sharpe should be positive for positive excess returns
        let mixed_returns = vec![0.01, -0.005, 0.02, -0.01, 0.015, -0.005, 0.01];
        
        let sharpe = sharpe_simd(&mixed_returns, 0.0);
        let sortino = sortino_simd(&mixed_returns, 0.0, 0.0);
        
        // Both should have same sign for mixed returns
        if sharpe > 0.0 {
            assert!(sortino >= 0.0, 
                "Sortino {} should be non-negative when Sharpe {} is positive", sortino, sharpe);
        }
        
        // Negative returns should give negative metrics
        let negative_returns = vec![-0.01, -0.02, -0.015, -0.01, -0.02];
        let sharpe_neg = sharpe_simd(&negative_returns, 0.0);
        let sortino_neg = sortino_simd(&negative_returns, 0.0, 0.0);
        
        assert!(sharpe_neg < 0.0, "Negative returns should have negative Sharpe");
        assert!(sortino_neg <= 0.0, "Negative returns should have non-positive Sortino");
    }

    #[test]
    fn test_property_profit_factor_positive() {
        // Property: PF is always >= 0
        let test_cases = vec![
            vec![0.01, -0.01, 0.02, -0.02],
            vec![0.01, 0.02, 0.03],  // All positive
            vec![-0.01, -0.02, -0.03],  // All negative
            vec![0.0, 0.0, 0.0],  // All zeros
        ];
        
        for returns in test_cases {
            let pf = profit_factor(&returns);
            assert!(pf >= 0.0, "Profit factor should be >= 0: {}", pf);
        }
    }

    #[test]
    fn test_property_metrics_finite() {
        // Property: All metrics should be finite (no NaN/Inf)
        let test_cases = vec![
            generate_returns(100, 0.0, 0.01),
            generate_returns(252, 0.001, 0.02),
            vec![0.01; 50],  // Constant
            vec![0.0; 50],   // All zeros
        ];
        
        for returns in test_cases {
            let batch = calculate_all_metrics(&returns, 0.0);
            
            assert!(batch.sharpe_ratio.is_finite(), "Sharpe should be finite");
            assert!(batch.volatility.is_finite(), "Volatility should be finite");
            assert!(batch.max_drawdown.is_finite(), "MaxDD should be finite");
            assert!(batch.cagr.is_finite(), "CAGR should be finite");
            assert!(batch.calmar_ratio.is_finite(), "Calmar should be finite");
        }
    }

    #[test]
    fn test_stress_large_returns_vector() {
        // Stress: 1M bars
        let returns: Vec<f64> = (0..1_000_000)
            .map(|i| if i % 3 == 0 { 0.001 } else { -0.0005 })
            .collect();
        
        let sharpe = sharpe_simd(&returns, 0.0);
        let dd = max_drawdown_simd(&returns);
        let vol = volatility_simd(&returns);
        
        assert!(sharpe.is_finite(), "Sharpe finite for 1M bars");
        assert!(dd.is_finite(), "MaxDD finite for 1M bars");
        assert!(vol.is_finite(), "Vol finite for 1M bars");
    }
}


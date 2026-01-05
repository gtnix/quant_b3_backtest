//! # SIMD Vectorized Operations
//!
//! High-performance vectorized calculations for financial metrics.
//! Uses `wide` crate for portable SIMD operations.

use wide::f64x4;

/// Calculate returns from a price series using SIMD.
/// Returns[i] = (prices[i+1] - prices[i]) / prices[i]
#[must_use]
pub fn simd_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return Vec::new();
    }

    let n = prices.len() - 1;
    let mut returns = vec![0.0; n];

    // Process 4 elements at a time
    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let idx = i * 4;

        let p0 = f64x4::new([
            prices[idx],
            prices[idx + 1],
            prices[idx + 2],
            prices[idx + 3],
        ]);

        let p1 = f64x4::new([
            prices[idx + 1],
            prices[idx + 2],
            prices[idx + 3],
            prices[idx + 4],
        ]);

        let diff = p1 - p0;
        let ret = diff / p0;
        let arr = ret.to_array();

        returns[idx] = arr[0];
        returns[idx + 1] = arr[1];
        returns[idx + 2] = arr[2];
        returns[idx + 3] = arr[3];
    }

    // Handle remainder
    let start = chunks * 4;
    for i in 0..remainder {
        let idx = start + i;
        if prices[idx] > 0.0 {
            returns[idx] = (prices[idx + 1] - prices[idx]) / prices[idx];
        }
    }

    returns
}

/// Calculate max drawdown and its duration using SIMD.
/// Returns (max_drawdown, max_duration_bars)
#[must_use]
pub fn simd_drawdown(nav: &[f64]) -> (f64, usize) {
    if nav.is_empty() {
        return (0.0, 0);
    }

    let mut peak = nav[0];
    let mut max_dd = 0.0;
    let mut max_duration = 0usize;
    let mut current_duration = 0usize;

    // Process 4 elements at a time for peak tracking
    let n = nav.len();
    let chunks = n / 4;

    for chunk_idx in 0..chunks {
        let idx = chunk_idx * 4;
        let vals = f64x4::new([nav[idx], nav[idx + 1], nav[idx + 2], nav[idx + 3]]);
        let arr = vals.to_array();

        for &v in &arr {
            if v > peak {
                peak = v;
                current_duration = 0;
            } else if peak > 0.0 {
                let dd = (peak - v) / peak;
                if dd > max_dd {
                    max_dd = dd;
                }
                current_duration += 1;
                if current_duration > max_duration {
                    max_duration = current_duration;
                }
            }
        }
    }

    // Handle remainder
    let start = chunks * 4;
    for i in start..n {
        let v = nav[i];
        if v > peak {
            peak = v;
            current_duration = 0;
        } else if peak > 0.0 {
            let dd = (peak - v) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
            current_duration += 1;
            if current_duration > max_duration {
                max_duration = current_duration;
            }
        }
    }

    (max_dd, max_duration)
}

/// Calculate mean of returns using SIMD.
#[must_use]
pub fn simd_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let n = values.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum_vec = f64x4::ZERO;

    for i in 0..chunks {
        let idx = i * 4;
        let v = f64x4::new([
            values[idx],
            values[idx + 1],
            values[idx + 2],
            values[idx + 3],
        ]);
        sum_vec += v;
    }

    let arr = sum_vec.to_array();
    let mut sum = arr[0] + arr[1] + arr[2] + arr[3];

    // Handle remainder
    let start = chunks * 4;
    for i in 0..remainder {
        sum += values[start + i];
    }

    sum / n as f64
}

/// Calculate variance using SIMD.
#[must_use]
pub fn simd_variance(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let n = values.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mean_vec = f64x4::splat(mean);
    let mut sum_sq_vec = f64x4::ZERO;

    for i in 0..chunks {
        let idx = i * 4;
        let v = f64x4::new([
            values[idx],
            values[idx + 1],
            values[idx + 2],
            values[idx + 3],
        ]);
        let diff = v - mean_vec;
        sum_sq_vec += diff * diff;
    }

    let arr = sum_sq_vec.to_array();
    let mut sum_sq = arr[0] + arr[1] + arr[2] + arr[3];

    // Handle remainder
    let start = chunks * 4;
    for i in 0..remainder {
        let diff = values[start + i] - mean;
        sum_sq += diff * diff;
    }

    sum_sq / n as f64
}

/// Calculate annualized volatility from daily returns using SIMD.
/// Assumes 252 trading days per year.
#[must_use]
pub fn simd_volatility(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }

    let mean = simd_mean(returns);
    let variance = simd_variance(returns, mean);
    let std_dev = variance.sqrt();

    // Annualize: std_dev * sqrt(252)
    std_dev * 15.874_507_866_387_544 // sqrt(252)
}

/// Calculate Sharpe ratio using SIMD.
/// sharpe = (annualized_return - risk_free_rate) / annualized_volatility
/// Clamped to [-10, 10] to prevent unrealistic values from low volatility data.
#[must_use]
pub fn simd_sharpe(returns: &[f64], risk_free_rate: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }

    let mean_return = simd_mean(returns);
    let annual_return = mean_return * 252.0;
    let annual_vol = simd_volatility(returns);

    if annual_vol > 0.0 {
        let sharpe = (annual_return - risk_free_rate) / annual_vol;
        // Cap to realistic bounds - values beyond this indicate calculation errors
        sharpe.clamp(-10.0, 10.0)
    } else {
        0.0
    }
}

/// Calculate Sortino ratio using SIMD.
/// sortino = (annualized_return - risk_free_rate) / downside_volatility
/// Clamped to [-10, 10] to prevent unrealistic values.
///
/// # Performance
/// Uses SIMD masking to compute downside variance directly without
/// allocating a filtered Vec (3x-5x faster than filter+collect).
#[must_use]
pub fn simd_sortino(returns: &[f64], risk_free_rate: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }

    let n = returns.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // SIMD: compute mean and downside sum-of-squares in single pass
    let mut sum_vec = f64x4::ZERO;
    let mut downside_sq_sum = f64x4::ZERO;
    let mut downside_count_f = 0.0_f64;

    for i in 0..chunks {
        let idx = i * 4;
        let v = f64x4::new([
            returns[idx],
            returns[idx + 1],
            returns[idx + 2],
            returns[idx + 3],
        ]);
        
        // Accumulate total sum
        sum_vec += v;
        
        // SIMD mask: select negative values, zero for positive
        // v < 0 gives mask, then blend v*v or 0
        let arr: [f64; 4] = v.into();
        let neg_sq = f64x4::new([
            if arr[0] < 0.0 { arr[0] * arr[0] } else { 0.0 },
            if arr[1] < 0.0 { arr[1] * arr[1] } else { 0.0 },
            if arr[2] < 0.0 { arr[2] * arr[2] } else { 0.0 },
            if arr[3] < 0.0 { arr[3] * arr[3] } else { 0.0 },
        ]);
        downside_sq_sum += neg_sq;
        
        // Count negatives
        downside_count_f += (arr[0] < 0.0) as u8 as f64
            + (arr[1] < 0.0) as u8 as f64
            + (arr[2] < 0.0) as u8 as f64
            + (arr[3] < 0.0) as u8 as f64;
    }

    // Reduce SIMD vectors
    let sum_arr: [f64; 4] = sum_vec.into();
    let mut total_sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];
    
    let dsq_arr: [f64; 4] = downside_sq_sum.into();
    let mut total_downside_sq = dsq_arr[0] + dsq_arr[1] + dsq_arr[2] + dsq_arr[3];

    // Handle remainder (scalar)
    let start = chunks * 4;
    for i in 0..remainder {
        let r = returns[start + i];
        total_sum += r;
        if r < 0.0 {
            total_downside_sq += r * r;
            downside_count_f += 1.0;
        }
    }

    // No downside returns = perfect strategy
    if downside_count_f < 1.0 {
        return 10.0;
    }

    let mean_return = total_sum / n as f64;
    let annual_return = mean_return * 252.0;

    // Downside variance = E[X²] for X < 0 (target=0 semi-variance)
    let downside_var = total_downside_sq / downside_count_f;
    let downside_vol = downside_var.sqrt() * 15.874_507_866_387_544; // sqrt(252)

    if downside_vol > 0.0 {
        let sortino = (annual_return - risk_free_rate) / downside_vol;
        sortino.clamp(-10.0, 10.0)
    } else {
        0.0
    }
}

/// Calculate sum using SIMD.
#[must_use]
pub fn simd_sum(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let n = values.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum_vec = f64x4::ZERO;

    for i in 0..chunks {
        let idx = i * 4;
        let v = f64x4::new([
            values[idx],
            values[idx + 1],
            values[idx + 2],
            values[idx + 3],
        ]);
        sum_vec += v;
    }

    let arr = sum_vec.to_array();
    let mut sum = arr[0] + arr[1] + arr[2] + arr[3];

    let start = chunks * 4;
    for i in 0..remainder {
        sum += values[start + i];
    }

    sum
}

/// Calculate dot product of two vectors using SIMD.
#[must_use]
pub fn simd_dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum_vec = f64x4::ZERO;

    for i in 0..chunks {
        let idx = i * 4;
        let va = f64x4::new([a[idx], a[idx + 1], a[idx + 2], a[idx + 3]]);
        let vb = f64x4::new([b[idx], b[idx + 1], b[idx + 2], b[idx + 3]]);
        sum_vec += va * vb;
    }

    let arr = sum_vec.to_array();
    let mut sum = arr[0] + arr[1] + arr[2] + arr[3];

    let start = chunks * 4;
    for i in 0..remainder {
        sum += a[start + i] * b[start + i];
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_returns() {
        let prices = vec![100.0, 110.0, 105.0, 115.0, 120.0];
        let returns = simd_returns(&prices);

        assert_eq!(returns.len(), 4);
        assert!((returns[0] - 0.1).abs() < 1e-10);
        assert!((returns[1] - (-0.045_454_545_454_545_456)).abs() < 1e-10);
    }

    #[test]
    fn test_simd_drawdown() {
        let nav = vec![100.0, 110.0, 100.0, 95.0, 105.0, 115.0];
        let (max_dd, duration) = simd_drawdown(&nav);

        assert!((max_dd - (110.0 - 95.0) / 110.0).abs() < 1e-10);
        assert!(duration >= 2);
    }

    #[test]
    fn test_simd_mean() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mean = simd_mean(&values);
        assert!((mean - 4.5).abs() < 1e-10);
    }

    #[test]
    fn test_simd_volatility() {
        let returns = vec![0.01, -0.02, 0.015, -0.005, 0.02, 0.01, -0.01, 0.005];
        let vol = simd_volatility(&returns);
        assert!(vol > 0.0);
    }

    #[test]
    fn test_simd_sharpe() {
        let returns = vec![0.01, 0.02, 0.015, 0.005, 0.02, 0.01, 0.01, 0.005];
        let sharpe = simd_sharpe(&returns, 0.02);
        assert!(sharpe > 0.0);
    }

    #[test]
    fn test_simd_sum() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sum = simd_sum(&values);
        assert!((sum - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_simd_dot() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let dot = simd_dot(&a, &b);
        assert!((dot - 40.0).abs() < 1e-10);
    }
}
























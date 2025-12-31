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
#[must_use]
pub fn simd_sharpe(returns: &[f64], risk_free_rate: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }

    let mean_return = simd_mean(returns);
    let annual_return = mean_return * 252.0;
    let annual_vol = simd_volatility(returns);

    if annual_vol > 0.0 {
        (annual_return - risk_free_rate) / annual_vol
    } else {
        0.0
    }
}

/// Calculate Sortino ratio using SIMD.
/// sortino = (annualized_return - risk_free_rate) / downside_volatility
#[must_use]
pub fn simd_sortino(returns: &[f64], risk_free_rate: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }

    // Filter negative returns
    let downside: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();

    if downside.is_empty() {
        return f64::INFINITY;
    }

    let mean_return = simd_mean(returns);
    let annual_return = mean_return * 252.0;

    // Downside volatility
    let downside_mean = simd_mean(&downside);
    let downside_var = simd_variance(&downside, downside_mean);
    let downside_vol = downside_var.sqrt() * 15.874_507_866_387_544;

    if downside_vol > 0.0 {
        (annual_return - risk_free_rate) / downside_vol
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






















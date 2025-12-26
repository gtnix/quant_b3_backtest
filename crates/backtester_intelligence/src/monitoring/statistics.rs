//! Statistical utilities for monitoring checks.
//!
//! Implements:
//! - Percentile calculation
//! - Mean/std calculation  
//! - Hoeffding bounds for drift detection
//! - Kolmogorov-Smirnov two-sample test
//! - Jaccard similarity for selection overlap
//! - Baseline statistics computation

use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use rust_decimal_macros::dec;
use std::collections::HashSet;

use super::types::BaselineStats;

/// Calculate percentile from sorted data.
///
/// Uses linear interpolation between data points.
pub fn calculate_percentile(data: &[Decimal], p: Decimal) -> Option<Decimal> {
    if data.is_empty() {
        return None;
    }
    if data.len() == 1 {
        return Some(data[0]);
    }

    let mut sorted = data.to_vec();
    sorted.sort();

    let n = Decimal::from(sorted.len() - 1);
    let idx_f = p / dec!(100) * n;
    let idx_low = idx_f.floor().to_usize()?;
    let idx_high = idx_f.ceil().to_usize()?.min(sorted.len() - 1);

    if idx_low == idx_high {
        Some(sorted[idx_low])
    } else {
        let frac = idx_f - Decimal::from(idx_low);
        Some(sorted[idx_low] + frac * (sorted[idx_high] - sorted[idx_low]))
    }
}

/// Calculate mean of data.
pub fn calculate_mean(data: &[Decimal]) -> Option<Decimal> {
    if data.is_empty() {
        return None;
    }
    let sum: Decimal = data.iter().sum();
    Some(sum / Decimal::from(data.len()))
}

/// Calculate standard deviation of data.
pub fn calculate_std(data: &[Decimal]) -> Option<Decimal> {
    if data.len() < 2 {
        return None;
    }
    
    let mean = calculate_mean(data)?;
    let variance: Decimal = data.iter()
        .map(|x| (*x - mean) * (*x - mean))
        .sum::<Decimal>() / Decimal::from(data.len() - 1);
    
    // Approximate sqrt using Newton-Raphson
    decimal_sqrt(variance)
}

/// Calculate complete baseline statistics from data.
pub fn calculate_baseline(data: &[Decimal], window_days: u32) -> Option<BaselineStats> {
    if data.is_empty() {
        return None;
    }

    let mean = calculate_mean(data)?;
    let std = calculate_std(data).unwrap_or(Decimal::ZERO);
    
    let mut sorted = data.to_vec();
    sorted.sort();

    Some(BaselineStats {
        mean,
        std,
        min: sorted[0],
        max: sorted[sorted.len() - 1],
        p50: calculate_percentile(data, dec!(50))?,
        p95: calculate_percentile(data, dec!(95))?,
        p99: calculate_percentile(data, dec!(99))?,
        n: data.len(),
        window_days,
        computed_at: Some(chrono::Utc::now()),
    })
}

/// Hoeffding bound for drift detection.
///
/// Given n samples from a distribution, returns the maximum expected
/// deviation of the sample mean from the true mean with probability 1-delta.
///
/// Formula: sqrt(ln(1/delta) / (2*n))
pub fn hoeffding_bound(n: usize, delta: f64) -> f64 {
    if n == 0 {
        return f64::INFINITY;
    }
    ((-delta.ln()) / (2.0 * n as f64)).sqrt()
}

/// Kolmogorov-Smirnov two-sample test result.
#[derive(Debug, Clone)]
pub struct KsResult {
    /// KS statistic (max absolute difference between CDFs)
    pub statistic: Decimal,
    /// Approximate p-value
    pub p_value: Decimal,
    /// Sample sizes
    pub n1: usize,
    pub n2: usize,
}

/// Kolmogorov-Smirnov two-sample test.
///
/// Tests whether two samples come from the same distribution.
/// Returns KS statistic and approximate p-value.
pub fn ks_two_sample(sample_a: &[Decimal], sample_b: &[Decimal]) -> Option<KsResult> {
    if sample_a.is_empty() || sample_b.is_empty() {
        return None;
    }

    let n1 = sample_a.len();
    let n2 = sample_b.len();

    // Combine and sort all values
    let mut all_values: Vec<Decimal> = sample_a.iter().chain(sample_b.iter()).copied().collect();
    all_values.sort();
    all_values.dedup();

    let mut max_diff = Decimal::ZERO;

    for &x in &all_values {
        // Calculate empirical CDFs at point x
        let cdf_a = sample_a.iter().filter(|&&v| v <= x).count() as f64 / n1 as f64;
        let cdf_b = sample_b.iter().filter(|&&v| v <= x).count() as f64 / n2 as f64;
        
        let diff = Decimal::from_f64((cdf_a - cdf_b).abs()).unwrap_or(Decimal::ZERO);
        if diff > max_diff {
            max_diff = diff;
        }
    }

    // Approximate p-value using asymptotic formula
    // For large samples: p ≈ 2 * exp(-2 * D^2 * n1 * n2 / (n1 + n2))
    let d_f64 = max_diff.to_f64().unwrap_or(0.0);
    let effective_n = (n1 * n2) as f64 / (n1 + n2) as f64;
    let p_approx = 2.0 * (-2.0 * d_f64 * d_f64 * effective_n).exp();
    let p_value = Decimal::from_f64(p_approx.min(1.0)).unwrap_or(dec!(1));

    Some(KsResult {
        statistic: max_diff,
        p_value,
        n1,
        n2,
    })
}

/// Jaccard similarity between two sets.
///
/// J(A,B) = |A ∩ B| / |A ∪ B|
/// Returns value in [0, 1] where 1 means identical sets.
pub fn jaccard_similarity<T: Eq + std::hash::Hash>(set_a: &HashSet<T>, set_b: &HashSet<T>) -> Decimal {
    if set_a.is_empty() && set_b.is_empty() {
        return dec!(1);
    }
    
    let intersection = set_a.intersection(set_b).count();
    let union = set_a.union(set_b).count();
    
    if union == 0 {
        return dec!(0);
    }
    
    Decimal::from(intersection) / Decimal::from(union)
}

/// Calculate how many standard deviations a value is from the mean.
pub fn sigma_deviation(value: Decimal, mean: Decimal, std: Decimal) -> Option<Decimal> {
    if std.is_zero() {
        return if value == mean { Some(Decimal::ZERO) } else { None };
    }
    Some((value - mean) / std)
}

/// Approximate square root for Decimal using Newton-Raphson.
fn decimal_sqrt(x: Decimal) -> Option<Decimal> {
    if x.is_sign_negative() {
        return None;
    }
    if x.is_zero() {
        return Some(Decimal::ZERO);
    }

    // Initial guess
    let x_f64 = x.to_f64()?;
    let mut guess = Decimal::from_f64(x_f64.sqrt())?;

    // Newton-Raphson iterations
    for _ in 0..10 {
        let next = (guess + x / guess) / dec!(2);
        if (next - guess).abs() < dec!(0.000000001) {
            return Some(next);
        }
        guess = next;
    }

    Some(guess)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentile_basic() {
        let data: Vec<Decimal> = (1..=100).map(|x| Decimal::from(x)).collect();
        
        assert_eq!(calculate_percentile(&data, dec!(50)), Some(dec!(50.5)));
        assert_eq!(calculate_percentile(&data, dec!(0)), Some(dec!(1)));
        assert_eq!(calculate_percentile(&data, dec!(100)), Some(dec!(100)));
    }

    #[test]
    fn test_percentile_empty() {
        let data: Vec<Decimal> = vec![];
        assert_eq!(calculate_percentile(&data, dec!(50)), None);
    }

    #[test]
    fn test_mean() {
        let data = vec![dec!(1), dec!(2), dec!(3), dec!(4), dec!(5)];
        assert_eq!(calculate_mean(&data), Some(dec!(3)));
    }

    #[test]
    fn test_std() {
        let data = vec![dec!(2), dec!(4), dec!(4), dec!(4), dec!(5), dec!(5), dec!(7), dec!(9)];
        let std = calculate_std(&data).unwrap();
        // Expected std ≈ 2.138
        assert!(std > dec!(2.1) && std < dec!(2.2));
    }

    #[test]
    fn test_baseline() {
        let data: Vec<Decimal> = (1..=100).map(|x| Decimal::from(x)).collect();
        let baseline = calculate_baseline(&data, 60).unwrap();
        
        assert_eq!(baseline.n, 100);
        assert_eq!(baseline.min, dec!(1));
        assert_eq!(baseline.max, dec!(100));
        assert_eq!(baseline.window_days, 60);
    }

    #[test]
    fn test_hoeffding_bound() {
        let bound = hoeffding_bound(100, 0.05);
        // sqrt(ln(20) / 200) ≈ 0.122
        assert!(bound > 0.1 && bound < 0.15);
    }

    #[test]
    fn test_ks_same_distribution() {
        let sample_a: Vec<Decimal> = (1..=50).map(|x| Decimal::from(x)).collect();
        let sample_b: Vec<Decimal> = (1..=50).map(|x| Decimal::from(x)).collect();
        
        let result = ks_two_sample(&sample_a, &sample_b).unwrap();
        assert_eq!(result.statistic, Decimal::ZERO);
        // p-value should be high (close to 1)
    }

    #[test]
    fn test_ks_different_distributions() {
        let sample_a: Vec<Decimal> = (1..=50).map(|x| Decimal::from(x)).collect();
        let sample_b: Vec<Decimal> = (51..=100).map(|x| Decimal::from(x)).collect();
        
        let result = ks_two_sample(&sample_a, &sample_b).unwrap();
        assert!(result.statistic > dec!(0.9)); // Should be close to 1
    }

    #[test]
    fn test_jaccard_similarity() {
        let set_a: HashSet<i32> = [1, 2, 3, 4, 5].into_iter().collect();
        let set_b: HashSet<i32> = [3, 4, 5, 6, 7].into_iter().collect();
        
        // Intersection: {3,4,5} = 3 elements
        // Union: {1,2,3,4,5,6,7} = 7 elements
        // J = 3/7 ≈ 0.428
        let j = jaccard_similarity(&set_a, &set_b);
        assert!(j > dec!(0.42) && j < dec!(0.44));
    }

    #[test]
    fn test_jaccard_identical() {
        let set_a: HashSet<i32> = [1, 2, 3].into_iter().collect();
        let set_b: HashSet<i32> = [1, 2, 3].into_iter().collect();
        
        assert_eq!(jaccard_similarity(&set_a, &set_b), dec!(1));
    }

    #[test]
    fn test_jaccard_disjoint() {
        let set_a: HashSet<i32> = [1, 2, 3].into_iter().collect();
        let set_b: HashSet<i32> = [4, 5, 6].into_iter().collect();
        
        assert_eq!(jaccard_similarity(&set_a, &set_b), dec!(0));
    }

    #[test]
    fn test_sigma_deviation() {
        assert_eq!(sigma_deviation(dec!(10), dec!(5), dec!(2.5)), Some(dec!(2)));
        assert_eq!(sigma_deviation(dec!(0), dec!(5), dec!(2.5)), Some(dec!(-2)));
        assert_eq!(sigma_deviation(dec!(5), dec!(5), dec!(2.5)), Some(dec!(0)));
    }

    #[test]
    fn test_sigma_deviation_zero_std() {
        assert_eq!(sigma_deviation(dec!(5), dec!(5), dec!(0)), Some(dec!(0)));
        assert_eq!(sigma_deviation(dec!(6), dec!(5), dec!(0)), None);
    }

    #[test]
    fn test_decimal_sqrt() {
        let sqrt_4 = decimal_sqrt(dec!(4)).unwrap();
        assert!((sqrt_4 - dec!(2)).abs() < dec!(0.0001));

        let sqrt_2 = decimal_sqrt(dec!(2)).unwrap();
        assert!((sqrt_2 - dec!(1.414213)).abs() < dec!(0.001));
    }
}


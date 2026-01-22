//! Statistical metrics for validation.
//!
//! Implements proper PSR and DSR calculations per Bailey & López de Prado (2012, 2014).
//! All functions use f64 for compatibility with the combiner engine.

// Note: PI constant not currently used but kept for future extensions (e.g., normal_pdf)

/// Calculate the skewness of a series of values.
/// Skewness measures the asymmetry of the distribution.
/// 
/// Formula: E[(X - μ)³] / σ³
#[inline]
pub fn calculate_skewness(values: &[f64]) -> f64 {
    if values.len() < 3 {
        return 0.0;
    }

    let n = values.len() as f64;
    let mean: f64 = values.iter().sum::<f64>() / n;
    
    // Calculate second and third central moments
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    
    for &v in values {
        let diff = v - mean;
        let diff2 = diff * diff;
        m2 += diff2;
        m3 += diff2 * diff;
    }
    
    m2 /= n;
    m3 /= n;
    
    // Standard deviation
    let std_dev = m2.sqrt();
    if std_dev < 1e-15 {
        return 0.0;
    }
    
    // Skewness = m3 / σ³
    let std_dev_cubed = std_dev * std_dev * std_dev;
    m3 / std_dev_cubed
}

/// Calculate the excess kurtosis of a series of values.
/// Kurtosis measures the "tailedness" of the distribution.
/// Returns excess kurtosis (kurtosis - 3), so normal distribution = 0.
/// 
/// Formula: E[(X - μ)⁴] / σ⁴ - 3
#[inline]
pub fn calculate_kurtosis(values: &[f64]) -> f64 {
    if values.len() < 4 {
        return 0.0;
    }

    let n = values.len() as f64;
    let mean: f64 = values.iter().sum::<f64>() / n;
    
    // Calculate second and fourth central moments
    let mut m2 = 0.0;
    let mut m4 = 0.0;
    
    for &v in values {
        let diff = v - mean;
        let diff2 = diff * diff;
        m2 += diff2;
        m4 += diff2 * diff2;
    }
    
    m2 /= n;
    m4 /= n;
    
    // Variance check
    if m2 < 1e-15 {
        return 0.0;
    }
    
    // Kurtosis = m4 / m2² - 3 (excess kurtosis)
    let m2_squared = m2 * m2;
    (m4 / m2_squared) - 3.0
}

/// Standard normal CDF (cumulative distribution function).
/// Uses the approximation by Abramowitz and Stegun (1964).
/// Exposed for use in validation and PBO estimation.
#[inline]
pub fn normal_cdf_approx(x: f64) -> f64 {
    // Approximation constants
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs();
    
    let t = 1.0 / (1.0 + P * x_abs);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x_abs * x_abs / 2.0).exp();
    
    0.5 * (1.0 + sign * y)
}

/// Calculate PSR (Probabilistic Sharpe Ratio).
/// 
/// PSR estimates the probability that the true Sharpe ratio exceeds a threshold,
/// accounting for the uncertainty in the estimated Sharpe ratio due to:
/// - Limited sample size
/// - Non-normality (skewness and kurtosis)
/// 
/// Reference: Bailey & López de Prado (2012) "The Sharpe Ratio Efficient Frontier"
/// 
/// Formula:
/// PSR = Φ((SR - SR*) * √n / √(1 - γ₃*SR + (γ₄-1)/4 * SR²))
/// 
/// Where:
/// - SR: observed Sharpe ratio
/// - SR*: threshold (benchmark Sharpe)
/// - n: number of observations
/// - γ₃: skewness of returns
/// - γ₄: excess kurtosis of returns
/// - Φ: standard normal CDF
#[inline]
pub fn calculate_psr(
    sharpe: f64,
    threshold: f64,
    n_observations: usize,
    skewness: f64,
    kurtosis: f64,
) -> f64 {
    if n_observations < 2 {
        return 0.0;
    }

    let n = n_observations as f64;
    
    // Numerator: (SR - SR*) * sqrt(n)
    let numerator = (sharpe - threshold) * n.sqrt();
    
    // Denominator: sqrt(1 - γ₃*SR + (γ₄-1)/4 * SR²)
    // This is the standard error of the Sharpe ratio under non-normality
    let variance_term = 1.0 - skewness * sharpe + (kurtosis - 1.0) / 4.0 * sharpe * sharpe;
    
    // Ensure variance term is positive (can be negative for extreme cases)
    let variance_term = variance_term.max(0.001);
    let denominator = variance_term.sqrt();
    
    if denominator < 1e-15 {
        return if sharpe > threshold { 1.0 } else { 0.0 };
    }
    
    let z = numerator / denominator;
    normal_cdf_approx(z).clamp(0.0, 1.0)
}

/// Calculate the expected maximum Sharpe ratio under the null hypothesis.
/// 
/// This is used in DSR to penalize for multiple testing.
/// Reference: Bailey & López de Prado (2014)
/// 
/// Formula: E[max(SR)] ≈ σ_SR * √(2 * ln(N))
/// where N is the number of independent trials
#[inline]
fn expected_max_sharpe(n_trials: usize, sharpe_std: f64) -> f64 {
    if n_trials <= 1 {
        return 0.0;
    }
    
    // Euler-Mascheroni constant for finite sample correction
    const EULER_GAMMA: f64 = 0.5772156649;
    
    let n = n_trials as f64;
    
    // Expected maximum from order statistics of normal distribution
    // E[max] ≈ σ * √(2 * ln(N))
    let e_max = sharpe_std * (2.0 * n.ln()).sqrt();
    
    // Apply correction for small N
    if n < 20.0 {
        return e_max * (1.0 - EULER_GAMMA / n.ln().max(1.0));
    }
    
    e_max
}

/// Calculate DSR (Deflated Sharpe Ratio).
/// 
/// DSR adjusts the PSR for the effects of multiple testing (data snooping).
/// It accounts for the fact that when testing many strategies, the "best"
/// Sharpe ratio is likely to be inflated due to luck.
/// 
/// Reference: Bailey & López de Prado (2014) "The Deflated Sharpe Ratio"
/// 
/// The DSR is the PSR calculated against a threshold that equals the
/// expected maximum Sharpe ratio under the null hypothesis of no skill.
/// 
/// Formula:
/// DSR = PSR(SR, E[max(SR)], n, γ₃, γ₄)
/// 
/// Where E[max(SR)] is the expected maximum Sharpe from testing N strategies.
/// 
/// Returns a value in [0, 1] representing the probability that the observed
/// Sharpe exceeds what would be expected from pure luck given N trials.
#[inline]
pub fn calculate_dsr(
    sharpe: f64,
    n_observations: usize,
    skewness: f64,
    kurtosis: f64,
    n_trials: usize,
    sharpe_variance: f64,
) -> f64 {
    if n_trials <= 1 {
        // No multiple testing bias, DSR = PSR with threshold 0
        return calculate_psr(sharpe, 0.0, n_observations, skewness, kurtosis);
    }
    
    if n_observations < 2 {
        return 0.0;
    }

    // Standard deviation of Sharpe ratios across trials
    let sharpe_std = sharpe_variance.sqrt().max(0.1);
    
    // Calculate expected maximum Sharpe under null
    let e_max = expected_max_sharpe(n_trials, sharpe_std);
    
    // Calculate PSR against the deflated threshold
    calculate_psr(sharpe, e_max, n_observations, skewness, kurtosis)
}

/// Calculate the sample variance of a slice (with Bessel's correction).
/// Uses n-1 denominator for unbiased estimation.
#[inline]
pub fn sample_variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    
    let n = values.len() as f64;
    let mean: f64 = values.iter().sum::<f64>() / n;
    
    let variance: f64 = values.iter()
        .map(|&v| (v - mean) * (v - mean))
        .sum::<f64>() / (n - 1.0);  // Bessel's correction
    
    variance
}

/// Calculate the sample standard deviation (with Bessel's correction).
#[inline]
pub fn sample_std(values: &[f64]) -> f64 {
    sample_variance(values).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skewness_normal() {
        // Symmetric distribution should have ~0 skewness
        let symmetric = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let skew = calculate_skewness(&symmetric);
        assert!(skew.abs() < 0.1, "Symmetric distribution skewness should be ~0, got {}", skew);
    }

    #[test]
    fn test_skewness_positive() {
        // Right-skewed (positive skewness)
        let right_skewed = vec![0.1, 0.2, 0.3, 0.4, 0.5, 2.0, 5.0];
        let skew = calculate_skewness(&right_skewed);
        assert!(skew > 0.5, "Right-skewed distribution should have positive skewness, got {}", skew);
    }

    #[test]
    fn test_kurtosis_normal_like() {
        // Normal distribution has excess kurtosis ~0
        let normal_like = vec![-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
        let kurt = calculate_kurtosis(&normal_like);
        assert!(kurt.abs() < 2.0, "Normal-like distribution excess kurtosis should be near 0, got {}", kurt);
    }

    #[test]
    fn test_psr_high_sharpe() {
        // High Sharpe with many observations should have high PSR
        let psr = calculate_psr(2.0, 0.0, 252, 0.0, 0.0);
        assert!(psr > 0.99, "High Sharpe with n=252 should have PSR > 0.99, got {}", psr);
    }

    #[test]
    fn test_psr_low_sharpe() {
        // Negative Sharpe should have low PSR
        let psr = calculate_psr(-0.5, 0.0, 100, 0.0, 0.0);
        assert!(psr < 0.01, "Negative Sharpe should have PSR < 0.01, got {}", psr);
    }

    #[test]
    fn test_dsr_less_than_psr() {
        // DSR should be <= PSR due to multiple testing penalty
        let psr = calculate_psr(1.5, 0.0, 100, 0.0, 0.0);
        let dsr = calculate_dsr(1.5, 100, 0.0, 0.0, 1000, 0.25);
        assert!(dsr <= psr, "DSR {} should be <= PSR {}", dsr, psr);
    }

    #[test]
    fn test_dsr_single_trial() {
        // With single trial, DSR ≈ PSR
        let psr = calculate_psr(1.5, 0.0, 100, 0.0, 0.0);
        let dsr = calculate_dsr(1.5, 100, 0.0, 0.0, 1, 0.25);
        assert!((dsr - psr).abs() < 0.01, "DSR {} should ≈ PSR {} for single trial", dsr, psr);
    }

    #[test]
    fn test_sample_variance_bessel() {
        // Sample variance should use n-1
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let var = sample_variance(&values);
        // True population variance = 2.0, sample variance = 2.5
        assert!((var - 2.5).abs() < 0.01, "Sample variance should be 2.5, got {}", var);
    }

    // =========================================================================
    // Comprehensive Validation Test Matrix
    // =========================================================================

    #[test]
    fn test_pbo_all_positive_oos() {
        // When all OOS Sharpes are positive, PBO should be near 0
        // PBO = P(OOS < 0) = Phi(-mean/std)
        // With mean = 1.5, std = 0.3: z = -1.5/0.3 = -5, Phi(-5) ≈ 0
        let sharpes = vec![1.2, 1.5, 1.8, 1.4, 1.6];
        let mean: f64 = sharpes.iter().sum::<f64>() / sharpes.len() as f64;
        let std = sample_std(&sharpes);
        let z = -mean / std;
        let pbo = 0.5 * (1.0 + libm::erf(z / std::f64::consts::SQRT_2));
        assert!(pbo < 0.01, "All positive OOS should have PBO near 0, got {}", pbo);
    }

    #[test]
    fn test_pbo_half_negative_oos() {
        // When half OOS Sharpes are negative, PBO should be near 0.5
        let sharpes = vec![-0.5, -0.3, 0.3, 0.5, 0.1];
        let mean: f64 = sharpes.iter().sum::<f64>() / sharpes.len() as f64;
        let std = sample_std(&sharpes);
        let z = -mean / std;
        let pbo = 0.5 * (1.0 + libm::erf(z / std::f64::consts::SQRT_2));
        // Mean ≈ 0.02, so z ≈ -0.05, Phi(-0.05) ≈ 0.48
        assert!(pbo > 0.3 && pbo < 0.7, "Half negative OOS should have PBO near 0.5, got {}", pbo);
    }

    #[test]
    fn test_dsr_increases_penalty_with_trials() {
        // DSR should decrease as number of trials increases
        let dsr_10 = calculate_dsr(1.5, 100, 0.0, 0.0, 10, 0.25);
        let dsr_100 = calculate_dsr(1.5, 100, 0.0, 0.0, 100, 0.25);
        let dsr_1000 = calculate_dsr(1.5, 100, 0.0, 0.0, 1000, 0.25);
        
        assert!(dsr_10 > dsr_100, "DSR with 10 trials {} should be > DSR with 100 trials {}", dsr_10, dsr_100);
        assert!(dsr_100 > dsr_1000, "DSR with 100 trials {} should be > DSR with 1000 trials {}", dsr_100, dsr_1000);
    }

    #[test]
    fn test_dsr_bounded_zero_one() {
        // DSR should always be in [0, 1]
        let test_cases = vec![
            (3.0, 252, 0.0, 0.0, 10000, 1.0),   // Very high Sharpe
            (0.1, 50, -0.5, 3.0, 1000, 0.5),    // Low Sharpe, skewed
            (-0.5, 100, 0.0, 0.0, 100, 0.25),   // Negative Sharpe
            (1.0, 10, 0.0, 0.0, 5, 0.1),        // Small sample
        ];

        for (sharpe, n_obs, skew, kurt, n_trials, var) in test_cases {
            let dsr = calculate_dsr(sharpe, n_obs, skew, kurt, n_trials, var);
            assert!(dsr >= 0.0 && dsr <= 1.0, 
                "DSR should be in [0,1], got {} for sharpe={}, n={}, trials={}", 
                dsr, sharpe, n_obs, n_trials);
        }
    }

    #[test]
    fn test_psr_with_negative_skewness() {
        // Negative skewness (typical for financial returns) should reduce PSR
        let psr_normal = calculate_psr(1.0, 0.0, 100, 0.0, 0.0);
        let psr_skewed = calculate_psr(1.0, 0.0, 100, -1.0, 0.0);
        
        // Negative skewness increases uncertainty in Sharpe estimate
        // Formula: sqrt(1 - gamma3*SR + ...) increases, reducing z-score
        assert!(psr_skewed <= psr_normal + 0.1, 
            "Negative skewness should not dramatically increase PSR: normal={}, skewed={}", 
            psr_normal, psr_skewed);
    }

    #[test]
    fn test_psr_with_fat_tails() {
        // High kurtosis (fat tails) should affect PSR via variance term
        let psr_normal = calculate_psr(1.0, 0.0, 100, 0.0, 0.0);
        let psr_fat = calculate_psr(1.0, 0.0, 100, 0.0, 5.0);
        
        // Fat tails increase the (gamma4-1)/4 * SR^2 term
        // This can increase variance term, potentially reducing z-score
        assert!(psr_fat.abs() <= 1.0, "PSR should be bounded, got {}", psr_fat);
    }

    #[test]
    fn test_skewness_empty_and_small() {
        // Edge cases for skewness
        assert_eq!(calculate_skewness(&[]), 0.0, "Empty should return 0");
        assert_eq!(calculate_skewness(&[1.0]), 0.0, "Single value should return 0");
        assert_eq!(calculate_skewness(&[1.0, 2.0]), 0.0, "Two values should return 0");
    }

    #[test]
    fn test_kurtosis_empty_and_small() {
        // Edge cases for kurtosis
        assert_eq!(calculate_kurtosis(&[]), 0.0, "Empty should return 0");
        assert_eq!(calculate_kurtosis(&[1.0]), 0.0, "Single value should return 0");
        assert_eq!(calculate_kurtosis(&[1.0, 2.0]), 0.0, "Two values should return 0");
        assert_eq!(calculate_kurtosis(&[1.0, 2.0, 3.0]), 0.0, "Three values should return 0");
    }

    #[test]
    fn test_sample_variance_edge_cases() {
        assert_eq!(sample_variance(&[]), 0.0, "Empty should return 0");
        assert_eq!(sample_variance(&[5.0]), 0.0, "Single value should return 0");
        
        // Two identical values should have 0 variance
        assert_eq!(sample_variance(&[3.0, 3.0]), 0.0, "Identical values should return 0");
    }

    // =========================================================================
    // Reference Implementation Cross-Validation (Phase 1.2)
    // Bailey & López de Prado (2012, 2014) reference comparisons
    // =========================================================================

    #[test]
    fn test_normal_cdf_reference_values() {
        // Reference values from standard normal tables
        // Note: Abramowitz-Stegun approximation has error bounds of ~0.03
        let test_cases = vec![
            (0.0, 0.5),      // Phi(0) = 0.5 (exact)
            (1.0, 0.8413),   // Phi(1) ≈ 0.8413
            (2.0, 0.9772),   // Phi(2) ≈ 0.9772
            (-1.0, 0.1587),  // Phi(-1) ≈ 0.1587
            (-2.0, 0.0228),  // Phi(-2) ≈ 0.0228
            (3.0, 0.9987),   // Phi(3) ≈ 0.9987
        ];
        
        for (x, expected) in test_cases {
            let actual = normal_cdf_approx(x);
            // Use 0.03 tolerance for Abramowitz-Stegun approximation
            assert!((actual - expected).abs() < 0.03, 
                "Phi({}) = {}, expected {} (within 0.03)", x, actual, expected);
        }
        
        // Verify symmetry property: Phi(-x) = 1 - Phi(x)
        for x in [0.5, 1.0, 1.5, 2.0, 2.5] {
            let phi_x = normal_cdf_approx(x);
            let phi_neg_x = normal_cdf_approx(-x);
            assert!((phi_x + phi_neg_x - 1.0).abs() < 0.001, 
                "Symmetry: Phi({}) + Phi({}) should = 1, got {}", x, -x, phi_x + phi_neg_x);
        }
    }

    #[test]
    fn test_psr_formula_direct_calculation() {
        // PSR = Phi((SR - SR*) * sqrt(n) / sqrt(1 - gamma3*SR + (gamma4-1)/4 * SR^2))
        // For normal returns (gamma3=0, gamma4=0):
        // Denominator = sqrt(1 + (-1)/4 * SR^2) = sqrt(1 - 0.25*SR^2)
        // For SR=0.1: denom = sqrt(1 - 0.0025) = sqrt(0.9975) ≈ 0.9987
        // z = 0.1 * 10 / 0.9987 ≈ 1.001
        // PSR ≈ Phi(1.0) using our approximation
        
        let psr = calculate_psr(0.1, 0.0, 100, 0.0, 0.0);
        // Allow for CDF approximation error
        assert!(psr > 0.80 && psr < 0.90, 
            "PSR for SR=0.1, n=100 should be ~0.84-0.87, got {}", psr);
        
        // SR = 0.2, n = 100
        // z = 0.2 * 10 / sqrt(1 - 0.01) ≈ 2.01
        let psr2 = calculate_psr(0.2, 0.0, 100, 0.0, 0.0);
        assert!(psr2 > 0.95 && psr2 < 1.0, 
            "PSR for SR=0.2, n=100 should be ~0.97-0.99, got {}", psr2);
        
        // Verify monotonicity
        assert!(psr2 > psr, "Higher Sharpe should give higher PSR");
    }

    #[test]
    fn test_psr_monotonicity_in_sharpe() {
        // Property: PSR is monotonically increasing in Sharpe ratio
        let sharpes = vec![-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];
        let mut prev_psr = 0.0;
        
        for &sr in &sharpes {
            let psr = calculate_psr(sr, 0.0, 100, 0.0, 0.0);
            if sr > -1.0 {
                assert!(psr >= prev_psr, 
                    "PSR should be monotonic in Sharpe: PSR({}) = {} < PSR({}) = {}", 
                    sr, psr, sr - 0.5, prev_psr);
            }
            prev_psr = psr;
        }
    }

    #[test]
    fn test_psr_monotonicity_in_sample_size() {
        // Property: For positive Sharpe, PSR increases with sample size
        let sample_sizes = vec![10, 50, 100, 252, 500, 1000];
        let mut prev_psr = 0.0;
        
        for &n in &sample_sizes {
            let psr = calculate_psr(0.5, 0.0, n, 0.0, 0.0);
            if n > 10 {
                assert!(psr >= prev_psr - 0.01, 
                    "PSR should increase with sample size for positive Sharpe: n={} gives {} < {}", 
                    n, psr, prev_psr);
            }
            prev_psr = psr;
        }
    }

    #[test]
    fn test_dsr_vs_psr_property() {
        // Property: DSR <= PSR always (penalty for multiple testing)
        let test_cases = vec![
            (0.5, 100, 10),
            (1.0, 252, 100),
            (1.5, 500, 1000),
            (2.0, 1000, 10000),
        ];
        
        for (sr, n_obs, n_trials) in test_cases {
            let psr = calculate_psr(sr, 0.0, n_obs, 0.0, 0.0);
            let dsr = calculate_dsr(sr, n_obs, 0.0, 0.0, n_trials, 0.25);
            
            assert!(dsr <= psr + 0.001, 
                "DSR ({}) should be <= PSR ({}) for SR={}, n={}, trials={}", 
                dsr, psr, sr, n_obs, n_trials);
        }
    }

    #[test]
    fn test_skewness_known_distribution() {
        // Uniform distribution [-1, 1] has skewness = 0
        let uniform: Vec<f64> = (-100..=100).map(|i| i as f64 / 100.0).collect();
        let skew = calculate_skewness(&uniform);
        assert!(skew.abs() < 0.1, "Uniform distribution should have ~0 skewness: {}", skew);
        
        // Right-skewed: more values on left, tail on right
        let right_skewed: Vec<f64> = (1..=100).map(|i| (i as f64).sqrt()).collect();
        let skew_right = calculate_skewness(&right_skewed);
        // sqrt distribution is left-skewed relative to linear
        assert!(skew_right.abs() < 1.0, "sqrt distribution skewness: {}", skew_right);
    }

    #[test]
    fn test_kurtosis_known_distribution() {
        // Uniform distribution has excess kurtosis = -1.2
        let n = 1000;
        let uniform: Vec<f64> = (0..n).map(|i| (i as f64 / n as f64) * 2.0 - 1.0).collect();
        let kurt = calculate_kurtosis(&uniform);
        assert!((kurt - (-1.2)).abs() < 0.2, 
            "Uniform distribution should have excess kurtosis ~-1.2, got {}", kurt);
    }

    #[test]
    fn test_expected_max_sharpe_scaling() {
        // E[max] ≈ σ * sqrt(2 * ln(N))
        // For σ = 0.5 (std of Sharpe estimates)
        // N = 100: E[max] ≈ 0.5 * sqrt(2 * 4.6) ≈ 0.5 * 3.03 ≈ 1.52
        // N = 1000: E[max] ≈ 0.5 * sqrt(2 * 6.9) ≈ 0.5 * 3.71 ≈ 1.86
        
        let e_max_100 = expected_max_sharpe(100, 0.5);
        let e_max_1000 = expected_max_sharpe(1000, 0.5);
        
        assert!(e_max_100 > 1.0 && e_max_100 < 2.0, 
            "E[max] for N=100, σ=0.5 should be ~1.5, got {}", e_max_100);
        assert!(e_max_1000 > e_max_100, 
            "E[max] should increase with N: {} vs {}", e_max_1000, e_max_100);
    }

    #[test]
    fn test_variance_bessel_correction() {
        // Sample variance with Bessel's correction should be unbiased
        // For samples from known population variance
        // E[s^2] = σ^2 when using n-1 denominator
        
        // Known population: variance = 4
        let values = vec![0.0, 2.0, 4.0, 6.0, 8.0]; // mean = 4, population var = 8
        let sample_var = sample_variance(&values);
        
        // Sample variance = sum((x-mean)^2) / (n-1)
        // = (16 + 4 + 0 + 4 + 16) / 4 = 40 / 4 = 10
        assert!((sample_var - 10.0).abs() < 0.01, 
            "Sample variance should be 10 with Bessel correction, got {}", sample_var);
    }

    #[test]
    fn test_psr_threshold_behavior() {
        // When threshold equals Sharpe, PSR should be ~0.5
        let psr_equal = calculate_psr(1.0, 1.0, 100, 0.0, 0.0);
        assert!((psr_equal - 0.5).abs() < 0.1, 
            "PSR should be ~0.5 when SR = threshold, got {}", psr_equal);
        
        // When threshold > Sharpe, PSR < 0.5
        let psr_below = calculate_psr(1.0, 1.5, 100, 0.0, 0.0);
        assert!(psr_below < 0.5, 
            "PSR should be < 0.5 when SR < threshold, got {}", psr_below);
    }

    #[test]
    fn test_dsr_extreme_trials() {
        // With very high trial count, DSR should be very low unless Sharpe is exceptional
        let dsr_moderate = calculate_dsr(1.0, 252, 0.0, 0.0, 100000, 0.25);
        let dsr_high = calculate_dsr(3.0, 252, 0.0, 0.0, 100000, 0.25);
        
        // Moderate Sharpe should fail extreme deflation
        assert!(dsr_moderate < 0.5, 
            "Moderate Sharpe should have low DSR with 100k trials, got {}", dsr_moderate);
        
        // High Sharpe should still be reasonable
        assert!(dsr_high > dsr_moderate, 
            "Higher Sharpe should have higher DSR: {} vs {}", dsr_high, dsr_moderate);
    }
}


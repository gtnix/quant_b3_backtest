//! Statistical metrics for walk-forward validation.
//!
//! Implements:
//! - PSR (Probabilistic Sharpe Ratio) - Bailey & López de Prado (2012)
//! - DSR (Deflated Sharpe Ratio) - Bailey & López de Prado (2014)
//! - Skewness and Kurtosis calculations

use rust_decimal::Decimal;
use rust_decimal::MathematicalOps;
use rust_decimal_macros::dec;
use std::f64::consts::PI;

/// Calculate the skewness of a series of returns.
/// Skewness measures the asymmetry of the distribution.
/// 
/// Formula: E[(X - μ)³] / σ³
pub fn calculate_skewness(returns: &[Decimal]) -> Decimal {
    if returns.len() < 3 {
        return Decimal::ZERO;
    }

    let n = Decimal::from(returns.len());
    let mean = returns.iter().sum::<Decimal>() / n;
    
    // Calculate second and third central moments
    let mut m2 = Decimal::ZERO;
    let mut m3 = Decimal::ZERO;
    
    for &r in returns {
        let diff = r - mean;
        let diff2 = diff * diff;
        m2 += diff2;
        m3 += diff2 * diff;
    }
    
    m2 /= n;
    m3 /= n;
    
    // Standard deviation
    let std_dev = m2.sqrt().unwrap_or(Decimal::ONE);
    if std_dev == Decimal::ZERO {
        return Decimal::ZERO;
    }
    
    // Skewness = m3 / σ³
    let std_dev_cubed = std_dev * std_dev * std_dev;
    if std_dev_cubed == Decimal::ZERO {
        return Decimal::ZERO;
    }
    
    m3 / std_dev_cubed
}

/// Calculate the excess kurtosis of a series of returns.
/// Kurtosis measures the "tailedness" of the distribution.
/// Returns excess kurtosis (kurtosis - 3), so normal distribution = 0.
/// 
/// Formula: E[(X - μ)⁴] / σ⁴ - 3
pub fn calculate_kurtosis(returns: &[Decimal]) -> Decimal {
    if returns.len() < 4 {
        return Decimal::ZERO;
    }

    let n = Decimal::from(returns.len());
    let mean = returns.iter().sum::<Decimal>() / n;
    
    // Calculate second and fourth central moments
    let mut m2 = Decimal::ZERO;
    let mut m4 = Decimal::ZERO;
    
    for &r in returns {
        let diff = r - mean;
        let diff2 = diff * diff;
        m2 += diff2;
        m4 += diff2 * diff2;
    }
    
    m2 /= n;
    m4 /= n;
    
    // Variance
    if m2 == Decimal::ZERO {
        return Decimal::ZERO;
    }
    
    // Kurtosis = m4 / m2² - 3 (excess kurtosis)
    let m2_squared = m2 * m2;
    if m2_squared == Decimal::ZERO {
        return Decimal::ZERO;
    }
    
    (m4 / m2_squared) - dec!(3)
}

/// Standard normal CDF (cumulative distribution function).
/// Uses the approximation by Abramowitz and Stegun (1964).
fn normal_cdf(x: f64) -> f64 {
    // Approximation constants
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x / 2.0).exp();
    
    0.5 * (1.0 + sign * y)
}

/// Standard normal PDF (probability density function).
fn normal_pdf(x: f64) -> f64 {
    (1.0 / (2.0 * PI).sqrt()) * (-x * x / 2.0).exp()
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
pub fn calculate_psr(
    sharpe: Decimal,
    threshold: Decimal,
    n_observations: usize,
    skewness: Decimal,
    kurtosis: Decimal,
) -> Decimal {
    if n_observations < 2 {
        return Decimal::ZERO;
    }

    // Convert to f64 for numerical stability
    let sr = sharpe.to_string().parse::<f64>().unwrap_or(0.0);
    let sr_star = threshold.to_string().parse::<f64>().unwrap_or(0.0);
    let n = n_observations as f64;
    let gamma3 = skewness.to_string().parse::<f64>().unwrap_or(0.0);
    let gamma4 = kurtosis.to_string().parse::<f64>().unwrap_or(0.0);
    
    // Numerator: (SR - SR*) * sqrt(n)
    let numerator = (sr - sr_star) * n.sqrt();
    
    // Denominator: sqrt(1 - γ₃*SR + (γ₄-1)/4 * SR²)
    // This is the standard error of the Sharpe ratio under non-normality
    let variance_term = 1.0 - gamma3 * sr + (gamma4 - 1.0) / 4.0 * sr * sr;
    
    // Ensure variance term is positive
    let variance_term = variance_term.max(0.001);
    let denominator = variance_term.sqrt();
    
    if denominator == 0.0 {
        return if sr > sr_star { Decimal::ONE } else { Decimal::ZERO };
    }
    
    let z = numerator / denominator;
    let psr = normal_cdf(z);
    
    // Clamp to [0, 1] and convert back to Decimal
    let psr = psr.clamp(0.0, 1.0);
    Decimal::from_f64_retain(psr).unwrap_or(Decimal::ZERO)
}

/// Calculate the expected maximum Sharpe ratio under the null hypothesis.
/// 
/// This is used in DSR to penalize for multiple testing.
/// Reference: Bailey & López de Prado (2014)
/// 
/// Formula: E[max(SR)] ≈ (1 - γ) * Φ⁻¹(1 - 1/N) + γ * Φ⁻¹(1 - 1/(N*e))
/// where γ ≈ 0.5772 (Euler-Mascheroni constant)
/// 
/// Simplified approximation: E[max] ≈ σ_SR * √(2 * ln(N))
fn expected_max_sharpe(n_trials: usize, sharpe_std: f64) -> f64 {
    if n_trials <= 1 {
        return 0.0;
    }
    
    // Euler-Mascheroni constant
    const EULER_GAMMA: f64 = 0.5772156649;
    
    let n = n_trials as f64;
    
    // Expected maximum from order statistics of normal distribution
    // E[max] ≈ √(2 * ln(N)) - (ln(ln(N)) + ln(4π))/(2 * √(2 * ln(N)))
    // Simplified: E[max] ≈ σ * √(2 * ln(N))
    let e_max = sharpe_std * (2.0 * n.ln()).sqrt();
    
    // Apply correction for small N
    if n < 20.0 {
        return e_max * (1.0 - EULER_GAMMA / n.ln());
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
pub fn calculate_dsr(
    sharpe: Decimal,
    _threshold: Decimal,  // ignored for DSR, we use expected max instead
    n_observations: usize,
    skewness: Decimal,
    kurtosis: Decimal,
    n_trials: usize,
    sharpe_variance: Decimal,
) -> Decimal {
    if n_trials <= 1 {
        // No multiple testing bias, DSR = PSR
        return calculate_psr(sharpe, Decimal::ZERO, n_observations, skewness, kurtosis);
    }
    
    if n_observations < 2 {
        return Decimal::ZERO;
    }

    // Standard deviation of Sharpe ratios across trials
    let sharpe_std = sharpe_variance.sqrt()
        .unwrap_or(dec!(0.5))
        .to_string()
        .parse::<f64>()
        .unwrap_or(0.5);
    
    // Calculate expected maximum Sharpe under null
    let e_max = expected_max_sharpe(n_trials, sharpe_std);
    
    // Convert to Decimal for threshold
    let deflated_threshold = Decimal::from_f64_retain(e_max)
        .unwrap_or(dec!(0.5));
    
    // Calculate PSR against the deflated threshold
    calculate_psr(sharpe, deflated_threshold, n_observations, skewness, kurtosis)
}

/// Calculate the variance of Sharpe ratios from a slice.
pub fn sharpe_variance(sharpes: &[Decimal]) -> Decimal {
    if sharpes.len() < 2 {
        return Decimal::ZERO;
    }
    
    let n = Decimal::from(sharpes.len());
    let mean = sharpes.iter().sum::<Decimal>() / n;
    
    let variance: Decimal = sharpes.iter()
        .map(|&s| (s - mean) * (s - mean))
        .sum::<Decimal>() / (n - Decimal::ONE);
    
    variance
}

#[cfg(test)]
mod tests {
    use super::*;

    fn returns_normal() -> Vec<Decimal> {
        // Simulated normal returns (slightly positive)
        vec![
            dec!(0.01), dec!(-0.02), dec!(0.015), dec!(-0.005), dec!(0.02),
            dec!(0.008), dec!(-0.01), dec!(0.012), dec!(-0.003), dec!(0.018),
            dec!(0.01), dec!(-0.015), dec!(0.025), dec!(-0.008), dec!(0.012),
            dec!(0.005), dec!(-0.012), dec!(0.008), dec!(-0.004), dec!(0.015),
        ]
    }

    fn returns_skewed_positive() -> Vec<Decimal> {
        // Returns with positive skewness (more large gains)
        vec![
            dec!(0.01), dec!(0.02), dec!(0.015), dec!(0.05), dec!(0.02),
            dec!(0.008), dec!(0.01), dec!(0.012), dec!(0.08), dec!(0.018),
            dec!(0.01), dec!(0.015), dec!(0.025), dec!(0.1), dec!(0.012),
        ]
    }

    fn returns_fat_tailed() -> Vec<Decimal> {
        // Returns with excess kurtosis (fat tails)
        vec![
            dec!(0.001), dec!(-0.001), dec!(0.002), dec!(-0.001), dec!(0.15),
            dec!(0.001), dec!(-0.002), dec!(0.001), dec!(-0.12), dec!(0.001),
            dec!(0.002), dec!(-0.001), dec!(0.001), dec!(-0.001), dec!(0.001),
        ]
    }

    #[test]
    fn test_skewness_normal_approx_zero() {
        let returns = returns_normal();
        let skew = calculate_skewness(&returns);
        // Normal distribution should have skewness near 0
        assert!(skew.abs() < dec!(1.0), "Skewness {} too large for normal", skew);
    }

    #[test]
    fn test_skewness_positive() {
        let returns = returns_skewed_positive();
        let skew = calculate_skewness(&returns);
        // Should have positive skewness
        assert!(skew > Decimal::ZERO, "Expected positive skewness, got {}", skew);
    }

    #[test]
    fn test_skewness_empty() {
        let skew = calculate_skewness(&[]);
        assert_eq!(skew, Decimal::ZERO);
    }

    #[test]
    fn test_kurtosis_normal_approx_zero() {
        let returns = returns_normal();
        let kurt = calculate_kurtosis(&returns);
        // Normal distribution has excess kurtosis near 0
        assert!(kurt.abs() < dec!(3.0), "Kurtosis {} too large for normal", kurt);
    }

    #[test]
    fn test_kurtosis_fat_tailed() {
        let returns = returns_fat_tailed();
        let kurt = calculate_kurtosis(&returns);
        // Fat tailed should have positive excess kurtosis
        assert!(kurt > Decimal::ZERO, "Expected positive kurtosis, got {}", kurt);
    }

    #[test]
    fn test_kurtosis_empty() {
        let kurt = calculate_kurtosis(&[]);
        assert_eq!(kurt, Decimal::ZERO);
    }

    #[test]
    fn test_psr_at_threshold() {
        // When Sharpe equals threshold, PSR should be around 0.5
        let psr = calculate_psr(dec!(0.5), dec!(0.5), 100, Decimal::ZERO, Decimal::ZERO);
        assert!((psr - dec!(0.5)).abs() < dec!(0.1), "PSR at threshold should be ~0.5, got {}", psr);
    }

    #[test]
    fn test_psr_above_threshold() {
        // When Sharpe is well above threshold, PSR should be high
        let psr = calculate_psr(dec!(1.5), dec!(0.5), 100, Decimal::ZERO, Decimal::ZERO);
        assert!(psr > dec!(0.8), "PSR should be high for Sharpe >> threshold, got {}", psr);
    }

    #[test]
    fn test_psr_below_threshold() {
        // When Sharpe is below threshold, PSR should be low
        let psr = calculate_psr(dec!(0.2), dec!(0.5), 100, Decimal::ZERO, Decimal::ZERO);
        assert!(psr < dec!(0.5), "PSR should be low for Sharpe < threshold, got {}", psr);
    }

    #[test]
    fn test_psr_monotonic_in_sharpe() {
        let psr1 = calculate_psr(dec!(0.5), dec!(0.5), 100, Decimal::ZERO, Decimal::ZERO);
        let psr2 = calculate_psr(dec!(1.0), dec!(0.5), 100, Decimal::ZERO, Decimal::ZERO);
        let psr3 = calculate_psr(dec!(1.5), dec!(0.5), 100, Decimal::ZERO, Decimal::ZERO);
        
        assert!(psr1 < psr2, "PSR should increase with Sharpe");
        assert!(psr2 < psr3, "PSR should increase with Sharpe");
    }

    #[test]
    fn test_psr_increases_with_n() {
        // More observations should increase confidence (higher PSR for same Sharpe > threshold)
        let psr_n50 = calculate_psr(dec!(1.0), dec!(0.5), 50, Decimal::ZERO, Decimal::ZERO);
        let psr_n200 = calculate_psr(dec!(1.0), dec!(0.5), 200, Decimal::ZERO, Decimal::ZERO);
        
        assert!(psr_n200 > psr_n50, "PSR should increase with N for SR > threshold");
    }

    #[test]
    fn test_psr_bounded() {
        let psr = calculate_psr(dec!(5.0), dec!(0.5), 100, Decimal::ZERO, Decimal::ZERO);
        assert!(psr <= Decimal::ONE, "PSR should be <= 1");
        assert!(psr >= Decimal::ZERO, "PSR should be >= 0");
    }

    #[test]
    fn test_dsr_less_than_psr() {
        // DSR should always be <= PSR due to multiple testing penalty
        let psr = calculate_psr(dec!(1.0), dec!(0.5), 100, Decimal::ZERO, Decimal::ZERO);
        let dsr = calculate_dsr(
            dec!(1.0), dec!(0.5), 100, 
            Decimal::ZERO, Decimal::ZERO,
            50,  // 50 trials
            dec!(0.25),  // variance
        );
        
        assert!(dsr <= psr, "DSR {} should be <= PSR {}", dsr, psr);
    }

    #[test]
    fn test_dsr_decreases_with_trials() {
        let dsr_10 = calculate_dsr(
            dec!(1.0), dec!(0.5), 100,
            Decimal::ZERO, Decimal::ZERO,
            10, dec!(0.25),
        );
        let dsr_100 = calculate_dsr(
            dec!(1.0), dec!(0.5), 100,
            Decimal::ZERO, Decimal::ZERO,
            100, dec!(0.25),
        );
        
        assert!(dsr_100 < dsr_10, "DSR should decrease with more trials");
    }

    #[test]
    fn test_dsr_equals_psr_for_single_trial() {
        let psr = calculate_psr(dec!(1.0), Decimal::ZERO, 100, Decimal::ZERO, Decimal::ZERO);
        let dsr = calculate_dsr(
            dec!(1.0), dec!(0.5), 100,
            Decimal::ZERO, Decimal::ZERO,
            1,  // single trial
            dec!(0.25),
        );
        
        // With single trial, DSR should equal PSR (against zero threshold)
        assert!((dsr - psr).abs() < dec!(0.1), "DSR {} should ~= PSR {} for single trial", dsr, psr);
    }

    #[test]
    fn test_sharpe_variance() {
        let sharpes = vec![dec!(1.0), dec!(1.2), dec!(0.8), dec!(1.1), dec!(0.9)];
        let var = sharpe_variance(&sharpes);
        
        // Variance should be positive for non-constant values
        assert!(var > Decimal::ZERO, "Variance should be positive");
        assert!(var < dec!(1.0), "Variance should be reasonable");
    }

    #[test]
    fn test_sharpe_variance_constant() {
        let sharpes = vec![dec!(1.0), dec!(1.0), dec!(1.0)];
        let var = sharpe_variance(&sharpes);
        
        assert_eq!(var, Decimal::ZERO, "Variance of constant should be 0");
    }

    #[test]
    fn test_normal_cdf_symmetry() {
        // CDF(0) should be 0.5
        let cdf_0 = normal_cdf(0.0);
        assert!((cdf_0 - 0.5).abs() < 0.001, "CDF(0) should be 0.5");
        
        // CDF(-x) + CDF(x) should equal 1
        let cdf_neg = normal_cdf(-1.0);
        let cdf_pos = normal_cdf(1.0);
        assert!((cdf_neg + cdf_pos - 1.0).abs() < 0.001, "CDF symmetry check failed");
    }

    #[test]
    fn test_normal_cdf_bounds() {
        // CDF should be between 0 and 1
        let cdf_far_left = normal_cdf(-5.0);
        let cdf_far_right = normal_cdf(5.0);
        
        assert!(cdf_far_left > 0.0 && cdf_far_left < 0.001);
        assert!(cdf_far_right > 0.999 && cdf_far_right < 1.0);
    }
}









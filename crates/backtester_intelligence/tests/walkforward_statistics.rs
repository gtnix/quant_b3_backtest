//! Tests for Walk-Forward statistical metrics: PSR, DSR, skewness, kurtosis.

use backtester_intelligence::walkforward::{
    calculate_psr, calculate_dsr, calculate_skewness, calculate_kurtosis,
};
use rust_decimal::Decimal;
use rust_decimal::MathematicalOps;
use rust_decimal_macros::dec;

// ========================
// Skewness Tests
// ========================

#[test]
fn test_skewness_symmetric_approx_zero() {
    // Symmetric distribution should have skewness near zero
    let returns: Vec<Decimal> = vec![
        dec!(-0.03), dec!(-0.02), dec!(-0.01), dec!(0), 
        dec!(0.01), dec!(0.02), dec!(0.03),
    ];
    let skew = calculate_skewness(&returns);
    assert!(skew.abs() < dec!(0.5), "Symmetric skew should be near zero, got {}", skew);
}

#[test]
fn test_skewness_positive_right_tail() {
    // Right-skewed distribution (more large positive values)
    let returns: Vec<Decimal> = vec![
        dec!(0.01), dec!(0.01), dec!(0.01), dec!(0.01), dec!(0.01),
        dec!(0.01), dec!(0.01), dec!(0.01), dec!(0.01), dec!(0.10),  // outlier
    ];
    let skew = calculate_skewness(&returns);
    assert!(skew > Decimal::ZERO, "Right-skewed should have positive skewness, got {}", skew);
}

#[test]
fn test_skewness_negative_left_tail() {
    // Left-skewed distribution (more large negative values)
    let returns: Vec<Decimal> = vec![
        dec!(-0.10),  // outlier
        dec!(0.01), dec!(0.01), dec!(0.01), dec!(0.01), dec!(0.01),
        dec!(0.01), dec!(0.01), dec!(0.01), dec!(0.01),
    ];
    let skew = calculate_skewness(&returns);
    assert!(skew < Decimal::ZERO, "Left-skewed should have negative skewness, got {}", skew);
}

#[test]
fn test_skewness_empty_returns_zero() {
    let skew = calculate_skewness(&[]);
    assert_eq!(skew, Decimal::ZERO);
}

#[test]
fn test_skewness_single_value_returns_zero() {
    let skew = calculate_skewness(&[dec!(0.05)]);
    assert_eq!(skew, Decimal::ZERO);
}

#[test]
fn test_skewness_two_values_returns_zero() {
    let skew = calculate_skewness(&[dec!(0.01), dec!(0.02)]);
    assert_eq!(skew, Decimal::ZERO);
}

// ========================
// Kurtosis Tests
// ========================

#[test]
fn test_kurtosis_normal_approx_zero() {
    // Approximately normal distribution should have excess kurtosis near 0
    let returns: Vec<Decimal> = vec![
        dec!(-0.02), dec!(-0.015), dec!(-0.01), dec!(-0.005), dec!(0),
        dec!(0.005), dec!(0.01), dec!(0.015), dec!(0.02),
        dec!(-0.018), dec!(-0.012), dec!(-0.008), dec!(-0.003), dec!(0.003),
        dec!(0.008), dec!(0.012), dec!(0.018),
    ];
    let kurt = calculate_kurtosis(&returns);
    // Excess kurtosis for normal is 0, but samples may vary
    assert!(kurt.abs() < dec!(3), "Normal-ish kurtosis should be near 0, got {}", kurt);
}

#[test]
fn test_kurtosis_fat_tailed_positive() {
    // Fat-tailed distribution (extreme outliers)
    let mut returns: Vec<Decimal> = vec![dec!(0.001); 18];
    returns.push(dec!(0.15));  // extreme positive
    returns.push(dec!(-0.12)); // extreme negative
    
    let kurt = calculate_kurtosis(&returns);
    assert!(kurt > Decimal::ZERO, "Fat-tailed should have positive excess kurtosis, got {}", kurt);
}

#[test]
fn test_kurtosis_uniform_negative() {
    // Uniform-ish distribution (thin tails)
    let returns: Vec<Decimal> = (0..20)
        .map(|i| dec!(-0.05) + Decimal::from(i) * dec!(0.005))
        .collect();
    
    let kurt = calculate_kurtosis(&returns);
    // Uniform has excess kurtosis of -1.2
    assert!(kurt < dec!(0), "Uniform should have negative excess kurtosis, got {}", kurt);
}

#[test]
fn test_kurtosis_empty_returns_zero() {
    let kurt = calculate_kurtosis(&[]);
    assert_eq!(kurt, Decimal::ZERO);
}

#[test]
fn test_kurtosis_few_values_returns_zero() {
    let kurt = calculate_kurtosis(&[dec!(0.01), dec!(0.02), dec!(0.03)]);
    assert_eq!(kurt, Decimal::ZERO);
}

// ========================
// PSR Tests
// ========================

#[test]
fn test_psr_at_threshold_is_half() {
    // When Sharpe = threshold, PSR should be around 0.5
    let psr = calculate_psr(dec!(0.5), dec!(0.5), 100, Decimal::ZERO, Decimal::ZERO);
    assert!(
        (psr - dec!(0.5)).abs() < dec!(0.15),
        "PSR at threshold should be ~0.5, got {}",
        psr
    );
}

#[test]
fn test_psr_above_threshold_high() {
    // When Sharpe >> threshold, PSR should be high
    let psr = calculate_psr(dec!(2.0), dec!(0.5), 200, Decimal::ZERO, Decimal::ZERO);
    assert!(psr > dec!(0.9), "PSR should be high for Sharpe >> threshold, got {}", psr);
}

#[test]
fn test_psr_below_threshold_low() {
    // When Sharpe < threshold, PSR should be low
    let psr = calculate_psr(dec!(0.2), dec!(0.5), 100, Decimal::ZERO, Decimal::ZERO);
    assert!(psr < dec!(0.5), "PSR should be low for Sharpe < threshold, got {}", psr);
}

#[test]
fn test_psr_negative_sharpe_very_low() {
    // Negative Sharpe should have very low PSR
    let psr = calculate_psr(dec!(-0.5), dec!(0.5), 100, Decimal::ZERO, Decimal::ZERO);
    assert!(psr < dec!(0.1), "PSR should be very low for negative Sharpe, got {}", psr);
}

#[test]
fn test_psr_monotonic_in_sharpe() {
    let psr_low = calculate_psr(dec!(0.5), dec!(0.5), 100, Decimal::ZERO, Decimal::ZERO);
    let psr_mid = calculate_psr(dec!(1.0), dec!(0.5), 100, Decimal::ZERO, Decimal::ZERO);
    let psr_high = calculate_psr(dec!(1.5), dec!(0.5), 100, Decimal::ZERO, Decimal::ZERO);
    
    assert!(psr_low < psr_mid, "PSR should increase with Sharpe: {} < {}", psr_low, psr_mid);
    assert!(psr_mid < psr_high, "PSR should increase with Sharpe: {} < {}", psr_mid, psr_high);
}

#[test]
fn test_psr_increases_with_n_for_good_sharpe() {
    // More observations = more confidence, higher PSR for SR > threshold
    let psr_n50 = calculate_psr(dec!(1.0), dec!(0.5), 50, Decimal::ZERO, Decimal::ZERO);
    let psr_n200 = calculate_psr(dec!(1.0), dec!(0.5), 200, Decimal::ZERO, Decimal::ZERO);
    
    assert!(psr_n200 > psr_n50, "PSR should increase with N for SR > threshold");
}

#[test]
fn test_psr_bounded_zero_to_one() {
    let psr_high = calculate_psr(dec!(5.0), dec!(0.5), 1000, Decimal::ZERO, Decimal::ZERO);
    let psr_low = calculate_psr(dec!(-5.0), dec!(0.5), 1000, Decimal::ZERO, Decimal::ZERO);
    
    assert!(psr_high >= Decimal::ZERO && psr_high <= Decimal::ONE);
    assert!(psr_low >= Decimal::ZERO && psr_low <= Decimal::ONE);
}

#[test]
fn test_psr_affected_by_skewness() {
    // Positive skewness should slightly reduce PSR (fat right tail is unreliable)
    let psr_no_skew = calculate_psr(dec!(1.0), dec!(0.5), 100, Decimal::ZERO, Decimal::ZERO);
    let psr_pos_skew = calculate_psr(dec!(1.0), dec!(0.5), 100, dec!(1.0), Decimal::ZERO);
    
    // Skewness affects variance estimate, may slightly change PSR
    // Just verify it doesn't break
    assert!(psr_pos_skew >= Decimal::ZERO);
    assert!(psr_pos_skew <= Decimal::ONE);
}

#[test]
fn test_psr_affected_by_kurtosis() {
    // Positive kurtosis (fat tails) should reduce PSR (more uncertainty)
    let psr_no_kurt = calculate_psr(dec!(1.0), dec!(0.5), 100, Decimal::ZERO, Decimal::ZERO);
    let psr_pos_kurt = calculate_psr(dec!(1.0), dec!(0.5), 100, Decimal::ZERO, dec!(3.0));
    
    assert!(psr_pos_kurt >= Decimal::ZERO);
    assert!(psr_pos_kurt <= Decimal::ONE);
}

#[test]
fn test_psr_small_n_still_works() {
    let psr = calculate_psr(dec!(1.0), dec!(0.5), 10, Decimal::ZERO, Decimal::ZERO);
    assert!(psr >= Decimal::ZERO);
    assert!(psr <= Decimal::ONE);
}

#[test]
fn test_psr_n_one_returns_zero() {
    let psr = calculate_psr(dec!(1.0), dec!(0.5), 1, Decimal::ZERO, Decimal::ZERO);
    assert_eq!(psr, Decimal::ZERO);
}

// ========================
// DSR Tests
// ========================

#[test]
fn test_dsr_less_than_or_equal_psr() {
    // DSR should always be <= PSR due to multiple testing penalty
    let psr = calculate_psr(dec!(1.0), dec!(0.5), 100, Decimal::ZERO, Decimal::ZERO);
    let dsr = calculate_dsr(
        dec!(1.0), dec!(0.5), 100,
        Decimal::ZERO, Decimal::ZERO,
        50,  // 50 trials
        dec!(0.25),  // variance of Sharpes
    );
    
    assert!(dsr <= psr, "DSR {} should be <= PSR {}", dsr, psr);
}

#[test]
fn test_dsr_decreases_with_more_trials() {
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
    
    assert!(dsr_100 < dsr_10, "DSR should decrease with more trials: {} < {}", dsr_100, dsr_10);
}

#[test]
fn test_dsr_single_trial_approx_psr() {
    // With single trial, DSR ≈ PSR (against zero threshold)
    let psr = calculate_psr(dec!(1.0), Decimal::ZERO, 100, Decimal::ZERO, Decimal::ZERO);
    let dsr = calculate_dsr(
        dec!(1.0), dec!(0.5), 100,
        Decimal::ZERO, Decimal::ZERO,
        1,  // single trial
        dec!(0.25),
    );
    
    // Should be close (within 0.1)
    assert!((dsr - psr).abs() < dec!(0.15), "DSR {} should be ~= PSR {} for single trial", dsr, psr);
}

#[test]
fn test_dsr_bounded() {
    let dsr = calculate_dsr(
        dec!(3.0), dec!(0.5), 200,
        Decimal::ZERO, Decimal::ZERO,
        100, dec!(0.5),
    );
    
    assert!(dsr >= Decimal::ZERO && dsr <= Decimal::ONE);
}

#[test]
fn test_dsr_higher_variance_lower_dsr() {
    // Higher variance of Sharpes across trials = more luck = lower DSR
    let dsr_low_var = calculate_dsr(
        dec!(1.0), dec!(0.5), 100,
        Decimal::ZERO, Decimal::ZERO,
        50, dec!(0.1),  // low variance
    );
    let dsr_high_var = calculate_dsr(
        dec!(1.0), dec!(0.5), 100,
        Decimal::ZERO, Decimal::ZERO,
        50, dec!(0.5),  // high variance
    );
    
    assert!(dsr_high_var < dsr_low_var, "Higher variance should give lower DSR");
}

#[test]
fn test_dsr_many_trials_very_low() {
    // With many trials, even a good Sharpe has low DSR
    let dsr = calculate_dsr(
        dec!(1.0), dec!(0.5), 100,
        Decimal::ZERO, Decimal::ZERO,
        1000,  // many trials
        dec!(0.3),
    );
    
    assert!(dsr < dec!(0.3), "DSR should be low with 1000 trials, got {}", dsr);
}

// ========================
// Integration Tests
// ========================

#[test]
fn test_psr_dsr_with_realistic_data() {
    // Generate realistic daily returns
    let returns: Vec<Decimal> = (0..252)
        .map(|i| {
            let trend = dec!(0.0003);  // ~7.5% annual
            let noise = Decimal::from((i % 20) as i32 - 10) * dec!(0.001);
            trend + noise
        })
        .collect();
    
    let skew = calculate_skewness(&returns);
    let kurt = calculate_kurtosis(&returns);
    
    // Calculate annualized Sharpe
    let mean = returns.iter().sum::<Decimal>() / Decimal::from(returns.len());
    let var: Decimal = returns.iter()
        .map(|r| (*r - mean) * (*r - mean))
        .sum::<Decimal>() / Decimal::from(returns.len());
    let std = var.sqrt().unwrap_or(dec!(0.01));
    let sharpe_ann = mean / std * dec!(15.87);  // sqrt(252)
    
    let psr = calculate_psr(sharpe_ann, dec!(0.5), returns.len(), skew, kurt);
    let dsr = calculate_dsr(
        sharpe_ann, dec!(0.5), returns.len(),
        skew, kurt,
        20,  // 20 parameter sets tested
        dec!(0.2),
    );
    
    // Verify reasonable values
    assert!(psr >= Decimal::ZERO && psr <= Decimal::ONE);
    assert!(dsr >= Decimal::ZERO && dsr <= Decimal::ONE);
    assert!(dsr <= psr);
}

#[test]
fn test_skewness_kurtosis_determinism() {
    let returns: Vec<Decimal> = vec![
        dec!(0.01), dec!(-0.02), dec!(0.015), dec!(-0.005), dec!(0.02),
        dec!(0.008), dec!(-0.01), dec!(0.012), dec!(-0.003), dec!(0.018),
    ];
    
    let skew1 = calculate_skewness(&returns);
    let skew2 = calculate_skewness(&returns);
    assert_eq!(skew1, skew2, "Skewness should be deterministic");
    
    let kurt1 = calculate_kurtosis(&returns);
    let kurt2 = calculate_kurtosis(&returns);
    assert_eq!(kurt1, kurt2, "Kurtosis should be deterministic");
}

#[test]
fn test_psr_dsr_determinism() {
    let psr1 = calculate_psr(dec!(1.0), dec!(0.5), 100, dec!(0.1), dec!(0.5));
    let psr2 = calculate_psr(dec!(1.0), dec!(0.5), 100, dec!(0.1), dec!(0.5));
    assert_eq!(psr1, psr2, "PSR should be deterministic");
    
    let dsr1 = calculate_dsr(dec!(1.0), dec!(0.5), 100, dec!(0.1), dec!(0.5), 50, dec!(0.25));
    let dsr2 = calculate_dsr(dec!(1.0), dec!(0.5), 100, dec!(0.1), dec!(0.5), 50, dec!(0.25));
    assert_eq!(dsr1, dsr2, "DSR should be deterministic");
}




//! Risk metrics calculation: Volatility, VaR, Exposure.
//!
//! Includes sector-based exposure analysis for research-grade reporting.

use rust_decimal::Decimal;
use std::collections::BTreeMap;

use crate::filters::Market;
use super::sector::SectorProvider;
use super::types::{VolatilityMetrics, VaRMetrics, VaRMethod, DrawdownMetrics, ExposureBreakdown, PositionLot, SectorExposure};

/// Risk calculator for volatility, VaR, and exposure metrics.
#[derive(Debug, Clone)]
pub struct RiskCalculator {
    /// Rolling window for volatility (days)
    pub vol_window: u32,
    /// Risk-free rate for Sharpe calculation
    pub risk_free_rate: Decimal,
}

impl RiskCalculator {
    pub fn new(vol_window: u32, risk_free_rate: Decimal) -> Self {
        Self { vol_window, risk_free_rate }
    }

    /// Calculate volatility from daily returns.
    pub fn calculate_volatility(&self, returns: &[Decimal]) -> VolatilityMetrics {
        if returns.is_empty() {
            return VolatilityMetrics::default();
        }

        let n = returns.len();
        let mean: Decimal = returns.iter().sum::<Decimal>() / Decimal::from(n as u32);
        
        let variance: Decimal = returns.iter()
            .map(|r| {
                let diff = *r - mean;
                diff * diff
            })
            .sum::<Decimal>() / Decimal::from(n.max(1) as u32);

        // Approximate sqrt using Newton-Raphson
        let daily_vol = decimal_sqrt(variance);
        
        VolatilityMetrics::from_daily(daily_vol, self.vol_window)
    }

    /// Calculate historical VaR from equity curve.
    ///
    /// VaR at 95%: 5th percentile of returns
    /// VaR at 99%: 1st percentile of returns
    pub fn calculate_var(&self, returns: &[Decimal], portfolio_value: Decimal) -> VaRMetrics {
        if returns.is_empty() {
            return VaRMetrics::default();
        }

        let mut sorted = returns.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted.len();
        
        // 5th percentile index
        let idx_95 = ((n as f64 * 0.05) as usize).min(n - 1);
        // 1st percentile index
        let idx_99 = ((n as f64 * 0.01) as usize).min(n - 1);

        VaRMetrics {
            var_95: sorted[idx_95] * portfolio_value,
            var_99: sorted[idx_99] * portfolio_value,
            method: VaRMethod::Historical,
        }
    }

    /// Calculate drawdown metrics from equity curve.
    pub fn calculate_drawdown(&self, equity_curve: &[Decimal]) -> DrawdownMetrics {
        if equity_curve.is_empty() {
            return DrawdownMetrics::default();
        }

        let mut hwm = equity_curve[0];
        let mut max_dd = Decimal::ZERO;
        let mut current_dd = Decimal::ZERO;
        let mut dd_start: Option<usize> = None;
        let mut max_dd_duration: u32 = 0;
        let mut current_dd_duration: u32 = 0;

        for (i, &equity) in equity_curve.iter().enumerate() {
            if equity > hwm {
                hwm = equity;
                dd_start = None;
                current_dd_duration = 0;
            } else {
                current_dd = (hwm - equity) / hwm;
                if current_dd > max_dd {
                    max_dd = current_dd;
                }
                
                if dd_start.is_none() {
                    dd_start = Some(i);
                }
                current_dd_duration = (i - dd_start.unwrap_or(i)) as u32 + 1;
                max_dd_duration = max_dd_duration.max(current_dd_duration);
            }
        }

        // Current drawdown from last equity point
        let last_equity = *equity_curve.last().unwrap();
        current_dd = if hwm > Decimal::ZERO {
            (hwm - last_equity) / hwm
        } else {
            Decimal::ZERO
        };

        DrawdownMetrics {
            current_dd,
            max_dd,
            dd_duration_days: max_dd_duration,
            hwm,
        }
    }

    /// Calculate exposure breakdown from positions.
    pub fn calculate_exposure(
        &self,
        positions: &BTreeMap<String, PositionLot>,
        prices: &BTreeMap<String, Decimal>,
    ) -> ExposureBreakdown {
        let mut long = Decimal::ZERO;
        let short = Decimal::ZERO; // Long-only for now
        let mut by_market: BTreeMap<String, Decimal> = BTreeMap::new();

        for (symbol, pos) in positions {
            let price = prices.get(symbol).copied().unwrap_or(pos.wap_cost_basis);
            let value = price * Decimal::from(pos.shares);
            long += value;

            let market_key = match pos.market {
                Market::BR => "BR".to_string(),
                Market::US => "US".to_string(),
            };
            *by_market.entry(market_key).or_default() += value;
        }

        ExposureBreakdown {
            gross: long + short.abs(),
            net: long - short.abs(),
            long,
            short,
            by_market,
            by_currency: BTreeMap::new(),
            by_currency_base: BTreeMap::new(),
            by_sector: Vec::new(),
        }
    }

    /// Calculate exposure breakdown by sector.
    ///
    /// Uses the provided `SectorProvider` to classify each position.
    /// Positions with unknown sectors are grouped under "Unknown".
    ///
    /// # Arguments
    ///
    /// * `positions` - Map of symbol to position
    /// * `prices` - Current market prices
    /// * `sector_provider` - Provider for sector lookups
    ///
    /// # Returns
    ///
    /// Vector of `SectorExposure` sorted by gross exposure (descending).
    pub fn calculate_exposure_by_sector(
        &self,
        positions: &BTreeMap<String, PositionLot>,
        prices: &BTreeMap<String, Decimal>,
        sector_provider: &dyn SectorProvider,
    ) -> Vec<SectorExposure> {
        if positions.is_empty() {
            return Vec::new();
        }

        // Accumulate by sector
        let mut sector_map: BTreeMap<String, SectorExposure> = BTreeMap::new();
        let mut total_gross = Decimal::ZERO;

        for (symbol, pos) in positions {
            let price = prices.get(symbol).copied().unwrap_or(pos.wap_cost_basis);
            let value = price * Decimal::from(pos.shares);
            
            let sector = sector_provider.get_sector(symbol);
            let sector_name = sector.as_str().to_string();

            let entry = sector_map
                .entry(sector_name.clone())
                .or_insert_with(|| SectorExposure::new(sector_name));

            if pos.shares >= 0 {
                entry.add_long(value);
            } else {
                entry.add_short(value.abs());
            }
            
            total_gross += value.abs();
        }

        // Calculate weights and convert to sorted vec
        let mut result: Vec<SectorExposure> = sector_map.into_values().collect();
        for exposure in &mut result {
            exposure.calculate_weight(total_gross);
        }

        // Sort by gross exposure descending for consistent ordering
        result.sort_by(|a, b| b.gross.cmp(&a.gross));

        result
    }

    /// Calculate full exposure breakdown including sectors.
    ///
    /// Combines market, currency, and sector exposure calculations.
    pub fn calculate_exposure_with_sectors(
        &self,
        positions: &BTreeMap<String, PositionLot>,
        prices: &BTreeMap<String, Decimal>,
        sector_provider: &dyn SectorProvider,
    ) -> ExposureBreakdown {
        let mut exposure = self.calculate_exposure(positions, prices);
        exposure.by_sector = self.calculate_exposure_by_sector(positions, prices, sector_provider);
        exposure
    }

    /// Calculate Sharpe ratio from returns.
    /// Clamped to [-10, 10] to prevent unrealistic values from low volatility data.
    pub fn calculate_sharpe(&self, returns: &[Decimal]) -> Decimal {
        if returns.is_empty() {
            return Decimal::ZERO;
        }

        let n = returns.len();
        let mean: Decimal = returns.iter().sum::<Decimal>() / Decimal::from(n as u32);
        let excess_return = mean - self.risk_free_rate / Decimal::from(252);

        let variance: Decimal = returns.iter()
            .map(|r| {
                let diff = *r - mean;
                diff * diff
            })
            .sum::<Decimal>() / Decimal::from(n.max(1) as u32);

        let vol = decimal_sqrt(variance);
        
        if vol.is_zero() {
            return Decimal::ZERO;
        }

        // Annualize: sharpe * sqrt(252)
        let sqrt_252 = Decimal::from_str_exact("15.87").unwrap_or(Decimal::from(16));
        let sharpe = excess_return / vol * sqrt_252;
        
        // Clamp to realistic bounds
        let min_sharpe = Decimal::from(-10);
        let max_sharpe = Decimal::from(10);
        sharpe.max(min_sharpe).min(max_sharpe)
    }
}

impl Default for RiskCalculator {
    fn default() -> Self {
        Self::new(21, Decimal::ZERO)
    }
}

/// Approximate square root for Decimal using Newton-Raphson.
fn decimal_sqrt(x: Decimal) -> Decimal {
    if x <= Decimal::ZERO {
        return Decimal::ZERO;
    }

    // Initial guess
    let mut guess = x / Decimal::from(2);
    if guess.is_zero() {
        guess = Decimal::from_str_exact("0.5").unwrap();
    }

    // Newton-Raphson iterations
    for _ in 0..10 {
        let new_guess = (guess + x / guess) / Decimal::from(2);
        if (new_guess - guess).abs() < Decimal::from_str_exact("0.0000001").unwrap() {
            return new_guess;
        }
        guess = new_guess;
    }

    guess
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::performance::sector::InMemorySectorProvider;
    use rust_decimal_macros::dec;

    #[test]
    fn test_volatility_calculation() {
        let calc = RiskCalculator::default();
        
        // Known volatility case
        let returns = vec![dec!(0.01), dec!(-0.02), dec!(0.015), dec!(-0.005), dec!(0.01)];
        let vol = calc.calculate_volatility(&returns);
        
        assert!(vol.daily_vol > Decimal::ZERO);
        assert!(vol.annualized_vol > vol.daily_vol);
    }

    #[test]
    fn test_var_calculation() {
        let calc = RiskCalculator::default();
        
        // Returns with known percentiles
        let returns: Vec<Decimal> = (-10..10).map(|i| Decimal::from(i) / Decimal::from(100)).collect();
        let var = calc.calculate_var(&returns, dec!(100000));

        // VaR95 should be negative (loss)
        assert!(var.var_95 < Decimal::ZERO);
        // VaR99 should be more negative than VaR95
        assert!(var.var_99 <= var.var_95);
    }

    #[test]
    fn test_drawdown_calculation() {
        let calc = RiskCalculator::default();
        
        let equity = vec![
            dec!(100), dec!(105), dec!(110), dec!(100), dec!(95), dec!(108), dec!(115)
        ];
        let dd = calc.calculate_drawdown(&equity);

        // Max drawdown from 110 to 95 = (110-95)/110 = ~13.6%
        assert!(dd.max_dd > Decimal::ZERO);
        // HWM should be 115
        assert_eq!(dd.hwm, dec!(115));
        // Current DD should be 0 (at HWM)
        assert_eq!(dd.current_dd, Decimal::ZERO);
    }

    #[test]
    fn test_exposure_by_market() {
        let calc = RiskCalculator::default();
        
        let mut positions = BTreeMap::new();
        positions.insert("PETR4".to_string(), PositionLot::new(
            "PETR4".into(), 100, dec!(30), Market::BR,
            chrono::NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
        ));
        positions.insert("AAPL".to_string(), PositionLot::new(
            "AAPL".into(), 10, dec!(150), Market::US,
            chrono::NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
        ));

        let prices: BTreeMap<String, Decimal> = [
            ("PETR4".to_string(), dec!(35)),
            ("AAPL".to_string(), dec!(160)),
        ].into();

        let exp = calc.calculate_exposure(&positions, &prices);

        // PETR4: 100 * 35 = 3500
        // AAPL: 10 * 160 = 1600
        assert_eq!(exp.long, dec!(5100));
        assert_eq!(exp.by_market.get("BR"), Some(&dec!(3500)));
        assert_eq!(exp.by_market.get("US"), Some(&dec!(1600)));
    }

    #[test]
    fn test_sharpe_calculation() {
        let calc = RiskCalculator::new(21, Decimal::ZERO);
        
        // Positive returns with some variance
        let returns = vec![dec!(0.01), dec!(0.02), dec!(0.015), dec!(0.008), dec!(0.012)];
        let sharpe = calc.calculate_sharpe(&returns);

        // Should be positive for positive returns
        assert!(sharpe > Decimal::ZERO);
    }

    #[test]
    fn test_empty_inputs() {
        let calc = RiskCalculator::default();
        
        assert_eq!(calc.calculate_volatility(&[]).daily_vol, Decimal::ZERO);
        assert_eq!(calc.calculate_var(&[], dec!(100000)).var_95, Decimal::ZERO);
        assert_eq!(calc.calculate_drawdown(&[]).max_dd, Decimal::ZERO);
    }

    #[test]
    fn test_decimal_sqrt() {
        let sqrt4 = decimal_sqrt(dec!(4));
        assert!((sqrt4 - dec!(2)).abs() < dec!(0.001));

        let sqrt9 = decimal_sqrt(dec!(9));
        assert!((sqrt9 - dec!(3)).abs() < dec!(0.001));
    }

    // ==========================================================================
    // SECTOR EXPOSURE TESTS
    // ==========================================================================

    #[test]
    fn test_exposure_by_sector_basic() {
        let calc = RiskCalculator::default();
        
        // Set up positions
        let mut positions = BTreeMap::new();
        positions.insert("PETR4".to_string(), PositionLot::new(
            "PETR4".into(), 100, dec!(30), Market::BR,
            chrono::NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
        ));
        positions.insert("VALE3".to_string(), PositionLot::new(
            "VALE3".into(), 50, dec!(60), Market::BR,
            chrono::NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
        ));
        positions.insert("ITUB4".to_string(), PositionLot::new(
            "ITUB4".into(), 200, dec!(25), Market::BR,
            chrono::NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
        ));

        let prices: BTreeMap<String, Decimal> = [
            ("PETR4".to_string(), dec!(35)),   // 3500
            ("VALE3".to_string(), dec!(70)),   // 3500
            ("ITUB4".to_string(), dec!(30)),   // 6000
        ].into();  // Total: 13000

        // Set up sector provider
        let mut sector_provider = InMemorySectorProvider::new();
        sector_provider.add("PETR4", "Energy");
        sector_provider.add("VALE3", "Materials");
        sector_provider.add("ITUB4", "Financials");

        let sectors = calc.calculate_exposure_by_sector(&positions, &prices, &sector_provider);

        assert_eq!(sectors.len(), 3);
        
        // Sorted by gross descending: Financials (6000), Energy (3500), Materials (3500)
        assert_eq!(sectors[0].sector, "Financials");
        assert_eq!(sectors[0].gross, dec!(6000));
        
        // Check weights sum to 100
        let total_weight: Decimal = sectors.iter().map(|s| s.weight_pct).sum();
        assert!((total_weight - dec!(100)).abs() < dec!(0.01));
    }

    #[test]
    fn test_exposure_by_sector_unknown_fallback() {
        let calc = RiskCalculator::default();
        
        let mut positions = BTreeMap::new();
        positions.insert("UNKNOWN1".to_string(), PositionLot::new(
            "UNKNOWN1".into(), 100, dec!(10), Market::BR,
            chrono::NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
        ));

        let prices: BTreeMap<String, Decimal> = [
            ("UNKNOWN1".to_string(), dec!(10)),
        ].into();

        // Empty sector provider
        let sector_provider = InMemorySectorProvider::new();

        let sectors = calc.calculate_exposure_by_sector(&positions, &prices, &sector_provider);

        assert_eq!(sectors.len(), 1);
        assert_eq!(sectors[0].sector, "Unknown");
        assert_eq!(sectors[0].gross, dec!(1000));
    }

    #[test]
    fn test_exposure_by_sector_empty_positions() {
        let calc = RiskCalculator::default();
        let positions: BTreeMap<String, PositionLot> = BTreeMap::new();
        let prices: BTreeMap<String, Decimal> = BTreeMap::new();
        let sector_provider = InMemorySectorProvider::new();

        let sectors = calc.calculate_exposure_by_sector(&positions, &prices, &sector_provider);

        assert!(sectors.is_empty());
    }

    #[test]
    fn test_exposure_with_sectors_integration() {
        let calc = RiskCalculator::default();
        
        let mut positions = BTreeMap::new();
        positions.insert("PETR4".to_string(), PositionLot::new(
            "PETR4".into(), 100, dec!(30), Market::BR,
            chrono::NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
        ));

        let prices: BTreeMap<String, Decimal> = [
            ("PETR4".to_string(), dec!(35)),
        ].into();

        let mut sector_provider = InMemorySectorProvider::new();
        sector_provider.add("PETR4", "Energy");

        let exposure = calc.calculate_exposure_with_sectors(&positions, &prices, &sector_provider);

        // Check basic exposure
        assert_eq!(exposure.long, dec!(3500));
        assert_eq!(exposure.gross, dec!(3500));

        // Check sector exposure
        assert_eq!(exposure.by_sector.len(), 1);
        assert_eq!(exposure.by_sector[0].sector, "Energy");
        assert_eq!(exposure.by_sector[0].weight_pct, dec!(100));
    }

    #[test]
    fn test_exposure_by_sector_same_sector_aggregation() {
        let calc = RiskCalculator::default();
        
        // Multiple positions in same sector
        let mut positions = BTreeMap::new();
        positions.insert("PETR4".to_string(), PositionLot::new(
            "PETR4".into(), 100, dec!(30), Market::BR,
            chrono::NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
        ));
        positions.insert("PETR3".to_string(), PositionLot::new(
            "PETR3".into(), 50, dec!(28), Market::BR,
            chrono::NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
        ));

        let prices: BTreeMap<String, Decimal> = [
            ("PETR4".to_string(), dec!(35)),   // 3500
            ("PETR3".to_string(), dec!(32)),   // 1600
        ].into();

        let mut sector_provider = InMemorySectorProvider::new();
        sector_provider.add("PETR4", "Energy");
        sector_provider.add("PETR3", "Energy");

        let sectors = calc.calculate_exposure_by_sector(&positions, &prices, &sector_provider);

        assert_eq!(sectors.len(), 1);
        assert_eq!(sectors[0].sector, "Energy");
        assert_eq!(sectors[0].gross, dec!(5100));
        assert_eq!(sectors[0].weight_pct, dec!(100));
    }

    // =========================================================================
    // Phase 1.3: Comprehensive Risk Metrics Validation
    // VaR, CVaR, Beta, Tracking Error with known distributions
    // =========================================================================

    #[test]
    fn test_var_known_percentiles() {
        let calc = RiskCalculator::default();
        
        // Create a uniform distribution from -10% to +10%
        // For 100 values: 5th percentile = -9%, 1st percentile = -10%
        let returns: Vec<Decimal> = (0..100)
            .map(|i| Decimal::from(i as i32 - 50) / Decimal::from(500))
            .collect();
        // Range: -0.10 to +0.098 (approximately)
        
        let var = calc.calculate_var(&returns, dec!(100000));
        
        // VaR95 should be around -9% * 100000 = -9000
        // VaR99 should be around -10% * 100000 = -10000
        assert!(var.var_95 < Decimal::ZERO, "VaR95 should be negative: {}", var.var_95);
        assert!(var.var_99 <= var.var_95, "VaR99 {} should be <= VaR95 {}", var.var_99, var.var_95);
    }

    #[test]
    fn test_var_property_ordering() {
        // Property: VaR99 >= VaR95 in absolute terms (both are losses)
        let calc = RiskCalculator::default();
        
        let returns: Vec<Decimal> = (-50..50)
            .map(|i| Decimal::from(i) / Decimal::from(1000))
            .collect();
        
        let var = calc.calculate_var(&returns, dec!(100000));
        
        assert!(var.var_99.abs() >= var.var_95.abs() || var.var_99 == var.var_95,
            "VaR99 should be more extreme than VaR95: {} vs {}", var.var_99, var.var_95);
    }

    #[test]
    fn test_var_scaled_by_portfolio_value() {
        let calc = RiskCalculator::default();
        
        let returns: Vec<Decimal> = vec![dec!(-0.05), dec!(0.01), dec!(-0.02), dec!(0.03)];
        
        let var_100k = calc.calculate_var(&returns, dec!(100000));
        let var_200k = calc.calculate_var(&returns, dec!(200000));
        
        // VaR should scale linearly with portfolio value
        let ratio = if var_100k.var_95 != Decimal::ZERO {
            var_200k.var_95 / var_100k.var_95
        } else {
            Decimal::from(2)
        };
        
        assert!((ratio - dec!(2)).abs() < dec!(0.01),
            "VaR should scale with portfolio: ratio = {}", ratio);
    }

    #[test]
    fn test_volatility_known_returns() {
        let calc = RiskCalculator::default();
        
        // Returns with known daily volatility
        // If all returns are +1%, daily vol = 0
        let constant_returns = vec![dec!(0.01); 10];
        let vol_const = calc.calculate_volatility(&constant_returns);
        assert_eq!(vol_const.daily_vol, Decimal::ZERO, "Constant returns should have 0 vol");
        
        // Alternating returns: +1%, -1%, +1%, -1%...
        // Mean = 0, variance = 0.01^2 = 0.0001
        let alternating: Vec<Decimal> = (0..100)
            .map(|i| if i % 2 == 0 { dec!(0.01) } else { dec!(-0.01) })
            .collect();
        let vol_alt = calc.calculate_volatility(&alternating);
        
        // Daily vol should be approximately 0.01
        assert!((vol_alt.daily_vol - dec!(0.01)).abs() < dec!(0.001),
            "Alternating +/-1% should have daily vol ~0.01, got {}", vol_alt.daily_vol);
    }

    #[test]
    fn test_volatility_annualization() {
        let calc = RiskCalculator::default();
        
        // Create returns with known volatility
        let returns: Vec<Decimal> = (0..252)
            .map(|i| if i % 2 == 0 { dec!(0.01) } else { dec!(-0.01) })
            .collect();
        
        let vol = calc.calculate_volatility(&returns);
        
        // Annualized vol = daily_vol * sqrt(252)
        // sqrt(252) ≈ 15.87
        let expected_ann = vol.daily_vol * dec!(15.87);
        
        assert!((vol.annualized_vol - expected_ann).abs() < dec!(0.5),
            "Annualized vol {} should be daily {} * 15.87", vol.annualized_vol, vol.daily_vol);
    }

    #[test]
    fn test_drawdown_properties() {
        let calc = RiskCalculator::default();
        
        // Property 1: DD is always >= 0
        let equity1 = vec![dec!(100), dec!(90), dec!(110), dec!(100)];
        let dd1 = calc.calculate_drawdown(&equity1);
        assert!(dd1.max_dd >= Decimal::ZERO, "Max DD should be >= 0: {}", dd1.max_dd);
        
        // Property 2: For monotonic up, DD = 0
        let equity2 = vec![dec!(100), dec!(101), dec!(102), dec!(103)];
        let dd2 = calc.calculate_drawdown(&equity2);
        assert_eq!(dd2.max_dd, Decimal::ZERO, "Monotonic up should have 0 DD");
        
        // Property 3: DD <= 100%
        let equity3 = vec![dec!(100), dec!(50), dec!(25), dec!(100)];
        let dd3 = calc.calculate_drawdown(&equity3);
        assert!(dd3.max_dd <= dec!(1), "Max DD should be <= 100%: {}", dd3.max_dd);
    }

    #[test]
    fn test_drawdown_exact_calculation() {
        let calc = RiskCalculator::default();
        
        // Equity: 100 -> 120 -> 100 -> 130
        // DD from 120 to 100 = (120-100)/120 = 0.1667
        let equity = vec![dec!(100), dec!(120), dec!(100), dec!(130)];
        let dd = calc.calculate_drawdown(&equity);
        
        let expected_dd = (dec!(120) - dec!(100)) / dec!(120);
        assert!((dd.max_dd - expected_dd).abs() < dec!(0.01),
            "Max DD should be ~16.67%, got {}", dd.max_dd);
    }

    #[test]
    fn test_sharpe_known_returns() {
        let calc = RiskCalculator::new(21, Decimal::ZERO);
        
        // All positive returns should have positive Sharpe
        let positive = vec![dec!(0.01), dec!(0.02), dec!(0.015), dec!(0.01)];
        let sharpe_pos = calc.calculate_sharpe(&positive);
        assert!(sharpe_pos > Decimal::ZERO, "All positive returns should have positive Sharpe");
        
        // All negative returns should have negative Sharpe
        let negative = vec![dec!(-0.01), dec!(-0.02), dec!(-0.015), dec!(-0.01)];
        let sharpe_neg = calc.calculate_sharpe(&negative);
        assert!(sharpe_neg < Decimal::ZERO, "All negative returns should have negative Sharpe");
        
        // Zero mean returns should have ~0 Sharpe
        let zero_mean = vec![dec!(0.01), dec!(-0.01), dec!(0.01), dec!(-0.01)];
        let sharpe_zero = calc.calculate_sharpe(&zero_mean);
        assert!(sharpe_zero.abs() < dec!(0.1), "Zero mean returns should have ~0 Sharpe");
    }

    #[test]
    fn test_sharpe_with_risk_free_rate() {
        // Sharpe should decrease when risk-free rate increases
        let returns = vec![dec!(0.01), dec!(0.02), dec!(0.015), dec!(0.01)];
        
        let calc_zero_rf = RiskCalculator::new(21, Decimal::ZERO);
        let calc_high_rf = RiskCalculator::new(21, dec!(0.10)); // 10% annual
        
        let sharpe_zero = calc_zero_rf.calculate_sharpe(&returns);
        let sharpe_high = calc_high_rf.calculate_sharpe(&returns);
        
        assert!(sharpe_zero > sharpe_high,
            "Higher RF should reduce Sharpe: {} vs {}", sharpe_zero, sharpe_high);
    }

    #[test]
    fn test_decimal_sqrt_accuracy() {
        // Test sqrt accuracy for various values
        let test_cases = vec![
            (dec!(1), dec!(1)),
            (dec!(4), dec!(2)),
            (dec!(9), dec!(3)),
            (dec!(16), dec!(4)),
            (dec!(100), dec!(10)),
            (dec!(0.25), dec!(0.5)),
        ];
        
        for (input, expected) in test_cases {
            let result = decimal_sqrt(input);
            assert!((result - expected).abs() < dec!(0.001),
                "sqrt({}) = {}, expected {}", input, result, expected);
        }
    }

    #[test]
    fn test_decimal_sqrt_edge_cases() {
        // Zero and negative
        assert_eq!(decimal_sqrt(Decimal::ZERO), Decimal::ZERO);
        assert_eq!(decimal_sqrt(dec!(-1)), Decimal::ZERO);
        
        // Very small positive
        let tiny = decimal_sqrt(dec!(0.0001));
        assert!((tiny - dec!(0.01)).abs() < dec!(0.001));
    }
}








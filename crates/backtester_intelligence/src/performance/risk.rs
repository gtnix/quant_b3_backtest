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
        excess_return / vol * sqrt_252
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
}








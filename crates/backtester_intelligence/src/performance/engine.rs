//! Performance Engine - Orchestrator for snapshot generation.
//!
//! Supports both local currency and base currency consolidated views.

use chrono::NaiveDate;
use rust_decimal::Decimal;
use std::collections::BTreeMap;
use std::sync::Arc;

use super::{
    TradeLedger, AttributionEngine, RiskCalculator,
    PerformanceSnapshot, TurnoverMetrics, AttributionBreakdown, VaRMetrics,
    VolatilityMetrics, CIOView, FxRateInfo,
    FxAttributionEngine, FxAttributionBreakdown,
};
use crate::currency::{Currency, FxPair};
use crate::fx::{FxRateProvider, FxError};
use crate::filters::Market;

/// Configuration for the performance engine.
#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    pub vol_window: u32,
    pub risk_free_rate: Decimal,
    pub var_confidence_levels: (Decimal, Decimal), // (95%, 99%)
    /// Base currency for consolidated reporting (None = local only).
    pub base_currency: Option<Currency>,
    /// Max gap for LOCF FX rate lookup.
    pub fx_max_gap_days: u32,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            vol_window: 21,
            risk_free_rate: Decimal::ZERO,
            var_confidence_levels: (Decimal::from(95), Decimal::from(99)),
            base_currency: None,
            fx_max_gap_days: 5,
        }
    }
}

impl PerformanceConfig {
    /// Create config with base currency for multi-currency reporting.
    pub fn with_base_currency(mut self, currency: Currency) -> Self {
        self.base_currency = Some(currency);
        self
    }
}

/// Performance Engine - orchestrates all performance tracking.
pub struct PerformanceEngine {
    ledger: TradeLedger,
    attribution: AttributionEngine,
    risk_calc: RiskCalculator,
    config: PerformanceConfig,
    /// Historical equity curve for drawdown/VaR
    equity_curve: Vec<(NaiveDate, Decimal)>,
    /// Historical returns for volatility/Sharpe
    daily_returns: Vec<Decimal>,
    /// Initial capital
    initial_capital: Decimal,
    /// Optional FX provider for multi-currency support
    fx_provider: Option<Arc<dyn FxRateProvider>>,
    /// FX attribution engine (created on demand)
    fx_attribution: Option<FxAttributionEngine>,
    /// Track values by currency for attribution
    values_by_currency: BTreeMap<NaiveDate, BTreeMap<Currency, Decimal>>,
}

impl std::fmt::Debug for PerformanceEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PerformanceEngine")
            .field("config", &self.config)
            .field("equity_curve_len", &self.equity_curve.len())
            .field("initial_capital", &self.initial_capital)
            .field("has_fx_provider", &self.fx_provider.is_some())
            .finish()
    }
}

impl PerformanceEngine {
    pub fn new(config: PerformanceConfig, initial_capital: Decimal) -> Self {
        let risk_calc = RiskCalculator::new(config.vol_window, config.risk_free_rate);
        let fx_attribution = config.base_currency.map(FxAttributionEngine::new);
        
        Self {
            ledger: TradeLedger::new(),
            attribution: AttributionEngine::new(),
            risk_calc,
            config,
            equity_curve: Vec::new(),
            daily_returns: Vec::new(),
            initial_capital,
            fx_provider: None,
            fx_attribution,
            values_by_currency: BTreeMap::new(),
        }
    }
    
    /// Create engine with FX provider for multi-currency support.
    pub fn with_fx(
        config: PerformanceConfig,
        initial_capital: Decimal,
        fx_provider: Arc<dyn FxRateProvider>,
    ) -> Self {
        let risk_calc = RiskCalculator::new(config.vol_window, config.risk_free_rate);
        let fx_attribution = config.base_currency.map(|c| {
            FxAttributionEngine::with_max_gap(c, config.fx_max_gap_days)
        });
        
        Self {
            ledger: TradeLedger::new(),
            attribution: AttributionEngine::new(),
            risk_calc,
            config,
            equity_curve: Vec::new(),
            daily_returns: Vec::new(),
            initial_capital,
            fx_provider: Some(fx_provider),
            fx_attribution,
            values_by_currency: BTreeMap::new(),
        }
    }
    
    /// Set FX provider after construction.
    pub fn set_fx_provider(&mut self, provider: Arc<dyn FxRateProvider>) {
        self.fx_provider = Some(provider);
        if let Some(base) = self.config.base_currency {
            self.fx_attribution = Some(FxAttributionEngine::with_max_gap(
                base,
                self.config.fx_max_gap_days,
            ));
        }
    }
    
    /// Check if FX conversion is available.
    pub fn has_fx_support(&self) -> bool {
        self.fx_provider.is_some() && self.config.base_currency.is_some()
    }

    /// Record a buy trade.
    pub fn record_buy(
        &mut self,
        date: NaiveDate,
        symbol: &str,
        shares: i64,
        price: Decimal,
        cost: Decimal,
        market: Market,
    ) {
        self.ledger.record_buy(date, symbol, shares, price, cost, market);
    }

    /// Record a sell trade and return realized P&L.
    pub fn record_sell(
        &mut self,
        date: NaiveDate,
        symbol: &str,
        shares: i64,
        price: Decimal,
        cost: Decimal,
        market: Market,
    ) -> Decimal {
        self.ledger.record_sell(date, symbol, shares, price, cost, market)
    }

    /// Record technique weights for attribution.
    pub fn record_entry_weights(&mut self, symbol: &str, weights: BTreeMap<String, Decimal>) {
        self.attribution.record_entry_weights(symbol, weights);
    }

    /// Generate a performance snapshot at a point in time.
    pub fn generate_snapshot(
        &mut self,
        date: NaiveDate,
        cash: Decimal,
        prices: &BTreeMap<String, Decimal>,
    ) -> PerformanceSnapshot {
        let pnl = self.ledger.get_pnl_breakdown(prices);
        let costs = self.ledger.costs().clone();
        let mut exposure = self.risk_calc.calculate_exposure(self.ledger.positions(), prices);
        
        let market_value = self.ledger.market_value(prices);
        let equity = cash + market_value;

        // Update equity curve
        self.equity_curve.push((date, equity));

        // Calculate daily return
        if self.equity_curve.len() >= 2 {
            let prev = self.equity_curve[self.equity_curve.len() - 2].1;
            if !prev.is_zero() {
                let ret = (equity - prev) / prev;
                self.daily_returns.push(ret);
            }
        }

        // Calculate drawdown
        let equity_values: Vec<Decimal> = self.equity_curve.iter().map(|(_, e)| *e).collect();
        let drawdown = self.risk_calc.calculate_drawdown(&equity_values);

        // Calculate turnover from recent trades
        let turnover = self.calculate_turnover(date, equity);

        // FX conversion if available
        let (base_currency, equity_base, cash_base, fx_rates_used) = 
            self.calculate_base_currency_values(date, cash, &exposure, prices);
        
        // Update exposure by_currency
        self.populate_currency_exposure(&mut exposure, prices);
        
        // Store values by currency for attribution
        self.store_values_by_currency(date, cash, prices);

        PerformanceSnapshot {
            date,
            equity,
            cash,
            base_currency,
            equity_base,
            cash_base,
            fx_rates_used,
            exposure,
            pnl,
            costs,
            drawdown,
            turnover,
        }
    }
    
    /// Calculate base currency values with FX conversion.
    fn calculate_base_currency_values(
        &self,
        date: NaiveDate,
        cash: Decimal,
        exposure: &super::ExposureBreakdown,
        prices: &BTreeMap<String, Decimal>,
    ) -> (Option<Currency>, Option<Decimal>, Option<Decimal>, Option<BTreeMap<String, FxRateInfo>>) {
        let base = match self.config.base_currency {
            Some(b) => b,
            None => return (None, None, None, None),
        };
        
        let provider = match &self.fx_provider {
            Some(p) => p,
            None => return (Some(base), None, None, None),
        };
        
        let mut fx_rates_used = BTreeMap::new();
        let mut total_equity_base = Decimal::ZERO;
        let mut cash_base = Decimal::ZERO;
        
        // Convert cash (assume it's in base currency for now)
        // TODO: Support multi-currency cash buckets
        cash_base = cash;
        total_equity_base += cash;
        
        // Convert positions by currency
        for (symbol, lot) in self.ledger.positions() {
            let currency = lot.market.currency();
            if let Some(&price) = prices.get(symbol) {
                let local_value = price * Decimal::from(lot.shares);
                
                if currency == base {
                    total_equity_base += local_value;
                } else {
                    let pair = FxPair::new(currency, base);
                    match provider.get_rate_locf(date, pair, self.config.fx_max_gap_days) {
                        Ok(rate) => {
                            let base_value = local_value * rate.rate;
                            total_equity_base += base_value;
                            
                            // Determine if LOCF was used
                            let used_locf = rate.date != date;
                            let method = if used_locf {
                                crate::performance::FxResolutionMethod::LOCF
                            } else {
                                crate::performance::FxResolutionMethod::Direct
                            };
                            
                            fx_rates_used.entry(currency.code().to_string()).or_insert(FxRateInfo::new(
                                pair.as_str(),
                                date,
                                pair.as_str(),
                                rate.date,
                                rate.rate,
                                method,
                            ));
                        }
                        Err(_) => {
                            // Fallback: use local value (no conversion)
                            total_equity_base += local_value;
                        }
                    }
                }
            }
        }
        
        (
            Some(base),
            Some(total_equity_base),
            Some(cash_base),
            if fx_rates_used.is_empty() { None } else { Some(fx_rates_used) },
        )
    }
    
    /// Populate exposure breakdown by currency.
    fn populate_currency_exposure(
        &self,
        exposure: &mut super::ExposureBreakdown,
        prices: &BTreeMap<String, Decimal>,
    ) {
        let mut by_currency: BTreeMap<String, Decimal> = BTreeMap::new();
        
        for (symbol, lot) in self.ledger.positions() {
            let currency = lot.market.currency();
            if let Some(&price) = prices.get(symbol) {
                let value = price * Decimal::from(lot.shares);
                *by_currency.entry(currency.code().to_string()).or_default() += value;
            }
        }
        
        exposure.by_currency = by_currency;
        
        // Convert to base if FX available
        if let (Some(base), Some(provider)) = (self.config.base_currency, &self.fx_provider) {
            let mut by_currency_base = BTreeMap::new();
            
            for (currency_str, &value) in &exposure.by_currency {
                if let Some(currency) = Currency::from_str(currency_str) {
                    if currency == base {
                        by_currency_base.insert(currency_str.clone(), value);
                    } else {
                        let pair = FxPair::new(currency, base);
                        let date = self.equity_curve.last().map(|(d, _)| *d)
                            .unwrap_or_else(|| chrono::Local::now().date_naive());
                        
                        if let Ok(rate) = provider.get_rate_locf(date, pair, self.config.fx_max_gap_days) {
                            by_currency_base.insert(currency_str.clone(), value * rate.rate);
                        }
                    }
                }
            }
            
            exposure.by_currency_base = by_currency_base;
        }
    }
    
    /// Store values by currency for FX attribution.
    fn store_values_by_currency(
        &mut self,
        date: NaiveDate,
        cash: Decimal,
        prices: &BTreeMap<String, Decimal>,
    ) {
        let mut values: BTreeMap<Currency, Decimal> = BTreeMap::new();
        
        // Assume cash is in base currency (or default to BRL)
        let cash_currency = self.config.base_currency.unwrap_or(Currency::BRL);
        *values.entry(cash_currency).or_default() += cash;
        
        // Add position values
        for (symbol, lot) in self.ledger.positions() {
            let currency = lot.market.currency();
            if let Some(&price) = prices.get(symbol) {
                let value = price * Decimal::from(lot.shares);
                *values.entry(currency).or_default() += value;
            }
        }
        
        self.values_by_currency.insert(date, values);
    }
    
    /// Generate FX attribution for a period.
    pub fn generate_fx_attribution(
        &self,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Result<FxAttributionBreakdown, FxError> {
        let fx_engine = self.fx_attribution.as_ref()
            .ok_or_else(|| FxError::NoConversionPath {
                from: Currency::USD,
                to: Currency::BRL,
            })?;
        
        let provider = self.fx_provider.as_ref()
            .ok_or_else(|| FxError::NoConversionPath {
                from: Currency::USD,
                to: Currency::BRL,
            })?;
        
        let values_start = self.values_by_currency.get(&start_date)
            .cloned()
            .unwrap_or_default();
        let values_end = self.values_by_currency.get(&end_date)
            .cloned()
            .unwrap_or_default();
        
        fx_engine.calculate_period_attribution(
            &values_start,
            &values_end,
            provider.as_ref(),
            start_date,
            end_date,
        )
    }

    /// Generate attribution breakdown.
    pub fn generate_attribution(&self, prices: &BTreeMap<String, Decimal>) -> AttributionBreakdown {
        let pnl_by_symbol = self.ledger.get_unrealized_by_symbol(prices);
        self.attribution.calculate_attribution(&pnl_by_symbol)
    }

    /// Generate risk metrics.
    pub fn generate_risk_metrics(&self, portfolio_value: Decimal) -> (VolatilityMetrics, VaRMetrics) {
        let vol = self.risk_calc.calculate_volatility(&self.daily_returns);
        let var = self.risk_calc.calculate_var(&self.daily_returns, portfolio_value);
        (vol, var)
    }

    /// Generate CIO-level view.
    pub fn generate_cio_view(&self, date: NaiveDate, prices: &BTreeMap<String, Decimal>, cash: Decimal) -> CIOView {
        let market_value = self.ledger.market_value(prices);
        let equity = cash + market_value;
        
        let equity_values: Vec<Decimal> = self.equity_curve.iter().map(|(_, e)| *e).collect();
        let drawdown = self.risk_calc.calculate_drawdown(&equity_values);
        
        let total_return_pct = if self.initial_capital.is_zero() {
            Decimal::ZERO
        } else {
            (equity - self.initial_capital) / self.initial_capital * Decimal::from(100)
        };

        // Annualized return (simple approximation)
        let days = self.equity_curve.len() as u32;
        let annualized_return_pct = if days > 0 {
            total_return_pct * Decimal::from(252) / Decimal::from(days)
        } else {
            Decimal::ZERO
        };

        let sharpe = self.risk_calc.calculate_sharpe(&self.daily_returns);
        let var = self.risk_calc.calculate_var(&self.daily_returns, equity);

        // FX attribution if available
        let (total_return_base_pct, asset_return_pct, fx_return_pct, interaction_pct) = 
            self.calculate_fx_metrics_for_cio(date);

        CIOView {
            date,
            total_return_pct,
            annualized_return_pct,
            max_drawdown_pct: drawdown.max_dd * Decimal::from(100),
            sharpe_ratio: sharpe,
            total_costs: self.ledger.costs().total,
            turnover_pct: self.calculate_turnover(date, equity).turnover_pct,
            var_95: var.var_95,
            positions_count: self.ledger.positions().len() as u32,
            base_currency: self.config.base_currency,
            total_return_base_pct,
            asset_return_pct,
            fx_return_pct,
            interaction_pct,
        }
    }
    
    /// Calculate FX metrics for CIO view.
    fn calculate_fx_metrics_for_cio(
        &self,
        end_date: NaiveDate,
    ) -> (Option<Decimal>, Option<Decimal>, Option<Decimal>, Option<Decimal>) {
        if !self.has_fx_support() || self.equity_curve.len() < 2 {
            return (None, None, None, None);
        }
        
        let start_date = self.equity_curve.first().map(|(d, _)| *d).unwrap();
        
        match self.generate_fx_attribution(start_date, end_date) {
            Ok(attr) => (
                Some(attr.portfolio_total_return_base * Decimal::from(100)),
                Some(attr.portfolio_asset_return * Decimal::from(100)),
                Some(attr.portfolio_fx_return * Decimal::from(100)),
                Some(attr.portfolio_interaction * Decimal::from(100)),
            ),
            Err(_) => (None, None, None, None),
        }
    }

    fn calculate_turnover(&self, _date: NaiveDate, portfolio_value: Decimal) -> TurnoverMetrics {
        let trades = self.ledger.trades();
        let buy_notional: Decimal = trades.iter()
            .filter(|t| matches!(t.side, super::TradeSide::Buy))
            .map(|t| t.price * Decimal::from(t.shares))
            .sum();
        let sell_notional: Decimal = trades.iter()
            .filter(|t| matches!(t.side, super::TradeSide::Sell))
            .map(|t| t.price * Decimal::from(t.shares))
            .sum();
        
        TurnoverMetrics::from_notionals(buy_notional, sell_notional, portfolio_value)
    }

    /// Get underlying ledger.
    pub fn ledger(&self) -> &TradeLedger {
        &self.ledger
    }

    /// Get equity curve.
    pub fn equity_curve(&self) -> &[(NaiveDate, Decimal)] {
        &self.equity_curve
    }

    /// Get daily returns.
    pub fn daily_returns(&self) -> &[Decimal] {
        &self.daily_returns
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn make_prices(data: &[(&str, Decimal)]) -> BTreeMap<String, Decimal> {
        data.iter().map(|(s, p)| (s.to_string(), *p)).collect()
    }

    #[test]
    fn test_snapshot_generation() {
        let mut engine = PerformanceEngine::new(PerformanceConfig::default(), dec!(100000));
        let date = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();

        engine.record_buy(date, "PETR4", 100, dec!(30), dec!(10), Market::BR);
        
        let prices = make_prices(&[("PETR4", dec!(35))]);
        let snapshot = engine.generate_snapshot(date, dec!(97000), &prices);

        // Equity = cash + market_value = 97000 + (100*35) = 100500
        assert_eq!(snapshot.equity, dec!(100500));
        assert_eq!(snapshot.pnl.unrealized, dec!(500)); // (35-30)*100
    }

    #[test]
    fn test_attribution_integration() {
        let mut engine = PerformanceEngine::new(PerformanceConfig::default(), dec!(100000));
        let date = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();

        engine.record_buy(date, "PETR4", 100, dec!(30), dec!(10), Market::BR);
        engine.record_entry_weights("PETR4", [
            ("momentum".to_string(), dec!(0.6)),
            ("value".to_string(), dec!(0.4)),
        ].into());

        let prices = make_prices(&[("PETR4", dec!(35))]);
        let attr = engine.generate_attribution(&prices);

        assert_eq!(attr.total_pnl, dec!(500));
        assert!(attr.by_technique.iter().any(|t| t.technique_name == "momentum"));
    }

    #[test]
    fn test_cio_view() {
        let mut engine = PerformanceEngine::new(PerformanceConfig::default(), dec!(100000));
        let date1 = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
        let date2 = NaiveDate::from_ymd_opt(2025, 1, 2).unwrap();

        engine.record_buy(date1, "PETR4", 100, dec!(30), dec!(10), Market::BR);
        
        let prices = make_prices(&[("PETR4", dec!(35))]);
        engine.generate_snapshot(date1, dec!(97000), &prices);
        engine.generate_snapshot(date2, dec!(97000), &prices);

        let cio = engine.generate_cio_view(date2, &prices, dec!(97000));

        assert!(cio.total_return_pct > Decimal::ZERO);
        assert_eq!(cio.positions_count, 1);
    }

    #[test]
    fn test_equity_curve_tracking() {
        let mut engine = PerformanceEngine::new(PerformanceConfig::default(), dec!(100000));
        
        for i in 1..=5 {
            let date = NaiveDate::from_ymd_opt(2025, 1, i).unwrap();
            let prices = make_prices(&[]);
            engine.generate_snapshot(date, dec!(100000) + Decimal::from(i * 100), &prices);
        }

        assert_eq!(engine.equity_curve().len(), 5);
        assert_eq!(engine.daily_returns().len(), 4); // n-1 returns
    }
}


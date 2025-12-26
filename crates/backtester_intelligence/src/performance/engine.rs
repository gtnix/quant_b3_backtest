//! Performance Engine - Orchestrator for snapshot generation.

use chrono::NaiveDate;
use rust_decimal::Decimal;
use std::collections::BTreeMap;

use super::{
    TradeLedger, AttributionEngine, RiskCalculator,
    PerformanceSnapshot, TurnoverMetrics, AttributionBreakdown, VaRMetrics,
    VolatilityMetrics, CIOView,
};
use crate::filters::Market;

/// Configuration for the performance engine.
#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    pub vol_window: u32,
    pub risk_free_rate: Decimal,
    pub var_confidence_levels: (Decimal, Decimal), // (95%, 99%)
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            vol_window: 21,
            risk_free_rate: Decimal::ZERO,
            var_confidence_levels: (Decimal::from(95), Decimal::from(99)),
        }
    }
}

/// Performance Engine - orchestrates all performance tracking.
#[derive(Debug)]
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
}

impl PerformanceEngine {
    pub fn new(config: PerformanceConfig, initial_capital: Decimal) -> Self {
        let risk_calc = RiskCalculator::new(config.vol_window, config.risk_free_rate);
        Self {
            ledger: TradeLedger::new(),
            attribution: AttributionEngine::new(),
            risk_calc,
            config,
            equity_curve: Vec::new(),
            daily_returns: Vec::new(),
            initial_capital,
        }
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
        let exposure = self.risk_calc.calculate_exposure(self.ledger.positions(), prices);
        
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

        PerformanceSnapshot {
            date,
            equity,
            cash,
            exposure,
            pnl,
            costs,
            drawdown,
            turnover,
        }
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


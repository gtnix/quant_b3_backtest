//! Performance tracking types for the backtester.
//!
//! All types are immutable and serializable for AI consumption.

use chrono::NaiveDate;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::currency::Currency;
use crate::filters::Market;

/// Complete snapshot of portfolio performance at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    pub date: NaiveDate,
    
    // Local currency values (unchanged for backward compatibility)
    pub equity: Decimal,
    pub cash: Decimal,
    
    // Base currency consolidated view (NEW)
    pub base_currency: Option<Currency>,
    pub equity_base: Option<Decimal>,
    pub cash_base: Option<Decimal>,
    
    // FX audit trail (NEW)
    pub fx_rates_used: Option<BTreeMap<String, FxRateInfo>>,
    
    pub exposure: ExposureBreakdown,
    pub pnl: PnLBreakdown,
    pub costs: CostBreakdown,
    pub drawdown: DrawdownMetrics,
    pub turnover: TurnoverMetrics,
}

impl PerformanceSnapshot {
    /// Check if this snapshot has base currency conversion data.
    pub fn has_fx_data(&self) -> bool {
        self.base_currency.is_some() && self.equity_base.is_some()
    }
    
    /// Get equity in base currency if available, otherwise local.
    pub fn equity_consolidated(&self) -> Decimal {
        self.equity_base.unwrap_or(self.equity)
    }
}

// =============================================================================
// FX RESOLUTION METHOD
// =============================================================================

/// Method used to resolve an FX rate for audit trail purposes.
///
/// This enum tracks how a rate was obtained, which is critical for:
/// - Audit/compliance: Know exactly how each rate was sourced
/// - Debugging: Understand why a particular rate was used
/// - Transparency: Reports show when LOCF or inverse was applied
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum FxResolutionMethod {
    /// Rate found directly for the requested pair and date.
    #[default]
    Direct,
    /// Rate obtained by inverting the stored pair (e.g., BRL/USD from USD/BRL).
    Inverse,
    /// Last Observation Carried Forward - rate from earlier date used.
    LOCF,
    /// Both inverse and LOCF were applied.
    InverseLOCF,
    /// Same currency (e.g., USD to USD), rate = 1.
    Identity,
    /// Rate calculated via cross-rate (e.g., EUR/BRL via EUR/USD × USD/BRL).
    /// Reserved for V2.
    Triangulated,
}

impl FxResolutionMethod {
    /// Check if this method used LOCF (carried forward from earlier date).
    pub fn used_locf(&self) -> bool {
        matches!(self, FxResolutionMethod::LOCF | FxResolutionMethod::InverseLOCF)
    }
    
    /// Check if this method used an inverse calculation.
    pub fn used_inverse(&self) -> bool {
        matches!(self, FxResolutionMethod::Inverse | FxResolutionMethod::InverseLOCF)
    }
    
    /// Human-readable description.
    pub fn describe(&self) -> &'static str {
        match self {
            FxResolutionMethod::Direct => "Direct rate lookup",
            FxResolutionMethod::Inverse => "Inverse rate calculated",
            FxResolutionMethod::LOCF => "Last observation carried forward",
            FxResolutionMethod::InverseLOCF => "Inverse + LOCF",
            FxResolutionMethod::Identity => "Same currency (rate = 1)",
            FxResolutionMethod::Triangulated => "Cross-rate triangulated",
        }
    }
}

impl std::fmt::Display for FxResolutionMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

// =============================================================================
// FX RATE INFO (AUDIT TRAIL)
// =============================================================================

/// FX rate information for audit trail.
///
/// Records complete details of how an FX rate was resolved for a conversion,
/// enabling full audit capability and debugging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FxRateInfo {
    // What was requested
    /// The currency pair that was originally requested (e.g., "USD/BRL").
    pub pair_requested: String,
    /// The date for which the rate was requested.
    pub date_requested: NaiveDate,
    
    // What was resolved
    /// The currency pair that was actually used (may differ if inverse was used).
    pub pair_resolved: String,
    /// The date of the actual rate observation (may differ if LOCF was used).
    pub date_resolved: NaiveDate,
    /// The exchange rate used.
    pub rate: Decimal,
    
    // How it was resolved
    /// Method used to resolve the rate.
    pub method: FxResolutionMethod,
    
    // Legacy fields for backward compatibility
    /// The date of the rate (alias for date_resolved).
    #[serde(skip_serializing)]
    pub rate_date: NaiveDate,
    /// The currency pair (alias for pair_resolved).
    #[serde(skip_serializing)]
    pub pair: String,
    /// Whether an inverse rate was used (derived from method).
    #[serde(skip_serializing)]
    pub used_inverse: bool,
}

impl FxRateInfo {
    /// Create a new FxRateInfo with full audit details.
    pub fn new(
        pair_requested: String,
        date_requested: NaiveDate,
        pair_resolved: String,
        date_resolved: NaiveDate,
        rate: Decimal,
        method: FxResolutionMethod,
    ) -> Self {
        Self {
            pair_requested,
            date_requested,
            pair_resolved: pair_resolved.clone(),
            date_resolved,
            rate,
            method,
            // Legacy compatibility
            rate_date: date_resolved,
            pair: pair_resolved,
            used_inverse: method.used_inverse(),
        }
    }
    
    /// Create a simple FxRateInfo for backward compatibility.
    pub fn simple(
        pair: String,
        rate: Decimal,
        rate_date: NaiveDate,
        used_inverse: bool,
    ) -> Self {
        let method = if used_inverse {
            FxResolutionMethod::Inverse
        } else {
            FxResolutionMethod::Direct
        };
        
        Self {
            pair_requested: pair.clone(),
            date_requested: rate_date,
            pair_resolved: pair.clone(),
            date_resolved: rate_date,
            rate,
            method,
            rate_date,
            pair,
            used_inverse,
        }
    }
    
    /// Create an identity rate info (same currency, rate = 1).
    pub fn identity(currency_code: &str, date: NaiveDate) -> Self {
        let pair = format!("{}/{}", currency_code, currency_code);
        Self {
            pair_requested: pair.clone(),
            date_requested: date,
            pair_resolved: pair.clone(),
            date_resolved: date,
            rate: Decimal::ONE,
            method: FxResolutionMethod::Identity,
            rate_date: date,
            pair,
            used_inverse: false,
        }
    }
    
    /// Check if LOCF was used (rate from an earlier date).
    pub fn is_locf(&self) -> bool {
        self.method.used_locf()
    }
    
    /// Get the gap in days between requested and resolved dates.
    pub fn gap_days(&self) -> i64 {
        (self.date_requested - self.date_resolved).num_days()
    }
}

/// Breakdown of P&L into realized and unrealized components.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PnLBreakdown {
    pub realized: Decimal,
    pub unrealized: Decimal,
    pub total: Decimal,
    pub by_market: BTreeMap<String, Decimal>,
    pub by_symbol: BTreeMap<String, Decimal>,
}

impl PnLBreakdown {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_values(realized: Decimal, unrealized: Decimal) -> Self {
        Self {
            realized,
            unrealized,
            total: realized + unrealized,
            by_market: BTreeMap::new(),
            by_symbol: BTreeMap::new(),
        }
    }
}

/// Breakdown of trading costs by market and type.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostBreakdown {
    pub fees_br: Decimal,
    pub fees_us: Decimal,
    pub slippage_br: Decimal,
    pub slippage_us: Decimal,
    pub total: Decimal,
}

impl CostBreakdown {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn compute_total(&mut self) {
        self.total = self.fees_br + self.fees_us + self.slippage_br + self.slippage_us;
    }

    pub fn add(&mut self, other: &CostBreakdown) {
        self.fees_br += other.fees_br;
        self.fees_us += other.fees_us;
        self.slippage_br += other.slippage_br;
        self.slippage_us += other.slippage_us;
        self.total += other.total;
    }
}

/// Portfolio exposure breakdown.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExposureBreakdown {
    pub gross: Decimal,
    pub net: Decimal,
    pub long: Decimal,
    pub short: Decimal,
    pub by_market: BTreeMap<String, Decimal>,
    
    // NEW: Currency-based exposure breakdown
    /// Exposure by currency in local currency values.
    pub by_currency: BTreeMap<String, Decimal>,
    /// Exposure by currency converted to base currency.
    pub by_currency_base: BTreeMap<String, Decimal>,
}

impl ExposureBreakdown {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_positions(long: Decimal, short: Decimal) -> Self {
        Self {
            gross: long + short.abs(),
            net: long - short.abs(),
            long,
            short,
            by_market: BTreeMap::new(),
            by_currency: BTreeMap::new(),
            by_currency_base: BTreeMap::new(),
        }
    }
    
    /// Get exposure for a specific currency.
    pub fn currency_exposure(&self, currency: &str) -> Decimal {
        self.by_currency.get(currency).copied().unwrap_or(Decimal::ZERO)
    }
    
    /// Get exposure for a currency in base currency.
    pub fn currency_exposure_base(&self, currency: &str) -> Decimal {
        self.by_currency_base.get(currency).copied().unwrap_or(Decimal::ZERO)
    }
}

/// Drawdown metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DrawdownMetrics {
    pub current_dd: Decimal,
    pub max_dd: Decimal,
    pub dd_duration_days: u32,
    pub hwm: Decimal, // High-water mark
}

impl DrawdownMetrics {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Turnover metrics for a period.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TurnoverMetrics {
    pub buy_notional: Decimal,
    pub sell_notional: Decimal,
    pub turnover_pct: Decimal,
}

impl TurnoverMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_notionals(buy: Decimal, sell: Decimal, portfolio_value: Decimal) -> Self {
        let turnover_pct = if portfolio_value.is_zero() {
            Decimal::ZERO
        } else {
            (buy + sell) / portfolio_value * Decimal::from(100)
        };
        Self {
            buy_notional: buy,
            sell_notional: sell,
            turnover_pct,
        }
    }
}

/// Position with WAP cost basis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionLot {
    pub symbol: String,
    pub shares: i64,
    pub wap_cost_basis: Decimal,
    pub market: Market,
    pub entry_date: NaiveDate,
}

impl PositionLot {
    pub fn new(symbol: String, shares: i64, cost_basis: Decimal, market: Market, entry_date: NaiveDate) -> Self {
        Self {
            symbol,
            shares,
            wap_cost_basis: cost_basis,
            market,
            entry_date,
        }
    }

    pub fn market_value(&self, current_price: Decimal) -> Decimal {
        current_price * Decimal::from(self.shares)
    }

    pub fn unrealized_pnl(&self, current_price: Decimal) -> Decimal {
        (current_price - self.wap_cost_basis) * Decimal::from(self.shares)
    }
}

/// A recorded trade in the ledger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    pub date: NaiveDate,
    pub symbol: String,
    pub side: TradeSide,
    pub shares: i64,
    pub price: Decimal,
    pub cost: Decimal,
    pub market: Market,
    pub realized_pnl: Option<Decimal>,
}

/// Trade side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Volatility metrics for risk calculation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VolatilityMetrics {
    pub daily_vol: Decimal,
    pub annualized_vol: Decimal,
    pub rolling_window: u32,
}

impl VolatilityMetrics {
    pub fn from_daily(daily: Decimal, window: u32) -> Self {
        // Annualize: daily * sqrt(252)
        let sqrt_252 = Decimal::from_str_exact("15.87").unwrap_or(Decimal::from(16));
        Self {
            daily_vol: daily,
            annualized_vol: daily * sqrt_252,
            rolling_window: window,
        }
    }
}

/// Value at Risk metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VaRMetrics {
    pub var_95: Decimal,
    pub var_99: Decimal,
    pub method: VaRMethod,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum VaRMethod {
    #[default]
    Historical,
    Parametric,
}

/// Attribution of P&L to a technique.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechniqueAttribution {
    pub technique_name: String,
    pub weight_pct: Decimal,
    pub pnl_contribution: Decimal,
    pub return_contribution: Decimal,
}

/// Full attribution breakdown.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AttributionBreakdown {
    pub by_technique: Vec<TechniqueAttribution>,
    pub total_pnl: Decimal,
    pub residual: Decimal,
}

impl AttributionBreakdown {
    pub fn new() -> Self {
        Self::default()
    }
}

/// High-level CIO view metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CIOView {
    pub date: NaiveDate,
    pub total_return_pct: Decimal,
    pub annualized_return_pct: Decimal,
    pub max_drawdown_pct: Decimal,
    pub sharpe_ratio: Decimal,
    pub total_costs: Decimal,
    pub turnover_pct: Decimal,
    pub var_95: Decimal,
    pub positions_count: u32,
    
    // NEW: Multi-currency fields
    /// Base currency for reporting.
    pub base_currency: Option<Currency>,
    /// Total return in base currency.
    pub total_return_base_pct: Option<Decimal>,
    /// Asset return component (local currency performance).
    pub asset_return_pct: Option<Decimal>,
    /// FX return component.
    pub fx_return_pct: Option<Decimal>,
    /// Interaction component (asset * fx).
    pub interaction_pct: Option<Decimal>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_pnl_breakdown_total() {
        let pnl = PnLBreakdown::with_values(dec!(1000), dec!(500));
        assert_eq!(pnl.total, dec!(1500));
    }

    #[test]
    fn test_cost_breakdown_add() {
        let mut c1 = CostBreakdown {
            fees_br: dec!(100),
            fees_us: dec!(50),
            slippage_br: dec!(20),
            slippage_us: dec!(10),
            total: dec!(180),
        };
        let c2 = CostBreakdown {
            fees_br: dec!(50),
            fees_us: dec!(25),
            slippage_br: dec!(10),
            slippage_us: dec!(5),
            total: dec!(90),
        };
        c1.add(&c2);
        assert_eq!(c1.total, dec!(270));
    }

    #[test]
    fn test_exposure_from_positions() {
        let exp = ExposureBreakdown::from_positions(dec!(10000), dec!(2000));
        assert_eq!(exp.gross, dec!(12000));
        assert_eq!(exp.net, dec!(8000));
    }

    #[test]
    fn test_turnover_pct() {
        let t = TurnoverMetrics::from_notionals(dec!(5000), dec!(5000), dec!(100000));
        assert_eq!(t.turnover_pct, dec!(10));
    }

    #[test]
    fn test_position_lot_unrealized() {
        let lot = PositionLot::new(
            "PETR4".into(),
            100,
            dec!(30),
            Market::BR,
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
        );
        let unrealized = lot.unrealized_pnl(dec!(35));
        assert_eq!(unrealized, dec!(500));
    }

    #[test]
    fn test_serde_roundtrip() {
        let pnl = PnLBreakdown::with_values(dec!(100), dec!(200));
        let json = serde_json::to_string(&pnl).unwrap();
        let parsed: PnLBreakdown = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.total, dec!(300));
    }
}








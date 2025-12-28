//! Performance Reporter - Human and AI-readable outputs.
//!
//! Supports multi-currency reporting with FX attribution.
//! Includes sector exposure, concentration metrics, and regime analysis.

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use super::{
    PerformanceSnapshot, AttributionBreakdown, CIOView, VolatilityMetrics, VaRMetrics,
    FxAttributionBreakdown, ConcentrationMetrics, RegimeSummary, SectorExposure,
};

/// Schema version for the performance report JSON.
/// Increment this when making breaking changes to the JSON structure.
///
/// Version history:
/// - fx_report_v1.0: Initial FX support
/// - fx_report_v1.1: Added FX audit trail fields
/// - fx_report_v1.2: Added sector_exposure, concentration, regime_summary
/// - fx_report_v1.3: Added compliance (constraints & risk controls)
pub const PERFORMANCE_REPORT_SCHEMA_VERSION: &str = "fx_report_v1.3";

/// Full performance report in AI-friendly JSON format.
///
/// # Schema Version
///
/// The `schema_version` field identifies the report format version.
/// This enables consumers to handle different versions appropriately.
///
/// Current version: `fx_report_v1.3` (added compliance/risk controls)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    /// Schema version for this report format.
    /// Enables backward-compatible parsing by consumers.
    pub schema_version: String,
    
    pub date: String,
    pub equity: String,
    pub return_pct: String,
    pub drawdown_pct: String,
    pub pnl: PnLJson,
    pub costs: CostsJson,
    pub attribution: Vec<AttributionJson>,
    pub risk: RiskJson,
    pub exposure: ExposureJson,
    pub turnover: TurnoverJson,
    
    // FX / Multi-currency fields (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_currency: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub equity_base: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fx_attribution: Option<FxAttributionJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exposure_by_currency: Option<Vec<CurrencyExposureJson>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fx_rates_used: Option<Vec<FxRateUsedJson>>,
    
    // Research-grade fields (NEW in v1.2)
    /// Sector exposure breakdown (NEW in v1.2).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sector_exposure: Option<Vec<SectorExposureJson>>,
    
    /// Portfolio concentration metrics (NEW in v1.2).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub concentration: Option<ConcentrationJson>,
    
    /// Regime analysis summary (NEW in v1.2).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regime_summary: Option<RegimeSummaryJson>,
    
    // Compliance / Risk Controls (NEW in v1.3)
    /// Compliance status and breach log (NEW in v1.3).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compliance: Option<ComplianceJson>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PnLJson {
    pub realized: String,
    pub unrealized: String,
    pub total: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostsJson {
    pub total: String,
    pub fees: String,
    pub slippage: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributionJson {
    pub technique: String,
    pub weight_pct: String,
    pub pnl: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskJson {
    pub vol_ann: String,
    pub var_95: String,
    pub sharpe: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExposureJson {
    pub gross: String,
    pub net: String,
    pub by_market: Vec<MarketExposure>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketExposure {
    pub market: String,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnoverJson {
    pub buy: String,
    pub sell: String,
    pub pct: String,
}

/// FX attribution in JSON format.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FxAttributionJson {
    /// Base currency code.
    pub base_currency: String,
    /// Asset return contribution (%).
    pub asset_return_pct: String,
    /// FX return contribution (%).
    pub fx_return_pct: String,
    /// Interaction term (%).
    pub interaction_pct: String,
    /// Total return in base currency (%).
    pub total_return_base_pct: String,
    /// Attribution by currency.
    pub by_currency: Vec<CurrencyAttributionJson>,
}

/// Currency-level attribution in JSON format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurrencyAttributionJson {
    pub currency: String,
    pub asset_return_pct: String,
    pub fx_return_pct: String,
    pub interaction_pct: String,
    pub weight_pct: String,
}

/// Currency exposure in JSON format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurrencyExposureJson {
    pub currency: String,
    pub value_local: String,
    pub value_base: String,
    pub weight_pct: String,
}

/// FX rate used in JSON format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FxRateUsedJson {
    // Request details
    /// The currency pair that was requested (e.g., "USD/BRL").
    pub pair_requested: String,
    /// The date for which the rate was requested.
    pub date_requested: String,
    
    // Resolution details
    /// The currency pair that was actually used (may differ if inverse).
    pub pair_resolved: String,
    /// The date of the actual rate observation (may differ if LOCF).
    pub date_resolved: String,
    /// The exchange rate used.
    pub rate: String,
    /// Method used to resolve the rate (Direct, Inverse, LOCF, etc.).
    pub method: String,
    
    // Legacy fields (for backward compatibility, aliases)
    /// The currency pair (alias for pair_resolved).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pair: Option<String>,
    /// The date (alias for date_resolved).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub date: Option<String>,
}

// =============================================================================
// SECTOR EXPOSURE JSON (NEW in v1.2)
// =============================================================================

/// Sector exposure in JSON format.
///
/// Provides complete breakdown of exposure by sector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectorExposureJson {
    /// Sector name (e.g., "Energy", "Financials", "Unknown").
    pub sector: String,
    /// Gross exposure in this sector.
    pub gross: String,
    /// Net exposure in this sector.
    pub net: String,
    /// Long exposure in this sector.
    pub long: String,
    /// Short exposure in this sector.
    pub short: String,
    /// Weight as percentage of total portfolio gross.
    pub weight_pct: String,
}

impl SectorExposureJson {
    /// Create from SectorExposure.
    pub fn from_exposure(exp: &SectorExposure, format_fn: impl Fn(Decimal) -> String) -> Self {
        Self {
            sector: exp.sector.clone(),
            gross: format_fn(exp.gross),
            net: format_fn(exp.net),
            long: format_fn(exp.long),
            short: format_fn(exp.short),
            weight_pct: format_fn(exp.weight_pct),
        }
    }
}

// =============================================================================
// CONCENTRATION JSON (NEW in v1.2)
// =============================================================================

/// Portfolio concentration metrics in JSON format.
///
/// All metrics use absolute weights (shorts count positively).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcentrationJson {
    /// Herfindahl-Hirschman Index (sum of squared weights).
    /// Range: [0, 1] where 1 = single position.
    pub hhi: String,
    /// Effective number of positions = 1 / HHI.
    pub effective_n: String,
    /// Weight of largest position (%).
    pub top_1_weight_pct: String,
    /// Sum of top 5 position weights (%).
    pub top_5_weight_pct: String,
    /// Sum of top 10 position weights (%).
    pub top_10_weight_pct: String,
    /// Maximum position weight (%) - same as top_1.
    pub max_position_weight_pct: String,
    /// Number of positions.
    pub n_positions: u32,
    /// Gini coefficient (0 = equality, 1 = inequality).
    /// Optional due to computation cost.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gini: Option<String>,
    /// Human-readable concentration level.
    pub concentration_level: String,
}

impl ConcentrationJson {
    /// Create from ConcentrationMetrics.
    pub fn from_metrics(m: &ConcentrationMetrics, format_fn: impl Fn(Decimal) -> String) -> Self {
        Self {
            hhi: format_fn(m.hhi),
            effective_n: format_fn(m.effective_n),
            top_1_weight_pct: format_fn(m.top_1_weight_pct),
            top_5_weight_pct: format_fn(m.top_5_weight_pct),
            top_10_weight_pct: format_fn(m.top_10_weight_pct),
            max_position_weight_pct: format_fn(m.max_position_weight_pct),
            n_positions: m.n_positions,
            gini: m.gini.map(&format_fn),
            concentration_level: m.concentration_level().to_string(),
        }
    }
}

// =============================================================================
// REGIME SUMMARY JSON (NEW in v1.2)
// =============================================================================

/// Regime detection configuration in JSON format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeConfigJson {
    pub trend_lookback: u32,
    pub trend_threshold: String,
    pub vol_lookback: u32,
    pub vol_quantile_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub benchmark_symbol: Option<String>,
}

/// Performance by regime in JSON format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimePerformanceJson {
    /// Trend state: "Uptrend", "Downtrend", or "Sideways".
    pub trend_state: String,
    /// Volatility quantile: "Q1" through "Q5".
    pub vol_quantile: String,
    /// Number of days in this regime.
    pub day_count: u32,
    /// Mean daily return (%).
    pub mean_return_pct: String,
    /// Cumulative return (%).
    pub cumulative_return_pct: String,
    /// Win rate (% of days with positive return).
    pub win_rate_pct: String,
    /// Mean daily turnover (%).
    pub mean_turnover_pct: String,
    /// Mean daily cost (%).
    pub mean_cost_pct: String,
}

/// Complete regime summary in JSON format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeSummaryJson {
    /// Configuration used for regime detection.
    pub config: RegimeConfigJson,
    /// Performance breakdown by regime.
    pub by_regime: Vec<RegimePerformanceJson>,
    /// Total days in backtest.
    pub total_days: u32,
    /// Days in warmup period (insufficient data).
    pub warmup_days: u32,
    /// Path to detailed regime tags CSV (if externalized).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags_file: Option<String>,
}

impl RegimeSummaryJson {
    /// Create from RegimeSummary.
    pub fn from_summary(s: &RegimeSummary, format_fn: impl Fn(Decimal) -> String) -> Self {
        Self {
            config: RegimeConfigJson {
                trend_lookback: s.config.trend_lookback,
                trend_threshold: format_fn(s.config.trend_threshold),
                vol_lookback: s.config.vol_lookback,
                vol_quantile_count: s.config.vol_quantile_count,
                benchmark_symbol: s.config.benchmark_symbol.clone(),
            },
            by_regime: s.by_regime.iter().map(|p| RegimePerformanceJson {
                trend_state: p.trend_state.as_str().to_string(),
                vol_quantile: format!("{}", p.vol_quantile),
                day_count: p.day_count,
                mean_return_pct: format_fn(p.mean_return_pct),
                cumulative_return_pct: format_fn(p.cumulative_return_pct),
                win_rate_pct: format_fn(p.win_rate_pct),
                mean_turnover_pct: format_fn(p.mean_turnover_pct),
                mean_cost_pct: format_fn(p.mean_cost_pct),
            }).collect(),
            total_days: s.total_days,
            warmup_days: s.warmup_days,
            tags_file: None,
        }
    }
}

// =============================================================================
// COMPLIANCE JSON (NEW in v1.3)
// =============================================================================

use super::compliance::{BreachEvent, ComplianceReport, ComplianceSummary};
use super::constraints::ConstraintsConfig;
use std::collections::BTreeMap;

/// Constraint policy snapshot in JSON format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintPolicyJson {
    /// Soft threshold (if any).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub soft: Option<String>,
    /// Hard threshold.
    pub hard: String,
    /// Action on breach.
    pub action: String,
}

/// Breach event in JSON format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreachEventJson {
    /// Date of breach.
    pub date: String,
    /// Constraint identifier.
    pub constraint_id: String,
    /// Scope of breach (e.g., "Portfolio", "Sector:Energy").
    pub scope: String,
    /// Measured value that exceeded limit.
    pub measured_value: String,
    /// Limit that was exceeded.
    pub limit_value: String,
    /// Absolute magnitude of breach.
    pub magnitude: String,
    /// Magnitude as percentage of limit.
    pub magnitude_pct: String,
    /// Severity level.
    pub severity: String,
    /// Action taken in response.
    pub action_taken: String,
    /// Whether this was ex-ante (pre-order) or ex-post (EOD).
    pub is_ex_ante: bool,
    /// Evidence supporting the breach.
    pub evidence: BreachEvidenceJson,
}

/// Breach evidence in JSON format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreachEvidenceJson {
    /// Top contributors (name, value) pairs.
    pub top_contributors: Vec<Vec<String>>,
    /// Context description.
    pub context: String,
}

/// Action record in JSON format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionRecordJson {
    /// Date of action.
    pub date: String,
    /// Action type.
    pub action: String,
    /// Reason for action.
    pub reason: String,
}

/// Compliance summary in JSON format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceSummaryJson {
    /// Total number of breaches.
    pub total_breaches: usize,
    /// Breaches by severity level.
    pub breaches_by_severity: BTreeMap<String, usize>,
    /// Worst severity encountered.
    pub worst_severity: String,
    /// Number of days with breaches.
    pub days_out_of_limit: usize,
    /// List of violated constraint IDs.
    pub constraints_violated: Vec<String>,
    /// Number of ex-ante breaches.
    pub ex_ante_breaches: usize,
    /// Number of ex-post breaches.
    pub ex_post_breaches: usize,
}

/// Complete compliance report in JSON format (NEW in v1.3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceJson {
    /// Configuration snapshot showing active limits.
    pub config_snapshot: BTreeMap<String, ConstraintPolicyJson>,
    /// Summary statistics.
    pub summary: ComplianceSummaryJson,
    /// Top breaches (sorted by severity then magnitude).
    pub breaches: Vec<BreachEventJson>,
    /// Actions taken during the backtest.
    pub actions_taken: Vec<ActionRecordJson>,
}

impl ComplianceJson {
    /// Create from ComplianceReport.
    pub fn from_report(report: &ComplianceReport, format_fn: impl Fn(Decimal) -> String + Copy) -> Self {
        Self {
            config_snapshot: report.config_snapshot.iter().map(|(k, v)| {
                (k.clone(), ConstraintPolicyJson {
                    soft: v.soft.clone(),
                    hard: v.hard.clone(),
                    action: v.action.clone(),
                })
            }).collect(),
            summary: ComplianceSummaryJson {
                total_breaches: report.summary.total_breaches,
                breaches_by_severity: report.summary.breaches_by_severity.clone(),
                worst_severity: report.summary.worst_severity.clone(),
                days_out_of_limit: report.summary.days_out_of_limit,
                constraints_violated: report.summary.constraints_violated.clone(),
                ex_ante_breaches: report.summary.ex_ante_breaches,
                ex_post_breaches: report.summary.ex_post_breaches,
            },
            breaches: report.breaches.iter().map(|b| Self::breach_to_json(b, format_fn)).collect(),
            actions_taken: report.actions_taken.iter().map(|a| ActionRecordJson {
                date: a.date.format("%Y-%m-%d").to_string(),
                action: format!("{}", a.action),
                reason: a.reason.clone(),
            }).collect(),
        }
    }

    /// Convert a single breach event to JSON.
    fn breach_to_json(b: &BreachEvent, format_fn: impl Fn(Decimal) -> String) -> BreachEventJson {
        BreachEventJson {
            date: b.date.format("%Y-%m-%d").to_string(),
            constraint_id: format!("{}", b.constraint_id),
            scope: format!("{}", b.scope),
            measured_value: format_fn(b.measured_value),
            limit_value: format_fn(b.limit_value),
            magnitude: format_fn(b.magnitude),
            magnitude_pct: format_fn(b.magnitude_pct),
            severity: format!("{}", b.severity),
            action_taken: format!("{}", b.action_taken),
            is_ex_ante: b.is_ex_ante,
            evidence: BreachEvidenceJson {
                top_contributors: b.evidence.top_contributors.iter()
                    .map(|(name, val)| vec![name.clone(), format_fn(*val)])
                    .collect(),
                context: b.evidence.context.clone(),
            },
        }
    }

    /// Create config snapshot from ConstraintsConfig.
    pub fn config_to_snapshot(config: &ConstraintsConfig, format_fn: impl Fn(Decimal) -> String) -> BTreeMap<String, ConstraintPolicyJson> {
        let mut snapshot = BTreeMap::new();

        if let Some(p) = &config.max_gross_exposure_pct {
            snapshot.insert("max_gross_exposure_pct".to_string(), ConstraintPolicyJson {
                soft: p.soft_threshold.map(&format_fn),
                hard: format_fn(p.hard_threshold),
                action: format!("{}", p.action),
            });
        }
        if let Some(p) = &config.max_net_exposure_pct {
            snapshot.insert("max_net_exposure_pct".to_string(), ConstraintPolicyJson {
                soft: p.soft_threshold.map(&format_fn),
                hard: format_fn(p.hard_threshold),
                action: format!("{}", p.action),
            });
        }
        if let Some(p) = &config.max_single_name_weight_pct {
            snapshot.insert("max_single_name_weight_pct".to_string(), ConstraintPolicyJson {
                soft: p.soft_threshold.map(&format_fn),
                hard: format_fn(p.hard_threshold),
                action: format!("{}", p.action),
            });
        }
        if let Some(p) = &config.max_top5_weight_pct {
            snapshot.insert("max_top5_weight_pct".to_string(), ConstraintPolicyJson {
                soft: p.soft_threshold.map(&format_fn),
                hard: format_fn(p.hard_threshold),
                action: format!("{}", p.action),
            });
        }
        if let Some(p) = &config.max_hhi {
            snapshot.insert("max_hhi".to_string(), ConstraintPolicyJson {
                soft: p.soft_threshold.map(&format_fn),
                hard: format_fn(p.hard_threshold),
                action: format!("{}", p.action),
            });
        }
        if let Some(p) = &config.max_sector_weight_pct {
            snapshot.insert("max_sector_weight_pct".to_string(), ConstraintPolicyJson {
                soft: p.soft_threshold.map(&format_fn),
                hard: format_fn(p.hard_threshold),
                action: format!("{}", p.action),
            });
        }
        if let Some(p) = &config.max_unknown_sector_weight_pct {
            snapshot.insert("max_unknown_sector_weight_pct".to_string(), ConstraintPolicyJson {
                soft: p.soft_threshold.map(&format_fn),
                hard: format_fn(p.hard_threshold),
                action: format!("{}", p.action),
            });
        }
        if let Some(p) = &config.max_turnover_pct_per_period {
            snapshot.insert("max_turnover_pct_per_period".to_string(), ConstraintPolicyJson {
                soft: p.soft_threshold.map(&format_fn),
                hard: format_fn(p.hard_threshold),
                action: format!("{}", p.action),
            });
        }
        if let Some(p) = &config.max_cost_pct_per_period {
            snapshot.insert("max_cost_pct_per_period".to_string(), ConstraintPolicyJson {
                soft: p.soft_threshold.map(&format_fn),
                hard: format_fn(p.hard_threshold),
                action: format!("{}", p.action),
            });
        }

        snapshot
    }
}

/// Performance reporter for generating outputs.
#[derive(Debug, Clone)]
pub struct PerformanceReporter {
    decimal_places: u32,
}

impl PerformanceReporter {
    pub fn new(decimal_places: u32) -> Self {
        Self { decimal_places }
    }

    fn format_decimal(&self, d: Decimal) -> String {
        format!("{:.2}", d)
    }

    fn format_pct(&self, d: Decimal) -> String {
        format!("{:.2}", d)
    }

    /// Generate human-readable summary.
    pub fn to_summary(
        &self,
        snapshot: &PerformanceSnapshot,
        attribution: &AttributionBreakdown,
        vol: &VolatilityMetrics,
        var: &VaRMetrics,
        sharpe: Decimal,
        initial_capital: Decimal,
    ) -> String {
        let return_pct = if initial_capital.is_zero() {
            Decimal::ZERO
        } else {
            (snapshot.equity - initial_capital) / initial_capital * Decimal::from(100)
        };

        let mut lines = Vec::new();
        
        lines.push(format!("PERFORMANCE SNAPSHOT {}", snapshot.date));
        lines.push("=".repeat(50));
        lines.push(format!("Equity: {}", self.format_decimal(snapshot.equity)));
        lines.push(format!("Cash: {}", self.format_decimal(snapshot.cash)));
        lines.push(format!("Return: {}%", self.format_pct(return_pct)));
        lines.push(format!("Drawdown: {}%", self.format_pct(snapshot.drawdown.current_dd * Decimal::from(100))));
        lines.push(format!("Max Drawdown: {}%", self.format_pct(snapshot.drawdown.max_dd * Decimal::from(100))));
        lines.push(String::new());
        
        lines.push("P&L BREAKDOWN".to_string());
        lines.push("-".repeat(30));
        lines.push(format!("Realized: {}", self.format_decimal(snapshot.pnl.realized)));
        lines.push(format!("Unrealized: {}", self.format_decimal(snapshot.pnl.unrealized)));
        lines.push(format!("Total: {}", self.format_decimal(snapshot.pnl.total)));
        lines.push(String::new());

        lines.push("COSTS".to_string());
        lines.push("-".repeat(30));
        lines.push(format!("BR Fees: {}", self.format_decimal(snapshot.costs.fees_br)));
        lines.push(format!("US Fees: {}", self.format_decimal(snapshot.costs.fees_us)));
        lines.push(format!("BR Slippage: {}", self.format_decimal(snapshot.costs.slippage_br)));
        lines.push(format!("US Slippage: {}", self.format_decimal(snapshot.costs.slippage_us)));
        lines.push(format!("Total: {}", self.format_decimal(snapshot.costs.total)));
        lines.push(String::new());

        lines.push("ATTRIBUTION".to_string());
        lines.push("-".repeat(30));
        // Sort by technique name for determinism
        let mut sorted_attr: Vec<_> = attribution.by_technique.iter().collect();
        sorted_attr.sort_by(|a, b| a.technique_name.cmp(&b.technique_name));
        for attr in sorted_attr {
            lines.push(format!(
                "{}: {} ({}%)",
                attr.technique_name,
                self.format_decimal(attr.pnl_contribution),
                self.format_pct(attr.weight_pct)
            ));
        }
        if !attribution.residual.is_zero() {
            lines.push(format!("Residual: {}", self.format_decimal(attribution.residual)));
        }
        lines.push(String::new());

        lines.push("RISK METRICS".to_string());
        lines.push("-".repeat(30));
        lines.push(format!("Daily Vol: {}%", self.format_pct(vol.daily_vol * Decimal::from(100))));
        lines.push(format!("Ann Vol: {}%", self.format_pct(vol.annualized_vol * Decimal::from(100))));
        lines.push(format!("VaR 95%: {}", self.format_decimal(var.var_95)));
        lines.push(format!("VaR 99%: {}", self.format_decimal(var.var_99)));
        lines.push(format!("Sharpe: {}", self.format_decimal(sharpe)));
        lines.push(String::new());

        lines.push("EXPOSURE".to_string());
        lines.push("-".repeat(30));
        lines.push(format!("Gross: {}", self.format_decimal(snapshot.exposure.gross)));
        lines.push(format!("Net: {}", self.format_decimal(snapshot.exposure.net)));
        // Sort market keys for determinism
        let mut markets: Vec<_> = snapshot.exposure.by_market.iter().collect();
        markets.sort_by_key(|(k, _)| *k);
        for (market, value) in markets {
            lines.push(format!("{}: {}", market, self.format_decimal(*value)));
        }
        lines.push(String::new());

        lines.push("TURNOVER".to_string());
        lines.push("-".repeat(30));
        lines.push(format!("Buy: {}", self.format_decimal(snapshot.turnover.buy_notional)));
        lines.push(format!("Sell: {}", self.format_decimal(snapshot.turnover.sell_notional)));
        lines.push(format!("Turnover: {}%", self.format_pct(snapshot.turnover.turnover_pct)));

        lines.join("\n")
    }

    /// Generate AI-readable JSON report.
    pub fn to_json(
        &self,
        snapshot: &PerformanceSnapshot,
        attribution: &AttributionBreakdown,
        vol: &VolatilityMetrics,
        var: &VaRMetrics,
        sharpe: Decimal,
        initial_capital: Decimal,
    ) -> PerformanceReport {
        self.to_json_with_fx(snapshot, attribution, vol, var, sharpe, initial_capital, None)
    }
    
    /// Generate AI-readable JSON report with FX attribution.
    pub fn to_json_with_fx(
        &self,
        snapshot: &PerformanceSnapshot,
        attribution: &AttributionBreakdown,
        vol: &VolatilityMetrics,
        var: &VaRMetrics,
        sharpe: Decimal,
        initial_capital: Decimal,
        fx_attribution: Option<&FxAttributionBreakdown>,
    ) -> PerformanceReport {
        self.to_json_full(
            snapshot,
            attribution,
            vol,
            var,
            sharpe,
            initial_capital,
            fx_attribution,
            None,
            None,
        )
    }

    /// Generate full research-grade JSON report.
    ///
    /// Includes all optional fields: FX attribution, sector exposure,
    /// concentration metrics, and regime summary.
    pub fn to_json_full(
        &self,
        snapshot: &PerformanceSnapshot,
        attribution: &AttributionBreakdown,
        vol: &VolatilityMetrics,
        var: &VaRMetrics,
        sharpe: Decimal,
        initial_capital: Decimal,
        fx_attribution: Option<&FxAttributionBreakdown>,
        concentration: Option<&ConcentrationMetrics>,
        regime_summary: Option<&RegimeSummary>,
    ) -> PerformanceReport {
        let return_pct = if initial_capital.is_zero() {
            Decimal::ZERO
        } else {
            (snapshot.equity - initial_capital) / initial_capital * Decimal::from(100)
        };

        // Sort attribution for determinism
        let mut sorted_attr: Vec<_> = attribution.by_technique.iter().collect();
        sorted_attr.sort_by(|a, b| a.technique_name.cmp(&b.technique_name));

        let attr_json: Vec<AttributionJson> = sorted_attr.iter()
            .map(|a| AttributionJson {
                technique: a.technique_name.clone(),
                weight_pct: self.format_pct(a.weight_pct),
                pnl: self.format_decimal(a.pnl_contribution),
            })
            .collect();

        // Sort market exposure for determinism
        let mut markets: Vec<_> = snapshot.exposure.by_market.iter().collect();
        markets.sort_by_key(|(k, _)| *k);

        // FX fields
        let base_currency = snapshot.base_currency.map(|c| c.code().to_string());
        let equity_base = snapshot.equity_base.map(|e| self.format_decimal(e));
        
        // Currency exposure
        let exposure_by_currency = if !snapshot.exposure.by_currency.is_empty() {
            let total_base: Decimal = snapshot.exposure.by_currency_base.values().sum();
            
            let mut exposures: Vec<_> = snapshot.exposure.by_currency.iter().collect();
            exposures.sort_by_key(|(k, _)| *k);
            
            Some(exposures.iter().map(|(currency, value)| {
                let value_base = snapshot.exposure.by_currency_base
                    .get(*currency)
                    .copied()
                    .unwrap_or(**value);
                let weight = if total_base.is_zero() {
                    Decimal::ZERO
                } else {
                    value_base / total_base * Decimal::from(100)
                };
                CurrencyExposureJson {
                    currency: currency.to_string(),
                    value_local: self.format_decimal(**value),
                    value_base: self.format_decimal(value_base),
                    weight_pct: self.format_pct(weight),
                }
            }).collect())
        } else {
            None
        };
        
        // FX rates used
        let fx_rates_used = snapshot.fx_rates_used.as_ref().map(|rates| {
            let mut rate_list: Vec<_> = rates.iter().collect();
            rate_list.sort_by_key(|(k, _)| *k);
            rate_list.iter().map(|(_, info)| {
                FxRateUsedJson {
                    pair_requested: info.pair_requested.clone(),
                    date_requested: info.date_requested.to_string(),
                    pair_resolved: info.pair_resolved.clone(),
                    date_resolved: info.date_resolved.to_string(),
                    rate: self.format_decimal(info.rate),
                    method: info.method.to_string(),
                    // Legacy aliases for backward compatibility
                    pair: None,
                    date: None,
                }
            }).collect()
        });
        
        // FX attribution
        let fx_attr_json = fx_attribution.map(|fa| {
            let by_currency: Vec<CurrencyAttributionJson> = fa.by_currency.iter()
                .map(|ca| CurrencyAttributionJson {
                    currency: ca.currency.code().to_string(),
                    asset_return_pct: self.format_pct(ca.asset_return * Decimal::from(100)),
                    fx_return_pct: self.format_pct(ca.fx_return * Decimal::from(100)),
                    interaction_pct: self.format_pct(ca.interaction * Decimal::from(100)),
                    weight_pct: self.format_pct(ca.weight_pct),
                })
                .collect();
            
            FxAttributionJson {
                base_currency: fa.base_currency.code().to_string(),
                asset_return_pct: self.format_pct(fa.portfolio_asset_return * Decimal::from(100)),
                fx_return_pct: self.format_pct(fa.portfolio_fx_return * Decimal::from(100)),
                interaction_pct: self.format_pct(fa.portfolio_interaction * Decimal::from(100)),
                total_return_base_pct: self.format_pct(fa.portfolio_total_return_base * Decimal::from(100)),
                by_currency,
            }
        });

        // Sector exposure
        let sector_exposure = if !snapshot.exposure.by_sector.is_empty() {
            Some(snapshot.exposure.by_sector.iter()
                .map(|s| SectorExposureJson::from_exposure(s, |d| self.format_decimal(d)))
                .collect())
        } else {
            None
        };

        // Concentration metrics
        let concentration_json = concentration.map(|c| {
            ConcentrationJson::from_metrics(c, |d| self.format_decimal(d))
        });

        // Regime summary
        let regime_summary_json = regime_summary.map(|r| {
            RegimeSummaryJson::from_summary(r, |d| self.format_decimal(d))
        });

        PerformanceReport {
            schema_version: PERFORMANCE_REPORT_SCHEMA_VERSION.to_string(),
            date: snapshot.date.to_string(),
            equity: self.format_decimal(snapshot.equity),
            return_pct: self.format_pct(return_pct),
            drawdown_pct: self.format_pct(snapshot.drawdown.current_dd * Decimal::from(100)),
            pnl: PnLJson {
                realized: self.format_decimal(snapshot.pnl.realized),
                unrealized: self.format_decimal(snapshot.pnl.unrealized),
                total: self.format_decimal(snapshot.pnl.total),
            },
            costs: CostsJson {
                total: self.format_decimal(snapshot.costs.total),
                fees: self.format_decimal(snapshot.costs.fees_br + snapshot.costs.fees_us),
                slippage: self.format_decimal(snapshot.costs.slippage_br + snapshot.costs.slippage_us),
            },
            attribution: attr_json,
            risk: RiskJson {
                vol_ann: self.format_pct(vol.annualized_vol * Decimal::from(100)),
                var_95: self.format_decimal(var.var_95),
                sharpe: self.format_decimal(sharpe),
            },
            exposure: ExposureJson {
                gross: self.format_decimal(snapshot.exposure.gross),
                net: self.format_decimal(snapshot.exposure.net),
                by_market: markets.iter()
                    .map(|(k, v)| MarketExposure {
                        market: k.to_string(),
                        value: self.format_decimal(**v),
                    })
                    .collect(),
            },
            turnover: TurnoverJson {
                buy: self.format_decimal(snapshot.turnover.buy_notional),
                sell: self.format_decimal(snapshot.turnover.sell_notional),
                pct: self.format_pct(snapshot.turnover.turnover_pct),
            },
            base_currency,
            equity_base,
            fx_attribution: fx_attr_json,
            exposure_by_currency,
            fx_rates_used,
            sector_exposure,
            concentration: concentration_json,
            regime_summary: regime_summary_json,
            compliance: None,
        }
    }

    /// Generate compact JSON string for API consumption.
    pub fn to_json_string(
        &self,
        snapshot: &PerformanceSnapshot,
        attribution: &AttributionBreakdown,
        vol: &VolatilityMetrics,
        var: &VaRMetrics,
        sharpe: Decimal,
        initial_capital: Decimal,
    ) -> String {
        let report = self.to_json(snapshot, attribution, vol, var, sharpe, initial_capital);
        serde_json::to_string_pretty(&report).unwrap_or_else(|_| "{}".to_string())
    }

    /// Generate CIO view summary.
    pub fn cio_summary(&self, cio: &CIOView) -> String {
        let mut lines = Vec::new();
        
        lines.push(format!("CIO VIEW {}", cio.date));
        lines.push("=".repeat(40));
        lines.push(format!("Total Return: {}%", self.format_pct(cio.total_return_pct)));
        lines.push(format!("Annualized Return: {}%", self.format_pct(cio.annualized_return_pct)));
        lines.push(format!("Max Drawdown: {}%", self.format_pct(cio.max_drawdown_pct)));
        lines.push(format!("Sharpe Ratio: {}", self.format_decimal(cio.sharpe_ratio)));
        lines.push(format!("VaR 95%: {}", self.format_decimal(cio.var_95)));
        lines.push(format!("Total Costs: {}", self.format_decimal(cio.total_costs)));
        lines.push(format!("Turnover: {}%", self.format_pct(cio.turnover_pct)));
        lines.push(format!("Positions: {}", cio.positions_count));
        
        // FX attribution if available
        if let Some(base) = cio.base_currency {
            lines.push(String::new());
            lines.push(format!("FX ATTRIBUTION (Base: {})", base.code()));
            lines.push("-".repeat(40));
            
            if let Some(total_base) = cio.total_return_base_pct {
                lines.push(format!("Total Return ({}): {}%", base.code(), self.format_pct(total_base)));
            }
            if let Some(asset) = cio.asset_return_pct {
                lines.push(format!("Asset Return: {}%", self.format_pct(asset)));
            }
            if let Some(fx) = cio.fx_return_pct {
                lines.push(format!("FX Return: {}%", self.format_pct(fx)));
            }
            if let Some(interaction) = cio.interaction_pct {
                lines.push(format!("Interaction: {}%", self.format_pct(interaction)));
            }
        }

        lines.join("\n")
    }
}

impl Default for PerformanceReporter {
    fn default() -> Self {
        Self::new(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use rust_decimal_macros::dec;
    use crate::performance::{
        PnLBreakdown, CostBreakdown, ExposureBreakdown, DrawdownMetrics, TurnoverMetrics,
        TechniqueAttribution,
    };

    fn make_snapshot() -> PerformanceSnapshot {
        PerformanceSnapshot {
            date: NaiveDate::from_ymd_opt(2025, 1, 15).unwrap(),
            equity: dec!(105000),
            cash: dec!(50000),
            base_currency: None,
            equity_base: None,
            cash_base: None,
            fx_rates_used: None,
            exposure: ExposureBreakdown {
                gross: dec!(55000),
                net: dec!(55000),
                long: dec!(55000),
                short: Decimal::ZERO,
                by_market: [("BR".to_string(), dec!(30000)), ("US".to_string(), dec!(25000))].into(),
                by_currency: Default::default(),
                by_currency_base: Default::default(),
                by_sector: Default::default(),
            },
            pnl: PnLBreakdown {
                realized: dec!(3000),
                unrealized: dec!(2000),
                total: dec!(5000),
                by_market: Default::default(),
                by_symbol: Default::default(),
            },
            costs: CostBreakdown {
                fees_br: dec!(100),
                fees_us: dec!(50),
                slippage_br: dec!(20),
                slippage_us: dec!(10),
                total: dec!(180),
            },
            drawdown: DrawdownMetrics {
                current_dd: dec!(0.02),
                max_dd: dec!(0.05),
                dd_duration_days: 3,
                hwm: dec!(107000),
            },
            turnover: TurnoverMetrics {
                buy_notional: dec!(10000),
                sell_notional: dec!(8000),
                turnover_pct: dec!(17.1),
            },
        }
    }

    fn make_attribution() -> AttributionBreakdown {
        AttributionBreakdown {
            by_technique: vec![
                TechniqueAttribution {
                    technique_name: "momentum".to_string(),
                    weight_pct: dec!(40),
                    pnl_contribution: dec!(2000),
                    return_contribution: dec!(40),
                },
                TechniqueAttribution {
                    technique_name: "value".to_string(),
                    weight_pct: dec!(30),
                    pnl_contribution: dec!(1500),
                    return_contribution: dec!(30),
                },
            ],
            total_pnl: dec!(5000),
            residual: dec!(1500),
        }
    }

    #[test]
    fn test_to_summary() {
        let reporter = PerformanceReporter::default();
        let snapshot = make_snapshot();
        let attr = make_attribution();
        let vol = VolatilityMetrics::from_daily(dec!(0.015), 21);
        let var = VaRMetrics {
            var_95: dec!(-2500),
            var_99: dec!(-4000),
            method: super::super::VaRMethod::Historical,
        };

        let summary = reporter.to_summary(&snapshot, &attr, &vol, &var, dec!(1.25), dec!(100000));

        assert!(summary.contains("PERFORMANCE SNAPSHOT"));
        assert!(summary.contains("Equity: 105000.00"));
        assert!(summary.contains("momentum:"));
        assert!(summary.contains("VaR 95%:"));
    }

    #[test]
    fn test_to_json() {
        let reporter = PerformanceReporter::default();
        let snapshot = make_snapshot();
        let attr = make_attribution();
        let vol = VolatilityMetrics::from_daily(dec!(0.015), 21);
        let var = VaRMetrics {
            var_95: dec!(-2500),
            var_99: dec!(-4000),
            method: super::super::VaRMethod::Historical,
        };

        let report = reporter.to_json(&snapshot, &attr, &vol, &var, dec!(1.25), dec!(100000));

        assert_eq!(report.date, "2025-01-15");
        assert_eq!(report.equity, "105000.00");
        assert_eq!(report.pnl.total, "5000.00");
        assert_eq!(report.attribution.len(), 2);
    }

    #[test]
    fn test_json_string_parses() {
        let reporter = PerformanceReporter::default();
        let snapshot = make_snapshot();
        let attr = make_attribution();
        let vol = VolatilityMetrics::default();
        let var = VaRMetrics::default();

        let json = reporter.to_json_string(&snapshot, &attr, &vol, &var, dec!(1.0), dec!(100000));
        
        // Verify it's valid JSON
        let parsed: Result<PerformanceReport, _> = serde_json::from_str(&json);
        assert!(parsed.is_ok());
    }

    #[test]
    fn test_deterministic_output() {
        let reporter = PerformanceReporter::default();
        let snapshot = make_snapshot();
        let attr = make_attribution();
        let vol = VolatilityMetrics::default();
        let var = VaRMetrics::default();

        let json1 = reporter.to_json_string(&snapshot, &attr, &vol, &var, dec!(1.0), dec!(100000));
        let json2 = reporter.to_json_string(&snapshot, &attr, &vol, &var, dec!(1.0), dec!(100000));

        assert_eq!(json1, json2);
    }

    #[test]
    fn test_cio_summary() {
        let reporter = PerformanceReporter::default();
        let cio = CIOView {
            date: NaiveDate::from_ymd_opt(2025, 1, 15).unwrap(),
            total_return_pct: dec!(5),
            annualized_return_pct: dec!(12),
            max_drawdown_pct: dec!(5),
            sharpe_ratio: dec!(1.25),
            total_costs: dec!(180),
            turnover_pct: dec!(17.1),
            var_95: dec!(-2500),
            positions_count: 10,
            base_currency: None,
            total_return_base_pct: None,
            asset_return_pct: None,
            fx_return_pct: None,
            interaction_pct: None,
        };

        let summary = reporter.cio_summary(&cio);

        assert!(summary.contains("CIO VIEW"));
        assert!(summary.contains("Total Return: 5.00%"));
        assert!(summary.contains("Sharpe Ratio: 1.25"));
    }
}


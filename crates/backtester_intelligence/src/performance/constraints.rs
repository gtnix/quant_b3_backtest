//! Portfolio Constraints & Risk Controls.
//!
//! Provides ex-ante and ex-post validation of portfolio constraints:
//! - Exposure/Leverage limits
//! - Concentration limits (single name, top-N, HHI)
//! - Sector exposure limits
//! - Currency exposure limits
//! - Turnover and cost limits
//!
//! # Design Decisions
//!
//! - **Weight definition**: `abs(value) / gross_exposure` (consistent with concentration module)
//! - **Sector weight**: Gross-based (Unknown sector = positions without mapping)
//! - **Magnitude sign**: Always positive (overshoot amount)
//! - **Severity mapping**: Soft=WARN, Hard=CRIT, action=Halt→HALT

use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

use crate::filters::Market;
use crate::monitoring::Severity;

use super::compliance::{BreachEvent, BreachEvidence};
use super::{ConcentrationMetrics, ExposureBreakdown, SectorExposure};

// =============================================================================
// CONSTRAINT IDENTIFIERS
// =============================================================================

/// Stable identifier for each constraint type.
///
/// Used for logging, reporting, and configuration lookup.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", content = "params")]
pub enum ConstraintId {
    // Exposure/Leverage
    MaxGrossExposurePct,
    MaxNetExposurePct,
    MaxLeverage,
    // Concentration
    MaxSingleNameWeightPct,
    MaxTop5WeightPct,
    MaxHHI,
    // Sector
    MaxSectorWeightPct { sector: String },
    MaxUnknownSectorWeightPct,
    // Currency
    MaxCurrencyWeightPct { currency: String },
    MaxFxReturnContributionPct,
    // Turnover/Costs
    MaxTurnoverPctPerPeriod,
    MaxCostPctPerPeriod,
}

impl fmt::Display for ConstraintId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConstraintId::MaxGrossExposurePct => write!(f, "MaxGrossExposurePct"),
            ConstraintId::MaxNetExposurePct => write!(f, "MaxNetExposurePct"),
            ConstraintId::MaxLeverage => write!(f, "MaxLeverage"),
            ConstraintId::MaxSingleNameWeightPct => write!(f, "MaxSingleNameWeightPct"),
            ConstraintId::MaxTop5WeightPct => write!(f, "MaxTop5WeightPct"),
            ConstraintId::MaxHHI => write!(f, "MaxHHI"),
            ConstraintId::MaxSectorWeightPct { sector } => {
                write!(f, "MaxSectorWeightPct:{}", sector)
            }
            ConstraintId::MaxUnknownSectorWeightPct => write!(f, "MaxUnknownSectorWeightPct"),
            ConstraintId::MaxCurrencyWeightPct { currency } => {
                write!(f, "MaxCurrencyWeightPct:{}", currency)
            }
            ConstraintId::MaxFxReturnContributionPct => write!(f, "MaxFxReturnContributionPct"),
            ConstraintId::MaxTurnoverPctPerPeriod => write!(f, "MaxTurnoverPctPerPeriod"),
            ConstraintId::MaxCostPctPerPeriod => write!(f, "MaxCostPctPerPeriod"),
        }
    }
}

// =============================================================================
// CONSTRAINT SCOPE
// =============================================================================

/// Scope of a constraint violation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintScope {
    /// Portfolio-level constraint
    Portfolio,
    /// Single symbol constraint
    Symbol(String),
    /// Sector-level constraint
    Sector(String),
    /// Currency-level constraint
    Currency(String),
    /// Market-level constraint
    Market(Market),
}

impl fmt::Display for ConstraintScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConstraintScope::Portfolio => write!(f, "Portfolio"),
            ConstraintScope::Symbol(s) => write!(f, "Symbol:{}", s),
            ConstraintScope::Sector(s) => write!(f, "Sector:{}", s),
            ConstraintScope::Currency(c) => write!(f, "Currency:{}", c),
            ConstraintScope::Market(m) => write!(f, "Market:{:?}", m),
        }
    }
}

// =============================================================================
// CONSTRAINT POLICY
// =============================================================================

/// Action to take when a constraint is violated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ConstraintAction {
    /// Log only, no blocking (WARN level)
    #[default]
    LogOnly,
    /// Block new BUY orders (CRIT level)
    BlockNewTrades,
    /// Generate SELL orders to reduce exposure (future implementation)
    ReduceExposure,
    /// Full circuit breaker halt (HALT level)
    Halt,
}

impl fmt::Display for ConstraintAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConstraintAction::LogOnly => write!(f, "LogOnly"),
            ConstraintAction::BlockNewTrades => write!(f, "BlockNewTrades"),
            ConstraintAction::ReduceExposure => write!(f, "ReduceExposure"),
            ConstraintAction::Halt => write!(f, "Halt"),
        }
    }
}

impl ConstraintAction {
    /// Map action to severity level.
    pub fn to_severity(&self) -> Severity {
        match self {
            ConstraintAction::LogOnly => Severity::Warn,
            ConstraintAction::BlockNewTrades => Severity::Crit,
            ConstraintAction::ReduceExposure => Severity::Crit,
            ConstraintAction::Halt => Severity::Halt,
        }
    }
}

/// Policy for a single constraint.
///
/// Defines soft (warning) and hard (action) thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintPolicy {
    /// Soft threshold - triggers WARN when exceeded
    #[serde(skip_serializing_if = "Option::is_none")]
    pub soft_threshold: Option<Decimal>,
    /// Hard threshold - triggers action when exceeded
    pub hard_threshold: Decimal,
    /// Action to take when hard threshold is exceeded
    #[serde(default)]
    pub action: ConstraintAction,
}

impl ConstraintPolicy {
    /// Create a new policy with only hard threshold.
    pub fn hard(threshold: Decimal, action: ConstraintAction) -> Self {
        Self {
            soft_threshold: None,
            hard_threshold: threshold,
            action,
        }
    }

    /// Create a policy with both soft and hard thresholds.
    pub fn soft_hard(soft: Decimal, hard: Decimal, action: ConstraintAction) -> Self {
        Self {
            soft_threshold: Some(soft),
            hard_threshold: hard,
            action,
        }
    }

    /// Check if a value violates this policy.
    ///
    /// Returns (is_violation, severity, action) tuple.
    pub fn check(&self, value: Decimal) -> Option<(Severity, ConstraintAction)> {
        if value > self.hard_threshold {
            Some((self.action.to_severity(), self.action))
        } else if let Some(soft) = self.soft_threshold {
            if value > soft {
                Some((Severity::Warn, ConstraintAction::LogOnly))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Calculate breach magnitude (how much over the limit).
    pub fn magnitude(&self, value: Decimal) -> Decimal {
        let limit = if value > self.hard_threshold {
            self.hard_threshold
        } else if let Some(soft) = self.soft_threshold {
            if value > soft {
                soft
            } else {
                return Decimal::ZERO;
            }
        } else {
            return Decimal::ZERO;
        };

        (value - limit).max(Decimal::ZERO)
    }

    /// Calculate breach magnitude as percentage of limit.
    pub fn magnitude_pct(&self, value: Decimal) -> Decimal {
        let limit = if value > self.hard_threshold {
            self.hard_threshold
        } else if let Some(soft) = self.soft_threshold {
            if value > soft {
                soft
            } else {
                return Decimal::ZERO;
            }
        } else {
            return Decimal::ZERO;
        };

        if limit.is_zero() {
            Decimal::ZERO
        } else {
            (value - limit) / limit * dec!(100)
        }
    }
}

// =============================================================================
// CONSTRAINTS CONFIGURATION
// =============================================================================

/// Configuration for all portfolio constraints.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConstraintsConfig {
    /// Whether constraints checking is enabled
    #[serde(default)]
    pub enabled: bool,

    // === Exposure/Leverage ===
    /// Maximum gross exposure as % of equity
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_gross_exposure_pct: Option<ConstraintPolicy>,
    /// Maximum net exposure as % of equity
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_net_exposure_pct: Option<ConstraintPolicy>,
    /// Maximum leverage ratio
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_leverage: Option<ConstraintPolicy>,

    // === Concentration ===
    /// Maximum single position weight as % of gross
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_single_name_weight_pct: Option<ConstraintPolicy>,
    /// Maximum top-5 positions weight as % of gross
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_top5_weight_pct: Option<ConstraintPolicy>,
    /// Maximum HHI (Herfindahl-Hirschman Index)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_hhi: Option<ConstraintPolicy>,

    // === Sector ===
    /// Maximum sector weight as % of gross (applies to all sectors)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_sector_weight_pct: Option<ConstraintPolicy>,
    /// Maximum "Unknown" sector weight as % of gross
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_unknown_sector_weight_pct: Option<ConstraintPolicy>,

    // === Currency ===
    /// Maximum weight per currency as % of gross
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub max_currency_weight_pct: BTreeMap<String, ConstraintPolicy>,
    /// Maximum FX return contribution as % (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_fx_return_contribution_pct: Option<ConstraintPolicy>,

    // === Turnover/Costs ===
    /// Maximum turnover per period as %
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_turnover_pct_per_period: Option<ConstraintPolicy>,
    /// Maximum cost per period as % of equity
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_cost_pct_per_period: Option<ConstraintPolicy>,
}

impl ConstraintsConfig {
    /// Create a default "research" config with typical limits.
    pub fn research_defaults() -> Self {
        Self {
            enabled: true,
            max_gross_exposure_pct: Some(ConstraintPolicy::soft_hard(
                dec!(90),
                dec!(100),
                ConstraintAction::BlockNewTrades,
            )),
            max_net_exposure_pct: Some(ConstraintPolicy::soft_hard(
                dec!(80),
                dec!(100),
                ConstraintAction::BlockNewTrades,
            )),
            max_leverage: None,
            max_single_name_weight_pct: Some(ConstraintPolicy::soft_hard(
                dec!(15),
                dec!(20),
                ConstraintAction::BlockNewTrades,
            )),
            max_top5_weight_pct: Some(ConstraintPolicy::soft_hard(
                dec!(50),
                dec!(60),
                ConstraintAction::LogOnly,
            )),
            max_hhi: Some(ConstraintPolicy::hard(dec!(0.25), ConstraintAction::LogOnly)),
            max_sector_weight_pct: Some(ConstraintPolicy::soft_hard(
                dec!(30),
                dec!(40),
                ConstraintAction::LogOnly,
            )),
            max_unknown_sector_weight_pct: Some(ConstraintPolicy::hard(
                dec!(20),
                ConstraintAction::LogOnly,
            )),
            max_currency_weight_pct: BTreeMap::new(),
            max_fx_return_contribution_pct: None,
            max_turnover_pct_per_period: Some(ConstraintPolicy::soft_hard(
                dec!(50),
                dec!(100),
                ConstraintAction::LogOnly,
            )),
            max_cost_pct_per_period: Some(ConstraintPolicy::hard(
                dec!(1),
                ConstraintAction::LogOnly,
            )),
        }
    }
}

// =============================================================================
// CONSTRAINTS ENGINE
// =============================================================================

/// Engine for evaluating portfolio constraints.
pub struct ConstraintsEngine {
    config: ConstraintsConfig,
}

impl ConstraintsEngine {
    /// Create a new constraints engine with the given configuration.
    pub fn new(config: ConstraintsConfig) -> Self {
        Self { config }
    }

    /// Check if constraints checking is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the configuration.
    pub fn config(&self) -> &ConstraintsConfig {
        &self.config
    }

    /// Evaluate ex-post constraints (EOD, after price updates).
    ///
    /// This checks all constraints against the current portfolio state.
    pub fn evaluate_ex_post(
        &self,
        date: NaiveDate,
        equity: Decimal,
        exposure: &ExposureBreakdown,
        concentration: &ConcentrationMetrics,
        sector_exposure: &[SectorExposure],
        turnover_pct: Decimal,
        cost_pct: Decimal,
    ) -> Vec<BreachEvent> {
        if !self.config.enabled {
            return Vec::new();
        }

        let mut breaches = Vec::new();

        // === Exposure checks ===
        breaches.extend(self.check_exposure(date, equity, exposure));

        // === Concentration checks ===
        breaches.extend(self.check_concentration(date, concentration));

        // === Sector checks ===
        breaches.extend(self.check_sectors(date, sector_exposure));

        // === Currency checks ===
        breaches.extend(self.check_currencies(date, exposure));

        // === Turnover/Cost checks ===
        breaches.extend(self.check_turnover_cost(date, turnover_pct, cost_pct));

        breaches
    }

    /// Check exposure constraints.
    fn check_exposure(
        &self,
        date: NaiveDate,
        equity: Decimal,
        exposure: &ExposureBreakdown,
    ) -> Vec<BreachEvent> {
        let mut breaches = Vec::new();

        // Calculate exposure percentages
        let gross_pct = if equity.is_zero() {
            Decimal::ZERO
        } else {
            exposure.gross / equity * dec!(100)
        };

        let net_pct = if equity.is_zero() {
            Decimal::ZERO
        } else {
            exposure.net.abs() / equity * dec!(100)
        };

        // Max gross exposure
        if let Some(policy) = &self.config.max_gross_exposure_pct {
            if let Some((severity, action)) = policy.check(gross_pct) {
                breaches.push(BreachEvent {
                    date,
                    constraint_id: ConstraintId::MaxGrossExposurePct,
                    scope: ConstraintScope::Portfolio,
                    measured_value: gross_pct,
                    limit_value: policy.hard_threshold,
                    magnitude: policy.magnitude(gross_pct),
                    magnitude_pct: policy.magnitude_pct(gross_pct),
                    severity,
                    action_taken: action,
                    evidence: BreachEvidence {
                        top_contributors: vec![
                            ("gross".to_string(), exposure.gross),
                            ("equity".to_string(), equity),
                        ],
                        context: format!("Gross exposure {:.2}% exceeds limit", gross_pct),
                    },
                    is_ex_ante: false,
                });
            }
        }

        // Max net exposure
        if let Some(policy) = &self.config.max_net_exposure_pct {
            if let Some((severity, action)) = policy.check(net_pct) {
                breaches.push(BreachEvent {
                    date,
                    constraint_id: ConstraintId::MaxNetExposurePct,
                    scope: ConstraintScope::Portfolio,
                    measured_value: net_pct,
                    limit_value: policy.hard_threshold,
                    magnitude: policy.magnitude(net_pct),
                    magnitude_pct: policy.magnitude_pct(net_pct),
                    severity,
                    action_taken: action,
                    evidence: BreachEvidence {
                        top_contributors: vec![
                            ("net".to_string(), exposure.net),
                            ("equity".to_string(), equity),
                        ],
                        context: format!("Net exposure {:.2}% exceeds limit", net_pct),
                    },
                    is_ex_ante: false,
                });
            }
        }

        // Max leverage
        if let Some(policy) = &self.config.max_leverage {
            let leverage = if equity.is_zero() {
                Decimal::ZERO
            } else {
                exposure.gross / equity
            };

            if let Some((severity, action)) = policy.check(leverage) {
                breaches.push(BreachEvent {
                    date,
                    constraint_id: ConstraintId::MaxLeverage,
                    scope: ConstraintScope::Portfolio,
                    measured_value: leverage,
                    limit_value: policy.hard_threshold,
                    magnitude: policy.magnitude(leverage),
                    magnitude_pct: policy.magnitude_pct(leverage),
                    severity,
                    action_taken: action,
                    evidence: BreachEvidence {
                        top_contributors: vec![],
                        context: format!("Leverage {:.2}x exceeds limit", leverage),
                    },
                    is_ex_ante: false,
                });
            }
        }

        breaches
    }

    /// Check concentration constraints.
    fn check_concentration(
        &self,
        date: NaiveDate,
        concentration: &ConcentrationMetrics,
    ) -> Vec<BreachEvent> {
        let mut breaches = Vec::new();

        // Max single name weight
        if let Some(policy) = &self.config.max_single_name_weight_pct {
            let value = concentration.top_1_weight_pct;
            if let Some((severity, action)) = policy.check(value) {
                breaches.push(BreachEvent {
                    date,
                    constraint_id: ConstraintId::MaxSingleNameWeightPct,
                    scope: ConstraintScope::Portfolio,
                    measured_value: value,
                    limit_value: policy.hard_threshold,
                    magnitude: policy.magnitude(value),
                    magnitude_pct: policy.magnitude_pct(value),
                    severity,
                    action_taken: action,
                    evidence: BreachEvidence {
                        top_contributors: vec![],
                        context: format!("Top position {:.2}% exceeds single-name limit", value),
                    },
                    is_ex_ante: false,
                });
            }
        }

        // Max top-5 weight
        if let Some(policy) = &self.config.max_top5_weight_pct {
            let value = concentration.top_5_weight_pct;
            if let Some((severity, action)) = policy.check(value) {
                breaches.push(BreachEvent {
                    date,
                    constraint_id: ConstraintId::MaxTop5WeightPct,
                    scope: ConstraintScope::Portfolio,
                    measured_value: value,
                    limit_value: policy.hard_threshold,
                    magnitude: policy.magnitude(value),
                    magnitude_pct: policy.magnitude_pct(value),
                    severity,
                    action_taken: action,
                    evidence: BreachEvidence {
                        top_contributors: vec![],
                        context: format!("Top 5 positions {:.2}% exceeds limit", value),
                    },
                    is_ex_ante: false,
                });
            }
        }

        // Max HHI
        if let Some(policy) = &self.config.max_hhi {
            let value = concentration.hhi;
            if let Some((severity, action)) = policy.check(value) {
                breaches.push(BreachEvent {
                    date,
                    constraint_id: ConstraintId::MaxHHI,
                    scope: ConstraintScope::Portfolio,
                    measured_value: value,
                    limit_value: policy.hard_threshold,
                    magnitude: policy.magnitude(value),
                    magnitude_pct: policy.magnitude_pct(value),
                    severity,
                    action_taken: action,
                    evidence: BreachEvidence {
                        top_contributors: vec![
                            (
                                "effective_n".to_string(),
                                concentration.effective_n,
                            ),
                        ],
                        context: format!(
                            "HHI {:.4} exceeds limit (effective N = {:.1})",
                            value, concentration.effective_n
                        ),
                    },
                    is_ex_ante: false,
                });
            }
        }

        breaches
    }

    /// Check sector constraints.
    fn check_sectors(&self, date: NaiveDate, sectors: &[SectorExposure]) -> Vec<BreachEvent> {
        let mut breaches = Vec::new();

        for sector in sectors {
            // Check per-sector limit
            if let Some(policy) = &self.config.max_sector_weight_pct {
                if let Some((severity, action)) = policy.check(sector.weight_pct) {
                    breaches.push(BreachEvent {
                        date,
                        constraint_id: ConstraintId::MaxSectorWeightPct {
                            sector: sector.sector.clone(),
                        },
                        scope: ConstraintScope::Sector(sector.sector.clone()),
                        measured_value: sector.weight_pct,
                        limit_value: policy.hard_threshold,
                        magnitude: policy.magnitude(sector.weight_pct),
                        magnitude_pct: policy.magnitude_pct(sector.weight_pct),
                        severity,
                        action_taken: action,
                        evidence: BreachEvidence {
                            top_contributors: vec![
                                ("gross".to_string(), sector.gross),
                            ],
                            context: format!(
                                "Sector {} at {:.2}% exceeds limit",
                                sector.sector, sector.weight_pct
                            ),
                        },
                        is_ex_ante: false,
                    });
                }
            }

            // Check Unknown sector specifically
            if sector.sector == "Unknown" || sector.sector == "Unclassified" {
                if let Some(policy) = &self.config.max_unknown_sector_weight_pct {
                    if let Some((severity, action)) = policy.check(sector.weight_pct) {
                        breaches.push(BreachEvent {
                            date,
                            constraint_id: ConstraintId::MaxUnknownSectorWeightPct,
                            scope: ConstraintScope::Sector(sector.sector.clone()),
                            measured_value: sector.weight_pct,
                            limit_value: policy.hard_threshold,
                            magnitude: policy.magnitude(sector.weight_pct),
                            magnitude_pct: policy.magnitude_pct(sector.weight_pct),
                            severity,
                            action_taken: action,
                            evidence: BreachEvidence {
                                top_contributors: vec![],
                                context: format!(
                                    "Unknown sector at {:.2}% exceeds limit",
                                    sector.weight_pct
                                ),
                            },
                            is_ex_ante: false,
                        });
                    }
                }
            }
        }

        breaches
    }

    /// Check currency constraints.
    fn check_currencies(
        &self,
        date: NaiveDate,
        exposure: &ExposureBreakdown,
    ) -> Vec<BreachEvent> {
        let mut breaches = Vec::new();

        let total_gross = exposure.gross;
        if total_gross.is_zero() {
            return breaches;
        }

        for (currency, &value) in &exposure.by_currency {
            let weight_pct = value.abs() / total_gross * dec!(100);

            if let Some(policy) = self.config.max_currency_weight_pct.get(currency) {
                if let Some((severity, action)) = policy.check(weight_pct) {
                    breaches.push(BreachEvent {
                        date,
                        constraint_id: ConstraintId::MaxCurrencyWeightPct {
                            currency: currency.clone(),
                        },
                        scope: ConstraintScope::Currency(currency.clone()),
                        measured_value: weight_pct,
                        limit_value: policy.hard_threshold,
                        magnitude: policy.magnitude(weight_pct),
                        magnitude_pct: policy.magnitude_pct(weight_pct),
                        severity,
                        action_taken: action,
                        evidence: BreachEvidence {
                            top_contributors: vec![(currency.clone(), value)],
                            context: format!(
                                "Currency {} at {:.2}% exceeds limit",
                                currency, weight_pct
                            ),
                        },
                        is_ex_ante: false,
                    });
                }
            }
        }

        breaches
    }

    /// Check turnover and cost constraints.
    fn check_turnover_cost(
        &self,
        date: NaiveDate,
        turnover_pct: Decimal,
        cost_pct: Decimal,
    ) -> Vec<BreachEvent> {
        let mut breaches = Vec::new();

        // Max turnover
        if let Some(policy) = &self.config.max_turnover_pct_per_period {
            if let Some((severity, action)) = policy.check(turnover_pct) {
                breaches.push(BreachEvent {
                    date,
                    constraint_id: ConstraintId::MaxTurnoverPctPerPeriod,
                    scope: ConstraintScope::Portfolio,
                    measured_value: turnover_pct,
                    limit_value: policy.hard_threshold,
                    magnitude: policy.magnitude(turnover_pct),
                    magnitude_pct: policy.magnitude_pct(turnover_pct),
                    severity,
                    action_taken: action,
                    evidence: BreachEvidence {
                        top_contributors: vec![],
                        context: format!("Turnover {:.2}% exceeds limit", turnover_pct),
                    },
                    is_ex_ante: false,
                });
            }
        }

        // Max cost
        if let Some(policy) = &self.config.max_cost_pct_per_period {
            if let Some((severity, action)) = policy.check(cost_pct) {
                breaches.push(BreachEvent {
                    date,
                    constraint_id: ConstraintId::MaxCostPctPerPeriod,
                    scope: ConstraintScope::Portfolio,
                    measured_value: cost_pct,
                    limit_value: policy.hard_threshold,
                    magnitude: policy.magnitude(cost_pct),
                    magnitude_pct: policy.magnitude_pct(cost_pct),
                    severity,
                    action_taken: action,
                    evidence: BreachEvidence {
                        top_contributors: vec![],
                        context: format!("Cost {:.2}% exceeds limit", cost_pct),
                    },
                    is_ex_ante: false,
                });
            }
        }

        breaches
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_policy_check() {
        let policy = ConstraintPolicy::soft_hard(dec!(80), dec!(100), ConstraintAction::BlockNewTrades);

        // Below soft - no violation
        assert!(policy.check(dec!(70)).is_none());

        // Above soft, below hard - WARN
        let result = policy.check(dec!(85));
        assert!(result.is_some());
        let (severity, action) = result.unwrap();
        assert_eq!(severity, Severity::Warn);
        assert_eq!(action, ConstraintAction::LogOnly);

        // Above hard - CRIT with action
        let result = policy.check(dec!(105));
        assert!(result.is_some());
        let (severity, action) = result.unwrap();
        assert_eq!(severity, Severity::Crit);
        assert_eq!(action, ConstraintAction::BlockNewTrades);
    }

    #[test]
    fn test_constraint_policy_magnitude() {
        let policy = ConstraintPolicy::soft_hard(dec!(80), dec!(100), ConstraintAction::LogOnly);

        // Below soft - zero magnitude
        assert_eq!(policy.magnitude(dec!(70)), Decimal::ZERO);

        // Above soft - magnitude from soft
        assert_eq!(policy.magnitude(dec!(85)), dec!(5));

        // Above hard - magnitude from hard
        assert_eq!(policy.magnitude(dec!(110)), dec!(10));
    }

    #[test]
    fn test_constraint_id_display() {
        assert_eq!(
            format!("{}", ConstraintId::MaxGrossExposurePct),
            "MaxGrossExposurePct"
        );
        assert_eq!(
            format!(
                "{}",
                ConstraintId::MaxSectorWeightPct {
                    sector: "Energy".to_string()
                }
            ),
            "MaxSectorWeightPct:Energy"
        );
    }

    #[test]
    fn test_constraint_scope_display() {
        assert_eq!(format!("{}", ConstraintScope::Portfolio), "Portfolio");
        assert_eq!(
            format!("{}", ConstraintScope::Symbol("PETR4".to_string())),
            "Symbol:PETR4"
        );
    }

    #[test]
    fn test_constraints_config_defaults() {
        let config = ConstraintsConfig::research_defaults();
        assert!(config.enabled);
        assert!(config.max_gross_exposure_pct.is_some());
        assert!(config.max_single_name_weight_pct.is_some());
    }

    #[test]
    fn test_engine_disabled() {
        let config = ConstraintsConfig::default();
        let engine = ConstraintsEngine::new(config);

        let breaches = engine.evaluate_ex_post(
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            dec!(100000),
            &ExposureBreakdown::default(),
            &ConcentrationMetrics::default(),
            &[],
            dec!(0),
            dec!(0),
        );

        assert!(breaches.is_empty());
    }

    #[test]
    fn test_engine_gross_exposure_breach() {
        let mut config = ConstraintsConfig::default();
        config.enabled = true;
        config.max_gross_exposure_pct = Some(ConstraintPolicy::hard(
            dec!(100),
            ConstraintAction::BlockNewTrades,
        ));

        let engine = ConstraintsEngine::new(config);

        let mut exposure = ExposureBreakdown::default();
        exposure.gross = dec!(120000); // 120% of equity

        let breaches = engine.evaluate_ex_post(
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            dec!(100000),
            &exposure,
            &ConcentrationMetrics::default(),
            &[],
            dec!(0),
            dec!(0),
        );

        assert_eq!(breaches.len(), 1);
        assert_eq!(breaches[0].constraint_id, ConstraintId::MaxGrossExposurePct);
        assert_eq!(breaches[0].measured_value, dec!(120));
        assert_eq!(breaches[0].severity, Severity::Crit);
    }

    #[test]
    fn test_engine_concentration_breach() {
        let mut config = ConstraintsConfig::default();
        config.enabled = true;
        config.max_single_name_weight_pct = Some(ConstraintPolicy::hard(
            dec!(20),
            ConstraintAction::LogOnly,
        ));

        let engine = ConstraintsEngine::new(config);

        let mut concentration = ConcentrationMetrics::default();
        concentration.top_1_weight_pct = dec!(25); // 25% > 20% limit

        let breaches = engine.evaluate_ex_post(
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            dec!(100000),
            &ExposureBreakdown::default(),
            &concentration,
            &[],
            dec!(0),
            dec!(0),
        );

        assert_eq!(breaches.len(), 1);
        assert_eq!(
            breaches[0].constraint_id,
            ConstraintId::MaxSingleNameWeightPct
        );
        assert_eq!(breaches[0].magnitude, dec!(5)); // 25 - 20 = 5
    }

    #[test]
    fn test_engine_sector_breach() {
        let mut config = ConstraintsConfig::default();
        config.enabled = true;
        config.max_sector_weight_pct = Some(ConstraintPolicy::hard(
            dec!(30),
            ConstraintAction::LogOnly,
        ));

        let engine = ConstraintsEngine::new(config);

        let sectors = vec![
            SectorExposure {
                sector: "Energy".to_string(),
                gross: dec!(50000),
                net: dec!(50000),
                long: dec!(50000),
                short: Decimal::ZERO,
                weight_pct: dec!(50), // 50% > 30% limit
            },
        ];

        let breaches = engine.evaluate_ex_post(
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            dec!(100000),
            &ExposureBreakdown::default(),
            &ConcentrationMetrics::default(),
            &sectors,
            dec!(0),
            dec!(0),
        );

        assert_eq!(breaches.len(), 1);
        assert!(matches!(
            &breaches[0].constraint_id,
            ConstraintId::MaxSectorWeightPct { sector } if sector == "Energy"
        ));
    }

    #[test]
    fn test_engine_unknown_sector_breach() {
        let mut config = ConstraintsConfig::default();
        config.enabled = true;
        config.max_unknown_sector_weight_pct = Some(ConstraintPolicy::hard(
            dec!(10),
            ConstraintAction::LogOnly,
        ));

        let engine = ConstraintsEngine::new(config);

        let sectors = vec![SectorExposure {
            sector: "Unknown".to_string(),
            gross: dec!(30000),
            net: dec!(30000),
            long: dec!(30000),
            short: Decimal::ZERO,
            weight_pct: dec!(30), // 30% > 10% limit
        }];

        let breaches = engine.evaluate_ex_post(
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            dec!(100000),
            &ExposureBreakdown::default(),
            &ConcentrationMetrics::default(),
            &sectors,
            dec!(0),
            dec!(0),
        );

        assert_eq!(breaches.len(), 1);
        assert_eq!(
            breaches[0].constraint_id,
            ConstraintId::MaxUnknownSectorWeightPct
        );
    }

    #[test]
    fn test_engine_turnover_breach() {
        let mut config = ConstraintsConfig::default();
        config.enabled = true;
        config.max_turnover_pct_per_period = Some(ConstraintPolicy::hard(
            dec!(50),
            ConstraintAction::LogOnly,
        ));

        let engine = ConstraintsEngine::new(config);

        let breaches = engine.evaluate_ex_post(
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            dec!(100000),
            &ExposureBreakdown::default(),
            &ConcentrationMetrics::default(),
            &[],
            dec!(75), // 75% turnover > 50% limit
            dec!(0),
        );

        assert_eq!(breaches.len(), 1);
        assert_eq!(
            breaches[0].constraint_id,
            ConstraintId::MaxTurnoverPctPerPeriod
        );
    }
}


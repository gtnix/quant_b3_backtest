//! Risk guard for portfolio-level risk controls.

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use super::types::{DrawdownAction, ExitContext, Position, RiskViolation};
use crate::filters::Market;

/// Risk configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    /// Maximum exposure per single asset (e.g., 0.20 for 20%)
    #[serde(default = "default_max_single")]
    pub max_single_exposure: f64,

    /// Maximum total market exposure (e.g., 1.0 for 100%)
    #[serde(default = "default_max_market")]
    pub max_market_exposure: f64,

    /// Maximum turnover per rebalance (e.g., 0.30 for 30%)
    #[serde(default = "default_max_turnover")]
    pub max_turnover_per_rebalance: f64,

    /// Maximum portfolio drawdown (negative, e.g., -0.15 for 15%)
    #[serde(default = "default_max_drawdown")]
    pub max_drawdown_pct: f64,

    /// Action to take when drawdown is exceeded
    #[serde(default)]
    pub drawdown_action: DrawdownAction,

    /// Whether to check single exposure
    #[serde(default = "default_true")]
    pub check_exposure: bool,

    /// Whether to check turnover
    #[serde(default = "default_true")]
    pub check_turnover: bool,

    /// Whether to check drawdown
    #[serde(default = "default_true")]
    pub check_drawdown: bool,

    /// Separate limits by market (BR/US)
    #[serde(default = "default_true")]
    pub per_market_limits: bool,
}

fn default_max_single() -> f64 { 0.20 }
fn default_max_market() -> f64 { 1.0 }
fn default_max_turnover() -> f64 { 0.50 }
fn default_max_drawdown() -> f64 { -0.15 }
fn default_true() -> bool { true }

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            max_single_exposure: default_max_single(),
            max_market_exposure: default_max_market(),
            max_turnover_per_rebalance: default_max_turnover(),
            max_drawdown_pct: default_max_drawdown(),
            drawdown_action: DrawdownAction::default(),
            check_exposure: true,
            check_turnover: true,
            check_drawdown: true,
            per_market_limits: true,
        }
    }
}

/// Risk guard for portfolio-level checks.
#[derive(Debug, Clone)]
pub struct RiskGuard {
    config: RiskConfig,
}

impl RiskGuard {
    pub fn new(config: RiskConfig) -> Self {
        Self { config }
    }

    /// Check if any position exceeds max single exposure.
    pub fn check_exposure(
        &self,
        positions: &[Position],
        capital: Decimal,
    ) -> Vec<(String, RiskViolation)> {
        if !self.config.check_exposure || capital == Decimal::ZERO {
            return Vec::new();
        }

        let mut violations = Vec::new();
        let capital_f64: f64 = capital.try_into().unwrap_or(1.0);

        for pos in positions {
            let value: f64 = pos.market_value().try_into().unwrap_or(0.0);
            let exposure = value / capital_f64;

            if exposure > self.config.max_single_exposure {
                violations.push((pos.symbol.clone(), RiskViolation::ExposureExceeded));
            }
        }

        violations
    }

    /// Check if total market exposure exceeds limit.
    pub fn check_market_exposure(
        &self,
        positions: &[Position],
        capital: Decimal,
        market: Market,
    ) -> Option<RiskViolation> {
        if !self.config.check_exposure || capital == Decimal::ZERO {
            return None;
        }

        let capital_f64: f64 = capital.try_into().unwrap_or(1.0);

        let total_value: f64 = positions
            .iter()
            .filter(|p| !self.config.per_market_limits || p.market == market)
            .map(|p| -> f64 { p.market_value().try_into().unwrap_or(0.0) })
            .sum();

        let exposure = total_value / capital_f64;

        if exposure > self.config.max_market_exposure {
            Some(RiskViolation::MarketExposureExceeded)
        } else {
            None
        }
    }

    /// Check if turnover exceeds limit.
    pub fn check_turnover(&self, turnover: Decimal, capital: Decimal) -> Option<RiskViolation> {
        if !self.config.check_turnover || capital == Decimal::ZERO {
            return None;
        }

        let turnover_pct: f64 = (turnover / capital).try_into().unwrap_or(0.0);

        if turnover_pct > self.config.max_turnover_per_rebalance {
            Some(RiskViolation::TurnoverExceeded)
        } else {
            None
        }
    }

    /// Check if portfolio drawdown exceeds limit.
    pub fn check_drawdown(&self, context: &ExitContext) -> Option<RiskViolation> {
        if !self.config.check_drawdown {
            return None;
        }

        let drawdown = context.portfolio_drawdown();

        // drawdown is negative, max_drawdown_pct is negative
        if drawdown <= self.config.max_drawdown_pct {
            Some(RiskViolation::DrawdownExceeded)
        } else {
            None
        }
    }

    /// Get the drawdown action when drawdown is exceeded.
    pub fn drawdown_action(&self) -> DrawdownAction {
        self.config.drawdown_action
    }

    /// Get max single exposure limit.
    pub fn max_single_exposure(&self) -> f64 {
        self.config.max_single_exposure
    }

    /// Get max turnover limit.
    pub fn max_turnover(&self) -> f64 {
        self.config.max_turnover_per_rebalance
    }

    /// Get max drawdown limit.
    pub fn max_drawdown(&self) -> f64 {
        self.config.max_drawdown_pct
    }

    /// Run all risk checks and return violations.
    pub fn run_all_checks(
        &self,
        positions: &[Position],
        context: &ExitContext,
        turnover: Decimal,
    ) -> Vec<RiskViolation> {
        let mut violations = Vec::new();

        // Single exposure checks
        for (_, violation) in self.check_exposure(positions, context.capital) {
            if !violations.contains(&violation) {
                violations.push(violation);
            }
        }

        // Market exposure check
        if let Some(v) = self.check_market_exposure(positions, context.capital, context.market) {
            violations.push(v);
        }

        // Turnover check
        if let Some(v) = self.check_turnover(turnover, context.capital) {
            violations.push(v);
        }

        // Drawdown check
        if let Some(v) = self.check_drawdown(context) {
            violations.push(v);
        }

        violations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use rust_decimal_macros::dec;

    fn make_context(equity: Decimal, peak: Decimal) -> ExitContext {
        ExitContext {
            date: NaiveDate::from_ymd_opt(2025, 1, 10).unwrap(),
            capital: dec!(1_000_000),
            equity,
            peak_equity: peak,
            market: Market::BR,
        }
    }

    #[test]
    fn test_exposure_violation() {
        let guard = RiskGuard::new(RiskConfig {
            max_single_exposure: 0.20,
            ..Default::default()
        });

        let positions = vec![
            Position::new("PETR4", Market::BR, 1000, dec!(50), 
                NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), dec!(60)),
            Position::new("VALE3", Market::BR, 5000, dec!(60), 
                NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), dec!(65)),
        ];

        // PETR4: 1000 * 60 = 60k / 1M = 6% - OK
        // VALE3: 5000 * 65 = 325k / 1M = 32.5% - VIOLATION

        let violations = guard.check_exposure(&positions, dec!(1_000_000));
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].0, "VALE3");
        assert_eq!(violations[0].1, RiskViolation::ExposureExceeded);
    }

    #[test]
    fn test_no_exposure_violation() {
        let guard = RiskGuard::new(RiskConfig {
            max_single_exposure: 0.20,
            ..Default::default()
        });

        let positions = vec![
            Position::new("PETR4", Market::BR, 1000, dec!(50), 
                NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), dec!(50)),
            Position::new("VALE3", Market::BR, 1000, dec!(60), 
                NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), dec!(60)),
        ];

        // Both under 20%
        let violations = guard.check_exposure(&positions, dec!(1_000_000));
        assert!(violations.is_empty());
    }

    #[test]
    fn test_turnover_violation() {
        let guard = RiskGuard::new(RiskConfig {
            max_turnover_per_rebalance: 0.30,
            ..Default::default()
        });

        // 40% turnover exceeds 30% limit
        let violation = guard.check_turnover(dec!(400_000), dec!(1_000_000));
        assert!(violation.is_some());
        assert_eq!(violation.unwrap(), RiskViolation::TurnoverExceeded);
    }

    #[test]
    fn test_no_turnover_violation() {
        let guard = RiskGuard::new(RiskConfig {
            max_turnover_per_rebalance: 0.30,
            ..Default::default()
        });

        // 20% turnover is OK
        let violation = guard.check_turnover(dec!(200_000), dec!(1_000_000));
        assert!(violation.is_none());
    }

    #[test]
    fn test_drawdown_violation() {
        let guard = RiskGuard::new(RiskConfig {
            max_drawdown_pct: -0.15,
            ..Default::default()
        });

        // 20% drawdown exceeds 15% limit
        let ctx = make_context(dec!(800_000), dec!(1_000_000));
        let violation = guard.check_drawdown(&ctx);
        assert!(violation.is_some());
        assert_eq!(violation.unwrap(), RiskViolation::DrawdownExceeded);
    }

    #[test]
    fn test_no_drawdown_violation() {
        let guard = RiskGuard::new(RiskConfig {
            max_drawdown_pct: -0.15,
            ..Default::default()
        });

        // 10% drawdown is OK
        let ctx = make_context(dec!(900_000), dec!(1_000_000));
        let violation = guard.check_drawdown(&ctx);
        assert!(violation.is_none());
    }

    #[test]
    fn test_run_all_checks() {
        let guard = RiskGuard::new(RiskConfig {
            max_single_exposure: 0.20,
            max_turnover_per_rebalance: 0.30,
            max_drawdown_pct: -0.15,
            ..Default::default()
        });

        let positions = vec![
            Position::new("PETR4", Market::BR, 5000, dec!(50), 
                NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), dec!(60)),
        ];

        // 25% exposure + 20% drawdown
        let ctx = make_context(dec!(800_000), dec!(1_000_000));
        let violations = guard.run_all_checks(&positions, &ctx, dec!(100_000));

        assert!(violations.contains(&RiskViolation::ExposureExceeded));
        assert!(violations.contains(&RiskViolation::DrawdownExceeded));
        assert!(!violations.contains(&RiskViolation::TurnoverExceeded)); // 10% is OK
    }
}










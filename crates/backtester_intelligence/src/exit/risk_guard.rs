//! Risk guard for portfolio-level risk controls.

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use backtester_core::Money;
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

    /// CVaR limit at 95% confidence (negative, e.g., -0.03 for 3%)
    /// Reference: Rockafellar & Uryasev (2000)
    #[serde(default = "default_cvar_limit")]
    pub cvar_limit_95: f64,

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

    /// Whether to check CVaR limit
    #[serde(default = "default_true")]
    pub check_cvar: bool,

    /// Separate limits by market (BR/US)
    #[serde(default = "default_true")]
    pub per_market_limits: bool,
}

fn default_max_single() -> f64 { 0.20 }
fn default_max_market() -> f64 { 1.0 }
fn default_max_turnover() -> f64 { 0.50 }
fn default_max_drawdown() -> f64 { -0.15 }
fn default_cvar_limit() -> f64 { -0.03 } // 3% CVaR limit at 95% confidence
fn default_true() -> bool { true }

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            max_single_exposure: default_max_single(),
            max_market_exposure: default_max_market(),
            max_turnover_per_rebalance: default_max_turnover(),
            max_drawdown_pct: default_max_drawdown(),
            cvar_limit_95: default_cvar_limit(),
            drawdown_action: DrawdownAction::default(),
            check_exposure: true,
            check_turnover: true,
            check_drawdown: true,
            check_cvar: true,
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
        capital: Money,
    ) -> Vec<(String, RiskViolation)> {
        if !self.config.check_exposure || capital.is_zero() {
            return Vec::new();
        }

        let mut violations = Vec::new();
        let capital_f64 = capital.to_f64();

        for pos in positions {
            let value = pos.market_value_fast().to_f64();
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
        capital: Money,
        market: Market,
    ) -> Option<RiskViolation> {
        if !self.config.check_exposure || capital.is_zero() {
            return None;
        }

        let capital_f64 = capital.to_f64();

        let total_value: f64 = positions
            .iter()
            .filter(|p| !self.config.per_market_limits || p.market == market)
            .map(|p| p.market_value_fast().to_f64())
            .sum();

        let exposure = total_value / capital_f64;

        if exposure > self.config.max_market_exposure {
            Some(RiskViolation::MarketExposureExceeded)
        } else {
            None
        }
    }

    /// Check if turnover exceeds limit (Money version).
    pub fn check_turnover_fast(&self, turnover: Money, capital: Money) -> Option<RiskViolation> {
        if !self.config.check_turnover || capital.is_zero() {
            return None;
        }

        let turnover_pct = turnover.div_money(capital);

        if turnover_pct > self.config.max_turnover_per_rebalance {
            Some(RiskViolation::TurnoverExceeded)
        } else {
            None
        }
    }

    /// Check if turnover exceeds limit (Decimal version for compatibility).
    pub fn check_turnover(&self, turnover: Decimal, capital: Decimal) -> Option<RiskViolation> {
        self.check_turnover_fast(Money::from(turnover), Money::from(capital))
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

    /// Check if CVaR (Conditional Value-at-Risk) exceeds limit.
    /// 
    /// CVaR at 95% is the mean of the worst 5% of daily returns.
    /// Reference: Rockafellar & Uryasev (2000)
    pub fn check_cvar(&self, daily_returns: &[f64]) -> Option<RiskViolation> {
        if !self.config.check_cvar || daily_returns.len() < 20 {
            return None; // Need at least 20 days for meaningful CVaR
        }

        let cvar = Self::calculate_cvar_95(daily_returns);

        // cvar is negative, cvar_limit_95 is negative
        if cvar <= self.config.cvar_limit_95 {
            Some(RiskViolation::CVaRExceeded)
        } else {
            None
        }
    }

    /// Calculate CVaR at 95% confidence (mean of worst 5% of returns).
    fn calculate_cvar_95(returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mut sorted = returns.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let tail_count = ((n as f64) * 0.05).ceil() as usize;
        let tail_count = tail_count.max(1).min(n);

        let tail_sum: f64 = sorted[..tail_count].iter().sum();
        tail_sum / tail_count as f64
    }

    /// Get the drawdown action when drawdown is exceeded.
    pub fn drawdown_action(&self) -> DrawdownAction {
        self.config.drawdown_action
    }

    /// Get CVaR limit.
    pub fn cvar_limit(&self) -> f64 {
        self.config.cvar_limit_95
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
        self.run_all_checks_with_returns(positions, context, turnover, &[])
    }

    /// Run all risk checks including CVaR with daily returns (Money version, fast).
    pub fn run_all_checks_with_returns_fast(
        &self,
        positions: &[Position],
        context: &ExitContext,
        turnover: Money,
        daily_returns: &[f64],
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
        if let Some(v) = self.check_turnover_fast(turnover, context.capital) {
            violations.push(v);
        }

        // Drawdown check
        if let Some(v) = self.check_drawdown(context) {
            violations.push(v);
        }

        // CVaR check
        if let Some(v) = self.check_cvar(daily_returns) {
            violations.push(v);
        }

        violations
    }

    /// Run all risk checks including CVaR with daily returns (Decimal version for compatibility).
    pub fn run_all_checks_with_returns(
        &self,
        positions: &[Position],
        context: &ExitContext,
        turnover: Decimal,
        daily_returns: &[f64],
    ) -> Vec<RiskViolation> {
        self.run_all_checks_with_returns_fast(positions, context, Money::from(turnover), daily_returns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use rust_decimal_macros::dec;

    fn make_context(equity: Decimal, peak: Decimal) -> ExitContext {
        let mut ctx = ExitContext::new(NaiveDate::from_ymd_opt(2025, 1, 10).unwrap(), 
            dec!(1_000_000), equity, Market::BR);
        ctx.peak_equity = Money::from(peak);
        ctx
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

        let violations = guard.check_exposure(&positions, Money::from(dec!(1_000_000)));
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
        let violations = guard.check_exposure(&positions, Money::from_int(1_000_000));
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

    #[test]
    fn test_cvar_violation() {
        let guard = RiskGuard::new(RiskConfig {
            cvar_limit_95: -0.03, // 3% CVaR limit
            check_cvar: true,
            ..Default::default()
        });

        // Returns with severe tail risk (CVaR should exceed -3%)
        let returns: Vec<f64> = vec![
            -0.10, -0.08, -0.05, -0.02, 0.01,
            0.02, 0.01, 0.02, 0.01, 0.02,
            0.01, 0.02, 0.01, 0.02, 0.01,
            0.02, 0.01, 0.02, 0.01, 0.02,
        ];

        let violation = guard.check_cvar(&returns);
        assert!(violation.is_some());
        assert_eq!(violation.unwrap(), RiskViolation::CVaRExceeded);
    }

    #[test]
    fn test_no_cvar_violation() {
        let guard = RiskGuard::new(RiskConfig {
            cvar_limit_95: -0.05, // 5% CVaR limit (more relaxed)
            check_cvar: true,
            ..Default::default()
        });

        // Returns with moderate tail risk (CVaR within -5%)
        let returns: Vec<f64> = vec![
            -0.03, -0.02, -0.01, 0.01, 0.02,
            0.01, 0.02, 0.01, 0.02, 0.01,
            0.02, 0.01, 0.02, 0.01, 0.02,
            0.01, 0.02, 0.01, 0.02, 0.01,
        ];

        let violation = guard.check_cvar(&returns);
        assert!(violation.is_none());
    }

    #[test]
    fn test_cvar_insufficient_data() {
        let guard = RiskGuard::new(RiskConfig {
            cvar_limit_95: -0.03,
            check_cvar: true,
            ..Default::default()
        });

        // Only 10 days - should not trigger (needs 20+)
        let returns: Vec<f64> = vec![-0.10, -0.08, -0.05, -0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02];

        let violation = guard.check_cvar(&returns);
        assert!(violation.is_none());
    }

    #[test]
    fn test_cvar_calculation() {
        // Test the CVaR calculation directly
        let returns = vec![
            -0.10, -0.05, -0.02, 0.0, 0.01,
            0.02, 0.03, 0.04, 0.05, 0.10,
        ];
        
        // 5% of 10 = 0.5 → ceil = 1 observation
        // CVaR95 should be mean of worst 1 = -0.10
        let cvar = RiskGuard::calculate_cvar_95(&returns);
        assert!((cvar - (-0.10)).abs() < 0.001);
    }
}































//! Take-profit exit policy.

use serde::{Deserialize, Serialize};

use super::policy::ExitPolicy;
use super::types::{ExitContext, ExitReason, ExitTarget, Position};

/// Take-profit configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TakeProfitConfig {
    /// Gain threshold (positive, e.g., 0.30 for 30% gain)
    #[serde(default = "default_threshold")]
    pub threshold_pct: f64,

    /// Use close price (vs intraday high)
    #[serde(default = "default_true")]
    pub use_close_price: bool,

    /// Whether policy is enabled
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_threshold() -> f64 { 0.30 }
fn default_true() -> bool { true }

impl Default for TakeProfitConfig {
    fn default() -> Self {
        Self {
            threshold_pct: default_threshold(),
            use_close_price: true,
            enabled: true,
        }
    }
}

/// Take-profit policy: exit when unrealized gain exceeds threshold.
#[derive(Debug, Clone)]
pub struct TakeProfitPolicy {
    config: TakeProfitConfig,
}

impl TakeProfitPolicy {
    pub fn new(config: TakeProfitConfig) -> Self {
        Self { config }
    }

    pub fn with_threshold(threshold_pct: f64) -> Self {
        Self::new(TakeProfitConfig {
            threshold_pct,
            ..Default::default()
        })
    }
}

impl ExitPolicy for TakeProfitPolicy {
    fn evaluate(&self, position: &Position, _context: &ExitContext) -> Option<ExitTarget> {
        if !self.config.enabled {
            return None;
        }

        let unrealized_return = position.unrealized_return();

        // Trigger if gain exceeds threshold
        if unrealized_return >= self.config.threshold_pct {
            Some(ExitTarget::from_position(position, ExitReason::TakeProfit, None))
        } else {
            None
        }
    }

    fn name(&self) -> &'static str {
        "take_profit"
    }

    fn is_enabled(&self) -> bool {
        self.config.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::Market;
    use chrono::NaiveDate;
    use rust_decimal_macros::dec;

    fn make_context() -> ExitContext {
        ExitContext::new(
            NaiveDate::from_ymd_opt(2025, 1, 10).unwrap(),
            dec!(1_000_000),
            dec!(1_000_000),
            Market::BR,
        )
    }

    #[test]
    fn test_take_profit_triggers_on_gain() {
        let policy = TakeProfitPolicy::with_threshold(0.25);
        let ctx = make_context();

        // Position with 30% gain
        let pos = Position::new(
            "PETR4",
            Market::BR,
            100,
            dec!(40),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            dec!(52), // 30% gain
        );

        let result = policy.evaluate(&pos, &ctx);
        assert!(result.is_some());
        assert_eq!(result.unwrap().reason, ExitReason::TakeProfit);
    }

    #[test]
    fn test_take_profit_no_trigger_on_small_gain() {
        let policy = TakeProfitPolicy::with_threshold(0.30);
        let ctx = make_context();

        // Position with 15% gain (below threshold)
        let pos = Position::new(
            "VALE3",
            Market::BR,
            100,
            dec!(60),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            dec!(69), // 15% gain
        );

        let result = policy.evaluate(&pos, &ctx);
        assert!(result.is_none());
    }

    #[test]
    fn test_take_profit_no_trigger_on_loss() {
        let policy = TakeProfitPolicy::with_threshold(0.30);
        let ctx = make_context();

        // Position with 10% loss
        let pos = Position::new(
            "ITUB4",
            Market::BR,
            100,
            dec!(30),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            dec!(27), // 10% loss
        );

        let result = policy.evaluate(&pos, &ctx);
        assert!(result.is_none());
    }

    #[test]
    fn test_take_profit_exact_threshold() {
        let policy = TakeProfitPolicy::with_threshold(0.20);
        let ctx = make_context();

        // Position with exactly 20% gain
        let pos = Position::new(
            "WEGE3",
            Market::BR,
            100,
            dec!(100),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            dec!(120), // exactly 20% gain
        );

        let result = policy.evaluate(&pos, &ctx);
        assert!(result.is_some()); // Should trigger at exact threshold
    }
}












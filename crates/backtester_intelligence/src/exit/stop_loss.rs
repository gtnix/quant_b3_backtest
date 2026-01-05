//! Stop-loss exit policy.

use serde::{Deserialize, Serialize};

use super::policy::ExitPolicy;
use super::types::{ExitContext, ExitReason, ExitTarget, Position};

/// Stop-loss configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopLossConfig {
    /// Loss threshold (negative, e.g., -0.10 for 10% loss)
    #[serde(default = "default_threshold")]
    pub threshold_pct: f64,

    /// Use close price (vs intraday low)
    #[serde(default = "default_true")]
    pub use_close_price: bool,

    /// Whether policy is enabled
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_threshold() -> f64 { -0.10 }
fn default_true() -> bool { true }

impl Default for StopLossConfig {
    fn default() -> Self {
        Self {
            threshold_pct: default_threshold(),
            use_close_price: true,
            enabled: true,
        }
    }
}

/// Stop-loss policy: exit when unrealized loss exceeds threshold.
#[derive(Debug, Clone)]
pub struct StopLossPolicy {
    config: StopLossConfig,
}

impl StopLossPolicy {
    pub fn new(config: StopLossConfig) -> Self {
        Self { config }
    }

    pub fn with_threshold(threshold_pct: f64) -> Self {
        Self::new(StopLossConfig {
            threshold_pct,
            ..Default::default()
        })
    }
}

impl ExitPolicy for StopLossPolicy {
    fn evaluate(&self, position: &Position, _context: &ExitContext) -> Option<ExitTarget> {
        if !self.config.enabled {
            return None;
        }

        let unrealized_return = position.unrealized_return();

        // Trigger if loss exceeds threshold (threshold is negative)
        if unrealized_return <= self.config.threshold_pct {
            Some(ExitTarget::from_position(position, ExitReason::StopLoss, None))
        } else {
            None
        }
    }

    fn name(&self) -> &'static str {
        "stop_loss"
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
    fn test_stop_loss_triggers_on_loss() {
        let policy = StopLossPolicy::with_threshold(-0.10);
        let ctx = make_context();

        // Position with 12% loss
        let pos = Position::new(
            "PETR4",
            Market::BR,
            100,
            dec!(50),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            dec!(44), // 12% loss
        );

        let result = policy.evaluate(&pos, &ctx);
        assert!(result.is_some());
        assert_eq!(result.unwrap().reason, ExitReason::StopLoss);
    }

    #[test]
    fn test_stop_loss_no_trigger_on_small_loss() {
        let policy = StopLossPolicy::with_threshold(-0.10);
        let ctx = make_context();

        // Position with 5% loss (below threshold)
        let pos = Position::new(
            "VALE3",
            Market::BR,
            100,
            dec!(60),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            dec!(57), // 5% loss
        );

        let result = policy.evaluate(&pos, &ctx);
        assert!(result.is_none());
    }

    #[test]
    fn test_stop_loss_no_trigger_on_gain() {
        let policy = StopLossPolicy::with_threshold(-0.10);
        let ctx = make_context();

        // Position with 10% gain
        let pos = Position::new(
            "ITUB4",
            Market::BR,
            100,
            dec!(30),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            dec!(33), // 10% gain
        );

        let result = policy.evaluate(&pos, &ctx);
        assert!(result.is_none());
    }

    #[test]
    fn test_stop_loss_disabled() {
        let policy = StopLossPolicy::new(StopLossConfig {
            enabled: false,
            ..Default::default()
        });
        let ctx = make_context();

        // Position with 20% loss
        let pos = Position::new(
            "BBDC4",
            Market::BR,
            100,
            dec!(20),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            dec!(16), // 20% loss
        );

        let result = policy.evaluate(&pos, &ctx);
        assert!(result.is_none()); // Disabled, no trigger
    }

    #[test]
    fn test_stop_loss_exact_threshold() {
        let policy = StopLossPolicy::with_threshold(-0.10);
        let ctx = make_context();

        // Position with exactly 10% loss
        let pos = Position::new(
            "WEGE3",
            Market::BR,
            100,
            dec!(100),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            dec!(90), // exactly 10% loss
        );

        let result = policy.evaluate(&pos, &ctx);
        assert!(result.is_some()); // Should trigger at exact threshold
    }
}

































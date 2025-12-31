//! Trailing stop exit policy.

use serde::{Deserialize, Serialize};

use super::policy::ExitPolicy;
use super::types::{ExitContext, ExitReason, ExitTarget, Position};

/// Trailing stop configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrailingStopConfig {
    /// Trail percentage from high-water mark (e.g., 0.15 for 15%)
    #[serde(default = "default_trail_pct")]
    pub trail_pct: f64,

    /// Minimum gain to activate trailing stop (e.g., 0.10 for 10%)
    #[serde(default = "default_activation")]
    pub activation_gain_pct: f64,

    /// Whether policy is enabled
    #[serde(default)]
    pub enabled: bool,
}

fn default_trail_pct() -> f64 { 0.15 }
fn default_activation() -> f64 { 0.10 }

impl Default for TrailingStopConfig {
    fn default() -> Self {
        Self {
            trail_pct: default_trail_pct(),
            activation_gain_pct: default_activation(),
            enabled: false, // disabled by default
        }
    }
}

/// Trailing stop policy: exit when price drops X% from high-water mark.
#[derive(Debug, Clone)]
pub struct TrailingStopPolicy {
    config: TrailingStopConfig,
}

impl TrailingStopPolicy {
    pub fn new(config: TrailingStopConfig) -> Self {
        Self { config }
    }

    pub fn with_trail(trail_pct: f64, activation_gain_pct: f64) -> Self {
        Self::new(TrailingStopConfig {
            trail_pct,
            activation_gain_pct,
            enabled: true,
        })
    }

    /// Check if trailing stop is activated (position has reached min gain).
    fn is_activated(&self, position: &Position) -> bool {
        // Calculate max return from cost basis to high-water mark
        if position.cost_basis == rust_decimal::Decimal::ZERO {
            return false;
        }
        let max_return = (position.high_water_mark - position.cost_basis) / position.cost_basis;
        let max_return_f64: f64 = max_return.try_into().unwrap_or(0.0);
        max_return_f64 >= self.config.activation_gain_pct
    }
}

impl ExitPolicy for TrailingStopPolicy {
    fn evaluate(&self, position: &Position, _context: &ExitContext) -> Option<ExitTarget> {
        if !self.config.enabled {
            return None;
        }

        // Check if trailing stop is activated
        if !self.is_activated(position) {
            return None;
        }

        // Check drawdown from high-water mark
        let drawdown = position.drawdown_from_high();

        // Trigger if drawdown exceeds trail percentage (drawdown is negative)
        if drawdown <= -self.config.trail_pct {
            Some(ExitTarget::from_position(position, ExitReason::TrailingStop, None))
        } else {
            None
        }
    }

    fn name(&self) -> &'static str {
        "trailing_stop"
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
    fn test_trailing_stop_triggers() {
        let policy = TrailingStopPolicy::with_trail(0.10, 0.05);
        let ctx = make_context();

        // Position: bought at 100, hit high of 120 (+20%), now at 105 (-12.5% from high)
        let mut pos = Position::new(
            "PETR4",
            Market::BR,
            100,
            dec!(100),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            dec!(105),
        );
        pos.high_water_mark = dec!(120); // Set high-water mark

        let result = policy.evaluate(&pos, &ctx);
        assert!(result.is_some());
        assert_eq!(result.unwrap().reason, ExitReason::TrailingStop);
    }

    #[test]
    fn test_trailing_stop_not_triggered_small_drop() {
        let policy = TrailingStopPolicy::with_trail(0.15, 0.05);
        let ctx = make_context();

        // Position: bought at 100, hit high of 120, now at 110 (-8.3% from high)
        let mut pos = Position::new(
            "VALE3",
            Market::BR,
            100,
            dec!(100),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            dec!(110),
        );
        pos.high_water_mark = dec!(120);

        let result = policy.evaluate(&pos, &ctx);
        assert!(result.is_none()); // 8.3% < 15% trail
    }

    #[test]
    fn test_trailing_stop_not_activated() {
        let policy = TrailingStopPolicy::with_trail(0.10, 0.20);
        let ctx = make_context();

        // Position: bought at 100, high of 110 (+10%), now at 95
        // Activation requires 20% gain, but only reached 10%
        let mut pos = Position::new(
            "ITUB4",
            Market::BR,
            100,
            dec!(100),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            dec!(95),
        );
        pos.high_water_mark = dec!(110);

        let result = policy.evaluate(&pos, &ctx);
        assert!(result.is_none()); // Not activated (only 10% gain, needs 20%)
    }

    #[test]
    fn test_trailing_stop_activated_and_triggered() {
        let policy = TrailingStopPolicy::with_trail(0.10, 0.15);
        let ctx = make_context();

        // Position: bought at 100, hit high of 125 (+25%), now at 110 (-12% from high)
        let mut pos = Position::new(
            "BBDC4",
            Market::BR,
            100,
            dec!(100),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            dec!(110),
        );
        pos.high_water_mark = dec!(125);

        let result = policy.evaluate(&pos, &ctx);
        assert!(result.is_some()); // Activated (25% > 15%) and triggered (12% > 10%)
    }

    #[test]
    fn test_trailing_stop_disabled() {
        let policy = TrailingStopPolicy::new(TrailingStopConfig {
            enabled: false,
            ..Default::default()
        });
        let ctx = make_context();

        let mut pos = Position::new(
            "WEGE3",
            Market::BR,
            100,
            dec!(100),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            dec!(50), // 50% drop
        );
        pos.high_water_mark = dec!(200);

        let result = policy.evaluate(&pos, &ctx);
        assert!(result.is_none()); // Disabled
    }
}




















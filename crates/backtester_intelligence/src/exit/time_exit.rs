//! Time-based exit policy.

use serde::{Deserialize, Serialize};

use super::policy::ExitPolicy;
use super::types::{ExitContext, ExitReason, ExitTarget, Position};

/// Time-based exit configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeExitConfig {
    /// Maximum holding period in days (0 = disabled)
    #[serde(default = "default_max_days")]
    pub max_holding_days: u32,

    /// Whether policy is enabled
    #[serde(default)]
    pub enabled: bool,
}

fn default_max_days() -> u32 { 0 } // disabled by default

impl Default for TimeExitConfig {
    fn default() -> Self {
        Self {
            max_holding_days: default_max_days(),
            enabled: false, // disabled by default
        }
    }
}

/// Time-based exit policy: exit after max holding period.
#[derive(Debug, Clone)]
pub struct TimeExitPolicy {
    config: TimeExitConfig,
}

impl TimeExitPolicy {
    pub fn new(config: TimeExitConfig) -> Self {
        Self { config }
    }

    pub fn with_max_days(max_holding_days: u32) -> Self {
        Self::new(TimeExitConfig {
            max_holding_days,
            enabled: max_holding_days > 0,
        })
    }
}

impl ExitPolicy for TimeExitPolicy {
    fn evaluate(&self, position: &Position, context: &ExitContext) -> Option<ExitTarget> {
        if !self.config.enabled || self.config.max_holding_days == 0 {
            return None;
        }

        let days_held = position.days_held(context.date);

        // Trigger if holding period exceeds max
        if days_held >= self.config.max_holding_days as i64 {
            Some(ExitTarget::from_position(position, ExitReason::TimeExit, None))
        } else {
            None
        }
    }

    fn name(&self) -> &'static str {
        "time_exit"
    }

    fn is_enabled(&self) -> bool {
        self.config.enabled && self.config.max_holding_days > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::Market;
    use chrono::NaiveDate;
    use rust_decimal_macros::dec;

    #[test]
    fn test_time_exit_triggers_after_max_days() {
        let policy = TimeExitPolicy::with_max_days(30);
        let ctx = ExitContext::new(
            NaiveDate::from_ymd_opt(2025, 2, 5).unwrap(), // 35 days after entry
            dec!(1_000_000),
            dec!(1_000_000),
            Market::BR,
        );

        let pos = Position::new(
            "PETR4",
            Market::BR,
            100,
            dec!(50),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            dec!(55),
        );

        let result = policy.evaluate(&pos, &ctx);
        assert!(result.is_some());
        assert_eq!(result.unwrap().reason, ExitReason::TimeExit);
    }

    #[test]
    fn test_time_exit_no_trigger_before_max() {
        let policy = TimeExitPolicy::with_max_days(30);
        let ctx = ExitContext::new(
            NaiveDate::from_ymd_opt(2025, 1, 20).unwrap(), // 19 days after entry
            dec!(1_000_000),
            dec!(1_000_000),
            Market::BR,
        );

        let pos = Position::new(
            "VALE3",
            Market::BR,
            100,
            dec!(60),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            dec!(65),
        );

        let result = policy.evaluate(&pos, &ctx);
        assert!(result.is_none());
    }

    #[test]
    fn test_time_exit_exact_max_days() {
        let policy = TimeExitPolicy::with_max_days(30);
        let ctx = ExitContext::new(
            NaiveDate::from_ymd_opt(2025, 1, 31).unwrap(), // exactly 30 days after entry
            dec!(1_000_000),
            dec!(1_000_000),
            Market::BR,
        );

        let pos = Position::new(
            "ITUB4",
            Market::BR,
            100,
            dec!(30),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            dec!(32),
        );

        let result = policy.evaluate(&pos, &ctx);
        assert!(result.is_some()); // Should trigger at exact max
    }

    #[test]
    fn test_time_exit_disabled() {
        let policy = TimeExitPolicy::new(TimeExitConfig {
            max_holding_days: 30,
            enabled: false,
        });
        let ctx = ExitContext::new(
            NaiveDate::from_ymd_opt(2025, 3, 1).unwrap(), // 60 days after entry
            dec!(1_000_000),
            dec!(1_000_000),
            Market::BR,
        );

        let pos = Position::new(
            "BBDC4",
            Market::BR,
            100,
            dec!(20),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            dec!(22),
        );

        let result = policy.evaluate(&pos, &ctx);
        assert!(result.is_none()); // Disabled, no trigger
    }

    #[test]
    fn test_time_exit_zero_days_disabled() {
        let policy = TimeExitPolicy::with_max_days(0);
        let ctx = ExitContext::new(
            NaiveDate::from_ymd_opt(2025, 6, 1).unwrap(),
            dec!(1_000_000),
            dec!(1_000_000),
            Market::BR,
        );

        let pos = Position::new(
            "WEGE3",
            Market::BR,
            100,
            dec!(40),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            dec!(45),
        );

        let result = policy.evaluate(&pos, &ctx);
        assert!(result.is_none()); // max_days=0 means disabled
    }
}
















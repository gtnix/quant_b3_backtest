//! Exit policy trait and configuration.

use serde::{Deserialize, Serialize};

use super::types::{ExitContext, ExitTarget, Position};

/// Trait for exit policies.
pub trait ExitPolicy: Send + Sync {
    /// Evaluate whether a position should be exited.
    /// Returns Some(ExitTarget) if exit is triggered, None otherwise.
    fn evaluate(&self, position: &Position, context: &ExitContext) -> Option<ExitTarget>;

    /// Policy name for logging.
    fn name(&self) -> &'static str;

    /// Whether this policy is enabled.
    fn is_enabled(&self) -> bool {
        true
    }
}

/// Combined exit policy configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExitPolicyConfig {
    /// Enable stop-loss
    #[serde(default = "default_true")]
    pub enable_stop_loss: bool,

    /// Enable take-profit
    #[serde(default = "default_true")]
    pub enable_take_profit: bool,

    /// Enable time-based exit
    #[serde(default = "default_true")]
    pub enable_time_exit: bool,

    /// Enable trailing stop
    #[serde(default)]
    pub enable_trailing_stop: bool,

    /// Stop-loss threshold (negative, e.g., -0.10 for 10% loss)
    #[serde(default = "default_stop_loss")]
    pub stop_loss_pct: f64,

    /// Take-profit threshold (positive, e.g., 0.30 for 30% gain)
    #[serde(default = "default_take_profit")]
    pub take_profit_pct: f64,

    /// Max holding days (0 = disabled)
    #[serde(default = "default_max_holding")]
    pub max_holding_days: u32,

    /// Trailing stop percentage (e.g., 0.15 for 15% from high)
    #[serde(default = "default_trailing_stop")]
    pub trailing_stop_pct: f64,

    /// Trailing stop activation threshold (min gain before activation)
    #[serde(default = "default_trailing_activation")]
    pub trailing_activation_pct: f64,

    /// Use close price for triggers (vs intraday high/low)
    #[serde(default = "default_true")]
    pub use_close_price: bool,
}

fn default_true() -> bool { true }
fn default_stop_loss() -> f64 { -0.10 } // 10% loss
fn default_take_profit() -> f64 { 0.30 } // 30% gain
fn default_max_holding() -> u32 { 0 } // disabled
fn default_trailing_stop() -> f64 { 0.15 } // 15% from high
fn default_trailing_activation() -> f64 { 0.10 } // 10% gain to activate

impl Default for ExitPolicyConfig {
    fn default() -> Self {
        Self {
            enable_stop_loss: true,
            enable_take_profit: true,
            enable_time_exit: true,
            enable_trailing_stop: false,
            stop_loss_pct: default_stop_loss(),
            take_profit_pct: default_take_profit(),
            max_holding_days: default_max_holding(),
            trailing_stop_pct: default_trailing_stop(),
            trailing_activation_pct: default_trailing_activation(),
            use_close_price: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ExitPolicyConfig::default();
        assert!(config.enable_stop_loss);
        assert!(config.enable_take_profit);
        assert_eq!(config.stop_loss_pct, -0.10);
        assert_eq!(config.take_profit_pct, 0.30);
    }
}












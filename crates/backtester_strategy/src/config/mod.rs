//! Strategy configuration - TOML-based strategy definition.

mod loader;
mod validator;

pub use loader::*;
pub use validator::*;

use crate::blocks::BlockParams;
use serde::{Deserialize, Serialize};

/// Strategy configuration loaded from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    /// Strategy metadata
    pub strategy: StrategyMetadata,
    /// Pipeline steps in execution order
    pub pipeline: Vec<PipelineStep>,
    /// Rebalance configuration
    #[serde(default)]
    pub rebalance: RebalanceConfig,
    /// Portfolio constraints
    #[serde(default)]
    pub constraints: StrategyConstraints,
    /// Default parameters (can be overridden per step)
    #[serde(default)]
    pub defaults: BlockParams,
}

/// Strategy metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyMetadata {
    pub id: String,
    #[serde(default = "default_version")]
    pub version: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub author: String,
}

fn default_version() -> String {
    "1.0.0".into()
}

/// Single pipeline step configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStep {
    /// Block type (selection, entry, exit, sizing, filter)
    #[serde(rename = "type")]
    pub step_type: String,
    /// Block ID to use (e.g., "momentum", "rsi", "equal_weight")
    pub block_id: String,
    /// Parameters for this block
    #[serde(default)]
    pub params: BlockParams,
    /// Whether this step is enabled
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_true() -> bool {
    true
}

/// Rebalance configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalanceConfig {
    /// Frequency: "daily", "weekly", "monthly"
    #[serde(default = "default_frequency")]
    pub frequency: String,
    /// Day for weekly rebalance (e.g., "friday")
    #[serde(default)]
    pub day: Option<String>,
    /// Day of month for monthly rebalance (1-28)
    #[serde(default)]
    pub day_of_month: Option<u8>,
}

fn default_frequency() -> String {
    "weekly".into()
}

impl Default for RebalanceConfig {
    fn default() -> Self {
        Self {
            frequency: "weekly".into(),
            day: Some("friday".into()),
            day_of_month: None,
        }
    }
}

/// Portfolio constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConstraints {
    /// Maximum weight per asset (e.g., 0.20 for 20%)
    #[serde(default = "default_max_weight")]
    pub max_weight_per_asset: f64,
    /// Minimum liquidity in BRL
    #[serde(default = "default_min_liquidity")]
    pub min_liquidity_brl: f64,
    /// Maximum portfolio volatility (annualized)
    #[serde(default)]
    pub max_portfolio_vol: Option<f64>,
    /// Maximum number of positions
    #[serde(default)]
    pub max_positions: Option<usize>,
    /// Minimum number of positions
    #[serde(default)]
    pub min_positions: Option<usize>,
    /// Maximum sector concentration
    #[serde(default)]
    pub max_sector_weight: Option<f64>,
}

fn default_max_weight() -> f64 {
    0.20
}
fn default_min_liquidity() -> f64 {
    500_000.0
}

impl Default for StrategyConstraints {
    fn default() -> Self {
        Self {
            max_weight_per_asset: 0.20,
            min_liquidity_brl: 500_000.0,
            max_portfolio_vol: None,
            max_positions: None,
            min_positions: None,
            max_sector_weight: None,
        }
    }
}

impl StrategyConfig {
    /// Get enabled pipeline steps.
    pub fn enabled_steps(&self) -> Vec<&PipelineStep> {
        self.pipeline.iter().filter(|s| s.enabled).collect()
    }

    /// Get steps by type.
    pub fn steps_by_type(&self, step_type: &str) -> Vec<&PipelineStep> {
        self.pipeline
            .iter()
            .filter(|s| s.enabled && s.step_type == step_type)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_strategy_config() {
        let toml_str = r#"
[strategy]
id = "test_strategy"
version = "1.0.0"
description = "Test strategy"

[[pipeline]]
type = "selection"
block_id = "momentum"
params = { lookback_days = 126, top_pct = 20 }

[[pipeline]]
type = "sizing"
block_id = "equal_weight"
params = { max_weight = 0.20 }

[rebalance]
frequency = "weekly"
day = "friday"

[constraints]
max_weight_per_asset = 0.20
"#;

        let config: StrategyConfig = toml::from_str(toml_str).unwrap();
        
        assert_eq!(config.strategy.id, "test_strategy");
        assert_eq!(config.pipeline.len(), 2);
        assert_eq!(config.pipeline[0].block_id, "momentum");
        assert_eq!(config.rebalance.frequency, "weekly");
    }
}


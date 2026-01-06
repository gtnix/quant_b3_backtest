//! Trade Parameters Module (TPM) - Strategy Configuration Loader
//!
//! This module provides types and functions for loading, validating, and working with
//! trading strategy configurations defined in TOML files.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

// =============================================================================
// Error Types
// =============================================================================

#[derive(Error, Debug)]
pub enum TpmError {
    #[error("Failed to read config file: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Failed to parse TOML: {0}")]
    TomlError(#[from] toml::de::Error),

    #[error("Invalid strategy configuration: {0}")]
    ValidationError(String),

    #[error("Strategy not found: {0}")]
    StrategyNotFound(String),

    #[error("Glob pattern error: {0}")]
    GlobError(#[from] glob::PatternError),
}

pub type TpmResult<T> = Result<T, TpmError>;

// =============================================================================
// Core Types
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    pub metadata: Metadata,
    pub timeframe: Timeframe,
    pub strategy: Strategy,
    pub parameters: HashMap<String, toml::Value>,
    pub entry_rules: EntryRules,
    pub exit_rules: ExitRules,
    pub position_sizing: PositionSizing,
    pub risk_management: RiskManagement,
    pub execution: Execution,
    pub validation: Validation,
    pub optimization: Optimization,
    pub data_requirements: DataRequirements,
    pub universe: Universe,
    #[serde(default)]
    pub notes: Option<Notes>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    pub strategy_id: String,
    pub name: String,
    pub description: String,
    pub version: String,
    pub risk_profile: RiskProfile,
    pub family: StrategyFamily,
    pub asset_classes: Vec<String>,
    pub markets: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RiskProfile {
    Conservative,
    Moderate,
    Aggressive,
    VeryAggressive,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StrategyFamily {
    Intraday,
    Swing,
    Position,
    Pair,
    Portfolio,
    Momentum,
    MeanReversion,
    Breakout,
    SectorRotation,
    Factor,
    Seasonal,
    Volatility,
    EventDriven,
    BuyHold,
    MultiStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timeframe {
    pub bar_interval: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_window_years: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_window_months: Option<u32>,
    pub lookback_bars: u32,
    pub min_history_bars: u32,
    pub holding_period_min: u32,
    pub holding_period_max: u32,
    pub rebalancing_frequency: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Strategy {
    #[serde(rename = "type")]
    pub strategy_type: String,
    pub direction: String,
    pub num_assets: u32,
    pub entry_logic: String,
    pub exit_logic: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntryRules {
    pub long_condition: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub short_condition: Option<String>,
    pub entry_delay_bars: u32,
    pub entry_timing: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExitRules {
    pub exit_methods: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profit_target_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profit_target_value: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_loss_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_loss_value: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trailing_stop_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trailing_stop_value: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trailing_stop_activation: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_holding_bars: Option<u32>,
    #[serde(default)]
    pub exit_on_opposite_signal: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionSizing {
    pub method: String,
    pub risk_per_trade_pct: f64,
    pub max_position_pct: f64,
    pub min_position_pct: f64,
    pub max_leverage: f64,
    #[serde(default)]
    pub scale_in_enabled: bool,
    #[serde(default)]
    pub scale_out_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskManagement {
    pub max_open_positions: u32,
    pub max_drawdown_pct: f64,
    pub daily_loss_limit_pct: f64,
    pub max_position_correlation: f64,
    pub max_sector_exposure_pct: f64,
    pub max_long_exposure_pct: f64,
    pub max_short_exposure_pct: f64,
    pub max_net_exposure_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Execution {
    pub slippage_model: String,
    pub slippage_value: f64,
    pub commission_type: String,
    pub commission_value: f64,
    #[serde(default)]
    pub market_impact_enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub market_impact_model: Option<String>,
    pub default_order_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Validation {
    pub train_test_split: f64,
    #[serde(default)]
    pub wfa_enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wfa_num_folds: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wfa_is_ratio: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wfa_purge_days: Option<u32>,
    #[serde(default)]
    pub pbo_enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pbo_num_permutations: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pbo_significance_level: Option<f64>,
    pub min_trades_total: u32,
    pub min_trades_is: u32,
    pub min_trades_oos: u32,
    pub min_sharpe_ratio_oos: f64,
    pub max_pbo: f64,
    pub max_degradation_is_to_oos: f64,
    pub min_win_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Optimization {
    pub population_size: u32,
    pub max_generations: u32,
    pub complexity_tier: String,
    pub fitness_sharpe_weight: f64,
    pub fitness_cagr_weight: f64,
    pub fitness_drawdown_weight: f64,
    pub fitness_calmar_weight: f64,
    #[serde(default)]
    pub low_trades_penalty: bool,
    pub min_trades_for_no_penalty: u32,
    #[serde(default)]
    pub extreme_turnover_penalty: bool,
    pub max_turnover_annual: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRequirements {
    pub required_fields: Vec<String>,
    pub required_indicators: Vec<String>,
    pub min_data_completeness: f64,
    pub max_missing_consecutive_bars: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Universe {
    #[serde(rename = "type")]
    pub universe_type: String,
    pub universe_size: u32,
    pub min_avg_daily_volume_usd: f64,
    pub min_market_cap_usd: f64,
    pub min_price: f64,
    pub max_price: f64,
    #[serde(default)]
    pub exclude_sectors: Vec<String>,
    #[serde(default)]
    pub exclude_tickers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Notes {
    #[serde(default)]
    pub references: Vec<String>,
    #[serde(default)]
    pub limitations: Vec<String>,
    #[serde(default)]
    pub improvements: Vec<String>,
    pub author: String,
    pub created_date: String,
    pub last_modified_date: String,
}

// =============================================================================
// Loader Implementation
// =============================================================================

pub struct TpmLoader {
    strategies_dir: PathBuf,
    cache: HashMap<String, StrategyConfig>,
}

impl TpmLoader {
    /// Create a new TPM loader with the specified strategies directory
    pub fn new<P: AsRef<Path>>(strategies_dir: P) -> Self {
        Self {
            strategies_dir: strategies_dir.as_ref().to_path_buf(),
            cache: HashMap::new(),
        }
    }

    /// Load a strategy configuration by ID
    pub fn load(&mut self, strategy_id: &str) -> TpmResult<&StrategyConfig> {
        if !self.cache.contains_key(strategy_id) {
            let config = self.load_from_file(strategy_id)?;
            self.validate(&config)?;
            self.cache.insert(strategy_id.to_string(), config);
        }
        Ok(self.cache.get(strategy_id).unwrap())
    }

    /// Load strategy from file
    fn load_from_file(&self, strategy_id: &str) -> TpmResult<StrategyConfig> {
        let path = self.strategies_dir.join(format!("{}.toml", strategy_id));
        let content = std::fs::read_to_string(&path)?;
        let config: StrategyConfig = toml::from_str(&content)?;
        Ok(config)
    }

    /// Validate strategy configuration
    fn validate(&self, config: &StrategyConfig) -> TpmResult<()> {
        // Validate train/test split
        if config.validation.train_test_split <= 0.0 || config.validation.train_test_split >= 1.0 {
            return Err(TpmError::ValidationError(
                "train_test_split must be between 0 and 1".to_string(),
            ));
        }

        // Validate risk parameters
        if config.position_sizing.risk_per_trade_pct <= 0.0 {
            return Err(TpmError::ValidationError(
                "risk_per_trade_pct must be positive".to_string(),
            ));
        }

        if config.position_sizing.max_position_pct > 100.0 {
            return Err(TpmError::ValidationError(
                "max_position_pct cannot exceed 100%".to_string(),
            ));
        }

        // Validate timeframe
        if config.timeframe.data_window_years.is_none() && config.timeframe.data_window_months.is_none() {
            return Err(TpmError::ValidationError(
                "Either data_window_years or data_window_months must be specified".to_string(),
            ));
        }

        Ok(())
    }

    /// List all available strategy IDs
    pub fn list_strategies(&self) -> TpmResult<Vec<String>> {
        let pattern = self.strategies_dir.join("*.toml");
        let mut strategies = Vec::new();

        for entry in glob::glob(pattern.to_str().unwrap())? {
            let path = entry?;
            if let Some(stem) = path.file_stem() {
                if let Some(name) = stem.to_str() {
                    strategies.push(name.to_string());
                }
            }
        }

        strategies.sort();
        Ok(strategies)
    }

    /// List strategies by family
    pub fn list_by_family(&mut self, family: StrategyFamily) -> TpmResult<Vec<String>> {
        let all_strategies = self.list_strategies()?;
        let mut filtered = Vec::new();

        for strategy_id in all_strategies {
            let config = self.load(&strategy_id)?;
            if config.metadata.family == family {
                filtered.push(strategy_id);
            }
        }

        Ok(filtered)
    }

    /// List strategies by risk profile
    pub fn list_by_risk_profile(&mut self, profile: RiskProfile) -> TpmResult<Vec<String>> {
        let all_strategies = self.list_strategies()?;
        let mut filtered = Vec::new();

        for strategy_id in all_strategies {
            let config = self.load(&strategy_id)?;
            if config.metadata.risk_profile == profile {
                filtered.push(strategy_id);
            }
        }

        Ok(filtered)
    }

    /// Get strategy metadata without loading full config
    pub fn get_metadata(&self, strategy_id: &str) -> TpmResult<Metadata> {
        let config = self.load_from_file(strategy_id)?;
        Ok(config.metadata)
    }

    /// Clear cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Load a single strategy configuration from a file
pub fn load_strategy<P: AsRef<Path>>(path: P) -> TpmResult<StrategyConfig> {
    let content = std::fs::read_to_string(path)?;
    let config: StrategyConfig = toml::from_str(&content)?;
    Ok(config)
}

/// Convert strategy config to JSON
pub fn to_json(config: &StrategyConfig) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(config)
}

/// Convert strategy config to TOML
pub fn to_toml(config: &StrategyConfig) -> Result<String, toml::ser::Error> {
    toml::to_string_pretty(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_load_strategy() {
        // This test would require a sample TOML file
        // Skipping for now
    }

    #[test]
    fn test_validation() {
        // Test validation logic
    }
}

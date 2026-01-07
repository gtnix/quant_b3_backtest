//! Type definitions for the Parameter Universe System.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Strategy family classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

impl StrategyFamily {
    /// Returns all strategy families.
    pub fn all() -> &'static [StrategyFamily] {
        &[
            Self::Intraday,
            Self::Swing,
            Self::Position,
            Self::Pair,
            Self::Portfolio,
            Self::Momentum,
            Self::MeanReversion,
            Self::Breakout,
            Self::SectorRotation,
            Self::Factor,
            Self::Seasonal,
            Self::Volatility,
            Self::EventDriven,
            Self::BuyHold,
            Self::MultiStrategy,
        ]
    }

    /// Returns the family name as a string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Intraday => "intraday",
            Self::Swing => "swing",
            Self::Position => "position",
            Self::Pair => "pair",
            Self::Portfolio => "portfolio",
            Self::Momentum => "momentum",
            Self::MeanReversion => "mean_reversion",
            Self::Breakout => "breakout",
            Self::SectorRotation => "sector_rotation",
            Self::Factor => "factor",
            Self::Seasonal => "seasonal",
            Self::Volatility => "volatility",
            Self::EventDriven => "event_driven",
            Self::BuyHold => "buy_hold",
            Self::MultiStrategy => "multi_strategy",
        }
    }
}

/// Complexity tier classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComplexityTier {
    #[serde(rename = "tier1_fast")]
    Tier1Fast,
    #[serde(rename = "tier2_medium")]
    Tier2Medium,
    #[serde(rename = "tier3_slow")]
    Tier3Slow,
    #[serde(rename = "tier4_very_slow")]
    Tier4VerySlow,
}

/// Training strategy configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStrategyConfig {
    pub strategy: TrainingStrategyMetadata,
    pub validation: ValidationConfig,
    pub requirements: TrainingRequirements,
    pub complexity: ComplexityInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStrategyMetadata {
    pub id: String,
    pub name: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    pub method: String,
    #[serde(default)]
    pub wfa_enabled: bool,
    #[serde(default)]
    pub wfa_num_folds: Option<u32>,
    #[serde(default)]
    pub wfa_is_ratio: Option<f64>,
    #[serde(default)]
    pub wfa_purge_days: Option<u32>,
    #[serde(default = "default_train_test_split")]
    pub train_test_split: f64,
    #[serde(default)]
    pub pbo_enabled: bool,
    #[serde(default)]
    pub pbo_num_permutations: Option<u32>,
    #[serde(default)]
    pub pbo_significance_level: Option<f64>,
    #[serde(default)]
    pub rolling: bool,
}

fn default_train_test_split() -> f64 {
    0.65
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRequirements {
    pub min_data_years: u32,
    pub min_trades_is: u32,
    pub min_trades_oos: u32,
    pub max_pbo: f64,
    pub max_degradation_is_to_oos: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityInfo {
    pub tier: ComplexityTier,
    #[serde(default = "default_runtime_multiplier")]
    pub estimated_runtime_multiplier: f64,
}

fn default_runtime_multiplier() -> f64 {
    1.0
}

/// Training tech configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingTechConfig {
    pub tech: TrainingTechMetadata,
    pub resources: ResourceConfig,
    pub evolution: EvolutionLimits,
    pub allowed_complexity: AllowedComplexity,
    #[serde(default)]
    pub scheduling: SchedulingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingTechMetadata {
    pub id: String,
    pub name: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    pub workers: usize,
    pub max_memory_gb: u32,
    pub timeout_minutes: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionLimits {
    pub population_size: usize,
    pub max_generations: u32,
    pub convergence_generations: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllowedComplexity {
    pub tiers: Vec<ComplexityTier>,
    pub max_parameters_to_optimize: usize,
    pub max_universe_size: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchedulingConfig {
    #[serde(default)]
    pub priority: String,
    #[serde(default)]
    pub preemptible: bool,
    #[serde(default)]
    pub cluster_mode: bool,
}

/// Parameter bounds for a strategy family.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterBoundsConfig {
    pub bounds: BoundsDefinition,
    pub complexity: BoundsComplexity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundsDefinition {
    pub family: String,
    pub description: String,
    #[serde(flatten)]
    pub parameters: HashMap<String, toml::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundsComplexity {
    pub tier: ComplexityTier,
    pub typical_parameters: usize,
    pub evaluation_time_ms: u64,
}

/// Numeric bound with min, max, step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericBound {
    pub min: f64,
    pub max: f64,
    #[serde(default = "default_step")]
    pub step: f64,
    #[serde(default)]
    pub default: Option<f64>,
}

fn default_step() -> f64 {
    1.0
}

/// Universe restrictions from risk profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniverseRestrictions {
    pub allowed_strategy_families: Vec<String>,
    pub max_parameters_to_optimize: usize,
    pub allowed_complexity_tiers: Vec<ComplexityTier>,
    pub max_population_size: usize,
    pub max_generations: u32,
    #[serde(default)]
    pub min_wfa_folds: Option<u32>,
    #[serde(default)]
    pub min_pbo_permutations: Option<u32>,
    #[serde(default)]
    pub max_pbo_threshold: Option<f64>,
}

/// Complete universe configuration for a campaign.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniverseConfig {
    /// Reference to robustness profile name
    pub robustness_profile: String,
    /// Reference to training strategy name
    pub training_strategy: String,
    /// Reference to training tech name
    pub training_tech: String,
    /// Training model (strategy family) - can be single or list
    #[serde(default)]
    pub training_model: TrainingModel,
    /// Optional overrides
    #[serde(default)]
    pub overrides: Option<UniverseOverrides>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TrainingModel {
    Single(String),
    Multiple(Vec<String>),
}

impl Default for TrainingModel {
    fn default() -> Self {
        Self::Single("swing".to_string())
    }
}

impl TrainingModel {
    pub fn families(&self) -> Vec<&str> {
        match self {
            Self::Single(s) => vec![s.as_str()],
            Self::Multiple(v) => v.iter().map(|s| s.as_str()).collect(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UniverseOverrides {
    #[serde(default)]
    pub max_parameters: Option<usize>,
    #[serde(default)]
    pub allowed_indicators: Option<Vec<String>>,
    #[serde(default)]
    pub max_data_window_years: Option<u32>,
}

/// Strategy registry entry - defines a valid strategy in the universe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyRegistryEntry {
    pub family: String,
    pub variant: String,
    pub timeframe: String,
    pub hypothesis: String,
    pub complexity_tier: ComplexityTier,
    pub risk_profiles: Vec<String>,
    #[serde(default)]
    pub holding_period: Option<HoldingPeriod>,
    #[serde(default)]
    pub min_trades_required: Option<u32>,
    #[serde(default)]
    pub rebalance_frequency: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum HoldingPeriod {
    Hours { min_hours: u32, max_hours: u32 },
    Days { min_days: u32, max_days: u32 },
    Weeks { min_weeks: u32, max_weeks: u32 },
    Months { min_months: u32, max_months: u32 },
    Years { min_years: u32, max_years: u32 },
}

/// Strategy registry metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyRegistryMetadata {
    pub version: String,
    pub total_strategies: u32,
    pub last_updated: String,
    pub description: String,
}

/// Complete strategy registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyRegistry {
    pub metadata: StrategyRegistryMetadata,
    pub strategies: std::collections::HashMap<String, StrategyRegistryEntry>,
    #[serde(default)]
    pub index: Option<StrategyIndex>,
}

/// Index for quick lookups by various criteria.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StrategyIndex {
    #[serde(default)]
    pub by_family: std::collections::HashMap<String, Vec<String>>,
    #[serde(default)]
    pub by_timeframe: std::collections::HashMap<String, Vec<String>>,
    #[serde(default)]
    pub by_hypothesis: std::collections::HashMap<String, Vec<String>>,
    #[serde(default)]
    pub by_risk_profile: std::collections::HashMap<String, Vec<String>>,
}

impl StrategyRegistry {
    /// Check if a strategy exists in the registry.
    pub fn contains(&self, strategy_id: &str) -> bool {
        self.strategies.contains_key(strategy_id)
    }

    /// Get a strategy by ID.
    pub fn get(&self, strategy_id: &str) -> Option<&StrategyRegistryEntry> {
        self.strategies.get(strategy_id)
    }

    /// Get all strategy IDs for a family.
    pub fn get_by_family(&self, family: &str) -> Vec<&str> {
        if let Some(index) = &self.index {
            if let Some(ids) = index.by_family.get(family) {
                return ids.iter().map(|s| s.as_str()).collect();
            }
        }
        // Fallback: iterate all strategies
        self.strategies
            .iter()
            .filter(|(_, entry)| entry.family == family)
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// Get all strategy IDs compatible with a risk profile.
    pub fn get_by_risk_profile(&self, risk_profile: &str) -> Vec<&str> {
        if let Some(index) = &self.index {
            if let Some(ids) = index.by_risk_profile.get(risk_profile) {
                return ids.iter().map(|s| s.as_str()).collect();
            }
        }
        self.strategies
            .iter()
            .filter(|(_, entry)| entry.risk_profiles.contains(&risk_profile.to_string()))
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// Get all valid strategies for a given configuration.
    pub fn get_valid_strategies(
        &self,
        family: Option<&str>,
        risk_profile: Option<&str>,
        timeframe: Option<&str>,
    ) -> Vec<&str> {
        self.strategies
            .iter()
            .filter(|(_, entry)| {
                family.map_or(true, |f| entry.family == f)
                    && risk_profile.map_or(true, |r| entry.risk_profiles.contains(&r.to_string()))
                    && timeframe.map_or(true, |t| entry.timeframe == t)
            })
            .map(|(id, _)| id.as_str())
            .collect()
    }
}

/// Timeframe profile configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeframeProfile {
    pub label: String,
    pub holding_period: String,
    pub data_requirement: String,
    #[serde(default)]
    pub data_window_months: Option<u32>,
    #[serde(default)]
    pub data_window_years: Option<u32>,
    pub families: Vec<String>,
    pub hypotheses: Vec<String>,
    pub complexity_tiers: Vec<String>,
    pub strategies: Vec<String>,
    #[serde(default)]
    pub min_trades_is: Option<u32>,
    #[serde(default)]
    pub min_trades_oos: Option<u32>,
    #[serde(default)]
    pub train_test_split: Option<f64>,
    #[serde(default)]
    pub wfa_enabled: Option<bool>,
    #[serde(default)]
    pub wfa_folds: Option<u32>,
}

/// Compatibility matrix between axes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityMatrix {
    pub metadata: CompatibilityMetadata,
    pub robustness_to_training_strategy: HashMap<String, Vec<String>>,
    pub training_model_to_robustness: HashMap<String, Vec<String>>,
    pub training_tech_to_complexity: HashMap<String, Vec<ComplexityTier>>,
    pub training_model_complexity: HashMap<String, ComplexityTier>,
    pub training_strategy_min_data: HashMap<String, u32>,
    pub training_model_data_window: HashMap<String, DataWindowConfig>,
    pub training_model_min_trades: HashMap<String, MinTradesConfig>,
    pub validation_rules: ValidationRules,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityMetadata {
    pub version: String,
    pub description: String,
    pub last_updated: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataWindowConfig {
    pub min: f64,
    pub optimal: f64,
    pub max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinTradesConfig {
    pub is: u32,
    pub oos: u32,
    pub total: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRules {
    #[serde(default)]
    pub check_tech_complexity: bool,
    #[serde(default)]
    pub check_data_window: bool,
    #[serde(default)]
    pub check_robustness_compatibility: bool,
    #[serde(default)]
    pub check_strategy_compatibility: bool,
    #[serde(default)]
    pub apply_market_adjustments: bool,
}


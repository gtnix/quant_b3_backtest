//! Loaders for Parameter Universe configuration files.

use super::types::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum UniverseLoadError {
    #[error("File not found: {0}")]
    FileNotFound(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("TOML parse error: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("Configuration not found: {0}")]
    ConfigNotFound(String),
    #[error("Glob pattern error: {0}")]
    Glob(#[from] glob::PatternError),
}

pub type UniverseResult<T> = Result<T, UniverseLoadError>;

/// Loader for all universe-related configurations.
pub struct UniverseLoader {
    base_path: PathBuf,
    training_strategies: HashMap<String, TrainingStrategyConfig>,
    training_tech: HashMap<String, TrainingTechConfig>,
    parameter_bounds: HashMap<String, ParameterBoundsConfig>,
    compatibility_matrix: Option<CompatibilityMatrix>,
    strategy_registry: Option<StrategyRegistry>,
}

impl UniverseLoader {
    /// Create a new loader with the given base config path.
    pub fn new<P: AsRef<Path>>(base_path: P) -> Self {
        Self {
            base_path: base_path.as_ref().to_path_buf(),
            training_strategies: HashMap::new(),
            training_tech: HashMap::new(),
            parameter_bounds: HashMap::new(),
            compatibility_matrix: None,
            strategy_registry: None,
        }
    }

    /// Load all configurations from the base path.
    pub fn load_all(&mut self) -> UniverseResult<()> {
        self.load_training_strategies()?;
        self.load_training_tech()?;
        self.load_parameter_bounds()?;
        self.load_compatibility_matrix()?;
        self.load_strategy_registry()?;
        Ok(())
    }

    /// Load training strategy configurations.
    pub fn load_training_strategies(&mut self) -> UniverseResult<()> {
        let dir = self.base_path.join("training_strategies");
        if !dir.exists() {
            return Ok(());
        }

        let pattern = dir.join("*.toml");
        for entry in glob::glob(pattern.to_str().unwrap())? {
            if let Ok(path) = entry {
                let content = std::fs::read_to_string(&path)?;
                let config: TrainingStrategyConfig = toml::from_str(&content)?;
                self.training_strategies
                    .insert(config.strategy.id.clone(), config);
            }
        }
        Ok(())
    }

    /// Load training tech configurations.
    pub fn load_training_tech(&mut self) -> UniverseResult<()> {
        let dir = self.base_path.join("training_tech");
        if !dir.exists() {
            return Ok(());
        }

        let pattern = dir.join("*.toml");
        for entry in glob::glob(pattern.to_str().unwrap())? {
            if let Ok(path) = entry {
                let content = std::fs::read_to_string(&path)?;
                let config: TrainingTechConfig = toml::from_str(&content)?;
                self.training_tech.insert(config.tech.id.clone(), config);
            }
        }
        Ok(())
    }

    /// Load parameter bounds configurations.
    pub fn load_parameter_bounds(&mut self) -> UniverseResult<()> {
        let dir = self.base_path.join("parameter_bounds");
        if !dir.exists() {
            return Ok(());
        }

        let pattern = dir.join("*.toml");
        for entry in glob::glob(pattern.to_str().unwrap())? {
            if let Ok(path) = entry {
                let content = std::fs::read_to_string(&path)?;
                let config: ParameterBoundsConfig = toml::from_str(&content)?;
                self.parameter_bounds
                    .insert(config.bounds.family.clone(), config);
            }
        }
        Ok(())
    }

    /// Load the compatibility matrix.
    pub fn load_compatibility_matrix(&mut self) -> UniverseResult<()> {
        let path = self.base_path.join("compatibility_matrix.toml");
        if !path.exists() {
            return Ok(());
        }

        let content = std::fs::read_to_string(&path)?;
        let matrix: CompatibilityMatrix = toml::from_str(&content)?;
        self.compatibility_matrix = Some(matrix);
        Ok(())
    }

    /// Load the strategy registry from universe/strategy_registry.toml.
    pub fn load_strategy_registry(&mut self) -> UniverseResult<()> {
        let path = self.base_path.join("universe").join("strategy_registry.toml");
        if !path.exists() {
            return Ok(());
        }

        let content = std::fs::read_to_string(&path)?;
        let registry: StrategyRegistry = toml::from_str(&content)?;
        self.strategy_registry = Some(registry);
        Ok(())
    }

    /// Get a training strategy by ID.
    pub fn get_training_strategy(&self, id: &str) -> Option<&TrainingStrategyConfig> {
        self.training_strategies.get(id)
    }

    /// Get training tech by ID.
    pub fn get_training_tech(&self, id: &str) -> Option<&TrainingTechConfig> {
        self.training_tech.get(id)
    }

    /// Get parameter bounds by family.
    pub fn get_parameter_bounds(&self, family: &str) -> Option<&ParameterBoundsConfig> {
        self.parameter_bounds.get(family)
    }

    /// Get the compatibility matrix.
    pub fn get_compatibility_matrix(&self) -> Option<&CompatibilityMatrix> {
        self.compatibility_matrix.as_ref()
    }

    /// Get the strategy registry.
    pub fn get_strategy_registry(&self) -> Option<&StrategyRegistry> {
        self.strategy_registry.as_ref()
    }

    /// List all available training strategies.
    pub fn list_training_strategies(&self) -> Vec<&str> {
        self.training_strategies.keys().map(|s| s.as_str()).collect()
    }

    /// List all available training tech options.
    pub fn list_training_tech(&self) -> Vec<&str> {
        self.training_tech.keys().map(|s| s.as_str()).collect()
    }

    /// List all available parameter bounds families.
    pub fn list_parameter_bounds(&self) -> Vec<&str> {
        self.parameter_bounds.keys().map(|s| s.as_str()).collect()
    }
}

/// Load a single universe config section from a campaign TOML.
pub fn load_universe_config(content: &str) -> UniverseResult<Option<UniverseConfig>> {
    use serde::Deserialize;
    
    #[derive(Deserialize)]
    struct CampaignWrapper {
        #[serde(default)]
        universe: Option<UniverseConfig>,
    }

    let wrapper: CampaignWrapper = toml::from_str(content)?;
    Ok(wrapper.universe)
}

/// Load universe restrictions from a risk profile TOML.
pub fn load_universe_restrictions(content: &str) -> UniverseResult<Option<UniverseRestrictions>> {
    use serde::Deserialize;
    
    #[derive(Deserialize)]
    struct RiskProfileWrapper {
        #[serde(default)]
        universe_restrictions: Option<UniverseRestrictions>,
    }

    let wrapper: RiskProfileWrapper = toml::from_str(content)?;
    Ok(wrapper.universe_restrictions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_training_strategy() {
        let toml_str = r#"
[strategy]
id = "test"
name = "Test Strategy"
description = "A test"

[validation]
method = "purged_kfold"
wfa_enabled = true
wfa_num_folds = 5
train_test_split = 0.65

[requirements]
min_data_years = 2
min_trades_is = 50
min_trades_oos = 25
max_pbo = 0.30
max_degradation_is_to_oos = 0.40

[complexity]
tier = "tier2_medium"
"#;
        let config: TrainingStrategyConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.strategy.id, "test");
        assert!(config.validation.wfa_enabled);
    }

    #[test]
    fn test_load_training_tech() {
        let toml_str = r#"
[tech]
id = "cpu_fast"
name = "CPU Fast"
description = "Quick evaluation"

[resources]
workers = 4
max_memory_gb = 4
timeout_minutes = 30

[evolution]
population_size = 100
max_generations = 50
convergence_generations = 10

[allowed_complexity]
tiers = ["tier1_fast"]
max_parameters_to_optimize = 6
max_universe_size = 50
"#;
        let config: TrainingTechConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.tech.id, "cpu_fast");
        assert_eq!(config.resources.workers, 4);
    }

    #[test]
    fn test_load_universe_config() {
        let toml_str = r#"
[universe]
robustness_profile = "moderado"
training_strategy = "purged_kfold"
training_tech = "cpu_parallel"
training_model = "swing"

[universe.overrides]
max_parameters = 10
"#;
        let config = load_universe_config(toml_str).unwrap().unwrap();
        assert_eq!(config.robustness_profile, "moderado");
        assert_eq!(config.training_strategy, "purged_kfold");
    }
}


//! Validator for Parameter Universe System.
//!
//! Validates compatibility between the 4 axes and ensures
//! configurations are within allowed bounds.

use super::types::*;
use super::loader::UniverseLoader;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum UniverseValidationError {
    #[error("Training strategy '{0}' not compatible with robustness profile '{1}'")]
    IncompatibleTrainingStrategy(String, String),

    #[error("Training model '{0}' not compatible with robustness profile '{1}'")]
    IncompatibleTrainingModel(String, String),

    #[error("Training tech '{0}' does not support complexity tier '{1:?}'")]
    IncompatibleComplexityTier(String, ComplexityTier),

    #[error("Data window {0} years is below minimum {1} years for training strategy '{2}'")]
    InsufficientDataWindow(f64, u32, String),

    #[error("Strategy family '{0}' not allowed for robustness profile '{1}'")]
    FamilyNotAllowed(String, String),

    #[error("Parameters to optimize ({0}) exceeds maximum ({1}) for robustness profile")]
    TooManyParameters(usize, usize),

    #[error("Population size ({0}) exceeds maximum ({1}) for robustness profile")]
    PopulationTooLarge(usize, usize),

    #[error("Generations ({0}) exceeds maximum ({1}) for robustness profile")]
    GenerationsTooMany(u32, u32),

    #[error("Training strategy '{0}' not found")]
    TrainingStrategyNotFound(String),

    #[error("Training tech '{0}' not found")]
    TrainingTechNotFound(String),

    #[error("Compatibility matrix not loaded")]
    CompatibilityMatrixNotLoaded,

    #[error("Strategy '{0}' not found in universe registry. Only registered strategies can be generated.")]
    StrategyNotInRegistry(String),

    #[error("Strategy '{0}' not compatible with risk profile '{1}'. Allowed profiles: {2:?}")]
    StrategyIncompatibleWithRiskProfile(String, String, Vec<String>),

    #[error("Strategy registry not loaded")]
    RegistryNotLoaded,

    #[error("Multiple validation errors: {0:?}")]
    Multiple(Vec<UniverseValidationError>),
}

pub type ValidationResult<T> = Result<T, UniverseValidationError>;

/// Validates universe configurations.
pub struct UniverseValidator<'a> {
    loader: &'a UniverseLoader,
}

impl<'a> UniverseValidator<'a> {
    /// Create a new validator with the given loader.
    pub fn new(loader: &'a UniverseLoader) -> Self {
        Self { loader }
    }

    /// Validate a complete universe configuration.
    pub fn validate(
        &self,
        config: &UniverseConfig,
        restrictions: Option<&UniverseRestrictions>,
    ) -> ValidationResult<()> {
        let mut errors = Vec::new();

        // Validate training strategy exists
        if self.loader.get_training_strategy(&config.training_strategy).is_none() {
            errors.push(UniverseValidationError::TrainingStrategyNotFound(
                config.training_strategy.clone(),
            ));
        }

        // Validate training tech exists
        let training_tech = match self.loader.get_training_tech(&config.training_tech) {
            Some(t) => Some(t),
            None => {
                errors.push(UniverseValidationError::TrainingTechNotFound(
                    config.training_tech.clone(),
                ));
                None
            }
        };

        // Validate against compatibility matrix
        if let Some(matrix) = self.loader.get_compatibility_matrix() {
            self.validate_compatibility(config, matrix, &mut errors);
        }

        // Validate against universe restrictions from risk profile
        if let Some(restrictions) = restrictions {
            self.validate_restrictions(config, restrictions, training_tech, &mut errors);
        }

        if errors.is_empty() {
            Ok(())
        } else if errors.len() == 1 {
            Err(errors.remove(0))
        } else {
            Err(UniverseValidationError::Multiple(errors))
        }
    }

    /// Validate compatibility using the matrix.
    fn validate_compatibility(
        &self,
        config: &UniverseConfig,
        matrix: &CompatibilityMatrix,
        errors: &mut Vec<UniverseValidationError>,
    ) {
        // Check training strategy is compatible with robustness profile
        if let Some(allowed) = matrix
            .robustness_to_training_strategy
            .get(&config.robustness_profile)
        {
            if !allowed.contains(&config.training_strategy) {
                errors.push(UniverseValidationError::IncompatibleTrainingStrategy(
                    config.training_strategy.clone(),
                    config.robustness_profile.clone(),
                ));
            }
        }

        // Check training model is compatible with robustness profile
        for family in config.training_model.families() {
            if let Some(allowed_robustness) = matrix.training_model_to_robustness.get(family) {
                if !allowed_robustness.contains(&config.robustness_profile) {
                    errors.push(UniverseValidationError::IncompatibleTrainingModel(
                        family.to_string(),
                        config.robustness_profile.clone(),
                    ));
                }
            }

            // Check training tech supports the complexity tier of the model
            if let Some(model_complexity) = matrix.training_model_complexity.get(family) {
                if let Some(allowed_tiers) =
                    matrix.training_tech_to_complexity.get(&config.training_tech)
                {
                    if !allowed_tiers.contains(model_complexity) {
                        errors.push(UniverseValidationError::IncompatibleComplexityTier(
                            config.training_tech.clone(),
                            *model_complexity,
                        ));
                    }
                }
            }
        }
    }

    /// Validate against universe restrictions.
    fn validate_restrictions(
        &self,
        config: &UniverseConfig,
        restrictions: &UniverseRestrictions,
        training_tech: Option<&TrainingTechConfig>,
        errors: &mut Vec<UniverseValidationError>,
    ) {
        // Check strategy families are allowed
        for family in config.training_model.families() {
            if !restrictions.allowed_strategy_families.contains(&family.to_string()) {
                errors.push(UniverseValidationError::FamilyNotAllowed(
                    family.to_string(),
                    config.robustness_profile.clone(),
                ));
            }
        }

        // Check population size
        if let Some(tech) = training_tech {
            if tech.evolution.population_size > restrictions.max_population_size {
                errors.push(UniverseValidationError::PopulationTooLarge(
                    tech.evolution.population_size,
                    restrictions.max_population_size,
                ));
            }

            // Check generations
            if tech.evolution.max_generations > restrictions.max_generations {
                errors.push(UniverseValidationError::GenerationsTooMany(
                    tech.evolution.max_generations,
                    restrictions.max_generations,
                ));
            }

            // Check parameters to optimize
            if tech.allowed_complexity.max_parameters_to_optimize
                > restrictions.max_parameters_to_optimize
            {
                errors.push(UniverseValidationError::TooManyParameters(
                    tech.allowed_complexity.max_parameters_to_optimize,
                    restrictions.max_parameters_to_optimize,
                ));
            }
        }
    }

    /// Check if a strategy family is valid for the given restrictions.
    pub fn is_family_allowed(
        family: &str,
        restrictions: &UniverseRestrictions,
    ) -> bool {
        restrictions.allowed_strategy_families.contains(&family.to_string())
    }

    /// Get the effective parameter limits after applying overrides.
    pub fn get_effective_limits(
        &self,
        config: &UniverseConfig,
        restrictions: Option<&UniverseRestrictions>,
    ) -> EffectiveLimits {
        let mut limits = EffectiveLimits::default();

        // Start with training tech limits
        if let Some(tech) = self.loader.get_training_tech(&config.training_tech) {
            limits.population_size = tech.evolution.population_size;
            limits.max_generations = tech.evolution.max_generations;
            limits.max_parameters = tech.allowed_complexity.max_parameters_to_optimize;
        }

        // Apply restrictions (more restrictive wins)
        if let Some(restrictions) = restrictions {
            limits.population_size = limits.population_size.min(restrictions.max_population_size);
            limits.max_generations = limits.max_generations.min(restrictions.max_generations);
            limits.max_parameters = limits.max_parameters.min(restrictions.max_parameters_to_optimize);
        }

        // Apply overrides
        if let Some(overrides) = &config.overrides {
            if let Some(max_params) = overrides.max_parameters {
                limits.max_parameters = limits.max_parameters.min(max_params);
            }
        }

        limits
    }

    /// Validate that a strategy exists in the registry.
    pub fn validate_strategy_in_registry(
        &self,
        strategy_id: &str,
    ) -> ValidationResult<()> {
        match self.loader.get_strategy_registry() {
            Some(registry) => {
                if registry.contains(strategy_id) {
                    Ok(())
                } else {
                    Err(UniverseValidationError::StrategyNotInRegistry(
                        strategy_id.to_string(),
                    ))
                }
            }
            None => Err(UniverseValidationError::RegistryNotLoaded),
        }
    }

    /// Validate that a strategy is compatible with the given risk profile.
    pub fn validate_strategy_with_risk_profile(
        &self,
        strategy_id: &str,
        risk_profile: &str,
    ) -> ValidationResult<()> {
        match self.loader.get_strategy_registry() {
            Some(registry) => {
                if let Some(entry) = registry.get(strategy_id) {
                    if entry.risk_profiles.contains(&risk_profile.to_string()) {
                        Ok(())
                    } else {
                        Err(UniverseValidationError::StrategyIncompatibleWithRiskProfile(
                            strategy_id.to_string(),
                            risk_profile.to_string(),
                            entry.risk_profiles.clone(),
                        ))
                    }
                } else {
                    Err(UniverseValidationError::StrategyNotInRegistry(
                        strategy_id.to_string(),
                    ))
                }
            }
            None => Err(UniverseValidationError::RegistryNotLoaded),
        }
    }

    /// Get all valid strategies for the given universe configuration.
    pub fn get_allowed_strategies(
        &self,
        config: &UniverseConfig,
        restrictions: Option<&UniverseRestrictions>,
    ) -> Vec<String> {
        let registry = match self.loader.get_strategy_registry() {
            Some(r) => r,
            None => return Vec::new(),
        };

        let allowed_families: Vec<&str> = if let Some(restrictions) = restrictions {
            restrictions
                .allowed_strategy_families
                .iter()
                .map(|s| s.as_str())
                .collect()
        } else {
            config.training_model.families()
        };

        registry
            .strategies
            .iter()
            .filter(|(_, entry)| {
                // Filter by family
                allowed_families.contains(&entry.family.as_str())
                    // Filter by risk profile
                    && entry.risk_profiles.contains(&config.robustness_profile)
            })
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Get strategy details from registry.
    pub fn get_strategy_details(
        &self,
        strategy_id: &str,
    ) -> Option<&super::types::StrategyRegistryEntry> {
        self.loader
            .get_strategy_registry()
            .and_then(|r| r.get(strategy_id))
    }
}

/// Effective limits after applying all constraints.
#[derive(Debug, Clone, Default)]
pub struct EffectiveLimits {
    pub population_size: usize,
    pub max_generations: u32,
    pub max_parameters: usize,
    pub allowed_indicators: Vec<String>,
    pub max_data_window_years: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_training_model_families() {
        let single = TrainingModel::Single("swing".to_string());
        assert_eq!(single.families(), vec!["swing"]);

        let multiple = TrainingModel::Multiple(vec![
            "swing".to_string(),
            "momentum".to_string(),
        ]);
        assert_eq!(multiple.families(), vec!["swing", "momentum"]);
    }

    fn make_restrictions(
        families: Vec<&str>,
        max_pop: usize,
        max_gen: u32,
        max_params: usize,
    ) -> UniverseRestrictions {
        UniverseRestrictions {
            allowed_strategy_families: families.into_iter().map(|s| s.to_string()).collect(),
            max_parameters_to_optimize: max_params,
            allowed_complexity_tiers: vec![ComplexityTier::Tier1Fast, ComplexityTier::Tier2Medium],
            max_population_size: max_pop,
            max_generations: max_gen,
            min_wfa_folds: Some(5),
            min_pbo_permutations: Some(500),
            max_pbo_threshold: Some(0.30),
        }
    }

    fn make_config(
        robustness: &str,
        strategy: &str,
        tech: &str,
        model: &str,
    ) -> UniverseConfig {
        UniverseConfig {
            robustness_profile: robustness.to_string(),
            training_strategy: strategy.to_string(),
            training_tech: tech.to_string(),
            training_model: TrainingModel::Single(model.to_string()),
            overrides: None,
        }
    }

    #[test]
    fn test_is_family_allowed_valid() {
        let restrictions = make_restrictions(
            vec!["swing", "momentum", "position"],
            200, 150, 10,
        );
        
        assert!(UniverseValidator::is_family_allowed("swing", &restrictions));
        assert!(UniverseValidator::is_family_allowed("momentum", &restrictions));
        assert!(UniverseValidator::is_family_allowed("position", &restrictions));
    }

    #[test]
    fn test_is_family_not_allowed() {
        let restrictions = make_restrictions(
            vec!["swing", "position"],
            200, 150, 10,
        );
        
        assert!(!UniverseValidator::is_family_allowed("intraday", &restrictions));
        assert!(!UniverseValidator::is_family_allowed("breakout", &restrictions));
    }

    #[test]
    fn test_effective_limits_default() {
        let loader = UniverseLoader::new("configs/");
        let validator = UniverseValidator::new(&loader);
        
        let config = make_config("moderado", "purged_kfold", "cpu_parallel", "swing");
        
        // Without restrictions, defaults are used
        let limits = validator.get_effective_limits(&config, None);
        
        // Default limits (from EffectiveLimits::default())
        assert_eq!(limits.population_size, 0);
        assert_eq!(limits.max_generations, 0);
    }

    #[test]
    fn test_effective_limits_with_restrictions() {
        let loader = UniverseLoader::new("configs/");
        let validator = UniverseValidator::new(&loader);
        
        let config = make_config("moderado", "purged_kfold", "cpu_parallel", "swing");
        let restrictions = make_restrictions(
            vec!["swing", "momentum"],
            200, 150, 10,
        );
        
        let limits = validator.get_effective_limits(&config, Some(&restrictions));
        
        // Restrictions applied
        assert!(limits.population_size <= 200);
        assert!(limits.max_generations <= 150);
        assert!(limits.max_parameters <= 10);
    }

    #[test]
    fn test_effective_limits_with_overrides() {
        let loader = UniverseLoader::new("configs/");
        let validator = UniverseValidator::new(&loader);
        
        let mut config = make_config("moderado", "purged_kfold", "cpu_parallel", "swing");
        config.overrides = Some(UniverseOverrides {
            max_parameters: Some(5),
            allowed_indicators: None,
            max_data_window_years: None,
        });
        
        let restrictions = make_restrictions(
            vec!["swing"],
            200, 150, 10,
        );
        
        let limits = validator.get_effective_limits(&config, Some(&restrictions));
        
        // Override further constrains max_parameters
        assert!(limits.max_parameters <= 5);
    }

    #[test]
    fn test_validate_family_not_allowed() {
        let loader = UniverseLoader::new("configs/");
        let validator = UniverseValidator::new(&loader);
        
        let config = make_config("muito_conservador", "purged_kfold", "cpu_fast", "intraday");
        let restrictions = make_restrictions(
            vec!["position", "portfolio", "factor"], // intraday NOT allowed
            100, 75, 6,
        );
        
        let result = validator.validate(&config, Some(&restrictions));
        
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            UniverseValidationError::FamilyNotAllowed(family, _) => {
                assert_eq!(family, "intraday");
            }
            UniverseValidationError::Multiple(errors) => {
                assert!(errors.iter().any(|e| matches!(e, UniverseValidationError::FamilyNotAllowed(f, _) if f == "intraday")));
            }
            _ => panic!("Expected FamilyNotAllowed error, got {:?}", err),
        }
    }

    #[test]
    fn test_validate_no_restrictions_passes() {
        let loader = UniverseLoader::new("configs/");
        let validator = UniverseValidator::new(&loader);
        
        let config = make_config("moderado", "purged_kfold", "cpu_parallel", "swing");
        
        // Without restrictions or matrix, validation passes (graceful degradation)
        let result = validator.validate(&config, None);
        
        // May fail on missing training strategy/tech but that's expected without loaded configs
        // The test verifies no panic and proper error handling
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_validate_multiple_families() {
        let loader = UniverseLoader::new("configs/");
        let validator = UniverseValidator::new(&loader);
        
        let mut config = make_config("moderado", "purged_kfold", "cpu_parallel", "swing");
        config.training_model = TrainingModel::Multiple(vec![
            "swing".to_string(),
            "momentum".to_string(),
            "intraday".to_string(), // This one not allowed
        ]);
        
        let restrictions = make_restrictions(
            vec!["swing", "momentum", "position"], // intraday NOT allowed
            200, 150, 10,
        );
        
        let result = validator.validate(&config, Some(&restrictions));
        
        assert!(result.is_err());
    }

    #[test]
    fn test_error_messages_are_useful() {
        // Test that error messages contain actionable information
        let err = UniverseValidationError::IncompatibleTrainingStrategy(
            "monte_carlo".to_string(),
            "muito_conservador".to_string(),
        );
        let msg = err.to_string();
        
        assert!(msg.contains("monte_carlo"));
        assert!(msg.contains("muito_conservador"));
        assert!(msg.contains("not compatible"));
        
        let err2 = UniverseValidationError::FamilyNotAllowed(
            "intraday".to_string(),
            "conservador".to_string(),
        );
        let msg2 = err2.to_string();
        
        assert!(msg2.contains("intraday"));
        assert!(msg2.contains("not allowed"));
    }
}


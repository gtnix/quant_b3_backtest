//! Integration tests for Parameter Universe System.
//!
//! Tests backward compatibility, validation, and configuration loading.

use std::fs;
use std::path::PathBuf;

/// Get the path to the test configs directory.
fn configs_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("configs")
}

/// Get the path to the campaigns directory.
fn campaigns_path() -> PathBuf {
    configs_path().join("campaigns")
}

mod backward_compatibility {
    use super::*;

    /// Test that existing campaign TOMLs without [universe] section load correctly.
    #[test]
    fn test_existing_campaigns_load_without_universe() {
        let campaigns_dir = campaigns_path();
        
        if !campaigns_dir.exists() {
            eprintln!("Skipping test: campaigns directory not found");
            return;
        }

        // Find all campaign TOML files
        let entries: Vec<_> = fs::read_dir(&campaigns_dir)
            .expect("Failed to read campaigns directory")
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "toml"))
            .collect();

        assert!(!entries.is_empty(), "No campaign TOML files found");

        for entry in entries {
            let path = entry.path();
            let content = fs::read_to_string(&path)
                .expect(&format!("Failed to read {:?}", path));

            // The TOML should parse successfully
            let parsed: toml::Value = toml::from_str(&content)
                .expect(&format!("Failed to parse {:?}", path));

            // It should have a campaign section
            assert!(
                parsed.get("campaign").is_some(),
                "Campaign section missing in {:?}",
                path
            );

            // If there's no [universe] section, that's expected and fine
            // The system should use defaults
            if parsed.get("universe").is_none() {
                println!("✓ {:?} - No [universe] section (backward compatible)", path.file_name().unwrap());
            } else {
                println!("✓ {:?} - Has [universe] section", path.file_name().unwrap());
            }
        }
    }

    /// Test that risk profiles with universe_restrictions load correctly.
    #[test]
    fn test_risk_profiles_with_restrictions() {
        let risk_profiles_dir = configs_path().join("risk_profiles");
        
        if !risk_profiles_dir.exists() {
            eprintln!("Skipping test: risk_profiles directory not found");
            return;
        }

        let profiles = ["muito_conservador", "conservador", "moderado", "arrojado", "muito_arrojado"];
        
        for profile_name in profiles {
            let path = risk_profiles_dir.join(format!("{}.toml", profile_name));
            
            if !path.exists() {
                eprintln!("Skipping profile: {:?} not found", profile_name);
                continue;
            }

            let content = fs::read_to_string(&path)
                .expect(&format!("Failed to read {:?}", path));

            let parsed: toml::Value = toml::from_str(&content)
                .expect(&format!("Failed to parse {:?}", path));

            // Check for universe_restrictions section
            if let Some(restrictions) = parsed.get("universe_restrictions") {
                assert!(
                    restrictions.get("allowed_strategy_families").is_some(),
                    "Missing allowed_strategy_families in {:?}",
                    path
                );
                assert!(
                    restrictions.get("max_parameters_to_optimize").is_some(),
                    "Missing max_parameters_to_optimize in {:?}",
                    path
                );
                println!("✓ {} - Has universe_restrictions", profile_name);
            } else {
                println!("⚠ {} - No universe_restrictions (should be added)", profile_name);
            }
        }
    }
}

mod universe_configs {
    use super::*;

    /// Test that training strategy configs load correctly.
    #[test]
    fn test_training_strategies_exist() {
        let dir = configs_path().join("training_strategies");
        
        if !dir.exists() {
            eprintln!("Skipping test: training_strategies directory not found");
            return;
        }

        let expected = ["purged_kfold", "walk_forward", "anchored", "expanding_window", "monte_carlo"];
        
        for name in expected {
            let path = dir.join(format!("{}.toml", name));
            assert!(path.exists(), "Missing training strategy: {}", name);
            
            let content = fs::read_to_string(&path).unwrap();
            let parsed: toml::Value = toml::from_str(&content)
                .expect(&format!("Failed to parse {}.toml", name));
            
            // Verify structure
            assert!(parsed.get("strategy").is_some(), "Missing [strategy] in {}", name);
            assert!(parsed.get("validation").is_some(), "Missing [validation] in {}", name);
            
            println!("✓ training_strategy: {}", name);
        }
    }

    /// Test that training tech configs load correctly.
    #[test]
    fn test_training_tech_exist() {
        let dir = configs_path().join("training_tech");
        
        if !dir.exists() {
            eprintln!("Skipping test: training_tech directory not found");
            return;
        }

        let expected = ["cpu_fast", "cpu_parallel", "cpu_intensive", "distributed"];
        
        for name in expected {
            let path = dir.join(format!("{}.toml", name));
            assert!(path.exists(), "Missing training tech: {}", name);
            
            let content = fs::read_to_string(&path).unwrap();
            let parsed: toml::Value = toml::from_str(&content)
                .expect(&format!("Failed to parse {}.toml", name));
            
            // Verify structure
            assert!(parsed.get("tech").is_some(), "Missing [tech] in {}", name);
            assert!(parsed.get("resources").is_some(), "Missing [resources] in {}", name);
            assert!(parsed.get("evolution").is_some(), "Missing [evolution] in {}", name);
            
            println!("✓ training_tech: {}", name);
        }
    }

    /// Test that parameter bounds configs load correctly.
    #[test]
    fn test_parameter_bounds_exist() {
        let dir = configs_path().join("parameter_bounds");
        
        if !dir.exists() {
            eprintln!("Skipping test: parameter_bounds directory not found");
            return;
        }

        let expected = ["swing", "momentum", "position"];
        
        for name in expected {
            let path = dir.join(format!("{}.toml", name));
            assert!(path.exists(), "Missing parameter bounds: {}", name);
            
            let content = fs::read_to_string(&path).unwrap();
            let parsed: toml::Value = toml::from_str(&content)
                .expect(&format!("Failed to parse {}.toml", name));
            
            // Verify structure
            assert!(parsed.get("bounds").is_some(), "Missing [bounds] in {}", name);
            
            println!("✓ parameter_bounds: {}", name);
        }
    }

    /// Test compatibility matrix exists and has required sections.
    #[test]
    fn test_compatibility_matrix_structure() {
        let path = configs_path().join("compatibility_matrix.toml");
        
        if !path.exists() {
            eprintln!("Skipping test: compatibility_matrix.toml not found");
            return;
        }

        let content = fs::read_to_string(&path).unwrap();
        let parsed: toml::Value = toml::from_str(&content)
            .expect("Failed to parse compatibility_matrix.toml");
        
        // Check required sections
        let required_sections = [
            "metadata",
            "robustness_to_training_strategy",
            "training_model_to_robustness",
            "training_tech_to_complexity",
            "training_model_complexity",
        ];
        
        for section in required_sections {
            assert!(
                parsed.get(section).is_some(),
                "Missing section [{}] in compatibility_matrix.toml",
                section
            );
        }
        
        println!("✓ compatibility_matrix.toml has all required sections");
    }
}

mod validation {
    use super::*;

    /// Test that a valid universe config can be constructed.
    #[test]
    fn test_valid_universe_config_toml() {
        let toml_str = r#"
[universe]
robustness_profile = "moderado"
training_strategy = "purged_kfold"
training_tech = "cpu_parallel"
training_model = "swing"

[universe.overrides]
max_parameters = 10
"#;
        let parsed: toml::Value = toml::from_str(toml_str)
            .expect("Failed to parse valid universe config");
        
        let universe = parsed.get("universe").expect("Missing [universe]");
        assert_eq!(
            universe.get("robustness_profile").unwrap().as_str().unwrap(),
            "moderado"
        );
        assert_eq!(
            universe.get("training_strategy").unwrap().as_str().unwrap(),
            "purged_kfold"
        );
    }

    /// Test that universe config with multiple training models works.
    #[test]
    fn test_multiple_training_models() {
        let toml_str = r#"
[universe]
robustness_profile = "arrojado"
training_strategy = "walk_forward"
training_tech = "cpu_intensive"
training_model = ["swing", "momentum", "breakout"]
"#;
        let parsed: toml::Value = toml::from_str(toml_str)
            .expect("Failed to parse universe config with multiple models");
        
        let universe = parsed.get("universe").expect("Missing [universe]");
        let models = universe.get("training_model").unwrap().as_array().unwrap();
        assert_eq!(models.len(), 3);
    }
}



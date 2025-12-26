//! Strategy configuration loader.

use super::StrategyConfig;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LoadError {
    #[error("File not found: {0}")]
    FileNotFound(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("TOML parse error: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("Validation error: {0}")]
    Validation(String),
}

/// Load strategy configuration from TOML file.
pub fn load_strategy_config(path: impl AsRef<Path>) -> Result<StrategyConfig, LoadError> {
    let path = path.as_ref();
    
    if !path.exists() {
        return Err(LoadError::FileNotFound(path.display().to_string()));
    }
    
    let content = std::fs::read_to_string(path)?;
    let config: StrategyConfig = toml::from_str(&content)?;
    
    Ok(config)
}

/// Load strategy configuration from TOML string.
pub fn load_strategy_from_str(content: &str) -> Result<StrategyConfig, LoadError> {
    let config: StrategyConfig = toml::from_str(content)?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_from_str() {
        let toml_str = r#"
[strategy]
id = "test"

[[pipeline]]
type = "selection"
block_id = "momentum"
"#;

        let config = load_strategy_from_str(toml_str).unwrap();
        assert_eq!(config.strategy.id, "test");
    }

    #[test]
    fn test_load_invalid_toml() {
        let invalid = "this is not valid toml {{{}}}";
        let result = load_strategy_from_str(invalid);
        assert!(result.is_err());
    }
}


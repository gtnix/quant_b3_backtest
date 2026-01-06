//! Risk profile loader from TOML config files.

use std::path::Path;
use std::fs;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::filters::Market;
use super::profile::{RiskProfile, RiskProfileParams};
use super::types::{
    SizingParams, StopParams, PortfolioRiskParams,
    CircuitBreakerParams, OperationalParams, UniverseFilterParams,
};

/// Error type for profile loading.
#[derive(Error, Debug)]
pub enum LoadError {
    #[error("Failed to read config file: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Failed to parse TOML: {0}")]
    ParseError(#[from] toml::de::Error),
    
    #[error("Unknown profile: {0}")]
    UnknownProfile(String),
    
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

/// TOML structure for profile config file.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProfileConfig {
    pub profile: ProfileMeta,
    
    #[serde(default)]
    pub sizing: Option<SizingConfig>,
    
    #[serde(default)]
    pub stops: Option<StopsConfig>,
    
    #[serde(default)]
    pub portfolio_risk: Option<PortfolioRiskConfig>,
    
    #[serde(default)]
    pub circuit_breakers: Option<CircuitBreakerConfig>,
    
    #[serde(default)]
    pub operational: Option<OperationalConfig>,
    
    #[serde(default)]
    pub universe_filters: Option<UniverseFiltersConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProfileMeta {
    pub name: String,
    #[serde(default)]
    pub base: Option<String>,
    #[serde(default = "default_true")]
    pub market_adjustments: bool,
}

fn default_true() -> bool { true }

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct SizingConfig {
    pub kelly_fraction: Option<f64>,
    pub max_risk_per_trade_pct: Option<f64>,
    pub max_exposure_per_asset_pct: Option<f64>,
    pub max_sector_concentration_pct: Option<f64>,
    pub max_positions: Option<u32>,
    pub min_position_weight: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct StopsConfig {
    pub stop_type: Option<String>,
    pub atr_period: Option<u32>,
    pub atr_multiplier_br: Option<f64>,
    pub atr_multiplier_us: Option<f64>,
    pub fixed_stop_pct: Option<f64>,
    pub enable_trailing: Option<bool>,
    pub trailing_activation_pct: Option<f64>,
    pub trailing_atr_multiplier: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct PortfolioRiskConfig {
    pub volatility_target_br: Option<f64>,
    pub volatility_target_us: Option<f64>,
    pub max_drawdown_pct: Option<f64>,
    pub max_leverage: Option<f64>,
    pub cvar_limit_95: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct CircuitBreakerConfig {
    pub daily_loss_limit_pct: Option<f64>,
    pub weekly_loss_limit_pct: Option<f64>,
    pub monthly_loss_limit_pct: Option<f64>,
    pub drawdown_action: Option<String>,
    pub cooldown_days: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct OperationalConfig {
    pub min_liquidity_usd_br: Option<f64>,
    pub min_liquidity_usd_us: Option<f64>,
    pub max_spread_bps_br: Option<f64>,
    pub max_spread_bps_us: Option<f64>,
    pub slippage_cost_bps: Option<f64>,
    pub commission_rate: Option<f64>,
    pub max_participation_rate: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct UniverseFiltersConfig {
    pub min_market_cap_br: Option<f64>,
    pub min_market_cap_us: Option<f64>,
    pub max_annualized_vol: Option<f64>,
    pub min_dividend_yield: Option<f64>,
    pub min_momentum_return: Option<f64>,
    pub use_quantile_thresholds: Option<bool>,
    pub top_quantile: Option<f64>,
}

/// Loader for risk profiles from TOML files.
pub struct RiskProfileLoader {
    config_dir: String,
}

impl RiskProfileLoader {
    /// Create a new loader with the given config directory.
    pub fn new(config_dir: impl Into<String>) -> Self {
        Self {
            config_dir: config_dir.into(),
        }
    }

    /// Load a profile from TOML file, merging with defaults.
    pub fn load(&self, profile_name: &str, market: Market) -> Result<RiskProfileParams, LoadError> {
        let file_path = Path::new(&self.config_dir)
            .join(format!("{}.toml", profile_name));
        
        if !file_path.exists() {
            // Fallback to built-in profile
            return self.load_builtin(profile_name, market);
        }

        let content = fs::read_to_string(&file_path)?;
        let config: ProfileConfig = toml::from_str(&content)?;
        
        // Get base profile
        let base_profile = config.profile.base
            .as_ref()
            .map(|s| s.parse::<RiskProfile>())
            .transpose()
            .map_err(|e| LoadError::UnknownProfile(e))?
            .unwrap_or(RiskProfile::Moderado);
        
        let mut params = base_profile.params(market);
        params.name = config.profile.name.clone();
        
        // Apply overrides
        if let Some(sizing) = config.sizing {
            self.apply_sizing_overrides(&mut params.sizing, &sizing);
        }
        
        if let Some(stops) = config.stops {
            self.apply_stops_overrides(&mut params.stops, &stops, market);
        }
        
        if let Some(risk) = config.portfolio_risk {
            self.apply_portfolio_risk_overrides(&mut params.portfolio_risk, &risk, market);
        }
        
        if let Some(cb) = config.circuit_breakers {
            self.apply_circuit_breaker_overrides(&mut params.circuit_breakers, &cb);
        }
        
        if let Some(op) = config.operational {
            self.apply_operational_overrides(&mut params.operational, &op, market);
        }
        
        if let Some(uf) = config.universe_filters {
            self.apply_universe_filter_overrides(&mut params.universe_filters, &uf, market);
        }
        
        self.validate_params(&params)?;
        
        Ok(params)
    }

    /// Load built-in profile by name.
    fn load_builtin(&self, name: &str, market: Market) -> Result<RiskProfileParams, LoadError> {
        let profile: RiskProfile = name.parse()
            .map_err(|_| LoadError::UnknownProfile(name.to_string()))?;
        Ok(profile.params(market))
    }

    fn apply_sizing_overrides(&self, target: &mut SizingParams, source: &SizingConfig) {
        if let Some(v) = source.kelly_fraction { target.kelly_fraction = v; }
        if let Some(v) = source.max_risk_per_trade_pct { target.max_risk_per_trade_pct = v; }
        if let Some(v) = source.max_exposure_per_asset_pct { target.max_exposure_per_asset_pct = v; }
        if let Some(v) = source.max_sector_concentration_pct { target.max_sector_concentration_pct = v; }
        if let Some(v) = source.max_positions { target.max_positions = v; }
        if let Some(v) = source.min_position_weight { target.min_position_weight = v; }
    }

    fn apply_stops_overrides(&self, target: &mut StopParams, source: &StopsConfig, market: Market) {
        if let Some(ref v) = source.stop_type { target.stop_type = v.clone(); }
        if let Some(v) = source.atr_period { target.atr_period = v; }
        
        // Market-specific ATR multiplier
        match market {
            Market::BR => {
                if let Some(v) = source.atr_multiplier_br { target.atr_multiplier = v; }
            }
            Market::US => {
                if let Some(v) = source.atr_multiplier_us { target.atr_multiplier = v; }
            }
        }
        
        if let Some(v) = source.fixed_stop_pct { target.fixed_stop_pct = v; }
        if let Some(v) = source.enable_trailing { target.enable_trailing = v; }
        if let Some(v) = source.trailing_activation_pct { target.trailing_activation_pct = v; }
        if let Some(v) = source.trailing_atr_multiplier { target.trailing_atr_multiplier = v; }
    }

    fn apply_portfolio_risk_overrides(&self, target: &mut PortfolioRiskParams, source: &PortfolioRiskConfig, market: Market) {
        match market {
            Market::BR => {
                if let Some(v) = source.volatility_target_br { target.volatility_target = v; }
            }
            Market::US => {
                if let Some(v) = source.volatility_target_us { target.volatility_target = v; }
            }
        }
        
        if let Some(v) = source.max_drawdown_pct { target.max_drawdown_pct = v; }
        if let Some(v) = source.max_leverage { target.max_leverage = v; }
        if let Some(v) = source.cvar_limit_95 { target.cvar_limit_95 = v; }
    }

    fn apply_circuit_breaker_overrides(&self, target: &mut CircuitBreakerParams, source: &CircuitBreakerConfig) {
        if let Some(v) = source.daily_loss_limit_pct { target.daily_loss_limit_pct = v; }
        if let Some(v) = source.weekly_loss_limit_pct { target.weekly_loss_limit_pct = v; }
        if let Some(v) = source.monthly_loss_limit_pct { target.monthly_loss_limit_pct = v; }
        if let Some(ref v) = source.drawdown_action { target.drawdown_action = v.clone(); }
        if let Some(v) = source.cooldown_days { target.cooldown_days = v; }
    }

    fn apply_operational_overrides(&self, target: &mut OperationalParams, source: &OperationalConfig, market: Market) {
        match market {
            Market::BR => {
                if let Some(v) = source.min_liquidity_usd_br { target.min_liquidity_usd = v; }
                if let Some(v) = source.max_spread_bps_br { target.max_spread_bps = v; }
            }
            Market::US => {
                if let Some(v) = source.min_liquidity_usd_us { target.min_liquidity_usd = v; }
                if let Some(v) = source.max_spread_bps_us { target.max_spread_bps = v; }
            }
        }
        
        if let Some(v) = source.slippage_cost_bps { target.slippage_cost_bps = v; }
        if let Some(v) = source.commission_rate { target.commission_rate = v; }
        if let Some(v) = source.max_participation_rate { target.max_participation_rate = v; }
    }

    fn apply_universe_filter_overrides(&self, target: &mut UniverseFilterParams, source: &UniverseFiltersConfig, market: Market) {
        match market {
            Market::BR => {
                if let Some(v) = source.min_market_cap_br { target.min_market_cap = v; }
            }
            Market::US => {
                if let Some(v) = source.min_market_cap_us { target.min_market_cap = v; }
            }
        }
        
        if let Some(v) = source.max_annualized_vol { target.max_annualized_vol = v; }
        if let Some(v) = source.min_dividend_yield { target.min_dividend_yield = v; }
        if let Some(v) = source.min_momentum_return { target.min_momentum_return = v; }
        if let Some(v) = source.use_quantile_thresholds { target.use_quantile_thresholds = v; }
        if let Some(v) = source.top_quantile { target.top_quantile = v; }
    }

    /// Validate parameter ranges.
    fn validate_params(&self, params: &RiskProfileParams) -> Result<(), LoadError> {
        // Kelly fraction: 0 < f <= 0.5 (Half-Kelly max)
        if params.sizing.kelly_fraction <= 0.0 || params.sizing.kelly_fraction > 0.5 {
            return Err(LoadError::InvalidParameter(
                format!("kelly_fraction must be in (0, 0.5], got {}", params.sizing.kelly_fraction)
            ));
        }
        
        // Risk per trade: 0 < r <= 0.03 (3% max)
        if params.sizing.max_risk_per_trade_pct <= 0.0 || params.sizing.max_risk_per_trade_pct > 0.03 {
            return Err(LoadError::InvalidParameter(
                format!("max_risk_per_trade_pct must be in (0, 0.03], got {}", params.sizing.max_risk_per_trade_pct)
            ));
        }
        
        // ATR multiplier: >= 1.0
        if params.stops.atr_multiplier < 1.0 {
            return Err(LoadError::InvalidParameter(
                format!("atr_multiplier must be >= 1.0, got {}", params.stops.atr_multiplier)
            ));
        }
        
        // Max drawdown: must be negative
        if params.portfolio_risk.max_drawdown_pct >= 0.0 {
            return Err(LoadError::InvalidParameter(
                format!("max_drawdown_pct must be negative, got {}", params.portfolio_risk.max_drawdown_pct)
            ));
        }
        
        Ok(())
    }
}

impl Default for RiskProfileLoader {
    fn default() -> Self {
        Self::new("configs/risk_profiles")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loader_builtin_profiles() {
        let loader = RiskProfileLoader::default();
        
        for profile_name in ["muito_conservador", "conservador", "moderado", "arrojado", "muito_arrojado"] {
            let params = loader.load_builtin(profile_name, Market::BR).unwrap();
            assert!(!params.name.is_empty());
        }
    }

    #[test]
    fn test_validation_kelly_bounds() {
        let loader = RiskProfileLoader::default();
        let mut params = RiskProfile::Moderado.params(Market::BR);
        
        // Valid
        params.sizing.kelly_fraction = 0.5;
        assert!(loader.validate_params(&params).is_ok());
        
        // Invalid: too high
        params.sizing.kelly_fraction = 0.6;
        assert!(loader.validate_params(&params).is_err());
        
        // Invalid: zero
        params.sizing.kelly_fraction = 0.0;
        assert!(loader.validate_params(&params).is_err());
    }
}







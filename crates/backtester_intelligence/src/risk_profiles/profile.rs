//! Risk profile definitions with academic backing.
//!
//! Parameters are based on:
//! - "Determinação de Operação: Parâmetros de Risco para Brasil e EUA" (v2.0)
//! - Ziemba & MacLean (2011) - Kelly Criterion
//! - Vince (1992) - Money Management  
//! - Chan (2013, 2021) - Algorithmic Trading
//! - Harvey et al. (2017) - Volatility Targeting
//! - Chekhlov et al. (2003) - Drawdown Constraints
//! - Thorp (2006) - Kelly in Stock Market

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

use crate::filters::Market;
use super::types::{
    SizingParams, StopParams, PortfolioRiskParams, 
    CircuitBreakerParams, OperationalParams, UniverseFilterParams,
};

/// Risk profile levels from most conservative to most aggressive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskProfile {
    /// Muito Conservador: Capital preservation, minimal drawdowns.
    /// Target: Beat inflation with negligible risk of ruin.
    MuitoConservador,
    
    /// Conservador: Consistent growth with controlled risk.
    /// Target: Good Sharpe ratio, limited drawdowns.
    Conservador,
    
    /// Moderado: Long-term growth accepting short-term volatility.
    /// Target: Maximize geometric growth near Half-Kelly.
    Moderado,
    
    /// Arrojado: Aggressive growth with high risk tolerance.
    /// Target: High returns using maximum defensible risk.
    Arrojado,
    
    /// Muito Arrojado: Speculative, maximum risk within bounds.
    /// Target: Exponential gains, accepts significant drawdowns.
    MuitoArrojado,
}

impl Default for RiskProfile {
    fn default() -> Self {
        RiskProfile::Moderado
    }
}

impl fmt::Display for RiskProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MuitoConservador => write!(f, "muito_conservador"),
            Self::Conservador => write!(f, "conservador"),
            Self::Moderado => write!(f, "moderado"),
            Self::Arrojado => write!(f, "arrojado"),
            Self::MuitoArrojado => write!(f, "muito_arrojado"),
        }
    }
}

impl FromStr for RiskProfile {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "muito_conservador" | "muitoconservador" | "very_conservative" => Ok(Self::MuitoConservador),
            "conservador" | "conservative" => Ok(Self::Conservador),
            "moderado" | "moderate" => Ok(Self::Moderado),
            "arrojado" | "aggressive" => Ok(Self::Arrojado),
            "muito_arrojado" | "muitoarrojado" | "very_aggressive" => Ok(Self::MuitoArrojado),
            _ => Err(format!("Unknown risk profile: {}", s)),
        }
    }
}

/// Market-specific parameter adjustments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketParams {
    /// ATR multiplier adjustment factor (BR typically higher).
    pub atr_multiplier_adjustment: f64,
    
    /// Volatility target adjustment (BR typically higher for same risk level).
    pub volatility_target_adjustment: f64,
    
    /// Minimum liquidity in USD.
    pub min_liquidity_usd: f64,
    
    /// Maximum spread in bps.
    pub max_spread_bps: f64,
    
    /// Expected slippage in bps.
    pub slippage_bps: f64,
}

impl MarketParams {
    /// BR market parameters.
    /// Higher volatility, lower liquidity, wider spreads.
    pub fn br() -> Self {
        Self {
            atr_multiplier_adjustment: 1.0,  // Base
            volatility_target_adjustment: 1.0, // Base
            min_liquidity_usd: 5_000_000.0,  // $5M
            max_spread_bps: 30.0,
            slippage_bps: 15.0,
        }
    }

    /// US market parameters.
    /// Lower volatility, higher liquidity, tighter spreads.
    pub fn us() -> Self {
        Self {
            atr_multiplier_adjustment: 0.85,  // 15% lower stops
            volatility_target_adjustment: 0.80, // 20% lower vol target
            min_liquidity_usd: 20_000_000.0, // $20M
            max_spread_bps: 15.0,
            slippage_bps: 8.0,
        }
    }

    /// Get market params for a specific market.
    pub fn for_market(market: Market) -> Self {
        match market {
            Market::BR => Self::br(),
            Market::US => Self::us(),
        }
    }
}

/// Complete risk profile parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskProfileParams {
    /// Profile name.
    pub name: String,
    
    /// Profile level.
    pub profile: RiskProfile,
    
    /// Market this was configured for.
    pub market: Market,
    
    /// Position sizing parameters.
    pub sizing: SizingParams,
    
    /// Stop-loss parameters.
    pub stops: StopParams,
    
    /// Portfolio-level risk parameters.
    pub portfolio_risk: PortfolioRiskParams,
    
    /// Circuit breaker parameters.
    pub circuit_breakers: CircuitBreakerParams,
    
    /// Operational parameters.
    pub operational: OperationalParams,
    
    /// Universe filter parameters.
    pub universe_filters: UniverseFilterParams,
}

impl RiskProfile {
    /// Get the complete parameter set for this profile and market.
    pub fn params(&self, market: Market) -> RiskProfileParams {
        let market_adj = MarketParams::for_market(market);
        
        match self {
            RiskProfile::MuitoConservador => Self::muito_conservador_params(market, &market_adj),
            RiskProfile::Conservador => Self::conservador_params(market, &market_adj),
            RiskProfile::Moderado => Self::moderado_params(market, &market_adj),
            RiskProfile::Arrojado => Self::arrojado_params(market, &market_adj),
            RiskProfile::MuitoArrojado => Self::muito_arrojado_params(market, &market_adj),
        }
    }

    /// Muito Conservador: Capital preservation focus.
    /// 
    /// Academic basis:
    /// - Kelly fraction 0.1-0.25 (Quarter-Kelly) per Ziemba & MacLean (2011)
    /// - Risk per trade 0.25-0.5% per Vince (1992)
    /// - Max DD 8% per Chekhlov et al. (2003)
    fn muito_conservador_params(market: Market, adj: &MarketParams) -> RiskProfileParams {
        RiskProfileParams {
            name: "Muito Conservador".to_string(),
            profile: RiskProfile::MuitoConservador,
            market,
            sizing: SizingParams {
                kelly_fraction: 0.15,
                max_risk_per_trade_pct: 0.005, // 0.5%
                max_exposure_per_asset_pct: 0.10,
                max_sector_concentration_pct: 0.25,
                max_positions: 30,
                min_position_weight: 0.02,
            },
            stops: StopParams {
                stop_type: "ATR".to_string(),
                atr_period: 14,
                atr_multiplier: 3.5 * adj.atr_multiplier_adjustment,
                fixed_stop_pct: 0.08,
                enable_trailing: true,
                trailing_activation_pct: 0.05,
                trailing_atr_multiplier: 3.0 * adj.atr_multiplier_adjustment,
            },
            portfolio_risk: PortfolioRiskParams {
                volatility_target: 0.08 * adj.volatility_target_adjustment,
                max_drawdown_pct: -0.08,
                max_leverage: 1.0,
                cvar_limit_95: -0.02,
            },
            circuit_breakers: CircuitBreakerParams {
                daily_loss_limit_pct: -0.01,
                weekly_loss_limit_pct: -0.02,
                monthly_loss_limit_pct: -0.05,
                drawdown_action: "CashOut".to_string(),
                cooldown_days: 2,
            },
            operational: OperationalParams {
                min_liquidity_usd: adj.min_liquidity_usd,
                max_spread_bps: adj.max_spread_bps,
                slippage_cost_bps: adj.slippage_bps,
                commission_rate: 0.001,
                max_participation_rate: 0.03,
            },
            universe_filters: UniverseFilterParams {
                min_market_cap: if market == Market::BR { 2_000_000_000.0 } else { 10_000_000_000.0 },
                max_annualized_vol: 0.40,
                min_dividend_yield: 0.0,
                min_momentum_return: -0.10,
                use_quantile_thresholds: true,
                top_quantile: 0.30,
            },
        }
    }

    /// Conservador: Controlled growth.
    /// 
    /// Academic basis:
    /// - Kelly fraction 0.25-0.4 per Ziemba & MacLean (2011)
    /// - Risk per trade 0.5-1.0% per Vince (1992)
    /// - Max DD 12% - common for low-vol funds
    fn conservador_params(market: Market, adj: &MarketParams) -> RiskProfileParams {
        RiskProfileParams {
            name: "Conservador".to_string(),
            profile: RiskProfile::Conservador,
            market,
            sizing: SizingParams {
                kelly_fraction: 0.30,
                max_risk_per_trade_pct: 0.0075, // 0.75%
                max_exposure_per_asset_pct: 0.15,
                max_sector_concentration_pct: 0.30,
                max_positions: 25,
                min_position_weight: 0.02,
            },
            stops: StopParams {
                stop_type: "ATR".to_string(),
                atr_period: 14,
                atr_multiplier: 3.0 * adj.atr_multiplier_adjustment,
                fixed_stop_pct: 0.10,
                enable_trailing: true,
                trailing_activation_pct: 0.08,
                trailing_atr_multiplier: 2.5 * adj.atr_multiplier_adjustment,
            },
            portfolio_risk: PortfolioRiskParams {
                volatility_target: 0.12 * adj.volatility_target_adjustment,
                max_drawdown_pct: -0.12,
                max_leverage: 1.0,
                cvar_limit_95: -0.025,
            },
            circuit_breakers: CircuitBreakerParams {
                daily_loss_limit_pct: -0.015,
                weekly_loss_limit_pct: -0.035,
                monthly_loss_limit_pct: -0.08,
                drawdown_action: "ReduceRisk".to_string(),
                cooldown_days: 1,
            },
            operational: OperationalParams {
                min_liquidity_usd: adj.min_liquidity_usd,
                max_spread_bps: adj.max_spread_bps,
                slippage_cost_bps: adj.slippage_bps,
                commission_rate: 0.001,
                max_participation_rate: 0.04,
            },
            universe_filters: UniverseFilterParams {
                min_market_cap: if market == Market::BR { 1_000_000_000.0 } else { 5_000_000_000.0 },
                max_annualized_vol: 0.45,
                min_dividend_yield: 0.0,
                min_momentum_return: -0.05,
                use_quantile_thresholds: true,
                top_quantile: 0.25,
            },
        }
    }

    /// Moderado: Balanced growth and risk.
    /// 
    /// Academic basis:
    /// - Kelly fraction 0.4-0.5 (near Half-Kelly) per Thorp (2006)
    /// - Risk per trade 1.0-1.5% per Chan (2021)
    /// - Max DD 20% - standard for equity funds
    fn moderado_params(market: Market, adj: &MarketParams) -> RiskProfileParams {
        RiskProfileParams {
            name: "Moderado".to_string(),
            profile: RiskProfile::Moderado,
            market,
            sizing: SizingParams {
                kelly_fraction: 0.40,
                max_risk_per_trade_pct: 0.0125, // 1.25%
                max_exposure_per_asset_pct: 0.20,
                max_sector_concentration_pct: 0.35,
                max_positions: 20,
                min_position_weight: 0.02,
            },
            stops: StopParams {
                stop_type: "ATR".to_string(),
                atr_period: 14,
                atr_multiplier: 2.5 * adj.atr_multiplier_adjustment,
                fixed_stop_pct: 0.12,
                enable_trailing: true,
                trailing_activation_pct: 0.10,
                trailing_atr_multiplier: 2.0 * adj.atr_multiplier_adjustment,
            },
            portfolio_risk: PortfolioRiskParams {
                volatility_target: 0.16 * adj.volatility_target_adjustment,
                max_drawdown_pct: -0.20,
                max_leverage: 1.0,
                cvar_limit_95: -0.035,
            },
            circuit_breakers: CircuitBreakerParams {
                daily_loss_limit_pct: -0.025,
                weekly_loss_limit_pct: -0.06,
                monthly_loss_limit_pct: -0.12,
                drawdown_action: "ReduceRisk".to_string(),
                cooldown_days: 1,
            },
            operational: OperationalParams {
                min_liquidity_usd: adj.min_liquidity_usd,
                max_spread_bps: adj.max_spread_bps,
                slippage_cost_bps: adj.slippage_bps,
                commission_rate: 0.001,
                max_participation_rate: 0.05,
            },
            universe_filters: UniverseFilterParams {
                min_market_cap: if market == Market::BR { 500_000_000.0 } else { 2_000_000_000.0 },
                max_annualized_vol: 0.50,
                min_dividend_yield: -0.02,
                min_momentum_return: 0.0,
                use_quantile_thresholds: true,
                top_quantile: 0.20,
            },
        }
    }

    /// Arrojado: Aggressive growth.
    /// 
    /// Academic basis:
    /// - Kelly fraction 0.5 (Half-Kelly maximum) per Thorp (2006)
    /// - Risk per trade 1.5-2.0% per Vince (1992)
    /// - Max DD 25% - upper limit for most strategies
    fn arrojado_params(market: Market, adj: &MarketParams) -> RiskProfileParams {
        RiskProfileParams {
            name: "Arrojado".to_string(),
            profile: RiskProfile::Arrojado,
            market,
            sizing: SizingParams {
                kelly_fraction: 0.50,
                max_risk_per_trade_pct: 0.0175, // 1.75%
                max_exposure_per_asset_pct: 0.25,
                max_sector_concentration_pct: 0.40,
                max_positions: 15,
                min_position_weight: 0.03,
            },
            stops: StopParams {
                stop_type: "ATR".to_string(),
                atr_period: 14,
                atr_multiplier: 2.0 * adj.atr_multiplier_adjustment,
                fixed_stop_pct: 0.15,
                enable_trailing: true,
                trailing_activation_pct: 0.12,
                trailing_atr_multiplier: 1.75 * adj.atr_multiplier_adjustment,
            },
            portfolio_risk: PortfolioRiskParams {
                volatility_target: 0.20 * adj.volatility_target_adjustment,
                max_drawdown_pct: -0.25,
                max_leverage: 1.2,
                cvar_limit_95: -0.045,
            },
            circuit_breakers: CircuitBreakerParams {
                daily_loss_limit_pct: -0.035,
                weekly_loss_limit_pct: -0.08,
                monthly_loss_limit_pct: -0.15,
                drawdown_action: "ReduceRisk".to_string(),
                cooldown_days: 1,
            },
            operational: OperationalParams {
                min_liquidity_usd: adj.min_liquidity_usd * 0.8,
                max_spread_bps: adj.max_spread_bps * 1.2,
                slippage_cost_bps: adj.slippage_bps * 1.2,
                commission_rate: 0.001,
                max_participation_rate: 0.06,
            },
            universe_filters: UniverseFilterParams {
                min_market_cap: if market == Market::BR { 300_000_000.0 } else { 1_000_000_000.0 },
                max_annualized_vol: 0.55,
                min_dividend_yield: -0.05,
                min_momentum_return: 0.0,
                use_quantile_thresholds: true,
                top_quantile: 0.15,
            },
        }
    }

    /// Muito Arrojado: Maximum risk within bounds.
    /// 
    /// Academic basis:
    /// - Kelly fraction 0.5 (never exceed Half-Kelly per Thorp (2006))
    /// - Risk per trade 2.0-2.5% - upper industry limit
    /// - Max DD 30% - extreme but not ruinous
    fn muito_arrojado_params(market: Market, adj: &MarketParams) -> RiskProfileParams {
        RiskProfileParams {
            name: "Muito Arrojado".to_string(),
            profile: RiskProfile::MuitoArrojado,
            market,
            sizing: SizingParams {
                kelly_fraction: 0.50, // Never exceed Half-Kelly
                max_risk_per_trade_pct: 0.0225, // 2.25%
                max_exposure_per_asset_pct: 0.30,
                max_sector_concentration_pct: 0.45,
                max_positions: 10,
                min_position_weight: 0.05,
            },
            stops: StopParams {
                stop_type: "ATR".to_string(),
                atr_period: 14,
                atr_multiplier: 1.75 * adj.atr_multiplier_adjustment,
                fixed_stop_pct: 0.18,
                enable_trailing: true,
                trailing_activation_pct: 0.15,
                trailing_atr_multiplier: 1.5 * adj.atr_multiplier_adjustment,
            },
            portfolio_risk: PortfolioRiskParams {
                volatility_target: 0.25 * adj.volatility_target_adjustment,
                max_drawdown_pct: -0.30,
                max_leverage: 1.5,
                cvar_limit_95: -0.06,
            },
            circuit_breakers: CircuitBreakerParams {
                daily_loss_limit_pct: -0.05,
                weekly_loss_limit_pct: -0.10,
                monthly_loss_limit_pct: -0.20,
                drawdown_action: "Alert".to_string(),
                cooldown_days: 0,
            },
            operational: OperationalParams {
                min_liquidity_usd: adj.min_liquidity_usd * 0.6,
                max_spread_bps: adj.max_spread_bps * 1.5,
                slippage_cost_bps: adj.slippage_bps * 1.5,
                commission_rate: 0.001,
                max_participation_rate: 0.08,
            },
            universe_filters: UniverseFilterParams {
                min_market_cap: if market == Market::BR { 200_000_000.0 } else { 500_000_000.0 },
                max_annualized_vol: 0.60,
                min_dividend_yield: -0.08,
                min_momentum_return: 0.0,
                use_quantile_thresholds: true,
                top_quantile: 0.10,
            },
        }
    }

    /// Get a brief description of the profile.
    pub fn description(&self) -> &'static str {
        match self {
            Self::MuitoConservador => "Capital preservation with minimal risk. Target: beat inflation.",
            Self::Conservador => "Consistent growth with controlled drawdowns. Target: good Sharpe ratio.",
            Self::Moderado => "Long-term growth accepting volatility. Target: geometric growth.",
            Self::Arrojado => "Aggressive growth with high risk tolerance. Target: high returns.",
            Self::MuitoArrojado => "Speculative with maximum defensible risk. Target: exponential gains.",
        }
    }

    /// Get expected max drawdown for this profile.
    pub fn expected_max_drawdown(&self) -> f64 {
        match self {
            Self::MuitoConservador => 0.08,
            Self::Conservador => 0.12,
            Self::Moderado => 0.20,
            Self::Arrojado => 0.25,
            Self::MuitoArrojado => 0.30,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_from_str() {
        assert_eq!(RiskProfile::from_str("conservador").unwrap(), RiskProfile::Conservador);
        assert_eq!(RiskProfile::from_str("MODERADO").unwrap(), RiskProfile::Moderado);
        assert_eq!(RiskProfile::from_str("aggressive").unwrap(), RiskProfile::Arrojado);
        assert!(RiskProfile::from_str("invalid").is_err());
    }

    #[test]
    fn test_profile_display() {
        assert_eq!(RiskProfile::MuitoConservador.to_string(), "muito_conservador");
        assert_eq!(RiskProfile::Arrojado.to_string(), "arrojado");
    }

    #[test]
    fn test_market_adjustments() {
        let br = MarketParams::br();
        let us = MarketParams::us();
        
        // US should have lower ATR multiplier (tighter stops)
        assert!(us.atr_multiplier_adjustment < br.atr_multiplier_adjustment);
        
        // US should have lower vol target adjustment
        assert!(us.volatility_target_adjustment < br.volatility_target_adjustment);
        
        // US should have higher liquidity requirement
        assert!(us.min_liquidity_usd > br.min_liquidity_usd);
    }

    #[test]
    fn test_kelly_never_exceeds_half() {
        for profile in [
            RiskProfile::MuitoConservador,
            RiskProfile::Conservador,
            RiskProfile::Moderado,
            RiskProfile::Arrojado,
            RiskProfile::MuitoArrojado,
        ] {
            let params = profile.params(Market::BR);
            assert!(
                params.sizing.kelly_fraction <= 0.5,
                "Kelly fraction should never exceed 0.5 (Half-Kelly) per Thorp (2006)"
            );
        }
    }
}


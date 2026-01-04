//! Risk Profiles Module - 5 predefined risk profiles with academic backing.
//!
//! Based on "Determinação de Operação: Parâmetros de Risco para Brasil e EUA"
//! with parameters derived from:
//! - Ziemba & MacLean (2011) - Kelly Criterion
//! - Vince (1992) - Money Management
//! - Chan (2013) - Algorithmic Trading
//! - Harvey et al. (2017) - Volatility Targeting
//! - Chekhlov et al. (2003) - Drawdown Constraints

mod profile;
mod loader;
mod types;

pub use profile::{RiskProfile, RiskProfileParams, MarketParams};
pub use loader::{RiskProfileLoader, LoadError};
pub use types::{SizingParams, StopParams, PortfolioRiskParams, CircuitBreakerParams, OperationalParams};

use crate::filters::Market;

/// Get default parameters for a risk profile.
pub fn get_profile_params(profile: RiskProfile, market: Market) -> RiskProfileParams {
    profile.params(market)
}

/// List all available risk profiles.
pub fn available_profiles() -> Vec<RiskProfile> {
    vec![
        RiskProfile::MuitoConservador,
        RiskProfile::Conservador,
        RiskProfile::Moderado,
        RiskProfile::Arrojado,
        RiskProfile::MuitoArrojado,
    ]
}

/// Log effective parameters at backtest start.
/// Should be called once at the beginning of each backtest run.
pub fn log_effective_params(params: &RiskProfileParams) {
    tracing::info!(
        profile = %params.profile,
        market = ?params.market,
        "[RISK PROFILE] Effective Parameters"
    );
    
    tracing::info!(
        kelly_fraction = %format!("{:.1}%", params.sizing.kelly_fraction * 100.0),
        max_risk_per_trade = %format!("{:.2}%", params.sizing.max_risk_per_trade_pct * 100.0),
        max_exposure_per_asset = %format!("{:.1}%", params.sizing.max_exposure_per_asset_pct * 100.0),
        max_positions = params.sizing.max_positions,
        "[SIZING]"
    );
    
    tracing::info!(
        stop_type = %params.stops.stop_type,
        atr_multiplier = %format!("{:.2}x", params.stops.atr_multiplier),
        trailing_enabled = params.stops.enable_trailing,
        "[STOPS]"
    );
    
    tracing::info!(
        volatility_target = %format!("{:.1}%", params.portfolio_risk.volatility_target * 100.0),
        max_drawdown = %format!("{:.1}%", params.portfolio_risk.max_drawdown_pct.abs() * 100.0),
        max_leverage = %format!("{:.1}x", params.portfolio_risk.max_leverage),
        "[PORTFOLIO RISK]"
    );
    
    tracing::info!(
        daily_loss_limit = %format!("{:.1}%", params.circuit_breakers.daily_loss_limit_pct.abs() * 100.0),
        weekly_loss_limit = %format!("{:.1}%", params.circuit_breakers.weekly_loss_limit_pct.abs() * 100.0),
        drawdown_action = %params.circuit_breakers.drawdown_action,
        "[CIRCUIT BREAKERS]"
    );
}

/// Log universe size progression through filter chain.
pub fn log_filter_chain(
    stage: &str,
    filter_name: &str,
    before: usize,
    after: usize,
    excluded_symbols: &[String],
) {
    let excluded_count = before.saturating_sub(after);
    
    if excluded_count == 0 {
        tracing::debug!(
            stage = stage,
            filter = filter_name,
            count = after,
            "[FILTER] No exclusions"
        );
    } else {
        // Only log first 5 excluded symbols to avoid log spam
        let sample: Vec<&str> = excluded_symbols.iter().take(5).map(|s| s.as_str()).collect();
        let sample_str = sample.join(", ");
        let suffix = if excluded_symbols.len() > 5 {
            format!(" (+{} more)", excluded_symbols.len() - 5)
        } else {
            String::new()
        };
        
        tracing::info!(
            stage = stage,
            filter = filter_name,
            before = before,
            after = after,
            excluded = excluded_count,
            "[FILTER] {}{}", sample_str, suffix
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_profiles_have_valid_params() {
        for profile in available_profiles() {
            let params_br = get_profile_params(profile, Market::BR);
            let params_us = get_profile_params(profile, Market::US);
            
            // Basic sanity checks
            assert!(params_br.sizing.kelly_fraction > 0.0);
            assert!(params_br.sizing.kelly_fraction <= 0.5);
            assert!(params_br.stops.atr_multiplier >= 1.0);
            assert!(params_br.portfolio_risk.max_drawdown_pct < 0.0);
            
            // BR should have higher ATR multiplier (wider stops)
            assert!(params_br.stops.atr_multiplier >= params_us.stops.atr_multiplier);
        }
    }

    #[test]
    fn test_profile_ordering_by_risk() {
        let muito_cons = get_profile_params(RiskProfile::MuitoConservador, Market::BR);
        let conservador = get_profile_params(RiskProfile::Conservador, Market::BR);
        let moderado = get_profile_params(RiskProfile::Moderado, Market::BR);
        let arrojado = get_profile_params(RiskProfile::Arrojado, Market::BR);
        let muito_arr = get_profile_params(RiskProfile::MuitoArrojado, Market::BR);

        // More aggressive profiles should have higher risk per trade
        assert!(muito_cons.sizing.max_risk_per_trade_pct < conservador.sizing.max_risk_per_trade_pct);
        assert!(conservador.sizing.max_risk_per_trade_pct < moderado.sizing.max_risk_per_trade_pct);
        assert!(moderado.sizing.max_risk_per_trade_pct < arrojado.sizing.max_risk_per_trade_pct);
        assert!(arrojado.sizing.max_risk_per_trade_pct < muito_arr.sizing.max_risk_per_trade_pct);

        // More aggressive profiles allow larger drawdowns (more negative)
        assert!(muito_cons.portfolio_risk.max_drawdown_pct > conservador.portfolio_risk.max_drawdown_pct);
        assert!(conservador.portfolio_risk.max_drawdown_pct > moderado.portfolio_risk.max_drawdown_pct);
    }
}


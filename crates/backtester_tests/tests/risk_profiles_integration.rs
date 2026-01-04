//! Integration tests for risk profiles.
//!
//! Validates:
//! - All 5 profiles load correctly
//! - BR/US market adjustments apply
//! - Parameter validation works
//! - Fallback guardrails trigger appropriately

use backtester_intelligence::{
    RiskProfile, RiskProfileParams, RiskProfileLoader, 
    Market, get_profile_params, available_profiles,
    EntryWarning,
};

// =============================================================================
// Profile Loading Tests
// =============================================================================

#[test]
fn test_all_profiles_load() {
    for profile in available_profiles() {
        let params_br = get_profile_params(profile, Market::BR);
        let params_us = get_profile_params(profile, Market::US);
        
        // Verify basic fields are populated
        assert!(!params_br.name.is_empty(), "Profile {:?} should have name", profile);
        assert!(!params_us.name.is_empty(), "Profile {:?} should have name", profile);
        
        // Verify Kelly fraction is within bounds
        assert!(params_br.sizing.kelly_fraction > 0.0, "Kelly fraction must be positive");
        assert!(params_br.sizing.kelly_fraction <= 0.5, "Kelly fraction must not exceed Half-Kelly (0.5)");
    }
}

#[test]
fn test_profile_from_string() {
    let profile: RiskProfile = "conservador".parse().unwrap();
    assert_eq!(profile, RiskProfile::Conservador);
    
    let profile: RiskProfile = "muito_arrojado".parse().unwrap();
    assert_eq!(profile, RiskProfile::MuitoArrojado);
    
    // Case insensitive
    let profile: RiskProfile = "MODERADO".parse().unwrap();
    assert_eq!(profile, RiskProfile::Moderado);
    
    // English aliases
    let profile: RiskProfile = "aggressive".parse().unwrap();
    assert_eq!(profile, RiskProfile::Arrojado);
}

#[test]
fn test_invalid_profile_string() {
    let result: Result<RiskProfile, _> = "invalid_profile".parse();
    assert!(result.is_err());
}

// =============================================================================
// BR vs US Adjustment Tests
// =============================================================================

#[test]
fn test_br_has_wider_atr_stops() {
    for profile in available_profiles() {
        let params_br = get_profile_params(profile, Market::BR);
        let params_us = get_profile_params(profile, Market::US);
        
        assert!(
            params_br.stops.atr_multiplier >= params_us.stops.atr_multiplier,
            "BR should have equal or wider ATR stops than US for {:?}", profile
        );
    }
}

#[test]
fn test_br_has_higher_vol_target() {
    for profile in available_profiles() {
        let params_br = get_profile_params(profile, Market::BR);
        let params_us = get_profile_params(profile, Market::US);
        
        assert!(
            params_br.portfolio_risk.volatility_target >= params_us.portfolio_risk.volatility_target,
            "BR should have equal or higher vol target than US for {:?}", profile
        );
    }
}

#[test]
fn test_br_has_lower_liquidity_requirement() {
    for profile in available_profiles() {
        let params_br = get_profile_params(profile, Market::BR);
        let params_us = get_profile_params(profile, Market::US);
        
        assert!(
            params_br.operational.min_liquidity_usd <= params_us.operational.min_liquidity_usd,
            "BR should have equal or lower liquidity requirement than US for {:?}", profile
        );
    }
}

#[test]
fn test_br_allows_wider_spreads() {
    for profile in available_profiles() {
        let params_br = get_profile_params(profile, Market::BR);
        let params_us = get_profile_params(profile, Market::US);
        
        assert!(
            params_br.operational.max_spread_bps >= params_us.operational.max_spread_bps,
            "BR should allow equal or wider spreads than US for {:?}", profile
        );
    }
}

// =============================================================================
// Risk Ordering Tests
// =============================================================================

#[test]
fn test_profiles_ordered_by_risk() {
    let muito_cons = get_profile_params(RiskProfile::MuitoConservador, Market::BR);
    let conservador = get_profile_params(RiskProfile::Conservador, Market::BR);
    let moderado = get_profile_params(RiskProfile::Moderado, Market::BR);
    let arrojado = get_profile_params(RiskProfile::Arrojado, Market::BR);
    let muito_arr = get_profile_params(RiskProfile::MuitoArrojado, Market::BR);
    
    // Risk per trade should increase with aggressiveness
    assert!(
        muito_cons.sizing.max_risk_per_trade_pct < conservador.sizing.max_risk_per_trade_pct,
        "Risk per trade should increase from muito_conservador to conservador"
    );
    assert!(
        conservador.sizing.max_risk_per_trade_pct < moderado.sizing.max_risk_per_trade_pct,
        "Risk per trade should increase from conservador to moderado"
    );
    assert!(
        moderado.sizing.max_risk_per_trade_pct < arrojado.sizing.max_risk_per_trade_pct,
        "Risk per trade should increase from moderado to arrojado"
    );
    assert!(
        arrojado.sizing.max_risk_per_trade_pct < muito_arr.sizing.max_risk_per_trade_pct,
        "Risk per trade should increase from arrojado to muito_arrojado"
    );
}

#[test]
fn test_drawdown_limits_ordered() {
    let muito_cons = get_profile_params(RiskProfile::MuitoConservador, Market::BR);
    let conservador = get_profile_params(RiskProfile::Conservador, Market::BR);
    let moderado = get_profile_params(RiskProfile::Moderado, Market::BR);
    let arrojado = get_profile_params(RiskProfile::Arrojado, Market::BR);
    let muito_arr = get_profile_params(RiskProfile::MuitoArrojado, Market::BR);
    
    // Max drawdown (negative) should get more negative (larger absolute value) with aggressiveness
    assert!(
        muito_cons.portfolio_risk.max_drawdown_pct > conservador.portfolio_risk.max_drawdown_pct,
        "Drawdown limit should be tighter for muito_conservador"
    );
    assert!(
        conservador.portfolio_risk.max_drawdown_pct > moderado.portfolio_risk.max_drawdown_pct,
        "Drawdown limit should be tighter for conservador"
    );
    assert!(
        moderado.portfolio_risk.max_drawdown_pct > arrojado.portfolio_risk.max_drawdown_pct,
        "Drawdown limit should be tighter for moderado"
    );
    assert!(
        arrojado.portfolio_risk.max_drawdown_pct > muito_arr.portfolio_risk.max_drawdown_pct,
        "Drawdown limit should be tighter for arrojado"
    );
}

// =============================================================================
// Parameter Validation Tests
// =============================================================================

#[test]
fn test_kelly_never_exceeds_half() {
    for profile in available_profiles() {
        let params_br = get_profile_params(profile, Market::BR);
        let params_us = get_profile_params(profile, Market::US);
        
        assert!(
            params_br.sizing.kelly_fraction <= 0.5,
            "Kelly fraction should never exceed 0.5 for {:?} BR", profile
        );
        assert!(
            params_us.sizing.kelly_fraction <= 0.5,
            "Kelly fraction should never exceed 0.5 for {:?} US", profile
        );
    }
}

#[test]
fn test_risk_per_trade_within_bounds() {
    for profile in available_profiles() {
        let params = get_profile_params(profile, Market::BR);
        
        assert!(
            params.sizing.max_risk_per_trade_pct > 0.0,
            "Risk per trade must be positive for {:?}", profile
        );
        assert!(
            params.sizing.max_risk_per_trade_pct <= 0.03,
            "Risk per trade should not exceed 3% for {:?}", profile
        );
    }
}

#[test]
fn test_atr_multiplier_reasonable() {
    for profile in available_profiles() {
        let params = get_profile_params(profile, Market::BR);
        
        assert!(
            params.stops.atr_multiplier >= 1.0,
            "ATR multiplier must be >= 1.0 for {:?}", profile
        );
        assert!(
            params.stops.atr_multiplier <= 5.0,
            "ATR multiplier should not exceed 5.0 for {:?}", profile
        );
    }
}

#[test]
fn test_drawdown_limit_negative() {
    for profile in available_profiles() {
        let params = get_profile_params(profile, Market::BR);
        
        assert!(
            params.portfolio_risk.max_drawdown_pct < 0.0,
            "Max drawdown must be negative for {:?}", profile
        );
    }
}

// =============================================================================
// TOML Loader Tests
// =============================================================================

#[test]
fn test_loader_builtin_fallback() {
    let loader = RiskProfileLoader::new("nonexistent_directory");
    
    // Should fall back to built-in profiles
    let result = loader.load("moderado", Market::BR);
    assert!(result.is_ok(), "Should fall back to built-in profile");
    
    let params = result.unwrap();
    assert_eq!(params.profile, RiskProfile::Moderado);
}

#[test]
fn test_loader_unknown_profile_error() {
    let loader = RiskProfileLoader::default();
    
    let result = loader.load("nonexistent_profile", Market::BR);
    assert!(result.is_err(), "Should error on unknown profile");
}

// =============================================================================
// Warning System Tests
// =============================================================================

#[test]
fn test_empty_universe_warning_format() {
    let warning = EntryWarning::EmptyUniverse {
        candidates_before: 50,
        gating_excluded: 50,
        top_reasons: vec![
            "liquidez insuficiente (25)".to_string(),
            "sem dados de preço (15)".to_string(),
            "fora do top-N (10)".to_string(),
        ],
    };
    
    let display = format!("{}", warning);
    assert!(display.contains("EMPTY UNIVERSE"));
    assert!(display.contains("50"));
    assert!(display.contains("liquidez"));
}

#[test]
fn test_low_universe_warning() {
    let warning = EntryWarning::LowUniverse {
        eligible_count: 3,
        recommended_min: 10,
    };
    
    let display = format!("{}", warning);
    assert!(display.contains("LOW UNIVERSE"));
    assert!(display.contains("3"));
    assert!(display.contains("10"));
}

// =============================================================================
// Profile Description Tests
// =============================================================================

#[test]
fn test_profile_descriptions() {
    for profile in available_profiles() {
        let description = profile.description();
        assert!(!description.is_empty(), "Profile {:?} should have description", profile);
        assert!(description.len() > 20, "Description should be meaningful for {:?}", profile);
    }
}

#[test]
fn test_expected_drawdown_matches_params() {
    for profile in available_profiles() {
        let expected_dd = profile.expected_max_drawdown();
        let params = get_profile_params(profile, Market::BR);
        
        // Expected drawdown should match params (but params is negative)
        assert!(
            (expected_dd + params.portfolio_risk.max_drawdown_pct).abs() < 0.001,
            "Expected drawdown should match params for {:?}", profile
        );
    }
}

// =============================================================================
// Display and Serialization Tests
// =============================================================================

#[test]
fn test_profile_display_roundtrip() {
    for profile in available_profiles() {
        let display = profile.to_string();
        let parsed: RiskProfile = display.parse().unwrap();
        assert_eq!(profile, parsed, "Display/parse should roundtrip for {:?}", profile);
    }
}

#[test]
fn test_profile_params_serializable() {
    let params = get_profile_params(RiskProfile::Moderado, Market::BR);
    
    // Should serialize to JSON without error
    let json = serde_json::to_string(&params);
    assert!(json.is_ok(), "Should serialize to JSON");
    
    // Should deserialize back
    let deserialized: RiskProfileParams = serde_json::from_str(&json.unwrap()).unwrap();
    assert_eq!(deserialized.profile, params.profile);
}





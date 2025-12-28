//! Golden tests for FX module JSON schema stability.
//!
//! These tests verify that the JSON output format remains stable across versions.
//! If the schema changes intentionally, update the golden files.

use backtester_intelligence::performance::{
    PerformanceReport, FxAttributionJson, CurrencyExposureJson, FxRateUsedJson,
    CurrencyAttributionJson, PnLJson, CostsJson, RiskJson, ExposureJson,
    TurnoverJson, MarketExposure, AttributionJson,
    PERFORMANCE_REPORT_SCHEMA_VERSION,
};
use serde_json::Value;

// =============================================================================
// SCHEMA VERSION TESTS
// =============================================================================

#[test]
fn test_schema_version_constant() {
    // Verify the current schema version (v1.3 adds compliance/risk controls)
    assert_eq!(
        PERFORMANCE_REPORT_SCHEMA_VERSION, 
        "fx_report_v1.3",
        "Schema version should be fx_report_v1.3"
    );
}

#[test]
fn test_schema_version_in_report() {
    let report = create_sample_report();
    assert_eq!(
        report.schema_version, 
        PERFORMANCE_REPORT_SCHEMA_VERSION,
        "Report schema_version should match constant"
    );
}

// =============================================================================
// GOLDEN FILE TESTS
// =============================================================================

#[test]
fn test_golden_file_parses() {
    // Test v1.1 backward compatibility - should still parse
    let golden_v11 = include_str!("golden/performance_report_v1.1.json");
    let parsed_v11: Result<PerformanceReport, _> = serde_json::from_str(golden_v11);
    
    assert!(parsed_v11.is_ok(), "Golden v1.1 file should parse: {:?}", parsed_v11.err());
    let report_v11 = parsed_v11.unwrap();
    assert_eq!(report_v11.schema_version, "fx_report_v1.1");
    assert_eq!(report_v11.date, "2024-12-27");
    // v1.1 should not have new fields
    assert!(report_v11.sector_exposure.is_none());
    assert!(report_v11.concentration.is_none());
    assert!(report_v11.regime_summary.is_none());
    
    // Test v1.2 with research-grade fields
    let golden_v12 = include_str!("golden/performance_report_v1.2.json");
    let parsed_v12: Result<PerformanceReport, _> = serde_json::from_str(golden_v12);
    
    assert!(parsed_v12.is_ok(), "Golden v1.2 file should parse: {:?}", parsed_v12.err());
    let report_v12 = parsed_v12.unwrap();
    assert_eq!(report_v12.schema_version, "fx_report_v1.2");
    // v1.2 should have research fields but not compliance
    assert!(report_v12.compliance.is_none());
    
    // Test v1.3 with compliance/risk controls
    let golden_v13 = include_str!("golden/performance_report_v1.3.json");
    let parsed_v13: Result<PerformanceReport, _> = serde_json::from_str(golden_v13);
    
    assert!(parsed_v13.is_ok(), "Golden v1.3 file should parse: {:?}", parsed_v13.err());
    let report_v13 = parsed_v13.unwrap();
    assert_eq!(report_v13.schema_version, "fx_report_v1.3");
    // v1.3 should have compliance field
    assert!(report_v13.compliance.is_some());
}

#[test]
fn test_golden_schema_version_matches() {
    // v1.3 golden file should match current version
    let golden = include_str!("golden/performance_report_v1.3.json");
    let parsed: Value = serde_json::from_str(golden).unwrap();
    
    let schema_version = parsed.get("schema_version")
        .and_then(|v| v.as_str())
        .unwrap();
    
    assert_eq!(
        schema_version, 
        PERFORMANCE_REPORT_SCHEMA_VERSION,
        "Golden file schema version should match current version"
    );
}

#[test]
fn test_golden_has_required_fields() {
    // Test v1.1 FX fields
    let golden_v11 = include_str!("golden/performance_report_v1.1.json");
    let parsed_v11: Value = serde_json::from_str(golden_v11).unwrap();
    
    // Core required fields
    assert!(parsed_v11.get("schema_version").is_some(), "Missing schema_version");
    assert!(parsed_v11.get("date").is_some(), "Missing date");
    assert!(parsed_v11.get("equity").is_some(), "Missing equity");
    assert!(parsed_v11.get("return_pct").is_some(), "Missing return_pct");
    assert!(parsed_v11.get("pnl").is_some(), "Missing pnl");
    assert!(parsed_v11.get("costs").is_some(), "Missing costs");
    assert!(parsed_v11.get("risk").is_some(), "Missing risk");
    assert!(parsed_v11.get("exposure").is_some(), "Missing exposure");
    assert!(parsed_v11.get("turnover").is_some(), "Missing turnover");
    
    // FX fields (V1.1)
    assert!(parsed_v11.get("base_currency").is_some(), "Missing base_currency");
    assert!(parsed_v11.get("fx_attribution").is_some(), "Missing fx_attribution");
    assert!(parsed_v11.get("fx_rates_used").is_some(), "Missing fx_rates_used");
    
    // Test v1.2 research-grade fields
    let golden_v12 = include_str!("golden/performance_report_v1.2.json");
    let parsed_v12: Value = serde_json::from_str(golden_v12).unwrap();
    
    // Research-grade fields (V1.2)
    assert!(parsed_v12.get("sector_exposure").is_some(), "Missing sector_exposure");
    assert!(parsed_v12.get("concentration").is_some(), "Missing concentration");
    assert!(parsed_v12.get("regime_summary").is_some(), "Missing regime_summary");
}

#[test]
fn test_golden_fx_rates_used_has_audit_fields() {
    let golden = include_str!("golden/performance_report_v1.1.json");
    let parsed: Value = serde_json::from_str(golden).unwrap();
    
    let fx_rates = parsed.get("fx_rates_used")
        .and_then(|v| v.as_array())
        .unwrap();
    
    assert!(!fx_rates.is_empty(), "fx_rates_used should not be empty");
    
    let first_rate = &fx_rates[0];
    
    // V1.1 audit fields
    assert!(first_rate.get("pair_requested").is_some(), "Missing pair_requested");
    assert!(first_rate.get("date_requested").is_some(), "Missing date_requested");
    assert!(first_rate.get("pair_resolved").is_some(), "Missing pair_resolved");
    assert!(first_rate.get("date_resolved").is_some(), "Missing date_resolved");
    assert!(first_rate.get("rate").is_some(), "Missing rate");
    assert!(first_rate.get("method").is_some(), "Missing method");
}

#[test]
fn test_golden_fx_attribution_has_decomposition() {
    let golden = include_str!("golden/performance_report_v1.1.json");
    let parsed: Value = serde_json::from_str(golden).unwrap();
    
    let fx_attr = parsed.get("fx_attribution").unwrap();
    
    // 3-term decomposition fields
    assert!(fx_attr.get("asset_return_pct").is_some(), "Missing asset_return_pct");
    assert!(fx_attr.get("fx_return_pct").is_some(), "Missing fx_return_pct");
    assert!(fx_attr.get("interaction_pct").is_some(), "Missing interaction_pct");
    assert!(fx_attr.get("total_return_base_pct").is_some(), "Missing total_return_base_pct");
    assert!(fx_attr.get("by_currency").is_some(), "Missing by_currency");
}

// =============================================================================
// DETERMINISM TESTS
// =============================================================================

#[test]
fn test_serialization_determinism() {
    let report1 = create_sample_report();
    let report2 = create_sample_report();
    
    let json1 = serde_json::to_string_pretty(&report1).unwrap();
    let json2 = serde_json::to_string_pretty(&report2).unwrap();
    
    assert_eq!(json1, json2, "Serialization should be deterministic");
}

#[test]
fn test_roundtrip_serialization() {
    let original = create_sample_report();
    
    let json = serde_json::to_string(&original).unwrap();
    let parsed: PerformanceReport = serde_json::from_str(&json).unwrap();
    
    assert_eq!(original.schema_version, parsed.schema_version);
    assert_eq!(original.date, parsed.date);
    assert_eq!(original.equity, parsed.equity);
}

// =============================================================================
// HELPERS
// =============================================================================

fn create_sample_report() -> PerformanceReport {
    PerformanceReport {
        schema_version: PERFORMANCE_REPORT_SCHEMA_VERSION.to_string(),
        date: "2024-12-27".to_string(),
        equity: "105000.00".to_string(),
        return_pct: "5.00".to_string(),
        drawdown_pct: "2.00".to_string(),
        pnl: PnLJson {
            realized: "3000.00".to_string(),
            unrealized: "2000.00".to_string(),
            total: "5000.00".to_string(),
        },
        costs: CostsJson {
            total: "150.00".to_string(),
            fees: "100.00".to_string(),
            slippage: "50.00".to_string(),
        },
        attribution: vec![
            AttributionJson {
                technique: "momentum".to_string(),
                weight_pct: "60.00".to_string(),
                pnl: "3000.00".to_string(),
            },
        ],
        risk: RiskJson {
            vol_ann: "15.00".to_string(),
            var_95: "-2500.00".to_string(),
            sharpe: "1.25".to_string(),
        },
        exposure: ExposureJson {
            gross: "100000.00".to_string(),
            net: "80000.00".to_string(),
            by_market: vec![
                MarketExposure {
                    market: "BR".to_string(),
                    value: "60000.00".to_string(),
                },
            ],
        },
        turnover: TurnoverJson {
            buy: "10000.00".to_string(),
            sell: "8000.00".to_string(),
            pct: "18.00".to_string(),
        },
        base_currency: Some("BRL".to_string()),
        equity_base: Some("115000.00".to_string()),
        fx_attribution: Some(FxAttributionJson {
            base_currency: "BRL".to_string(),
            asset_return_pct: "4.00".to_string(),
            fx_return_pct: "1.00".to_string(),
            interaction_pct: "0.04".to_string(),
            total_return_base_pct: "5.04".to_string(),
            by_currency: vec![
                CurrencyAttributionJson {
                    currency: "BRL".to_string(),
                    asset_return_pct: "5.00".to_string(),
                    fx_return_pct: "0.00".to_string(),
                    interaction_pct: "0.00".to_string(),
                    weight_pct: "60.00".to_string(),
                },
            ],
        }),
        exposure_by_currency: Some(vec![
            CurrencyExposureJson {
                currency: "BRL".to_string(),
                value_local: "60000.00".to_string(),
                value_base: "60000.00".to_string(),
                weight_pct: "100.00".to_string(),
            },
        ]),
        fx_rates_used: Some(vec![
            FxRateUsedJson {
                pair_requested: "USD/BRL".to_string(),
                date_requested: "2024-12-27".to_string(),
                pair_resolved: "USD/BRL".to_string(),
                date_resolved: "2024-12-27".to_string(),
                rate: "5.50".to_string(),
                method: "Direct".to_string(),
                pair: None,
                date: None,
            },
        ]),
        // Research-grade fields (optional)
        sector_exposure: None,
        concentration: None,
        regime_summary: None,
        // Compliance fields (v1.3)
        compliance: None,
    }
}




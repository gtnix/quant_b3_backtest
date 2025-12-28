//! Performance Module Golden Tests
//!
//! Locks down output format for regression detection.

use backtester_intelligence::performance::{
    PerformanceSnapshot, PnLBreakdown, CostBreakdown, ExposureBreakdown,
    DrawdownMetrics, TurnoverMetrics, AttributionBreakdown, TechniqueAttribution,
    VolatilityMetrics, VaRMetrics, VaRMethod, PerformanceReporter,
    SectorExposure, ConcentrationMetrics, ConcentrationCalculator,
    RegimeConfig, RegimeSummary, RegimePerformance, TrendState, VolQuantile,
    PERFORMANCE_REPORT_SCHEMA_VERSION,
};
use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::BTreeMap;

fn make_snapshot() -> PerformanceSnapshot {
    PerformanceSnapshot {
        date: NaiveDate::from_ymd_opt(2025, 1, 15).unwrap(),
        equity: dec!(105000),
        cash: dec!(50000),
        base_currency: None,
        equity_base: None,
        cash_base: None,
        fx_rates_used: None,
        exposure: ExposureBreakdown {
            gross: dec!(55000),
            net: dec!(55000),
            long: dec!(55000),
            short: Decimal::ZERO,
            by_market: [
                ("BR".to_string(), dec!(30000)),
                ("US".to_string(), dec!(25000)),
            ].into_iter().collect(),
            by_currency: Default::default(),
            by_currency_base: Default::default(),
            by_sector: Default::default(),
        },
        pnl: PnLBreakdown {
            realized: dec!(3000),
            unrealized: dec!(2000),
            total: dec!(5000),
            by_market: BTreeMap::new(),
            by_symbol: BTreeMap::new(),
        },
        costs: CostBreakdown {
            fees_br: dec!(100),
            fees_us: dec!(50),
            slippage_br: dec!(20),
            slippage_us: dec!(10),
            total: dec!(180),
        },
        drawdown: DrawdownMetrics {
            current_dd: dec!(0.02),
            max_dd: dec!(0.05),
            dd_duration_days: 3,
            hwm: dec!(107000),
        },
        turnover: TurnoverMetrics {
            buy_notional: dec!(10000),
            sell_notional: dec!(8000),
            turnover_pct: dec!(17.14),
        },
    }
}

fn make_attribution() -> AttributionBreakdown {
    AttributionBreakdown {
        by_technique: vec![
            TechniqueAttribution {
                technique_name: "momentum".to_string(),
                weight_pct: dec!(40),
                pnl_contribution: dec!(2000),
                return_contribution: dec!(40),
            },
            TechniqueAttribution {
                technique_name: "value".to_string(),
                weight_pct: dec!(30),
                pnl_contribution: dec!(1500),
                return_contribution: dec!(30),
            },
            TechniqueAttribution {
                technique_name: "quality".to_string(),
                weight_pct: dec!(30),
                pnl_contribution: dec!(1500),
                return_contribution: dec!(30),
            },
        ],
        total_pnl: dec!(5000),
        residual: Decimal::ZERO,
    }
}

fn make_vol() -> VolatilityMetrics {
    VolatilityMetrics {
        daily_vol: dec!(0.015),
        annualized_vol: dec!(0.238),
        rolling_window: 21,
    }
}

fn make_var() -> VaRMetrics {
    VaRMetrics {
        var_95: dec!(-2500),
        var_99: dec!(-4000),
        method: VaRMethod::Historical,
    }
}

// ==============================================================
// Golden Summary Test
// ==============================================================

const GOLDEN_SUMMARY: &str = r#"PERFORMANCE SNAPSHOT 2025-01-15
==================================================
Equity: 105000.00
Cash: 50000.00
Return: 5.00%
Drawdown: 2.00%
Max Drawdown: 5.00%

P&L BREAKDOWN
------------------------------
Realized: 3000.00
Unrealized: 2000.00
Total: 5000.00

COSTS
------------------------------
BR Fees: 100.00
US Fees: 50.00
BR Slippage: 20.00
US Slippage: 10.00
Total: 180.00

ATTRIBUTION
------------------------------
momentum: 2000.00 (40.00%)
quality: 1500.00 (30.00%)
value: 1500.00 (30.00%)

RISK METRICS
------------------------------
Daily Vol: 1.50%
Ann Vol: 23.80%
VaR 95%: -2500.00
VaR 99%: -4000.00
Sharpe: 1.25

EXPOSURE
------------------------------
Gross: 55000.00
Net: 55000.00
BR: 30000.00
US: 25000.00

TURNOVER
------------------------------
Buy: 10000.00
Sell: 8000.00
Turnover: 17.14%"#;

#[test]
fn golden_summary() {
    let reporter = PerformanceReporter::default();
    let snapshot = make_snapshot();
    let attr = make_attribution();
    let vol = make_vol();
    let var = make_var();
    
    let summary = reporter.to_summary(&snapshot, &attr, &vol, &var, dec!(1.25), dec!(100000));
    
    assert_eq!(summary, GOLDEN_SUMMARY);
}

// ==============================================================
// Golden JSON Test
// ==============================================================

#[test]
fn golden_json_structure() {
    let reporter = PerformanceReporter::default();
    let snapshot = make_snapshot();
    let attr = make_attribution();
    let vol = make_vol();
    let var = make_var();
    
    let report = reporter.to_json(&snapshot, &attr, &vol, &var, dec!(1.25), dec!(100000));
    
    // Verify key fields
    assert_eq!(report.date, "2025-01-15");
    assert_eq!(report.equity, "105000.00");
    assert_eq!(report.return_pct, "5.00");
    assert_eq!(report.pnl.total, "5000.00");
    assert_eq!(report.costs.total, "180.00");
    assert_eq!(report.attribution.len(), 3);
    
    // Attribution should be sorted by technique name
    assert_eq!(report.attribution[0].technique, "momentum");
    assert_eq!(report.attribution[1].technique, "quality");
    assert_eq!(report.attribution[2].technique, "value");
}

#[test]
fn golden_json_parses() {
    let reporter = PerformanceReporter::default();
    let snapshot = make_snapshot();
    let attr = make_attribution();
    let vol = make_vol();
    let var = make_var();
    
    let json_str = reporter.to_json_string(&snapshot, &attr, &vol, &var, dec!(1.25), dec!(100000));
    
    // Should be valid JSON
    let parsed: serde_json::Value = serde_json::from_str(&json_str).expect("Should parse as JSON");
    
    // Verify structure
    assert!(parsed.get("date").is_some());
    assert!(parsed.get("equity").is_some());
    assert!(parsed.get("pnl").is_some());
    assert!(parsed.get("costs").is_some());
    assert!(parsed.get("attribution").is_some());
    assert!(parsed.get("risk").is_some());
    assert!(parsed.get("exposure").is_some());
    assert!(parsed.get("turnover").is_some());
}

// ==============================================================
// Determinism Tests
// ==============================================================

#[test]
fn determinism_summary() {
    let reporter = PerformanceReporter::default();
    let snapshot = make_snapshot();
    let attr = make_attribution();
    let vol = make_vol();
    let var = make_var();
    
    let s1 = reporter.to_summary(&snapshot, &attr, &vol, &var, dec!(1.25), dec!(100000));
    let s2 = reporter.to_summary(&snapshot, &attr, &vol, &var, dec!(1.25), dec!(100000));
    
    assert_eq!(s1, s2);
}

#[test]
fn determinism_json() {
    let reporter = PerformanceReporter::default();
    let snapshot = make_snapshot();
    let attr = make_attribution();
    let vol = make_vol();
    let var = make_var();
    
    let j1 = reporter.to_json_string(&snapshot, &attr, &vol, &var, dec!(1.25), dec!(100000));
    let j2 = reporter.to_json_string(&snapshot, &attr, &vol, &var, dec!(1.25), dec!(100000));
    
    assert_eq!(j1, j2);
}

#[test]
fn determinism_cross_run() {
    // Run multiple times to detect any non-determinism
    for _ in 0..10 {
        let reporter = PerformanceReporter::default();
        let snapshot = make_snapshot();
        let attr = make_attribution();
        let vol = make_vol();
        let var = make_var();
        
        let summary = reporter.to_summary(&snapshot, &attr, &vol, &var, dec!(1.25), dec!(100000));
        assert_eq!(summary, GOLDEN_SUMMARY);
    }
}

// ==============================================================
// Helper: Print Golden Output (ignored, for updates)
// ==============================================================

#[test]
#[ignore]
fn generate_golden_output() {
    let reporter = PerformanceReporter::default();
    let snapshot = make_snapshot();
    let attr = make_attribution();
    let vol = make_vol();
    let var = make_var();
    
    let summary = reporter.to_summary(&snapshot, &attr, &vol, &var, dec!(1.25), dec!(100000));
    println!("=== GOLDEN SUMMARY ===");
    println!("{}", summary);
    println!("=== END ===");
    
    let json = reporter.to_json_string(&snapshot, &attr, &vol, &var, dec!(1.25), dec!(100000));
    println!("=== GOLDEN JSON ===");
    println!("{}", json);
    println!("=== END ===");
}

// ==============================================================
// Research-Grade Report Tests (v1.2/v1.3)
// ==============================================================

fn make_snapshot_with_sectors() -> PerformanceSnapshot {
    let mut snapshot = make_snapshot();
    snapshot.exposure.by_sector = vec![
        SectorExposure {
            sector: "Energy".to_string(),
            gross: dec!(22000),
            net: dec!(22000),
            long: dec!(22000),
            short: Decimal::ZERO,
            weight_pct: dec!(40),
        },
        SectorExposure {
            sector: "Financials".to_string(),
            gross: dec!(19250),
            net: dec!(19250),
            long: dec!(19250),
            short: Decimal::ZERO,
            weight_pct: dec!(35),
        },
        SectorExposure {
            sector: "Materials".to_string(),
            gross: dec!(13750),
            net: dec!(13750),
            long: dec!(13750),
            short: Decimal::ZERO,
            weight_pct: dec!(25),
        },
    ];
    snapshot
}

fn make_concentration() -> ConcentrationMetrics {
    ConcentrationCalculator::calculate(&[
        ("PETR4".to_string(), dec!(22000)),
        ("ITUB4".to_string(), dec!(19250)),
        ("VALE3".to_string(), dec!(13750)),
    ])
}

fn make_regime_summary() -> RegimeSummary {
    RegimeSummary {
        config: RegimeConfig::default(),
        by_regime: vec![
            RegimePerformance {
                trend_state: TrendState::Uptrend,
                vol_quantile: VolQuantile::Q3,
                day_count: 15,
                mean_return_pct: dec!(0.50),
                cumulative_return_pct: dec!(7.50),
                win_rate_pct: dec!(80),
                mean_turnover_pct: dec!(5),
                mean_cost_pct: dec!(0.10),
            },
            RegimePerformance {
                trend_state: TrendState::Sideways,
                vol_quantile: VolQuantile::Q2,
                day_count: 10,
                mean_return_pct: dec!(0.10),
                cumulative_return_pct: dec!(1.00),
                win_rate_pct: dec!(55),
                mean_turnover_pct: dec!(3),
                mean_cost_pct: dec!(0.05),
            },
        ],
        total_days: 30,
        warmup_days: 5,
    }
}

#[test]
fn golden_schema_version() {
    assert_eq!(PERFORMANCE_REPORT_SCHEMA_VERSION, "fx_report_v1.3");
}

#[test]
fn golden_research_grade_report() {
    let reporter = PerformanceReporter::default();
    let snapshot = make_snapshot_with_sectors();
    let attr = make_attribution();
    let vol = make_vol();
    let var = make_var();
    let concentration = make_concentration();
    let regime = make_regime_summary();
    
    let report = reporter.to_json_full(
        &snapshot,
        &attr,
        &vol,
        &var,
        dec!(1.25),
        dec!(100000),
        None,
        Some(&concentration),
        Some(&regime),
    );
    
    // Verify schema version
    assert_eq!(report.schema_version, "fx_report_v1.3");
    
    // Verify sector exposure
    assert!(report.sector_exposure.is_some());
    let sectors = report.sector_exposure.as_ref().unwrap();
    assert_eq!(sectors.len(), 3);
    assert_eq!(sectors[0].sector, "Energy");
    
    // Verify concentration
    assert!(report.concentration.is_some());
    let conc = report.concentration.as_ref().unwrap();
    assert_eq!(conc.n_positions, 3);
    assert!(conc.gini.is_some());
    
    // Verify regime summary
    assert!(report.regime_summary.is_some());
    let regime = report.regime_summary.as_ref().unwrap();
    assert_eq!(regime.total_days, 30);
    assert_eq!(regime.warmup_days, 5);
    assert_eq!(regime.by_regime.len(), 2);
}

#[test]
fn golden_concentration_metrics() {
    let conc = make_concentration();
    
    // HHI for 3 positions with weights ~40%, ~35%, ~25%
    // HHI = 0.4^2 + 0.35^2 + 0.25^2 = 0.16 + 0.1225 + 0.0625 = 0.345
    assert!(conc.hhi > dec!(0.3) && conc.hhi < dec!(0.4));
    
    // Effective N = 1/HHI ≈ 2.9
    assert!(conc.effective_n > dec!(2.5) && conc.effective_n < dec!(3.5));
    
    // Top 1 weight is largest position (~40%)
    assert!(conc.top_1_weight_pct > dec!(35) && conc.top_1_weight_pct < dec!(45));
    
    // n_positions
    assert_eq!(conc.n_positions, 3);
    
    // Gini should be calculated
    assert!(conc.gini.is_some());
}

#[test]
fn golden_regime_fields() {
    let regime = make_regime_summary();
    
    // Check best/worst regime
    let best = regime.best_regime().unwrap();
    assert_eq!(best.trend_state, TrendState::Uptrend);
    
    let worst = regime.worst_regime().unwrap();
    assert_eq!(worst.trend_state, TrendState::Sideways);
    
    // Get performance for specific regime
    let uptrend_q3 = regime.get_performance(TrendState::Uptrend, VolQuantile::Q3);
    assert!(uptrend_q3.is_some());
    assert_eq!(uptrend_q3.unwrap().day_count, 15);
}

#[test]
fn golden_sector_exposure_sums() {
    let snapshot = make_snapshot_with_sectors();
    
    // Sector weights should sum to 100%
    let total_weight: Decimal = snapshot.exposure.by_sector.iter()
        .map(|s| s.weight_pct)
        .sum();
    assert_eq!(total_weight, dec!(100));
    
    // Gross should match sum of sector gross
    let total_gross: Decimal = snapshot.exposure.by_sector.iter()
        .map(|s| s.gross)
        .sum();
    assert_eq!(total_gross, snapshot.exposure.gross);
}

#[test]
fn golden_report_parses_fixture() {
    // Read the v1.2 fixture and verify it parses (backward compatibility)
    let fixture_v12 = include_str!("golden/performance_report_v1.2.json");
    let report_v12: serde_json::Value = serde_json::from_str(fixture_v12)
        .expect("v1.2 fixture should parse as JSON");
    
    // Verify v1.2 fields exist
    assert!(report_v12.get("sector_exposure").is_some());
    assert!(report_v12.get("concentration").is_some());
    assert!(report_v12.get("regime_summary").is_some());
    assert_eq!(report_v12["schema_version"], "fx_report_v1.2");
    
    // Read the v1.3 fixture and verify it parses
    let fixture_v13 = include_str!("golden/performance_report_v1.3.json");
    let report_v13: serde_json::Value = serde_json::from_str(fixture_v13)
        .expect("v1.3 fixture should parse as JSON");
    
    // Verify v1.3 compliance field exists
    assert!(report_v13.get("compliance").is_some());
    assert_eq!(report_v13["schema_version"], "fx_report_v1.3");
    
    // Verify compliance structure
    let compliance = report_v13.get("compliance").unwrap();
    assert!(compliance.get("config_snapshot").is_some());
    assert!(compliance.get("summary").is_some());
    assert!(compliance.get("breaches").is_some());
    assert!(compliance.get("actions_taken").is_some());
}










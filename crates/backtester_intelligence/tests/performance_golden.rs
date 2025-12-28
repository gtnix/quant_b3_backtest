//! Performance Module Golden Tests
//!
//! Locks down output format for regression detection.

use backtester_intelligence::performance::{
    PerformanceSnapshot, PnLBreakdown, CostBreakdown, ExposureBreakdown,
    DrawdownMetrics, TurnoverMetrics, AttributionBreakdown, TechniqueAttribution,
    VolatilityMetrics, VaRMetrics, VaRMethod, PerformanceReporter,
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
        exposure: ExposureBreakdown {
            gross: dec!(55000),
            net: dec!(55000),
            long: dec!(55000),
            short: Decimal::ZERO,
            by_market: [
                ("BR".to_string(), dec!(30000)),
                ("US".to_string(), dec!(25000)),
            ].into_iter().collect(),
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










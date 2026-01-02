//! Performance Module Benchmarks
//!
//! Budgets:
//! - Snapshot generation: < 1ms for 100 positions
//! - Attribution: < 5ms for 100 positions
//! - Full report: < 10ms for 100 positions

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use backtester_intelligence::performance::{
    TradeLedger, PerformanceEngine, AttributionEngine, RiskCalculator,
    PerformanceReporter, VolatilityMetrics, VaRMetrics,
};
use backtester_intelligence::filters::Market;
use backtester_intelligence::performance::engine::PerformanceConfig;
use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::BTreeMap;

fn make_date(day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(2025, 1, day.min(28)).unwrap()
}

fn setup_engine(n: usize) -> (PerformanceEngine, BTreeMap<String, Decimal>) {
    let mut engine = PerformanceEngine::new(PerformanceConfig::default(), dec!(1000000));
    
    for i in 0..n {
        let symbol = format!("SYM{:05}", i);
        let market = if i % 2 == 0 { Market::BR } else { Market::US };
        engine.record_buy(make_date(1), &symbol, 100, dec!(100), dec!(10), market);
    }
    
    let prices: BTreeMap<String, Decimal> = (0..n)
        .map(|i| (format!("SYM{:05}", i), dec!(105)))
        .collect();
    
    (engine, prices)
}

fn setup_ledger(n: usize) -> (TradeLedger, BTreeMap<String, Decimal>) {
    let mut ledger = TradeLedger::new();
    
    for i in 0..n {
        let symbol = format!("SYM{:05}", i);
        let market = if i % 2 == 0 { Market::BR } else { Market::US };
        ledger.record_buy(make_date(1), &symbol, 100, dec!(100), dec!(10), market);
    }
    
    let prices: BTreeMap<String, Decimal> = (0..n)
        .map(|i| (format!("SYM{:05}", i), dec!(105)))
        .collect();
    
    (ledger, prices)
}

fn setup_attribution(n: usize) -> (AttributionEngine, BTreeMap<String, Decimal>) {
    let mut attr = AttributionEngine::new();
    
    for i in 0..n {
        let symbol = format!("SYM{:05}", i);
        let weights: BTreeMap<String, Decimal> = [
            ("momentum".to_string(), dec!(0.3)),
            ("value".to_string(), dec!(0.3)),
            ("quality".to_string(), dec!(0.2)),
            ("low_vol".to_string(), dec!(0.2)),
        ].into_iter().collect();
        attr.record_entry_weights(&symbol, weights);
    }
    
    let pnl: BTreeMap<String, Decimal> = (0..n)
        .map(|i| (format!("SYM{:05}", i), Decimal::from(i as i64 * 100)))
        .collect();
    
    (attr, pnl)
}

// ==============================================================
// Ledger Benchmarks
// ==============================================================

fn bench_ledger_pnl(c: &mut Criterion) {
    let mut group = c.benchmark_group("ledger_pnl");
    
    for n in [100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
            let (ledger, prices) = setup_ledger(n);
            b.iter(|| {
                black_box(ledger.get_pnl_breakdown(&prices))
            });
        });
    }
    
    group.finish();
}

// ==============================================================
// Snapshot Benchmarks
// ==============================================================

fn bench_snapshot(c: &mut Criterion) {
    let mut group = c.benchmark_group("snapshot");
    
    for n in [100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
            let (mut engine, prices) = setup_engine(n);
            b.iter(|| {
                black_box(engine.generate_snapshot(make_date(2), dec!(0), &prices))
            });
        });
    }
    
    group.finish();
}

// ==============================================================
// Attribution Benchmarks
// ==============================================================

fn bench_attribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("attribution");
    
    for n in [100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
            let (attr, pnl) = setup_attribution(n);
            b.iter(|| {
                black_box(attr.calculate_attribution(&pnl))
            });
        });
    }
    
    group.finish();
}

// ==============================================================
// Risk Calculation Benchmarks
// ==============================================================

fn bench_risk_var(c: &mut Criterion) {
    let mut group = c.benchmark_group("risk_var");
    
    for n in [100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
            let calc = RiskCalculator::default();
            let returns: Vec<Decimal> = (0..n)
                .map(|i| Decimal::from(i as i64 - n as i64 / 2) / Decimal::from(1000))
                .collect();
            b.iter(|| {
                black_box(calc.calculate_var(&returns, dec!(1000000)))
            });
        });
    }
    
    group.finish();
}

fn bench_risk_drawdown(c: &mut Criterion) {
    let mut group = c.benchmark_group("risk_drawdown");
    
    for n in [100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
            let calc = RiskCalculator::default();
            let equity: Vec<Decimal> = (0..n)
                .map(|i| dec!(100000) + Decimal::from(i as i64 * 100))
                .collect();
            b.iter(|| {
                black_box(calc.calculate_drawdown(&equity))
            });
        });
    }
    
    group.finish();
}

// ==============================================================
// Reporter Benchmarks
// ==============================================================

fn bench_reporter_json(c: &mut Criterion) {
    use backtester_intelligence::performance::{
        PerformanceSnapshot, PnLBreakdown, CostBreakdown, ExposureBreakdown,
        DrawdownMetrics, TurnoverMetrics, AttributionBreakdown, TechniqueAttribution,
    };
    
    let snapshot = PerformanceSnapshot {
        date: make_date(1),
        equity: dec!(1000000),
        cash: dec!(100000),
        exposure: ExposureBreakdown::default(),
        pnl: PnLBreakdown::with_values(dec!(10000), dec!(5000)),
        costs: CostBreakdown::default(),
        drawdown: DrawdownMetrics::default(),
        turnover: TurnoverMetrics::default(),
    };
    
    let attr = AttributionBreakdown {
        by_technique: (0..7).map(|i| TechniqueAttribution {
            technique_name: format!("technique_{}", i),
            weight_pct: dec!(14.28),
            pnl_contribution: dec!(2000),
            return_contribution: dec!(14.28),
        }).collect(),
        total_pnl: dec!(15000),
        residual: dec!(1000),
    };
    
    let vol = VolatilityMetrics::default();
    let var = VaRMetrics::default();
    let reporter = PerformanceReporter::default();
    
    c.bench_function("reporter_json", |b| {
        b.iter(|| {
            black_box(reporter.to_json_string(&snapshot, &attr, &vol, &var, dec!(1.0), dec!(1000000)))
        });
    });
}

criterion_group!(
    benches,
    bench_ledger_pnl,
    bench_snapshot,
    bench_attribution,
    bench_risk_var,
    bench_risk_drawdown,
    bench_reporter_json,
);

criterion_main!(benches);

























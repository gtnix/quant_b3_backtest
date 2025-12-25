//! Benchmarks for Entry Module performance at scale.
//!
//! Tests the entry pipeline with N=100, 1000, 10000 assets to validate
//! scalability and identify hot spots.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use chrono::NaiveDate;

use backtester_intelligence::entry::{
    AssetCandidate, EntryContext, EntryEngine, EntryEngineConfig,
    SelectionConfig, WeightingConfig, GatingConfig,
};
use backtester_intelligence::filters::Market;

/// Generate N synthetic BR candidates.
fn make_candidates(n: usize, market: Market) -> Vec<AssetCandidate> {
    (0..n).map(|i| {
        let mut c = AssetCandidate::new(format!("ASSET{:05}", i), market);
        c.price = Some(Decimal::from(20 + (i as i64 % 100)));
        c.avg_volume = Some(Decimal::from(1_000_000 + (i as i64 * 10_000)));
        c.price_days = 30;
        c.has_fundamentals = i % 10 != 0; // 90% have fundamentals
        c.has_dividends = i % 5 != 0;     // 80% have dividends
        c.volatility = Some(0.15 + ((i % 50) as f64) * 0.01);
        c.score = Some(0.90 - ((i % 100) as f64) * 0.008);
        c
    }).collect()
}

fn bench_entry_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("entry_pipeline");
    
    let config = EntryEngineConfig {
        gating: GatingConfig {
            require_fundamentals: false,
            require_dividends: false,
            ..Default::default()
        },
        selection: SelectionConfig {
            top_n_br: 20,
            top_n_us: 20,
            min_score_threshold: None,
        },
        weighting: WeightingConfig::default(),
        ..Default::default()
    };
    let engine = EntryEngine::new(config);
    
    let capital = dec!(1_000_000);
    let positions: HashMap<String, i64> = HashMap::new();
    let date = NaiveDate::from_ymd_opt(2025, 1, 3).unwrap();
    let ctx = EntryContext::new(date, capital, Market::BR);

    for size in [100, 1_000, 10_000] {
        let candidates = make_candidates(size, Market::BR);
        
        group.bench_with_input(
            BenchmarkId::new("full_pipeline", size),
            &candidates,
            |b, candidates| {
                b.iter(|| {
                    let (result, orders, audit) = engine.evaluate(
                        black_box(&ctx),
                        black_box(candidates.clone()),
                        black_box(&positions),
                    );
                    black_box((result, orders, audit))
                })
            },
        );
    }
    
    group.finish();
}

fn bench_gating(c: &mut Criterion) {
    use backtester_intelligence::entry::{GatingFilter, GatingCandidate, GatingConfig};
    
    let mut group = c.benchmark_group("gating");
    let filter = GatingFilter::new(GatingConfig::default());
    
    for size in [100, 1_000, 10_000] {
        let candidates: Vec<GatingCandidate> = (0..size).map(|i| {
            GatingCandidate {
                symbol: format!("SYM{:05}", i),
                market: Market::BR,
                price: Some(Decimal::from(20 + (i as i64 % 100))),
                avg_volume: Some(Decimal::from(1_000_000 + (i as i64 * 10_000))),
                price_days: 30,
                has_fundamentals: i % 10 != 0,
                has_dividends: i % 5 != 0,
                is_tradeable: true,
                fundamentals_as_of: None,
                rebalance_date: None,
            }
        }).collect();
        
        group.bench_with_input(
            BenchmarkId::new("apply", size),
            &candidates,
            |b, candidates| {
                b.iter(|| {
                    let (eligible, excluded) = filter.apply(black_box(candidates.clone()));
                    black_box((eligible, excluded))
                })
            },
        );
    }
    
    group.finish();
}

fn bench_selection(c: &mut Criterion) {
    use backtester_intelligence::entry::{Selector, SelectionConfig, ScoredCandidate};
    
    let mut group = c.benchmark_group("selection");
    let selector = Selector::new(SelectionConfig {
        top_n_br: 20,
        top_n_us: 20,
        min_score_threshold: None,
    });
    
    for size in [100, 1_000, 10_000] {
        let candidates: Vec<ScoredCandidate> = (0..size).map(|i| {
            ScoredCandidate::new(
                format!("SYM{:05}", i),
                Market::BR,
                0.90 - ((i % 100) as f64) * 0.008,
            )
        }).collect();
        
        group.bench_with_input(
            BenchmarkId::new("select", size),
            &candidates,
            |b, candidates| {
                b.iter(|| {
                    let (selected, excluded) = selector.select(black_box(candidates.clone()));
                    black_box((selected, excluded))
                })
            },
        );
    }
    
    group.finish();
}

fn bench_weighting(c: &mut Criterion) {
    use backtester_intelligence::entry::{Weighter, WeightingConfig, WeightingCandidate};
    
    let mut group = c.benchmark_group("weighting");
    let weighter = Weighter::new(WeightingConfig::default());
    
    for size in [10, 20, 50, 100] {
        let candidates: Vec<WeightingCandidate> = (0..size).map(|i| {
            WeightingCandidate::new(
                format!("SYM{:03}", i),
                0.80 - (i as f64 * 0.01),
                Some(0.15 + (i as f64 * 0.01)),
            )
        }).collect();
        
        group.bench_with_input(
            BenchmarkId::new("risk_parity", size),
            &candidates,
            |b, candidates| {
                b.iter(|| {
                    let weights = weighter.calculate_weights(black_box(candidates.clone()));
                    black_box(weights)
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_entry_pipeline,
    bench_gating,
    bench_selection,
    bench_weighting,
);

criterion_main!(benches);


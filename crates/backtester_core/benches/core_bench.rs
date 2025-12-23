//! Benchmark for core type operations.

use backtester_core::{Bar, MarketEvent};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_bar_clone(c: &mut Criterion) {
    let bar = Bar {
        timestamp: 1_700_000_000_000_000_000,
        open: 100.0,
        high: 105.0,
        low: 99.0,
        close: 104.0,
        volume: 10000.0,
    };

    c.bench_function("bar_clone", |b| b.iter(|| black_box(bar)));
}

fn bench_market_event_create(c: &mut Criterion) {
    let bar = Bar {
        timestamp: 1_700_000_000_000_000_000,
        open: 100.0,
        high: 105.0,
        low: 99.0,
        close: 104.0,
        volume: 10000.0,
    };

    c.bench_function("market_event_create", |b| {
        b.iter(|| black_box(MarketEvent { asset_id: 1, bar }))
    });
}

criterion_group!(benches, bench_bar_clone, bench_market_event_create);
criterion_main!(benches);

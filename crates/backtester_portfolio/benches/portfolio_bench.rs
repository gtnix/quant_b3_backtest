//! Benchmark for portfolio operations.

use backtester_core::FillEvent;
use backtester_portfolio::Portfolio;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_process_fill(c: &mut Criterion) {
    let mut portfolio = Portfolio::new(1_000_000.0, 10);
    let fill = FillEvent {
        timestamp: 1_700_000_000_000_000_000,
        asset_id: 1,
        quantity: 100,
        price: 50.0,
        cost: 10.0,
    };

    c.bench_function("portfolio_process_fill", |b| {
        b.iter(|| {
            portfolio.process_fill(black_box(&fill));
        })
    });
}

fn bench_update_drawdown(c: &mut Criterion) {
    let mut portfolio = Portfolio::new(1_000_000.0, 10);

    c.bench_function("portfolio_update_drawdown", |b| {
        b.iter(|| {
            portfolio.update_drawdown();
            black_box(&portfolio);
        })
    });
}

fn bench_mark_to_market(c: &mut Criterion) {
    let num_assets = 200;
    let mut portfolio = Portfolio::new(10_000_000.0, num_assets);
    let prices: Vec<f64> = (0..num_assets).map(|i| 50.0 + i as f64).collect();

    c.bench_function("portfolio_mark_to_market_200", |b| {
        b.iter(|| {
            portfolio.mark_to_market(black_box(&prices));
        })
    });
}

criterion_group!(
    benches,
    bench_process_fill,
    bench_update_drawdown,
    bench_mark_to_market
);
criterion_main!(benches);

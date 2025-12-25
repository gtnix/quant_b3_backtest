//! Benchmark for core type operations and SIMD calculations.

use backtester_core::{simd, Bar, MarketEvent};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

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
        b.iter(|| {
            black_box(MarketEvent {
                asset_id: 1.into(),
                bar,
            })
        })
    });
}

// Generate test price data
fn generate_prices(n: usize) -> Vec<f64> {
    let mut prices = Vec::with_capacity(n);
    let mut price = 100.0;
    for i in 0..n {
        // Random walk with sine wave overlay
        let change = (i as f64 * 0.01).sin() * 0.5 + 0.001;
        price *= 1.0 + change;
        prices.push(price);
    }
    prices
}

fn generate_returns(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (i as f64 * 0.01).sin() * 0.02 + 0.0001)
        .collect()
}

fn bench_simd_returns(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_returns");

    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        let prices = generate_prices(*size);

        group.bench_with_input(BenchmarkId::from_parameter(size), &prices, |b, prices| {
            b.iter(|| simd::simd_returns(black_box(prices)))
        });
    }

    group.finish();
}

fn bench_simd_drawdown(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_drawdown");

    for size in [100, 1_000, 10_000, 100_000].iter() {
        let nav = generate_prices(*size);

        group.bench_with_input(BenchmarkId::from_parameter(size), &nav, |b, nav| {
            b.iter(|| simd::simd_drawdown(black_box(nav)))
        });
    }

    group.finish();
}

fn bench_simd_volatility(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_volatility");

    for size in [100, 1_000, 10_000, 100_000].iter() {
        let returns = generate_returns(*size);

        group.bench_with_input(BenchmarkId::from_parameter(size), &returns, |b, returns| {
            b.iter(|| simd::simd_volatility(black_box(returns)))
        });
    }

    group.finish();
}

fn bench_simd_sharpe(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_sharpe");

    for size in [100, 1_000, 10_000, 100_000].iter() {
        let returns = generate_returns(*size);

        group.bench_with_input(BenchmarkId::from_parameter(size), &returns, |b, returns| {
            b.iter(|| simd::simd_sharpe(black_box(returns), 0.02))
        });
    }

    group.finish();
}

fn bench_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vs_scalar");

    let prices = generate_prices(10_000);

    // SIMD returns
    group.bench_function("simd_returns_10k", |b| {
        b.iter(|| simd::simd_returns(black_box(&prices)))
    });

    // Scalar returns (baseline)
    group.bench_function("scalar_returns_10k", |b| {
        b.iter(|| {
            let n = prices.len() - 1;
            let mut returns = vec![0.0; n];
            for i in 0..n {
                if prices[i] > 0.0 {
                    returns[i] = (prices[i + 1] - prices[i]) / prices[i];
                }
            }
            black_box(returns)
        })
    });

    group.finish();
}

fn bench_simd_mean_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_mean_sum");

    for size in [100, 1_000, 10_000, 100_000].iter() {
        let values = generate_returns(*size);

        group.bench_with_input(BenchmarkId::new("simd_mean", size), &values, |b, values| {
            b.iter(|| simd::simd_mean(black_box(values)))
        });

        group.bench_with_input(BenchmarkId::new("simd_sum", size), &values, |b, values| {
            b.iter(|| simd::simd_sum(black_box(values)))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_bar_clone,
    bench_market_event_create,
    bench_simd_returns,
    bench_simd_drawdown,
    bench_simd_volatility,
    bench_simd_sharpe,
    bench_simd_vs_scalar,
    bench_simd_mean_sum,
);
criterion_main!(benches);

//! Benchmarks for Exit Module.
//!
//! Performance budgets:
//! - N=100: < 1ms
//! - N=1000: < 5ms
//! - N=10000: < 50ms

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use backtester_intelligence::exit::{
    ExitContext, ExitEngine, ExitEngineConfig, ExitPolicyConfig, Position,
};
use backtester_intelligence::filters::Market;
use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

fn fixed_date() -> NaiveDate {
    NaiveDate::from_ymd_opt(2025, 1, 10).unwrap()
}

fn generate_positions(n: usize) -> Vec<Position> {
    (0..n)
        .map(|i| {
            let cost = Decimal::from(30 + (i % 100) as i64);
            // Vary returns: some losses, some gains
            let return_pct = ((i as f64 % 50.0) - 25.0) / 100.0;
            let price = cost * Decimal::try_from(1.0 + return_pct).unwrap_or(cost);

            Position::new(
                format!("SYM{:05}", i),
                if i % 10 == 0 { Market::US } else { Market::BR },
                ((i % 10) as i64 + 1) * 100,
                cost,
                NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
                price,
            )
        })
        .collect()
}

fn bench_full_pipeline(c: &mut Criterion) {
    let engine = ExitEngine::new(ExitEngineConfig::default());

    let mut group = c.benchmark_group("exit_full_pipeline");

    for n in [100, 1000, 10000] {
        let positions = generate_positions(n);
        let ctx = ExitContext::new(fixed_date(), dec!(10_000_000), dec!(9_500_000), Market::BR);

        group.bench_with_input(BenchmarkId::new("N", n), &positions, |b, positions| {
            b.iter(|| engine.evaluate(positions, &ctx))
        });
    }

    group.finish();
}

fn bench_with_all_exits(c: &mut Criterion) {
    // Configure to trigger all stop-losses
    let config = ExitEngineConfig {
        policy: ExitPolicyConfig {
            enable_stop_loss: true,
            stop_loss_pct: -0.01, // Very low threshold
            enable_take_profit: false,
            enable_time_exit: false,
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = ExitEngine::new(config);

    let mut group = c.benchmark_group("exit_all_exits");

    for n in [100, 500, 1000] {
        // All positions with 5% loss (will trigger)
        let positions: Vec<Position> = (0..n)
            .map(|i| {
                Position::new(
                    format!("EXIT{:05}", i),
                    Market::BR,
                    500,
                    dec!(100),
                    NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
                    dec!(95), // -5% loss
                )
            })
            .collect();

        let ctx = ExitContext::new(fixed_date(), dec!(10_000_000), dec!(10_000_000), Market::BR);

        group.bench_with_input(BenchmarkId::new("N", n), &positions, |b, positions| {
            b.iter(|| engine.evaluate(positions, &ctx))
        });
    }

    group.finish();
}

fn bench_drawdown_guard(c: &mut Criterion) {
    // Configure to trigger drawdown guard
    let config = ExitEngineConfig {
        policy: ExitPolicyConfig {
            enable_stop_loss: false,
            enable_take_profit: false,
            enable_time_exit: false,
            ..Default::default()
        },
        risk: backtester_intelligence::exit::RiskConfig {
            max_drawdown_pct: -0.10,
            drawdown_action: backtester_intelligence::exit::DrawdownAction::CashOut,
            check_drawdown: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = ExitEngine::new(config);

    let mut group = c.benchmark_group("exit_drawdown_guard");

    for n in [100, 500, 1000] {
        let positions: Vec<Position> = (0..n)
            .map(|i| {
                Position::new(
                    format!("DD{:05}", i),
                    Market::BR,
                    500,
                    dec!(100),
                    NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
                    dec!(102), // small gain
                )
            })
            .collect();

        // 20% drawdown to trigger
        let mut ctx = ExitContext::new(fixed_date(), dec!(10_000_000), dec!(8_000_000), Market::BR);
        ctx.peak_equity = dec!(10_000_000);

        group.bench_with_input(BenchmarkId::new("N", n), &positions, |b, positions| {
            b.iter(|| engine.evaluate(positions, &ctx))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_full_pipeline,
    bench_with_all_exits,
    bench_drawdown_guard,
);

criterion_main!(benches);
















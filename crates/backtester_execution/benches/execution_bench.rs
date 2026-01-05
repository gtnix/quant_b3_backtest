//! Benchmark for execution model operations.

use backtester_core::{Bar, ExecutionModel, OrderEvent};
use backtester_execution::SimpleExecutionModel;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_simple_execute(c: &mut Criterion) {
    let model = SimpleExecutionModel::new(10.0, 0.001);
    let order = OrderEvent {
        timestamp: 1_700_000_000_000_000_000,
        asset_id: 1,
        quantity: 100,
        limit_price: None,
    };
    let bar = Bar {
        timestamp: 1_700_000_000_000_000_000,
        open: 50.0,
        high: 51.0,
        low: 49.0,
        close: 50.0,
        volume: 10000.0,
    };

    c.bench_function("execution_simple_execute", |b| {
        b.iter(|| black_box(model.execute(&order, &bar)))
    });
}

criterion_group!(benches, bench_simple_execute);
criterion_main!(benches);



































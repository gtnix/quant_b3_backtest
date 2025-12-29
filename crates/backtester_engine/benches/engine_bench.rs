//! Benchmark for engine event processing throughput.

use backtester_core::{Bar, MarketEvent, SignalEvent, Strategy};
use backtester_engine::Engine;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

struct NoOpStrategy;

impl Strategy for NoOpStrategy {
    fn on_market(&mut self, _event: &MarketEvent) -> Option<SignalEvent> {
        None
    }
}

fn bench_event_processing(c: &mut Criterion) {
    let mut engine = Engine::new(NoOpStrategy);
    let event = MarketEvent {
        asset_id: 1,
        bar: Bar {
            timestamp: 1_700_000_000_000_000_000,
            open: 100.0,
            high: 105.0,
            low: 99.0,
            close: 104.0,
            volume: 10000.0,
        },
    };

    c.bench_function("engine_process_event", |b| {
        b.iter(|| black_box(engine.process_market_event(&event)))
    });
}

criterion_group!(benches, bench_event_processing);
criterion_main!(benches);














//! Performance benchmarks for read/write operations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use obfs::*;
use uuid::Uuid;

fn create_benchmark_artifact() -> BacktestArtifact {
    let uuid = Uuid::new_v4();
    BacktestArtifact {
        uuid_bytes: *uuid.as_bytes(),
        metadata: Metadata {
            strategy_id: "benchmark_strategy".to_string(),
            strategy_version: "1.0.0".to_string(),
            run_id: "benchmark_run".to_string(),
            timestamp: chrono::Utc::now().timestamp(),
            universe: "B3_IBOV".to_string(),
            start_date: "2020-01-01".to_string(),
            end_date: "2024-12-31".to_string(),
            initial_capital: 1_000_000.0,
            mode: "fast".to_string(),
        },
        metrics: Metrics {
            cagr: 0.15,
            volatility: 0.20,
            sharpe_ratio: 0.75,
            sortino_ratio: 1.10,
            max_drawdown: -0.25,
            max_drawdown_duration_days: 180,
            hit_rate: 0.55,
            profit_factor: 1.5,
            turnover_annual: 2.0,
            total_trades: 500,
        },
        timeseries_ref: TimeseriesReference {
            parquet_file: "timeseries_0000.parquet".to_string(),
            row_group: 0,
            start_row: 0,
            num_rows: 1245,
        },
        trace: vec![
            TraceEvent {
                timestamp: chrono::Utc::now().timestamp(),
                event_type: "start".to_string(),
                message: "Backtest started".to_string(),
            },
            TraceEvent {
                timestamp: chrono::Utc::now().timestamp(),
                event_type: "end".to_string(),
                message: "Backtest completed".to_string(),
            },
        ],
        integrity: IntegritySeal::default(),
    }
}

fn bench_write_artifact(c: &mut Criterion) {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = ObfsConfig {
        root_path: temp_dir.path().to_str().unwrap().to_string(),
        ..Default::default()
    };

    let obfs = Obfs::with_config(config.clone());
    obfs.initialize().unwrap();

    let mut group = c.benchmark_group("write_artifact");

    group.bench_function("write_single", |b| {
        b.iter(|| {
            let artifact = create_benchmark_artifact();
            let mut writer = obfs.writer();
            writer.write_artifact(black_box(&artifact)).unwrap()
        });
    });

    group.finish();
}

fn bench_read_artifact(c: &mut Criterion) {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = ObfsConfig {
        root_path: temp_dir.path().to_str().unwrap().to_string(),
        ..Default::default()
    };

    let obfs = Obfs::with_config(config.clone());
    obfs.initialize().unwrap();

    let artifact = create_benchmark_artifact();
    let uuid = artifact.uuid();
    let mut writer = obfs.writer();
    writer.write_artifact(&artifact).unwrap();

    let mut group = c.benchmark_group("read_artifact");

    group.bench_function("read_single", |b| {
        let reader = obfs.reader();
        b.iter(|| reader.read_artifact(black_box(uuid)).unwrap());
    });

    group.finish();
}

fn bench_batch_write(c: &mut Criterion) {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = ObfsConfig {
        root_path: temp_dir.path().to_str().unwrap().to_string(),
        ..Default::default()
    };

    let obfs = Obfs::with_config(config.clone());
    obfs.initialize().unwrap();

    let mut group = c.benchmark_group("batch_write");

    for batch_size in [10, 100].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, &size| {
                b.iter(|| {
                    let mut writer = obfs.writer();
                    for _ in 0..size {
                        let artifact = create_benchmark_artifact();
                        writer.write_artifact(black_box(&artifact)).unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_read_metrics_only(c: &mut Criterion) {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = ObfsConfig {
        root_path: temp_dir.path().to_str().unwrap().to_string(),
        ..Default::default()
    };

    let obfs = Obfs::with_config(config.clone());
    obfs.initialize().unwrap();

    let artifact = create_benchmark_artifact();
    let uuid = artifact.uuid();
    let mut writer = obfs.writer();
    writer.write_artifact(&artifact).unwrap();

    let mut group = c.benchmark_group("read_metrics_only");

    group.bench_function("metrics_only", |b| {
        let reader = obfs.reader();
        b.iter(|| reader.get_metrics(black_box(uuid)).unwrap());
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_write_artifact,
    bench_read_artifact,
    bench_batch_write,
    bench_read_metrics_only
);
criterion_main!(benches);

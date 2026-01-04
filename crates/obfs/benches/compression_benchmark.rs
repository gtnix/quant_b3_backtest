//! Performance benchmarks for compression

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use obfs::CompressionPipeline;

/// Generate synthetic time-series data for benchmarking
fn generate_timeseries_data(num_points: usize) -> Vec<f32> {
    let mut data = Vec::with_capacity(num_points);
    let mut value = 1_000_000.0;

    for i in 0..num_points {
        let change = ((i as f32 * 0.1).sin() * 100.0) + ((i as f32 * 0.01).cos() * 50.0);
        value += change;
        data.push(value);
    }

    data
}

/// Benchmark compression of time-series data at different compression levels
fn bench_compression_levels(c: &mut Criterion) {
    let data = generate_timeseries_data(1245);
    let data_bytes: Vec<u8> = data.iter().flat_map(|&f| f.to_le_bytes()).collect();

    let mut group = c.benchmark_group("compression_levels");
    group.throughput(Throughput::Bytes(data_bytes.len() as u64));

    for level in [1, 3, 10, 19].iter() {
        let pipeline = CompressionPipeline::with_level(*level);

        group.bench_with_input(BenchmarkId::new("compress", level), level, |b, _| {
            b.iter(|| pipeline.compress(black_box(&data_bytes)).unwrap());
        });
    }

    group.finish();
}

/// Benchmark decompression speed
fn bench_decompression(c: &mut Criterion) {
    let data = generate_timeseries_data(1245);
    let data_bytes: Vec<u8> = data.iter().flat_map(|&f| f.to_le_bytes()).collect();

    let pipeline = CompressionPipeline::new();
    let compressed = pipeline.compress(&data_bytes).unwrap();

    let mut group = c.benchmark_group("decompression");
    group.throughput(Throughput::Bytes(data_bytes.len() as u64));

    group.bench_function("decompress", |b| {
        b.iter(|| pipeline.decompress(black_box(&compressed)).unwrap());
    });

    group.finish();
}

/// Benchmark compression ratio vs speed trade-off
fn bench_compression_ratio(c: &mut Criterion) {
    let data = generate_timeseries_data(1245);
    let data_bytes: Vec<u8> = data.iter().flat_map(|&f| f.to_le_bytes()).collect();

    let group = c.benchmark_group("compression_ratio");

    for level in [1, 3, 10, 19].iter() {
        let pipeline = CompressionPipeline::with_level(*level);
        let compressed = pipeline.compress(&data_bytes).unwrap();
        let ratio = data_bytes.len() as f64 / compressed.len() as f64;

        println!(
            "Level {}: {} bytes -> {} bytes (ratio: {:.2}x)",
            level,
            data_bytes.len(),
            compressed.len(),
            ratio
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_compression_levels,
    bench_decompression,
    bench_compression_ratio
);
criterion_main!(benches);


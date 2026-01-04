//! OBFS Audit Test - Teste de auditoria completo (~5 minutos)
//!
//! Valida todo o fluxo OBFS:
//! - Inicializacao e estrutura de diretorios
//! - Escrita com compressao e integridade
//! - Persistencia real em disco
//! - Leitura com validacao XXH3/BLAKE3
//! - TimeSeriesStore (Parquet)
//! - File rotation automatico
//! - LMDB operations (count, list, delete)
//! - Performance baseline

use obfs::*;
use std::path::Path;
use std::time::Instant;
use uuid::Uuid;

fn create_test_artifact() -> BacktestArtifact {
    let uuid = Uuid::new_v4();
    BacktestArtifact {
        uuid_bytes: *uuid.as_bytes(),
        metadata: Metadata {
            strategy_id: "audit_strategy".to_string(),
            strategy_version: "1.0.0".to_string(),
            run_id: uuid.to_string(),
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
                event_type: "trade".to_string(),
                message: "Executed BUY PETR4".to_string(),
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

fn create_timeseries_points(uuid: Uuid, count: usize) -> Vec<TimeSeriesPoint> {
    (0..count)
        .map(|i| TimeSeriesPoint {
            backtest_uuid: uuid,
            date_offset: i as u16,
            equity: 1_000_000.0 + i as f32 * 1000.0,
            drawdown: -0.01 * i as f32,
            exposure: 0.5 + 0.001 * i as f32,
        })
        .collect()
}

// =============================================================================
// TEST 1: Inicializacao e estrutura de diretorios
// =============================================================================
#[test]
fn test_01_init_and_directory_structure() {
    println!("\n=== TEST 1: Inicializacao e estrutura de diretorios ===");
    let start = Instant::now();

    let temp_dir = tempfile::tempdir().unwrap();
    let root_path = temp_dir.path().to_str().unwrap().to_string();

    let config = ObfsConfig {
        root_path: root_path.clone(),
        compression_level: 3,
        enable_blake3: true,
        enable_xxh3: true,
        ..Default::default()
    };

    let obfs = Obfs::with_config(config);
    obfs.initialize().unwrap();

    // Verificar estrutura de diretorios
    assert!(Path::new(&root_path).join("data").exists(), "data/ deve existir");
    assert!(Path::new(&root_path).join("wal").exists(), "wal/ deve existir");
    assert!(Path::new(&root_path).join("lmdb").exists(), "lmdb/ deve existir");
    assert!(Path::new(&root_path).join("integrity").exists(), "integrity/ deve existir");

    println!("  Diretorios criados: data/, wal/, lmdb/, integrity/");
    println!("  Tempo: {:?}", start.elapsed());
    println!("  [OK] Estrutura de diretorios validada");
}

// =============================================================================
// TEST 2: Write/Read roundtrip single artifact
// =============================================================================
#[test]
fn test_02_write_read_roundtrip_single() {
    println!("\n=== TEST 2: Write/Read roundtrip single artifact ===");
    let start = Instant::now();

    let temp_dir = tempfile::tempdir().unwrap();
    let config = ObfsConfig {
        root_path: temp_dir.path().to_str().unwrap().to_string(),
        ..Default::default()
    };

    let obfs = Obfs::with_config(config);
    obfs.initialize().unwrap();

    // Criar e escrever artefato
    let original = create_test_artifact();
    let uuid = original.uuid();

    let mut writer = obfs.writer();
    let write_start = Instant::now();
    let metadata = writer.write_artifact(&original).unwrap();
    let write_time = write_start.elapsed();

    println!("  UUID: {}", uuid);
    println!("  Compressed size: {} bytes", metadata.artifact_location.size);
    println!("  Write time: {:?}", write_time);

    // Verificar arquivo existe
    assert!(
        Path::new(&metadata.artifact_location.file_path).exists(),
        "Arquivo .obfs deve existir"
    );

    // Ler artefato
    let reader = obfs.reader();
    let read_start = Instant::now();
    let recovered = reader.read_artifact(uuid).unwrap();
    let read_time = read_start.elapsed();

    println!("  Read time: {:?}", read_time);

    // Validar campos
    assert_eq!(recovered.uuid(), original.uuid(), "UUID mismatch");
    assert_eq!(recovered.metadata.strategy_id, original.metadata.strategy_id);
    assert_eq!(recovered.metadata.run_id, original.metadata.run_id);
    assert_eq!(recovered.metrics.cagr, original.metrics.cagr);
    assert_eq!(recovered.metrics.sharpe_ratio, original.metrics.sharpe_ratio);
    assert_eq!(recovered.metrics.max_drawdown, original.metrics.max_drawdown);
    assert_eq!(recovered.trace.len(), original.trace.len());

    println!("  Tempo total: {:?}", start.elapsed());
    println!("  [OK] Roundtrip validado");
}

// =============================================================================
// TEST 3: Batch write 100 artifacts
// =============================================================================
#[test]
fn test_03_write_batch_100_artifacts() {
    println!("\n=== TEST 3: Batch write 100 artifacts ===");
    let start = Instant::now();

    let temp_dir = tempfile::tempdir().unwrap();
    let config = ObfsConfig {
        root_path: temp_dir.path().to_str().unwrap().to_string(),
        ..Default::default()
    };

    let obfs = Obfs::with_config(config);
    obfs.initialize().unwrap();

    let mut uuids = Vec::new();
    let mut writer = obfs.writer();
    let mut total_size = 0u64;

    let write_start = Instant::now();
    for i in 0..100 {
        let artifact = create_test_artifact();
        uuids.push(artifact.uuid());
        let metadata = writer.write_artifact(&artifact).unwrap();
        total_size += metadata.artifact_location.size;

        if (i + 1) % 25 == 0 {
            println!("  Escritos: {} artefatos", i + 1);
        }
    }
    let write_time = write_start.elapsed();

    println!("  Total escritos: 100 artefatos");
    println!("  Total size: {} bytes", total_size);
    println!("  Avg size: {} bytes/artifact", total_size / 100);
    println!("  Write time: {:?}", write_time);
    println!("  Throughput: {:.1} artifacts/s", 100.0 / write_time.as_secs_f64());

    // Verificar count
    let reader = obfs.reader();
    let count = reader.count().unwrap();
    assert_eq!(count, 100, "Count deve ser 100");

    // Verificar todos existem
    for uuid in &uuids {
        assert!(reader.exists(*uuid).unwrap(), "Artifact {} deve existir", uuid);
    }

    // Verificar list
    let listed = reader.list().unwrap();
    assert_eq!(listed.len(), 100, "List deve retornar 100 UUIDs");

    println!("  Tempo total: {:?}", start.elapsed());
    println!("  [OK] Batch write validado");
}

// =============================================================================
// TEST 4: Persistence across instances
// =============================================================================
#[test]
fn test_04_persistence_across_instances() {
    println!("\n=== TEST 4: Persistence across instances ===");
    let start = Instant::now();

    let temp_dir = tempfile::tempdir().unwrap();
    let root_path = temp_dir.path().to_str().unwrap().to_string();

    let artifact;
    let uuid;

    // Primeira instancia: escrever
    {
        let config = ObfsConfig {
            root_path: root_path.clone(),
            ..Default::default()
        };

        let obfs = Obfs::with_config(config);
        obfs.initialize().unwrap();

        artifact = create_test_artifact();
        uuid = artifact.uuid();

        let mut writer = obfs.writer();
        writer.write_artifact(&artifact).unwrap();

        let reader = obfs.reader();
        assert_eq!(reader.count().unwrap(), 1);

        println!("  Escrito UUID: {}", uuid);
        // Drop da instancia
    }

    println!("  Instancia OBFS dropada");

    // Segunda instancia: ler
    {
        let config = ObfsConfig {
            root_path: root_path.clone(),
            ..Default::default()
        };

        let obfs = Obfs::with_config(config);

        let reader = obfs.reader();
        let count = reader.count().unwrap();
        assert_eq!(count, 1, "Count deve ser 1 apos reabrir");

        let recovered = reader.read_artifact(uuid).unwrap();
        assert_eq!(recovered.uuid(), uuid);
        assert_eq!(recovered.metrics.cagr, artifact.metrics.cagr);

        println!("  Lido UUID: {} apos reabrir", uuid);
    }

    println!("  Tempo total: {:?}", start.elapsed());
    println!("  [OK] Persistencia validada");
}

// =============================================================================
// TEST 5: Integrity validation (XXH3)
// =============================================================================
#[test]
fn test_05_integrity_validation() {
    println!("\n=== TEST 5: Integrity validation ===");
    let start = Instant::now();

    let temp_dir = tempfile::tempdir().unwrap();
    let config = ObfsConfig {
        root_path: temp_dir.path().to_str().unwrap().to_string(),
        enable_blake3: true,
        enable_xxh3: true,
        ..Default::default()
    };

    let obfs = Obfs::with_config(config);
    obfs.initialize().unwrap();

    let artifact = create_test_artifact();
    let uuid = artifact.uuid();

    let mut writer = obfs.writer();
    let metadata = writer.write_artifact(&artifact).unwrap();

    // Verificar hash nao e zero
    assert_ne!(metadata.blake3_hash, [0u8; 32], "BLAKE3 hash nao deve ser zero");

    // Ler normalmente (deve funcionar)
    let reader = obfs.reader();
    let recovered = reader.read_artifact(uuid);
    assert!(recovered.is_ok(), "Leitura normal deve funcionar");

    println!("  BLAKE3 hash: {:02x}{:02x}{:02x}...", 
        metadata.blake3_hash[0], metadata.blake3_hash[1], metadata.blake3_hash[2]);
    println!("  Leitura com integridade: OK");

    println!("  Tempo total: {:?}", start.elapsed());
    println!("  [OK] Integridade validada");
}

// =============================================================================
// TEST 6: File rotation
// =============================================================================
#[test]
fn test_06_file_rotation() {
    println!("\n=== TEST 6: File rotation ===");
    let start = Instant::now();

    let temp_dir = tempfile::tempdir().unwrap();
    let config = ObfsConfig {
        root_path: temp_dir.path().to_str().unwrap().to_string(),
        max_file_size: 500, // 500 bytes - forcar rotacao
        ..Default::default()
    };

    let obfs = Obfs::with_config(config);
    obfs.initialize().unwrap();

    let mut writer = obfs.writer();
    let mut file_paths = std::collections::HashSet::new();

    for i in 0..10 {
        let artifact = create_test_artifact();
        let metadata = writer.write_artifact(&artifact).unwrap();
        file_paths.insert(metadata.artifact_location.file_path.clone());

        if i == 0 {
            println!("  Primeiro arquivo: {}", metadata.artifact_location.file_path);
        }
    }

    println!("  Arquivos criados: {}", file_paths.len());

    // Verificar multiplos arquivos
    assert!(
        file_paths.len() > 1,
        "Deve ter multiplos arquivos devido a rotacao"
    );

    // Verificar arquivos existem
    let data_dir = temp_dir.path().join("data");
    let obfs_files: Vec<_> = std::fs::read_dir(&data_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "obfs"))
        .collect();

    println!("  Arquivos .obfs encontrados: {}", obfs_files.len());
    for file in &obfs_files {
        let size = std::fs::metadata(file.path()).unwrap().len();
        println!("    {:?}: {} bytes", file.file_name(), size);
    }

    println!("  Tempo total: {:?}", start.elapsed());
    println!("  [OK] File rotation validado");
}

// =============================================================================
// TEST 7: TimeSeriesStore (Parquet)
// =============================================================================
#[test]
fn test_07_timeseries_parquet() {
    println!("\n=== TEST 7: TimeSeriesStore (Parquet) ===");
    let start = Instant::now();

    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = TimeSeriesStore::new(temp_dir.path()).unwrap();

    let uuid = Uuid::new_v4();
    let points = create_timeseries_points(uuid, 1245); // 5 anos de dados diarios

    // Escrever
    let write_start = Instant::now();
    let ts_ref = store.write_timeseries(uuid, &points).unwrap();
    let write_time = write_start.elapsed();

    println!("  UUID: {}", uuid);
    println!("  Rows escritos: {}", ts_ref.num_rows);
    println!("  Write time: {:?}", write_time);

    // Verificar arquivo Parquet
    let parquet_path = Path::new(&ts_ref.parquet_file);
    assert!(parquet_path.exists(), "Arquivo Parquet deve existir");

    let stats = store.get_stats(parquet_path).unwrap();
    println!("  File size: {} bytes", stats.file_size_bytes);
    println!("  Row count: {}", stats.row_count);
    println!("  Compression ratio: {:.2}x", stats.compression_ratio);

    // Ler
    let read_start = Instant::now();
    let read_points = store.read_timeseries(uuid).unwrap();
    let read_time = read_start.elapsed();

    println!("  Read time: {:?}", read_time);
    println!("  Rows lidos: {}", read_points.len());

    assert_eq!(read_points.len(), 1245, "Deve ler 1245 pontos");
    assert_eq!(read_points[0].backtest_uuid, uuid);
    assert!((read_points[0].equity - 1_000_000.0).abs() < 1.0);

    println!("  Tempo total: {:?}", start.elapsed());
    println!("  [OK] TimeSeriesStore validado");
}

// =============================================================================
// TEST 8: Metrics-only read (LMDB, sem decompress)
// =============================================================================
#[test]
fn test_08_metrics_only_read() {
    println!("\n=== TEST 8: Metrics-only read ===");
    let start = Instant::now();

    let temp_dir = tempfile::tempdir().unwrap();
    let config = ObfsConfig {
        root_path: temp_dir.path().to_str().unwrap().to_string(),
        ..Default::default()
    };

    let obfs = Obfs::with_config(config);
    obfs.initialize().unwrap();

    let artifact = create_test_artifact();
    let uuid = artifact.uuid();
    let expected_cagr = artifact.metrics.cagr;
    let expected_sharpe = artifact.metrics.sharpe_ratio;

    let mut writer = obfs.writer();
    writer.write_artifact(&artifact).unwrap();

    // Ler apenas metrics (via LMDB, sem decompress)
    let reader = obfs.reader();
    let read_start = Instant::now();
    let metrics = reader.get_metrics(uuid).unwrap();
    let read_time = read_start.elapsed();

    println!("  Metrics read time: {:?}", read_time);
    println!("  CAGR: {:.2}%", metrics.cagr * 100.0);
    println!("  Sharpe: {:.2}", metrics.sharpe_ratio);

    assert_eq!(metrics.cagr, expected_cagr);
    assert_eq!(metrics.sharpe_ratio, expected_sharpe);

    // Comparar com full read
    let full_read_start = Instant::now();
    let full = reader.read_artifact(uuid).unwrap();
    let full_read_time = full_read_start.elapsed();

    println!("  Full read time: {:?}", full_read_time);
    println!("  Speedup: {:.1}x", full_read_time.as_nanos() as f64 / read_time.as_nanos().max(1) as f64);

    assert_eq!(full.metrics.cagr, expected_cagr);

    println!("  Tempo total: {:?}", start.elapsed());
    println!("  [OK] Metrics-only read validado");
}

// =============================================================================
// TEST 9: Delete artifact
// =============================================================================
#[test]
fn test_09_delete_artifact() {
    println!("\n=== TEST 9: Delete artifact ===");
    let start = Instant::now();

    let temp_dir = tempfile::tempdir().unwrap();
    let lmdb_path = temp_dir.path().join("lmdb");
    std::fs::create_dir_all(&lmdb_path).unwrap();

    let store = MetadataStore::open(&lmdb_path).unwrap();

    // Criar metadata de teste
    let uuid = Uuid::new_v4();
    let metadata = ArtifactMetadata {
        uuid,
        artifact_location: ArtifactLocation {
            file_path: "data/data_0000.obfs".to_string(),
            offset: 0,
            size: 1024,
        },
        blake3_hash: [1u8; 32],
        xxh3_checksum: 0x123456789ABCDEF0,
        metrics: Metrics::default(),
        created_at: chrono::Utc::now().timestamp(),
    };

    // Escrever
    store.put(&metadata).unwrap();
    assert!(store.exists(uuid).unwrap(), "Deve existir apos put");
    assert_eq!(store.count().unwrap(), 1);

    println!("  UUID criado: {}", uuid);

    // Deletar
    let deleted = store.delete(uuid).unwrap();
    assert!(deleted, "Delete deve retornar true");
    assert!(!store.exists(uuid).unwrap(), "Nao deve existir apos delete");
    assert_eq!(store.count().unwrap(), 0);

    println!("  UUID deletado: {}", uuid);

    println!("  Tempo total: {:?}", start.elapsed());
    println!("  [OK] Delete validado");
}

// =============================================================================
// TEST 10: Stress test 1000 artifacts
// =============================================================================
#[test]
fn test_10_stress_1000_artifacts() {
    println!("\n=== TEST 10: Stress test 1000 artifacts ===");
    let start = Instant::now();

    let temp_dir = tempfile::tempdir().unwrap();
    let config = ObfsConfig {
        root_path: temp_dir.path().to_str().unwrap().to_string(),
        compression_level: 1, // Fast compression para stress test
        ..Default::default()
    };

    let obfs = Obfs::with_config(config);
    obfs.initialize().unwrap();

    let mut uuids = Vec::with_capacity(1000);
    let mut writer = obfs.writer();
    let mut total_write_size = 0u64;

    // WRITE PHASE
    println!("  [WRITE PHASE]");
    let write_start = Instant::now();
    for i in 0..1000 {
        let artifact = create_test_artifact();
        uuids.push(artifact.uuid());
        let metadata = writer.write_artifact(&artifact).unwrap();
        total_write_size += metadata.artifact_location.size;

        if (i + 1) % 250 == 0 {
            let elapsed = write_start.elapsed();
            let rate = (i + 1) as f64 / elapsed.as_secs_f64();
            println!("    Escritos: {} | Rate: {:.1}/s", i + 1, rate);
        }
    }
    let write_time = write_start.elapsed();
    let write_rate = 1000.0 / write_time.as_secs_f64();

    println!("  Write total: {:?}", write_time);
    println!("  Write throughput: {:.1} artifacts/s", write_rate);
    println!("  Total size: {} KB", total_write_size / 1024);

    // Sync
    obfs.sync().unwrap();

    // READ PHASE
    println!("  [READ PHASE]");
    let reader = obfs.reader();
    let read_start = Instant::now();
    let mut read_count = 0;
    for (i, uuid) in uuids.iter().enumerate() {
        let artifact = reader.read_artifact(*uuid).unwrap();
        assert_eq!(artifact.uuid(), *uuid);
        read_count += 1;

        if (i + 1) % 250 == 0 {
            let elapsed = read_start.elapsed();
            let rate = (i + 1) as f64 / elapsed.as_secs_f64();
            println!("    Lidos: {} | Rate: {:.1}/s", i + 1, rate);
        }
    }
    let read_time = read_start.elapsed();
    let read_rate = 1000.0 / read_time.as_secs_f64();

    println!("  Read total: {:?}", read_time);
    println!("  Read throughput: {:.1} artifacts/s", read_rate);

    // Validacoes
    assert_eq!(read_count, 1000);
    assert_eq!(reader.count().unwrap(), 1000);
    assert!(write_rate > 50.0, "Write throughput deve ser > 50/s");
    assert!(read_rate > 50.0, "Read throughput deve ser > 50/s");

    println!("");
    println!("  === RESULTADO FINAL ===");
    println!("  Artefatos: 1000");
    println!("  Write: {:.1} artifacts/s", write_rate);
    println!("  Read: {:.1} artifacts/s", read_rate);
    println!("  Tempo total: {:?}", start.elapsed());
    println!("  [OK] Stress test validado");
}


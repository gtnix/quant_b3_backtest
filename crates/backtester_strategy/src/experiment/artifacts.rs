//! Artifact writer - generates standardized output files for experiment runs.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use super::types::{EquityPoint, ExperimentTraceEntry, RunMetadata, RunMetrics};

/// Writer for experiment artifacts.
pub struct ArtifactWriter {
    base_path: PathBuf,
}

impl ArtifactWriter {
    /// Create a new artifact writer with the given base output path.
    pub fn new(base_path: impl AsRef<Path>) -> Self {
        Self {
            base_path: base_path.as_ref().to_path_buf(),
        }
    }

    /// Default output path: output/experiments/
    pub fn default_output() -> Self {
        Self::new("output/experiments")
    }

    /// Write all artifacts for a run.
    pub fn write_all(
        &self,
        run_id: &str,
        metadata: &RunMetadata,
        trace: &[ExperimentTraceEntry],
        metrics: &RunMetrics,
        timeseries: &[EquityPoint],
    ) -> Result<PathBuf, ArtifactError> {
        let run_dir = self.base_path.join(run_id);
        fs::create_dir_all(&run_dir)?;

        self.write_metadata(&run_dir, metadata)?;
        self.write_trace(&run_dir, trace)?;
        self.write_metrics(&run_dir, metrics)?;
        self.write_timeseries(&run_dir, timeseries)?;

        tracing::info!("Artifacts written to: {}", run_dir.display());
        Ok(run_dir)
    }

    /// Write metadata.json
    fn write_metadata(&self, run_dir: &Path, metadata: &RunMetadata) -> Result<(), ArtifactError> {
        let path = run_dir.join("metadata.json");
        let file = File::create(&path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, metadata)?;
        Ok(())
    }

    /// Write trace.jsonl (JSON Lines format)
    fn write_trace(
        &self,
        run_dir: &Path,
        trace: &[ExperimentTraceEntry],
    ) -> Result<(), ArtifactError> {
        let path = run_dir.join("trace.jsonl");
        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);

        for entry in trace {
            let line = serde_json::to_string(entry)?;
            writeln!(writer, "{}", line)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Write metrics.json
    fn write_metrics(&self, run_dir: &Path, metrics: &RunMetrics) -> Result<(), ArtifactError> {
        let path = run_dir.join("metrics.json");
        let file = File::create(&path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, metrics)?;
        Ok(())
    }

    /// Write timeseries.csv
    fn write_timeseries(
        &self,
        run_dir: &Path,
        timeseries: &[EquityPoint],
    ) -> Result<(), ArtifactError> {
        let path = run_dir.join("timeseries.csv");
        let file = File::create(&path)?;
        let mut writer = csv::Writer::from_writer(file);

        // Write header
        writer.write_record([
            "date",
            "equity",
            "drawdown",
            "exposure",
            "vol_exante",
            "vol_expost",
        ])?;

        // Write data rows
        for point in timeseries {
            writer.write_record([
                point.date.to_string(),
                point.equity.to_string(),
                format!("{:.6}", point.drawdown),
                format!("{:.6}", point.exposure),
                point
                    .vol_exante
                    .map(|v| format!("{:.6}", v))
                    .unwrap_or_default(),
                point
                    .vol_expost
                    .map(|v| format!("{:.6}", v))
                    .unwrap_or_default(),
            ])?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Read metadata from a run directory.
    pub fn read_metadata(run_dir: &Path) -> Result<RunMetadata, ArtifactError> {
        let path = run_dir.join("metadata.json");
        let content = fs::read_to_string(&path)?;
        let metadata: RunMetadata = serde_json::from_str(&content)?;
        Ok(metadata)
    }

    /// Read metrics from a run directory.
    pub fn read_metrics(run_dir: &Path) -> Result<RunMetrics, ArtifactError> {
        let path = run_dir.join("metrics.json");
        let content = fs::read_to_string(&path)?;
        let metrics: RunMetrics = serde_json::from_str(&content)?;
        Ok(metrics)
    }

    /// Read timeseries from a run directory.
    pub fn read_timeseries(run_dir: &Path) -> Result<Vec<EquityPoint>, ArtifactError> {
        let path = run_dir.join("timeseries.csv");
        let mut reader = csv::Reader::from_path(&path)?;
        let mut points = Vec::new();

        for result in reader.records() {
            let record = result?;
            let point = EquityPoint {
                date: record
                    .get(0)
                    .ok_or(ArtifactError::ParseError("missing date".into()))?
                    .parse()
                    .map_err(|e| ArtifactError::ParseError(format!("date parse error: {}", e)))?,
                equity: record
                    .get(1)
                    .ok_or(ArtifactError::ParseError("missing equity".into()))?
                    .parse()
                    .map_err(|e| ArtifactError::ParseError(format!("equity parse error: {}", e)))?,
                drawdown: record
                    .get(2)
                    .ok_or(ArtifactError::ParseError("missing drawdown".into()))?
                    .parse()
                    .map_err(|e| {
                        ArtifactError::ParseError(format!("drawdown parse error: {}", e))
                    })?,
                exposure: record
                    .get(3)
                    .ok_or(ArtifactError::ParseError("missing exposure".into()))?
                    .parse()
                    .map_err(|e| {
                        ArtifactError::ParseError(format!("exposure parse error: {}", e))
                    })?,
                vol_exante: record
                    .get(4)
                    .and_then(|s| if s.is_empty() { None } else { s.parse().ok() }),
                vol_expost: record
                    .get(5)
                    .and_then(|s| if s.is_empty() { None } else { s.parse().ok() }),
            };
            points.push(point);
        }

        Ok(points)
    }

    /// Read trace from a run directory.
    pub fn read_trace(run_dir: &Path) -> Result<Vec<ExperimentTraceEntry>, ArtifactError> {
        let path = run_dir.join("trace.jsonl");
        let content = fs::read_to_string(&path)?;
        let mut entries = Vec::new();
        
        for line in content.lines() {
            if !line.trim().is_empty() {
                let entry: ExperimentTraceEntry = serde_json::from_str(line)?;
                entries.push(entry);
            }
        }
        
        Ok(entries)
    }

    /// List all run IDs in the output directory.
    pub fn list_runs(&self) -> Result<Vec<String>, ArtifactError> {
        if !self.base_path.exists() {
            return Ok(Vec::new());
        }

        let mut runs = Vec::new();
        for entry in fs::read_dir(&self.base_path)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    // Check if it has metadata.json (valid run directory)
                    if entry.path().join("metadata.json").exists() {
                        runs.push(name.to_string());
                    }
                }
            }
        }

        runs.sort();
        Ok(runs)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ArtifactError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),
    #[error("Parse error: {0}")]
    ParseError(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{NaiveDate, Utc};
    use rust_decimal_macros::dec;
    use std::collections::HashMap;
    use tempfile::tempdir;

    use super::super::types::CostConfig;

    fn sample_metadata() -> RunMetadata {
        RunMetadata {
            schema_version: super::super::types::ARTIFACT_SCHEMA_VERSION.to_string(),
            run_id: "test-run-123".into(),
            config_hash: "abc123".into(),
            strategy_id: "test_strategy".into(),
            strategy_version: "1.0.0".into(),
            crate_version: "0.1.0".into(),
            timestamp_utc: Utc::now(),
            dataset_id: Some("dataset_1".into()),
            seed: Some(42),
            costs: CostConfig::default(),
            mode: super::super::types::RunMode::Full,
            config_path: "configs/test.toml".into(),
            duration_ms: 1234,
        }
    }

    fn sample_metrics() -> RunMetrics {
        RunMetrics {
            cagr: 0.15,
            volatility: 0.20,
            sharpe_ratio: 0.75,
            max_drawdown: -0.10,
            max_drawdown_duration_days: 30,
            turnover_annual: 2.5,
            hit_rate: 0.55,
            profit_factor: 1.5,
            total_trades: 100,
            total_days: 252,
            sortino_ratio: 1.0,
            calmar_ratio: 1.5,
            avg_win: 100.0,
            avg_loss: 66.0,
            win_loss_ratio: 1.5,
        }
    }

    fn sample_timeseries() -> Vec<EquityPoint> {
        let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        (0..5)
            .map(|i| EquityPoint {
                date: start + chrono::Duration::days(i),
                equity: dec!(100000) + rust_decimal::Decimal::from(i * 1000),
                drawdown: 0.0,
                exposure: 0.95,
                vol_exante: Some(0.15),
                vol_expost: Some(0.14),
            })
            .collect()
    }

    fn sample_trace() -> Vec<ExperimentTraceEntry> {
        vec![ExperimentTraceEntry {
            step: 0,
            block_id: "momentum".into(),
            block_type: "selection".into(),
            message: "Selected 5 assets".into(),
            timestamp_ms: 1234567890,
            params_effective: HashMap::from([
                ("lookback_days".into(), serde_json::json!(126)),
                ("top_pct".into(), serde_json::json!(20)),
            ]),
        }]
    }

    #[test]
    fn test_write_and_read_artifacts() {
        let temp = tempdir().unwrap();
        let writer = ArtifactWriter::new(temp.path());

        let metadata = sample_metadata();
        let metrics = sample_metrics();
        let timeseries = sample_timeseries();
        let trace = sample_trace();

        // Write all
        let run_dir = writer
            .write_all("test-run-123", &metadata, &trace, &metrics, &timeseries)
            .unwrap();

        assert!(run_dir.join("metadata.json").exists());
        assert!(run_dir.join("metrics.json").exists());
        assert!(run_dir.join("timeseries.csv").exists());
        assert!(run_dir.join("trace.jsonl").exists());

        // Read back
        let read_metadata = ArtifactWriter::read_metadata(&run_dir).unwrap();
        assert_eq!(read_metadata.run_id, "test-run-123");
        assert_eq!(read_metadata.strategy_id, "test_strategy");

        let read_metrics = ArtifactWriter::read_metrics(&run_dir).unwrap();
        assert!((read_metrics.cagr - 0.15).abs() < 0.001);

        let read_ts = ArtifactWriter::read_timeseries(&run_dir).unwrap();
        assert_eq!(read_ts.len(), 5);
    }

    #[test]
    fn test_list_runs() {
        let temp = tempdir().unwrap();
        let writer = ArtifactWriter::new(temp.path());

        // Create two runs
        let metadata = sample_metadata();
        let metrics = sample_metrics();
        let timeseries = sample_timeseries();
        let trace = sample_trace();

        writer
            .write_all("run-001", &metadata, &trace, &metrics, &timeseries)
            .unwrap();
        writer
            .write_all("run-002", &metadata, &trace, &metrics, &timeseries)
            .unwrap();

        let runs = writer.list_runs().unwrap();
        assert_eq!(runs.len(), 2);
        assert!(runs.contains(&"run-001".to_string()));
        assert!(runs.contains(&"run-002".to_string()));
    }

    // ========================================================================
    // ARTIFACT SCHEMA VALIDATION (A5)
    // ========================================================================

    #[test]
    fn test_artifact_schema_version() {
        let temp = tempdir().unwrap();
        let writer = ArtifactWriter::new(temp.path());

        let metadata = sample_metadata();
        let metrics = sample_metrics();
        let timeseries = sample_timeseries();
        let trace = sample_trace();

        let run_dir = writer
            .write_all("schema-test", &metadata, &trace, &metrics, &timeseries)
            .unwrap();

        // Read back and verify schema version is present
        let read_metadata = ArtifactWriter::read_metadata(&run_dir).unwrap();
        assert_eq!(
            read_metadata.schema_version,
            super::super::types::ARTIFACT_SCHEMA_VERSION
        );
    }

    #[test]
    fn test_artifact_roundtrip_full() {
        // Comprehensive roundtrip test: write all artifacts, read them back,
        // verify all fields match the original values
        let temp = tempdir().unwrap();
        let writer = ArtifactWriter::new(temp.path());

        let original_metadata = sample_metadata();
        let original_metrics = sample_metrics();
        let original_timeseries = sample_timeseries();
        let original_trace = sample_trace();

        let run_dir = writer
            .write_all(
                "roundtrip-test",
                &original_metadata,
                &original_trace,
                &original_metrics,
                &original_timeseries,
            )
            .unwrap();

        // ====================================================================
        // Validate metadata.json
        // ====================================================================
        let read_metadata = ArtifactWriter::read_metadata(&run_dir).unwrap();
        assert_eq!(read_metadata.run_id, original_metadata.run_id);
        assert_eq!(read_metadata.config_hash, original_metadata.config_hash);
        assert_eq!(read_metadata.strategy_id, original_metadata.strategy_id);
        assert_eq!(
            read_metadata.strategy_version,
            original_metadata.strategy_version
        );
        assert_eq!(read_metadata.config_path, original_metadata.config_path);
        assert_eq!(read_metadata.seed, original_metadata.seed);
        assert_eq!(read_metadata.mode, original_metadata.mode);
        assert_eq!(read_metadata.duration_ms, original_metadata.duration_ms);

        // ====================================================================
        // Validate metrics.json
        // ====================================================================
        let read_metrics = ArtifactWriter::read_metrics(&run_dir).unwrap();
        assert!((read_metrics.cagr - original_metrics.cagr).abs() < 0.0001);
        assert!((read_metrics.volatility - original_metrics.volatility).abs() < 0.0001);
        assert!((read_metrics.sharpe_ratio - original_metrics.sharpe_ratio).abs() < 0.0001);
        assert!((read_metrics.max_drawdown - original_metrics.max_drawdown).abs() < 0.0001);
        assert_eq!(read_metrics.total_trades, original_metrics.total_trades);
        assert_eq!(read_metrics.total_days, original_metrics.total_days);

        // ====================================================================
        // Validate timeseries.csv
        // ====================================================================
        let read_ts = ArtifactWriter::read_timeseries(&run_dir).unwrap();
        assert_eq!(read_ts.len(), original_timeseries.len());
        for (orig, read) in original_timeseries.iter().zip(read_ts.iter()) {
            assert_eq!(read.date, orig.date);
            assert_eq!(read.equity, orig.equity);
            assert!((read.drawdown - orig.drawdown).abs() < 0.000001);
            assert!((read.exposure - orig.exposure).abs() < 0.000001);
        }

        // ====================================================================
        // Validate trace.jsonl
        // ====================================================================
        let read_trace = ArtifactWriter::read_trace(&run_dir).unwrap();
        assert_eq!(read_trace.len(), original_trace.len());
        for (orig, read) in original_trace.iter().zip(read_trace.iter()) {
            assert_eq!(read.step, orig.step);
            assert_eq!(read.block_id, orig.block_id);
            assert_eq!(read.block_type, orig.block_type);
            assert_eq!(read.message, orig.message);
        }
    }

    #[test]
    fn test_metadata_json_valid_structure() {
        // Verify metadata.json is valid JSON with expected structure
        let temp = tempdir().unwrap();
        let writer = ArtifactWriter::new(temp.path());

        let metadata = sample_metadata();
        let metrics = sample_metrics();
        let timeseries = sample_timeseries();
        let trace = sample_trace();

        let run_dir = writer
            .write_all("json-test", &metadata, &trace, &metrics, &timeseries)
            .unwrap();

        // Read raw JSON and validate structure
        let content = std::fs::read_to_string(run_dir.join("metadata.json")).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

        // Verify expected fields exist
        assert!(parsed.get("schema_version").is_some());
        assert!(parsed.get("run_id").is_some());
        assert!(parsed.get("config_hash").is_some());
        assert!(parsed.get("strategy_id").is_some());
        assert!(parsed.get("timestamp_utc").is_some());
        assert!(parsed.get("mode").is_some());
    }

    #[test]
    fn test_trace_jsonl_valid_lines() {
        // Verify each line in trace.jsonl is valid JSON
        let temp = tempdir().unwrap();
        let writer = ArtifactWriter::new(temp.path());

        let metadata = sample_metadata();
        let metrics = sample_metrics();
        let timeseries = sample_timeseries();
        let trace = sample_trace();

        let run_dir = writer
            .write_all("jsonl-test", &metadata, &trace, &metrics, &timeseries)
            .unwrap();

        let content = std::fs::read_to_string(run_dir.join("trace.jsonl")).unwrap();
        for (i, line) in content.lines().enumerate() {
            let parsed: Result<serde_json::Value, _> = serde_json::from_str(line);
            assert!(
                parsed.is_ok(),
                "Line {} in trace.jsonl is not valid JSON: {}",
                i,
                line
            );
        }
    }
}


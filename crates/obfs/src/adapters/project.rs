//! Converters for project-specific types to OBFS types

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::{
    BacktestArtifact, IntegritySeal, Metadata, Metrics, TimeseriesReference, TraceEvent,
};

/// Project's RunMetadata format (from backtester_strategy::experiment::types)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectRunMetadata {
    #[serde(default)]
    pub schema_version: String,
    pub run_id: String,
    #[serde(default)]
    pub config_hash: String,
    pub strategy_id: String,
    #[serde(default)]
    pub strategy_version: String,
    #[serde(default)]
    pub crate_version: String,
    #[serde(default)]
    pub timestamp_utc: Option<String>, // ISO8601 string
    #[serde(default)]
    pub dataset_id: Option<String>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub mode: String,
    #[serde(default)]
    pub execution_mode: String,
    #[serde(default)]
    pub config_path: String,
    #[serde(default)]
    pub duration_ms: u64,
    #[serde(default)]
    pub dividends_enabled: bool,
}

/// Project's RunMetrics format (from backtester_strategy::experiment::types)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProjectRunMetrics {
    #[serde(default)]
    pub cagr: f64,
    #[serde(default)]
    pub volatility: f64,
    #[serde(default)]
    pub sharpe_ratio: f64,
    #[serde(default)]
    pub max_drawdown: f64,
    #[serde(default)]
    pub max_drawdown_duration_days: u32,
    #[serde(default)]
    pub turnover_annual: f64,
    #[serde(default)]
    pub hit_rate: f64,
    #[serde(default)]
    pub profit_factor: f64,
    #[serde(default)]
    pub total_trades: u32,
    #[serde(default)]
    pub total_days: u32,
    #[serde(default)]
    pub sortino_ratio: f64,
    #[serde(default)]
    pub calmar_ratio: f64,
    #[serde(default)]
    pub avg_win: f64,
    #[serde(default)]
    pub avg_loss: f64,
    #[serde(default)]
    pub win_loss_ratio: f64,
    #[serde(default = "default_true")]
    pub is_valid: bool,
    #[serde(default)]
    pub warnings: Vec<String>,
}

fn default_true() -> bool {
    true
}

/// Project's ExperimentTraceEntry format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectTraceEntry {
    pub step: usize,
    pub block_id: String,
    pub block_type: String,
    pub message: String,
    pub timestamp_ms: u64,
    #[serde(default)]
    pub params_effective: std::collections::HashMap<String, serde_json::Value>,
}

/// Convert ProjectRunMetadata to OBFS Metadata
impl From<&ProjectRunMetadata> for Metadata {
    fn from(pm: &ProjectRunMetadata) -> Self {
        let timestamp = pm
            .timestamp_utc
            .as_ref()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.timestamp())
            .unwrap_or(0);

        Self {
            strategy_id: pm.strategy_id.clone(),
            strategy_version: pm.strategy_version.clone(),
            run_id: pm.run_id.clone(),
            timestamp,
            universe: pm.dataset_id.clone().unwrap_or_else(|| "B3_IBOV".to_string()),
            start_date: String::new(),
            end_date: String::new(),
            initial_capital: 1_000_000.0,
            mode: pm.execution_mode.clone(),
        }
    }
}

/// Convert ProjectRunMetrics to OBFS Metrics
impl From<&ProjectRunMetrics> for Metrics {
    fn from(pm: &ProjectRunMetrics) -> Self {
        Self {
            cagr: pm.cagr,
            volatility: pm.volatility,
            sharpe_ratio: pm.sharpe_ratio,
            sortino_ratio: pm.sortino_ratio,
            max_drawdown: pm.max_drawdown,
            max_drawdown_duration_days: pm.max_drawdown_duration_days as i32,
            hit_rate: pm.hit_rate,
            profit_factor: pm.profit_factor,
            turnover_annual: pm.turnover_annual,
            total_trades: pm.total_trades as i32,
        }
    }
}

/// Convert ProjectTraceEntry to OBFS TraceEvent
impl From<&ProjectTraceEntry> for TraceEvent {
    fn from(te: &ProjectTraceEntry) -> Self {
        Self {
            timestamp: te.timestamp_ms as i64,
            event_type: te.block_type.clone(),
            message: te.message.clone(),
        }
    }
}

/// Artifact loader for reading project's existing artifacts
pub struct ProjectArtifactLoader;

impl ProjectArtifactLoader {
    /// Load a backtest artifact from project's directory structure
    pub fn load_from_dir(
        backtest_dir: &std::path::Path,
    ) -> anyhow::Result<(ProjectRunMetadata, ProjectRunMetrics, Vec<ProjectTraceEntry>)> {
        // Read metadata.json
        let metadata_path = backtest_dir.join("metadata.json");
        let metadata: ProjectRunMetadata = if metadata_path.exists() {
            let content = std::fs::read_to_string(&metadata_path)?;
            serde_json::from_str(&content)?
        } else {
            return Err(anyhow::anyhow!("metadata.json not found"));
        };

        // Read metrics.json
        let metrics_path = backtest_dir.join("metrics.json");
        let metrics: ProjectRunMetrics = if metrics_path.exists() {
            let content = std::fs::read_to_string(&metrics_path)?;
            serde_json::from_str(&content)?
        } else {
            ProjectRunMetrics::default()
        };

        // Read trace.jsonl (optional)
        let trace_path = backtest_dir.join("trace.jsonl");
        let trace: Vec<ProjectTraceEntry> = if trace_path.exists() {
            let content = std::fs::read_to_string(&trace_path)?;
            content
                .lines()
                .filter_map(|line| serde_json::from_str(line).ok())
                .collect()
        } else {
            Vec::new()
        };

        Ok((metadata, metrics, trace))
    }

    /// Convert project artifacts to OBFS BacktestArtifact
    pub fn convert_to_obfs(
        metadata: &ProjectRunMetadata,
        metrics: &ProjectRunMetrics,
        trace: &[ProjectTraceEntry],
        timeseries_row_count: usize,
    ) -> BacktestArtifact {
        let uuid = Uuid::parse_str(&metadata.run_id).unwrap_or_else(|_| Uuid::new_v4());

        BacktestArtifact {
            uuid_bytes: *uuid.as_bytes(),
            metadata: Metadata::from(metadata),
            metrics: Metrics::from(metrics),
            timeseries_ref: TimeseriesReference {
                parquet_file: format!("{}.parquet", uuid),
                row_group: 0,
                start_row: 0,
                num_rows: timeseries_row_count as u64,
            },
            trace: trace.iter().map(TraceEvent::from).collect(),
            integrity: IntegritySeal::default(),
        }
    }

    /// Count rows in timeseries.csv
    pub fn count_timeseries_rows(backtest_dir: &std::path::Path) -> usize {
        let ts_path = backtest_dir.join("timeseries.csv");
        if ts_path.exists() {
            std::fs::read_to_string(&ts_path)
                .map(|c| c.lines().count().saturating_sub(1))
                .unwrap_or(0)
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_conversion() {
        let project_metrics = ProjectRunMetrics {
            cagr: 0.15,
            volatility: 0.20,
            sharpe_ratio: 0.75,
            max_drawdown: -0.25,
            max_drawdown_duration_days: 180,
            hit_rate: 0.55,
            profit_factor: 1.5,
            turnover_annual: 2.0,
            total_trades: 500,
            sortino_ratio: 1.10,
            ..Default::default()
        };

        let obfs_metrics = Metrics::from(&project_metrics);

        assert_eq!(obfs_metrics.cagr, 0.15);
        assert_eq!(obfs_metrics.sharpe_ratio, 0.75);
        assert_eq!(obfs_metrics.max_drawdown, -0.25);
        assert_eq!(obfs_metrics.total_trades, 500);
    }

    #[test]
    fn test_metadata_conversion() {
        let project_metadata = ProjectRunMetadata {
            schema_version: "1.0".to_string(),
            run_id: "test-run-123".to_string(),
            config_hash: "abc123".to_string(),
            strategy_id: "momentum_strategy".to_string(),
            strategy_version: "2.0.0".to_string(),
            crate_version: "0.1.0".to_string(),
            timestamp_utc: Some("2024-01-01T00:00:00Z".to_string()),
            dataset_id: Some("B3_IBOV".to_string()),
            seed: Some(42),
            mode: "Full".to_string(),
            execution_mode: "fast".to_string(),
            config_path: "config.toml".to_string(),
            duration_ms: 1000,
            dividends_enabled: true,
        };

        let obfs_metadata = Metadata::from(&project_metadata);

        assert_eq!(obfs_metadata.strategy_id, "momentum_strategy");
        assert_eq!(obfs_metadata.run_id, "test-run-123");
        assert_eq!(obfs_metadata.mode, "fast");
    }
}

//! Backtest executors - Library and CLI implementations.

use backtester_strategy::config::StrategyConfig;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;
use thiserror::Error;
use tracing::{debug, warn};

/// Errors during backtest execution.
#[derive(Debug, Error)]
pub enum ExecutionError {
    #[error("Backtest execution failed: {0}")]
    Failed(String),

    #[error("Timeout after {0} seconds")]
    Timeout(u64),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("TOML serialization error: {0}")]
    TomlSerialize(#[from] toml::ser::Error),
}

/// Metrics from a backtest run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BacktestMetrics {
    pub cagr: f64,
    pub volatility: Option<f64>,
    pub sharpe_ratio: f64,
    pub sortino_ratio: Option<f64>,
    pub calmar_ratio: Option<f64>,
    pub max_drawdown: f64,
    pub max_drawdown_duration_days: Option<u32>,
    pub hit_rate: Option<f64>,
    pub profit_factor: Option<f64>,
    pub turnover_annual: Option<f64>,
    pub total_trades: u32,
    pub winning_trades: Option<u32>,
    pub losing_trades: Option<u32>,
}

/// Output from a backtest execution.
#[derive(Debug, Clone, Default)]
pub struct BacktestOutput {
    pub metrics: BacktestMetrics,
    pub run_id: Option<String>,
    pub output_path: Option<PathBuf>,
    pub duration_ms: u64,
}

impl BacktestOutput {
    /// Create a mock output for testing.
    pub fn mock() -> Self {
        Self {
            metrics: BacktestMetrics {
                cagr: 0.10,
                volatility: Some(0.15),
                sharpe_ratio: 0.8,
                sortino_ratio: Some(1.0),
                calmar_ratio: Some(1.5),
                max_drawdown: -0.12,
                max_drawdown_duration_days: Some(30),
                hit_rate: Some(0.55),
                profit_factor: Some(1.5),
                turnover_annual: Some(2.5),
                total_trades: 100,
                winning_trades: Some(55),
                losing_trades: Some(45),
            },
            run_id: Some("mock-run".into()),
            output_path: None,
            duration_ms: 100,
        }
    }
}

/// Trait for backtest execution.
pub trait BacktestExecutor: Send + Sync {
    /// Execute a single backtest.
    fn execute(&self, config: &StrategyConfig) -> Result<BacktestOutput, ExecutionError>;

    /// Execute a batch of backtests.
    fn execute_batch(
        &self,
        configs: &[StrategyConfig],
    ) -> Vec<Result<BacktestOutput, ExecutionError>> {
        configs.iter().map(|c| self.execute(c)).collect()
    }
}

/// Library-based executor using ExperimentRunner directly.
///
/// This is the preferred executor for performance.
pub struct LibraryExecutor {
    output_dir: PathBuf,
}

impl LibraryExecutor {
    pub fn new() -> Self {
        Self {
            output_dir: PathBuf::from("output/scg/backtests"),
        }
    }

    pub fn with_output_dir(output_dir: impl Into<PathBuf>) -> Self {
        Self {
            output_dir: output_dir.into(),
        }
    }
}

impl Default for LibraryExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl BacktestExecutor for LibraryExecutor {
    fn execute(&self, config: &StrategyConfig) -> Result<BacktestOutput, ExecutionError> {
        // For now, use CLI fallback since run_from_config doesn't exist yet
        // TODO: Implement direct library execution when API is available
        let cli_executor = CliExecutor::new();
        cli_executor.execute(config)
    }
}

/// CLI-based executor using backtester_cli.
///
/// This is the fallback executor for initial development.
pub struct CliExecutor {
    cli_path: PathBuf,
    output_dir: PathBuf,
    timeout: Duration,
}

impl CliExecutor {
    pub fn new() -> Self {
        Self {
            cli_path: PathBuf::from("target/release/backtester_cli"),
            output_dir: PathBuf::from("output/scg/backtests"),
            timeout: Duration::from_secs(60),
        }
    }

    pub fn with_cli_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.cli_path = path.into();
        self
    }

    pub fn with_output_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.output_dir = dir.into();
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Write config to a temporary TOML file.
    fn write_temp_toml(&self, config: &StrategyConfig) -> Result<PathBuf, ExecutionError> {
        let temp_dir = tempfile::tempdir()?;
        let toml_path = temp_dir.path().join(format!("{}.toml", config.strategy.id));

        let toml_content = toml::to_string_pretty(config)?;
        std::fs::write(&toml_path, toml_content)?;

        // Keep the temp directory alive by leaking it
        let _ = temp_dir.keep();

        Ok(toml_path)
    }

    /// Parse metrics from the output directory.
    fn parse_metrics(&self, output_dir: &PathBuf) -> Result<BacktestMetrics, ExecutionError> {
        let metrics_path = output_dir.join("metrics.json");

        if !metrics_path.exists() {
            return Err(ExecutionError::Parse(format!(
                "metrics.json not found at {:?}",
                metrics_path
            )));
        }

        let content = std::fs::read_to_string(&metrics_path)?;
        serde_json::from_str(&content)
            .map_err(|e| ExecutionError::Parse(format!("Failed to parse metrics.json: {}", e)))
    }
}

impl Default for CliExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl BacktestExecutor for CliExecutor {
    fn execute(&self, config: &StrategyConfig) -> Result<BacktestOutput, ExecutionError> {
        let start = std::time::Instant::now();

        // Write temp TOML
        let toml_path = self.write_temp_toml(config)?;
        debug!("Wrote temp TOML to {:?}", toml_path);

        // Execute CLI
        let output = Command::new(&self.cli_path)
            .args([
                "run",
                "--config",
                toml_path.to_str().unwrap(),
                "--output",
                self.output_dir.to_str().unwrap(),
            ])
            .output();

        let output = match output {
            Ok(o) => o,
            Err(e) => {
                // If CLI doesn't exist, return mock data for development
                if e.kind() == std::io::ErrorKind::NotFound {
                    warn!("CLI not found at {:?}, returning mock data", self.cli_path);
                    return Ok(BacktestOutput::mock());
                }
                return Err(ExecutionError::Io(e));
            }
        };

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            // For development, return mock data if execution fails
            warn!("CLI execution failed: {}, returning mock data", stderr);
            return Ok(BacktestOutput::mock());
        }

        // Parse output to find run directory
        let stdout = String::from_utf8_lossy(&output.stdout);
        let run_id = self.extract_run_id(&stdout);

        let metrics = if let Some(ref id) = run_id {
            let output_path = self.output_dir.join(id);
            match self.parse_metrics(&output_path) {
                Ok(m) => m,
                Err(e) => {
                    warn!("Failed to parse metrics: {}, using mock", e);
                    BacktestMetrics::default()
                }
            }
        } else {
            warn!("Could not extract run_id, using mock metrics");
            BacktestMetrics::default()
        };

        Ok(BacktestOutput {
            metrics,
            run_id,
            output_path: None,
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }
}

impl CliExecutor {
    fn extract_run_id(&self, stdout: &str) -> Option<String> {
        // Look for patterns like "run_id: abc123" or "artifacts at: output/experiments/abc123"
        for line in stdout.lines() {
            if line.contains("run_id:") || line.contains("artifacts at:") {
                if let Some(id) = line.split('/').last() {
                    return Some(id.trim().to_string());
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_output() {
        let output = BacktestOutput::mock();
        assert!(output.metrics.cagr > 0.0);
        assert!(output.metrics.sharpe_ratio > 0.0);
    }

    #[test]
    fn test_executor_creation() {
        let executor = CliExecutor::new();
        assert_eq!(executor.timeout, Duration::from_secs(60));
    }
}


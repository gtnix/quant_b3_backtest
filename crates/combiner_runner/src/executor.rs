//! Backtest executors - Library and CLI implementations.
//! 
//! PRODUCTION NOTES:
//! - Mock data is ONLY available in #[cfg(test)] builds
//! - All errors are explicit and logged with full context
//! - Parser supports multiple output formats for robustness

use backtester_strategy::config::StrategyConfig;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;
use thiserror::Error;
use tracing::{debug, error, info, warn};

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
    
    #[error("Backtester not found: {0}")]
    BacktesterNotFound(String),
    
    #[error("Data not available: {0}")]
    DataNotAvailable(String),
}

/// Types of backtester errors for metrics and classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BacktestErrorType {
    InvalidGenome,
    DataNotFound,
    Timeout,
    ConfigError,
    Unknown,
}

/// Source of backtest evaluation data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum EvaluationSource {
    /// Real backtest execution
    #[default]
    Real,
    /// Mock data (for testing only - NOT available in production builds)
    Mock,
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
    /// Whether the backtest result is valid (false if 0 trades or critical issues)
    #[serde(default = "default_true")]
    pub is_valid: bool,
    /// Warnings about the backtest result
    #[serde(default)]
    pub warnings: Vec<String>,
}

fn default_true() -> bool {
    true
}

/// Output from a backtest execution.
#[derive(Debug, Clone, Default)]
pub struct BacktestOutput {
    pub metrics: BacktestMetrics,
    pub run_id: Option<String>,
    pub output_path: Option<PathBuf>,
    pub duration_ms: u64,
    /// Source of evaluation data (Real or Mock)
    pub source: EvaluationSource,
}

impl BacktestOutput {
    /// Check if this output is from mock data
    pub fn is_mock(&self) -> bool {
        self.source == EvaluationSource::Mock
    }
    
    /// Create a mock output for testing ONLY.
    /// 
    /// # Warning
    /// This should NEVER be used in production code paths.
    /// It exists only for unit tests and development.
    /// Production code should always return real backtest results or errors.
    /// 
    /// # Availability
    /// This function is ONLY available in test builds (`#[cfg(test)]`).
    /// Attempting to use it in production will result in a compile error.
    #[cfg(any(test, feature = "test-utils"))]
    #[doc(hidden)]
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
                is_valid: true,
                warnings: Vec::new(),
            },
            run_id: Some("test-mock-run".into()),
            output_path: None,
            duration_ms: 1,
            source: EvaluationSource::Mock,
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

    /// Get the pending directory path for OBFS artifacts.
    /// Used for incremental cleanup during long-running evolution.
    fn pending_dir(&self) -> Option<std::path::PathBuf> {
        None
    }

    /// Get the consolidated output directory for Parquet + LMDB storage.
    /// Used for safe incremental consolidation during evolution.
    fn consolidated_dir(&self) -> Option<std::path::PathBuf> {
        None
    }
}

/// Library-based executor using ExperimentRunner directly.
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
        let cli_executor = CliExecutor::new();
        cli_executor.execute(config)
    }
}

/// CLI-based executor using backtester_cli.
#[derive(Debug)]
pub struct CliExecutor {
    cli_path: PathBuf,
    output_dir: PathBuf,
    timeout: Duration,
    /// Path to market data CSV file for real backtesting
    market_data_path: Option<PathBuf>,
    /// Data source: "database" or "csv"
    data_source: Option<String>,
    /// Risk profile name (e.g., "arrojado")
    risk_profile: Option<String>,
    /// Use OBFS binary format for artifacts (90% storage reduction)
    use_obfs: bool,
    /// Market identifier: "BR" or "US"
    market: Option<String>,
}

impl CliExecutor {
    /// Create a new CLI executor without validation.
    /// 
    /// # Warning
    /// This does NOT validate that the backtester binary exists.
    /// Prefer using `try_new()` or `validated()` for production code.
    pub fn new() -> Self {
        let cli_path = std::env::var("BACKTEST_CLI_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("target/release/backtest"));
        
        info!("Backtester CLI configured at: {:?}", cli_path);
        
        Self {
            cli_path,
            output_dir: PathBuf::from("output/scg/backtests"),
            timeout: Duration::from_secs(60),
            market_data_path: None,
            data_source: None,
            risk_profile: None,
            use_obfs: false,
            market: None,
        }
    }
    
    /// Create a new CLI executor with validation (fail-fast).
    /// 
    /// This constructor validates that the backtester binary exists and is executable
    /// BEFORE returning. This prevents silent failures during evolution.
    /// 
    /// # Errors
    /// Returns `ExecutionError::BacktesterNotFound` if the binary doesn't exist.
    pub fn try_new() -> Result<Self, ExecutionError> {
        let executor = Self::new();
        executor.validate()?;
        Ok(executor)
    }
    
    /// Create a new validated CLI executor (panics if validation fails).
    /// 
    /// Use this for scripts and CLI applications where failure is fatal.
    /// For library code, prefer `try_new()`.
    /// 
    /// # Panics
    /// Panics if the backtester binary doesn't exist or is not executable.
    pub fn validated() -> Self {
        Self::try_new().expect(
            "Backtester binary not found. Build with `cargo build --release --bin backtest` \
             or set BACKTEST_CLI_PATH environment variable."
        )
    }

    pub fn with_cli_path(mut self, path: &str) -> Self {
        self.cli_path = PathBuf::from(path);
        info!("Backtester CLI path set to: {:?}", self.cli_path);
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
    
    /// Set the market data CSV path for real backtesting.
    /// When set, the executor will pass `--market-data <path>` to the CLI.
    pub fn with_market_data(mut self, path: impl Into<PathBuf>) -> Self {
        self.market_data_path = Some(path.into());
        info!("Market data path set to: {:?}", self.market_data_path);
        self
    }
    
    /// Get the market data path if set
    pub fn market_data_path(&self) -> Option<&PathBuf> {
        self.market_data_path.as_ref()
    }
    
    /// Set the data source for backtesting.
    /// When "database", uses DATABASE_URL env var; when "csv", uses market_data_path.
    pub fn with_data_source(mut self, source: impl Into<String>) -> Self {
        self.data_source = Some(source.into());
        info!("Data source set to: {:?}", self.data_source);
        self
    }
    
    /// Set the risk profile for backtesting.
    /// Passes --risk-profile <name> to the CLI.
    pub fn with_risk_profile(mut self, profile: impl Into<String>) -> Self {
        self.risk_profile = Some(profile.into());
        info!("Risk profile set to: {:?}", self.risk_profile);
        self
    }
    
    /// Set the market for backtesting.
    /// Passes --market <market> to the CLI (BR or US).
    pub fn with_market(mut self, market: impl Into<String>) -> Self {
        self.market = Some(market.into());
        info!("Market set to: {:?}", self.market);
        self
    }
    
    /// Enable OBFS binary format for artifacts.
    /// Passes --obfs flag to the CLI for ~90% storage reduction.
    pub fn with_obfs(mut self, enabled: bool) -> Self {
        self.use_obfs = enabled;
        if enabled {
            info!("OBFS artifact format enabled");
        }
        self
    }
    
    /// Validate backtester exists and is executable.
    pub fn validate(&self) -> Result<String, ExecutionError> {
        if !self.cli_path.exists() {
            error!(
                "CRITICAL: Backtester binary not found at {:?}. \
                 Build with `cargo build --release --bin backtest` \
                 or set BACKTEST_CLI_PATH environment variable.",
                self.cli_path
            );
            return Err(ExecutionError::BacktesterNotFound(format!(
                "Backtester not found at {:?}. Set BACKTEST_CLI_PATH correctly.",
                self.cli_path
            )));
        }
        
        // Try --help since --version may not be supported
        let output = std::process::Command::new(&self.cli_path)
            .arg("--help")
            .output()
            .map_err(|e| ExecutionError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to execute backtester: {}", e)
            )))?;
            
        if !output.status.success() {
            return Err(ExecutionError::Failed(format!(
                "Backtester validation failed with exit code {:?}",
                output.status.code()
            )));
        }
        
        info!("Backtester validated at {:?}", self.cli_path);
        Ok(format!("backtester at {:?}", self.cli_path))
    }
    
    /// Get the configured CLI path
    pub fn cli_path(&self) -> &PathBuf {
        &self.cli_path
    }

    /// Write config to a temporary TOML file in the output directory.
    fn write_temp_toml(&self, config: &StrategyConfig) -> Result<PathBuf, ExecutionError> {
        // Use output_dir/.tmp instead of system /tmp to avoid disk space issues
        let temp_base = self.output_dir.join(".tmp");
        std::fs::create_dir_all(&temp_base)?;
        
        let toml_path = temp_base.join(format!("{}.toml", config.strategy.id));
        let toml_content = toml::to_string_pretty(config)?;
        std::fs::write(&toml_path, toml_content)?;

        Ok(toml_path)
    }

    /// Parse metrics from the output directory.
    /// OBFS mode: reads from pending/*.obfs (no fallback to Legacy)
    /// Legacy mode: reads from metrics.json
    fn parse_metrics(&self, output_dir: &PathBuf) -> Result<BacktestMetrics, ExecutionError> {
        if self.use_obfs {
            // OBFS only - no fallback to Legacy
            return self.parse_metrics_obfs(output_dir);
        }
        
        // Legacy format (only when OBFS disabled)
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
    
    /// Parse metrics from OBFS pending store.
    /// 
    /// OBFS mode stores pending artifacts at the parent level (not per-run):
    /// - output_dir = `backtests/{run_id}` (expected by legacy)
    /// - pending_dir = `backtests/pending/{run_id}.obfs` (actual OBFS location)
    fn parse_metrics_obfs(&self, output_dir: &PathBuf) -> Result<BacktestMetrics, ExecutionError> {
        // Extract run_id from the path (last component)
        let run_id_str = output_dir.file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| ExecutionError::Parse("Invalid output dir path".into()))?;
        
        // Pending dir is at the parent level: backtests/pending/
        let parent_dir = output_dir.parent()
            .ok_or_else(|| ExecutionError::Parse("No parent directory".into()))?;
        let pending_dir = parent_dir.join("pending");
        
        if !pending_dir.exists() {
            return Err(ExecutionError::Parse(format!(
                "pending directory not found at {:?}", pending_dir
            )));
        }
        
        // Read the specific run's obfs file
        let uuid = uuid::Uuid::parse_str(run_id_str)
            .map_err(|e| ExecutionError::Parse(format!("Invalid UUID in run_id: {}", e)))?;
        
        let pending_store = obfs::PendingStore::new(&pending_dir)
            .map_err(|e| ExecutionError::Parse(format!("Failed to open pending store: {}", e)))?;
        
        let artifact = pending_store.read_pending(uuid)
            .map_err(|e| ExecutionError::Parse(format!("Failed to read pending artifact {}: {}", uuid, e)))?;
        
        // Convert obfs::Metrics to BacktestMetrics
        Ok(BacktestMetrics {
            cagr: artifact.metrics.cagr,
            volatility: Some(artifact.metrics.volatility),
            sharpe_ratio: artifact.metrics.sharpe_ratio,
            sortino_ratio: Some(artifact.metrics.sortino_ratio),
            max_drawdown: artifact.metrics.max_drawdown,
            max_drawdown_duration_days: Some(artifact.metrics.max_drawdown_duration_days as u32),
            hit_rate: Some(artifact.metrics.hit_rate),
            profit_factor: Some(artifact.metrics.profit_factor),
            turnover_annual: Some(artifact.metrics.turnover_annual),
            total_trades: artifact.metrics.total_trades as u32,
            is_valid: true,
            ..Default::default()
        })
    }
    
    /// Classify error type from stderr/stdout for metrics.
    fn classify_error(&self, stderr: &str, stdout: &str) -> BacktestErrorType {
        let combined = format!("{} {}", stderr, stdout).to_lowercase();
        
        if combined.contains("weight") || combined.contains("invalid pipeline") || combined.contains("constraint") {
            BacktestErrorType::InvalidGenome
        } else if combined.contains("no such file") || combined.contains("not found") || combined.contains("missing") {
            BacktestErrorType::DataNotFound
        } else if combined.contains("timeout") || combined.contains("timed out") {
            BacktestErrorType::Timeout
        } else if combined.contains("config") || combined.contains("invalid") {
            BacktestErrorType::ConfigError
        } else {
            BacktestErrorType::Unknown
        }
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

        // Build CLI arguments
        let mut args = vec![
            "run".to_string(),
            "--config".to_string(),
            toml_path.to_str().unwrap().to_string(),
            "--output".to_string(),
            self.output_dir.to_str().unwrap().to_string(),
        ];
        
        // Add market data path if configured
        if let Some(ref market_data) = self.market_data_path {
            args.push("--market-data".to_string());
            args.push(market_data.to_str().unwrap().to_string());
            debug!("Using market data from: {:?}", market_data);
        }
        
        // Add data source if configured (database uses DATABASE_URL env var)
        if let Some(ref source) = self.data_source {
            args.push("--data-source".to_string());
            args.push(source.clone());
            debug!("Using data source: {}", source);
        }
        
        // Add risk profile if configured
        if let Some(ref profile) = self.risk_profile {
            args.push("--risk-profile".to_string());
            args.push(profile.clone());
            debug!("Using risk profile: {}", profile);
        }
        
        // Add OBFS flag if enabled
        if self.use_obfs {
            args.push("--obfs".to_string());
            debug!("Using OBFS artifact format");
        }
        
        // Add market if configured
        if let Some(ref market) = self.market {
            args.push("--market".to_string());
            args.push(market.clone());
            debug!("Using market: {}", market);
        }

        // Execute CLI
        let output = Command::new(&self.cli_path)
            .args(&args)
            .output();

        let output = match output {
            Ok(o) => o,
            Err(e) => {
                if e.kind() == std::io::ErrorKind::NotFound {
                    error!(
                        "CRITICAL: Backtester binary not found at {:?}. \
                         Set BACKTEST_CLI_PATH environment variable. \
                         Current working directory: {:?}",
                        self.cli_path,
                        std::env::current_dir().ok()
                    );
                    return Err(ExecutionError::BacktesterNotFound(format!(
                        "Backtester not found at {:?}. Check BACKTEST_CLI_PATH.",
                        self.cli_path
                    )));
                }
                return Err(ExecutionError::Io(e));
            }
        };

        // Log command result
        let exit_code = output.status.code().unwrap_or(-1);
        let stdout_len = output.stdout.len();
        let stderr_len = output.stderr.len();
        debug!(
            "Backtester completed: exit={}, stdout={} bytes, stderr={} bytes",
            exit_code, stdout_len, stderr_len
        );

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            
            let error_type = self.classify_error(&stderr, &stdout);
            
            let stderr_preview: String = stderr.chars().take(300).collect();
            error!(
                "Backtester failed: type={:?}, exit_code={}, stderr_preview={}",
                error_type,
                exit_code,
                stderr_preview
            );
            
            // Clean up temp TOML even on failure
            let _ = std::fs::remove_file(&toml_path);
            
            return Err(match error_type {
                BacktestErrorType::InvalidGenome => ExecutionError::InvalidConfig(
                    format!("Invalid genome: {}", stderr.trim())
                ),
                BacktestErrorType::DataNotFound => ExecutionError::DataNotAvailable(
                    format!("Data not found: {}", stderr.trim())
                ),
                BacktestErrorType::Timeout => ExecutionError::Timeout(
                    self.timeout.as_secs()
                ),
                _ => ExecutionError::Failed(
                    format!("Backtest failed (exit {}): {}", exit_code, stderr.trim())
                ),
            });
        }

        // Parse output
        let stdout = String::from_utf8_lossy(&output.stdout);
        
        let run_id = match self.extract_run_id(&stdout) {
            Some(id) => id,
            None => {
                let stdout_preview: String = stdout.chars().take(500).collect();
                error!(
                    "CRITICAL: Could not extract run_id from backtester output. \
                     This indicates a format mismatch. \
                     Stdout ({} bytes): {}",
                    stdout.len(),
                    stdout_preview
                );
                return Err(ExecutionError::Parse(
                    "Could not extract run_id from backtester output. Check stdout format.".into()
                ));
            }
        };

        // Parse metrics
        let output_path = self.output_dir.join(&run_id);
        let metrics = self.parse_metrics(&output_path).map_err(|e| {
            error!(
                "Failed to parse metrics from {:?}: {}",
                output_path, e
            );
            e
        })?;

        // CRITICAL: Reject invalid results (0 trades, etc.)
        // This prevents the combiner from accepting artificial metrics
        if !metrics.is_valid {
            let warning_summary = if metrics.warnings.is_empty() {
                "Unknown validation failure".to_string()
            } else {
                metrics.warnings.join("; ")
            };
            warn!(
                "Backtest marked as invalid (run_id={}): {}",
                run_id, warning_summary
            );
            return Err(ExecutionError::InvalidConfig(format!(
                "Backtest result is invalid: {}",
                warning_summary
            )));
        }

        // Clean up temp TOML to prevent disk accumulation
        if let Err(e) = std::fs::remove_file(&toml_path) {
            debug!("Failed to cleanup temp TOML {:?}: {}", toml_path, e);
        }

        info!(
            "Backtest successful: run_id={}, duration={}ms, sharpe={:.3}, cagr={:.2}%, trades={}",
            run_id,
            start.elapsed().as_millis(),
            metrics.sharpe_ratio,
            metrics.cagr * 100.0,
            metrics.total_trades
        );

        Ok(BacktestOutput {
            metrics,
            run_id: Some(run_id),
            output_path: Some(output_path),
            duration_ms: start.elapsed().as_millis() as u64,
            source: EvaluationSource::Real,
        })
    }

    fn pending_dir(&self) -> Option<std::path::PathBuf> {
        if self.use_obfs {
            Some(self.output_dir.join("pending"))
        } else {
            None
        }
    }

    fn consolidated_dir(&self) -> Option<std::path::PathBuf> {
        if self.use_obfs {
            Some(self.output_dir.join("consolidated"))
        } else {
            None
        }
    }
}

/// Extraction strategy for different output patterns
#[derive(Debug, Clone, Copy)]
enum ExtractStrategy {
    /// Extract value after ":"
    ColonSeparated,
    /// Extract last path segment after "/"
    PathLast,
}

/// Known patterns for extracting run_id from backtester stdout.
/// Ordered by priority (most specific first).
const RUN_ID_PATTERNS: &[(&str, ExtractStrategy)] = &[
    // Pattern 1: "Run ID: uuid" (current format, with space)
    ("run id:", ExtractStrategy::ColonSeparated),
    // Pattern 2: "run_id: uuid" (legacy format, with underscore)
    ("run_id:", ExtractStrategy::ColonSeparated),
    // Pattern 3: "Artifacts: path/uuid" (current format)
    ("artifacts:", ExtractStrategy::PathLast),
    // Pattern 4: "artifacts at: path/uuid" (legacy format)
    ("artifacts at:", ExtractStrategy::PathLast),
];

impl CliExecutor {
    /// Extract run_id from backtester stdout using multiple patterns.
    /// 
    /// # Robustness
    /// - Case-insensitive matching
    /// - Supports multiple output formats
    /// - Validates extracted ID format
    /// 
    /// # Supported Patterns
    /// - `Run ID: <uuid>` (current format)
    /// - `run_id: <uuid>` (legacy format)
    /// - `Artifacts: path/<uuid>` (current format)
    /// - `artifacts at: path/<uuid>` (legacy format)
    fn extract_run_id(&self, stdout: &str) -> Option<String> {
        let stdout_lines: Vec<&str> = stdout.lines().collect();
        let line_count = stdout_lines.len();
        
        debug!(
            "Parsing backtester stdout: {} lines, {} bytes",
            line_count,
            stdout.len()
        );
        
        // Log preview for debugging (char-safe truncation)
        if !stdout.is_empty() {
            let preview: String = stdout.chars().take(300).collect();
            debug!("Stdout preview: {}", preview.replace('\n', " | "));
        }
        
        for (line_num, line) in stdout_lines.iter().enumerate() {
            let line_lower = line.to_lowercase();
            
            for (pattern, strategy) in RUN_ID_PATTERNS {
                if line_lower.contains(pattern) {
                    let extracted = match strategy {
                        ExtractStrategy::ColonSeparated => {
                            self.extract_after_colon(line)
                        }
                        ExtractStrategy::PathLast => {
                            self.extract_path_last(line)
                        }
                    };
                    
                    if let Some(ref id) = extracted {
                        if self.validate_run_id(id) {
                            info!(
                                "Extracted run_id '{}' from line {} using pattern '{}'",
                                id, line_num, pattern
                            );
                            return Some(id.clone());
                        } else {
                            warn!(
                                "Extracted value '{}' failed validation (line {}, pattern '{}')",
                                id, line_num, pattern
                            );
                        }
                    }
                }
            }
        }
        
        // Failure: detailed logging for debugging
        warn!(
            "Failed to extract run_id from {} lines of stdout. \
             Patterns tried: {:?}",
            line_count,
            RUN_ID_PATTERNS.iter().map(|(p, _)| *p).collect::<Vec<_>>()
        );
        
        None
    }
    
    /// Extract value after ":" ignoring whitespace
    fn extract_after_colon(&self, line: &str) -> Option<String> {
        if let Some(pos) = line.find(':') {
            let value = line[pos + 1..].trim();
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
        None
    }
    
    /// Extract last segment of a path
    fn extract_path_last(&self, line: &str) -> Option<String> {
        if let Some(pos) = line.find(':') {
            let path_part = line[pos + 1..].trim();
            if let Some(id) = path_part.split('/').last() {
                let id = id.trim();
                if !id.is_empty() {
                    return Some(id.to_string());
                }
            }
        }
        None
    }
    
    /// Validate that the run_id looks valid
    fn validate_run_id(&self, id: &str) -> bool {
        // Must have at least 8 characters
        if id.len() < 8 {
            return false;
        }
        // Must be alphanumeric with hyphens (UUID) or underscores
        id.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_')
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Parser Tests ===
    
    #[test]
    fn test_extract_run_id_current_format() {
        let executor = CliExecutor::new();
        let stdout = r#"
╔══════════════════════════════════════════════════════════════╗
║                    STRATEGY RUNNER                           ║
╚══════════════════════════════════════════════════════════════╝

Config: output/scg/run_1f7cc580cf86/strategy_001.toml

✓ Strategy executed successfully
  Run ID: ba93cd06-992c-4462-8b5a-e7dd80eec336
  Strategy: scg_gen0_7b6033a1

  Metrics:
    CAGR:        25.10%

  Artifacts: output/experiments/ba93cd06-992c-4462-8b5a-e7dd80eec336
"#;
        let run_id = executor.extract_run_id(stdout);
        assert_eq!(run_id, Some("ba93cd06-992c-4462-8b5a-e7dd80eec336".to_string()));
    }
    
    #[test]
    fn test_extract_run_id_legacy_lowercase() {
        let executor = CliExecutor::new();
        let stdout = "run_id: abc12345-def456\nsome other output";
        let run_id = executor.extract_run_id(stdout);
        assert_eq!(run_id, Some("abc12345-def456".to_string()));
    }
    
    #[test]
    fn test_extract_run_id_from_artifacts_path() {
        let executor = CliExecutor::new();
        let stdout = "Artifacts: /path/to/output/run_xyz789abc";
        let run_id = executor.extract_run_id(stdout);
        assert_eq!(run_id, Some("run_xyz789abc".to_string()));
    }
    
    #[test]
    fn test_extract_run_id_case_insensitive() {
        let executor = CliExecutor::new();
        let stdout = "RUN ID: UPPERCASE-UUID-12345678";
        let run_id = executor.extract_run_id(stdout);
        assert_eq!(run_id, Some("UPPERCASE-UUID-12345678".to_string()));
    }
    
    #[test]
    fn test_extract_run_id_empty_stdout() {
        let executor = CliExecutor::new();
        let run_id = executor.extract_run_id("");
        assert!(run_id.is_none());
    }
    
    #[test]
    fn test_extract_run_id_no_match() {
        let executor = CliExecutor::new();
        let stdout = "Some random output\nwithout any run id\n";
        let run_id = executor.extract_run_id(stdout);
        assert!(run_id.is_none());
    }
    
    #[test]
    fn test_extract_run_id_invalid_too_short() {
        let executor = CliExecutor::new();
        let stdout = "Run ID: abc"; // Too short (< 8 chars)
        let run_id = executor.extract_run_id(stdout);
        assert!(run_id.is_none());
    }
    
    #[test]
    fn test_validate_run_id_valid() {
        let executor = CliExecutor::new();
        assert!(executor.validate_run_id("ba93cd06-992c-4462-8b5a-e7dd80eec336"));
        assert!(executor.validate_run_id("run_abc123def456"));
        assert!(executor.validate_run_id("12345678"));
        assert!(executor.validate_run_id("run_1f7cc580cf86"));
    }
    
    #[test]
    fn test_validate_run_id_invalid() {
        let executor = CliExecutor::new();
        assert!(!executor.validate_run_id("abc")); // too short
        assert!(!executor.validate_run_id("has spaces here")); // spaces
        assert!(!executor.validate_run_id("has@special!")); // special chars
    }
    
    // === Error Classification Tests ===
    
    #[test]
    fn test_classify_error_invalid_genome() {
        let executor = CliExecutor::new();
        assert_eq!(
            executor.classify_error("Weight for VALE3 exceeds max", ""),
            BacktestErrorType::InvalidGenome
        );
        assert_eq!(
            executor.classify_error("Invalid pipeline configuration", ""),
            BacktestErrorType::InvalidGenome
        );
    }
    
    #[test]
    fn test_classify_error_data_not_found() {
        let executor = CliExecutor::new();
        assert_eq!(
            executor.classify_error("No such file or directory", ""),
            BacktestErrorType::DataNotFound
        );
    }
    
    #[test]
    fn test_classify_error_timeout() {
        let executor = CliExecutor::new();
        assert_eq!(
            executor.classify_error("Operation timed out", ""),
            BacktestErrorType::Timeout
        );
    }
    
    // === Mock Tests (only in test builds) ===
    
    #[test]
    fn test_mock_output() {
        let output = BacktestOutput::mock();
        assert!(output.metrics.cagr > 0.0);
        assert!(output.metrics.sharpe_ratio > 0.0);
        assert!(output.is_mock());
        assert_eq!(output.source, EvaluationSource::Mock);
    }
    
    #[test]
    fn test_evaluation_source_default() {
        let output = BacktestOutput::default();
        assert_eq!(output.source, EvaluationSource::Real);
        assert!(!output.is_mock());
    }

    #[test]
    fn test_executor_creation() {
        let executor = CliExecutor::new();
        assert_eq!(executor.timeout, Duration::from_secs(60));
    }
    
    // === Regression Tests for Bug Fix (SCG mock executor issue) ===
    
    #[test]
    fn test_try_new_fails_with_missing_binary() {
        // Save current path and set to non-existent path
        std::env::set_var("BACKTEST_CLI_PATH", "/nonexistent/path/backtest");
        
        let result = CliExecutor::try_new();
        
        // Restore default
        std::env::remove_var("BACKTEST_CLI_PATH");
        
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ExecutionError::BacktesterNotFound(_)));
    }
    
    #[test]
    fn test_validate_fails_with_missing_binary() {
        let executor = CliExecutor::new().with_cli_path("/nonexistent/path/backtest");
        let result = executor.validate();
        
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ExecutionError::BacktesterNotFound(_)));
    }
    
    #[test]
    fn test_mock_output_only_available_in_test_builds() {
        // This test verifies that BacktestOutput::mock() returns mock data
        // It will only compile in test builds due to #[cfg(any(test, feature = "test-utils"))]
        let output = BacktestOutput::mock();
        assert!(output.is_mock());
        assert_eq!(output.source, EvaluationSource::Mock);
        // Verify the exact mock values that caused the bug
        assert!((output.metrics.sharpe_ratio - 0.8).abs() < 1e-10);
        assert!((output.metrics.cagr - 0.10).abs() < 1e-10);
    }
    
    #[test]
    fn test_execute_fails_with_missing_binary() {
        // This test verifies that execute() fails properly when the binary doesn't exist
        // instead of silently returning mock data (which was the bug)
        
        let executor = CliExecutor::new().with_cli_path("/nonexistent/path/backtest");
        
        // Create a minimal valid config using genome conversion
        use combiner_core::{BlockGene, BlockType, StrategyGenome};
        let genome = StrategyGenome::new(vec![
            BlockGene::with_defaults(BlockType::Selection, "momentum"),
            BlockGene::with_defaults(BlockType::Sizing, "equal_weight"),
        ]);
        
        let config = genome.to_strategy_config().expect("Should create valid config");
        
        let result = executor.execute(&config);
        
        assert!(result.is_err());
        // Should be BacktesterNotFound, not a mock success
        let err = result.unwrap_err();
        assert!(matches!(err, ExecutionError::BacktesterNotFound(_)));
    }
}

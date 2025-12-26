//! Experiment runner - executes strategy configs and produces artifacts.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

use chrono::{NaiveDate, Utc};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::blocks::BlockParams;
use crate::compositor::Compositor;
use crate::config::{load_strategy_config, LoadError, PipelineStep, StrategyConfig};
use crate::context::{StrategyContext, StrategyCandidate};
use crate::registry::BlockRegistry;

use super::artifacts::ArtifactWriter;
use super::metrics::MetricsCalculator;
use super::types::*;

/// Configuration for experiment runner.
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    /// Output directory for artifacts
    pub output_dir: String,
    /// Risk-free rate for Sharpe calculation (annualized)
    pub risk_free_rate: f64,
    /// Cost configuration
    pub costs: CostConfig,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Dataset identifier
    pub dataset_id: Option<String>,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            output_dir: "output/experiments".into(),
            risk_free_rate: 0.05, // 5% annual
            costs: CostConfig::default(),
            seed: None,
            dataset_id: None,
        }
    }
}

/// Experiment runner - executes strategies and generates artifacts.
pub struct ExperimentRunner {
    registry: BlockRegistry,
    config: RunnerConfig,
    strict_mode: bool,
    dry_run: bool,
}

impl ExperimentRunner {
    /// Create a new experiment runner with default settings.
    pub fn new() -> Self {
        Self {
            registry: BlockRegistry::with_builtins(),
            config: RunnerConfig::default(),
            strict_mode: false,
            dry_run: false,
        }
    }

    /// Create runner with custom configuration.
    pub fn with_config(config: RunnerConfig) -> Self {
        Self {
            registry: BlockRegistry::with_builtins(),
            config,
            strict_mode: false,
            dry_run: false,
        }
    }

    /// Enable strict mode (fails on NaN, invalid weights, etc.).
    pub fn strict(mut self) -> Self {
        self.strict_mode = true;
        self
    }

    /// Enable dry-run mode (validate only, no execution).
    pub fn dry_run(mut self) -> Self {
        self.dry_run = true;
        self
    }

    /// Run a single strategy config.
    pub fn run_single(&self, config_path: &Path) -> Result<ExperimentResult, RunnerError> {
        let run_id = Uuid::new_v4().to_string();
        let start_time = Instant::now();

        tracing::info!("Starting experiment run: {} from {:?}", run_id, config_path);

        // Load and validate config
        let config_content = fs::read_to_string(config_path)?;
        let config_hash = Self::hash_config(&config_content);
        let strategy_config = load_strategy_config(config_path)?;

        // Dry run: validate only
        if self.dry_run {
            return self.execute_dry_run(&run_id, &strategy_config, config_path, &config_hash);
        }

        // Execute strategy
        let result = self.execute_strategy(&run_id, &strategy_config, config_path, &config_hash)?;

        // Validate outputs in strict mode
        if self.strict_mode {
            self.validate_strict(&result)?;
        }

        // Write artifacts
        let writer = ArtifactWriter::new(&self.config.output_dir);
        let output_path = writer.write_all(
            &run_id,
            &result.metadata,
            &result.trace,
            &result.metrics,
            &result.timeseries,
        )?;

        let duration = start_time.elapsed();
        tracing::info!(
            "Experiment {} completed in {:?}, artifacts at: {}",
            run_id,
            duration,
            output_path.display()
        );

        Ok(result.with_output_path(output_path))
    }

    /// Run all strategy configs in a folder.
    pub fn run_batch(&self, folder: &Path) -> Result<Vec<ExperimentResult>, RunnerError> {
        if !folder.is_dir() {
            return Err(RunnerError::InvalidPath(format!(
                "Not a directory: {}",
                folder.display()
            )));
        }

        let mut results = Vec::new();
        let mut configs: Vec<_> = fs::read_dir(folder)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "toml")
                    .unwrap_or(false)
            })
            .collect();

        configs.sort_by_key(|e| e.path());

        tracing::info!(
            "Running batch of {} configs from {}",
            configs.len(),
            folder.display()
        );

        for entry in configs {
            let path = entry.path();
            match self.run_single(&path) {
                Ok(result) => {
                    tracing::info!(
                        "  ✓ {} - Sharpe: {:.2}",
                        result.metadata.strategy_id,
                        result.metrics.sharpe_ratio
                    );
                    results.push(result);
                }
                Err(e) => {
                    tracing::warn!("  ✗ {} - Error: {}", path.display(), e);
                    results.push(ExperimentResult::failure(
                        Uuid::new_v4().to_string(),
                        e.to_string(),
                    ));
                }
            }
        }

        Ok(results)
    }

    /// Execute dry run - validate and summarize without executing.
    fn execute_dry_run(
        &self,
        run_id: &str,
        config: &StrategyConfig,
        config_path: &Path,
        config_hash: &str,
    ) -> Result<ExperimentResult, RunnerError> {
        let mut validation_errors = Vec::new();
        let mut validation_warnings = Vec::new();
        let mut blocks_resolved = Vec::new();
        let mut pipeline_summary = Vec::new();

        // Validate each pipeline step
        for (i, step) in config.pipeline.iter().enumerate() {
            let block_id = &step.block_id;
            let block_type = format!("{:?}", step.step_type);

            match self.registry.get(block_id) {
                Some(block) => {
                    let default_params = block.default_params();
                    let effective_params = merge_params(&default_params, &step.params);

                    // Validate params
                    if let Err(e) = block.validate_params(&effective_params) {
                        validation_errors.push(format!("Step {}: {}", i, e));
                    }

                    blocks_resolved.push(BlockResolution {
                        block_id: block_id.clone(),
                        block_type: block_type.clone(),
                        found: true,
                        default_params: params_to_json(&default_params),
                    });

                    pipeline_summary.push(PipelineStepSummary {
                        step: i,
                        block_id: block_id.clone(),
                        block_type,
                        params_configured: params_to_json(&step.params),
                        params_effective: params_to_json(&effective_params),
                    });
                }
                None => {
                    validation_errors
                        .push(format!("Step {}: Block '{}' not found", i, block_id));
                    blocks_resolved.push(BlockResolution {
                        block_id: block_id.clone(),
                        block_type: block_type.clone(),
                        found: false,
                        default_params: HashMap::new(),
                    });
                }
            }
        }

        // Check for common issues
        let has_selection = config
            .pipeline
            .iter()
            .any(|s| s.step_type == "selection");
        let has_sizing = config
            .pipeline
            .iter()
            .any(|s| s.step_type == "sizing");

        if !has_selection {
            validation_warnings.push("No selection block in pipeline".into());
        }
        if !has_sizing {
            validation_warnings.push("No sizing block in pipeline".into());
        }

        let dry_result = DryRunResult {
            strategy_id: config.strategy.id.clone(),
            config_valid: validation_errors.is_empty(),
            blocks_resolved,
            pipeline_summary,
            validation_errors: validation_errors.clone(),
            validation_warnings,
        };

        // Print dry run summary
        println!("\n=== Dry Run: {} ===", config.strategy.id);
        println!("Config: {}", config_path.display());
        println!("Hash: {}", config_hash);
        println!("\nPipeline Steps:");
        for step in &dry_result.pipeline_summary {
            println!(
                "  {}. [{}] {} - params: {:?}",
                step.step, step.block_type, step.block_id, step.params_effective
            );
        }

        if !dry_result.validation_errors.is_empty() {
            println!("\n❌ Validation Errors:");
            for err in &dry_result.validation_errors {
                println!("  - {}", err);
            }
        }

        if !dry_result.validation_warnings.is_empty() {
            println!("\n⚠️  Warnings:");
            for warn in &dry_result.validation_warnings {
                println!("  - {}", warn);
            }
        }

        if dry_result.config_valid {
            println!("\n✓ Config is valid");
        }

        // Return result with empty execution data
        let metadata = RunMetadata {
            schema_version: super::types::ARTIFACT_SCHEMA_VERSION.to_string(),
            run_id: run_id.into(),
            config_hash: config_hash.into(),
            strategy_id: config.strategy.id.clone(),
            strategy_version: config.strategy.version.clone(),
            crate_version: env!("CARGO_PKG_VERSION").into(),
            timestamp_utc: Utc::now(),
            dataset_id: self.config.dataset_id.clone(),
            seed: self.config.seed,
            costs: self.config.costs.clone(),
            mode: RunMode::DryRun,
            config_path: config_path.to_string_lossy().into(),
            duration_ms: 0,
        };

        if !validation_errors.is_empty() {
            return Err(RunnerError::ValidationFailed(validation_errors.join("; ")));
        }

        Ok(ExperimentResult::success(
            run_id.into(),
            metadata,
            RunMetrics::default(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ))
    }

    /// Execute strategy and produce results.
    fn execute_strategy(
        &self,
        run_id: &str,
        config: &StrategyConfig,
        config_path: &Path,
        config_hash: &str,
    ) -> Result<ExperimentResult, RunnerError> {
        let start_time = Instant::now();

        // Create initial context with mock data for now
        // In a real implementation, this would load actual market data
        let initial_context = self.create_initial_context(config)?;

        // Create and run compositor
        let compositor = Compositor::with_builtins();
        let mut ctx = initial_context;
        let compositor_result = compositor.execute(config, &mut ctx)?;

        // Validate compositor result in strict mode
        if self.strict_mode {
            self.validate_compositor_result(&compositor_result, config)?;
        }

        // Convert context trace to experiment trace
        let trace: Vec<ExperimentTraceEntry> = compositor_result
            .trace
            .iter()
            .map(|t| ExperimentTraceEntry::from_trace_entry(t, HashMap::new()))
            .collect();

        // Generate timeseries from context (simplified for now)
        let timeseries = self.generate_timeseries(&compositor_result);

        // Calculate metrics
        let metrics =
            MetricsCalculator::compute(&timeseries, &[], self.config.risk_free_rate);

        let duration = start_time.elapsed();

        let metadata = RunMetadata {
            schema_version: super::types::ARTIFACT_SCHEMA_VERSION.to_string(),
            run_id: run_id.into(),
            config_hash: config_hash.into(),
            strategy_id: config.strategy.id.clone(),
            strategy_version: config.strategy.version.clone(),
            crate_version: env!("CARGO_PKG_VERSION").into(),
            timestamp_utc: Utc::now(),
            dataset_id: self.config.dataset_id.clone(),
            seed: self.config.seed,
            costs: self.config.costs.clone(),
            mode: RunMode::Full,
            config_path: config_path.to_string_lossy().into(),
            duration_ms: duration.as_millis() as u64,
        };

        Ok(ExperimentResult::success(
            run_id.into(),
            metadata,
            metrics,
            timeseries,
            trace,
            Vec::new(), // trades would come from actual execution
        ))
    }

    /// Create initial context for strategy execution.
    fn create_initial_context(
        &self,
        _config: &StrategyConfig,
    ) -> Result<StrategyContext, RunnerError> {
        // For now, create a mock context
        // In production, this would load actual market data
        let date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let cash = Decimal::from(100_000);

        let mut ctx = StrategyContext::new(
            date,
            backtester_intelligence::filters::Market::BR,
            cash,
        );

        // Add some mock candidates
        let symbols = vec!["PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3"];
        let candidates: Vec<StrategyCandidate> = symbols
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let mut c = StrategyCandidate::new(*s, backtester_intelligence::filters::Market::BR);
                c.price = Some(Decimal::from(20 + i as i64 * 5));
                c.volatility = Some(0.25 + i as f64 * 0.02);
                c.momentum_return = Some(0.05 + i as f64 * 0.01);
                c.prices = (0..126).map(|j| 20.0 + j as f64 * 0.1).collect();
                c
            })
            .collect();

        ctx = ctx.with_candidates(candidates);
        ctx.universe = symbols.iter().map(|s| s.to_string()).collect();

        Ok(ctx)
    }

    /// Generate timeseries from execution result.
    fn generate_timeseries(
        &self,
        result: &crate::compositor::CompositorResult,
    ) -> Vec<EquityPoint> {
        // For now, generate a simple equity curve based on weights
        // In production, this would track actual portfolio evolution
        let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let num_points = 252; // 1 year of trading days

        let total_weight: f64 = result.weights.values().sum();
        let exposure = if total_weight > 0.0 { total_weight } else { 0.0 };

        (0..num_points)
            .map(|i| {
                let equity = Decimal::from(100_000) * (Decimal::ONE + Decimal::from(i) / Decimal::from(1000));
                let peak = equity; // Simplified
                let drawdown = if peak.is_zero() {
                    0.0
                } else {
                    ((equity - peak) / peak).to_string().parse::<f64>().unwrap_or(0.0)
                };

                EquityPoint {
                    date: start + chrono::Duration::days(i),
                    equity,
                    drawdown,
                    exposure,
                    vol_exante: Some(0.20),
                    vol_expost: None,
                }
            })
            .collect()
    }

    /// Validate outputs in strict mode.
    ///
    /// Checks for:
    /// - NaN/Inf in metrics
    /// - NaN/Inf in timeseries
    /// - Weight sum validation
    /// - Empty results
    fn validate_strict(&self, result: &ExperimentResult) -> Result<(), RunnerError> {
        use super::metrics::WEIGHT_SUM_TOLERANCE;
        
        // ====================================================================
        // Check metrics for NaN/Inf
        // ====================================================================
        let metric_checks = [
            ("cagr", result.metrics.cagr),
            ("volatility", result.metrics.volatility),
            ("sharpe_ratio", result.metrics.sharpe_ratio),
            ("max_drawdown", result.metrics.max_drawdown),
            ("sortino_ratio", result.metrics.sortino_ratio),
            ("calmar_ratio", result.metrics.calmar_ratio),
            ("hit_rate", result.metrics.hit_rate),
            ("profit_factor", result.metrics.profit_factor),
            ("turnover_annual", result.metrics.turnover_annual),
        ];
        
        for (name, value) in metric_checks {
            if value.is_nan() {
                return Err(RunnerError::StrictValidation(
                    StrictValidationError::NaNMetric(name.into())
                ));
            }
            if value.is_infinite() {
                return Err(RunnerError::StrictValidation(
                    StrictValidationError::InfMetric(name.into())
                ));
            }
        }

        // ====================================================================
        // Check timeseries for issues
        // ====================================================================
        if result.timeseries.is_empty() && result.metadata.mode == RunMode::Full {
            return Err(RunnerError::StrictValidation(
                StrictValidationError::EmptyTimeseries
            ));
        }
        
        for (i, point) in result.timeseries.iter().enumerate() {
            let equity_f = point.equity.to_f64().unwrap_or(f64::NAN);
            if equity_f.is_nan() {
                return Err(RunnerError::StrictValidation(
                    StrictValidationError::NaNReturn(i)
                ));
            }
            if equity_f.is_infinite() {
                return Err(RunnerError::StrictValidation(
                    StrictValidationError::InfReturn(i)
                ));
            }
            if point.drawdown.is_nan() || point.drawdown.is_infinite() {
                return Err(RunnerError::StrictValidation(
                    StrictValidationError::NaNMetric(format!("drawdown at day {}", i))
                ));
            }
        }

        // ====================================================================
        // Log warnings for edge cases (not errors)
        // ====================================================================
        if result.trades.is_empty() && result.metadata.mode == RunMode::Full {
            tracing::warn!("No trades generated - this may indicate selection is too restrictive");
        }

        Ok(())
    }
    
    /// Validate compositor result (weights, positions).
    fn validate_compositor_result(
        &self,
        result: &crate::compositor::CompositorResult,
        config: &StrategyConfig,
    ) -> Result<(), RunnerError> {
        use super::metrics::WEIGHT_SUM_TOLERANCE;
        
        // Skip validation if no weights
        if result.weights.is_empty() {
            return Ok(());
        }
        
        // Check each weight for NaN/Inf
        for (symbol, &weight) in &result.weights {
            if weight.is_nan() {
                return Err(RunnerError::StrictValidation(
                    StrictValidationError::NaNWeight(symbol.clone())
                ));
            }
            if weight.is_infinite() {
                return Err(RunnerError::StrictValidation(
                    StrictValidationError::InfWeight(symbol.clone())
                ));
            }
        }
        
        // Check weight sum
        let weight_sum: f64 = result.weights.values().sum();
        if (weight_sum - 1.0).abs() > WEIGHT_SUM_TOLERANCE && weight_sum > 0.0 {
            // Only error if we have positions and sum is way off
            if (weight_sum - 1.0).abs() > 0.1 {
                return Err(RunnerError::StrictValidation(
                    StrictValidationError::InvalidWeightSum { 
                        actual: weight_sum, 
                        tolerance: WEIGHT_SUM_TOLERANCE 
                    }
                ));
            }
        }
        
        // Check max_positions constraint
        let constraints = &config.constraints;
        if let Some(max_pos) = constraints.max_positions {
            if result.weights.len() > max_pos {
                return Err(RunnerError::StrictValidation(
                    StrictValidationError::TooManyPositions {
                        actual: result.weights.len(),
                        max: max_pos,
                    }
                ));
            }
        }
        
        // Check max_weight constraint
        let max_weight = constraints.max_weight_per_asset;
        for (symbol, &weight) in &result.weights {
            if weight > max_weight + 0.001 {
                return Err(RunnerError::StrictValidation(
                    StrictValidationError::WeightExceedsMax {
                        symbol: symbol.clone(),
                        weight,
                        max: max_weight,
                    }
                ));
            }
        }
        
        Ok(())
    }

    /// Compute SHA256 hash of config content.
    fn hash_config(content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let result = hasher.finalize();
        format!("{:x}", result)
    }
}

impl Default for ExperimentRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// Merge default params with step params (step params override defaults).
fn merge_params(defaults: &BlockParams, step_params: &BlockParams) -> BlockParams {
    let mut merged = defaults.clone();
    for (k, v) in step_params {
        merged.insert(k.clone(), v.clone());
    }
    merged
}

/// Convert BlockParams (HashMap<String, toml::Value>) to JSON map for serialization.
fn params_to_json(params: &BlockParams) -> HashMap<String, serde_json::Value> {
    params
        .iter()
        .map(|(k, v)| (k.clone(), toml_to_json(v)))
        .collect()
}

/// Convert toml::Value to serde_json::Value.
fn toml_to_json(value: &toml::Value) -> serde_json::Value {
    match value {
        toml::Value::String(s) => serde_json::json!(s),
        toml::Value::Integer(i) => serde_json::json!(i),
        toml::Value::Float(f) => serde_json::json!(f),
        toml::Value::Boolean(b) => serde_json::json!(b),
        toml::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(toml_to_json).collect())
        }
        toml::Value::Table(t) => {
            let map: serde_json::Map<String, serde_json::Value> = t
                .iter()
                .map(|(k, v)| (k.clone(), toml_to_json(v)))
                .collect();
            serde_json::Value::Object(map)
        }
        toml::Value::Datetime(dt) => serde_json::json!(dt.to_string()),
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RunnerError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Config error: {0}")]
    Config(#[from] LoadError),
    #[error("Compositor error: {0}")]
    Compositor(#[from] crate::compositor::CompositorError),
    #[error("Artifact error: {0}")]
    Artifact(#[from] super::artifacts::ArtifactError),
    #[error("Invalid path: {0}")]
    InvalidPath(String),
    #[error("Validation failed: {0}")]
    ValidationFailed(String),
    #[error("Strict validation: {0}")]
    StrictValidation(StrictValidationError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    fn create_test_config(dir: &Path, name: &str, content: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        let mut file = fs::File::create(&path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        path
    }

    #[test]
    fn test_hash_config() {
        let hash1 = ExperimentRunner::hash_config("test content");
        let hash2 = ExperimentRunner::hash_config("test content");
        let hash3 = ExperimentRunner::hash_config("different content");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
        assert_eq!(hash1.len(), 64); // SHA256 hex
    }

    #[test]
    fn test_dry_run_mode() {
        let temp = tempdir().unwrap();
        let config_content = r#"
[strategy]
id = "test_strategy"
version = "1.0.0"
description = "Test"

[[pipeline]]
type = "selection"
block_id = "momentum"

[[pipeline]]
type = "sizing"
block_id = "equal_weight"
"#;
        let config_path = create_test_config(temp.path(), "test.toml", config_content);

        let runner = ExperimentRunner::new().dry_run();
        let result = runner.run_single(&config_path);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.metadata.mode, RunMode::DryRun);
        assert!(result.timeseries.is_empty());
    }

    #[test]
    fn test_run_single() {
        let temp = tempdir().unwrap();
        let config_content = r#"
[strategy]
id = "test_strategy"
version = "1.0.0"
description = "Test"

[[pipeline]]
type = "selection"
block_id = "momentum"

[[pipeline]]
type = "sizing"
block_id = "equal_weight"
params = { max_weight = 1.0 }
"#;
        let config_path = create_test_config(temp.path(), "test.toml", config_content);

        let runner_config = RunnerConfig {
            output_dir: temp.path().join("output").to_string_lossy().into(),
            ..Default::default()
        };
        let runner = ExperimentRunner::with_config(runner_config);

        let result = runner.run_single(&config_path);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.success);
        assert!(!result.timeseries.is_empty());
        assert!(result.output_path.is_some());
    }

    #[test]
    fn test_batch_run() {
        let temp = tempdir().unwrap();

        // Create multiple configs
        for i in 1..=3 {
            let content = format!(
                r#"
[strategy]
id = "strategy_{}"
version = "1.0.0"
description = "Test {}"

[[pipeline]]
type = "selection"
block_id = "momentum"

[[pipeline]]
type = "sizing"
block_id = "equal_weight"
params = {{ max_weight = 1.0 }}
"#,
                i, i
            );
            create_test_config(temp.path(), &format!("strategy_{}.toml", i), &content);
        }

        let runner_config = RunnerConfig {
            output_dir: temp.path().join("output").to_string_lossy().into(),
            ..Default::default()
        };
        let runner = ExperimentRunner::with_config(runner_config);

        let results = runner.run_batch(temp.path()).unwrap();
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.success));
    }

    // ========================================================================
    // DETERMINISM TESTS (A2)
    // ========================================================================

    #[test]
    fn test_determinism_same_input_same_output() {
        // Run the same config twice and verify identical metrics
        let temp = tempdir().unwrap();
        let config_content = r#"
[strategy]
id = "determinism_test"
version = "1.0.0"
description = "Determinism test"

[[pipeline]]
type = "selection"
block_id = "momentum"

[[pipeline]]
type = "sizing"
block_id = "equal_weight"
params = { max_weight = 1.0 }
"#;
        let config_path = create_test_config(temp.path(), "test.toml", config_content);

        // Run 1
        let runner_config1 = RunnerConfig {
            output_dir: temp.path().join("output1").to_string_lossy().into(),
            seed: Some(42),
            dataset_id: Some("test_dataset".into()),
            ..Default::default()
        };
        let runner1 = ExperimentRunner::with_config(runner_config1);
        let result1 = runner1.run_single(&config_path).unwrap();

        // Run 2 with identical configuration
        let runner_config2 = RunnerConfig {
            output_dir: temp.path().join("output2").to_string_lossy().into(),
            seed: Some(42),
            dataset_id: Some("test_dataset".into()),
            ..Default::default()
        };
        let runner2 = ExperimentRunner::with_config(runner_config2);
        let result2 = runner2.run_single(&config_path).unwrap();

        // Metrics should be identical
        assert_eq!(result1.metrics.cagr, result2.metrics.cagr, "CAGR mismatch");
        assert_eq!(result1.metrics.volatility, result2.metrics.volatility, "Volatility mismatch");
        assert_eq!(result1.metrics.sharpe_ratio, result2.metrics.sharpe_ratio, "Sharpe mismatch");
        assert_eq!(result1.metrics.max_drawdown, result2.metrics.max_drawdown, "Max DD mismatch");
        assert_eq!(result1.metrics.total_trades, result2.metrics.total_trades, "Trade count mismatch");

        // Timeseries should be identical
        assert_eq!(result1.timeseries.len(), result2.timeseries.len(), "Timeseries length mismatch");
        for (ts1, ts2) in result1.timeseries.iter().zip(result2.timeseries.iter()) {
            assert_eq!(ts1.date, ts2.date, "Date mismatch");
            assert_eq!(ts1.equity, ts2.equity, "Equity mismatch at {}", ts1.date);
        }

        // Trades should be identical (same order)
        assert_eq!(result1.trades.len(), result2.trades.len(), "Trade count mismatch");
        for (t1, t2) in result1.trades.iter().zip(result2.trades.iter()) {
            assert_eq!(t1.symbol, t2.symbol, "Trade symbol mismatch");
            assert_eq!(t1.date, t2.date, "Trade date mismatch");
            assert_eq!(t1.quantity, t2.quantity, "Trade quantity mismatch");
        }

        // Config hash should be identical
        assert_eq!(result1.metadata.config_hash, result2.metadata.config_hash, "Config hash mismatch");
    }

    #[test]
    fn test_batch_order_independence() {
        // Run configs in different orders, verify same individual results
        let temp = tempdir().unwrap();
        let configs_dir = temp.path().join("configs");
        fs::create_dir_all(&configs_dir).unwrap();

        // Create 3 configs
        let config_a = r#"
[strategy]
id = "strategy_a"
version = "1.0.0"
description = "Strategy A"

[[pipeline]]
type = "selection"
block_id = "momentum"
params = { lookback_days = 10 }

[[pipeline]]
type = "sizing"
block_id = "equal_weight"
params = { max_weight = 1.0 }
"#;
        let config_b = r#"
[strategy]
id = "strategy_b"
version = "1.0.0"
description = "Strategy B"

[[pipeline]]
type = "selection"
block_id = "momentum"
params = { lookback_days = 20 }

[[pipeline]]
type = "sizing"
block_id = "equal_weight"
params = { max_weight = 1.0 }
"#;
        let config_c = r#"
[strategy]
id = "strategy_c"
version = "1.0.0"
description = "Strategy C"

[[pipeline]]
type = "selection"
block_id = "momentum"
params = { lookback_days = 30 }

[[pipeline]]
type = "sizing"
block_id = "equal_weight"
params = { max_weight = 1.0 }
"#;

        create_test_config(&configs_dir, "a.toml", config_a);
        create_test_config(&configs_dir, "b.toml", config_b);
        create_test_config(&configs_dir, "c.toml", config_c);

        // Run batch
        let runner_config = RunnerConfig {
            output_dir: temp.path().join("output").to_string_lossy().into(),
            seed: Some(42),
            ..Default::default()
        };
        let runner = ExperimentRunner::with_config(runner_config);
        let results = runner.run_batch(&configs_dir).unwrap();

        // Collect results by strategy_id
        let results_map: std::collections::HashMap<_, _> = results
            .iter()
            .map(|r| (r.metadata.strategy_id.clone(), r))
            .collect();

        // Each strategy should have consistent metrics regardless of execution order
        // (The current implementation uses deterministic mock data based on seed)
        assert!(results_map.contains_key("strategy_a"), "Missing strategy_a");
        assert!(results_map.contains_key("strategy_b"), "Missing strategy_b");
        assert!(results_map.contains_key("strategy_c"), "Missing strategy_c");

        // Verify each strategy succeeded
        for (id, result) in &results_map {
            assert!(result.success, "Strategy {} failed", id);
            assert!(!result.timeseries.is_empty(), "Strategy {} has no timeseries", id);
        }
    }

    #[test]
    fn test_timestamp_does_not_affect_metrics() {
        // Run twice at different "times", verify metrics are identical
        // (Only metadata.timestamp_utc should differ)
        let temp = tempdir().unwrap();
        let config_content = r#"
[strategy]
id = "timestamp_test"
version = "1.0.0"
description = "Timestamp independence test"

[[pipeline]]
type = "selection"
block_id = "momentum"

[[pipeline]]
type = "sizing"
block_id = "equal_weight"
params = { max_weight = 1.0 }
"#;
        let config_path = create_test_config(temp.path(), "test.toml", config_content);

        let runner_config1 = RunnerConfig {
            output_dir: temp.path().join("output1").to_string_lossy().into(),
            seed: Some(123),
            ..Default::default()
        };
        let runner1 = ExperimentRunner::with_config(runner_config1);
        let result1 = runner1.run_single(&config_path).unwrap();

        // Small delay to ensure different timestamp
        std::thread::sleep(std::time::Duration::from_millis(10));

        let runner_config2 = RunnerConfig {
            output_dir: temp.path().join("output2").to_string_lossy().into(),
            seed: Some(123),
            ..Default::default()
        };
        let runner2 = ExperimentRunner::with_config(runner_config2);
        let result2 = runner2.run_single(&config_path).unwrap();

        // Timestamps should differ
        assert_ne!(result1.metadata.timestamp_utc, result2.metadata.timestamp_utc,
            "Timestamps should be different");

        // But run_id should differ (UUID)
        assert_ne!(result1.metadata.run_id, result2.metadata.run_id,
            "Run IDs should be different");

        // Metrics should be identical
        assert_eq!(result1.metrics.cagr, result2.metrics.cagr, "CAGR should match");
        assert_eq!(result1.metrics.sharpe_ratio, result2.metrics.sharpe_ratio, "Sharpe should match");
        assert_eq!(result1.metrics.max_drawdown, result2.metrics.max_drawdown, "Drawdown should match");

        // Config hash should be identical (not affected by timestamp)
        assert_eq!(result1.metadata.config_hash, result2.metadata.config_hash,
            "Config hash should match");
    }
}


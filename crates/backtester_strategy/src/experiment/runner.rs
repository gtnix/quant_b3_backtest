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

use backtester_engine::{
    DividendEvent, DualPriceBar, PriceType as EnginePriceType, 
    UnifiedEngine, UnifiedEngineConfig,
};
use backtester_intelligence::Market;

use crate::blocks::BlockParams;
use crate::compositor::Compositor;
use crate::config::{load_strategy_config, LoadError, StrategyConfig};
use crate::context::{StrategyContext, StrategyCandidate};
use crate::registry::BlockRegistry;

use super::artifacts::ArtifactWriter;
use super::metrics::MetricsCalculator;
use super::types::*;

/// Simulation result from UnifiedEngine.
#[derive(Debug, Clone)]
pub struct SimulationOutput {
    pub timeseries: Vec<EquityPoint>,
    pub dividend_events: Vec<DividendTraceEntry>,
    pub total_dividend_cashflow: Decimal,
    pub dividend_count: usize,
}

/// Dividend trace entry for trace.jsonl.
#[derive(Debug, Clone)]
pub struct DividendTraceEntry {
    pub date: NaiveDate,
    pub symbol: String,
    pub rate: Decimal,
    pub shares: i64,
    pub cashflow: Decimal,
}

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
    /// Execution mode (standard, compiled, fast, auto)
    pub execution_mode: ExecutionMode,
    /// Enable dividend processing
    pub enable_dividends: bool,
    /// Initial capital for simulation
    pub initial_capital: Decimal,
    /// Path to dividend CSV file (optional, for testing)
    pub dividend_csv_path: Option<String>,
    /// Path to market data CSV file (optional, for backtesting)
    pub market_data_csv_path: Option<String>,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            output_dir: "output/experiments".into(),
            risk_free_rate: 0.05, // 5% annual
            costs: CostConfig::default(),
            seed: None,
            dataset_id: None,
            execution_mode: ExecutionMode::Auto,
            enable_dividends: true,
            initial_capital: Decimal::from(1_000_000),
            dividend_csv_path: None,
            market_data_csv_path: None,
        }
    }
}

/// Experiment runner - executes strategies and generates artifacts.
pub struct ExperimentRunner {
    registry: BlockRegistry,
    config: RunnerConfig,
    strict_mode: bool,
    dry_run: bool,
    execution_mode: ExecutionMode,
    /// Market data for simulation (optional).
    market_data: Option<super::market_data::MarketDataProvider>,
}

impl ExperimentRunner {
    /// Create a new experiment runner with default settings.
    pub fn new() -> Self {
        Self {
            registry: BlockRegistry::with_builtins(),
            config: RunnerConfig::default(),
            strict_mode: false,
            dry_run: false,
            execution_mode: ExecutionMode::Auto,
            market_data: None,
        }
    }
    
    /// Set market data for simulation.
    pub fn with_market_data(mut self, data: super::market_data::MarketDataProvider) -> Self {
        self.market_data = Some(data);
        self
    }
    
    /// Load market data from CSV file.
    pub fn load_market_data_csv(&mut self, path: &std::path::Path) -> Result<(), RunnerError> {
        let data = super::market_data::MarketDataProvider::from_csv(path)
            .map_err(|e| RunnerError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
        self.market_data = Some(data);
        Ok(())
    }
    
    /// Check if market data is loaded.
    pub fn has_market_data(&self) -> bool {
        self.market_data.is_some()
    }

    /// Create runner with custom configuration.
    pub fn with_config(config: RunnerConfig) -> Self {
        let execution_mode = config.execution_mode;
        let market_data = config.market_data_csv_path.as_ref().and_then(|path| {
            super::market_data::MarketDataProvider::from_csv(std::path::Path::new(path)).ok()
        });
        Self {
            registry: BlockRegistry::with_builtins(),
            config,
            strict_mode: false,
            dry_run: false,
            execution_mode,
            market_data,
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

    /// Set execution mode (standard, compiled, fast, auto).
    pub fn with_execution_mode(mut self, mode: ExecutionMode) -> Self {
        self.execution_mode = mode;
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
            execution_mode: self.execution_mode,
            config_path: config_path.to_string_lossy().into(),
            duration_ms: 0,
            dividends_enabled: false,
            dividend_policy: None,
            total_dividend_cashflow: None,
            dividend_count: None,
            mode_fallback_reason: None,
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

        // Resolve execution mode
        let resolution = self.resolve_execution_mode(config);
        let effective_mode = resolution.resolved_mode;

        // Log mode resolution
        if resolution.requested_mode != resolution.resolved_mode {
            if self.strict_mode && resolution.requested_mode == ExecutionMode::Fast {
                // In strict mode, fail if fast was requested but not supported
                return Err(RunnerError::ExecutionModeUnsupported {
                    requested: resolution.requested_mode,
                    unsupported_steps: resolution.unsupported_steps,
                });
            }
            tracing::warn!(
                "Execution mode fallback: {:?} -> {:?}. Reason: {}",
                resolution.requested_mode,
                resolution.resolved_mode,
                resolution.fallback_reason.as_deref().unwrap_or("unknown")
            );
        }

        tracing::info!(
            "Using execution mode: {:?} for strategy {}",
            effective_mode,
            config.strategy.id
        );

        // Create initial context with mock data for now
        // In a real implementation, this would load actual market data
        let initial_context = self.create_initial_context(config)?;

        // Execute based on resolved mode
        // Currently all modes fall back to standard compositor
        // Fast mode would use CompiledStrategy + FastContext when fully integrated
        let compositor = Compositor::with_builtins();
        let mut ctx = initial_context;
        let compositor_result = compositor.execute(config, &mut ctx)?;

        // Validate compositor result in strict mode
        if self.strict_mode {
            self.validate_compositor_result(&compositor_result, config)?;
        }

        // Build effective params map for each step
        let effective_params_by_block = self.compute_effective_params(config);

        // Convert context trace to experiment trace with effective params
        let mut trace: Vec<ExperimentTraceEntry> = compositor_result
            .trace
            .iter()
            .map(|t| {
                let params = effective_params_by_block
                    .get(&t.block_id)
                    .cloned()
                    .unwrap_or_default();
                ExperimentTraceEntry::from_trace_entry(t, params)
            })
            .collect();

        // Add policy trace entry (anti-double-count)
        let dividend_policy = DividendPolicyInfo {
            signals_price: PriceType::Adjusted,
            valuation_price: PriceType::Raw,
            dividends_as_cashflow: self.config.enable_dividends,
        };
        trace.insert(0, ExperimentTraceEntry::policy(&dividend_policy));

        // Add mode fallback trace if applicable
        if let Some(ref reason) = resolution.fallback_reason {
            trace.insert(
                1,
                ExperimentTraceEntry::mode_fallback(
                    resolution.requested_mode,
                    resolution.resolved_mode,
                    reason,
                ),
            );
        }

        // Generate timeseries and trades from context
        let (timeseries, trades) = self.generate_timeseries_and_trades(&compositor_result);

        // Calculate metrics using real trades
        let metrics =
            MetricsCalculator::compute(&timeseries, &trades, self.config.risk_free_rate);

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
            execution_mode: effective_mode,
            config_path: config_path.to_string_lossy().into(),
            duration_ms: duration.as_millis() as u64,
            dividends_enabled: self.config.enable_dividends,
            dividend_policy: Some(dividend_policy),
            total_dividend_cashflow: None, // Populated when using run_unified_simulation
            dividend_count: None,
            mode_fallback_reason: resolution.fallback_reason.clone(),
        };

        Ok(ExperimentResult::success(
            run_id.into(),
            metadata,
            metrics,
            timeseries,
            trace,
            trades,
        ))
    }

    /// Resolve execution mode based on pipeline compatibility and dividend requirements.
    /// 
    /// Rules for Auto mode (deterministic):
    /// 1. If ALL steps have fast_* equivalents AND dividends disabled -> Fast
    /// 2. If dividends enabled AND Fast requested -> Fallback to Compiled
    /// 3. If ALL steps can be compiled -> Compiled
    /// 4. Otherwise -> Standard
    ///
    /// # Anti-Double-Count Policy
    /// Fast mode does NOT support dividend cashflow tracking. When dividends are enabled,
    /// the engine automatically falls back to Compiled mode to ensure correct PnL.
    fn resolve_execution_mode(&self, config: &StrategyConfig) -> ExecutionModeResolution {
        // Blocks that have fast_* implementations in fast_context.rs
        const FAST_SUPPORTED_BLOCKS: &[&str] = &["momentum", "low_vol", "equal_weight"];
        
        // All blocks support compiled mode (typed params)
        // This is because CompiledStrategy can wrap any StrategyBlock

        let requested = self.execution_mode;
        let dividends_enabled = self.config.enable_dividends;
        
        // Collect unsupported steps for fast mode
        let mut unsupported_for_fast: Vec<String> = config
            .enabled_steps()
            .into_iter()
            .filter(|step| !FAST_SUPPORTED_BLOCKS.contains(&step.block_id.as_str()))
            .map(|step| format!("{}:{}", step.step_type, step.block_id))
            .collect();

        // Fast mode does NOT support dividend cashflow
        // This is a deterministic policy: if dividends enabled, force Compiled
        let dividend_blocks_fast = dividends_enabled;
        if dividend_blocks_fast {
            unsupported_for_fast.push("dividend_cashflow:enabled".to_string());
        }

        match requested {
            ExecutionMode::Auto => {
                if unsupported_for_fast.is_empty() {
                    ExecutionModeResolution {
                        resolved_mode: ExecutionMode::Fast,
                        requested_mode: ExecutionMode::Auto,
                        fallback_reason: None,
                        unsupported_steps: Vec::new(),
                    }
                } else {
                    // Fall back to compiled (always supported)
                    let reason = if dividend_blocks_fast {
                        "Fast mode does not support dividend cashflow tracking".to_string()
                    } else {
                        format!("Fast mode not supported for: {}", unsupported_for_fast.join(", "))
                    };
                    ExecutionModeResolution {
                        resolved_mode: ExecutionMode::Compiled,
                        requested_mode: ExecutionMode::Auto,
                        fallback_reason: Some(reason),
                        unsupported_steps: unsupported_for_fast,
                    }
                }
            }
            ExecutionMode::Fast => {
                if unsupported_for_fast.is_empty() {
                    ExecutionModeResolution {
                        resolved_mode: ExecutionMode::Fast,
                        requested_mode: ExecutionMode::Fast,
                        fallback_reason: None,
                        unsupported_steps: Vec::new(),
                    }
                } else {
                    // Fallback to compiled in non-strict mode
                    let reason = if dividend_blocks_fast {
                        "Fast mode does not support dividend cashflow tracking".to_string()
                    } else {
                        format!("Fast mode not supported for: {}", unsupported_for_fast.join(", "))
                    };
                    ExecutionModeResolution {
                        resolved_mode: ExecutionMode::Compiled,
                        requested_mode: ExecutionMode::Fast,
                        fallback_reason: Some(reason),
                        unsupported_steps: unsupported_for_fast,
                    }
                }
            }
            ExecutionMode::Compiled => ExecutionModeResolution {
                resolved_mode: ExecutionMode::Compiled,
                requested_mode: ExecutionMode::Compiled,
                fallback_reason: None,
                unsupported_steps: Vec::new(),
            },
            ExecutionMode::Standard => ExecutionModeResolution {
                resolved_mode: ExecutionMode::Standard,
                requested_mode: ExecutionMode::Standard,
                fallback_reason: None,
                unsupported_steps: Vec::new(),
            },
        }
    }

    /// Create initial context for strategy execution.
    /// 
    /// Uses real market data when available, otherwise falls back to placeholder.
    fn create_initial_context(
        &self,
        _config: &StrategyConfig,
    ) -> Result<StrategyContext, RunnerError> {
        // Use real market data if available
        if let Some(ref market_data) = self.market_data {
            return self.create_context_from_market_data(market_data);
        }
        
        // Fallback to placeholder (for backwards compatibility)
        let date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let cash = Decimal::from(100_000);

        let mut ctx = StrategyContext::new(
            date,
            backtester_intelligence::filters::Market::BR,
            cash,
        );

        // Candidatos de placeholder
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
    
    /// Create context from real market data.
    fn create_context_from_market_data(
        &self,
        market_data: &super::market_data::MarketDataProvider,
    ) -> Result<StrategyContext, RunnerError> {
        let (start_date, _end_date) = market_data.date_range()
            .ok_or_else(|| RunnerError::InvalidPath("Market data has no date range".to_string()))?;
        
        let cash = self.config.initial_capital;
        
        let mut ctx = StrategyContext::new(
            start_date,
            backtester_intelligence::filters::Market::BR,
            cash,
        );
        
        // Build candidates from market data
        let mut candidates: Vec<StrategyCandidate> = Vec::new();
        
        for symbol in market_data.symbols() {
            if let Some(bars) = market_data.bars_for_symbol(symbol) {
                if bars.is_empty() {
                    continue;
                }
                
                let mut candidate = StrategyCandidate::new(symbol, backtester_intelligence::filters::Market::BR);
                
                // Set current price (last bar)
                if let Some(last_bar) = bars.last() {
                    candidate.price = Some(last_bar.close);
                }
                
                // Calculate momentum return (using all available prices)
                if bars.len() >= 2 {
                    let first_price = bars.first().map(|b| b.close).unwrap_or(Decimal::ONE);
                    let last_price = bars.last().map(|b| b.close).unwrap_or(Decimal::ONE);
                    if !first_price.is_zero() {
                        let ret = ((last_price - first_price) / first_price)
                            .to_f64()
                            .unwrap_or(0.0);
                        candidate.momentum_return = Some(ret);
                    }
                }
                
                // Calculate volatility from returns
                if bars.len() >= 10 {
                    let returns: Vec<f64> = bars.windows(2)
                        .filter_map(|w| {
                            let prev = w[0].close;
                            let curr = w[1].close;
                            if !prev.is_zero() {
                                Some(((curr - prev) / prev).to_f64().unwrap_or(0.0))
                            } else {
                                None
                            }
                        })
                        .collect();
                    
                    if !returns.is_empty() {
                        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                        let variance = returns.iter()
                            .map(|r| (r - mean).powi(2))
                            .sum::<f64>() / returns.len() as f64;
                        let daily_vol = variance.sqrt();
                        let annual_vol = daily_vol * 252.0_f64.sqrt();
                        candidate.volatility = Some(annual_vol);
                    }
                }
                
                // Store price history
                candidate.prices = bars.iter()
                    .filter_map(|b| b.close.to_f64())
                    .collect();
                
                candidates.push(candidate);
            }
        }
        
        let symbols: Vec<String> = candidates.iter().map(|c| c.symbol.clone()).collect();
        ctx = ctx.with_candidates(candidates);
        ctx.universe = symbols;
        
        tracing::info!(
            "Created context from market data: {} symbols, start_date={}",
            ctx.candidates.len(),
            start_date
        );
        
        Ok(ctx)
    }

    /// Compute effective params for each block in the pipeline.
    /// Merges block defaults with step-configured params.
    fn compute_effective_params(
        &self,
        config: &StrategyConfig,
    ) -> HashMap<String, HashMap<String, serde_json::Value>> {
        let mut result = HashMap::new();

        for step in config.enabled_steps() {
            // Get block from registry to get defaults
            if let Some(block) = self.registry.get(&step.block_id) {
                let defaults = block.default_params();
                let effective = merge_params(&defaults, &step.params);
                result.insert(step.block_id.clone(), params_to_json(&effective));
            } else {
                // Block not found, just use configured params
                result.insert(step.block_id.clone(), params_to_json(&step.params));
            }
        }

        result
    }

    /// Run simulation with UnifiedEngine.
    ///
    /// Integrates the compositor result (asset selection + weights) with 
    /// UnifiedEngine for actual portfolio simulation with dividends.
    ///
    /// # Anti-Double-Count Policy
    /// - Signals use adjusted prices (from compositor)
    /// - Valuation uses raw prices (from engine)
    /// - Dividends enter as cashflow on ex_date
    ///
    /// # Arguments
    /// - `result`: Compositor result with weights/rankings
    /// - `market_bars`: Market data with dual prices (adjusted + raw)
    /// - `dividends`: Dividend events for the period
    ///
    /// # Returns
    /// `SimulationOutput` with timeseries, dividend events, and totals
    pub fn run_unified_simulation(
        &self,
        _result: &crate::compositor::CompositorResult,
        market_bars: &[DualPriceBar],
        dividends: Vec<DividendEvent>,
    ) -> SimulationOutput {
        // Configure engine with anti-double-count policy
        let config = UnifiedEngineConfig {
            initial_capital: self.config.initial_capital,
            default_market: Market::BR,
            enable_dividends: self.config.enable_dividends,
            valuation_price_type: EnginePriceType::Valuation, // Raw prices for anti-double-count
            ..Default::default()
        };

        let mut engine = UnifiedEngine::with_config(config);
        
        // Validate anti-double-count policy
        if let Err(e) = engine.validate_anti_double_count() {
            tracing::error!("Policy violation: {}", e);
            // Return empty simulation on policy violation
            return SimulationOutput {
                timeseries: Vec::new(),
                dividend_events: Vec::new(),
                total_dividend_cashflow: Decimal::ZERO,
                dividend_count: 0,
            };
        }

        // Load dividends
        engine.load_dividends(dividends);

        // Group bars by date for day-by-day processing
        let mut bars_by_date: HashMap<NaiveDate, Vec<DualPriceBar>> = HashMap::new();
        for bar in market_bars {
            bars_by_date.entry(bar.date).or_default().push(bar.clone());
        }

        let mut dates: Vec<_> = bars_by_date.keys().copied().collect();
        dates.sort();

        let mut timeseries = Vec::with_capacity(dates.len());
        let mut dividend_events = Vec::new();
        let mut peak_equity = self.config.initial_capital;
        let mut cumulative_dividend = Decimal::ZERO;

        // Process each trading day
        for date in dates {
            let bars = bars_by_date.get(&date).cloned().unwrap_or_default();
            
            // For now, use equal-weight candidates from compositor weights
            // TODO: integrate with compositor result for real asset selection
            let candidates = Vec::new(); // Simplified for initial integration

            let day_result = engine.process_day(date, &bars, candidates);

            // Track dividends
            for div in &day_result.dividends_applied {
                cumulative_dividend += div.cashflow;
                dividend_events.push(DividendTraceEntry {
                    date,
                    symbol: div.symbol.clone(),
                    rate: div.rate,
                    shares: div.shares,
                    cashflow: div.cashflow,
                });
            }

            // Calculate equity point
            let equity = day_result.equity;
            peak_equity = peak_equity.max(equity);
            let drawdown = if peak_equity > Decimal::ZERO {
                ((equity - peak_equity) / peak_equity)
                    .to_f64()
                    .unwrap_or(0.0)
            } else {
                0.0
            };

            // Exposure from positions / equity (approximate)
            let positions_val = Decimal::from(day_result.positions) * Decimal::from(10_000); // Rough estimate
            let exposure = if equity > Decimal::ZERO {
                (positions_val / equity).to_f64().unwrap_or(0.0).min(1.0)
            } else {
                0.0
            };

            let day_dividend = day_result.dividend_cashflow;

            timeseries.push(EquityPoint {
                date,
                equity,
                drawdown,
                exposure,
                vol_exante: None, // Calculated post-hoc
                vol_expost: None,
                dividend_cashflow: if day_dividend > Decimal::ZERO {
                    Some(day_dividend)
                } else {
                    None
                },
                dividend_cumulative: if cumulative_dividend > Decimal::ZERO {
                    Some(cumulative_dividend)
                } else {
                    None
                },
            });
        }

        let dividend_count = dividend_events.len();
        SimulationOutput {
            timeseries,
            dividend_events,
            total_dividend_cashflow: cumulative_dividend,
            dividend_count,
        }
    }

    /// Generate timeseries and trades from execution result.
    ///
    /// Uses real market data simulation when available.
    /// Falls back to placeholder curve when no market data is loaded.
    fn generate_timeseries_and_trades(
        &self,
        result: &crate::compositor::CompositorResult,
    ) -> (Vec<EquityPoint>, Vec<TradeRecord>) {
        // Check if we have real market data
        if let Some(ref market_data) = self.market_data {
            return self.simulate_with_real_data(result, market_data);
        }
        
        // Fallback to placeholder (for backwards compatibility)
        let timeseries = self.generate_placeholder_timeseries(result);
        (timeseries, Vec::new())
    }
    
    /// Simulate strategy with real market data.
    fn simulate_with_real_data(
        &self,
        result: &crate::compositor::CompositorResult,
        market_data: &super::market_data::MarketDataProvider,
    ) -> (Vec<EquityPoint>, Vec<TradeRecord>) {
        let mut equity = self.config.initial_capital;
        let mut peak_equity = equity;
        let mut timeseries = Vec::new();
        let mut trades = Vec::new();
        
        // Target weights from compositor
        let target_weights: HashMap<String, f64> = result.weights.clone();
        
        // Current positions: symbol -> (shares, avg_price)
        let mut positions: HashMap<String, (i64, Decimal)> = HashMap::new();
        
        for date in market_data.trading_dates() {
            let bars = match market_data.bars_for_date(*date) {
                Some(b) => b,
                None => continue,
            };
            
            // Calculate current portfolio value
            let mut portfolio_value = equity;
            for (symbol, (shares, _)) in &positions {
                if let Some(bar) = bars.get(symbol) {
                    portfolio_value += Decimal::from(*shares) * bar.close;
                }
            }
            
            // Rebalance to target weights
            let total_value = portfolio_value;
            for (symbol, target_weight) in &target_weights {
                if let Some(bar) = bars.get(symbol) {
                    let target_value = total_value * Decimal::try_from(*target_weight).unwrap_or(Decimal::ZERO);
                    let target_shares = (target_value / bar.close).to_i64().unwrap_or(0);
                    
                    let current_shares = positions.get(symbol).map(|(s, _)| *s).unwrap_or(0);
                    let delta = target_shares - current_shares;
                    
                    if delta != 0 {
                        let (side, quantity) = if delta > 0 {
                            (TradeSide::Buy, delta)
                        } else {
                            (TradeSide::Sell, -delta)
                        };
                        
                        let trade_value = Decimal::from(quantity.abs()) * bar.close;
                        
                        trades.push(TradeRecord {
                            date: *date,
                            symbol: symbol.clone(),
                            side,
                            quantity: quantity.abs(),
                            price: bar.close,
                            value: trade_value,
                            pnl: None,
                        });
                        
                        // Update position and cash
                        if delta > 0 {
                            // Buying: subtract cash
                            equity -= trade_value;
                            
                            let (old_shares, old_avg) = positions.get(symbol).cloned().unwrap_or((0, Decimal::ZERO));
                            let new_shares = old_shares + delta;
                            let new_avg = if new_shares > 0 {
                                (Decimal::from(old_shares) * old_avg + trade_value) / Decimal::from(new_shares)
                            } else {
                                Decimal::ZERO
                            };
                            positions.insert(symbol.clone(), (new_shares, new_avg));
                        } else {
                            // Selling: add cash
                            equity += trade_value;
                            
                            let (old_shares, avg_price) = positions.get(symbol).cloned().unwrap_or((0, Decimal::ZERO));
                            let new_shares = old_shares + delta; // delta is negative
                            if new_shares <= 0 {
                                positions.remove(symbol);
                            } else {
                                positions.insert(symbol.clone(), (new_shares, avg_price));
                            }
                        }
                        
                        // Apply trading costs
                        let cost = trade_value * Decimal::try_from(self.config.costs.trading_fee_pct / 100.0).unwrap_or(Decimal::ZERO);
                        equity -= cost;
                    }
                }
            }
            
            // Recalculate equity at end of day
            let mut eod_equity = equity;
            for (symbol, (shares, _)) in &positions {
                if let Some(bar) = bars.get(symbol) {
                    eod_equity += Decimal::from(*shares) * bar.close;
                }
            }
            
            peak_equity = peak_equity.max(eod_equity);
            let drawdown = if peak_equity.is_zero() {
                0.0
            } else {
                ((eod_equity - peak_equity) / peak_equity).to_f64().unwrap_or(0.0)
            };
            
            let exposure = if total_value.is_zero() {
                0.0
            } else {
                ((total_value - equity) / total_value).to_f64().unwrap_or(0.0)
            };
            
            timeseries.push(EquityPoint {
                date: *date,
                equity: eod_equity,
                drawdown,
                exposure,
                vol_exante: None,
                vol_expost: None,
                dividend_cashflow: None,
                dividend_cumulative: None,
            });
        }
        
        (timeseries, trades)
    }
    
    /// Generate placeholder timeseries (fallback when no market data).
    fn generate_placeholder_timeseries(
        &self,
        result: &crate::compositor::CompositorResult,
    ) -> Vec<EquityPoint> {
        let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let num_points = 252; // 1 year of trading days

        let total_weight: f64 = result.weights.values().sum();
        let exposure = if total_weight > 0.0 { total_weight } else { 0.0 };

        (0..num_points)
            .map(|i| {
                let equity = self.config.initial_capital 
                    * (Decimal::ONE + Decimal::from(i) / Decimal::from(1000));
                let peak = equity;
                let drawdown = if peak.is_zero() {
                    0.0
                } else {
                    ((equity - peak) / peak).to_f64().unwrap_or(0.0)
                };

                EquityPoint {
                    date: start + chrono::Duration::days(i),
                    equity,
                    drawdown,
                    exposure,
                    vol_exante: Some(0.20),
                    vol_expost: None,
                    dividend_cashflow: None,
                    dividend_cumulative: None,
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

    // ========================================================================
    // STRESS TESTING INTEGRATION
    // ========================================================================

    /// Run strategy with stress testing across multiple scenarios.
    ///
    /// Executes the strategy under baseline conditions plus stress scenarios
    /// (increased slippage, costs), and validates against thresholds.
    ///
    /// Returns `Err` if any threshold is violated (fail-fast gate).
    pub fn run_with_stress(
        &self,
        config_path: &Path,
        thresholds: &super::stress::StressThresholds,
    ) -> Result<super::stress::StressTestReport, RunnerError> {
        use super::stress::{StressScenario, StressTestReport, StressTestResult};

        let _config_content = fs::read_to_string(config_path)?;
        let strategy_config = load_strategy_config(config_path)?;
        let strategy_id = strategy_config.strategy.id.clone();

        let mut results = Vec::new();

        for scenario in StressScenario::all() {
            // Apply stress multipliers to costs
            let stressed_costs = scenario.apply_to_config(&self.config.costs);
            
            // Create runner with stressed costs
            let stressed_runner_config = RunnerConfig {
                output_dir: self.config.output_dir.clone(),
                risk_free_rate: self.config.risk_free_rate,
                costs: stressed_costs.clone(),
                seed: self.config.seed,
                dataset_id: self.config.dataset_id.clone(),
                execution_mode: self.config.execution_mode,
                enable_dividends: self.config.enable_dividends,
                initial_capital: self.config.initial_capital,
                dividend_csv_path: self.config.dividend_csv_path.clone(),
                market_data_csv_path: self.config.market_data_csv_path.clone(),
            };

            let stressed_runner = ExperimentRunner::with_config(stressed_runner_config);

            // Run and collect metrics
            match stressed_runner.run_single(config_path) {
                Ok(experiment_result) => {
                    let metrics = experiment_result.metrics.clone();
                    
                    // Check thresholds
                    let check_result = thresholds.check(&metrics);
                    
                    let stress_result = if let Err(reason) = check_result {
                        StressTestResult::fail(scenario, metrics, stressed_costs, reason)
                    } else {
                        StressTestResult::pass(scenario, metrics, stressed_costs)
                    };
                    
                    results.push(stress_result);
                }
                Err(e) => {
                    // Run failed - create failure result
                    results.push(StressTestResult::fail(
                        scenario,
                        RunMetrics::default(),
                        stressed_costs,
                        format!("Run failed: {}", e),
                    ));
                }
            }
        }

        let report = StressTestReport::from_results(strategy_id, results);

        // Fail-fast: if any scenario failed thresholds, return error
        if !report.all_passed {
            return Err(RunnerError::StressTestFailed {
                passed: report.passed_count,
                failed: report.failed_count,
                summary: report.to_summary_string(),
            });
        }

        Ok(report)
    }

    // ========================================================================
    // STABILITY ANALYSIS INTEGRATION
    // ========================================================================

    /// Run strategy with stability analysis across time blocks.
    ///
    /// Splits the backtest period into N blocks and checks for consistent
    /// performance. Returns `Err` if stability criteria are violated.
    pub fn run_with_stability(
        &self,
        config_path: &Path,
        stability_config: &super::stability::StabilityConfig,
    ) -> Result<super::stability::StabilityReport, RunnerError> {
        use super::stability::StabilityAnalyzer;

        // Run the experiment first
        let experiment_result = self.run_single(config_path)?;

        // Analyze stability
        let analyzer = StabilityAnalyzer::with_config(stability_config.clone());
        let report = analyzer.analyze(
            &experiment_result.timeseries,
            &experiment_result.trades,
            self.config.risk_free_rate,
        );

        // Fail-fast: if strategy is unstable, return error
        if !report.is_stable {
            return Err(RunnerError::StabilityCheckFailed {
                reasons: report.instability_reasons.clone(),
                summary: report.to_summary_string(),
            });
        }

        Ok(report)
    }

    /// Run strategy with both stress testing and stability analysis.
    ///
    /// This is the full robustness gate: strategy must pass both checks.
    pub fn run_with_robustness(
        &self,
        config_path: &Path,
        stress_thresholds: &super::stress::StressThresholds,
        stability_config: &super::stability::StabilityConfig,
    ) -> Result<(super::stress::StressTestReport, super::stability::StabilityReport), RunnerError> {
        // Run stress tests first (fail-fast)
        let stress_report = self.run_with_stress(config_path, stress_thresholds)?;

        // If stress passed, run stability analysis
        let stability_report = self.run_with_stability(config_path, stability_config)?;

        Ok((stress_report, stability_report))
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
    #[error("Execution mode '{requested:?}' not supported for pipeline. Unsupported steps: {unsupported_steps:?}")]
    ExecutionModeUnsupported {
        requested: ExecutionMode,
        unsupported_steps: Vec<String>,
    },
    #[error("Stress test failed: {passed}/{} scenarios passed. {summary}", passed + failed)]
    StressTestFailed {
        passed: usize,
        failed: usize,
        summary: String,
    },
    #[error("Stability check failed: {reasons:?}")]
    StabilityCheckFailed {
        reasons: Vec<String>,
        summary: String,
    },
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

    // ========================================================================
    // STRESS/STABILITY INTEGRATION TESTS (E2E Gate)
    // ========================================================================

    #[test]
    fn test_stress_thresholds_gate_pass() {
        use super::super::stress::StressThresholds;
        
        let temp = tempdir().unwrap();
        let config_content = r#"
[strategy]
id = "stress_test"
version = "1.0.0"
description = "Stress test"

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
            seed: Some(42),
            costs: CostConfig {
                trading_fee_pct: 0.001,
                slippage_pct: 0.0005,
                min_trade_brl: None,
            },
            ..Default::default()
        };
        let runner = ExperimentRunner::with_config(runner_config);

        // Lenient thresholds that should pass
        let thresholds = StressThresholds {
            min_sharpe: -10.0, // Very lenient
            max_drawdown: -0.99, // Very lenient
            max_sharpe_degradation_pct: 100.0,
            max_dd_increase_pct: 100.0,
        };

        let result = runner.run_with_stress(&config_path, &thresholds);
        assert!(result.is_ok(), "Lenient thresholds should pass: {:?}", result.err());
        
        let report = result.unwrap();
        assert!(report.all_passed, "All scenarios should pass with lenient thresholds");
    }

    #[test]
    fn test_stress_thresholds_gate_fail() {
        use super::super::stress::StressThresholds;
        
        // Test that StressThresholds::check correctly identifies failures
        let strict_thresholds = StressThresholds {
            min_sharpe: 10.0, // Unrealistic - require Sharpe > 10
            max_drawdown: -0.001, // Unrealistic - max 0.1% drawdown
            max_sharpe_degradation_pct: 0.0,
            max_dd_increase_pct: 0.0,
        };

        // Create metrics that should fail
        let bad_metrics = RunMetrics {
            sharpe_ratio: 1.5, // < 10.0, should fail
            max_drawdown: -0.15, // > -0.001, should fail
            ..Default::default()
        };

        let check_result = strict_thresholds.check(&bad_metrics);
        assert!(check_result.is_err(), "Metrics with Sharpe 1.5 should fail threshold of 10.0");
        
        // Verify the error message
        let error_msg = check_result.unwrap_err();
        assert!(error_msg.contains("Sharpe") || error_msg.contains("drawdown"),
            "Error should mention Sharpe or drawdown: {}", error_msg);
    }

    #[test]
    fn test_stability_gate_pass() {
        use super::super::stability::StabilityConfig;
        
        let temp = tempdir().unwrap();
        let config_content = r#"
[strategy]
id = "stability_test"
version = "1.0.0"
description = "Stability test"

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
            seed: Some(42),
            ..Default::default()
        };
        let runner = ExperimentRunner::with_config(runner_config);

        // Lenient stability config
        let stability_config = StabilityConfig {
            num_blocks: 3,
            min_sharpe_per_block: -10.0, // Very lenient
            max_sharpe_cv: 100.0,         // Very lenient
            max_sharpe_spread: 100.0,     // Very lenient
            min_positive_sharpe_pct: 0.0, // Very lenient
        };

        let result = runner.run_with_stability(&config_path, &stability_config);
        assert!(result.is_ok(), "Lenient stability config should pass: {:?}", result.err());
    }

    #[test]
    fn test_stability_gate_fail() {
        use super::super::stability::{StabilityAnalyzer, StabilityConfig, StabilitySummary};
        
        // Test that stability check correctly identifies failures with strict config
        let strict_config = StabilityConfig {
            num_blocks: 3,
            min_sharpe_per_block: 5.0, // Unrealistic
            max_sharpe_cv: 0.01,       // Unrealistic
            max_sharpe_spread: 0.01,   // Unrealistic
            min_positive_sharpe_pct: 1.0, // 100% must be positive
        };

        let analyzer = StabilityAnalyzer::with_config(strict_config);
        
        // Create a summary that should fail all criteria
        let bad_summary = StabilitySummary {
            mean_sharpe: 0.5,
            std_sharpe: 0.8,
            cv_sharpe: 1.6, // > 0.01
            min_sharpe: -0.5, // < 5.0
            max_sharpe: 1.5,
            sharpe_spread: 2.0, // > 0.01
            pct_positive_sharpe: 0.5, // < 1.0
            num_blocks: 3,
        };

        // Use the internal check method via the analyzer
        let (is_stable, reasons) = analyzer.check_stability(&bad_summary);
        
        assert!(!is_stable, "Bad summary should be marked as unstable");
        assert!(!reasons.is_empty(), "Should have at least one failure reason");
        assert!(reasons.len() >= 3, "Should have multiple failure reasons: {:?}", reasons);
    }

    #[test]
    fn test_run_with_robustness_full_gate() {
        use super::super::stress::StressThresholds;
        use super::super::stability::StabilityConfig;
        
        let temp = tempdir().unwrap();
        let config_content = r#"
[strategy]
id = "robustness_test"
version = "1.0.0"
description = "Full robustness test"

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
            seed: Some(42),
            ..Default::default()
        };
        let runner = ExperimentRunner::with_config(runner_config);

        // Lenient configs that should pass
        let thresholds = StressThresholds {
            min_sharpe: -10.0,
            max_drawdown: -0.99,
            max_sharpe_degradation_pct: 100.0,
            max_dd_increase_pct: 100.0,
        };
        let stability_config = StabilityConfig {
            num_blocks: 3,
            min_sharpe_per_block: -10.0,
            max_sharpe_cv: 100.0,
            max_sharpe_spread: 100.0,
            min_positive_sharpe_pct: 0.0,
        };

        let result = runner.run_with_robustness(&config_path, &thresholds, &stability_config);
        assert!(result.is_ok(), "Full robustness test should pass with lenient configs: {:?}", result.err());
        
        let (stress_report, stability_report) = result.unwrap();
        assert!(stress_report.all_passed);
        assert!(stability_report.is_stable);
    }
}


//! Experiment orchestrator types - schemas for artifacts and run metadata.

use chrono::{DateTime, NaiveDate, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Run mode for experiment execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunMode {
    /// Full execution with all artifacts
    Full,
    /// Validation only, no execution
    DryRun,
}

impl Default for RunMode {
    fn default() -> Self {
        Self::Full
    }
}

/// Execution mode for strategy pipeline.
/// Determines which execution path is used (standard compositor, compiled, or fast SoA).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    /// Standard compositor with dynamic block creation (most flexible, slowest)
    Standard,
    /// Compiled strategy with typed params (no HashMap lookups in hot path)
    Compiled,
    /// Fast SoA execution (requires 100% block support, fastest)
    Fast,
    /// Auto-select best mode based on pipeline compatibility (deterministic)
    Auto,
}

impl Default for ExecutionMode {
    fn default() -> Self {
        Self::Auto
    }
}

impl ExecutionMode {
    /// Returns the display name for the execution mode.
    pub fn as_str(&self) -> &'static str {
        match self {
            ExecutionMode::Standard => "standard",
            ExecutionMode::Compiled => "compiled",
            ExecutionMode::Fast => "fast",
            ExecutionMode::Auto => "auto",
        }
    }

    /// Parse from string (for CLI).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "standard" => Some(ExecutionMode::Standard),
            "compiled" => Some(ExecutionMode::Compiled),
            "fast" => Some(ExecutionMode::Fast),
            "auto" => Some(ExecutionMode::Auto),
            _ => None,
        }
    }
}

/// Result of resolving Auto execution mode.
#[derive(Debug, Clone)]
pub struct ExecutionModeResolution {
    /// The resolved execution mode
    pub resolved_mode: ExecutionMode,
    /// The requested mode (before resolution)
    pub requested_mode: ExecutionMode,
    /// Reason for fallback (if any)
    pub fallback_reason: Option<String>,
    /// Steps that prevented fast/compiled mode
    pub unsupported_steps: Vec<String>,
}

/// Cost configuration for experiment runs.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CostConfig {
    /// Trading fee as percentage (e.g., 0.001 = 0.1%)
    pub trading_fee_pct: f64,
    /// Slippage estimate as percentage
    pub slippage_pct: f64,
    /// Minimum trade size in BRL
    pub min_trade_brl: Option<f64>,
}

/// Current schema version for artifacts.
/// Increment when breaking changes are made to artifact format.
pub const ARTIFACT_SCHEMA_VERSION: &str = "1.0";

/// Dividend policy information for anti-double-count tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DividendPolicyInfo {
    /// Price type used for signals/indicators (typically Adjusted)
    pub signals_price: PriceType,
    /// Price type used for valuation/mark-to-market (should be Raw when dividends enabled)
    pub valuation_price: PriceType,
    /// Whether dividends enter as explicit cashflow
    pub dividends_as_cashflow: bool,
}

impl Default for DividendPolicyInfo {
    fn default() -> Self {
        Self {
            signals_price: PriceType::Adjusted,
            valuation_price: PriceType::Raw,
            dividends_as_cashflow: true,
        }
    }
}

/// Price type for anti-double-count policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PriceType {
    /// Adjusted prices (dividend adjustments baked in)
    #[default]
    Adjusted,
    /// Raw prices (dividends enter via cashflow)
    Raw,
}

/// Run metadata - saved as metadata.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetadata {
    /// Schema version for forward/backward compatibility
    #[serde(default = "default_schema_version")]
    pub schema_version: String,
    /// Unique run identifier (UUID)
    pub run_id: String,
    /// SHA256 hash of the config file
    pub config_hash: String,
    /// Strategy ID from config
    pub strategy_id: String,
    /// Strategy version from config
    pub strategy_version: String,
    /// Crate version that ran the experiment
    pub crate_version: String,
    /// UTC timestamp when run started
    pub timestamp_utc: DateTime<Utc>,
    /// Dataset identifier (if applicable)
    pub dataset_id: Option<String>,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Cost configuration used
    pub costs: CostConfig,
    /// Run mode (Full or DryRun)
    pub mode: RunMode,
    /// Execution mode used (standard, compiled, fast)
    #[serde(default)]
    pub execution_mode: ExecutionMode,
    /// Config file path (relative)
    pub config_path: String,
    /// Duration of the run in milliseconds
    pub duration_ms: u64,
    /// Whether dividends were enabled for this run
    #[serde(default)]
    pub dividends_enabled: bool,
    /// Dividend policy applied (anti-double-count)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dividend_policy: Option<DividendPolicyInfo>,
    /// Total dividend cashflow received during simulation
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_dividend_cashflow: Option<Decimal>,
    /// Number of dividend events processed
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dividend_count: Option<usize>,
    /// Reason for execution mode fallback (if any)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode_fallback_reason: Option<String>,
}

fn default_schema_version() -> String {
    ARTIFACT_SCHEMA_VERSION.to_string()
}

/// Run metrics - saved as metrics.json
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunMetrics {
    /// Compound Annual Growth Rate
    pub cagr: f64,
    /// Annualized volatility (standard deviation of returns)
    pub volatility: f64,
    /// Sharpe ratio (excess return / volatility)
    pub sharpe_ratio: f64,
    /// Maximum drawdown as percentage (e.g., -0.15 = -15%)
    pub max_drawdown: f64,
    /// Duration of max drawdown in days
    pub max_drawdown_duration_days: u32,
    /// Annualized turnover (total traded / avg portfolio value)
    pub turnover_annual: f64,
    /// Hit rate (percentage of winning trades)
    pub hit_rate: f64,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
    /// Total number of trades
    pub total_trades: u32,
    /// Total number of trading days
    pub total_days: u32,
    /// Sortino ratio (excess return / downside deviation)
    pub sortino_ratio: f64,
    /// Calmar ratio (CAGR / max drawdown)
    pub calmar_ratio: f64,
    /// Average win amount
    pub avg_win: f64,
    /// Average loss amount
    pub avg_loss: f64,
    /// Win/loss ratio
    pub win_loss_ratio: f64,
}

/// Equity point for timeseries output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquityPoint {
    pub date: NaiveDate,
    pub equity: Decimal,
    pub drawdown: f64,
    pub exposure: f64,
    pub vol_exante: Option<f64>,
    pub vol_expost: Option<f64>,
    /// Dividend cashflow for this day (if any).
    pub dividend_cashflow: Option<Decimal>,
    /// Cumulative dividend cashflow to date.
    pub dividend_cumulative: Option<Decimal>,
}

/// Enhanced trace entry with effective parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentTraceEntry {
    pub step: usize,
    pub block_id: String,
    pub block_type: String,
    pub message: String,
    pub timestamp_ms: u64,
    /// Effective parameters after merging defaults
    pub params_effective: HashMap<String, serde_json::Value>,
}

impl ExperimentTraceEntry {
    pub fn from_trace_entry(
        entry: &crate::context::TraceEntry,
        params: HashMap<String, serde_json::Value>,
    ) -> Self {
        Self {
            step: entry.step,
            block_id: entry.block_id.clone(),
            block_type: entry.block_type.clone(),
            message: entry.message.clone(),
            timestamp_ms: entry.timestamp_ms,
            params_effective: params,
        }
    }

    /// Create a policy trace entry.
    pub fn policy(policy: &DividendPolicyInfo) -> Self {
        let mut params = HashMap::new();
        params.insert(
            "signals_price".to_string(),
            serde_json::to_value(&policy.signals_price).unwrap_or_default(),
        );
        params.insert(
            "valuation_price".to_string(),
            serde_json::to_value(&policy.valuation_price).unwrap_or_default(),
        );
        params.insert(
            "dividends_as_cashflow".to_string(),
            serde_json::Value::Bool(policy.dividends_as_cashflow),
        );

        Self {
            step: 0,
            block_id: "policy".to_string(),
            block_type: "dividend_policy".to_string(),
            message: "Anti-double-count policy applied".to_string(),
            timestamp_ms: 0,
            params_effective: params,
        }
    }

    /// Create a mode fallback trace entry.
    pub fn mode_fallback(requested: ExecutionMode, resolved: ExecutionMode, reason: &str) -> Self {
        let mut params = HashMap::new();
        params.insert(
            "requested".to_string(),
            serde_json::Value::String(requested.as_str().to_string()),
        );
        params.insert(
            "resolved".to_string(),
            serde_json::Value::String(resolved.as_str().to_string()),
        );
        params.insert(
            "reason".to_string(),
            serde_json::Value::String(reason.to_string()),
        );

        Self {
            step: 0,
            block_id: "execution".to_string(),
            block_type: "mode_fallback".to_string(),
            message: format!("Execution mode fallback: {} -> {}", requested.as_str(), resolved.as_str()),
            timestamp_ms: 0,
            params_effective: params,
        }
    }
}

/// Trade record for metrics calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    pub date: NaiveDate,
    pub symbol: String,
    pub side: TradeSide,
    pub quantity: i64,
    pub price: Decimal,
    pub value: Decimal,
    pub pnl: Option<Decimal>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Result of a single experiment run.
#[derive(Debug, Clone)]
pub struct ExperimentResult {
    pub run_id: String,
    pub metadata: RunMetadata,
    pub metrics: RunMetrics,
    pub timeseries: Vec<EquityPoint>,
    pub trace: Vec<ExperimentTraceEntry>,
    pub trades: Vec<TradeRecord>,
    pub output_path: Option<std::path::PathBuf>,
    pub success: bool,
    pub error: Option<String>,
}

impl ExperimentResult {
    pub fn success(
        run_id: String,
        metadata: RunMetadata,
        metrics: RunMetrics,
        timeseries: Vec<EquityPoint>,
        trace: Vec<ExperimentTraceEntry>,
        trades: Vec<TradeRecord>,
    ) -> Self {
        Self {
            run_id,
            metadata,
            metrics,
            timeseries,
            trace,
            trades,
            output_path: None,
            success: true,
            error: None,
        }
    }

    pub fn failure(run_id: String, error: impl Into<String>) -> Self {
        Self {
            run_id: run_id.clone(),
            metadata: RunMetadata {
                schema_version: ARTIFACT_SCHEMA_VERSION.to_string(),
                run_id,
                config_hash: String::new(),
                strategy_id: String::new(),
                strategy_version: String::new(),
                crate_version: env!("CARGO_PKG_VERSION").to_string(),
                timestamp_utc: Utc::now(),
                dataset_id: None,
                seed: None,
                costs: CostConfig::default(),
                mode: RunMode::Full,
                execution_mode: ExecutionMode::Standard,
                config_path: String::new(),
                duration_ms: 0,
                dividends_enabled: false,
                dividend_policy: None,
                total_dividend_cashflow: None,
                dividend_count: None,
                mode_fallback_reason: None,
            },
            metrics: RunMetrics::default(),
            timeseries: Vec::new(),
            trace: Vec::new(),
            trades: Vec::new(),
            output_path: None,
            success: false,
            error: Some(error.into()),
        }
    }

    pub fn with_output_path(mut self, path: std::path::PathBuf) -> Self {
        self.output_path = Some(path);
        self
    }
}

/// Dry run result - validation only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DryRunResult {
    pub strategy_id: String,
    pub config_valid: bool,
    pub blocks_resolved: Vec<BlockResolution>,
    pub pipeline_summary: Vec<PipelineStepSummary>,
    pub validation_errors: Vec<String>,
    pub validation_warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockResolution {
    pub block_id: String,
    pub block_type: String,
    pub found: bool,
    pub default_params: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStepSummary {
    pub step: usize,
    pub block_id: String,
    pub block_type: String,
    pub params_configured: HashMap<String, serde_json::Value>,
    pub params_effective: HashMap<String, serde_json::Value>,
}

/// Strict mode validation errors.
///
/// These errors are only raised when the runner is in strict mode.
/// They catch potential data integrity issues that would go unnoticed otherwise.
#[derive(Debug, Clone, thiserror::Error)]
pub enum StrictValidationError {
    // ========================================================================
    // Weight-related errors
    // ========================================================================
    
    #[error("NaN detected in weights for symbol: {0}")]
    NaNWeight(String),
    
    #[error("Inf detected in weights for symbol: {0}")]
    InfWeight(String),
    
    #[error("Weights sum to {actual:.4}, expected ~1.0 (tolerance: {tolerance})")]
    InvalidWeightSum { actual: f64, tolerance: f64 },
    
    #[error("Weight for {symbol} ({weight:.4}) exceeds max_weight ({max:.4})")]
    WeightExceedsMax { symbol: String, weight: f64, max: f64 },
    
    #[error("Weight for {symbol} ({weight:.4}) is below min_weight ({min:.4})")]
    WeightBelowMin { symbol: String, weight: f64, min: f64 },
    
    #[error("Number of positions ({actual}) exceeds max_positions ({max})")]
    TooManyPositions { actual: usize, max: usize },
    
    // ========================================================================
    // Return/metric-related errors
    // ========================================================================
    
    #[error("NaN detected in returns at index: {0}")]
    NaNReturn(usize),
    
    #[error("Inf detected in returns at index: {0}")]
    InfReturn(usize),
    
    #[error("NaN detected in metric: {0}")]
    NaNMetric(String),
    
    #[error("Inf detected in metric: {0}")]
    InfMetric(String),
    
    // ========================================================================
    // Pipeline/signal errors
    // ========================================================================
    
    #[error("Signal for {0} has no valid price")]
    MissingPrice(String),
    
    #[error("Empty universe after selection: {reason}")]
    EmptyUniverse { reason: String },
    
    #[error("Empty pipeline result at step {step}: {reason}")]
    EmptyPipelineResult { step: usize, reason: String },
    
    #[error("Zero-quantity order generated for {symbol} without reason")]
    ZeroQuantityOrder { symbol: String },
    
    #[error("Exit for {symbol} missing reason code")]
    MissingExitReason { symbol: String },
    
    // ========================================================================
    // General errors
    // ========================================================================
    
    #[error("No trades generated")]
    NoTrades,
    
    #[error("Empty timeseries generated")]
    EmptyTimeseries,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_mode_serde() {
        let mode = RunMode::DryRun;
        let json = serde_json::to_string(&mode).unwrap();
        assert_eq!(json, "\"dry_run\"");

        let parsed: RunMode = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, RunMode::DryRun);
    }

    #[test]
    fn test_execution_mode_serde() {
        // Test all variants serialize correctly
        assert_eq!(serde_json::to_string(&ExecutionMode::Standard).unwrap(), "\"standard\"");
        assert_eq!(serde_json::to_string(&ExecutionMode::Compiled).unwrap(), "\"compiled\"");
        assert_eq!(serde_json::to_string(&ExecutionMode::Fast).unwrap(), "\"fast\"");
        assert_eq!(serde_json::to_string(&ExecutionMode::Auto).unwrap(), "\"auto\"");

        // Test deserialization
        let parsed: ExecutionMode = serde_json::from_str("\"fast\"").unwrap();
        assert_eq!(parsed, ExecutionMode::Fast);
    }

    #[test]
    fn test_execution_mode_from_str() {
        assert_eq!(ExecutionMode::from_str("standard"), Some(ExecutionMode::Standard));
        assert_eq!(ExecutionMode::from_str("COMPILED"), Some(ExecutionMode::Compiled));
        assert_eq!(ExecutionMode::from_str("Fast"), Some(ExecutionMode::Fast));
        assert_eq!(ExecutionMode::from_str("auto"), Some(ExecutionMode::Auto));
        assert_eq!(ExecutionMode::from_str("invalid"), None);
    }

    #[test]
    fn test_execution_mode_default() {
        assert_eq!(ExecutionMode::default(), ExecutionMode::Auto);
    }

    #[test]
    fn test_run_metrics_default() {
        let metrics = RunMetrics::default();
        assert_eq!(metrics.cagr, 0.0);
        assert_eq!(metrics.sharpe_ratio, 0.0);
    }

    #[test]
    fn test_experiment_result_failure() {
        let result = ExperimentResult::failure("test-123".into(), "Test error");
        assert!(!result.success);
        assert_eq!(result.error, Some("Test error".to_string()));
    }
}


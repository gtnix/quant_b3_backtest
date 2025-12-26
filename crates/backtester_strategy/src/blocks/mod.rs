//! Strategy blocks - modular units for strategy composition.

pub mod entry;
pub mod exit;
pub mod selection;
pub mod sizing;

use crate::context::StrategyContext;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Block type categorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockType {
    /// Asset selection/filtering (Techniques 1-7)
    Selection,
    /// Entry signal generation (Techniques 8-15)
    Entry,
    /// Exit signal generation (Techniques 16-22)
    Exit,
    /// Position sizing/weighting
    Sizing,
    /// Risk/liquidity filters (Techniques 23-27)
    Filter,
}

/// Signal direction for entry/exit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SignalDirection {
    /// Go long
    Long,
    /// Go short
    Short,
    /// No position / flat
    #[default]
    Flat,
    /// Exit current position
    Exit,
}

/// Trading signal produced by entry/exit blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    pub symbol: String,
    pub direction: SignalDirection,
    /// Signal strength (0.0 to 1.0)
    pub strength: f64,
    /// Block that generated this signal
    pub source_block: String,
    /// Additional metadata
    pub metadata: HashMap<String, f64>,
}

impl Signal {
    pub fn new(symbol: impl Into<String>, direction: SignalDirection, strength: f64) -> Self {
        Self {
            symbol: symbol.into(),
            direction,
            strength: strength.clamp(0.0, 1.0),
            source_block: String::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source_block = source.into();
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: f64) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    pub fn long(symbol: impl Into<String>, strength: f64) -> Self {
        Self::new(symbol, SignalDirection::Long, strength)
    }

    pub fn short(symbol: impl Into<String>, strength: f64) -> Self {
        Self::new(symbol, SignalDirection::Short, strength)
    }

    pub fn flat(symbol: impl Into<String>) -> Self {
        Self::new(symbol, SignalDirection::Flat, 0.0)
    }

    pub fn exit(symbol: impl Into<String>, strength: f64) -> Self {
        Self::new(symbol, SignalDirection::Exit, strength)
    }
}

/// Parameters passed to a block for execution.
pub type BlockParams = HashMap<String, toml::Value>;

/// Validation error for block parameters.
#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("Missing required parameter: {0}")]
    MissingParam(String),
    #[error("Invalid parameter type for '{0}': expected {1}")]
    InvalidType(String, String),
    #[error("Parameter '{0}' out of range: {1}")]
    OutOfRange(String, String),
    #[error("Unknown parameter: {0}")]
    UnknownParam(String),
}

/// Result of block execution.
#[derive(Debug, Clone)]
pub struct BlockResult {
    /// Whether execution succeeded
    pub success: bool,
    /// Signals generated (for entry/exit blocks)
    pub signals: Vec<Signal>,
    /// Weights calculated (for sizing blocks)
    pub weights: HashMap<String, f64>,
    /// Symbols that passed selection (for selection blocks)
    pub selected: Vec<String>,
    /// Symbols excluded with reasons
    pub excluded: Vec<(String, String)>,
    /// Execution message/summary
    pub message: String,
}

impl BlockResult {
    pub fn success(message: impl Into<String>) -> Self {
        Self {
            success: true,
            signals: Vec::new(),
            weights: HashMap::new(),
            selected: Vec::new(),
            excluded: Vec::new(),
            message: message.into(),
        }
    }

    pub fn failure(message: impl Into<String>) -> Self {
        Self {
            success: false,
            signals: Vec::new(),
            weights: HashMap::new(),
            selected: Vec::new(),
            excluded: Vec::new(),
            message: message.into(),
        }
    }

    pub fn with_signals(mut self, signals: Vec<Signal>) -> Self {
        self.signals = signals;
        self
    }

    pub fn with_weights(mut self, weights: HashMap<String, f64>) -> Self {
        self.weights = weights;
        self
    }

    pub fn with_selected(mut self, selected: Vec<String>) -> Self {
        self.selected = selected;
        self
    }

    pub fn with_excluded(mut self, excluded: Vec<(String, String)>) -> Self {
        self.excluded = excluded;
        self
    }
}

/// Trait for all strategy blocks.
pub trait StrategyBlock: Send + Sync {
    /// Unique identifier for this block type.
    fn block_id(&self) -> &'static str;

    /// Block category.
    fn block_type(&self) -> BlockType;

    /// Execute the block with given parameters and context.
    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult;

    /// Validate parameters before execution.
    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError>;

    /// Get default parameters for this block.
    fn default_params(&self) -> BlockParams;

    /// Human-readable description of this block.
    fn description(&self) -> &'static str {
        "No description available"
    }
}

/// Helper to extract typed parameter from BlockParams.
pub fn get_param<T: TryFrom<toml::Value>>(
    params: &BlockParams,
    key: &str,
    default: T,
) -> T {
    params
        .get(key)
        .cloned()
        .and_then(|v| T::try_from(v).ok())
        .unwrap_or(default)
}

/// Helper to extract f64 parameter.
pub fn get_f64(params: &BlockParams, key: &str, default: f64) -> f64 {
    params
        .get(key)
        .and_then(|v| match v {
            toml::Value::Float(f) => Some(*f),
            toml::Value::Integer(i) => Some(*i as f64),
            _ => None,
        })
        .unwrap_or(default)
}

/// Helper to extract i64 parameter.
pub fn get_i64(params: &BlockParams, key: &str, default: i64) -> i64 {
    params
        .get(key)
        .and_then(|v| match v {
            toml::Value::Integer(i) => Some(*i),
            toml::Value::Float(f) => Some(*f as i64),
            _ => None,
        })
        .unwrap_or(default)
}

/// Helper to extract usize parameter.
pub fn get_usize(params: &BlockParams, key: &str, default: usize) -> usize {
    get_i64(params, key, default as i64) as usize
}

/// Helper to extract bool parameter.
pub fn get_bool(params: &BlockParams, key: &str, default: bool) -> bool {
    params
        .get(key)
        .and_then(|v| v.as_bool())
        .unwrap_or(default)
}


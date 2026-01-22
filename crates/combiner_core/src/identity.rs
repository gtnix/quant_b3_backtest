//! Strategy Identity - Comprehensive metadata for strategy traceability.
//!
//! This module provides a standardized identity block that appears in all
//! strategy artifacts for auditability and reproducibility.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use crate::{BlockGene, BlockType, StrategyGenome};

/// Summary of a strategy block for the identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockSummary {
    /// Block type: "selection", "entry", "exit", "sizing"
    pub block_type: String,
    /// Block identifier: "momentum", "ma_crossover", etc.
    pub block_id: String,
    /// Key parameters with string values for display
    pub key_params: HashMap<String, String>,
}

impl BlockSummary {
    /// Create from a BlockGene.
    pub fn from_gene(gene: &BlockGene) -> Self {
        let key_params: HashMap<String, String> = gene.params
            .iter()
            .map(|(k, v)| (k.clone(), format!("{:.4}", v.as_f64())))
            .collect();
        
        Self {
            block_type: gene.block_type.to_string(),
            block_id: gene.block_id.clone(),
            key_params,
        }
    }
}

/// Comprehensive strategy identity for traceability and auditability.
///
/// This struct contains all metadata needed to:
/// - Identify the strategy uniquely
/// - Understand what it does (type, blocks, rules)
/// - Reproduce the results (parameters, seed, version)
/// - Audit the context (market, period, costs)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyIdentity {
    // ========== Core Identification ==========
    /// Unique strategy ID (genome UUID or derived)
    pub strategy_id: String,
    /// Human-readable strategy name
    pub strategy_name: String,
    /// Template slug that originated this strategy (if from GA)
    pub template_slug: Option<String>,
    
    // ========== Market Context ==========
    /// Market: "BR" or "US"
    pub market: String,
    /// Universe: "IBOV", "SP500", etc.
    pub universe: String,
    /// Timeframe: "daily", "intraday", "weekly"
    pub timeframe: String,
    /// Data period start
    pub period_start: String,
    /// Data period end
    pub period_end: String,
    
    // ========== Strategy Classification ==========
    /// Primary strategy type: "momentum", "mean_reversion", "breakout", etc.
    pub strategy_type: String,
    /// Strategy family: "swing", "position", "intraday", etc.
    pub strategy_family: String,
    
    // ========== Blocks/Components ==========
    /// List of blocks composing the strategy
    pub blocks: Vec<BlockSummary>,
    
    // ========== Parameters ==========
    /// All effective parameters as key-value pairs
    pub effective_parameters: HashMap<String, f64>,
    
    // ========== Rules (Human Readable) ==========
    /// Entry rules description
    pub entry_rules: String,
    /// Exit rules description  
    pub exit_rules: String,
    
    // ========== Risk Management ==========
    /// Position sizing method: "equal_weight", "risk_parity", etc.
    pub position_sizing: String,
    /// Stop loss percentage (if applicable)
    pub stop_loss_pct: Option<f64>,
    /// Maximum positions allowed
    pub max_positions: u32,
    
    // ========== Cost Assumptions ==========
    /// Slippage in basis points
    pub slippage_bps: f64,
    /// Commission rate
    pub commission_rate: f64,
    
    // ========== Reproducibility ==========
    /// Random seed used (if applicable)
    pub seed: Option<u64>,
    /// Software version
    pub version: String,
    /// Git commit hash (if available)
    pub commit_hash: Option<String>,
    /// Generation number when created (for GA)
    pub generation: u32,
}

impl Default for StrategyIdentity {
    fn default() -> Self {
        Self {
            strategy_id: String::new(),
            strategy_name: String::new(),
            template_slug: None,
            market: "BR".into(),
            universe: "IBOV".into(),
            timeframe: "daily".into(),
            period_start: String::new(),
            period_end: String::new(),
            strategy_type: "unknown".into(),
            strategy_family: "unknown".into(),
            blocks: Vec::new(),
            effective_parameters: HashMap::new(),
            entry_rules: String::new(),
            exit_rules: String::new(),
            position_sizing: "equal_weight".into(),
            stop_loss_pct: None,
            max_positions: 10,
            slippage_bps: 10.0,
            commission_rate: 0.001,
            seed: None,
            version: env!("CARGO_PKG_VERSION").into(),
            commit_hash: None,
            generation: 0,
        }
    }
}

impl StrategyIdentity {
    /// Create a new identity with basic fields.
    pub fn new(strategy_id: impl Into<String>, strategy_name: impl Into<String>) -> Self {
        Self {
            strategy_id: strategy_id.into(),
            strategy_name: strategy_name.into(),
            ..Default::default()
        }
    }
    
    /// Create identity from a StrategyGenome.
    pub fn from_genome(genome: &StrategyGenome, market: &str, universe: &str) -> Self {
        let strategy_id = genome.id.to_string();
        
        // Extract blocks
        let blocks: Vec<BlockSummary> = genome.genes.iter()
            .map(BlockSummary::from_gene)
            .collect();
        
        // Determine strategy type from first selection block
        let strategy_type = genome.genes.iter()
            .find(|g| g.block_type == BlockType::Selection)
            .map(|g| g.block_id.clone())
            .unwrap_or_else(|| "unknown".into());
        
        // Determine strategy family from template slug or blocks
        let strategy_family = genome.template_slug.as_ref()
            .and_then(|s| s.split('_').next().map(String::from))
            .unwrap_or_else(|| "unknown".into());
        
        // Generate human-readable name
        let strategy_name = genome.human_readable_name();
        
        // Collect all parameters
        let effective_parameters: HashMap<String, f64> = genome.genes.iter()
            .flat_map(|g| g.params.iter())
            .map(|(k, v)| (k.clone(), v.as_f64()))
            .collect();
        
        // Generate entry rules description
        let entry_rules = Self::generate_entry_description(&genome.genes);
        
        // Generate exit rules description
        let exit_rules = Self::generate_exit_description(&genome.genes);
        
        // Get position sizing method
        let position_sizing = genome.genes.iter()
            .find(|g| g.block_type == BlockType::Sizing)
            .map(|g| g.block_id.clone())
            .unwrap_or_else(|| "equal_weight".into());
        
        // Get stop loss from exit blocks
        let stop_loss_pct = genome.genes.iter()
            .find(|g| g.block_type == BlockType::Exit && g.block_id == "stop_loss")
            .and_then(|g| g.params.get("stop_pct").map(|v| v.as_f64()));
        
        Self {
            strategy_id,
            strategy_name,
            template_slug: genome.template_slug.clone(),
            market: market.into(),
            universe: universe.into(),
            timeframe: "daily".into(),
            period_start: String::new(),
            period_end: String::new(),
            strategy_type,
            strategy_family,
            blocks,
            effective_parameters,
            entry_rules,
            exit_rules,
            position_sizing,
            stop_loss_pct,
            max_positions: 10,
            slippage_bps: 10.0,
            commission_rate: 0.001,
            seed: None,
            version: env!("CARGO_PKG_VERSION").into(),
            commit_hash: option_env!("GIT_HASH").map(String::from),
            generation: genome.generation,
        }
    }
    
    /// Generate human-readable entry rules from genes.
    fn generate_entry_description(genes: &[BlockGene]) -> String {
        let entry_blocks: Vec<_> = genes.iter()
            .filter(|g| g.block_type == BlockType::Entry)
            .collect();
        
        if entry_blocks.is_empty() {
            return "No explicit entry rules".into();
        }
        
        entry_blocks.iter()
            .map(|g| {
                match g.block_id.as_str() {
                    "ma_crossover" => {
                        let fast = g.params.get("fast_period").map(|v| v.as_i64() as u32).unwrap_or(10);
                        let slow = g.params.get("slow_period").map(|v| v.as_i64() as u32).unwrap_or(20);
                        format!("MA Crossover: Enter when {}-day MA crosses above {}-day MA", fast, slow)
                    }
                    "rsi" => {
                        let period = g.params.get("period").map(|v| v.as_i64() as u32).unwrap_or(14);
                        let oversold = g.params.get("oversold").map(|v| v.as_f64()).unwrap_or(30.0);
                        format!("RSI({}): Enter when RSI < {:.0} (oversold)", period, oversold)
                    }
                    "macd" => {
                        format!("MACD: Enter on bullish crossover")
                    }
                    "bollinger" => {
                        let period = g.params.get("period").map(|v| v.as_i64() as u32).unwrap_or(20);
                        format!("Bollinger({}): Enter on lower band touch", period)
                    }
                    _ => format!("{}: Default rules", g.block_id)
                }
            })
            .collect::<Vec<_>>()
            .join("; ")
    }
    
    /// Generate human-readable exit rules from genes.
    fn generate_exit_description(genes: &[BlockGene]) -> String {
        let exit_blocks: Vec<_> = genes.iter()
            .filter(|g| g.block_type == BlockType::Exit)
            .collect();
        
        if exit_blocks.is_empty() {
            return "No explicit exit rules (rebalance-based)".into();
        }
        
        exit_blocks.iter()
            .map(|g| {
                match g.block_id.as_str() {
                    "stop_loss" => {
                        let pct = g.params.get("stop_pct").map(|v| v.as_f64()).unwrap_or(0.05);
                        format!("Stop Loss at {:.1}%", pct * 100.0)
                    }
                    "take_profit" => {
                        let pct = g.params.get("target_pct").map(|v| v.as_f64()).unwrap_or(0.10);
                        format!("Take Profit at {:.1}%", pct * 100.0)
                    }
                    "trailing_stop" => {
                        let pct = g.params.get("trail_pct").map(|v| v.as_f64()).unwrap_or(0.05);
                        format!("Trailing Stop at {:.1}%", pct * 100.0)
                    }
                    "time_exit" => {
                        let days = g.params.get("max_days").map(|v| v.as_i64() as u32).unwrap_or(20);
                        format!("Time Exit after {} days", days)
                    }
                    _ => format!("{}: Default exit", g.block_id)
                }
            })
            .collect::<Vec<_>>()
            .join("; ")
    }
    
    /// Set market context.
    pub fn with_market(mut self, market: &str, universe: &str) -> Self {
        self.market = market.into();
        self.universe = universe.into();
        self
    }
    
    /// Set period.
    pub fn with_period(mut self, start: &str, end: &str) -> Self {
        self.period_start = start.into();
        self.period_end = end.into();
        self
    }
    
    /// Set cost assumptions.
    pub fn with_costs(mut self, slippage_bps: f64, commission_rate: f64) -> Self {
        self.slippage_bps = slippage_bps;
        self.commission_rate = commission_rate;
        self
    }
    
    /// Set reproducibility info.
    pub fn with_reproducibility(mut self, seed: Option<u64>, commit_hash: Option<String>) -> Self {
        self.seed = seed;
        self.commit_hash = commit_hash;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BlockGene;
    use uuid::Uuid;
    
    #[test]
    fn test_identity_default() {
        let identity = StrategyIdentity::default();
        assert_eq!(identity.market, "BR");
        assert_eq!(identity.timeframe, "daily");
    }
    
    #[test]
    fn test_identity_from_genome() {
        let genes = vec![
            BlockGene::with_defaults(BlockType::Selection, "momentum"),
            BlockGene::with_defaults(BlockType::Sizing, "equal_weight"),
        ];
        let genome = StrategyGenome::new(genes);
        
        let identity = StrategyIdentity::from_genome(&genome, "BR", "IBOV");
        
        assert_eq!(identity.market, "BR");
        assert_eq!(identity.universe, "IBOV");
        assert_eq!(identity.strategy_type, "momentum");
        assert_eq!(identity.blocks.len(), 2);
    }
}

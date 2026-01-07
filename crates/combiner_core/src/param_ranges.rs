//! Parameter ranges for mutation and validation.
//!
//! Defines valid ranges and steps for all block parameters,
//! extracted from the block catalog.

use crate::genome::{BlockType, ParamValue};
use std::collections::HashMap;

/// Parameter specification with range and default.
#[derive(Debug, Clone)]
pub struct ParamSpec {
    /// Parameter name.
    pub name: String,
    /// Default value.
    pub default: ParamValue,
    /// Description.
    pub description: String,
}

impl ParamSpec {
    /// Create a float parameter spec.
    pub fn float(
        name: impl Into<String>,
        default: f64,
        min: f64,
        max: f64,
        step: f64,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            default: ParamValue::float(default, min, max, step),
            description: description.into(),
        }
    }

    /// Create an int parameter spec.
    pub fn int(
        name: impl Into<String>,
        default: i64,
        min: i64,
        max: i64,
        step: i64,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            default: ParamValue::int(default, min, max, step),
            description: description.into(),
        }
    }

    /// Create a bool parameter spec.
    pub fn bool(name: impl Into<String>, default: bool, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            default: ParamValue::bool(default),
            description: description.into(),
        }
    }
}

/// Block specification with all parameters.
#[derive(Debug, Clone)]
pub struct BlockSpec {
    /// Block ID.
    pub block_id: String,
    /// Block type.
    pub block_type: BlockType,
    /// Parameter specifications.
    pub params: Vec<ParamSpec>,
    /// Description.
    pub description: String,
    /// Whether this block supports fast SoA execution.
    pub fast_supported: bool,
}

impl BlockSpec {
    /// Get default parameters as a HashMap.
    pub fn default_params(&self) -> HashMap<String, ParamValue> {
        self.params
            .iter()
            .map(|p| (p.name.clone(), p.default.clone()))
            .collect()
    }
}

/// Registry of all block parameter ranges.
#[derive(Debug, Clone)]
pub struct ParamRanges {
    blocks: HashMap<String, BlockSpec>,
}

impl Default for ParamRanges {
    fn default() -> Self {
        Self::new()
    }
}

impl ParamRanges {
    /// Create a new parameter ranges registry with all built-in blocks.
    pub fn new() -> Self {
        let mut blocks = HashMap::new();

        // Selection blocks
        Self::register_selection_blocks(&mut blocks);

        // Entry blocks
        Self::register_entry_blocks(&mut blocks);

        // Exit blocks
        Self::register_exit_blocks(&mut blocks);

        // Sizing blocks
        Self::register_sizing_blocks(&mut blocks);

        Self { blocks }
    }

    fn register_selection_blocks(blocks: &mut HashMap<String, BlockSpec>) {
        // B3-calibrated: min_return range -0.10 to +0.15 (realista para 3-6 meses)
        blocks.insert(
            "momentum".into(),
            BlockSpec {
                block_id: "momentum".into(),
                block_type: BlockType::Selection,
                params: vec![
                    ParamSpec::int("lookback_days", 126, 21, 252, 21, "Lookback period in days"),
                    ParamSpec::float("top_pct", 20.0, 5.0, 50.0, 5.0, "Top % of assets to select"),
                    ParamSpec::float("min_return", 0.0, -0.10, 0.15, 0.02, "Minimum return threshold"),
                    ParamSpec::int("skip_last_days", 21, 0, 63, 7, "Days to skip at end"),
                ],
                description: "Momentum selection - ranks by 6-12 month returns".into(),
                fast_supported: true,
            },
        );

        blocks.insert(
            "value".into(),
            BlockSpec {
                block_id: "value".into(),
                block_type: BlockType::Selection,
                params: vec![
                    ParamSpec::float("max_pe", 15.0, 5.0, 50.0, 5.0, "Maximum P/E ratio"),
                    ParamSpec::float("max_pb", 2.0, 0.5, 10.0, 0.5, "Maximum P/B ratio"),
                    ParamSpec::float("top_pct", 20.0, 5.0, 50.0, 5.0, "Top % of assets"),
                ],
                description: "Value selection - low P/E, P/B".into(),
                fast_supported: false,
            },
        );

        blocks.insert(
            "quality".into(),
            BlockSpec {
                block_id: "quality".into(),
                block_type: BlockType::Selection,
                params: vec![
                    ParamSpec::float("min_roe", 0.15, 0.05, 0.50, 0.05, "Minimum ROE"),
                    ParamSpec::float("max_debt_equity", 0.5, 0.1, 2.0, 0.1, "Maximum D/E ratio"),
                    ParamSpec::float("top_pct", 20.0, 5.0, 50.0, 5.0, "Top % of assets"),
                ],
                description: "Quality selection - high ROE, low debt".into(),
                fast_supported: false,
            },
        );

        // B3-calibrated: vol tipica B3 = 25-40%, max_annualized_vol range 0.20 to 0.60
        blocks.insert(
            "low_vol".into(),
            BlockSpec {
                block_id: "low_vol".into(),
                block_type: BlockType::Selection,
                params: vec![
                    ParamSpec::float(
                        "max_annualized_vol",
                        0.30,
                        0.20,
                        0.60,
                        0.05,
                        "Maximum annualized volatility",
                    ),
                    ParamSpec::int("lookback_days", 60, 20, 252, 20, "Lookback period"),
                    ParamSpec::float("top_pct", 20.0, 5.0, 50.0, 5.0, "Top % of assets"),
                ],
                description: "Low volatility selection".into(),
                fast_supported: true,
            },
        );

        // B3-calibrated: boas pagadoras DY 4-7%, max realista 8%
        blocks.insert(
            "dividend".into(),
            BlockSpec {
                block_id: "dividend".into(),
                block_type: BlockType::Selection,
                params: vec![
                    ParamSpec::float("min_yield", 0.03, 0.01, 0.08, 0.01, "Minimum dividend yield"),
                    ParamSpec::float("top_pct", 20.0, 5.0, 50.0, 5.0, "Top % of assets"),
                ],
                description: "Dividend yield selection".into(),
                fast_supported: false,
            },
        );

        // B3-calibrated: min_market_cap max 25B (inclui mid-caps), max_market_cap max 500B
        blocks.insert(
            "size".into(),
            BlockSpec {
                block_id: "size".into(),
                block_type: BlockType::Selection,
                params: vec![
                    ParamSpec::float(
                        "min_market_cap",
                        1_000_000_000.0,
                        100_000_000.0,
                        25_000_000_000.0,
                        1_000_000_000.0,
                        "Minimum market cap",
                    ),
                    ParamSpec::float(
                        "max_market_cap",
                        100_000_000_000.0,
                        1_000_000_000.0,
                        500_000_000_000.0,
                        10_000_000_000.0,
                        "Maximum market cap",
                    ),
                    ParamSpec::float("top_pct", 20.0, 5.0, 50.0, 5.0, "Top % of assets"),
                ],
                description: "Size selection by market cap".into(),
                fast_supported: false,
            },
        );

        // B3-calibrated: com Selic ~10.5%, carry > 4% e impossivel (DY - Selic)
        blocks.insert(
            "carry".into(),
            BlockSpec {
                block_id: "carry".into(),
                block_type: BlockType::Selection,
                params: vec![
                    ParamSpec::float(
                        "min_carry",
                        0.0,
                        -0.08,
                        0.04,
                        0.01,
                        "Minimum carry (yield - risk-free)",
                    ),
                    ParamSpec::float("top_pct", 20.0, 5.0, 50.0, 5.0, "Top % of assets"),
                ],
                description: "Carry selection - dividend vs risk-free".into(),
                fast_supported: false,
            },
        );
    }

    fn register_entry_blocks(blocks: &mut HashMap<String, BlockSpec>) {
        blocks.insert(
            "ma_crossover".into(),
            BlockSpec {
                block_id: "ma_crossover".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("fast_period", 50, 5, 100, 5, "Fast MA period"),
                    ParamSpec::int("slow_period", 200, 50, 400, 25, "Slow MA period"),
                ],
                description: "MA Crossover entry".into(),
                fast_supported: false,
            },
        );

        blocks.insert(
            "rsi".into(),
            BlockSpec {
                block_id: "rsi".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("period", 14, 5, 30, 1, "RSI period"),
                    ParamSpec::float("oversold", 30.0, 10.0, 40.0, 5.0, "Oversold threshold"),
                    ParamSpec::float("overbought", 70.0, 60.0, 90.0, 5.0, "Overbought threshold"),
                ],
                description: "RSI entry".into(),
                fast_supported: false,
            },
        );

        blocks.insert(
            "macd".into(),
            BlockSpec {
                block_id: "macd".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("fast_ema", 12, 5, 20, 1, "Fast EMA period"),
                    ParamSpec::int("slow_ema", 26, 15, 50, 1, "Slow EMA period"),
                    ParamSpec::int("signal", 9, 5, 15, 1, "Signal line period"),
                ],
                description: "MACD entry".into(),
                fast_supported: false,
            },
        );

        blocks.insert(
            "bollinger".into(),
            BlockSpec {
                block_id: "bollinger".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("period", 20, 10, 50, 5, "Bollinger period"),
                    ParamSpec::float("std_dev", 2.0, 1.0, 3.0, 0.5, "Standard deviation multiplier"),
                ],
                description: "Bollinger Bands entry".into(),
                fast_supported: false,
            },
        );

        // B3-calibrated: Z=3 e evento rarissimo, max 2.5 para eventos mais frequentes
        blocks.insert(
            "zscore".into(),
            BlockSpec {
                block_id: "zscore".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("period", 20, 10, 60, 5, "Z-score lookback period"),
                    ParamSpec::float("threshold", 1.5, 1.0, 2.5, 0.25, "Z-score threshold"),
                ],
                description: "Z-Score mean reversion entry".into(),
                fast_supported: false,
            },
        );
    }

    fn register_exit_blocks(blocks: &mut HashMap<String, BlockSpec>) {
        blocks.insert(
            "stop_loss".into(),
            BlockSpec {
                block_id: "stop_loss".into(),
                block_type: BlockType::Exit,
                params: vec![ParamSpec::float(
                    "threshold_pct",
                    0.10,
                    0.02,
                    0.25,
                    0.02,
                    "Stop loss percentage",
                )],
                description: "Stop-loss exit".into(),
                fast_supported: false,
            },
        );

        blocks.insert(
            "take_profit".into(),
            BlockSpec {
                block_id: "take_profit".into(),
                block_type: BlockType::Exit,
                params: vec![ParamSpec::float(
                    "target_pct",
                    0.30,
                    0.05,
                    1.0,
                    0.05,
                    "Take profit percentage",
                )],
                description: "Take-profit exit".into(),
                fast_supported: false,
            },
        );

        blocks.insert(
            "trailing_stop".into(),
            BlockSpec {
                block_id: "trailing_stop".into(),
                block_type: BlockType::Exit,
                params: vec![
                    ParamSpec::float(
                        "trailing_pct",
                        0.15,
                        0.03,
                        0.30,
                        0.03,
                        "Trailing stop percentage",
                    ),
                    ParamSpec::float(
                        "activation_pct",
                        0.10,
                        0.0,
                        0.30,
                        0.05,
                        "Activation gain percentage",
                    ),
                ],
                description: "Trailing stop exit".into(),
                fast_supported: false,
            },
        );

        blocks.insert(
            "time_exit".into(),
            BlockSpec {
                block_id: "time_exit".into(),
                block_type: BlockType::Exit,
                params: vec![ParamSpec::int(
                    "max_days",
                    20,
                    5,
                    252,
                    5,
                    "Maximum days in position",
                )],
                description: "Time-based exit".into(),
                fast_supported: false,
            },
        );
    }

    fn register_sizing_blocks(blocks: &mut HashMap<String, BlockSpec>) {
        // ARROJADO-COMPATIBLE: Adjusted max_weight range from 0.05-0.50 to 0.10-0.40
        // This prevents weight violations when fewer assets are selected
        // (e.g., with 3 assets, equal weight would be 0.33 which needs max_weight >= 0.33)
        blocks.insert(
            "equal_weight".into(),
            BlockSpec {
                block_id: "equal_weight".into(),
                block_type: BlockType::Sizing,
                params: vec![
                    ParamSpec::float("max_weight", 0.25, 0.10, 0.40, 0.05, "Maximum weight per position"),
                    ParamSpec::float("min_weight", 0.02, 0.01, 0.10, 0.01, "Minimum weight per position"),
                    ParamSpec::int("max_positions", 15, 5, 30, 5, "Maximum number of positions"),
                ],
                description: "Equal weight sizing (1/N)".into(),
                fast_supported: true,
            },
        );

        blocks.insert(
            "risk_parity".into(),
            BlockSpec {
                block_id: "risk_parity".into(),
                block_type: BlockType::Sizing,
                params: vec![
                    ParamSpec::float("max_weight", 0.20, 0.05, 0.50, 0.05, "Maximum weight"),
                    ParamSpec::float("min_weight", 0.02, 0.01, 0.10, 0.01, "Minimum weight"),
                    ParamSpec::int("max_positions", 20, 5, 50, 5, "Maximum positions"),
                    ParamSpec::float("fallback_vol", 0.25, 0.10, 0.50, 0.05, "Fallback volatility"),
                ],
                description: "Risk parity sizing (inverse volatility)".into(),
                fast_supported: false,
            },
        );

        blocks.insert(
            "vol_targeting".into(),
            BlockSpec {
                block_id: "vol_targeting".into(),
                block_type: BlockType::Sizing,
                params: vec![
                    ParamSpec::float("target_vol", 0.12, 0.05, 0.30, 0.02, "Target portfolio volatility"),
                    ParamSpec::float("max_weight", 0.30, 0.10, 0.50, 0.05, "Maximum weight"),
                    ParamSpec::float("min_weight", 0.02, 0.01, 0.10, 0.01, "Minimum weight"),
                    ParamSpec::float("max_leverage", 1.0, 0.5, 2.0, 0.1, "Maximum leverage"),
                    ParamSpec::float("correlation", 0.5, 0.0, 1.0, 0.1, "Assumed correlation"),
                    ParamSpec::int("max_positions", 10, 5, 30, 5, "Maximum positions"),
                ],
                description: "Volatility targeting sizing".into(),
                fast_supported: false,
            },
        );
    }

    /// Get block specification by ID.
    pub fn get_block(&self, block_id: &str) -> Option<&BlockSpec> {
        self.blocks.get(block_id)
    }

    /// Get all blocks of a specific type.
    pub fn blocks_by_type(&self, block_type: BlockType) -> Vec<&BlockSpec> {
        self.blocks
            .values()
            .filter(|b| b.block_type == block_type)
            .collect()
    }

    /// Get all block IDs of a specific type.
    pub fn block_ids_by_type(&self, block_type: BlockType) -> Vec<&str> {
        self.blocks
            .values()
            .filter(|b| b.block_type == block_type)
            .map(|b| b.block_id.as_str())
            .collect()
    }

    /// Check if a block ID exists.
    pub fn contains(&self, block_id: &str) -> bool {
        self.blocks.contains_key(block_id)
    }

    /// Get all block IDs.
    pub fn all_block_ids(&self) -> Vec<&str> {
        self.blocks.keys().map(|s| s.as_str()).collect()
    }

    /// Apply restrictions to narrow the parameter space.
    /// 
    /// This method returns a new ParamRanges with:
    /// - Only blocks from allowed families (if specified)
    /// - Narrowed numeric ranges based on provided bounds
    /// 
    /// # Arguments
    /// * `allowed_blocks` - Optional list of block IDs to keep (filters out others)
    /// * `max_parameters` - Optional maximum number of parameters per block
    /// 
    /// # Example
    /// ```ignore
    /// let ranges = ParamRanges::new()
    ///     .with_restrictions(Some(&["momentum", "equal_weight"]), Some(5));
    /// ```
    pub fn with_restrictions(
        mut self,
        allowed_blocks: Option<&[&str]>,
        max_parameters: Option<usize>,
    ) -> Self {
        // Filter blocks if allowed list provided
        if let Some(allowed) = allowed_blocks {
            self.blocks.retain(|id, _| allowed.contains(&id.as_str()));
        }

        // Limit parameters per block if specified
        if let Some(max_params) = max_parameters {
            for block in self.blocks.values_mut() {
                if block.params.len() > max_params {
                    // Keep only the first N parameters (typically the most important)
                    block.params.truncate(max_params);
                }
            }
        }

        self
    }

    /// Apply numeric bounds to narrow parameter ranges.
    /// 
    /// # Arguments
    /// * `block_id` - The block to modify
    /// * `param_name` - The parameter to narrow
    /// * `new_min` - New minimum value (None to keep current)
    /// * `new_max` - New maximum value (None to keep current)
    /// 
    /// The resulting range is the intersection of old and new bounds.
    pub fn narrow_param_range(
        &mut self,
        block_id: &str,
        param_name: &str,
        new_min: Option<f64>,
        new_max: Option<f64>,
    ) -> bool {
        if let Some(block) = self.blocks.get_mut(block_id) {
            for param in &mut block.params {
                if param.name == param_name {
                    // Get current bounds and narrow them
                    match &mut param.default {
                        ParamValue::Float { min, max, .. } => {
                            if let Some(nm) = new_min {
                                *min = min.max(nm);
                            }
                            if let Some(nx) = new_max {
                                *max = max.min(nx);
                            }
                            return true;
                        }
                        ParamValue::Int { min, max, .. } => {
                            if let Some(nm) = new_min {
                                *min = (*min).max(nm as i64);
                            }
                            if let Some(nx) = new_max {
                                *max = (*max).min(nx as i64);
                            }
                            return true;
                        }
                        _ => {}
                    }
                }
            }
        }
        false
    }

    /// Get the total count of optimizable parameters across all blocks.
    pub fn total_parameter_count(&self) -> usize {
        self.blocks.values().map(|b| b.params.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_ranges_creation() {
        let ranges = ParamRanges::new();

        // Check selection blocks
        assert!(ranges.contains("momentum"));
        assert!(ranges.contains("value"));
        assert!(ranges.contains("quality"));

        // Check entry blocks
        assert!(ranges.contains("ma_crossover"));
        assert!(ranges.contains("rsi"));

        // Check exit blocks
        assert!(ranges.contains("stop_loss"));
        assert!(ranges.contains("trailing_stop"));

        // Check sizing blocks
        assert!(ranges.contains("equal_weight"));
        assert!(ranges.contains("risk_parity"));
    }

    #[test]
    fn test_blocks_by_type() {
        let ranges = ParamRanges::new();

        let selection = ranges.blocks_by_type(BlockType::Selection);
        assert_eq!(selection.len(), 7);

        let sizing = ranges.blocks_by_type(BlockType::Sizing);
        assert_eq!(sizing.len(), 3);
    }

    #[test]
    fn test_default_params() {
        let ranges = ParamRanges::new();
        let momentum = ranges.get_block("momentum").unwrap();

        let defaults = momentum.default_params();
        assert!(defaults.contains_key("lookback_days"));
        assert!(defaults.contains_key("top_pct"));
    }
}


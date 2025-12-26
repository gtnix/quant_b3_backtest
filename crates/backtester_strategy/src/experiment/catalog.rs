//! Block catalog generator - produces documentation of available blocks.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::blocks::{BlockParams, BlockType};
use crate::registry::BlockRegistry;

/// Generator for block catalog documentation.
pub struct BlockCatalog;

impl BlockCatalog {
    /// Generate markdown documentation of all registered blocks.
    pub fn generate_markdown(registry: &BlockRegistry) -> String {
        let mut output = String::new();

        output.push_str("# Block Catalog\n\n");
        output.push_str("This document lists all available strategy blocks in the Strategy Factory.\n\n");
        output.push_str("## Overview\n\n");
        output.push_str(&Self::generate_summary(registry));
        output.push_str("\n---\n\n");

        // Selection Blocks
        output.push_str("## Selection Blocks\n\n");
        output.push_str("Selection blocks filter and rank assets from the universe.\n\n");
        output.push_str(&Self::generate_block_table(registry, BlockType::Selection));
        output.push_str("\n");
        output.push_str(&Self::generate_block_details(registry, BlockType::Selection));

        // Entry Blocks
        output.push_str("## Entry Blocks\n\n");
        output.push_str("Entry blocks generate buy/sell signals based on technical indicators.\n\n");
        output.push_str(&Self::generate_block_table(registry, BlockType::Entry));
        output.push_str("\n");
        output.push_str(&Self::generate_block_details(registry, BlockType::Entry));

        // Exit Blocks
        output.push_str("## Exit Blocks\n\n");
        output.push_str("Exit blocks determine when to close positions.\n\n");
        output.push_str(&Self::generate_block_table(registry, BlockType::Exit));
        output.push_str("\n");
        output.push_str(&Self::generate_block_details(registry, BlockType::Exit));

        // Sizing Blocks
        output.push_str("## Sizing Blocks\n\n");
        output.push_str("Sizing blocks determine position weights.\n\n");
        output.push_str(&Self::generate_block_table(registry, BlockType::Sizing));
        output.push_str("\n");
        output.push_str(&Self::generate_block_details(registry, BlockType::Sizing));

        // Usage Examples
        output.push_str("---\n\n");
        output.push_str("## Usage Examples\n\n");
        output.push_str(&Self::generate_examples());

        output
    }

    /// Generate summary statistics.
    fn generate_summary(registry: &BlockRegistry) -> String {
        let selection_count = registry.blocks_by_type(BlockType::Selection).len();
        let entry_count = registry.blocks_by_type(BlockType::Entry).len();
        let exit_count = registry.blocks_by_type(BlockType::Exit).len();
        let sizing_count = registry.blocks_by_type(BlockType::Sizing).len();
        let total = selection_count + entry_count + exit_count + sizing_count;

        format!(
            "| Category | Count |\n|----------|-------|\n| Selection | {} |\n| Entry | {} |\n| Exit | {} |\n| Sizing | {} |\n| **Total** | **{}** |\n",
            selection_count, entry_count, exit_count, sizing_count, total
        )
    }

    /// Generate a table of blocks for a given type.
    fn generate_block_table(registry: &BlockRegistry, block_type: BlockType) -> String {
        let block_ids = registry.blocks_by_type(block_type);
        if block_ids.is_empty() {
            return "No blocks registered.\n".to_string();
        }

        let mut table = String::new();
        table.push_str("| block_id | Description | Key Parameters |\n");
        table.push_str("|----------|-------------|----------------|\n");

        for block_id in block_ids {
            if let Some(block) = registry.get(block_id) {
                let desc = block.description();
                let params = block.default_params();
                let param_keys: Vec<&String> = params.keys().collect();
                let param_str = if param_keys.is_empty() {
                    "-".to_string()
                } else {
                    param_keys.iter().map(|k| k.as_str()).collect::<Vec<_>>().join(", ")
                };
                table.push_str(&format!("| `{}` | {} | {} |\n", block_id, desc, param_str));
            }
        }

        table
    }

    /// Generate detailed documentation for blocks of a given type.
    fn generate_block_details(registry: &BlockRegistry, block_type: BlockType) -> String {
        let block_ids = registry.blocks_by_type(block_type);
        let mut details = String::new();

        for block_id in block_ids {
            if let Some(block) = registry.get(block_id) {
                details.push_str(&format!("### `{}`\n\n", block_id));
                details.push_str(&format!("{}\n\n", block.description()));
                
                let params = block.default_params();
                if !params.is_empty() {
                    details.push_str("**Parameters:**\n\n");
                    details.push_str("| Parameter | Type | Default | Description |\n");
                    details.push_str("|-----------|------|---------|-------------|\n");

                    for (k, v) in &params {
                        let desc = Self::get_param_description(block_id, k);
                        let (type_str, value_str) = Self::format_toml_value(v);
                        details.push_str(&format!("| `{}` | {} | {} | {} |\n", k, type_str, value_str, desc));
                    }

                    details.push_str("\n");
                }

                details.push_str("**Example:**\n\n");
                details.push_str("```toml\n");
                details.push_str("[[pipeline]]\n");
                let type_str = format!("{:?}", block_type).to_lowercase();
                details.push_str(&format!("type = \"{}\"\n", type_str));
                details.push_str(&format!("block_id = \"{}\"\n", block_id));
                if !params.is_empty() {
                    details.push_str("params = { ");
                    let param_strs: Vec<String> = params
                        .iter()
                        .map(|(k, v)| format!("{} = {}", k, Self::format_toml_value_inline(v)))
                        .collect();
                    details.push_str(&param_strs.join(", "));
                    details.push_str(" }\n");
                }
                details.push_str("```\n\n");
            }
        }

        details
    }

    /// Format a toml::Value to (type_name, value_string).
    fn format_toml_value(v: &toml::Value) -> (&'static str, String) {
        match v {
            toml::Value::Float(f) => ("float", format!("{}", f)),
            toml::Value::Integer(i) => ("int", format!("{}", i)),
            toml::Value::Boolean(b) => ("bool", format!("{}", b)),
            toml::Value::String(s) => ("string", format!("\"{}\"", s)),
            toml::Value::Array(_) => ("array", "[...]".into()),
            toml::Value::Table(_) => ("table", "{...}".into()),
            toml::Value::Datetime(dt) => ("datetime", dt.to_string()),
        }
    }

    /// Format toml::Value for inline display in TOML.
    fn format_toml_value_inline(v: &toml::Value) -> String {
        match v {
            toml::Value::Float(f) => format!("{}", f),
            toml::Value::Integer(i) => format!("{}", i),
            toml::Value::Boolean(b) => format!("{}", b),
            toml::Value::String(s) => format!("\"{}\"", s),
            toml::Value::Array(arr) => {
                let items: Vec<String> = arr.iter().map(Self::format_toml_value_inline).collect();
                format!("[{}]", items.join(", "))
            }
            toml::Value::Table(t) => {
                let items: Vec<String> = t.iter()
                    .map(|(k, v)| format!("{} = {}", k, Self::format_toml_value_inline(v)))
                    .collect();
                format!("{{ {} }}", items.join(", "))
            }
            toml::Value::Datetime(dt) => dt.to_string(),
        }
    }

    /// Get parameter description (hardcoded for now, could be metadata-driven).
    fn get_param_description(block_id: &str, param: &str) -> &'static str {
        match (block_id, param) {
            // Selection params
            (_, "lookback_days") => "Number of days to look back for calculation",
            (_, "min_return") => "Minimum return threshold",
            (_, "top_pct") => "Top percentage of assets to select",
            (_, "skip_last_days") => "Skip recent days (avoid mean reversion)",
            
            // Entry params
            (_, "fast_period") => "Fast moving average period",
            (_, "slow_period") => "Slow moving average period",
            (_, "period") => "Indicator period",
            (_, "std_dev") => "Standard deviation multiplier",
            (_, "overbought") => "Overbought threshold",
            (_, "oversold") => "Oversold threshold",
            (_, "signal_period") => "Signal line period",
            (_, "z_threshold") => "Z-score threshold for mean reversion",
            
            // Exit params
            (_, "threshold_pct") => "Exit threshold as percentage",
            (_, "max_days") => "Maximum holding period in days",
            (_, "trailing_pct") => "Trailing stop percentage",
            
            // Sizing params
            (_, "max_weight") => "Maximum weight per position",
            (_, "min_weight") => "Minimum weight per position",
            (_, "max_positions") => "Maximum number of positions",
            (_, "target_vol") => "Target portfolio volatility",
            (_, "lookback") => "Lookback period for volatility calculation",
            
            _ => "Parameter for block configuration",
        }
    }

    /// Generate usage examples.
    fn generate_examples() -> String {
        let mut examples = String::new();

        examples.push_str("### Momentum Strategy\n\n");
        examples.push_str("```toml\n");
        examples.push_str("[strategy]\n");
        examples.push_str("id = \"momentum_pure\"\n");
        examples.push_str("version = \"1.0.0\"\n");
        examples.push_str("description = \"Pure momentum with equal weights\"\n\n");
        examples.push_str("[[pipeline]]\n");
        examples.push_str("type = \"selection\"\n");
        examples.push_str("block_id = \"momentum\"\n");
        examples.push_str("params = { lookback_days = 126, top_pct = 20 }\n\n");
        examples.push_str("[[pipeline]]\n");
        examples.push_str("type = \"sizing\"\n");
        examples.push_str("block_id = \"equal_weight\"\n");
        examples.push_str("params = { max_weight = 0.20 }\n");
        examples.push_str("```\n\n");

        examples.push_str("### Multi-Factor Strategy\n\n");
        examples.push_str("```toml\n");
        examples.push_str("[strategy]\n");
        examples.push_str("id = \"value_quality\"\n");
        examples.push_str("version = \"1.0.0\"\n");
        examples.push_str("description = \"Value + Quality with risk parity\"\n\n");
        examples.push_str("[[pipeline]]\n");
        examples.push_str("type = \"selection\"\n");
        examples.push_str("block_id = \"value\"\n\n");
        examples.push_str("[[pipeline]]\n");
        examples.push_str("type = \"selection\"\n");
        examples.push_str("block_id = \"quality\"\n\n");
        examples.push_str("[[pipeline]]\n");
        examples.push_str("type = \"sizing\"\n");
        examples.push_str("block_id = \"risk_parity\"\n");
        examples.push_str("params = { max_weight = 0.20 }\n");
        examples.push_str("```\n\n");

        examples.push_str("### Trend Following with Exits\n\n");
        examples.push_str("```toml\n");
        examples.push_str("[strategy]\n");
        examples.push_str("id = \"trend_following\"\n");
        examples.push_str("version = \"1.0.0\"\n");
        examples.push_str("description = \"MA crossover with trailing stop\"\n\n");
        examples.push_str("[[pipeline]]\n");
        examples.push_str("type = \"entry\"\n");
        examples.push_str("block_id = \"ma_crossover\"\n");
        examples.push_str("params = { fast_period = 20, slow_period = 50 }\n\n");
        examples.push_str("[[pipeline]]\n");
        examples.push_str("type = \"exit\"\n");
        examples.push_str("block_id = \"trailing_stop\"\n");
        examples.push_str("params = { trailing_pct = 0.10 }\n\n");
        examples.push_str("[[pipeline]]\n");
        examples.push_str("type = \"sizing\"\n");
        examples.push_str("block_id = \"vol_targeting\"\n");
        examples.push_str("params = { target_vol = 0.15 }\n");
        examples.push_str("```\n\n");

        examples
    }

    /// Write catalog to a file.
    pub fn write_to_file(registry: &BlockRegistry, path: &Path) -> std::io::Result<()> {
        let content = Self::generate_markdown(registry);
        fs::write(path, content)
    }

    /// Generate a JSON representation of the catalog.
    pub fn generate_json(registry: &BlockRegistry) -> String {
        let mut blocks: HashMap<String, Vec<BlockInfo>> = HashMap::new();

        for block_type in [
            BlockType::Selection,
            BlockType::Entry,
            BlockType::Exit,
            BlockType::Sizing,
        ] {
            let type_name = format!("{:?}", block_type).to_lowercase();
            let mut type_blocks = Vec::new();

            for block_id in registry.blocks_by_type(block_type) {
                if let Some(block) = registry.get(block_id) {
                    let params = block.default_params();
                    let param_map: HashMap<String, serde_json::Value> = params
                        .iter()
                        .map(|(k, v)| (k.clone(), Self::toml_to_json(v)))
                        .collect();

                    type_blocks.push(BlockInfo {
                        block_id: block_id.to_string(),
                        description: block.description().to_string(),
                        default_params: param_map,
                    });
                }
            }

            blocks.insert(type_name, type_blocks);
        }

        serde_json::to_string_pretty(&blocks).unwrap_or_default()
    }

    /// Convert toml::Value to serde_json::Value.
    fn toml_to_json(value: &toml::Value) -> serde_json::Value {
        match value {
            toml::Value::String(s) => serde_json::json!(s),
            toml::Value::Integer(i) => serde_json::json!(i),
            toml::Value::Float(f) => serde_json::json!(f),
            toml::Value::Boolean(b) => serde_json::json!(b),
            toml::Value::Array(arr) => {
                serde_json::Value::Array(arr.iter().map(Self::toml_to_json).collect())
            }
            toml::Value::Table(t) => {
                let map: serde_json::Map<String, serde_json::Value> = t
                    .iter()
                    .map(|(k, v)| (k.clone(), Self::toml_to_json(v)))
                    .collect();
                serde_json::Value::Object(map)
            }
            toml::Value::Datetime(dt) => serde_json::json!(dt.to_string()),
        }
    }
}

#[derive(serde::Serialize)]
struct BlockInfo {
    block_id: String,
    description: String,
    default_params: HashMap<String, serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_markdown() {
        let registry = BlockRegistry::with_builtins();
        let markdown = BlockCatalog::generate_markdown(&registry);

        assert!(markdown.contains("# Block Catalog"));
        assert!(markdown.contains("## Selection Blocks"));
        assert!(markdown.contains("## Entry Blocks"));
        assert!(markdown.contains("`momentum`"));
        assert!(markdown.contains("`equal_weight`"));
    }

    #[test]
    fn test_generate_json() {
        let registry = BlockRegistry::with_builtins();
        let json = BlockCatalog::generate_json(&registry);

        let parsed: HashMap<String, Vec<serde_json::Value>> =
            serde_json::from_str(&json).unwrap();

        assert!(parsed.contains_key("selection"));
        assert!(parsed.contains_key("entry"));
        assert!(parsed.contains_key("sizing"));
    }

    #[test]
    fn test_summary() {
        let registry = BlockRegistry::with_builtins();
        let summary = BlockCatalog::generate_summary(&registry);

        assert!(summary.contains("Selection"));
        assert!(summary.contains("Entry"));
        assert!(summary.contains("Total"));
    }
}


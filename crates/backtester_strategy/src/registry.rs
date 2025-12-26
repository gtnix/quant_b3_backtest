//! Block registry - maps block_id to implementation.

use crate::blocks::{
    entry::create_entry_block,
    exit::create_exit_block,
    selection::create_selection_block,
    sizing::create_sizing_block,
    BlockParams, BlockType, StrategyBlock,
};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("Block not found: {0}")]
    BlockNotFound(String),
    #[error("Block already registered: {0}")]
    AlreadyRegistered(String),
    #[error("Invalid block type for {0}: expected {1:?}")]
    InvalidBlockType(String, BlockType),
}

/// Registry for strategy blocks.
#[derive(Default)]
pub struct BlockRegistry {
    blocks: HashMap<String, Box<dyn StrategyBlock>>,
}

impl BlockRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create registry with all built-in blocks registered.
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();
        registry.register_all_builtins();
        registry
    }

    /// Register a block.
    pub fn register(&mut self, block: Box<dyn StrategyBlock>) -> Result<(), RegistryError> {
        let id = block.block_id().to_string();
        if self.blocks.contains_key(&id) {
            return Err(RegistryError::AlreadyRegistered(id));
        }
        self.blocks.insert(id, block);
        Ok(())
    }

    /// Get a block by ID.
    pub fn get(&self, block_id: &str) -> Option<&dyn StrategyBlock> {
        self.blocks.get(block_id).map(|b| b.as_ref())
    }

    /// Check if block exists.
    pub fn contains(&self, block_id: &str) -> bool {
        self.blocks.contains_key(block_id)
    }

    /// Get all registered block IDs.
    pub fn block_ids(&self) -> Vec<&str> {
        self.blocks.keys().map(|s| s.as_str()).collect()
    }

    /// Get blocks by type.
    pub fn blocks_by_type(&self, block_type: BlockType) -> Vec<&str> {
        self.blocks
            .iter()
            .filter(|(_, b)| b.block_type() == block_type)
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// Register all built-in blocks.
    pub fn register_all_builtins(&mut self) {
        // Selection blocks (1-7)
        self.register_selection_blocks();
        
        // Entry blocks (8-12)
        self.register_entry_blocks();
        
        // Exit blocks
        self.register_exit_blocks();
        
        // Sizing blocks
        self.register_sizing_blocks();
    }

    fn register_selection_blocks(&mut self) {
        let params = BlockParams::new();
        
        for block_id in ["momentum", "value", "quality", "low_vol", "dividend", "size", "carry"] {
            if let Some(block) = create_selection_block(block_id, &params) {
                let _ = self.register(block);
            }
        }
    }

    fn register_entry_blocks(&mut self) {
        for block_id in ["ma_crossover", "bollinger", "rsi", "macd", "zscore"] {
            if let Some(block) = create_entry_block(block_id) {
                let _ = self.register(block);
            }
        }
    }

    fn register_exit_blocks(&mut self) {
        for block_id in ["stop_loss", "take_profit", "trailing_stop", "time_exit"] {
            if let Some(block) = create_exit_block(block_id) {
                let _ = self.register(block);
            }
        }
    }

    fn register_sizing_blocks(&mut self) {
        for block_id in ["equal_weight", "risk_parity", "vol_targeting"] {
            if let Some(block) = create_sizing_block(block_id) {
                let _ = self.register(block);
            }
        }
    }

    /// Create a block dynamically from type and id.
    pub fn create_block(
        &self,
        step_type: &str,
        block_id: &str,
        params: &BlockParams,
    ) -> Option<Box<dyn StrategyBlock>> {
        match step_type {
            "selection" | "filter" => create_selection_block(block_id, params),
            "entry" => create_entry_block(block_id),
            "exit" => create_exit_block(block_id),
            "sizing" => create_sizing_block(block_id),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_builtins() {
        let registry = BlockRegistry::with_builtins();
        
        // Check selection blocks
        assert!(registry.contains("momentum"));
        assert!(registry.contains("value"));
        assert!(registry.contains("quality"));
        
        // Check entry blocks
        assert!(registry.contains("ma_crossover"));
        assert!(registry.contains("rsi"));
        
        // Check sizing blocks
        assert!(registry.contains("equal_weight"));
        assert!(registry.contains("risk_parity"));
    }

    #[test]
    fn test_blocks_by_type() {
        let registry = BlockRegistry::with_builtins();
        
        let selection_blocks = registry.blocks_by_type(BlockType::Selection);
        assert!(selection_blocks.contains(&"momentum"));
        
        let sizing_blocks = registry.blocks_by_type(BlockType::Sizing);
        assert!(sizing_blocks.contains(&"equal_weight"));
    }
}


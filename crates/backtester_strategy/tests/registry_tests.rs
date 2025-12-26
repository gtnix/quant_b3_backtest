//! Registry tests.

use backtester_strategy::{BlockRegistry, BlockType};

#[test]
fn test_registry_has_all_selection_blocks() {
    let registry = BlockRegistry::with_builtins();
    
    let expected = ["momentum", "value", "quality", "low_vol", "dividend", "size", "carry"];
    for block_id in expected {
        assert!(
            registry.contains(block_id),
            "Missing selection block: {}",
            block_id
        );
    }
}

#[test]
fn test_registry_has_all_entry_blocks() {
    let registry = BlockRegistry::with_builtins();
    
    let expected = ["ma_crossover", "bollinger", "rsi", "macd", "zscore"];
    for block_id in expected {
        assert!(
            registry.contains(block_id),
            "Missing entry block: {}",
            block_id
        );
    }
}

#[test]
fn test_registry_has_all_exit_blocks() {
    let registry = BlockRegistry::with_builtins();
    
    let expected = ["stop_loss", "take_profit", "trailing_stop", "time_exit"];
    for block_id in expected {
        assert!(
            registry.contains(block_id),
            "Missing exit block: {}",
            block_id
        );
    }
}

#[test]
fn test_registry_has_all_sizing_blocks() {
    let registry = BlockRegistry::with_builtins();
    
    let expected = ["equal_weight", "risk_parity", "vol_targeting"];
    for block_id in expected {
        assert!(
            registry.contains(block_id),
            "Missing sizing block: {}",
            block_id
        );
    }
}

#[test]
fn test_blocks_by_type() {
    let registry = BlockRegistry::with_builtins();
    
    let selection = registry.blocks_by_type(BlockType::Selection);
    assert!(selection.len() >= 7, "Expected at least 7 selection blocks");
    
    let entry = registry.blocks_by_type(BlockType::Entry);
    assert!(entry.len() >= 5, "Expected at least 5 entry blocks");
    
    let sizing = registry.blocks_by_type(BlockType::Sizing);
    assert!(sizing.len() >= 3, "Expected at least 3 sizing blocks");
}

#[test]
fn test_block_default_params() {
    let registry = BlockRegistry::with_builtins();
    
    if let Some(block) = registry.get("momentum") {
        let params = block.default_params();
        assert!(!params.is_empty(), "Momentum should have default params");
    }
    
    if let Some(block) = registry.get("equal_weight") {
        let params = block.default_params();
        assert!(params.contains_key("max_weight"), "equal_weight should have max_weight param");
    }
}


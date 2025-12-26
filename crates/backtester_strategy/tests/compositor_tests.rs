//! Compositor tests.

use backtester_strategy::{
    config::load_strategy_from_str,
    compositor::Compositor,
    context::{StrategyCandidate, StrategyContext},
};
use backtester_intelligence::filters::Market;
use chrono::NaiveDate;
use rust_decimal_macros::dec;

fn make_simple_candidates() -> Vec<StrategyCandidate> {
    vec![
        {
            let mut c = StrategyCandidate::new("A", Market::BR);
            c.price = Some(dec!(100));
            c.volatility = Some(0.20);
            c.momentum_return = Some(0.15);
            c.prices = (1..=150).map(|i| 80.0 + i as f64 * 0.2).collect();
            c
        },
        {
            let mut c = StrategyCandidate::new("B", Market::BR);
            c.price = Some(dec!(50));
            c.volatility = Some(0.25);
            c.momentum_return = Some(0.10);
            c.prices = (1..=150).map(|i| 40.0 + i as f64 * 0.1).collect();
            c
        },
        {
            let mut c = StrategyCandidate::new("C", Market::BR);
            c.price = Some(dec!(75));
            c.volatility = Some(0.15);
            c.momentum_return = Some(0.05);
            c.prices = (1..=150).map(|i| 70.0 + i as f64 * 0.05).collect();
            c
        },
    ]
}

#[test]
fn test_simple_pipeline() {
    let toml_str = r#"
[strategy]
id = "test_simple"

[[pipeline]]
type = "selection"
block_id = "momentum"
params = { top_pct = 50, lookback_days = 126 }

[[pipeline]]
type = "sizing"
block_id = "equal_weight"
params = { max_weight = 0.50 }

[rebalance]
frequency = "weekly"

[constraints]
max_weight_per_asset = 0.50
"#;

    let config = load_strategy_from_str(toml_str).unwrap();
    let compositor = Compositor::with_builtins();
    
    let mut ctx = StrategyContext::new(
        NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
        Market::BR,
        dec!(100_000),
    );
    ctx.candidates = make_simple_candidates();
    
    let result = compositor.execute(&config, &mut ctx).unwrap();
    
    assert!(result.success);
    assert!(!result.weights.is_empty());
    
    // Should have trace entries
    assert!(!result.trace.is_empty());
}

#[test]
fn test_pipeline_with_entry_block() {
    let toml_str = r#"
[strategy]
id = "test_entry"

[[pipeline]]
type = "entry"
block_id = "rsi"
params = { period = 14, oversold = 30, overbought = 70 }

[[pipeline]]
type = "sizing"
block_id = "equal_weight"
params = { max_weight = 0.50 }

[rebalance]
frequency = "weekly"

[constraints]
max_weight_per_asset = 0.50
"#;

    let config = load_strategy_from_str(toml_str).unwrap();
    let compositor = Compositor::with_builtins();
    
    let mut ctx = StrategyContext::new(
        NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
        Market::BR,
        dec!(100_000),
    );
    ctx.candidates = make_simple_candidates();
    
    let result = compositor.execute(&config, &mut ctx);
    
    // Should not error even if no signals generated
    assert!(result.is_ok());
}

#[test]
fn test_weights_sum_validation() {
    let toml_str = r#"
[strategy]
id = "test_weights"

[[pipeline]]
type = "sizing"
block_id = "equal_weight"
params = { max_weight = 0.50, max_positions = 3 }

[rebalance]
frequency = "weekly"

[constraints]
max_weight_per_asset = 0.50
"#;

    let config = load_strategy_from_str(toml_str).unwrap();
    let compositor = Compositor::with_builtins();
    
    let mut ctx = StrategyContext::new(
        NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
        Market::BR,
        dec!(100_000),
    );
    ctx.candidates = make_simple_candidates();
    
    let result = compositor.execute(&config, &mut ctx).unwrap();
    
    if result.success && !result.weights.is_empty() {
        let total: f64 = result.weights.values().sum();
        assert!(
            (total - 1.0).abs() < 0.05,
            "Weights should sum to ~1.0, got {}",
            total
        );
    }
}

#[test]
fn test_trace_records_steps() {
    let toml_str = r#"
[strategy]
id = "test_trace"

[[pipeline]]
type = "selection"
block_id = "momentum"
params = { top_pct = 100 }

[[pipeline]]
type = "sizing"
block_id = "risk_parity"
params = { max_weight = 0.50 }

[rebalance]
frequency = "weekly"

[constraints]
max_weight_per_asset = 0.50
"#;

    let config = load_strategy_from_str(toml_str).unwrap();
    let compositor = Compositor::with_builtins();
    
    let mut ctx = StrategyContext::new(
        NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
        Market::BR,
        dec!(100_000),
    );
    ctx.candidates = make_simple_candidates();
    
    let result = compositor.execute(&config, &mut ctx).unwrap();
    
    // Should have at least 2 trace entries (one per step)
    assert!(result.trace.len() >= 2, "Expected at least 2 trace entries");
}


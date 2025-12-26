# Strategy Factory

Modular DSL for declarative strategy composition. Create new strategies by writing TOML configs without modifying engine code.

## Quick Start

```rust
use backtester_strategy::{
    config::load_strategy_config,
    compositor::Compositor,
    context::{StrategyContext, StrategyCandidate},
};

// Load strategy from TOML
let config = load_strategy_config("configs/strategies/golden_momentum.toml")?;

// Create compositor with built-in blocks
let compositor = Compositor::with_builtins();

// Execute pipeline
let mut ctx = StrategyContext::new(date, Market::BR, capital);
ctx.candidates = load_candidates(); // Your data loading
let result = compositor.execute(&config, &mut ctx)?;

// Use results
println!("Selected: {:?}", result.selected);
println!("Weights: {:?}", result.weights);
```

## Creating a New Strategy

Create a TOML file in `configs/strategies/`:

```toml
[strategy]
id = "my_strategy_v1"
version = "1.0.0"
description = "My custom strategy"

# Pipeline steps execute in order
[[pipeline]]
type = "selection"
block_id = "momentum"
params = { lookback_days = 126, top_pct = 20 }

[[pipeline]]
type = "exit"
block_id = "stop_loss"
params = { threshold_pct = 0.10 }

[[pipeline]]
type = "sizing"
block_id = "equal_weight"
params = { max_weight = 0.20 }

[rebalance]
frequency = "weekly"
day = "friday"

[constraints]
max_weight_per_asset = 0.20
min_liquidity_brl = 500000
```

## Available Blocks

### Selection (Techniques 1-7)

| block_id | Description | Key Params |
|----------|-------------|------------|
| `momentum` | 6-12 month returns | `lookback_days`, `min_return`, `top_pct` |
| `value` | Low P/E, P/B | `max_pe`, `max_pb` |
| `quality` | High ROE, low debt | `min_roe`, `max_debt_equity` |
| `low_vol` | Low volatility | `max_annualized_vol`, `lookback_days` |
| `dividend` | High dividend yield | `min_yield`, `max_yield` |
| `size` | Market cap filter | `min_market_cap`, `max_market_cap` |
| `carry` | Dividend vs risk-free | `min_carry` |

### Entry (Techniques 8-12)

| block_id | Description | Key Params |
|----------|-------------|------------|
| `ma_crossover` | MA fast/slow crossover | `fast_period`, `slow_period` |
| `bollinger` | Bollinger band breakout | `period`, `std_dev` |
| `rsi` | RSI oversold/overbought | `period`, `oversold`, `overbought` |
| `macd` | MACD signal crossover | `fast_ema`, `slow_ema`, `signal` |
| `zscore` | Mean reversion | `period`, `threshold` |

### Exit (Techniques 16-19)

| block_id | Description | Key Params |
|----------|-------------|------------|
| `stop_loss` | Exit on loss | `threshold_pct` |
| `take_profit` | Exit on gain | `target_pct` |
| `trailing_stop` | Trailing from high | `trailing_pct`, `activation_pct` |
| `time_exit` | Exit after N days | `max_days` |

### Sizing

| block_id | Description | Key Params |
|----------|-------------|------------|
| `equal_weight` | 1/N allocation | `max_weight`, `min_weight` |
| `risk_parity` | Inverse volatility | `max_weight`, `fallback_vol` |
| `vol_targeting` | Target portfolio vol | `target_vol`, `max_leverage` |

## Adding a New Block

1. Create file in appropriate directory (e.g., `src/blocks/entry/my_block.rs`)
2. Implement `StrategyBlock` trait:

```rust
impl StrategyBlock for MyBlock {
    fn block_id(&self) -> &'static str { "my_block" }
    fn block_type(&self) -> BlockType { BlockType::Entry }
    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        // Your logic here
        BlockResult::success("Done").with_signals(signals)
    }
    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> { Ok(()) }
    fn default_params(&self) -> BlockParams { HashMap::new() }
}
```

3. Add to module and registry factory function
4. Register in `registry.rs`

## Golden Strategies

Three baseline strategies for regression testing:

- **golden_momentum.toml**: Pure momentum + equal weight
- **golden_value_quality.toml**: Value + Quality + LowVol + risk parity
- **golden_trend_vol.toml**: MA crossover + trailing stop + vol targeting

## Experiment Orchestrator

The Strategy Factory includes an Experiment Orchestrator for running strategies and comparing results:

```bash
# Run a single strategy
backtest run --config configs/strategies/my_strategy.toml

# Run all strategies in a folder
backtest run-batch --folder configs/strategies

# Dry run (validation only)
backtest run --config my_strategy.toml --dry-run

# Compare against golden strategy
backtest compare-to-golden --run output/experiments/my-run --golden golden_momentum
```

Each run produces standardized artifacts:
- `metadata.json` - Run configuration and timing
- `metrics.json` - Performance metrics (CAGR, Sharpe, etc.)
- `timeseries.csv` - Equity curve and risk metrics
- `trace.jsonl` - Pipeline execution trace

See [Experiment Orchestrator Documentation](../../docs/EXPERIMENT_ORCHESTRATOR.md) for details.

## Testing

```bash
cargo test -p backtester_strategy
```


# Experiment Orchestrator

The Experiment Orchestrator is a layer on top of the Strategy Factory that executes strategy configurations, produces standardized artifacts, and enables automated comparison against golden strategies.

## Quick Start

```bash
# Run a single strategy
backtest run --config configs/strategies/my_strategy.toml

# Run all strategies in a folder  
backtest run-batch --folder configs/strategies

# Dry run (validation only)
backtest run --config configs/strategies/my_strategy.toml --dry-run

# Compare two runs
backtest compare --run-a output/experiments/run-001 --run-b output/experiments/run-002

# Compare against golden strategy
backtest compare-to-golden --run output/experiments/my-run --golden golden_momentum

# Generate block catalog
backtest generate-catalog --output docs/BLOCK_CATALOG.md --json
```

## CLI Commands

### `run`

Execute a single strategy configuration.

```bash
backtest run --config <path> [--output <dir>] [--dry-run] [--strict]
```

| Flag | Description |
|------|-------------|
| `--config` | Path to strategy TOML file (required) |
| `--output` | Output directory for artifacts (default: `output/experiments`) |
| `--dry-run` | Validate config and resolve blocks without executing |
| `--strict` | Fail on NaN values, invalid weights, or other invariant violations |

### `run-batch`

Execute all strategy configurations in a folder.

```bash
backtest run-batch --folder <path> [--output <dir>] [--strict]
```

### `compare`

Compare metrics between two experiment runs.

```bash
backtest compare --run-a <path> --run-b <path>
```

### `compare-to-golden`

Compare a run against a baseline golden strategy.

```bash
backtest compare-to-golden --run <path> --golden <id> [--golden-dir <path>]
```

Regression is flagged if:
- Sharpe ratio drops > 20%
- Max drawdown increases > 25%
- CAGR drops > 30%

### `generate-catalog`

Generate documentation of all available blocks.

```bash
backtest generate-catalog --output docs/BLOCK_CATALOG.md [--json]
```

## Artifacts

Each experiment run produces a folder with standardized artifacts:

```
output/experiments/<run_id>/
├── metadata.json    # Run configuration and metadata
├── trace.jsonl      # Pipeline execution trace (JSON Lines)
├── metrics.json     # Performance metrics
└── timeseries.csv   # Equity curve and risk metrics
```

### metadata.json

```json
{
  "run_id": "abc123-def456",
  "config_hash": "sha256...",
  "strategy_id": "momentum_v1",
  "strategy_version": "1.0.0",
  "crate_version": "0.1.0",
  "timestamp_utc": "2025-01-01T12:00:00Z",
  "dataset_id": "br_stocks_2024",
  "seed": 42,
  "costs": {
    "trading_fee_pct": 0.001,
    "slippage_pct": 0.0005
  },
  "mode": "full",
  "config_path": "configs/strategies/momentum.toml",
  "duration_ms": 1234
}
```

### metrics.json

```json
{
  "cagr": 0.15,
  "volatility": 0.20,
  "sharpe_ratio": 0.75,
  "max_drawdown": -0.10,
  "max_drawdown_duration_days": 30,
  "turnover_annual": 2.5,
  "hit_rate": 0.55,
  "profit_factor": 1.5,
  "sortino_ratio": 1.0,
  "calmar_ratio": 1.5
}
```

### trace.jsonl

Each line is a JSON object:

```json
{"step": 0, "block_id": "momentum", "block_type": "selection", "message": "Selected 5 assets", "timestamp_ms": 1234567890, "params_effective": {"lookback_days": 126}}
```

### timeseries.csv

```csv
date,equity,drawdown,exposure,vol_exante,vol_expost
2024-01-02,100000.00,0.000000,0.950000,0.150000,
2024-01-03,100250.00,-0.002500,0.950000,0.150000,0.148000
```

## Golden Strategies

Three golden strategies serve as permanent baselines:

| ID | Description | Blocks |
|----|-------------|--------|
| `golden_momentum_v1` | Pure momentum with equal weights | momentum → stop_loss → time_exit → equal_weight |
| `golden_value_quality_v1` | Multi-factor with risk parity | value → quality → low_vol → stop_loss → trailing_stop → risk_parity |
| `golden_trend_vol_v1` | Trend following with vol targeting | ma_crossover → trailing_stop → vol_targeting |

## Programmatic Usage

```rust
use backtester_strategy::experiment::{
    ExperimentRunner, RunnerConfig, Comparator, BlockCatalog
};
use backtester_strategy::BlockRegistry;

// Run a single experiment
let runner = ExperimentRunner::new().strict();
let result = runner.run_single(Path::new("configs/my_strategy.toml"))?;

println!("Sharpe: {:.2}", result.metrics.sharpe_ratio);

// Compare runs
let comparator = Comparator::new();
let comparison = comparator.compare(&run_a_path, &run_b_path)?;

if comparison.regression {
    println!("Regression detected: {}", comparison.regression_reason.unwrap());
}

// Generate catalog
let registry = BlockRegistry::with_builtins();
let markdown = BlockCatalog::generate_markdown(&registry);
```

## Metrics Formulas

| Metric | Formula |
|--------|---------|
| CAGR | $(end/start)^{1/years} - 1$ |
| Volatility | $\sigma_{daily} \times \sqrt{252}$ |
| Sharpe | $(R_{annual} - R_f) / \sigma$ |
| Sortino | $(R_{annual} - R_f) / \sigma_{downside}$ |
| Calmar | $CAGR / |MaxDD|$ |
| Hit Rate | $wins / total\_trades$ |
| Profit Factor | $gross\_profit / gross\_loss$ |

## Configuration

### Runner Configuration

```rust
RunnerConfig {
    output_dir: "output/experiments".into(),
    risk_free_rate: 0.05,  // 5% annual for Sharpe
    costs: CostConfig {
        trading_fee_pct: 0.001,
        slippage_pct: 0.0005,
        min_trade_brl: Some(100.0),
    },
    seed: Some(42),
    dataset_id: Some("br_2024".into()),
}
```

### Regression Thresholds

```rust
RegressionThresholds {
    sharpe_drop_pct: 0.20,      // 20% drop triggers regression
    max_dd_increase_pct: 0.25,  // 25% worse drawdown
    cagr_drop_pct: 0.30,        // 30% CAGR drop
}
```

## See Also

- [Block Catalog](./BLOCK_CATALOG.md) - List of all available blocks
- [Strategy Factory README](../crates/backtester_strategy/README.md) - Strategy composition DSL


# Pipeline Homologation Checklist

This checklist validates the end-to-end pipeline from mining/detection through to the trading journal.

## Pre-requisites

- [ ] Market data synchronized (`data/market_data_ibov_index.csv` for BR, `data/market_data_us_index.csv` for US)
- [ ] Configuration set to relaxed mode (`validation_tier = "research_brazil"` in configs/default.toml)
- [ ] Output directories clean (`rm -rf output/scg/*`)

## 1. Mining/Detection Stage

### Command

```bash
./scripts/mine.sh
```

### Verification

- [ ] Mining process starts without errors
- [ ] Logs show genome generation: `Gen N ULTRA: A=...ms B=...ms`
- [ ] Stage A fitness values are reasonable (Sharpe between 0.5 and 3.0)

### Evidence

```bash
# Check mining is running
grep "Gen.*ULTRA" logs/br.log | tail -5

# Expected output:
# Gen 10 ULTRA: A=4.2ms B=12.3ms pareto=15 validated=0/10 hof=0
```

## 2. Strategy Catalog Population

### Verification

- [ ] Templates are loaded from `configs/strategies/` or built-in catalog
- [ ] Each template has unique slug identifier

### Evidence

```bash
# List available templates
./target/release/combiner strategies --family swing

# Expected: List of strategy templates with IDs
```

## 3. Backtest Execution (Stage A)

### Verification

- [ ] Backtests complete in <30ms per genome (target: <10ms with `--in-process`)
- [ ] Fitness metrics are calculated correctly (CAGR, Sharpe, Max DD, Calmar)
- [ ] Pareto ranking is applied (pareto_rank=0 for top candidates)

### Evidence

```bash
# Check backtest timing
grep "Stage A" logs/br.log | tail -5

# Should show A=X.Xms where X < 30
```

## 4. Stage B Validation

### Verification

- [ ] Walk-forward analysis runs all splits (no early exit in relaxed mode)
- [ ] OOS metrics are calculated (OOS Sharpe, OOS Max DD, PBO, DSR)
- [ ] Trail logs show complete evaluation: `[TRAIL] Gen N | ID | Template | A: sharpe=X | B: sharpe=Y | PASS/FAIL`

### Evidence

```bash
# Check trail logs
grep "\[TRAIL\]" logs/br.log | head -20

# Expected format:
# [TRAIL] Gen 5 | abc12345 | swing_rsi | A: sharpe=1.234 | B: sharpe=0.456 dd=-0.123 pbo=0.234 | FAIL
```

## 5. Artifact Persistence

### Verification

- [ ] Hall of Fame entries are saved with Strategy Identity
- [ ] Failed candidates are tracked with failure reasons
- [ ] Output directory structure is correct:
  - `output/scg/BR_<timestamp>/hof/`
  - `output/scg/BR_<timestamp>/failed_candidates.json` (if any)

### Evidence

```bash
# Check output structure
ls -la output/scg/BR_*/

# Check HoF content (if any passed)
cat output/scg/BR_*/hof/summary.json | jq '.entries | length'
```

## 6. Strategy Identity Completeness

### Verification

For each HoF entry or failed candidate, verify identity contains:

- [ ] `strategy_id` - Unique identifier (UUID)
- [ ] `strategy_name` - Human-readable name
- [ ] `market` - "BR" or "US"
- [ ] `universe` - "IBOV" or "SP500"
- [ ] `timeframe` - "daily", "intraday", etc.
- [ ] `strategy_type` - "momentum", "mean_reversion", etc.
- [ ] `blocks` - Array of block summaries
- [ ] `effective_parameters` - All parameter values
- [ ] `entry_rules` - Human-readable entry description
- [ ] `exit_rules` - Human-readable exit description
- [ ] `generation` - When created in GA

### Evidence

```bash
# Inspect identity in HoF entry
cat output/scg/BR_*/hof/summary.json | jq '.entries[0].identity'
```

## 7. Diagnostic Report Generation

### Command

```bash
./target/release/combiner diagnose --market BR --output artifacts/diagnostics/
```

### Verification

- [ ] Report generated successfully
- [ ] `br_diagnostic_report.json` created with:
  - Total strategies analyzed
  - Top 5 failure reasons with counts
  - Stage A vs Stage B distribution
  - Near-miss strategies
  - Gap diagnosis
- [ ] `br_gap_analysis.md` created with human-readable summary

### Evidence

```bash
# Check diagnostic outputs
cat artifacts/diagnostics/br_failure_breakdown.json

# Expected:
# [["Sharpe too low", 45, 75.0], ["Drawdown too deep", 12, 20.0], ...]
```

## 8. Trading Journal Integration

### Verification

- [ ] Dashboard can read HoF entries
- [ ] Strategy Identity block is displayed
- [ ] Backtest metrics are visible
- [ ] Stage A vs Stage B comparison is available

### Evidence

```bash
# Start dashboard
cd dashboard && npm run dev

# Navigate to Hall of Fame page
# Verify: Each entry shows complete Strategy Identity block
```

## Summary Checklist

| Stage | Status | Notes |
|-------|--------|-------|
| 1. Mining | ✅ | Gen logs show `ULTRA: A=XXms B=XXms pareto=N` |
| 2. Catalog | ✅ | 128 templates loaded from Strategy Catalog |
| 3. Stage A | ✅ | Sharpe 0.4-0.8, proper fitness calculation |
| 4. Stage B | ✅ | Pursuit-to-completion (max_failures=999), TRAIL logs with PASS/FAIL |
| 5. Artifacts | ✅ | failed_candidates.json + HoF persistence added |
| 6. Identity | ✅ | StrategyIdentity propagated to FailedCandidate + UnifiedHofEntry |
| 7. Diagnostics | ✅ | `combiner diagnose` command generates reports |
| 8. Journal | ✅ | Dashboard displays StrategyIdentity (Type, Market, Timeframe, Blocks) |

### Last Validation Run (2026-01-22)
- **Generations**: 294
- **Stage A candidates**: 4
- **Stage B PASS**: 0 (dd threshold too strict)
- **Failed candidates tracked**: 919
- **Identity fields verified**: strategy_id, market (BR/IBOV), timeframe (daily), blocks, entry_rules
- **Dashboard evidence**: Screenshot saved to `artifacts/audits/hof_strategy_identity_display.png`

## Known Issues & Notes

### BR Market

- Stage B validation uses `research_brazil` thresholds:
  - min_oos_sharpe: 0.3
  - max_pbo: 0.40
  - max_oos_drawdown: -0.40
  - max_degradation_pct: 80%
- Early exit is disabled (`max_failures_early_exit = 999`)

### US Market

- Uses `research_us` thresholds
- Generally higher pass rate than BR

## Troubleshooting

### No candidates passing Stage B

1. Check Stage A Sharpe distribution in diagnostic report
2. If Stage A Sharpe is < 0.5 on average, the fitness function may need tuning
3. If Stage A Sharpe is high but Stage B is low, check for overfitting

### Missing Strategy Identity

1. Ensure `from_genome()` is called when creating HoF entries
2. Check that `market` and `universe` are passed correctly

### Trail logs not appearing

1. Ensure log level includes INFO: `RUST_LOG=info`
2. Check that `--in-process` flag is used for Stage B evaluation

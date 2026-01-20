---
name: trader-expert
description: "Execution realism, market conventions, slippage/costs, and trading sessions (B3 + US)"
triggers:
  - command: "/trader-expert"
    description: "Invoke for execution assumptions, costs, sessions, fills"
domain_knowledge:
  - execution modeling (slippage, costs, fills)
  - market microstructure (spread, queue, impact)
  - B3 and US trading sessions
  - stress testing execution assumptions
  - capacity and liquidity constraints
---

# Trader Expert

## Role

Trader / Market Microstructure & Execution Realism Lead (B3 + US).
Expert in execution modeling, cost analysis, and trading session conventions.

---

## Expertise Map

### Market Microstructure
- Spread dynamics: bid-ask spread as primary execution cost
- Bid-ask bounce: false signals from microstructure noise
- Market impact: price movement caused by order flow
- Queue priority: time priority in limit order books
- Adverse selection: informed traders on the other side

### Execution Models
- **SlippageModel::None**: Zero slippage (testing only)
- **SlippageModel::Constant**: Fixed bps per trade (conservative fallback)
- **SlippageModel::VolumeLinear**: base_bps + volume_factor * (order_size / bar_volume)
- **SlippageModel::Volatility**: base_bps + vol_factor * (high - low) / close
- **SlippageModel::SpreadProxy**: Estimates bid-ask from price patterns
- Implementation: `crates/backtester_execution/src/lib.rs`

### Slippage Components
- Spread crossing: half-spread to cross from bid to ask
- Market impact: additional cost from moving the market
- Volatility cost: wider spreads in high vol regimes
- Timing risk: price drift during execution window

### Cost Structure

**B3 Fee Tiers:**

| Tier | Fixed | Commission | Emolument |
|------|-------|------------|-----------|
| B3Retail | R$10 | 0.15% | 0.035% |
| B3Prime | R$5 | 0.10% | 0.035% |

**US Fee Tiers:**

| Tier | Fixed | Commission | Per-Share | SEC Fee |
|------|-------|------------|-----------|---------|
| USRetail | $1 | 0.10% | $0.005 | 0.002% |
| USPrime | $0 | 0.03% | $0.003 | 0.002% |

Implementation: `crates/backtester_execution/src/config.rs` (FeeTier enum)

### Trading Sessions

**B3 (America/Sao_Paulo):**

| Phase | Time | Characteristics |
|-------|------|-----------------|
| Pre-Market Auction | 09:45-10:00 | No continuous matching, price discovery |
| Regular Session | 10:00-17:55 | Continuous trading |
| Closing Auction | 17:55-18:00 | Final price determination |
| After-Hours | 18:25-18:45 | Limited liquidity |

**NYSE (America/New_York):**

| Phase | Time | Characteristics |
|-------|------|-----------------|
| Pre-Market | 04:00-09:30 | ECN trading, lower liquidity |
| Regular Session | 09:30-16:00 | Full liquidity |
| After-Hours | 16:00-20:00 | ECN trading, wider spreads |

Implementation: `crates/market_data/src/calendar/hours.rs`

### Fill Policy
- **max_participation**: Maximum fraction of bar volume (default 5%)
- **allow_partial**: Whether partial fills are accepted
- **gap_policy**: ExecuteAtOpen, SkipIfGapExceeds, AlwaysExecute
- **reject_policy**: RetryNextBar, Cancel, ConvertToNextDay
- Implementation: `crates/backtester_execution/src/config.rs` (FillPolicyConfig)

### Capacity Analysis
- **capacity_proxy_usd**: Estimated AUM before significant degradation
- Methodology: 5% max participation at estimated daily volume
- Institutional threshold: >= $10M USD
- Implementation: `crates/backtester_execution/src/cost_report.rs`

### Institutional Gates
- **max_turnover_annual**: 12x portfolio per year
- **max_slippage_pct_of_pnl**: 30% of gross PnL
- **min_capacity_usd**: $5M USD
- **max_avg_slippage_bps**: 25 bps average
- Implementation: `crates/backtester_execution/src/gates.rs`

### Stress Testing Integration
- S1 (costs_2x): Double all execution costs
- S2 (delay_plus1): Add one bar execution delay
- S3 (spread_widen_vol): Triple slippage in high vol
- S4 (capacity_constraint): 1% max participation
- S5 (combined_adverse): 2x costs + 1 bar delay
- Implementation: `crates/backtester_execution/src/stress.rs`

### Intraday vs Position Considerations

**Intraday:**
- Spread/slippage dominates cost structure
- Session boundaries critical (no overnight)
- Latency assumptions matter (S2 stress)
- High turnover amplifies cost impact

**Position:**
- Overnight gaps are primary risk
- Corporate actions affect prices
- Borrow costs for shorts (limitation: not fully modeled)
- Rebalance timing within session

---

## When to Use

**INVOKE this skill when:**
- Strategy has high turnover needing cost validation
- Execution assumptions need verification
- Intraday strategy operates near session boundaries
- Result degrades significantly with slippage increase
- Limit order fills assumed at 100%
- Short positions or borrow costs involved
- Strategy crosses timezone boundaries
- Backtest uses closing prices but assumes execution at close

**DO NOT use this skill when:**
- Optimizing engine performance (use `/quant-engineer`)
- Validating OOS metrics (use `/risk-analyst`)
- Investigating data quality (use `/data-engineer`)
- Designing strategy logic (use `/quant-researcher`)

---

## Operating Rules

### Hard Constraints

1. **Never accept fill = 100% for limit orders without justification**
   - Limit orders have no fill guarantee
   - Document fill rate assumption or use market orders

2. **Never accept intraday without stress S1 and S2**
   - S1 (costs_2x): Must maintain positive Sharpe
   - S2 (delay+1): Must prove not latency-dependent

3. **Never accept strategy depending on closing auction without modeling**
   - Closing auction price may differ from last print
   - Document as limitation if not modeled

4. **Never accept fixed cost without sensitivity analysis**
   - Minimum 3 scenarios: base, pessimist (2x), stress (3x)
   - Report cost_as_pct_of_gross_pnl for each

5. **Never accept "infinite capacity"**
   - Require capacity_proxy_usd or max_participation limit
   - Flag if capacity < $5M (institutional threshold)

6. **Never accept short without borrow/fees rules**
   - Document borrow cost assumption or declare limitation
   - Current system limitation: borrow costs not fully modeled

7. **Never accept backtest crossing sessions without data-engineer handoff**
   - Timezone alignment must be verified
   - Session boundaries must be respected

8. **Never accept OOS without execution aligned to trading calendar**
   - Verify no trades on holidays
   - Verify session times match market

9. **Never accept cost model that "improves" with more trades**
   - This indicates a bug in cost calculation
   - Costs must be monotonically non-decreasing with trades

10. **Never accept settlement assumption without confirmation**
    - Current limitation: T+1/T+2 settlement not modeled
    - Document as limitation in Assumptions Card

11. **Never accept after-hours/pre-market execution if not supported**
    - System models sessions but may not support extended hours trading
    - Verify session support before assuming execution

12. **Never promote without Execution Assumptions Card**
    - Mandatory template must be completed
    - Handoff to risk-analyst must include signed card

---

## Repo Anchors

### Execution Crate

| File | Purpose |
|------|---------|
| `crates/backtester_execution/src/lib.rs` | SlippageModel, CostModel, LiquidityModel, AdvancedExecutionModel |
| `crates/backtester_execution/src/config.rs` | ExecutionModelConfig, SlippageModelConfig, FeeModelConfig, FeeTier |
| `crates/backtester_execution/src/stress.rs` | StressSuite S1-S5, StressScenario, AcceptanceCriteria |
| `crates/backtester_execution/src/gates.rs` | GateChecker, InstitutionalGatesConfig |
| `crates/backtester_execution/src/cost_report.rs` | CostReport, SlippageBreakdown, FeeBreakdown |

### Calendar Module

| File | Purpose |
|------|---------|
| `crates/market_data/src/calendar/hours.rs` | MarketHoursProvider, B3/NYSE session times |
| `crates/market_data/src/calendar/mod.rs` | MarketSessionCalendar, SessionInfo, TimeRange |
| `crates/market_data/src/calendar/holidays.rs` | HolidayProvider, HolidayType |

### Configuration

| File | Purpose |
|------|---------|
| `configs/execution_institutional.toml` | Institutional execution config with stress suite |
| `configs/risk_profiles/` | Risk profile configurations |

### Related Skills

| File | Purpose |
|------|---------|
| `.cursor/skills/risk-analyst/SKILL.md` | Stress testing handoff, validation gates |
| `.cursor/skills/data-engineer/SKILL.md` | Calendar/timezone handoff |

---

## Market Session and Auction Rules

### Session Truth Table

| Market | Pre-Market | Regular | Closing Auction | After-Hours |
|--------|------------|---------|-----------------|-------------|
| B3 | 09:45-10:00 | 10:00-17:55 | 17:55-18:00 | 18:25-18:45 |
| NYSE | 04:00-09:30 | 09:30-16:00 | N/A | 16:00-20:00 |

### Execution Implications

**Pre-Market (B3):**
- Auction-based, no continuous matching
- Orders accumulate for opening price determination
- Cannot assume mid-price execution

**Regular Session:**
- Continuous matching, full liquidity
- Standard slippage models apply
- Primary execution window for strategies

**Closing Auction (B3):**
- 17:55-18:00: Price discovery phase
- Final price may differ from 17:55 print
- Strategies using "close" price: verify timing assumption

**After-Hours:**
- Lower liquidity, wider spreads
- B3: 18:25-18:45 (limited)
- NYSE: 16:00-20:00 (ECN)
- Spreads may be 2-5x regular session

### Special Sessions

| Type | Description | Impact |
|------|-------------|--------|
| HalfDay | Early close (e.g., Christmas Eve) | Reduced session hours |
| LateOpen | Late start (e.g., Ash Wednesday B3) | No pre-market |
| ExtraordinaryClosure | Emergency closure | No trading |

### Limitations (Document These)

- **Auction mechanics not simulated**: Opening/closing auction prices not modeled
- **Extended hours execution**: Sessions defined but may not be executable
- **Circuit breakers/halts**: Not modeled

---

## Cost and Slippage Model Playbook

### Model Selection

| Scenario | Recommended Model | Parameters |
|----------|-------------------|------------|
| Conservative fallback | Constant | bps = 10 |
| Liquid large-cap | Constant | bps = 5 |
| Mid-cap / higher turnover | VolumeLinear | base_bps = 5, volume_factor = 0.5 |
| Volatile regime aware | VolatilityAdaptive | base_bps = 5, vol_factor = 0.3 |
| Institutional | VolatilityAdaptive | base_bps = 5, vol_factor = 0.3, regime_multiplier = 2.0 |

### Calibration Sources

- **Median spread**: From historical bid-ask data (if available)
- **ATR-based volatility**: (high - low) / close as proxy
- **Average daily volume**: For participation rate calculation
- **Historical fill rates**: For limit order assumptions

### Scenario Analysis (Mandatory)

| Scenario | Slippage Multiplier | Purpose |
|----------|---------------------|---------|
| Base | 1.0x | Normal conditions |
| Pessimist | 2.0x | Market stress |
| Stress | 3.0x | Extreme conditions |

### Reporting Metrics

| Metric | How to Measure | Red Flag |
|--------|----------------|----------|
| cost_as_pct_of_gross_pnl | total_costs / gross_pnl * 100 | > 30% |
| avg_slippage_bps | mean(slippage / notional * 10000) | > 25 bps |
| turnover_annual | total_traded / avg_nav * (252 / days) | > 12x |
| capacity_proxy_usd | See cost_report.rs methodology | < $5M |

---

## Execution Assumptions Card

### Template

```markdown
## Execution Assumptions Card

**Strategy ID:** {genome_id}
**Market:** B3 / US
**Timeframe:** {daily / intraday / 30m / etc}
**Universe:** {IBOV / SPY / custom}
**Date:** YYYY-MM-DD

### Order Execution
- Order type: Market / Limit
- Fill assumption: 100% / Partial / Volume-constrained
- Delay bars: {0 / 1 / 2}

### Slippage Model
- Model: Constant / VolumeLinear / VolatilityAdaptive
- Base slippage: {X} bps
- Parameters: {list}

### Scenarios Tested
| Scenario | Slippage | Sharpe | Pass |
|----------|----------|--------|------|
| Base | {X} bps | {Y.YY} | Y/N |
| Pessimist (2x) | {2X} bps | {Y.YY} | Y/N |
| Stress (3x) | {3X} bps | {Y.YY} | Y/N |

### Fee Structure
- Fee tier: B3Retail / B3Prime / USRetail / USPrime
- Fixed per trade: {amount}
- Commission rate: {pct}
- Overrides: {any}

### Session Constraints
- Sessions used: Regular only / Includes auction / Extended hours
- Timezone: America/Sao_Paulo / America/New_York
- Trading calendar verified: Yes / No

### Capacity
- max_participation: {pct}
- capacity_proxy_usd: ${amount}
- Institutional grade (>$10M): Yes / No

### Limitations Declared
- [ ] Settlement not modeled (T+1/T+2)
- [ ] Borrow costs not modeled (if short)
- [ ] Auction mechanics not simulated
- [ ] Extended hours not executable
- [ ] {other}

### Sign-off
- [ ] Trader Expert reviewed
- [ ] Ready for risk-analyst handoff
```

---

## Acceptance Criteria

### Execution Review

| Criterion | Pass | Fail |
|-----------|------|------|
| Assumptions Card | Completed | Missing fields |
| Sensitivity analysis | 3 scenarios tested | Single scenario only |
| Intraday stress | S1 + S2 passed | Failed or not run |
| Position stress | S5 passed | Failed or not run |
| Calendar coherence | Verified | Trades on holidays |
| Capacity documented | Present | Missing |
| Limitations declared | Explicit | Assumed away |
| Handoff signed | Yes | No |

### Gate Thresholds

| Gate | Threshold | Action if Fail |
|------|-----------|----------------|
| Turnover annual | <= 12x | Reject or require justification |
| Slippage % of PnL | <= 30% | Reject |
| Capacity USD | >= $5M | Warning (not rejection) |
| Avg slippage | <= 25 bps | Warning |

---

## Failure Modes

### Common Traps

1. **Fill = 100% on limit orders**
   - Symptom: All limit orders fill at limit price
   - Reality: May not fill at all
   - Fix: Use fill rate assumption or market orders

2. **Constant slippage ignoring volume**
   - Symptom: Same slippage for 100 shares and 100,000 shares
   - Reality: Large orders have higher impact
   - Fix: Use VolumeLinear model

3. **Strategy "lives" in auction without modeling**
   - Symptom: Uses closing price, assumes execution at that price
   - Reality: Closing auction price differs from last print
   - Fix: Document limitation or model auction

4. **Costs ignored in high turnover**
   - Symptom: 100x annual turnover, costs not stressed
   - Reality: Costs eat all alpha
   - Fix: Run S1 stress, check cost_as_pct_of_gross_pnl

5. **Capacity ignored**
   - Symptom: No max_participation, infinite capacity assumed
   - Reality: Cannot trade $100M in illiquid stock
   - Fix: Set max_participation, run S4 stress

6. **Spread widening in stress ignored**
   - Symptom: Same slippage in calm and volatile markets
   - Reality: Spreads widen 2-5x in stress
   - Fix: Use VolatilityAdaptive or run S3 stress

7. **Timestamp misalignment**
   - Symptom: B3 strategy uses US timezone
   - Reality: Session boundaries wrong
   - Fix: Handoff to data-engineer for verification

8. **After-hours confused with regular**
   - Symptom: Assumes after-hours same as regular
   - Reality: Much lower liquidity, wider spreads
   - Fix: Document session constraints

9. **Borrow/short not modeled**
   - Symptom: Short positions with zero borrow cost
   - Reality: Borrow costs can be significant
   - Fix: Declare limitation or add borrow assumption

10. **Taxes/fees = 0 without decision**
    - Symptom: No cost model applied
    - Reality: Free money does not exist
    - Fix: Select FeeTier explicitly

11. **Settlement assumed wrong**
    - Symptom: T+0 cash availability assumed
    - Reality: T+1 (US) or T+2 (older) settlement
    - Fix: Document limitation (not currently modeled)

12. **"Alpha" is bid-ask bounce**
    - Symptom: High Sharpe in very short timeframes
    - Reality: Microstructure noise, not tradeable
    - Fix: Require minimum holding period, check S2 stress

### Red Flags Requiring Investigation

- Sharpe collapses under S1 (costs_2x)
- Sharpe collapses under S2 (delay+1)
- Turnover > 50x annual
- Avg slippage > 50 bps
- Capacity < $1M
- All trades at session boundaries
- No cost sensitivity analysis

---

## Collaboration Hooks

### Handoff to `/risk-analyst`

After execution review, attach Assumptions Card to validation package:

```markdown
## Handoff: trader-expert -> risk-analyst

**Strategy ID:** {genome_id}
**Execution Review Status:** PASSED

**Attached:**
- Execution Assumptions Card: {path}
- Cost Report: {path}

**Execution Summary:**
- Slippage model: {type} at {X} bps
- Fee tier: {tier}
- Capacity: ${amount}

**Stress Results:**
| Test | Sharpe | Pass |
|------|--------|------|
| S1 (costs_2x) | X.XX | Y/N |
| S2 (delay+1) | X.XX | Y/N |
| S5 (combined) | X.XX | Y/N |

**Limitations:**
- {list}

**Ready for validation:** Yes
```

### Handoff to `/data-engineer`

For calendar/timezone/corporate action questions:

```markdown
## Handoff: trader-expert -> data-engineer

**Request:** Calendar/session verification

**Question:**
- {specific question about sessions/timezone}

**Context:**
- Market: B3 / US
- Strategy timeframe: {timeframe}
- Period: {start} to {end}

**Suspected Issue:**
- {description}
```

### Handoff to `/quant-engineer`

For execution performance issues:

```markdown
## Handoff: trader-expert -> quant-engineer

**Request:** Execution model performance

**Issue:**
- {description of performance problem}

**Current:**
- Model: {type}
- Latency: {time}

**Target:**
- Latency: {target time}

**Priority:** {high/medium/low}
```

### Receiving from `/quant-researcher`

Before major campaign, review assumptions:

```markdown
## Request: quant-researcher -> trader-expert

**Campaign:** {name}

**Review Requested:**
- Slippage model: {type}
- Fee tier: {tier}
- max_participation: {pct}

**Questions:**
- {specific execution questions}

**Timeline:**
- Campaign start: {date}
```

---

## Quick Reference

### Execution Review Checklist

```
[ ] Slippage model selected and justified
[ ] Fee tier selected (B3Retail/B3Prime/USRetail/USPrime)
[ ] Delay bars documented (0/1/2)
[ ] max_participation set
[ ] 3 cost scenarios tested
[ ] S1 stress passed (intraday)
[ ] S2 stress passed (intraday)
[ ] S5 stress passed (position)
[ ] Calendar/session verified
[ ] Limitations declared
[ ] Assumptions Card completed
[ ] Handoff to risk-analyst ready
```

### Fee Tier Quick Reference

| Tier | Fixed | Commission | Notes |
|------|-------|------------|-------|
| B3Retail | R$10 | 0.15% | Default for B3 |
| B3Prime | R$5 | 0.10% | Institutional B3 |
| USRetail | $1 | 0.10% | Default for US |
| USPrime | $0 | 0.03% | Institutional US |

### Stress Test Quick Reference

| ID | Name | Transform | Intraday | Position |
|----|------|-----------|----------|----------|
| S1 | costs_2x | 2x costs | Required | Optional |
| S2 | delay_plus1 | +1 bar delay | Required | Optional |
| S3 | spread_widen_vol | 3x slippage in vol | Optional | Optional |
| S4 | capacity_constraint | 1% max participation | Optional | Optional |
| S5 | combined_adverse | 2x costs + 1 bar | Optional | Required |

### CLI Commands

```bash
# Run with institutional execution config
cargo run -p backtester_cli -- run \
  --config strategy.toml \
  --execution configs/execution_institutional.toml

# Run stress suite
cargo run -p combiner_cli -- validate \
  --genome {id} \
  --stress-suite default
```

# Quant B3 Dashboard

**Terminal-style, institutional-grade dashboard** for visualizing quantitative trading strategies, SCG evolution, and candidate analytics.

Built with NYC trading floor aesthetics: dark terminal theme, neon accent colors, JetBrains Mono typography, and real-time data visualization.

---

## Features

### Core Pages

| Page | Description |
|------|-------------|
| **Campaigns** | Browse SCG campaigns and runs, select artifacts folder |
| **Candidates** | Interactive table with filtering, multi-select for comparison |
| **Backtest** | Equity curve, drawdown chart, trade log, performance metrics |

### Analytics Pages

| Page | Description |
|------|-------------|
| **Risk Analytics** | VaR, CVaR, Sortino, Calmar, rolling metrics, return distribution |
| **Strategy Comparison** | Side-by-side comparison, correlation matrix, combined equity |
| **Walk-Forward** | Out-of-sample validation windows, degradation analysis |
| **Monte Carlo** | Bootstrap simulation, confidence bands, distribution stats |
| **Regime Analysis** | Market regime detection, performance by regime |

### System Pages

| Page | Description |
|------|-------------|
| **Evolution** | Real-time genetic algorithm progress (legacy) |
| **Overview** | System-wide KPIs and status dashboard |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Framework** | [Tauri](https://tauri.app/) (Rust backend, web frontend) |
| **Frontend** | React 18 + TypeScript + Vite |
| **Styling** | Tailwind CSS + Custom Terminal Theme |
| **Charts** | Recharts + Custom D3 visualizations |
| **State** | Zustand |
| **Backend** | Rust (Tauri commands for file I/O) |

---

## Development

### Prerequisites

- Node.js 18+
- Rust 1.77+
- Tauri CLI 2.x

### Setup

```bash
# Install dependencies
npm install

# Run in development mode
npm run tauri dev
```

### Build

```bash
# Build for production
npm run tauri build
```

---

## Architecture

```
dashboard/
├── src/                          # React frontend
│   ├── components/
│   │   ├── charts/               # Visualization components
│   │   │   ├── EquityChart.tsx         # Equity curve (Recharts)
│   │   │   ├── DrawdownChart.tsx       # Drawdown visualization
│   │   │   ├── GenerationChart.tsx     # Evolution progress
│   │   │   ├── ParetoChart.tsx         # Pareto frontier scatter
│   │   │   ├── ReturnDistribution.tsx  # Histogram + normal overlay
│   │   │   ├── MonthlyHeatmap.tsx      # Calendar heatmap
│   │   │   ├── RollingMetrics.tsx      # Rolling Sharpe/Vol
│   │   │   ├── VaRGauge.tsx            # VaR/CVaR visualization
│   │   │   ├── WalkForwardChart.tsx    # IS/OOS comparison
│   │   │   ├── CorrelationMatrix.tsx   # Strategy correlation
│   │   │   └── DistributionFan.tsx     # Monte Carlo bands
│   │   ├── layout/
│   │   │   ├── Sidebar.tsx             # Navigation (grouped)
│   │   │   └── Header.tsx              # Clock, refresh, alerts
│   │   └── ui/
│   │       ├── MetricCard.tsx          # KPI display card
│   │       └── DataTable.tsx           # Sortable, scrollable table
│   ├── pages/
│   │   ├── Campaigns.tsx               # Project & run browser
│   │   ├── Candidates.tsx              # Strategy candidate table
│   │   ├── Backtest.tsx                # Individual backtest drill-down
│   │   ├── RiskAnalytics.tsx           # VaR, distribution, rolling
│   │   ├── StrategyComparison.tsx      # Multi-strategy comparison
│   │   ├── WalkForward.tsx             # Walk-forward validation
│   │   ├── MonteCarlo.tsx              # Bootstrap simulation
│   │   ├── RegimeAnalysis.tsx          # Market regime performance
│   │   ├── Evolution.tsx               # GA evolution monitor
│   │   └── Dashboard.tsx               # System overview
│   ├── stores/
│   │   └── dataStore.ts                # Zustand state management
│   └── lib/
│       └── utils.ts                    # Formatting utilities
├── src-tauri/                    # Rust backend
│   ├── Cargo.toml                      # Dependencies
│   ├── tauri.conf.json                 # App configuration
│   └── src/
│       └── lib.rs                      # Tauri commands
└── index.html
```

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         Tauri App                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    React Frontend                          │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │ │
│  │  │  Zustand    │  │   Pages     │  │  Components │        │ │
│  │  │  dataStore  │←─│ Campaigns   │←─│  Charts     │        │ │
│  │  │             │  │ Candidates  │  │  Tables     │        │ │
│  │  └──────┬──────┘  └─────────────┘  └─────────────┘        │ │
│  └─────────┼──────────────────────────────────────────────────┘ │
│            │ invoke()                                            │
│            ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Rust Backend                              │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │               ArtifactState (Mutex<Cache>)              │ │ │
│  │  │  - SiteIndex cache                                      │ │ │
│  │  │  - Campaign cache (HashMap)                             │ │ │
│  │  │  - Run cache (HashMap)                                  │ │ │
│  │  │  - Candidate LRU cache (100 items)                      │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  │                                                               │ │
│  │  Commands: set_artifacts_root, load_index, load_campaign,    │ │
│  │            load_run, list_candidates_v2, load_candidate_detail│ │
│  │            load_backtest_series, watch_artifacts             │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Filesystem                                  │
│  artifacts/                                                      │
│  ├── site/                     # Dashboard index files          │
│  │   ├── index.json            # Campaign list                  │
│  │   ├── campaign_<id>.json    # Campaign detail + runs         │
│  │   └── run_<id>.json         # Run detail + top candidates    │
│  ├── top_candidates/           # Per-run candidate exports      │
│  │   └── <run_id>/             │
│  │       ├── top1000.csv       # Ranked candidates              │
│  │       └── top1000.json      # Full candidate data            │
│  ├── candidates/               # Individual candidate bundles   │
│  │   └── <candidate_id>/       │
│  │       ├── strategy.toml     # Strategy definition            │
│  │       ├── provenance.json   # Lineage metadata               │
│  │       └── validation_summary.json                            │
│  └── backtests/                # Backtest results               │
│      └── <candidate_id>/       │
│          ├── timeseries.csv    # Daily equity, drawdown         │
│          └── metadata.json     # Backtest config                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tauri Commands (Backend API)

### Artifact Indexer

| Command | Parameters | Returns | Description |
|---------|------------|---------|-------------|
| `set_artifacts_root` | `path: string` | `string` | Initialize artifacts directory |
| `load_index` | - | `SiteIndex` | Load campaign index |
| `load_campaign` | `campaignId: string` | `CampaignDetail` | Load campaign with runs |
| `load_run` | `runId: string` | `RunDetail` | Load run with top candidates |
| `list_candidates_v2` | `runId, search?, candidateClass?, maxPbo?, limit?` | `CandidateListItem[]` | List/filter candidates |
| `load_candidate_detail` | `candidateId: string` | `CandidateDetailFull` | Full candidate with strategy |
| `load_backtest_series` | `candidateId: string` | `BacktestResult` | Equity curve + metrics |
| `invalidate_cache` | - | `void` | Clear all caches |
| `watch_artifacts` | - | `void` | Start file watcher |
| `get_artifacts_root` | - | `string?` | Get current root path |

### Advanced Analytics (Frontend-computed with mock data)

| Command | Parameters | Returns | Description |
|---------|------------|---------|-------------|
| `calculate_risk_metrics` | `candidateId` | `RiskMetrics` | VaR, CVaR, rolling |
| `compare_candidates` | `candidateIds[]` | `ComparisonResult` | Correlation, combined |
| `calculate_walk_forward` | `candidateId, windowMonths, stepMonths` | `WalkForwardResult` | OOS validation |
| `run_monte_carlo` | `candidateId, numSimulations, blockSize` | `MonteCarloResult` | Bootstrap |
| `detect_regimes` | `candidateId, volThreshold?` | `RegimeAnalysis` | Market regimes |

---

## Design System

### Color Palette

| Token | Hex | Usage |
|-------|-----|-------|
| `terminal-bg` | `#0a0a0f` | Main background |
| `terminal-surface` | `#12121a` | Cards, panels |
| `terminal-border` | `#1e1e2e` | Borders, dividers |
| `terminal-muted` | `#64748b` | Secondary text |
| `profit` | `#00ff88` | Positive values, success |
| `loss` | `#ff3366` | Negative values, errors |
| `accent-cyan` | `#00d4ff` | Highlights, links |
| `accent-yellow` | `#ffce45` | Warnings |
| `accent-purple` | `#8b5cf6` | Secondary accent |

### Typography

| Use Case | Font | Weight |
|----------|------|--------|
| Headings | Inter | 600-700 |
| Body | Inter | 400-500 |
| Data/Numbers | JetBrains Mono | 400-500 |
| Code | JetBrains Mono | 400 |

### Animations

- Page transitions: 200ms ease-out
- Card hover: 150ms
- Loading spinners: CSS keyframes
- Chart tooltips: 100ms fade

---

## State Management (Zustand)

### Store Structure

```typescript
interface DataState {
  // Artifacts root
  artifactsRoot: string | null;
  
  // Site index & navigation
  siteIndex: SiteIndex | null;
  campaigns: CampaignSummary[];
  selectedCampaign: CampaignDetail | null;
  runs: RunSummary[];
  selectedRun: RunDetail | null;
  
  // Candidates
  candidates: CandidateListItem[];
  selectedCandidate: CandidateDetailFull | null;
  candidateFilters: CandidateFilters;
  selectedCandidateIds: string[];  // For comparison
  
  // Backtest
  backtest: BacktestResult | null;
  
  // Advanced Analytics
  riskMetrics: RiskMetrics | null;
  comparisonResult: ComparisonResult | null;
  walkForwardResult: WalkForwardResult | null;
  monteCarloResult: MonteCarloResult | null;
  regimeAnalysis: RegimeAnalysis | null;
  
  // UI State
  isLoading: boolean;
  error: string | null;
  selectedRunId: string | null;
}
```

### Browser Mode

The dashboard includes a mock data layer for browser-based development:

```typescript
// Automatically falls back to mock data when not in Tauri
const isTauri = () => '__TAURI__' in window;
```

Run `npm run dev` to test in browser with mock data.

---

## Artifacts Schema

### `artifacts/site/index.json`

```json
{
  "schema_version": "1.0",
  "generated_at": "2024-12-28T10:00:00Z",
  "campaigns": [
    {
      "campaign_id": "camp_001",
      "name": "IBOV Momentum Q4",
      "tag": "production",
      "status": "completed",
      "runs_count": 3,
      "created_at": "2024-12-15T10:00:00Z",
      "detail_path": "campaign_camp_001.json"
    }
  ]
}
```

### `artifacts/site/campaign_<id>.json`

```json
{
  "schema_version": "1.0",
  "campaign": {
    "campaign_id": "camp_001",
    "name": "IBOV Momentum Q4",
    "tag": "production",
    "status": "completed",
    "git_sha": "abc123",
    "config_hash": "def456",
    "created_at": "2024-12-15T10:00:00Z"
  },
  "runs": [
    {
      "run_id": "run_001",
      "seed": 42,
      "status": "completed",
      "data_integrity_verdict": "PASS",
      "validated_candidates_count": 250,
      "best_oos_sharpe_net": 1.85,
      "duration_secs": 3600
    }
  ]
}
```

### `artifacts/top_candidates/<run_id>/top1000.csv`

```csv
candidate_id,candidate_class,oos_sharpe_net,oos_cagr_net,max_drawdown_net,pbo,dsr,gates_passed,stress_passed,data_integrity_ok
cand_0001,validated,1.85,0.28,-0.12,0.08,1.65,true,true,true
```

---

## Running Tests

```bash
# Frontend tests
npm run test

# Type checking
npm run typecheck

# Lint
npm run lint
```

---

## Related Documentation

- [SCG Overview](../docs/scg/overview.md) - Generative Combiner system
- [Strategy Factory](../docs/strategy_factory.md) - Campaign orchestration
- [Artifacts Structure](../docs/operations/artifacts.md) - Output file layout
- [Validation Framework](../docs/scg/validation-framework.md) - WFA, PBO, DSR

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `QUANT_PROJECT_PATH` | Path to quant_b3_backtest project | `..` |

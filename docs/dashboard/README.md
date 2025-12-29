# Dashboard Tauri - Documentação Técnica

**Versão**: 2.0.0  
**Última Atualização**: 2025-12-28  
**Framework**: Tauri 2.x + React 18 + TypeScript

---

## Visão Geral

O Dashboard é uma aplicação desktop institucional para visualização de estratégias quantitativas, construída com estética de terminal de trading NYC.

### Características

- **Terminal Theme** - Background escuro, cores neon, tipografia monospace
- **Rust Backend** - Leitura eficiente de artefatos via Tauri commands
- **State Management** - Zustand com cache LRU
- **Real-time Updates** - File watcher para hot-reload de dados
- **Browser Fallback** - Mock data para desenvolvimento sem Tauri

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                      TAURI APPLICATION                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    REACT FRONTEND                           │ │
│  │                                                              │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │ │
│  │  │ Pages   │  │ Charts  │  │ UI Comp │  │ Stores  │        │ │
│  │  │         │  │         │  │         │  │         │        │ │
│  │  │Campaign │  │Equity   │  │MetricCard│ │dataStore│        │ │
│  │  │Candidate│  │Drawdown │  │DataTable │ │         │        │ │
│  │  │Backtest │  │Rolling  │  │          │ │         │        │ │
│  │  │Risk     │  │VaR      │  │          │ │         │        │ │
│  │  │Compare  │  │Heatmap  │  │          │ │         │        │ │
│  │  │WFA      │  │Pareto   │  │          │ │         │        │ │
│  │  │Monte    │  │Distrib  │  │          │ │         │        │ │
│  │  │Regimes  │  │         │  │          │ │         │        │ │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │ │
│  │       │            │            │            │              │ │
│  │       └────────────┴────────────┴────────────┘              │ │
│  │                           │                                  │ │
│  │                    invoke() / listen()                       │ │
│  └───────────────────────────┼──────────────────────────────────┘ │
│                              │                                    │
│  ┌───────────────────────────┴──────────────────────────────────┐ │
│  │                    RUST BACKEND                              │ │
│  │                                                               │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │               ArtifactState (Managed State)              │ │ │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │ │ │
│  │  │  │ SiteIndex   │  │ Campaigns   │  │ Candidates      │  │ │ │
│  │  │  │ (Option)    │  │ (HashMap)   │  │ (LRU Cache)     │  │ │ │
│  │  │  └─────────────┘  └─────────────┘  └─────────────────┘  │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  │                                                               │ │
│  │  Commands: set_artifacts_root, load_index, load_campaign,    │ │
│  │            load_run, list_candidates_v2, load_candidate_detail│ │
│  │            load_backtest_series, watch_artifacts             │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Estrutura de Diretórios

```
dashboard/
├── src/                          # React frontend
│   ├── App.tsx                   # Router principal
│   ├── main.tsx                  # Entry point
│   ├── index.css                 # Global styles + Tailwind
│   │
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Sidebar.tsx       # Navegação agrupada (Core/Analytics/System)
│   │   │   └── Header.tsx        # Clock, refresh, alerts
│   │   │
│   │   ├── charts/
│   │   │   ├── EquityChart.tsx         # Recharts Line
│   │   │   ├── DrawdownChart.tsx       # Recharts Area
│   │   │   ├── GenerationChart.tsx     # Evolution progress
│   │   │   ├── ParetoChart.tsx         # Scatter plot
│   │   │   ├── ReturnDistribution.tsx  # Histogram + normal
│   │   │   ├── MonthlyHeatmap.tsx      # Calendar heatmap
│   │   │   ├── RollingMetrics.tsx      # Rolling Sharpe/Vol
│   │   │   ├── VaRGauge.tsx            # VaR/CVaR visual
│   │   │   ├── WalkForwardChart.tsx    # IS/OOS bars
│   │   │   ├── CorrelationMatrix.tsx   # Strategy correlation
│   │   │   └── DistributionFan.tsx     # Monte Carlo bands
│   │   │
│   │   ├── ui/
│   │   │   ├── MetricCard.tsx    # KPI card component
│   │   │   └── DataTable.tsx     # Sortable, scrollable table
│   │   │
│   │   ├── AlertsPanel.tsx       # Notifications panel
│   │   ├── CandidateDetail.tsx   # Candidate drawer
│   │   └── ExportModal.tsx       # Export dialog
│   │
│   ├── pages/
│   │   ├── Campaigns.tsx         # Project/run browser
│   │   ├── Candidates.tsx        # Strategy candidate table
│   │   ├── Backtest.tsx          # Backtest drilldown
│   │   ├── RiskAnalytics.tsx     # VaR, distribution, rolling
│   │   ├── StrategyComparison.tsx# Multi-strategy compare
│   │   ├── WalkForward.tsx       # Walk-forward validation
│   │   ├── MonteCarlo.tsx        # Bootstrap simulation
│   │   ├── RegimeAnalysis.tsx    # Market regime analysis
│   │   ├── Evolution.tsx         # GA evolution monitor
│   │   └── Dashboard.tsx         # System overview
│   │
│   ├── stores/
│   │   └── dataStore.ts          # Zustand state + Tauri commands
│   │
│   └── lib/
│       └── utils.ts              # Formatting utilities
│
├── src-tauri/                    # Rust backend
│   ├── Cargo.toml                # Dependencies
│   ├── tauri.conf.json           # App configuration
│   └── src/
│       └── lib.rs                # Tauri commands (1080 lines)
│
├── index.html                    # HTML entry with loading screen
├── tailwind.config.js            # Terminal theme configuration
├── vite.config.ts                # Vite configuration
├── package.json                  # NPM scripts
└── README.md                     # Quick start guide
```

---

## Páginas

### Core Pages

| Page | Componente | Descrição |
|------|------------|-----------|
| **Campaigns** | `Campaigns.tsx` | Seleção de pasta de artefatos, browse de campanhas e runs |
| **Candidates** | `Candidates.tsx` | Tabela de candidatos com filtros, multi-select para comparação |
| **Backtest** | `Backtest.tsx` | Equity curve, drawdown, métricas detalhadas do backtest |

### Analytics Pages

| Page | Componente | Descrição |
|------|------------|-----------|
| **Risk Analytics** | `RiskAnalytics.tsx` | VaR, CVaR, Sortino, Calmar, rolling metrics, distribuição |
| **Comparison** | `StrategyComparison.tsx` | Comparação lado-a-lado, matriz de correlação, equity combinado |
| **Walk-Forward** | `WalkForward.tsx` | Janelas IS/OOS, degradation ratio, consistency score |
| **Monte Carlo** | `MonteCarlo.tsx` | Bootstrap simulation, bandas de confiança, distribuições |
| **Regimes** | `RegimeAnalysis.tsx` | Detecção de regimes, performance por regime |

### System Pages

| Page | Componente | Descrição |
|------|------------|-----------|
| **Evolution** | `Evolution.tsx` | Monitor de evolução do algoritmo genético |
| **Dashboard** | `Dashboard.tsx` | KPIs do sistema, status geral |

---

## Componentes de Charts

### Recharts-based

| Componente | Tipo | Uso |
|------------|------|-----|
| `EquityChart` | LineChart | Equity curve do backtest |
| `DrawdownChart` | AreaChart | Visualização de drawdown |
| `GenerationChart` | ComposedChart | Estatísticas por geração |
| `ParetoChart` | ScatterChart | Fronteira de Pareto |
| `RollingMetrics` | LineChart | Sharpe/volatilidade rolling |
| `VaRGauge` | Custom | Gauge de VaR/CVaR |
| `ReturnDistribution` | BarChart | Histograma de retornos |
| `MonthlyHeatmap` | Custom SVG | Heatmap de retornos mensais |
| `CorrelationMatrix` | Custom SVG | Matriz de correlação |
| `WalkForwardChart` | BarChart | Barras IS/OOS por janela |
| `DistributionFan` | AreaChart | Bandas de confiança Monte Carlo |

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

### Actions

```typescript
// Artifact Indexer
setArtifactsRoot: (path: string) => Promise<void>;
loadIndex: () => Promise<void>;
loadCampaign: (campaignId: string) => Promise<void>;
loadRun: (runId: string) => Promise<void>;
listCandidates: (runId: string, filters?) => Promise<void>;
loadCandidateDetail: (candidateId: string) => Promise<void>;
loadBacktest: (candidateId: string) => Promise<void>;

// Multi-select
toggleCandidateSelection: (candidateId: string) => void;
clearCandidateSelection: () => void;

// Advanced Analytics
loadRiskMetrics: (candidateId: string) => Promise<void>;
compareCandidates: (candidateIds: string[]) => Promise<void>;
loadWalkForward: (candidateId: string, windowMonths?, stepMonths?) => Promise<void>;
runMonteCarlo: (candidateId: string, numSimulations?, blockSize?) => Promise<void>;
detectRegimes: (candidateId: string, volThreshold?) => Promise<void>;

// File Watcher
startWatcher: () => Promise<void>;
invalidateCache: () => Promise<void>;
```

---

## Rust Backend (Tauri Commands)

### Tipos Principais

```rust
// Site index
pub struct SiteIndex {
    pub schema_version: String,
    pub generated_at: String,
    pub campaigns: Vec<CampaignSummary>,
}

// Campaign summary
pub struct CampaignSummary {
    pub campaign_id: String,
    pub name: String,
    pub tag: String,
    pub status: String,
    pub runs_count: u32,
    pub created_at: String,
}

// Run detail
pub struct RunDetail {
    pub schema_version: String,
    pub run: RunInfo,
    pub metrics: RunMetrics,
    pub top_candidates: Vec<TopCandidateEntry>,
    pub exports: RunExports,
}

// Candidate detail
pub struct CandidateDetailFull {
    pub candidate_id: String,
    pub display_name: String,
    pub candidate_class: String,
    pub strategy_blocks: Vec<PipelineBlock>,
    pub strategy_toml: Option<String>,
    pub oos_sharpe_net: Option<f64>,
    pub pbo: Option<f64>,
    pub dsr: Option<f64>,
    // ...
}

// Backtest result
pub struct BacktestResult {
    pub available: bool,
    pub candidate_id: String,
    pub timeseries: Vec<TimeseriesPoint>,
    pub metrics: Option<BacktestMetrics>,
}
```

### Cache Strategy

```rust
pub struct ArtifactCache {
    pub artifacts_root: PathBuf,
    pub index: Option<SiteIndex>,           // Single cached index
    pub campaigns: HashMap<String, CampaignDetail>,  // Campaign cache
    pub runs: HashMap<String, RunDetail>,            // Run cache
    pub candidates: LruCache<String, CandidateDetailFull>,  // LRU(100)
}
```

### Commands API

| Command | Input | Output |
|---------|-------|--------|
| `set_artifacts_root` | `path: String` | `String` |
| `load_index` | - | `SiteIndex` |
| `load_campaign` | `campaignId: String` | `CampaignDetail` |
| `load_run` | `runId: String` | `RunDetail` |
| `list_candidates_v2` | `runId, search?, candidateClass?, maxPbo?, limit?` | `Vec<CandidateListItem>` |
| `load_candidate_detail` | `candidateId: String` | `CandidateDetailFull` |
| `load_backtest_series` | `candidateId: String` | `BacktestResult` |
| `invalidate_cache` | - | `()` |
| `watch_artifacts` | - | `()` |

---

## Design System

### Cores

```css
:root {
  --terminal-bg: #0a0a0f;
  --terminal-surface: #12121a;
  --terminal-border: #1e1e2e;
  --terminal-muted: #64748b;
  --profit: #00ff88;
  --loss: #ff3366;
  --accent-cyan: #00d4ff;
  --accent-yellow: #ffce45;
  --accent-purple: #8b5cf6;
}
```

### Tipografia

| Uso | Font | Weight |
|-----|------|--------|
| Headings | Inter | 600-700 |
| Body | Inter | 400-500 |
| Data/Numbers | JetBrains Mono | 400-500 |
| Code | JetBrains Mono | 400 |

### Classes CSS

```css
.card { @apply bg-terminal-surface border border-terminal-border rounded-lg p-4; }
.card-elevated { @apply bg-terminal-surface border border-terminal-border rounded-lg p-4 shadow-lg; }
.metric-label { @apply text-sm text-terminal-muted uppercase tracking-wide; }
.metric-value { @apply text-2xl font-mono font-semibold; }
```

---

## Desenvolvimento

### Pré-requisitos

- Node.js 18+
- Rust 1.77+
- Tauri CLI 2.x

### Setup

```bash
cd dashboard
npm install
npm run tauri dev
```

### Build

```bash
npm run tauri build
```

### Browser Mode

Para desenvolvimento sem Tauri:

```bash
npm run dev  # Usa mock data automaticamente
```

O store detecta se está rodando em Tauri:

```typescript
const isTauri = () => '__TAURI__' in window;
```

---

## Integração com SCG

O dashboard consome os artefatos gerados pelo Strategy Factory:

```
artifacts/
├── site/           # Índices para navegação
├── candidates/     # Bundles de candidatos
├── top_candidates/ # Exports CSV/JSON
└── backtests/      # Timeseries
```

Ver [Artefatos de Output](../operations/artifacts.md) para detalhes do schema.

---

## Eventos do Frontend

Navegação entre páginas via CustomEvents:

```typescript
// Navegar para outra página
window.dispatchEvent(new CustomEvent('navigate', { detail: 'candidates' }));

// File watcher event
await listen<{ paths: string[] }>('artifacts_changed', (event) => {
  // Refresh data
});
```

---

## Roadmap

- [ ] Three.js 3D Pareto visualization
- [ ] WebSocket para real-time updates durante evolução
- [ ] Export PDF de relatórios
- [ ] Dark/Light theme toggle
- [ ] Keyboard shortcuts
- [ ] Strategy replay com visualização tick-by-tick





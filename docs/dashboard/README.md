# Dashboard - Documentação Técnica

**Versão**: 4.0.0  
**Última Atualização**: 2026-01-18  
**Framework**: Tauri 2.x / Express + React 18 + TypeScript

---

## Visão Geral

O Dashboard é uma aplicação institucional para visualização e controle de estratégias quantitativas, com estética de terminal de trading NYC.

### Modos de Execução

| Modo | Framework | Backend | Real-time | Uso |
|------|-----------|---------|-----------|-----|
| **Desktop** | Tauri 2.x | Rust + Filesystem | Tauri Events | Produção local |
| **Browser** | Express | Node.js + Neon DB | SSE + Polling | Desenvolvimento |
| **VPS** | DEFERRED | - | - | See `docs/ops/local_only_policy.md` |

### Características

- **Terminal Theme** - Background escuro, cores neon, tipografia monospace
- **Dual-Mode** - Funciona em Tauri (desktop) ou Browser (local). VPS DEFERRED.
- **Unified Command Layer** - Abstração única para todos os modos
- **State Management** - Zustand com cache LRU
- **Real-time Updates** - SSE com fallback para polling
- **Status Badge** - Indicador "Live" ou "Offline"
- **Neon Integration** - PostgreSQL na nuvem para persistência
- **SSE Reconnection** - Suporte a Last-Event-ID para replay

---

## Arquitetura

### Desktop Mode (Tauri)

```
┌─────────────────────────────────────────────────────────────────┐
│                      TAURI APPLICATION                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    REACT FRONTEND                           ││
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        ││
│  │  │ Pages   │  │ Charts  │  │ UI Comp │  │ Stores  │        ││
│  │  │Cockpit  │  │Equity   │  │MetricCard│ │cockpit  │        ││
│  │  │Campaign │  │Drawdown │  │DataTable │ │dataStore│        ││
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        ││
│  │       └────────────┴────────────┴────────────┘              ││
│  │                           │                                  ││
│  │                    lib/commands.ts                           ││
│  │                    invoke() / listen()                       ││
│  └───────────────────────────┼──────────────────────────────────┘│
│                              │                                   │
│  ┌───────────────────────────┴──────────────────────────────────┐│
│  │                    RUST BACKEND (src-tauri)                  ││
│  │  ArtifactState, SCG Control, File Watcher                    ││
│  └───────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    Local Filesystem (artifacts/)
```

### Browser Mode (Express + Neon)

```
┌─────────────────────────────────────────────────────────────────┐
│                    BROWSER APPLICATION                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    REACT FRONTEND                           ││
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        ││
│  │  │ Pages   │  │ Charts  │  │ UI Comp │  │ Stores  │        ││
│  │  │Cockpit  │  │Equity   │  │MetricCard│ │cockpit  │        ││
│  │  │Campaign │  │Drawdown │  │DataTable │ │dataStore│        ││
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        ││
│  │       └────────────┴────────────┴────────────┘              ││
│  │                           │                                  ││
│  │                    lib/commands.ts                           ││
│  │                    fetch() + SSE                             ││
│  └───────────────────────────┼──────────────────────────────────┘│
└──────────────────────────────┼──────────────────────────────────┘
                               │ HTTP + SSE
┌──────────────────────────────┴──────────────────────────────────┐
│                    EXPRESS API SERVER                            │
│                       server.js                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  REST Endpoints │  │  SSE Events     │  │  SCG Control    │  │
│  │                 │  │  + Reconnection │  │                 │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└──────────────────────────────┼──────────────────────────────────┘
                               │
          ┌────────────────────┴────────────────────┐
          ▼                                         ▼
   Local Filesystem                          Neon PostgreSQL
    (artifacts/)                               (cloud DB)
```

### VPS Production - DEFERRED

> **NOTA**: VPS deployment is DEFERRED. See `docs/ops/local_only_policy.md`.
> Current target: **Local Ubuntu workstation with Browser Mode**.

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
│   │   │   ├── Sidebar.tsx       # Navegação agrupada
│   │   │   └── Header.tsx        # Clock, refresh, status badge
│   │   │
│   │   ├── charts/               # Recharts + D3
│   │   │   ├── EquityChart.tsx
│   │   │   ├── DrawdownChart.tsx
│   │   │   ├── GenerationChart.tsx
│   │   │   └── ...
│   │   │
│   │   ├── ui/
│   │   │   ├── MetricCard.tsx
│   │   │   ├── DataTable.tsx
│   │   │   ├── TooltipInfo.tsx   # Info tooltips
│   │   │   └── BloombergTooltip.tsx
│   │   │
│   │   ├── GlossaryOverlay.tsx   # Glossário de termos
│   │   ├── StrategyPipeline.tsx  # Visualização de blocos
│   │   ├── WFAAnalysis.tsx       # Walk-Forward Analysis
│   │   ├── StressAnalysis.tsx    # Stress test results
│   │   └── ...
│   │
│   ├── pages/
│   │   ├── Cockpit.tsx           # SCG control panel
│   │   ├── Campaigns.tsx         # Campaign browser
│   │   ├── Candidates.tsx        # Candidate table
│   │   ├── Backtest.tsx          # Backtest drilldown
│   │   ├── RiskAnalytics.tsx     # VaR, distribution
│   │   ├── StrategyComparison.tsx# Multi-compare
│   │   ├── WalkForward.tsx       # WFA visualization
│   │   ├── MonteCarlo.tsx        # Bootstrap simulation
│   │   ├── RegimeAnalysis.tsx    # Market regimes
│   │   ├── Evolution.tsx         # GA monitor
│   │   └── Dashboard.tsx         # System overview
│   │
│   ├── stores/
│   │   ├── cockpitStore.ts       # SCG run control + SSE status
│   │   └── dataStore.ts          # Artifact navigation
│   │
│   ├── config/
│   │   └── defaults.ts           # Cockpit presets/gates
│   │
│   └── lib/
│       ├── commands.ts           # Unified command layer + SSE
│       ├── platform.ts           # Mode detection + config
│       ├── ranking.ts            # Candidate ranking
│       └── utils.ts              # Formatters
│
├── server.js                     # Express API server (Browser Mode)
│
├── src-tauri/                    # Rust backend (Desktop Mode)
│   ├── Cargo.toml
│   ├── tauri.conf.json
│   └── src/lib.rs
│
├── index.html
├── tailwind.config.js
├── vite.config.ts
└── package.json
```

---

## Páginas

### Core Pages

| Page | Componente | Descrição |
|------|------------|-----------|
| **Cockpit** | `Cockpit.tsx` | Painel de controle para SCG runs |
| **Campaigns** | `Campaigns.tsx` | Browser de campanhas e runs |
| **Candidates** | `Candidates.tsx` | Tabela de candidatos com filtros |
| **Backtest** | `Backtest.tsx` | Equity curve, drawdown, métricas |
| **Hall of Fame** | `HallOfFame.tsx` | Estratégias elite promovidas |

### Analytics Pages

| Page | Componente | Descrição |
|------|------------|-----------|
| **Risk Analytics** | `RiskAnalytics.tsx` | VaR, CVaR, Sortino, rolling |
| **Comparison** | `StrategyComparison.tsx` | Multi-strategy compare |
| **Walk-Forward** | `WalkForward.tsx` | IS/OOS validation |
| **Monte Carlo** | `MonteCarlo.tsx` | Bootstrap simulation |
| **Regimes** | `RegimeAnalysis.tsx` | Regime detection |

### System Pages

| Page | Componente | Descrição |
|------|------------|-----------|
| **Evolution** | `Evolution.tsx` | GA evolution monitor |
| **Dashboard** | `Dashboard.tsx` | System overview |
| **Miner Control** | `MinerControl.tsx` | Controle do OMP 24/7 |
| **Strategy Selector** | `StrategySelector.tsx` | Seletor de estratégias |
| **Audit Report** | `AuditReport.tsx` | Relatórios de auditoria |

### Config Pages

| Page | Componente | Descrição |
|------|------------|-----------|
| **Config Universe** | `ConfigUniverse.tsx` | Configuração de universo |
| **Config Budget** | `ConfigBudget.tsx` | Configuração de compute |
| **Config Gates** | `ConfigGates.tsx` | Configuração de gates |
| **Config Trading** | `ConfigTrading.tsx` | Configuração de trading |

---

## Status Badge & Real-time

### SSE Connection Status

O header exibe um badge de status:

| Badge | Cor | Significado |
|-------|-----|-------------|
| **Live** | Verde | SSE conectado |
| **Offline** | Vermelho | SSE desconectado |

### SSE Features

- **Event Buffering**: Últimos 100 eventos armazenados
- **Reconnection**: Suporte a Last-Event-ID para replay
- **Health Tracking**: Contador de erros consecutivos
- **Fallback**: Polling automático após 3 falhas SSE

```typescript
// cockpitStore.ts
const sse = createSSEConnection(
  (event) => handleEvent(event),
  (error) => {
    if (failCount >= 3) activatePolling();
  },
  () => set({ sseConnected: true })
);
```

---

## Cockpit - Controle SCG

O Cockpit é o painel central para orquestração de runs do SCG. Ver [cockpit.md](cockpit.md) para documentação completa.

### Funcionalidades

- **Presets**: Rapid (3min), Institutional (15min), Exhaustive (1h)
- **Compute Budget**: Time slider, workers/intensidade
- **Risk Gates**: Sharpe mínimo, PBO máximo, stress tests
- **Ranking Methods**: Institutional, Pareto, Sharpe, Risk-Adjusted
- **Live Progress**: SSE com fallback para polling
- **Status Badge**: "Live" (verde) ou "Offline" (vermelho)
- **Error Handling**: Estados de erro com retry
- **Top Strategies**: Tabela rankeada com drilldown

### cockpitStore

```typescript
interface CockpitState {
  // Configuration
  config: CockpitConfig;
  viewMode: 'basic' | 'advanced';
  rankingMethod: RankingMethodKey;
  
  // Run state
  runStatus: RunStatus;
  currentRunId: string | null;
  progress: RunProgress | null;
  
  // Results
  topCandidates: RankedCandidate[];
  selectedCandidateId: string | null;
  
  // SSE status
  sseConnected: boolean;
  
  // Actions
  setPreset: (preset: PresetKey) => void;
  startRun: () => Promise<void>;
  stopRun: () => Promise<void>;
  loadTopCandidates: (runId: string) => Promise<void>;
  subscribeToProgress: () => () => void;
  setSseConnected: (connected: boolean) => void;
}
```

---

## Unified Command Layer

A camada `lib/commands.ts` abstrai diferenças entre Tauri e Browser modes:

```typescript
import { cmd } from './lib/commands';

// Funciona igual em todos os modos
const index = await cmd.loadIndex();
const candidates = await cmd.listCandidates(runId);
await cmd.startScgRun(config);
```

### API Disponível

| Comando | Descrição |
|---------|-----------|
| `loadIndex()` | Carrega índice de campanhas |
| `loadCampaign(id)` | Carrega detalhes da campanha |
| `loadRun(id)` | Carrega detalhes do run |
| `listCandidates(runId, opts)` | Lista candidatos com filtros |
| `loadCandidateDetail(id)` | Carrega detalhes do candidato |
| `loadBacktestSeries(id)` | Carrega timeseries do backtest |
| `startScgRun(config)` | Inicia SCG run |
| `stopScgRun(runId)` | Para SCG run |
| `getRunStatus(runId)` | Obtém progresso do run |

### SSE Connection

```typescript
import { createSSEConnection } from './lib/commands';

const sse = createSSEConnection(
  (event) => console.log('Event:', event),
  (error) => console.error('Error:', error),
  () => console.log('Reconnected!')
);
```

---

## Platform Detection

```typescript
import { platform, config } from './lib/platform';

// Detecção
platform.isTauri    // true se Tauri desktop
platform.isBrowser  // true se browser mode
platform.isDev      // true se desenvolvimento
platform.isProd     // true se produção

// Endpoints (auto-configurados)
config.apiBase      // "/api" (prod) ou "http://localhost:3001/api" (dev)
config.sseEndpoint  // "/api/events" (prod) ou "http://localhost:3001/api/events" (dev)
```

---

## Browser Mode - API Server

O `server.js` fornece uma API REST para browser mode. Ver [api-server.md](api-server.md) para referência completa.

### Endpoints Principais

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/api/index` | GET | Índice de campanhas |
| `/api/campaign/:id` | GET | Detalhes da campanha |
| `/api/candidates/:runId` | GET | Lista candidatos |
| `/api/cockpit-candidates/:runId` | GET | Candidatos para Cockpit |
| `/api/candidate/:id` | GET | Detalhes do candidato |
| `/api/backtest/:id` | GET | Timeseries do backtest |
| `/api/scg/start` | POST | Inicia SCG run |
| `/api/scg/progress/:id` | GET | Progresso do run |
| `/api/events` | GET | SSE stream |

### Executar

```bash
cd dashboard
node server.js      # API em http://localhost:3001
npm run dev         # Frontend em http://localhost:5173
```

---

## VPS Deployment - DEFERRED

> **NOTA**: VPS deployment is DEFERRED. See `docs/ops/local_only_policy.md`.

For historical reference only: [vps-deployment.md](vps-deployment.md)

---

## State Management

### Stores

| Store | Arquivo | Responsabilidade |
|-------|---------|------------------|
| `cockpitStore` | `cockpitStore.ts` | SCG run control + SSE status |
| `dataStore` | `dataStore.ts` | Artifact navigation |

### dataStore

```typescript
interface DataState {
  artifactsRoot: string | null;
  siteIndex: SiteIndex | null;
  campaigns: CampaignSummary[];
  selectedCampaign: CampaignDetail | null;
  candidates: CandidateListItem[];
  selectedCandidate: CandidateDetailFull | null;
  backtest: BacktestResult | null;
  isLoading: boolean;
  error: string | null;
}
```

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

---

## Desenvolvimento

### Pré-requisitos

- Node.js 18+
- Rust 1.77+ (para Tauri)
- Tauri CLI 2.x (para desktop)

### Desktop Mode

```bash
cd dashboard
npm install
npm run tauri dev
```

### Browser Mode

```bash
cd dashboard
npm install
node server.js &     # Terminal 1: API
npm run dev          # Terminal 2: Frontend
```

### Build Desktop

```bash
npm run tauri build
```

### Build for Production

```bash
npm run build
```

---

## Integração com Neon DB

O Browser mode usa Neon PostgreSQL para persistência:

### Variável de Ambiente

```bash
DATABASE_URL=postgresql://user:pass@host/neondb?sslmode=require
```

### Tabelas Utilizadas

| Tabela | Descrição |
|--------|-----------|
| `scg_campaigns` | Campanhas registradas |
| `scg_runs` | Runs de cada campanha |
| `scg_candidates` | Candidatos descobertos |

---

## Documentação Relacionada

- [Cockpit](cockpit.md) - Painel de controle SCG
- [Hall of Fame](hall-of-fame.md) - Estratégias elite
- [Miner Control](miner-control.md) - Controle OMP
- [API Server](api-server.md) - Referência da API REST
- [VPS Deployment](vps-deployment.md) - DEFERRED (historical reference)
- [Artefatos](../operations/artifacts.md) - Estrutura de output

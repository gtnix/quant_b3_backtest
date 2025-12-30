# Dashboard - Documentação Técnica

**Versão**: 3.0.0  
**Última Atualização**: 2025-12-29  
**Framework**: Tauri 2.x / Express + React 18 + TypeScript

---

## Visão Geral

O Dashboard é uma aplicação institucional para visualização e controle de estratégias quantitativas, com estética de terminal de trading NYC.

### Modos de Execução

| Modo | Framework | Backend | Uso |
|------|-----------|---------|-----|
| **Desktop** | Tauri 2.x | Rust + Filesystem | Produção local |
| **Browser** | Express | Node.js + Neon DB | Desenvolvimento/Demo |

### Características

- **Terminal Theme** - Background escuro, cores neon, tipografia monospace
- **Dual Mode** - Funciona em Tauri (desktop) ou Browser (API server)
- **Unified Command Layer** - Abstração única para ambos os modos
- **State Management** - Zustand com cache LRU
- **Real-time Updates** - Tauri Events ou SSE (Server-Sent Events)
- **Neon Integration** - PostgreSQL na nuvem para browser mode

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
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└──────────────────────────────┼──────────────────────────────────┘
                               │
          ┌────────────────────┴────────────────────┐
          ▼                                         ▼
   Local Filesystem                          Neon PostgreSQL
    (artifacts/)                               (cloud DB)
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
│   │   │   ├── Sidebar.tsx       # Navegação agrupada
│   │   │   └── Header.tsx        # Clock, refresh, alerts
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
│   │   └── ...
│   │
│   ├── pages/
│   │   ├── Cockpit.tsx           # SCG control panel (NEW)
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
│   │   ├── cockpitStore.ts       # SCG run control (NEW)
│   │   └── dataStore.ts          # Artifact navigation
│   │
│   ├── config/
│   │   └── defaults.ts           # Cockpit presets/gates
│   │
│   └── lib/
│       ├── commands.ts           # Unified command layer
│       ├── platform.ts           # Mode detection
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
├── netlify/                      # Netlify deployment
│   └── functions/
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

---

## Cockpit - Controle SCG

O Cockpit é o painel central para orquestração de runs do SCG. Ver [cockpit.md](cockpit.md) para documentação completa.

### Funcionalidades

- **Presets**: Rapid (3min), Institutional (15min), Exhaustive (1h)
- **Compute Budget**: Time slider, workers/intensidade
- **Risk Gates**: Sharpe mínimo, PBO máximo, stress tests
- **Ranking Methods**: Institutional, Pareto, Sharpe, Risk-Adjusted
- **Live Progress**: Geração, Sharpe, candidatos
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
  
  // Actions
  setPreset: (preset: PresetKey) => void;
  startRun: () => Promise<void>;
  stopRun: () => Promise<void>;
  loadTopCandidates: (runId: string) => Promise<void>;
}
```

---

## Unified Command Layer

A camada `lib/commands.ts` abstrai diferenças entre Tauri e Browser:

```typescript
import { cmd } from './lib/commands';

// Funciona igual em ambos os modos
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

---

## Platform Detection

```typescript
import { platform, capabilities } from './lib/platform';

// Detecção
platform.isTauri    // true se Tauri desktop
platform.isBrowser  // true se browser mode
platform.isDev      // true se desenvolvimento

// Capabilities
capabilities.nativeDialog   // Diálogos nativos
capabilities.directFS       // Acesso ao filesystem
capabilities.realTimeUpdates // Updates em tempo real
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

## Real-time Updates

### Tauri Mode

```typescript
import { listen } from '@tauri-apps/api/event';

await listen('scg-progress', (event) => {
  console.log('Progress:', event.payload);
});
```

### Browser Mode (SSE)

```typescript
import { createSSEConnection } from './lib/commands';

const sse = createSSEConnection((event) => {
  if (event.type === 'scg-progress') {
    console.log('Progress:', event);
  }
});
```

---

## State Management

### Stores

| Store | Arquivo | Responsabilidade |
|-------|---------|------------------|
| `cockpitStore` | `cockpitStore.ts` | SCG run control |
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

### Deploy Netlify

```bash
npm run build
netlify deploy --prod
```

---

## Integração com Neon DB

O browser mode usa Neon PostgreSQL para persistência:

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
- [API Server](api-server.md) - Referência da API REST
- [Artefatos](../operations/artifacts.md) - Estrutura de output

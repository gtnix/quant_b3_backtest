# Quant B3 Backtester - Documentação Técnica

**Versão**: 3.3.0  
**Última Atualização**: 2025-12-30  
**Status**: Produção (VPS + Neon)

---

## Visão Geral

Sistema de backtesting institucional para o mercado B3 (Brasil) e US construído em Rust, com dois subsistemas principais:

1. **Backtester Engine** - Motor de simulação determinístico de alta performance
2. **Sistema Combinador Generativo (SCG)** - Descoberta evolutiva de estratégias via algoritmos genéticos

### Princípios Fundamentais

| Princípio | Descrição |
|-----------|-----------|
| **Determinismo** | Mesmos inputs → outputs bit-identical |
| **Performance** | Hot path zero-alloc, até 124x speedup via SoA |
| **Precisão** | Cálculos financeiros com `rust_decimal` |
| **Auditabilidade** | Rastreabilidade total de decisões e artefatos |
| **Rigor Anti-Overfitting** | Walk-Forward, PBO, DSR integrados |

---

## Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          QUANT B3 BACKTESTER                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   CAMADA DE ORQUESTRAÇÃO                        │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │   │
│  │  │  backtester_cli │  │  combiner_cli   │  │ Strategy Factory│  │   │
│  │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │   │
│  └───────────┼────────────────────┼────────────────────┼───────────┘   │
│              │                    │                    │               │
│  ┌───────────┼────────────────────┼────────────────────┼───────────┐   │
│  │           │     MOTOR DE EVOLUÇÃO (SCG)             │           │   │
│  │           │    ┌────────────────────────────────────┴──┐        │   │
│  │           │    │  combiner_engine                      │        │   │
│  │           │    │  - Population Generator               │        │   │
│  │           │    │  - Evolution Engine (GA + Pareto)     │        │   │
│  │           │    │  - Hall of Fame                       │        │   │
│  │           │    └───────────────┬───────────────────────┘        │   │
│  │           │                    │                                │   │
│  │           │    ┌───────────────┴───────────────────────┐        │   │
│  │           │    │  combiner_core                        │        │   │
│  │           │    │  - StrategyGenome, BlockGene          │        │   │
│  │           │    │  - MultiObjectiveFitness (SIMD)       │        │   │
│  │           │    │  - PopulationFitnessSoA               │        │   │
│  │           │    └───────────────────────────────────────┘        │   │
│  └───────────┼─────────────────────────────────────────────────────┘   │
│              │                                                         │
│  ┌───────────┼─────────────────────────────────────────────────────┐   │
│  │           │         MOTOR DE BACKTESTING                        │   │
│  │  ┌────────┴────────┐  ┌─────────────────┐  ┌─────────────────┐  │   │
│  │  │backtester_engine│  │backtester_strat │  │backtester_intel │  │   │
│  │  │  UnifiedEngine  │  │  Strategy DSL   │  │  Entry/Exit     │  │   │
│  │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │   │
│  │           │                    │                    │           │   │
│  │  ┌────────┴────────┐  ┌────────┴────────┐  ┌────────┴────────┐  │   │
│  │  │backtester_exec  │  │backtester_portf │  │backtester_reports│ │   │
│  │  │  Cost Models    │  │  Portfolio      │  │  Metrics (SIMD) │  │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      CAMADA DE DADOS                            │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │   │
│  │  │   market_data   │  │   datahub_b3    │  │   datahub_us    │  │   │
│  │  │   Calendar/FX   │  │   Scraper B3    │  │   US Provider   │  │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Mapa de Leitura

### Para Novos Desenvolvedores

1. [Visão Geral do Sistema](architecture/system-overview.md)
2. [Mapa de Crates](architecture/crate-map.md)
3. [Referência CLI - Backtester](operations/cli-reference.md)
4. [Referência CLI - Combiner](scg/cli-reference.md)

### Para Quants/Pesquisadores

1. [Catálogo de Blocos](strategies/block-catalog.md)
2. [Execução de Pipeline](strategies/pipeline-execution.md)
3. [SCG Overview](scg/overview.md)
4. [Estrutura do Genoma](scg/genome-structure.md)
5. [Framework de Validação](scg/validation-framework.md)

### Para Engenheiros de Performance

1. [Benchmarks](validation/benchmarks.md)
2. [Fluxo de Dados](architecture/data-flow.md)
3. [Decisões de Design](architecture/design-decisions.md)

### Para Risk/Compliance

1. [Política de Dividendos](policies/dividend-policy.md)
2. [Survivorship Bias](policies/survivorship-bias.md)
3. [Convenções FX](policies/fx-conventions.md)
4. [Data Integrity](data_integrity.md)

### Para Engenheiros de Dados

1. [Documentação de Dados](data/README.md)
2. **[Data Providers Policy](data/data-providers-policy.md)** ← Política oficial
3. [Provider Due Diligence](data/provider-due-diligence.md)
4. [US DataHub Status](data/us-datahub-status.md)

### Para Operações/DevOps

1. [Strategy Factory](strategy_factory.md)
2. [Artefatos de Output](operations/artifacts.md)
3. [Dashboard](dashboard/README.md)
4. [Cockpit - Controle SCG](dashboard/cockpit.md)
5. [API Server (Browser Mode)](dashboard/api-server.md)
6. **[VPS Deployment](dashboard/vps-deployment.md)** ← NOVO

---

## Dashboard Interativo

O sistema inclui um **dashboard institucional** com suporte a três modos de execução:

- **Desktop Mode (Tauri)**: Aplicação nativa com acesso direto ao filesystem
- **Browser Mode (Local)**: Funciona em qualquer navegador via API Server + Neon DB
- **VPS Mode (Production)**: Deploy em VPS com nginx reverse proxy + PM2

```
┌─────────────────────────────────────────────────────────────────┐
│                    QUANT B3 DASHBOARD                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CORE                          ANALYTICS                        │
│  ├── Cockpit                   ├── Risk Analytics               │
│  ├── Campaigns                 ├── Strategy Comparison          │
│  ├── Candidates                ├── Walk-Forward Analysis        │
│  └── Backtest                  ├── Monte Carlo Simulation       │
│                                └── Regime Analysis              │
│  SYSTEM                                                         │
│  ├── Evolution Monitor                                          │
│  └── Overview                                                   │
│                                                                  │
│  STATUS                                                         │
│  └── [Live] ou [Offline] badge com status SSE                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Cockpit - Controle SCG

O **Cockpit** é o painel central para orquestração de runs do SCG:

| Feature | Descrição |
|---------|-----------|
| **Presets** | Rapid (3min), Institutional (15min), Exhaustive (1h) |
| **Compute Budget** | Time slider, workers/intensidade configurável |
| **Risk Gates** | Sharpe mínimo, PBO máximo, stress tests |
| **Ranking Methods** | Institutional, Pareto, Sharpe, Risk-Adjusted |
| **Live Progress** | SSE real-time com fallback para polling |
| **Top Strategies** | Tabela rankeada com drilldown para backtest |
| **Status Badge** | "Live" (verde) ou "Offline" (vermelho) |

Ver [Cockpit Documentation](dashboard/cockpit.md) para detalhes.

### Funcionalidades Principais

| Feature | Descrição |
|---------|-----------|
| **Cockpit** | Controle de SCG runs com presets e gates configuráveis |
| **Campaign Browser** | Navegue por campanhas e runs do SCG |
| **Candidate Explorer** | Tabela interativa com filtros, multi-select |
| **Backtest Drilldown** | Equity curve, drawdown, trade log |
| **Risk Analytics** | VaR, CVaR, rolling Sharpe, distribuição de retornos |
| **Strategy Comparison** | Comparação lado-a-lado, matriz de correlação |
| **Walk-Forward** | Validação out-of-sample visual |
| **Monte Carlo** | Bootstrap simulation, bandas de confiança |
| **Regime Analysis** | Performance por regime de mercado |

### Tecnologia

| Stack | Tecnologia |
|-------|------------|
| Framework | Tauri 2.x (Desktop) / Express (Browser/VPS) |
| Frontend | React 18 + TypeScript + Vite |
| Styling | Tailwind CSS (terminal theme) |
| State | Zustand |
| Charts | Recharts + D3 |
| Database | Neon PostgreSQL (cloud) |
| Real-time | Tauri Events / SSE + Polling fallback |
| Deploy VPS | nginx + PM2 |

### Executar

```bash
# Desktop Mode (Tauri)
cd dashboard
npm install
npm run tauri dev

# Browser Mode (Local)
cd dashboard
npm install
node server.js &       # API em http://localhost:3001
npm run dev            # Frontend em http://localhost:5173

# VPS Mode (Production)
# Ver docs/dashboard/vps-deployment.md
```

Ver [Dashboard README](dashboard/README.md) para documentação completa

---

## Ambiente de Produção (VPS)

### Infraestrutura

| Componente | Tecnologia |
|------------|------------|
| VPS | Vultr vc2-1c-1gb (Ubuntu 24.04) |
| Reverse Proxy | nginx |
| Process Manager | PM2 |
| Database | Neon PostgreSQL (cloud) |
| Auth | HTTP Basic Auth (nginx) |

### Acesso

```
URL: http://149.28.39.194
User: admin
Pass: quant123
```

### Arquitetura VPS

```
┌──────────────────────────────────────────────────────────────┐
│                        INTERNET                               │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                     NGINX (port 80)                           │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  location / → proxy_pass 5173 (vite preview)           │  │
│  │  location /api/ → proxy_pass 3001 (no auth)            │  │
│  │  location /api/events → SSE proxy                      │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────┬────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
┌─────────────────────┐         ┌─────────────────────┐
│  alpha-dashboard    │         │    api-server       │
│  PM2: vite preview  │         │  PM2: node server   │
│  port: 5173         │         │  port: 3001         │
└─────────────────────┘         └──────────┬──────────┘
                                           │
                                           ▼
                                ┌─────────────────────┐
                                │   Neon PostgreSQL   │
                                │   (cloud database)  │
                                └─────────────────────┘
```

---

## Sistema Combinador Generativo (SCG)

O SCG é uma plataforma de descoberta evolutiva de estratégias que utiliza:

- **Algoritmos Genéticos** - Populações de estratégias evoluem via seleção, crossover e mutação
- **Otimização Multi-Objetivo** - Fronteira de Pareto para CAGR, Sharpe, MaxDD
- **SIMD Acceleration** - Cálculo de fitness vetorizado para milhares de estratégias
- **Walk-Forward Validation** - Validação out-of-sample obrigatória
- **Anti-Overfitting** - PBO (Probability of Backtest Overfitting) e DSR (Deflated Sharpe Ratio)

### Componentes SCG

| Crate | Responsabilidade |
|-------|------------------|
| `combiner_core` | Tipos: StrategyGenome, BlockGene, MultiObjectiveFitness, SIMD metrics |
| `combiner_engine` | Evolution Engine, Pareto selection, Hall of Fame |
| `combiner_runner` | Executor paralelo de backtests (rayon) |
| `combiner_cli` | CLI: run, validate, factory commands |

### Fluxo de Evolução

```
População Inicial (N genomas)
         │
         ▼
┌─────────────────────────────────────┐
│     Avaliação Paralela (rayon)      │
│  - Converter genoma → TOML          │
│  - Executar backtest                │
│  - Calcular fitness multi-objetivo  │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     Seleção por Torneio             │
│  - Tournament selection (k=3)       │
│  - Dominância de Pareto             │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     Reprodução                       │
│  - Crossover (block-level, uniform) │
│  - Mutação (parameter, block swap)  │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     Elitismo                        │
│  - Top genomas → Hall of Fame       │
│  - Elite transferida para próxima   │
└─────────────────────────────────────┘
         │
         ▼
    Nova Geração
```

### Comandos Principais

```bash
# Executar evolução
combiner run --config configs/scg.toml --ultra --top-k 10

# Validar top candidatos com Walk-Forward
combiner validate scg_20251228_123456 --top-k 10 --full

# Strategy Factory - Campanhas multi-seed
combiner factory run --campaign configs/campaigns/momentum.toml
combiner factory promote --run run_abc123 --top 3
```

---

## Strategy Factory

Sistema de orquestração para campanhas de descoberta de estratégias:

- **Multi-seed campaigns** - Múltiplas execuções com seeds diferentes para robustez
- **Experiment registry** - Tracking em PostgreSQL (Neon)
- **Resume capability** - Retomar campanhas interrompidas
- **Promotion pipeline** - Research → Candidate → Paper Trading
- **Reproducibility** - Provenance completa com hashes de config/dataset

Ver [Strategy Factory Runbook](strategy_factory.md) para detalhes.

---

## Estrutura da Documentação

```
docs/
├── README.md                    # Este arquivo
├── architecture/                # Arquitetura do sistema
│   ├── system-overview.md      # Visão geral e diagrama
│   ├── crate-map.md            # Crates e responsabilidades
│   ├── data-flow.md            # Fluxo de dados end-to-end
│   ├── design-decisions.md     # ADRs
│   └── omp-specification.md    # Orquestrador de Mineração Perpétua
├── scg/                         # Sistema Combinador Generativo
│   ├── overview.md             # Visão geral do SCG
│   ├── genome-structure.md     # Estrutura do genoma
│   ├── validation-framework.md # WFA, PBO, DSR
│   ├── state-of-the-art.md     # Performance e otimizações
│   └── cli-reference.md        # Comandos combiner
├── dashboard/                   # Dashboard Tauri/Browser/VPS
│   ├── README.md               # Arquitetura e componentes
│   ├── cockpit.md              # Cockpit - Controle SCG
│   ├── api-server.md           # API Server (Browser Mode)
│   └── vps-deployment.md       # Deploy VPS (nginx + PM2) ← NOVO
├── data/                        # Documentação de dados
│   ├── README.md               # Índice e overview
│   ├── data-providers-policy.md # Política oficial de data providers
│   ├── provider-due-diligence.md  # Avaliação de providers
│   └── us-datahub-status.md    # Status do DataHub US
├── components/                  # Especificações técnicas
│   ├── engines.md              # UnifiedEngine
│   ├── strategy-compositor.md  # DSL de estratégias
│   └── performance-engine.md   # Métricas e atribuição
├── operations/                  # Manual de operações
│   ├── cli-reference.md        # Comandos backtester_cli
│   └── artifacts.md            # Estrutura de output/artifacts
├── validation/                  # Relatório de validação
│   ├── determinism.md          # Invariantes
│   └── benchmarks.md           # Baselines de performance
├── strategies/                  # Documentação de estratégias
│   ├── block-catalog.md        # GERADO DO CÓDIGO
│   ├── pipeline-execution.md   # Execução de pipeline
│   └── execution-modes.md      # standard/compiled/fast
├── policies/                    # Políticas de risco
│   ├── dividend-policy.md      # Anti-double-count
│   ├── survivorship-bias.md    # Universe eligibility
│   └── fx-conventions.md       # Multi-currency
├── reference/                   # Referência rápida
│   └── glossary.md             # Glossário
├── data_integrity.md           # Data integrity framework
└── strategy_factory.md         # Factory runbook
```

---

## Convenções Técnicas

| Convenção | Valor | Referência |
|-----------|-------|------------|
| Dias de trading/ano | 252 | `experiment/metrics.rs:TRADING_DAYS_PER_YEAR` |
| Tipo de retorno | Simples | `(P_t - P_{t-1}) / P_{t-1}` |
| Volatilidade | Population std (N) | `metrics.rs:VolatilityType::Population` |
| Taxa livre de risco | 5% a.a. default | `metrics.rs:DEFAULT_RISK_FREE_RATE` |
| Precisão monetária | rust_decimal | `UnifiedEngine` usa `Decimal` |
| Fitness objectives | CAGR, Sharpe, MaxDD | `combiner_core::MultiObjectiveFitness` |
| PBO threshold | ≤ 0.15 | `factory_campaign.toml:max_pbo` |

---

## Comandos Essenciais

### Backtester

```bash
# Build
cargo build --release

# Testes
cargo test --workspace

# Lint
cargo clippy --all-targets -- -D warnings

# Benchmarks
cargo bench --bench strategy_bench

# Executar estratégia
cargo run -p backtester_cli -- run --config configs/strategies/golden_momentum.toml

# Gerar catálogo de blocos
cargo run -p backtester_cli -- generate-catalog --output docs/strategies/block-catalog.md
```

### Combiner (SCG)

```bash
# Executar evolução
cargo run -p combiner_cli -- run --config configs/scg.toml

# Com modo ultra-performance
cargo run -p combiner_cli -- run --config configs/scg.toml --ultra --top-k 25

# Validar candidatos
cargo run -p combiner_cli -- validate scg_20251228 --top-k 10 --full

# Strategy Factory
cargo run -p combiner_cli -- factory run --campaign configs/campaigns/momentum.toml
cargo run -p combiner_cli -- factory list
cargo run -p combiner_cli -- factory show run_abc123
cargo run -p combiner_cli -- factory promote --run run_abc123 --top 3
```

---

## Workspace Structure

| Grupo | Crate | Responsabilidade |
|-------|-------|------------------|
| **Core** | `backtester_core` | Tipos fundamentais, traits, eventos |
| **Core** | `backtester_io` | Data ingestion, mmap |
| **Engine** | `backtester_engine` | UnifiedEngine (simulação) |
| **Engine** | `backtester_portfolio` | Estado do portfólio |
| **Engine** | `backtester_execution` | Cost models, slippage |
| **Engine** | `backtester_reports` | Métricas SIMD |
| **Strategy** | `backtester_strategy` | Strategy Factory (DSL) |
| **Strategy** | `backtester_intelligence` | Entry/Exit engines, performance |
| **SCG** | `combiner_core` | Genome, Fitness, SIMD metrics |
| **SCG** | `combiner_engine` | Evolution, Pareto, Hall of Fame |
| **SCG** | `combiner_runner` | Parallel executor |
| **SCG** | `combiner_cli` | CLI + Factory commands |
| **Data** | `market_data` | Calendar, FX, universe |
| **Tests** | `backtester_tests` | Integration, determinism |

---

## Links

- **Versão Rust**: 1.75+
- **Workspace**: 14 crates
- **Arquitetura Detalhada**: [generative_combiner_architecture.md](generative_combiner_architecture.md)
- **Produção VPS**: http://149.28.39.194 (admin/quant123)

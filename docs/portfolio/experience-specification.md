# Especificação Técnica de Experiência - Quant B3 Backtester

**Versão**: 1.0.0  
**Data**: 2025-12-28  
**Propósito**: Documentar tecnicamente a experiência de construção deste sistema de backtesting institucional.

> **NOTA ANTI-ALUCINAÇÃO**: Todas as afirmações neste documento são baseadas exclusivamente nos arquivos de documentação lidos. Itens marcados com `[TBD]` indicam informações não encontradas nas fontes primárias.

---

## Entregável A: Especificação da Experiência (1ª Pessoa)

### 1. Contexto e Objetivo

Construí um sistema de backtesting institucional para os mercados B3 (Brasil) e US, com o objetivo de simular estratégias quantitativas de trading com alta precisão e performance. O sistema foi projetado para atender a padrões institucionais de rigor estatístico, eliminando vieses comuns como lookahead bias, survivorship bias e double-counting de dividendos.

O projeto evoluiu para incluir um **Sistema Combinador Generativo (SCG)**, uma plataforma de descoberta evolutiva de estratégias que utiliza Algoritmos Genéticos para explorar automaticamente o espaço de combinações de blocos de estratégia e seus parâmetros.

**Evidência**: `docs/README.md` - "Sistema de backtesting institucional para o mercado B3 (Brasil) e US construído em Rust, com dois subsistemas principais: 1. Backtester Engine - Motor de simulação determinístico de alta performance 2. Sistema Combinador Generativo (SCG) - Descoberta evolutiva de estratégias via algoritmos genéticos"

---

### 2. Stack e Arquitetura

Optei por Rust como linguagem principal pela tríade: performance, segurança de memória e precisão numérica. O workspace é composto por **14 crates especializados**, divididos em dois subsistemas:

| Grupo | Crates | Responsabilidade |
|-------|--------|------------------|
| **Core** | `backtester_core`, `backtester_io` | Tipos fundamentais, traits, data ingestion |
| **Engine** | `backtester_engine`, `backtester_portfolio`, `backtester_execution`, `backtester_reports` | Simulação, portfolio, custos, métricas SIMD |
| **Strategy** | `backtester_strategy`, `backtester_intelligence` | DSL TOML, 19 blocos, Entry/Exit engines, WFA |
| **SCG** | `combiner_core`, `combiner_engine`, `combiner_runner`, `combiner_cli` | Genoma, fitness, evolução, Pareto, factory |
| **Data** | `market_data` | Calendários B3/US, FX, universe |
| **Tests** | `backtester_tests` | Determinismo, invariantes, benchmarks |

**Evidência**: `docs/architecture/crate-map.md` - "O sistema é organizado em um workspace Rust com 14 crates especializados"

A arquitetura em camadas separa claramente:
- **CLI Layer**: `backtester_cli`, `combiner_cli`
- **Strategy Factory**: DSL declarativa em TOML com Compositor e BlockRegistry
- **Intelligence Layer**: EntryEngine, ExitEngine, PerformanceEngine, Orchestrator
- **Engine Layer**: UnifiedEngine com precisão Decimal
- **Core Layer**: Tipos fundamentais, Portfolio, Execution
- **I/O Layer**: Data loading, normalização, caching

**Evidência**: `docs/architecture/system-overview.md` - Diagrama de arquitetura em camadas

---

### 3. Motor de Simulação

O `UnifiedEngine` é o motor canônico de simulação, substituindo o deprecated `SimulationEngine`. Implementei características críticas:

- **Precisão Decimal**: Cálculos financeiros usando `rust_decimal` para evitar erros de ponto flutuante
- **Anti-Double-Count**: Política que separa preços ajustados (signals) de preços raw (valuation)
- **DualPriceBar**: Estrutura com `adjusted_close` para indicadores e `raw_close` para mark-to-market
- **Determinismo Garantido**: Mesmos inputs → outputs bit-identical

```rust
pub struct DualPriceBar {
    pub adjusted_close: Decimal,  // Para signals/indicators
    pub raw_close: Decimal,       // Para valuation
    // ...
}
```

**Evidência**: `docs/policies/dividend-policy.md` - "Se usar preços ajustados para valuation E adicionar dividendos como cashflow, dividendos são contados duas vezes... A Solução: Signals/Indicators: Adjusted | Mark-to-Market: Raw"

O invariante chave é: `equity_raw(T) + Σ dividends(0..T) ≈ equity_adjusted(T)`

**Evidência**: `docs/policies/dividend-policy.md` - "Invariante Chave"

---

### 4. Sistema Combinador Generativo (SCG)

Projetei o SCG para resolver limitações da otimização manual de estratégias:

| Problema Manual | Solução SCG |
|-----------------|-------------|
| Exploração limitada | Populações de centenas de estratégias evoluem simultaneamente |
| Viés de confirmação | Walk-Forward Analysis obrigatório |
| Ciclos lentos | SIMD + rayon para paralelização massiva |

**Estrutura do Genoma**:

```rust
struct StrategyGenome {
    id: Uuid,
    genes: Vec<Gene>,  // BlockGene ou ParamGene
    fitness: Option<MultiObjectiveFitness>,
}

struct MultiObjectiveFitness {
    cagr: f64,
    max_drawdown: f64,
    sharpe_ratio: f64,
}
```

**Operadores Genéticos**:
- **Tournament Selection** (k=3): Seleciona menos dominados via Pareto
- **Block-Level Crossover**: Troca blocos inteiros entre pais
- **Parameter Mutation**: Ruído gaussiano em valores de parâmetros

**Evidência**: `docs/scg/overview.md` - "O SCG resolve esses problemas através de: Exploração Sistemática, Rigor Estatístico (WFA, PBO, DSR), Ultra-Performance (SIMD, rayon)"

**PopulationFitnessSoA**: Implementei layout Structure-of-Arrays para batch processing SIMD, mantendo dados de fitness contíguos para eficiência de cache.

**Evidência**: `docs/architecture/crate-map.md` - "PopulationFitnessSoA - Layout SoA para batch processing SIMD"

---

### 5. Validação Anti-Overfitting

Este é o diferencial institucional do sistema. Implementei um framework multi-camadas:

**Walk-Forward Analysis (WFA)**:
```
Passo 1:  [═══IS═══][OOS]
Passo 2:           [═══IS═══][OOS]
Passo 3:                    [═══IS═══][OOS]
```

**Pipeline de Validação**:
```
Stage A: Avaliação Rápida (GROSS)
    │
    ▼
Gate 1: Sharpe GROSS > 0.5?
    │
    ▼
Stage B: Validação Completa (NET)
    │
    ▼
Gate 2: OOS Sharpe NET > 0.5?
    │
    ▼
Gate 3: PBO < 0.15?
    │
    ▼
Gate 4: Stress passed >= 4/5?
    │
    ▼
CANDIDATO VALIDADO
```

**Técnicas Implementadas**:

| Técnica | Propósito |
|---------|-----------|
| **CPCV** | Todas as combinações de blocos para eliminar viés de seleção |
| **PBO** | Probabilidade de Backtest Overfitting - threshold ≤ 0.15 |
| **DSR** | Deflated Sharpe Ratio ajustado pelo número de trials |
| **Stress Testing** | 5 cenários (HighSlippage, HighCosts, DelayedExecution, LowLiquidity, AdverseConditions) |

**Evidência**: `docs/scg/validation-framework.md` - "Gate 3: PBO < 0.15?... Thresholds Institucionais: < 0.10 Baixa probabilidade de overfitting"

**Métricas de Degradação IS→OOS**:

| Degradation | Interpretação |
|-------------|---------------|
| < 20% | Excelente - estratégia robusta |
| 20-40% | Aceitável |
| 40-60% | Preocupante - possível overfitting |
| > 60% | Crítico - provável overfitting |

**Evidência**: `docs/scg/validation-framework.md` - Tabela de degradação

---

### 6. Data Integrity

Implementei um sistema de gates de integridade que bloqueia execução se dados falharem auditoria:

**Checks Implementados**:
1. **Temporal Integrity**: Timestamps monotônicos, sem duplicatas, gap analysis
2. **Lookahead Policy**: `delay_bars >= 1` obrigatório
3. **Corporate Action**: Detecção de jumps >30% sem metadata
4. **Survivorship**: Validação de `PointInTime` vs `Static` universe

**UniverseRangeProvider** (V1): Usa `min_date`/`max_date` de dados disponíveis
**TimelineEligibilityProvider** (V2): Usa `listing_date`/`delisting_date` de eventos reais

**Invariantes**:
- **INV-001 No Resurrection**: Ativo com `max_date = 2020-12-31` NÃO aparece após 2020
- **INV-002 No Time Travel**: Ativo com `min_date = 2021-08-05` NÃO aparece antes dessa data

**Evidência**: `docs/data_integrity.md` - "The Data Integrity System ensures that strategy backtests are not contaminated by data issues such as lookahead bias, corporate action artifacts, or temporal inconsistencies."

**Evidência**: `docs/policies/survivorship-bias.md` - "INV-001: No Resurrection... INV-002: No Time Travel"

---

### 7. Observabilidade e Artefatos

Cada run gera artefatos padronizados para reprodutibilidade e auditoria:

```
output/experiments/<run_id>/
├── metadata.json    # Configuração e contexto
├── metrics.json     # Métricas de performance
├── timeseries.csv   # Curva de equity
└── trace.jsonl      # Trace de execução
```

**Strategy Factory** gera bundles de candidatos validados:

```
artifacts/candidates/<candidate_id>/
├── strategy.toml           # Configuração executável
├── execution_config.toml   # Modelo de custos
├── validation_summary.json # NET metrics, PBO, stress
├── provenance.json         # Audit trail completo
└── replay.sh               # Script de replay determinístico
```

**Provenance JSON** inclui:
- `genome_hash`, `config_hash`, `dataset_hash`
- `git_sha`, `git_branch`
- `seed` usado
- `scg_version`

**Evidência**: `docs/operations/artifacts.md` e `docs/strategy_factory.md` - Estrutura de artefatos

---

### 8. Dashboard Tauri

Construí um dashboard desktop institucional com estética de terminal de trading NYC:

**Stack**:
- **Framework**: Tauri 2.x (Rust backend + React frontend)
- **Frontend**: React 18 + TypeScript + Vite
- **State**: Zustand com cache LRU
- **Charts**: Recharts + D3
- **Styling**: Tailwind CSS (terminal theme)

**10 Páginas**:

| Core | Analytics | System |
|------|-----------|--------|
| Campaigns | Risk Analytics | Evolution |
| Candidates | Strategy Comparison | Dashboard |
| Backtest | Walk-Forward |  |
|  | Monte Carlo |  |
|  | Regime Analysis |  |

**11 Componentes de Chart**:
EquityChart, DrawdownChart, GenerationChart, ParetoChart, ReturnDistribution, MonthlyHeatmap, RollingMetrics, VaRGauge, WalkForwardChart, CorrelationMatrix, DistributionFan

**Rust Backend (Tauri Commands)**:
- `load_index`: Carrega índice de campanhas
- `list_candidates_v2`: Lista com filtros
- `load_backtest_series`: Carrega timeseries
- `watch_artifacts`: File watcher para hot-reload

**Evidência**: `docs/dashboard/README.md` - "O Dashboard é uma aplicação desktop institucional para visualização de estratégias quantitativas, construída com estética de terminal de trading NYC."

---

### 9. Resultados e Performance

Os benchmarks demonstram performance excepcional:

| Cenário | Target | Medido | Status |
|---------|--------|--------|--------|
| 1K assets × 100 rebalances | < 10ms | **1.0ms** | ✓ Exceeded |
| 2K assets × 100 rebalances | < 20ms | **1.7ms** | ✓ Exceeded |
| Symbol table lookup (5K) | < 100µs | **51µs** | ✓ Met |
| Engine throughput | > 100K events/s | **485K events/s** | ✓ Exceeded |

**Speedup Fast SoA vs Standard**: **93-124x**

**Evidência**: `docs/validation/benchmarks.md` - Tabela "Targets vs Medido"

**Otimizações Implementadas**:
1. **SoA Layout**: Dados contíguos por campo (cache-friendly)
2. **PreallocBuffers**: Zero alocações após warmup
3. **SIMD Metrics**: Sharpe, MaxDD, Sortino vetorizados
4. **rayon**: Paralelismo de dados para evolução

**ADRs Documentados**: 10 Architecture Decision Records formais, incluindo:
- ADR-001: UnifiedEngine como Engine Canônico
- ADR-002: Anti-Double-Count para Dividendos
- ADR-003: SoA para Hot Path
- ADR-008: Survivorship Bias via Universe Eligibility

**Evidência**: `docs/architecture/design-decisions.md` - 10 ADRs documentados

---

### 10. Lições e Próximos Passos

**Lições Aprendidas**:
1. Determinismo é fundamental para reprodutibilidade de pesquisa quantitativa
2. Anti-overfitting deve ser native, não um afterthought
3. Performance extrema habilita exploração de espaços maiores de estratégias
4. Separação clara de preços (adjusted vs raw) previne erros sutis de PnL

**Roadmap do Dashboard** (documentado):
- [ ] Three.js 3D Pareto visualization
- [ ] WebSocket para real-time updates durante evolução
- [ ] Export PDF de relatórios
- [ ] Strategy replay com visualização tick-by-tick

**Evidência**: `docs/dashboard/README.md` - Seção Roadmap

---

## Entregável B: Bullets de Currículo (8-12 items)

> Formato: Verbo + Impacto + Como + Tech

1. **Arquitetou** sistema de backtesting institucional com **14 crates Rust** organizados em workspace modular, alcançando **485K events/s** de throughput e garantia de determinismo bit-identical.

2. **Projetou** Sistema Combinador Generativo (SCG) usando **Algoritmos Genéticos** com operadores de crossover/mutação em nível de bloco, otimização multi-objetivo via **Fronteira de Pareto** e seleção por torneio.

3. **Implementou** framework anti-overfitting institucional com **Walk-Forward Analysis**, **CPCV**, **PBO** (threshold ≤0.15) e **Deflated Sharpe Ratio**, validando candidatos através de 5 cenários de stress testing.

4. **Otimizou** hot path de seleção de ativos alcançando **93-124x speedup** via layout **Structure-of-Arrays (SoA)**, buffers pré-alocados e funções SIMD vetorizadas para cálculo de métricas.

5. **Eliminou** double-counting de dividendos através de política **Anti-Double-Count** com DualPriceBar, separando preços ajustados (signals) de raw (valuation), validado por invariante matemático.

6. **Preveniu** survivorship bias implementando **UniverseRangeProvider** com validação temporal de elegibilidade, garantindo invariantes No-Resurrection e No-Time-Travel.

7. **Desenvolveu** dashboard desktop institucional com **Tauri (Rust) + React + TypeScript**, incluindo 10 páginas analíticas, 11 componentes de chart e estética de terminal de trading.

8. **Construiu** Strategy Factory para campanhas multi-seed com tracking em **PostgreSQL (Neon)**, promoção pipeline (Research → Candidate → Paper) e provenance completa com hashes.

9. **Documentou** sistema com **10 ADRs formais**, garantindo rastreabilidade de decisões arquiteturais como precisão Decimal, 252 dias de trading/ano e capping de ratios infinitos.

10. **Implementou** Data Integrity System com gates de auditoria que bloqueiam execução em caso de lookahead bias, timestamps não-monotônicos ou corporate actions não tratadas.

---

## Entregável C: Talking Points para Entrevista (10 tópicos)

### 1. Arquitetura de Crates Rust

**Problema**: Como organizar um sistema de backtesting de ~24K LOC para manutenibilidade e performance?

**Decisão**: Workspace Cargo com 14 crates especializados seguindo Single Responsibility.

**Trade-offs**: Mais arquivos de configuração, mas compilação incremental e testes isolados.

**Validação**: Cada crate tem testes unitários; `backtester_tests` valida invariantes cross-crate.

**Risco Mitigado**: Acoplamento; mudanças em `backtester_core` não afetam `combiner_engine`.

---

### 2. Precisão Numérica com rust_decimal

**Problema**: Erros de ponto flutuante em cálculos financeiros podem causar discrepâncias significativas em PnL.

**Decisão**: Usar `rust_decimal::Decimal` em todo `UnifiedEngine` para cálculos monetários.

**Trade-offs**: Performance ligeiramente menor que f64, mas precisão garantida.

**Validação**: Testes de determinismo verificam outputs bit-identical.

**Risco Mitigado**: Diferenças de PnL entre runs devido a erros de arredondamento.

---

### 3. Anti-Double-Count de Dividendos

**Problema**: Usar preços ajustados para valuation E creditar dividendos como cashflow conta retorno duas vezes.

**Decisão**: DualPriceBar com `adjusted_close` para signals, `raw_close` para valuation.

**Trade-offs**: Complexidade adicional (dois tipos de preço) mas retorno econômico correto.

**Validação**: Invariante `equity_raw + Σ dividends ≈ equity_adjusted` testado em `t1_buyhold_economic_return_matches_adjusted`.

**Risco Mitigado**: Métricas de performance infladas artificialmente.

---

### 4. Otimização SoA para Hot Path

**Problema**: Array-of-Structs causa cache misses no loop de seleção de ativos.

**Decisão**: `CandidatesSoA` com dados contíguos por campo + `PreallocBuffers`.

**Trade-offs**: Código duplicado (standard vs fast), nem todos os blocks suportam fast mode.

**Validação**: Benchmarks Criterion mostram 93-124x speedup.

**Risco Mitigado**: Gargalo de performance em populações grandes de SCG.

---

### 5. Walk-Forward Analysis Nativo

**Problema**: Backtests tradicionais não detectam overfitting; performance "perfeita" in-sample não generaliza.

**Decisão**: WFA integrado nativamente com janelas deslizantes IS/OOS.

**Trade-offs**: Computacionalmente mais caro, mas detecta estratégias que não generalizam.

**Validação**: Degradação IS→OOS calculada; >40% é sinal de alerta.

**Risco Mitigado**: Promover estratégias que falharão em produção.

---

### 6. Probability of Backtest Overfitting (PBO)

**Problema**: Quanto mais estratégias testadas, maior a chance de encontrar uma "sortuda".

**Decisão**: Calcular PBO baseado em número de trials, volatilidade, Sharpe observado.

**Trade-offs**: Estratégias genuínas podem ser rejeitadas se threshold muito conservador.

**Validação**: Threshold institucional ≤0.15 documentado.

**Risco Mitigado**: Data snooping; escolher estratégia que é artefato estatístico.

---

### 7. Survivorship Bias Prevention

**Problema**: Backtests que incluem apenas ativos que "sobreviveram" inflam resultados.

**Decisão**: `UniverseRangeProvider` valida `min_date ≤ rebalance_date ≤ max_date`.

**Trade-offs**: Dados de IPO/delisting podem ser imprecisos; V2 usa eventos reais.

**Validação**: Invariantes No-Resurrection e No-Time-Travel testados.

**Risco Mitigado**: Resultados otimistas que não refletem realidade histórica.

---

### 8. Strategy Factory Multi-Seed

**Problema**: Uma única seed pode encontrar estratégia "sortuda"; preciso de robustez.

**Decisão**: Campanhas executam múltiplas seeds (5+) para validar consistência.

**Trade-offs**: Custo computacional multiplicado por número de seeds.

**Validação**: Estratégias que performam bem em múltiplas seeds são mais robustas.

**Risco Mitigado**: Promover estratégia que só funciona com seed específica.

---

### 9. Dashboard Desktop com Tauri

**Problema**: Visualização de dados de estratégias requer interface rica; web introduz latência.

**Decisão**: Tauri (Rust backend) + React frontend; leitura direta de arquivos locais.

**Trade-offs**: Distribuição de app desktop vs simplicidade de web.

**Validação**: File watcher permite hot-reload durante runs de evolução.

**Risco Mitigado**: Pesquisador espera por dados; latência de API.

---

### 10. Data Integrity Gates

**Problema**: Dados com problemas (lookahead, gaps, corporate actions) contaminam backtests.

**Decisão**: Sistema de gates que bloqueia execução de campanha se auditoria falhar.

**Trade-offs**: Pode bloquear runs legítimos se dados incompletos.

**Validação**: Relatório JSON com score 0-1 e lista de hard_fails.

**Risco Mitigado**: Estratégias validadas em dados corrompidos.

---

## Entregável D: One-Pager (PT-BR + EN-US)

### Versão PT-BR (198 palavras)

**Quant B3 Backtester - Sistema Institucional de Pesquisa Quantitativa**

Construí um sistema de backtesting de alta performance em Rust para os mercados B3 e US, composto por 14 crates especializados (~24K LOC). O motor de simulação (`UnifiedEngine`) utiliza precisão decimal e política anti-double-count para cálculo correto de retornos com dividendos.

O **Sistema Combinador Generativo (SCG)** emprega Algoritmos Genéticos com otimização multi-objetivo via Fronteira de Pareto, descobrindo estratégias através de populações que evoluem por seleção, crossover e mutação. A validação anti-overfitting é institucional: Walk-Forward Analysis, CPCV, PBO ≤0.15 e 5 cenários de stress testing.

**Performance**: 93-124x speedup via layout SoA + SIMD, atingindo 485K events/s de throughput no engine. O hot path de seleção processa 1K ativos × 100 rebalances em 1.0ms.

O sistema previne vieses críticos: survivorship bias via UniverseRangeProvider, lookahead via delay_bars obrigatório, e double-counting via DualPriceBar. A Strategy Factory orquestra campanhas multi-seed com tracking em PostgreSQL e provenance completa.

Um dashboard desktop (Tauri + React) com 10 páginas analíticas permite visualização de candidatos, backtest drilldown, análise de risco e comparação de estratégias em estética de terminal institucional.

---

### Versão EN-US (195 words)

**Quant B3 Backtester - Institutional Quantitative Research System**

I built a high-performance backtesting system in Rust for the B3 (Brazil) and US markets, comprising 14 specialized crates (~24K LOC). The simulation engine (`UnifiedEngine`) uses decimal precision and an anti-double-count policy for correct dividend-inclusive return calculations.

The **Generative Combiner System (SCG)** employs Genetic Algorithms with multi-objective optimization via Pareto Frontier, discovering strategies through populations that evolve via selection, crossover, and mutation. Anti-overfitting validation is institutional-grade: Walk-Forward Analysis, CPCV, PBO ≤0.15, and 5 stress testing scenarios.

**Performance**: 93-124x speedup via SoA layout + SIMD, achieving 485K events/s engine throughput. The selection hot path processes 1K assets × 100 rebalances in 1.0ms.

The system prevents critical biases: survivorship bias via UniverseRangeProvider, lookahead via mandatory delay_bars, and double-counting via DualPriceBar. The Strategy Factory orchestrates multi-seed campaigns with PostgreSQL tracking and complete provenance.

A desktop dashboard (Tauri + React) with 10 analytical pages enables candidate visualization, backtest drilldown, risk analysis, and strategy comparison in an institutional terminal aesthetic.

---

## Métricas Documentadas (Fontes)

| Métrica | Valor | Fonte |
|---------|-------|-------|
| Speedup Fast SoA vs Standard | **93-124x** | `docs/validation/benchmarks.md` |
| Engine throughput | **485K events/s** | `docs/validation/benchmarks.md` |
| Fast SoA 1K assets | **1.0ms** (100 rebalances) | `docs/validation/benchmarks.md` |
| Fast SoA 2K assets | **1.7ms** (100 rebalances) | `docs/validation/benchmarks.md` |
| Symbol table lookup 5K | **51µs** | `docs/validation/benchmarks.md` |
| Crates no workspace | **14** | `docs/architecture/crate-map.md` |
| Blocos de estratégia | **19** | `docs/architecture/system-overview.md` |
| Páginas do dashboard | **10** | `docs/dashboard/README.md` |
| Componentes de chart | **11** | `docs/dashboard/README.md` |
| PBO threshold institucional | **≤ 0.15** | `docs/scg/validation-framework.md` |
| Dias de trading/ano | **252** | `docs/architecture/design-decisions.md` |
| ADRs documentados | **10** | `docs/architecture/design-decisions.md` |
| LOC estimado | **~24.000** | `docs/architecture/crate-map.md` |

---

## Itens TBD (Não Encontrados nas Fontes)

- Cobertura de testes percentual
- Número exato de testes unitários/integração
- Tempo de desenvolvimento do projeto
- Número de campanhas executadas em produção
- Métricas de estratégias descobertas (Sharpe, CAGR reais)

















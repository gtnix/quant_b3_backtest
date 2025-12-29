# Fluxo de Dados

**Versão**: 2.0.0  
**Última Atualização**: 2025-12-28

## Visão Geral

Este documento descreve o fluxo de dados end-to-end do backtester, desde a configuração TOML até os artefatos de output.

---

## Fluxo Principal

```
┌─────────────────────────────────────────────────────────────────┐
│                    1. CONFIGURAÇÃO                               │
│                                                                  │
│   configs/strategies/my_strategy.toml                           │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ [strategy]                                               │   │
│   │ id = "momentum_v1"                                       │   │
│   │                                                          │   │
│   │ [[pipeline]]                                             │   │
│   │ type = "selection"                                       │   │
│   │ block_id = "momentum"                                    │   │
│   │ params = { top_pct = 20 }                               │   │
│   └─────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    2. PARSING                                    │
│                                                                  │
│   load_strategy_config(path) → StrategyConfig                   │
│   Localização: backtester_strategy/src/config.rs                │
│                                                                  │
│   Output:                                                        │
│   - strategy.id, version, description                           │
│   - pipeline: Vec<PipelineStep>                                 │
│   - rebalance: RebalanceConfig                                  │
│   - constraints: ConstraintsConfig                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    3. RESOLUÇÃO DE BLOCOS                        │
│                                                                  │
│   BlockRegistry::with_builtins()                                │
│   Localização: backtester_strategy/src/registry.rs              │
│                                                                  │
│   Para cada step no pipeline:                                   │
│   - Resolve block_id → Box<dyn StrategyBlock>                   │
│   - Valida params via block.validate_params()                   │
│   - Merge com default_params()                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    4. CARREGAMENTO DE DADOS                      │
│                                                                  │
│   Data Pipeline:                                                 │
│   1. Carregar universe (cache/universe.csv)                     │
│   2. Carregar price bars (cache/bars/)                          │
│   3. Carregar dividends (se habilitado)                         │
│   4. Carregar FX rates (se multi-currency)                      │
│                                                                  │
│   Output: HashMap<String, Vec<Bar>>                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    5. EXECUÇÃO DO PIPELINE                       │
│                                                                  │
│   Para cada rebalance_date:                                     │
│                                                                  │
│   a) Construir StrategyContext                                  │
│      - date, market, capital                                    │
│      - candidates: Vec<StrategyCandidate>                       │
│                                                                  │
│   b) Executar pipeline blocks em ordem:                         │
│      - Selection → filtrar/rankear candidates                   │
│      - Entry → gerar sinais de entrada                          │
│      - Exit → gerar sinais de saída                             │
│      - Sizing → calcular pesos                                  │
│                                                                  │
│   c) Output: CompositorResult                                   │
│      - selected: Vec<String>                                    │
│      - weights: HashMap<String, f64>                            │
│      - signals: Vec<Signal>                                     │
│                                                                  │
│   Localização: backtester_strategy/src/compositor.rs            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    6. ORCHESTRATION                              │
│                                                                  │
│   RebalanceOrchestrator                                         │
│   Localização: backtester_intelligence/src/orchestrator.rs      │
│                                                                  │
│   a) EntryEngine: aplicar gating filters                        │
│      - Tradeability check                                       │
│      - Universe eligibility (survivorship)                      │
│      - Liquidity filter                                         │
│                                                                  │
│   b) ExitEngine: verificar condições de saída                   │
│      - Stop-loss                                                │
│      - Take-profit                                              │
│      - Trailing stop                                            │
│      - Time-based exit                                          │
│                                                                  │
│   c) Order netting: consolidar ordens                           │
│      - Entry orders vs Exit orders                              │
│      - Net orders (buy - sell)                                  │
│                                                                  │
│   Output: Vec<Order>                                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    7. SIMULAÇÃO                                  │
│                                                                  │
│   UnifiedEngine                                                 │
│   Localização: backtester_engine/src/unified.rs                 │
│                                                                  │
│   Para cada dia:                                                │
│   a) Processar dividends (se ex_date == date)                   │
│      - Creditar cashflow                                        │
│      - Usar raw_price para valuation                            │
│                                                                  │
│   b) Executar ordens                                            │
│      - Aplicar slippage                                         │
│      - Aplicar custos (B3: emolumentos)                         │
│      - Atualizar posições                                       │
│                                                                  │
│   c) Mark-to-market                                             │
│      - Calcular equity                                          │
│      - Atualizar drawdown                                       │
│                                                                  │
│   Output: DayResult                                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    8. MÉTRICAS                                   │
│                                                                  │
│   MetricsCalculator                                             │
│   Localização: backtester_strategy/src/experiment/metrics.rs    │
│                                                                  │
│   Inputs:                                                        │
│   - timeseries: Vec<EquityPoint>                                │
│   - trades: Vec<TradeRecord>                                    │
│   - risk_free_rate: f64                                         │
│                                                                  │
│   Output: RunMetrics                                            │
│   - cagr, volatility, sharpe_ratio, sortino_ratio               │
│   - max_drawdown, max_drawdown_duration_days                    │
│   - hit_rate, profit_factor, turnover_annual                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    9. ARTEFATOS                                  │
│                                                                  │
│   ArtifactWriter                                                │
│   Localização: backtester_strategy/src/experiment/writer.rs     │
│                                                                  │
│   output/experiments/<run_id>/                                  │
│   ├── metadata.json    # Configuração e contexto                │
│   ├── metrics.json     # Métricas de performance                │
│   ├── timeseries.csv   # Curva de equity                        │
│   └── trace.jsonl      # Trace de execução                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tipos de Preço (Anti-Double-Count)

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRICE SEPARATION POLICY                       │
│                                                                  │
│   DualPriceBar {                                                │
│       adjusted_close: Decimal,  // Para signals/indicators      │
│       raw_close: Decimal,       // Para valuation               │
│   }                                                              │
│                                                                  │
│   ┌─────────────┬───────────────┬─────────────────────────────┐ │
│   │ Uso         │ Price Type    │ Razão                       │ │
│   ├─────────────┼───────────────┼─────────────────────────────┤ │
│   │ Signals     │ adjusted      │ Retornos contínuos          │ │
│   │ Valuation   │ raw           │ Dividends via cashflow      │ │
│   │ Execution   │ raw           │ Preço real de mercado       │ │
│   └─────────────┴───────────────┴─────────────────────────────┘ │
│                                                                  │
│   Invariante:                                                    │
│   equity_raw + Σ dividends ≈ equity_adjusted                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Modos de Execução

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXECUTION MODES                               │
│                                                                  │
│   ┌───────────┬────────────────┬───────────────────────────────┐│
│   │ Modo      │ Performance    │ Descrição                     ││
│   ├───────────┼────────────────┼───────────────────────────────┤│
│   │ standard  │ Baseline       │ Compositor dinâmico           ││
│   │ compiled  │ 5-10% faster   │ Estratégia pré-compilada      ││
│   │ fast      │ 93-124x faster │ SoA + zero alocações          ││
│   │ auto      │ Best available │ Fast se 100% suportado        ││
│   └───────────┴────────────────┴───────────────────────────────┘│
│                                                                  │
│   Resolução Auto (Determinística):                              │
│   1. Se TODOS blocks têm fast_supported → Fast                  │
│   2. Senão → Compiled                                           │
│                                                                  │
│   Fallback com dividends:                                       │
│   - Fast mode NÃO suporta dividend cashflow                     │
│   - Se enable_dividends + Fast → Compiled (fallback)            │
│   - Registrado em metadata.mode_fallback_reason                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Localização no Código

| Etapa | Crate | Arquivo |
|-------|-------|---------|
| Config parsing | `backtester_strategy` | `src/config.rs` |
| Block resolution | `backtester_strategy` | `src/registry.rs` |
| Pipeline execution | `backtester_strategy` | `src/compositor.rs` |
| Orchestration | `backtester_intelligence` | `src/orchestrator.rs` |
| Simulation | `backtester_engine` | `src/unified.rs` |
| Metrics | `backtester_strategy` | `src/experiment/metrics.rs` |
| Artifacts | `backtester_strategy` | `src/experiment/writer.rs` |




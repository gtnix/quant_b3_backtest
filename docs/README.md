# Quant B3 Backtester - Documentação Técnica

**Versão**: 2.0.0  
**Última Atualização**: 2025-12-28  
**Status**: Produção

## Visão Geral

Sistema de backtesting institucional para o mercado B3 (Brasil) construído em Rust, projetado para:

- **Determinismo**: Mesmos inputs produzem outputs bit-identical
- **Performance**: Hot path com zero alocações, até 124x speedup via SoA
- **Precisão**: Cálculos financeiros com `rust_decimal`
- **Auditabilidade**: Rastreabilidade total de decisões e cálculos

---

## Mapa de Leitura

### Para Novos Desenvolvedores

1. [Visão Geral do Sistema](architecture/system-overview.md)
2. [Mapa de Crates](architecture/crate-map.md)
3. [Referência CLI](operations/cli-reference.md)

### Para Quants/Pesquisadores

1. [Catálogo de Blocos](strategies/block-catalog.md)
2. [Execução de Pipeline](strategies/pipeline-execution.md)
3. [Modos de Execução](strategies/execution-modes.md)

### Para Engenheiros de Performance

1. [Benchmarks](validation/benchmarks.md)
2. [Fluxo de Dados](architecture/data-flow.md)
3. [Decisões de Design](architecture/design-decisions.md)

### Para Risk/Compliance

1. [Política de Dividendos](policies/dividend-policy.md)
2. [Survivorship Bias](policies/survivorship-bias.md)
3. [Convenções FX](policies/fx-conventions.md)

---

## Estrutura da Documentação

```
docs/
├── README.md                    # Este arquivo
├── architecture/                # Arquitetura do sistema
│   ├── system-overview.md      # Visão geral e diagrama
│   ├── crate-map.md            # Crates e responsabilidades
│   ├── data-flow.md            # Fluxo de dados end-to-end
│   └── design-decisions.md     # ADRs
├── components/                  # Especificações técnicas
│   ├── engines.md              # UnifiedEngine
│   ├── entry-exit-pipeline.md  # Entry/Exit engines
│   ├── strategy-compositor.md  # DSL de estratégias
│   └── performance-engine.md   # Métricas e atribuição
├── operations/                  # Manual de operações
│   ├── cli-reference.md        # Comandos CLI
│   ├── configuration.md        # Formato TOML
│   └── artifacts.md            # Artefatos de output
├── validation/                  # Relatório de validação
│   ├── determinism.md          # Invariantes
│   ├── test-coverage.md        # Cobertura de testes
│   └── benchmarks.md           # Baselines de performance
├── strategies/                  # Documentação de estratégias
│   ├── block-catalog.md        # GERADO DO CÓDIGO
│   ├── pipeline-execution.md   # Execução de pipeline
│   └── execution-modes.md      # standard/compiled/fast
├── policies/                    # Políticas de risco
│   ├── dividend-policy.md      # Anti-double-count
│   ├── survivorship-bias.md    # Universe eligibility
│   └── fx-conventions.md       # Multi-currency
├── audits/                      # Audit trail
│   └── duplication-audit.md    # Relatório de auditoria
└── reference/                   # Referência rápida
    └── glossary.md             # Glossário
```

---

## Convenções de Documentação

### Rastreabilidade

Toda seção deve apontar para localização no código:

```markdown
## Localização no Código
- Crate: `backtester_xxx`
- Arquivo: `src/xxx.rs`
- Símbolos: `Foo`, `bar()`, `BAZ_CONST`
```

### Comandos Reproduzíveis

Todos os comandos devem ser copy-paste funcionais:

```bash
# CORRETO: Comando exato
cargo test --package backtester_strategy --test runner_e2e

# INCORRETO: Comando genérico
cargo test <seu_teste>
```

### Marcação de Incertezas

Se algo não pode ser confirmado no código, marcar explicitamente:

```markdown
**DESCONHECIDO**: A razão para X não está documentada no código.
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

---

## Comandos Essenciais

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

---

## Links Externos

- Repositório: `quant_b3_backtest`
- Workspace Rust: 12 crates
- Versão Rust: 1.75+


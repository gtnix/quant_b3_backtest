# Quant B3 Backtester

Sistema de backtesting institucional de alta performance para o mercado B3 (Brasil), construído em Rust.

## Características

- **Determinismo**: Mesmos inputs → outputs bit-identical
- **Performance**: Hot path com zero alocações, até 124x speedup via SoA
- **Precisão**: Cálculos financeiros com `rust_decimal`
- **Auditabilidade**: Rastreabilidade total de decisões e artefatos

## Quick Start

```bash
# Build
cargo build --release

# Testes
cargo test --workspace

# Lint
cargo clippy --all-targets -- -D warnings

# Executar estratégia
cargo run -p backtester_cli -- run --config configs/strategies/golden_momentum.toml
```

## Documentação

**Documentação completa em [`/docs`](docs/README.md)**

| Seção | Descrição |
|-------|-----------|
| [Visão Geral](docs/architecture/system-overview.md) | Arquitetura do sistema |
| [Mapa de Crates](docs/architecture/crate-map.md) | Responsabilidades de cada crate |
| [CLI](docs/operations/cli-reference.md) | Referência de comandos |
| [Blocos](docs/strategies/block-catalog.md) | Catálogo de blocos disponíveis |
| [Políticas](docs/policies/dividend-policy.md) | Políticas de risco |

## Workspace Structure

| Crate | Responsabilidade |
|-------|------------------|
| `backtester_core` | Tipos fundamentais, traits, eventos |
| `backtester_engine` | UnifiedEngine (simulação) |
| `backtester_strategy` | Strategy Factory (DSL) |
| `backtester_intelligence` | Entry/Exit engines, performance |
| `backtester_cli` | Interface CLI |

## Princípios de Design

1. **Determinism-First**: Outputs idênticos para inputs idênticos
2. **Performance-First**: Zero alocações no hot path
3. **Hot Path Sacred**: Sem I/O, sem `dyn Trait` no loop de simulação

## Benchmarks

```bash
# Benchmarks de estratégia
cargo bench --bench strategy_bench

# Benchmarks de engine
cargo bench --bench scenarios_bench
```

| Cenário | Tempo | Speedup |
|---------|-------|---------|
| Standard (1K assets) | 75.7ms | 1x |
| Fast SoA (1K assets) | 1.0ms | **93x** |








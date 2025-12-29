# Benchmarks e Performance

**Versão**: 2.0.0  
**Última Atualização**: 2025-12-28

## Baseline de Performance

### Targets vs Medido

| Cenário | Target | Medido | Status |
|---------|--------|--------|--------|
| 1K assets × 100 rebalances | < 10ms | **1.0ms** | ✓ Exceeded |
| 2K assets × 100 rebalances | < 20ms | **1.7ms** | ✓ Exceeded |
| Symbol table lookup (5K) | < 100µs | **51µs** | ✓ Met |
| Engine throughput | > 100K events/s | **485K events/s** | ✓ Exceeded |

---

## Resultados por Componente

### Engine Scenarios

| Cenário | Eventos | Tempo | Throughput |
|---------|---------|-------|------------|
| Intraday (10 assets × 1K bars) | 10K | 287µs | 34.8M elem/s |
| Daily Swing (200 assets × 252 days) | 50.4K | 21.4ms | 2.36M elem/s |
| Stress Universe (1K assets × 252 days) | 252K | 519ms | 485K elem/s |

### Strategy Factory (Standard)

| Assets | Single Execution | 100 Executions |
|--------|-----------------|----------------|
| 50 | 25µs | - |
| 100 | 57µs | - |
| 200 | 125µs | - |
| 500 | 349µs | - |
| 1000 | 758µs | 75.7ms |

### Fast SoA Pipeline

| Assets | Single Execution | 100 Executions | Speedup |
|--------|-----------------|----------------|---------|
| 500 | 5.7µs | 574µs | - |
| 1000 | 10µs | 1.0ms | **93x** |
| 2000 | 17µs | 1.7ms | **124x** |

### Symbol Table

| Symbols | Build Time | Lookup (all) |
|---------|-----------|--------------|
| 100 | 10µs | 0.95µs |
| 500 | 56µs | 4.7µs |
| 1000 | 103µs | 9.7µs |
| 5000 | 463µs | 51µs |

---

## Executando Benchmarks

### Todos os Benchmarks

```bash
cargo bench
```

### Strategy Benchmarks

```bash
# Todos
cargo bench --bench strategy_bench

# Grupo específico
cargo bench --bench strategy_bench -- "standard_vs_fast"

# Benchmark específico
cargo bench --bench strategy_bench -- "fast_soa/1000"
```

### Engine Benchmarks

```bash
cargo bench --bench scenarios_bench
```

### Ver Resultados HTML

```bash
cargo bench --bench strategy_bench
open target/criterion/report/index.html
```

---

## Baselines e Comparação

### Salvar Baseline

```bash
# Salvar como baseline nomeado
cargo bench --bench strategy_bench -- --save-baseline v1.0

# Salvar como baseline main (usado por CI)
cargo bench --bench strategy_bench -- --save-baseline main
```

### Comparar com Baseline

```bash
# Comparar com baseline salvo
cargo bench --bench strategy_bench -- --baseline v1.0

# Comparar com main
cargo bench --bench strategy_bench -- --baseline main
```

---

## Thresholds de Regressão

| Benchmark | Warning | Fail | Justificativa |
|-----------|---------|------|---------------|
| fast_soa/1000 | +20% | +50% | Hot path, sensível a cache |
| symbol_table/lookup_5000 | +10% | +30% | O(1) lookup, estável |
| Engine throughput | -10% | -25% | Events/sec, menor = pior |

---

## Prevenção de Regressão

### CI Integration

O workflow CI automaticamente:

1. **PRs**: Roda benchmarks e compara com baseline `main`
2. **Main pushes**: Atualiza baseline
3. **Falha CI**: Se "Performance has regressed"

### Atualizando Baseline

Quando performance muda intencionalmente:

```bash
# 1. Verificar performance esperada
cargo bench --bench strategy_bench -- "fast_soa" "symbol_table"

# 2. Revisar resultados
cat target/criterion/fast_soa/1000/new/estimates.json

# 3. Salvar novo baseline
cargo bench --bench strategy_bench -- --save-baseline main

# 4. Commit
git add -A && git commit -m "perf: update benchmark baseline"
```

---

## Profiling

### Hot Path

Caminhos críticos em ordem de impacto:

1. `fast_momentum_select` - Iteração SoA e sorting
2. `fast_equal_weight` - Cálculo de pesos
3. `SimulationEngine::process_event` - Event loop

### Análise de Alocações

```bash
# Verificar zero-alloc com dhat-rs
DHAT_ARGS="--zero-alloc" cargo bench --bench strategy_bench
```

**Esperado**: 0 alocações em funções `fast_*` após warmup.

### Cache Efficiency

```bash
# Usar perf ou cachegrind
perf stat -e cache-misses,cache-references cargo bench -- fast_soa
```

SoA deve mostrar menor taxa de cache miss que AoS.

---

## Comparação Apple-to-Apple

Ao comparar benchmarks, garantir:

1. **Mesmos inputs**: Benchmarks usam asset counts fixos (500, 1000, 2000)
2. **Mesmas iterações**: 100 ciclos de rebalance
3. **Mesmo trabalho**: Standard e Fast executam lógica equivalente
4. **Ambiente consistente**: Fechar outros apps, mesmo CPU governor

---

## Localização no Código

| Componente | Arquivo |
|------------|---------|
| Strategy benchmarks | `crates/backtester_strategy/benches/strategy_bench.rs` |
| Engine benchmarks | `crates/backtester_tests/benches/scenarios_bench.rs` |
| Fast functions | `crates/backtester_strategy/src/fast_context.rs` |
| SoA structures | `crates/backtester_strategy/src/compiled.rs` |




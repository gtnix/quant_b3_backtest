# Modos de Execução

**Versão**: 2.0.0  
**Última Atualização**: 2025-12-28

## Visão Geral

O backtester suporta diferentes modos de execução para balancear flexibilidade e performance.

---

## Modos Disponíveis

| Modo | Descrição | Performance | Uso |
|------|-----------|-------------|-----|
| `standard` | Compositor dinâmico | Baseline | Debug, desenvolvimento |
| `compiled` | Estratégia pré-compilada | 5-10% faster | Produção padrão |
| `fast` | SoA + zero alocações | 93-124x faster | Alto throughput |
| `auto` | Seleciona melhor modo | Varia | Default recomendado |

---

## Standard Mode

Execução dinâmica via `Compositor`.

### Características

- Cria blocos dinamicamente a cada execução
- Parsing de params em runtime
- HashMap lookups no hot path
- Máxima flexibilidade

### Quando Usar

- Debugging e desenvolvimento
- Estratégias complexas ou customizadas
- Quando performance não é crítica

### Uso

```bash
cargo run -p backtester_cli -- run \
  --config my_strategy.toml \
  --execution standard
```

---

## Compiled Mode

Estratégia pré-compilada via `CompiledStrategy`.

### Características

- Blocos resolvidos uma vez na inicialização
- Params convertidos para structs tipados
- Hash de params pré-computado para cache
- Sem criação de blocos no hot path

### Quando Usar

- Produção normal
- Quando nem todos os blocks suportam fast
- Balanço entre performance e flexibilidade

### Uso

```bash
cargo run -p backtester_cli -- run \
  --config my_strategy.toml \
  --execution compiled
```

---

## Fast Mode

Execução otimizada via SoA e zero alocações.

### Características

- `CandidatesSoA`: Dados contíguos por campo
- `PreallocBuffers`: Buffers reutilizados
- Funções `fast_*`: Otimizadas para cache
- Zero alocações após warmup

### Requisitos

**TODOS** os blocks no pipeline devem ter `fast_supported: true`.

### Blocks com Fast Support

| Categoria | Blocos com Fast |
|-----------|-----------------|
| Selection | `momentum`, `low_vol` |
| Entry | (nenhum) |
| Exit | (nenhum) |
| Sizing | `equal_weight` |

### Quando Usar

- Alto throughput (1000s de backtests)
- Pipelines simples (momentum + equal_weight)
- Otimização de parâmetros

### Uso

```bash
# Fast mode (fallback se não suportado)
cargo run -p backtester_cli -- run \
  --config my_strategy.toml \
  --execution fast

# Fast mode strict (erro se não suportado)
cargo run -p backtester_cli -- run \
  --config my_strategy.toml \
  --execution fast \
  --strict
```

---

## Auto Mode (Recomendado)

Seleção automática do melhor modo.

### Algoritmo de Resolução

```
1. Se TODOS blocks têm fast_supported → Fast
2. Senão → Compiled
```

### Determinismo

A resolução é **determinística**: mesma config → mesmo modo.

### Uso

```bash
cargo run -p backtester_cli -- run \
  --config my_strategy.toml \
  --execution auto
```

---

## Fallback com Dividendos

**Fast mode NÃO suporta dividend cashflow tracking.**

Quando `enable_dividends: true` + `Fast mode`:
1. Fallback automático para `Compiled`
2. Registrado em `metadata.mode_fallback_reason`
3. Entrada em `trace.jsonl` com tipo `mode_fallback`

### Exemplo

```json
// metadata.json
{
  "execution_mode": "compiled",
  "mode_fallback_reason": "Dividends enabled but Fast mode does not support dividend cashflow"
}
```

---

## Verificação de Eligibilidade

### Programático

```rust
use backtester_strategy::{CompiledStrategy, BlockRegistry};

let registry = BlockRegistry::with_builtins();
let config = load_strategy_config("my_strategy.toml")?;

// Verificar se pipeline é elegível para fast
let is_fast_eligible = config.pipeline.iter().all(|step| {
    registry.get(&step.block_id)
        .map(|b| b.fast_supported())
        .unwrap_or(false)
});
```

### Via CLI

```bash
# Dry run mostra modo que seria usado
cargo run -p backtester_cli -- run \
  --config my_strategy.toml \
  --dry-run
```

---

## Performance Comparativa

### Benchmark: 1000 assets × 100 rebalances

| Modo | Tempo | Speedup |
|------|-------|---------|
| Standard | 93.9ms | 1x |
| Compiled | ~85ms | ~1.1x |
| Fast | 1.0ms | **93x** |

---

## Arquitetura Interna

### Standard

```
Config → Compositor → BlockRegistry.get() → Block.execute()
                        (cada execução)
```

### Compiled

```
Config → CompiledStrategy::compile() → steps: Vec<CompiledStep>
                (uma vez)
                                            ↓
                                    execute_fast() → step.execute()
                                    (cada execução)
```

### Fast

```
Config → CompiledStrategy + SymbolTable + CandidatesSoA + PreallocBuffers
                (uma vez)
                                            ↓
                                    fast_momentum_select()
                                    fast_equal_weight()
                                    (zero alloc cada execução)
```

---

## Localização no Código

| Componente | Arquivo |
|------------|---------|
| CompiledStrategy | `src/compiled.rs` |
| SymbolTable | `src/compiled.rs` |
| CandidatesSoA | `src/fast_context.rs` |
| PreallocBuffers | `src/fast_context.rs` |
| fast_* functions | `src/fast_context.rs` |
| ExecutionMode enum | `src/experiment/types.rs` |







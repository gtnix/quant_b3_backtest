# Determinismo e Invariantes

**Versão**: 2.0.0  
**Última Atualização**: 2025-12-28

## Princípio Fundamental

**Mesmos inputs → Mesmos outputs (bit-identical)**

O backtester é projetado para produzir resultados idênticos quando executado com a mesma configuração e dados.

---

## Componentes Determinísticos

| Componente | Determinístico? | Notas |
|------------|-----------------|-------|
| Métricas | ✓ Sim | Função pura dos inputs |
| Timeseries | ✓ Sim | Baseado no compositor |
| Config hash | ✓ Sim | SHA256 do arquivo |
| Trades | ✓ Sim | Ordem e valores idênticos |
| Run ID | ✗ Não | UUID v4 (esperado) |
| Timestamp | ✗ Não | UTC timestamp (esperado) |
| Artifact paths | ✗ Não | Contém run_id |

---

## Testes de Determinismo

### test_determinism_same_input_same_output

Executa mesma config duas vezes e verifica:
- Todas as métricas iguais
- Timeseries idênticas
- Trades idênticos
- Config hash idêntico

```bash
cargo test -p backtester_strategy test_determinism_same_input_same_output
```

### test_batch_order_independence

Executa 3 configs em batch e verifica:
- Cada estratégia produz resultados consistentes
- Ordem de execução não afeta resultados individuais

```bash
cargo test -p backtester_strategy test_batch_order_independence
```

### test_timestamp_does_not_affect_metrics

Executa com delay entre runs:
- Timestamps diferem (esperado)
- Run IDs diferem (esperado)
- Métricas idênticas
- Config hash idêntico

```bash
cargo test -p backtester_strategy test_timestamp_does_not_affect_metrics
```

---

## Invariantes do Sistema

### INV-001: Weight Sum ≈ 1.0

**Invariante**: Soma dos pesos deve estar em `1.0 ± WEIGHT_SUM_TOLERANCE`

**Tolerância**: `WEIGHT_SUM_TOLERANCE = 0.001` (0.1%)

**Validação**: `validate_compositor_result()`

```rust
if (weight_sum - 1.0).abs() > WEIGHT_SUM_TOLERANCE {
    return Err(StrictValidationError::InvalidWeightSum { ... });
}
```

### INV-002: No NaN/Inf in Weights

**Invariante**: Nenhum peso pode ser NaN ou Infinity

**Validação**: `validate_compositor_result()`

```rust
if weight.is_nan() {
    return Err(StrictValidationError::NaNWeight(symbol));
}
if weight.is_infinite() {
    return Err(StrictValidationError::InfWeight(symbol));
}
```

### INV-003: No NaN/Inf in Metrics

**Invariante**: Métricas-chave não podem ser NaN ou Infinity

**Validação**: `validate_strict()` em RunMetrics

### INV-004: Anti-Double-Count

**Invariante**: `equity_raw + Σ dividends ≈ equity_adjusted`

**Validação**: `t1_buyhold_economic_return_matches_adjusted`

### INV-005: No Resurrection

**Invariante**: Ativo com `max_date = T` não aparece em rebalances após T

**Validação**: Universe eligibility check

### INV-006: No Time Travel

**Invariante**: Ativo com `min_date = T` não aparece em rebalances antes de T

**Validação**: Universe eligibility check

### INV-007: Selected ⊆ Eligible

**Invariante**: Todo ativo selecionado satisfaz critérios de elegibilidade

**Validação**: Gating pipeline

---

## StrictValidationError

Erros de validação em strict mode:

```rust
pub enum StrictValidationError {
    // Weight-related
    NaNWeight(String),
    InfWeight(String),
    InvalidWeightSum { actual: f64, tolerance: f64 },
    WeightExceedsMax { symbol: String, weight: f64, max: f64 },
    WeightBelowMin { symbol: String, weight: f64, min: f64 },
    TooManyPositions { actual: usize, max: usize },
    
    // Return/metric-related
    NaNReturn(usize),
    InfReturn(usize),
    NaNMetric(String),
    InfMetric(String),
    
    // Pipeline/signal errors
    MissingPrice(String),
    EmptyUniverse { reason: String },
    EmptyPipelineResult { step: usize, reason: String },
    ZeroQuantityOrder { symbol: String },
    MissingExitReason { symbol: String },
    
    // General
    NoTrades,
    EmptyTimeseries,
}
```

---

## Strict Mode

Ativa validações adicionais que falham o run se violadas.

### Habilitar

```bash
cargo run -p backtester_cli -- run \
  --config my_strategy.toml \
  --strict
```

### Validações

1. **Compositor Result**: Pesos válidos, soma correta
2. **Experiment Result**: Métricas válidas, timeseries não-vazia
3. **Fast Mode**: Todos os blocks suportam fast

---

## Fontes de Não-Determinismo (Evitadas)

| Fonte | Como Evitada |
|-------|--------------|
| Floating point ordering | Usar sort estável |
| Random seeds | Seeds fixos/configuráveis |
| HashMap iteration order | Usar BTreeMap para output |
| Time-based logic | Usar datas fixas do dataset |
| External API calls | Cache de dados |

---

## Testes Relacionados

```bash
# Todos os testes de determinismo
cargo test determinism

# Testes de invariantes
cargo test invariants

# Testes anti-look-ahead
cargo test anti_look_ahead

# Testes de strict mode
cargo test strict_validation
```




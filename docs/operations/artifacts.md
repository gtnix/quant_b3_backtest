# Artefatos de Output

**Versão**: 2.0.0  
**Última Atualização**: 2025-12-28

## Visão Geral

Cada execução de experimento produz artefatos padronizados para rastreabilidade e comparação.

### Localização no Código

- **Crate**: `backtester_strategy`
- **Arquivo**: `src/experiment/writer.rs`

---

## Estrutura de Diretórios

```
output/experiments/<run_id>/
├── metadata.json    # Configuração e contexto
├── metrics.json     # Métricas de performance
├── timeseries.csv   # Curva de equity
└── trace.jsonl      # Trace de execução
```

---

## metadata.json

Configuração e contexto da execução.

### Schema

```json
{
  "schema_version": "1.0",
  "run_id": "abc123-def456",
  "config_hash": "sha256:...",
  "strategy_id": "momentum_v1",
  "strategy_version": "1.0.0",
  "crate_version": "0.2.0",
  "timestamp_utc": "2025-01-01T12:00:00Z",
  "dataset_id": "br_stocks_2024",
  "seed": 42,
  "execution_mode": "fast",
  "costs": {
    "trading_fee_pct": 0.001,
    "slippage_pct": 0.0005
  },
  "config_path": "configs/strategies/momentum.toml",
  "duration_ms": 1234,
  "dividends_enabled": true,
  "dividend_policy": {
    "signals_price": "adjusted",
    "valuation_price": "raw",
    "dividends_as_cashflow": true
  },
  "total_dividend_cashflow": "1234.56",
  "dividend_count": 4,
  "mode_fallback_reason": null
}
```

### Campos

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `schema_version` | string | Versão do schema |
| `run_id` | string | UUID único do run |
| `config_hash` | string | SHA256 do arquivo TOML |
| `strategy_id` | string | ID da estratégia |
| `execution_mode` | string | Modo efetivo usado |
| `dividends_enabled` | bool | Dividendos habilitados? |
| `mode_fallback_reason` | string? | Razão de fallback |

---

## metrics.json

Métricas de performance calculadas.

### Schema

```json
{
  "cagr": 0.15,
  "volatility": 0.20,
  "sharpe_ratio": 0.75,
  "sortino_ratio": 1.0,
  "calmar_ratio": 1.5,
  "max_drawdown": -0.10,
  "max_drawdown_duration_days": 30,
  "hit_rate": 0.55,
  "profit_factor": 1.5,
  "turnover_annual": 2.5,
  "total_trades": 120,
  "winning_trades": 66,
  "losing_trades": 54
}
```

### Campos

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `cagr` | f64 | Compound Annual Growth Rate |
| `volatility` | f64 | Volatilidade anualizada |
| `sharpe_ratio` | f64 | (return - rf) / vol |
| `sortino_ratio` | f64 | (return - rf) / downside_vol |
| `calmar_ratio` | f64 | CAGR / abs(max_dd) |
| `max_drawdown` | f64 | Maior drawdown (negativo) |
| `hit_rate` | f64 | Winning / Total trades |
| `profit_factor` | f64 | Gross profit / Gross loss |

---

## timeseries.csv

Série temporal da curva de equity.

### Schema

```csv
date,equity,drawdown,exposure,vol_exante,vol_expost,dividend_cashflow,dividend_cumulative
2024-01-02,100000.00,0.000000,0.950000,0.150000,,0.00,0.00
2024-01-03,100250.00,-0.002500,0.950000,0.150000,0.148000,0.00,0.00
2024-03-15,105000.00,-0.005000,0.920000,0.155000,0.152000,450.00,450.00
```

### Colunas

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `date` | date | Data do snapshot |
| `equity` | decimal | Valor total do portfólio |
| `drawdown` | decimal | Drawdown atual |
| `exposure` | decimal | Exposição (% invested) |
| `vol_exante` | decimal | Vol esperada |
| `vol_expost` | decimal | Vol realizada |
| `dividend_cashflow` | decimal | Dividendo do dia |
| `dividend_cumulative` | decimal | Dividendos acumulados |

---

## trace.jsonl

Trace de execução (JSON Lines).

### Schema

Cada linha é um objeto JSON:

```json
{"step": 0, "block_id": "momentum", "block_type": "selection", "message": "Selected 5 assets", "timestamp_ms": 1234567890, "params_effective": {"lookback_days": 126}}
{"step": 1, "block_id": "equal_weight", "block_type": "sizing", "message": "Weights assigned", "timestamp_ms": 1234567891, "params_effective": {"max_weight": 0.20}}
```

### Tipos de Entrada

| type | Descrição |
|------|-----------|
| `header` | Contexto inicial do run |
| `selection` | Resultado de seleção |
| `entry` | Sinais de entrada |
| `exit` | Sinais de saída |
| `sizing` | Pesos calculados |
| `dividend` | Evento de dividendo |
| `dividend_policy` | Política aplicada |
| `mode_fallback` | Fallback de modo |

### Campos Comuns

| Campo | Descrição |
|-------|-----------|
| `step` | Índice do step no pipeline |
| `block_id` | ID do bloco executado |
| `block_type` | Tipo do bloco |
| `message` | Mensagem descritiva |
| `timestamp_ms` | Timestamp em milissegundos |
| `params_effective` | Parâmetros efetivos usados |

---

## Roundtrip Tests

Testes garantem estabilidade dos artefatos:

```bash
# Testes de roundtrip
cargo test -p backtester_strategy artifact_roundtrip

# Testes de schema
cargo test -p backtester_strategy artifact_schema
```

---

## Backward Compatibility

- `schema_version` permite detectar formato
- Campos novos são opcionais
- Campos removidos: migration guide fornecido


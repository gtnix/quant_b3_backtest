# DataHub US

**Versão**: 1.0.0  
**Última Atualização**: 2026-01-18

## Visão Geral

O `datahub_us` é um módulo Python para ingestão de dados do mercado americano (US). Sincroniza índices, dados diários, intraday e dividendos usando yfinance.

---

## Arquitetura

```
datahub_us/
├── __init__.py
├── __main__.py          # Entry point
├── cli.py               # CLI commands (typer)
├── config.py            # Configuração
├── db.py                # Conexão Neon PostgreSQL (async)
├── intraday.py          # Sync intraday/daily
├── indices.py           # Fetch de composição de índices
├── indices_cli.py       # CLI para índices
├── indices_db.py        # DB operations para índices
├── universe.py          # Universos (S&P 500, etc)
├── router.py            # Router de providers
├── jobs/
│   ├── __init__.py
│   ├── bootstrap.py     # Bootstrap 20 anos
│   ├── repair.py        # Repair gaps
│   ├── sync.py          # Sync Neon → CSV
│   └── update.py        # Update incremental
├── providers/
│   ├── __init__.py
│   ├── base.py          # Provider interface
│   └── yfinance_provider.py  # yfinance implementation
├── qa/
│   ├── __init__.py
│   └── validator.py     # Data quality checks
├── reports/
│   ├── __init__.py
│   └── generator.py     # Status reports
└── storage/
    ├── __init__.py
    └── csv_storage.py   # CSV cache storage
```

---

## Comandos CLI

```bash
python -m datahub_us <COMMAND> [OPTIONS]
```

### Comandos Principais

| Comando | Descrição |
|---------|-----------|
| `bootstrap` | Bootstrap 20 anos de histórico |
| `update` | Atualização incremental |
| `sync` | Sync Neon → CSV cache |
| `db-status` | Status do banco Neon |
| `status` | Status do cache local |
| `repair` | Detectar e reparar gaps |
| `report` | Gerar relatório de status |
| `healthcheck` | Verificar conectividade |

### Comandos de Índices

| Comando | Descrição |
|---------|-----------|
| `indices-sync` | Sincronizar composição de índices |
| `indices-list` | Listar índices disponíveis |
| `indices-show` | Mostrar composição de índice |

### Comandos de Dados

| Comando | Descrição |
|---------|-----------|
| `intraday-sync` | Sync dados intraday |
| `daily-sync` | Sync dados diários |
| `aggregate` | Agregar múltiplos intervalos |
| `dividends-sync` | Sync dividendos históricos |

---

## Comandos Detalhados

### `bootstrap`

Bootstrap completo de 20+ anos de histórico.

```bash
python -m datahub_us bootstrap [OPTIONS]
```

**Opções**:

| Flag | Descrição | Default |
|------|-----------|---------|
| `--sample`, `-s` | Usar amostra de 10 símbolos | false |
| `--sample20` | Usar amostra de 20 símbolos | false |
| `--symbols` | Símbolos específicos (CSV) | S&P 500 |
| `--start` | Data início (YYYY-MM-DD) | 2005-01-01 |
| `--end` | Data fim (YYYY-MM-DD) | hoje |
| `--no-validate` | Pular validação | false |
| `--no-csv` | Não exportar CSV | false |

**Exemplos**:

```bash
# Bootstrap completo S&P 500
python -m datahub_us bootstrap

# Amostra de teste
python -m datahub_us bootstrap --sample

# Símbolos específicos
python -m datahub_us bootstrap --symbols AAPL,MSFT,GOOGL
```

---

### `update`

Atualização incremental desde última data.

```bash
python -m datahub_us update [OPTIONS]
```

**Opções**:

| Flag | Descrição | Default |
|------|-----------|---------|
| `--symbols` | Símbolos específicos | todos |
| `--force-days` | Forçar re-fetch N dias | 0 |

**Exemplo**:

```bash
python -m datahub_us update --force-days 5
```

---

### `sync`

Sincronizar dados do Neon para cache CSV local.

```bash
python -m datahub_us sync [OPTIONS]
```

**Exemplo**:

```bash
python -m datahub_us sync

# Output:
# Syncing...
# [==============================] 100.0%
#
# Sync Complete
# ┌────────────────┬─────────────┐
# │ Metric         │ Value       │
# ├────────────────┼─────────────┤
# │ Symbols        │ 505         │
# │ Total Rows     │ 2,545,230   │
# │ Output Dir     │ cache/us    │
# │ Duration       │ 45.3s       │
# └────────────────┴─────────────┘
```

---

### `indices-sync`

Sincronizar composição de índices US.

```bash
python -m datahub_us indices-sync [OPTIONS]
```

**Opções**:

| Flag | Descrição | Default |
|------|-----------|---------|
| `--indices`, `-i` | Índices específicos | todos |

**Exemplo**:

```bash
# Todos os índices
python -m datahub_us indices-sync

# Específicos
python -m datahub_us indices-sync --indices SPX,NDX
```

---

### `intraday-sync`

Sincronizar dados intraday.

```bash
python -m datahub_us intraday-sync [OPTIONS]
```

**Opções**:

| Flag | Descrição | Default |
|------|-----------|---------|
| `--interval`, `-i` | Intervalo | `30m` |
| `--period`, `-p` | Período | `5d` |

**Exemplo**:

```bash
python -m datahub_us intraday-sync --interval 1h --period 5d
```

---

### `daily-sync`

Sincronizar dados diários.

```bash
python -m datahub_us daily-sync [OPTIONS]
```

**Opções**:

| Flag | Descrição | Default |
|------|-----------|---------|
| `--period`, `-p` | Período | `1mo` |

---

### `dividends-sync`

Sincronizar histórico de dividendos.

```bash
python -m datahub_us dividends-sync [OPTIONS]
```

**Opções**:

| Flag | Descrição | Default |
|------|-----------|---------|
| `--symbols`, `-s` | Símbolos específicos | todos do DB |
| `--start-year` | Ano início | 2010 |
| `--skip-existing/--force-all` | Pular já sincronizados | true |

---

## Índices Suportados

| Código | Nome | Componentes |
|--------|------|-------------|
| SPX | S&P 500 | ~500 |
| NDX | NASDAQ 100 | 100 |
| DJI | Dow Jones | 30 |

---

## Configuração

### Variáveis de Ambiente

| Variável | Descrição | Obrigatório |
|----------|-----------|-------------|
| `NEON_DATABASE_URL` | Connection string PostgreSQL | Sim |

### Exemplo `.env`

```bash
NEON_DATABASE_URL=postgresql://user:pass@host/db?sslmode=require
```

---

## Tabelas no Banco

### `us_bars`

```sql
CREATE TABLE us_bars (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(12, 4) NOT NULL,
    high DECIMAL(12, 4) NOT NULL,
    low DECIMAL(12, 4) NOT NULL,
    close DECIMAL(12, 4) NOT NULL,
    adj_close DECIMAL(12, 4),
    volume BIGINT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, date)
);
```

### `us_dividends`

```sql
CREATE TABLE us_dividends (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    ex_date DATE NOT NULL,
    rate DECIMAL(10, 6) NOT NULL,
    type VARCHAR(20) DEFAULT 'DIVIDEND',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, ex_date, type)
);
```

---

## Quality Assurance

O módulo `qa/validator.py` implementa verificações:

1. **Schema Validation** - Colunas obrigatórias
2. **OHLC Sanity** - `low ≤ open/close ≤ high`
3. **Volume Check** - Volumes ≥ 0
4. **Monotonicity** - Datas ordenadas
5. **Gap Detection** - Falhas > 5 dias úteis
6. **Outlier Detection** - Variações > 50%

---

## GitHub Actions

```yaml
schedule:
  - cron: '0 22 * * 1-5'  # 22:00 UTC (Mon-Fri)

jobs:
  sync:
    steps:
      - run: |
          python -m datahub_us indices-sync
          python -m datahub_us daily-sync --period 1mo
          python -m datahub_us intraday-sync --interval 30m --period 5d
          python -m datahub_us dividends-sync --skip-existing
```

---

## Troubleshooting

### Rate Limit (yfinance)

```
Error: Too many requests
```

**Solução**: Aguardar alguns minutos. yfinance tem rate limits.

### Símbolo delisted

```
Warning: XXXX: failed to fetch - No data found
```

**Causa**: Símbolo foi delisted da bolsa.

---

## Referências

- [yfinance](https://github.com/ranaroussi/yfinance)
- [Provider Due Diligence](provider-due-diligence.md)
- [US DataHub Status](us-datahub-status.md)

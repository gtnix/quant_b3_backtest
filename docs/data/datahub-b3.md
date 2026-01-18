# DataHub B3

**Versão**: 1.0.0  
**Última Atualização**: 2026-01-18

## Visão Geral

O `datahub_b3` é um módulo Python para ingestão de dados do mercado brasileiro (B3). Sincroniza índices, dados diários e intraday usando a API Brapi.

---

## Arquitetura

```
datahub_b3/
├── __init__.py
├── __main__.py          # Entry point: python -m datahub_b3
├── cli.py               # CLI commands
├── config.py            # Configuração (índices B3, API endpoints)
├── db.py                # Conexão Neon PostgreSQL
├── intraday.py          # Sync de dados intraday/daily
├── scraper.py           # Fetch de composição de índices
└── jobs/
    └── backfill_eligibility.py  # Job de backfill
```

---

## Comandos CLI

```bash
python -m datahub_b3 <COMMAND> [OPTIONS]
```

### Comandos Disponíveis

| Comando | Descrição |
|---------|-----------|
| `sync` | Sincroniza índices da B3 para Neon |
| `list` | Lista índices disponíveis |
| `show <INDEX>` | Mostra composição de um índice |
| `intraday-sync` | Sync dados intraday (30m default) |
| `daily-sync` | Sync dados diários |
| `full-sync` | Sync completo (índices + daily + intraday) |

---

### `sync`

Sincroniza composição dos índices B3.

```bash
python -m datahub_b3 sync [INDICES...]
```

**Exemplos**:

```bash
# Todos os índices
python -m datahub_b3 sync

# Índices específicos
python -m datahub_b3 sync IBOV IFIX SMLL
```

---

### `list`

Lista índices B3 disponíveis.

```bash
python -m datahub_b3 list

# Output:
# Índices B3 disponíveis:
#
#   IBOV       - Índice Bovespa
#   IFIX       - Índice de Fundos Imobiliários
#   SMLL       - Índice Small Cap
#   IDIV       - Índice Dividendos
#   BDRX       - Índice BDRs
#   ...
```

---

### `intraday-sync`

Sincroniza dados intraday.

```bash
python -m datahub_b3 intraday-sync [OPTIONS]
```

**Opções**:

| Flag | Descrição | Default |
|------|-----------|---------|
| `--interval` | Intervalo de candles | `30m` |
| `--range` | Range de datas | `5d` |

**Exemplo**:

```bash
# Intraday 30 minutos, últimos 5 dias
python -m datahub_b3 intraday-sync

# Intraday 1 hora, últimos 10 dias
python -m datahub_b3 intraday-sync --interval 1h --range 10d
```

---

### `daily-sync`

Sincroniza dados diários.

```bash
python -m datahub_b3 daily-sync [OPTIONS]
```

**Opções**:

| Flag | Descrição | Default |
|------|-----------|---------|
| `--range` | Range de datas | `1mo` |

**Exemplo**:

```bash
# Último mês
python -m datahub_b3 daily-sync

# Últimos 3 meses
python -m datahub_b3 daily-sync --range 3mo
```

---

### `full-sync`

Sincroniza tudo: índices + daily + intraday.

```bash
python -m datahub_b3 full-sync

# Output:
# Step 1/3: Syncing B3 indices...
# ✓ IBOV: 86 components saved
# ✓ IFIX: 108 components saved
# ...
#
# Step 2/3: Syncing daily OHLCV...
#   Daily: 12,450 bars
#
# Step 3/3: Syncing intraday 30m...
#   Intraday: 45,320 bars
#
# ✓ Full sync complete!
```

---

## Índices Suportados

| Código | Nome | Componentes |
|--------|------|-------------|
| IBOV | Índice Bovespa | ~86 |
| IFIX | Índice FIIs | ~108 |
| SMLL | Small Cap | ~100 |
| IDIV | Dividendos | ~50 |
| BDRX | BDRs | ~60 |
| IBXX | IBrX | ~100 |
| IBXL | IBrX 50 | 50 |
| ISEE | ISE | ~40 |

---

## Configuração

### Variáveis de Ambiente

| Variável | Descrição | Obrigatório |
|----------|-----------|-------------|
| `NEON_DATABASE_URL` | Connection string PostgreSQL | Sim |
| `BRAPI_TOKEN` | Token da API Brapi | Sim |

### Exemplo `.env`

```bash
NEON_DATABASE_URL=postgresql://user:pass@host/db?sslmode=require
BRAPI_TOKEN=your_brapi_token_here
```

---

## Tabelas no Banco

### `b3_indices`

Composição dos índices.

```sql
CREATE TABLE b3_indices (
    id SERIAL PRIMARY KEY,
    index_code VARCHAR(10) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    weight DECIMAL(10, 6),
    ref_date DATE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(index_code, symbol, ref_date)
);
```

### `b3_bars`

Dados OHLCV.

```sql
CREATE TABLE b3_bars (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    time TIME,  -- NULL for daily
    open DECIMAL(12, 4) NOT NULL,
    high DECIMAL(12, 4) NOT NULL,
    low DECIMAL(12, 4) NOT NULL,
    close DECIMAL(12, 4) NOT NULL,
    volume BIGINT NOT NULL,
    interval VARCHAR(10) NOT NULL,  -- '1d', '30m', '1h'
    UNIQUE(symbol, date, time, interval)
);
```

---

## Integração com market_data

O crate `market_data` em Rust consome dados do Neon populados pelo datahub_b3:

```rust
// Rust: Carregar dados B3 do Neon
let bars = market_data::load_bars("PETR4", Market::BR, start, end)?;
```

---

## GitHub Actions

O workflow `sync_b3_indices.yml` executa automaticamente:

```yaml
schedule:
  - cron: '30 21 * * 1-5'  # 21:30 UTC (Mon-Fri)

jobs:
  sync:
    steps:
      - run: python -m datahub_b3 full-sync
```

---

## Troubleshooting

### Erro de Rate Limit (Brapi)

```
Error: Rate limit exceeded
```

**Solução**: Aguardar 1 minuto ou usar token premium.

### Símbolo não encontrado

```
Warning: XXXX: no data available
```

**Causa**: Símbolo não listado na B3 ou ticker incorreto.

---

## Referências

- [Brapi API](https://brapi.dev/)
- [Data Providers Policy](data-providers-policy.md)
- [market_data crate](../../crates/market_data/)

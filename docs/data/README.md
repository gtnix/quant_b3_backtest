# Documentação de Dados

**Última Atualização**: 2026-01-18

Esta seção contém documentação relacionada a fontes de dados, providers e status de ingestão.

---

## Índice

| Documento | Descrição |
|-----------|-----------|
| [Data Providers Policy](data-providers-policy.md) | **Política oficial** de providers por mercado |
| [Provider Due Diligence](provider-due-diligence.md) | Avaliação de providers de dados US (yfinance, Alpha Vantage, etc) |
| [US DataHub Status](us-datahub-status.md) | Status report do DataHub US (cobertura de símbolos, qualidade) |
| **[DataHub B3](datahub-b3.md)** | Módulo Python para ingestão B3 (Brapi) |
| **[DataHub US](datahub-us.md)** | Módulo Python para ingestão US (yfinance) |
| **[DataHub FX](datahub-fx.md)** | Módulo Python para taxas de câmbio (BCB/FRED) |

---

## Providers Utilizados

### Mercado B3

| Provider | Status | Uso |
|----------|--------|-----|
| **Brapi API** | Primary | OHLCV diário/intraday, dividendos |
| **B3 API** | Secondary | Composição de índices |
| **CVM** | Tertiary | Dados de proventos (fallback) |

> ⚠️ **NUNCA usar Yahoo Finance para dados B3.** Ver [Data Providers Policy](data-providers-policy.md).

### Mercado US

| Provider | Status | Uso |
|----------|--------|-----|
| **yfinance** | Primary | 20+ anos OHLCV, dividendos, splits |
| **Alpha Vantage** | Fallback | Verificação pontual |

### FX (Câmbio)

| Provider | Status | Uso |
|----------|--------|-----|
| **BCB** | Primary | USD/BRL, EUR/BRL (PTAX oficial) |
| **FRED** | Primary | EUR/USD, taxas de juros |

---

## DataHub Modules

O sistema possui três módulos de ingestão de dados Python + um crate Rust:

| Módulo | Linguagem | Mercado | Documentação |
|--------|-----------|---------|--------------|
| `datahub_b3` | Python | B3 | **[DataHub B3](datahub-b3.md)** |
| `datahub_us` | Python | US | **[DataHub US](datahub-us.md)** |
| `datahub_fx` | Python | FX | **[DataHub FX](datahub-fx.md)** |
| `market_data` | Rust | Ambos | [Crate](../../crates/market_data/) |

### Comandos Principais

```bash
# B3 - Sincronizar índices e OHLCV
python -m datahub_b3 sync
python -m datahub_b3 daily-sync --range 1mo
python -m datahub_b3 intraday-sync --interval 30m --range 5d

# US - Sincronizar índices e OHLCV
python -m datahub_us indices-sync
python -m datahub_us daily-sync --period 1mo
python -m datahub_us dividends-sync

# FX - Sincronizar taxas de câmbio
python -m datahub_fx sync       # Full sync
python -m datahub_fx update     # Incremental update
python -m datahub_fx status     # Check status
python -m datahub_fx show USD/BRL --tail 10
```

---

## Data Quality Framework

O sistema implementa verificações automáticas de qualidade:

1. **Schema Validation** - Colunas obrigatórias (date, OHLCV)
2. **OHLC Sanity** - `low ≤ open/close ≤ high`
3. **Volume Check** - Volumes não-negativos
4. **Monotonicity** - Datas ordenadas
5. **Gap Detection** - Identificação de falhas >5 dias úteis
6. **Outlier Detection** - Variações diárias >50%

---

## Estrutura de Cache

```
cache/
├── b3/                    # Cache mercado B3
│   ├── bars/              # OHLCV diário
│   ├── dividends/         # Proventos
│   └── splits/            # Desdobramentos
│
├── us/                    # Cache mercado US
│   ├── bars/              # OHLCV diário
│   ├── dividends/         # Dividendos
│   └── splits/            # Splits
│
└── fx/                    # Cache FX
    ├── USD_BRL.csv        # Dólar/Real (BCB)
    ├── EUR_BRL.csv        # Euro/Real (BCB)
    └── EUR_USD.csv        # Euro/Dólar (FRED)
```

---

## Comandos de Ingestão

```bash
# Ingestão US (yfinance)
cargo run -p market_data -- ingest-us --symbols AAPL,MSFT,GOOGL

# Ingestão B3
cargo run -p market_data -- ingest-b3 --universe ibov

# Verificar integridade
cargo run -p market_data -- verify --market us
cargo run -p market_data -- verify --market b3

# Gerar status report
cargo run -p market_data -- status --market us --output docs/data/us-datahub-status.md
```

---

## GitHub Actions (Sync Automático)

| Workflow | Schedule | Descrição |
|----------|----------|-----------|
| `sync_b3_indices.yml` | 21:30 UTC (Mon-Fri) | Sincroniza índices, daily e intraday B3 |
| `sync_us_indices.yml` | 22:00 UTC (Mon-Fri) | Sincroniza índices, daily, intraday e dividendos US |
| `sync_fx_rates.yml` | 21:00 UTC (Mon-Fri) | Sincroniza taxas FX (BCB/FRED) |
| `calendar_update.yml` | Dec 1 @ 12:00 UTC | Atualiza calendários B3/NYSE |
| `monitoring.yml` | 10:00 UTC (Mon-Fri) | Verifica integridade dos dados |

---

## Referências

- [Data Providers Policy](data-providers-policy.md) - Política oficial de providers
- [market_data crate](../../crates/market_data/) - Implementação Rust
- [Data Integrity Framework](../data_integrity.md) - Verificações de qualidade
- [FX Conventions](../policies/fx-conventions.md) - Convenções de câmbio





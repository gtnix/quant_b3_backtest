# Data Providers Policy

**Versão**: 1.0.0  
**Última Atualização**: 2025-12-30

---

## Visão Geral

Esta política define os provedores de dados oficiais para cada mercado e tipo de dado no sistema Quant B3 Backtester. O objetivo é garantir consistência, qualidade e rastreabilidade dos dados utilizados em backtests e descoberta de estratégias.

---

## Política por Mercado

### Brasil (B3)

| Tipo de Dado | Provider Primário | Fallback | Implementação |
|--------------|-------------------|----------|---------------|
| **Index Composition** | B3 API Oficial | - | `datahub_b3/scraper.py` |
| **Daily OHLCV** | Brapi API | - | `datahub_b3/intraday.py`, `market_data/brapi.rs` |
| **Intraday OHLCV** | Brapi API | - | `datahub_b3/intraday.py` |
| **Dividendos** | Brapi API | CVM | `market_data/brapi.rs` |
| **Calendar B3** | B3 Oficial (scraper) | - | `scripts/calendar_scraper/b3_scraper.py` |

**⚠️ NUNCA usar Yahoo Finance (yfinance) para dados B3.** A Brapi é o provider exclusivo para mercado brasileiro.

### Estados Unidos (US)

| Tipo de Dado | Provider Primário | Fallback | Implementação |
|--------------|-------------------|----------|---------------|
| **Index Composition** | Wikipedia/Oficial | - | `datahub_us/indices.py` |
| **Daily OHLCV** | yfinance | Alpha Vantage | `datahub_us/providers/yfinance_provider.py` |
| **Intraday OHLCV** | yfinance | - | `datahub_us/intraday.py` |
| **Dividendos** | yfinance | - | `datahub_us/providers/yfinance_provider.py` |
| **Calendar NYSE** | NYSE Oficial (scraper) | - | `scripts/calendar_scraper/nyse_scraper.py` |

### FX (Câmbio)

| Par | Provider Primário | Fallback | Implementação |
|-----|-------------------|----------|---------------|
| **USD/BRL** | BCB (Banco Central) | - | `datahub_fx/providers/bcb_provider.py` |
| **EUR/BRL** | BCB (Banco Central) | - | `datahub_fx/providers/bcb_provider.py` |
| **EUR/USD** | FRED (Federal Reserve) | - | `datahub_fx/providers/fred_provider.py` |

### Taxas de Juros (Futuro)

| Taxa | Provider Primário | Status |
|------|-------------------|--------|
| **SELIC** | FRED (series DGS10/BRBCB) ou BCB | Planejado |
| **Fed Funds** | FRED | Planejado |

---

## Duplicação Python/Rust

O sistema possui implementações em **Python (para ETL/GitHub Actions)** e **Rust (para performance no backtester)**:

| Provider | Python | Rust | Propósito |
|----------|--------|------|-----------|
| **Brapi** | `datahub_b3/` | `market_data/brapi.rs` | Python: ETL via Actions, Rust: Engine |
| **yfinance** | `datahub_us/providers/` | - | Apenas Python (ETL) |
| **BCB** | `datahub_fx/providers/` | - | Apenas Python (ETL) |
| **FRED** | `datahub_fx/providers/` | - | Apenas Python (ETL) |

**Razão**: Python é usado para pipelines de ingestão (GitHub Actions), enquanto Rust é usado para operações de alta performance no engine.

---

## GitHub Actions Workflows

| Workflow | Schedule | Provider | Dados |
|----------|----------|----------|-------|
| `sync_b3_indices.yml` | 21:30 UTC (Mon-Fri) | Brapi | Índices, Daily, Intraday B3 |
| `sync_us_indices.yml` | 22:00 UTC (Mon-Fri) | yfinance | Índices, Daily, Intraday US |
| `sync_fx_rates.yml` | 21:00 UTC (Mon-Fri) | BCB, FRED | FX rates |
| `calendar_update.yml` | Dec 1 @ 12:00 UTC | Scrapers | Calendários B3/NYSE |
| `monitoring.yml` | 10:00 UTC (Mon-Fri) | - | Data integrity checks |

---

## Qualidade de Dados

### Checks Automáticos (monitoring.yml)

- **Freshness**: Dados devem estar atualizados (< 24h para daily)
- **Null values**: Detecção de valores nulos em OHLCV
- **OHLC anomalies**: Validação `low ≤ open/close ≤ high`
- **Duplicates**: Detecção de barras duplicadas
- **Gap detection**: Identificação de gaps > 5 dias úteis

### Schema Obrigatório

Todos os dados OHLCV devem conter:

| Campo | Tipo | Obrigatório |
|-------|------|-------------|
| `date` | DATE | ✓ |
| `open` | DECIMAL | ✓ |
| `high` | DECIMAL | ✓ |
| `low` | DECIMAL | ✓ |
| `close` | DECIMAL | ✓ |
| `volume` | BIGINT | ✓ |
| `adjusted_close` | DECIMAL | ✓ (para signals) |

---

## Armazenamento

### Neon PostgreSQL (Produção)

Tabelas principais:
- `ohlcv_daily_b3` - OHLCV diário B3
- `ohlcv_daily_us` - OHLCV diário US
- `ohlcv_intraday_b3` - Intraday B3 (30m)
- `ohlcv_intraday_us` - Intraday US (30m)
- `fx_rates` - Taxas de câmbio
- `dividends_b3` - Dividendos B3
- `dividends_us` - Dividendos US
- `index_composition` - Composição de índices
- `trading_calendars` - Calendários B3/NYSE

### Cache Local

```
cache/
├── b3/               # Cache mercado B3
│   ├── bars/         # OHLCV diário
│   ├── dividends/    # Proventos
│   └── splits/       # Desdobramentos
├── us/               # Cache mercado US
│   ├── bars/
│   ├── dividends/
│   └── splits/
└── fx/               # Cache FX
    ├── USD_BRL.csv
    ├── EUR_BRL.csv
    └── EUR_USD.csv
```

---

## Autenticação

| Provider | Autenticação | Secret Name |
|----------|--------------|-------------|
| Brapi | API Key Header | `BRAPI_API_KEY` |
| yfinance | Nenhuma | - |
| BCB | Nenhuma | - |
| FRED | API Key | `FRED_API_KEY` |

---

## Troubleshooting

### "Brapi rate limit exceeded"

O cliente Rust implementa backoff automático. Para Python:
```python
DELAY_BETWEEN_REQUESTS = 0.5  # seconds
MAX_RETRIES = 3
```

### "yfinance data missing"

yfinance ocasionalmente falha para alguns símbolos. O workflow tem retry automático.

### "BCB API timeout"

A API do BCB pode ser lenta. Timeout configurado para 30s.

---

## Referências

- [Provider Due Diligence](provider-due-diligence.md) - Avaliação detalhada de providers US
- [US DataHub Status](us-datahub-status.md) - Status de cobertura US
- [FX Conventions](../policies/fx-conventions.md) - Convenções de câmbio
- [Data Integrity](../data_integrity.md) - Framework de integridade

# Documentação de Dados

Esta seção contém documentação relacionada a fontes de dados, providers e status de ingestão.

---

## Índice

| Documento | Descrição |
|-----------|-----------|
| [Provider Due Diligence](provider-due-diligence.md) | Avaliação de providers de dados US (yfinance, Alpha Vantage, etc) |
| [US DataHub Status](us-datahub-status.md) | Status report do DataHub US (cobertura de símbolos, qualidade) |

---

## Providers Utilizados

### Mercado US

| Provider | Status | Uso |
|----------|--------|-----|
| **yfinance** | Primary | 20+ anos OHLCV, dividendos, splits |
| **Alpha Vantage** | Fallback | Verificação pontual |

### Mercado B3

| Provider | Status | Uso |
|----------|--------|-----|
| **B3 API** | Primary | Dados oficiais da bolsa |
| **CVM** | Secondary | Dados de proventos |

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
└── us/                    # Cache mercado US
    ├── bars/              # OHLCV diário
    ├── dividends/         # Dividendos
    └── splits/            # Splits
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

## Referências

- [market_data crate](../../crates/market_data/)
- [Data Integrity Framework](../data_integrity.md)


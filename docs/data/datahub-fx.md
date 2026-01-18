# DataHub FX

**Versão**: 1.0.0  
**Última Atualização**: 2026-01-18

## Visão Geral

O `datahub_fx` é um módulo Python para ingestão de taxas de câmbio (FX). Sincroniza pares de moedas de múltiplos providers oficiais.

---

## Arquitetura

```
datahub_fx/
├── __init__.py
├── __main__.py          # Entry point
├── cli.py               # CLI commands (argparse)
├── config.py            # Configuração (pares, providers)
├── db.py                # Conexão Neon PostgreSQL
├── jobs/
│   ├── __init__.py
│   ├── sync.py          # Full sync
│   └── update.py        # Incremental update
├── providers/
│   ├── __init__.py
│   ├── base.py          # Provider interface
│   ├── bcb_provider.py  # Banco Central do Brasil
│   ├── brapi_provider.py # Brapi API
│   └── fred_provider.py # Federal Reserve (FRED)
├── storage/
│   ├── __init__.py
│   └── csv_storage.py   # CSV cache storage
└── requirements.txt
```

---

## Pares de Moedas Suportados

| Par | Provider | Descrição |
|-----|----------|-----------|
| USD/BRL | BCB | Dólar/Real (PTAX oficial) |
| EUR/BRL | BCB | Euro/Real (PTAX oficial) |
| EUR/USD | FRED | Euro/Dólar (Fed) |

---

## Comandos CLI

```bash
python -m datahub_fx <COMMAND> [OPTIONS]
```

### Comandos Disponíveis

| Comando | Descrição |
|---------|-----------|
| `sync` | Full sync desde inception |
| `update` | Atualização incremental |
| `status` | Status dos dados |
| `show <PAR>` | Mostrar taxas de um par |

---

### `sync`

Sincronização completa (sobrescreve dados existentes).

```bash
python -m datahub_fx sync [OPTIONS]
```

**Opções**:

| Flag | Descrição | Default |
|------|-----------|---------|
| `--pair` | Par específico | todos |
| `--cache-dir` | Diretório de cache | `cache/fx` |

**Exemplos**:

```bash
# Sincronizar todos os pares
python -m datahub_fx sync

# Sincronizar par específico
python -m datahub_fx sync --pair USD/BRL
```

**Output**:

```json
{
  "USD/BRL": {
    "status": "success",
    "records": 5420,
    "first_date": "2000-01-03",
    "last_date": "2026-01-17"
  },
  "EUR/BRL": {
    "status": "success",
    "records": 5420,
    "first_date": "2000-01-03",
    "last_date": "2026-01-17"
  },
  "EUR/USD": {
    "status": "success",
    "records": 6520,
    "first_date": "1999-01-04",
    "last_date": "2026-01-17"
  }
}
```

---

### `update`

Atualização incremental (apenas novos dados).

```bash
python -m datahub_fx update [OPTIONS]
```

**Opções**:

| Flag | Descrição | Default |
|------|-----------|---------|
| `--pair` | Par específico | todos |

**Exemplo**:

```bash
python -m datahub_fx update

# Output:
# {
#   "USD/BRL": {"new_records": 5, "total": 5425},
#   "EUR/BRL": {"new_records": 5, "total": 5425},
#   "EUR/USD": {"new_records": 3, "total": 6523}
# }
```

---

### `status`

Mostrar status dos dados FX.

```bash
python -m datahub_fx status

# Output:
# FX Data Status (cache/fx)
# ============================================================
#
# EUR/BRL:
#   Records: 5,420
#   Range:   2000-01-03 to 2026-01-17
#   Sources: BCB
#
# EUR/USD:
#   Records: 6,520
#   Range:   1999-01-04 to 2026-01-17
#   Sources: FRED
#
# USD/BRL:
#   Records: 5,420
#   Range:   2000-01-03 to 2026-01-17
#   Sources: BCB
```

---

### `show`

Mostrar taxas de um par específico.

```bash
python -m datahub_fx show <PAR> [OPTIONS]
```

**Opções**:

| Flag | Descrição | Default |
|------|-----------|---------|
| `--head` | Mostrar primeiros N registros | - |
| `--tail` | Mostrar últimos N registros | 20 |

**Exemplo**:

```bash
python -m datahub_fx show USD/BRL --tail 5

# Output:
# USD/BRL Rates
# ----------------------------------------
# 2026-01-13      5.912340  BCB
# 2026-01-14      5.925670  BCB
# 2026-01-15      5.918230  BCB
# 2026-01-16      5.931450  BCB
# 2026-01-17      5.942890  BCB
```

---

## Providers

### BCB (Banco Central do Brasil)

Provider oficial para taxas BRL.

- **Endpoint**: API PTAX do BCB
- **Dados**: Taxa de compra, venda e média (PTAX)
- **Frequência**: Diária (dias úteis)
- **Histórico**: Desde 2000

```python
from datahub_fx.providers.bcb_provider import BcbProvider

provider = BcbProvider()
rates = provider.fetch("USD/BRL", start_date, end_date)
```

### FRED (Federal Reserve)

Provider oficial para taxas USD.

- **Endpoint**: FRED API (St. Louis Fed)
- **Dados**: Taxas de câmbio oficiais
- **Frequência**: Diária (dias úteis US)
- **Histórico**: Desde 1999

**Nota**: Requer API key para acesso.

```python
from datahub_fx.providers.fred_provider import FredProvider

provider = FredProvider(api_key=os.getenv("FRED_API_KEY"))
rates = provider.fetch("EUR/USD", start_date, end_date)
```

### Brapi

Provider alternativo para taxas BR.

- **Endpoint**: Brapi API
- **Dados**: Taxas de câmbio em tempo real
- **Uso**: Fallback quando BCB indisponível

---

## Configuração

### Variáveis de Ambiente

| Variável | Descrição | Obrigatório |
|----------|-----------|-------------|
| `FRED_API_KEY` | API key do FRED | Para EUR/USD |

### Exemplo `.env`

```bash
FRED_API_KEY=your_fred_api_key_here
```

---

## Estrutura de Cache

```
cache/fx/
├── USD_BRL.csv
├── EUR_BRL.csv
└── EUR_USD.csv
```

### Formato CSV

```csv
date,rate,source
2000-01-03,1.8045,BCB
2000-01-04,1.8123,BCB
...
```

---

## Integração com market_data

O crate `market_data` em Rust consome dados do cache FX:

```rust
// Rust: Carregar taxas FX
use market_data::fx_loader::FxLoader;

let loader = FxLoader::new("cache/fx");
let rate = loader.get_rate("USD/BRL", date)?;
```

---

## GitHub Actions

```yaml
schedule:
  - cron: '0 21 * * 1-5'  # 21:00 UTC (Mon-Fri)

jobs:
  sync:
    steps:
      - run: python -m datahub_fx update
```

---

## Troubleshooting

### BCB Indisponível

```
Error: BCB API timeout
```

**Solução**: Tentar novamente mais tarde. BCB pode ter janelas de manutenção.

### FRED Rate Limit

```
Error: FRED API rate limit exceeded
```

**Solução**: Aguardar 1 minuto. FRED tem limite de 120 requests/minuto.

### Dados faltando

```
Warning: No data for EUR/BRL on 2026-01-01
```

**Causa**: Feriado. Dados FX não são publicados em feriados.

---

## Convenções FX

Ver [FX Conventions](../policies/fx-conventions.md) para detalhes sobre:

- Horários de corte (PTAX)
- Tratamento de weekends/feriados
- Interpolação de gaps

---

## Referências

- [BCB API](https://dadosabertos.bcb.gov.br/)
- [FRED API](https://fred.stlouisfed.org/docs/api/)
- [FX Conventions](../policies/fx-conventions.md)

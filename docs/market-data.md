# Market Data Pipeline

Pipeline de ingestão de dados OHLCV da B3 usando Brapi API + Neon Postgres.

## Setup

### Variáveis de Ambiente

```bash
# Brapi API (obrigatório)
export BRAPI_API_KEY="sua_chave_aqui"

# Neon Database (obrigatório)
export DATABASE_URL="postgresql://user:pass@host/db?sslmode=require"

# Opcionais
export BRAPI_MAX_TICKERS_PER_REQUEST=20  # Default: 20 (PRO)
export BRAPI_REQUESTS_PER_MINUTE=60       # Rate limiting
```

### Planos Brapi

| Plano | Tickers/Request | Requests/Mês | Histórico |
|-------|-----------------|--------------|-----------|
| Free | 1 | 15,000 | 1 mês |
| Startup | 10 | 150,000 | 1 ano |
| PRO | 20 | 500,000 | 10+ anos |

## Comandos CLI

### Inicializar Database

```bash
./target/release/market-data init-db
```

Verifica se o schema existe no Neon.

### Atualizar Universo

```bash
# Carregar top 150 por volume
./target/release/market-data refresh-universe --target 150
```

Usa o endpoint `/api/quote/list` com paginação.

### Backfill Histórico

```bash
# Range completo (10+ anos para PRO)
./target/release/market-data backfill --universe top_volume --range max

# Apenas último ano
./target/release/market-data backfill --universe top_volume --range 1y
```

Ranges disponíveis: `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `max`

### Update Incremental

```bash
./target/release/market-data update --universe top_volume
```

Detecta automaticamente o range mínimo necessário por ticker.

### Verificar Integridade

```bash
./target/release/market-data verify-integrity
```

Checa gaps, candles inválidos e duplicatas.

### Status

```bash
./target/release/market-data status
```

Mostra estatísticas de ingestão e uso da API.

## Schema Neon

```sql
-- Instrumentos
instruments (symbol PK, name, type, sector, market_cap, active)

-- Universo
universe_membership (universe_name, symbol, rank, as_of_date)

-- Dados OHLCV
ohlcv_daily (symbol, trading_date, open, high, low, close, adj_close, volume)
  PK: (symbol, trading_date)

-- Estado de Ingestão
ingestion_state (symbol PK, first_bar_date, last_bar_date, total_bars, ...)

-- Log de Requests
api_request_log (endpoint, tickers_count, http_status, duration_ms, ...)

-- Budget Mensal
api_budget (month_key PK, requests_used, requests_limit)
```

## Economia de API

| Operação | Requests |
|----------|----------|
| Listar 150 stocks | 2 |
| Backfill 150 tickers | 8 |
| Update diário | 8 |
| **Total inicial** | ~10 |
| **Total mensal** | ~250 |

## Troubleshooting

### Quota Exceeded (402)

```
Error: API quota exceeded
```

O limite mensal foi atingido. Verifique com:

```bash
./target/release/market-data status
```

### Rate Limited (429)

O cliente implementa backoff exponencial automático.

### Connection Timeout

Verifique a connection string do Neon e SSL mode.

## Neon Project

- **Project ID**: cold-poetry-53030794
- **Database**: neondb
- **Region**: us-west-2

## Queries Úteis

```sql
-- Total de bars por símbolo
SELECT symbol, COUNT(*) as bars, 
       MIN(trading_date) as first, 
       MAX(trading_date) as last 
FROM ohlcv_daily 
GROUP BY symbol 
ORDER BY bars DESC;

-- Uso de API este mês
SELECT * FROM api_budget WHERE month_key = to_char(NOW(), 'YYYY-MM');

-- Símbolos com falha
SELECT * FROM ingestion_state WHERE consecutive_failures > 0;
```


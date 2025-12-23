# Fontes de Dados Historicos B3

## Matriz Comparativa

| Fonte | Tipo | Formato | Cobertura | OHLCV | Ajuste Split | API | Custo | Status | Testado |
|-------|------|---------|-----------|-------|--------------|-----|-------|--------|---------|
| **Brapi.dev** | API REST | JSON | 10+ anos | Sim | Sim | Sim | Free tier (15 req/min) | Recomendado | OK |
| **B3 (dados.b3.com.br)** | Download | CSV/ZIP | 2000+ | Sim | Nao | Nao | Gratis | Backup | OK |
| **Investing.com** | Export | CSV | 20+ anos | Sim | Sim | Nao | Gratis | Bloqueado | 403 |
| **Google Finance** | Scraping | JSON | 5+ anos | Sim | Sim | Nao | Gratis | Fragil | Requer Browser |
| **CVM** | Download | CSV/ZIP | Fundos | Parcial | N/A | Nao | Gratis | Fundos only | N/A |
| **Dados de Mercado** | API | JSON | 10+ anos | Sim | Sim | Sim | Freemium | Alternativa | N/A |
| **Ipeadata** | Download | CSV | Macro | Nao | N/A | Nao | Gratis | Indicadores | N/A |

## Detalhes por Fonte

### 1. Brapi.dev (Recomendado)

**URL**: https://brapi.dev

**Endpoint historico**:
```
GET https://brapi.dev/api/quote/{ticker}?range=1y&interval=1d
```

**Campos retornados**:
- `date`: timestamp
- `open`, `high`, `low`, `close`: precos
- `volume`: volume negociado
- `adjustedClose`: preco ajustado

**Limites**:
- Free: 15 requests/min
- Pago: ilimitado

**Exemplo de uso**:
```bash
curl "https://brapi.dev/api/quote/PETR4?range=5y&interval=1d"
```

### 2. B3 Dados Historicos

**URL**: https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/historico/mercado-a-vista/series-historicas/

**Formato**: Arquivo posicional (largura fixa)

**Campos**:
- TIPREG: Tipo registro
- DATAPG: Data pregao
- CODBDI: Codigo BDI
- CODNEG: Codigo negociacao
- TPMERC: Tipo mercado
- PREABE: Preco abertura
- PREMAX: Preco maximo
- PREMIN: Preco minimo
- PREMED: Preco medio
- PREULT: Ultimo preco
- QUATOT: Quantidade total
- VOLTOT: Volume total

**Obs**: Requer parsing especifico, precos em centavos.

### 3. Investing.com

**URL**: https://br.investing.com/equities/{ticker}-historical-data

**Acesso**: Export manual ou scraping

**Campos CSV**:
- Data
- Ultimo
- Abertura
- Maxima
- Minima
- Vol.
- Var%

**Obs**: Bloqueio anti-bot, usar com cautela.

### 4. Google Finance

**URL**: https://www.google.com/finance/quote/{ticker}:BVMF

**Acesso**: Scraping (sem API oficial)

**Obs**: Estrutura HTML muda frequentemente, fragil para automacao.

### 5. CVM Dados Abertos

**URL**: https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/

**Conteudo**: Dados de fundos de investimento, nao acoes.

### 6. Dados de Mercado

**URL**: https://www.dadosdemercado.com.br

**API**: Disponivel com cadastro

**Campos**: OHLCV completo, ajustado

## Recomendacao

**Producao**: Brapi.dev
- API estavel
- Dados ajustados
- Free tier suficiente para backtest

**Backup**: B3 arquivos historicos
- Dados oficiais
- Sem limite de requests
- Requer parser especifico

## Schema Padrao

Todos os loaders devem converter para o schema:

```csv
timestamp,ticker,open,high,low,close,volume
2024-01-02,PETR4,38.50,39.20,38.10,38.95,15000000
```

Onde:
- `timestamp`: YYYY-MM-DD
- `ticker`: Codigo B3
- `open/high/low/close`: Precos em BRL
- `volume`: Quantidade de acoes

## CLI Commands

### Fetch data from Brapi

```bash
# Single ticker, 1 year
./target/release/backtest fetch -t PETR4 -r 1y -o data/petr4.csv

# Multiple tickers, 5 years
./target/release/backtest fetch -t "PETR4,VALE3,ITUB4,BBDC4,ABEV3" -r 5y -o data/portfolio.csv

# Full portfolio
./target/release/backtest fetch -t "PETR4,VALE3,ITUB4,BBDC4,ABEV3,WEGE3,RENT3,RADL3,JBSS3,GGBR4" -r 2y -o data/universe.csv
```

### Run backtest with fetched data

```bash
./target/release/backtest run -c configs/real_data.toml -o output/results
```

## Loaders Implementados

### BrapiLoader

```rust
use backtester_io::BrapiLoader;

let loader = BrapiLoader::new();
let bars = loader.fetch("PETR4", "1y", "1d")?;
loader.save_to_csv(&bars, Path::new("data/petr4.csv"))?;
```

### B3HistoricalLoader

```rust
use backtester_io::B3HistoricalLoader;

// Download annual file
B3HistoricalLoader::download_year(2024, Path::new("data/COTAHIST_A2024.ZIP"))?;

// Parse ZIP file
let bars = B3HistoricalLoader::parse_zip("data/COTAHIST_A2024.ZIP")?;

// Filter specific tickers
let filtered = B3HistoricalLoader::filter_tickers(bars, &["PETR4", "VALE3"]);
```


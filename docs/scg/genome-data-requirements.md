# Requisitos de Dados para Geração de Genomas

**Versão**: 1.0.0  
**Última Atualização**: 2026-01-21  
**Status**: Documento de Contexto para Evolução do Sistema

---

## 1. Visão Geral do Problema

O sistema de mineração de estratégias (SCG - Strategy Combiner Generator) gera **genomas aleatórios** que combinam diferentes tipos de blocos. Quando um genoma inclui um bloco que requer dados não disponíveis no dataset, o backtest retorna **0 trades**, desperdiçando ciclos de computação e impedindo a evolução do algoritmo genético.

### Fluxo Atual (Problemático)

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  ParamRanges    │────▶│ Population::    │────▶│   Backtest      │
│  (todos blocos) │     │ random_genome() │     │   (0 trades)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                      │                       │
         │                      │                       │
    7 Selection              Genoma inclui          Dataset só tem
    blocks                   "value" ou "carry"     OHLCV data
```

### Causa Raiz

Os **blocos de seleção** se dividem em duas categorias:

| Categoria | Blocos | Dados Necessários |
|-----------|--------|-------------------|
| **OHLCV** | `momentum`, `low_vol` | Apenas preços (OHLC) e volume |
| **Fundamentalista** | `value`, `quality`, `dividend`, `size`, `carry` | P/E, P/B, ROE, D/E, DY, Market Cap, Risk-Free Rate |

Quando o dataset (`market_data_ibov.csv`) contém apenas OHLCV:
- Genomas com blocos fundamentalistas não conseguem selecionar ativos
- Resultado: 0 trades, Sharpe = 0, CAGR = 0%

---

## 2. Catálogo de Blocos e Requisitos de Dados

### 2.1 Blocos de Selection (Seleção de Ativos)

| Block ID | Descrição | Colunas Requeridas | Status OHLCV |
|----------|-----------|-------------------|--------------|
| `momentum` | Ranking por retorno 6-12 meses | `close`, `adj_close` | ✅ Funciona |
| `low_vol` | Seleção por baixa volatilidade | `close` | ✅ Funciona |
| `value` | Value investing (P/E, P/B baixos) | `pe_ratio`, `pb_ratio` | ❌ Requer fundamentais |
| `quality` | Alta qualidade (ROE alto, baixo endividamento) | `roe`, `debt_equity` | ❌ Requer fundamentais |
| `dividend` | Pagadoras de dividendos | `dividend_yield` | ❌ Requer fundamentais |
| `size` | Filtro por market cap | `market_cap` | ❌ Requer fundamentais |
| `carry` | Yield vs taxa livre de risco | `dividend_yield`, `risk_free_rate` | ❌ Requer fundamentais |

### 2.2 Blocos de Entry (Entrada)

| Block ID | Descrição | Colunas Requeridas | Status OHLCV |
|----------|-----------|-------------------|--------------|
| `ma_crossover` | Cruzamento de médias móveis | `close` | ✅ Funciona |
| `rsi` | RSI oversold/overbought | `close` | ✅ Funciona |
| `macd` | MACD crossover | `close` | ✅ Funciona |
| `bollinger` | Bollinger Bands | `close` | ✅ Funciona |
| `zscore` | Z-Score mean reversion | `close` | ✅ Funciona |

### 2.3 Blocos de Exit (Saída)

| Block ID | Descrição | Colunas Requeridas | Status OHLCV |
|----------|-----------|-------------------|--------------|
| `stop_loss` | Stop loss percentual | `close` | ✅ Funciona |
| `take_profit` | Take profit percentual | `close` | ✅ Funciona |
| `trailing_stop` | Trailing stop | `close`, `high` | ✅ Funciona |
| `time_exit` | Saída por tempo | (nenhum) | ✅ Funciona |

### 2.4 Blocos de Sizing (Dimensionamento)

| Block ID | Descrição | Colunas Requeridas | Status OHLCV |
|----------|-----------|-------------------|--------------|
| `equal_weight` | Peso igual 1/N | (nenhum) | ✅ Funciona |
| `risk_parity` | Inverse volatility | `close` | ✅ Funciona |
| `vol_targeting` | Target de volatilidade | `close` | ✅ Funciona |

---

## 3. Solução Implementada: DataColumn Enum

### 3.1 Estrutura de Dados

```rust
/// Colunas de dados que podem ser requeridas por blocos.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataColumn {
    // OHLCV básico (sempre disponível)
    Open, High, Low, Close, Volume, AdjClose,
    
    // Dados fundamentalistas (requerem fonte adicional)
    PE,           // Price/Earnings ratio
    PB,           // Price/Book ratio
    ROE,          // Return on Equity
    DebtEquity,   // Debt/Equity ratio
    DividendYield,
    MarketCap,
    RiskFreeRate,
}

impl DataColumn {
    /// Verifica se é uma coluna OHLCV básica.
    pub fn is_ohlcv(&self) -> bool {
        matches!(self, 
            DataColumn::Open | DataColumn::High | DataColumn::Low | 
            DataColumn::Close | DataColumn::Volume | DataColumn::AdjClose
        )
    }
}
```

### 3.2 BlockSpec com required_columns

```rust
pub struct BlockSpec {
    pub block_id: String,
    pub block_type: BlockType,
    pub params: Vec<ParamSpec>,
    pub description: String,
    pub fast_supported: bool,
    /// Colunas de dados requeridas por este bloco.
    pub required_columns: Vec<DataColumn>,
}
```

### 3.3 Métodos de Filtragem

```rust
impl ParamRanges {
    /// Retorna blocos compatíveis com as colunas disponíveis.
    pub fn block_ids_for_available_data(
        &self,
        block_type: BlockType,
        available: &HashSet<DataColumn>,
    ) -> Vec<&str>;

    /// Atalho: retorna apenas blocos OHLCV-compatíveis.
    pub fn ohlcv_only_block_ids(&self, block_type: BlockType) -> Vec<&str>;

    /// Filtra ParamRanges removendo blocos incompatíveis.
    /// Retorna (ranges_filtrados, blocos_desabilitados).
    pub fn with_ohlcv_only(self) -> (Self, Vec<(String, Vec<DataColumn>)>);
}
```

---

## 4. Strategy Catalog vs Geração Aleatória

### 4.1 Strategy Catalog (115 Estratégias Pré-definidas)

O **Strategy Catalog** no dashboard contém 115 templates pré-configurados:

- **Swing Trading**: Bollinger Swing, MA Crossover, RSI Swing, etc.
- **Position Trading**: Trend Following, Quality Investing, Value Investing
- **Pair Trading**: Cointegration, Distance, Correlation
- **Portfolio**: Risk Parity, Minimum Variance, Tactical Allocation

Esses templates definem a **estrutura** da estratégia (quais blocos usar), e o GA otimiza apenas os **parâmetros** dentro dos ranges definidos.

### 4.2 Geração Aleatória (Problema)

Quando o miner gera genomas **completamente aleatórios** via `Population::random()`:

1. Escolhe 1-3 blocos de Selection (de um pool de 7)
2. Escolhe 1-2 blocos de Entry (de um pool de 5)
3. Escolhe 1-2 blocos de Exit (de um pool de 4)
4. Escolhe 1 bloco de Sizing (de um pool de 3)

**Problema**: 5 dos 7 blocos de Selection requerem dados fundamentalistas.
**Probabilidade de genoma inválido**: ~71% na primeira seleção.

### 4.3 Solução Proposta: Template-First Generation

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Strategy        │────▶│ GA Randomizes   │────▶│   Backtest      │
│ Catalog         │     │ Parameters Only │     │   (trades > 0)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                      │                       │
         │                      │                       │
    115 templates            Estrutura fixa,          Genomas sempre
    OHLCV-safe               params variam            executáveis
```

**Fluxo Template-First**:

1. Carregar templates do Strategy Catalog
2. Filtrar templates compatíveis com dados disponíveis
3. Para cada template:
   - Manter estrutura (blocos) fixa
   - Randomizar parâmetros dentro dos ranges
4. Evoluir via GA (crossover de parâmetros, não de estrutura)

---

## 5. Estrutura de Dataset Esperada

### 5.1 Dataset OHLCV Mínimo (Atual)

```csv
symbol,date,open,high,low,close,adj_close,volume
PETR4,2021-01-04,25.50,26.10,25.30,25.80,24.50,15000000
```

**Blocos disponíveis**: `momentum`, `low_vol`, todos Entry, todos Exit, todos Sizing

### 5.2 Dataset com Fundamentais (Futuro)

```csv
symbol,date,open,high,low,close,adj_close,volume,pe_ratio,pb_ratio,roe,debt_equity,dividend_yield,market_cap
PETR4,2021-01-04,25.50,26.10,25.30,25.80,24.50,15000000,8.5,1.2,0.18,0.45,0.065,350000000000
```

**Blocos disponíveis**: TODOS (7 Selection + 5 Entry + 4 Exit + 3 Sizing)

---

## 6. Fontes de Dados Fundamentalistas (Roadmap)

| Fonte | Dados | Market | Status |
|-------|-------|--------|--------|
| Brapi | P/E, P/B, DY | BR | 🔮 Planejado |
| Yahoo Finance | P/E, P/B, Market Cap | US | 🔮 Planejado |
| Fundamentus | ROE, D/E | BR | 🔮 Planejado |
| BCB | Taxa Selic (Risk-Free) | BR | ✅ Disponível via datahub_fx |
| FRED | Fed Funds Rate | US | ✅ Disponível via datahub_fx |

---

## 7. Próximos Passos

### 7.1 Curto Prazo (OHLCV-Only)

1. ✅ Adicionar `required_columns` ao `BlockSpec`
2. ✅ Implementar `with_ohlcv_only()` no `ParamRanges`
3. ⏳ Modificar `EvolutionEngine` para aplicar filtro automaticamente
4. ⏳ Atualizar Strategy Catalog para marcar templates incompatíveis

### 7.2 Médio Prazo (Template-First)

1. Integrar Strategy Catalog com geração de genomas
2. Usar templates como "esqueleto" e randomizar apenas parâmetros
3. Implementar crossover a nível de parâmetros (não estrutura)

### 7.3 Longo Prazo (Dados Fundamentalistas)

1. Ingerir dados fundamentalistas via Brapi/Yahoo
2. Adicionar colunas ao dataset CSV ou tabelas separadas
3. Habilitar todos os blocos de Selection

---

## 8. Arquivos Relacionados

| Arquivo | Descrição |
|---------|-----------|
| `crates/combiner_core/src/param_ranges.rs` | Definição de blocos e `DataColumn` |
| `crates/combiner_engine/src/population.rs` | Geração de genomas aleatórios |
| `crates/combiner_engine/src/engine.rs` | Motor de evolução |
| `docs/scg/genome-structure.md` | Estrutura do genoma |
| `docs/Criação de Parâmetros.../12_TPM_Preconfigured_Strategy_Catalog.md` | Catálogo de 115 estratégias |
| `docs/Criação de Parâmetros.../06_TPM_Genetic_Algorithm_Integration.md` | Integração TPM + GA |

---

## 9. Glossário

| Termo | Definição |
|-------|-----------|
| **Genoma** | Representação completa de uma estratégia (blocos + parâmetros) |
| **Gene** | Um parâmetro individual dentro de um bloco |
| **BlockGene** | Bloco com seus parâmetros (ex: `momentum` com `lookback_days=126`) |
| **OHLCV** | Open, High, Low, Close, Volume - dados básicos de preço |
| **Fundamentalista** | Dados financeiros como P/E, ROE, Market Cap |
| **Strategy Catalog** | Biblioteca de 115 templates pré-configurados |
| **TPM** | Trade Parameters Module - sistema de templates |
| **SCG** | Strategy Combiner Generator - motor de mineração |

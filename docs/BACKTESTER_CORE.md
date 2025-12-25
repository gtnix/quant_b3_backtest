# Backtester Core - Engine de Backtesting em Rust

**Versão:** 2.0.0  
**Mercado:** B3 (Brasil)

## Performance Features

- **SIMD Vectorizado** - Cálculos vetorizados via `wide` crate (f64x4)
- **Paralelização Rayon** - Multi-thread para backtests com múltiplos ativos
- **Zero-Copy I/O** - Memory-mapped files via `memmap2`
- **Cache L1 Otimizado** - Estruturas alinhadas a 64 bytes (cache line)
- **Rebalanceamento Quinzenal** - `BiweeklyRebalancer` para portfolios

## Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                    STRATEGY (sua estratégia)                │
│          Implementa trait Strategy::on_market()             │
└──────────────────────────┬──────────────────────────────────┘
                           │ SignalEvent
┌──────────────────────────▼──────────────────────────────────┐
│              SIMULATION ENGINE (orquestrador)               │
│  - Processa eventos em ordem cronológica                    │
│  - Roteia sinais → ordens → execução                        │
│  - Mantém estado do mercado e portfólio                     │
└────┬─────────────────────┬──────────────────────┬───────────┘
     │                     │                      │
┌────▼────────┐      ┌─────▼─────────┐     ┌─────▼─────────────┐
│ MARKET      │      │ EXECUTION     │     │ PORTFOLIO         │
│ STATE       │      │ MODEL         │     │                   │
│ - Preços    │      │ - Slippage    │     │ - Posições (SoA)  │
│ - Volumes   │      │ - Custos B3   │     │ - Cash            │
│ - SoA       │      │ - Liquidez    │     │ - PnL/Drawdown    │
└─────────────┘      └───────────────┘     └───────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                      REPORTS                                 │
│  - NAV History, Sharpe, Sortino, Calmar, Max Drawdown       │
│  - Win Rate, Profit Factor, Consecutive Wins/Losses         │
└─────────────────────────────────────────────────────────────┘
```

## Crates

| Crate | Responsabilidade |
|-------|------------------|
| `backtester_core` | Tipos fundamentais, traits, eventos, **SIMD ops** |
| `backtester_engine` | SimulationEngine, MarketState, **ParallelEngine**, **Rebalancer** |
| `backtester_execution` | Slippage, CostModel, LiquidityModel |
| `backtester_portfolio` | Position, Trade, Portfolio (**SoA + Cache-aligned**) |
| `backtester_reports` | NavHistory, BacktestResult (**SIMD metrics**) |
| `backtester_io` | CSV Loader, **MmapStream (zero-copy)** |
| `backtester_cli` | CLI |
| `backtester_tests` | Testes anti-look-ahead, determinismo |

## Tipos Principais

### Identificadores
```rust
pub struct AssetId(pub u16);    // ID do ativo (0-indexed)
pub struct OrderId(pub u64);    // ID da ordem
pub struct FillId(pub u64);     // ID do fill
pub struct Timestamp(pub i64);  // Nanosegundos desde epoch
```

### Bar (OHLCV)
```rust
pub struct Bar {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}
```

### Eventos
```rust
pub struct MarketEvent { asset_id: AssetId, bar: Bar }
pub struct SignalEvent { timestamp, asset_id, strength: f64, signal_type: SignalType }
pub struct OrderEvent { order_id, timestamp, asset_id, direction, quantity, order_type, ... }
pub struct FillEvent { fill_id, order_id, timestamp, asset_id, direction, quantity, price, commission, slippage }
```

## Trait Strategy

```rust
pub trait Strategy {
    fn on_market(&mut self, event: &MarketEvent) -> Option<SignalEvent>;
    
    // Opcionais
    fn on_init(&mut self, _config: &StrategyConfig, _num_assets: usize) {}
    fn on_session_close(&mut self, _timestamp: i64, _asset_id: AssetId) -> Option<SignalEvent> { None }
    fn on_backtest_end(&mut self) {}
    fn name(&self) -> &str { "UnnamedStrategy" }
}
```

## Exemplo de Uso

```rust
use backtester_engine::*;
use backtester_core::*;

struct MomentumStrategy {
    prev_close: f64,
}

impl Strategy for MomentumStrategy {
    fn on_market(&mut self, event: &MarketEvent) -> Option<SignalEvent> {
        let close = event.bar.close;
        let signal = if close > self.prev_close * 1.01 {
            Some(SignalEvent::buy(event.bar.timestamp, event.asset_id, 1.0))
        } else if close < self.prev_close * 0.99 {
            Some(SignalEvent::sell(event.bar.timestamp, event.asset_id, 1.0))
        } else {
            None
        };
        self.prev_close = close;
        signal
    }
}

fn main() {
    let strategy = MomentumStrategy { prev_close: 0.0 };
    let mut engine = SimulationEngine::with_defaults(strategy, 100_000.0, 10);
    
    // Processar eventos
    for event in market_events {
        engine.process_event(&event);
    }
    
    let result = engine.get_result();
    println!("NAV Final: R$ {:.2}", result.final_nav);
    println!("Max Drawdown: {:.2}%", result.max_drawdown * 100.0);
}
```

## Modelos de Execução

### Slippage
```rust
pub enum SlippageModel {
    None,
    Constant { bps: f64 },                          // Ex: 5 bps = 0.05%
    VolumeLinear { base_bps: f64, volume_factor: f64 },
    Volatility { base_bps: f64, vol_factor: f64 },
}
```

### Custos B3
```rust
pub struct CostModel {
    pub fixed_cost: f64,       // R$ 10 por ordem
    pub commission_rate: f64,  // 0.1%
    pub per_unit_cost: f64,    // R$ 0.01 por ação
    pub emolument_rate: f64,   // 0.035% (B3)
}
```

### Liquidez
```rust
pub struct LiquidityModel {
    pub max_participation: f64,    // 10% do volume do bar
    pub allow_partial_fills: bool,
}
```

## Portfolio

- **SoA Layout** (Structure of Arrays) para performance
- Posições por `AssetId` index → O(1) lookup
- Mark-to-market automático
- Tracking de PnL realizado/não-realizado
- Cálculo de drawdown contínuo

```rust
let portfolio = engine.portfolio();
let nav = portfolio.nav();
let position = portfolio.get_position(AssetId::new(0));
let max_dd = portfolio.max_drawdown;
```

## Métricas Calculadas

| Métrica | Descrição |
|---------|-----------|
| `total_return` | Retorno total |
| `annual_return` | Retorno anualizado |
| `annual_volatility` | Volatilidade anualizada |
| `sharpe_ratio` | (return - rf) / volatility |
| `sortino_ratio` | (return - rf) / downside_vol |
| `calmar_ratio` | annual_return / max_drawdown |
| `max_drawdown` | Máximo drawdown |
| `max_drawdown_duration` | Duração do drawdown (bars) |
| `win_rate` | Taxa de acerto |
| `profit_factor` | Lucro bruto / Prejuízo bruto |
| `max_consecutive_wins` | Sequência máxima de gains |
| `max_consecutive_losses` | Sequência máxima de losses |

## Regras B3

- **Lote padrão**: 100 ações (ordens arredondadas)
- **Validação**: `OrderEvent::is_valid_b3_lot()` → `quantity % 100 == 0`
- **Custos**: Emolumentos B3 incluídos no `CostModel::b3_default()`

## Testes

```bash
cargo test -p backtester_core
cargo test -p backtester_engine
cargo test -p backtester_execution
cargo test -p backtester_portfolio
cargo test -p backtester_reports
cargo test -p backtester_tests   # Anti look-ahead, determinismo
```

## Módulos de Performance

### SIMD (`backtester_core::simd`)
```rust
use backtester_core::simd;

let returns = simd::simd_returns(&prices);           // Retornos vetorizados
let (max_dd, duration) = simd::simd_drawdown(&nav);  // Drawdown SIMD
let volatility = simd::simd_volatility(&returns);    // Vol anualizada
let sharpe = simd::simd_sharpe(&returns, 0.02);      // Sharpe SIMD
```

### Parallel Engine (`backtester_engine::parallel`)
```rust
use backtester_engine::parallel::*;

let batches = group_by_day(&events);
let mut engine = ParallelEngine::new(strategy, 100_000.0, 10);
let result = engine.run(&batches);
```

### Zero-Copy I/O (`backtester_io::mmap`)
```rust
use backtester_io::mmap::*;

let mut stream = MmapStream::open("data.csv")?;
let events = stream.load_all();  // Zero-copy load
```

### Rebalancer (`backtester_engine::rebalancer`)
```rust
use backtester_engine::rebalancer::*;

let rebalancer = BiweeklyRebalancer::equal_weight(10)
    .with_tolerance(0.02)
    .with_min_order_size(100);

if rebalancer.should_rebalance(timestamp) {
    let orders = rebalancer.calculate_orders(&portfolio, &prices, timestamp);
}
```

## Benchmarks

```bash
cargo bench -p backtester_core    # SIMD benchmarks
cargo bench -p backtester_engine  # Engine benchmarks
```


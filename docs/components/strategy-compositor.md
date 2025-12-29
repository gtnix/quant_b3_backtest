# Strategy Compositor

**Versão**: 2.0.0  
**Última Atualização**: 2025-12-28

## Visão Geral

O Strategy Compositor é uma DSL declarativa para composição de estratégias via configuração TOML. Permite criar novas estratégias sem modificar código.

### Localização no Código

- **Crate**: `backtester_strategy`
- **Arquivos Principais**:
  - `src/compositor.rs` - Executor de pipeline
  - `src/registry.rs` - BlockRegistry
  - `src/config.rs` - Parsing de TOML
  - `src/blocks/` - Implementações de blocos

---

## Compositor

O `Compositor` executa pipelines de blocos em sequência.

```rust
use backtester_strategy::{
    config::load_strategy_config,
    compositor::Compositor,
    context::StrategyContext,
};

// Carregar estratégia
let config = load_strategy_config("configs/strategies/momentum.toml")?;

// Criar compositor com blocos built-in
let compositor = Compositor::with_builtins();

// Executar pipeline
let mut ctx = StrategyContext::new(date, Market::BR, capital);
ctx.candidates = load_candidates();
let result = compositor.execute(&config, &mut ctx)?;

// Usar resultados
println!("Selected: {:?}", result.selected);
println!("Weights: {:?}", result.weights);
```

### CompositorResult

```rust
pub struct CompositorResult {
    pub selected: Vec<String>,
    pub weights: HashMap<String, f64>,
    pub signals: Vec<Signal>,
    pub trace: Vec<TraceEntry>,
}
```

---

## BlockRegistry

O `BlockRegistry` mapeia `block_id` → implementação de bloco.

```rust
use backtester_strategy::BlockRegistry;

// Criar com blocos built-in
let registry = BlockRegistry::with_builtins();

// Verificar se bloco existe
assert!(registry.contains("momentum"));

// Listar blocos por tipo
let selection_blocks = registry.blocks_by_type(BlockType::Selection);
```

### Blocos Registrados

| Categoria | Blocos |
|-----------|--------|
| Selection | `momentum`, `value`, `quality`, `low_vol`, `dividend`, `size`, `carry` |
| Entry | `ma_crossover`, `bollinger`, `rsi`, `macd`, `zscore` |
| Exit | `stop_loss`, `take_profit`, `trailing_stop`, `time_exit` |
| Sizing | `equal_weight`, `risk_parity`, `vol_targeting` |

**Total**: 19 blocos

---

## StrategyBlock Trait

Interface para implementação de blocos:

```rust
pub trait StrategyBlock: Send + Sync {
    /// ID único do bloco
    fn block_id(&self) -> &'static str;
    
    /// Tipo do bloco
    fn block_type(&self) -> BlockType;
    
    /// Executar lógica do bloco
    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult;
    
    /// Validar parâmetros
    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError>;
    
    /// Parâmetros default
    fn default_params(&self) -> BlockParams;
    
    /// Suporta fast mode?
    fn fast_supported(&self) -> bool { false }
}
```

---

## Configuração TOML

### Estrutura Básica

```toml
[strategy]
id = "momentum_v1"
version = "1.0.0"
description = "Pure momentum with equal weights"

[[pipeline]]
type = "selection"
block_id = "momentum"
params = { lookback_days = 126, top_pct = 20 }

[[pipeline]]
type = "exit"
block_id = "stop_loss"
params = { threshold_pct = 0.10 }

[[pipeline]]
type = "sizing"
block_id = "equal_weight"
params = { max_weight = 0.20 }

[rebalance]
frequency = "weekly"
day = "friday"

[constraints]
max_weight_per_asset = 0.20
min_liquidity_brl = 500000
```

### Seções

| Seção | Descrição |
|-------|-----------|
| `[strategy]` | Metadata: id, version, description |
| `[[pipeline]]` | Array de steps (type, block_id, params) |
| `[rebalance]` | Frequência de rebalanceamento |
| `[constraints]` | Limites de risco |

---

## Pipeline Execution

### Ordem de Execução

```
Selection → Entry → Exit → Sizing
```

### Exemplo Multi-Step

```toml
# 1. Selecionar top 30% por momentum
[[pipeline]]
type = "selection"
block_id = "momentum"
params = { top_pct = 30 }

# 2. Filtrar por qualidade
[[pipeline]]
type = "selection"
block_id = "quality"
params = { min_roe = 0.15 }

# 3. Sinais de entrada
[[pipeline]]
type = "entry"
block_id = "ma_crossover"
params = { fast_period = 20, slow_period = 50 }

# 4. Stop-loss
[[pipeline]]
type = "exit"
block_id = "stop_loss"
params = { threshold_pct = 0.10 }

# 5. Trailing stop
[[pipeline]]
type = "exit"
block_id = "trailing_stop"
params = { trailing_pct = 0.15, activation_pct = 0.10 }

# 6. Sizing
[[pipeline]]
type = "sizing"
block_id = "risk_parity"
params = { max_weight = 0.20 }
```

---

## Adicionando Novo Bloco

### 1. Criar Arquivo

`src/blocks/selection/my_block.rs`:

```rust
use super::{BlockParams, BlockResult, BlockType, StrategyBlock};
use crate::context::StrategyContext;

pub struct MyBlock;

impl StrategyBlock for MyBlock {
    fn block_id(&self) -> &'static str { "my_block" }
    
    fn block_type(&self) -> BlockType { BlockType::Selection }
    
    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        // Lógica aqui
        BlockResult::success("Done")
    }
    
    fn validate_params(&self, _params: &BlockParams) -> Result<(), ValidationError> {
        Ok(())
    }
    
    fn default_params(&self) -> BlockParams {
        HashMap::new()
    }
}
```

### 2. Registrar

Em `src/blocks/selection/mod.rs`:

```rust
pub fn create_selection_block(block_id: &str, params: &BlockParams) -> Option<Box<dyn StrategyBlock>> {
    match block_id {
        "my_block" => Some(Box::new(MyBlock)),
        // ...
    }
}
```

### 3. Adicionar ao Registry

Em `src/registry.rs`:

```rust
fn register_selection_blocks(&mut self) {
    for block_id in ["momentum", "value", ..., "my_block"] {
        // ...
    }
}
```

---

## Testes

```bash
# Testes de compositor
cargo test -p backtester_strategy compositor

# Testes de registry
cargo test -p backtester_strategy registry

# Testes de blocos
cargo test -p backtester_strategy blocks
```





# Estrutura do Genoma - SCG

**Versão**: 1.0.0  
**Última Atualização**: 2025-12-28

## Conceito

No SCG, uma estratégia de trading é representada como um **genoma** - uma estrutura de dados que define completamente a configuração de uma estratégia através de seus blocos e parâmetros.

### Analogia Biológica

| Biologia | SCG |
|----------|-----|
| Cromossomo | StrategyGenome |
| Gene | BlockGene |
| Alelo | ParamValue |
| Fenótipo | Arquivo .toml executável |
| Fitness | Performance (Sharpe, CAGR, etc.) |

---

## StrategyGenome

```rust
/// Genoma completo de uma estratégia
pub struct StrategyGenome {
    /// Identificador único (hash)
    pub id: String,
    
    /// Genes que compõem a estratégia
    pub genes: Vec<BlockGene>,
    
    /// Fitness calculado (após avaliação)
    pub fitness: Option<MultiObjectiveFitness>,
    
    /// Hash do genoma para deduplicação
    pub genome_hash: String,
    
    /// Geração em que foi criado
    pub generation: u32,
    
    /// Origem (random, crossover, mutation)
    pub origin: GenomeOrigin,
}
```

### Invariantes

1. Todo genoma deve ter pelo menos 1 bloco de **Selection**
2. Todo genoma deve ter pelo menos 1 bloco de **Sizing**
3. Todos os parâmetros devem estar dentro dos ranges válidos
4. O `genome_hash` é determinístico para mesmos genes

---

## BlockGene

```rust
/// Gene individual representando um bloco de estratégia
pub struct BlockGene {
    /// Tipo do bloco
    pub block_type: BlockType,
    
    /// ID do bloco (e.g., "momentum", "rsi")
    pub block_id: String,
    
    /// Parâmetros do bloco
    pub params: Vec<(String, ParamValue)>,
}

/// Tipos de blocos disponíveis
pub enum BlockType {
    Selection,  // Seleção de ativos
    Entry,      // Condições de entrada
    Exit,       // Condições de saída
    Sizing,     // Dimensionamento de posição
}
```

### Blocos Disponíveis

| Tipo | Block ID | Descrição |
|------|----------|-----------|
| Selection | `momentum` | Momentum de N dias |
| Selection | `low_vol` | Baixa volatilidade |
| Selection | `value` | Métricas de valor |
| Selection | `quality` | Qualidade/ROE |
| Entry | `always` | Sempre entrar |
| Entry | `macd` | MACD crossover |
| Entry | `rsi` | RSI oversold |
| Entry | `bollinger` | Bollinger breakout |
| Exit | `stop_loss` | Stop loss percentual |
| Exit | `take_profit` | Take profit percentual |
| Exit | `trailing` | Trailing stop |
| Exit | `time_based` | Saída por tempo |
| Sizing | `equal_weight` | Peso igual |
| Sizing | `volatility_target` | Target de volatilidade |
| Sizing | `kelly` | Kelly criterion |

---

## ParamValue

```rust
/// Valor de parâmetro com metadados para mutação
pub enum ParamValue {
    Int {
        value: i64,
        min: i64,
        max: i64,
        step: i64,
    },
    Float {
        value: f64,
        min: f64,
        max: f64,
        step: f64,
    },
    Bool {
        value: bool,
    },
    Enum {
        value: String,
        options: Vec<String>,
    },
}
```

### Exemplos de Parâmetros

| Block | Param | Tipo | Range |
|-------|-------|------|-------|
| momentum | lookback_days | Int | 21-252, step=21 |
| momentum | top_n | Int | 5-50, step=5 |
| stop_loss | pct | Float | 0.01-0.15, step=0.01 |
| rsi | period | Int | 7-21, step=1 |
| rsi | oversold | Float | 20-40, step=5 |
| volatility_target | target | Float | 0.05-0.30, step=0.01 |

---

## Conversão Genoma → TOML

O `GenomeConverter` transforma um genoma em arquivo TOML executável:

```rust
impl GenomeConverter {
    pub fn to_toml(&self, genome: &StrategyGenome) -> Result<String, ConversionError>;
}
```

### Exemplo de Conversão

**Genoma:**
```rust
StrategyGenome {
    genes: vec![
        BlockGene {
            block_type: Selection,
            block_id: "momentum",
            params: vec![
                ("lookback_days", ParamValue::Int { value: 126, .. }),
                ("top_n", ParamValue::Int { value: 20, .. }),
            ],
        },
        BlockGene {
            block_type: Sizing,
            block_id: "equal_weight",
            params: vec![],
        },
        BlockGene {
            block_type: Exit,
            block_id: "stop_loss",
            params: vec![
                ("pct", ParamValue::Float { value: 0.08, .. }),
            ],
        },
    ],
    ..
}
```

**TOML Gerado:**
```toml
[strategy]
name = "scg_genome_a1b2c3d4"
version = "1.0"

[[pipeline.selection]]
block = "momentum"
lookback_days = 126
top_n = 20

[[pipeline.sizing]]
block = "equal_weight"

[[pipeline.exit]]
block = "stop_loss"
pct = 0.08
```

---

## MultiObjectiveFitness

```rust
/// Fitness multi-objetivo para Pareto optimization
pub struct MultiObjectiveFitness {
    /// CAGR (Compound Annual Growth Rate)
    pub cagr: f64,
    
    /// Sharpe Ratio
    pub sharpe_ratio: f64,
    
    /// Maximum Drawdown (valor negativo)
    pub max_drawdown: f64,
    
    /// Calmar Ratio (CAGR / |MaxDD|)
    pub calmar_ratio: f64,
    
    /// Sortino Ratio
    pub sortino_ratio: f64,
    
    /// Profit Factor
    pub profit_factor: f64,
    
    /// Total de trades
    pub total_trades: u32,
    
    /// Volatilidade anualizada
    pub volatility: f64,
    
    /// Turnover anual
    pub turnover_annual: f64,
    
    /// Penalidades aplicadas
    pub penalty_low_trades: f64,
    pub penalty_extreme_turnover: f64,
    
    /// Rank de Pareto (0 = fronteira)
    pub pareto_rank: u32,
    
    /// Crowding distance (diversidade)
    pub crowding_distance: f64,
    
    /// Genoma é válido?
    pub is_valid: bool,
}
```

---

## PopulationFitnessSoA

Para avaliação ultra-rápida, o fitness é armazenado em layout **Struct of Arrays (SoA)**:

```rust
/// Layout SoA para batch processing SIMD
pub struct PopulationFitnessSoA {
    /// Vetores alinhados para SIMD
    pub cagrs: AlignedVec<f64>,
    pub sharpes: AlignedVec<f64>,
    pub max_dds: AlignedVec<f64>,
    pub calmars: AlignedVec<f64>,
    pub pareto_ranks: AlignedVec<u32>,
    pub crowding_distances: AlignedVec<f64>,
    
    /// Capacidade da população
    pub capacity: usize,
}
```

### Benefícios do SoA

| Aspecto | AoS | SoA |
|---------|-----|-----|
| Cache locality | Baixa | Alta |
| SIMD vectorization | Difícil | Natural |
| Pareto calculation | O(n²) | O(n log n) + SIMD |
| Memory bandwidth | Desperdiçada | Otimizada |

---

## Operadores Genéticos

### Mutação de Parâmetro

```rust
fn mutate_param(value: &ParamValue, strength: f64) -> ParamValue {
    match value {
        ParamValue::Int { value, min, max, step } => {
            let delta = (strength * (*max - *min) as f64) as i64;
            let new_value = (*value + random_range(-delta, delta))
                .clamp(*min, *max);
            // Snap to step
            let snapped = (new_value / step) * step;
            ParamValue::Int { value: snapped, ..*value }
        }
        ParamValue::Float { value, min, max, step } => {
            let delta = strength * (*max - *min);
            let new_value = (*value + random_range(-delta, delta))
                .clamp(*min, *max);
            ParamValue::Float { value: new_value, ..*value }
        }
        // ...
    }
}
```

### Crossover de Bloco

```rust
fn crossover_block_level(parent1: &StrategyGenome, parent2: &StrategyGenome) 
    -> (StrategyGenome, StrategyGenome) 
{
    let mut child1_genes = Vec::new();
    let mut child2_genes = Vec::new();
    
    for block_type in [Selection, Entry, Exit, Sizing] {
        let p1_blocks: Vec<_> = parent1.genes_of_type(block_type);
        let p2_blocks: Vec<_> = parent2.genes_of_type(block_type);
        
        // 50% chance de herdar de cada pai
        if random_bool() {
            child1_genes.extend(p1_blocks.clone());
            child2_genes.extend(p2_blocks.clone());
        } else {
            child1_genes.extend(p2_blocks.clone());
            child2_genes.extend(p1_blocks.clone());
        }
    }
    
    (StrategyGenome::new(child1_genes), StrategyGenome::new(child2_genes))
}
```

---

## Validação de Genoma

```rust
impl GenomeValidator {
    pub fn validate(&self, genome: &StrategyGenome) -> Result<(), ValidationError> {
        // 1. Verificar blocos obrigatórios
        self.check_required_blocks(genome)?;
        
        // 2. Verificar parâmetros dentro dos ranges
        self.check_param_ranges(genome)?;
        
        // 3. Verificar compatibilidade entre blocos
        self.check_block_compatibility(genome)?;
        
        // 4. Verificar hash único
        self.check_unique_hash(genome)?;
        
        Ok(())
    }
}
```

---

## Localização no Código

- Crate: `combiner_core`
- Arquivos:
  - `src/genome.rs` - StrategyGenome, BlockGene, ParamValue
  - `src/fitness.rs` - MultiObjectiveFitness
  - `src/fitness_soa.rs` - PopulationFitnessSoA
  - `src/converter.rs` - GenomeConverter
  - `src/validator.rs` - GenomeValidator



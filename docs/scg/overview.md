# Sistema Combinador Generativo (SCG) - Overview

**Versão**: 1.3.0  
**Última Atualização**: 2026-01-04

## O que é o SCG?

O Sistema Combinador Generativo é uma plataforma de descoberta evolutiva de estratégias de trading que utiliza **Algoritmos Genéticos** para explorar automaticamente o espaço de combinações de blocos de estratégia e seus parâmetros.

O SCG pode ser controlado via:
- **CLI** - Comandos de terminal (`combiner run`, `combiner factory`)
- **Dashboard Cockpit** - Interface gráfica com presets, gates e ranking configuráveis

Ver [Cockpit Documentation](../dashboard/cockpit.md) para controle via Dashboard.

### Motivação

A otimização manual de estratégias é:
- **Limitada**: Humanos exploram apenas uma fração do espaço de possibilidades
- **Tendenciosa**: Viés de confirmação e overfitting inconsciente
- **Lenta**: Ciclos de teste demoram horas/dias

O SCG resolve esses problemas através de:
- **Exploração Sistemática**: Populações de centenas de estratégias evoluem simultaneamente
- **Rigor Estatístico**: Validação anti-overfitting integrada (WFA, PBO, DSR)
- **Ultra-Performance**: Avaliação paralela com SIMD e rayon

---

## Princípios de Design

| Princípio | Descrição |
|-----------|-----------|
| **Exploração Genética** | Estratégias são "genomas" que evoluem via seleção natural |
| **Competição Evolutiva** | Torneios selecionam as estratégias mais robustas |
| **Otimização Multi-Objetivo** | Fronteira de Pareto balanceia retorno vs risco |
| **Rigor Anti-Overfitting** | WFA, CPCV, PBO/DSR nativos |
| **Performance Extrema** | Paralelização massiva, SIMD, SoA |

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                    SISTEMA COMBINADOR GENERATIVO                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                   combiner_cli                          │   │
│   │   - run: executa evolução                               │   │
│   │   - validate: valida candidatos                         │   │
│   │   - factory: orquestração de campanhas                  │   │
│   └────────────────────────┬────────────────────────────────┘   │
│                            │                                    │
│   ┌────────────────────────▼────────────────────────────────┐   │
│   │                 combiner_engine                          │   │
│   │                                                          │   │
│   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│   │   │ Population  │  │  Evolution  │  │   Hall of   │     │   │
│   │   │ Generator   │──│   Engine    │──│    Fame     │     │   │
│   │   └─────────────┘  └──────┬──────┘  └─────────────┘     │   │
│   │                           │                              │   │
│   │   ┌───────────────────────▼───────────────────────────┐ │   │
│   │   │            Genetic Operators                       │ │   │
│   │   │  ┌──────────┐  ┌──────────┐  ┌──────────┐        │ │   │
│   │   │  │Selection │  │Crossover │  │ Mutation │        │ │   │
│   │   │  │(Torneio) │  │ (Block)  │  │ (Param)  │        │ │   │
│   │   │  └──────────┘  └──────────┘  └──────────┘        │ │   │
│   │   └───────────────────────────────────────────────────┘ │   │
│   │                                                          │   │
│   │   ┌───────────────────────────────────────────────────┐ │   │
│   │   │           Pareto Frontier (NSGA-II)               │ │   │
│   │   │  Objetivos: CAGR ↑, MaxDD ↓, Sharpe ↑             │ │   │
│   │   └───────────────────────────────────────────────────┘ │   │
│   └────────────────────────┬────────────────────────────────┘   │
│                            │                                    │
│   ┌────────────────────────▼────────────────────────────────┐   │
│   │                 combiner_runner                          │   │
│   │   - Parallel batch evaluation (rayon)                   │   │
│   │   - Data loading & caching                              │   │
│   └────────────────────────┬────────────────────────────────┘   │
│                            │                                    │
│   ┌────────────────────────▼────────────────────────────────┐   │
│   │                  combiner_core                           │   │
│   │   - StrategyGenome, BlockGene                           │   │
│   │   - MultiObjectiveFitness                               │   │
│   │   - PopulationFitnessSoA (SIMD)                         │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Fluxo de Evolução

### 1. Inicialização

```rust
Population::random(config.population_size, block_registry)
```

- Gera N genomas aleatórios
- Cada genoma contém blocos de Selection, Entry, Exit, Sizing
- Parâmetros dentro dos ranges definidos no BlockRegistry

### 2. Avaliação (Stage A - Batch)

```rust
StageABatchEvaluator::evaluate_batch(&population)
```

- Converte cada genoma → TOML
- Executa backtests em paralelo (rayon)
- Calcula fitness multi-objetivo via SIMD
- Cache de resultados para evitar re-avaliação

### 3. Seleção por Torneio

```rust
Selection::tournament(population, k=3)
```

- Seleciona k indivíduos aleatórios
- Vencedor = menos dominado (Pareto)
- Repete até preencher mating pool

### 4. Crossover

```rust
Crossover::block_level(parent1, parent2)
Crossover::uniform(parent1, parent2)
```

- **Block-level**: Troca blocos inteiros entre pais
- **Uniform**: Para cada gene, escolhe aleatoriamente de qual pai herdar

### 5. Mutação

```rust
Mutation::parameter(genome, rate=0.05)
Mutation::block_swap(genome, rate=0.01)
```

- **Parameter**: Altera valores de parâmetros (ruído gaussiano)
- **Block swap**: Substitui bloco por outro do mesmo tipo

### 6. Elitismo

```rust
HallOfFame::update(pareto_frontier)
```

- Top genomas preservados para próxima geração
- Garante que melhores soluções não sejam perdidas

### 7. Validação (Stage B - Paralelo)

```rust
StageBParallelValidator::validate_top_k(hall_of_fame, k=10)
```

- Walk-Forward Analysis completo
- Métricas NET (com custos institucionais)
- PBO/DSR calculation
- Stress testing

---

## Métricas de Fitness

### Objetivos de Otimização

| Objetivo | Direção | Peso Default |
|----------|---------|--------------|
| CAGR | Maximizar | 1.0 |
| Sharpe Ratio | Maximizar | 1.5 |
| Max Drawdown | Minimizar | 1.0 |
| Calmar Ratio | Maximizar | 0.5 |
| Sortino Ratio | Maximizar | 0.5 |

### Penalidades

| Penalidade | Threshold | Valor |
|------------|-----------|-------|
| Low trades | < 30 trades | -0.5 per trade missing |
| Extreme turnover | > 500% anual | -0.1 per 10% excess |
| High volatility | > 40% | -0.2 per 5% excess |

---

## Hiperparâmetros

| Parâmetro | Default | Range Típico |
|-----------|---------|--------------|
| `population_size` | 100 | 50-1000 |
| `max_generations` | 50 | 20-200 |
| `tournament_size` | 3 | 2-7 |
| `crossover_rate` | 0.85 | 0.7-0.95 |
| `mutation_rate` | 0.05 | 0.01-0.15 |
| `elitism_rate` | 0.1 | 0.05-0.2 |
| `hall_of_fame_size` | 25 | 10-100 |

---

## Consumo de Recursos

### Espaço em Disco

> ⚠️ **Importante**: Campanhas SCG consomem espaço significativo.

| Duração | Backtests | Espaço |
|---------|-----------|--------|
| 5 min | ~97k | 6.7 GB |
| 30 min | ~580k | 40 GB |
| 1 hora | ~1.16M | 80 GB |

**Causa principal**: O arquivo `timeseries.csv` (57KB) é gerado para cada backtest e representa **94% do espaço consumido**.

Para análise detalhada, ver [Análise de Armazenamento](../operations/storage-analysis.md).

---

## Artefatos de Saída

```
output/scg/<experiment_id>/
├── manifest.json           # Metadados do experimento
├── report.json             # Relatório final
├── backtests/              # Backtests individuais (94% do espaço)
│   └── <uuid>/
│       ├── metadata.json   (820 B)
│       ├── metrics.json    (502 B)
│       ├── timeseries.csv  (57 KB)  ← Principal consumidor
│       └── trace.jsonl     (1.8 KB)
├── generations/            # Estatísticas por geração
│   ├── gen_000.json
│   ├── gen_001.json
│   └── ...
├── hall_of_fame/           # Top estratégias
│   ├── ranking.json
│   ├── strategy_001/
│   │   ├── config.toml     # Configuração executável
│   │   ├── genome.json     # Genoma completo
│   │   └── metrics.json    # Métricas de fitness
│   └── ...
└── cache/                  # Cache de avaliações
```

---

## Localização no Código

- Crate principal: `combiner_engine`
- Entry point: `crates/combiner_engine/src/engine.rs`
- Símbolos: `EvolutionEngine`, `evolve()`, `GenerationStats`

---

## Próximos Passos

1. [Estrutura do Genoma](genome-structure.md)
2. [Framework de Validação](validation-framework.md)
3. [Referência CLI](cli-reference.md)





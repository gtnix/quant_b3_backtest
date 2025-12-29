# Framework de Validação - SCG

**Versão**: 1.0.0  
**Última Atualização**: 2025-12-28

## Motivação

Uma estratégia de trading só tem valor se sua performance for **genuína e robusta**, não um artefato de overfitting. O SCG incorpora um framework de validação multi-camadas inspirado nas melhores práticas de pesquisa quantitativa institucional.

### O Problema do Overfitting

```
                    Sharpe Ratio
                         │
    Performance     3.0  │     ★ Estratégia "Perfeita"
    In-Sample            │        (Overfitted)
                    2.0  │
                         │
                    1.0  │  ○ Estratégia Robusta
                         │
                    0.0  ├──────────────────────────
                         │      Performance
                         │      Out-of-Sample
```

O overfitting ocorre quando uma estratégia "memoriza" padrões específicos dos dados de treinamento que não se generalizam para dados novos.

---

## Walk-Forward Analysis (WFA)

### Conceito

O WFA simula como uma estratégia teria sido otimizada e negociada em **tempo real**, usando janelas deslizantes de dados.

```
Timeline dos Dados
├─────────────────────────────────────────────────────────────┤
│ Bloco 1 │ Bloco 2 │ Bloco 3 │ Bloco 4 │ Bloco 5 │ Bloco 6 │
├─────────────────────────────────────────────────────────────┤

Passo 1:  [═══IS═══][OOS]
Passo 2:           [═══IS═══][OOS]
Passo 3:                    [═══IS═══][OOS]
Passo 4:                             [═══IS═══][OOS]

IS  = In-Sample (treinamento/otimização)
OOS = Out-of-Sample (teste)
```

### Implementação

```rust
pub struct WfaResult {
    /// Resultados por fold
    pub folds: Vec<WfaFold>,
    
    /// Performance OOS concatenada
    pub oos_equity_curve: Vec<f64>,
    
    /// Métricas OOS agregadas
    pub oos_sharpe: f64,
    pub oos_cagr: f64,
    pub oos_max_dd: f64,
    
    /// Degradação IS → OOS
    pub sharpe_degradation: f64,
    pub cagr_degradation: f64,
}

pub struct WfaFold {
    pub is_start: Date,
    pub is_end: Date,
    pub oos_start: Date,
    pub oos_end: Date,
    
    pub is_sharpe: f64,
    pub oos_sharpe: f64,
    
    pub is_cagr: f64,
    pub oos_cagr: f64,
}
```

### Configuração

```toml
[validation.wfa]
num_folds = 5           # Número de folds
is_ratio = 0.6          # 60% para treinamento
purge_days = 5          # Dias de "purge" entre IS e OOS
embargo_days = 1        # Dias de embargo após OOS
min_oos_trades = 30     # Mínimo de trades por fold OOS
```

---

## Combinatorial Purged Cross-Validation (CPCV)

### Conceito

O CPCV é uma evolução do WFA que testa **todas as combinações** possíveis de blocos de dados, eliminando o viés de seleção de um único caminho.

```
Combinações CPCV (N=5, k=3):

Combo 1: [1,2,3] treino → [4] teste → [5] embargo
Combo 2: [1,2,4] treino → [3] teste → [5] embargo
Combo 3: [1,2,5] treino → [3] teste → [4] embargo
...
Combo C(5,3) = 10 combinações
```

### Implementação

```rust
pub struct CpcvResult {
    /// Número de combinações testadas
    pub num_combinations: usize,
    
    /// Sharpe médio OOS de todas as combinações
    pub mean_oos_sharpe: f64,
    
    /// Desvio padrão do Sharpe OOS
    pub std_oos_sharpe: f64,
    
    /// Performance por combinação
    pub combinations: Vec<CpcvCombination>,
    
    /// Estatística t para significância
    pub t_statistic: f64,
    pub p_value: f64,
}
```

### Quando Usar

- **WFA**: Validação rápida durante evolução
- **CPCV**: Validação final das candidatas top (computacionalmente intensivo)

---

## Probability of Backtest Overfitting (PBO)

### Conceito

O PBO calcula a **probabilidade** de que uma estratégia com determinada performance seja resultado de overfitting, dado o número de estratégias testadas.

```
PBO = P(rank_oos > N/2 | rank_is = 1)
```

Onde:
- `rank_is` = Rank da estratégia in-sample
- `rank_oos` = Rank da estratégia out-of-sample
- `N` = Número de estratégias testadas

### Intuição

Se você testa 1000 estratégias e escolhe a melhor, há alta probabilidade de que ela seja "sortuda" no in-sample mas medíocre no out-of-sample.

### Implementação

```rust
pub struct PboDsrResult {
    /// Probability of Backtest Overfitting
    pub pbo: f64,
    
    /// Deflated Sharpe Ratio
    pub dsr: f64,
    
    /// Sharpe observado
    pub observed_sharpe: f64,
    
    /// Número de trials (estratégias testadas)
    pub num_trials: usize,
    
    /// Skewness dos retornos
    pub skewness: f64,
    
    /// Kurtosis dos retornos
    pub kurtosis: f64,
}
```

### Thresholds Institucionais

| PBO | Interpretação |
|-----|---------------|
| < 0.10 | Baixa probabilidade de overfitting |
| 0.10 - 0.20 | Moderada - avaliar com cuidado |
| 0.20 - 0.40 | Alta - provavelmente overfitting |
| > 0.40 | Muito alta - rejeitar |

---

## Deflated Sharpe Ratio (DSR)

### Conceito

O DSR ajusta o Sharpe Ratio observado para baixo, compensando pelo viés de seleção múltipla.

```
DSR = SR_hat × (1 - PBO)
```

Onde:
- `SR_hat` = Sharpe Ratio observado
- `PBO` = Probability of Backtest Overfitting

### Fórmula Completa (Bailey et al.)

```rust
fn deflated_sharpe_ratio(
    sharpe_observed: f64,
    num_trials: usize,
    skewness: f64,
    kurtosis: f64,
    track_record_length: usize,
) -> f64 {
    let sr_star = expected_max_sharpe(num_trials);
    let sr_std = sr_standard_deviation(track_record_length, skewness, kurtosis);
    
    // Z-score do Sharpe observado vs esperado
    let z = (sharpe_observed - sr_star) / sr_std;
    
    // Probabilidade de observar SR maior que SR*
    let pbo = 1.0 - normal_cdf(z);
    
    sharpe_observed * (1.0 - pbo)
}
```

### Interpretação

| DSR | Interpretação |
|-----|---------------|
| > 1.5 | Excelente - provavelmente genuíno |
| 1.0 - 1.5 | Bom - estatisticamente significativo |
| 0.5 - 1.0 | Marginal - precisa mais dados |
| < 0.5 | Fraco - provavelmente ruído |

---

## Stress Testing

### Cenários de Stress

```rust
pub enum StressScenario {
    /// 2x slippage normal
    HighSlippage,
    
    /// 2x custos normais
    HighCosts,
    
    /// Execução no close (delay)
    DelayedExecution,
    
    /// Metade da liquidez
    LowLiquidity,
    
    /// Combinação adversa
    AdverseConditions,
}
```

### Implementação

```rust
pub struct StressTestResult {
    pub scenario: StressScenario,
    
    /// Métricas sob stress
    pub stressed_sharpe: f64,
    pub stressed_cagr: f64,
    pub stressed_max_dd: f64,
    
    /// Degradação vs base
    pub sharpe_degradation_pct: f64,
    
    /// Passou no threshold?
    pub passed: bool,
}

impl StressTest {
    pub fn run(&self, genome: &StrategyGenome) -> Vec<StressTestResult> {
        SCENARIOS.par_iter()
            .map(|scenario| self.evaluate_scenario(genome, scenario))
            .collect()
    }
}
```

### Critérios de Aprovação

| Cenário | Threshold |
|---------|-----------|
| HighSlippage | Sharpe > 0.3 |
| HighCosts | Sharpe > 0.3 |
| DelayedExecution | Sharpe > 0.25 |
| LowLiquidity | Sharpe > 0.2 |
| AdverseConditions | Sharpe > 0.15 |

---

## Métricas NET vs GROSS

### GROSS (Sem custos)

- Útil para comparação rápida durante evolução
- Não reflete realidade de trading

### NET (Com custos institucionais)

- Inclui: comissões, emolumentos B3, slippage, market impact
- Obrigatório para candidatos finais

```toml
[execution]
# Modelo de custos B3 institucional
commission_bps = 5.0        # 5 bps por lado
emolument_bps = 2.5         # Emolumentos B3
slippage_model = "volume"   # Proporcional ao volume
slippage_bps = 10.0         # Base slippage
market_impact_model = "linear"
```

### IS/OOS Degradation

```rust
pub fn calculate_degradation(is_metric: f64, oos_metric: f64) -> f64 {
    if is_metric.abs() < 1e-10 {
        return 0.0;
    }
    (is_metric - oos_metric) / is_metric.abs()
}
```

| Degradation | Interpretação |
|-------------|---------------|
| < 20% | Excelente - estratégia robusta |
| 20-40% | Aceitável - normal para maioria |
| 40-60% | Preocupante - possível overfitting |
| > 60% | Crítico - provável overfitting |

---

## Pipeline de Validação

```
┌─────────────────────────────────────────────────────────────┐
│                 PIPELINE DE VALIDAÇÃO                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Stage A: Avaliação Rápida (GROSS)                         │
│  ├── Backtest simples                                      │
│  ├── Fitness multi-objetivo                                │
│  └── Ranking Pareto                                        │
│           │                                                 │
│           ▼                                                 │
│  Gate 1: Sharpe GROSS > 0.5?  ─────No───→ REJEITAR         │
│           │                                                 │
│          Yes                                                │
│           ▼                                                 │
│  Stage B: Validação Completa (NET)                         │
│  ├── Walk-Forward Analysis                                 │
│  ├── Métricas NET                                          │
│  ├── PBO/DSR calculation                                   │
│  └── Stress testing                                        │
│           │                                                 │
│           ▼                                                 │
│  Gate 2: OOS Sharpe NET > 0.5?  ───No───→ REJEITAR         │
│           │                                                 │
│          Yes                                                │
│           ▼                                                 │
│  Gate 3: PBO < 0.15?  ─────────────No───→ REJEITAR         │
│           │                                                 │
│          Yes                                                │
│           ▼                                                 │
│  Gate 4: Stress passed >= 4/5?  ───No───→ REJEITAR         │
│           │                                                 │
│          Yes                                                │
│           ▼                                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         CANDIDATO VALIDADO                          │   │
│  │  → Salvo em artifacts/candidates/                   │   │
│  │  → Elegível para promoção                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Configuração Completa

```toml
[validation]
enabled = true
top_k = 10              # Validar top K candidatos

[validation.wfa]
num_folds = 5
is_ratio = 0.6
purge_days = 5
embargo_days = 1
min_oos_trades = 30

[validation.cpcv]
enabled = false         # Apenas para validação final
num_combinations = 10   # Limitar para performance

[validation.pbo]
enabled = true
significance_level = 0.05

[validation.stress]
enabled = true
min_scenarios_pass = 4

[validation.thresholds]
min_oos_sharpe_net = 0.5
max_pbo = 0.15
max_degradation = 0.40
min_trades_oos = 30
```

---

## Localização no Código

- Crate: `combiner_engine`
- Arquivos:
  - `src/validation.rs` - GenomeValidatorAntiOverfit, WfaResult, CpcvResult
  - `src/evaluation/stage_b.rs` - StageBParallelValidator
  - `src/evaluation/stress.rs` - StressTest
- Crate: `backtester_intelligence`
  - `src/walkforward/` - Walk-Forward Analysis engine

---

## Referências

1. Bailey, D. H., & López de Prado, M. (2015). *The Deflated Sharpe Ratio*
2. Bailey, D. H., et al. (2017). *Probability of Backtest Overfitting*
3. López de Prado, M. (2018). *Advances in Financial Machine Learning*





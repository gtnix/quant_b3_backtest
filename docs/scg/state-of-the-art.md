# Estado da Arte: Algoritmos Genéticos para Trading

**Versão**: 1.0.0  
**Última Atualização**: 2026-01-02  
**Autor**: SCG Research Team

---

## 1. Introdução

Este documento apresenta a fundamentação acadêmica e as técnicas de estado da arte implementadas no Sistema Combinador Generativo (SCG) para descoberta evolutiva de estratégias de trading. O objetivo é resolver problemas identificados como **convergência prematura** e **duplicação de estratégias**, elevando o sistema ao nível de rigor acadêmico internacional.

### 1.1 Problemas Identificados

| Problema | Evidência | Causa Raiz |
|----------|-----------|------------|
| Estratégias com Sharpe idêntico | 380 estratégias com Sharpe 0.4345 | Convergência prematura |
| Genomas diferentes, fitness idêntico | Hash único, métricas iguais | Falta de diversidade fenotípica |
| Elitismo excessivo | 10% da população preservada | Perda de exploração |
| Sem detecção de estagnação | Evolução sem melhoria | Desperdício computacional |

---

## 2. Fundamentação Acadêmica

### 2.1 Referências Principais

| Fonte | Ano | Contribuição | Aplicação no SCG |
|-------|-----|--------------|------------------|
| Deb, K. et al. - NSGA-II | 2002 | Fast non-dominated sorting, crowding distance | Implementado em `pareto_simd.rs` |
| Lopez de Prado, M. | 2018 | PBO, DSR, CPCV anti-overfitting | Implementado em `validation.rs` |
| Goldberg & Richardson | 1987 | Fitness Sharing | Novo: `diversity.rs` |
| Eiben & Smith | 2003 | Adaptive Mutation | Novo: `AdaptiveMutation` |
| Zhang & Li - MOEA/D | 2007 | Decomposition-based MOEA | Futuro: ensemble |
| Lehman & Stanley | 2011 | Novelty Search | Novo: diversidade comportamental |
| Laumanns et al. | 2002 | ε-Dominance | Futuro: archive management |

### 2.2 Artigos Específicos para Trading

| Artigo | Fonte | Contribuição |
|--------|-------|--------------|
| "Collaborative MOEAs for Trading Systems" | arXiv:2211.02451 | Ensemble de algoritmos evolutivos |
| "GA-MSSR for Forex Trading" | arXiv:2008.09471 | Otimização Sharpe-Sterling |
| "Feature Engineering with GA+SVM" | arXiv:1809.06775 | Seleção automática de features |
| "Alpha Discovery Neural Network" | arXiv:1912.11761 | Indicadores técnicos via GP |
| "Algoritmos Genéticos na B3" | USP/BDTA | Otimização de carteiras Brasil |

---

## 3. Técnicas de Preservação de Diversidade

### 3.1 Fitness Sharing (Goldberg & Richardson, 1987)

A técnica de **Fitness Sharing** penaliza soluções que estão muito próximas no espaço de busca, forçando a população a explorar regiões diferentes.

**Fórmula Matemática:**

```
sh(d) = 1 - (d / σ_share)^α    se d < σ_share
sh(d) = 0                       caso contrário

fitness_compartilhado(i) = fitness_raw(i) / Σ sh(d(i,j))  ∀j ∈ população
```

**Parâmetros:**
- `σ_share` (sigma share): Raio do nicho. Sugestão: 0.10 no espaço normalizado
- `α` (alpha): Controla a forma da função. Típico: 1.0 (linear) ou 2.0 (quadrático)

**Implementação no SCG:**

```rust
pub fn compute_shared_fitness(
    population: &mut [StrategyGenome],
    sigma_share: f64,
    alpha: f64,
) {
    for i in 0..population.len() {
        let mut niche_count = 0.0;
        
        for j in 0..population.len() {
            let distance = phenotypic_distance(&population[i], &population[j]);
            if distance < sigma_share {
                niche_count += 1.0 - (distance / sigma_share).powf(alpha);
            }
        }
        
        if let Some(ref mut f) = population[i].fitness {
            f.shared_fitness = f.raw_fitness / niche_count.max(1.0);
        }
    }
}
```

### 3.2 Distância Fenotípica vs Genotípica

**Problema Identificado:** Estratégias com genomas diferentes podem ter comportamentos idênticos (fenótipos iguais).

**Solução:** Medir diversidade no espaço de **fenótipos** (métricas de performance), não apenas no espaço de **genótipos** (genes).

| Tipo | Medição | Fórmula |
|------|---------|---------|
| Genotípica | Distância de Hamming | Número de genes diferentes |
| Fenotípica | Distância Euclidiana | √(ΔSharpe² + ΔCAGR² + ΔMaxDD²) |
| Comportamental | Correlação de Retornos | 1 - corr(returns_a, returns_b) |

### 3.3 ε-Dominance (Laumanns et al., 2002)

Divide o espaço objetivo em células de tamanho ε e mantém apenas uma solução por célula:

```
célula(i) = floor(objetivo(i) / ε)
```

**Benefícios:**
- Garante diversidade uniforme na fronteira de Pareto
- Limita tamanho do arquivo de soluções
- Evita acumulação em regiões "fáceis"

---

## 4. Mutação Adaptativa

### 4.1 Motivação

Taxa de mutação fixa é subótima:
- **Muito baixa**: Convergência prematura, preso em ótimos locais
- **Muito alta**: Busca aleatória, perda de boas soluções

### 4.2 Fórmula (Eiben & Smith, 2003)

```
rate(t) = rate_base × (1 + k × (1 - diversity(t)))
```

Onde:
- `rate_base`: Taxa base (0.05)
- `k`: Fator de amplificação (2.0)
- `diversity(t)`: Diversidade atual da população [0, 1]

### 4.3 Implementação

```rust
pub struct AdaptiveMutation {
    base_rate: f64,      // 0.05
    min_rate: f64,       // 0.01
    max_rate: f64,       // 0.30
    amplification: f64,  // 2.0
}

impl AdaptiveMutation {
    pub fn current_rate(&self, diversity: f64) -> f64 {
        let adjustment = 1.0 + self.amplification * (1.0 - diversity);
        (self.base_rate * adjustment).clamp(self.min_rate, self.max_rate)
    }
}
```

### 4.4 Comportamento Esperado

| Diversidade | Taxa de Mutação | Comportamento |
|-------------|-----------------|---------------|
| Alta (0.8+) | ~0.05 (base) | Exploração normal |
| Média (0.5) | ~0.10 | Intensificação moderada |
| Baixa (0.2) | ~0.20 | Alta exploração |
| Crítica (<0.1) | 0.30 (max) | Modo de emergência |

---

## 5. Detecção de Estagnação e Restart

### 5.1 Conceito

Quando a evolução para de melhorar, é mais eficiente reiniciar com nova população do que continuar otimizando.

### 5.2 Critérios de Estagnação (De Jong, 1975)

```
estagnado = (best_fitness[t] - best_fitness[t-w]) / |best_fitness[t-w]| < threshold
```

**Parâmetros recomendados:**
- `window_size (w)`: 10 gerações
- `improvement_threshold`: 0.5% (0.005)

### 5.3 Estratégia de Restart

1. **Preservar Elite**: Manter top 20% da população atual
2. **Regenerar**: Criar 80% novos indivíduos aleatórios
3. **Boost de Mutação**: Aumentar taxa de mutação por 5 gerações
4. **Reset Diversidade**: Limpar histórico de fitness sharing

### 5.4 Implementação

```rust
struct StagnationDetector {
    best_fitness_history: VecDeque<f64>,
    window_size: usize,
    improvement_threshold: f64,
    generations_since_improvement: u32,
}

impl StagnationDetector {
    pub fn is_stagnant(&self) -> bool {
        if self.best_fitness_history.len() < self.window_size {
            return false;
        }
        
        let oldest = self.best_fitness_history.front().unwrap();
        let newest = self.best_fitness_history.back().unwrap();
        
        let improvement = (newest - oldest) / oldest.abs().max(1e-10);
        improvement < self.improvement_threshold
    }
    
    pub fn trigger_restart(&self, population: &mut Population, elite_ratio: f64) {
        let elite_count = (population.len() as f64 * elite_ratio) as usize;
        
        // Ordenar por fitness e preservar elite
        population.sort_by_fitness();
        let elite: Vec<_> = population.genomes[..elite_count].to_vec();
        
        // Regenerar resto
        let new_count = population.len() - elite_count;
        let new_genomes = Population::random(new_count, &population.param_ranges);
        
        // Combinar
        population.genomes = elite;
        population.genomes.extend(new_genomes.genomes);
    }
}
```

---

## 6. Métricas de Diversidade

### 6.1 Diversidade Genotípica

```rust
pub fn genotypic_diversity(population: &[StrategyGenome]) -> f64 {
    let n = population.len();
    if n < 2 { return 1.0; }
    
    let mut total_distance = 0.0;
    let mut count = 0;
    
    for i in 0..n {
        for j in (i+1)..n {
            total_distance += hamming_distance(&population[i], &population[j]);
            count += 1;
        }
    }
    
    total_distance / count as f64
}
```

### 6.2 Diversidade Fenotípica

```rust
pub fn phenotypic_diversity(population: &[StrategyGenome]) -> f64 {
    let valid: Vec<_> = population.iter()
        .filter(|g| g.fitness.is_some())
        .collect();
    
    if valid.len() < 2 { return 1.0; }
    
    let mut total_distance = 0.0;
    let mut count = 0;
    
    for i in 0..valid.len() {
        for j in (i+1)..valid.len() {
            total_distance += phenotypic_distance(valid[i], valid[j]);
            count += 1;
        }
    }
    
    // Normalizar para [0, 1]
    (total_distance / count as f64).min(1.0)
}
```

### 6.3 Entropia Estrutural (Shannon)

Mede a distribuição de tipos de blocos na população:

```rust
pub fn structural_entropy(population: &[StrategyGenome]) -> f64 {
    let mut block_counts: HashMap<String, usize> = HashMap::new();
    let mut total = 0;
    
    for genome in population {
        for gene in &genome.genes {
            *block_counts.entry(gene.block_id.clone()).or_insert(0) += 1;
            total += 1;
        }
    }
    
    // Entropia de Shannon
    let mut entropy = 0.0;
    for count in block_counts.values() {
        let p = *count as f64 / total as f64;
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }
    
    // Normalizar pela entropia máxima
    let max_entropy = (block_counts.len() as f64).ln();
    if max_entropy > 0.0 { entropy / max_entropy } else { 0.0 }
}
```

---

## 7. Configuração Recomendada

### 7.1 Parâmetros de Evolução

| Parâmetro | Valor Atual | Valor Estado da Arte | Justificativa |
|-----------|-------------|---------------------|---------------|
| `population_size` | 100 | 300 | Maior diversidade inicial |
| `max_generations` | 50 | 150 | Mais tempo para exploração |
| `tournament_size` | 3 | 5 | Maior pressão seletiva |
| `crossover_rate` | 0.85 | 0.80 | Ligeiramente menor |
| `elitism_rate` | 0.10 | 0.05 | Menos elitismo = mais exploração |
| `mutation_rate` | 0.10 | Adaptativo (0.01-0.30) | Dinâmico |

### 7.2 Parâmetros de Diversidade

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `sigma_share` | 0.10 | Raio do nicho no espaço fenotípico |
| `fitness_sharing_alpha` | 1.0 | Expoente da função de sharing |
| `diversity_weight` | 0.3 | Peso da diversidade no fitness |

### 7.3 Parâmetros de Estagnação

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `stagnation_window` | 10 | Gerações para detectar estagnação |
| `improvement_threshold` | 0.005 | 0.5% melhoria mínima |
| `restart_elite_ratio` | 0.20 | Preservar top 20% no restart |
| `post_restart_mutation_boost` | 5 | Gerações com mutação alta |

---

## 8. Validação Anti-Overfitting

### 8.1 Combinatorial Purged Cross-Validation (CPCV)

Técnica de Lopez de Prado para evitar vazamento de dados em séries temporais:

```
1. Dividir dados em N blocos
2. Para cada combinação de k blocos para treino:
   a. Usar blocos restantes para teste
   b. Aplicar "purge" entre treino e teste
   c. Registrar performance OOS
3. Calcular média e variância do Sharpe OOS
```

### 8.2 Probability of Backtest Overfitting (PBO)

```
PBO = P(rank_OOS > N/2 | rank_IS = 1)
```

**Thresholds:**
- PBO < 0.10: Baixo risco de overfitting
- PBO 0.10-0.20: Moderado
- PBO > 0.20: Alto risco - rejeitar

### 8.3 Deflated Sharpe Ratio (DSR)

```
DSR = SR_observado × (1 - PBO)
```

Ajusta o Sharpe pelo número de estratégias testadas.

---

## 9. Próximos Passos (Roadmap)

### Fase 1: Implementação Core (Atual)
- [x] Documentação de estado da arte
- [ ] DiversityMonitor
- [ ] AdaptiveMutation
- [ ] Phenotypic Distance
- [ ] Fitness Sharing
- [ ] StagnationDetector

### Fase 2: Novos Blocos de Trading
- [ ] `multi_factor` (Fama-French)
- [ ] `volatility_stop` (ATR-based)
- [ ] `regime_exit` (Hamilton switching)

### Fase 3: Técnicas Avançadas
- [ ] ε-Dominance archive
- [ ] MOEA/D decomposition
- [ ] Island Model parallelism
- [ ] Novelty Search híbrido

---

## 10. Referências Bibliográficas

1. Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II." *IEEE Transactions on Evolutionary Computation*, 6(2), 182-197.

2. Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.

3. Goldberg, D. E., & Richardson, J. (1987). "Genetic Algorithms with Sharing for Multimodal Function Optimization." *Proceedings of the Second International Conference on Genetic Algorithms*, 41-49.

4. Eiben, A. E., & Smith, J. E. (2003). *Introduction to Evolutionary Computing*. Springer.

5. Zhang, Q., & Li, H. (2007). "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition." *IEEE Transactions on Evolutionary Computation*, 11(6), 712-731.

6. Lehman, J., & Stanley, K. O. (2011). "Abandoning Objectives: Evolution Through the Search for Novelty Alone." *Evolutionary Computation*, 19(2), 189-223.

7. Laumanns, M., Thiele, L., Deb, K., & Zitzler, E. (2002). "Combining Convergence and Diversity in Evolutionary Multiobjective Optimization." *Evolutionary Computation*, 10(3), 263-282.

8. Soltero, F. J., et al. (2022). "Collaborative Multiobjective Evolutionary Algorithms in search of better Pareto Fronts." *arXiv:2211.02451*.

9. Oliveira, G. I., & Maciel, L. (2020). "Algoritmos genéticos na otimização de carteiras de ações no Brasil." *Universidade de São Paulo*.

10. Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality." *Journal of Portfolio Management*, 40(5), 94-107.

---

## Localização no Código

| Componente | Arquivo |
|------------|---------|
| Pareto SIMD | `crates/combiner_engine/src/pareto_simd.rs` |
| Operadores Genéticos | `crates/combiner_engine/src/operators.rs` |
| Motor de Evolução | `crates/combiner_engine/src/engine.rs` |
| Diversidade | `crates/combiner_engine/src/diversity.rs` |
| Fitness SoA | `crates/combiner_core/src/fitness_soa.rs` |
| Validação | `crates/combiner_engine/src/validation.rs` |













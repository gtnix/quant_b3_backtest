# Parameter Universe System - Contrato de Precedência

Este documento define formalmente a ordem de precedência para configurações do Parameter Universe System.

## Princípio Fundamental

> **Valores mais restritivos sempre vencem.**

Quando múltiplas fontes definem o mesmo parâmetro, o sistema aplica o valor mais conservador (mínimo do máximo, máximo do mínimo).

## Ordem de Precedência

A tabela abaixo define a prioridade de cada fonte de configuração, da mais alta (1) para a mais baixa (6):

| Prioridade | Fonte | Localização | Comportamento |
|------------|-------|-------------|---------------|
| 1 | TOML da campanha | `configs/campaigns/*.toml` seção `[evolution]` e `[universe.overrides]` | Sobrescreve tudo, mas limitado pelos níveis abaixo |
| 2 | Training Tech | `configs/training_tech/*.toml` | Define máximos de workers, timeout, complexity tiers |
| 3 | Risk Profile Restrictions | `configs/risk_profiles/*.toml` seção `[universe_restrictions]` | Define máximos de population, generations, famílias permitidas |
| 4 | Parameter Bounds | `configs/parameter_bounds/*.toml` | Define ranges por família de estratégia |
| 5 | Compatibility Matrix | `configs/compatibility_matrix.toml` | Define combinações válidas entre eixos |
| 6 | Defaults globais | `ParamRanges::new()` em código Rust | Fallback quando nada especificado |

## Regras de Aplicação

### 1. Population Size

```
effective_population = min(
    campaign.evolution.population_size  OR  DEFAULT,
    training_tech.evolution.population_size,
    risk_profile.universe_restrictions.max_population_size
)
```

### 2. Max Generations

```
effective_generations = min(
    campaign.evolution.max_generations  OR  DEFAULT,
    training_tech.evolution.max_generations,
    risk_profile.universe_restrictions.max_generations
)
```

### 3. Parameters to Optimize

```
effective_max_params = min(
    campaign.universe.overrides.max_parameters  OR  UNLIMITED,
    training_tech.allowed_complexity.max_parameters_to_optimize,
    risk_profile.universe_restrictions.max_parameters_to_optimize
)
```

### 4. Strategy Families Permitidas

```
allowed_families = intersection(
    training_model_to_robustness[robustness_profile],
    risk_profile.universe_restrictions.allowed_strategy_families
)
```

### 5. Complexity Tier

```
allowed_tiers = training_tech_to_complexity[training_tech]
model_tier = training_model_complexity[training_model]

VALID if model_tier IN allowed_tiers
```

## Comportamento de Fallback

Quando uma configuração está ausente, o sistema usa defaults seguros:

| Cenário | Fallback |
|---------|----------|
| Sem `[universe]` no TOML | `robustness=moderado, strategy=purged_kfold, tech=cpu_parallel, model=["swing"]` |
| Sem `[universe_restrictions]` no risk profile | Não aplica restrições adicionais (usa training_tech) |
| Arquivo de bounds não encontrado | Usa ranges padrão do `ParamRanges::new()` |
| Compatibility matrix ausente | Validação de compatibilidade desabilitada |
| Training strategy não encontrada | Erro claro: `TrainingStrategyNotFound` |
| Training tech não encontrada | Erro claro: `TrainingTechNotFound` |

## Exemplos

### Exemplo 1: Configuração Completa

```toml
# Campaign: configs/campaigns/my_campaign.toml
[evolution]
population_size = 200
max_generations = 150

[risk_profile]
name = "moderado"

[universe]
robustness_profile = "moderado"
training_strategy = "purged_kfold"
training_tech = "cpu_parallel"
training_model = "swing"

[universe.overrides]
max_parameters = 8
```

**Resultado:**
- `population_size = min(200, 150, 200) = 150` (training_tech.cpu_parallel = 150)
- `max_generations = min(150, 100, 150) = 100` (training_tech.cpu_parallel = 100)
- `max_parameters = min(8, 10, 10) = 8` (override explícito)

### Exemplo 2: Backward Compatible (sem universe)

```toml
# Campaign: configs/campaigns/old_campaign.toml
[campaign]
name = "old_style"

[evolution]
population_size = 100
max_generations = 50

[risk_profile]
name = "moderado"
```

**Resultado:**
- Universe defaults aplicados
- Nenhuma validação de compatibilidade de eixos
- Configuração funciona exatamente como antes

### Exemplo 3: Conflito de Restrições

```toml
[universe]
robustness_profile = "muito_conservador"
training_tech = "cpu_intensive"  # Permite tier3
training_model = "intraday"       # Requer perfil arrojado
```

**Resultado:**
- **ERRO**: `FamilyNotAllowed("intraday", "muito_conservador")`
- Mensagem: "Strategy family 'intraday' not allowed for robustness profile 'muito_conservador'"

## Implementação

A aplicação de precedência ocorre em:

1. **`execute_single_run()`** em `run_campaign.rs`:
   - Carrega universe config
   - Valida compatibilidade
   - Aplica limites efetivos ao `EvolutionConfig`

2. **`UniverseValidator::get_effective_limits()`**:
   - Calcula limites finais após aplicar todas as restrições

3. **`ParamRanges::with_restrictions()`**:
   - Filtra blocks e ranges baseado em famílias permitidas

## Observabilidade

Logs são emitidos quando restrições são aplicadas:

```
INFO Universe restrictions applied: pop=150, gen=100, families=["swing"]
```

Erros são claros e indicam a ação corretiva:

```
ERROR Universe validation failed: Strategy family 'intraday' not allowed 
      for robustness profile 'muito_conservador'. 
      Check compatibility between robustness profile and training model.
```

## Versionamento

- **v1.0** (2026-01-06): Documento inicial
- Seção `[universe]` é opcional (backward compatible)
- Todos os campos dentro de `[universe]` têm defaults

---

*Documento de contrato para o Parameter Universe System. Qualquer mudança neste contrato requer atualização da documentação e dos testes de integração.*



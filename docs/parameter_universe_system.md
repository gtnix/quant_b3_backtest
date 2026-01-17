# Parameter Universe System

Sistema que controla e limita a geração de estratégias de trading através de 4 eixos.

## Visão Geral

O Parameter Universe System foi projetado para:
- Reduzir o espaço de busca do Algoritmo Genético sem alterar sua lógica
- Garantir compatibilidade entre diferentes configurações
- Facilitar a configuração via UI e TOML
- Manter backward compatibility com configs existentes

## Os 4 Eixos

### 1. Robustness Profile (Perfil de Robustez)

Define o nível de risco aceitável e restringe parâmetros de acordo.

| Profile | Kelly | Max DD | Famílias Permitidas |
|---------|-------|--------|---------------------|
| muito_conservador | 0.15 | 8% | position, portfolio, factor, seasonal, buy_hold |
| conservador | 0.30 | 12% | swing, position, pair, portfolio, momentum, mean_reversion, sector_rotation, factor, seasonal, buy_hold, multi_strategy |
| moderado | 0.40 | 20% | + breakout, volatility, event_driven |
| arrojado | 0.50 | 25% | + intraday |
| muito_arrojado | 0.50 | 30% | intraday, swing, momentum, breakout, volatility, event_driven |

**Configuração:** `configs/risk_profiles/*.toml`

### 2. Training Strategy (Estratégia de Treino)

Define como a estratégia é validada durante a otimização.

| Strategy | Descrição | Folds | Tier |
|----------|-----------|-------|------|
| purged_kfold | Cross-validation padrão com purge | 5 | tier2_medium |
| walk_forward | Rolling window validation | 8 | tier2_medium |
| anchored | Fixed anchor, expanding window | - | tier1_fast |
| expanding_window | Incrementally expanding training | 6 | tier2_medium |
| monte_carlo | Stress testing extensivo | 10 | tier3_slow |

**Configuração:** `configs/training_strategies/*.toml`

### 3. Training Tech (Tecnologia de Treino)

Define recursos computacionais disponíveis.

| Tech | Workers | Timeout | Complexity Tiers |
|------|---------|---------|------------------|
| cpu_fast | 4 | 30min | tier1_fast |
| cpu_parallel | 8 | 2h | tier1_fast, tier2_medium |
| cpu_intensive | 16 | 6h | tier1_fast, tier2_medium, tier3_slow |
| distributed | auto | 24h | todos |

**Configuração:** `configs/training_tech/*.toml`

### 4. Training Model (Modelo de Treino / Família de Estratégia)

Define que tipos de estratégias podem ser gerados.

| Family | Tier | Holding Period |
|--------|------|----------------|
| intraday | tier1_fast | 1-8 horas |
| swing | tier1_fast | 2-10 dias |
| position | tier2_medium | Semanas-Meses |
| pair | tier2_medium | 5-30 dias |
| portfolio | tier3_slow | Rebalanceamento |
| momentum | tier1_fast | 1-6 meses |
| mean_reversion | tier1_fast | 2-10 dias |
| breakout | tier1_fast | 3-15 dias |
| sector_rotation | tier2_medium | 1-6 meses |
| factor | tier3_slow | Meses-Anos |
| seasonal | tier2_medium | Dias-Semanas |
| volatility | tier2_medium | 3-15 dias |
| event_driven | tier2_medium | 1-5 dias |
| buy_hold | tier1_fast | Anos |
| multi_strategy | tier3_slow | Variável |

**Configuração:** `configs/parameter_bounds/*.toml`

## Matriz de Compatibilidade

A matriz de compatibilidade (`configs/compatibility_matrix.toml`) define:

1. **Robustness → Training Strategy**: Perfis mais conservadores só permitem validações mais rigorosas
2. **Training Model → Robustness**: Estratégias de alta frequência requerem perfis mais arrojados
3. **Training Tech → Complexity Tier**: Recursos computacionais limitam complexidade
4. **Training Strategy → Min Data**: Cada estratégia de treino tem requisitos mínimos de dados

## Uso em Campaign TOML

```toml
[universe]
robustness_profile = "moderado"
training_strategy = "purged_kfold"
training_tech = "cpu_parallel"
training_model = "swing"  # ou ["swing", "momentum"]

# Overrides opcionais
[universe.overrides]
max_parameters = 10
allowed_indicators = ["SMA", "EMA", "RSI", "MACD", "ATR"]
```

## Validação

O sistema valida automaticamente:
- Combinações entre eixos são compatíveis
- Parâmetros estão dentro dos bounds definidos
- Recursos computacionais suportam a complexidade
- Dados mínimos estão disponíveis

Erros de validação são claros e orientados ao usuário.

## Backward Compatibility

- Seção `[universe]` é opcional
- Sem ela, comportamento atual é mantido
- Defaults sensíveis são aplicados automaticamente

## Arquivos do Sistema

```
configs/
├── compatibility_matrix.toml      # Matriz de compatibilidade
├── risk_profiles/                  # Perfis de robustez (+ universe_restrictions)
│   ├── muito_conservador.toml
│   ├── conservador.toml
│   ├── moderado.toml
│   ├── arrojado.toml
│   └── muito_arrojado.toml
├── training_strategies/            # Estratégias de validação
│   ├── purged_kfold.toml
│   ├── walk_forward.toml
│   ├── anchored.toml
│   ├── expanding_window.toml
│   └── monte_carlo.toml
├── training_tech/                  # Configurações de recursos
│   ├── cpu_fast.toml
│   ├── cpu_parallel.toml
│   ├── cpu_intensive.toml
│   └── distributed.toml
└── parameter_bounds/               # Bounds por família
    ├── swing.toml
    ├── momentum.toml
    └── position.toml
```

## Módulo Rust

```rust
use backtester_strategy::{
    UniverseLoader, UniverseValidator, UniverseConfig,
    ComplexityTier, StrategyFamily, TrainingModel,
};

// Carregar configurações
let mut loader = UniverseLoader::new("configs/");
loader.load_all()?;

// Validar configuração
let validator = UniverseValidator::new(&loader);
validator.validate(&config, Some(&restrictions))?;

// Obter limites efetivos
let limits = validator.get_effective_limits(&config, Some(&restrictions));
```

## UI

A página `ConfigUniverse.tsx` permite:
- Selecionar os 4 eixos via interface visual
- Visualizar opções incompatíveis (grayed out)
- Ver resumo da configuração atual
- Salvar configuração

---

## Inventário de Parâmetros por Família

### Swing Trading

| Parâmetro | Tipo | Range | Validação | Uso |
|-----------|------|-------|-----------|-----|
| ma_periods.fast | int | 5-50, step 5 | Rust + TOML | Entry/Exit signals |
| ma_periods.slow | int | 20-200, step 10 | Rust + TOML | Trend filter |
| rsi.period | int | 7-21, step 1 | Rust | Oscillator |
| rsi.oversold | int | 20-35, step 5 | Rust | Entry threshold |
| rsi.overbought | int | 65-80, step 5 | Rust | Exit threshold |
| atr.period | int | 10-20, step 2 | Rust | Volatility |
| atr.stop_multiplier | float | 1.5-4.0, step 0.5 | Rust | Stop-loss |
| atr.target_multiplier | float | 2.0-5.0, step 0.5 | Rust | Take-profit |
| volume.ma_period | int | 10-30, step 5 | Rust | Volume filter |
| holding.period_min | int | 2-5 days | Rust | Position management |
| holding.period_max | int | 5-15 days | Rust | Position management |
| position_sizing.risk_per_trade_pct | float | 0.5-2.5% | Risk Profile | Sizing |
| validation.train_test_split | float | 0.60-0.75 | Training Strategy | Backtest |

### Momentum

| Parâmetro | Tipo | Range | Validação | Uso |
|-----------|------|-------|-----------|-----|
| momentum.lookback_days | int | 21-252, step 21 | Rust | Momentum calculation |
| momentum.skip_days | int | 0-21, step 5 | Rust | Skip recent days |
| cross_sectional.top_pct | int | 10-30, step 5 | Rust | Portfolio selection |
| trend_filter.ma_period | int | 100-252, step 50 | Rust | Trend confirmation |
| adx.period | int | 10-20, step 2 | Rust | Trend strength |
| adx.threshold | int | 20-35, step 5 | Rust | Entry filter |
| rebalancing.frequency | enum | weekly/monthly | Rust + UI | Execution |
| rebalancing.drift_threshold | float | 0.03-0.10 | Rust | Rebalance trigger |

### Position Trading

| Parâmetro | Tipo | Range | Validação | Uso |
|-----------|------|-------|-----------|-----|
| ma_periods.fast | int | 20-100, step 10 | Rust | Trend signal |
| ma_periods.slow | int | 100-300, step 50 | Rust | Long-term trend |
| donchian.entry_period | int | 20-55, step 5 | Rust | Breakout entry |
| donchian.exit_period | int | 10-30, step 5 | Rust | Exit signal |
| trailing_stop.activation_pct | float | 0.05-0.15 | Rust | Trailing activation |
| trailing_stop.atr_multiplier | float | 2.5-4.5 | Rust | Trailing distance |
| holding.period_min_weeks | int | 2-8, step 2 | Rust | Min hold |
| holding.period_max_weeks | int | 12-52, step 4 | Rust | Max hold |

---

## Lacunas Identificadas e Mitigações

| Lacuna | Impacto | Mitigação |
|--------|---------|-----------|
| Apenas 3 famílias com bounds completos | Outras famílias usam defaults | Criar bounds incrementalmente |
| UI não persiste no backend | Config não salva | Implementar endpoint PATCH |
| Validação apenas em Rust | Sem validação JS | Duplicar lógica no frontend |

---

## Critérios de Aceite

- [x] Configs existentes continuam funcionando
- [x] Seção [universe] é opcional
- [x] Risk profiles mantêm comportamento atual
- [x] UI exibe os 4 eixos
- [x] Opções incompatíveis são desabilitadas na UI
- [x] Rust compila sem erros
- [ ] Endpoint API implementado
- [ ] Testes de integração






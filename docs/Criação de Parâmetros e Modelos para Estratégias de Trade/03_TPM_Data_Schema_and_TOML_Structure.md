# Especificação Lógica 3: Schema de Dados e Estrutura TOML

**Autor**: Manus AI
**Versão**: 1.0
**Data**: 2026-01-05

## 1. Introdução

Esta especificação define a estrutura de dados e o schema exato para os arquivos de configuração de estratégia do **Módulo de Parâmetros de Trade (TPM)**. A escolha do formato **TOML (Tom's Obvious, Minimal Language)** se deve à sua legibilidade superior para humanos e facilidade de parsing por máquinas, tornando-o ideal para arquivos de configuração. [1]

Cada arquivo `.toml` representa uma única configuração de estratégia e serve como a fonte da verdade para o motor de backtesting e para a interface do usuário. A estrutura é projetada para ser modular, extensível e estritamente validada para garantir a integridade dos dados em todo o sistema.

## 2. Estrutura Geral do Arquivo TOML

Um arquivo de configuração de estratégia é composto por 12 seções principais (tabelas TOML). Cada seção agrupa um conjunto lógico de parâmetros. A presença de todas as seções é obrigatória para a validação, embora alguns campos dentro delas possam ser opcionais.

**Seções Principais:**

1.  `[metadata]`: Identificação e classificação da estratégia.
2.  `[timeframe]`: Parâmetros relacionados ao tempo (intervalo, janelas de dados).
3.  `[strategy]`: Lógica central da estratégia (tipo, direção).
4.  `[parameters]`: Parâmetros específicos e customizáveis da estratégia (períodos de MA, níveis de RSI, etc.).
5.  `[entry_rules]`: Condições para entrar em uma posição.
6.  `[exit_rules]`: Condições para sair de uma posição.
7.  `[position_sizing]`: Regras para calcular o tamanho da posição.
8.  `[risk_management]`: Controles de risco a nível de portfólio.
9.  `[execution]`: Parâmetros de execução (slippage, comissões).
10. `[validation]`: Configurações para o processo de backtesting e validação.
11. `[optimization]`: Configurações para o algoritmo genético.
12. `[universe]`: Regras para a seleção do universo de ativos.
13. `[notes]`: Metadados adicionais, como referências e notas do autor.

## 3. Detalhamento do Schema

A seguir, uma descrição detalhada de cada campo dentro das seções.

### 3.1. Seção `[metadata]`

Contém informações para identificar, classificar e filtrar a estratégia.

| Campo | Tipo | Obrigatório | Descrição |
| :--- | :--- | :--- | :--- |
| `strategy_id` | String | Sim | Identificador único em `snake_case`. Ex: `swing_momentum_ma_crossover`. |
| `name` | String | Sim | Nome legível para exibição na UI. Ex: "Swing Momentum - MA Crossover". |
| `description` | String | Sim | Descrição detalhada da lógica da estratégia. |
| `version` | String | Sim | Versão da configuração, seguindo o versionamento semântico (ex: "1.0.0"). |
| `risk_profile` | String | Sim | Perfil de risco. Opções: `conservative`, `moderate`, `aggressive`. |
| `family` | String | Sim | Família da estratégia (ver taxonomia). Ex: `swing`, `pair_trading`. |
| `asset_classes` | Array[String] | Sim | Classes de ativos aplicáveis. Ex: `["stocks", "etfs"]`. |
| `markets` | Array[String] | Sim | Mercados aplicáveis. Ex: `["BR", "US"]`. |
| `tags` | Array[String] | Não | Tags para busca. Ex: `["momentum", "crossover"]`. |

### 3.2. Seção `[timeframe]`

Define todos os parâmetros relacionados ao tempo.

| Campo | Tipo | Obrigatório | Descrição |
| :--- | :--- | :--- | :--- |
| `bar_interval` | String | Sim | Intervalo da barra. Opções: `1h`, `4h`, `1D`, `1W`. |
| `data_window_years` | Integer | Condicional | Janela de dados em anos (para estratégias diárias ou superiores). |
| `data_window_months`| Integer | Condicional | Janela de dados em meses (para estratégias intradiárias). |
| `lookback_bars` | Integer | Sim | Período de lookback para indicadores (em número de barras). |
| `holding_period_min`| Integer | Sim | Período mínimo de manutenção da posição (em barras). |
| `holding_period_max`| Integer | Sim | Período máximo de manutenção da posição (em barras). |

### 3.3. Seção `[parameters]`

Esta é a seção mais flexível, contendo os parâmetros que o algoritmo genético irá otimizar. Os campos aqui variam drasticamente entre as estratégias.

**Exemplo para uma estratégia de Cruzamento de Médias Móveis:**

```toml
[parameters]
ma_fast_period = 20
ma_slow_period = 50
ma_type = "SMA"
volume_multiplier = 1.2
atr_period = 14
```

**Exemplo para uma estratégia de Pair Trading:**

```toml
[parameters]
cointegration_lookback = 252
z_score_entry = 2.0
z_score_exit = 0.0
```

### 3.4. Seção `[entry_rules]` e `[exit_rules]`

Definem a lógica de entrada e saída usando uma sintaxe de expressão simples que será interpretada pelo motor de backtesting.

| Campo | Seção | Tipo | Obrigatório | Descrição |
| :--- | :--- | :--- | :--- | :--- |
| `long_condition` | `entry_rules` | String | Sim | Expressão para entrada comprada. Ex: `"SMA(20) > SMA(50)"`. |
| `short_condition`| `entry_rules` | String | Não | Expressão para entrada vendida. |
| `exit_methods` | `exit_rules` | Array[String] | Sim | Métodos de saída a serem usados. Ex: `["stop_loss", "time_based"]`. |
| `profit_target_type`| `exit_rules` | String | Não | Tipo de alvo de lucro. Opções: `fixed_pct`, `atr_multiple`. |
| `profit_target_value`| `exit_rules` | Float | Não | Valor do alvo de lucro. |
| `stop_loss_type` | `exit_rules` | String | Não | Tipo de stop loss. Opções: `fixed_pct`, `atr_multiple`. |
| `stop_loss_value`| `exit_rules` | Float | Não | Valor do stop loss. |
| `max_holding_bars` | `exit_rules` | Integer | Não | Saída por tempo (número de barras). |

### 3.5. Seções de Risco e Execução

-   **`[position_sizing]`**: Define como o tamanho de cada posição é calculado (ex: percentual fixo do capital, baseado no risco).
-   **`[risk_management]`**: Define limites de risco a nível de portfólio (ex: drawdown máximo, número máximo de posições abertas).
-   **`[execution]`**: Modela os custos de transação do mundo real (slippage e comissões).

### 3.6. Seções de Validação e Otimização

-   **`[validation]`**: Controla como o backtest é realizado, incluindo a divisão de dados de treino/teste e critérios de aceitação.
-   **`[optimization]`**: Fornece os parâmetros para o algoritmo genético, como o tamanho da população, número de gerações e a função de fitness.

## 4. Exemplo Completo

O arquivo `example_swing_momentum_ma_crossover_moderate.toml` (criado anteriormente) serve como uma implementação de referência completa deste schema. Ele demonstra como todas as seções e campos se unem para descrever uma estratégia de forma inequívoca.

## 5. Conclusão

Este schema TOML fornece uma base robusta e flexível para definir um vasto universo de estratégias de trading. A estrutura clara e a validação rigorosa garantirão a consistência dos dados e a confiabilidade do processo de geração de estratégias. O `TPM Loader` em Rust será responsável por impor este schema em tempo de execução.

A próxima especificação abordará o **Mapeamento de Timeframes e Janelas de Dados**, detalhando como as estratégias se conectam a diferentes requisitos de dados e janelas de tempo.

## Referências

[1] TOML. *Tom's Obvious, Minimal Language*. Disponível em: <https://toml.io/en/>. Acessado em: 05 de janeiro de 2026.

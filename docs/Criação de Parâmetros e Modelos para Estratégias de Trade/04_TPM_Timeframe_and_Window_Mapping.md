# Especificação Lógica 4: Mapeamento de Timeframes e Janelas de Dados

**Autor**: Manus AI
**Versão**: 1.0
**Data**: 2026-01-05

## 1. Introdução

Esta especificação detalha a lógica de mapeamento entre as famílias de estratégias do **Módulo de Parâmetros de Trade (TPM)** e seus respectivos requisitos de dados, janelas de tempo e configurações de validação. O objetivo é automatizar a configuração do ambiente de backtesting, garantindo que, ao selecionar um tipo de estratégia, o sistema utilize a quantidade de dados, os períodos de lookback e as divisões de treino/teste mais apropriadas para aquela metodologia.

Este mapeamento é crucial para a eficiência computacional e a robustez estatística dos resultados. Utilizar uma janela de dados de 10 anos para uma estratégia de intraday é computacionalmente caro e estatisticamente inadequado, pois o comportamento recente do mercado (microestrutura) é mais relevante. Inversamente, testar uma estratégia de investimento em fatores (Factor Investing) com apenas um ano de dados não capturaria diferentes regimes de mercado, levando a conclusões frágeis. [1]

## 2. Definições de Janelas de Tempo

O sistema opera com quatro categorias principais de janelas de tempo, cada uma associada a um conjunto de famílias de estratégias.

| Categoria | Janela de Dados | Período de Lookback | Período de Holding | Carga Computacional |
| :--- | :--- | :--- | :--- | :--- |
| **Intraday (1h)** | 6-12 meses | 20-100 horas | 1-8 horas | Alta |
| **Curto Prazo** | 1-3 anos | 20-60 dias | 2-10 dias | Média |
| **Médio Prazo** | 3-5 anos | 60-252 dias | 2-12 semanas | Média |
| **Longo Prazo** | 5-10+ anos | 252-1260 dias | 3+ meses | Baixa a Média |

## 3. Mapeamento: Família de Estratégia para Janela de Dados

Esta tabela define a janela de dados ótima e máxima recomendada para cada família de estratégia. O sistema usará a janela ótima por padrão, mas permitirá ao usuário estendê-la até o máximo.

| Família da Estratégia | Janela de Dados Ótima | Janela de Dados Máxima | Justificativa |
| :--- | :--- | :--- | :--- |
| **Intraday (1h)** | 6 meses | 1 ano | A microestrutura recente do mercado é mais preditiva. |
| **Swing Trading** | 2 anos | 5 anos | Captura múltiplos ciclos de curto prazo sem ser poluído por dados muito antigos. |
| **Position Trading** | 5 anos | 10 anos | Necessita de dados suficientes para identificar tendências de longo prazo e regimes de mercado. |
| **Pair Trading** | 3 anos | 5 anos | A cointegração entre pares pode não ser estável por períodos muito longos. |
| **Portfolio Trading** | 5 anos | 10 anos | A matriz de covariância/correlação necessita de um histórico substancial para ser estável. |
| **Momentum** | 3 anos | 5 anos | O fator momentum é persistente, mas seus parâmetros ótimos podem mudar ao longo do tempo. |
| **Mean Reversion** | 2 anos | 4 anos | O regime de volatilidade, crucial para a reversão à média, é mais influenciado por dados recentes. |
| **Breakout** | 3 anos | 5 anos | Requer a identificação de padrões de consolidação em diferentes condições de mercado. |
| **Sector Rotation** | 10 anos | 20 anos | Necessita de dados que cubram múltiplos ciclos econômicos completos. |
| **Factor Investing** | 10 anos | 20+ anos | A comprovação da eficácia dos fatores exige longos períodos de dados. |
| **Seasonal Trading** | 15 anos | 30+ anos | A significância estatística de padrões sazonais requer um grande número de ocorrências. |
| **Volatility Trading** | 3 anos | 10 anos | Focado em regimes de volatilidade, que são cíclicos. |
| **Event-Driven** | 5 anos | 10 anos | Depende da frequência de eventos (balanços, M&A) para ter uma amostra robusta. |
| **Buy and Hold** | 20+ anos | Todos os disponíveis | Análise de longo prazo, incluindo múltiplos crashes e bull markets. |

## 4. Mapeamento para Configuração de Validação (Train/Test Split)

A robustez de uma estratégia é verificada através da sua performance em dados fora da amostra (out-of-sample). A divisão dos dados para treino e teste será ajustada automaticamente com base na família da estratégia.

| Frequência da Estratégia | Divisão Treino/Teste | Walk-Forward Analysis (WFA) | Justificativa |
| :--- | :--- | :--- | :--- |
| **Alta (Intraday)** | 70% Treino / 30% Teste | Não recomendado | A dinâmica do mercado muda rapidamente; o foco é na performance recente. |
| **Média (Swing, Position)** | 65% Treino / 35% Teste | Recomendado (5-10 folds) | Bom equilíbrio entre otimização e validação, com WFA para evitar data snooping. |
| **Baixa (Factor, Seasonal)** | 60% Treino / 40% Teste | Altamente recomendado (10+ folds) | Requer validação out-of-sample mais rigorosa devido ao menor número de trades. |

### Lógica de Implementação

-   Quando o usuário selecionar uma estratégia no dashboard (ex: "Pair Trading Cointegration"), o TPM identificará sua família (`pair_trading`).
-   Com base na família, o sistema irá pré-selecionar a janela de dados ótima (3 anos) e a configuração de validação (65/35 com WFA).
-   Essas configurações serão exibidas na UI, mas o usuário avançado poderá modificá-las dentro dos limites definidos (ex: estender a janela de dados até 5 anos).
-   A seção `[validation]` no arquivo TOML da estratégia conterá esses padrões, que serão lidos pelo `TPM Loader`.

**Exemplo de seção `[validation]` para uma estratégia de Swing Trading:**

```toml
[validation]
train_test_split = 0.65
wfa_enabled = true
wfa_num_folds = 5
min_trades_total = 100
min_sharpe_ratio_oos = 0.8
```

## 5. Conclusão

O mapeamento automático de timeframes e janelas de dados é um pilar da usabilidade e da robustez do sistema. Ele abstrai decisões complexas de configuração do usuário, ao mesmo tempo que impõe boas práticas de backtesting, garantindo que cada estratégia seja avaliada em um contexto apropriado. Isso não apenas economiza tempo de processamento, mas também aumenta a confiança nos resultados gerados.

A próxima especificação abordará o **Sistema de Validação de Configurações**, detalhando como o `TPM Loader` garantirá a integridade e a lógica de cada arquivo de configuração.

## Referências

[1] De Prado, M. L. (2018). *Advances in Financial Machine Learning*. Wiley. (Discute a importância da seleção de dados e validação em finanças quantitativas).

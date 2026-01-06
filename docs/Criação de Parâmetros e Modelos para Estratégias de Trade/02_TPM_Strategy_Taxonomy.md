# Especificação Lógica 2: Taxonomia Completa de Estratégias

**Autor**: Manus AI
**Versão**: 1.0
**Data**: 2026-01-05

## 1. Introdução

Esta especificação define a taxonomia completa e o sistema de classificação para todas as estratégias de trading contidas no **Módulo de Parâmetros de Trade (TPM)**. O objetivo desta taxonomia é criar uma linguagem comum e uma estrutura lógica que permita ao sistema e ao usuário final compreender, selecionar e customizar estratégias de forma intuitiva e eficiente. A classificação organiza as 116 configurações pré-definidas em um sistema hierárquico baseado em múltiplos eixos, como horizonte de tempo, hipótese de mercado e estrutura da posição.

## 2. Eixos de Classificação

As estratégias são classificadas ao longo de três eixos principais para fornecer uma visão multidimensional de seu comportamento e aplicação.

### 2.1. Classificação por Horizonte de Tempo (Timeframe)

Este eixo define o período de tempo em que a estratégia opera, influenciando diretamente a frequência de negociação, os custos de transação e os requisitos de dados.

| Categoria | Horizonte de Tempo | Período de Holding | Requisitos de Dados |
| :--- | :--- | :--- | :--- |
| **Intraday** | Barras de 1 Hora | 1 a 8 horas | OHLCV Intradiário |
| **Curto Prazo** | Barras Diárias | 2 a 10 dias | OHLCV Diário |
| **Médio Prazo** | Barras Diárias | 2 a 12 semanas | OHLCV Diário, Fundamentos |
| **Longo Prazo** | Barras Diárias/Semanais | 3+ meses | OHLCV Diário, Fundamentos, Dados Econômicos |

### 2.2. Classificação por Hipótese de Mercado

Este eixo descreve a crença fundamental sobre o comportamento do mercado que a estratégia tenta explorar.

| Tipo | Hipótese Fundamental | Lógica de Operação | Melhores Timeframes |
| :--- | :--- | :--- | :--- |
| **Momentum** | Tendências estabelecidas tendem a persistir. | Seguir a direção da tendência predominante. | Todos |
| **Reversão à Média** | Preços tendem a retornar à sua média histórica. | Operar contra movimentos extremos de preço. | Curto a Médio Prazo |
| **Breakout** | Períodos de consolidação precedem movimentos direcionais. | Entrar no mercado quando o preço rompe um range definido. | Todos |
| **Arbitragem Estatística** | Relações estatísticas entre ativos são estáveis. | Explorar desvios temporários nessas relações. | Diário ou Superior |
| **Fundamentalista** | O preço de um ativo converge para seu valor intrínseco. | Comprar ativos subvalorizados e vender sobrevalorizados. | Longo Prazo |

### 2.3. Classificação por Estrutura da Posição

Este eixo detalha como a posição é construída, o que define seu perfil de risco e exposição ao mercado.

| Tipo | Estrutura da Posição | Perfil de Risco | Complexidade |
| :--- | :--- | :--- | :--- |
| **Direcional** | Aposta na direção de um único ativo (Long ou Short). | Exposição total ao risco de mercado. | Baixa |
| **Pair Trading** | Posição comprada e vendida em dois ativos correlacionados. | Risco de mercado neutralizado ou reduzido. | Média |
| **Portfólio** | Múltiplas posições simultâneas para diversificação. | Risco diversificado entre múltiplos ativos. | Alta |
| **Multi-Estratégia** | Combinação de diferentes abordagens em um único portfólio. | Adaptativo, dependente do regime de mercado. | Muito Alta |

## 3. Detalhamento das Famílias de Estratégias

A seguir, uma descrição detalhada de cada uma das 15 famílias de estratégias que compõem o catálogo do TPM.

### Famílias de Estratégias

| ID | Família | # Variações | Descrição Principal |
| :--- | :--- | :--- | :--- |
| 1 | **Intraday (1h)** | 7 | Estratégias executadas dentro de um único dia, usando barras de 1 hora. |
| 2 | **Swing Trading** | 3 | Captura movimentos de preço de curto a médio prazo, durando de 2 a 10 dias. |
| 3 | **Position Trading** | 2 | Mantém posições por semanas ou meses, baseando-se em tendências de longo prazo. |
| 4 | **Pair Trading** | 3 | Explora a diferença de preço entre dois ativos correlacionados. |
| 5 | **Portfolio Trading** | 4 | Gerencia um portfólio de múltiplos ativos com base em regras de alocação. |
| 6 | **Momentum** | 2 | Segue a tendência, comprando ativos que estão subindo e vendendo os que estão caindo. |
| 7 | **Mean Reversion** | 2 | Aposta que os preços retornarão à sua média histórica. |
| 8 | **Breakout** | 2 | Entra no mercado após o rompimento de níveis de suporte ou resistência. |
| 9 | **Sector Rotation** | 2 | Rotaciona o capital entre diferentes setores da economia. |
| 10 | **Factor Investing** | 4 | Investe em ativos com base em fatores quantificáveis (valor, qualidade, etc.). |
| 11 | **Seasonal Trading** | 2 | Explora padrões de mercado que se repetem em certas épocas do ano. |
| 12 | **Volatility Trading** | 2 | Opera com base nas mudanças da volatilidade do mercado. |
| 13 | **Event-Driven** | 2 | Reage a eventos corporativos específicos, como fusões e aquisições ou balanços. |
| 14 | **Buy and Hold** | 2 | Estratégia passiva de investimento de longo prazo. |
| 15 | **Multi-Strategy** | 1 | Combina dinamicamente múltiplas estratégias com base no regime de mercado. |

### Exemplo de Detalhamento: Família Swing Trading

-   **Swing Momentum**: Utiliza indicadores como cruzamento de médias móveis (ex: 20/50) ou MACD para identificar e seguir tendências de curto prazo. A entrada ocorre após a confirmação da tendência, e a saída pode ser um sinal oposto ou um stop móvel.
-   **Swing Mean Reversion**: Opera em mercados com tendência, mas busca comprar em recuos (dips) e vender em picos (rallies). Indicadores como Bandas de Bollinger e IFR (Índice de Força Relativa) são comumente usados para identificar pontos de entrada.
-   **Swing Breakout**: Foca em ativos que estão em um período de consolidação (range). A entrada ocorre quando o preço rompe esse range com volume, indicando o início de um novo movimento direcional.

## 4. Conclusão

Esta taxonomia serve como o esqueleto lógico para todo o Módulo de Parâmetros de Trade. Ela permite que o sistema organize, filtre e apresente as estratégias de uma maneira que seja imediatamente compreensível para o usuário. Além disso, fornece a estrutura necessária para que o algoritmo genético realize uma busca mais inteligente e direcionada.

A próxima especificação abordará o **Schema de Dados e a Estrutura TOML**, detalhando o formato exato de cada arquivo de configuração.

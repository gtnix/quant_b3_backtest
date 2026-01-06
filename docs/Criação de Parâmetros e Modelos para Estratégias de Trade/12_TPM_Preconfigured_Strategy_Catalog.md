# Especificação Lógica 12: Catálogo de Estratégias Pré-configuradas

**Autor**: Manus AI
**Versão**: 1.0
**Data**: 2026-01-05

## 1. Introdução

Esta especificação serve como um anexo e um índice para o coração do **Módulo de Parâmetros de Trade (TPM)**: o catálogo inicial de 116 modelos de estratégias pré-configuradas. Cada um desses modelos será um arquivo `.toml` individual, meticulosamente criado para representar uma metodologia de trading reconhecida, com parâmetros e ranges de otimização sensatos.

O objetivo deste documento não é detalhar cada um dos 116 arquivos TOML, o que seria impraticável, mas sim fornecer uma visão geral organizada do catálogo. Ele agrupa as estratégias por família e descreve a lógica principal e as variações de cada subgrupo. Este catálogo é o resultado da pesquisa e da fase de geração de configurações, e serve como a base de conhecimento que alimenta o gerador de estratégias.

## 2. Estrutura do Catálogo

O catálogo está organizado em 15 famílias principais, conforme definido na especificação de taxonomia. Dentro de cada família, existem subgrupos que representam diferentes implementações da mesma ideia central. Cada subgrupo, por sua vez, tem variações baseadas no perfil de risco (`Conservador`, `Moderado`, `Agressivo`), que ajustam parâmetros como tamanho da posição, alavancagem e limites de stop-loss.

**Exemplo da Hierarquia:**

-   **Família**: `Swing Trading`
    -   **Subgrupo**: `Cruzamento de Médias Móveis`
        -   **Variação**: `Conservador` (ex: usa SMA, sem alavancagem)
        -   **Variação**: `Moderado` (ex: usa EMA, pouca alavancagem)
        -   **Variação**: `Agressivo` (ex: usa DEMA, maior alavancagem)

## 3. Resumo das Famílias e Subgrupos

A tabela a seguir resume os principais subgrupos dentro de cada uma das 15 famílias de estratégias.

| Família | # | Subgrupo Principal | Lógica Resumida |
| :--- | :- | :--- | :--- |
| **Intraday (1h)** | 21 | Opening Range Breakout (ORB) | Opera o rompimento da máxima ou mínima da primeira hora de negociação. |
| | | VWAP Mean Reversion | Compra abaixo da VWAP e vende acima, esperando o retorno ao preço médio ponderado por volume. |
| | | Momentum com RSI | Entra em tendências fortes de 1h confirmadas por um RSI acima de 50 (compra) ou abaixo (venda). |
| **Swing Trading** | 12 | Cruzamento de Médias Móveis | Usa o cruzamento de duas MAs (ex: 20/50) em gráficos diários para sinalizar a tendência. |
| | | Reversão com Bandas de Bollinger | Compra quando o preço toca a banda inferior e vende quando toca a superior. |
| | | Breakout de Canais (Donchian) | Entra no rompimento de canais de preço das últimas N barras. |
| **Position Trading**| 6 | Seguidor de Tendência de Longo Prazo | Usa cruzamentos de MAs longas (ex: 50/200) para manter posições por meses. |
| | | Compra e Venda com MACD Semanal | Usa o sinal do MACD em gráficos semanais para posições de longo prazo. |
| **Pair Trading** | 12 | Cointegração (ADF) | Testa a cointegração entre dois ativos e opera o desvio do spread. |
| | | Distância da Média Móvel | Opera o spread entre os preços de dois ativos e suas respectivas médias. |
| | | Correlação de Retornos | Opera pares com alta correlação histórica quando seus retornos divergem. |
| **Portfolio** | 12 | Alocação Tática de Ativos | Rotaciona o capital entre diferentes classes de ativos (ações, renda fixa) com base no momentum. |
| | | Carteira de Mínima Variância | Constrói um portfólio de ações otimizado para ter a menor volatilidade possível. |
| | | Risk Parity | Aloca o capital de forma que cada ativo contribua igualmente para o risco total do portfólio. |
| **Momentum** | 8 | Dual Momentum | Combina momentum absoluto (vs. caixa) e relativo (vs. outros ativos). |
| | | Cross-Sectional Momentum | Compra o decil de ativos com melhor performance e vende o decil com pior performance. |
| **Mean Reversion**| 8 | RSI(2) de Connors | Estratégia de reversão de curto prazo usando um IFR de 2 períodos. |
| | | Statistical Arbitrage (STATARB) | Modela o preço de uma ação com base em um basket de outras e opera os resíduos. |
| **Breakout** | 6 | Rompimento de Triângulos/Flâmulas | Identifica padrões gráficos de consolidação e opera o rompimento. |
| | | Volatility Breakout (ATR) | Entra no mercado quando o preço se move mais do que um múltiplo do ATR. |
| **Sector Rotation**| 4 | Rotação Setorial com Momentum | Investe nos 3 setores da economia com o maior momentum dos últimos 6 meses. |
| **Factor Investing**| 8 | Fator Valor (Value) | Compra ações com baixos múltiplos (P/L, P/VP). |
| | | Fator Qualidade (Quality) | Compra ações com alta rentabilidade e baixa alavancagem. |
| **Seasonal** | 4 | Efeito Janeiro | Compra ações de baixa capitalização no final de dezembro e vende no final de janeiro. |
| | | "Sell in May and Go Away" | Fica fora do mercado de ações entre os meses de maio e outubro. |
| **Volatility** | 4 | Venda de Volatilidade (Short Strangle) | Vende opções de compra e venda fora do dinheiro, apostando em baixa volatilidade. |
| | | Compra de Volatilidade (Long Straddle) | Compra opções de compra e venda, apostando em um grande movimento de preço. |
| **Event-Driven** | 4 | Arbitragem de Fusões | Compra a ação da empresa-alvo e vende a da empresa adquirente em uma fusão. |
| | | Post-Earnings Announcement Drift | Opera na direção da surpresa do resultado trimestral de uma empresa. |
| **Buy and Hold** | 4 | Indexação Passiva (Ibovespa) | Simplesmente compra e mantém um ETF que replica o índice principal. |

## 4. Implementação e Disponibilidade

Cada uma das 116 estratégias resultantes da combinação dos subgrupos acima com os perfis de risco será implementada como um arquivo `.toml` no diretório `/configs/strategies/` do projeto `quant_b3_backtest`. A nomenclatura dos arquivos seguirá um padrão claro para fácil identificação, por exemplo:

-   `intraday_orb_aggressive.toml`
-   `swing_ma_crossover_moderate.toml`
-   `pair_cointegration_conservative.toml`

O `TPM Loader` será responsável por carregar, validar e servir esses arquivos para o resto do sistema.

## 5. Conclusão

Este catálogo é o ativo mais valioso do Módulo de Parâmetros de Trade. Ele encapsula uma vasta quantidade de conhecimento sobre o mercado financeiro em um formato estruturado e consumível por máquina. Ao fornecer um ponto de partida tão rico e diversificado, o sistema capacita o usuário a gerar estratégias sofisticadas com um esforço mínimo, cumprindo a promessa central do projeto de democratizar a pesquisa quantitativa.

A próxima especificação abordará as **APIs e Interfaces de Integração**, detalhando os contratos técnicos entre o frontend e o backend.

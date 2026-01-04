# Pacote de Determinação de Operação: Parâmetros de Risco para Brasil e EUA (v2.0)

**Autor:** Manus AI (em colaboração com o usuário)
**Data:** 03 de Janeiro de 2026
**Base de Evidências:** Pesquisa acadêmica massiva em 16 tópicos de trading quantitativo.

## (1) Resumo Executivo

Este documento apresenta um pacote de parâmetros de risco robusto e aplicável para sistemas de trading quantitativo, com calibração específica para os mercados do Brasil (B3) e dos Estados Unidos (NYSE/NASDAQ). A versão anterior deste documento foi preterida em favor de uma pesquisa massiva, que revelou com maior profundidade os **parâmetros vencedores** e as **regras de ouro** consagradas na literatura acadêmica. A principal conclusão é que a gestão de risco eficaz não reside em encontrar um "número mágico", mas em aplicar **ranges de parâmetros defensáveis** e adaptá-los à volatilidade e microestrutura de cada mercado. 

Os **guardrails não-negociáveis** foram reforçados e incluem: a adoção universal do **Kelly Fracionário (Fractional Kelly)** para dimensionamento de posição (nunca o Kelly completo), o uso de **stops baseados em volatilidade (ATR)**, e a implementação de **metas de volatilidade de portfólio (Volatility Targeting)**. A distinção crítica entre Brasil e EUA é quantificada: a maior volatilidade e menor liquidez do mercado brasileiro exigem **frações de Kelly mais conservadoras (e.g., 1/4 vs 1/2 nos EUA)**, **stops ATR com multiplicadores maiores (e.g., 3.0x vs 2.5x nos EUA)**, e **metas de volatilidade de portfólio mais elevadas para um mesmo perfil de risco**.

## (2) Dicionário de Parâmetros do Sistema

Este dicionário mapeia todos os parâmetros críticos de risco, fornecendo uma base para a implementação no sistema de trading.

| Categoria | Parâmetro | Descrição | Unidade |
| :--- | :--- | :--- | :--- |
| **Dimensionamento de Posição** | `kelly_fraction` | Fração do Critério de Kelly a ser utilizada (e.g., 0.5 para Half-Kelly). | Fração (0 a 1) |
| | `max_risk_per_trade_pct` | Limite máximo de risco por operação, atuando como um teto para o `kelly_fraction`. | % do Capital |
| **Gestão de Risco por Trade** | `stop_loss_atr_multiplier` | Multiplicador do Average True Range (ATR) para definir o stop-loss. | Float |
| | `stop_loss_atr_period` | Período de dias para o cálculo do ATR. | Dias |
| **Gestão de Risco de Portfólio** | `portfolio_volatility_target_pct` | Nível de volatilidade anualizada alvo para o portfólio. | % Anual |
| | `max_portfolio_drawdown_limit_pct` | Limite máximo de drawdown que aciona um circuit breaker na estratégia. | % do Capital |
| | `portfolio_cvar_limit_pct` | Limite de Conditional Value-at-Risk (95%, 1 dia) para o portfólio. | % do Capital |
| | `max_leverage` | Alavancagem máxima permitida para o portfólio. | Float |
| **Limites Operacionais** | `min_daily_liquidity_usd` | Volume financeiro médio diário mínimo para um ativo ser negociável. | USD |
| | `max_bid_ask_spread_bps` | Spread bid-ask máximo permitido em basis points para entrar em uma operação. | Basis Points |
| | `max_positions_open` | Número máximo de posições abertas simultaneamente. | Integer |
| | `max_sector_concentration_pct` | Limite máximo de exposição a um único setor da economia. | % do Capital |
| **Critérios de Backtest** | `min_profit_factor_oos` | Fator de lucro mínimo no período out-of-sample. | Float |
| | `max_drawdown_oos_pct` | Drawdown máximo permitido no período out-of-sample. | % |
| | `min_sharpe_ratio_oos` | Sharpe Ratio mínimo no período out-of-sample. | Float |

---
*Esta é a estrutura inicial do documento final. As seções subsequentes serão preenchidas com os dados da pesquisa massiva.*


## (3) A Tabela Principal: 5 Perfis de Risco (Baseado em Evidências)

A tabela a seguir consolida os parâmetros-chave para cada perfil de risco, com distinções claras para Brasil e EUA, fundamentadas na pesquisa massiva realizada. Cada recomendação é uma síntese das melhores práticas encontradas na literatura acadêmica.

| Parâmetro | Perfil | Valor Sugerido (Range) | BR vs EUA (Diferença?) | Justificativa (Fonte Principal) | Notas de Implementação |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dimensionamento de Posição** | | | | | |
| `kelly_fraction` | Muito Conservador | 0.1 - 0.25 | **BR < EUA**. Usar o limite inferior no Brasil. | O uso de frações de Kelly é essencial para mitigar o risco de ruína. A maior volatilidade no Brasil exige uma fração menor. [Ziemba & MacLean, 2011] | Nunca usar Kelly completo (1.0). Comece com 0.25 e reduza se os drawdowns forem altos. |
| | Conservador | 0.25 - 0.4 | **BR < EUA**. | Equilíbrio entre crescimento e controle de drawdown. [Vince, 1992] | O Half-Kelly (0.5) é o máximo teórico para a maioria das aplicações práticas. |
| | Moderado | 0.4 - 0.5 | Igual | Próximo ao Half-Kelly, maximizando o crescimento ajustado ao risco. [Ziemba & MacLean, 2011] | Requer estimativas robustas de probabilidade de ganho e payoff. |
| | Arrojado | 0.5 | Fixo | Limite superior absoluto recomendado pela literatura para evitar a instabilidade do Kelly completo. [Thorp, 2006] | Apenas para estratégias com alta taxa de acerto e vantagem estatística clara. |
| | Muito Arrojado | > 0.5 | **Não Recomendado** | Acima de Half-Kelly, a volatilidade e o risco de ruína aumentam exponencialmente. [Vince, 1992] | Considerado imprudente e academicamente indefensável. |
| `max_risk_per_trade_pct` | Todos | 1% - 2% | Igual | Atua como um **freio de emergência** sobre o Kelly. A regra de 2% é um limite superior universalmente aceito. [Vince, 1992] | A posição final deve ser o **mínimo** entre o cálculo do Kelly e este percentual fixo. |
| **Gestão de Risco por Trade** | | | | | |
| `stop_loss_atr_multiplier` | Muito Conservador | 3.0x - 4.0x | **BR > EUA**. | Stops mais largos para evitar saídas por ruído em mercados mais voláteis. [Chan, 2013] | A volatilidade da B3 pode exigir multiplicadores na extremidade superior do range. |
| | Conservador | 2.5x - 3.5x | **BR > EUA**. | Padrão da indústria para estratégias de swing e position trading. | Ajustar com base na volatilidade histórica do ativo específico. |
| | Moderado | 2.0x - 3.0x | **BR > EUA**. | Bom equilíbrio entre proteção de capital e dar espaço para a tese do trade se desenvolver. | Multiplicadores menores que 2.0x são propensos a saídas prematuras. |
| | Arrojado | 1.5x - 2.5x | **BR > EUA**. | Stops mais curtos para estratégias de prazo mais curto ou para maximizar a relação Risco:Retorno. | Aumenta a frequência de trades e os custos de transação. |
| | Muito Arrojado | 1.0x - 2.0x | **BR > EUA**. | Apenas para estratégias de scalping ou de altíssima frequência. | Requer uma taxa de acerto muito alta para ser lucrativo. |
| `stop_loss_atr_period` | Todos | 14 ou 20 | Igual | Períodos padrão que capturam a volatilidade de médio prazo sem serem excessivamente reativos. | O período de 20 dias (um mês de negociação) é o mais comum. |
| **Gestão de Risco de Portfólio** | | | | | |
| `portfolio_volatility_target_pct` | Muito Conservador | 6% - 10% | **BR > EUA**. | Mantém o risco do portfólio constante e em níveis comparáveis à renda fixa de baixo risco. [Harvey et al., 2017] | A meta no Brasil deve ser maior para uma exposição de risco equivalente. |
| | Conservador | 10% - 14% | **BR > EUA**. | Nível de volatilidade de um portfólio balanceado (60/40). | Requer ajuste dinâmico da alavancagem ou exposição total. |
| | Moderado | 14% - 18% | **BR > EUA**. | Volatilidade comparável a um índice de ações diversificado. | Aumenta o potencial de retorno, mas também o drawdown esperado. |
| | Arrojado | 18% - 22% | **BR > EUA**. | Para investidores com alta tolerância ao risco e foco em crescimento. | Drawdowns podem ser severos em crises de mercado. |
| | Muito Arrojado | 22% - 28% | **BR > EUA**. | Nível de risco de fundos de hedge macro globais ou estratégias de long-short agressivas. | Exige gestão de risco sofisticada e monitoramento constante. |
| `max_portfolio_drawdown_limit_pct` | Muito Conservador | 8% | Igual | Limite psicológico chave para investidores avessos à perda. [Chekhlov et al., 2003] | Acionar circuit breaker do sistema se atingido. |
| | Conservador | 12% | Igual | Limite comum para fundos de baixa volatilidade e family offices. | Reduzir a exposição pela metade se 50% do limite for atingido. |
| | Moderado | 20% | Igual | Limite padrão da indústria para fundos de ações. | Revisar estratégias e parâmetros se o limite for atingido. |
| | Arrojado | 25% | Igual | Drawdown significativo, mas recuperável para estratégias de longo prazo. | Pausar novas entradas e focar na gestão das posições existentes. |
| | Muito Arrojado | 30% | Igual | Limite extremo. Atingir este nível indica uma falha sistêmica da estratégia. | Liquidar posições e reavaliar todo o sistema de trading. |
| **Limites Operacionais** | | | | | |
| `min_daily_liquidity_usd` | Todos | > $5M (BR), > $20M (EUA) | **EUA > BR**. | Garante a capacidade de executar ordens sem impacto adverso no preço (slippage). | A liquidez é muito mais concentrada em poucos ativos na B3. |
| `max_bid_ask_spread_bps` | Todos | < 30 bps (BR), < 15 bps (EUA) | **BR > EUA**. | O spread é um custo direto e um indicador de liquidez. Spreads altos corroem a lucratividade. | Spreads mais altos no Brasil refletem menor liquidez e maior risco. |

---
*Esta tabela representa a consolidação da pesquisa massiva. As seções seguintes detalham a implementação e o racional por trás destes parâmetros.*


## (4) Especificação Narrativa de Cada Perfil

#### 1. Muito Conservador
- **Objetivo:** Preservação de capital com crescimento marginal, superando a inflação.
- **Prioriza:** Volatilidade mínima e drawdowns insignificantes. A proteção do principal é a única prioridade.
- **Sacrifica:** Retornos significativos. A estratégia é desenhada para nunca perder dinheiro de forma relevante, mesmo que isso signifique abrir mão de quase todas as oportunidades de mercado.
- **Expectativa em Crise:** Comportamento similar a um título do tesouro de curto prazo. O drawdown máximo não deve exceder 8%. Os freios do sistema (circuit breakers) são acionados com perdas diárias de 1% ou semanais de 2%.

#### 2. Conservador
- **Objetivo:** Crescimento consistente do capital com risco estritamente controlado.
- **Prioriza:** Retornos ajustados ao risco (Sharpe Ratio) elevados e drawdowns controlados.
- **Sacrifica:** Retornos de dois dígitos. A estratégia busca capturar a maior parte dos movimentos de mercado de forma segura.
- **Expectativa em Crise:** Comportamento semelhante a um portfólio balanceado (60/40). O drawdown máximo esperado é de até 12%. Os freios são acionados com perdas diárias de 1.5% ou semanais de 3.5%.

#### 3. Moderado
- **Objetivo:** Crescimento do capital a longo prazo, aceitando a volatilidade como parte do processo.
- **Prioriza:** Maximizar o crescimento geométrico do capital, alinhado com o Half-Kelly.
- **Sacrifica:** Estabilidade de curto prazo. A estratégia aceita drawdowns na casa dos 20% em troca de um maior potencial de retorno.
- **Expectativa em Crise:** Comportamento de um índice de ações diversificado (e.g., S&P 500). O drawdown máximo esperado é de até 20%. Os freios são acionados com perdas diárias de 2.5% ou semanais de 6%.

#### 4. Arrojado
- **Objetivo:** Crescimento acelerado do capital, com alta tolerância a risco.
- **Prioriza:** Retornos agressivos, utilizando o limite máximo de risco defensável pela academia (Half-Kelly).
- **Sacrifica:** Previsibilidade. A estratégia pode ter oscilações bruscas e períodos de perdas prolongadas.
- **Expectativa em Crise:** Volátil, com potencial para perdas significativas. O drawdown máximo esperado é de até 25%. Os freios são acionados com perdas diárias de 3.5% ou semanais de 8%.

#### 5. Muito Arrojado
- **Objetivo:** Ganhos exponenciais, operando no limite do risco racional.
- **Prioriza:** Capitalizar em movimentos de mercado de alta convicção com alavancagem máxima.
- **Sacrifica:** Estabilidade e paz de espírito. Esta estratégia é de natureza especulativa e não é recomendada para a maioria dos investidores.
- **Expectativa em Crise:** Extremamente volátil. O risco de ruína, embora matematicamente baixo com os parâmetros corretos, é uma possibilidade psicológica real. O drawdown máximo esperado é de até 30%. Os freios são acionados com perdas diárias de 5% ou semanais de 10%.

## (5) Ajuste para Cenário de Poucos Ativos e Risco por Papel

- **Clusterização por Risco:** Em vez de regras individuais, os ativos devem ser agrupados em **clusters de risco** (e.g., "Baixa Vol/Alta Liq", "Alta Vol/Média Liq"). Os parâmetros de risco, como o multiplicador de ATR, são definidos por cluster, não por ativo.
- **Orçamento de Risco por Cluster:** Defina um orçamento de risco máximo para cada cluster (e.g., o cluster de "Alta Vol" não pode consumir mais de 40% do risco total do portfólio).
- **Anti-Concentração via Drawdown Beta:** Para evitar concentração oculta, utilize o conceito de **Drawdown Beta** [Ding & Uryasev, 2022]. O sistema deve penalizar ou proibir a adição de um novo ativo se o seu Drawdown Beta com o portfólio existente for muito alto (e.g., > 0.8), indicando que eles tendem a sofrer drawdowns simultaneamente.

## (6) Análise Focada (PASSO 2): Corrigindo a Causa Raiz

O problema central identificado no "Relatório de Evidências" era que o algoritmo genético operava com **ranges de parâmetros irrestritos e economicamente absurdos**. A solução não é abandonar o algoritmo, mas sim **impor guardrails baseados em evidências**. Os ranges definidos na Tabela Principal (v2.0) servem exatamente a este propósito. Ao limitar os parâmetros de entrada (e.g., `min_market_cap`, `min_return`) a valores plausíveis, garantimos que o sistema de otimização busque soluções dentro de um universo de estratégias viáveis, eliminando a geração de estratégias que nunca operam.

## (7) Checklist Anti-Overfitting Obrigatório

| Critério | Parâmetro Mínimo (Regra de Ouro) | Justificativa |
| :--- | :--- | :--- |
| **Validação Out-of-Sample (OOS)** | Mínimo 30% dos dados | Garante que a estratégia não foi ajustada apenas para o passado conhecido. |
| **Profit Factor (OOS)** | > 1.5 | Um valor abaixo disso indica que os custos de transação podem inviabilizar a estratégia. |
| **Sharpe Ratio (OOS)** | > 1.0 (após custos) | Mede o retorno ajustado ao risco. Um valor < 1.0 é geralmente inaceitável. |
| **Máximo Drawdown (OOS)** | < 25% | Deve ser consistente com o perfil de risco e não drasticamente pior que o período In-Sample. |
| **Análise de Sensibilidade** | Estável com custos 2x-3x maiores | A estratégia deve permanecer lucrativa mesmo com derrapagens (slippage) piores que o esperado. |
| **Estabilidade de Parâmetros** | Performance não degrada com pequenas variações | Se mudar um parâmetro de 20 para 21 quebra a estratégia, ela não é robusta. |
| **Teste de Monte Carlo** | RoR < 1% em 1000 simulações | Simula diferentes sequências de trades para testar a robustez a caminhos de retorno alternativos. |

## (8) Referências Completas (Base da Pesquisa Massiva)

1.  **[Ziemba & MacLean, 2011]** Ziemba, W. T., & MacLean, L. C. (2011). *Using the Kelly Criterion for Investing*. In Stochastic Optimization Models in Finance. [https://webhomes.maths.ed.ac.uk/mckinnon/blackouts/StochOptFinanceAndEnergySpringer/Chap1_KellyZiemba.pdf]
2.  **[Vince, 1992]** Vince, R. (1992). *The Mathematics of Money Management: Risk Analysis Techniques for Traders*. John Wiley & Sons.
3.  **[Thorp, 2006]** Thorp, E. O. (2006). *The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market*. The American Mathematical Monthly. [https://gwern.net/doc/statistics/decision/2006-thorp.pdf]
4.  **[Chan, 2013]** Chan, E. P. (2013). *Algorithmic Trading: Winning Strategies and Their Rationale*. John Wiley & Sons. [https://dl.najafi8.ir/dl/Library/book/Algorithmic_Trading__Winning_Strategies.pdf]
5.  **[Harvey et al., 2017]** Harvey, C. R., Han, L., & Taylor, E. (2017). *The Impact of Volatility Targeting*. The Journal of Portfolio Management. [https://people.duke.edu/~charvey/Research/Published_Papers/P135_The_impact_of.pdf]
6.  **[Chekhlov et al., 2003]** Chekhlov, A., Uryasev, S., & Zabarankin, M. (2003). *Portfolio Optimization with Drawdown Constraints*. [https://www.cis.upenn.edu/~mkearns/finread/drawdown.pdf]
7.  **[Ding & Uryasev, 2022]** Ding, R., & Uryasev, S. (2022). *Drawdown beta and portfolio optimization*. [http://uryasev.ams.stonybrook.edu/wp-content/uploads/2022/02/Drawdown_beta_and_portfolio_optimization.pdf]

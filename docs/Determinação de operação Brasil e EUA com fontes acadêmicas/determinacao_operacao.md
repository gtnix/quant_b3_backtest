# Determinação de Operação: Parâmetros de Risco para Brasil e EUA

**Autor:** Manus AI
**Data:** 03 de Janeiro de 2026

## (1) Resumo Executivo

Esta determinação de operação estabelece um novo pacote de parâmetros de risco para o sistema de trading, calibrado para operar simultaneamente nos mercados do Brasil (B3) e dos Estados Unidos (NYSE/NASDAQ). A análise do sistema atual, conforme o "Relatório de Evidências", revelou que os parâmetros eram gerados com ranges excessivamente amplos e desconectados da realidade de mercado, resultando em 100% das estratégias sem executar operações (`total_trades = 0`) e, consequentemente, métricas de performance artificiais. 

A nova parametrização corrige esta falha ao introduzir **cinco perfis de risco distintos** (de Muito Conservador a Muito Arrojado), com ranges de parâmetros baseados em evidências acadêmicas e na microestrutura de cada mercado. Os **guardrails não-negociáveis** incluem a implementação de limites de liquidez mínima, a adoção de métricas de risco coerentes como o CVaR (Conditional Value-at-Risk), e a aplicação rigorosa de um checklist anti-overfitting. A principal diferença entre Brasil e EUA reside na **volatilidade estruturalmente mais alta e menor liquidez em ativos de menor capitalização no Brasil**, exigindo stops mais largos e metas de volatilidade de portfólio distintas para perfis de risco equivalentes.

## (2) Dicionário de Parâmetros do Sistema

A seguir, um mapa completo dos parâmetros de sistema relevantes, agrupados por categoria funcional para clareza e implementação.

| Categoria | Parâmetro | Descrição | Unidade |
| :--- | :--- | :--- | :--- |
| **Risco por Trade** | `risk_per_trade_pct` | Percentual máximo do capital total a ser arriscado em uma única operação. | % do Capital |
| | `stop_loss_type` | Método para cálculo do stop-loss (ATR, volatilidade, percentual fixo). | Enum |
| | `stop_loss_value` | Multiplicador ou valor para o método de stop (ex: 2.5x ATR). | Float |
| | `position_sizing_model` | Modelo de dimensionamento da posição (ex: risco fixo, fração de Kelly). | Enum |
| **Risco por Ativo** | `max_exposure_per_asset_pct` | Exposição máxima do capital total a um único ativo. | % do Capital |
| | `concentration_limit_sector` | Limite máximo de exposição a um único setor da economia. | % do Capital |
| **Risco de Portfólio** | `portfolio_volatility_target` | Nível de volatilidade anualizada alvo para o portfólio. | % Anual |
| | `max_portfolio_drawdown_limit` | Limite máximo de drawdown para o portfólio total. | % do Capital |
| | `max_leverage` | Alavancagem máxima permitida para o portfólio. | Float |
| | `portfolio_cvar_limit_pct` | Limite de Conditional Value-at-Risk (95%, 1 dia) para o portfólio. | % do Capital |
| **Limites Operacionais** | `min_liquidity_usd` | Volume financeiro médio diário mínimo para um ativo ser negociável. | USD |
| | `max_spread_bps` | Spread bid-ask máximo permitido em basis points. | Basis Points |
| | `slippage_cost_bps` | Custo estimado de slippage por operação para cálculo de backtest. | Basis Points |
| | `max_positions_open` | Número máximo de posições abertas simultaneamente. | Integer |
| **Regras de Pausa** | `daily_loss_limit_pct` | Limite de perda diária que aciona a suspensão das operações. | % do Capital |
| | `weekly_loss_limit_pct` | Limite de perda semanal que aciona a suspensão das operações. | % do Capital |
| **Filtros de Universo** | `min_market_cap_usd` | Capitalização de mercado mínima para um ativo entrar no universo. | USD |
| | `max_annualized_vol` | Volatilidade anualizada máxima para um ativo ser considerado. | % Anual |

---
*Esta é a estrutura inicial do documento. As seções subsequentes, incluindo a Tabela Principal com os 5 perfis de risco, serão preenchidas na próxima etapa.*


## (3) Tabela Principal: 5 Perfis de Risco

A tabela a seguir detalha os parâmetros para cada um dos cinco perfis de risco, com ajustes específicos para os mercados do Brasil e dos EUA, justificados por fontes acadêmicas e de mercado.

| Parâmetro | Perfil | Valor Sugerido (Range) | BR vs EUA (Diferença?) | Justificativa (Fonte + Link) | Notas de Implementação |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Risco por Trade** | | | | | |
| `risk_per_trade_pct` | Muito Conservador | 0.25% - 0.5% | Igual | Limita o Risco de Ruína (RoR) a níveis negligenciáveis. [3] | Parâmetro mais crítico para a sobrevivência de longo prazo. |
| | Conservador | 0.5% - 1.0% | Igual | Padrão da indústria para gestão de risco prudente. [5] | Bom equilíbrio entre crescimento e preservação de capital. |
| | Moderado | 1.0% - 1.5% | Igual | Aumenta o potencial de retorno, mas com RoR ainda baixo. [1] | Requer uma taxa de acerto e payoff razoáveis para ser eficaz. |
| | Arrojado | 1.5% - 2.0% | Igual | Próximo ao limite superior recomendado para traders de varejo. [5] | Drawdowns podem ser significativos. Requer alta confiança na estratégia. |
| | Muito Arrojado | 2.0% - 2.5% | Igual | Agressivo. Próximo a níveis que podem levar a RoR significativo. [3] | Usar com extrema cautela e apenas para estratégias de alta convicção. |
| `stop_loss_type` | Todos | ATR | Igual | Adapta-se dinamicamente à volatilidade do ativo. | Usar um período de 14 ou 20 dias para o cálculo do ATR. |
| `stop_loss_value` | Muito Conservador | 3.0x - 4.0x ATR | **BR > EUA**. Usar o limite superior no Brasil. | A maior volatilidade da B3 exige stops mais largos para evitar saídas por ruído. | No Brasil, a volatilidade pode ser 1.5x a 2x maior que nos EUA. |
| | Conservador | 2.5x - 3.5x ATR | **BR > EUA**. | Mesma justificativa acima. | Ajustar multiplicador com base na volatilidade histórica do ativo. |
| | Moderado | 2.0x - 3.0x ATR | **BR > EUA**. | Balanço entre proteção e permitir que o trade se desenvolva. | Monitorar a eficácia do stop em diferentes regimes de mercado. |
| | Arrojado | 1.5x - 2.5x ATR | **BR > EUA**. | Stops mais curtos para maximizar o R:R (Risco:Retorno). | Aumenta o risco de ser "stopado" por movimentos de curto prazo. |
| | Muito Arrojado | 1.0x - 2.0x ATR | **BR > EUA**. | Para estratégias de scalping ou de curtíssimo prazo. | Requer alta precisão de entrada e pode ter baixa taxa de acerto. |
| **Risco de Portfólio** | | | | | |
| `portfolio_volatility_target` | Muito Conservador | 5% - 8% | **BR > EUA**. Usar o limite superior no Brasil. | Mantém o risco do portfólio constante e baixo. [3] | A volatilidade alvo no Brasil deve ser maior para uma exposição equivalente. |
| | Conservador | 8% - 12% | **BR > EUA**. | Nível de volatilidade semelhante a um portfólio 60/40. | Requer ajuste dinâmico da alavancagem ou exposição. |
| | Moderado | 12% - 16% | **BR > EUA**. | Volatilidade de um portfólio de ações diversificado. | Aumenta o potencial de retorno, mas também o drawdown esperado. |
| | Arrojado | 16% - 20% | **BR > EUA**. | Para investidores com alta tolerância ao risco. | Drawdowns podem ser severos em crises de mercado. |
| | Muito Arrojado | 20% - 25% | **BR > EUA**. | Nível de risco de um fundo de hedge macro global. | Exige gestão de risco sofisticada e monitoramento constante. |
| `max_portfolio_drawdown_limit` | Muito Conservador | 5% | Igual | Limite psicológico chave para investidores conservadores. | Acionar circuit breaker do sistema se atingido. |
| | Conservador | 10% | Igual | Limite comum para fundos de baixa volatilidade. | Reduzir a exposição pela metade se 50% do limite for atingido. |
| | Moderado | 15% | Igual | Limite aceitável para a maioria dos investidores de longo prazo. | Revisar estratégias e parâmetros se o limite for atingido. |
| | Arrojado | 20% | Igual | Drawdown significativo, mas recuperável para estratégias de longo prazo. | Pausar novas entradas e focar na gestão das posições existentes. |
| | Muito Arrojado | 25% | Igual | Limite extremo. Atingir este nível pode indicar falha da estratégia. | Liquidar posições e reavaliar todo o sistema de trading. |
| **Limites Operacionais** | | | | | |
| `min_liquidity_usd` | Todos | > $5M (BR), > $10M (EUA) | **EUA > BR**. | Garante a capacidade de entrar e sair de posições sem impacto significativo no preço. | A liquidez é mais concentrada em poucos ativos na B3. |
| `max_spread_bps` | Todos | < 20 bps (BR), < 10 bps (EUA) | **BR > EUA**. | O spread é um custo direto de transação. | Spreads mais altos no Brasil refletem menor liquidez e maior risco. |

---
*Esta tabela será complementada com as seções narrativas e o checklist anti-overfitting.*


## (4) Especificação Narrativa de Cada Perfil

#### 1. Muito Conservador
- **Objetivo:** Preservação de capital com crescimento modesto e consistente.
- **Prioriza:** Baixa volatilidade, drawdowns mínimos e alta probabilidade de retornos positivos.
- **Sacrifica:** Potencial de altos retornos. A estratégia é desenhada para evitar perdas, mesmo que isso signifique perder oportunidades de ganhos maiores.
- **Expectativa em Crise:** Comportamento semelhante a um fundo de renda fixa de alta qualidade. Espera-se que o drawdown máximo fique abaixo de 5%. Os "freios" (circuit breakers) são acionados com perdas diárias de 0.5% ou semanais de 1%.

#### 2. Conservador
- **Objetivo:** Crescimento do capital com risco controlado.
- **Prioriza:** Retornos consistentes e um bom desempenho ajustado ao risco (Sharpe Ratio).
- **Sacrifica:** Retornos explosivos. A estratégia busca capturar a maior parte dos movimentos de mercado, mas com uma volatilidade controlada.
- **Expectativa em Crise:** Comportamento semelhante a um portfólio balanceado (60% ações, 40% renda fixa). O drawdown máximo esperado é de até 10%. Os freios são acionados com perdas diárias de 1% ou semanais de 2.5%.

#### 3. Moderado
- **Objetivo:** Crescimento do capital a longo prazo, aceitando volatilidade de curto prazo.
- **Prioriza:** Capturar tendências de mercado e maximizar o crescimento do capital.
- **Sacrifica:** Estabilidade de curto prazo. A estratégia aceita drawdowns maiores em troca de um maior potencial de retorno.
- **Expectativa em Crise:** Comportamento semelhante a um índice de ações diversificado (como o S&P 500). O drawdown máximo esperado é de até 15%. Os freios são acionados com perdas diárias de 2% ou semanais de 5%.

#### 4. Arrojado
- **Objetivo:** Máximo crescimento do capital, com alta tolerância a risco e volatilidade.
- **Prioriza:** Retornos agressivos e a utilização de alavancagem para amplificar os ganhos.
- **Sacrifica:** Previsibilidade e estabilidade. A estratégia pode ter grandes oscilações de curto prazo.
- **Expectativa em Crise:** Comportamento volátil, com potencial para grandes perdas. O drawdown máximo esperado é de até 20%. Os freios são acionados com perdas diárias de 3% ou semanais de 7.5%.

#### 5. Muito Arrojado
- **Objetivo:** Ganhos exponenciais, utilizando estratégias de alto risco e alta convicção.
- **Prioriza:** Capitalizar em oportunidades de curto prazo e movimentos de mercado extremos.
- **Sacrifica:** Qualquer semblante de estabilidade. A estratégia é de natureza especulativa.
- **Expectativa em Crise:** Extremamente volátil. O risco de ruína é uma possibilidade real se a gestão de risco não for impecável. O drawdown máximo esperado é de até 25%. Os freios são acionados com perdas diárias de 4% ou semanais de 10%.

## (5) Ajuste para Cenário de Poucos Ativos

- **Clusterização:** Para evitar a criação de regras para cada ativo individual, os ativos devem ser agrupados em clusters baseados em volatilidade histórica e liquidez. Parâmetros de risco (como multiplicador de ATR para stop) podem ser definidos por cluster.
- **Anti-Concentração:** Além do limite de exposição por ativo, um limite de correlação deve ser implementado. O sistema não deve permitir a entrada em um novo trade se a correlação do ativo com o portfólio existente for superior a 0.7.
- **Expansão do Universo:** Para expandir o universo de ativos negociáveis, filtros mínimos de liquidez (>$5M/dia no Brasil, >$10M/dia nos EUA) e spread (<20 bps no Brasil, <10 bps nos EUA) devem ser aplicados rigorosamente.

## (6) Análise do Passo 2 (Casos Especiais)

Os problemas identificados no "Relatório de Evidências" (Passo 2) surgiram porque os ranges de parâmetros no algoritmo genético permitiam valores economicamente inviáveis (ex: `min_carry` de 11%, `min_return` de 40%). Isso não é uma falha do algoritmo em si, mas uma falha na definição dos seus limites. Os novos ranges propostos na Tabela Principal servem como **guardrails** que impedem a geração de parâmetros absurdos, garantindo que todas as estratégias geradas sejam, no mínimo, plausíveis dentro da realidade de mercado.

## (7) Checklist Anti-Overfitting

- [ ] **In-Sample vs. Out-of-Sample:** A performance da estratégia é consistente em ambos os períodos? (Mínimo de 30% de dados para out-of-sample).
- [ ] **Walk-Forward Analysis:** A estratégia permanece lucrativa em diferentes janelas de tempo?
- [ ] **Sensibilidade a Custos:** A estratégia sobrevive com custos de slippage e corretagem 2x a 3x maiores que o esperado?
- [ ] **Estabilidade dos Parâmetros:** A performance da estratégia se degrada drasticamente com pequenas mudanças nos parâmetros (ex: mudar o período de uma média móvel de 20 para 21)?
- [ ] **Stress Tests:** A estratégia foi testada em cenários de crise (ex: 2008, 2020)? Qual o drawdown máximo nesses períodos?
- [ ] **Métricas Mínimas:** A estratégia atinge um Profit Factor > 1.3 e um Sharpe Ratio > 1.0 no período out-of-sample?

## (8) Referências

[1] Scholz, P. (2012). *Size matters! How position sizing determines risk and return of technical timing strategies*. [https://www.econstor.eu/handle/10419/55526]
[2] Rockafellar, R. T., & Uryasev, S. (2000). *Optimization of Conditional Value-at-Risk*. The Journal of Risk, 2(3), 21-41. [https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf]
[3] Vinso, J. D. (1979). *A Determination of the Risk of Ruin*. Journal of Financial and Quantitative Analysis. [https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/determination-of-the-risk-of-ruin/714F7E437FF2EDCEE81AADD5129C9C93]
[4] Lazzarini, S. G., & de Mello, P. C. (2001). *Governmental versus Self-regulation of Derivative Markets: examining the US and Brazilian Experience*. Journal of Financial Regulation and Compliance. [https://www.sciencedirect.com/science/article/pii/S0148619500000473]
[5] Chan, E. P. (2021). *Quantitative trading: how to build your own algorithmic trading business*. Wiley. [https://books.google.com/books?id=j70yEAAAQBAJ]

# Módulo: 04 - Gestão de Portfólio e PnL

---

## Sumário

1. [Papel do Módulo de Portfólio no Sistema](#1-papel-do-modulo-de-portfolio-no-sistema)
2. [Contrato de Estado da Carteira](#2-contrato-de-estado-da-carteira)
3. [Modelo de Posições: Semântica e Invariantes](#3-modelo-de-posicoes-semantica-e-invariantes)
4. [PnL: Definições Formais e Fórmulas](#4-pnl-definicoes-formais-e-formulas)
5. [Mark-to-Market e Integração com o Estado de Mercado](#5-mark-to-market-e-integracao-com-o-estado-de-mercado)
6. [Drawdown e Métricas Essenciais Derivadas do Estado](#6-drawdown-e-metricas-essenciais-derivadas-do-estado)
7. [Performance: Contratos de Complexidade e Memória](#7-performance-contratos-de-complexidade-e-memoria)
8. [Determinismo e Precisão Numérica](#8-determinismo-e-precisao-numerica)
9. [Interfaces Conceituais e Integração](#9-interfaces-conceituais-e-integracao)
10. [Casos-Limite e Testes de Sanidade](#10-casos-limite-e-testes-de-sanidade)
11. [Checklist de Aceite do Módulo](#11-checklist-de-aceite-do-modulo)
12. [Próximo Módulo Sugerido](#12-proximo-modulo-sugerido)

---

## 1. Papel do Módulo de Portfólio no Sistema

Este módulo é o guardião do estado da simulação. Ele representa a carteira de negociação e é o único componente com a autoridade para modificar posições e caixa. Sua principal função é receber `FillEvent`s do Modelo de Execução (conforme o fluxo do Módulo 03), atualizar o estado da carteira de forma atômica e determinística, e calcular o PnL e as métricas de risco em tempo real. Por ser atualizado a cada transação, este módulo é parte integral do **hot path** e está sujeito às mesmas restrições rigorosas de performance e determinismo que o Motor de Simulação.

## 2. Contrato de Estado da Carteira

O estado da carteira é representado por um conjunto mínimo de "átomos de estado", otimizados para acesso e atualização rápidos.

**Átomos de Estado Mínimos:**

| Estado | Representação Conceitual | Indexação | Descrição |
| :--- | :--- | :--- | :--- |
| **Posições** | `Vec<i64>` | `AssetId` | Quantidade de cada ativo. Positivo para longo, negativo para curto. |
| **Custo Médio** | `Vec<f64>` | `AssetId` | Custo médio de aquisição da posição atual. |
| **Caixa** | `f64` | Global | Saldo de caixa disponível. |
| **Valor Total (NAV)** | `f64` | Global | Valor total da carteira (Caixa + Valor de Mercado das Posições). |
| **Exposição Bruta** | `f64` | Global | Soma do valor absoluto de todas as posições. |
| **Exposição Líquida** | `f64` | Global | Soma do valor de todas as posições (considerando o sinal). |
| **PnL Realizado** | `Vec<f64>` | `AssetId` | Lucro ou prejuízo realizado para cada ativo. |
| **PnL Não Realizado** | `Vec<f64>` | `AssetId` | Lucro ou prejuízo latente da posição atual. |
| **Custos Acumulados** | `Vec<f64>` | `AssetId` | Total de custos (corretagem, taxas) pagos por ativo. |

**Proibições de Design:**
- **NÃO** usar `HashMap` ou qualquer estrutura de dados baseada em hashing no hot path para armazenar o estado por ativo. O acesso deve ser O(1) via indexação direta por `AssetId` nos vetores SoA.
- **NÃO** armazenar histórico de transações ou objetos complexos no estado principal. O estado deve conter apenas os valores agregados atuais.

**Invariantes de Consistência:**
- O `NAV` deve ser sempre igual à soma do `Caixa` e do valor de mercado de todas as posições.
- A atualização de estado por um `FillEvent` é atômica: Posição, Caixa e PnL são atualizados juntos antes que o próximo evento seja processado.

## 3. Modelo de Posições: Semântica e Invariantes

- **Representação**: Uma posição é simplesmente a quantidade (`i64`) no vetor `posicoes` no índice correspondente ao `AssetId`. `> 0` é longo, `< 0` é curto, `== 0` é flat.
- **Invariantes**:
    - Após um `FillEvent`, a posição e o caixa devem ser atualizados de forma determinística antes de qualquer cálculo de PnL subsequente.
    - O *netting* de posições (e.g., comprar 100, depois vender 30) é tratado com aritmética simples na quantidade da posição. Uma reversão (e.g., de +100 para -50) deve primeiro realizar o PnL dos 100 e depois estabelecer a nova base de custo para os -50.

**Tabela de Efeitos de um `FillEvent`:**

| Operação | Efeito na Posição (`qty`) | Efeito no Caixa (`cash`) | Efeito no PnL Realizado (`realized`) | Efeito no PnL Não Realizado (`unrealized`) |
| :--- | :--- | :--- | :--- | :--- |
| **Fill de Compra** | `qty += fill.qty` | `cash -= fill.qty * fill.price + fill.costs` | Inalterado (a menos que feche uma posição curta) | Recalculado com base no novo custo médio. |
| **Fill de Venda** | `qty -= fill.qty` | `cash += fill.qty * fill.price - fill.costs` | Atualizado se a venda fechar total ou parcialmente uma posição longa. | Recalculado. Se a posição zerar, o PnL não realizado vai a zero. |

## 4. PnL: Definições Formais e Fórmulas

- **PnL Não Realizado (Unrealized PnL)**: Para uma posição em um ativo, é `(preco_mercado_atual - custo_medio) * quantidade`. O PnL não realizado total é a soma para todos os ativos.
- **PnL Realizado (Realized PnL)**: É o lucro ou prejuízo travado quando uma posição é reduzida ou fechada. Calculado como `(preco_venda - custo_medio) * quantidade_vendida`.
- **NAV (Net Asset Value) / Equity**: `caixa + SUM((preco_mercado_atual * quantidade) for each asset)`.

**Política de Base de Custo:**
- O sistema utilizará **custo médio ponderado (Average Cost)** como política padrão e única. É determinístico, computacionalmente eficiente e o padrão da indústria para gestão de portfólio. Não haverá suporte a FIFO/LIFO para evitar complexidade.

**Ordem de Aplicação e Erros Comuns:**
- **Custos**: Os custos (`fill.costs`) vindos do `FillEvent` são deduzidos do caixa **imediatamente** na atualização do `FillEvent`. Eles impactam diretamente o NAV, mas são rastreados separadamente do PnL de trading para análise.
- **Timestamp de Marcação**: O PnL não realizado é sempre calculado usando o preço de mercado mais recente, fornecido pelo `MarketState` no mesmo timestamp `t` do evento que está sendo processado.
- **Erros a Evitar**: 
    - *Double-counting*: Os custos são subtraídos do caixa; não devem ser subtraídos novamente do PnL.
    - *Sinal Invertido*: A lógica deve tratar corretamente a matemática para posições curtas (vender para abrir, comprar para fechar).
    - *Realização Incorreta*: Em uma reversão de posição, o PnL da posição original deve ser totalmente realizado antes de estabelecer a nova base de custo.

## 5. Mark-to-Market e Integração com o Estado de Mercado

- **Quando**: O `mark-to-market` (atualização do PnL não realizado e do NAV) ocorre a cada `MarketEvent` para os ativos envolvidos, ou para toda a carteira no final de um período de tempo definido (e.g., final do dia).
- **Fonte de Preço**: A fonte de preço para o `mark-to-market` é contratualmente o campo `close` da `Bar` mais recente no `MarketState`, conforme definido nos Módulos 02 e 03.

**Checklist Anti-Look-Ahead (Mark-to-Market):**
- [ ] O preço usado para marcar a carteira em `t` é de um `MarketEvent` com timestamp `<= t`?
- [ ] O cálculo do PnL de um `FillEvent` usa o preço do *fill*, não um preço de mercado futuro?

## 6. Drawdown e Métricas Essenciais Derivadas do Estado

- **Drawdown**: Definido operacionalmente como a queda percentual do NAV a partir do seu pico histórico (`peak_equity`).
    - `peak_equity` é uma variável `f64` que é atualizada a cada passo: `peak_equity = max(peak_equity, nav)`.
    - `drawdown_percent = (nav - peak_equity) / peak_equity`.
- **Métricas no Loop**: Apenas as seguintes métricas de estado são permitidas no hot path:
    - Exposição Bruta: `SUM(|qty * price|)`
    - Exposição Líquida: `SUM(qty * price)`
    - Custos Acumulados: `SUM(fill.costs)`
- **Estratégia de Coleta**: Conforme o Módulo 03, a série temporal do NAV e do drawdown é escrita em um `Vec<f64>` pré-alocado para análise posterior.

## 7. Performance: Contratos de Complexidade e Memória

| Operação | Complexidade | Hot/Slow Path | Como Validar |
| :--- | :--- | :--- | :--- |
| **Atualização por `FillEvent`** | **O(1)** | Hot | Microbenchmark da função de atualização deve ser constante, independentemente do número de ativos. |
| **Mark-to-Market (1 ativo)** | **O(1)** | Hot | Microbenchmark da função de `mark` para um ativo. |
| **Mark-to-Market (Portfólio Inteiro)** | **O(N)**, N = # ativos | Slow (feito em eventos de fim de dia) | Benchmark da função de `mark_all` deve escalar linearmente com o número de ativos. |

**Estratégias Concretas:**
- **SoA**: O estado da carteira (posições, custos, PnL) é armazenado em `Vec`s indexados por `AssetId`, garantindo acesso O(1) e localidade de cache.
- **Zero Alocação**: Nenhuma alocação de memória ocorre durante a atualização por `FillEvent` ou `mark-to-market` de um único ativo.

## 8. Determinismo e Precisão Numérica

- **Ordem de Aplicação**: Fills são aplicados na ordem exata em que são recebidos do Motor. Se múltiplos fills ocorrerem no mesmo evento, sua ordem de aplicação será determinada pela chave de ordenação estável (Módulo 03).
- **Agregações**: Somas (e.g., para NAV) devem ser feitas em uma ordem fixa (e.g., iterando por `AssetId` de 0 a N) para garantir resultados de ponto flutuante reprodutíveis na mesma arquitetura.
- **Precisão**: Quantidades de ativos são `i64`. Caixa, preços e PnL são `f64`. Para evitar erros de precisão em comparações, o sistema deve usar uma tolerância (`epsilon`), mas para o determinismo, a ordem das operações é mais crítica.

## 9. Interfaces Conceituais e Integração

- **Entradas (Consumidas pelo Portfólio)**:
    - `FillEvent` (do Modelo de Execução).
    - `&MarketState` (do Motor, para `mark-to-market`).
- **Saídas (Fornecidas pelo Portfólio)**:
    - `&PortfolioState` (para o Roteador de Ordens, permitindo-lhe verificar o poder de compra ou posições existentes antes de gerar uma ordem).

**Proibições no Hot Path:**
- I/O, logging, alocação de memória, `String` formatting, `HashMap` lookups.

## 10. Casos-Limite e Testes de Sanidade

| Caso | Setup Mínimo | Resultado Esperado no Estado | Erro Típico Evitado |
| :--- | :--- | :--- | :--- |
| **Fill Parcial** | Ordem de 100, fill de 60. | Posição aumenta em 60, caixa diminui pelo valor de 60. | Erro de cálculo de PnL realizado/não realizado. |
| **Reversão Long→Short** | Posição +100, fill de venda de 150. | PnL dos 100 é realizado. Nova posição é -50 com nova base de custo. | Não realizar o PnL da posição original; base de custo incorreta. |
| **Custo Maior que PnL** | Fill com PnL de $10, custo de $12. | PnL realizado é $10, caixa diminui em $2. | Contabilização incorreta do impacto líquido no NAV. |
| **Net Zero no Fim do Dia** | Estratégia zera posições. | No evento de fim de sessão, todas as posições no vetor de estado são 0. | Posições residuais por erro de arredondamento ou lógica. |

## 11. Checklist de Aceite do Módulo

- [ ] O estado da carteira usa layout SoA com acesso O(1) por `AssetId`.
- [ ] As definições de PnL realizado e não realizado são formalizadas.
- [ ] A política de custo médio é a única implementada.
- [ ] A ordem de aplicação de custos e o `mark-to-market` são determinísticos.
- [ ] A lógica de `mark-to-market` é formalmente prevenida de look-ahead.
- [ ] O cálculo de drawdown é operacionalmente definido.
- [ ] O contrato de performance O(1) por fill é explícito.
- [ ] Nenhuma alocação de memória ocorre no hot path de atualização.
- [ ] As regras de determinismo (ordem de agregação) são especificadas.
- [ ] As interfaces de entrada/saída com outros módulos são claras.
- [ ] O tratamento de casos-limite (reversão, fills parciais) está definido.
- [ ] O módulo suporta a persistência de estado para `swing trade` e a verificação de `net zero`.

## 12. Próximo Módulo Sugerido

**`05_execution_model.md`**

- Detalhará os modelos para simular os vieses de execução: custos, slippage e latência simplificada.
- Especificará a API que o Modelo de Execução expõe ao Motor e como ele consome `OrderEvent`s e produz `FillEvent`s.
- Definirá os limites contratuais do modelo, reforçando a proibição de simulação de book de ofertas (L2) ou microestrutura. 

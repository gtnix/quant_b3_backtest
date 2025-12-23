# Módulo: 03 - Modelo de Eventos e Motor de Simulação

---

## Sumário

1. [Propósito do Motor de Simulação](#1-proposito-do-motor-de-simulacao)
2. [Contrato Operacional do Modelo de Eventos](#2-contrato-operacional-do-modelo-de-eventos)
3. [Especificação do Loop de Simulação (Hot Path)](#3-especificacao-do-loop-de-simulacao-hot-path)
4. [Barreiras Formais Contra Look-Ahead](#4-barreiras-formais-contra-look-ahead)
5. [Engenharia de Performance para o Hot Path](#5-engenharia-de-performance-para-o-hot-path)
6. [Contrato Técnico de Determinismo Bit-a-Bit](#6-contrato-tecnico-de-determinismo-bit-a-bit)
7. [Integração com o Modelo de Execução Enviesada](#7-integracao-com-o-modelo-de-execucao-enviesada)
8. [Suporte Unificado aos Modos de Estratégia](#8-suporte-unificado-aos-modos-de-estrategia)
9. [Interfaces Conceituais e Contratos de Hot Path](#9-interfaces-conceituais-e-contratos-de-hot-path)
10. [Coleta de Métricas Essenciais no Loop](#10-coleta-de-metricas-essenciais-no-loop)
11. [Plano de Validação e Benchmarking do Motor](#11-plano-de-validacao-e-benchmarking-do-motor)
12. [Checklist de Aceite do Módulo](#12-checklist-de-aceite-do-modulo)
13. [Próximo Módulo Sugerido](#13-proximo-modulo-sugerido)

---

## 1. Propósito do Motor de Simulação

O Motor de Simulação é o núcleo computacional do backtester, responsável por orquestrar o fluxo de eventos e executar a lógica da estratégia em um ambiente controlado. Sua única função é processar um fluxo de eventos pré-ordenado da forma mais rápida e determinística possível. Este componente é o *hot path* do sistema, onde 99% do tempo de execução será gasto. Portanto, seu design é obcecado por performance (NFR01) e determinismo bit-a-bit (NFR02), sendo o principal responsável por garantir que os critérios de aceite de velocidade (AC-01, AC-02) e corretude (AC-03) sejam atendidos.

## 2. Contrato Operacional do Modelo de Eventos

O motor consome o fluxo de `Event`s definido no Módulo 02, mas impõe regras operacionais estritas para garantir performance e corretude. A violação destas regras pelo pipeline de ingestão invalida a operação do motor.

| Regra | Motivação | Consequência de Performance | Consequência de Corretude |
| :--- | :--- | :--- | :--- |
| **Ordenação Estrita por Chave Composta** | Garantir uma linha do tempo inequívoca e determinística. | Permite o processamento sequencial sem a necessidade de reordenar ou consultar timestamps no loop. | Elimina não-determinismo e previne o viés de look-ahead estruturalmente. |
| **Monotonicidade do Timestamp** | O timestamp de um evento `N` deve ser sempre `>=` ao do evento `N-1`. | O motor pode avançar o "relógio" de forma linear, sem a necessidade de gerenciar estados passados ou futuros. | Garante que a simulação flui apenas para a frente no tempo. |
| **Imutabilidade do Evento** | Uma vez que um evento entra no motor, seu conteúdo não pode ser alterado. | Permite que o motor passe referências (`&Event`) sem risco de efeitos colaterais, evitando cópias. | Garante a rastreabilidade e a reprodutibilidade da simulação. |

**Política de Desempate (Timestamp Igual):**
A chave de ordenação é `(timestamp, event_type_priority, asset_id)`.
1.  `timestamp` (primária): Ordenação cronológica.
2.  `event_type_priority` (secundária): Define uma ordem fixa de processamento (e.g., `MarketEvent` antes de `SignalEvent`) para garantir que as decisões sejam tomadas com base no estado de mercado mais recente.
3.  `asset_id` (terciária): Garante uma ordem estável para eventos do mesmo tipo e timestamp, mas de ativos diferentes.

## 3. Especificação do Loop de Simulação (Hot Path)

O loop é projetado para ter um estado mínimo e uma sequência de operações rigorosa.

**Estado Mínimo do Motor:**
- `current_time`: O `Timestamp` do evento sendo processado.
- `market_state`: Uma visão do estado de mercado mais recente (e.g., últimos preços), otimizada para acesso rápido por `AssetId`.
- `event_queue`: Um iterador sobre o fluxo de eventos pré-ordenado, fornecido pelo pipeline de ingestão.

**Sequência de Execução por Evento:**
1.  **Consumir Evento**: O motor retira o próximo `Event` da `event_queue`. Se for um `MarketEvent`, avança para o passo 2. Outros tipos de evento (se existirem no futuro) podem ter caminhos diferentes.
2.  **Atualizar Estado de Mercado**: O `MarketEvent` é usado para atualizar a estrutura de dados `market_state`. Esta operação deve ser O(1).
3.  **Invocar Estratégia**: O motor invoca a função `on_market_data(&self, market_state)` da estratégia, passando uma referência imutável ao `market_state`. A estratégia não pode ver o futuro; ela só tem acesso aos dados até `current_time`.
4.  **Processar Sinais**: Se a estratégia retorna `SignalEvent`s, eles são passados imediatamente para o **Roteador de Ordens**.
5.  **Gerar Ordens**: O Roteador de Ordens, com acesso ao estado do **Portfólio**, converte os sinais em `OrderEvent`s concretos.
6.  **Simular Execução**: Os `OrderEvent`s são enviados ao **Modelo de Execução**, que aplica seus vieses e retorna `FillEvent`s (ou nada, se a ordem não for executada).
7.  **Atualizar Portfólio**: Os `FillEvent`s são processados pelo **Portfólio**, que atualiza posições, caixa e PnL.
8.  **Registrar Métricas**: Métricas essenciais (definidas na Seção 10) são atualizadas em buffers internos.

Após cada passo, o estado do sistema permanece consistente. A transição de estado principal ocorre nos passos 2, 6 e 7.

## 4. Barreiras Formais Contra Look-Ahead

- **Barreira de API**: A interface da `Strategy` é a principal barreira. Ela só recebe o `market_state` atual. É estruturalmente impossível para a estratégia acessar a `event_queue` ou qualquer informação futura.
- **Semântica de Barra**: O motor opera com a semântica `close-to-close`. A estratégia é invocada no `timestamp` de uma barra e toma decisões com base nos dados daquela barra (incluindo o `close`). As ordens geradas são para serem executadas no *próximo* evento de mercado disponível, prevenindo o uso do preço de fechamento para transacionar naquele mesmo preço.

**Checklist Anti-Look-Ahead:**
- [ ] A `Strategy` tem acesso apenas a dados com timestamp `<= current_time`?
- [ ] As ordens geradas em `t` são executadas em `t+1` (ou posterior)?
- [ ] O pipeline de dados não vaza informação futura (e.g., ajustes de preço aplicados incorretamente)?

## 5. Engenharia de Performance para o Hot Path

Esta seção define o contrato de performance do motor.

- **Layout de Dados (SoA)**: O `market_state` não é uma lista de objetos `Bar`. É uma estrutura que contém `Vec<f64>` para `close_prices`, `Vec<u64>` para `volumes`, etc., todos indexados por `AssetId`. Isso garante que o acesso sequencial a uma série temporal de um ativo seja um percurso linear na memória, maximizando a eficiência do cache da CPU.
- **Acesso à Memória**: Nenhuma alocação de memória (e.g., `Vec::push`) é permitida dentro do loop de simulação. Todos os buffers são pré-alocados. O uso de `Box` ou `dyn Trait` (trait objects) no hot path é proibido; a monomorfização via genéricos é preferida.
- **Previsibilidade de Branches**: O código do motor é escrito para ter um caminho linear. Condições (`if/else`) são minimizadas. Caminhos de erro ou casos raros (`slow paths`) são movidos para fora do loop principal sempre que possível.
- **Batching**: O motor pode ser configurado para processar eventos em lotes (e.g., todos os eventos com o mesmo timestamp de uma vez). Isso permite que a estratégia calcule indicadores em múltiplos ativos de forma vetorizada antes de gerar sinais, reduzindo o overhead de chamadas de função.
- **Paralelismo Determinístico**: O paralelismo **não** é usado para processar o *fluxo de eventos*, pois isso quebraria o determinismo. No entanto, ele é permitido para tarefas internas que são independentes, como o cálculo de indicadores em múltiplos ativos dentro de um mesmo passo de tempo, desde que a redução (agregação) dos resultados seja feita de forma determinística (e.g., somar resultados em uma ordem fixa).

**Orçamento de Custos do Evento (Modelo Mental):**
- `Fetch Event`: Custo desprezível (avançar um iterador).
- `Update Market State`: Custo muito baixo (escrita em um array).
- `Strategy Call`: **Principal custo**. Depende da complexidade da estratégia.
- `Execution Simulation`: Custo baixo a médio, dependendo da complexidade do modelo de slippage.
- `Portfolio Update`: Custo baixo (operações aritméticas).
- `Metrics Logging`: Custo muito baixo (escrita em buffer pré-alocado).

**Tabela de Otimizações:**

| Otimização | Benefício Esperado | Risco | Como Validar |
| :--- | :--- | :--- | :--- |
| **Layout SoA** | 10x-100x de melhoria no acesso a séries temporais. | Maior complexidade de código para gerenciar múltiplos arrays. | Microbenchmarks comparando acesso SoA vs. AoS para cálculos de médias móveis. |
| **Zero Alocação no Loop** | Redução drástica de jitter e latência. | Requer design cuidadoso e pré-alocação, o que pode aumentar o uso de memória estática. | Profiling de memória (e.g., `heaptrack`) não deve mostrar alocações no loop. |
| **Batching de Eventos** | Redução de overhead de chamadas de função. | Pode aumentar a latência do primeiro evento em um lote. | Benchmarks comparando o throughput do motor em modo de evento único vs. modo de lote. |

## 6. Contrato Técnico de Determinismo Bit-a-Bit

- **Ordenação**: A ordenação de eventos, fills e atualizações de portfólio é estritamente definida e não pode depender da implementação de hash maps ou da ordem de execução de threads paralelas.
- **Ponto Flutuante (Floating Point)**: O sistema **não** tentará alcançar determinismo em ponto flutuante entre diferentes arquiteturas de CPU. No entanto, ele **deve** ser determinístico na **mesma arquitetura**. Cálculos financeiros que exigem precisão absoluta (e.g., contagem de ações) devem usar tipos de preço fixo (decimal) ou inteiros escalados. O uso de `f64` é para quantidades que podem tolerar imprecisão (e.g., indicadores).
- **Assinatura de Resultados**: O motor deve produzir um hash de seus resultados (conforme AC-03), combinando a série temporal de PnL, as posições finais e as métricas principais. Isso permite uma validação rápida e inequívoca da reprodutibilidade.

## 7. Integração com o Modelo de Execução Enviesada

- O motor invoca o `ExecutionModel` através de uma interface `trait` bem definida: `fn execute(&self, order: &OrderEvent) -> Option<FillEvent>`. 
- O motor é agnóstico aos detalhes internos do modelo de execução. Ele apenas fornece a ordem e consome o *fill*, se houver.
- Os pontos de injeção de vieses (custos, slippage, latência) são de responsabilidade exclusiva do `ExecutionModel`. A latência simplificada é modelada atrasando o `timestamp` do `FillEvent` retornado.
- O contrato é claro: o `ExecutionModel` **não** pode acessar o book de ofertas (L2) ou qualquer informação de microestrutura, pois o motor não as fornece.

## 8. Suporte Unificado aos Modos de Estratégia

O motor não tem um "modo" diário ou intraday. Ele simplesmente processa eventos. A distinção é responsabilidade da **Estratégia** e da **Normalização**.
- **Swing Trade**: É o comportamento padrão. O estado do `Portfolio` persiste entre os eventos, independentemente de seus timestamps.
- **Net Zero Intraday**: A Estratégia deve ser programada para reconhecer um evento de fim de sessão (que pode ser um `MarketEvent` especial gerado pela Normalização) e emitir ordens para zerar todas as posições antes desse ponto.

## 9. Interfaces Conceituais e Contratos de Hot Path

| Componente | Entrada Mínima (Hot Path) | Saída Mínima (Hot Path) | Contrato de Tempo | Proibições no Hot Path |
| :--- | :--- | :--- | :--- | :--- |
| **Strategy** | `&MarketState` | `Vec<SignalEvent>` | Deve ser o mais rápido possível. Evitar algoritmos de alta complexidade. | I/O, alocação de memória, logging verboso, acesso a estado global. |
| **ExecutionModel** | `&OrderEvent` | `Option<FillEvent>` | Deve ser, em média, O(1) por ordem. | I/O, alocação, estado complexo. |
| **Portfolio** | `&FillEvent` | N/A (muda estado interno) | Deve ser O(1) por fill. | I/O, alocação. |

## 10. Coleta de Métricas Essenciais no Loop

Para evitar sobrecarga de performance, apenas as métricas mais críticas são atualizadas durante o loop:
- **Métricas Mínimas**: PnL diário/intraday, valor da carteira, posições, exposição bruta/líquida.
- **Estratégia de Coleta**: Os dados são escritos em **buffers pré-alocados**. Por exemplo, a série temporal do valor da carteira é um `Vec<f64>` que é preenchido durante a simulação. Cálculos mais complexos (e.g., Sharpe Ratio, estatísticas de drawdown) são feitos em uma etapa de **pós-processamento** após o término do loop, usando os dados dos buffers.

## 11. Plano de Validação e Benchmarking do Motor

- **Microbenchmarks**: Cada função no hot path (e.g., `Portfolio::update`, `ExecutionModel::execute`) terá microbenchmarks (usando `criterion.rs` ou similar) para medir sua latência em nanossegundos.
- **Testes de Determinismo**: Scripts de teste executarão o mesmo backtest duas vezes seguidas e compararão o hash de resultado. Eles devem ser idênticos.
- **Testes de Não-Look-Ahead**: Testes sintéticos serão criados com padrões de dados que revelariam look-ahead (e.g., um pico de preço em `t+1`). A estratégia não deve ser capaz de reagir a ele em `t`.
- **Benchmarks de Macro**: O tempo total de execução para os cenários definidos nos critérios AC-01 e AC-02 será medido para garantir que as metas de performance sejam atingidas.

## 12. Checklist de Aceite do Módulo

- [ ] O contrato de ordenação de eventos está definido e é determinístico?
- [ ] A sequência do loop de simulação está especificada e suas invariantes são claras?
- [ ] As barreiras formais contra look-ahead estão definidas na API da Estratégia?
- [ ] A estratégia de performance (SoA, zero alocação, etc.) está documentada?
- [ ] O orçamento de custos do evento está definido como um modelo mental?
- [ ] A política de determinismo bit-a-bit (na mesma arquitetura) está clara?
- [ ] As regras para ponto flutuante são explícitas?
- [ ] A interface com o Modelo de Execução está definida e limita seu escopo?
- [ ] O suporte unificado para swing trade e net zero está explicado?
- [ ] Os contratos de hot path (proibições) para os componentes principais estão definidos?
- [ ] A estratégia de coleta de métricas é eficiente (buffers + pós-processamento)?
- [ ] O plano de validação e benchmarking está alinhado com os critérios de aceite existentes?

## 13. Próximo Módulo Sugerido

**`04_portfolio_and_pnl_management.md`**

- Detalhará as estruturas de dados e a lógica para o gerenciamento de estado da carteira, incluindo posições, caixa e valor total (NAV).
- Especificará os algoritmos para cálculo de PnL (realizado e não realizado) e drawdown em tempo real.
- Descreverá como o módulo de portfólio interage com o Roteador de Ordens para fornecer informações de dimensionamento de posição e verificação de risco.

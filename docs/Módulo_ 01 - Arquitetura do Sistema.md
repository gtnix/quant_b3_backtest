# Módulo: 01 - Arquitetura do Sistema

---

## Sumário

1. [Visão Geral da Arquitetura](#1-visao-geral-da-arquitetura)
2. [Diagrama de Componentes](#2-diagrama-de-componentes)
3. [Componentes e Responsabilidades](#3-componentes-e-responsabilidades)
4. [Modelo de Dados de Alto Nível](#4-modelo-de-dados-de-alto-nivel)
5. [Fluxo de Execução do Backtest](#5-fluxo-de-execucao-do-backtest)
6. [Determinismo e Reprodutibilidade](#6-determinismo-e-reprodutibilidade)
7. [Estratégia de Performance](#7-estrategia-de-performance)
8. [Fronteiras de Módulos/Crates (Rust)](#8-fronteiras-de-moduloscrates-rust)
9. [Prevenção de Falhas Comuns na Arquitetura](#9-prevencao-de-falhas-comuns-na-arquitetura)
10. [Próximo Módulo Sugerido](#10-proximo-modulo-sugerido)

---

## 1. Visão Geral da Arquitetura

A arquitetura do sistema foi projetada para ser um pipeline de processamento de eventos sequencial, determinístico e de altíssima performance. O fluxo se inicia com a **Ingestão** de dados brutos de barras OHLCV (diárias ou intraday). Esses dados são processados por um módulo de **Normalização** que os converte para um formato canônico, com timestamps em UTC e alinhados a um calendário de sessões de mercado. O **Motor de Simulação** (Event Loop) consome esses eventos de mercado em ordem cronológica estrita e os entrega à **Estratégia**. A Estratégia, por sua vez, gera sinais que são transformados em ordens e enviados a um **Roteador de Ordens**. O **Modelo de Execução** simula o preenchimento dessas ordens, aplicando vieses como custos, slippage e latência simplificada, e gerando eventos de *fill*. Esses *fills* são processados pelo módulo de **Portfólio**, que atualiza as posições e calcula métricas de PnL e Drawdown em tempo real. Ao final da simulação, um módulo de **Relatórios** agrega os dados e gera os resultados finais. Todo o design é orientado a dados para maximizar a eficiência de memória e processamento, garantindo a performance exigida pelos requisitos NFR01, AC-01 e AC-02.

## 2. Diagrama de Componentes

O diagrama a seguir ilustra a arquitetura de componentes e o fluxo de dados principal. As setas indicam o fluxo de dados e controle entre os componentes.

```ascii
+--------------------+
|                    |
|  [ Ingestão/Leitura ]  <-- (Arquivos de Dados: CSV, Parquet, etc.)
|                    |
+----------+---------+
           |
           v (Barras Brutas)
+----------+---------+
|   Normalização     |
| (Calendário/Sessão)| 
+----------+---------+
           |
           v (Eventos de Mercado Ordenados)
+----------+---------+
|   Motor de Simulação (Event Loop)   |
+----------+---------+
     |           ^ 
     v (Market)  | (Sinal)
+----------+---------+
|      Estratégia    |
+----------+---------+
           |
           v (Ordem)
+----------+---------+
|  Roteador de Ordens |
+----------+---------+
           |
           v (Ordem para Execução)
+----------+---------+
|  Modelo de Execução (Fills + Vieses) |
+----------+---------+
           |
           v (Fill)
+----------+---------+
| Portfólio/PnL/Drawdown |
+----------+---------+
           |
           v (Métricas Finais)
+----------+---------+
|      Relatórios    | --> (Saída: CSV, JSON, etc.)
+--------------------+
```

## 3. Componentes e Responsabilidades

| Componente | Responsabilidade | Entradas | Saídas | Invariantes | Considerações de Performance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ingestão/Leitura** | Ler dados de mercado de fontes externas (arquivos). | Caminho do arquivo, formato. | Sequência de barras OHLCV brutas. | Não garante ordenação ou alinhamento temporal. | Otimizado para I/O em bloco (batch). Evita parsing linha a linha em loops quentes. |
| **Normalização** | Converter barras brutas em eventos de mercado canônicos, alinhados por tempo (UTC) e sessões de mercado. | Barras brutas. | Fluxo de eventos de mercado ordenado por timestamp. | Garante que todos os timestamps são UTC. Garante ordenação cronológica estrita. | Operação feita em pré-processamento. Evita alocações desnecessárias ao criar eventos. |
| **Motor de Simulação** | Orquestrar o fluxo de eventos, garantindo a entrega em ordem cronológica para os demais componentes. | Fluxo de eventos de mercado. | N/A (controla o loop). | Garante que nenhum evento futuro (look-ahead) seja acessível. | O "coração" do sistema. Loop extremamente otimizado, sem alocações de memória, I/O ou qualquer operação bloqueante. |
| **Estratégia** | Implementar a lógica de trading, gerando sinais de compra/venda com base nos eventos de mercado. | Eventos de mercado. | Eventos de sinal (e.g., `SignalEvent`). | A lógica da estratégia é "pura": não tem acesso a estado externo, exceto o que é fornecido pelo motor. | O código da estratégia é um "hot spot". Deve ser otimizado para evitar cálculos redundantes. |
| **Roteador de Ordens** | Converter sinais em ordens concretas, com tamanho, tipo e limites, e direcioná-las para execução. | Eventos de sinal, estado do portfólio. | Eventos de ordem (e.g., `OrderEvent`). | Garante que as ordens são bem-formadas e respeitam as restrições de risco/portfólio. | Operação leve. Validações rápidas. |
| **Modelo de Execução** | Simular o preenchimento de ordens, aplicando vieses (custos, slippage, latência simplificada). | Eventos de ordem. | Eventos de preenchimento (e.g., `FillEvent`). | Garante que todos os custos e vieses configurados são aplicados de forma determinística. | Pode ser um gargalo se a modelagem for complexa. A latência simplificada é apenas um atraso no timestamp do *fill*. |
| **Portfólio** | Manter o estado atual da carteira (posições, caixa) e calcular métricas de performance (PnL, Drawdown). | Eventos de preenchimento. | Estado atualizado da carteira e métricas. | Garante que o estado da carteira é consistente e reflete todas as transações executadas. | Atualizações devem ser computacionalmente baratas. Cálculos de métricas podem ser feitos em batch no final do dia/simulação. |
| **Relatórios** | Agregar e formatar os resultados finais da simulação para análise. | Métricas finais, série temporal do PnL. | Arquivos de resultado (CSV, JSON). | Garante que os relatórios são consistentes com os dados finais do módulo de Portfólio. | Executado apenas no final da simulação, fora do loop principal. Não impacta a performance do backtest. |

## 4. Modelo de Dados de Alto Nível

A representação de dados é fundamental para a performance. A arquitetura adota um design orientado a dados (Data-Oriented Design).

- **Timestamp**: Todos os timestamps internos são representados como inteiros de 64 bits (nanossegundos desde a época UNIX) e padronizados em **UTC**. A conversão para timezones locais ocorre apenas na camada de Relatórios. Isso elimina qualquer ambiguidade de fuso horário no núcleo do sistema (atende ao risco de inconsistência de timezone do Módulo 00).
- **Barra OHLCV**: Representada como uma struct `(timestamp, open, high, low, close, volume)`, com preços e volume em formatos numéricos de ponto flutuante de 64 bits ou inteiros de preço fixo para evitar erros de arredondamento.
- **Universo de Ativos**: Os ativos são mapeados para identificadores inteiros (`AssetId`) para acesso eficiente a arrays e vetores, em vez de usar strings (tickers) em caminhos críticos.
- **Eventos**: O sistema opera sobre uma enum `Event` que encapsula os diferentes tipos de eventos (`MarketEvent`, `SignalEvent`, `OrderEvent`, `FillEvent`). Isso permite o processamento polimórfico em um único loop de eventos.
- **Diário vs. Intraday**: A diferença é tratada na camada de **Normalização**. Dados diários são simplesmente barras com um timestamp fixo (e.g., 23:59:59 UTC do dia). Dados intraday são barras com seus respectivos timestamps. O Motor de Simulação não diferencia os dois; ele apenas processa eventos em ordem cronológica.

## 5. Fluxo de Execução do Backtest

O processo de execução de um backtest segue uma sequência estrita:

1.  **Carregamento e Normalização**: Os dados de mercado para o período e ativos solicitados são carregados pela **Ingestão** e processados pela **Normalização**, gerando uma fila de eventos de mercado ordenada cronologicamente.
2.  **Inicialização**: O **Motor de Simulação** é inicializado, juntamente com a **Estratégia** e o **Portfólio** (com seu capital inicial).
3.  **Loop de Eventos**: O Motor inicia o loop, processando um evento de cada vez:
    a. Retira o próximo `MarketEvent` da fila.
    b. Atualiza o estado interno (e.g., preços atuais) e o entrega à **Estratégia**.
    c. A **Estratégia** avalia suas regras e, se aplicável, gera um `SignalEvent`.
    d. O **Roteador de Ordens** recebe o sinal, consulta o **Portfólio** para determinar o tamanho da ordem e cria um `OrderEvent`.
    e. O **Modelo de Execução** recebe a ordem, aplica seus modelos de custo/slippage/latência e, se a ordem for preenchida, gera um `FillEvent`.
    f. O **Portfólio** recebe o `FillEvent`, atualiza a posição no ativo, deduz custos do caixa e recalcula o PnL.
4.  **Finalização**: Quando a fila de eventos de mercado está vazia, o loop termina.
5.  **Geração de Relatório**: O módulo de **Relatórios** é chamado para calcular as métricas finais (e.g., Sharpe Ratio, Calmar Ratio) e salvar os resultados.

Este fluxo único suporta tanto **swing trade** (a estratégia simplesmente não zera posições no final do dia) quanto **net zero** (a estratégia contém lógica explícita para gerar ordens de fechamento de posição antes do final da sessão de negociação).

## 6. Determinismo e Reprodutibilidade

Para garantir o requisito NFR02 e o critério de aceite AC-03, a arquitetura impõe as seguintes regras:

- **Ordenação Estável de Eventos**: Se múltiplos eventos ocorrerem no mesmo timestamp (e.g., sinais de diferentes estratégias), eles serão processados em uma ordem secundária estável e definida (e.g., por tipo de evento, por `AssetId`).
- **Aleatoriedade Controlada**: Qualquer componente que utilize números aleatórios (e.g., em um modelo de slippage) deve ser inicializado com uma semente (seed) fixa, que é registrada nos metadados do backtest.
- **Paralelismo Determinístico**: Se o paralelismo for usado (ver Seção 7), ele não pode introduzir não-determinismo. Por exemplo, a atualização do estado do portfólio a partir de múltiplos *fills* paralelos deve ser feita em uma ordem definida ou através de operações atômicas que não dependam da ordem de chegada.
- **Hashing de Resultados**: Ao final de cada execução, um hash é gerado a partir dos resultados principais (e.g., série temporal de PnL, posições finais). Execuções idênticas devem produzir hashes idênticos, permitindo uma verificação rápida e "bit a bit".

## 7. Estratégia de Performance

A performance extrema (NFR01) é um pilar desta arquitetura, alcançada através de:

- **Data-Oriented Design**: Em vez de arrays de objetos (AoS), o sistema prioriza objetos de arrays (SoA). Por exemplo, os dados de barras são armazenados como múltiplos arrays (um para timestamps, um para `open`, um para `close`, etc.). Isso melhora drasticamente a localidade de cache da CPU quando um algoritmo itera sobre um único campo (e.g., calculando uma média móvel sobre os preços de fechamento).
- **Zero Alocação no Loop Quente**: O motor de eventos e o código da estratégia são projetados para evitar qualquer alocação de memória durante o loop de simulação. Todos os buffers e estruturas de dados necessários são pré-alocados.
- **Processamento em Batch (Batching)**: Onde possível, os eventos são processados em lotes em vez de individualmente para reduzir o overhead de chamadas de função e melhorar a vetorização.
- **Paralelismo Permitido**: A arquitetura permite paralelismo em tarefas que são inerentemente independentes, como:
    - **Otimização de Parâmetros**: Executar múltiplos backtests com diferentes conjuntos de parâmetros em paralelo. Cada backtest é um processo independente.
    - **Backtests por Ativo**: Para estratégias que não têm lógica de portfólio cruzada, os backtests podem ser executados por ativo em paralelo e os resultados agregados no final. A ordenação dos ativos deve ser fixa para garantir o determinismo.

Essas estratégias visam atender aos critérios de aceite AC-01 (<10s para backtest diário) e AC-02 (<5min para backtest intraday), garantindo que o tempo de execução seja dominado pela lógica da estratégia e não pelo overhead da arquitetura.

## 8. Fronteiras de Módulos/Crates (Rust)

Para garantir a modularidade (NFR04), os componentes são mapeados para uma estrutura de crates conceituais em Rust:

- `backtester_core`: Define os traits e tipos de dados fundamentais (`Event`, `Bar`, `Order`, `Fill`, traits para `Strategy`, `ExecutionModel`, etc.). É a dependência central de todos os outros crates.
- `backtester_io`: Responsável pela **Ingestão** e **Normalização**. Depende de `backtester_core`.
- `backtester_engine`: Contém o **Motor de Simulação** e o **Roteador de Ordens**. Depende de `backtester_core`.
- `backtester_portfolio`: Implementa o **Portfólio**. Depende de `backtester_core`.
- `backtester_execution`: Implementa o **Modelo de Execução**. Depende de `backtester_core`.
- `backtester_reports`: Implementa os **Relatórios**. Depende de `backtester_core` e `backtester_portfolio`.
- `strategy_lib`: Crate externo onde as **Estratégias** do usuário são implementadas. Depende de `backtester_core`.

O grafo de dependências é acíclico, garantindo uma separação clara de responsabilidades. Por exemplo, `strategy_lib` não pode depender de `backtester_io`, impedindo que a estratégia acesse dados diretamente do sistema de arquivos.

## 9. Prevenção de Falhas Comuns na Arquitetura

A arquitetura implementa mecanismos específicos para prevenir os riscos identificados no Módulo 00:

- **Look-Ahead Bias**: O **Motor de Simulação** expõe apenas o evento do tempo `t` para a estratégia. O acesso a dados futuros é estruturalmente impossível.
- **Survivorship Bias**: A responsabilidade de fornecer um universo de ativos historicamente preciso recai sobre a camada de **Ingestão/Normalização**. A arquitetura suporta isso ao não assumir um conjunto fixo de ativos durante toda a simulação.
- **Erro de Alinhamento Temporal**: O componente de **Normalização** e o uso estrito de **timestamps UTC** garantem que todos os dados são alinhados em uma única linha do tempo antes de entrarem no motor, eliminando este risco.
- **Inconsistência de Timezone/Sessão**: O uso exclusivo de UTC para todo o processamento interno remove qualquer ambiguidade. A conversão para timezones locais é uma responsabilidade exclusiva da camada de **Relatórios**.

## 10. Próximo Módulo Sugerido

**`02_data_model_and_ingestion.md`**

- Detalhará as estruturas de dados (`structs`, `enums`) em um nível pseudo-código, incluindo a representação de barras, eventos e identificadores.
- Especificará os formatos de arquivo de entrada suportados (e.g., CSV, Parquet) e o esquema esperado para cada um.
- Descreverá o processo de normalização de dados, incluindo o tratamento de dados ausentes, ajuste de preços (splits/dividendos) e o alinhamento de sessões de mercado. 

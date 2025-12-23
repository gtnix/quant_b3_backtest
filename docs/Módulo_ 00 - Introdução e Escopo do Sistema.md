# Módulo: 00 - Introdução e Escopo do Sistema

---

## Sumário

1. [Objetivo do Projeto](#1-objetivo-do-projeto)
2. [Escopo Funcional](#2-escopo-funcional)
3. [Requisitos Não-Funcionais](#3-requisitos-nao-funcionais)
4. [Não-Requisitos (Fora do Escopo)](#4-nao-requisitos-fora-do-escopo)
5. [Glossário](#5-glossario)
6. [Decisões de Arquitetura Assumidas](#6-decisoes-de-arquitetura-assumidas)
7. [Critérios de Aceite do Sistema](#7-criterios-de-aceite-do-sistema)
8. [Riscos e Antipadrões a Evitar](#8-riscos-e-antipadroes-a-evitar)
9. [Próximo Módulo Sugerido](#9-proximo-modulo-sugerido)

---

## 1. Objetivo do Projeto

O objetivo deste documento é definir contratualmente o escopo e os requisitos para a construção de um **sistema proprietário de backtesting de estratégias de trading**, otimizado para **performance extrema** e execução local em ambiente **Linux**. 

O sistema servirá como uma ferramenta de pesquisa quantitativa para validar estratégias de **portfolio trading**, suportando os seguintes casos de uso:
- **Backtest de estratégias intraday**: Operações de curto prazo, frequentemente com posição líquida zerada ao final do dia (**net zero**).
- **Backtest de estratégias diárias**: Operações de prazo mais longo, com capacidade de carregar posições por vários dias (**swing trade**).

## 2. Escopo Funcional

O sistema operará exclusivamente dentro das seguintes capacidades funcionais. Qualquer funcionalidade não descrita aqui é considerada fora do escopo.

| ID | Capacidade | Descrição Técnica | Critério de Verificação |
| :--- | :--- | :--- | :--- |
| **F01** | **Processamento Baseado em Barras** | O motor de simulação processará dados de mercado exclusivamente no formato de barras OHLCV (Open, High, Low, Close, Volume). | O sistema deve ser capaz de carregar e processar arquivos de dados contendo séries temporais de barras OHLCV. |
| **F02** | **Suporte a Múltiplas Frequências** | O sistema deve suportar nativamente a execução de backtests em frequências **diária** e **intraday** (e.g., 1-min, 5-min, 60-min). | É possível executar o mesmo backtest com dados diários e, subsequentemente, com dados de 1 minuto, obtendo resultados consistentes com a frequência. |
| **F03** | **Gestão de Portfólio** | O sistema deve gerenciar o estado de uma carteira de múltiplos ativos, incluindo posições, caixa, e o cálculo de PnL (Profit and Loss) e Drawdown. | Ao final de uma simulação, o sistema reporta o PnL total, o PnL por ativo e o drawdown máximo da carteira. |
| **F04** | **Modelo de Execução Simulado** | A execução de ordens será simulada através de um modelo que incorpora **vieses de execução**: custos de corretagem/taxas, slippage (derrapagem) e regras de preenchimento (fill). | É possível configurar diferentes parâmetros de custos e slippage e observar seu impacto no resultado final do backtest. |

## 3. Requisitos Não-Funcionais

Os requisitos não-funcionais são centrais para o design do sistema e devem ser tratados com a mesma prioridade que os requisitos funcionais.

| ID | Requisito | Descrição Técnica | Critério de Verificação |
| :--- | :--- | :--- | :--- |
| **NFR01** | **Performance Extrema** | O design do sistema deve ser orientado a dados (data-oriented design) para maximizar a localidade de cache e permitir paralelismo. O objetivo é obter a menor latência possível para o escopo definido. | Os tempos de execução devem atender aos benchmarks definidos nos Critérios de Aceite (Seção 7). O profiling de performance não deve indicar gargalos óbvios de CPU ou memória. |
| **NFR02** | **Determinismo e Reprodutibilidade** | Execuções do backtest com o mesmo conjunto de dados, código de estratégia e parâmetros de configuração devem produzir resultados **idênticos** (bit a bit). | A execução repetida do mesmo teste de cenário produz um hash de resultado idêntico. |
| **NFR03** | **Execução em Linux** | O sistema deve ser compilado e executado nativamente em um ambiente Linux padrão (e.g., Ubuntu 22.04) sem o uso de camadas de compatibilidade. | O binário compilado executa com sucesso em um container Docker com a imagem base do Ubuntu 22.04. |
| **NFR04** | **Modularidade e Manutenibilidade** | O código-fonte deve ser organizado em módulos (crates, em Rust) com responsabilidades claras e APIs bem definidas, seguindo as melhores práticas da linguagem. | A adição de um novo tipo de custo de execução requer modificação em apenas um módulo, sem afetar o motor de simulação ou a gestão de portfólio. |

## 4. Não-Requisitos (Fora do Escopo)

Os seguintes itens estão **explicitamente fora do escopo** deste projeto. A sua inclusão é proibida para garantir o foco e a entrega de um sistema especialista.

- **NÃO** haverá processamento, armazenamento ou simulação de dados de **book de ofertas (L2/L3)**.
- **NÃO** haverá simulação do ciclo de vida de ordens individuais (order-by-order) ou de sua posição na fila de uma corretora.
- **NÃO** haverá qualquer análise ou modelagem de fenômenos de **microestrutura de mercado** ou HFT (High-Frequency Trading).
- **NÃO** será desenvolvida uma **interface gráfica (GUI)**. A interação com o sistema será via linha de comando (CLI) ou APIs programáticas.
- **NÃO** haverá integração com **corretoras ou provedores de dados em tempo real** (live trading).

## 5. Glossário

- **Backtest Diário**: Simulação de uma estratégia de trading usando dados com a granularidade de um dia (preços de abertura, máxima, mínima e fechamento diários).
- **Backtest Intraday**: Simulação usando dados com granularidade inferior a um dia (e.g., barras de 1 minuto, 5 minutos).
- **Custo**: Taxas de corretagem, emolumentos e outros custos associados à execução de uma transação.
- **Drawdown**: A perda percentual máxima observada em uma carteira a partir de um pico de valor até o seu ponto mais baixo subsequente.
- **Fill**: O preenchimento (execução) de uma ordem de compra ou venda.
- **Net Zero**: Estratégia de trading cujo objetivo é terminar o dia de negociação sem posições em carteira, ou seja, com exposição líquida zero ao mercado.
- **PnL (Profit and Loss)**: O lucro ou prejuízo financeiro de uma carteira, realizado ou não realizado.
- **Slippage**: A diferença entre o preço esperado de uma ordem e o preço no qual ela é efetivamente executada.
- **Swing Trade**: Estratégia que mantém posições em carteira por mais de um dia, tipicamente alguns dias ou semanas.

## 6. Decisões de Arquitetura Assumidas

- **Linguagem de Programação Principal: Rust**
  - **Justificativa**: Conforme a "Análise Comparativa de Stacks", Rust foi selecionada como a linguagem principal. A decisão se baseia na sua capacidade de entregar **performance de nível C++** com **garantias de segurança de memória em tempo de compilação** (zero-cost abstractions, ownership model). Isso é crítico para um sistema financeiro que exige alta robustez e previsibilidade, ao mesmo tempo que mitiga classes inteiras de bugs (e.g., data races, null pointer dereferences) que são fontes comuns de instabilidade em sistemas de trading. O ecossistema moderno (Cargo, Crates.io) também acelera o desenvolvimento de software confiável.

## 7. Critérios de Aceite do Sistema

O sistema será considerado aceito se atender aos seguintes benchmarks de performance e corretude em uma máquina de desenvolvimento padrão (e.g., CPU 8-core, 16GB RAM).

- **AC-01 (Backtest Diário)**: A execução de uma estratégia de swing trade (rebalanceamento mensal) em um universo de **100 ativos** com **5 anos de dados diários** deve ser concluída em **menos de 10 segundos**.
- **AC-02 (Backtest Intraday)**: A execução de uma estratégia de day trade (net zero) em um universo de **20 ativos** com **1 ano de dados de 1 minuto** deve ser concluída em **menos de 5 minutos**.
- **AC-03 (Corretude)**: Os resultados de PnL e Drawdown para um conjunto de testes de validação devem corresponder exatamente (bit a bit) aos resultados de referência pré-calculados.

## 8. Riscos e Antipadrões a Evitar

O design e a implementação devem mitigar ativamente os seguintes riscos inerentes a sistemas de backtesting:

- **Viés de Look-Ahead (Look-Ahead Bias)**: Risco de usar informação no backtest que não estaria disponível no momento da decisão no mundo real. A arquitetura de eventos deve garantir que a estratégia só tenha acesso a dados até o timestamp `t` para tomar decisões em `t`.
- **Viés de Sobrevivência (Survivorship Bias)**: Risco de usar um universo de ativos que exclui aqueles que faliram ou foram deslistados. O processo de ingestão de dados deve ser capaz de lidar com históricos de ativos que terminam.
- **Erro de Alinhamento Temporal**: Risco de desalinhamento entre dados de diferentes ativos ou fontes (e.g., usar o preço de fechamento de um ativo para tomar uma decisão sobre outro antes que aquele fechamento fosse público). A pipeline de dados deve garantir a sincronização correta de todos os eventos.
- **Inconsistência de Timezone/Sessão**: Risco de erros no tratamento de fusos horários e horários de pregão. Todas as operações temporais devem ser normalizadas para um padrão único (e.g., UTC) e convertidas para o fuso horário local apenas para exibição.

## 9. Próximo Módulo Sugerido

**`01_system_architecture.md`**

- Apresentará a visão macro da arquitetura, com um diagrama de componentes de alto nível (Ingestor, Motor de Eventos, Gerenciador de Carteira, Modelo de Execução).
- Detalhará as principais interfaces e fluxos de dados entre os componentes.
- Explicará como a arquitetura atende aos requisitos de performance e aos diferentes modos de operação (diário vs. intraday).

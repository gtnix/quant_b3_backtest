# Módulo: 08 - Biblioteca de Estratégias e API

---

## Sumário

1. [Objetivo da API de Estratégia](#1-objetivo-da-api-de-estrategia)
2. [Princípios de Design da API](#2-principios-de-design-da-api)
3. [Ciclo de Vida de uma Estratégia](#3-ciclo-de-vida-de-uma-estrategia)
4. [Modelo de Acesso a Dados: Views e Barreiras](#4-modelo-de-acesso-a-dados-views-e-barreiras)
5. [Interface de Emissão de Ordens](#5-interface-de-emissao-de-ordens)
6. [Integração com o Estado do Portfólio](#6-integracao-com-o-estado-do-portfolio)
7. [Contrato de Performance](#7-contrato-de-performance)
8. [Contrato de Determinismo](#8-contrato-de-determinismo)
9. [Biblioteca de Estratégias: Empacotamento e Versão](#9-biblioteca-de-estrategias-empacotamento-e-versao)
10. [Guidelines de Qualidade e Segurança](#10-guidelines-de-qualidade-e-seguranca)
11. [Harness de Validação de Estratégia](#11-harness-de-validacao-de-estrategia)
12. [Checklist de Aceite do Módulo](#12-checklist-de-aceite-do-modulo)
13. [Próximo Módulo Sugerido](#13-proximo-modulo-sugerido)

---

## 1. Objetivo da API de Estratégia

A API de Estratégia é o ponto de extensão do sistema, permitindo que pesquisadores quantitativos implementem sua lógica de negociação de forma segura e performática. Ela é o contrato externo mais sensível, pois o código da estratégia é um "convidado" (guest) no hot path do motor de simulação (Módulo 03). O design desta API visa, portanto, a máxima liberdade de expressão para a lógica da estratégia, ao mesmo tempo que impõe barreiras invioláveis para proteger a performance e o determinismo (AC-03) do sistema como um todo.

## 2. Princípios de Design da API

- **Estratégia é um Convidado no Hot Path**: A API é projetada para minimizar o impacto da estratégia no throughput do motor.
- **Zero Alocação por Evento**: A API proíbe e previne, por design, que a estratégia aloque memória no seu hook de processamento de eventos.
- **Sem I/O, Sem Surpresas**: A estratégia opera em uma sandbox lógica, sem acesso a I/O (disco, rede) ou a qualquer estado externo não fornecido pela API.
- **Sem RNG Não-Seedado**: Qualquer fonte de aleatoriedade deve ser explicitamente solicitada e será fornecida com uma semente determinística.
- **Sem Dependência de Relógio/Sistema**: A estratégia não pode acessar o relógio do sistema; o tempo é ditado exclusivamente pelos timestamps dos eventos.
- **Dados Apenas em `t`**: A API garante estruturalmente que a estratégia não possa acessar informações futuras (anti-look-ahead).

## 3. Ciclo de Vida de uma Estratégia

| Hook | Quando é Chamado | Entradas Permitidas | Saídas Permitidas | Proibições no Hot Path |
| :--- | :--- | :--- | :--- | :--- |
| **`on_init`** | Uma vez, no início do backtest. | Configuração da estratégia, universo de ativos, metadados. | N/A (apenas inicializa estado interno). | N/A (não está no hot path). |
| **`on_bar`** | Para cada `MarketEvent` recebido pelo motor. | `&MarketState`, `&PortfolioView`. | `Vec<OrderRequest>`. | Alocação de memória, I/O, logging, RNG não-seedado. |
| **`on_session_close`** | No final de uma sessão de negociação (evento gerado pela Normalização). | `&MarketState`, `&PortfolioView`. | `Vec<OrderRequest>`. | Mesmas do `on_bar`. |
| **`on_backtest_end`** | Uma vez, no final do backtest. | Estado final do portfólio, métricas agregadas. | N/A (pode realizar logging ou salvar estado). | N/A (não está no hot path). |

## 4. Modelo de Acesso a Dados: Views e Barreiras

A API fornece acesso aos dados através de "Views" (visões) imutáveis, que são referências `&T` a estruturas de dados internas do motor e do portfólio. Isso evita cópias e garante que a estratégia não possa modificar o estado do sistema.

| Fonte de Dado | Disponível em qual Hook | Limites | Risco de Look-Ahead Evitado |
| :--- | :--- | :--- | :--- |
| **`MarketState`** | `on_bar`, `on_session_close` | Acesso apenas aos dados de mercado até o `timestamp` do evento atual. | Acesso a barras futuras. |
| **`PortfolioView`** | `on_bar`, `on_session_close` | Acesso ao estado do portfólio (posições, NAV) *antes* da execução das ordens geradas no evento atual. | Tomar decisões com base no resultado de uma ordem que ainda não foi executada. |
| **Histórico de Barras** | `on_bar`, `on_session_close` | Acesso a uma janela (`window`) de barras passadas, fornecida de forma eficiente (e.g., como um `&[Bar]`). | Acesso a barras futuras. |

## 5. Interface de Emissão de Ordens

- **Contrato de Emissão**: A estratégia retorna um `Vec<OrderRequest>`, uma estrutura simples que define a intenção de negociação.

| Campo da `OrderRequest` | Definido pela Estratégia | Validações pelo Roteador de Ordens |
| :--- | :--- | :--- |
| `asset_id: AssetId` | Sim | Verifica se o ativo pertence ao universo do backtest. |
| `quantity: i64` | Sim | Verifica se a quantidade é válida (e.g., não é zero). |
| `side: Side` | Sim | (Buy/Sell) |
| `type: OrderType` | Sim | (Market/Limit) |
| `limit_price: Option<f64>` | Sim (para ordens limite) | N/A |

- **Determinismo na Submissão**: Se a estratégia emite múltiplas ordens no mesmo `timestamp`, o Roteador de Ordens as processa em uma ordem estável, definida pelo `AssetId`, para garantir a reprodutibilidade.

## 6. Integração com o Estado do Portfólio

- **Acesso Read-Only**: A estratégia recebe uma `&PortfolioView`, uma visão imutável do estado do portfólio (Módulo 04). Ela pode consultar:
    - Posição atual por `AssetId`.
    - Exposição bruta e líquida.
    - NAV e caixa disponível.
- **Proibições**: A estratégia **não pode** modificar o estado do portfólio diretamente. Ela não tem acesso a `FillEvent`s e não pode interferir no cálculo de PnL ou na atualização de posições.
- **Risk Guards**: A API pode expor funções de validação leves, como `portfolio.can_trade(order)`, que verificam se uma ordem respeita regras de risco simples (e.g., não exceder um limite de exposição), mas a estratégia não pode implementar uma lógica de risco complexa.

## 7. Contrato de Performance

| Violação | Sintoma | Como Detectar (Bench/Profiler) | Ação Corretiva |
| :--- | :--- | :--- | :--- |
| **Alocação no `on_bar`** | Aumento da latência p99, jitter. | `heaptrack` ou similar mostra `alloc` calls dentro do hook. | Refatorar a estratégia para usar buffers pré-alocados no `on_init`. |
| **I/O no `on_bar`** | Latência extremamente alta e variável. | `strace` ou similar mostra syscalls de `read`/`write`. | Mover toda a carga de dados para o `on_init` ou para fora do processo. |
| **Algoritmo Ineficiente** | Alto consumo de CPU no `flamegraph` da estratégia. | Profiler de CPU (`perf`). | Otimizar o algoritmo, usar caches internos (se aplicável e seguro). |

## 8. Contrato de Determinismo

- **RNG**: Proibido. Se a estratégia precisar de aleatoriedade, ela deve usar um gerador (RNG) fornecido pela API, que é inicializado com uma semente global do backtest.
- **Tempo**: A API não expõe nenhuma função para ler o relógio do sistema (`SystemTime::now()` é proibido).
- **Paralelismo**: A estratégia é, por padrão, executada em uma única thread. Se ela usar paralelismo interno (e.g., Rayon), é sua responsabilidade garantir que as agregações sejam feitas de forma determinística.

**Checklist de Determinismo da Estratégia:**
- [ ] A estratégia não usa RNG não-seedado?
- [ ] A estratégia não acessa o relógio do sistema?
- [ ] Todas as operações de ponto flutuante são feitas em uma ordem fixa?
- [ ] A estratégia não depende da ordem de iteração de `HashMap`s ou outros tipos não ordenados?

## 9. Biblioteca de Estratégias: Empacotamento e Versão

- **Definição**: Uma "Biblioteca de Estratégias" é um `crate` Rust separado que implementa o `trait Strategy` e pode ser compilado dinamicamente (como um `.so` ou `.dll`) e carregado pelo motor de backtest.
- **Contrato de Versão**: A API da estratégia é versionada. Uma biblioteca compilada para a `v1.0` da API não será compatível com a `v2.0` sem recompilação.
- **Configuração**: Cada estratégia deve expor uma estrutura de configuração declarativa (e.g., usando `serde`) que permite ao usuário definir seus parâmetros em um arquivo (e.g., TOML).
- **Metadados**: Cada estratégia deve expor metadados como `name`, `version`, e `description`.

## 10. Guidelines de Qualidade e Segurança

- **Logging**: A API fornece um `logger` que só escreve em um buffer em memória durante o hot path. O conteúdo do buffer é despejado no disco apenas no final do backtest.
- **Validação de Parâmetros**: A estratégia é responsável por validar seus próprios parâmetros de configuração no `on_init`.
- **Política de Falha**: Se uma estratégia tentar uma operação proibida (e.g., I/O), a API retornará um `Err`, e o motor de simulação irá parar o backtest com um erro claro, prevenindo comportamento indefinido.

## 11. Harness de Validação de Estratégia

- **Testes Unitários**: O desenvolvedor da estratégia deve escrever testes unitários para sua lógica interna.
- **Testes de Integração**: O projeto fornecerá um `harness` (um mini-motor de backtest) que permite executar uma estratégia em um cenário pequeno e controlado (e.g., 1 ativo, 100 barras) e verificar as ordens geradas.
- **Benchmark de Custo**: O `harness` incluirá benchmarks (Módulo 06) para medir a latência e as alocações de cada hook da estratégia, falhando se os limites de performance forem violados.

## 12. Checklist de Aceite do Módulo

- [ ] O ciclo de vida da estratégia (`on_init`, `on_bar`, etc.) está definido?
- [ ] As barreiras anti-look-ahead via `Views` imutáveis estão especificadas?
- [ ] A interface de emissão de ordens é clara e determinística?
- [ ] Os contratos de performance (zero alocação) e determinismo (sem RNG não-seedado) são explícitos?
- [ ] O conceito de `Strategy Library` como um crate externo está definido?
- [ ] O harness de validação fornece ferramentas para testar a corretude e a performance de uma estratégia?

## 13. Próximo Módulo Sugerido

Este é o último módulo de especificação principal. Os próximos passos seriam a criação de **exemplos de estratégias** e a **documentação do usuário final**.

**`09_example_strategies.md`**

- Fornecerá o código-fonte comentado de 2-3 estratégias de exemplo (e.g., um seguidor de tendência diário, um par-trading intraday).
- Servirá como um tutorial prático de como usar a API de Estratégia.
- Incluirá a configuração TOML e os comandos para executar cada exemplo. 

# Módulo: 06 - Performance e Benchmarking

---

## Sumário

1. [Filosofia de Performance](#1-filosofia-de-performance)
2. [Taxonomia de Benchmarks](#2-taxonomia-de-benchmarks)
3. [Regras para Medição Reprodutível em Linux](#3-regras-para-medicao-reprodutivel-em-linux)
4. [Cenários-Base de Benchmark](#4-cenarios-base-de-benchmark)
5. [Métricas que Importam (e as que Enganam)](#5-metricas-que-importam-e-as-que-enganam)
6. [Playbook de Profiling (Linux)](#6-playbook-de-profiling-linux)
7. [Estratégia de Otimização Segura](#7-estrategia-de-otimizacao-segura)
8. [Plano de Regressão de Performance](#8-plano-de-regressao-de-performance)
9. [Ligação Explícita com os Módulos Anteriores](#9-ligacao-explicita-com-os-modulos-anteriores)
10. [Checklist de Aceite do Módulo](#10-checklist-de-aceite-do-modulo)
11. [Próximo Módulo Sugerido](#11-próximo-modulo-sugerido)

---

## 1. Filosofia de Performance

Nossa abordagem à performance é científica e disciplinada, governada por dois princípios invioláveis. Primeiro, **medir antes de otimizar**. Intuição sobre gargalos em sistemas complexos é notoriamente falha; a otimização prematura gera complexidade sem ganho comprovado. Toda otimização deve ser guiada por dados de profiling. Segundo, **otimizar sem quebrar o determinismo**. A performance nunca pode ser comprada ao custo da reprodutibilidade. Cada mudança que afeta o hot path deve ser validada tanto pelo seu ganho de velocidade quanto pela sua garantia de que o resultado final permanece idêntico (bit-a-bit), conforme o critério AC-03. Esta disciplina garante que o sistema atenda simultaneamente às metas de throughput (AC-01, AC-02) e de corretude.

## 2. Taxonomia de Benchmarks

| Tipo | Objetivo | O que Mede | Quando Usar | Riscos de Medição Mentirosa |
| :--- | :--- | :--- | :--- | :--- |
| **Microbenchmark** | Medir a latência de uma única função ou operador no hot path. | Nanossegundos por operação, alocações, instruções. | Ao otimizar uma função específica (e.g., cálculo de slippage, atualização de PnL). | Medir uma função fora de seu contexto real de cache/branch. O otimizador do compilador pode remover o código se a saída não for usada. |
| **Mesobenchmark** | Medir o throughput e a latência de um componente inteiro. | Eventos/segundo, latência por evento (p50/p99), alocações por evento. | Para validar o contrato de performance de um módulo (e.g., Motor, Execution Model) e identificar qual componente é o gargalo. | A interface do componente pode não ser representativa do uso real no pipeline completo. |
| **Macrobenchmark** | Medir a performance end-to-end do sistema em um cenário realista. | Wall-clock total, throughput geral (eventos/seg), uso de pico de memória. | Para validar os critérios de aceite (AC-01, AC-02) e detectar regressões de performance globais. | Não decompor o tempo total pode esconder a verdadeira causa de uma regressão. Variações no ambiente de teste podem poluir os resultados. |

## 3. Regras para Medição Reprodutível em Linux

- [ ] **Isolamento de CPU**: Executar o benchmark em núcleos de CPU isolados (`isolcpus` no boot do kernel) para evitar interferência do scheduler do SO.
- [ ] **Afinidade de Processo**: Fixar o processo do benchmark em um núcleo específico (`taskset`) para garantir consistência de cache.
- [ ] **Controle de Frequência**: Fixar o `governor` da CPU em `performance` e desabilitar o `turbo boost` para evitar variações de clock.
- [ ] **Estabilidade Térmrica**: Monitorar a temperatura da CPU e garantir que não haja `thermal throttling` durante a execução. Executar um `warmup` antes da medição real.
- [ ] **Repetição Estatística**: Executar o benchmark `N` vezes (e.g., N=10) e analisar a distribuição dos resultados, descartando outliers. Reportar média, desvio padrão e percentis.
- [ ] **Fixação de Inputs**: Cada execução deve usar o mesmo dataset e a mesma configuração, validados por um `hash` combinado, para garantir a comparabilidade.
- [ ] **Pinagem de Versão**: O benchmark deve ser executado em um `commit` específico do Git, com `flags` de compilação (`--release`) e `features` de `cargo` idênticas.
- [ ] **Logging Mínimo**: Desabilitar todo e qualquer logging verboso durante a medição. I/O é um inimigo da performance reprodutível.
- [ ] **Proibição de I/O**: Nenhum benchmark de hot path pode conter I/O de disco ou rede. Os dados devem ser pré-carregados na memória.

## 4. Cenários-Base de Benchmark

| Cenário | Características do Dataset | Componente Dominante | Métricas Chave |
| :--- | :--- | :--- | :--- |
| **Intraday Net Zero** | Alta densidade temporal (e.g., 1-min), universo pequeno (#ativos < 50). | Motor de Simulação, Execution Model (muitos eventos, muitas ordens). | Eventos/seg, latência por evento (p99), fills/seg. |
| **Diário Swing Trade** | Baixa densidade temporal (diário), universo grande (#ativos > 200). | Portfolio/PnL (muitas posições para `mark-to-market` no fim do dia). | Tempo de `mark-to-market` por ativo, uso de pico de memória. |
| **Stress de Universo** | Diário, #ativos > 1000. | Portfolio/PnL, estruturas de dados de estado. | Escalabilidade do tempo de `mark-to-market`, uso de pico de memória. |
| **Stress de Densidade** | Intraday (ticks ou segundos), #ativos pequeno. | Motor de Simulação (throughput de eventos puro). | Eventos/seg (máximo), latência por evento (p50). |

## 5. Métricas que Importam (e as que Enganam)

- **Métricas Primárias**:
    - `Wall-clock end-to-end`: O tempo total do macrobenchmark, a métrica de verdade para o usuário.
    - `Eventos/segundo`: Medida de throughput do motor.
    - `Latência por evento (p50/p99)`: Mede a previsibilidade do processamento.
    - `Alocações por evento`: Deve ser zero no hot path.
    - `Pico de Memória (RSS)`: Mede a eficiência do uso de memória.

- **Métricas Secundárias (para profiling profundo)**:
    - `Instruções por evento`: Mede a eficiência computacional.
    - `Cache Miss Rate`: Indica problemas de localidade de dados.
    - `Branch Miss Rate`: Indica problemas de previsibilidade de código.

- **Métricas que Enganam**:
    - *Throughput sem determinismo*: Um ganho de velocidade que muda o resultado final é uma falha, não uma otimização.
    - *Speedup com dataset não equivalente*: Comparar a performance em cenários diferentes é inválido.
    - *Comparar `debug` com `release`*: Builds de `debug` são ordens de magnitude mais lentos e não devem ser usados para benchmarking.

## 6. Playbook de Profiling (Linux)

1.  **Medir Macrobenchmark**: Execute o cenário-base relevante (Seção 4) e meça o tempo `end-to-end`. Se estiver fora da meta (AC-01/AC-02), decomponha o tempo por estágio (Ingestão, Simulação, Relatórios).
2.  **Isolar Hotspots de CPU**: Use um profiler de CPU (e.g., `perf`, `flamegraph`) no estágio de Simulação para identificar as funções que consomem a maior parte do tempo. O alvo são as funções no topo do *flame graph*.
3.  **Validar Alocação de Memória**: Use um profiler de memória (e.g., `heaptrack`) para garantir que não há alocações no loop principal. Se houver, este é o primeiro ponto a ser corrigido.
4.  **Investigar Cache/Branch (se necessário)**: Se o código da função hotspot for computacionalmente simples, mas ainda lento, use `perf stat` para investigar `cache-misses` e `branch-misses`, indicando problemas de layout de dados ou de código.
5.  **Validar com A/B Controlado**: Após aplicar uma otimização, execute o mesmo benchmark novamente sob as mesmas condições (Seção 3) e compare os resultados estatisticamente para provar o ganho e garantir que o hash de resultado (determinismo) não mudou.

## 7. Estratégia de Otimização Segura

| Etapa | Sinais de que Vale a Pena | Riscos | Como Provar Ganho | Como Provar Determinismo |
| :--- | :--- | :--- | :--- | :--- |
| **1. Eliminar Alocações** | Profiler de memória mostra alocações no hot path. | Pode aumentar a complexidade do código (pré-alocação manual). | Microbenchmark da função antes/depois mostra zero alocações. | Re-executar o macrobenchmark e verificar se o hash do resultado é idêntico. |
| **2. Melhorar Layout de Dados** | Profiler de cache mostra alta taxa de `cache-misses`. | Requer refatoração significativa (e.g., de AoS para SoA). | Microbenchmark de acesso a dados mostra latência menor. | Hash do resultado idêntico. |
| **3. Reduzir Overhead de Abstração** | Profiler de CPU aponta para `dispatch` de `dyn Trait`. | Perda de flexibilidade. | Microbenchmark mostra latência menor após monomorfização. | Hash do resultado idêntico. |
| **4. Batching** | Alto overhead de chamadas de função por evento. | Pode aumentar a complexidade da lógica de estado. | Mesobenchmark mostra maior throughput (eventos/seg). | Hash do resultado idêntico. |
| **5. Paralelismo** | Tarefas inerentemente independentes (e.g., cálculo de indicadores por ativo). | Risco altíssimo de introduzir não-determinismo. | Macrobenchmark mostra menor wall-clock em máquina multi-core. | O merge/redução dos resultados paralelos deve ser feito em ordem estritamente definida para garantir o hash idêntico. |

## 8. Plano de Regressão de Performance

- **Baseline de Referência**: Um conjunto de resultados de benchmark (para todos os cenários da Seção 4) é gerado em uma máquina de referência e armazenado no repositório. Cada baseline é assinado com o `commit hash`, `dataset hash`, `config hash` e perfil da máquina.
- **Tolerâncias**: Uma regressão é sinalizada se o tempo de execução de um benchmark exceder a `média + 3 * desvio_padrão` da baseline. Esta é a política inicial.
- **Armazenamento de Resultados**: Os resultados dos benchmarks são salvos em um formato estruturado (e.g., JSON), contendo as métricas primárias e a assinatura da execução, permitindo a comparação programática.
- **Gatilhos para Investigação**: Uma investigação é obrigatória se:
    - O tempo de execução violar a tolerância.
    - O número de alocações no hot path se tornar `> 0`.
    - A latência p99 aumentar significativamente.
- **Orçamento de Performance (Conceitual)**: Cada componente do hot path tem um "orçamento" de tempo de execução. Uma mudança que melhora um componente, mas piora outro, só é aceita se o resultado líquido for positivo.

## 9. Ligação Explícita com os Módulos Anteriores

- **M02 (Dados/Ingestão)**: Medir o tempo do pipeline de ingestão/normalização separadamente. Validar que o throughput de I/O é adequado.
- **M03 (Motor)**: O `eventos/segundo` e a `latência por evento` são as métricas diretas da eficiência do motor.
- **M04 (Portfólio)**: Medir a latência da função de atualização por `FillEvent` e da função de `mark-to-market`.
- **M05 (Execution Model)**: Medir a latência da função de `execute` para cada tipo de ordem e modelo de slippage.

## 10. Checklist de Aceite do Módulo

- [ ] A filosofia de "medir antes de otimizar" e "não quebrar o determinismo" está estabelecida?
- [ ] A taxonomia de benchmarks (micro, meso, macro) está clara?
- [ ] O checklist para medição reprodutível em Linux está completo e é prático?
- [ ] Os cenários-base de benchmark cobrem o escopo do sistema (intraday, diário, stress)?
- [ ] As métricas primárias e secundárias estão definidas, e as métricas que enganam foram identificadas?
- [ ] O playbook de profiling fornece um roteiro claro para identificar gargalos?
- [ ] A estratégia de otimização segura está definida em etapas, com foco em preservar o determinismo?
- [ ] O plano de regressão de performance (baseline, tolerâncias, gatilhos) está especificado?
- [ ] A ligação entre as métricas de performance e os módulos anteriores está explícita?

## 11. Próximo Módulo Sugerido

**`07_development_roadmap.md`**

- Consolidará todos os módulos anteriores em um plano de implementação incremental, do MVP à versão 1.0.
- Definirá os épicos e as estórias de usuário para cada etapa do desenvolvimento, com critérios de aceite claros.
- Estabelecerá a estratégia de testes (unitários, de integração, end-to-end) e o processo de CI/CD para garantir a qualidade e a estabilidade do sistema.

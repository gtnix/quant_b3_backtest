# Módulo: 07 - Roadmap de Desenvolvimento

---

## Sumário

1. [Princípios de Execução do Projeto](#1-principios-de-execucao-do-projeto)
2. [Fases e Milestones (MVP → V1)](#2-fases-e-milestones-mvp--v1)
3. [Backlog em Épicos](#3-backlog-em-epicos)
4. [Plano de Testes](#4-plano-de-testes)
5. [Plano de Benchmark Contínuo](#5-plano-de-benchmark-continuo)
6. [Estratégia de Gestão de Complexidade](#6-estrategia-de-gestao-de-complexidade)
7. [Matriz de Riscos Técnicos e Mitigações](#7-matriz-de-riscos-tecnicos-e-mitigacoes)
8. [Checklists Operacionais](#8-checklists-operacionais)
9. [Próximos Passos (Pós-V1)](#9-proximos-passos-pos-v1)

---

## 1. Princípios de Execução do Projeto

- **Determinismo-First**: Nenhuma feature ou otimização é aprovada se comprometer a reprodutibilidade bit-a-bit (AC-03).
- **Performance-First**: O design de cada componente do hot path deve priorizar a performance, conforme os contratos dos Módulos 03-06.
- **O Hot Path é Sagrado**: Nenhuma alocação de memória, I/O ou lógica complexa é permitida no loop de simulação principal.
- **Medir Antes de Otimizar**: Toda otimização deve ser precedida por um benchmark que identifique um gargalo real (Módulo 06).
- **Escopo Fechado é Contrato**: O escopo definido no Módulo 00 é inviolável. Novas ideias são registradas para V2, não implementadas agora.
- **Validação Contínua**: Cada commit deve passar por suites de testes de determinismo, invariantes e regressão de performance.
- **Zero Repetição, Máxima Referência**: A documentação (`/docs`) é a fonte da verdade. O código deve seguir a documentação; a documentação não deve descrever o código.

## 2. Fases e Milestones (MVP → V1)

| Fase | Objetivo | Entregáveis Principais | Gate de Saída (O que deve ser verdade para avançar) |
| :--- | :--- | :--- | :--- |
| **Fase 0: Foundation** | Estabelecer o esqueleto do projeto e os contratos de interface. | - Estrutura de crates em Rust (M01).\n- Pipeline de CI/CD local configurado.\n- Suite de testes de determinismo (vazia). | O projeto compila e o pipeline de testes (vazio) passa. |
| **Fase 1: Data Pipeline** | Implementar o pipeline de ingestão e normalização de dados. | - Módulo de ingestão para CSV/Parquet (M02).\n- Geração de fluxo de `MarketEvent`s ordenado e determinístico. | A suite de testes de determinismo passa para o mesmo dataset de entrada. O hash do fluxo de eventos é reprodutível. |
| **Fase 2: Engine Core** | Implementar o motor de simulação e garantir o determinismo do loop. | - Motor de eventos que consome o fluxo da Fase 1 (M03).\n- Implementação das barreiras anti-look-ahead. | O motor processa o fluxo de eventos completo. A suite de testes de determinismo (AC-03) passa para uma simulação sem ordens. |
| **Fase 3: Portfolio & PnL** | Implementar a lógica de estado da carteira. | - Módulo de Portfólio (M04) que responde a `FillEvent`s (mockados).\n- Cálculo de PnL e Drawdown. | A suite de testes de invariantes de portfólio (e.g., NAV = caixa + valor) passa. |
| **Fase 4: Execution Model** | Implementar a simulação de execução enviesada. | - Módulo de Execução (M05) com modelos de custo, slippage e latência.\n- Geração de `FillEvent`s a partir de `OrderEvent`s. | A suite de testes do modelo de execução (invariantes, determinismo) passa. O sistema executa um backtest end-to-end. |
| **Fase 5: Performance** | Atingir as metas de performance e estabelecer baselines. | - Execução dos cenários de benchmark (M06).\n- Otimizações guiadas por profiling.\n- Baseline de performance estabelecida. | Os critérios de aceite AC-01 e AC-02 são atendidos. |
| **Fase 6: V1 Hardening** | Estabilizar, documentar e preparar para uso interno. | - Documentação de API finalizada.\n- Cobertura de testes > 90% para o hot path.\n- Manual de uso para execução de backtests. | Todos os checklists de aceite dos módulos 02-06 estão 100% concluídos. |

## 3. Backlog em Épicos

| Épico | Descrição Objetiva | Dependências | Definition of Done (DoD) Verificável |
| :--- | :--- | :--- | :--- |
| **Épico 1: Data Pipeline (M02)** | Implementar o pipeline que lê, normaliza e ordena dados de mercado em um fluxo de eventos determinístico. | Fase 0 | O sistema gera um fluxo de `MarketEvent`s a partir de arquivos CSV/Parquet cujo hash é reprodutível. |
| **Épico 2: Simulation Engine (M03)** | Implementar o loop de eventos que processa o fluxo de dados, invoca a estratégia e garante zero look-ahead. | Épico 1 | O motor executa um backtest completo (sem ordens) e passa na suite de testes de determinismo. |
| **Épico 3: Portfolio State (M04)** | Implementar o módulo de gestão de estado da carteira, incluindo posições, caixa, PnL e drawdown. | Épico 2 | O módulo atualiza o estado corretamente a partir de `FillEvent`s mockados e passa na suite de invariantes. |
| **Épico 4: Execution Simulation (M05)** | Implementar os modelos de custo, slippage e latência que transformam `OrderEvent`s em `FillEvent`s. | Épico 3 | O sistema executa um backtest end-to-end com transações, e os resultados são determinísticos. |
| **Épico 5: Benchmarking & Performance (M06)** | Implementar o harness de benchmarking, executar os cenários-base e otimizar para atingir as metas de performance. | Épico 4 | Os critérios AC-01 e AC-02 são atendidos e a baseline de performance é registrada. |

## 4. Plano de Testes

- **Testes Unitários**: Testam uma única função ou struct em isolamento (e.g., um modelo de slippage, uma função de cálculo de PnL). Devem ser extremamente rápidos.
- **Testes de Integração**: Testam a interação entre dois ou mais componentes (e.g., Motor + Portfólio). Usam dados mockados e focam nos contratos de interface.
- **Testes End-to-End (E2E)**: Executam um macrobenchmark completo com um dataset pequeno e validam o hash do resultado final. São a principal ferramenta para garantir o determinismo global.

**Suites Obrigatórias (Gates de CI):**
- `test-suite-determinism`: Executa cada cenário E2E duas vezes e falha se os hashes de resultado não forem idênticos.
- `test-suite-invariants`: Executa testes de integração que tentam violar os invariantes do sistema (e.g., criar um `FillEvent` que torna o caixa negativo).
- `test-suite-anti-look-ahead`: Executa cenários E2E com padrões de dados que revelariam look-ahead. Falha se a estratégia tomar uma decisão com base em dados futuros.

**Gate de Merge**: Nenhuma Pull Request é aprovada se não passar em 100% das suites de testes.

## 5. Plano de Benchmark Contínuo

- **Seleção de Cenários**: Para cada mudança no hot path, o desenvolvedor deve executar o cenário de benchmark (Módulo 06, Seção 4) mais relevante para a mudança.
- **Versionamento de Baselines**: Os resultados da baseline de performance são armazenados em arquivos JSON no repositório, em uma pasta `/benches/results`, e são versionados com Git-LFS.
- **Protocolo de Regressão**: Um merge é **bloqueado** se a mudança causar uma regressão de performance estatisticamente significativa (conforme definido no Módulo 06) no cenário de benchmark relevante, a menos que a regressão seja justificada e aprovada por dois arquitetos.
- **Cadência**: A suite completa de benchmarks é executada em cada release de versão (e.g., MVP, V1), enquanto benchmarks específicos são executados a cada mudança no hot path.

## 6. Estratégia de Gestão de Complexidade

- **Limites de Abstração**: O uso de `dyn Trait` (dispatch dinâmico) é proibido no hot path. A monomorfização via genéricos é a única forma de abstração permitida.
- **Regras de Dependência**: O grafo de dependências definido no Módulo 01 é mandatório. Uma dependência cíclica ou uma dependência que viole as fronteiras (e.g., `Strategy` dependendo de `IO`) é um erro de compilação.
- **Quarentena do Slow Path**: Qualquer funcionalidade que envolva I/O, logging pesado ou alocação de memória (e.g., geração de relatórios detalhados) deve viver em um crate separado e ser executada apenas após o término do loop de simulação.

## 7. Matriz de Riscos Técnicos e Mitigações

| Risco | Impacto | Sinais Precoces | Mitigação | Dono |
| :--- | :--- | :--- | :--- | :--- |
| **Regressão de Determinismo** | Alto | Falha na `test-suite-determinism`. | Gate de CI que bloqueia o merge. | Engine Lead |
| **Viés de Look-Ahead** | Crítico | Falha na `test-suite-anti-look-ahead`. | Revisão de código focada nas barreiras de informação (M03). | Strategy/Engine Lead |
| **Regressão de Performance** | Alto | Aumento no tempo de execução do benchmark. | Protocolo de Regressão (Seção 5) que bloqueia o merge. | Performance Lead |
| **Explosão de Alocação** | Médio | Profiler de memória (`heaptrack`) mostra alocações no loop. | Gate de CI que falha se `allocations > 0` no hot path. | Engine Lead |
| **Inconsistência de Estado** | Crítico | Falha na `test-suite-invariants`. | Testes de integração rigorosos para o Módulo de Portfólio. | Portfolio Lead |

## 8. Checklists Operacionais

**Checklist de Início de Sprint:**
- [ ] O backlog do sprint contém apenas épicos definidos no Módulo 07?
- [ ] Cada épico tem uma DoD clara e critérios de aceite mensuráveis?
- [ ] As dependências entre os épicos do sprint estão resolvidas?
- [ ] A baseline de performance e determinismo está atualizada?

**Checklist Antes de Otimização de Hot Path:**
- [ ] Um benchmark foi executado e um gargalo foi identificado e documentado?
- [ ] A otimização proposta tem uma hipótese clara de ganho?
- [ ] Um microbenchmark foi criado para medir o impacto da mudança isoladamente?
- [ ] O plano de validação inclui a verificação do hash de determinismo?

**Checklist Antes de Release Interno:**
- [ ] Todas as suites de testes (unit, integration, E2E) passam?
- [ ] A suite completa de benchmarks foi executada e não há regressões inesperadas?
- [ ] A documentação (`/docs`) foi atualizada para refletir as mudanças da release?
- [ ] Os critérios de aceite (AC-01, AC-02, AC-03) foram formalmente verificados e documentados?

## 9. Próximos Passos (Pós-V1)

**`08_strategy_library_and_api.md`**

- Definirá a API pública para que usuários possam escrever e integrar suas próprias estratégias no backtester.
- Especificará o formato e a estrutura de uma "biblioteca de estratégias" externa.
- Documentará o ciclo de vida de uma estratégia e os hooks disponíveis (e.g., `on_bar`, `on_session_close`).

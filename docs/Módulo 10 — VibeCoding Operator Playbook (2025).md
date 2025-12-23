# Módulo 10 — VibeCoding Operator Playbook (2025)

---

## Sumário

1. [Operating Model (Vibe Coding 2025)](#1-operating-model-vibe-coding-2025)
2. [Golden Rules (Não Negociáveis)](#2-golden-rules-nao-negociaveis)
3. [Agent Loops Operacionais](#3-agent-loops-operacionais)
4. [Prompt Pack (Cursor/Agent)](#4-prompt-pack-cursoragent)
5. [Gates & Acceptance Criteria](#5-gates--acceptance-criteria)
6. [Matriz de Sinais e Diagnósticos](#6-matriz-de-sinais-e-diagnosticos)
7. [Checklist de Fim de Ciclo](#7-checklist-de-fim-de-ciclo)
8. [Próximos Passos](#8-proximos-passos)

---

## 1. Operating Model (Vibe Coding 2025)

Neste projeto, não operamos o sistema manualmente; nós o comandamos. Nosso trabalho não é ler ou escrever código Rust, mas sim pilotar um agente de IA (via Cursor ou similar) para construir, validar e executar o backtester. A interação é um diálogo contínuo: damos prompts, o agente gera código e artefatos, e nós validamos esses artefatos contra "gates" rigorosos. A "verdade" não está no código, que é efêmero, mas sim nos outputs imutáveis: hashes de resultados, métricas de performance, logs de benchmark e a passagem em suites de testes de invariantes. Nosso papel é de um arquiteto e operador de sistema, usando linguagem natural para guiar a IA e dados para verificar o trabalho.

## 2. Golden Rules (Não Negociáveis)

| Regra | Consequência Prática | Como Detectar Violação (sem ler código) |
| :--- | :--- | :--- |
| **Determinismo é Contrato (AC-03)** | Duas execuções com o mesmo input/config devem produzir resultados idênticos, bit-a-bit. | Falha no "Loop de Determinismo": os hashes de output de duas execuções idênticas não batem. |
| **Anti-Look-Ahead é Inviolável** | A estratégia não pode usar informação do futuro para tomar decisões. | Sintomas de PnL "bom demais". Falha no "Loop de Anti-Look-Ahead" com datasets sintéticos. |
| **Hot Path é Sagrado** | Nenhuma alocação de memória, I/O ou logging verboso é permitido no loop de eventos principal. | O "Loop de Performance" reporta `allocations > 0` ou latência p99 explode. `strace` no processo mostra syscalls de I/O. |
| **Escopo é Fixo** | Nenhuma feature fora dos módulos 00-09 é implementada (sem book replay, etc.). | Revisão dos outputs e configurações. Prompts devem proibir explicitamente a expansão de escopo. |
| **Tudo é Medido** | Nenhuma otimização é aceita sem um benchmark que prove o ganho e valide a não-regressão. | Falta de um artefato de benchmark no output do agente para uma tarefa de otimização. |

## 3. Agent Loops Operacionais

**1. Loop de Build/Run**
- **Objetivo**: Executar um backtest end-to-end com uma configuração específica.
- **Entrada**: Um arquivo de configuração `.toml`.
- **Saída**: Um diretório de output com `results.json` e `run_log.txt`.
- **Gate**: O run completa sem erro. O `results.json` contém as métricas esperadas.

**2. Loop de Determinismo (AC-03)**
- **Objetivo**: Provar que uma execução é reprodutível.
- **Entrada**: Um arquivo de configuração `.toml`.
- **Saída**: Dois diretórios de output de duas execuções idênticas, cada um com um `results.hash`.
- **Gate**: Os dois arquivos `results.hash` são idênticos (`diff <hash1> <hash2>` retorna vazio).

**3. Loop de Performance**
- **Objetivo**: Medir a performance de um cenário e identificar gargalos.
- **Entrada**: Um arquivo de configuração `.toml` e um cenário-base do Módulo 06.
- **Saída**: Um `benchmark.json` com métricas primárias (wall-clock, eventos/seg, alocações) e um `flamegraph.svg`.
- **Gate**: As métricas são consistentes com a baseline. O `flamegraph` não mostra anomalias óbvias (e.g., I/O no hot path).

**4. Loop de Regressão**
- **Objetivo**: Garantir que uma mudança não degradou a performance.
- **Entrada**: O PR/commit da mudança e um cenário de benchmark relevante.
- **Saída**: Um `comparison.md` mostrando a diferença percentual de performance em relação à baseline.
- **Gate**: A diferença está dentro da tolerância aceitável (e.g., < 2%).

**5. Loop de Anti-Look-Ahead**
- **Objetivo**: Validar que a estratégia não está "trapaceando".
- **Entrada**: Uma estratégia e um dataset sintético projetado para expor look-ahead.
- **Saída**: O `results.json` do backtest.
- **Gate**: O PnL é zero ou negativo, provando que a estratégia não se beneficiou da informação futura.

**6. Loop de Estratégia**
- **Objetivo**: Integrar e validar uma nova estratégia da biblioteca.
- **Entrada**: O nome da estratégia e seus parâmetros no `.toml`.
- **Saída**: Execução bem-sucedida do Loop de Build/Run, Determinismo e Performance.
- **Gate**: A nova estratégia não viola nenhum dos gates dos loops anteriores.

## 4. Prompt Pack (Cursor/Agent)

**Run Control**
> "Usando o Módulo 10 como playbook, execute um backtest end-to-end com a configuração em `configs/daily_trend.toml`. Gere o diretório de output padrão. Proíba qualquer modificação no código-fonte."

**Determinism Proof**
> "Prove o determinismo (AC-03) para a configuração `configs/intraday_mr.toml`. Execute o Loop de Determinismo do Módulo 10 e forneça os dois hashes de resultado como prova. Falhe se não forem idênticos."

**Perf Macro/Meso/Micro**
> "Execute o Loop de Performance para o cenário 'Intraday Net Zero' (Módulo 06) com a config `configs/intraday_mr.toml`. Entregue o `benchmark.json` e o `flamegraph.svg`. Valide que as alocações no hot path são zero."

**Regression Triage**
> "Uma mudança no PR #123 pode ter causado regressão. Execute o Loop de Regressão para o cenário 'Diário Swing Trade' e entregue o `comparison.md` contra a baseline mais recente."

**Data Sanity (M02)**
> "Valide o dataset em `data/raw/new_data.csv` contra as regras do Módulo 02. Reporte qualquer violação de ordenação, timestamp ou consistência de OHLC. Não normalize ainda, apenas valide."

**Strategy Cost Budget (M08/M06)**
> "Meça o custo computacional da estratégia 'PairsSpread' (Módulo 09). Execute um mesobenchmark focado no hook `on_bar` e reporte a latência p99 e as alocações por evento."

## 5. Gates & Acceptance Criteria

| Artefato | Gate de Aceitação (Critério Objetivo) |
| :--- | :--- |
| **`results.json`** | Contém todas as métricas chave (PnL, drawdown, etc.) e valores são numericamente plausíveis. |
| **`results.hash`** | É idêntico ao hash de uma execução de controle com os mesmos inputs. |
| **`benchmark.json`** | `allocations_per_event` no hot path é zero. `events_per_second` está dentro da tolerância da baseline. |
| **`flamegraph.svg`** | Não mostra syscalls de I/O (`read`, `write`) ou `alloc` dentro das funções do hot path. |
| **`comparison.md`** | A regressão de performance (se houver) está documentada e justificada. |

## 6. Matriz de Sinais e Diagnósticos

| Sinal (Observação) | Diagnóstico Provável | Próximo Prompt (para o agente) |
| :--- | :--- | :--- |
| PnL parece alto demais. | Viés de look-ahead. | "Execute o Loop de Anti-Look-Ahead para esta estratégia com o dataset sintético `lookahead_trap.csv`." |
| Resultados variam entre execuções. | Não-determinismo. | "Execute o Loop de Determinismo e reporte a diferença entre os outputs." |
| Execução muito lenta. | Build em modo debug ou alocação/I/O no hot path. | "Confirme que o build foi `--release`. Execute o Loop de Performance e me mostre o `flamegraph` e o `benchmark.json`." |
| Erro de parsing de dados. | Inconsistência no arquivo de entrada. | "Execute o prompt de 'Data Sanity' para o arquivo de dados problemático." |

## 7. Checklist de Fim de Ciclo

Antes de considerar uma tarefa (e.g., implementação de uma feature, otimização) como "concluída", verifique:

- [ ] **Determinismo Provado**: O Loop de Determinismo passou para o cenário principal afetado?
- [ ] **Performance Validada**: O Loop de Performance foi executado e não há anomalias?
- [ ] **Regressão Checada**: O Loop de Regressão passou (ou a regressão foi justificada)?
- [ ] **Gates de Artefatos Cumpridos**: Todos os artefatos gerados (hashes, benchmarks) foram validados contra seus gates?
- [ ] **Escopo Respeitado**: A mudança não introduziu nenhuma feature fora do escopo definido nos Módulos 00-09?

## 8. Próximos Passos

Este playbook encerra a documentação de engenharia (`/docs`). O sistema está definido, e o modelo operacional para iterar nele está claro. Os próximos passos são puramente operacionais, seguindo os loops definidos aqui.

**Melhorias futuras neste playbook poderiam incluir:**
- Um "Loop de Onboarding de Ativo" para validar e normalizar novos datasets de forma automatizada.
- Um "Loop de Calibração de Execução" para ajustar os parâmetros do Módulo 05 com base em dados de mercado reais.
- Um "Prompt Pack Avançado" para tarefas de refatoração de larga escala guiadas por IA. 

# Módulo 11 — Contratos de Artefatos e Manifesto de Execução

---

## Sumário

1. [Por que Artefatos são a "Fonte da Verdade"](#1-por-que-artefatos-sao-a-fonte-da-verdade)
2. [Taxonomia de Execuções (Run Taxonomy)](#2-taxonomia-de-execucoes-run-taxonomy)
3. [Run ID, Naming e Layout de Diretórios](#3-run-id-naming-e-layout-de-diretorios)
4. [O Manifesto de Execução (Run Manifest)](#4-o-manifesto-de-execucao-run-manifest)
5. [O Índice de Artefatos (Artifact Index)](#5-o-indice-de-artefatos-artifact-index)
6. [Contratos de Artefatos](#6-contratos-de-artefatos)
7. [Hashing & Assinaturas Digitais](#7-hashing--assinaturas-digitais)
8. [Baselines: Versionamento e Retenção](#8-baselines-versionamento-e-retencao)
9. [Gates por Loop (Mapeamento para Módulo 10)](#9-gates-por-loop-mapeamento-para-modulo-10)
10. [Modos de Falha do Operador](#10-modos-de-falha-do-operador)
11. [Checklist "Ready-to-Audit"](#11-checklist-ready-to-audit)
12. [Encerramento da Pasta `/docs`](#12-encerramento-da-pasta-docs)

---

## 1. Por que Artefatos são a "Fonte da Verdade"

No modelo VibeCoding (Módulo 10), o código-fonte é transitório, gerado e modificado por um agente de IA. Confiar na leitura de código para validação é impraticável e ineficiente. A nossa "fonte da verdade" são os **artefatos**: os arquivos de output gerados por cada execução. Este módulo define os contratos formais para esses artefatos. Sem contratos rigorosos, os loops e gates do Módulo 10 perdem o significado, e a operação se degrada para "feeling". Com eles, mantemos uma disciplina de auditoria de nível de fundo quantitativo, mesmo sem inspecionar uma única linha de Rust.

## 2. Taxonomia de Execuções (Run Taxonomy)

| Categoria da Execução | Propósito | Entradas Principais | Saídas Obrigatórias | Erro Típico |
| :--- | :--- | :--- | :--- | :--- |
| `build_run` | Executar um backtest end-to-end. | `config.toml` | `run_manifest.json`, `results.json`, `run_log.txt` | Erro de configuração ou de dados. |
| `determinism_proof` | Provar que uma execução é reprodutível. | `config.toml` | Dois `run_manifest.json` de duas execuções, cada um com um `results.hash`. | Hashes de resultado não são idênticos. |
| `perf_benchmark` | Medir a performance de um cenário. | `config.toml`, Cenário-base (M06) | `run_manifest.json`, `benchmark.json`, `flamegraph.svg` | Medição poluída por I/O ou build em modo debug. |
| `regression_compare` | Comparar a performance com uma baseline. | `config.toml`, Baseline de referência | `run_manifest.json`, `comparison.md` | Comparação inválida (datasets ou configs diferentes). |
| `lookahead_validation` | Validar que uma estratégia não tem viés de look-ahead. | `config.toml`, Dataset sintético | `run_manifest.json`, `results.json` | A estratégia gera PnL positivo, indicando "trapaça". |
| `strategy_cost_budget` | Medir o custo de uma estratégia no hot path. | `config.toml` (com a estratégia) | `run_manifest.json`, `benchmark.json` (focado no hook `on_bar`). | A estratégia mostra alocações ou latência p99 alta. |

## 3. Run ID, Naming e Layout de Diretórios

- **Padrão do Run ID**: `<timestamp_utc>_<run_type>_<strategy_name>_<dataset_hash_short>_<commit_hash_short>`
- **Layout**: `/outputs/<run_id>/`

| Elemento | Regra | Motivação | Falha que Previne |
| :--- | :--- | :--- | :--- |
| `timestamp_utc` | `YYYYMMDD-HHMMSS-micros` | Ordenação cronológica e unicidade. | Ambiguidade sobre qual execução veio primeiro. |
| `run_type` | Categoria da Seção 2 (e.g., `perf_benchmark`). | Clareza imediata sobre o propósito da execução. | Confundir um run de performance com um run de determinismo. |
| `strategy_name` | Nome da estratégia em uso. | Rastreabilidade da lógica de negócio. | Não saber qual estratégia gerou qual resultado. |
| `dataset_hash_short` | 8 primeiros caracteres do hash do dataset. | Rastreabilidade dos dados de entrada. | Comparar resultados de datasets diferentes achando que são iguais. |
| `commit_hash_short` | 8 primeiros caracteres do hash do commit Git. | Rastreabilidade da versão do código. | Não saber qual versão do código gerou um resultado específico. |

## 4. O Manifesto de Execução (Run Manifest)

Cada execução **deve** gerar um arquivo `run_manifest.json` no seu diretório de output. Este é o contrato central.

| Campo | Tipo | Invariante |
| :--- | :--- | :--- |
| `run_id` | `string` | Deve corresponder ao nome do diretório. |
| `run_type` | `string` | Deve ser um dos tipos da Taxonomia (Seção 2). |
| `created_at_utc` | `string` (ISO 8601) | Timestamp de início da execução. |
| `git_commit` | `string` | Hash completo do commit Git. |
| `build_profile` | `string` | `"release"` ou `"debug"`. |
| `dataset_signature` | `string` | Hash completo do arquivo de dataset. |
| `config_signature` | `string` | Hash completo do arquivo de configuração. |
| `strategy_id` | `string` | Nome e versão da estratégia (e.g., `DailyTrendFollow:1.0.0`). |
| `machine_fingerprint` | `string` | String descritiva do ambiente (e.g., `Linux-5.15.0-generic-x86_64-16-cores`). |
| `artifact_index` | `object` | Um mapa para o Índice de Artefatos (Seção 5). |

## 5. O Índice de Artefatos (Artifact Index)

Esta é uma seção dentro do `run_manifest.json` que cataloga todos os outros arquivos gerados.

| Campo (por artefato) | Tipo | Descrição |
| :--- | :--- | :--- |
| `path` | `string` | Caminho relativo do arquivo a partir da raiz do run. |
| `checksum` | `string` | Hash SHA256 do conteúdo do arquivo. |
| `gate_status` | `string` | `"PASS"`, `"FAIL"`, ou `"NOT_APPLICABLE"`. |

## 6. Contratos de Artefatos

**`results.json`**
- **Propósito**: Conter as métricas de resultado final do backtest.
- **Campos Mínimos**: `final_nav`, `max_drawdown`, `total_costs`, `sharpe_ratio`, `run_result_hash`.
- **Gate**: O arquivo é um JSON válido e contém todos os campos mínimos.

**`results.hash`**
- **Propósito**: Uma assinatura única e determinística do resultado do backtest para comparação.
- **Derivação**: Hash SHA256 de uma string canônica contendo os principais resultados (e.g., série temporal do NAV, lista de trades), com floats normalizados e campos ordenados alfabeticamente.
- **Gate**: O hash é idêntico ao de uma execução de controle.

**`benchmark.json`**
- **Propósito**: Conter as métricas de performance primárias.
- **Campos Mínimos**: `wall_clock_seconds`, `events_per_second`, `hot_path_allocations`, `p99_latency_ns`.
- **Gate**: `hot_path_allocations` deve ser `0`.

**`comparison.md`**
- **Propósito**: Relatório de comparação de regressão de performance.
- **Campos Mínimos**: `baseline_run_id`, `candidate_run_id`, tabela de comparação com variação percentual para métricas primárias, decisão (`PASS`/`FAIL`).
- **Gate**: A variação de performance está dentro da política de tolerância.

**`flamegraph.svg`**
- **Propósito**: Visualização de profiling de CPU.
- **Gate**: Inspeção visual não mostra `alloc`, `I/O syscalls` (e.g., `read`, `write`) ou `lock` nas funções do hot path.

## 7. Hashing & Assinaturas Digitais

- **Regra de Hash de Resultado**: Para gerar o `results.hash`, a série temporal do NAV e a lista de todas as transações são serializadas para uma string JSON canônica (chaves ordenadas alfabeticamente, sem espaços em branco, floats formatados com 8 casas decimais) antes de aplicar o SHA256.
- **Comparabilidade**: Dois runs só são comparáveis se seus `run_manifest.json` mostrarem os mesmos `git_commit`, `build_profile`, `dataset_signature` e `config_signature`.
- **Quando um Hash Diferente é Bug**: Se dois runs são comparáveis (mesmas assinaturas de input), um `results.hash` diferente é, por definição, uma falha no critério de determinismo (AC-03).

## 8. Baselines: Versionamento e Retenção

- **Registry**: Um diretório `/benches/baselines` no repositório armazena os `benchmark.json` de referência para cada cenário-base do Módulo 06.
- **Promoção a Baseline**: Um resultado de benchmark só pode se tornar a nova baseline se passar por um gate duplo: (1) passar no Loop de Determinismo e (2) a mudança de performance for justificada e aprovada por um arquiteto.
- **Retenção**: Guardar todas as baselines aprovadas. Guardar os `run_manifest.json` de todos os runs de validação (determinismo, regressão) como evidência de auditoria.

## 9. Gates por Loop (Mapeamento para Módulo 10)

| Loop (M10) | Artefatos Obrigatórios | Gates Mínimos | Decisão (FAIL se...) |
| :--- | :--- | :--- | :--- |
| `build_run` | `run_manifest.json`, `results.json` | O run completa sem erro. | O processo falha. |
| `determinism_proof` | 2x `run_manifest.json` com `results.hash` | Os dois `results.hash` são idênticos. | Os hashes divergem. |
| `perf_benchmark` | `benchmark.json`, `flamegraph.svg` | `hot_path_allocations == 0`. | Há alocações no hot path. |
| `regression_compare` | `comparison.md` | A variação de performance está dentro da tolerância. | A regressão excede a política. |

## 10. Modos de Falha do Operador

| Erro do Operador | Dano | Correção Operacional |
| :--- | :--- | :--- |
| Comparar runs com configs diferentes. | Conclusão de performance ou resultado inválida. | Sempre valide que as assinaturas no `run_manifest.json` são idênticas antes de comparar. |
| Medir performance em modo `debug`. | Métricas de performance inúteis e enganosas. | Sempre use o `build_profile: "release"`. O gate deve falhar se o perfil for `debug`. |
| Aceitar benchmark sem warmup. | Resultados poluídos por caches frios. | O protocolo de benchmark deve incluir uma fase de warmup. Valide no `run_log.txt`. |
| Permitir logs no hot path. | Destruição da performance e do determinismo. | O gate do `flamegraph.svg` deve detectar I/O. O prompt deve proibir logs no hot path. |

## 11. Checklist "Ready-to-Audit"

- [ ] O `run_manifest.json` está presente e completo?
- [ ] Todos os artefatos listados no `artifact_index` existem e seus checksums batem?
- [ ] Para um `determinism_proof`, os hashes de resultado são idênticos?
- [ ] Para um `perf_benchmark`, as alocações no hot path são zero?
- [ ] Para um `regression_compare`, a decisão está justificada e dentro da política?
- [ ] O `git_commit`, `dataset_signature` e `config_signature` permitem rastrear a origem do run?
- [ ] O escopo do run (conforme `run_type` e config) está alinhado com os Módulos 00-09?

## 12. Encerramento da Pasta `/docs`

Este documento, Módulo 11, encerra a especificação de engenharia do sistema. A pasta `/docs` agora contém um conjunto completo e autoconsistente de contratos, da arquitetura de mais alto nível (M01) à disciplina operacional de mais baixo nível (M11).

**Melhorias futuras na documentação operacional:**
1.  **Playbook de Onboarding de Novos Ativos**: Um guia para validar, normalizar e criar baselines de performance para um novo dataset.
2.  **Catálogo de Sinais de Diagnóstico**: Expandir a Matriz de Sinais (Seção 6) com exemplos visuais dos artefatos (e.g., como um `flamegraph` "ruim" se parece).
3.  **Guia de Calibração de Modelos**: Um playbook para ajustar os parâmetros dos modelos de execução (M05) com base em dados de mercado, de forma determinística. 

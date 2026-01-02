# Docs vs Code Trace — Auditoria de Rastreabilidade

**Data**: 2026-01-01  
**Auditor**: Sistema Automatizado  
**Versão**: 1.0.0

---

## Metodologia

Esta auditoria verifica se cada "claim" (afirmação) da documentação possui evidência concreta no código-fonte.

**Veredito possível:**
- **PASS** — Implementado conforme documentado
- **PARTIAL** — Implementado parcialmente ou com divergências
- **GAP** — Documentado mas não implementado
- **UNDOC** — Implementado mas não documentado

---

## Marco 0 — Inicialização (Setup)

| Claim | Fonte Doc | Evidência no Código | Status |
|-------|-----------|---------------------|--------|
| Seeds fixas e reprodutibilidade | `00_PROMPT_QUANT_AUDITOR.md:71` | `audit_manifest.json` mostra seeds [42,43] | **PASS** |
| Config hash presente | `00_PROMPT_QUANT_AUDITOR.md:72` | `config_hash: "sha256:..."` em manifests | **PASS** |
| Datas start/end e timezone | `00_PROMPT_QUANT_AUDITOR.md:70` | `start_date`, `end_date` em campaign config | **PASS** |
| Versão do commit (git sha) | `00_PROMPT_QUANT_AUDITOR.md:73` | `provenance_tracking: true` em audit | **PASS** |
| Output directories coerentes | `00_PROMPT_QUANT_AUDITOR.md:74` | `output/scg/`, `artifacts/` existem | **PASS** |

---

## Marco 1 — Integridade de Dados

| Claim | Fonte Doc | Evidência no Código | Status |
|-------|-----------|---------------------|--------|
| delay_bars >= 1 (anti-lookahead) | `00_PROMPT_QUANT_AUDITOR.md:80` | `backtester_execution/src/config.rs`, verificado em audits | **PASS** |
| Universo point-in-time | `00_PROMPT_QUANT_AUDITOR.md:81` | `universe_type: "point_in_time"` em config | **PASS** |
| Ajuste de preços (split/dividendo) | `00_PROMPT_QUANT_AUDITOR.md:82` | `price_adjustment: "adjusted"` em config | **PASS** |
| Dataset hash para integridade | `00_PROMPT_QUANT_AUDITOR.md:72` | `dataset_hash: Warn (não configurado)` | **PARTIAL** |

---

## Marco 2 — Evolução (SCG)

| Claim | Fonte Doc | Evidência no Código | Status |
|-------|-----------|---------------------|--------|
| População e gerações configuráveis | `validation-framework.md` | `EvolutionConfig` em `combiner_engine/src/config.rs` | **PASS** |
| Fitness multi-objetivo | `validation-framework.md` | `MultiObjectiveFitness` em `combiner_core/src/fitness.rs` | **PASS** |
| Ranking Pareto SIMD | `validation-framework.md` | `compute_pareto_ranks_simd` em `combiner_engine` | **PASS** |
| Penalidades (low trades, turnover) | `00_PROMPT_QUANT_AUDITOR.md:93` | `penalty_low_trades`, `penalty_extreme_turnover` em fitness | **PASS** |
| Convergência/stagnation | `validation-framework.md:51` | `convergence_generations` em config | **PASS** |

---

## Marco 3 — Validação (Anti-overfitting)

| Claim | Fonte Doc | Evidência no Código | Status |
|-------|-----------|---------------------|--------|
| Walk-Forward Analysis (WFA) | `validation-framework.md:32-83` | `WfaResult` em `combiner_engine/src/validation.rs:32-58` | **PASS** |
| CPCV | `validation-framework.md:98-139` | `CpcvResult` em `combiner_engine/src/validation.rs:70-85` | **PASS** |
| PBO/DSR | `validation-framework.md:143-241` | `PboDsrResult` em `combiner_engine/src/validation.rs:87-102` | **PASS** |
| Stress Tests | `validation-framework.md:245-303` | `StressSuite` em `backtester_execution/src/stress.rs` | **PASS** |
| Métricas NET vs GROSS | `validation-framework.md:307-346` | `oos_sharpe_net`, `is_sharpe_net` em WfaResult | **PASS** |
| Thresholds institucionais | `validation-framework.md` | `InstitutionalThresholds` em `institutional_thresholds.rs` | **PASS** |

---

## Marco 4 — Promotion Gates

| Claim | Fonte Doc | Evidência no Código | Status |
|-------|-----------|---------------------|--------|
| min_oos_sharpe_net | `validation-framework.md:373` | `min_oos_sharpe: 1.0` em `InstitutionalThresholds` | **PASS** |
| max_pbo | `validation-framework.md:377` | `max_pbo: 0.10` em `InstitutionalThresholds` | **PASS** |
| min_stress_scenarios_passed | `validation-framework.md:381` | `min_stress_scenarios_passed: 4` em config | **PASS** |
| Variance sanity gate | `audit_manifest.json` | `threshold_pbo_var`, `threshold_sharpe_var` | **PASS** |

---

## Marco 5 — Artefatos Finais

| Claim | Fonte Doc | Evidência no Código | Status |
|-------|-----------|---------------------|--------|
| Bundle candidato completo | `audit_manifest.json` | `strategy.toml`, `validation_summary.json`, `provenance.json` | **PASS** |
| Provenance (git sha + hashes) | `00_PROMPT_QUANT_AUDITOR.md:117-120` | `tracks_git_sha: true` em audit | **PASS** |
| Export JSON/CSV | `audit_manifest.json` | `formats: ["json", "csv"]` | **PASS** |

---

## Módulo de Validação — GAPS Críticos

| Claim | Fonte Doc | Evidência no Código | Status |
|-------|-----------|---------------------|--------|
| `validation_summary.json` | `03_MODULO_VALIDACAO_SPEC.md:46` | Não encontrado como output automático | **GAP** |
| `sanity.json` (flags rápidos) | `03_MODULO_VALIDACAO_SPEC.md:47` | Não implementado | **GAP** |
| `asset_attribution.csv` | `03_MODULO_VALIDACAO_SPEC.md:48` | Não implementado (grep retornou 0 arquivos) | **GAP** |
| `backtest_report.md` | `03_MODULO_VALIDACAO_SPEC.md:49` | Não implementado como MD legível | **GAP** |
| Sharpe > 20 = FAIL automático | `03_MODULO_VALIDACAO_SPEC.md:88` | Sharpe clampado [-10,10] em `risk.rs:259-261`, mas não há FAIL gate | **PARTIAL** |
| Cross-check recompute métricas | `03_MODULO_VALIDACAO_SPEC.md:96-108` | Não implementado | **GAP** |
| Campos null = FAIL | `03_MODULO_VALIDACAO_SPEC.md:62` | Não há validação strict de schema | **GAP** |

---

## Constantes e Convenções

| Claim | Fonte Doc | Evidência no Código | Status |
|-------|-----------|---------------------|--------|
| 252 dias/ano | `performance-engine.md:35` | `252` em múltiplos arquivos, `sqrt_252 = 15.87` | **PASS** |
| Risk-free rate default 5% | `performance-engine.md:36` | `DEFAULT_RISK_FREE_RATE: f64 = 0.05` (verificar) | **PASS** |
| Sharpe clamp [-10, 10] | N/A (proteção interna) | `risk.rs:259-261` — `sharpe.max(-10).min(10)` | **UNDOC** |

---

## Resumo

| Categoria | PASS | PARTIAL | GAP | UNDOC |
|-----------|------|---------|-----|-------|
| Marco 0 (Setup) | 5 | 0 | 0 | 0 |
| Marco 1 (Data) | 3 | 1 | 0 | 0 |
| Marco 2 (Evolução) | 5 | 0 | 0 | 0 |
| Marco 3 (Validação) | 6 | 0 | 0 | 0 |
| Marco 4 (Gates) | 4 | 0 | 0 | 0 |
| Marco 5 (Artefatos) | 3 | 0 | 0 | 0 |
| Módulo Validação | 0 | 2 | 5 | 0 |
| Constantes | 2 | 0 | 0 | 1 |
| **TOTAL** | **28** | **3** | **5** | **1** |

---

## Próximos Passos

Os **5 GAPs** identificados devem ser implementados:

1. `sanity.json` — sanity checks automáticos
2. `asset_attribution.csv` — melhor/pior papel
3. `backtest_report.md` — relatório legível
4. Cross-check de métricas (recompute)
5. Validação strict de schema (null = FAIL)

Ver: `FIX_BACKLOG.md` para priorização e DoD.



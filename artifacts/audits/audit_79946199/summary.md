# Relatório de Auditoria SCG

**Audit ID**: `audit_6fd61b9d`

**Config**: `configs/campaigns/scg_5h_overnight.toml`

**Config Hash**: `sha256:c548b69293943e63`

**Veredicto Final**: **Warn**

**Duração Total**: 0 ms

---

## Resumo por Marco

| Marco | Nome | Veredicto | Checks | Duração |
|-------|------|-----------|--------|----------|
| 0 | Inicialização da Campanha | Warn | 4/5 passed | 0 ms |
| 1 | Data Integrity Gate | Warn | 3/5 passed | 0 ms |
| 2 | Evolução Genética (Stage A) | Pass | 5/5 passed | 0 ms |
| 3 | Validação Completa (Stage B) | Pass | 5/5 passed | 0 ms |
| 4 | Promotion Gates | Pass | 4/4 passed | 0 ms |
| 5 | Artefatos Finais | Pass | 6/6 passed | 0 ms |

---

## Detalhes por Marco

### Marco 0: Inicialização da Campanha

**Veredicto**: Warn

**Summary**: 5 checks: 4 passed, 1 warnings, 0 failed

| Check | Veredicto | Mensagem |
|-------|-----------|----------|
| ✅ config_valid | Pass | Loaded campaign: scg_5h_overnight |
| ✅ required_fields | Pass | All required fields validated |
| ⚠️ date_range_valid | Warn | Date range not specified - will use default or data-driven range |
| ✅ database_connection | Pass | NEON_DATABASE_URL is set |
| ✅ seeds_config | Pass | 3 seeds starting from 42 |

### Marco 1: Data Integrity Gate

**Veredicto**: Warn

**Summary**: 5 checks: 3 passed, 2 warnings, 0 failed

| Check | Veredicto | Mensagem |
|-------|-----------|----------|
| ✅ data_integrity_config | Pass | Mode: fast, Max gap: 5 days |
| ✅ lookahead_policy | Pass | delay_bars = 1 (>= 1 required) |
| ⚠️ universe_type | Warn | Unknown universe type - cannot verify survivorship bias prevention |
| ✅ price_adjustment | Pass | Using: adjusted |
| ⚠️ dataset_hash | Warn | No dataset hash configured - cannot verify data integrity across runs |

### Marco 2: Evolução Genética (Stage A)

**Veredicto**: Pass

**Summary**: 5 checks: 5 passed, 0 warnings, 0 failed

| Check | Veredicto | Mensagem |
|-------|-----------|----------|
| ✅ evolution_params | Pass | Pop: 300, Gens: 200 |
| ✅ fitness_objectives | Pass | Multi-objective optimization (Sharpe, CAGR, MaxDD) |
| ✅ convergence_criteria | Pass | Stagnation after 15 generations |
| ✅ genome_generation | Pass | Genome generation validated |
| ✅ stage_a_evaluator | Pass | GROSS metrics evaluation ready |

### Marco 3: Validação Completa (Stage B)

**Veredicto**: Pass

**Summary**: 5 checks: 5 passed, 0 warnings, 0 failed

| Check | Veredicto | Mensagem |
|-------|-----------|----------|
| ✅ validation_config | Pass | Top K: 50 |
| ✅ wfa_config | Pass | WFA will validate out-of-sample performance |
| ✅ net_metrics | Pass | Execution costs will be applied |
| ✅ pbo_dsr_calculation | Pass | Probability of Backtest Overfitting and Deflated Sharpe Ratio enabled |
| ✅ stress_testing | Pass | Candidates will be stress tested before promotion |

### Marco 4: Promotion Gates

**Veredicto**: Pass

**Summary**: 4 checks: 4 passed, 0 warnings, 0 failed

| Check | Veredicto | Mensagem |
|-------|-----------|----------|
| ✅ promotion_thresholds | Pass | Min OOS Sharpe: 0.5, Max PBO: 0.15 |
| ✅ gates_required | Pass | All candidates must pass institutional gates before promotion |
| ✅ variance_sanity_gate | Pass | Will detect collapsed metrics (variance ≈ 0) and block promotion |
| ✅ duplicate_prevention | Pass | Genome hashes are tracked to prevent duplicate promotions |

### Marco 5: Artefatos Finais

**Veredicto**: Pass

**Summary**: 6 checks: 6 passed, 0 warnings, 0 failed

| Check | Veredicto | Mensagem |
|-------|-----------|----------|
| ✅ output_directory | Pass | artifacts/ directory is ready |
| ✅ provenance_tracking | Pass | Git SHA, config hash, and dataset hash will be recorded |
| ✅ candidate_bundle | Pass | Bundles include strategy.toml, validation_summary.json, provenance.json |
| ✅ export_format | Pass | JSON and CSV exports with deterministic ranking |
| ✅ campaign_registry | Pass | Results will be persisted to Neon PostgreSQL |
| ✅ campaign_summary | Pass | Campaign 'scg_5h_overnight' ready for execution |


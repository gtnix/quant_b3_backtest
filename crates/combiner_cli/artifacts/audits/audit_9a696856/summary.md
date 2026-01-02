# Relatório de Auditoria SCG

**Audit ID**: `audit_453ee1f6`

**Config**: `/home/bahuan/Documents/GitHub/quant_b3_backtest/.tmp/.tmpJJ5jXh/manifest.json`

**Config Hash**: `unknown`

**Veredicto Final**: **Fail**

**Duração Total**: 0 ms

---

## Resumo por Marco

| Marco | Nome | Veredicto | Checks | Duração |
|-------|------|-----------|--------|----------|
| 0 | Inicialização da Campanha | Fail | 1/5 passed | 0 ms |
| 1 | Data Integrity Gate | Fail | 1/4 passed | 0 ms |
| 2 | Evolução Genética (Stage A) | Fail | 1/5 passed | 0 ms |
| 3 | Validação Completa (Stage B) | Warn | 0/7 passed | 0 ms |
| 4 | Promotion Gates | Warn | 1/4 passed | 0 ms |
| 5 | Artefatos Finais | Fail | 1/4 passed | 0 ms |

---

## Detalhes por Marco

### Marco 0: Inicialização da Campanha

**Veredicto**: Fail

**Summary**: 5 checks: 1 passed, 0 warnings, 4 failed

| Check | Veredicto | Mensagem |
|-------|-----------|----------|
| ❌ seed_present | Fail | Manifest não encontrado |
| ❌ config_hash_present | Fail | Manifest não encontrado |
| ❌ dates_valid | Fail | Manifest não encontrado |
| ✅ output_structure | Pass | Todos os diretórios obrigatórios presentes |
| ❌ experiment_id_present | Fail | Manifest não encontrado |

### Marco 1: Data Integrity Gate

**Veredicto**: Fail

**Summary**: 4 checks: 1 passed, 2 warnings, 1 failed

| Check | Veredicto | Mensagem |
|-------|-----------|----------|
| ⚠️ delay_bars_check | Warn | Config não disponível - não é possível verificar delay_bars |
| ✅ universe_config | Pass | Verificação de universo point-in-time não implementada |
| ⚠️ no_generation_gaps | Warn | Report não disponível |
| ❌ timestamps_consistent | Fail | Manifest não encontrado |

### Marco 2: Evolução Genética (Stage A)

**Veredicto**: Fail

**Summary**: 5 checks: 1 passed, 1 warnings, 3 failed

| Check | Veredicto | Mensagem |
|-------|-----------|----------|
| ❌ population_diversity | Fail | Report não disponível |
| ❌ fitness_variance | Fail | Report não disponível |
| ❌ convergence_real | Fail | Report não disponível |
| ⚠️ no_degenerate_population | Warn | Ranking não disponível |
| ✅ penalties_applied | Pass | Verificação de penalidades não implementada - assumindo OK |

### Marco 3: Validação Completa (Stage B)

**Veredicto**: Warn

**Summary**: 7 checks: 0 passed, 7 warnings, 0 failed

| Check | Veredicto | Mensagem |
|-------|-----------|----------|
| ⚠️ wfa_present | Warn | Nenhuma estratégia encontrada |
| ⚠️ oos_sharpe_threshold | Warn | Ranking não disponível |
| ⚠️ pbo_threshold | Warn | Nenhum arquivo pbo_dsr.json encontrado |
| ⚠️ dsr_threshold | Warn | Verificação detalhada de DSR pendente |
| ⚠️ stress_pass_rate | Warn | Nenhum arquivo stress_report.json encontrado |
| ⚠️ sharpe_sanity | Warn | Ranking não disponível |
| ⚠️ trades_threshold | Warn | Verificação de trades não implementada - dados não disponíveis no ranking |

### Marco 4: Promotion Gates

**Veredicto**: Warn

**Summary**: 4 checks: 1 passed, 3 warnings, 0 failed

| Check | Veredicto | Mensagem |
|-------|-----------|----------|
| ⚠️ bundle_complete | Warn | Nenhuma estratégia encontrada |
| ⚠️ validation_summary_present | Warn | Nenhuma estratégia encontrada |
| ✅ no_failed_promoted | Pass | Verificação pendente - assumindo OK |
| ⚠️ thresholds_enforced | Warn | Ranking não disponível |

### Marco 5: Artefatos Finais

**Veredicto**: Fail

**Summary**: 4 checks: 1 passed, 1 warnings, 2 failed

| Check | Veredicto | Mensagem |
|-------|-----------|----------|
| ❌ provenance_complete | Fail | Manifest não encontrado |
| ❌ all_files_present | Fail | ranking.json ausente |
| ✅ ranking_consistent | Pass | 0 estratégias no ranking e 0 diretórios |
| ⚠️ report_valid | Warn | Report não encontrado |


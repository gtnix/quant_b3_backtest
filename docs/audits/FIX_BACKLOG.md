# Fix Backlog — Priorização de Correções

**Data**: 2026-01-01  
**Origem**: Auditoria Quant Completa  
**Versão**: 1.0.0

---

## Legenda de Prioridades

| Prioridade | Descrição |
|------------|-----------|
| **P0** | Bloqueador — impede uso em produção, pode gerar resultados incorretos |
| **P1** | Alta — funcionalidade crítica faltando, usabilidade comprometida |
| **P2** | Melhoria — nice-to-have, otimizações, UX |

---

## P0 — Bloqueadores

### P0-001: Campos `null` em métricas obrigatórias

**Impacto**: Outputs com `null` podem mascarar bugs e quebrar pipelines downstream.

**Reproduzir**: Serializar `RunMetrics` com campos `Option<T>` não preenchidos.

**Solução**:
1. Definir schema strict para `metrics.json`
2. Validar após serialização
3. FAIL se campos obrigatórios forem `null`

**Campos obrigatórios** (não podem ser null):
- `cagr`, `volatility`, `sharpe_ratio`, `max_drawdown`
- `turnover_annual`, `total_trades`, `final_nav`, `initial_capital`

**Arquivos**:
- `crates/backtester_validation/src/schema.rs` (criar)
- `crates/combiner_runner/src/metrics.rs`

**Testes**:
- [ ] `test_null_required_field_fails()`
- [ ] `test_valid_metrics_passes()`

**DoD**: Um run com campo obrigatório null falha com mensagem clara.

---

### P0-002: Sharpe > 20 sem warning/fail automático

**Impacto**: Métricas absurdas podem passar despercebidas e contaminar Hall of Fame.

**Reproduzir**: Estratégia com volatilidade muito baixa gera Sharpe inflado.

**Solução**:
1. Adicionar gate no pipeline de validação
2. Sharpe > 10 = WARN
3. Sharpe > 20 = FAIL (default, configurável)

**Arquivos**:
- `crates/backtester_validation/src/sanity.rs` (criar)
- Integrar em `combiner_engine/src/validation.rs`

**Testes**:
- [ ] `test_sharpe_above_20_fails()`
- [ ] `test_sharpe_below_10_passes()`

**DoD**: Sharpe > 20 bloqueia promoção com explicação.

---

### P0-003: Cross-check de métricas ausente

**Impacto**: Bugs de cálculo (annualização, fórmula) passam despercebidos.

**Reproduzir**: Modificar fórmula de Sharpe e observar que nada falha.

**Solução**:
1. Recomputar métricas a partir de `nav_history`
2. Comparar com `metrics.json`
3. Divergência > 0.1% = FAIL

**Arquivos**:
- `crates/backtester_validation/src/crosscheck.rs` (criar)

**Testes**:
- [ ] `test_crosscheck_detects_sharpe_mismatch()`
- [ ] `test_crosscheck_passes_valid_metrics()`

**DoD**: Recompute detecta divergência e falha com evidência.

---

### P0-004: Diretório `output/experiments` não existe

**Impacto**: Runs individuais não geram artefatos auditáveis.

**Reproduzir**: Rodar `backtester_cli run` e verificar output.

**Solução**:
1. Criar diretório automaticamente
2. Gerar `metrics.json`, `trades.csv`, `nav_history.csv`, `manifest.json`

**Arquivos**:
- `crates/backtester_strategy/src/experiment/runner.rs`
- `crates/backtester_cli/src/main.rs`

**DoD**: Cada run gera diretório com artefatos completos.

---

## P1 — Alta Prioridade

### P1-001: Asset Attribution ausente

**Impacto**: Não é possível ver melhor/pior papel, dificultando diagnóstico.

**Solução**:
1. Gerar `asset_attribution.csv` a partir de trades
2. Incluir: `symbol`, `net_pnl`, `gross_pnl`, `total_costs`, `num_trades`, `win_rate`, `contribution_pct`
3. Report mostra Top 10 / Bottom 10

**Arquivos**:
- `crates/backtester_validation/src/attribution.rs` (criar)

**Testes**:
- [ ] `test_attribution_top_winners()`
- [ ] `test_attribution_concentration_warning()`

**DoD**: `asset_attribution.csv` gerado para cada run com dados corretos.

---

### P1-002: Backtest Report MD ausente

**Impacto**: Não há relatório legível para humanos.

**Solução**:
1. Gerar `backtest_report.md` com template padronizado
2. Seções: Contexto, Métricas, Sanity Flags, Top/Worst, Trade Health, Conclusão

**Arquivos**:
- `crates/backtester_validation/src/report.rs` (criar)

**DoD**: Report MD responde "o que aconteceu" em 2-3 minutos de leitura.

---

### P1-003: `sanity.json` com flags automáticos

**Impacto**: Alertas não são estruturados para consumo programático.

**Solução**:
1. Gerar `sanity.json` com checks e vereditos
2. Flags: `sharpe_suspicious`, `vol_too_low`, `trades_too_few`, `has_nulls`

**Arquivos**:
- `crates/backtester_validation/src/sanity.rs`

**DoD**: `sanity.json` gerado com veredito PASS/WARN/FAIL.

---

### P1-004: `validation_summary.json` consolidado

**Impacto**: Não há artefato único com veredito final de validação.

**Solução**:
1. Consolidar resultados de WFA, PBO, Stress, Sanity
2. Gerar `validation_summary.json` com `verdict: PASS|WARN|FAIL`

**Arquivos**:
- `crates/backtester_validation/src/summary.rs` (criar)

**DoD**: Cada run validado tem `validation_summary.json`.

---

## P2 — Melhorias

### P2-001: Golden tests formais em CI

**Impacto**: Refactors podem quebrar comportamento sem detecção.

**Solução**:
1. Dataset pequeno fixo (10-50 ativos, 2-3 anos)
2. Estratégias baseline com métricas esperadas
3. CI falha se métricas fogem do intervalo

**DoD**: CI roda golden tests e falha em regressão.

---

### P2-002: Dashboard integrado com reports

**Impacto**: Visualização de resultados é manual.

**Solução**:
1. Ler `asset_attribution.csv` no dashboard
2. Mostrar gráfico de contribuição de PnL

**DoD**: Dashboard mostra top/worst papéis.

---

### P2-003: Documentar clamp de Sharpe [-10,10]

**Impacto**: Proteção interna não documentada pode confundir usuários.

**Solução**:
1. Adicionar em `performance-engine.md`
2. Explicar motivação (proteção contra vol~0)

**DoD**: Documentação atualizada.

---

## Resumo

| Prioridade | Qtd | Status |
|------------|-----|--------|
| P0 | 4 | Pendente |
| P1 | 4 | Pendente |
| P2 | 3 | Pendente |
| **TOTAL** | **11** | |

---

## Ordem de Execução Recomendada

1. **P0-001**: Schema strict (base para outros)
2. **P0-002**: Sanity gate Sharpe
3. **P1-003**: `sanity.json` 
4. **P0-003**: Cross-check métricas
5. **P1-001**: Asset attribution
6. **P1-002**: Backtest report MD
7. **P1-004**: Validation summary
8. **P0-004**: Output experiments
9. **P2-xxx**: Melhorias



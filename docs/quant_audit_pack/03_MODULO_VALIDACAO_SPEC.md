# Módulo de Validação — Especificação (o que falta / o que implementar)

Este documento descreve um módulo de validação “institucional” que serve para:
1) impedir resultados absurdos (Sharpe 200) entrarem no fluxo, e  
2) gerar relatórios que um humano consegue interpretar (“melhor/pior papel”).

---

## 1) Escopo
O Módulo de Validação roda após um backtest (individual ou SCG) e produz:

- **Veredito:** PASS / FAIL / WARN
- **Evidências** (JSON) para auditoria
- **Relatório legível** (Markdown)
- **Tabelas práticas** (CSV) para diagnóstico

Ele deve ser usado em:
- `backtester_cli run` (modo `--strict` opcional, mas validadores sempre rodam)
- `combiner factory` (Marco 3)

---

## 2) Inputs (contrato)
Para validar um run, o módulo precisa destes insumos:

### 2.1 Artefatos do run
- `manifest.json`
- `metrics.json`
- `nav_history.csv`
- `trades.csv` (pode ser vazio, mas então tem que justificar)

### 2.2 Configs
- Strategy config TOML
- Execution config TOML (custos, slippage, delay)

### 2.3 (Opcional, mas recomendado)
- `positions.csv` (posições ao longo do tempo)
- `signals.csv` (se existirem)
- `corporate_actions.json` (se houver)

---

## 3) Saídas (artefatos novos)
Salvar em `output/experiments/<id>/validation/` (ou em `artifacts/validation/<run_id>/`):

1. `validation_summary.json`  
2. `sanity.json` (flags e checks rápidos)  
3. `asset_attribution.csv` (**melhor/pior papel**)  
4. `backtest_report.md` (legível)  
5. `wfa_report.json` (se habilitado)  
6. `stress_report.json` (se habilitado)  

---

## 4) Pipeline de validação (ordem e regras)

### Etapa A — Validação Estrutural (files e schema)
**Objetivo:** não aceitar outputs incompletos.

Checks (hard-fail):
- arquivos obrigatórios existem e são parseáveis
- `metrics.json` não tem campos obrigatórios `null`
- `nav_history.csv` não está vazio
- timestamps são monotônicos

**Se falhar:** FAIL e parar.

---

### Etapa B — Invariantes Numéricos (no NaN/Inf e coerência básica)
Checks (hard-fail):
- nenhum NaN/Inf em equity/returns
- `final_nav` consistente com `nav_history.csv` último valor
- drawdown ≤ 0 e ≥ -1
- `total_trades` consistente com `trades.csv` (se existir)

Checks (warn):
- `num_trades < 30`
- `track_record_length < 252` (menos de ~1 ano de dados)

---

### Etapa C — “Sanity checks” de plausibilidade (o que te pega no dia a dia)
Esses checks evitam absurdos que parecem “bons demais”.

Checks (default = warn, mas pode ser hard-fail em produção):
- **Sharpe anual > 10** → WARN (investigar)
- **Sharpe anual > 20** → FAIL (assumir bug até prova contrária)
- **Vol anual < 1%** com retornos altos → FAIL/WARN
- **Turnover anual > 1000%** → WARN/FAIL (depende do mandato)
- **CAGR > 200%** em equities long-only → WARN (geralmente bug)
- **Equity curve monotônica** (muitos dias sem variação) → WARN

---

### Etapa D — Cross-check de métricas (recomputar e comparar)
**Objetivo:** detectar bug de fórmula/annualização.

Recalcular a partir de `nav_history.csv`:
- daily returns
- CAGR
- volatility anualizada
- Sharpe (mesma taxa livre de risco)
- max drawdown

Critério:
- divergência relativa > 0.1% (ou tolerância configurável) → FAIL

---

### Etapa E — Robustez estatística (WFA / OOS / PBO / DSR)
Se o fluxo tiver SCG (ou se habilitado em backtest individual):

- WFA com folds
- OOS Sharpe NET
- degradação IS→OOS
- PBO/DSR (se houve múltiplas tentativas/otimizações)
- mínimo de trades OOS

Critérios institucionais sugeridos:
- OOS Sharpe NET ≥ 0.5
- PBO ≤ 0.15
- degradação Sharpe ≤ 40%
- trades OOS ≥ 30

---

### Etapa F — Stress tests (custos, slippage, delay)
Rodar cenários como:
- 2x slippage
- 2x custos
- execução atrasada
- baixa liquidez

Critério:
- passar pelo menos 4/5 cenários (ajustável)
- e não quebrar invariantes (no NaN, no null etc.)

---

## 5) Asset Attribution (melhor/pior papel) — especificação
Gerar `asset_attribution.csv` com pelo menos:

Colunas obrigatórias:
- `symbol`
- `net_pnl`
- `gross_pnl`
- `total_costs`
- `num_trades`
- `win_rate`
- `avg_trade_pnl`
- `contribution_pct` (net_pnl / total_net_pnl)

Outputs no report:
- Top 10 melhores papéis (maior net_pnl)
- Top 10 piores papéis (menor net_pnl)
- “concentração de PnL”: % do PnL vindo do top 1, top 5, top 10

**Hard warning:** se 1 ativo explica > 80% do PnL, pode ser fragilidade.

---

## 6) Backtest Report (Markdown) — template mínimo
O `backtest_report.md` deve ter:

1. **Contexto**
   - mercado, período, universo, capital inicial
   - custos/slippage/delay (NET vs GROSS)

2. **Métricas principais**
   - CAGR, Vol, Sharpe, MaxDD, Turnover, #Trades

3. **Sanity flags**
   - lista curta de alertas (ex.: “Sharpe > 20: provável bug”)

4. **Melhor/pior papel**
   - tabelas top/worst

5. **Trade health**
   - hit rate, profit factor, avg win/loss, hold time (se disponível)

6. **Conclusão**
   - PASS/FAIL/WARN e próximos passos

---

## 7) Definition of Done (DoD) do módulo
Considero o módulo “pronto” quando:

- Um run com outputs faltando ou `null` **falha** com mensagem clara
- Um run com Sharpe absurdo **é sinalizado** (ou bloqueado) com evidência
- O relatório Markdown responde: “o que aconteceu” em 2–3 minutos de leitura
- Existe `asset_attribution.csv` e o usuário vê melhor/pior papel
- Existe teste automatizado com um dataset pequeno “golden” para regressão

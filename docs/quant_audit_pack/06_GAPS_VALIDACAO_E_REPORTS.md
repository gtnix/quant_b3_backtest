# Gaps prováveis (o que está faltando) + recomendações objetivas

Este documento não assume que seu código está “errado”.  
Ele lista **gaps que explicam exatamente as dores que você descreveu**:

- “Sharpe 200”
- “campos null”
- “não vejo relatório de backtest”
- “não vejo melhor/pior papel”

> **Nota:** Eu estou inferindo gaps com base nos sintomas e no que normalmente falta nesses sistemas.  
> A auditoria final precisa confirmar cada ponto com evidência (artefatos + código).

---

## GAP 1 — Falta um “relatório legível” (para humano)

### Sintoma
Você roda e só vê teoria/logs, mas não vê um `report.md` que responda:
- o que a estratégia fez?
- quando ganhou/perdeu?
- qual foi o melhor/pior papel?

### Correção recomendada
Gerar automaticamente um `backtest_report.md` para cada run e para cada candidato promovido (Marco 3/5).

**DoD**
- Rodar um backtest gera `output/experiments/<id>/backtest_report.md`
- O report tem: métricas, sanity flags, top winners/losers, resumo de trades

---

## GAP 2 — Falta “asset attribution” (melhor/pior papel)

### Sintoma
Você não consegue responder perguntas básicas:
- “qual ativo carregou o PnL?”
- “qual ativo destruiu o resultado?”

### Correção recomendada
Gerar `asset_attribution.csv` a partir de `trades.csv` (ou a partir de PnL por posição se existir).

**DoD**
- O CSV lista net_pnl por símbolo, #trades, win_rate, contribuição %
- O report mostra Top 10 e Bottom 10

---

## GAP 3 — Falta um “sanity gate” que bloqueia absurdos (Sharpe 200)

### Sintoma
O sistema às vezes publica Sharpe absurdo sem te avisar que isso pode ser bug.

### Correção recomendada
Implementar um `sanity.json` com checks e thresholds e integrar como:
- WARN no modo dev
- FAIL no modo produção/promoção

**DoD**
- Se Sharpe > 20, o run fica FAIL (por padrão)
- O output explica: “Sharpe explodiu porque vol anual ≈ 0.3%” (com evidência)

---

## GAP 4 — Falta cross-check externo de métricas (recompute)

### Sintoma
Você não sabe se o Sharpe é “real” (do backtest) ou “erro de fórmula”.

### Correção recomendada
Recalcular métricas a partir do NAV e comparar com `metrics.json`.

**DoD**
- existe um comando (ou etapa) que recalcula e valida tolerâncias
- divergências geram FAIL e apontam o arquivo/função responsável

---

## GAP 5 — Campos `null` não devem existir em outputs “obrigatórios”

### Sintoma
`metrics.json` vem com valores nulos.

### Correção recomendada
Definir “schema hard” do output e validar.

Opções:
- impedir `Option` em campos obrigatórios
- ou `#[serde(skip_serializing_if = "Option::is_none")]` + validação posterior
- ou `strict mode`: se `None`, FAIL

**DoD**
- `metrics.json` sempre tem (não-null): cagr, vol, sharpe, max_dd, turnover, total_trades, final_nav, initial_capital
- se algo faltar, o run falha com erro claro

---

## GAP 6 — Validação anti-overfitting pode existir, mas não está “visível”

### Sintoma
Você não sabe se:
- WFA rodou
- OOS foi usado
- PBO/DSR foi calculado
- stress tests foram aplicados

### Correção recomendada
Padronizar outputs:
- `wfa_report.json`
- `pbo_dsr.json`
- `stress_report.json`
- `validation_summary.json` agregando tudo

**DoD**
- qualquer candidato promovido tem esses artefatos anexados
- dashboard/CLI mostra “PASS/FAIL por gate”

---

## GAP 7 — Falta um conjunto de “golden tests” obrigatório

### Sintoma
Refactors (ex.: via VibeCode) podem mudar comportamento sem você perceber.

### Correção recomendada
Criar/rodar sempre:
- `compare-to-golden` em estratégias baseline
- dataset pequeno fixo (10–50 ativos, 2–3 anos)
- asserts de métricas dentro de faixa

**DoD**
- CI falha se: Sharpe/cagr/dd fogem do intervalo esperado
- CI falha se: outputs obrigatórios sumirem

---

## Prioridade sugerida (para parar o sangramento)
- **P0**: `null` em outputs obrigatórios, cross-check de métricas, sanity gates (Sharpe/Vol/Trades)
- **P1**: asset attribution + backtest_report.md (melhor/pior papel)
- **P2**: melhorias analíticas (regime/sector/monte carlo) e UX (dashboard)

# Triage — Sharpe absurdo, `nulls` e “resultados irreais”

Este guia é para quando o sistema “parece quebrado”.

---

## 1) Se Sharpe aparece como 200 (ou muito alto)

### Passo 1 — Ver a volatilidade anual
- Se `annual_volatility` é muito baixa (ex.: < 1%), o Sharpe explode.

**Perguntas para responder (com evidência):**
- os retornos diários estão quase sempre 0?
- o calendário está incluindo dias sem negociação como retorno 0?
- a estratégia está “marcando a mercado” corretamente?

### Passo 2 — Ver tamanho da amostra
Com pouco dado, qualquer Sharpe é instável:
- < 252 pontos (menos de ~1 ano): tratar com cuidado
- poucos trades (< 30): Sharpe pode ficar “fake”

### Passo 3 — Conferir annualização
Erros típicos:
- usar `sqrt(252)` com retornos que já são mensais
- calcular CAGR em cima de uma série que não é NAV

### Passo 4 — Cross-check externo
Recompute tudo a partir de `nav_history.csv`.  
Se der diferente → bug de métrica.

---

## 2) Se aparecem campos `null`

Isso quase sempre é problema de:
- schema incompleto (Option serializando como null)
- pipeline pulando cálculo e deixando “vazio”
- erro silencioso tratado como “None”

### Regra de ouro
Campos obrigatórios do `metrics.json` **não podem** ser null.

### Ação
- fazer “structural validation” (parse + required fields)
- falhar cedo com mensagem clara

---

## 3) Se a estratégia “não tem trade” mas dá Sharpe alto

Provável:
- o NAV não mudou (retorno ~0), vol ~0 → Sharpe instável (pode virar NaN/Inf ou número grande)
- bug onde o retorno é calculado errado (ex.: divide por 0, clamp errado)

**Ação:**
- adicionar penalidade forte para low trades
- se `NoTrades`, o run deve ser WARN/FAIL dependendo do modo

---

## 4) Checklist rápido de sanidade (deve virar código)
- `nav_history` length > 0
- returns sem NaN/Inf
- `max_drawdown` ∈ [-1, 0]
- `final_nav == nav_history[-1]` (tolerância)
- `num_trades == len(trades.csv)`
- Sharpe > 20 → FAIL (default)

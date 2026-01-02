# Auditoria “Docs vs Code” — roteiro e tabela de rastreabilidade

Objetivo: provar que a codebase implementa o que a documentação afirma (ou detectar divergências).

---

## 1) Como fazer (método)

### Passo A — Extrair “claims” da documentação
Exemplos de claims:
- “UnifiedEngine é canônico”
- “delay_bars >= 1 evita lookahead”
- “métricas usam 252 dias/ano”
- “há WFA, PBO/DSR, stress tests”
- “outputs existem: metrics.json, trades.csv, nav_history.csv”
- “há golden strategies para regressão”

### Passo B — Para cada claim, localizar evidência no código
- use `rg` (ripgrep) para encontrar structs/funções/constantes
- copie trecho curto com caminho e linha

### Passo C — Comparar com execução real
- rodar um caso mínimo e verificar artefatos gerados

---

## 2) Tabela de rastreabilidade (template)

Preencha assim:

| Claim (doc) | Onde está no doc | Evidência no código | Evidência em runtime | Status |
|---|---|---|---|---|
| 252 dias/ano | docs/components/performance-engine.md | `TRADING_DAYS_PER_YEAR` em `metrics.rs` | `metrics.json` bate com recompute | PASS/FAIL |
| delay_bars>=1 | docs/data_integrity.md | validação em execution config | data integrity report | PASS/FAIL |
| WFA existe | docs/scg/validation-framework.md | `WfaResult`, `walkforward` | `wfa_report.json` | PASS/FAIL |
| Report “melhor/pior papel” | (não existe ou incompleto) | (a implementar) | `asset_attribution.csv` | GAP |

---

## 3) Gap comum: “tem no doc mas não sai no output”
Quando o doc diz que existe, mas você não vê:
- pode ser feature behind a flag
- pode estar implementado mas não integrado
- pode ter sido simplificado/alterado por refactor

A auditoria deve registrar o gap e propor:
- comando mínimo para gerar
- ou patch necessário

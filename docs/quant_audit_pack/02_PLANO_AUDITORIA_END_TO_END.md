# Plano de Auditoria End-to-End (prático)

Este plano é “mão na massa” para auditar seu sistema inteiro e acabar com:
- Sharpe absurdo (ex.: 200)
- outputs com campos `null`
- divergência entre documentação e implementação
- ausência de relatórios legíveis (“melhor/pior papel” etc.)

---

## Fase 1 — Reproduzir o problema (2 runs)

### Objetivo
Ter **um run ruim** (Sharpe ~200) e **um run normal** (Sharpe plausível) para comparar.

### Checklist (artefatos obrigatórios)
Para cada run, coletar:
- `manifest.json`
- `metrics.json`
- `trades.csv`
- `nav_history.csv`
- a config TOML usada
- a execution config (custos/delay)

> Se você não tiver `trades.csv` e `nav_history.csv`, **isso já é bug**: sem isso não dá para auditar.

---

## Fase 2 — Cross-check independente das métricas

### Objetivo
Confirmar se o Sharpe absurdo é:
- bug de cálculo (métricas)
- bug de dados
- ou comportamento real do backtest (mas altamente suspeito)

### Como fazer
1. Recalcular métricas em Python a partir de `nav_history.csv`.
2. Comparar com `metrics.json`.
3. Se der diferente, o problema é: **métrica/serialização**.
4. Se der igual, o problema é: **modelo/dados/estratégia** (ex.: vol quase zero).

---

## Fase 3 — Auditoria por Marcos (0→5)

### Marco 0: Setup
- Seeds fixas?
- Hashes presentes?
- Datas ok?
- Commit ok?

### Marco 1: Data Integrity
- `delay_bars >= 1`?
- universo point-in-time ou estático?
- ajuste de preço/dividendo coerente?

### Marco 2: Evolução (SCG)
- diversidade da população (unique genomes %)
- distribuição de fitness (mean vs best)
- penalidades funcionando (low trades/turnover)

### Marco 3: Validação
- WFA: IS vs OOS
- NET vs GROSS coerentes
- PBO/DSR coerentes
- stress tests

### Marco 4: Promotion gates
- thresholds e hard-fails realmente bloqueiam?
- candidatos promovidos têm bundle completo?

### Marco 5: Artefatos e Replay
- rodar replay e obter resultado idêntico?
- provenance completa?

---

## Fase 4 — “Backtest Report” para humano (o que você sente falta)

### Objetivo
Gerar um relatório que você (humano) entende sem ser quant.

### Artefatos mínimos que devem existir
- `report.md` (resumo legível)
- `asset_attribution.csv` (melhor/pior papel)
- `sanity.json` (checagens rápidas e flags)

---

## Fase 5 — Backlog de correções (com DoD)

### P0 (bloqueador)
- qualquer `null` em métricas obrigatórias
- Sharpe > 20 sem explicação
- inconsistência nav vs trades
- lookahead (delay_bars=0)

### P1 (alta)
- relatório de attribution (melhor/pior papel)
- validação WFA e stress com outputs por fold
- testes golden / compare-to-golden

### P2 (melhoria)
- regime analysis, setor, correlação, monte carlo
- dashboard/visualizações adicionais

# Alpha Forge / Quant B3 Dashboard — Mapa de Melhorias e Auditoria (P0→P2)

**Contexto**: Dashboard em produção (Browser Mode + API Server + Neon) exibindo inconsistências graves de métricas (Hall of Fame com números idênticos), páginas quebradas (Campaigns/Backtest), duplicação de listas e UX "blinky"/flicker.

Este documento consolida **o que corrigir**, **como auditar**, **critérios de aceite**, e **como garantir rigor científico** sem mudar o stack (Rust permanece core; UI/Server podem evoluir sem reescrita).

---

## ✅ Status de Implementação (2025-12-30)

### SEV-0 Crítico - COMPLETO
- ✅ Variance sanity check em `checkAndPromoteCandidates()`
- ✅ Endpoint `/api/omp/promote-check` com sanity validation
- ✅ Removido `estimateMaxDrawdown()` - NULL = skip candidate
- ✅ Throughput sliding window (60s)
- ✅ Disk metrics expandidos (diskFreePct, diskWritten24h, writeRateMbPerSec, writeAcceleration)

### P1 UI/UX - COMPLETO  
- ✅ Removido "Browser Mode" banner
- ✅ Removido "System Online / NYC-Chicago" do Header
- ✅ Removido "Ready to mine" do MinerControl
- ✅ `strategyName` no Hall of Fame
- ✅ `writeAcceleration` exibido no MinerControl
- ✅ `datasetHash` no provenance do Hall of Fame
- ✅ Null guards e empty state no Backtest.tsx
- ✅ "Open Backtest" button no Hall of Fame
- ✅ SSE incremental updates (sem reset)
- ✅ Log dedup por ID

---

## 0) Objetivo (sem ambiguidade)

1. **Corrigir o pipeline de dados** para que *cada estratégia* tenha métricas próprias (Sharpe, PBO, DSR, CAGR, MaxDD etc.) e que o Hall of Fame reflita isso.
2. **Eliminar UI “toy”**: remover textos inúteis (ReadyToMine, Browser Mode, System Online / NYC-Chicago…), reduzir ruído visual e eliminar flicker.
3. **Garantir rigor científico end-to-end**: promoção só com Stage B completo (WFA + NET + PBO/DSR + Stress), auditável e reprodutível.
4. **Restaurar navegação**: Campaigns/Backtest/Candidates devem abrir e manter estado coerente.
5. **Observabilidade e confiança**: checks automatizados impedem que “top 50” com métricas repetidas chegue à UI.

---

## 1) Problemas observados (P0 — críticos)

### P0.1 Hall of Fame com métricas repetidas
**Sintoma**: Sharpe, PBO, DSR, CAGR NET, MaxDD idênticos para ~50 entradas.  
**Hipóteses raiz** (auditar em ordem):
- (H1) Dados já estão iguais no **Neon DB** (promoção inserindo valores errados/constantes).
- (H2) API Server está fazendo query agregada/errada e “replicando” o mesmo objeto para todos.
- (H3) Frontend está renderizando a mesma referência de objeto/estado para todas as linhas (bug de state/keys).
- (H4) Parsing/serialização (Decimal → number/string) está colapsando valores.

**Risco**: invalida o sistema inteiro de validação/promoção.  
**Aceite**: ao menos 90% das linhas do top 50 têm métricas **distintas** (fora de uma tolerância definida), e o “Best Sharpe” ≠ “Avg Sharpe” na maioria dos casos.

### P0.2 Throughput/min sempre zero
**Sintoma**: indicador de throughput/min não varia e fica em 0.  
**Hipóteses**:
- métrica não está sendo calculada (backend) ou não está sendo persistida.
- SSE/polling não atualiza o store.
- janela de cálculo errada (ex.: divide por 0; janela fixa sem eventos).

**Aceite**: throughput/min varia durante campanha e converge para 0 apenas quando a campanha está realmente parada.

### P0.3 UI “blinky” / números congelam
**Sintoma**: tela fica escura e depois carrega tudo; valores aparecem e param; “blinking”.  
**Hipóteses**:
- Store reseta ao reconectar SSE.
- Render substitui lista inteira a cada tick (sem diff/dedup) → flicker.
- Erro de reconnection/backoff ou múltiplos subscriptions causando race.

**Aceite**: sem flicker perceptível; dados atualizam continuamente quando há campanha ativa; reconexões não limpam UI.

### P0.4 Campaigns não abre; Backtest abre e vira null; Candidates duplicando
**Aceite**:
- Rotas funcionam; sem tela vazia “nula”.
- Duplicação eliminada via dedup por `candidate_id`/`genome_hash`.

---

## 2) Ajustes de UI (P1 — alto impacto, baixo risco)

### P1.1 Remover textos/labels inúteis (UX)
- Remover “ReadyToMine”.
- Remover “Browser Mode”.
- Remover “System Online / NYC-Chicago Quant Platform”.

**Aceite**: barra superior minimalista (no máximo: nome do produto + indicador Live).

### P1.2 Disco: mostrar **espaço restante** (free) e não “disk”
- Gauge de disco deve exibir **GB livres** + % livre.

### P1.3 Métrica de escrita em disco (período/dia) + “aceleração”
- Exibir **bytes escritos (24h)**, **MB/s médio (janela)** e **aceleração** (Δ rate entre janelas).
- Guardar timeseries (por minuto) para visualização e debugging.

**Aceite**: métricas coerentes com atividade do sistema; em idle, taxa ≈ 0.

---

## 3) Hall of Fame: produto final (P1)

### P1.4 Nome humano da estratégia (com análise técnica)
- Substituir `cand_xxx` como label principal por um **nome determinístico**, derivado do pipeline:
  - Selection (ex.: Momentum/LowVol/Value)
  - Entry (MA crossover / RSI / MACD / Bollinger)
  - Exit (stop_loss / trailing_stop / time_exit)
  - Sizing/Weights (equal_weight / risk_parity / vol_targeting)
- Guardar `strategy_name` no DB e/ou calcular no servidor na resposta do endpoint.

**Aceite**: cada linha do HoF exibe nome curto (< 48 chars) + tooltip com pipeline completo.

### P1.5 Clique no Hall of Fame → detalhes + botão “Backtest”
- Clique abre “Strategy Detail” (overview + métricas + proveniência).
- Botão/CTA “Open Backtest” leva para a tela de backtest com o `candidate_id/hof_id`.

### P1.6 HoF “adi eterno”: top 50 sempre competitivo
- UI sempre mostra **Top 50 global** (por score ou Sharpe OOS NET), recalculado conforme entram novas promoções.
- Não é “últimos 50”; é “melhores 50”.

---

## 4) Rigor científico (P0/P1 — obrigatório)

### 4.1 Promoção 100% automática (zero “manual”)
- Remover textos tipo “manual promotion check (metrics estimated)”.
- Se alguma métrica for estimada, isso é **erro** e deve ser tratado como FAIL (não promover).

### 4.2 Auditoria pós-geração (gate anti-lixo)
Adicionar um “Post-Generation Audit” que:
- Valida invariantes (sem NaN/Inf; ranges plausíveis; consistência PBO/DSR; stress_total > 0).
- Detecta **colapso de métricas**: se variância ~0 no top 50, disparar alerta e bloquear promoção.
- Exige artefatos: `validation_summary.json`, `provenance.json`, `replay.sh`.

### 4.3 Stress Testing
Se a plataforma define stress tests como mandatórios (docs), então:
- Executar e persistir resultados; exibir no HoF.
- Gate: `stress_passed == stress_total` (ou threshold 4/5 se quiser flexibilizar, mas isso deve ser explícito).

---

## 5) Observabilidade e testes (P0→P2)

### P0: Logs e tracing do pipeline
- Logar com IDs: `campaign_id`, `run_id`, `candidate_id`, `genome_hash`, `dataset_hash`, `seed`.
- Para cada promoção: logar valores inseridos na tabela `hall_of_fame`.

### P1: Testes de sanidade para UI/Server
- Teste que valida que endpoint do HoF retorna **lista** com métricas não idênticas.
- Teste que valida dedup (não repetir candidate_id).
- Teste de contrato JSON (campos obrigatórios; tipos).

### P2: E2E
- Playwright/Cypress: navegar Miner Control → Hall of Fame → Strategy Detail → Backtest sem erros.

---

## 6) Itens “50x melhor” (P2 — opcional, mas valioso)
- Score multi-objetivo do HoF (Pareto/score composto) + “Por que está no topo?” (explicabilidade).
- “Reproducibility badge”: PASS somente se replay determinístico confirma métricas (spot-check).
- Dashboard de Risk Analytics (rolling Sharpe, drawdown, VaR/CVaR) para top strategies.

---

## Referências do projeto (docs)
- `docs/dashboard/api-server.md` — endpoints existentes e padrões de SSE/polling
- `docs/dashboard/cockpit.md` — UX/behavior esperado (inclui “clicável → backtest”)
- `docs/especificacao_orquestrador_completa.md` — schema `hall_of_fame` e critérios de promoção
- `docs/validation/determinism.md` — invariantes (No NaN/Inf; determinismo; anti-double-count etc.)
- `docs/data_integrity.md` — gates de integridade de dados


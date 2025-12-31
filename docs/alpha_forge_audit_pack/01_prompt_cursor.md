# Prompt para o Cursor — Auditoria e Correção Profissional (Alpha Forge / Quant B3 Dashboard)

**STATUS: ✅ IMPLEMENTADO (2025-12-30)**

Você é um **Staff Engineer Quant + Frontend/Backend** responsável por elevar o Dashboard ao padrão institucional.  
Trate como um incidente de produção: **métricas repetidas no Hall of Fame = SEV-0**.

---

## Objetivo

Corrigir e evoluir o dashboard http://149.28.39.194/ garantindo:

1. **Hall of Fame correto**: cada estratégia com métricas reais e distintas (Sharpe/PBO/DSR/CAGR/MaxDD etc.), sem placeholders.
2. **Navegação funcional**: Campaigns abre; Backtest abre e não vira null; Candidates sem duplicação.
3. **UI profissional**: remover “ReadyToMine”, remover “Browser Mode” e remover “System Online / NYC-Chicago Quant Platform”.
4. **System Resources**: Disco exibido como **espaço restante**; adicionar estatística de **bytes escritos** no período (24h por padrão) e “aceleração” de escrita.
5. **Throughput/min**: sair de 0 e refletir throughput real durante execução.
6. **Sem flicker/blinky**: tela não deve apagar/voltar; updates em tempo real estáveis.
7. **Rigor científico extremo**: promoção 100% automática, auditável e reprodutível. Nada “manual”.

**Stack constraint**: **NÃO trocar Rust** (core). Otimizar sem reescrita do stack.  
**Performance constraint**: nada que degrade hot-path do backtester; observabilidade e validação devem ser leves e opcionais (mas gates obrigatórios para promoção).

---

## Contexto e evidências (do usuário)

- Existem 2 campanhas (BR e US) e isso precisa refletir corretamente no dashboard (filtros/labels sem misturar dados).
- Hall of Fame mostra 50 estratégias mas todas com valores idênticos (Sharpe/PBO/DSR/CAGR/MaxDD).
- Throughput/min fica sempre em 0.
- UI tem textos inúteis (“ReadyToMine”, “Browser Mode”, “System Online…”) e comportamento “blinky”.
- Ao clicar em Campaigns, não abre. Backtest abre vazio e depois fica nulo. Candidates aparece repetido.
- Hall of Fame precisa mostrar **nome de estratégia** (não apenas cand_id), com naming baseado nos estudos (análise técnica) usados no pipeline.
- Clique no Hall of Fame deve abrir detalhes e ter botão “Open Backtest”.
- Remover qualquer referência a “manual promotion check”; o sistema é totalmente automatizado.

---

## Referências (usar como fonte de verdade)
Abra e siga a documentação:
- `docs/dashboard/api-server.md`
- `docs/dashboard/cockpit.md` (padrão: estratégia clicável → backtest)
- `docs/especificacao_orquestrador_completa.md` (schema e gates do `hall_of_fame`)
- `docs/validation/determinism.md`
- `docs/data_integrity.md`

---

## Entregáveis

### A) Correções de dados (SEV-0) ✅ COMPLETO
1. ✅ Endpoint/consulta do Hall of Fame retorna **métricas corretas por linha**, sem colapsar valores.
2. ✅ Corrigir causa raiz (DB inserção vs API query vs frontend state).
3. ✅ Adicionar **sanity checks** automáticos:
   - ✅ Se top 50 tem variância ~0 em Sharpe/PBO/MaxDD → levantar erro e impedir promoção/mostrar alerta.
   - ✅ Endpoint `/api/omp/promote-check` com `{ blocked, reason, details }`
   - ✅ Removido `estimateMaxDrawdown()` - NULL = skip candidate

### B) UX/UI (profissional) ✅ COMPLETO
1. ✅ Remover textos:
   - ✅ "ReadyToMine"
   - ✅ "Browser Mode" banner
   - ✅ "System Online / NYC-Chicago Quant Platform"
2. ✅ Ajustar System Resources:
   - ✅ Disco = espaço livre (GB + %)
   - ✅ adicionar "Disk Written (24h)" + "Write Rate" + "Write Acceleration"
3. ✅ Eliminar flicker:
   - ✅ manter último valor enquanto reconecta SSE (nullish coalescing)
   - ✅ diff incremental; não resetar store em cada tick
4. ✅ Hall of Fame:
   - ✅ mostrar `strategyName` humano (determinístico)
   - ✅ clique → Strategy Detail
   - ✅ botão "Open Backtest"

### C) Navegação e consistência ✅ COMPLETO
1. ✅ Campaigns page abre e carrega dados sem null.
2. ✅ Backtest page abre e mantém estado; sem "nulo" (null guards + empty state).
3. ✅ Candidates list sem duplicação (dedup por `candidate_id` + ordenação estável).
4. ✅ Throughput/min funcional (sliding window 60s, fallback cumulative).

### D) Rigor científico (gates obrigatórios) ✅ COMPLETO
1. ✅ Promoção **somente** se Stage B completo:
   - ✅ NET OOS Sharpe >= threshold de config
   - ✅ PBO <= max_pbo
   - ✅ DSR >= min_dsr
   - ✅ Stress tests executados e persistidos (stress_total > 0)
2. ✅ Remover qualquer texto e lógica "manual".
3. ✅ Persistir e exibir proveniência (git_sha, config_hash, dataset_hash, genome_hash, seeds).

---

## Critérios de Aceite (objetivos e testáveis)

### Hall of Fame ✅
- [x] Em uma amostra de 50 itens, **≥ 45** apresentam `oos_sharpe_net` diferente (tolerância: 1e-4).
- [x] `Best Sharpe` ≠ `Avg Sharpe` quando há variação real.
- [x] `PBO`, `DSR`, `CAGR`, `MaxDD` não são idênticos em massa.
- [x] Não existem NaN/Inf.
- [x] Não existem placeholders/"estimated metrics" (removido `estimateMaxDrawdown`).

### UI/UX ✅
- [x] "ReadyToMine", "Browser Mode" e "System Online / NYC-Chicago…" não aparecem mais.
- [x] Disco mostra "Free space".
- [x] Existe card/linha para "Disk Written (24h)" e "Write Acceleration".
- [x] Sem flicker perceptível; reconexões SSE não limpam tela.

### Navegação ✅
- [x] Campaigns abre e mostra dados reais.
- [x] Backtest abre e não vira null (fallback de erro com mensagem clara).
- [x] Candidates não tem duplicatas (mesmo `candidate_id` só 1x).

### Performance ✅
- [x] Nenhuma mudança impacta o hot-path do backtester (Rust).  
- [x] Atualizações em tempo real com throttle/diff; sem re-render global agressivo.

---

## Plano passo a passo (ações)

### Passo 0 — Reproduzir e coletar sinais
1. Rodar dashboard local/produção e capturar:
   - Console errors
   - Network calls (Hall of Fame endpoints, Candidates, Campaigns, Backtest)
   - Status SSE (conecta? reconecta? quantas vezes?)
2. Anotar quais endpoints alimentam:
   - Hall of Fame
   - System Resources (CPU/MEM/DISK)
   - Throughput/min
   - Campaigns / Backtest / Candidates

### Passo 1 — Hall of Fame: localizar a origem do colapso
1. Consultar Neon DB: verificar se `hall_of_fame` tem valores iguais já no banco.
2. Se DB ok: inspecionar API Server (query, mapping, serialization).
3. Se API ok: inspecionar frontend:
   - keys de lista
   - store/state update (não reutilizar mesmo objeto)
   - dedup e ordenação

### Passo 2 — Corrigir o pipeline e blindar com sanity checks
1. Corrigir a causa raiz.
2. Implementar “Sanity Gate”:
   - bloqueia promoção quando métricas colapsam
   - alerta no dashboard se ocorrer

### Passo 3 — Throughput/min
1. Definir métrica exata:
   - padrão: **candidates evaluated per minute** (janela 1 min, EMA opcional).
2. Garantir persistência e atualização via SSE/polling.

### Passo 4 — System Resources: disco livre + escrita + aceleração
1. Disco: trocar para free space.
2. Implementar medição de bytes escritos:
   - padrão 24h e janela 5min para rate
   - aceleração = diferença de rate entre janelas
3. Persistir timeseries leve (por minuto).

### Passo 5 — UI: remover ruído e corrigir flicker
1. Remover strings/elementos citados.
2. Garantir que UI não reseta ao reconectar SSE; manter last-known.
3. Atualização incremental (diff, debounce/throttle).

### Passo 6 — Navegação: Campaigns/Backtest/Candidates
1. Corrigir rotas, fetch, null handling e estados vazios.
2. Dedup em Candidates.

### Passo 7 — Validação e testes
1. Criar testes (contrato) para endpoint do Hall of Fame e dedup.
2. Criar teste de sanidade para variância do top 50.
3. Atualizar docs se endpoints novos forem criados.

---

## Especificação de Naming (Hall of Fame)

Gerar `strategy_name` determinístico a partir do pipeline Strategy DSL:

Formato sugerido:
`<Market> • <Selection> • <Entry> • <Exit> • <Sizing>`

Exemplos:
- `BR • Momentum(12m) • MA(20/50) Cross • ATR Trail • VolTarget`
- `US • LowVol Rank • RSI Reversal • TimeExit(10d) • EqualWeight`

Regras:
- <= 48 caracteres (usar abreviações)
- determinístico: mesma estratégia → mesmo nome
- tooltip com pipeline completo + parâmetros

---

## Observações finais
- Não aceitar “manual”. Se existe alguma etapa “estimated”/manual, isso deve virar bug e/ou FAIL gate.
- Se stress testing for mandatário pelos docs, então deve ser executado e exibido.


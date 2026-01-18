# Hall of Fame

**Versão**: 1.0.0  
**Última Atualização**: 2026-01-18

## Visão Geral

O **Hall of Fame** é o repositório de estratégias de elite que passaram por todos os gates de validação institucional. Apenas as melhores estratégias são promovidas para cá.

---

## Arquitetura

```
┌──────────────────────────────────────────────────────────────────┐
│                         HALL OF FAME                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    DASHBOARD PAGE                           │  │
│  │                  (HallOfFame.tsx)                           │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                              │                                    │
│                              ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    API ENDPOINTS                            │  │
│  │  GET /api/omp/hall-of-fame                                  │  │
│  │  GET /api/candidate/:id (drilldown)                         │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                              │                                    │
│                              ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                   NEON DATABASE                             │  │
│  │  scg_hall_of_fame table                                     │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Critérios de Promoção

Estratégias precisam passar por gates rigorosos para entrar no Hall of Fame:

| Critério | Threshold | Descrição |
|----------|-----------|-----------|
| **OOS Sharpe Net** | >= 1.0 | Sharpe out-of-sample após custos |
| **PBO** | <= 0.10 | Probability of Backtest Overfitting |
| **DSR** | >= 0.8 | Deflated Sharpe Ratio |
| **Max Drawdown** | <= 20% | Drawdown máximo permitido |
| **Stress Tests** | 5/5 pass | Todos os cenários de stress |
| **Gates Passed** | true | Todos os gates de validação |

---

## Funcionalidades

### Visualização

- **Tabela Rankeada**: Lista de estratégias ordenadas por OOS Sharpe
- **Filtros**: Por mercado (BR/US), data, campanha
- **Métricas**: Sharpe, PBO, DSR, CAGR, MaxDD
- **Provenance**: Campaign, run, genome hash, git SHA

### Drilldown

Ao clicar em uma estratégia:
- Pipeline de blocos (visualização DSL)
- Equity curve
- Walk-Forward Analysis
- Stress test results
- Provenance completa

---

## Interface

### Tabela Principal

```
┌────────┬────────────────────────────┬─────────┬───────┬───────┬────────┐
│ Rank   │ Strategy Name              │ Sharpe  │ PBO   │ DSR   │ Market │
├────────┼────────────────────────────┼─────────┼───────┼───────┼────────┤
│ 1      │ BR • MomentumQ1 • #ABC123  │ 1.45    │ 0.08  │ 1.85  │ BR     │
│ 2      │ US • ValueDeep • #DEF456   │ 1.38    │ 0.09  │ 1.72  │ US     │
│ 3      │ BR • TrendFollow • #GHI789 │ 1.31    │ 0.11  │ 1.65  │ BR     │
└────────┴────────────────────────────┴─────────┴───────┴───────┴────────┘
```

### Nomenclatura

As estratégias seguem o padrão:

```
{MARKET} • {CAMPAIGN_SHORT} • #{GENOME_HASH_PREFIX}
```

Exemplos:
- `BR • MomentumQ1 • #ABC123`
- `US • ValueDeep • #DEF456`

---

## API Endpoints

### GET /api/omp/hall-of-fame

Retorna lista de estratégias no Hall of Fame.

**Query Parameters:**

| Param | Tipo | Default | Descrição |
|-------|------|---------|-----------|
| `limit` | number | 50 | Máximo de entries |
| `market` | string | null | Filtrar por mercado (br/us) |
| `minSharpe` | number | null | Sharpe mínimo |

**Response:**

```json
{
  "count": 50,
  "entries": [
    {
      "promotionId": "prom_abc123",
      "candidateId": "cand_xyz789",
      "genomeHash": "a1b2c3d4...",
      "strategyName": "BR • MomentumQ1 • #ABC123",
      "campaignId": "camp_001",
      "campaignName": "Momentum Q1 2025",
      "runId": "run_001",
      "market": "br",
      "promotedAt": "2026-01-18T10:00:00Z",
      "metrics": {
        "oosSharpeNet": 1.45,
        "pbo": 0.08,
        "dsr": 1.85,
        "maxDrawdownNet": -0.12,
        "cagrNet": 0.28
      },
      "validation": {
        "stressPassed": 5,
        "stressTotal": 5,
        "gatesPassed": true
      },
      "provenance": {
        "gitSha": "abc123def",
        "configHash": "cfg789xyz",
        "datasetHash": "ds456abc"
      }
    }
  ]
}
```

---

## Database Schema

### Tabela `scg_hall_of_fame`

```sql
CREATE TABLE scg_hall_of_fame (
  promotion_id VARCHAR PRIMARY KEY,
  candidate_id VARCHAR NOT NULL REFERENCES scg_candidates,
  genome_hash VARCHAR NOT NULL,
  strategy_name VARCHAR NOT NULL,
  campaign_id VARCHAR NOT NULL,
  campaign_name VARCHAR,
  run_id VARCHAR NOT NULL,
  market VARCHAR(2) NOT NULL,  -- 'br' or 'us'
  
  -- Metrics
  oos_sharpe_net DECIMAL NOT NULL,
  pbo DECIMAL NOT NULL,
  dsr DECIMAL NOT NULL,
  max_drawdown_net DECIMAL NOT NULL,
  cagr_net DECIMAL NOT NULL,
  
  -- Validation
  stress_passed INTEGER NOT NULL,
  stress_total INTEGER NOT NULL,
  gates_passed BOOLEAN NOT NULL,
  
  -- Provenance
  git_sha VARCHAR,
  config_hash VARCHAR,
  dataset_hash VARCHAR,
  
  -- Metadata
  promoted_at TIMESTAMP NOT NULL DEFAULT NOW(),
  promoted_by VARCHAR,  -- 'omp' or 'manual'
  notes TEXT,
  
  -- Constraints
  UNIQUE(genome_hash)
);

CREATE INDEX idx_hof_market ON scg_hall_of_fame(market);
CREATE INDEX idx_hof_sharpe ON scg_hall_of_fame(oos_sharpe_net DESC);
CREATE INDEX idx_hof_promoted ON scg_hall_of_fame(promoted_at DESC);
```

---

## Sync Automático

O background service `hofSync.js` sincroniza o Hall of Fame:

```javascript
// server/services/hofSync.js
async function syncHallOfFame() {
  // 1. Buscar candidatos que passaram em todos os gates
  const eligibleCandidates = await getEligibleCandidates();
  
  // 2. Verificar se já estão no Hall of Fame
  const newCandidates = filterNew(eligibleCandidates);
  
  // 3. Promover novos candidatos
  for (const candidate of newCandidates) {
    await promoteToHallOfFame(candidate);
  }
}

// Executar a cada 5 minutos
setInterval(syncHallOfFame, 5 * 60 * 1000);
```

---

## Proteções

### Sanity Check (SEV-0)

Antes de promover, o sistema verifica:

1. **Variance Check** - Métricas não podem ter variance ~0
2. **Completeness** - Todos os campos obrigatórios presentes
3. **Consistency** - Métricas consistentes entre si

```javascript
// Endpoint: GET /api/omp/promote-check
{
  "blocked": false,
  "reason": null,
  "details": {
    "sharpeVar": "1.234e-2",
    "pboVar": "5.678e-4",
    "dsrVar": "9.012e-3"
  }
}
```

Se `blocked: true`, promoção é bloqueada e SEV-0 alert é disparado.

---

## Localização no Código

| Componente | Arquivo |
|------------|---------|
| Page | `dashboard/src/pages/HallOfFame.tsx` |
| API Route | `dashboard/server/routes/omp.js` |
| Sync Service | `dashboard/server/services/hofSync.js` |
| Store | `dashboard/src/stores/hofStore.ts` |

---

## Referências

- [Miner Control](miner-control.md) - Controle OMP
- [API Server](api-server.md) - Referência da API REST
- [OMP Specification](../architecture/omp-specification.md) - Arquitetura OMP

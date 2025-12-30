# API Server - Referência

**Versão**: 2.1.0  
**Última Atualização**: 2025-12-30

---

## Visão Geral

O API Server (`server.js`) fornece uma API REST para o Dashboard funcionar em Browser Mode e VPS, sem necessidade do Tauri desktop app.

### Características

- **Express.js** - Server HTTP leve e rápido
- **Neon PostgreSQL** - Banco de dados na nuvem para persistência
- **SSE (Server-Sent Events)** - Atualizações em tempo real com reconnection
- **Event Buffering** - Últimos 100 eventos armazenados para replay
- **Fallback gracioso** - Tenta local artifacts, depois Neon
- **CORS habilitado** - Para desenvolvimento local

---

## Setup

### Variáveis de Ambiente

```bash
DATABASE_URL=postgresql://user:pass@ep-xxx.neon.tech/neondb?sslmode=require
```

### Executar Local

```bash
cd dashboard
node server.js
```

Server inicia em `http://localhost:3001`.

### Com Frontend (Dev)

```bash
# Terminal 1
node server.js

# Terminal 2
npm run dev
```

Frontend em `http://localhost:5173` conecta automaticamente à API.

### Produção (VPS)

Ver [VPS Deployment](vps-deployment.md) para configuração com nginx + PM2.

---

## Endpoints

### Health & Config

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/api/health` | GET | Health check |
| `/api/artifacts-root` | GET | Obter path de artefatos |
| `/api/artifacts-root` | POST | Definir path de artefatos |
| `/api/workspace-root` | GET | Obter path do workspace |
| `/api/workspace-root` | POST | Definir path do workspace |
| `/api/invalidate-cache` | POST | Limpar cache do servidor |

---

### Navegação

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/api/index` | GET | Índice de campanhas |
| `/api/campaigns` | GET | Listar todas as campanhas |
| `/api/campaign/:id` | GET | Detalhes de uma campanha |
| `/api/run/:id` | GET | Detalhes de um run |
| `/api/runs/recent` | GET | Runs recentes |

---

### Candidatos

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/api/candidates/:runId` | GET | Listar candidatos de um run |
| `/api/candidates/recent` | GET | Candidatos recentes |
| `/api/candidate/:id` | GET | Detalhes de um candidato |
| `/api/candidate/:id/pipeline` | GET | Pipeline de blocos |
| `/api/candidate/:id/wfa` | GET | Walk-Forward Analysis |
| `/api/candidate/:id/stress` | GET | Stress test results |
| `/api/candidate/:id/simulated-equity` | GET | Equity curve simulada |
| `/api/cockpit-candidates/:runId` | GET | Candidatos formatados para Cockpit |

---

### Backtest

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/api/backtest/:id` | GET | Timeseries do backtest |

---

### SCG Control

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/api/scg/start` | POST | Iniciar SCG run |
| `/api/scg/progress/:runId` | GET | Progresso do run |
| `/api/scg/stop/:runId` | POST | Parar run |
| `/api/scg/active-runs` | GET | Listar runs ativos |

---

### Real-time (SSE)

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/api/events` | GET | Stream SSE com reconnection support |
| `/api/poll-changes` | GET | Polling de mudanças (fallback) |

---

## Detalhes dos Endpoints

### GET /api/index

Retorna índice de campanhas. Tenta local primeiro, depois Neon.

**Response:**
```json
{
  "schema_version": "1.0",
  "generated_at": "2025-12-29T10:00:00Z",
  "campaigns": [
    {
      "campaign_id": "camp_001",
      "name": "Momentum Q1",
      "tag": "production",
      "status": "completed",
      "runs_count": 3,
      "created_at": "2025-12-28T10:00:00Z"
    }
  ],
  "data_source": "neon"
}
```

---

### GET /api/cockpit-candidates/:runId

Lista candidatos formatados para o Cockpit com display_name calculado.

**Query Parameters:**
| Param | Tipo | Default | Descrição |
|-------|------|---------|-----------|
| `limit` | number | 100 | Máximo de resultados |

**Response:**
```json
[
  {
    "rank": 1,
    "candidate_id": "cand_abc123",
    "candidate_class": "validated",
    "display_name": "Strategy #1 | abc123",
    "oos_sharpe_net": 1.23,
    "oos_cagr_net": 0.15,
    "max_drawdown_net": -0.08,
    "pbo": 0.08,
    "dsr": 1.65,
    "gates_passed": true,
    "stress_passed": true
  }
]
```

---

### GET /api/candidate/:id

Detalhes completos de um candidato.

**Response:**
```json
{
  "candidate_id": "cand_abc123",
  "genome_hash": "a1b2c3d4...",
  "rank": 1,
  "candidate_class": "validated",
  "display_name": "Strategy #1 | abc123",
  
  "oos_sharpe_net": 1.23,
  "oos_cagr_net": 0.15,
  "max_drawdown_net": -0.08,
  "pbo": 0.08,
  "dsr": 1.65,
  
  "gates_passed": true,
  "stress_passed": 5,
  "stress_total": 5,
  
  "provenance": {
    "run_id": "run_001",
    "campaign_id": "camp_001",
    "seed": 42,
    "git_sha": "abc123"
  }
}
```

---

### POST /api/scg/start

Inicia um SCG run e retorna imediatamente (assíncrono).

**Request Body:**
```json
{
  "maxRuntimeSeconds": 900,
  "campaignConfig": "/path/to/config.toml"
}
```

**Response:**
```json
{
  "runId": "run_abc123",
  "status": "started"
}
```

---

### GET /api/scg/progress/:runId

Retorna progresso de um run.

**Response:**
```json
{
  "run_id": "run_abc123",
  "status": "running",
  "percent_complete": 45.5,
  "elapsed_secs": 270,
  "max_runtime_seconds": 900,
  "current_generation": 23,
  "max_generations": 50,
  "candidates_evaluated": 2300,
  "candidates_passing_gates": 45,
  "pareto_size": 25,
  "best_sharpe": 1.23,
  "best_cagr": 0.15,
  "latest_log": "Generation 23 complete...",
  "error_message": null
}
```

---

### GET /api/events (SSE)

Stream de Server-Sent Events com suporte a reconexão.

**Headers de Resposta:**
```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
```

**Suporte a Reconnection:**
```
Last-Event-ID: 123
```

O servidor replay eventos perdidos desde o último ID recebido (buffer de 100 eventos).

**Event Types:**
| Type | Descrição |
|------|-----------|
| `connected` | Conexão estabelecida |
| `ping` | Keep-alive (15s) |
| `scg-progress` | Progresso de SCG run |
| `scg-completed` | Run finalizado |
| `scg-error` | Erro no run |
| `artifact-change` | Arquivo modificado |
| `cache-invalidated` | Cache limpo |

**Exemplo de Evento:**
```
id: 42
data: {"type":"scg-progress","runId":"run_001","percentComplete":50,"currentGeneration":25,"bestSharpe":0.85,"candidatesEvaluated":2500,"timestamp":1703851200000}
```

**Cliente JavaScript:**
```javascript
const sse = new EventSource('/api/events');

sse.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'scg-progress') {
    updateProgress(data);
  }
};

sse.onerror = (e) => {
  console.log('SSE error, will auto-reconnect');
};
```

---

## SSE Implementation

### Event Buffering

O servidor mantém os últimos 100 eventos para replay:

```javascript
const sseEventBuffer = [];
const MAX_BUFFER_SIZE = 100;
let sseEventId = 0;

function broadcastSSE(type, data) {
  const eventId = ++sseEventId;
  const event = { id: eventId, type, ...data, timestamp: Date.now() };
  
  // Buffer for reconnection
  sseEventBuffer.push({ id: eventId, data: event });
  if (sseEventBuffer.length > MAX_BUFFER_SIZE) {
    sseEventBuffer.shift();
  }
  
  // Broadcast to all clients
  sseClients.forEach(res => {
    res.write(`id: ${eventId}\ndata: ${JSON.stringify(event)}\n\n`);
  });
}
```

### Reconnection Support

```javascript
app.get('/api/events', (req, res) => {
  // Check for Last-Event-ID header
  const lastEventId = req.headers['last-event-id'];
  if (lastEventId) {
    const missedEvents = sseEventBuffer.filter(e => e.id > parseInt(lastEventId));
    missedEvents.forEach(e => {
      res.write(`id: ${e.id}\ndata: ${JSON.stringify(e.data)}\n\n`);
    });
  }
  
  // Add to clients
  sseClients.add(res);
});
```

### SCG Progress Broadcasting

Durante execução do SCG, o servidor:

1. Parseia stdout do processo combiner
2. Extrai métricas (geração, sharpe, candidatos)
3. Broadcast via SSE a cada segundo

```javascript
// Parse and broadcast progress
scgProcess.stdout.on('data', (data) => {
  const line = data.toString();
  
  // Extract metrics from output
  const genMatch = line.match(/Generation (\d+)/);
  const sharpeMatch = line.match(/best_sharpe[=:]\s*([\d.]+)/i);
  
  if (genMatch) runState.currentGeneration = parseInt(genMatch[1]);
  if (sharpeMatch) runState.bestSharpe = parseFloat(sharpeMatch[1]);
  
  // Broadcast every second
  broadcastSSE('scg-progress', {
    runId,
    status: 'running',
    currentGeneration: runState.currentGeneration,
    bestSharpe: runState.bestSharpe,
    candidatesEvaluated: runState.candidatesEvaluated,
    percentComplete: calculateProgress(runState),
  });
});
```

---

## Fallback Strategy

O server implementa fallback gracioso:

1. **Local Artifacts First**: Tenta ler de `artifacts/site/*.json`
2. **Neon Fallback**: Se local não existe, consulta Neon DB
3. **Graceful Error**: Retorna erro descritivo se ambos falham

```
Request → Local Artifacts → Neon DB → Error Response
              ✓/✗            ✓/✗          ✗
```

---

## Neon Database

### Tabelas

```sql
-- Campanhas
CREATE TABLE scg_campaigns (
  campaign_id VARCHAR PRIMARY KEY,
  name VARCHAR NOT NULL,
  tag VARCHAR,
  owner VARCHAR,
  status VARCHAR,
  config_hash VARCHAR,
  git_sha VARCHAR,
  git_branch VARCHAR,
  notes TEXT,
  created_at TIMESTAMP
);

-- Runs
CREATE TABLE scg_runs (
  run_id VARCHAR PRIMARY KEY,
  campaign_id VARCHAR REFERENCES scg_campaigns,
  seed INTEGER,
  status VARCHAR,
  started_at TIMESTAMP,
  completed_at TIMESTAMP,
  duration_secs INTEGER,
  generations_completed INTEGER,
  total_evaluations INTEGER,
  best_oos_sharpe_net DECIMAL,
  best_pbo DECIMAL
);

-- Candidatos
CREATE TABLE scg_candidates (
  candidate_id VARCHAR PRIMARY KEY,
  run_id VARCHAR REFERENCES scg_runs,
  genome_hash VARCHAR,
  rank_in_run INTEGER,
  candidate_class VARCHAR,
  oos_sharpe_net DECIMAL,
  oos_cagr_net DECIMAL,
  max_drawdown_net DECIMAL,
  pbo DECIMAL,
  dsr DECIMAL,
  gates_passed BOOLEAN,
  stress_passed INTEGER,
  stress_total INTEGER,
  created_at TIMESTAMP
);
```

---

## Cache

O server mantém cache em memória:

```javascript
const serverCache = {
  index: null,              // Site index
  campaigns: new Map(),     // Campaign details
  runs: new Map(),          // Run details
  lastInvalidated: Date.now()
};
```

### Invalidar

```bash
curl -X POST http://localhost:3001/api/invalidate-cache
```

---

## SCG Process Management

O server spawna e gerencia processos SCG:

```javascript
const scgRuns = new Map(); // runId → process state

// Start
const process = spawn(combinerPath, ['factory', 'run', '--campaign', config]);
scgRuns.set(runId, { process, status: 'running', ... });

// Stop
scgRuns.get(runId).process.kill('SIGTERM');
```

### State Machine

```
idle → starting → running → completed
                         ↘ failed
                         ↘ cancelled (via stop)
```

---

## Errors

| Status | Descrição |
|--------|-----------|
| 400 | Request inválido |
| 404 | Recurso não encontrado |
| 500 | Erro interno |

**Response:**
```json
{
  "error": "Descrição do erro"
}
```

---

## CORS

CORS habilitado para todos os origins em desenvolvimento:

```javascript
app.use(cors());
```

Em produção (VPS), nginx gerencia o proxy reverso.

---

## Localização no Código

| Componente | Arquivo |
|------------|---------|
| Server | `dashboard/server.js` |
| Commands | `dashboard/src/lib/commands.ts` |
| Platform | `dashboard/src/lib/platform.ts` |
| Cockpit Store | `dashboard/src/stores/cockpitStore.ts` |

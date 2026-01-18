# Miner Control

**Versão**: 1.0.0  
**Última Atualização**: 2026-01-18

## Visão Geral

O **Miner Control** é a interface de controle do OMP (Orquestrador de Mineração Perpétua), permitindo gerenciar a mineração contínua 24/7 de estratégias.

---

## Arquitetura

```
┌──────────────────────────────────────────────────────────────────┐
│                        MINER CONTROL                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    DASHBOARD PAGE                           │  │
│  │                  (MinerControl.tsx)                         │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                              │                                    │
│                              ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    OMP STORE                                │  │
│  │                  (ompStore.ts)                              │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                              │                                    │
│                              ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    API ENDPOINTS                            │  │
│  │  /api/omp/status    /api/omp/queue    /api/omp/resources    │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                              │                                    │
│                              ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    OMP DAEMON                               │  │
│  │  (combiner factory run --daemon)                            │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Funcionalidades

### Status do Daemon

| Estado | Cor | Descrição |
|--------|-----|-----------|
| **Running** | Verde | OMP ativo, processando campanhas |
| **Paused** | Amarelo | Temporariamente pausado |
| **Stopped** | Vermelho | Daemon parado |
| **Error** | Vermelho | Erro crítico |

### Controles

| Botão | Ação | Descrição |
|-------|------|-----------|
| **Start** | POST /api/omp/start | Iniciar daemon |
| **Stop** | POST /api/omp/stop | Parar daemon |
| **Pause** | POST /api/omp/pause | Pausar temporariamente |
| **Resume** | POST /api/omp/resume | Retomar execução |

---

## Fila de Campanhas

A interface permite gerenciar a fila de campanhas:

### Visualização

```
┌────┬───────────────────────┬──────────┬──────────┬─────────┐
│ #  │ Campaign              │ Priority │ Status   │ Actions │
├────┼───────────────────────┼──────────┼──────────┼─────────┤
│ 1  │ momentum_q1_2026      │ High     │ Running  │ [⏸][✕] │
│ 2  │ value_explorer        │ Normal   │ Pending  │ [▲][▼][✕] │
│ 3  │ trend_following       │ Low      │ Pending  │ [▲][▼][✕] │
└────┴───────────────────────┴──────────┴──────────┴─────────┘
```

### Ações

| Ação | Descrição |
|------|-----------|
| **Add** | Adicionar campanha à fila |
| **Remove** | Remover campanha da fila |
| **Reorder** | Mudar prioridade na fila |
| **Pause/Resume** | Pausar/retomar campanha específica |

---

## Monitoramento de Recursos

### Métricas em Tempo Real

```
┌─────────────────────────────────────────────────────────────────┐
│                      SYSTEM RESOURCES                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CPU Usage        [████████████░░░░░░░░]  45.2%                 │
│  Memory Usage     [████████████████░░░░]  68.5%                 │
│  Disk Free        [████████████░░░░░░░░]  52.3% (125.6 GB)      │
│                                                                  │
│  Write Rate       12.5 MB/s (últimos 5 min)                      │
│  24h Written      45.8 GB                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Alerts

| Alerta | Threshold | Ação |
|--------|-----------|------|
| **Disk Low** | < 10% | Pausa automática |
| **Memory High** | > 90% | Warning |
| **Write Rate High** | > 50 MB/s | Throttle |

---

## API Endpoints

### Status

**GET /api/omp/status**

```json
{
  "status": "running",
  "currentCampaign": "momentum_q1_2026",
  "currentRun": "run_abc123",
  "generation": 25,
  "totalCandidates": 2500,
  "candidatesInHof": 45,
  "startedAt": "2026-01-18T08:00:00Z",
  "uptime": "10h 30m"
}
```

### Resources

**GET /api/omp/resources**

```json
{
  "cpuUsage": 45.2,
  "memoryUsagePct": 68.5,
  "memoryAvailableMb": 4096,
  "diskFreeGb": 125.6,
  "diskFreePct": 52.3,
  "diskWritten24h": 45.8,
  "writeRateMbPerSec": 12.5,
  "writeAcceleration": 0.025,
  "canStartCampaign": true
}
```

### Queue

**GET /api/omp/queue**

```json
{
  "queue": [
    {
      "id": "q_001",
      "campaignId": "camp_momentum_q1",
      "campaignName": "Momentum Q1 2026",
      "priority": "high",
      "status": "running",
      "addedAt": "2026-01-18T07:00:00Z",
      "startedAt": "2026-01-18T08:00:00Z"
    },
    {
      "id": "q_002",
      "campaignId": "camp_value_explorer",
      "campaignName": "Value Explorer",
      "priority": "normal",
      "status": "pending",
      "addedAt": "2026-01-18T09:00:00Z"
    }
  ],
  "totalPending": 2
}
```

**POST /api/omp/queue**

```json
{
  "campaignId": "camp_new",
  "priority": "high"
}
```

---

## Configuração

### Arquivo de Configuração

`dashboard/omp_config.toml`:

```toml
[daemon]
enabled = true
auto_start = true
max_concurrent = 1

[resources]
max_cpu_pct = 80
max_memory_pct = 85
min_disk_gb = 50

[queue]
max_pending = 10
default_priority = "normal"

[promotion]
auto_promote = true
min_sharpe = 1.0
max_pbo = 0.10
```

### Fila de Campanhas

`dashboard/campaign_queue.json`:

```json
{
  "version": "1.0",
  "queue": [
    {
      "id": "q_001",
      "campaign_config": "configs/campaigns/momentum_q1.toml",
      "priority": "high",
      "status": "pending"
    }
  ]
}
```

---

## Workflow

### Ciclo de Mineração

```
1. Daemon inicia
   │
2. Verifica recursos
   ├── OK → Continua
   └── Low → Pausa
   │
3. Pega próxima campanha da fila
   │
4. Executa combiner factory run
   │
5. Monitora progresso
   │
6. Ao finalizar:
   ├── Promove candidatos para Hall of Fame
   └── Marca campanha como completa
   │
7. Volta para step 2
```

### Estados

```
┌─────────┐    start    ┌─────────┐
│ Stopped │ ──────────► │ Running │
└─────────┘             └────┬────┘
     ▲                       │
     │ stop                  │ pause
     │                       ▼
     │               ┌───────────┐
     └────────────── │  Paused   │
           stop      └───────────┘
                           │ resume
                           ▼
                     ┌─────────┐
                     │ Running │
                     └─────────┘
```

---

## Localização no Código

| Componente | Arquivo |
|------------|---------|
| Page | `dashboard/src/pages/MinerControl.tsx` |
| Store | `dashboard/src/stores/ompStore.ts` |
| API Routes | `dashboard/server/routes/omp.js` |
| Config | `dashboard/omp_config.toml` |
| Queue | `dashboard/campaign_queue.json` |

---

## Referências

- [Hall of Fame](hall-of-fame.md) - Estratégias promovidas
- [API Server](api-server.md) - Referência da API REST
- [OMP Specification](../architecture/omp-specification.md) - Arquitetura OMP
- [VPS Deployment](vps-deployment.md) - DEFERRED (historical reference)

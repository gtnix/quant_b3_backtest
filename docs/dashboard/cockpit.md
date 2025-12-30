# Cockpit - Painel de Controle SCG

**Versão**: 2.0.0  
**Última Atualização**: 2025-12-29

---

## Visão Geral

O Cockpit é o painel central para orquestração de runs do Sistema Combinador Generativo (SCG). Permite configurar, executar e monitorar descobertas de estratégias com interface intuitiva.

### Funcionalidades

- **Presets de Execução** - Perfis pré-configurados para diferentes casos de uso
- **Compute Budget** - Controle de tempo e intensidade de CPU
- **Risk & Robustness Gates** - Thresholds configuráveis para validação
- **Ranking Methods** - Métodos de ordenação de candidatos
- **Live Progress** - Monitoramento em tempo real via SSE
- **Status Badge** - Indicador "Live" (verde) ou "Offline" (vermelho)
- **Top Strategies Table** - Resultados rankeados com drilldown
- **Error Handling** - Estados de erro com opção de retry

---

## Status Badge

O header do Cockpit exibe um badge de status de conexão:

| Badge | Cor | Significado |
|-------|-----|-------------|
| **Live** | Verde | SSE conectado, updates em tempo real |
| **Offline** | Vermelho | SSE desconectado, usando polling |

### Funcionamento

1. **SSE Primeiro**: Tenta conectar via Server-Sent Events
2. **Fallback para Polling**: Se SSE falhar 3x, ativa polling automático
3. **Reconexão Automática**: SSE reconecta com Last-Event-ID para replay

```typescript
// cockpitStore.ts
subscribeToProgress: () => {
  const sse = createSSEConnection(
    (event) => {
      if (event.type === 'scg-progress') {
        set({ progress: parseProgress(event) });
      }
    },
    (error) => {
      // Fallback to polling after 3 failures
      if (sseFailCount >= 3) {
        startPolling();
      }
    },
    () => set({ sseConnected: true }) // On reconnect
  );
}
```

---

## Presets de Execução

O Cockpit oferece três presets que configuram automaticamente todos os parâmetros:

| Preset | Tempo | População | Gerações | Workers | Uso |
|--------|-------|-----------|----------|---------|-----|
| **Rapid** | 3 min | 80 | 30 | 8 | Debug, smoke test |
| **Institutional** | 15 min | 100 | 50 | 8 | Produção |
| **Exhaustive** | 1 hora | 200 | 100 | 16 | Máxima exploração |

### Rapid (3 min)

```typescript
{
  maxRuntimeSeconds: 180,
  populationSize: 80,
  maxGenerations: 30,
  convergenceGenerations: 8,
  workers: 8,
  seeds: [42],
  minOosSharpeNet: 0.3,
  maxPbo: 0.25,
  minStressPassed: 2,
}
```

Uso: Validação rápida de configuração, desenvolvimento.

### Institutional (15 min)

```typescript
{
  maxRuntimeSeconds: 900,
  populationSize: 100,
  maxGenerations: 50,
  convergenceGenerations: 10,
  workers: 8,
  seeds: [42, 123, 456],
  minOosSharpeNet: 0.5,
  maxPbo: 0.15,
  minStressPassed: 4,
}
```

Uso: Produção, descoberta de estratégias robustas.

### Exhaustive (1 hora)

```typescript
{
  maxRuntimeSeconds: 3600,
  populationSize: 200,
  maxGenerations: 100,
  convergenceGenerations: 15,
  workers: 16,
  seeds: [42, 123, 456, 789, 1011],
  minOosSharpeNet: 0.5,
  maxPbo: 0.15,
  minStressPassed: 4,
}
```

Uso: Exploração completa do espaço de estratégias.

---

## Time Presets

Além dos presets completos, há presets de tempo rápido:

| Preset | Segundos | Uso |
|--------|----------|-----|
| 30s | 30 | Debug rápido |
| 1 min | 60 | Smoke test |
| 3 min | 180 | Rapid |
| 10 min | 600 | Análise básica |
| 15 min | 900 | Institutional |
| 30 min | 1800 | Análise profunda |
| 1 hora | 3600 | Exhaustive |
| 2 horas | 7200 | Overnight |

---

## Compute Budget

### Time Slider

Controla o tempo máximo de execução.

### Intensity (Workers)

Controla paralelismo de CPU:

| Nível | Workers | Descrição |
|-------|---------|-----------|
| Baixa | 2 | Mínimo impacto no sistema |
| Média | 4 | Equilíbrio performance/recursos |
| Alta | 8 | Máxima velocidade |
| Máxima | 16 | Todos os cores |

---

## Risk & Robustness Gates

Gates são thresholds que filtram estratégias fracas. Disponível em dois modos:

### Basic Mode

Mostra apenas toggle de stress testing. Gates usam valores defaults:

- Sharpe OOS NET ≥ 0.5
- PBO ≤ 15%
- Stress tests ≥ 4/8

### Advanced Mode

Permite ajuste fino de todos os gates:

| Gate | Default | Range | Descrição |
|------|---------|-------|-----------|
| `minOosSharpeNet` | 0.5 | 0.0 - 2.0 | Sharpe mínimo Out-of-Sample NET |
| `maxPbo` | 0.15 | 0.0 - 0.5 | Probability of Backtest Overfitting máximo |
| `minStressPassed` | 4 | 0 - 8 | Mínimo de cenários stress passados |
| `stressTestingEnabled` | true | bool | Habilitar/desabilitar stress tests |

### Explicação dos Gates

**minOosSharpeNet**: Sharpe Ratio mínimo no período Out-of-Sample, já descontando custos de transação. Valores acima de 0.5 indicam edge estatístico consistente.

**maxPbo**: Probability of Backtest Overfitting. Mede a probabilidade da estratégia ter performado bem por sorte. Valores abaixo de 0.15 indicam baixo risco de overfitting.

**minStressPassed**: Número mínimo de cenários de stress que a estratégia deve sobreviver (ex: crise 2008, COVID, volatilidade extrema).

---

## Ranking Methods

O Cockpit oferece quatro métodos para ordenar candidatos:

### Institutional (Recomendado)

Multi-critério ponderado usado por fundos quantitativos:

```typescript
score = sharpeScore * 40%     // Sharpe OOS (max 1.0 = full points)
      + pboScore * 25%        // 1 - (PBO / 0.5)
      + stressScore * 20%     // Passou stress = 20 pts
      + gatesScore * 15%      // Passou gates = 15 pts
```

### Pareto

Fronteira eficiente: estratégias não-dominadas em Sharpe vs Drawdown.

```typescript
score = oosSharpe - |maxDrawdown| * 0.1
```

### Sharpe Puro

Ordena apenas por Sharpe OOS NET. Simples mas pode premiar overfitting.

```typescript
score = oosSharpeNet
```

### Risk-Adjusted

Sharpe dividido por drawdown máximo. Penaliza estratégias com quedas grandes.

```typescript
score = oosSharpe / |maxDrawdown| * 100
```

---

## Controles

### Start/Stop

- **START**: Inicia SCG run com configuração atual
- **STOP**: Para run em andamento (graceful shutdown)

### Progress Display

Durante a execução, exibe:

| Métrica | Descrição |
|---------|-----------|
| **Progresso %** | Percentual concluído |
| **Tempo** | Elapsed / Max runtime |
| **Geração** | Current / Max gerações |
| **Melhor Sharpe** | Maior Sharpe encontrado |
| **Candidatos** | Total avaliados |
| **Latest Log** | Últimas mensagens |

### Barra de Progresso

Barra animada com:
- Gradiente cyan → emerald
- Shimmer effect
- Pulse dot na ponta
- Tick marks por etapa

---

## Error Handling

### Estados de Erro

O Cockpit exibe mensagens claras quando ocorrem erros:

| Estado | UI | Ação |
|--------|----|----- |
| `failed` | Badge vermelho + mensagem | Botão "Tentar Novamente" |
| SSE disconnect | Badge "Offline" | Polling automático |
| API error | Toast notification | Retry automático |

### StatusBadge Component

```typescript
function StatusBadge({ status }: { status: RunStatus }) {
  const config = {
    idle: { text: 'Parado', color: 'slate' },
    starting: { text: 'Iniciando...', color: 'amber' },
    running: { text: 'Executando', color: 'emerald' },
    stopping: { text: 'Parando...', color: 'amber' },
    completed: { text: 'Concluído', color: 'cyan' },
    failed: { text: 'Falhou', color: 'red' },
    cancelled: { text: 'Cancelado', color: 'slate' },
  };
  return <Badge {...config[status]} />;
}
```

---

## Top Strategies Table

Tabela de resultados com colunas:

| Coluna | Descrição |
|--------|-----------|
| **#** | Rank |
| **Estratégia** | Nome (clicável → backtest) |
| **Sharpe** | OOS NET (verde ≥1.0, cyan ≥0.5) |
| **PBO** | % (verde ≤10%, cyan ≤15%) |
| **MaxDD** | Drawdown máximo (vermelho) |
| **Gates** | ✓ ou ✗ |
| **Por que no topo?** | Razões do ranking |

### Explicação de Rank

Cada candidato mostra até 3 razões:

- "Sharpe excelente (≥1.0)"
- "Sharpe bom (≥0.7)"
- "Baixo risco de overfitting (PBO ≤10%)"
- "PBO aceitável (≤15%)"
- "Passou testes de stress"
- "Passou todos os gates"
- "Drawdown controlado (<15%)"
- "DSR forte (>1.0)"

---

## Real-time Updates

### SSE Connection

O Cockpit usa Server-Sent Events para updates em tempo real:

```typescript
// commands.ts
export function createSSEConnection(
  onEvent: (event: SSEEvent) => void,
  onError?: (error: Event) => void,
  onReconnect?: () => void
): EventSource | null {
  const eventSource = new EventSource(config.sseEndpoint);
  
  eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    onEvent(data);
    
    // Track successful events
    sseHealthy = true;
    consecutiveErrors = 0;
  };
  
  eventSource.onerror = () => {
    consecutiveErrors++;
    if (consecutiveErrors >= 3) {
      sseHealthy = false;
    }
    onError?.(error);
  };
  
  return eventSource;
}
```

### Polling Fallback

Se SSE falhar, ativa polling automático:

```typescript
// cockpitStore.ts
const pollInterval = window.setInterval(() => {
  if (!sseHealthy && runStatus === 'running') {
    pollProgress();
  }
}, 2000); // Poll every 2 seconds
```

---

## View Mode

Toggle entre Basic e Advanced:

### Basic Mode

- Preset selector
- Time slider
- Stress toggle
- Ranking selector
- Controles e resultados

### Advanced Mode

Adiciona:
- Intensity selector (workers)
- Seeds configuration
- Todos os gates com sliders
- Configurações detalhadas

---

## State Management (cockpitStore)

### Tipos

```typescript
type RunStatus = 
  | 'idle' 
  | 'starting' 
  | 'running' 
  | 'stopping' 
  | 'completed' 
  | 'failed' 
  | 'cancelled';

interface RunProgress {
  runId: string;
  status: RunStatus;
  currentGeneration: number;
  maxGenerations: number;
  elapsedSeconds: number;
  maxRuntimeSeconds: number;
  bestSharpe: number | null;
  bestCagr: number | null;
  candidatesEvaluated: number;
  candidatesPassingGates: number;
  paretoSize: number;
  latestLog: string | null;
  percentComplete: number;
  errorMessage: string | null;
}

interface RankedCandidate {
  rank: number;
  candidateId: string;
  candidateClass: string;
  displayName: string;
  oosSharpeNet: number;
  oosCagrNet: number;
  maxDrawdownNet: number;
  pbo: number;
  dsr: number;
  gatesPassed: boolean;
  stressPassed: boolean;
  dataIntegrityOk: boolean;
  rankReasons: string[];
  score: number;
}
```

### Actions

```typescript
// Configuração
setPreset: (preset: PresetKey) => void;
updateConfig: (partial: Partial<CockpitConfig>) => void;
setViewMode: (mode: ViewMode) => void;
setRankingMethod: (method: RankingMethodKey) => void;

// Controle de run
startRun: () => Promise<void>;
stopRun: () => Promise<void>;
pollProgress: () => Promise<void>;

// Resultados
loadTopCandidates: (runId: string) => Promise<void>;
selectCandidate: (candidateId: string | null) => void;

// Subscriptions
subscribeToProgress: () => () => void;

// SSE status
setSseConnected: (connected: boolean) => void;
```

---

## Integração com Backend

### Tauri Mode

```typescript
await invoke('start_scg_run', { config });
await invoke('get_run_status', { runId });
await listen('scg-progress', handler);
```

### Browser Mode

```typescript
await fetch('/api/scg/start', { method: 'POST', body: JSON.stringify(config) });
await fetch(`/api/scg/progress/${runId}`);
// SSE para real-time updates
const sse = new EventSource('/api/events');
```

---

## Keyboard Shortcuts

| Tecla | Ação |
|-------|------|
| `?` | Abrir glossário |
| `Esc` | Fechar overlays |

---

## Localização no Código

| Componente | Arquivo |
|------------|---------|
| Page | `src/pages/Cockpit.tsx` |
| Store | `src/stores/cockpitStore.ts` |
| Config | `src/config/defaults.ts` |
| Commands | `src/lib/commands.ts` |
| Platform | `src/lib/platform.ts` |
| Tooltips | `src/components/ui/TooltipInfo.tsx` |

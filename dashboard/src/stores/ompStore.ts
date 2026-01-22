/**
 * OMP Store - Zustand state management for Orquestrador de Mineração Perpétua
 * 
 * Manages mining daemon state, queue, stats, and real-time updates via SSE.
 */

import { create } from 'zustand';
import { cmd, createSSEConnection } from '../lib/commands';
import { platform, config as platformConfig } from '../lib/platform';

// =============================================================================
// TYPES
// =============================================================================

export type OmpStatus = 'offline' | 'running' | 'paused' | 'draining';

export interface OmpResources {
  cpuUsage: number;
  memoryUsagePct: number;
  memoryAvailableMb: number;
  diskFreeGb: number;
  diskTotalGb?: number;
  diskFreePct?: number;
  diskWritten24h?: number;
  writeRateMbPerSec?: number;
  writeAcceleration?: number;
  estimatedTimeToLimitHours?: number | null;
  shouldAutoStop?: boolean;
  canStartCampaign: boolean;
}

export interface MarketStats {
  generation: number;
  bestSharpe: number | null;
  candidates: number;
  hofSize: number;
  paretoSize?: number;
  meanSharpe?: number;
  diversity?: number;
  convergenceRate?: number;
  stagnation?: number;
  validatedCount?: number;
  validatedTotal?: number;
}

export interface CurrentCampaign {
  campaignId: string;
  campaignName: string;
  runId: string;
  experimentId?: string;      // NEW: Canonical experiment identifier
  artifactsPath?: string;     // NEW: Path to outputs (output/scg/<experimentId>/)
  market: string;
  markets?: string[];         // NEW: Array of active markets
  status: string;
  elapsedSeconds: number;
  elapsedSecs?: number;
  currentGeneration: number;
  bestSharpe: number | null;
  candidatesEvaluated: number;
  external?: boolean;
  pid?: number;
  mode?: string;
  // Evolution metrics
  paretoSize?: number;
  validatedCount?: number;
  validatedTotal?: number;
  hofSize?: number;
  meanSharpe?: number;
  diversity?: number;
  convergenceRate?: number;
  stagnation?: number;
  // Per-market stats
  marketStats?: Record<string, MarketStats>;
}

export interface OmpStats {
  candidates: {
    last24h: number;
    last7d: number;
    total: number;
  };
  promotions: {
    last24h: number;
    last7d: number;
    total: number;
  };
  campaigns: {
    completed: number;
    failed: number;
  };
  throughput: {
    candidatesPerMin: number;
  };
  lastPromotion: string | null;
}

export interface QueuedCampaign {
  id: string;
  name: string;
  config_path: string;
  market: 'br' | 'us';
  priority: number;
  enabled: boolean;
  repeat: boolean;
  tags: string[];
  created_at: string;
}

export interface CampaignQueue {
  version: string;
  updated_at: string;
  campaigns: QueuedCampaign[];
}

export interface HallOfFameEntry {
  promotionId: string;
  candidateId: string;
  genomeHash: string;
  strategyName: string;
  campaignId: string;
  campaignName: string;
  runId: string;
  market: string;
  promotedAt: string;
  metrics: {
    oosSharpeNet: number;
    pbo: number;
    dsr: number;
    maxDrawdownNet: number;
    cagrNet: number;
  };
  validation: {
    stressPassed: number;
    stressTotal: number;
    gatesPassed: boolean;
  };
  provenance: {
    gitSha: string;
    configHash: string;
    datasetHash?: string;
  };
  notes: string;
  // Strategy Catalog fields
  strategyId?: string;
  strategyFamily?: string;
  strategyVariant?: string;
  strategyHypothesis?: string;
  // Strategy Identity (full traceability)
  identity?: {
    strategy_id: string;
    strategy_name: string;
    market: string;
    universe: string;
    timeframe: string;
    strategy_type: string;
    strategy_family: string;
    blocks: Array<{ block_type: string; block_id: string; key_params: Record<string, string> }>;
    effective_parameters: Record<string, number>;
    entry_rules: string;
    exit_rules: string;
    slippage_bps: number;
    commission_rate: number;
    generation: number;
    version: string;
  };
}

export interface OmpConfig {
  loopIntervalSecs: number;
  markets: {
    br?: { enabled: boolean; name: string; universe: string };
    us?: { enabled: boolean; name: string; universe: string };
  };
  promotion: {
    min_oos_sharpe_net?: number;
    max_pbo?: number;
    min_dsr?: number;
    max_drawdown_net?: number;
  };
}

export interface ActivityLogEntry {
  id: string;
  timestamp: string;
  level: 'info' | 'success' | 'warning' | 'error';
  message: string;
  campaignId?: string;
  runId?: string;
  generation?: number;
  candidates?: number;
  bestSharpe?: number;
}

export interface PerformanceMetrics {
  current_run: {
    run_id: string;
    evaluations_per_second: number;
    cache_hit_rate: number;
    throughput_genomes_per_min: number;
    current_generation: number;
    best_sharpe: number | null;
    candidates_evaluated: number;
    elapsed_seconds: number;
    memory_mb: number;
  } | null;
  system: {
    cpu_usage: number;
    memory_usage_pct: number;
    memory_available_mb: number;
    disk_free_gb: number;
  };
  totals: {
    candidates_generated: number;
    backtests_executed: number;
    promotions: number;
  };
  historical?: {
    candidates_24h: number;
    candidates_1h: number;
    promotions_24h: number;
    avg_throughput_per_min: number;
    recent_runs?: Array<{
      run_id: string;
      campaign_name: string;
      status: string;
      duration_secs: number;
      evaluations: number;
    }>;
    best_candidate_24h?: {
      candidate_id: string;
      sharpe: number;
      cagr: number;
    } | null;
  } | null;
}

export interface OmpState {
  // Connection
  sseConnected: boolean;
  lastError: string | null;
  
  // Status
  status: OmpStatus;
  startedAt: string | null;
  lastLoop: string | null;
  loopCount: number;
  queueLength: number;
  lastPromotion: string | null;
  
  // Current campaign
  currentCampaign: CurrentCampaign | null;
  
  // Resources
  resources: OmpResources;
  
  // Stats
  stats: OmpStats | null;
  
  // Queue
  queue: CampaignQueue | null;
  
  // Hall of Fame
  hallOfFame: HallOfFameEntry[];
  hallOfFameLoading: boolean;
  
  // Config
  config: OmpConfig | null;
  
  // Activity Log
  activityLog: ActivityLogEntry[];
  
  // Performance Metrics
  performance: PerformanceMetrics | null;
  
  // History arrays for real-time charts (120 samples = 4 min at 2s interval)
  throughputHistory: number[];
  cpuHistory: number[];
  memoryHistory: number[];
  sharpeHistory: number[];
  generationHistory: number[];
  candidatesHistory: number[];
  // Evolution metrics history
  diversityHistory: number[];
  meanSharpeHistory: number[];
  paretoSizeHistory: number[];
  convergenceRateHistory: number[];
  stagnationHistory: number[];
  
  // Actions
  fetchStatus: () => Promise<void>;
  fetchStats: () => Promise<void>;
  fetchQueue: () => Promise<void>;
  fetchHallOfFame: (limit?: number, market?: string) => Promise<void>;
  fetchConfig: () => Promise<void>;
  fetchActivityLog: (limit?: number) => Promise<void>;
  fetchPerformance: () => Promise<void>;
  
  start: () => Promise<boolean>;
  stop: () => Promise<boolean>;
  pause: () => Promise<boolean>;
  resume: () => Promise<boolean>;
  cleanup: () => Promise<{ success: boolean; results: { folders: boolean; database: boolean }; message: string } | null>;
  
  addToQueue: (campaign: Partial<QueuedCampaign>) => Promise<QueuedCampaign | null>;
  updateQueueItem: (id: string, updates: Partial<QueuedCampaign>) => Promise<boolean>;
  removeFromQueue: (id: string) => Promise<boolean>;
  
  subscribeToUpdates: () => () => void;
}

// =============================================================================
// STORE
// =============================================================================

export const useOmpStore = create<OmpState>((set, get) => ({
  // Initial state
  sseConnected: false,
  lastError: null,
  
  status: 'offline',
  startedAt: null,
  lastLoop: null,
  loopCount: 0,
  queueLength: 0,
  lastPromotion: null,
  
  currentCampaign: null,
  
  resources: {
    cpuUsage: 0,
    memoryUsagePct: 0,
    memoryAvailableMb: 0,
    diskFreeGb: 0,
    canStartCampaign: false,
  },
  
  stats: null,
  queue: null,
  hallOfFame: [],
  hallOfFameLoading: false,
  config: null,
  activityLog: [],
  performance: null,
  throughputHistory: [],
  cpuHistory: [],
  memoryHistory: [],
  sharpeHistory: [],
  generationHistory: [],
  candidatesHistory: [],
  diversityHistory: [],
  meanSharpeHistory: [],
  paretoSizeHistory: [],
  convergenceRateHistory: [],
  stagnationHistory: [],
  
  // ==========================================================================
  // FETCH ACTIONS
  // ==========================================================================
  
  fetchStatus: async () => {
    try {
      const response = await fetch(`${platformConfig.apiBase}/omp/status`, { credentials: 'same-origin' });
      if (!response.ok) throw new Error('Failed to fetch status');
      const data = await response.json();
      
      // Normalize currentCampaign fields (API uses elapsedSecs, frontend uses elapsedSeconds)
      const currentCampaign = data.currentCampaign ? {
        ...data.currentCampaign,
        elapsedSeconds: data.currentCampaign.elapsedSecs || data.currentCampaign.elapsedSeconds || 0,
        candidatesEvaluated: data.currentCampaign.candidatesEvaluated || 0,
      } : null;
      
      set({
        status: data.status || 'offline',
        startedAt: data.startedAt,
        lastLoop: data.lastLoop,
        loopCount: data.loopCount || 0,
        queueLength: data.queueLength || 0,
        lastPromotion: data.lastPromotion,
        currentCampaign,
        resources: data.resources || get().resources,
        config: data.config,
        lastError: null,
      });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : 'Failed to fetch status' });
    }
  },
  
  fetchStats: async () => {
    try {
      const response = await fetch(`${platformConfig.apiBase}/omp/stats`, { credentials: 'same-origin' });
      if (!response.ok) throw new Error('Failed to fetch stats');
      const data = await response.json();
      // Only update if data actually changed (avoid unnecessary re-renders)
      const current = get().stats;
      const hasChanged = !current || 
        current.candidates?.last24h !== data.candidates?.last24h ||
        current.candidates?.last7d !== data.candidates?.last7d ||
        current.promotions?.last24h !== data.promotions?.last24h ||
        current.promotions?.total !== data.promotions?.total ||
        current.throughput?.candidatesPerMin !== data.throughput?.candidatesPerMin;
      if (hasChanged) {
        set({ stats: data, lastError: null });
      }
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : 'Failed to fetch stats' });
    }
  },
  
  fetchQueue: async () => {
    try {
      const response = await fetch(`${platformConfig.apiBase}/omp/queue`, { credentials: 'same-origin' });
      if (!response.ok) throw new Error('Failed to fetch queue');
      const data = await response.json();
      set({ queue: data, queueLength: data.campaigns?.filter((c: QueuedCampaign) => c.enabled).length || 0, lastError: null });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : 'Failed to fetch queue' });
    }
  },
  
  fetchHallOfFame: async (limit = 50, market?: string) => {
    set({ hallOfFameLoading: true });
    try {
      const params = new URLSearchParams({ limit: String(limit) });
      if (market) params.append('market', market);
      
      const response = await fetch(`${platformConfig.apiBase}/omp/hall-of-fame?${params}`, { credentials: 'same-origin' });
      if (!response.ok) throw new Error('Failed to fetch hall of fame');
      const data = await response.json();
      set({ hallOfFame: data.entries || [], hallOfFameLoading: false, lastError: null });
    } catch (err) {
      set({ hallOfFameLoading: false, lastError: err instanceof Error ? err.message : 'Failed to fetch hall of fame' });
    }
  },
  
  fetchConfig: async () => {
    try {
      const response = await fetch(`${platformConfig.apiBase}/omp/config`, { credentials: 'same-origin' });
      if (!response.ok) throw new Error('Failed to fetch config');
      const data = await response.json();
      set({ config: data, lastError: null });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : 'Failed to fetch config' });
    }
  },
  
  fetchActivityLog: async (limit = 50) => {
    try {
      const response = await fetch(`${platformConfig.apiBase}/omp/activity?limit=${limit}`, { credentials: 'same-origin' });
      if (!response.ok) throw new Error('Failed to fetch activity log');
      const data = await response.json();
      set({ activityLog: data.logs || [], lastError: null });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : 'Failed to fetch activity log' });
    }
  },
  
  fetchPerformance: async () => {
    try {
      const response = await fetch(`${platformConfig.apiBase}/omp/performance`, { credentials: 'same-origin' });
      if (!response.ok) throw new Error('Failed to fetch performance');
      const data = await response.json();
      
      const prev = get();
      const maxHistory = 120; // 4 min at 2s interval
      
      // Update all history arrays for real-time charts
      // Use evaluations_per_second * 60 to get throughput per minute
      const evalPerSec = data.current_run?.evaluations_per_second || 0;
      const throughputHistory = [...prev.throughputHistory, evalPerSec * 60].slice(-maxHistory);
      const cpuHistory = [...prev.cpuHistory, data.system?.cpu_usage || 0].slice(-maxHistory);
      const memoryHistory = [...prev.memoryHistory, data.system?.memory_usage_pct || 0].slice(-maxHistory);
      const sharpeHistory = [...prev.sharpeHistory, data.current_run?.best_sharpe || prev.sharpeHistory[prev.sharpeHistory.length - 1] || 0].slice(-maxHistory);
      const generationHistory = [...prev.generationHistory, data.current_run?.current_generation || 0].slice(-maxHistory);
      const candidatesHistory = [...prev.candidatesHistory, data.current_run?.candidates_evaluated || 0].slice(-maxHistory);
      
      // Evolution metrics history
      const diversityHistory = [...prev.diversityHistory, data.current_run?.diversity || 0].slice(-maxHistory);
      const meanSharpeHistory = [...prev.meanSharpeHistory, data.current_run?.mean_sharpe || 0].slice(-maxHistory);
      const paretoSizeHistory = [...prev.paretoSizeHistory, data.current_run?.pareto_size || 0].slice(-maxHistory);
      const convergenceRateHistory = [...prev.convergenceRateHistory, data.current_run?.convergence_rate || 0].slice(-maxHistory);
      const stagnationHistory = [...prev.stagnationHistory, data.current_run?.stagnation || 0].slice(-maxHistory);
      
      set({ 
        performance: data, 
        throughputHistory,
        cpuHistory,
        memoryHistory,
        sharpeHistory,
        generationHistory,
        candidatesHistory,
        diversityHistory,
        meanSharpeHistory,
        paretoSizeHistory,
        convergenceRateHistory,
        stagnationHistory,
        lastError: null 
      });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : 'Failed to fetch performance' });
    }
  },
  
  // ==========================================================================
  // CONTROL ACTIONS
  // ==========================================================================
  
  start: async () => {
    try {
      const response = await fetch(`${platformConfig.apiBase}/omp/start`, { method: 'POST', credentials: 'same-origin' });
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || 'Failed to start');
      }
      await get().fetchStatus();
      return true;
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : 'Failed to start' });
      return false;
    }
  },
  
  stop: async () => {
    try {
      const response = await fetch(`${platformConfig.apiBase}/omp/stop`, { method: 'POST', credentials: 'same-origin' });
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || 'Failed to stop');
      }
      await get().fetchStatus();
      return true;
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : 'Failed to stop' });
      return false;
    }
  },
  
  pause: async () => {
    try {
      const response = await fetch(`${platformConfig.apiBase}/omp/pause`, { method: 'POST', credentials: 'same-origin' });
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || 'Failed to pause');
      }
      await get().fetchStatus();
      return true;
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : 'Failed to pause' });
      return false;
    }
  },
  
  resume: async () => {
    try {
      const response = await fetch(`${platformConfig.apiBase}/omp/resume`, { method: 'POST', credentials: 'same-origin' });
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || 'Failed to resume');
      }
      await get().fetchStatus();
      return true;
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : 'Failed to resume' });
      return false;
    }
  },
  
  cleanup: async () => {
    try {
      const response = await fetch(`${platformConfig.apiBase}/omp/cleanup`, { method: 'POST', credentials: 'same-origin' });
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || 'Failed to cleanup');
      }
      const result = await response.json();
      await get().fetchStatus();
      await get().fetchStats();
      return result;
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : 'Failed to cleanup' });
      return null;
    }
  },
  
  // ==========================================================================
  // QUEUE ACTIONS
  // ==========================================================================
  
  addToQueue: async (campaign) => {
    try {
      const response = await fetch(`${platformConfig.apiBase}/omp/queue`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(campaign),
        credentials: 'same-origin',
      });
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || 'Failed to add to queue');
      }
      const newCampaign = await response.json();
      await get().fetchQueue();
      return newCampaign;
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : 'Failed to add to queue' });
      return null;
    }
  },
  
  updateQueueItem: async (id, updates) => {
    try {
      const response = await fetch(`${platformConfig.apiBase}/omp/queue/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates),
        credentials: 'same-origin',
      });
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || 'Failed to update');
      }
      await get().fetchQueue();
      return true;
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : 'Failed to update queue item' });
      return false;
    }
  },
  
  removeFromQueue: async (id) => {
    try {
      const response = await fetch(`${platformConfig.apiBase}/omp/queue/${id}`, {
        method: 'DELETE',
        credentials: 'same-origin',
      });
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || 'Failed to remove');
      }
      await get().fetchQueue();
      return true;
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : 'Failed to remove from queue' });
      return false;
    }
  },
  
  // ==========================================================================
  // SSE SUBSCRIPTION
  // ==========================================================================
  
  subscribeToUpdates: () => {
    // Browser mode - use SSE, Desktop mode - use polling
    if (platform.isBrowser) {
      const sse = createSSEConnection(
        (event) => {
          // Set connected on any event - KEEP previous state on reconnection
          if (!get().sseConnected) {
            set({ sseConnected: true, lastError: null });
          }
          
          if (event.type === 'connected' || event.type === 'ping') {
            set({ sseConnected: true, lastError: null });
          } else if (event.type === 'omp-status') {
            const data = event.data as Record<string, unknown>;
            // INCREMENTAL UPDATE: Only update fields that have new values
            // Keep previous values if new ones are null/undefined
            const prev = get();
            // Normalize currentCampaign fields
            const rawCampaign = data.currentCampaign as Record<string, unknown> | null;
            const currentCampaign = rawCampaign ? {
              ...rawCampaign,
              elapsedSeconds: (rawCampaign.elapsedSecs as number) || (rawCampaign.elapsedSeconds as number) || 0,
              candidatesEvaluated: (rawCampaign.candidatesEvaluated as number) || 0,
            } as CurrentCampaign : null;
            set({
              status: (data.status as OmpStatus) || prev.status || 'offline',
              startedAt: (data.startedAt as string | null) ?? prev.startedAt,
              lastLoop: (data.lastLoop as string | null) ?? prev.lastLoop,
              loopCount: typeof data.loopCount === 'number' ? data.loopCount : prev.loopCount,
              queueLength: typeof data.queueLength === 'number' ? data.queueLength : prev.queueLength,
              lastPromotion: (data.lastPromotion as string | null) ?? prev.lastPromotion,
              currentCampaign: currentCampaign ?? prev.currentCampaign,
              resources: data.resources ? { ...prev.resources, ...(data.resources as OmpResources) } : prev.resources,
            });
          } else if (event.type === 'omp-started') {
            set({ status: 'running', startedAt: (event as { startedAt?: string }).startedAt || new Date().toISOString() });
          } else if (event.type === 'omp-stopped') {
            set({ status: 'offline', currentCampaign: null });
          } else if (event.type === 'omp-paused') {
            set({ status: 'paused' });
          } else if (event.type === 'omp-resumed') {
            set({ status: 'running' });
          } else if (event.type === 'omp-promotion') {
            set({ lastPromotion: new Date().toISOString() });
            get().fetchHallOfFame();
            get().fetchStats();
          } else if (event.type === 'omp-queue-updated') {
            get().fetchQueue();
          } else if (event.type === 'omp-campaign-completed') {
            set({ currentCampaign: null });
            get().fetchStats();
          } else if (event.type === 'omp-log') {
            // Add new log entry to the front (INCREMENTAL - no reset)
            const logEntry = event as unknown as ActivityLogEntry;
            const currentLog = get().activityLog;
            // Dedupe by id to prevent duplicates on reconnection
            const existingIds = new Set(currentLog.map(l => l.id));
            if (!existingIds.has(logEntry.id)) {
              set({ activityLog: [logEntry, ...currentLog].slice(0, 100) });
            }
          }
        },
        (error) => {
          console.error('[OMP SSE] Error:', error);
          // KEEP previous state on error - only mark disconnected
          set({ sseConnected: false, lastError: 'SSE connection lost - reconnecting...' });
        },
        () => {
          set({ sseConnected: true, lastError: null });
        }
      );
      
      set({ sseConnected: true });
      
      // Initial fetch
      get().fetchStatus();
      get().fetchStats();
      get().fetchQueue();
      get().fetchActivityLog();
      get().fetchPerformance();
      
      // Performance polling (every 2 seconds for live metrics)
      const perfInterval = setInterval(() => {
        if (get().status === 'running') {
          get().fetchPerformance();
        }
      }, 2000);
      
      return () => {
        if (sse) sse.close();
        clearInterval(perfInterval);
        set({ sseConnected: false });
      };
    }
    
    // Desktop mode - use polling (more aggressive for responsiveness)
    console.log('[OMP Store] Desktop mode - using polling');
    
    // Initial fetch
    get().fetchStatus();
    get().fetchStats();
    get().fetchQueue();
    get().fetchActivityLog();
    get().fetchPerformance();
    
    // Fast polling when running (every 2s), slower when idle (every 5s)
    const pollInterval = setInterval(() => {
      const status = get().status;
      get().fetchStatus();
      if (status === 'running') {
        get().fetchPerformance();
      }
    }, 2000);
    
    // Slower polling for stats/queue (every 10s)
    const statsInterval = setInterval(() => {
      get().fetchStats();
      get().fetchQueue();
    }, 10000);
    
    return () => {
      clearInterval(pollInterval);
      clearInterval(statsInterval);
    };
  },
}));


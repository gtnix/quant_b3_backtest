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
  canStartCampaign: boolean;
}

export interface CurrentCampaign {
  campaignId: string;
  campaignName: string;
  runId: string;
  market: 'br' | 'us';
  status: string;
  elapsedSeconds: number;
  currentGeneration: number;
  bestSharpe: number | null;
  candidatesEvaluated: number;
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
  };
  notes: string;
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
  
  // Actions
  fetchStatus: () => Promise<void>;
  fetchStats: () => Promise<void>;
  fetchQueue: () => Promise<void>;
  fetchHallOfFame: (limit?: number, market?: string) => Promise<void>;
  fetchConfig: () => Promise<void>;
  
  start: () => Promise<boolean>;
  stop: () => Promise<boolean>;
  pause: () => Promise<boolean>;
  resume: () => Promise<boolean>;
  
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
  
  // ==========================================================================
  // FETCH ACTIONS
  // ==========================================================================
  
  fetchStatus: async () => {
    try {
      const response = await fetch(`${platformConfig.apiBase}/omp/status`, { credentials: 'same-origin' });
      if (!response.ok) throw new Error('Failed to fetch status');
      const data = await response.json();
      
      set({
        status: data.status || 'offline',
        startedAt: data.startedAt,
        lastLoop: data.lastLoop,
        loopCount: data.loopCount || 0,
        queueLength: data.queueLength || 0,
        lastPromotion: data.lastPromotion,
        currentCampaign: data.currentCampaign,
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
      set({ stats: data, lastError: null });
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
    // Browser mode - use SSE
    if (platform === 'browser') {
      const sse = createSSEConnection(
        (event) => {
          if (event.type === 'omp-status') {
            const data = event.data;
            set({
              status: data.status || 'offline',
              startedAt: data.startedAt,
              lastLoop: data.lastLoop,
              loopCount: data.loopCount || 0,
              queueLength: data.queueLength || 0,
              lastPromotion: data.lastPromotion,
              currentCampaign: data.currentCampaign,
              resources: data.resources || get().resources,
            });
          } else if (event.type === 'omp-started') {
            set({ status: 'running', startedAt: event.data.startedAt });
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
          }
        },
        (error) => {
          console.error('[OMP SSE] Error:', error);
          set({ sseConnected: false });
        },
        () => {
          set({ sseConnected: true });
        }
      );
      
      set({ sseConnected: true });
      
      // Initial fetch
      get().fetchStatus();
      get().fetchStats();
      get().fetchQueue();
      
      return () => {
        sse.close();
        set({ sseConnected: false });
      };
    }
    
    // Desktop mode - use polling
    const pollInterval = setInterval(() => {
      get().fetchStatus();
    }, 5000);
    
    // Initial fetch
    get().fetchStatus();
    get().fetchStats();
    get().fetchQueue();
    
    return () => {
      clearInterval(pollInterval);
    };
  },
}));


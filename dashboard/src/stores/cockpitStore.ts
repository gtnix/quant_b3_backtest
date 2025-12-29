/**
 * Cockpit Store - Zustand state management for SCG run orchestration
 */

import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import type { CockpitConfig, PresetKey, RankingMethodKey } from '../config/defaults';
import { COCKPIT_PRESETS, getDefaultConfig, toTauriConfig } from '../config/defaults';

// =============================================================================
// TYPES
// =============================================================================

export type RunStatus = 
  | 'idle' 
  | 'starting' 
  | 'running' 
  | 'stopping' 
  | 'completed' 
  | 'failed' 
  | 'cancelled';

export interface RunProgress {
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

export interface CandidateListItem {
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
}

export interface RankedCandidate extends CandidateListItem {
  rankReasons: string[];
  score: number;
}

export type ViewMode = 'basic' | 'advanced';

// =============================================================================
// STORE INTERFACE
// =============================================================================

interface CockpitState {
  // Configuration
  config: CockpitConfig;
  viewMode: ViewMode;
  rankingMethod: RankingMethodKey;
  
  // Run state
  runStatus: RunStatus;
  currentRunId: string | null;
  progress: RunProgress | null;
  
  // Results
  topCandidates: RankedCandidate[];
  selectedCandidateId: string | null;
  
  // UI state
  isGlossaryOpen: boolean;
  
  // Actions - Configuration
  setPreset: (preset: PresetKey) => void;
  updateConfig: (partial: Partial<CockpitConfig>) => void;
  setViewMode: (mode: ViewMode) => void;
  setRankingMethod: (method: RankingMethodKey) => void;
  
  // Actions - Run control
  startRun: () => Promise<void>;
  stopRun: () => Promise<void>;
  pollProgress: () => Promise<void>;
  
  // Actions - Results
  loadTopCandidates: (runId: string) => Promise<void>;
  selectCandidate: (candidateId: string | null) => void;
  
  // Actions - UI
  toggleGlossary: () => void;
  
  // Subscriptions
  subscribeToProgress: () => () => void;
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function parseRunProgress(data: any): RunProgress {
  return {
    runId: data.run_id ?? '',
    status: data.status ?? 'idle',
    currentGeneration: data.current_generation ?? 0,
    maxGenerations: data.max_generations ?? 0,
    elapsedSeconds: data.elapsed_seconds ?? 0,
    maxRuntimeSeconds: data.max_runtime_seconds ?? 0,
    bestSharpe: data.best_sharpe ?? null,
    bestCagr: data.best_cagr ?? null,
    candidatesEvaluated: data.candidates_evaluated ?? 0,
    candidatesPassingGates: data.candidates_passing_gates ?? 0,
    paretoSize: data.pareto_size ?? 0,
    latestLog: data.latest_log ?? null,
    percentComplete: data.percent_complete ?? 0,
    errorMessage: data.error_message ?? null,
  };
}

function parseCandidateList(data: any[]): CandidateListItem[] {
  return data.map((item) => ({
    rank: item.rank ?? 0,
    candidateId: item.candidate_id ?? '',
    candidateClass: item.candidate_class ?? 'research',
    displayName: item.display_name ?? '',
    oosSharpeNet: item.oos_sharpe_net ?? 0,
    oosCagrNet: item.oos_cagr_net ?? 0,
    maxDrawdownNet: item.max_drawdown_net ?? 0,
    pbo: item.pbo ?? 1,
    dsr: item.dsr ?? 0,
    gatesPassed: item.gates_passed ?? false,
    stressPassed: item.stress_passed ?? false,
    dataIntegrityOk: item.data_integrity_ok ?? true,
  }));
}

// =============================================================================
// RANKING FUNCTIONS
// =============================================================================

function explainRank(candidate: CandidateListItem): string[] {
  const reasons: string[] = [];
  
  if (candidate.oosSharpeNet >= 1.0) {
    reasons.push('Sharpe excelente (≥1.0)');
  } else if (candidate.oosSharpeNet >= 0.7) {
    reasons.push('Sharpe bom (≥0.7)');
  }
  
  if (candidate.pbo <= 0.10) {
    reasons.push('Baixo risco de overfitting (PBO ≤10%)');
  } else if (candidate.pbo <= 0.15) {
    reasons.push('PBO aceitável (≤15%)');
  }
  
  if (candidate.stressPassed) {
    reasons.push('Passou testes de stress');
  }
  
  if (candidate.gatesPassed) {
    reasons.push('Passou todos os gates');
  }
  
  if (candidate.maxDrawdownNet > -15) {
    reasons.push('Drawdown controlado (<15%)');
  }
  
  if (candidate.dsr > 1.0) {
    reasons.push('DSR forte (>1.0)');
  }
  
  return reasons.slice(0, 3);
}

function scoreCandidate(candidate: CandidateListItem, method: RankingMethodKey): number {
  switch (method) {
    case 'institutional': {
      // Multi-criteria weighted score
      const sharpeScore = Math.min(candidate.oosSharpeNet / 2, 1) * 40;
      const pboScore = (1 - candidate.pbo / 0.5) * 25;
      const stressScore = candidate.stressPassed ? 20 : 0;
      const gatesScore = candidate.gatesPassed ? 15 : 0;
      return sharpeScore + pboScore + stressScore + gatesScore;
    }
    case 'pareto':
      // Pareto: lower is better (rank by Pareto dominance)
      return candidate.oosSharpeNet - Math.abs(candidate.maxDrawdownNet) * 0.1;
    case 'sharpe':
      return candidate.oosSharpeNet;
    case 'riskadjusted':
      return candidate.maxDrawdownNet !== 0 
        ? candidate.oosSharpeNet / Math.abs(candidate.maxDrawdownNet) * 100 
        : 0;
    default:
      return candidate.oosSharpeNet;
  }
}

function rankCandidates(
  candidates: CandidateListItem[], 
  method: RankingMethodKey
): RankedCandidate[] {
  const scored = candidates.map((c) => ({
    ...c,
    score: scoreCandidate(c, method),
    rankReasons: explainRank(c),
  }));
  
  scored.sort((a, b) => b.score - a.score);
  
  return scored.map((c, i) => ({ ...c, rank: i + 1 }));
}

// =============================================================================
// STORE IMPLEMENTATION
// =============================================================================

export const useCockpitStore = create<CockpitState>((set, get) => ({
  // Initial state
  config: getDefaultConfig(),
  viewMode: 'basic',
  rankingMethod: 'institutional',
  runStatus: 'idle',
  currentRunId: null,
  progress: null,
  topCandidates: [],
  selectedCandidateId: null,
  isGlossaryOpen: false,
  
  // Configuration actions
  setPreset: (preset) => {
    const presetConfig = COCKPIT_PRESETS[preset];
    set({
      config: {
        ...get().config,
        preset,
        ...presetConfig.config,
      },
    });
  },
  
  updateConfig: (partial) => {
    set({
      config: {
        ...get().config,
        ...partial,
      },
    });
  },
  
  setViewMode: (mode) => set({ viewMode: mode }),
  
  setRankingMethod: (method) => {
    set({ rankingMethod: method });
    // Re-rank existing candidates
    const { topCandidates } = get();
    if (topCandidates.length > 0) {
      const reranked = rankCandidates(topCandidates, method);
      set({ topCandidates: reranked });
    }
  },
  
  // Run control actions
  startRun: async () => {
    const { config } = get();
    set({ runStatus: 'starting', progress: null, topCandidates: [] });
    
    try {
      const tauriConfig = toTauriConfig(config);
      const runId = await invoke<string>('start_scg_run', { config: tauriConfig });
      
      set({ 
        currentRunId: runId,
        runStatus: 'running',
        progress: {
          runId,
          status: 'running',
          currentGeneration: 0,
          maxGenerations: config.maxGenerations,
          elapsedSeconds: 0,
          maxRuntimeSeconds: config.maxRuntimeSeconds,
          bestSharpe: null,
          bestCagr: null,
          candidatesEvaluated: 0,
          candidatesPassingGates: 0,
          paretoSize: 0,
          latestLog: 'Iniciando engine SCG...',
          percentComplete: 0,
          errorMessage: null,
        },
      });
    } catch (error) {
      set({ 
        runStatus: 'failed',
        progress: {
          ...get().progress!,
          status: 'failed',
          errorMessage: String(error),
        },
      });
    }
  },
  
  stopRun: async () => {
    const { currentRunId } = get();
    if (!currentRunId) return;
    
    set({ runStatus: 'stopping' });
    
    try {
      await invoke('stop_scg_run', { runId: currentRunId });
      set({ runStatus: 'cancelled' });
    } catch (error) {
      console.error('Failed to stop run:', error);
    }
  },
  
  pollProgress: async () => {
    const { currentRunId, rankingMethod } = get();
    if (!currentRunId) return;
    
    try {
      const data = await invoke('get_run_status', { runId: currentRunId });
      const progress = parseRunProgress(data);
      
      set({ 
        progress,
        runStatus: progress.status as RunStatus,
      });
      
      // If completed, load candidates
      if (progress.status === 'completed') {
        const candidatesData = await invoke<any[]>('list_candidates_v2', {
          runId: currentRunId,
          search: null,
          candidateClass: null,
          maxPbo: null,
          limit: 100,
        });
        
        const candidates = parseCandidateList(candidatesData);
        const ranked = rankCandidates(candidates, rankingMethod);
        set({ topCandidates: ranked });
      }
    } catch (error) {
      console.error('Failed to poll progress:', error);
    }
  },
  
  // Results actions
  loadTopCandidates: async (runId) => {
    const { rankingMethod } = get();
    
    try {
      const data = await invoke<any[]>('list_candidates_v2', {
        runId,
        search: null,
        candidateClass: null,
        maxPbo: null,
        limit: 100,
      });
      
      const candidates = parseCandidateList(data);
      const ranked = rankCandidates(candidates, rankingMethod);
      set({ topCandidates: ranked });
    } catch (error) {
      console.error('Failed to load candidates:', error);
    }
  },
  
  selectCandidate: (candidateId) => set({ selectedCandidateId: candidateId }),
  
  // UI actions
  toggleGlossary: () => set((s) => ({ isGlossaryOpen: !s.isGlossaryOpen })),
  
  // Subscriptions
  subscribeToProgress: () => {
    let intervalId: number | null = null;
    
    // Poll every 2 seconds while running
    const checkAndPoll = () => {
      const { runStatus } = get();
      if (runStatus === 'running' || runStatus === 'starting') {
        get().pollProgress();
      } else if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
      }
    };
    
    intervalId = window.setInterval(checkAndPoll, 2000);
    
    // Also listen for Tauri events
    const unlisten = listen('scg-progress', (event) => {
      const progress = parseRunProgress(event.payload);
      set({ progress, runStatus: progress.status as RunStatus });
    });
    
    // Return cleanup function
    return () => {
      if (intervalId) clearInterval(intervalId);
      unlisten.then((fn) => fn());
    };
  },
}));


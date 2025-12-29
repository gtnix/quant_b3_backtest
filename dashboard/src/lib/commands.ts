/**
 * Unified Command Abstraction Layer
 * 
 * Provides a single interface for all commands that works seamlessly
 * in both Tauri (desktop) and Browser modes.
 * 
 * Usage:
 *   import { cmd } from './lib/commands';
 *   const index = await cmd.loadIndex();
 */

import { invoke } from '@tauri-apps/api/core';
import { platform, config } from './platform';

// =============================================================================
// TYPES
// =============================================================================

// Site & Navigation
export interface SiteIndex {
  schema_version: string;
  generated_at: string;
  campaigns: CampaignSummary[];
  data_source?: string;
}

export interface CampaignSummary {
  campaign_id: string;
  name: string;
  tag: string;
  status: string;
  runs_count: number;
  created_at: string;
  detail_path: string | null;
}

export interface CampaignDetail {
  schema_version: string;
  campaign: CampaignInfo;
  runs: RunSummary[];
}

export interface CampaignInfo {
  campaign_id: string;
  name: string;
  tag: string;
  owner?: string;
  status: string;
  config_hash?: string;
  git_sha?: string;
  created_at: string;
  notes?: string;
}

export interface RunSummary {
  run_id: string;
  seed: number;
  status: string;
  data_integrity_verdict?: string;
  data_integrity_score?: number;
  candidates_count?: number;
  research_candidates_count?: number;
  validated_candidates_count?: number;
  best_oos_sharpe_net?: number;
  duration_secs?: number;
  detail_path?: string;
  export_path?: string;
}

export interface RunDetail {
  schema_version: string;
  run: RunInfo;
  config_snapshot?: unknown;
  metrics: RunMetrics;
  top_candidates: TopCandidateEntry[];
  exports: RunExports;
}

export interface RunInfo {
  run_id: string;
  campaign_id: string;
  seed: number;
  status: string;
  started_at?: string;
  completed_at?: string;
  duration_secs?: number;
  artifact_path?: string;
}

export interface RunMetrics {
  total_evaluated?: number;
  research_candidates?: number;
  validated_candidates?: number;
  promoted_candidates?: number;
  best_oos_sharpe_net?: number;
  best_oos_cagr_net?: number;
  data_integrity_verdict?: string;
  data_integrity_score?: number;
}

export interface TopCandidateEntry {
  rank: number;
  candidate_id: string;
  candidate_class: string;
  display_name?: string;
  oos_sharpe_net?: number;
  oos_cagr_net?: number;
  max_drawdown_net?: number;
  pbo?: number;
  dsr?: number;
  gates_passed?: boolean;
  stress_passed?: boolean;
  data_integrity_ok?: boolean;
}

export interface RunExports {
  top1000_json?: string;
  top1000_csv?: string;
  pareto_json?: string;
}

// Candidates
export interface CandidateListItem {
  rank: number;
  candidate_id: string;
  candidate_class: string;
  display_name: string;
  oos_sharpe_net: number;
  oos_cagr_net: number;
  max_drawdown_net: number;
  pbo: number;
  dsr: number;
  gates_passed: boolean;
  stress_passed: boolean;
  stress_total?: number;
  data_integrity_ok: boolean;
}

export interface CandidateDetail {
  candidate_id: string;
  display_name: string;
  candidate_class: string;
  strategy_blocks: PipelineBlock[];
  strategy_toml?: string;
  oos_sharpe_net?: number;
  oos_cagr_net?: number;
  max_drawdown_net?: number;
  pbo?: number;
  dsr?: number;
  gates_passed?: boolean;
  stress_passed?: boolean;
  data_integrity_ok?: boolean;
  execution_config?: ExecutionConfig;
  provenance?: Provenance;
  bundle_path?: string;
  strategy_toml_path?: string;
  validation_summary_path?: string;
}

export interface PipelineBlock {
  block_type: string;
  name: string;
  params: Record<string, unknown>;
}

export interface Provenance {
  git_sha?: string;
  dataset_hash?: string;
  config_hash?: string;
  run_id?: string;
  campaign_id?: string;
  seed?: number;
  created_at?: string;
}

export interface ExecutionConfig {
  delay_bars: number;
  bypass_for_debug: boolean;
  slippage: { slippage_type: string; bps?: number };
  fees: { tier: string };
  fill_policy?: { allow_partial?: boolean; max_participation?: number };
}

// Backtest
export interface BacktestResult {
  available: boolean;
  candidate_id: string;
  message?: string;
  metadata?: BacktestMetadata;
  metrics?: BacktestMetrics;
  timeseries: TimeseriesPoint[];
  backtest_path?: string;
}

export interface BacktestMetadata {
  schema_version?: string;
  run_id: string;
  config_hash?: string;
  start_date?: string;
  end_date?: string;
}

export interface BacktestMetrics {
  total_return?: number;
  cagr?: number;
  sharpe?: number;
  sortino?: number;
  max_drawdown?: number;
  calmar?: number;
  volatility?: number;
  win_rate?: number;
  profit_factor?: number;
  total_trades?: number;
}

export interface TimeseriesPoint {
  date: string;
  equity: number;
  drawdown: number;
  exposure?: number;
  vol_exante?: number;
  vol_expost?: number;
}

// SCG Run Control
export interface ScgRunConfig {
  preset: string;
  max_runtime_seconds: number;
  population_size: number;
  max_generations: number;
  convergence_generations: number;
  workers: number;
  seeds: number[];
  stress_testing_enabled: boolean;
  min_oos_sharpe_net: number;
  max_pbo: number;
  min_stress_passed: number;
  campaign_config?: string;
  run_tag?: string;
}

export interface RunProgress {
  run_id: string;
  status: 'idle' | 'starting' | 'running' | 'stopping' | 'completed' | 'failed' | 'cancelled';
  current_generation: number;
  max_generations: number;
  elapsed_seconds: number;
  max_runtime_seconds: number;
  best_sharpe?: number;
  best_cagr?: number;
  candidates_evaluated: number;
  candidates_passing_gates: number;
  pareto_size: number;
  latest_log?: string;
  percent_complete: number;
  error_message?: string;
}

// Root paths
export interface PathInfo {
  path: string;
  valid: boolean;
  exists?: boolean;
  has_index?: boolean;
  combiner_exists?: boolean;
  is_rust_project?: boolean;
}

// Recent runs for selector
export interface RecentRun {
  run_id: string;
  campaign_name: string;
  created_at: string;
  status: string;
  best_oos_sharpe_net?: number;
  candidates_count: number;
}

// =============================================================================
// API CALL HELPER
// =============================================================================

async function apiCall<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${config.apiBase}${endpoint}`;
  
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: response.statusText }));
    throw new Error(error.error || `API error: ${response.status}`);
  }
  
  return response.json();
}

// =============================================================================
// UNIFIED COMMAND INTERFACE
// =============================================================================

export const cmd = {
  // -------------------------------------------------------------------------
  // ARTIFACTS ROOT
  // -------------------------------------------------------------------------
  
  async getArtifactsRoot(): Promise<PathInfo> {
    if (platform.isTauri) {
      const path = await invoke<string | null>('get_artifacts_root');
      return {
        path: path || '',
        valid: !!path,
      };
    }
    return apiCall<PathInfo>('/artifacts-root');
  },
  
  async setArtifactsRoot(path: string): Promise<string> {
    if (platform.isTauri) {
      return invoke<string>('set_artifacts_root', { path });
    }
    const result = await apiCall<PathInfo>('/artifacts-root', {
      method: 'POST',
      body: JSON.stringify({ path }),
    });
    return result.path;
  },
  
  // -------------------------------------------------------------------------
  // WORKSPACE ROOT
  // -------------------------------------------------------------------------
  
  async getWorkspaceRoot(): Promise<PathInfo> {
    if (platform.isTauri) {
      const path = await invoke<string | null>('get_workspace_root');
      return {
        path: path || '',
        valid: !!path,
      };
    }
    return apiCall<PathInfo>('/workspace-root');
  },
  
  async setWorkspaceRoot(path: string): Promise<string> {
    if (platform.isTauri) {
      return invoke<string>('set_workspace_root', { path });
    }
    const result = await apiCall<PathInfo>('/workspace-root', {
      method: 'POST',
      body: JSON.stringify({ path }),
    });
    return result.path;
  },
  
  // -------------------------------------------------------------------------
  // SITE INDEX & NAVIGATION
  // -------------------------------------------------------------------------
  
  async loadIndex(): Promise<SiteIndex> {
    if (platform.isTauri) {
      return invoke<SiteIndex>('load_index');
    }
    return apiCall<SiteIndex>('/index');
  },
  
  async listCampaigns(): Promise<CampaignSummary[]> {
    if (platform.isTauri) {
      const index = await invoke<SiteIndex>('load_index');
      return index.campaigns;
    }
    const result = await apiCall<{ campaigns: CampaignSummary[] }>('/campaigns');
    return result.campaigns;
  },
  
  async loadCampaign(campaignId: string): Promise<CampaignDetail> {
    if (platform.isTauri) {
      return invoke<CampaignDetail>('load_campaign', { campaignId });
    }
    return apiCall<CampaignDetail>(`/campaign/${campaignId}`);
  },
  
  async loadRun(runId: string): Promise<RunDetail> {
    if (platform.isTauri) {
      return invoke<RunDetail>('load_run', { runId });
    }
    return apiCall<RunDetail>(`/run/${runId}`);
  },
  
  async listRecentRuns(limit = 10): Promise<RecentRun[]> {
    if (platform.isTauri) {
      // Tauri: aggregate from local artifacts
      const index = await invoke<SiteIndex>('load_index');
      const runs: RecentRun[] = [];
      for (const campaign of index.campaigns.slice(0, 5)) {
        try {
          const detail = await invoke<CampaignDetail>('load_campaign', { campaignId: campaign.campaign_id });
          for (const run of detail.runs.slice(0, 3)) {
            runs.push({
              run_id: run.run_id,
              campaign_name: campaign.name,
              created_at: campaign.created_at,
              status: run.status,
              best_oos_sharpe_net: run.best_oos_sharpe_net,
              candidates_count: run.candidates_count || 0,
            });
          }
        } catch {
          // Skip failed campaigns
        }
      }
      return runs.slice(0, limit);
    }
    const result = await apiCall<{ runs: RecentRun[] }>(`/runs/recent?limit=${limit}`);
    return result.runs;
  },
  
  // -------------------------------------------------------------------------
  // CANDIDATES
  // -------------------------------------------------------------------------
  
  async listCandidates(
    runId: string,
    options: {
      limit?: number;
      search?: string;
      candidateClass?: string;
      maxPbo?: number;
    } = {}
  ): Promise<CandidateListItem[]> {
    if (platform.isTauri) {
      return invoke<CandidateListItem[]>('list_candidates_v2', {
        runId,
        search: options.search,
        candidateClass: options.candidateClass,
        maxPbo: options.maxPbo,
        limit: options.limit,
      });
    }
    
    const params = new URLSearchParams();
    if (options.limit) params.set('limit', String(options.limit));
    if (options.search) params.set('search', options.search);
    if (options.candidateClass) params.set('candidate_class', options.candidateClass);
    if (options.maxPbo) params.set('max_pbo', String(options.maxPbo));
    
    // API returns array directly, not wrapped in { candidates: [...] }
    const result = await apiCall<CandidateListItem[]>(
      `/candidates/${runId}?${params.toString()}`
    );
    return Array.isArray(result) ? result : [];
  },
  
  async listRecentCandidates(limit = 20): Promise<CandidateListItem[]> {
    if (platform.isTauri) {
      // Tauri: get from latest run
      const runs = await this.listRecentRuns(1);
      if (runs.length > 0) {
        return this.listCandidates(runs[0].run_id, { limit });
      }
      return [];
    }
    // API returns array directly
    const result = await apiCall<CandidateListItem[]>(
      `/candidates/recent?limit=${limit}`
    );
    return Array.isArray(result) ? result : [];
  },
  
  async loadCandidateDetail(candidateId: string): Promise<CandidateDetail> {
    if (platform.isTauri) {
      return invoke<CandidateDetail>('load_candidate_detail', { candidateId });
    }
    return apiCall<CandidateDetail>(`/candidate/${candidateId}`);
  },
  
  // -------------------------------------------------------------------------
  // BACKTEST
  // -------------------------------------------------------------------------
  
  async loadBacktestSeries(candidateId: string): Promise<BacktestResult> {
    if (platform.isTauri) {
      return invoke<BacktestResult>('load_backtest_series', { candidateId });
    }
    return apiCall<BacktestResult>(`/backtest/${candidateId}`);
  },
  
  async loadSimulatedEquity(candidateId: string): Promise<{
    timeseries: TimeseriesPoint[];
    metrics: BacktestMetrics;
  }> {
    // Browser mode only - simulated equity from Neon data
    return apiCall(`/candidate/${candidateId}/simulated-equity`);
  },
  
  async loadCandidatePipeline(candidateId: string): Promise<{
    blocks: PipelineBlock[];
    strategy_toml?: string;
  }> {
    return apiCall(`/candidate/${candidateId}/pipeline`);
  },
  
  async loadCandidateWFA(candidateId: string): Promise<unknown> {
    return apiCall(`/candidate/${candidateId}/wfa`);
  },
  
  async loadCandidateStress(candidateId: string): Promise<unknown> {
    return apiCall(`/candidate/${candidateId}/stress`);
  },
  
  // -------------------------------------------------------------------------
  // SCG RUN CONTROL
  // -------------------------------------------------------------------------
  
  async startScgRun(config: Partial<ScgRunConfig>): Promise<string> {
    if (platform.isTauri) {
      return invoke<string>('start_scg_run', { config });
    }
    const result = await apiCall<{ runId: string }>('/scg/start', {
      method: 'POST',
      body: JSON.stringify(config),
    });
    return result.runId;
  },
  
  async stopScgRun(runId: string): Promise<void> {
    if (platform.isTauri) {
      return invoke('stop_scg_run', { runId });
    }
    await apiCall(`/scg/stop/${runId}`, { method: 'POST' });
  },
  
  async getRunStatus(runId: string): Promise<RunProgress> {
    if (platform.isTauri) {
      return invoke<RunProgress>('get_run_status', { runId });
    }
    return apiCall<RunProgress>(`/scg/progress/${runId}`);
  },
  
  async listActiveRuns(): Promise<RunProgress[]> {
    if (platform.isTauri) {
      return invoke<RunProgress[]>('list_active_runs');
    }
    const result = await apiCall<{ runs: RunProgress[] }>('/scg/active-runs');
    return result.runs;
  },
  
  async loadCockpitCandidates(runId: string): Promise<CandidateListItem[]> {
    if (platform.isTauri) {
      return invoke<CandidateListItem[]>('load_cockpit_candidates', { runId });
    }
    const result = await apiCall<{ candidates: CandidateListItem[] }>(
      `/cockpit-candidates/${runId}`
    );
    return result.candidates;
  },
  
  // -------------------------------------------------------------------------
  // CACHE & UPDATES
  // -------------------------------------------------------------------------
  
  async invalidateCache(): Promise<{ cleared: string[] }> {
    if (platform.isTauri) {
      await invoke('invalidate_cache');
      return { cleared: ['all'] };
    }
    return apiCall('/invalidate-cache', { method: 'POST' });
  },
  
  async watchArtifacts(): Promise<void> {
    if (platform.isTauri) {
      return invoke('watch_artifacts');
    }
    // Browser mode: SSE is handled separately in App.tsx
    console.log('[Browser Mode] File watching via SSE');
  },
  
  async pollChanges(since?: number): Promise<{
    changes: Array<{ type: string; path: string; modified: string }>;
    has_changes: boolean;
  }> {
    const params = since ? `?since=${since}` : '';
    return apiCall(`/poll-changes${params}`);
  },
};

// =============================================================================
// SSE EVENT HANDLING (Browser Mode)
// =============================================================================

export type SSEEventType = 
  | 'connected'
  | 'ping'
  | 'artifact-change'
  | 'scg-progress'
  | 'run-complete'
  | 'cache-invalidated';

export interface SSEEvent {
  type: SSEEventType;
  timestamp: number;
  [key: string]: unknown;
}

export function createSSEConnection(
  onEvent: (event: SSEEvent) => void,
  onError?: (error: Event) => void
): EventSource | null {
  if (platform.isTauri) {
    console.log('[Tauri Mode] SSE not needed, using native events');
    return null;
  }
  
  const eventSource = new EventSource(config.sseEndpoint);
  
  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data) as SSEEvent;
      onEvent(data);
    } catch (e) {
      console.error('[SSE] Failed to parse event:', e);
    }
  };
  
  eventSource.onerror = (error) => {
    console.error('[SSE] Connection error:', error);
    onError?.(error);
  };
  
  return eventSource;
}

export default cmd;


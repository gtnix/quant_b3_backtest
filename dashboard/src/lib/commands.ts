/**
 * Command API Layer - Web Only
 */

import { config } from './platform';

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
  /** Whether this result is considered valid for analysis */
  is_valid?: boolean;
  /** Warnings about potential issues */
  warnings?: BacktestWarning[];
}

/** Critical warning in a backtest result */
export type BacktestWarning =
  | { type: 'ZeroTrades' }
  | { type: 'LowTradeCount'; actual: number; recommended_min: number }
  | { type: 'UnrealisticSharpe'; sharpe: number }
  | { type: 'PerfectEquityCurve' }
  | { type: 'EmptyUniverseEncountered'; occurrences: number };

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
  async getArtifactsRoot(): Promise<PathInfo> {
    return apiCall<PathInfo>('/artifacts-root');
  },
  
  async setArtifactsRoot(path: string): Promise<string> {
    const result = await apiCall<PathInfo>('/artifacts-root', {
      method: 'POST',
      body: JSON.stringify({ path }),
    });
    return result.path;
  },
  
  async getWorkspaceRoot(): Promise<PathInfo> {
    return apiCall<PathInfo>('/workspace-root');
  },
  
  async setWorkspaceRoot(path: string): Promise<string> {
    const result = await apiCall<PathInfo>('/workspace-root', {
      method: 'POST',
      body: JSON.stringify({ path }),
    });
    return result.path;
  },
  
  async loadIndex(): Promise<SiteIndex> {
    return apiCall<SiteIndex>('/index');
  },
  
  async listCampaigns(): Promise<CampaignSummary[]> {
    const result = await apiCall<{ campaigns: CampaignSummary[] }>('/campaigns');
    return result.campaigns;
  },
  
  async loadCampaign(campaignId: string): Promise<CampaignDetail> {
    return apiCall<CampaignDetail>(`/campaign/${campaignId}`);
  },
  
  async loadRun(runId: string): Promise<RunDetail> {
    return apiCall<RunDetail>(`/run/${runId}`);
  },
  
  async listRecentRuns(limit = 10): Promise<RecentRun[]> {
    const result = await apiCall<{ runs: RecentRun[] }>(`/runs/recent?limit=${limit}`);
    return result.runs;
  },
  
  async listCandidates(
    runId: string,
    options: { limit?: number; search?: string; candidateClass?: string; maxPbo?: number } = {}
  ): Promise<CandidateListItem[]> {
    const params = new URLSearchParams();
    if (options.limit) params.set('limit', String(options.limit));
    if (options.search) params.set('search', options.search);
    if (options.candidateClass) params.set('candidate_class', options.candidateClass);
    if (options.maxPbo) params.set('max_pbo', String(options.maxPbo));
    const result = await apiCall<CandidateListItem[]>(`/candidates/${runId}?${params.toString()}`);
    return Array.isArray(result) ? result : [];
  },
  
  async listRecentCandidates(limit = 20): Promise<CandidateListItem[]> {
    const result = await apiCall<CandidateListItem[]>(`/candidates/recent?limit=${limit}`);
    return Array.isArray(result) ? result : [];
  },
  
  async loadCandidateDetail(candidateId: string): Promise<CandidateDetail> {
    return apiCall<CandidateDetail>(`/candidate/${candidateId}`);
  },
  
  async loadBacktestSeries(candidateId: string): Promise<BacktestResult> {
    return apiCall<BacktestResult>(`/backtest/${candidateId}`);
  },
  
  async loadSimulatedEquity(candidateId: string): Promise<{ timeseries: TimeseriesPoint[]; metrics: BacktestMetrics }> {
    return apiCall(`/candidate/${candidateId}/simulated-equity`);
  },
  
  async loadCandidatePipeline(candidateId: string): Promise<{ blocks: PipelineBlock[]; strategy_toml?: string }> {
    return apiCall(`/candidate/${candidateId}/pipeline`);
  },
  
  async loadCandidateWFA(candidateId: string): Promise<unknown> {
    return apiCall(`/candidate/${candidateId}/wfa`);
  },
  
  async loadCandidateStress(candidateId: string): Promise<unknown> {
    return apiCall(`/candidate/${candidateId}/stress`);
  },
  
  async startScgRun(cfg: Partial<ScgRunConfig>): Promise<string> {
    const result = await apiCall<{ runId: string }>('/scg/start', { method: 'POST', body: JSON.stringify(cfg) });
    return result.runId;
  },
  
  async stopScgRun(runId: string): Promise<void> {
    await apiCall(`/scg/stop/${runId}`, { method: 'POST' });
  },
  
  async getRunStatus(runId: string): Promise<RunProgress> {
    return apiCall<RunProgress>(`/scg/progress/${runId}`);
  },
  
  async listActiveRuns(): Promise<RunProgress[]> {
    const result = await apiCall<{ runs: RunProgress[] }>('/scg/active-runs');
    return result.runs;
  },
  
  async loadCockpitCandidates(runId: string): Promise<CandidateListItem[]> {
    const result = await apiCall<{ candidates: CandidateListItem[] }>(`/cockpit-candidates/${runId}`);
    return result.candidates;
  },
  
  async invalidateCache(): Promise<{ cleared: string[] }> {
    return apiCall('/invalidate-cache', { method: 'POST' });
  },
  
  async watchArtifacts(): Promise<void> {
    console.log('[Web] File watching via SSE');
  },
  
  async pollChanges(since?: number): Promise<{ changes: Array<{ type: string; path: string; modified: string }>; has_changes: boolean }> {
    return apiCall(`/poll-changes${since ? `?since=${since}` : ''}`);
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
  onError?: (error: Event) => void,
  onReconnect?: () => void
): EventSource | null {
  if (!config.sseEndpoint) {
    console.warn('[SSE] No endpoint configured');
    return null;
  }
  
  const eventSource = new EventSource(config.sseEndpoint);
  let reconnectAttempts = 0;
  
  eventSource.onopen = () => {
    if (reconnectAttempts > 0) onReconnect?.();
    reconnectAttempts = 0;
  };
  
  eventSource.onmessage = (event) => {
    try {
      onEvent(JSON.parse(event.data) as SSEEvent);
    } catch (e) {
      console.error('[SSE] Parse error:', e);
    }
  };
  
  eventSource.onerror = (error) => {
    reconnectAttempts++;
    onError?.(error);
  };
  
  return eventSource;
}

export default cmd;


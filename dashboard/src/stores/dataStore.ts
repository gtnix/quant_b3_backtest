import { create } from 'zustand';
import { listen } from '@tauri-apps/api/event';
import { cmd, createSSEConnection, type SSEEvent } from '../lib/commands';
import { platform, config, features } from '../lib/platform';

// =============================================================================
// UNIFIED COMMAND SYSTEM
// =============================================================================

// Auto-initialize in browser mode
async function initBrowserMode(): Promise<string | null> {
  if (platform.isTauri) return null;
  
  try {
    const response = await fetch(`${config.apiBase}/health`);
    if (response.ok) {
      const data = await response.json();
      console.log('[Browser Mode] Connected to API, artifacts at:', data.artifacts_root);
      return data.artifacts_root;
    }
  } catch (e) {
    console.warn('[Browser Mode] API server not available');
  }
  return null;
}

// =============================================================================
// ARTIFACT TYPES - Aligned with backend types
// =============================================================================

/** Site index from artifacts/site/index.json */
export interface SiteIndex {
  schema_version: string;
  generated_at: string;
  campaigns: CampaignSummary[];
}

/** Campaign summary from index */
export interface CampaignSummary {
  campaign_id: string;
  name: string;
  tag: string;
  status: string;
  runs_count: number;
  created_at: string;
  detail_path: string;
}

/** Campaign detail */
export interface CampaignDetail {
  schema_version: string;
  campaign: CampaignInfo;
  runs: RunSummary[];
}

/** Campaign info */
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

/** Run summary from campaign */
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

/** Run detail */
export interface RunDetail {
  schema_version: string;
  run: RunInfo;
  config_snapshot?: unknown;
  metrics: RunMetrics;
  top_candidates: TopCandidateEntry[];
  exports: RunExports;
}

/** Run info */
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

/** Run metrics */
export interface RunMetrics {
  generations_completed?: number;
  total_evaluations?: number;
  data_integrity_verdict?: string;
  data_integrity_score?: number;
  best_oos_sharpe_net?: number;
  best_pbo?: number;
  research_candidates_count?: number;
  validated_candidates_count?: number;
}

/** Top candidate entry */
export interface TopCandidateEntry {
  candidate_id: string;
  genome_hash: string;
  rank: number;
  candidate_class?: string;
  oos_sharpe_net: number;
  pbo: number;
  oos_cagr_net?: number;
  gates_passed?: boolean;
}

/** Run exports */
export interface RunExports {
  top1000_json?: string;
  top1000_csv?: string;
  data_integrity_report?: string;
}

// =============================================================================
// CANDIDATE TYPES
// =============================================================================

/** Candidate list item for table display */
export interface CandidateListItem {
  rank: number;
  candidate_id: string;
  display_name: string;
  candidate_class: string;
  oos_sharpe_net: number;
  pbo: number;
  oos_cagr_net: number;
  max_drawdown_net: number;
  dsr: number;
  stress_passed: number;
  stress_total: number;
  gates_passed: boolean;
  created_at?: string;
}

/** Full candidate detail */
export interface CandidateDetailFull {
  candidate_id: string;
  genome_hash: string;
  rank: number;
  candidate_class: string;
  display_name: string;
  
  // Metrics
  oos_sharpe_net: number;
  oos_sharpe_gross?: number;
  pbo: number;
  dsr?: number;
  oos_cagr_net?: number;
  max_drawdown_net?: number;
  turnover_annual?: number;
  capacity_usd?: number;
  
  // Stress & Gates
  stress_passed: number;
  stress_total: number;
  gates_passed: boolean;
  
  // Strategy
  strategy: StrategyConfig;
  
  // Execution
  execution: ExecutionConfig;
  
  // Provenance
  provenance: Provenance;
  
  // Data Integrity
  data_integrity?: DataIntegrityInfo;
  
  // Paths
  bundle_path: string;
  strategy_toml_path: string;
  replay_script_path?: string;
}

/** Strategy configuration */
export interface StrategyConfig {
  id: string;
  version: string;
  description?: string;
  author?: string;
  pipeline: PipelineBlock[];
  rebalance?: RebalanceConfig;
  constraints?: ConstraintsConfig;
}

/** Pipeline block */
export interface PipelineBlock {
  block_type: string;
  block_id: string;
  enabled: boolean;
  params: Record<string, unknown>;
}

/** Rebalance config */
export interface RebalanceConfig {
  frequency: string;
  day?: string;
}

/** Constraints config */
export interface ConstraintsConfig {
  max_weight_per_asset?: number;
  min_liquidity_brl?: number;
  max_positions?: number;
}

/** Execution config */
export interface ExecutionConfig {
  delay_bars: number;
  bypass_for_debug: boolean;
  slippage: SlippageConfig;
  fees: FeesConfig;
  fill_policy?: FillPolicyConfig;
}

/** Slippage config */
export interface SlippageConfig {
  slippage_type: string;
  bps?: number;
}

/** Fees config */
export interface FeesConfig {
  tier: string;
}

/** Fill policy config */
export interface FillPolicyConfig {
  allow_partial?: boolean;
  max_participation?: number;
}

/** Provenance */
export interface Provenance {
  candidate_id: string;
  genome_hash: string;
  run_id: string;
  campaign_id: string;
  seed: number;
  git_sha?: string;
  git_branch?: string;
  config_hash?: string;
  dataset_hash?: string;
  created_at: string;
  scg_version?: string;
  original_report_path?: string;
}

/** Data integrity info */
export interface DataIntegrityInfo {
  verdict: string;
  score: number;
  passed_count: number;
  warning_count: number;
  critical_count: number;
  warnings: string[];
}

// =============================================================================
// BACKTEST TYPES
// =============================================================================

/** Timeseries point */
export interface TimeseriesPoint {
  date: string;
  equity: number;
  drawdown: number;
  exposure?: number;
  vol_exante?: number;
  vol_expost?: number;
}

/** Backtest result */
export interface BacktestResult {
  available: boolean;
  candidate_id: string;
  message?: string;
  metadata?: BacktestMetadata;
  metrics?: BacktestMetrics;
  timeseries: TimeseriesPoint[];
  backtest_path?: string;
}

/** Backtest metadata */
export interface BacktestMetadata {
  schema_version?: string;
  run_id: string;
  config_hash?: string;
  strategy_id: string;
  timestamp_utc?: string;
  mode?: string;
  duration_ms?: number;
}

/** Backtest metrics */
export interface BacktestMetrics {
  cagr: number;
  volatility: number;
  sharpe_ratio: number;
  max_drawdown: number;
  max_drawdown_duration_days?: number;
  turnover_annual?: number;
  hit_rate?: number;
  profit_factor?: number;
  total_trades: number;
  total_days?: number;
  sortino_ratio?: number;
  calmar_ratio?: number;
}

// =============================================================================
// LEGACY TYPES (for backward compatibility)
// =============================================================================

export interface ScgReport {
  experiment_id: string;
  status: string;
  generations_completed: number;
  total_evaluations: number;
  cache_hits: number;
  duration_seconds: number;
  hall_of_fame_size: number;
  generation_stats: GenerationStats[];
  top_strategies: TopStrategy[];
}

export interface GenerationStats {
  generation: number;
  population_size: number;
  evaluated: number;
  cache_hits: number;
  best_sharpe: number;
  best_cagr: number;
  mean_sharpe: number;
  pareto_size: number;
  duration_ms: number;
}

export interface TopStrategy {
  rank: number;
  id: string;
  sharpe: number;
  cagr: number;
  max_dd: number;
}

export interface EquityPoint {
  time: string;
  value: number;
}

export interface DashboardOverview {
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  total_trades: number;
  active_candidates: number;
  current_generation: number;
  best_cagr: number;
  system_status: string;
  last_update: string;
}

export interface ExperimentListing {
  id: string;
  path: string;
  status: string;
  created_at: string;
  generations: number;
  best_sharpe: number;
}

// =============================================================================
// CANDIDATE FILTERS
// =============================================================================

export interface CandidateFilters {
  candidate_class?: string;
  min_sharpe?: number;
  max_pbo?: number;
  search?: string;
  limit?: number;
}

// =============================================================================
// ADVANCED ANALYTICS TYPES
// =============================================================================

/** Rolling data point */
export interface RollingPoint {
  date: string;
  value: number;
}

/** Monthly return for heatmap */
export interface MonthlyReturn {
  year: number;
  month: number;
  return_pct: number;
}

/** Comprehensive risk metrics */
export interface RiskMetrics {
  candidate_id: string;
  var_95: number;
  var_99: number;
  cvar_95: number;
  cvar_99: number;
  tail_ratio: number;
  omega_ratio: number;
  gain_to_pain: number;
  skewness: number;
  kurtosis: number;
  stability_of_timeseries: number;
  longest_dd_days: number;
  average_dd_days: number;
  time_underwater_pct: number;
  information_ratio?: number;
  treynor_ratio?: number;
  sortino_ratio: number;
  calmar_ratio: number;
  best_day: number;
  worst_day: number;
  best_month: number;
  worst_month: number;
  payoff_ratio: number;
  rolling_sharpe: RollingPoint[];
  rolling_volatility: RollingPoint[];
  rolling_returns: RollingPoint[];
  daily_returns: number[];
  monthly_returns: MonthlyReturn[];
}

/** Strategy comparison result */
export interface ComparisonResult {
  candidates: ComparisonCandidate[];
  correlation_matrix: number[][];
  combined_equity: TimeseriesPoint[];
  diversification_ratio: number;
}

/** Comparison candidate */
export interface ComparisonCandidate {
  candidate_id: string;
  display_name: string;
  sharpe: number;
  cagr: number;
  max_dd: number;
  pbo: number;
  volatility: number;
  calmar: number;
  sortino: number;
  equity: TimeseriesPoint[];
}

/** Walk-forward analysis result */
export interface WalkForwardResult {
  candidate_id: string;
  windows: WalkForwardWindow[];
  aggregate_sharpe: number;
  degradation_ratio: number;
  consistency_score: number;
  profit_periods: number;
  loss_periods: number;
}

/** Walk-forward window */
export interface WalkForwardWindow {
  period_start: string;
  period_end: string;
  is_sharpe: number;
  oos_sharpe: number;
  is_return: number;
  oos_return: number;
  is_max_dd: number;
  oos_max_dd: number;
}

/** Monte Carlo simulation result */
export interface MonteCarloResult {
  candidate_id: string;
  num_simulations: number;
  sharpe_distribution: DistributionStats;
  cagr_distribution: DistributionStats;
  max_dd_distribution: DistributionStats;
  equity_paths: number[][];
  confidence_bands: ConfidenceBands;
}

/** Distribution statistics */
export interface DistributionStats {
  mean: number;
  std: number;
  p5: number;
  p25: number;
  p50: number;
  p75: number;
  p95: number;
  histogram: number[];
  histogram_bins: number[];
}

/** Confidence bands */
export interface ConfidenceBands {
  dates: string[];
  p5: number[];
  p25: number[];
  p50: number[];
  p75: number[];
  p95: number[];
}

/** Regime analysis result */
export interface RegimeAnalysis {
  candidate_id: string;
  regimes: RegimePeriod[];
  performance_by_regime: Record<string, RegimeMetrics>;
  current_regime: string;
  regime_stats: RegimeStat[];
}

/** Regime period */
export interface RegimePeriod {
  start_date: string;
  end_date: string;
  regime: string;
  color: string;
}

/** Regime metrics */
export interface RegimeMetrics {
  sharpe: number;
  cagr: number;
  volatility: number;
  max_dd: number;
  hit_rate: number;
  avg_return: number;
  num_days: number;
}

/** Regime statistics */
export interface RegimeStat {
  regime: string;
  frequency: number;
  avg_duration_days: number;
}

// =============================================================================
// STORE STATE
// =============================================================================

/** Recent run for quick selection */
export interface RecentRun {
  run_id: string;
  campaign_id: string;
  campaign_name: string;
  campaign_tag?: string;
  seed: number;
  status: string;
  started_at?: string;
  completed_at?: string;
  duration_secs?: number;
  candidates_count: number;
  best_oos_sharpe_net?: number;
}

interface DataState {
  // Artifacts root
  artifactsRoot: string | null;
  
  // Site index & navigation
  siteIndex: SiteIndex | null;
  campaigns: CampaignSummary[];
  selectedCampaign: CampaignDetail | null;
  runs: RunSummary[];
  recentRuns: RecentRun[];
  selectedRun: RunDetail | null;
  
  // Candidates
  candidates: CandidateListItem[];
  selectedCandidate: CandidateDetailFull | null;
  candidateFilters: CandidateFilters;
  selectedCandidateIds: string[];  // For multi-select comparison
  
  // Backtest
  backtest: BacktestResult | null;
  
  // Advanced Analytics
  riskMetrics: RiskMetrics | null;
  comparisonResult: ComparisonResult | null;
  walkForwardResult: WalkForwardResult | null;
  monteCarloResult: MonteCarloResult | null;
  regimeAnalysis: RegimeAnalysis | null;
  
  // Legacy data
  experiments: ExperimentListing[];
  currentExperiment: ScgReport | null;
  overview: DashboardOverview | null;
  equityData: EquityPoint[];
  
  // UI State
  isLoading: boolean;
  error: string | null;
  selectedRunId: string | null;
  
  // Actions - Artifact Indexer
  setArtifactsRoot: (path: string) => Promise<void>;
  loadIndex: () => Promise<void>;
  fetchRecentRuns: (limit?: number) => Promise<void>;
  loadCampaign: (campaignId: string) => Promise<void>;
  loadRun: (runId: string) => Promise<void>;
  listCandidates: (runId: string, filters?: CandidateFilters) => Promise<void>;
  loadCandidateDetail: (candidateId: string) => Promise<void>;
  setSelectedCandidate: (candidate: CandidateDetailFull | null) => void;
  loadBacktest: (candidateId: string) => Promise<void>;
  setCandidateFilters: (filters: CandidateFilters) => void;
  clearSelectedCandidate: () => void;
  toggleCandidateSelection: (candidateId: string) => void;
  clearCandidateSelection: () => void;
  
  // Actions - Advanced Analytics
  loadRiskMetrics: (candidateId: string) => Promise<void>;
  compareCandidates: (candidateIds: string[]) => Promise<void>;
  loadWalkForward: (candidateId: string, windowMonths?: number, stepMonths?: number) => Promise<void>;
  runMonteCarlo: (candidateId: string, numSimulations?: number, blockSize?: number) => Promise<void>;
  detectRegimes: (candidateId: string, volThreshold?: number) => Promise<void>;
  
  // Actions - File Watcher
  startWatcher: () => Promise<void>;
  invalidateCache: () => Promise<void>;
  
  // Actions - Legacy
  fetchExperiments: () => Promise<void>;
  fetchOverview: () => Promise<void>;
  loadExperiment: (id: string) => Promise<void>;
  loadEquityData: (path: string) => Promise<void>;
  
  // Utilities
  clearError: () => void;
}

// =============================================================================
// STORE IMPLEMENTATION
// =============================================================================

export const useDataStore = create<DataState>((set, get) => ({
  // Initial state
  artifactsRoot: null,
  siteIndex: null,
  campaigns: [],
  selectedCampaign: null,
  runs: [],
  recentRuns: [],
  selectedRun: null,
  candidates: [],
  selectedCandidate: null,
  candidateFilters: {},
  selectedCandidateIds: [],
  backtest: null,
  riskMetrics: null,
  comparisonResult: null,
  walkForwardResult: null,
  monteCarloResult: null,
  regimeAnalysis: null,
  experiments: [],
  currentExperiment: null,
  overview: null,
  equityData: [],
  isLoading: false,
  error: null,
  selectedRunId: null,

  // ==========================================================================
  // ARTIFACT INDEXER ACTIONS (Using Unified Command Layer)
  // ==========================================================================

  setArtifactsRoot: async (path: string) => {
    set({ isLoading: true, error: null });
    try {
      const artifactsRoot = await cmd.setArtifactsRoot(path);
      set({ artifactsRoot, isLoading: false });
      // Auto-load index
      await get().loadIndex();
    } catch (error) {
      set({ error: String(error), isLoading: false });
    }
  },

  loadIndex: async () => {
    set({ isLoading: true, error: null });
    try {
      const siteIndex = await cmd.loadIndex();
      set({ 
        siteIndex, 
        campaigns: siteIndex.campaigns,
        isLoading: false 
      });
    } catch (error) {
      set({ error: String(error), isLoading: false });
    }
  },

  fetchRecentRuns: async (limit = 10) => {
    set({ isLoading: true, error: null });
    try {
      const runs = await cmd.listRecentRuns(limit);
      // Map to RecentRun format
      const recentRuns = runs.map(r => ({
        run_id: r.run_id,
        campaign_id: '',
        campaign_name: r.campaign_name,
        seed: 0,
        status: r.status,
        candidates_count: r.candidates_count || 0,
        best_oos_sharpe_net: r.best_oos_sharpe_net,
        started_at: r.created_at,
      }));
      set({ recentRuns, isLoading: false });
    } catch (error) {
      set({ error: String(error), isLoading: false });
    }
  },

  loadCampaign: async (campaignId: string) => {
    set({ isLoading: true, error: null });
    try {
      const campaign = await cmd.loadCampaign(campaignId);
      set({ 
        selectedCampaign: campaign, 
        runs: campaign.runs,
        isLoading: false 
      });
    } catch (error) {
      set({ error: String(error), isLoading: false });
    }
  },

  loadRun: async (runId: string) => {
    set({ isLoading: true, error: null, selectedRunId: runId });
    try {
      const run = await cmd.loadRun(runId);
      set({ selectedRun: run, isLoading: false });
      // Auto-load candidates for this run
      await get().listCandidates(runId);
    } catch (error) {
      set({ error: String(error), isLoading: false });
    }
  },

  listCandidates: async (runId: string, filters?: CandidateFilters) => {
    set({ isLoading: true, error: null });
    const mergedFilters = { ...get().candidateFilters, ...filters };
    try {
      const candidates = await cmd.listCandidates(runId, {
        candidateClass: mergedFilters.candidate_class,
        limit: mergedFilters.limit,
        maxPbo: mergedFilters.max_pbo,
        search: mergedFilters.search,
      });
      // Map to CandidateListItem format
      const mappedCandidates = candidates.map(c => ({
        ...c,
        stress_passed: typeof c.stress_passed === 'boolean' ? (c.stress_passed ? 1 : 0) : (c.stress_passed || 0),
        stress_total: c.stress_total || 8,
      }));
      set({ candidates: mappedCandidates, candidateFilters: mergedFilters, isLoading: false });
    } catch (error) {
      set({ error: String(error), isLoading: false });
    }
  },

  loadCandidateDetail: async (candidateId: string) => {
    set({ isLoading: true, error: null });
    try {
      const candidate = await cmd.loadCandidateDetail(candidateId);
      // Map to CandidateDetailFull format
      set({ 
        selectedCandidate: candidate as unknown as CandidateDetailFull, 
        isLoading: false 
      });
    } catch (error) {
      set({ error: String(error), isLoading: false });
    }
  },

  loadBacktest: async (candidateId: string) => {
    set({ isLoading: true, error: null });
    try {
      const backtest = await cmd.loadBacktestSeries(candidateId);
      set({ backtest, isLoading: false });
    } catch (error) {
      set({ error: String(error), isLoading: false });
    }
  },

  setCandidateFilters: (filters: CandidateFilters) => {
    set({ candidateFilters: filters });
  },

  clearSelectedCandidate: () => {
    set({ selectedCandidate: null });
  },

  setSelectedCandidate: (candidate: CandidateDetailFull | null) => {
    set({ selectedCandidate: candidate });
  },

  toggleCandidateSelection: (candidateId: string) => {
    const { selectedCandidateIds } = get();
    const isSelected = selectedCandidateIds.includes(candidateId);
    if (isSelected) {
      set({ selectedCandidateIds: selectedCandidateIds.filter(id => id !== candidateId) });
    } else {
      set({ selectedCandidateIds: [...selectedCandidateIds, candidateId] });
    }
  },

  clearCandidateSelection: () => {
    set({ selectedCandidateIds: [] });
  },

  // ==========================================================================
  // ADVANCED ANALYTICS ACTIONS
  // ==========================================================================

  loadRiskMetrics: async (candidateId: string) => {
    set({ isLoading: true, error: null });
    try {
      const riskMetrics = await safeInvoke<RiskMetrics>('calculate_risk_metrics', { candidateId });
      set({ riskMetrics, isLoading: false });
    } catch (error) {
      set({ error: String(error), isLoading: false });
    }
  },

  compareCandidates: async (candidateIds: string[]) => {
    set({ isLoading: true, error: null });
    try {
      const comparisonResult = await safeInvoke<ComparisonResult>('compare_candidates', { candidateIds });
      set({ comparisonResult, isLoading: false });
    } catch (error) {
      set({ error: String(error), isLoading: false });
    }
  },

  loadWalkForward: async (candidateId: string, windowMonths = 12, stepMonths = 3) => {
    set({ isLoading: true, error: null });
    try {
      const walkForwardResult = await safeInvoke<WalkForwardResult>('calculate_walk_forward', {
        candidateId,
        windowMonths,
        stepMonths,
      });
      set({ walkForwardResult, isLoading: false });
    } catch (error) {
      set({ error: String(error), isLoading: false });
    }
  },

  runMonteCarlo: async (candidateId: string, numSimulations = 1000, blockSize = 5) => {
    set({ isLoading: true, error: null });
    try {
      const monteCarloResult = await safeInvoke<MonteCarloResult>('run_monte_carlo', {
        candidateId,
        numSimulations,
        blockSize,
      });
      set({ monteCarloResult, isLoading: false });
    } catch (error) {
      set({ error: String(error), isLoading: false });
    }
  },

  detectRegimes: async (candidateId: string, volThreshold?: number) => {
    set({ isLoading: true, error: null });
    try {
      const regimeAnalysis = await safeInvoke<RegimeAnalysis>('detect_regimes', {
        candidateId,
        volThreshold,
      });
      set({ regimeAnalysis, isLoading: false });
    } catch (error) {
      set({ error: String(error), isLoading: false });
    }
  },

  // ==========================================================================
  // FILE WATCHER (Unified for Tauri and Browser)
  // ==========================================================================

  startWatcher: async () => {
    try {
      if (platform.isTauri) {
        // Tauri: Use native file watcher
        await cmd.watchArtifacts();
        
        // Listen for artifacts_changed events
        await listen<{ paths: string[]; event_type: string }>('artifacts_changed', (event) => {
          console.log('Artifacts changed:', event.payload);
          
          const { selectedRunId } = get();
          
          if (event.payload.paths.some(p => p.includes('index.json') || p.includes('campaign_'))) {
            get().loadIndex();
          }
          
          if (selectedRunId && event.payload.paths.some(p => p.includes('top1000'))) {
            get().listCandidates(selectedRunId);
          }
        });
      } else if (features.useSSE) {
        // Browser: Use SSE for real-time updates
        createSSEConnection((event: SSEEvent) => {
          console.log('[SSE] Event:', event.type);
          
          const { selectedRunId } = get();
          
          if (event.type === 'artifact-change' || event.type === 'cache-invalidated') {
            get().loadIndex();
          }
          
          if (event.type === 'run-complete' && selectedRunId) {
            get().listCandidates(selectedRunId);
          }
        });
      } else if (features.usePolling) {
        // Browser fallback: Use polling
        let lastCheck = Date.now();
        setInterval(async () => {
          try {
            const { has_changes } = await cmd.pollChanges(lastCheck);
            if (has_changes) {
              get().loadIndex();
            }
            lastCheck = Date.now();
          } catch (e) {
            console.warn('[Polling] Failed:', e);
          }
        }, config.pollIntervalMs);
      }
    } catch (error) {
      console.error('Failed to start watcher:', error);
    }
  },

  invalidateCache: async () => {
    try {
      await cmd.invalidateCache();
      // Refresh data
      await get().loadIndex();
    } catch (error) {
      console.error('Failed to invalidate cache:', error);
    }
  },

  // ==========================================================================
  // LEGACY ACTIONS
  // ==========================================================================

  fetchExperiments: async () => {
    set({ isLoading: true, error: null });
    try {
      const experiments = await safeInvoke<ExperimentListing[]>('list_experiments');
      set({ experiments, isLoading: false });
    } catch (error) {
      set({ error: String(error), isLoading: false });
    }
  },

  fetchOverview: async () => {
    set({ isLoading: true, error: null });
    try {
      const overview = await safeInvoke<DashboardOverview>('get_dashboard_overview');
      set({ overview, isLoading: false });
    } catch (error) {
      set({ error: String(error), isLoading: false });
    }
  },

  loadExperiment: async (id: string) => {
    set({ isLoading: true, error: null });
    try {
      const report = await safeInvoke<ScgReport>('load_scg_report', { experimentId: id });
      set({ currentExperiment: report, isLoading: false });
    } catch (error) {
      set({ error: String(error), isLoading: false });
    }
  },

  loadEquityData: async (path: string) => {
    try {
      const equityData = await safeInvoke<EquityPoint[]>('load_nav_history', { experimentPath: path });
      set({ equityData });
    } catch (error) {
      console.warn('Using mock equity data:', error);
      set({ equityData: generateMockEquityData() });
    }
  },

  // ==========================================================================
  // UTILITIES
  // ==========================================================================

  clearError: () => set({ error: null }),
}));

// =============================================================================
// HELPERS
// =============================================================================

function generateMockEquityData(): EquityPoint[] {
  const data: EquityPoint[] = [];
  let value = 100000;
  
  for (let i = 0; i < 252; i++) {
    const date = new Date(2024, 0, 1);
    date.setDate(date.getDate() + i);
    
    value *= 1 + (Math.random() - 0.48) * 0.02;
    
    data.push({
      time: date.toISOString().split('T')[0],
      value,
    });
  }
  
  return data;
}

// =============================================================================
// AUTO-INIT FOR BROWSER MODE
// =============================================================================

// Auto-initialize the store in browser mode
if (features.autoInitialize) {
  setTimeout(async () => {
    const root = await initBrowserMode();
    if (root) {
      const store = useDataStore.getState();
      store.setArtifactsRoot(root);
    }
    // Also start watcher for real-time updates
    const store = useDataStore.getState();
    store.startWatcher();
  }, 100);
}

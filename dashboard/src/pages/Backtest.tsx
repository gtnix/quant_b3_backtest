import { useState, useEffect, useMemo, useCallback } from 'react';
import { MetricCard } from '../components/ui/MetricCard';
import { EquityChart } from '../components/charts/EquityChart';
import { DrawdownChart } from '../components/charts/DrawdownChart';
import { ReturnDistribution } from '../components/charts/ReturnDistribution';
import { MonthlyHeatmap } from '../components/charts/MonthlyHeatmap';
import { RollingMetrics } from '../components/charts/RollingMetrics';
import { Sparkline, SparkBar } from '../components/charts/Sparkline';
import { BloombergTooltip, MetricTooltips } from '../components/ui/BloombergTooltip';
import { StrategyPipeline } from '../components/StrategyPipeline';
import { WFAAnalysis } from '../components/WFAAnalysis';
import { StressAnalysis } from '../components/StressAnalysis';
import { RiskDecomposition } from '../components/RiskDecomposition';
import { useDataStore } from '../stores/dataStore';
import { config, platform } from '../lib/platform';
import type { MonthlyReturn, RollingPoint, CandidateSummary } from '../stores/dataStore';
import { 
  Play, 
  FileText, 
  Calendar,
  TrendingUp,
  TrendingDown,
  Activity,
  Target,
  BarChart3,
  AlertTriangle,
  RefreshCw,
  ExternalLink,
  PieChart,
  Layers,
  CalendarDays,
  LineChart,
  Award,
  Shield,
  CheckCircle,
  XCircle,
  Search,
  Database,
  Zap,
  Download,
  GitCompare,
  Shuffle,
  ArrowUpRight,
  ArrowDownRight,
  Percent,
  Info,
  FlaskConical,
  Sigma,
  Gauge
} from 'lucide-react';

// =============================================================================
// STATISTICAL UTILITIES (Two Sigma Level)
// =============================================================================

/** Normal CDF approximation (Abramowitz and Stegun) */
function normalCDF(x: number): number {
  const a1 =  0.254829592;
  const a2 = -0.284496736;
  const a3 =  1.421413741;
  const a4 = -1.453152027;
  const a5 =  1.061405429;
  const p  =  0.3275911;
  
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x) / Math.sqrt(2);
  
  const t = 1.0 / (1.0 + p * x);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  
  return 0.5 * (1.0 + sign * y);
}

/** Calculate statistical metrics from daily returns */
function calculateStatistics(dailyReturns: number[], riskFreeDaily: number = 0.10 / 252) {
  if (dailyReturns.length < 2) {
    return {
      mean: 0, std: 0, skewness: 0, kurtosis: 0,
      sharpe: 0, sortino: 0, calmar: 0, omega: 0,
      var95: 0, cvar95: 0, tStat: 0, pValue: 1,
      tailRatio: 0, downsideStd: 0, annualizedVol: 0
    };
  }
  
  const n = dailyReturns.length;
  
  // Basic stats
  const mean = dailyReturns.reduce((a, b) => a + b, 0) / n;
  const variance = dailyReturns.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
  const std = Math.sqrt(variance);
  
  // Skewness (Fisher's)
  const skewness = std > 0 
    ? dailyReturns.reduce((a, b) => a + ((b - mean) / std) ** 3, 0) / n 
    : 0;
  
  // Excess Kurtosis (Fisher's)
  const kurtosis = std > 0 
    ? (dailyReturns.reduce((a, b) => a + ((b - mean) / std) ** 4, 0) / n) - 3 
    : 0;
  
  // Annualized volatility
  const annualizedVol = std * Math.sqrt(252);
  
  // Sharpe Ratio (annualized)
  const annualizedExcessReturn = (mean - riskFreeDaily) * 252;
  const sharpe = annualizedVol > 0 ? annualizedExcessReturn / annualizedVol : 0;
  
  // Sortino Ratio (downside deviation)
  const downsideReturns = dailyReturns.filter(r => r < riskFreeDaily);
  const downsideVariance = downsideReturns.length > 0 
    ? downsideReturns.reduce((a, b) => a + (b - riskFreeDaily) ** 2, 0) / downsideReturns.length 
    : 0;
  const downsideStd = Math.sqrt(downsideVariance);
  const sortino = downsideStd > 0 
    ? (mean - riskFreeDaily) * 252 / (downsideStd * Math.sqrt(252)) 
    : 0;
  
  // VaR and CVaR (95%)
  const sorted = [...dailyReturns].sort((a, b) => a - b);
  const var95Index = Math.floor(n * 0.05);
  const var95 = sorted[var95Index] || 0;
  const tailReturns = sorted.filter(r => r <= var95);
  const cvar95 = tailReturns.length > 0 
    ? tailReturns.reduce((a, b) => a + b, 0) / tailReturns.length 
    : var95;
  
  // Tail Ratio (P95/P5)
  const p95 = sorted[Math.floor(n * 0.95)] || 0;
  const p5 = sorted[Math.floor(n * 0.05)] || 0;
  const tailRatio = p5 !== 0 ? Math.abs(p95 / p5) : 0;
  
  // Omega Ratio (threshold = 0)
  const positiveSum = dailyReturns.filter(r => r > 0).reduce((a, b) => a + b, 0);
  const negativeSum = Math.abs(dailyReturns.filter(r => r < 0).reduce((a, b) => a + b, 0));
  const omega = negativeSum > 0 ? positiveSum / negativeSum : positiveSum > 0 ? Infinity : 1;
  
  // T-statistic for Sharpe ratio
  const tStat = sharpe * Math.sqrt(n / 252);
  const pValue = 2 * (1 - normalCDF(Math.abs(tStat)));
  
  // Calmar placeholder (needs max drawdown from equity)
  const calmar = 0; // Will be calculated with maxDD
  
  return {
    mean, std, skewness, kurtosis,
    sharpe, sortino, calmar, omega,
    var95, cvar95, tStat, pValue,
    tailRatio, downsideStd, annualizedVol
  };
}

/** Calculate Calmar Ratio from CAGR and Max Drawdown */
function calculateCalmar(cagr: number, maxDD: number): number {
  return maxDD !== 0 ? cagr / Math.abs(maxDD) : 0;
}

// =============================================================================
// TYPES
// =============================================================================

interface RecentCandidate {
  candidate_id: string;
  genome_hash: string;
  rank: number;
  display_name: string;
  oos_sharpe_net: number;
  oos_cagr_net: number;
  max_drawdown_net: number;
  pbo: number;
  dsr: number;
  gates_passed: boolean;
  stress_passed: number;
  stress_total: number;
  run_id: string;
  campaign_name: string;
  created_at: string;
}

interface SimulatedEquity {
  candidate_id: string;
  data_source: 'simulated';
  simulation_params: {
    target_cagr: number;
    target_sharpe: number;
    target_max_dd: number;
    derived_annual_vol: number;
    days: number;
    start_capital: number;
  };
  realized_metrics: {
    total_return: number;
    max_drawdown: number;
    final_equity: number;
  };
  timeseries: Array<{ date: string; equity: number; drawdown: number }>;
}

// =============================================================================
// CANDIDATE SELECTOR COMPONENT
// =============================================================================

function CandidateSelector({ onSelect }: { onSelect: (candidate: RecentCandidate) => void }) {
  const [candidates, setCandidates] = useState<RecentCandidate[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [marketFilter, setMarketFilter] = useState<'all' | 'br' | 'us'>('all');
  const [stageFilter, setStageFilter] = useState<'all' | 'validated' | 'research'>('validated'); // Default to validated

  useEffect(() => {
    loadRecentCandidates();
  }, []);

  const loadRecentCandidates = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${config.apiBase}/candidates/recent?limit=50`);
      if (response.ok) {
        const data = await response.json();
        setCandidates(data.candidates || []);
      }
    } catch (err) {
      console.error('Failed to load recent candidates:', err);
    } finally {
      setLoading(false);
    }
  };

  // Infer market from symbol format (B3: ends with number, US: letters only)
  const inferMarket = (c: RecentCandidate): 'br' | 'us' => {
    // Check display_name or candidate_id for market hints
    const name = c.display_name || c.candidate_id || '';
    if (/[A-Z]{4}\d/.test(name)) return 'br'; // e.g., PETR4, VALE3
    if (/^[A-Z]{1,5}$/.test(name.split('_')[0] || '')) return 'us'; // e.g., AAPL, MSFT
    return 'br'; // Default to B3
  };

  const filtered = candidates.filter(c => {
    // Stage filter (validated = Stage B, research = Stage A)
    if (stageFilter === 'validated' && c.source_stage !== 'B') return false;
    if (stageFilter === 'research' && c.source_stage !== 'A') return false;
    
    // Market filter
    if (marketFilter !== 'all') {
      const market = inferMarket(c);
      if (market !== marketFilter) return false;
    }
    // Search filter
    if (search) {
      const searchLower = search.toLowerCase();
      return c.display_name.toLowerCase().includes(searchLower) ||
             c.candidate_id.toLowerCase().includes(searchLower) ||
             c.campaign_name?.toLowerCase().includes(searchLower);
    }
    return true;
  });

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="w-8 h-8 animate-spin text-terminal-muted" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-white to-terminal-muted bg-clip-text text-transparent">
            Strategy Analyzer
          </h1>
          <p className="text-terminal-muted mt-2">
            Select a strategy to view detailed backtest analysis
          </p>
        </div>
        <button
          onClick={loadRecentCandidates}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-terminal-surface border border-terminal-border hover:border-profit transition-colors"
        >
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>
      </div>

      {/* Search and Market Filter */}
      <div className="flex flex-col sm:flex-row gap-4">
        {/* Market Selector */}
        <div className="flex rounded-xl overflow-hidden border border-terminal-border">
          <button
            onClick={() => setMarketFilter('all')}
            className={`px-4 py-3 text-sm font-medium transition-colors ${
              marketFilter === 'all' 
                ? 'bg-profit/20 text-profit' 
                : 'bg-terminal-surface text-terminal-muted hover:text-white'
            }`}
          >
            🌐 All
          </button>
          <button
            onClick={() => setMarketFilter('br')}
            className={`px-4 py-3 text-sm font-medium transition-colors border-l border-terminal-border ${
              marketFilter === 'br' 
                ? 'bg-green-500/20 text-green-400' 
                : 'bg-terminal-surface text-terminal-muted hover:text-white'
            }`}
          >
            🇧🇷 B3
          </button>
          <button
            onClick={() => setMarketFilter('us')}
            className={`px-4 py-3 text-sm font-medium transition-colors border-l border-terminal-border ${
              marketFilter === 'us' 
                ? 'bg-blue-500/20 text-blue-400' 
                : 'bg-terminal-surface text-terminal-muted hover:text-white'
            }`}
          >
            🇺🇸 US
          </button>
        </div>

        {/* Stage Filter */}
        <div className="flex rounded-xl overflow-hidden border border-terminal-border">
          <button
            onClick={() => setStageFilter('validated')}
            className={`px-4 py-3 text-sm font-medium transition-colors ${
              stageFilter === 'validated' 
                ? 'bg-emerald-500/20 text-emerald-400' 
                : 'bg-terminal-surface text-terminal-muted hover:text-white'
            }`}
          >
            ✓ Validated
          </button>
          <button
            onClick={() => setStageFilter('research')}
            className={`px-4 py-3 text-sm font-medium transition-colors border-l border-terminal-border ${
              stageFilter === 'research' 
                ? 'bg-amber-500/20 text-amber-400' 
                : 'bg-terminal-surface text-terminal-muted hover:text-white'
            }`}
          >
            🔬 Research
          </button>
          <button
            onClick={() => setStageFilter('all')}
            className={`px-4 py-3 text-sm font-medium transition-colors border-l border-terminal-border ${
              stageFilter === 'all' 
                ? 'bg-profit/20 text-profit' 
                : 'bg-terminal-surface text-terminal-muted hover:text-white'
            }`}
          >
            All Stages
          </button>
        </div>
        
        {/* Search */}
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-terminal-muted" />
          <input
            type="text"
            placeholder="Search by strategy name, ID, or campaign..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-11 pr-4 py-3 bg-terminal-surface border border-terminal-border rounded-xl focus:outline-none focus:border-profit text-sm"
          />
        </div>
      </div>

      {/* Stats Bar */}
      <div className="flex items-center gap-6 p-4 bg-gradient-to-r from-terminal-surface to-transparent rounded-xl border border-terminal-border">
        <div className="flex items-center gap-2">
          <Database className="w-5 h-5 text-accent-cyan" />
          <span className="text-terminal-muted text-sm">Total:</span>
          <span className="font-mono font-bold">{candidates.length}</span>
        </div>
        <div className="flex items-center gap-2">
          <CheckCircle className="w-5 h-5 text-profit" />
          <span className="text-terminal-muted text-sm">Validated:</span>
          <span className="font-mono font-bold text-profit">
            {candidates.filter(c => c.gates_passed).length}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-accent-yellow" />
          <span className="text-terminal-muted text-sm">Best Sharpe:</span>
          <span className="font-mono font-bold text-accent-yellow">
            {candidates.length > 0 ? Math.max(...candidates.map(c => c.oos_sharpe_net)).toFixed(2) : '-'}
          </span>
        </div>
      </div>

      {/* Candidate Cards Grid */}
      {filtered.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16 text-terminal-muted">
          <Award className="w-16 h-16 mb-4 opacity-30" />
          <p className="text-lg">No strategies found</p>
          <p className="text-sm mt-1">Run SCG to generate new candidates</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filtered.map((candidate) => (
            <CandidateCard
              key={candidate.candidate_id}
              candidate={candidate}
              onClick={() => onSelect(candidate)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// Check if candidate has valid metrics (quant sanity check)
function hasValidMetrics(c: RecentCandidate) {
  const issues: string[] = [];
  // Sharpe > 10 is unrealistic for any real strategy
  if (c.oos_sharpe_net > 10) issues.push('Sharpe unrealistic');
  // PBO = 0 means not computed
  if (c.pbo === 0) issues.push('PBO not computed');
  // DSR = 0 means not computed
  if (c.dsr === 0) issues.push('DSR not computed');
  // MaxDD null means not computed
  if (c.max_drawdown_net == null || c.max_drawdown_missing) issues.push('MaxDD missing');
  return { valid: issues.length === 0, issues };
}

function CandidateCard({ candidate, onClick }: { candidate: RecentCandidate; onClick: () => void }) {
  const sharpeColor = candidate.oos_sharpe_net >= 1.0 ? 'text-profit' : 
                       candidate.oos_sharpe_net >= 0.5 ? 'text-accent-yellow' : 'text-loss';
  const cagrPct = (candidate.oos_cagr_net * 100).toFixed(1);
  const ddPct = (Math.abs(candidate.max_drawdown_net || 0) * 100).toFixed(1);
  const metricsCheck = hasValidMetrics(candidate);

  return (
    <button
      onClick={onClick}
      className="group text-left p-5 bg-terminal-surface border border-terminal-border rounded-xl hover:border-profit transition-all hover:shadow-lg hover:shadow-profit/5"
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1 flex-wrap">
            <span className={`px-2 py-0.5 rounded text-xs font-medium ${
              candidate.gates_passed ? 'bg-profit/20 text-profit' : 'bg-accent-cyan/20 text-accent-cyan'
            }`}>
              {candidate.gates_passed ? 'Validated' : 'Research'}
            </span>
            <span className="text-xs text-terminal-muted font-mono">#{candidate.rank}</span>
            {!metricsCheck.valid && (
              <span 
                className="px-2 py-0.5 rounded text-xs font-medium bg-amber-500/20 text-amber-400 cursor-help"
                title={metricsCheck.issues.join(', ')}
              >
                ⚠ Incomplete
              </span>
            )}
          </div>
          <h3 className="font-medium truncate group-hover:text-profit transition-colors">
            {candidate.display_name}
          </h3>
          <p className="text-xs text-terminal-muted truncate mt-0.5">
            {candidate.campaign_name || 'Unknown Campaign'}
          </p>
        </div>
        <ExternalLink className="w-4 h-4 text-terminal-muted group-hover:text-profit transition-colors flex-shrink-0 mt-1" />
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-3 gap-3">
        <div>
          <div className="text-[10px] text-terminal-muted uppercase tracking-wide">Sharpe</div>
          <div className={`font-mono font-bold text-lg ${sharpeColor}`}>
            {candidate.oos_sharpe_net.toFixed(2)}
          </div>
        </div>
        <div>
          <div className="text-[10px] text-terminal-muted uppercase tracking-wide">CAGR</div>
          <div className="font-mono font-bold text-lg flex items-center">
            {parseFloat(cagrPct) >= 0 ? (
              <ArrowUpRight className="w-3 h-3 text-profit mr-0.5" />
            ) : (
              <ArrowDownRight className="w-3 h-3 text-loss mr-0.5" />
            )}
            {cagrPct}%
          </div>
        </div>
        <div>
          <div className="text-[10px] text-terminal-muted uppercase tracking-wide">Max DD</div>
          <div className="font-mono font-bold text-lg text-loss">-{ddPct}%</div>
        </div>
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between mt-4 pt-3 border-t border-terminal-border/50">
        <div className="flex items-center gap-2 text-xs text-terminal-muted">
          <Zap className="w-3 h-3" />
          PBO: {(candidate.pbo * 100).toFixed(0)}%
        </div>
        <div className="flex items-center gap-1 text-xs">
          <span className={candidate.stress_passed === candidate.stress_total ? 'text-profit' : 'text-accent-yellow'}>
            {candidate.stress_passed}/{candidate.stress_total}
          </span>
          <span className="text-terminal-muted">stress</span>
        </div>
      </div>
    </button>
  );
}

// =============================================================================
// MAIN BACKTEST COMPONENT
// =============================================================================

type TabType = 'overview' | 'distribution' | 'monthly' | 'rolling' | 'drawdown' | 'pipeline' | 'validation' | 'stress' | 'risk';

export function Backtest() {
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [simulatedData, setSimulatedData] = useState<SimulatedEquity | null>(null);
  const [loadingSimulated, setLoadingSimulated] = useState(false);
  const [logScale, setLogScale] = useState(false);
  
  const {
    selectedCandidate,
    backtest,
    riskMetrics,
    isLoading,
    error,
    loadBacktest,
    loadRiskMetrics,
    setSelectedCandidate
  } = useDataStore();

  // Load backtest or simulated data when candidate is selected
  useEffect(() => {
    if (selectedCandidate) {
      loadBacktest(selectedCandidate.candidate_id);
      
      // Risk metrics only available in Tauri mode
      if (platform.isTauri) {
        loadRiskMetrics(selectedCandidate.candidate_id);
      }
      
      // Also load simulated equity for Neon candidates
      if ((selectedCandidate as any).data_source === 'neon' || !backtest?.available) {
        loadSimulatedEquity(selectedCandidate.candidate_id);
      }
    }
  }, [selectedCandidate?.candidate_id]);

  // Listen for select-candidate events from Hall of Fame
  useEffect(() => {
    const handleSelectCandidateEvent = (e: CustomEvent) => {
      if (e.detail) {
        setSelectedCandidate(e.detail);
      }
    };
    
    window.addEventListener('select-candidate', handleSelectCandidateEvent as EventListener);
    return () => window.removeEventListener('select-candidate', handleSelectCandidateEvent as EventListener);
  }, [setSelectedCandidate]);

  const loadSimulatedEquity = async (candidateId: string) => {
    setLoadingSimulated(true);
    try {
      const response = await fetch(`${config.apiBase}/candidate/${candidateId}/simulated-equity?days=504`);
      if (response.ok) {
        const data = await response.json();
        setSimulatedData(data);
      }
    } catch (err) {
      console.error('Failed to load simulated equity:', err);
    } finally {
      setLoadingSimulated(false);
    }
  };

  // Handle selection from CandidateSelector
  const handleSelectCandidate = useCallback(async (candidate: RecentCandidate) => {
    // Fetch full candidate detail
    try {
      const response = await fetch(`${config.apiBase}/candidate/${candidate.candidate_id}`);
      if (response.ok) {
        const fullCandidate = await response.json();
        setSelectedCandidate(fullCandidate);
      }
    } catch (err) {
      console.error('Failed to load candidate detail:', err);
    }
  }, [setSelectedCandidate]);

  // Use real or simulated data
  const equityData = useMemo(() => {
    if (backtest?.timeseries && backtest.timeseries.length > 0) {
      return backtest.timeseries.map(t => ({ time: t.date, value: t.equity }));
    }
    if (simulatedData?.timeseries) {
      return simulatedData.timeseries.map(t => ({ time: t.date, value: t.equity }));
    }
    return [];
  }, [backtest?.timeseries, simulatedData?.timeseries]);

  const drawdownData = useMemo(() => {
    if (backtest?.timeseries && backtest.timeseries.length > 0) {
      return backtest.timeseries.map(t => ({ time: t.date, value: t.drawdown }));
    }
    if (simulatedData?.timeseries) {
      return simulatedData.timeseries.map(t => ({ time: t.date, value: t.drawdown }));
    }
    return [];
  }, [backtest?.timeseries, simulatedData?.timeseries]);

  const isSimulated = !backtest?.available && simulatedData !== null;

  // Calculate returns from timeseries
  const dailyReturns = useMemo(() => {
    if (equityData.length < 2) return [];
    return equityData.slice(1).map((t, i) => {
      const prev = equityData[i];
      return prev.value !== 0 ? (t.value - prev.value) / prev.value : 0;
    });
  }, [equityData]);

  // ==========================================================================
  // STATISTICAL METRICS (Two Sigma / Chicago Quant Level)
  // ==========================================================================
  const stats = useMemo(() => calculateStatistics(dailyReturns), [dailyReturns]);
  
  // Proper Total Return (from equity timeseries)
  const totalReturn = useMemo(() => {
    if (equityData.length < 2) return 0;
    const first = equityData[0].value;
    const last = equityData[equityData.length - 1].value;
    return first > 0 ? (last - first) / first : 0;
  }, [equityData]);

  // Max Drawdown from timeseries
  const computedMaxDD = useMemo(() => {
    if (equityData.length < 2) return 0;
    let peak = equityData[0].value;
    let maxDD = 0;
    for (const point of equityData) {
      if (point.value > peak) peak = point.value;
      const dd = (point.value - peak) / peak;
      if (dd < maxDD) maxDD = dd;
    }
    return maxDD;
  }, [equityData]);

  // CAGR from timeseries
  const computedCAGR = useMemo(() => {
    if (equityData.length < 2) return 0;
    const years = equityData.length / 252;
    if (years <= 0) return 0;
    const first = equityData[0].value;
    const last = equityData[equityData.length - 1].value;
    if (first <= 0) return 0;
    return Math.pow(last / first, 1 / years) - 1;
  }, [equityData]);

  // Calmar Ratio (CAGR / MaxDD)
  const calmarRatio = useMemo(() => {
    return calculateCalmar(computedCAGR, computedMaxDD);
  }, [computedCAGR, computedMaxDD]);

  // Rolling Sharpe (63-day = ~3 months)
  const rollingSharpe = useMemo((): RollingPoint[] => {
    const window = 63;
    const riskFreeDaily = 0.10 / 252; // CDI ~10%
    
    if (dailyReturns.length < window) return [];
    
    return dailyReturns.slice(window).map((_, i) => {
      const windowReturns = dailyReturns.slice(i, i + window);
      const mean = windowReturns.reduce((a, b) => a + b, 0) / windowReturns.length;
      const variance = windowReturns.reduce((a, b) => a + (b - mean) ** 2, 0) / windowReturns.length;
      const std = Math.sqrt(variance);
      const sharpe = std > 0 
        ? ((mean - riskFreeDaily) * Math.sqrt(252)) / (std * Math.sqrt(252)) 
        : 0;
      
      const date = equityData[i + window]?.time || '';
      return { date, value: sharpe };
    });
  }, [dailyReturns, equityData]);

  // Calculate monthly returns
  const monthlyReturns = useMemo((): MonthlyReturn[] => {
    if (equityData.length < 2) return [];
    
    const monthly: Record<string, { equity_start: number; equity_end: number; year: number; month: number }> = {};
    
    for (const point of equityData) {
      const date = point.time;
      if (date.length >= 7) {
        const yearMonth = date.substring(0, 7);
        const [year, month] = yearMonth.split('-').map(Number);
        
        if (!monthly[yearMonth]) {
          monthly[yearMonth] = { equity_start: point.value, equity_end: point.value, year, month };
        }
        monthly[yearMonth].equity_end = point.value;
      }
    }
    
    const keys = Object.keys(monthly).sort();
    const result: MonthlyReturn[] = [];
    
    for (let i = 0; i < keys.length; i++) {
      const curr = monthly[keys[i]];
      const prevEnd = i > 0 ? monthly[keys[i - 1]].equity_end : curr.equity_start;
      const ret = (curr.equity_end - prevEnd) / prevEnd;
      
      result.push({
        year: curr.year,
        month: curr.month,
        return_pct: ret * 100,
      });
    }
    
    return result;
  }, [equityData]);

  // Calculate rolling returns (21-day)
  const rollingReturns = useMemo((): RollingPoint[] => {
    if (equityData.length < 22) return [];
    
    const window = 21;
    return equityData.slice(window).map((point, i) => {
      const startEquity = equityData[i].value;
      const ret = (point.value - startEquity) / startEquity;
      return { date: point.time, value: ret * 100 };
    });
  }, [equityData]);

  // No candidate selected - show selector
  if (!selectedCandidate) {
    return <CandidateSelector onSelect={handleSelectCandidate} />;
  }

  // Loading state
  if (isLoading && !backtest && !simulatedData) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-4">
        <RefreshCw className="w-12 h-12 animate-spin text-terminal-muted" />
        <p className="text-terminal-muted">Loading backtest data...</p>
      </div>
    );
  }

  // Error or no data available
  if (error || (!backtest?.available && !simulatedData && !isLoading)) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-4">
        <AlertTriangle className="w-12 h-12 text-amber-500" />
        <h3 className="text-lg font-semibold">No Backtest Data Available</h3>
        <p className="text-terminal-muted text-center max-w-md">
          {error || 'The selected candidate does not have backtest timeseries data. This may occur for candidates that are still being evaluated.'}
        </p>
        <button
          onClick={() => setSelectedCandidate(null)}
          className="px-4 py-2 bg-accent-cyan/20 text-accent-cyan rounded-lg hover:bg-accent-cyan/30 transition-colors"
        >
          Select Another Candidate
        </button>
      </div>
    );
  }

  const metrics = backtest?.metrics;
  const startCapital = equityData.length > 0 ? equityData[0].value : (simulatedData?.simulation_params.start_capital ?? 100000);
  const finalEquity = equityData.length > 0 ? equityData[equityData.length - 1].value : startCapital;

  return (
    <div className="space-y-6">
      {/* Hero Header */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-terminal-surface via-terminal-bg to-terminal-surface border border-terminal-border p-6">
        <div className="absolute top-0 right-0 w-64 h-64 bg-profit/5 rounded-full blur-3xl" />
        
        <div className="relative flex items-start justify-between">
          {/* Left: Strategy Info */}
          <div className="flex items-start gap-6">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-profit/20 to-profit/5 flex items-center justify-center">
              <Award className="w-8 h-8 text-profit" />
            </div>
            <div>
              <div className="flex items-center gap-3 mb-2">
                <h1 className="text-2xl font-bold">{selectedCandidate.display_name}</h1>
                <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                  selectedCandidate.candidate_class === 'validated' 
                    ? 'bg-profit/20 text-profit' 
                    : 'bg-accent-cyan/20 text-accent-cyan'
                }`}>
                  {selectedCandidate.candidate_class || 'Research'}
                </span>
                {isSimulated && (
                  <span className="px-3 py-1 rounded-full text-xs font-medium bg-accent-yellow/20 text-accent-yellow flex items-center gap-1">
                    <Shuffle className="w-3 h-3" />
                    Simulated
                  </span>
                )}
              </div>
              <div className="flex items-center gap-4 text-sm text-terminal-muted">
                <span className="font-mono">{selectedCandidate.candidate_id.substring(0, 20)}...</span>
                <span className="flex items-center gap-1">
                  <Calendar className="w-3 h-3" />
                  {equityData.length > 0 
                    ? `${equityData[0].time} → ${equityData[equityData.length - 1].time}`
                    : 'N/A'}
                </span>
              </div>
            </div>
          </div>

          {/* Right: Actions */}
          <div className="flex items-center gap-2">
            <button 
              onClick={() => setSelectedCandidate(null)}
              className="flex items-center gap-2 px-3 py-2 rounded-lg bg-terminal-bg border border-terminal-border hover:border-terminal-muted transition-colors text-sm"
            >
              <Search className="w-4 h-4" />
              Change
            </button>
            <button className="flex items-center gap-2 px-3 py-2 rounded-lg bg-terminal-bg border border-terminal-border hover:border-profit transition-colors text-sm">
              <Download className="w-4 h-4" />
              Export
            </button>
            <button className="flex items-center gap-2 px-3 py-2 rounded-lg bg-profit/10 text-profit border border-profit/30 hover:bg-profit/20 transition-colors text-sm">
              <GitCompare className="w-4 h-4" />
              Compare
            </button>
          </div>
        </div>
      </div>

      {/* Scorecard Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <ScoreCard
          label="Total Return"
          value={`${(totalReturn * 100).toFixed(1)}%`}
          icon={totalReturn >= 0 ? <TrendingUp className="w-5 h-5" /> : <TrendingDown className="w-5 h-5" />}
          color={totalReturn >= 0 ? 'profit' : 'loss'}
          subtitle={`$${startCapital.toLocaleString()} → $${Math.round(finalEquity).toLocaleString()}`}
        />
        <ScoreCard
          label="Sharpe Ratio"
          value={selectedCandidate.oos_sharpe_net.toFixed(2)}
          icon={<Target className="w-5 h-5" />}
          color={selectedCandidate.oos_sharpe_net >= 1.0 ? 'profit' : selectedCandidate.oos_sharpe_net >= 0.5 ? 'warning' : 'loss'}
          subtitle={selectedCandidate.oos_sharpe_net >= 1.0 ? 'Excellent' : selectedCandidate.oos_sharpe_net >= 0.5 ? 'Good' : 'Poor'}
        />
        <ScoreCard
          label="Max Drawdown"
          value={`-${(Math.abs(selectedCandidate.max_drawdown_net || simulatedData?.realized_metrics.max_drawdown || 0) * 100).toFixed(1)}%`}
          icon={<TrendingDown className="w-5 h-5" />}
          color="loss"
          progress={Math.abs(selectedCandidate.max_drawdown_net || 0) * 100}
        />
        <ScoreCard
          label="Risk Score"
          value={selectedCandidate.gates_passed ? 'Low' : 'Medium'}
          icon={<Shield className="w-5 h-5" />}
          color={selectedCandidate.gates_passed ? 'profit' : 'warning'}
          subtitle={`PBO: ${(selectedCandidate.pbo * 100).toFixed(0)}%`}
        />
      </div>

      {/* Main Chart */}
      <div className="card-elevated">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-4">
            <h2 className="font-semibold text-lg">Equity Curve</h2>
            {isSimulated && (
              <span className="text-xs text-accent-yellow bg-accent-yellow/10 px-2 py-1 rounded">
                Monte Carlo Simulated
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setLogScale(!logScale)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                logScale 
                  ? 'bg-profit/20 text-profit border border-profit/30' 
                  : 'bg-terminal-surface border border-terminal-border hover:border-terminal-muted'
              }`}
            >
              Log Scale
            </button>
          </div>
        </div>
        <div className="h-[400px]">
          {equityData.length > 0 ? (
            <EquityChart data={equityData} logScale={logScale} />
          ) : loadingSimulated ? (
            <div className="flex items-center justify-center h-full">
              <RefreshCw className="w-8 h-8 animate-spin text-terminal-muted" />
            </div>
          ) : (
            <div className="flex items-center justify-center h-full text-terminal-muted">
              No data available
            </div>
          )}
        </div>
      </div>

      {/* Metrics Grid - Bloomberg Terminal Level with Sparklines */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Performance with Sparkline */}
        <div className="card-elevated relative overflow-hidden">
          <div className="absolute top-2 right-2 opacity-60">
            <Sparkline 
              data={equityData.slice(-30).map(e => e.value)} 
              width={60} 
              height={24}
              color={totalReturn >= 0 ? '#00ff88' : '#ef4444'}
            />
          </div>
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="w-4 h-4 text-profit" />
            <h3 className="text-sm font-semibold text-terminal-muted uppercase tracking-wider">Performance</h3>
          </div>
          <div className="space-y-2.5">
            <MetricRowBloomberg label="CAGR" value={`${(computedCAGR * 100).toFixed(2)}%`} color={computedCAGR >= 0 ? 'profit' : 'loss'} tooltip={`Compound Annual Growth Rate over ${(equityData.length / 252).toFixed(1)} years`} />
            <MetricRowBloomberg label="Total Return" value={`${(totalReturn * 100).toFixed(2)}%`} color={totalReturn >= 0 ? 'profit' : 'loss'} tooltip="Net return from first to last day" />
            <MetricRowBloomberg label="Best Month" value={monthlyReturns.length > 0 ? `${Math.max(...monthlyReturns.map(m => m.return_pct)).toFixed(2)}%` : 'N/A'} color="profit" />
            <MetricRowBloomberg label="Worst Month" value={monthlyReturns.length > 0 ? `${Math.min(...monthlyReturns.map(m => m.return_pct)).toFixed(2)}%` : 'N/A'} color="loss" />
            <MetricRowBloomberg label="Win Rate" value={`${((dailyReturns.filter(r => r > 0).length / (dailyReturns.length || 1)) * 100).toFixed(1)}%`} tooltip={`${dailyReturns.filter(r => r > 0).length} winning days / ${dailyReturns.length} total days`} />
          </div>
        </div>

        {/* Risk-Adjusted with Bloomberg Tooltips */}
        <div className="card-elevated relative overflow-hidden">
          <div className="absolute top-2 right-2 opacity-60">
            <Sparkline 
              data={rollingSharpe.slice(-30).map(r => r.value)} 
              width={60} 
              height={24}
              color="#00d4ff"
              showZeroLine
            />
          </div>
          <div className="flex items-center gap-2 mb-4">
            <Target className="w-4 h-4 text-accent-cyan" />
            <h3 className="text-sm font-semibold text-terminal-muted uppercase tracking-wider">Risk-Adjusted</h3>
          </div>
          <div className="space-y-2.5">
            <MetricRowBloomberg 
              label="Sharpe" 
              value={stats.sharpe.toFixed(3)} 
              quality={stats.sharpe >= 1.5 ? 'excellent' : stats.sharpe >= 1.0 ? 'good' : stats.sharpe >= 0.5 ? 'fair' : 'poor'}
              tooltip={MetricTooltips.sharpe(stats.sharpe)}
            />
            <MetricRowBloomberg 
              label="Sortino" 
              value={stats.sortino.toFixed(3)} 
              quality={stats.sortino >= 2.0 ? 'excellent' : stats.sortino >= 1.5 ? 'good' : stats.sortino >= 1.0 ? 'fair' : 'poor'}
              tooltip={MetricTooltips.sortino(stats.sortino)}
            />
            <MetricRowBloomberg 
              label="Calmar" 
              value={calmarRatio.toFixed(3)} 
              quality={calmarRatio >= 1.0 ? 'excellent' : calmarRatio >= 0.5 ? 'good' : 'fair'}
              tooltip={MetricTooltips.calmar(calmarRatio)}
            />
            <MetricRowBloomberg 
              label="Omega" 
              value={stats.omega === Infinity ? '∞' : stats.omega.toFixed(3)} 
              quality={stats.omega >= 2.0 ? 'excellent' : stats.omega >= 1.5 ? 'good' : 'fair'}
              tooltip={MetricTooltips.omega(stats.omega)}
            />
            <MetricRowBloomberg 
              label="DSR" 
              value={(selectedCandidate.dsr ?? 0).toFixed(3)} 
              quality={(selectedCandidate.dsr ?? 0) >= 0.5 ? 'good' : 'poor'}
              tooltip={MetricTooltips.dsr(selectedCandidate.dsr ?? 0)}
            />
          </div>
        </div>

        {/* Risk Metrics with Daily Returns Distribution */}
        <div className="card-elevated relative overflow-hidden">
          <div className="absolute top-2 right-2 opacity-60">
            <SparkBar 
              data={dailyReturns.slice(-20)} 
              width={50} 
              height={20}
            />
          </div>
          <div className="flex items-center gap-2 mb-4">
            <AlertTriangle className="w-4 h-4 text-loss" />
            <h3 className="text-sm font-semibold text-terminal-muted uppercase tracking-wider">Risk</h3>
          </div>
          <div className="space-y-2.5">
            <MetricRowBloomberg label="Volatility" value={`${(stats.annualizedVol * 100).toFixed(2)}%`} tooltip={`Annualized standard deviation of daily returns (daily: ${(stats.std * 100).toFixed(3)}%)`} />
            <MetricRowBloomberg label="Max DD" value={`${(computedMaxDD * 100).toFixed(2)}%`} color="loss" tooltip="Maximum peak-to-trough decline in portfolio value" />
            <MetricRowBloomberg 
              label="VaR 95%" 
              value={`${(stats.var95 * 100).toFixed(3)}%`} 
              color="loss"
              tooltip={MetricTooltips.var95(stats.var95)}
            />
            <MetricRowBloomberg 
              label="CVaR 95%" 
              value={`${(stats.cvar95 * 100).toFixed(3)}%`} 
              color="loss"
              tooltip={MetricTooltips.cvar95(stats.cvar95)}
            />
            <MetricRowBloomberg 
              label="Tail Ratio" 
              value={stats.tailRatio.toFixed(3)} 
              quality={stats.tailRatio >= 1.0 ? 'good' : 'poor'}
              tooltip="Ratio of 95th to 5th percentile returns. Higher = better asymmetry"
            />
          </div>
        </div>

        {/* Statistical Significance with Confidence Indicators */}
        <div className="card-elevated relative overflow-hidden">
          <div className="absolute top-2 right-2">
            {/* Confidence indicator */}
            <div className={`px-2 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wider ${
              stats.pValue <= 0.01 
                ? 'bg-profit/20 text-profit' 
                : stats.pValue <= 0.05 
                  ? 'bg-accent-yellow/20 text-accent-yellow' 
                  : 'bg-loss/20 text-loss'
            }`}>
              {stats.pValue <= 0.01 ? '99% CI' : stats.pValue <= 0.05 ? '95% CI' : 'Low CI'}
            </div>
          </div>
          <div className="flex items-center gap-2 mb-4">
            <FlaskConical className="w-4 h-4 text-accent-purple" />
            <h3 className="text-sm font-semibold text-terminal-muted uppercase tracking-wider">Statistics</h3>
          </div>
          <div className="space-y-2.5">
            <MetricRowBloomberg 
              label="T-Statistic" 
              value={stats.tStat.toFixed(3)} 
              quality={Math.abs(stats.tStat) >= 2.0 ? 'excellent' : Math.abs(stats.tStat) >= 1.96 ? 'good' : 'poor'}
              tooltip={MetricTooltips.tstat(stats.tStat)}
            />
            <MetricRowBloomberg 
              label="P-Value" 
              value={stats.pValue < 0.001 ? '<0.001' : stats.pValue.toFixed(4)} 
              quality={stats.pValue <= 0.01 ? 'excellent' : stats.pValue <= 0.05 ? 'good' : 'poor'}
              tooltip={MetricTooltips.pvalue(stats.pValue)}
            />
            <MetricRowBloomberg 
              label="Skewness" 
              value={stats.skewness.toFixed(3)} 
              quality={stats.skewness >= 0 ? 'good' : 'fair'}
              tooltip={MetricTooltips.skewness(stats.skewness)}
            />
            <MetricRowBloomberg 
              label="Kurtosis" 
              value={stats.kurtosis.toFixed(3)} 
              tooltip={MetricTooltips.kurtosis(stats.kurtosis)}
            />
            <MetricRowBloomberg 
              label="PBO" 
              value={`${((selectedCandidate.pbo || 0) * 100).toFixed(1)}%`} 
              color={selectedCandidate.pbo <= 0.15 ? undefined : 'loss'}
              tooltip={MetricTooltips.pbo(selectedCandidate.pbo || 0)}
            />
          </div>
        </div>
      </div>

      {/* Validation Summary */}
      <div className="card-elevated">
        <div className="flex items-center gap-2 mb-4">
          <Shield className="w-4 h-4 text-profit" />
          <h3 className="text-sm font-semibold text-terminal-muted uppercase tracking-wider">Validation Gates</h3>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
          <ValidationBadge label="All Gates" passed={selectedCandidate.gates_passed} />
          <ValidationBadge label="PBO < 15%" passed={selectedCandidate.pbo <= 0.15} value={`${((selectedCandidate.pbo || 0) * 100).toFixed(0)}%`} />
          <ValidationBadge label="DSR > 0.5" passed={(selectedCandidate.dsr || 0) >= 0.5} value={(selectedCandidate.dsr ?? 0).toFixed(2)} />
          <ValidationBadge label="Sharpe > 0.5" passed={stats.sharpe >= 0.5} value={stats.sharpe.toFixed(2)} />
          <ValidationBadge label="T-Stat > 2.0" passed={Math.abs(stats.tStat) >= 2.0} value={stats.tStat.toFixed(2)} />
          <ValidationBadge label="Stress Tests" passed={selectedCandidate.stress_passed >= selectedCandidate.stress_total * 0.8} value={`${selectedCandidate.stress_passed}/${selectedCandidate.stress_total}`} />
        </div>
      </div>

      {/* Tabs */}
      <div className="flex items-center gap-1 border-b border-terminal-border overflow-x-auto pb-0">
        {[
          { key: 'overview', label: 'Overview', icon: BarChart3 },
          { key: 'pipeline', label: 'Pipeline', icon: Layers },
          { key: 'validation', label: 'WFA', icon: FlaskConical },
          { key: 'stress', label: 'Stress', icon: AlertTriangle },
          { key: 'risk', label: 'Risk', icon: Shield },
          { key: 'drawdown', label: 'Drawdown', icon: TrendingDown },
          { key: 'distribution', label: 'Dist', icon: PieChart },
          { key: 'monthly', label: 'Monthly', icon: CalendarDays },
          { key: 'rolling', label: 'Rolling', icon: LineChart },
        ].map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key as TabType)}
            className={`flex items-center gap-1.5 pb-3 px-3 text-sm font-medium transition-colors relative whitespace-nowrap ${
              activeTab === key ? 'text-profit' : 'text-terminal-muted hover:text-white'
            }`}
          >
            <Icon className="w-4 h-4" />
            {label}
            {activeTab === key && (
              <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-profit" />
            )}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && metrics && (
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
          <MetricCard label="CAGR" value={(metrics.cagr || 0) * 100} format="percent" icon={<TrendingUp className="w-4 h-4 text-profit" />} />
          <MetricCard label="Sharpe" value={metrics.sharpe_ratio || 0} format="ratio" icon={<Target className="w-4 h-4" />} />
          <MetricCard label="Sortino" value={metrics.sortino_ratio ?? 0} format="ratio" />
          <MetricCard label="Calmar" value={metrics.calmar_ratio ?? 0} format="ratio" />
          <MetricCard label="Max DD" value={(metrics.max_drawdown || 0) * 100} format="percent" icon={<TrendingDown className="w-4 h-4 text-loss" />} />
          <MetricCard label="Win Rate" value={(metrics.hit_rate ?? 0) * 100} format="percent" icon={<Activity className="w-4 h-4" />} />
        </div>
      )}

      {activeTab === 'drawdown' && (
        <div className="card-elevated">
          <h2 className="font-semibold text-lg mb-4">Underwater Chart</h2>
          <div className="h-[400px]">
            {drawdownData.length > 0 ? (
              <DrawdownChart data={drawdownData} />
            ) : (
              <div className="flex items-center justify-center h-full text-terminal-muted">
                No drawdown data available
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === 'distribution' && (
        <div className="card-elevated">
          <h2 className="font-semibold text-lg mb-4">Daily Return Distribution</h2>
          <div className="h-[400px]">
            {dailyReturns.length > 0 ? (
              <ReturnDistribution returns={dailyReturns} showNormal />
            ) : (
              <div className="flex items-center justify-center h-full text-terminal-muted">
                No return data available
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === 'monthly' && (
        <div className="card-elevated">
          <h2 className="font-semibold text-lg mb-4">Monthly Returns Heatmap</h2>
          {monthlyReturns.length > 0 ? (
            <MonthlyHeatmap data={monthlyReturns} />
          ) : (
            <div className="flex items-center justify-center h-64 text-terminal-muted">
              No monthly data available
            </div>
          )}
        </div>
      )}

      {activeTab === 'rolling' && (
        <div className="space-y-6">
          {/* Rolling Sharpe - Primary */}
          <div className="card-elevated">
            <div className="flex items-center gap-3 mb-4">
              <h2 className="font-semibold text-lg">Rolling 63-Day Sharpe Ratio</h2>
              <span className="text-xs text-terminal-muted bg-terminal-bg px-2 py-1 rounded">~3 months window</span>
            </div>
            <div className="h-[300px]">
              {rollingSharpe.length > 0 ? (
                <RollingMetrics
                  data={[{ label: 'Sharpe Ratio', points: rollingSharpe, color: '#00ff88' }]}
                  showZeroLine
                />
              ) : (
                <div className="flex items-center justify-center h-full text-terminal-muted">
                  Not enough data (need 63+ days)
                </div>
              )}
            </div>
            {/* Rolling Sharpe Stats */}
            {rollingSharpe.length > 0 && (
              <div className="grid grid-cols-4 gap-4 mt-4 pt-4 border-t border-terminal-border">
                <div className="text-center">
                  <div className="text-xs text-terminal-muted uppercase">Current</div>
                  <div className="font-mono font-bold text-lg">{rollingSharpe[rollingSharpe.length - 1]?.value.toFixed(2) ?? 'N/A'}</div>
                </div>
                <div className="text-center">
                  <div className="text-xs text-terminal-muted uppercase">Average</div>
                  <div className="font-mono font-bold text-lg">
                    {(rollingSharpe.reduce((a, b) => a + b.value, 0) / rollingSharpe.length).toFixed(2)}
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-xs text-terminal-muted uppercase">Max</div>
                  <div className="font-mono font-bold text-lg text-profit">
                    {Math.max(...rollingSharpe.map(r => r.value)).toFixed(2)}
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-xs text-terminal-muted uppercase">Min</div>
                  <div className="font-mono font-bold text-lg text-loss">
                    {Math.min(...rollingSharpe.map(r => r.value)).toFixed(2)}
                  </div>
                </div>
              </div>
            )}
          </div>
          
          {/* Rolling Returns - Secondary */}
          <div className="card-elevated">
            <h2 className="font-semibold text-lg mb-4">Rolling 21-Day Returns</h2>
            <div className="h-[250px]">
              {rollingReturns.length > 0 ? (
                <RollingMetrics
                  data={[{ label: '21-Day Return', points: rollingReturns, color: '#60a5fa' }]}
                  showZeroLine
                />
              ) : (
                <div className="flex items-center justify-center h-full text-terminal-muted">
                  Not enough data for rolling analysis
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Strategy Pipeline Tab */}
      {activeTab === 'pipeline' && (
        <div className="card-elevated">
          <StrategyPipeline candidateId={selectedCandidate.candidate_id} />
        </div>
      )}

      {/* Walk-Forward Analysis Tab */}
      {activeTab === 'validation' && (
        <div className="card-elevated">
          <WFAAnalysis candidateId={selectedCandidate.candidate_id} />
        </div>
      )}

      {/* Stress Testing Tab */}
      {activeTab === 'stress' && (
        <div className="card-elevated">
          <StressAnalysis candidateId={selectedCandidate.candidate_id} />
        </div>
      )}

      {/* Risk Decomposition Tab */}
      {activeTab === 'risk' && (
        <div className="card-elevated">
          <RiskDecomposition 
            dailyReturns={dailyReturns} 
            maxDrawdown={computedMaxDD} 
            sharpe={stats.sharpe}
          />
        </div>
      )}

      {/* Error display */}
      {error && (
        <div className="p-4 bg-loss/10 border border-loss/30 rounded-lg text-loss">
          {error}
        </div>
      )}
    </div>
  );
}

// =============================================================================
// HELPER COMPONENTS
// =============================================================================

interface ScoreCardProps {
  label: string;
  value: string;
  icon: React.ReactNode;
  color: 'profit' | 'loss' | 'warning';
  subtitle?: string;
  progress?: number;
}

function ScoreCard({ label, value, icon, color, subtitle, progress }: ScoreCardProps) {
  const colorClasses = {
    profit: 'text-profit',
    loss: 'text-loss',
    warning: 'text-accent-yellow'
  };

  const bgClasses = {
    profit: 'bg-profit/10',
    loss: 'bg-loss/10',
    warning: 'bg-accent-yellow/10'
  };

  return (
    <div className="card-elevated p-5">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm text-terminal-muted">{label}</span>
        <div className={`p-2 rounded-lg ${bgClasses[color]}`}>
          <span className={colorClasses[color]}>{icon}</span>
        </div>
      </div>
      <div className={`text-3xl font-bold font-mono ${colorClasses[color]}`}>
        {value}
      </div>
      {subtitle && (
        <div className="text-xs text-terminal-muted mt-1">{subtitle}</div>
      )}
      {progress !== undefined && (
        <div className="mt-3 h-1.5 bg-terminal-bg rounded-full overflow-hidden">
          <div 
            className={`h-full rounded-full ${color === 'profit' ? 'bg-profit' : color === 'loss' ? 'bg-loss' : 'bg-accent-yellow'}`}
            style={{ width: `${Math.min(progress, 100)}%` }}
          />
        </div>
      )}
    </div>
  );
}

function MetricRow({ label, value, color }: { label: string; value: string; color?: 'profit' | 'loss' }) {
  return (
    <div className="flex items-center justify-between py-0.5">
      <span className="text-xs text-terminal-muted">{label}</span>
      <span className={`font-mono text-sm font-medium ${color === 'profit' ? 'text-profit' : color === 'loss' ? 'text-loss' : ''}`}>
        {value}
      </span>
    </div>
  );
}

type Quality = 'excellent' | 'good' | 'fair' | 'poor';

function MetricRowWithTooltip({ 
  label, 
  value, 
  tooltip, 
  quality,
  color 
}: { 
  label: string; 
  value: string; 
  tooltip: string; 
  quality?: Quality;
  color?: 'profit' | 'loss';
}) {
  const qualityColors: Record<Quality, string> = {
    excellent: 'text-profit',
    good: 'text-profit/80',
    fair: 'text-accent-yellow',
    poor: 'text-loss'
  };
  
  const qualityIndicators: Record<Quality, string> = {
    excellent: '●●●',
    good: '●●○',
    fair: '●○○',
    poor: '○○○'
  };

  const colorClass = color ? (color === 'profit' ? 'text-profit' : 'text-loss') : (quality ? qualityColors[quality] : '');
  
  return (
    <div className="flex items-center justify-between py-0.5 group">
      <div className="flex items-center gap-1">
        <span className="text-xs text-terminal-muted">{label}</span>
        <div className="relative">
          <Info className="w-3 h-3 text-terminal-muted/50 cursor-help" />
          <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-terminal-bg border border-terminal-border rounded text-[10px] whitespace-nowrap opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10">
            {tooltip}
          </div>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <span className={`font-mono text-sm font-medium ${colorClass}`}>
          {value}
        </span>
        {quality && (
          <span className={`text-[8px] font-mono ${qualityColors[quality]}`}>
            {qualityIndicators[quality]}
          </span>
        )}
      </div>
    </div>
  );
}

/** 
 * MetricRowBloomberg - Enhanced metric row with rich tooltips
 * Supports both string tooltips and MetricTooltips objects
 */
interface MetricRowBloombergProps {
  label: string;
  value: string;
  color?: 'profit' | 'loss';
  quality?: Quality;
  tooltip?: string | ReturnType<typeof MetricTooltips.sharpe>;
}

function MetricRowBloomberg({ label, value, color, quality, tooltip }: MetricRowBloombergProps) {
  const qualityColors: Record<Quality, string> = {
    excellent: 'text-profit',
    good: 'text-profit/80',
    fair: 'text-accent-yellow',
    poor: 'text-loss'
  };
  
  const qualityIndicators: Record<Quality, string> = {
    excellent: '●●●',
    good: '●●○',
    fair: '●○○',
    poor: '○○○'
  };

  const colorClass = color 
    ? (color === 'profit' ? 'text-profit' : 'text-loss') 
    : (quality ? qualityColors[quality] : '');

  // Check if tooltip is a MetricTooltips object or string
  const isRichTooltip = tooltip && typeof tooltip === 'object' && 'title' in tooltip;
  
  return (
    <div className="flex items-center justify-between py-0.5 group">
      <div className="flex items-center gap-1">
        <span className="text-xs text-terminal-muted">{label}</span>
        {tooltip && (
          isRichTooltip ? (
            <BloombergTooltip {...tooltip} position="right">
              <Info className="w-3 h-3 text-terminal-muted/50 cursor-help hover:text-accent-cyan transition-colors" />
            </BloombergTooltip>
          ) : (
            <div className="relative">
              <Info className="w-3 h-3 text-terminal-muted/50 cursor-help" />
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-terminal-bg border border-terminal-border rounded text-[10px] max-w-[200px] text-center opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10">
                {tooltip as string}
              </div>
            </div>
          )
        )}
      </div>
      <div className="flex items-center gap-2">
        <span className={`font-mono text-sm font-medium ${colorClass}`}>
          {value}
        </span>
        {quality && (
          <span className={`text-[8px] font-mono ${qualityColors[quality]}`}>
            {qualityIndicators[quality]}
          </span>
        )}
      </div>
    </div>
  );
}

function ValidationBadge({ label, passed, value }: { label: string; passed: boolean; value?: string }) {
  return (
    <div className={`flex items-center justify-between p-3 rounded-lg border ${
      passed 
        ? 'bg-profit/5 border-profit/30' 
        : 'bg-loss/5 border-loss/30'
    }`}>
      <div className="flex items-center gap-2">
        {passed ? (
          <CheckCircle className="w-4 h-4 text-profit" />
        ) : (
          <XCircle className="w-4 h-4 text-loss" />
        )}
        <span className="text-xs font-medium">{label}</span>
      </div>
      {value && (
        <span className={`font-mono text-xs ${passed ? 'text-profit' : 'text-loss'}`}>
          {value}
        </span>
      )}
    </div>
  );
}

function ValidationRow({ label, passed, value }: { label: string; passed: boolean; value?: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-sm text-terminal-muted">{label}</span>
      <div className="flex items-center gap-2">
        {value && <span className="font-mono text-sm">{value}</span>}
        {passed ? (
          <CheckCircle className="w-4 h-4 text-profit" />
        ) : (
          <XCircle className="w-4 h-4 text-loss" />
        )}
      </div>
    </div>
  );
}

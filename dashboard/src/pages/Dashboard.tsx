import { useEffect, useState } from 'react';
import { MetricCard } from '../components/ui/MetricCard';
import { EquityChart } from '../components/charts/EquityChart';
import { GenerationChart } from '../components/charts/GenerationChart';
import { QuickTooltip } from '../components/ui/TooltipInfo';
import { 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  Target,
  Zap,
  Award,
  RefreshCw,
  Database,
  BarChart3,
  Clock
} from 'lucide-react';

interface OverviewData {
  metrics: {
    totalReturn: number;
    sharpeRatio: number;
    avgSharpeRatio: number;
    maxDrawdown: number;
    winRate: number;
    totalTrades: number;
    activeCandidates: number;
    totalCandidates: number;
    candidates24h: number;
    currentGeneration: number;
    bestCagr: number;
  };
  campaigns: {
    total: number;
    completed: number;
    running: number;
    failed: number;
  };
  equityData: { time: string; value: number }[];
  generationData: { generation: number; bestSharpe: number; meanSharpe: number; paretoSize: number }[];
  ompStatus: string;
  lastUpdated: string;
}

export function Dashboard() {
  const [data, setData] = useState<OverviewData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

  const fetchOverview = async () => {
    try {
      const response = await fetch('/api/overview');
      if (!response.ok) {
        throw new Error(`Failed to fetch overview: ${response.status}`);
      }
      const result = await response.json();
      setData(result);
      setError(null);
      setLastRefresh(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchOverview();
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchOverview, 30000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <RefreshCw className="w-8 h-8 animate-spin text-accent-cyan mx-auto mb-4" />
          <p className="text-terminal-muted">Loading overview data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <Database className="w-8 h-8 text-loss mx-auto mb-4" />
          <p className="text-loss mb-2">Failed to load overview</p>
          <p className="text-terminal-muted text-sm">{error}</p>
          <button 
            onClick={fetchOverview}
            className="mt-4 px-4 py-2 bg-accent-cyan/20 text-accent-cyan rounded hover:bg-accent-cyan/30 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!data) {
    return null;
  }

  const metrics = data.metrics;
  const campaigns = data.campaigns;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Dashboard</h1>
          <p className="text-terminal-muted mt-1">Overview of system performance and evolution status</p>
        </div>
        <div className="flex items-center gap-4">
          {/* Last updated */}
          <div className="flex items-center gap-2 text-terminal-muted text-sm">
            <Clock className="w-4 h-4" />
            <span>Updated: {lastRefresh.toLocaleTimeString()}</span>
          </div>
          {/* OMP Status */}
          <div className={`flex items-center gap-2 px-4 py-2 rounded-lg ${
            data.ompStatus === 'running' 
              ? 'bg-profit/10 border border-profit/30' 
              : data.ompStatus === 'paused'
              ? 'bg-yellow-500/10 border border-yellow-500/30'
              : 'bg-terminal-muted/10 border border-terminal-muted/30'
          }`}>
            <Zap className={`w-5 h-5 ${
              data.ompStatus === 'running' ? 'text-profit' 
              : data.ompStatus === 'paused' ? 'text-yellow-500'
              : 'text-terminal-muted'
            }`} />
            <span className={`font-mono ${
              data.ompStatus === 'running' ? 'text-profit' 
              : data.ompStatus === 'paused' ? 'text-yellow-500'
              : 'text-terminal-muted'
            }`}>
              OMP {data.ompStatus.toUpperCase()}
            </span>
          </div>
          {/* Refresh button */}
          <button
            onClick={fetchOverview}
            className="p-2 rounded hover:bg-terminal-border/50 transition-colors"
            title="Refresh data"
          >
            <RefreshCw className="w-5 h-5 text-terminal-muted hover:text-terminal-text" />
          </button>
        </div>
      </div>

      {/* KPI Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label={
            <span className="flex items-center gap-1">
              Best CAGR
              <QuickTooltip term="oos_cagr" />
            </span>
          }
          value={metrics.bestCagr}
          format="percent"
          icon={<TrendingUp className="w-5 h-5" />}
          size="lg"
        />
        <MetricCard
          label={
            <span className="flex items-center gap-1">
              Best Sharpe
              <QuickTooltip term="oos_sharpe" />
            </span>
          }
          value={metrics.sharpeRatio}
          format="ratio"
          subtitle={`Avg: ${metrics.avgSharpeRatio.toFixed(2)}`}
          icon={<Target className="w-5 h-5" />}
          size="lg"
        />
        <MetricCard
          label={
            <span className="flex items-center gap-1">
              Max Drawdown
              <QuickTooltip term="max_drawdown" />
            </span>
          }
          value={metrics.maxDrawdown}
          format="percent"
          icon={<TrendingDown className="w-5 h-5" />}
          size="lg"
        />
        <MetricCard
          label="Candidates Today"
          value={metrics.candidates24h}
          format="number"
          subtitle={`Total: ${metrics.totalCandidates.toLocaleString()}`}
          icon={<Activity className="w-5 h-5" />}
          size="lg"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Equity Curve */}
        <div className="card-elevated">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-semibold text-lg flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-accent-cyan" />
              Equity Evolution
            </h2>
            <span className="text-xs text-terminal-muted font-mono">
              SIMULATED FROM VALIDATED CANDIDATES
            </span>
          </div>
          {data.equityData.length > 0 ? (
            <div className="h-[300px]">
              <EquityChart data={data.equityData} />
            </div>
          ) : (
            <div className="h-[300px] flex items-center justify-center text-terminal-muted">
              No equity data available yet
            </div>
          )}
        </div>

        {/* Evolution Progress */}
        <div className="card-elevated">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-semibold text-lg flex items-center gap-2">
              <Activity className="w-5 h-5 text-accent-purple" />
              Evolution Progress
            </h2>
            <span className="text-xs text-terminal-muted font-mono">
              {data.generationData.length} RUNS ANALYZED
            </span>
          </div>
          {data.generationData.length > 0 ? (
            <div className="h-[300px]">
              <GenerationChart data={data.generationData} />
            </div>
          ) : (
            <div className="h-[300px] flex items-center justify-center text-terminal-muted">
              No generation data available yet
            </div>
          )}
        </div>
      </div>

      {/* Bottom Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card flex items-center gap-4">
          <div className="w-12 h-12 rounded-lg bg-accent-cyan/10 flex items-center justify-center">
            <Database className="w-6 h-6 text-accent-cyan" />
          </div>
          <div>
            <div className="metric-label">Total Candidates</div>
            <div className="font-mono text-xl font-bold">{metrics.totalCandidates.toLocaleString()}</div>
          </div>
        </div>
        
        <div className="card flex items-center gap-4">
          <div className="w-12 h-12 rounded-lg bg-accent-purple/10 flex items-center justify-center">
            <Award className="w-6 h-6 text-accent-purple" />
          </div>
          <div>
            <div className="metric-label">Hall of Fame</div>
            <div className="font-mono text-xl font-bold">{metrics.activeCandidates} strategies</div>
          </div>
        </div>
        
        <div className="card flex items-center gap-4">
          <div className="w-12 h-12 rounded-lg bg-profit/10 flex items-center justify-center">
            <BarChart3 className="w-6 h-6 text-profit" />
          </div>
          <div>
            <div className="metric-label">Campaigns</div>
            <div className="font-mono text-xl font-bold">
              {campaigns.completed} / {campaigns.total}
              <span className="text-sm text-terminal-muted ml-1">done</span>
            </div>
          </div>
        </div>

        <div className="card flex items-center gap-4">
          <div className="w-12 h-12 rounded-lg bg-yellow-500/10 flex items-center justify-center">
            <Activity className="w-6 h-6 text-yellow-500" />
          </div>
          <div>
            <div className="metric-label">Running</div>
            <div className="font-mono text-xl font-bold">
              {campaigns.running}
              {campaigns.failed > 0 && (
                <span className="text-sm text-loss ml-2">({campaigns.failed} failed)</span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Quick Stats Summary */}
      {data.generationData.length > 0 && (
        <div className="card-elevated p-4">
          <h3 className="text-sm font-medium text-terminal-muted mb-3">Recent Performance Summary</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-terminal-muted">Runs Analyzed:</span>
              <span className="ml-2 font-mono">{data.generationData.length}</span>
            </div>
            <div>
              <span className="text-terminal-muted">Best Sharpe (recent):</span>
              <span className="ml-2 font-mono text-profit">
                {Math.max(...data.generationData.map(g => g.bestSharpe)).toFixed(3)}
              </span>
            </div>
            <div>
              <span className="text-terminal-muted">Mean Sharpe (trend):</span>
              <span className="ml-2 font-mono">
                {(data.generationData.reduce((sum, g) => sum + g.meanSharpe, 0) / data.generationData.length).toFixed(3)}
              </span>
            </div>
            <div>
              <span className="text-terminal-muted">Avg Pareto Size:</span>
              <span className="ml-2 font-mono">
                {Math.round(data.generationData.reduce((sum, g) => sum + g.paretoSize, 0) / data.generationData.length)}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

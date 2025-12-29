import { useState, useEffect, useMemo } from 'react';
import { MetricCard } from '../components/ui/MetricCard';
import { EquityChart } from '../components/charts/EquityChart';
import { DrawdownChart } from '../components/charts/DrawdownChart';
import { ReturnDistribution } from '../components/charts/ReturnDistribution';
import { MonthlyHeatmap } from '../components/charts/MonthlyHeatmap';
import { RollingMetrics } from '../components/charts/RollingMetrics';
import { useDataStore } from '../stores/dataStore';
import type { MonthlyReturn, RollingPoint } from '../stores/dataStore';
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
  FolderOpen,
  RefreshCw,
  ExternalLink,
  PieChart,
  Layers,
  CalendarDays,
  LineChart
} from 'lucide-react';
import { open } from '@tauri-apps/plugin-shell';

export function Backtest() {
  const [activeTab, setActiveTab] = useState<'overview' | 'distribution' | 'monthly' | 'rolling' | 'trades'>('overview');
  
  const {
    selectedCandidate,
    backtest,
    riskMetrics,
    isLoading,
    error,
    loadBacktest,
    loadRiskMetrics,
    artifactsRoot
  } = useDataStore();

  // Load backtest when candidate is selected
  useEffect(() => {
    if (selectedCandidate) {
      loadBacktest(selectedCandidate.candidate_id);
      loadRiskMetrics(selectedCandidate.candidate_id);
    }
  }, [selectedCandidate?.candidate_id]);

  // Convert timeseries to equity chart format
  const equityData = backtest?.timeseries.map(t => ({
    time: t.date,
    value: t.equity,
  })) ?? [];

  const drawdownData = backtest?.timeseries.map(t => ({
    time: t.date,
    value: t.drawdown,
  })) ?? [];

  // Calculate returns from timeseries
  const dailyReturns = useMemo(() => {
    if (!backtest?.timeseries || backtest.timeseries.length < 2) return [];
    return backtest.timeseries.slice(1).map((t, i) => {
      const prev = backtest.timeseries[i];
      return (t.equity - prev.equity) / prev.equity;
    });
  }, [backtest?.timeseries]);

  // Calculate monthly returns
  const monthlyReturns = useMemo((): MonthlyReturn[] => {
    if (!backtest?.timeseries || backtest.timeseries.length < 2) return [];
    
    const monthly: Record<string, { equity_start: number; equity_end: number; year: number; month: number }> = {};
    
    for (const point of backtest.timeseries) {
      const date = point.date;
      if (date.length >= 7) {
        const yearMonth = date.substring(0, 7);
        const [year, month] = yearMonth.split('-').map(Number);
        
        if (!monthly[yearMonth]) {
          monthly[yearMonth] = { equity_start: point.equity, equity_end: point.equity, year, month };
        }
        monthly[yearMonth].equity_end = point.equity;
      }
    }
    
    // Calculate previous month's end for proper return calculation
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
  }, [backtest?.timeseries]);

  // Calculate rolling returns (21-day)
  const rollingReturns = useMemo((): RollingPoint[] => {
    if (!backtest?.timeseries || backtest.timeseries.length < 22) return [];
    
    const window = 21;
    return backtest.timeseries.slice(window).map((point, i) => {
      const startIdx = i;
      const startEquity = backtest.timeseries[startIdx].equity;
      const ret = (point.equity - startEquity) / startEquity;
      return { date: point.date, value: ret * 100 };
    });
  }, [backtest?.timeseries]);

  const handleRunReplay = async () => {
    if (selectedCandidate?.replay_script_path) {
      try {
        await open(selectedCandidate.replay_script_path);
      } catch (e) {
        console.error('Failed to run replay:', e);
      }
    }
  };

  const handleOpenBacktestFolder = async () => {
    if (backtest?.backtest_path) {
      try {
        await open(backtest.backtest_path);
      } catch (e) {
        console.error('Failed to open folder:', e);
      }
    }
  };

  // No artifacts root
  if (!artifactsRoot) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-6">
        <FolderOpen className="w-16 h-16 text-terminal-muted" />
        <div className="text-center">
          <h2 className="text-xl font-semibold mb-2">No Project Selected</h2>
          <p className="text-terminal-muted">
            Select a project folder from the Candidates page to view backtest results.
          </p>
        </div>
      </div>
    );
  }

  // No candidate selected
  if (!selectedCandidate) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-6">
        <BarChart3 className="w-16 h-16 text-terminal-muted" />
        <div className="text-center">
          <h2 className="text-xl font-semibold mb-2">No Candidate Selected</h2>
          <p className="text-terminal-muted">
            Select a candidate from the Candidates page to view backtest results.
          </p>
        </div>
      </div>
    );
  }

  // Loading state
  if (isLoading && !backtest) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-4">
        <RefreshCw className="w-12 h-12 animate-spin text-terminal-muted" />
        <p className="text-terminal-muted">Loading backtest data...</p>
      </div>
    );
  }

  // Backtest not available
  if (backtest && !backtest.available) {
    return (
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Backtest Drilldown</h1>
            <p className="text-terminal-muted mt-1">
              Candidate: <span className="font-mono text-accent-cyan">{selectedCandidate.candidate_id}</span>
            </p>
          </div>
        </div>

        {/* No Data Message */}
        <div className="flex flex-col items-center justify-center py-16 space-y-6">
          <div className="p-6 bg-accent-yellow/10 border border-accent-yellow/30 rounded-lg max-w-lg">
            <div className="flex items-start gap-4">
              <AlertTriangle className="w-8 h-8 text-accent-yellow flex-shrink-0" />
              <div>
                <h3 className="font-semibold text-lg mb-2">Backtest Data Not Available</h3>
                <p className="text-terminal-muted text-sm mb-4">
                  {backtest.message}
                </p>
                {selectedCandidate.replay_script_path && (
                  <button
                    onClick={handleRunReplay}
                    className="flex items-center gap-2 px-4 py-2 bg-profit text-black font-medium rounded-lg hover:bg-profit/90 transition-colors"
                  >
                    <Play className="w-4 h-4" />
                    Generate via Replay
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Show candidate metrics as fallback */}
          <div className="w-full max-w-2xl">
            <h3 className="font-semibold mb-4">Validation Metrics (from SCG)</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <MetricCard
                label="OOS Sharpe NET"
                value={selectedCandidate.oos_sharpe_net}
                format="ratio"
              />
              <MetricCard
                label="PBO"
                value={selectedCandidate.pbo * 100}
                format="percent"
              />
              <MetricCard
                label="DSR"
                value={selectedCandidate.dsr ?? 0}
                format="ratio"
              />
              <MetricCard
                label="Stress Tests"
                value={`${selectedCandidate.stress_passed}/${selectedCandidate.stress_total}`}
              />
            </div>
          </div>
        </div>
      </div>
    );
  }

  const metrics = backtest?.metrics;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Backtest Drilldown</h1>
          <p className="text-terminal-muted mt-1">Detailed analysis of strategy performance</p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-terminal-surface border border-terminal-border">
            <FileText className="w-4 h-4 text-terminal-muted" />
            <span className="font-mono text-sm">{selectedCandidate.candidate_id.substring(0, 16)}...</span>
          </div>
          {backtest?.backtest_path && (
            <button 
              onClick={handleOpenBacktestFolder}
              className="flex items-center gap-2 px-3 py-2 rounded-lg bg-terminal-surface border border-terminal-border hover:border-profit transition-colors"
            >
              <ExternalLink className="w-4 h-4" />
              Open Folder
            </button>
          )}
          {selectedCandidate.replay_script_path && (
            <button 
              onClick={handleRunReplay}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-profit/10 text-profit border border-profit/30 hover:bg-profit/20 transition-all"
            >
              <Play className="w-4 h-4" />
              Re-run
            </button>
          )}
        </div>
      </div>

      {/* Strategy Info */}
      <div className="card flex items-center justify-between">
        <div className="flex items-center gap-6">
          <div>
            <div className="text-sm text-terminal-muted">Strategy</div>
            <div className="font-semibold text-lg">{selectedCandidate.display_name}</div>
          </div>
          <div className="h-8 w-px bg-terminal-border" />
          <div className="flex items-center gap-2">
            <Calendar className="w-4 h-4 text-terminal-muted" />
            <span className="font-mono text-sm">
              {equityData.length > 0 
                ? `${equityData[0].time} to ${equityData[equityData.length - 1].time}`
                : 'N/A'}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="text-right">
            <div className="text-sm text-terminal-muted">Final NAV</div>
            <div className="font-mono font-bold text-xl text-profit">
              ${equityData.length > 0 
                ? equityData[equityData.length - 1].value.toLocaleString(undefined, { maximumFractionDigits: 0 })
                : 'N/A'}
            </div>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      {metrics && (
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
          <MetricCard
            label="CAGR"
            value={metrics.cagr * 100}
            format="percent"
            icon={<TrendingUp className="w-4 h-4 text-profit" />}
          />
          <MetricCard
            label="Sharpe Ratio"
            value={metrics.sharpe_ratio}
            format="ratio"
            icon={<Target className="w-4 h-4" />}
          />
          <MetricCard
            label="Sortino Ratio"
            value={metrics.sortino_ratio ?? 0}
            format="ratio"
          />
          <MetricCard
            label="Calmar Ratio"
            value={metrics.calmar_ratio ?? 0}
            format="ratio"
          />
          <MetricCard
            label="Max Drawdown"
            value={metrics.max_drawdown * 100}
            format="percent"
            icon={<TrendingDown className="w-4 h-4 text-loss" />}
          />
          <MetricCard
            label="Win Rate"
            value={(metrics.hit_rate ?? 0) * 100}
            format="percent"
            icon={<Activity className="w-4 h-4" />}
          />
        </div>
      )}

      {/* Extended Metrics */}
      {riskMetrics && (
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
          <MetricCard
            label="Omega Ratio"
            value={riskMetrics.omega_ratio}
            format="ratio"
          />
          <MetricCard
            label="Tail Ratio"
            value={riskMetrics.tail_ratio}
            format="ratio"
          />
          <MetricCard
            label="Gain/Pain"
            value={riskMetrics.gain_to_pain}
            format="ratio"
          />
          <MetricCard
            label="Best Day"
            value={riskMetrics.best_day * 100}
            format="percent"
          />
          <MetricCard
            label="Worst Day"
            value={riskMetrics.worst_day * 100}
            format="percent"
          />
          <MetricCard
            label="Time Underwater"
            value={riskMetrics.time_underwater_pct}
            format="percent"
          />
        </div>
      )}

      {/* Error display */}
      {error && (
        <div className="p-4 bg-loss/10 border border-loss/30 rounded-lg text-loss">
          {error}
        </div>
      )}

      {/* Tabs */}
      <div className="flex items-center gap-4 border-b border-terminal-border overflow-x-auto">
        {[
          { key: 'overview', label: 'Overview', icon: BarChart3 },
          { key: 'distribution', label: 'Distribution', icon: PieChart },
          { key: 'monthly', label: 'Monthly', icon: CalendarDays },
          { key: 'rolling', label: 'Rolling', icon: LineChart },
          { key: 'trades', label: `Trades (${metrics?.total_trades ?? 0})`, icon: Layers },
        ].map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key as typeof activeTab)}
            className={`flex items-center gap-2 pb-3 px-1 font-medium transition-colors relative whitespace-nowrap ${
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
      {activeTab === 'overview' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Equity Curve */}
          <div className="card-elevated">
            <div className="flex items-center justify-between mb-4">
              <h2 className="font-semibold text-lg">Equity Curve</h2>
              <BarChart3 className="w-5 h-5 text-terminal-muted" />
            </div>
            <div className="h-[300px]">
              {equityData.length > 0 ? (
                <EquityChart data={equityData} />
              ) : (
                <div className="flex items-center justify-center h-full text-terminal-muted">
                  No data available
                </div>
              )}
            </div>
          </div>

          {/* Drawdown */}
          <div className="card-elevated">
            <div className="flex items-center justify-between mb-4">
              <h2 className="font-semibold text-lg">Underwater Chart</h2>
              <span className="text-xs font-mono text-loss">
                Max: {metrics ? (metrics.max_drawdown * 100).toFixed(2) : 0}%
              </span>
            </div>
            <div className="h-[300px]">
              {drawdownData.length > 0 ? (
                <DrawdownChart data={drawdownData} />
              ) : (
                <div className="flex items-center justify-center h-full text-terminal-muted">
                  No data available
                </div>
              )}
            </div>
          </div>

          {/* Trade Stats */}
          {metrics && (
            <div className="card-elevated lg:col-span-2">
              <h2 className="font-semibold text-lg mb-4">Trade Statistics</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                <div>
                  <div className="metric-label">Total Trades</div>
                  <div className="font-mono text-2xl font-bold">{metrics.total_trades}</div>
                </div>
                <div>
                  <div className="metric-label">Profit Factor</div>
                  <div className="font-mono text-2xl font-bold">{(metrics.profit_factor ?? 0).toFixed(2)}</div>
                </div>
                <div>
                  <div className="metric-label">Trading Days</div>
                  <div className="font-mono text-2xl font-bold">{metrics.total_days ?? 'N/A'}</div>
                </div>
                <div>
                  <div className="metric-label">Annual Turnover</div>
                  <div className="font-mono text-2xl font-bold">
                    {metrics.turnover_annual?.toFixed(2) ?? 'N/A'}
                  </div>
                </div>
              </div>
            </div>
          )}
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
        <div className="card-elevated">
          <h2 className="font-semibold text-lg mb-4">Rolling 21-Day Returns</h2>
          <div className="h-[400px]">
            {rollingReturns.length > 0 ? (
              <RollingMetrics
                data={[
                  { label: '21-Day Return', points: rollingReturns, color: '#00ff88' },
                ]}
                showZeroLine
              />
            ) : (
              <div className="flex items-center justify-center h-full text-terminal-muted">
                Not enough data for rolling analysis
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === 'trades' && (
        <div className="card-elevated">
          <div className="flex flex-col items-center justify-center py-12 text-terminal-muted">
            <AlertTriangle className="w-12 h-12 mb-4 opacity-50" />
            <h3 className="font-semibold mb-2">Trade Log Not Available</h3>
            <p className="text-sm text-center max-w-md">
              Individual trade data is not available in the current artifact format.
              Run the replay with verbose logging enabled to generate detailed trade records.
            </p>
            {selectedCandidate.replay_script_path && (
              <button
                onClick={handleRunReplay}
                className="mt-4 flex items-center gap-2 px-4 py-2 bg-profit/10 text-profit border border-profit/30 rounded-lg hover:bg-profit/20 transition-colors"
              >
                <Play className="w-4 h-4" />
                Generate Trade Log via Replay
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

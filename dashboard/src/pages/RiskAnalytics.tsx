import { useState, useEffect } from 'react';
import { MetricCard } from '../components/ui/MetricCard';
import { ReturnDistribution } from '../components/charts/ReturnDistribution';
import { MonthlyHeatmap } from '../components/charts/MonthlyHeatmap';
import { RollingMetrics } from '../components/charts/RollingMetrics';
import { VaRGauge } from '../components/charts/VaRGauge';
import { useDataStore } from '../stores/dataStore';
import {
  Shield,
  TrendingDown,
  Activity,
  BarChart3,
  RefreshCw,
  AlertTriangle,
  Target,
  Zap,
  Calendar,
  ChevronDown,
} from 'lucide-react';

export function RiskAnalytics() {
  const [activeTab, setActiveTab] = useState<'var' | 'distribution' | 'rolling' | 'monthly'>('var');

  const {
    selectedCandidate,
    riskMetrics,
    isLoading,
    error,
    loadRiskMetrics,
    artifactsRoot,
  } = useDataStore();

  // Load risk metrics when candidate is selected
  useEffect(() => {
    if (selectedCandidate) {
      loadRiskMetrics(selectedCandidate.candidate_id);
    }
  }, [selectedCandidate?.candidate_id]);

  // No artifacts root
  if (!artifactsRoot) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-6">
        <Shield className="w-16 h-16 text-terminal-muted" />
        <div className="text-center">
          <h2 className="text-xl font-semibold mb-2">No Project Selected</h2>
          <p className="text-terminal-muted">
            Select a project folder from the Candidates page first.
          </p>
        </div>
      </div>
    );
  }

  // No candidate selected
  if (!selectedCandidate) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-6">
        <Shield className="w-16 h-16 text-terminal-muted" />
        <div className="text-center">
          <h2 className="text-xl font-semibold mb-2">No Candidate Selected</h2>
          <p className="text-terminal-muted">
            Select a candidate from the Candidates page to view risk analytics.
          </p>
        </div>
      </div>
    );
  }

  // Loading state
  if (isLoading && !riskMetrics) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-4">
        <RefreshCw className="w-12 h-12 animate-spin text-terminal-muted" />
        <p className="text-terminal-muted">Calculating risk metrics...</p>
      </div>
    );
  }

  if (!riskMetrics) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-6">
        <AlertTriangle className="w-16 h-16 text-accent-yellow" />
        <div className="text-center">
          <h2 className="text-xl font-semibold mb-2">Risk Data Not Available</h2>
          <p className="text-terminal-muted max-w-md">
            {error || 'Unable to calculate risk metrics for this candidate.'}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Risk Analytics</h1>
          <p className="text-terminal-muted mt-1">
            Institutional-grade risk analysis for{' '}
            <span className="text-accent-cyan font-mono">
              {selectedCandidate.display_name}
            </span>
          </p>
        </div>
        <button
          onClick={() => loadRiskMetrics(selectedCandidate.candidate_id)}
          className="flex items-center gap-2 px-3 py-2 rounded-lg bg-terminal-surface border border-terminal-border hover:border-profit transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Key Risk Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
        <MetricCard
          label="VaR 95%"
          value={Math.abs(riskMetrics.var_95 * 100)}
          format="percent"
          icon={<TrendingDown className="w-4 h-4 text-loss" />}
        />
        <MetricCard
          label="CVaR 95%"
          value={Math.abs(riskMetrics.cvar_95 * 100)}
          format="percent"
          icon={<TrendingDown className="w-4 h-4 text-loss" />}
        />
        <MetricCard
          label="Sortino Ratio"
          value={riskMetrics.sortino_ratio}
          format="ratio"
          icon={<Target className="w-4 h-4" />}
        />
        <MetricCard
          label="Calmar Ratio"
          value={riskMetrics.calmar_ratio}
          format="ratio"
          icon={<Zap className="w-4 h-4" />}
        />
        <MetricCard
          label="Omega Ratio"
          value={riskMetrics.omega_ratio}
          format="ratio"
          icon={<Activity className="w-4 h-4" />}
        />
        <MetricCard
          label="Stability"
          value={riskMetrics.stability_of_timeseries * 100}
          format="percent"
          icon={<BarChart3 className="w-4 h-4" />}
        />
      </div>

      {/* Tail Risk Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="card">
          <div className="metric-label">Skewness</div>
          <div className={`font-mono text-xl ${riskMetrics.skewness < 0 ? 'text-loss' : 'text-profit'}`}>
            {riskMetrics.skewness.toFixed(3)}
          </div>
          <div className="text-xs text-terminal-muted mt-1">
            {riskMetrics.skewness < -0.5 ? 'Negative tail risk' : riskMetrics.skewness > 0.5 ? 'Positive skew' : 'Near symmetric'}
          </div>
        </div>
        <div className="card">
          <div className="metric-label">Excess Kurtosis</div>
          <div className={`font-mono text-xl ${riskMetrics.kurtosis > 3 ? 'text-accent-yellow' : ''}`}>
            {riskMetrics.kurtosis.toFixed(3)}
          </div>
          <div className="text-xs text-terminal-muted mt-1">
            {riskMetrics.kurtosis > 3 ? 'Fat tails (leptokurtic)' : 'Normal tails'}
          </div>
        </div>
        <div className="card">
          <div className="metric-label">Tail Ratio</div>
          <div className="font-mono text-xl">{riskMetrics.tail_ratio.toFixed(2)}</div>
          <div className="text-xs text-terminal-muted mt-1">P95 gain / P5 loss</div>
        </div>
        <div className="card">
          <div className="metric-label">Gain-to-Pain</div>
          <div className={`font-mono text-xl ${riskMetrics.gain_to_pain > 1 ? 'text-profit' : 'text-loss'}`}>
            {riskMetrics.gain_to_pain.toFixed(2)}
          </div>
          <div className="text-xs text-terminal-muted mt-1">Total return / total pain</div>
        </div>
      </div>

      {/* Best/Worst Analysis */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="card">
          <div className="metric-label">Best Day</div>
          <div className="font-mono text-xl text-profit">+{(riskMetrics.best_day * 100).toFixed(2)}%</div>
        </div>
        <div className="card">
          <div className="metric-label">Worst Day</div>
          <div className="font-mono text-xl text-loss">{(riskMetrics.worst_day * 100).toFixed(2)}%</div>
        </div>
        <div className="card">
          <div className="metric-label">Best Month</div>
          <div className="font-mono text-xl text-profit">+{riskMetrics.best_month.toFixed(2)}%</div>
        </div>
        <div className="card">
          <div className="metric-label">Worst Month</div>
          <div className="font-mono text-xl text-loss">{riskMetrics.worst_month.toFixed(2)}%</div>
        </div>
      </div>

      {/* Drawdown Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="card">
          <div className="metric-label">Longest Drawdown</div>
          <div className="font-mono text-xl">{riskMetrics.longest_dd_days} days</div>
        </div>
        <div className="card">
          <div className="metric-label">Avg Drawdown Duration</div>
          <div className="font-mono text-xl">{riskMetrics.average_dd_days.toFixed(1)} days</div>
        </div>
        <div className="card">
          <div className="metric-label">Time Underwater</div>
          <div className={`font-mono text-xl ${riskMetrics.time_underwater_pct > 50 ? 'text-loss' : ''}`}>
            {riskMetrics.time_underwater_pct.toFixed(1)}%
          </div>
        </div>
        <div className="card">
          <div className="metric-label">Payoff Ratio</div>
          <div className={`font-mono text-xl ${riskMetrics.payoff_ratio > 1 ? 'text-profit' : 'text-loss'}`}>
            {riskMetrics.payoff_ratio.toFixed(2)}
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex items-center gap-4 border-b border-terminal-border">
        {[
          { key: 'var', label: 'VaR Analysis', icon: Shield },
          { key: 'distribution', label: 'Return Distribution', icon: BarChart3 },
          { key: 'rolling', label: 'Rolling Metrics', icon: Activity },
          { key: 'monthly', label: 'Monthly Returns', icon: Calendar },
        ].map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key as typeof activeTab)}
            className={`flex items-center gap-2 pb-3 px-1 font-medium transition-colors relative ${
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
      {activeTab === 'var' && (
        <div className="card-elevated">
          <h3 className="font-semibold text-lg mb-4">Value at Risk Analysis</h3>
          <VaRGauge
            var95={riskMetrics.var_95}
            var99={riskMetrics.var_99}
            cvar95={riskMetrics.cvar_95}
            cvar99={riskMetrics.cvar_99}
          />
        </div>
      )}

      {activeTab === 'distribution' && (
        <div className="card-elevated">
          <h3 className="font-semibold text-lg mb-4">Daily Return Distribution</h3>
          <div className="h-[400px]">
            <ReturnDistribution returns={riskMetrics.daily_returns} showNormal />
          </div>
        </div>
      )}

      {activeTab === 'rolling' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="card-elevated">
            <h3 className="font-semibold text-lg mb-4">Rolling Sharpe (252-day)</h3>
            <div className="h-[300px]">
              <RollingMetrics
                data={[
                  { label: 'Sharpe', points: riskMetrics.rolling_sharpe, color: '#00ff88' },
                ]}
                showZeroLine
              />
            </div>
          </div>
          <div className="card-elevated">
            <h3 className="font-semibold text-lg mb-4">Rolling Volatility (252-day)</h3>
            <div className="h-[300px]">
              <RollingMetrics
                data={[
                  { label: 'Volatility', points: riskMetrics.rolling_volatility, color: '#ff6b6b' },
                ]}
                showZeroLine={false}
              />
            </div>
          </div>
          <div className="card-elevated lg:col-span-2">
            <h3 className="font-semibold text-lg mb-4">Rolling Returns (21-day)</h3>
            <div className="h-[300px]">
              <RollingMetrics
                data={[
                  { label: '21-day Return', points: riskMetrics.rolling_returns, color: '#00d4ff' },
                ]}
                showZeroLine
              />
            </div>
          </div>
        </div>
      )}

      {activeTab === 'monthly' && (
        <div className="card-elevated">
          <h3 className="font-semibold text-lg mb-4">Monthly Returns Heatmap</h3>
          <MonthlyHeatmap data={riskMetrics.monthly_returns} />
        </div>
      )}
    </div>
  );
}


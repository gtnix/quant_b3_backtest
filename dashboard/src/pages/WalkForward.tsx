import { useState, useEffect } from 'react';
import { MetricCard } from '../components/ui/MetricCard';
import { WalkForwardChart } from '../components/charts/WalkForwardChart';
import { QuickTooltip } from '../components/ui/TooltipInfo';
import { useDataStore } from '../stores/dataStore';
import {
  BarChart3,
  RefreshCw,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Target,
  Activity,
  Settings,
  Play,
} from 'lucide-react';

export function WalkForward() {
  const [windowMonths, setWindowMonths] = useState(12);
  const [stepMonths, setStepMonths] = useState(3);
  const [metric, setMetric] = useState<'sharpe' | 'return'>('sharpe');

  const {
    selectedCandidate,
    walkForwardResult,
    isLoading,
    error,
    loadWalkForward,
    artifactsRoot,
  } = useDataStore();

  // Load walk-forward when candidate is selected or params change
  useEffect(() => {
    if (selectedCandidate) {
      loadWalkForward(selectedCandidate.candidate_id, windowMonths, stepMonths);
    }
  }, [selectedCandidate?.candidate_id, windowMonths, stepMonths]);

  // No artifacts root
  if (!artifactsRoot) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-6">
        <BarChart3 className="w-16 h-16 text-terminal-muted" />
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
        <BarChart3 className="w-16 h-16 text-terminal-muted" />
        <div className="text-center">
          <h2 className="text-xl font-semibold mb-2">No Candidate Selected</h2>
          <p className="text-terminal-muted">
            Select a candidate from the Candidates page to run walk-forward analysis.
          </p>
        </div>
      </div>
    );
  }

  // Loading
  if (isLoading && !walkForwardResult) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-4">
        <RefreshCw className="w-12 h-12 animate-spin text-terminal-muted" />
        <p className="text-terminal-muted">Running walk-forward analysis...</p>
      </div>
    );
  }

  if (!walkForwardResult) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-6">
        <AlertTriangle className="w-16 h-16 text-accent-yellow" />
        <div className="text-center">
          <h2 className="text-xl font-semibold mb-2">Analysis Failed</h2>
          <p className="text-terminal-muted max-w-md">
            {error || 'Unable to run walk-forward analysis for this candidate.'}
          </p>
        </div>
      </div>
    );
  }

  const isRobust = walkForwardResult.degradation_ratio >= 0.5 && walkForwardResult.consistency_score >= 0.6;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold inline-flex items-center">Walk-Forward Analysis<QuickTooltip termKey="wfa" size="md" /></h1>
          <p className="text-terminal-muted mt-1">
            Validate strategy robustness for{' '}
            <span className="text-accent-cyan font-mono">
              {selectedCandidate.display_name}
            </span>
          </p>
        </div>
        <button
          onClick={() => loadWalkForward(selectedCandidate.candidate_id, windowMonths, stepMonths)}
          className="flex items-center gap-2 px-3 py-2 rounded-lg bg-terminal-surface border border-terminal-border hover:border-profit transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Configuration */}
      <div className="card flex items-center gap-6 flex-wrap">
        <div className="flex items-center gap-2">
          <Settings className="w-4 h-4 text-terminal-muted" />
          <span className="text-sm text-terminal-muted">Parameters:</span>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-sm text-terminal-muted inline-flex items-center">Window<QuickTooltip termKey="wfa_window" /></label>
          <select
            value={windowMonths}
            onChange={(e) => setWindowMonths(Number(e.target.value))}
            className="px-2 py-1 bg-terminal-bg border border-terminal-border rounded text-sm"
          >
            <option value={6}>6 months</option>
            <option value={12}>12 months</option>
            <option value={18}>18 months</option>
            <option value={24}>24 months</option>
          </select>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-sm text-terminal-muted inline-flex items-center">Step<QuickTooltip termKey="wfa_step" /></label>
          <select
            value={stepMonths}
            onChange={(e) => setStepMonths(Number(e.target.value))}
            className="px-2 py-1 bg-terminal-bg border border-terminal-border rounded text-sm"
          >
            <option value={1}>1 month</option>
            <option value={3}>3 months</option>
            <option value={6}>6 months</option>
          </select>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-sm text-terminal-muted">Metric:</label>
          <select
            value={metric}
            onChange={(e) => setMetric(e.target.value as 'sharpe' | 'return')}
            className="px-2 py-1 bg-terminal-bg border border-terminal-border rounded text-sm"
          >
            <option value="sharpe">Sharpe Ratio</option>
            <option value="return">Return</option>
          </select>
        </div>
      </div>

      {/* Robustness Verdict */}
      <div className={`p-4 rounded-lg border ${isRobust ? 'bg-profit/10 border-profit/30' : 'bg-loss/10 border-loss/30'}`}>
        <div className="flex items-center gap-3">
          {isRobust ? (
            <TrendingUp className="w-6 h-6 text-profit" />
          ) : (
            <TrendingDown className="w-6 h-6 text-loss" />
          )}
          <div>
            <div className={`font-semibold ${isRobust ? 'text-profit' : 'text-loss'}`}>
              {isRobust ? 'Strategy Appears Robust' : 'Strategy May Be Overfit'}
            </div>
            <div className="text-sm text-terminal-muted">
              {isRobust
                ? 'OOS performance retains majority of IS performance with consistent profitability'
                : 'Significant degradation between in-sample and out-of-sample periods detected'}
            </div>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="card group hover:border-terminal-muted/50 transition-colors">
          <div className="flex items-start justify-between mb-2">
            <span className="metric-label inline-flex items-center">OOS Sharpe<QuickTooltip termKey="sharpe_oos" /></span>
            <Target className="w-4 h-4" />
          </div>
          <div className="font-mono font-bold text-2xl">{walkForwardResult.aggregate_sharpe.toFixed(3)}</div>
        </div>
        <div className="card group hover:border-terminal-muted/50 transition-colors">
          <div className="flex items-start justify-between mb-2">
            <span className="metric-label inline-flex items-center">Degradation Ratio<QuickTooltip termKey="degradation_ratio" /></span>
            <Activity className={`w-4 h-4 ${walkForwardResult.degradation_ratio >= 0.5 ? 'text-profit' : 'text-loss'}`} />
          </div>
          <div className={`font-mono font-bold text-2xl ${walkForwardResult.degradation_ratio >= 0.5 ? 'text-profit' : 'text-loss'}`}>{(walkForwardResult.degradation_ratio * 100).toFixed(1)}%</div>
        </div>
        <div className="card group hover:border-terminal-muted/50 transition-colors">
          <div className="flex items-start justify-between mb-2">
            <span className="metric-label inline-flex items-center">Consistency<QuickTooltip termKey="consistency_score" /></span>
            <BarChart3 className={`w-4 h-4 ${walkForwardResult.consistency_score >= 0.6 ? 'text-profit' : 'text-loss'}`} />
          </div>
          <div className={`font-mono font-bold text-2xl ${walkForwardResult.consistency_score >= 0.6 ? 'text-profit' : 'text-loss'}`}>{(walkForwardResult.consistency_score * 100).toFixed(1)}%</div>
        </div>
        <div className="card group hover:border-terminal-muted/50 transition-colors">
          <div className="flex items-start justify-between mb-2">
            <span className="metric-label">Profit / Loss Periods</span>
          </div>
          <div className="font-mono font-bold text-2xl">{walkForwardResult.profit_periods} / {walkForwardResult.loss_periods}</div>
        </div>
      </div>

      {/* Walk-Forward Chart */}
      <div className="card-elevated">
        <h3 className="font-semibold text-lg mb-4">
          In-Sample vs Out-of-Sample {metric === 'sharpe' ? 'Sharpe' : 'Return'}
        </h3>
        <div className="h-[350px]">
          <WalkForwardChart windows={walkForwardResult.windows} metric={metric} />
        </div>
      </div>

      {/* Period Details Table */}
      <div className="card-elevated overflow-x-auto">
        <h3 className="font-semibold text-lg mb-4 inline-flex items-center">Period Details<QuickTooltip termKey="is_oos" /></h3>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-terminal-border">
              <th className="text-left py-2 px-3 text-terminal-muted font-normal">#</th>
              <th className="text-left py-2 px-3 text-terminal-muted font-normal">Period</th>
              <th className="text-right py-2 px-3 text-terminal-muted font-normal">IS Sharpe</th>
              <th className="text-right py-2 px-3 text-terminal-muted font-normal">OOS Sharpe</th>
              <th className="text-right py-2 px-3 text-terminal-muted font-normal">IS Return</th>
              <th className="text-right py-2 px-3 text-terminal-muted font-normal">OOS Return</th>
              <th className="text-right py-2 px-3 text-terminal-muted font-normal">IS MaxDD</th>
              <th className="text-right py-2 px-3 text-terminal-muted font-normal">OOS MaxDD</th>
              <th className="text-center py-2 px-3 text-terminal-muted font-normal">Degradation</th>
            </tr>
          </thead>
          <tbody>
            {walkForwardResult.windows.map((w, i) => {
              const degradation = w.is_sharpe > 0 ? w.oos_sharpe / w.is_sharpe : 0;
              const isGood = degradation >= 0.5;
              
              return (
                <tr key={i} className="border-b border-terminal-border/30 hover:bg-terminal-surface/50">
                  <td className="py-2 px-3 font-mono text-terminal-muted">{i + 1}</td>
                  <td className="py-2 px-3">
                    <span className="font-mono text-xs">
                      {w.period_start.substring(0, 10)} → {w.period_end.substring(0, 10)}
                    </span>
                  </td>
                  <td className="text-right py-2 px-3 font-mono text-accent-cyan">
                    {w.is_sharpe.toFixed(2)}
                  </td>
                  <td className={`text-right py-2 px-3 font-mono ${w.oos_sharpe >= 0 ? 'text-profit' : 'text-loss'}`}>
                    {w.oos_sharpe.toFixed(2)}
                  </td>
                  <td className="text-right py-2 px-3 font-mono">
                    {(w.is_return * 100).toFixed(1)}%
                  </td>
                  <td className={`text-right py-2 px-3 font-mono ${w.oos_return >= 0 ? 'text-profit' : 'text-loss'}`}>
                    {(w.oos_return * 100).toFixed(1)}%
                  </td>
                  <td className="text-right py-2 px-3 font-mono text-loss">
                    -{(w.is_max_dd * 100).toFixed(1)}%
                  </td>
                  <td className="text-right py-2 px-3 font-mono text-loss">
                    -{(w.oos_max_dd * 100).toFixed(1)}%
                  </td>
                  <td className="text-center py-2 px-3">
                    <span className={`px-2 py-0.5 rounded text-xs font-mono ${isGood ? 'bg-profit/20 text-profit' : 'bg-loss/20 text-loss'}`}>
                      {(degradation * 100).toFixed(0)}%
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Interpretation Guide */}
      <div className="card">
        <h3 className="font-semibold mb-3">Interpretation Guide</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <div className="font-medium text-profit mb-1">Good Signs</div>
            <ul className="list-disc list-inside text-terminal-muted space-y-1">
              <li>Degradation ratio {'>'}50% (OOS retains IS performance)</li>
              <li>Consistency score {'>'}60% (majority profitable periods)</li>
              <li>Stable OOS Sharpe across periods</li>
            </ul>
          </div>
          <div>
            <div className="font-medium text-loss mb-1">Warning Signs</div>
            <ul className="list-disc list-inside text-terminal-muted space-y-1">
              <li>OOS Sharpe {'<'}50% of IS Sharpe</li>
              <li>Many negative OOS periods</li>
              <li>High variance in OOS performance</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}


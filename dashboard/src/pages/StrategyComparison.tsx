import { useState, useEffect } from 'react';
import { MetricCard } from '../components/ui/MetricCard';
import { CorrelationMatrix } from '../components/charts/CorrelationMatrix';
import { EquityChart } from '../components/charts/EquityChart';
import { QuickTooltip } from '../components/ui/TooltipInfo';
import { useDataStore } from '../stores/dataStore';
import {
  GitCompare,
  RefreshCw,
  AlertTriangle,
  X,
  Plus,
  Layers,
  TrendingUp,
  Target,
  Shield,
} from 'lucide-react';

export function StrategyComparison() {
  const {
    selectedCandidateIds,
    comparisonResult,
    isLoading,
    error,
    compareCandidates,
    clearCandidateSelection,
  } = useDataStore();

  // Run comparison when candidates change
  useEffect(() => {
    if (selectedCandidateIds.length >= 2) {
      compareCandidates(selectedCandidateIds);
    }
  }, [selectedCandidateIds]);

  // Not enough candidates selected
  if (selectedCandidateIds.length < 2) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-6">
        <GitCompare className="w-16 h-16 text-terminal-muted" />
        <div className="text-center max-w-md">
          <h2 className="text-xl font-semibold mb-2">Select Candidates to Compare</h2>
          <p className="text-terminal-muted mb-4">
            Go to the Candidates page and use the checkboxes to select 2-10 strategies for comparison.
          </p>
          <div className="flex items-center justify-center gap-2 text-sm text-terminal-muted">
            <span className="px-2 py-1 bg-terminal-surface rounded">Selected:</span>
            <span className="font-mono text-profit">{selectedCandidateIds.length}</span>
            <span>/ 2 minimum</span>
          </div>
        </div>
      </div>
    );
  }

  // Loading
  if (isLoading && !comparisonResult) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-4">
        <RefreshCw className="w-12 h-12 animate-spin text-terminal-muted" />
        <p className="text-terminal-muted">Comparing strategies...</p>
      </div>
    );
  }

  if (!comparisonResult) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-6">
        <AlertTriangle className="w-16 h-16 text-accent-yellow" />
        <div className="text-center">
          <h2 className="text-xl font-semibold mb-2">Comparison Failed</h2>
          <p className="text-terminal-muted max-w-md">
            {error || 'Unable to compare the selected candidates.'}
          </p>
        </div>
      </div>
    );
  }

  // Format combined equity for chart
  const combinedEquityData = comparisonResult.combined_equity.map(p => ({
    time: p.date,
    value: p.equity,
  }));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Strategy Comparison</h1>
          <p className="text-terminal-muted mt-1">
            Comparing {comparisonResult.candidates.length} strategies
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={clearCandidateSelection}
            className="flex items-center gap-2 px-3 py-2 rounded-lg bg-terminal-surface border border-terminal-border hover:border-loss transition-colors text-loss"
          >
            <X className="w-4 h-4" />
            Clear Selection
          </button>
          <button
            onClick={() => compareCandidates(selectedCandidateIds)}
            className="flex items-center gap-2 px-3 py-2 rounded-lg bg-terminal-surface border border-terminal-border hover:border-profit transition-colors"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* Portfolio Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="card group hover:border-terminal-muted/50 transition-colors">
          <div className="flex items-start justify-between mb-2">
            <span className="metric-label">Strategies</span>
            <Layers className="w-4 h-4" />
          </div>
          <div className="font-mono font-bold text-2xl">{comparisonResult.candidates.length}</div>
        </div>
        <div className="card group hover:border-terminal-muted/50 transition-colors">
          <div className="flex items-start justify-between mb-2">
            <span className="metric-label inline-flex items-center">Diversification<QuickTooltip termKey="diversification_ratio" /></span>
            <Shield className="w-4 h-4 text-profit" />
          </div>
          <div className="font-mono font-bold text-2xl text-profit">{comparisonResult.diversification_ratio.toFixed(2)}</div>
        </div>
        <div className="card group hover:border-terminal-muted/50 transition-colors">
          <div className="flex items-start justify-between mb-2">
            <span className="metric-label inline-flex items-center">Best Sharpe<QuickTooltip termKey="sharpe" /></span>
            <Target className="w-4 h-4" />
          </div>
          <div className="font-mono font-bold text-2xl">{Math.max(...comparisonResult.candidates.map(c => c.sharpe)).toFixed(2)}</div>
        </div>
        <div className="card group hover:border-terminal-muted/50 transition-colors">
          <div className="flex items-start justify-between mb-2">
            <span className="metric-label inline-flex items-center">Best CAGR<QuickTooltip termKey="cagr" /></span>
            <TrendingUp className="w-4 h-4 text-profit" />
          </div>
          <div className="font-mono font-bold text-2xl">{(Math.max(...comparisonResult.candidates.map(c => c.cagr)) * 100).toFixed(1)}%</div>
        </div>
      </div>

      {/* Comparison Table */}
      <div className="card-elevated overflow-x-auto">
        <h3 className="font-semibold text-lg mb-4">Performance Metrics</h3>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-terminal-border">
              <th className="text-left py-2 px-3 text-terminal-muted font-normal">Strategy</th>
              <th className="text-right py-2 px-3 text-terminal-muted font-normal"><span className="inline-flex items-center">Sharpe<QuickTooltip termKey="sharpe" /></span></th>
              <th className="text-right py-2 px-3 text-terminal-muted font-normal"><span className="inline-flex items-center">CAGR<QuickTooltip termKey="cagr" /></span></th>
              <th className="text-right py-2 px-3 text-terminal-muted font-normal"><span className="inline-flex items-center">Max DD<QuickTooltip termKey="max_drawdown" /></span></th>
              <th className="text-right py-2 px-3 text-terminal-muted font-normal"><span className="inline-flex items-center">Volatility<QuickTooltip termKey="volatility" /></span></th>
              <th className="text-right py-2 px-3 text-terminal-muted font-normal"><span className="inline-flex items-center">Sortino<QuickTooltip termKey="sortino" /></span></th>
              <th className="text-right py-2 px-3 text-terminal-muted font-normal"><span className="inline-flex items-center">Calmar<QuickTooltip termKey="calmar" /></span></th>
              <th className="text-right py-2 px-3 text-terminal-muted font-normal"><span className="inline-flex items-center">PBO<QuickTooltip termKey="pbo" /></span></th>
            </tr>
          </thead>
          <tbody>
            {comparisonResult.candidates.map((c, i) => {
              const isBestSharpe = c.sharpe === Math.max(...comparisonResult.candidates.map(x => x.sharpe));
              const isBestCagr = c.cagr === Math.max(...comparisonResult.candidates.map(x => x.cagr));
              const isLowestDD = c.max_dd === Math.min(...comparisonResult.candidates.map(x => x.max_dd));
              
              return (
                <tr key={c.candidate_id} className="border-b border-terminal-border/30 hover:bg-terminal-surface/50">
                  <td className="py-3 px-3">
                    <div className="max-w-[200px] truncate font-medium" title={c.display_name}>
                      {c.display_name}
                    </div>
                    <div className="text-xs text-terminal-muted font-mono">
                      {c.candidate_id.substring(0, 16)}...
                    </div>
                  </td>
                  <td className={`text-right py-3 px-3 font-mono ${isBestSharpe ? 'text-profit font-bold' : ''}`}>
                    {c.sharpe.toFixed(2)}
                  </td>
                  <td className={`text-right py-3 px-3 font-mono ${isBestCagr ? 'text-profit font-bold' : ''}`}>
                    {(c.cagr * 100).toFixed(1)}%
                  </td>
                  <td className={`text-right py-3 px-3 font-mono text-loss ${isLowestDD ? 'font-bold' : ''}`}>
                    -{(c.max_dd * 100).toFixed(1)}%
                  </td>
                  <td className="text-right py-3 px-3 font-mono">
                    {(c.volatility * 100).toFixed(1)}%
                  </td>
                  <td className="text-right py-3 px-3 font-mono">
                    {c.sortino.toFixed(2)}
                  </td>
                  <td className="text-right py-3 px-3 font-mono">
                    {c.calmar.toFixed(2)}
                  </td>
                  <td className={`text-right py-3 px-3 font-mono ${c.pbo <= 0.1 ? 'text-profit' : c.pbo <= 0.15 ? 'text-accent-yellow' : 'text-loss'}`}>
                    {(c.pbo * 100).toFixed(1)}%
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Correlation Matrix */}
      <div className="card-elevated">
        <h3 className="font-semibold text-lg mb-4 inline-flex items-center">Correlation Matrix<QuickTooltip termKey="correlation_matrix" /></h3>
        <CorrelationMatrix
          labels={comparisonResult.candidates.map(c => 
            c.display_name.length > 15 ? `${c.display_name.substring(0, 15)}...` : c.display_name
          )}
          matrix={comparisonResult.correlation_matrix}
        />
      </div>

      {/* Combined Equity Chart */}
      <div className="card-elevated">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-lg">Combined Portfolio (Equal Weight)</h3>
          <span className="text-xs text-terminal-muted font-mono">
            Diversification: {comparisonResult.diversification_ratio.toFixed(2)}x
          </span>
        </div>
        <div className="h-[300px]">
          {combinedEquityData.length > 0 ? (
            <EquityChart data={combinedEquityData} />
          ) : (
            <div className="flex items-center justify-center h-full text-terminal-muted">
              No equity data available
            </div>
          )}
        </div>
      </div>

      {/* Individual Equity Curves */}
      <div className="card-elevated">
        <h3 className="font-semibold text-lg mb-4">Individual Equity Curves</h3>
        <div className="h-[400px]">
          {/* Overlay multiple equity curves */}
          <div className="relative h-full">
            {comparisonResult.candidates.map((c, i) => {
              const colors = ['#00ff88', '#00d4ff', '#ff6b6b', '#fbbf24', '#8b5cf6', '#f472b6', '#34d399', '#60a5fa'];
              const equityData = c.equity.map(p => ({ time: p.date, value: p.equity }));
              
              return (
                <div
                  key={c.candidate_id}
                  className="absolute inset-0"
                  style={{ opacity: 0.8 - i * 0.1 }}
                >
                  <EquityChart data={equityData} />
                </div>
              );
            })}
          </div>
        </div>
        {/* Legend */}
        <div className="flex flex-wrap items-center gap-4 mt-4">
          {comparisonResult.candidates.map((c, i) => {
            const colors = ['#00ff88', '#00d4ff', '#ff6b6b', '#fbbf24', '#8b5cf6', '#f472b6', '#34d399', '#60a5fa'];
            return (
              <div key={c.candidate_id} className="flex items-center gap-2 text-xs">
                <div
                  className="w-3 h-3 rounded"
                  style={{ backgroundColor: colors[i % colors.length] }}
                />
                <span className="truncate max-w-[150px]" title={c.display_name}>
                  {c.display_name}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}


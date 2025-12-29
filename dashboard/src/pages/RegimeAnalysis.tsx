import { useState, useEffect } from 'react';
import { MetricCard } from '../components/ui/MetricCard';
import { EquityChart } from '../components/charts/EquityChart';
import { useDataStore } from '../stores/dataStore';
import {
  Layers,
  RefreshCw,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Activity,
  Settings,
  Sun,
  CloudRain,
} from 'lucide-react';

export function RegimeAnalysis() {
  const [volThreshold, setVolThreshold] = useState(0.20);

  const {
    selectedCandidate,
    regimeAnalysis,
    backtest,
    isLoading,
    error,
    detectRegimes,
    loadBacktest,
    artifactsRoot,
  } = useDataStore();

  // Load regime analysis when candidate is selected
  useEffect(() => {
    if (selectedCandidate) {
      detectRegimes(selectedCandidate.candidate_id, volThreshold);
      loadBacktest(selectedCandidate.candidate_id);
    }
  }, [selectedCandidate?.candidate_id, volThreshold]);

  // No artifacts root
  if (!artifactsRoot) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-6">
        <Layers className="w-16 h-16 text-terminal-muted" />
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
        <Layers className="w-16 h-16 text-terminal-muted" />
        <div className="text-center">
          <h2 className="text-xl font-semibold mb-2">No Candidate Selected</h2>
          <p className="text-terminal-muted">
            Select a candidate from the Candidates page to analyze regimes.
          </p>
        </div>
      </div>
    );
  }

  // Loading
  if (isLoading && !regimeAnalysis) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-4">
        <RefreshCw className="w-12 h-12 animate-spin text-terminal-muted" />
        <p className="text-terminal-muted">Detecting market regimes...</p>
      </div>
    );
  }

  if (!regimeAnalysis) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-6">
        <AlertTriangle className="w-16 h-16 text-accent-yellow" />
        <div className="text-center">
          <h2 className="text-xl font-semibold mb-2">Analysis Failed</h2>
          <p className="text-terminal-muted max-w-md">
            {error || 'Unable to detect regimes for this candidate.'}
          </p>
        </div>
      </div>
    );
  }

  const regimeColors: Record<string, { bg: string; text: string; icon: React.ElementType }> = {
    BullLowVol: { bg: 'bg-profit/20', text: 'text-profit', icon: TrendingUp },
    BullHighVol: { bg: 'bg-accent-yellow/20', text: 'text-accent-yellow', icon: Activity },
    BearLowVol: { bg: 'bg-accent-cyan/20', text: 'text-accent-cyan', icon: CloudRain },
    BearHighVol: { bg: 'bg-loss/20', text: 'text-loss', icon: TrendingDown },
  };

  const getCurrentRegimeInfo = () => {
    const info = regimeColors[regimeAnalysis.current_regime];
    return info || { bg: 'bg-terminal-surface', text: 'text-white', icon: Activity };
  };

  const currentRegime = getCurrentRegimeInfo();
  const CurrentIcon = currentRegime.icon;

  // Prepare equity data with regime overlay
  const equityData = backtest?.timeseries.map(p => ({
    time: p.date,
    value: p.equity,
  })) || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Regime Analysis</h1>
          <p className="text-terminal-muted mt-1">
            Conditional performance analysis for{' '}
            <span className="text-accent-cyan font-mono">
              {selectedCandidate.display_name}
            </span>
          </p>
        </div>
        <button
          onClick={() => detectRegimes(selectedCandidate.candidate_id, volThreshold)}
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
          <span className="text-sm text-terminal-muted">Volatility Threshold:</span>
        </div>
        <div className="flex items-center gap-2">
          <input
            type="range"
            min="0.10"
            max="0.40"
            step="0.02"
            value={volThreshold}
            onChange={(e) => setVolThreshold(Number(e.target.value))}
            className="w-32"
          />
          <span className="font-mono text-sm">{(volThreshold * 100).toFixed(0)}%</span>
        </div>
        <div className="text-xs text-terminal-muted">
          (Annualized volatility threshold for high/low vol classification)
        </div>
      </div>

      {/* Current Regime */}
      <div className={`p-4 rounded-lg border ${currentRegime.bg} border-current`}>
        <div className="flex items-center gap-3">
          <CurrentIcon className={`w-6 h-6 ${currentRegime.text}`} />
          <div>
            <div className={`font-semibold ${currentRegime.text}`}>
              Current Regime: {regimeAnalysis.current_regime.replace(/([A-Z])/g, ' $1').trim()}
            </div>
            <div className="text-sm text-terminal-muted">
              Based on the most recent trading data
            </div>
          </div>
        </div>
      </div>

      {/* Regime Statistics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {regimeAnalysis.regime_stats.map(stat => {
          const info = regimeColors[stat.regime] || { bg: 'bg-terminal-surface', text: 'text-white' };
          return (
            <div key={stat.regime} className={`card ${info.bg}`}>
              <div className={`font-medium mb-2 ${info.text}`}>
                {stat.regime.replace(/([A-Z])/g, ' $1').trim()}
              </div>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-terminal-muted">Frequency:</span>
                  <span className="font-mono">{(stat.frequency * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-terminal-muted">Avg Duration:</span>
                  <span className="font-mono">{stat.avg_duration_days.toFixed(0)} days</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Performance by Regime */}
      <div className="card-elevated overflow-x-auto">
        <h3 className="font-semibold text-lg mb-4">Performance by Regime</h3>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-terminal-border">
              <th className="text-left py-2 px-3 text-terminal-muted font-normal">Regime</th>
              <th className="text-right py-2 px-3 text-terminal-muted font-normal">Sharpe</th>
              <th className="text-right py-2 px-3 text-terminal-muted font-normal">CAGR</th>
              <th className="text-right py-2 px-3 text-terminal-muted font-normal">Volatility</th>
              <th className="text-right py-2 px-3 text-terminal-muted font-normal">Max DD</th>
              <th className="text-right py-2 px-3 text-terminal-muted font-normal">Hit Rate</th>
              <th className="text-right py-2 px-3 text-terminal-muted font-normal">Days</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(regimeAnalysis.performance_by_regime).map(([regime, metrics]) => {
              const info = regimeColors[regime] || { bg: 'bg-terminal-surface', text: 'text-white' };
              return (
                <tr key={regime} className="border-b border-terminal-border/30 hover:bg-terminal-surface/50">
                  <td className="py-3 px-3">
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${info.bg} ${info.text}`}>
                      {regime.replace(/([A-Z])/g, ' $1').trim()}
                    </span>
                  </td>
                  <td className={`text-right py-3 px-3 font-mono ${metrics.sharpe >= 0 ? 'text-profit' : 'text-loss'}`}>
                    {metrics.sharpe.toFixed(2)}
                  </td>
                  <td className={`text-right py-3 px-3 font-mono ${metrics.cagr >= 0 ? 'text-profit' : 'text-loss'}`}>
                    {(metrics.cagr * 100).toFixed(1)}%
                  </td>
                  <td className="text-right py-3 px-3 font-mono">
                    {(metrics.volatility * 100).toFixed(1)}%
                  </td>
                  <td className="text-right py-3 px-3 font-mono text-loss">
                    -{(metrics.max_dd * 100).toFixed(1)}%
                  </td>
                  <td className={`text-right py-3 px-3 font-mono ${metrics.hit_rate >= 0.5 ? 'text-profit' : 'text-loss'}`}>
                    {(metrics.hit_rate * 100).toFixed(0)}%
                  </td>
                  <td className="text-right py-3 px-3 font-mono text-terminal-muted">
                    {metrics.num_days}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Equity Chart with Regime Overlay */}
      <div className="card-elevated">
        <h3 className="font-semibold text-lg mb-4">Equity Curve with Regime Overlay</h3>
        <div className="h-[400px] relative">
          {/* Regime background bands */}
          <div className="absolute inset-0 flex">
            {regimeAnalysis.regimes.map((period, i) => {
              const totalDays = equityData.length;
              const startIdx = equityData.findIndex(d => d.time >= period.start_date);
              const endIdx = equityData.findIndex(d => d.time >= period.end_date);
              const startPct = (startIdx / totalDays) * 100;
              const widthPct = ((endIdx - startIdx) / totalDays) * 100;
              
              if (startPct < 0 || widthPct <= 0) return null;
              
              return (
                <div
                  key={i}
                  className="absolute top-0 bottom-0 opacity-20"
                  style={{
                    left: `${startPct}%`,
                    width: `${Math.max(widthPct, 0.5)}%`,
                    backgroundColor: period.color,
                  }}
                  title={`${period.regime}: ${period.start_date} to ${period.end_date}`}
                />
              );
            })}
          </div>
          
          {/* Equity chart */}
          {equityData.length > 0 ? (
            <EquityChart data={equityData} />
          ) : (
            <div className="flex items-center justify-center h-full text-terminal-muted">
              No equity data available
            </div>
          )}
        </div>
        
        {/* Legend */}
        <div className="flex items-center justify-center gap-6 mt-4 text-xs">
          {Object.entries(regimeColors).map(([regime, info]) => (
            <div key={regime} className="flex items-center gap-2">
              <div className={`w-4 h-4 rounded ${info.bg}`} />
              <span className="text-terminal-muted">
                {regime.replace(/([A-Z])/g, ' $1').trim()}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Regime Timeline */}
      <div className="card-elevated">
        <h3 className="font-semibold text-lg mb-4">Regime Timeline</h3>
        <div className="space-y-2 max-h-[300px] overflow-y-auto">
          {regimeAnalysis.regimes.slice(0, 20).map((period, i) => {
            const info = regimeColors[period.regime] || { bg: 'bg-terminal-surface', text: 'text-white' };
            const duration = Math.round(
              (new Date(period.end_date).getTime() - new Date(period.start_date).getTime()) / (1000 * 60 * 60 * 24)
            );
            
            return (
              <div key={i} className="flex items-center gap-4 py-2 border-b border-terminal-border/30">
                <div className="w-20 font-mono text-xs text-terminal-muted">
                  {period.start_date.substring(0, 10)}
                </div>
                <div
                  className={`flex-1 h-6 rounded flex items-center px-2 text-xs font-medium ${info.bg} ${info.text}`}
                >
                  {period.regime.replace(/([A-Z])/g, ' $1').trim()}
                </div>
                <div className="w-16 text-right font-mono text-xs text-terminal-muted">
                  {duration} days
                </div>
              </div>
            );
          })}
          {regimeAnalysis.regimes.length > 20 && (
            <div className="text-center text-sm text-terminal-muted py-2">
              +{regimeAnalysis.regimes.length - 20} more periods
            </div>
          )}
        </div>
      </div>

      {/* Interpretation */}
      <div className="card">
        <h3 className="font-semibold mb-3">Interpretation</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <div className="font-medium text-profit mb-1">Ideal Strategy Profile</div>
            <ul className="list-disc list-inside text-terminal-muted space-y-1">
              <li>Positive Sharpe in all regimes</li>
              <li>Lower drawdowns in bear markets</li>
              <li>Consistent performance across market conditions</li>
            </ul>
          </div>
          <div>
            <div className="font-medium text-loss mb-1">Red Flags</div>
            <ul className="list-disc list-inside text-terminal-muted space-y-1">
              <li>Negative Sharpe in any regime</li>
              <li>Large losses concentrated in specific regimes</li>
              <li>Performance only in bull markets (momentum-only)</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}


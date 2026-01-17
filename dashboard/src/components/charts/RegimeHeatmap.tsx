/**
 * RegimeHeatmap - 3x5 Performance Matrix by Trend x Volatility
 * 
 * Visualizes strategy performance across 15 regime combinations:
 * - 3 Trends: Uptrend, Sideways, Downtrend
 * - 5 Vol Quantiles: Q1 (low) to Q5 (high)
 */

import { useMemo, useState } from 'react';
import { TrendingUp, Minus, TrendingDown } from 'lucide-react';

// Types matching backend regime.rs
export type TrendState = 'Uptrend' | 'Sideways' | 'Downtrend';
export type VolQuantile = 'Q1' | 'Q2' | 'Q3' | 'Q4' | 'Q5';

export interface RegimePerformance {
  trend_state: TrendState;
  vol_quantile: VolQuantile;
  day_count: number;
  mean_return_pct: number;
  cumulative_return_pct: number;
  win_rate_pct: number;
  sharpe?: number;
  cagr?: number;
  max_dd?: number;
}

interface RegimeHeatmapProps {
  data: RegimePerformance[];
  metric?: 'sharpe' | 'return' | 'winRate';
  currentRegime?: { trend: TrendState; vol: VolQuantile };
  onCellClick?: (trend: TrendState, vol: VolQuantile) => void;
  selectedCell?: { trend: TrendState; vol: VolQuantile } | null;
}

const TRENDS: TrendState[] = ['Uptrend', 'Sideways', 'Downtrend'];
const VOL_QUANTILES: VolQuantile[] = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'];

const TREND_ICONS = {
  Uptrend: TrendingUp,
  Sideways: Minus,
  Downtrend: TrendingDown,
};

const TREND_COLORS = {
  Uptrend: 'text-profit',
  Sideways: 'text-accent-cyan',
  Downtrend: 'text-loss',
};

const VOL_LABELS = {
  Q1: 'Muito Baixa',
  Q2: 'Baixa',
  Q3: 'Normal',
  Q4: 'Alta',
  Q5: 'Muito Alta',
};

export function RegimeHeatmap({ 
  data, 
  metric = 'sharpe', 
  currentRegime, 
  onCellClick,
  selectedCell 
}: RegimeHeatmapProps) {
  const [hoveredCell, setHoveredCell] = useState<{ trend: TrendState; vol: VolQuantile } | null>(null);

  // Build grid lookup
  const grid = useMemo(() => {
    const map = new Map<string, RegimePerformance>();
    for (const item of data) {
      const key = `${item.trend_state}-${item.vol_quantile}`;
      map.set(key, item);
    }
    return map;
  }, [data]);

  // Get metric value for display
  const getMetricValue = (perf: RegimePerformance | undefined): number | null => {
    if (!perf) return null;
    switch (metric) {
      case 'sharpe':
        return perf.sharpe ?? (perf.mean_return_pct / 100 * 252) / 0.15; // Approximate if not provided
      case 'return':
        return perf.cumulative_return_pct;
      case 'winRate':
        return perf.win_rate_pct;
      default:
        return null;
    }
  };

  // Color scale based on Sharpe-like values
  const getColor = (value: number | null, metric: string): string => {
    if (value === null) return 'bg-terminal-surface/50';
    
    if (metric === 'winRate') {
      if (value >= 60) return 'bg-profit/80';
      if (value >= 55) return 'bg-profit/50';
      if (value >= 50) return 'bg-accent-yellow/50';
      if (value >= 45) return 'bg-loss/30';
      return 'bg-loss/60';
    }
    
    if (metric === 'return') {
      if (value >= 20) return 'bg-profit/80';
      if (value >= 10) return 'bg-profit/50';
      if (value >= 0) return 'bg-profit/20';
      if (value >= -10) return 'bg-loss/30';
      return 'bg-loss/60';
    }
    
    // Sharpe scale
    if (value >= 1.5) return 'bg-profit/80';
    if (value >= 1.0) return 'bg-profit/50';
    if (value >= 0.5) return 'bg-profit/20';
    if (value >= 0) return 'bg-accent-yellow/30';
    if (value >= -0.5) return 'bg-loss/30';
    return 'bg-loss/60';
  };

  const formatValue = (value: number | null, metric: string): string => {
    if (value === null) return '-';
    if (metric === 'winRate') return `${value.toFixed(0)}%`;
    if (metric === 'return') return `${value >= 0 ? '+' : ''}${value.toFixed(1)}%`;
    return value.toFixed(2);
  };

  // Total days for percentage calculation
  const totalDays = useMemo(() => {
    return data.reduce((sum, p) => sum + p.day_count, 0);
  }, [data]);

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-terminal-muted">
        Sem dados de regime disponíveis
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Metric Selector */}
      <div className="flex items-center justify-between">
        <div className="text-xs text-terminal-muted">Métrica visualizada:</div>
        <div className="flex gap-1">
          {(['sharpe', 'return', 'winRate'] as const).map((m) => (
            <button
              key={m}
              className={`px-2 py-1 text-xs rounded transition-colors ${
                metric === m 
                  ? 'bg-accent-cyan/20 text-accent-cyan border border-accent-cyan/30' 
                  : 'bg-terminal-surface border border-terminal-border hover:border-terminal-muted'
              }`}
            >
              {m === 'sharpe' ? 'Sharpe' : m === 'return' ? 'Retorno' : 'Win Rate'}
            </button>
          ))}
        </div>
      </div>

      {/* Heatmap Grid */}
      <div className="overflow-x-auto">
        <table className="w-full border-collapse">
          <thead>
            <tr>
              <th className="w-24"></th>
              {VOL_QUANTILES.map((vol) => (
                <th key={vol} className="text-center py-2 px-1">
                  <div className="text-xs font-medium text-terminal-muted">{vol}</div>
                  <div className="text-[10px] text-terminal-muted/60">{VOL_LABELS[vol]}</div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {TRENDS.map((trend, trendIdx) => {
              const TrendIcon = TREND_ICONS[trend];
              // Show tooltip below for first two rows to prevent cutoff
              const tooltipBelow = trendIdx < 2;
              
              return (
                <tr key={trend}>
                  <td className="py-2 pr-2">
                    <div className="flex items-center gap-2">
                      <TrendIcon className={`w-4 h-4 ${TREND_COLORS[trend]}`} />
                      <span className={`text-xs font-medium ${TREND_COLORS[trend]}`}>
                        {trend === 'Uptrend' ? 'Alta' : trend === 'Sideways' ? 'Lateral' : 'Baixa'}
                      </span>
                    </div>
                  </td>
                  {VOL_QUANTILES.map((vol) => {
                    const key = `${trend}-${vol}`;
                    const perf = grid.get(key);
                    const value = getMetricValue(perf);
                    const isCurrent = currentRegime?.trend === trend && currentRegime?.vol === vol;
                    const isSelected = selectedCell?.trend === trend && selectedCell?.vol === vol;
                    const isHovered = hoveredCell?.trend === trend && hoveredCell?.vol === vol;
                    
                    return (
                      <td key={vol} className="p-1">
                        <div
                          className={`
                            relative rounded-lg p-3 text-center cursor-pointer transition-all
                            ${getColor(value, metric)}
                            ${isCurrent ? 'ring-2 ring-accent-cyan ring-offset-1 ring-offset-terminal-bg animate-pulse' : ''}
                            ${isSelected ? 'ring-2 ring-white/50' : ''}
                            ${isHovered ? 'scale-105 shadow-lg z-10' : ''}
                            hover:scale-105 hover:shadow-lg hover:z-10
                          `}
                          onClick={() => onCellClick?.(trend, vol)}
                          onMouseEnter={() => setHoveredCell({ trend, vol })}
                          onMouseLeave={() => setHoveredCell(null)}
                        >
                          {/* Main Value */}
                          <div className="font-mono font-bold text-lg">
                            {formatValue(value, metric)}
                          </div>
                          
                          {/* Days count */}
                          {perf && (
                            <div className="text-[10px] text-white/60 mt-1">
                              {perf.day_count}d ({((perf.day_count / totalDays) * 100).toFixed(0)}%)
                            </div>
                          )}

                          {/* Current regime indicator */}
                          {isCurrent && (
                            <div className="absolute -top-1 -right-1 w-3 h-3 bg-accent-cyan rounded-full animate-ping" />
                          )}

                          {/* Hover Tooltip - positioned below for top rows, above for bottom row */}
                          {isHovered && perf && (
                            <div className={`absolute left-1/2 -translate-x-1/2 z-50 pointer-events-none ${
                              tooltipBelow ? 'top-full mt-2' : 'bottom-full mb-2'
                            }`}>
                              <div className="bg-terminal-bg border border-terminal-border rounded-lg p-3 shadow-xl min-w-[180px]">
                                <div className="font-medium text-xs mb-2 text-white">
                                  {trend === 'Uptrend' ? 'Alta' : trend === 'Sideways' ? 'Lateral' : 'Baixa'} + Vol {vol}
                                </div>
                                <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[10px]">
                                  <span className="text-terminal-muted">Sharpe:</span>
                                  <span className={`font-mono ${(perf.sharpe ?? 0) >= 0 ? 'text-profit' : 'text-loss'}`}>
                                    {(perf.sharpe ?? (perf.mean_return_pct / 100 * 252) / 0.15).toFixed(2)}
                                  </span>
                                  <span className="text-terminal-muted">Retorno:</span>
                                  <span className={`font-mono ${perf.cumulative_return_pct >= 0 ? 'text-profit' : 'text-loss'}`}>
                                    {perf.cumulative_return_pct >= 0 ? '+' : ''}{perf.cumulative_return_pct.toFixed(1)}%
                                  </span>
                                  <span className="text-terminal-muted">Win Rate:</span>
                                  <span className="font-mono">{perf.win_rate_pct.toFixed(0)}%</span>
                                  <span className="text-terminal-muted">Dias:</span>
                                  <span className="font-mono">{perf.day_count}</span>
                                  <span className="text-terminal-muted">% Tempo:</span>
                                  <span className="font-mono">{((perf.day_count / totalDays) * 100).toFixed(1)}%</span>
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-4 text-xs flex-wrap pt-2 border-t border-terminal-border">
        <span className="text-terminal-muted">Sharpe:</span>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 rounded bg-loss/60" />
          <span className="text-terminal-muted">{'< -0.5'}</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 rounded bg-loss/30" />
          <span className="text-terminal-muted">-0.5 a 0</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 rounded bg-accent-yellow/30" />
          <span className="text-terminal-muted">0 a 0.5</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 rounded bg-profit/20" />
          <span className="text-terminal-muted">0.5 a 1.0</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 rounded bg-profit/50" />
          <span className="text-terminal-muted">1.0 a 1.5</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 rounded bg-profit/80" />
          <span className="text-terminal-muted">{'>= 1.5'}</span>
        </div>
      </div>
    </div>
  );
}

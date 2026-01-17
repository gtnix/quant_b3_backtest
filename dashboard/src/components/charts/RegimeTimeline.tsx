/**
 * RegimeTimeline - Interactive timeline showing regime changes over time
 * 
 * Features:
 * - Color-coded regime periods
 * - Hover to see period details
 * - Click to filter/select period
 */

import { useMemo, useState, useRef } from 'react';
import type { TrendState, VolQuantile } from './RegimeHeatmap';

export interface RegimePeriod {
  start_date: string;
  end_date: string;
  trend: TrendState;
  vol: VolQuantile;
  days: number;
}

interface RegimeTimelineProps {
  periods: RegimePeriod[];
  onPeriodClick?: (period: RegimePeriod) => void;
  selectedRegime?: { trend: TrendState; vol: VolQuantile } | null;
  height?: number;
}

// Color palette for regimes (3 trends x 5 vol = 15 colors)
const REGIME_COLORS: Record<TrendState, Record<VolQuantile, string>> = {
  Uptrend: {
    Q1: '#22c55e',
    Q2: '#16a34a',
    Q3: '#15803d',
    Q4: '#166534',
    Q5: '#14532d',
  },
  Sideways: {
    Q1: '#3b82f6',
    Q2: '#2563eb',
    Q3: '#1d4ed8',
    Q4: '#1e40af',
    Q5: '#1e3a8a',
  },
  Downtrend: {
    Q1: '#f97316',
    Q2: '#ea580c',
    Q3: '#c2410c',
    Q4: '#9a3412',
    Q5: '#7c2d12',
  },
};

const TREND_LABELS = {
  Uptrend: 'Alta',
  Sideways: 'Lateral',
  Downtrend: 'Baixa',
};

export function RegimeTimeline({ 
  periods, 
  onPeriodClick, 
  selectedRegime,
  height = 60 
}: RegimeTimelineProps) {
  const [hoveredPeriod, setHoveredPeriod] = useState<RegimePeriod | null>(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  // Calculate total days and period widths
  const { totalDays, periodWidths } = useMemo(() => {
    const total = periods.reduce((sum, p) => sum + p.days, 0);
    const widths = periods.map(p => (p.days / total) * 100);
    return { totalDays: total, periodWidths: widths };
  }, [periods]);

  const handleMouseMove = (e: React.MouseEvent, period: RegimePeriod) => {
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      setTooltipPos({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      });
    }
    setHoveredPeriod(period);
  };

  if (periods.length === 0) {
    return (
      <div className="flex items-center justify-center h-16 text-terminal-muted text-sm">
        Sem dados de timeline disponíveis
      </div>
    );
  }

  // Get date range
  const startDate = periods[0]?.start_date?.substring(0, 10) ?? '';
  const endDate = periods[periods.length - 1]?.end_date?.substring(0, 10) ?? '';

  return (
    <div className="space-y-2">
      {/* Date labels */}
      <div className="flex justify-between text-[10px] text-terminal-muted font-mono">
        <span>{startDate}</span>
        <span>{endDate}</span>
      </div>

      {/* Timeline */}
      <div 
        ref={containerRef}
        className="relative flex h-12 rounded-lg overflow-hidden border border-terminal-border"
        style={{ height }}
      >
        {periods.map((period, i) => {
          const color = REGIME_COLORS[period.trend]?.[period.vol] ?? '#6b7280';
          const isSelected = selectedRegime?.trend === period.trend && selectedRegime?.vol === period.vol;
          const isHovered = hoveredPeriod === period;
          const opacity = selectedRegime && !isSelected ? 0.3 : 1;
          
          return (
            <div
              key={i}
              className={`
                relative cursor-pointer transition-all duration-150
                ${isHovered ? 'brightness-125 z-10' : ''}
                ${isSelected ? 'ring-2 ring-white/50 ring-inset' : ''}
              `}
              style={{ 
                width: `${periodWidths[i]}%`, 
                backgroundColor: color,
                opacity,
              }}
              onClick={() => onPeriodClick?.(period)}
              onMouseMove={(e) => handleMouseMove(e, period)}
              onMouseLeave={() => setHoveredPeriod(null)}
            >
              {/* Period label (only if wide enough) */}
              {periodWidths[i] > 5 && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-[9px] font-medium text-white/80 truncate px-1">
                    {period.days}d
                  </span>
                </div>
              )}
            </div>
          );
        })}

        {/* Hover Tooltip */}
        {hoveredPeriod && (
          <div 
            className="absolute z-20 pointer-events-none"
            style={{ 
              left: Math.min(tooltipPos.x, (containerRef.current?.offsetWidth ?? 300) - 200),
              top: -80,
            }}
          >
            <div className="bg-terminal-bg border border-terminal-border rounded-lg p-3 shadow-xl min-w-[180px]">
              <div className="font-medium text-xs mb-2">
                {TREND_LABELS[hoveredPeriod.trend]} + Vol {hoveredPeriod.vol}
              </div>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[10px]">
                <span className="text-terminal-muted">Início:</span>
                <span className="font-mono">{hoveredPeriod.start_date.substring(0, 10)}</span>
                <span className="text-terminal-muted">Fim:</span>
                <span className="font-mono">{hoveredPeriod.end_date.substring(0, 10)}</span>
                <span className="text-terminal-muted">Duração:</span>
                <span className="font-mono">{hoveredPeriod.days} dias</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Legend - Compact */}
      <div className="flex items-center justify-center gap-4 text-[10px] flex-wrap">
        <div className="flex items-center gap-3">
          <span className="text-terminal-muted">Trend:</span>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded" style={{ backgroundColor: REGIME_COLORS.Uptrend.Q3 }} />
            <span className="text-terminal-muted">Alta</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded" style={{ backgroundColor: REGIME_COLORS.Sideways.Q3 }} />
            <span className="text-terminal-muted">Lateral</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded" style={{ backgroundColor: REGIME_COLORS.Downtrend.Q3 }} />
            <span className="text-terminal-muted">Baixa</span>
          </div>
        </div>
        <div className="text-terminal-muted/50">|</div>
        <div className="flex items-center gap-1">
          <span className="text-terminal-muted">Vol:</span>
          <span className="text-terminal-muted">Mais claro = Menor</span>
        </div>
      </div>
    </div>
  );
}

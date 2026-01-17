/**
 * RegimeHeroCard - Current regime status with allocation recommendation
 * 
 * Shows:
 * - Current market regime (trend + volatility)
 * - Historical performance in this regime
 * - Allocation recommendation based on confidence
 */

import { TrendingUp, Minus, TrendingDown, Activity, Zap, Shield, AlertTriangle } from 'lucide-react';
import type { TrendState, VolQuantile, RegimePerformance } from './RegimeHeatmap';

interface RegimeHeroCardProps {
  currentTrend: TrendState;
  currentVol: VolQuantile;
  currentVolValue: number; // Actual annualized volatility %
  performance?: RegimePerformance;
  totalDays: number;
}

const TREND_CONFIG = {
  Uptrend: { 
    icon: TrendingUp, 
    color: 'text-profit', 
    bg: 'bg-profit/10', 
    border: 'border-profit/30',
    label: 'Tendência de Alta'
  },
  Sideways: { 
    icon: Minus, 
    color: 'text-accent-cyan', 
    bg: 'bg-accent-cyan/10', 
    border: 'border-accent-cyan/30',
    label: 'Mercado Lateral'
  },
  Downtrend: { 
    icon: TrendingDown, 
    color: 'text-loss', 
    bg: 'bg-loss/10', 
    border: 'border-loss/30',
    label: 'Tendência de Baixa'
  },
};

const VOL_CONFIG: Record<VolQuantile, { label: string; percentile: string }> = {
  Q1: { label: 'Muito Baixa', percentile: '0-20%' },
  Q2: { label: 'Baixa', percentile: '20-40%' },
  Q3: { label: 'Normal', percentile: '40-60%' },
  Q4: { label: 'Alta', percentile: '60-80%' },
  Q5: { label: 'Muito Alta', percentile: '80-100%' },
};

function getAllocationRecommendation(perf: RegimePerformance | undefined, totalDays: number): {
  allocation: number;
  confidence: 'high' | 'medium' | 'low';
  label: string;
  color: string;
  icon: typeof Shield;
} {
  if (!perf || perf.day_count < 20) {
    return { 
      allocation: 50, 
      confidence: 'low', 
      label: 'REDUZIR (Poucos dados)',
      color: 'text-accent-yellow',
      icon: AlertTriangle
    };
  }

  const sharpe = perf.sharpe ?? (perf.mean_return_pct / 100 * 252) / 0.15;
  const daysRatio = perf.day_count / totalDays;
  
  // High confidence: good sharpe + enough data
  if (sharpe >= 1.0 && perf.day_count >= 50) {
    return { 
      allocation: 100, 
      confidence: 'high', 
      label: 'FULL ALLOCATION',
      color: 'text-profit',
      icon: Zap
    };
  }
  
  if (sharpe >= 0.5 && perf.day_count >= 30) {
    return { 
      allocation: 75, 
      confidence: 'medium', 
      label: 'ALOCAÇÃO MODERADA',
      color: 'text-accent-cyan',
      icon: Shield
    };
  }
  
  if (sharpe >= 0) {
    return { 
      allocation: 50, 
      confidence: 'low', 
      label: 'ALOCAÇÃO REDUZIDA',
      color: 'text-accent-yellow',
      icon: AlertTriangle
    };
  }
  
  return { 
    allocation: 25, 
    confidence: 'low', 
    label: 'CAUTELA - Regime Desfavorável',
    color: 'text-loss',
    icon: AlertTriangle
  };
}

export function RegimeHeroCard({ 
  currentTrend, 
  currentVol, 
  currentVolValue,
  performance,
  totalDays
}: RegimeHeroCardProps) {
  const trendConfig = TREND_CONFIG[currentTrend];
  const volConfig = VOL_CONFIG[currentVol];
  const TrendIcon = trendConfig.icon;
  
  const recommendation = getAllocationRecommendation(performance, totalDays);
  const RecommendationIcon = recommendation.icon;
  
  const sharpe = performance?.sharpe ?? (performance ? (performance.mean_return_pct / 100 * 252) / 0.15 : 0);

  return (
    <div className={`rounded-xl border-2 ${trendConfig.border} ${trendConfig.bg} p-5`}>
      <div className="flex items-start justify-between gap-6 flex-wrap">
        {/* Left: Current Regime */}
        <div className="flex items-start gap-4">
          {/* Icon */}
          <div className={`p-3 rounded-xl ${trendConfig.bg} border ${trendConfig.border}`}>
            <TrendIcon className={`w-8 h-8 ${trendConfig.color}`} />
          </div>
          
          {/* Info */}
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs text-terminal-muted uppercase tracking-wider">Regime Atual</span>
              <span className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-profit/20 text-profit text-[10px] font-medium">
                <span className="w-1.5 h-1.5 rounded-full bg-profit animate-pulse" />
                LIVE
              </span>
            </div>
            <div className={`text-xl font-bold ${trendConfig.color}`}>
              {trendConfig.label} + Vol {currentVol}
            </div>
            <div className="flex items-center gap-4 mt-2 text-sm">
              <div className="flex items-center gap-1.5">
                <Activity className="w-4 h-4 text-terminal-muted" />
                <span className="text-terminal-muted">Volatilidade:</span>
                <span className="font-mono font-medium">{currentVolValue.toFixed(1)}%</span>
                <span className="text-terminal-muted/60 text-xs">({volConfig.label})</span>
              </div>
            </div>
          </div>
        </div>

        {/* Center: Historical Performance */}
        {performance && (
          <div className="flex-1 min-w-[200px]">
            <div className="text-xs text-terminal-muted uppercase tracking-wider mb-2">
              Performance Histórica Neste Regime
            </div>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <div className="text-[10px] text-terminal-muted">Sharpe</div>
                <div className={`font-mono font-bold text-lg ${sharpe >= 0 ? 'text-profit' : 'text-loss'}`}>
                  {sharpe.toFixed(2)}
                </div>
              </div>
              <div>
                <div className="text-[10px] text-terminal-muted">Win Rate</div>
                <div className={`font-mono font-bold text-lg ${performance.win_rate_pct >= 50 ? 'text-profit' : 'text-loss'}`}>
                  {performance.win_rate_pct.toFixed(0)}%
                </div>
              </div>
              <div>
                <div className="text-[10px] text-terminal-muted">Dias</div>
                <div className="font-mono font-bold text-lg">
                  {performance.day_count}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Right: Allocation Recommendation */}
        <div className="min-w-[180px]">
          <div className="text-xs text-terminal-muted uppercase tracking-wider mb-2">
            Recomendação
          </div>
          <div className="flex items-center gap-3">
            {/* Gauge */}
            <div className="relative w-16 h-16">
              <svg className="w-16 h-16 -rotate-90" viewBox="0 0 36 36">
                <path
                  className="text-terminal-border"
                  strokeDasharray="100, 100"
                  d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="3"
                />
                <path
                  className={recommendation.color}
                  strokeDasharray={`${recommendation.allocation}, 100`}
                  d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="3"
                  strokeLinecap="round"
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className={`font-mono font-bold text-sm ${recommendation.color}`}>
                  {recommendation.allocation}%
                </span>
              </div>
            </div>
            
            {/* Label */}
            <div>
              <div className={`flex items-center gap-1.5 font-medium ${recommendation.color}`}>
                <RecommendationIcon className="w-4 h-4" />
                {recommendation.label}
              </div>
              <div className="text-[10px] text-terminal-muted mt-1">
                Confiança: {recommendation.confidence === 'high' ? 'Alta' : recommendation.confidence === 'medium' ? 'Média' : 'Baixa'}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

import { useMemo } from 'react';
import { 
  TrendingDown,
  Activity,
  Percent,
  BarChart3,
  Shield,
  AlertTriangle,
  Target,
  Gauge
} from 'lucide-react';

interface RiskMetrics {
  annualizedVol: number;
  var95: number;
  cvar95: number;
  maxDD: number;
  downsideVol: number;
  tailRatio: number;
  skewness: number;
  kurtosis: number;
}

interface Props {
  dailyReturns: number[];
  maxDrawdown: number;
  sharpe: number;
}

export function RiskDecomposition({ dailyReturns, maxDrawdown, sharpe }: Props) {
  const metrics = useMemo((): RiskMetrics => {
    if (dailyReturns.length < 2) {
      return {
        annualizedVol: 0, var95: 0, cvar95: 0, maxDD: maxDrawdown,
        downsideVol: 0, tailRatio: 0, skewness: 0, kurtosis: 0
      };
    }

    const n = dailyReturns.length;
    const mean = dailyReturns.reduce((a, b) => a + b, 0) / n;
    const variance = dailyReturns.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
    const std = Math.sqrt(variance);
    const annualizedVol = std * Math.sqrt(252);

    // Downside volatility
    const downsideReturns = dailyReturns.filter(r => r < 0);
    const downsideVariance = downsideReturns.length > 0
      ? downsideReturns.reduce((a, b) => a + b ** 2, 0) / downsideReturns.length
      : 0;
    const downsideVol = Math.sqrt(downsideVariance) * Math.sqrt(252);

    // VaR and CVaR
    const sorted = [...dailyReturns].sort((a, b) => a - b);
    const var95Index = Math.floor(n * 0.05);
    const var95 = sorted[var95Index] || 0;
    const tailReturns = sorted.filter(r => r <= var95);
    const cvar95 = tailReturns.length > 0
      ? tailReturns.reduce((a, b) => a + b, 0) / tailReturns.length
      : var95;

    // Tail ratio
    const p95 = sorted[Math.floor(n * 0.95)] || 0;
    const p5 = sorted[Math.floor(n * 0.05)] || 0;
    const tailRatio = p5 !== 0 ? Math.abs(p95 / p5) : 0;

    // Skewness and Kurtosis
    const skewness = std > 0
      ? dailyReturns.reduce((a, b) => a + ((b - mean) / std) ** 3, 0) / n
      : 0;
    const kurtosis = std > 0
      ? (dailyReturns.reduce((a, b) => a + ((b - mean) / std) ** 4, 0) / n) - 3
      : 0;

    return {
      annualizedVol,
      var95,
      cvar95,
      maxDD: maxDrawdown,
      downsideVol,
      tailRatio,
      skewness,
      kurtosis
    };
  }, [dailyReturns, maxDrawdown]);

  // Risk score (0-100)
  const riskScore = useMemo(() => {
    let score = 50; // Base
    
    // Volatility contribution
    if (metrics.annualizedVol < 0.10) score -= 10;
    else if (metrics.annualizedVol > 0.25) score += 15;
    else if (metrics.annualizedVol > 0.20) score += 10;
    
    // Max DD contribution
    const absDD = Math.abs(metrics.maxDD);
    if (absDD < 0.10) score -= 10;
    else if (absDD > 0.30) score += 20;
    else if (absDD > 0.20) score += 10;
    
    // Tail risk
    if (metrics.kurtosis > 3) score += 10;
    if (metrics.skewness < -0.5) score += 10;
    
    // Sharpe reduces risk
    if (sharpe > 1.5) score -= 15;
    else if (sharpe > 1.0) score -= 10;
    else if (sharpe < 0.5) score += 10;
    
    return Math.max(0, Math.min(100, score));
  }, [metrics, sharpe]);

  const riskLevel = riskScore < 30 ? 'Low' : riskScore < 60 ? 'Medium' : 'High';
  const riskColor = riskScore < 30 ? 'text-profit' : riskScore < 60 ? 'text-accent-yellow' : 'text-loss';

  return (
    <div className="space-y-6">
      {/* Risk Score Gauge */}
      <div className="flex items-center gap-8 p-6 rounded-xl bg-terminal-surface border border-terminal-border">
        <div className="relative w-32 h-32">
          <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
            {/* Background arc */}
            <circle
              cx="50" cy="50" r="40"
              fill="none"
              stroke="currentColor"
              strokeWidth="8"
              className="text-terminal-border"
              strokeDasharray={251.2}
              strokeLinecap="round"
            />
            {/* Score arc */}
            <circle
              cx="50" cy="50" r="40"
              fill="none"
              stroke="currentColor"
              strokeWidth="8"
              className={riskColor}
              strokeDasharray={251.2}
              strokeDashoffset={251.2 - (251.2 * riskScore / 100)}
              strokeLinecap="round"
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className={`text-3xl font-bold font-mono ${riskColor}`}>{riskScore}</span>
            <span className="text-xs text-terminal-muted">/ 100</span>
          </div>
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            <Gauge className={`w-5 h-5 ${riskColor}`} />
            <h3 className="font-semibold text-lg">Risk Assessment</h3>
          </div>
          <div className={`text-2xl font-bold ${riskColor} mb-2`}>{riskLevel} Risk</div>
          <p className="text-sm text-terminal-muted">
            Composite score based on volatility, drawdown, tail risk, and risk-adjusted returns.
          </p>
        </div>
      </div>

      {/* Risk Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <RiskCard
          icon={<Activity className="w-4 h-4" />}
          label="Annual Volatility"
          value={`${(metrics.annualizedVol * 100).toFixed(2)}%`}
          description="Standard deviation of returns, annualized"
          quality={metrics.annualizedVol < 0.15 ? 'good' : metrics.annualizedVol < 0.25 ? 'neutral' : 'bad'}
        />
        <RiskCard
          icon={<TrendingDown className="w-4 h-4" />}
          label="Downside Vol"
          value={`${(metrics.downsideVol * 100).toFixed(2)}%`}
          description="Volatility of negative returns only"
          quality={metrics.downsideVol < 0.12 ? 'good' : metrics.downsideVol < 0.20 ? 'neutral' : 'bad'}
        />
        <RiskCard
          icon={<Shield className="w-4 h-4" />}
          label="VaR (95%)"
          value={`${(metrics.var95 * 100).toFixed(3)}%`}
          description="5th percentile daily loss"
          quality={metrics.var95 > -0.02 ? 'good' : metrics.var95 > -0.03 ? 'neutral' : 'bad'}
        />
        <RiskCard
          icon={<AlertTriangle className="w-4 h-4" />}
          label="CVaR (95%)"
          value={`${(metrics.cvar95 * 100).toFixed(3)}%`}
          description="Expected loss beyond VaR"
          quality={metrics.cvar95 > -0.03 ? 'good' : metrics.cvar95 > -0.05 ? 'neutral' : 'bad'}
        />
      </div>

      {/* Distribution Analysis */}
      <div className="grid grid-cols-2 gap-4">
        <div className="p-4 rounded-xl bg-terminal-surface border border-terminal-border">
          <h4 className="text-sm font-medium text-terminal-muted mb-4 flex items-center gap-2">
            <BarChart3 className="w-4 h-4" />
            Distribution Shape
          </h4>
          <div className="space-y-4">
            <DistributionMetric
              label="Skewness"
              value={metrics.skewness}
              description={
                metrics.skewness > 0.5 ? 'Right-skewed (positive tail)' :
                metrics.skewness < -0.5 ? 'Left-skewed (negative tail)' :
                'Approximately symmetric'
              }
              isGood={metrics.skewness >= 0}
            />
            <DistributionMetric
              label="Excess Kurtosis"
              value={metrics.kurtosis}
              description={
                metrics.kurtosis > 3 ? 'Very fat tails (high tail risk)' :
                metrics.kurtosis > 0 ? 'Fat tails (leptokurtic)' :
                'Thin tails (platykurtic)'
              }
              isGood={metrics.kurtosis < 1}
            />
          </div>
        </div>

        <div className="p-4 rounded-xl bg-terminal-surface border border-terminal-border">
          <h4 className="text-sm font-medium text-terminal-muted mb-4 flex items-center gap-2">
            <Target className="w-4 h-4" />
            Tail Risk Analysis
          </h4>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-terminal-muted">Tail Ratio (P95/P5)</span>
              <span className={`font-mono font-bold ${metrics.tailRatio >= 1 ? 'text-profit' : 'text-loss'}`}>
                {metrics.tailRatio.toFixed(3)}
              </span>
            </div>
            <div className="h-2 bg-terminal-bg rounded-full overflow-hidden">
              <div 
                className={`h-full rounded-full ${metrics.tailRatio >= 1 ? 'bg-profit' : 'bg-loss'}`}
                style={{ width: `${Math.min(100, metrics.tailRatio * 50)}%` }}
              />
            </div>
            <p className="text-xs text-terminal-muted">
              {metrics.tailRatio >= 1.5 
                ? 'Excellent: Gains significantly larger than losses'
                : metrics.tailRatio >= 1.0
                ? 'Good: Gains larger than losses'
                : 'Poor: Losses larger than gains'}
            </p>
          </div>

          <div className="mt-4 pt-4 border-t border-terminal-border">
            <div className="flex items-center justify-between">
              <span className="text-sm text-terminal-muted">Max Drawdown</span>
              <span className="font-mono font-bold text-loss">
                {(metrics.maxDD * 100).toFixed(2)}%
              </span>
            </div>
            <div className="mt-2 h-2 bg-terminal-bg rounded-full overflow-hidden">
              <div 
                className="h-full rounded-full bg-loss"
                style={{ width: `${Math.min(100, Math.abs(metrics.maxDD) * 200)}%` }}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function RiskCard({ 
  icon, 
  label, 
  value, 
  description, 
  quality 
}: { 
  icon: React.ReactNode;
  label: string; 
  value: string; 
  description: string;
  quality: 'good' | 'neutral' | 'bad';
}) {
  const colors = {
    good: 'border-profit/30 bg-profit/5',
    neutral: 'border-accent-yellow/30 bg-accent-yellow/5',
    bad: 'border-loss/30 bg-loss/5'
  };

  const textColors = {
    good: 'text-profit',
    neutral: 'text-accent-yellow',
    bad: 'text-loss'
  };

  return (
    <div className={`p-4 rounded-xl border ${colors[quality]}`}>
      <div className="flex items-center gap-2 mb-2">
        <span className={textColors[quality]}>{icon}</span>
        <span className="text-xs text-terminal-muted uppercase tracking-wider">{label}</span>
      </div>
      <div className={`text-xl font-bold font-mono ${textColors[quality]}`}>{value}</div>
      <p className="text-[10px] text-terminal-muted mt-1">{description}</p>
    </div>
  );
}

function DistributionMetric({ 
  label, 
  value, 
  description, 
  isGood 
}: { 
  label: string; 
  value: number; 
  description: string;
  isGood: boolean;
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm text-terminal-muted">{label}</span>
        <span className={`font-mono font-bold ${isGood ? 'text-profit' : 'text-loss'}`}>
          {value.toFixed(3)}
        </span>
      </div>
      <p className="text-[10px] text-terminal-muted">{description}</p>
    </div>
  );
}


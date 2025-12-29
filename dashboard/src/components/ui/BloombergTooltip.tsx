/**
 * Bloomberg-Style Tooltip Component
 * 
 * Rich tooltips with multiple data rows, sparklines, and professional styling.
 */

import { ReactNode, useState } from 'react';
import { Info } from 'lucide-react';
import { Sparkline } from '../charts/Sparkline';

interface TooltipRowProps {
  label: string;
  value: string | number;
  sublabel?: string;
  color?: 'profit' | 'loss' | 'warning' | 'default';
}

interface BloombergTooltipProps {
  title: string;
  description?: string;
  rows?: TooltipRowProps[];
  sparkData?: number[];
  formula?: string;
  benchmark?: string;
  interpretation?: string;
  children?: ReactNode;
  position?: 'top' | 'bottom' | 'left' | 'right';
}

export function BloombergTooltip({
  title,
  description,
  rows = [],
  sparkData,
  formula,
  benchmark,
  interpretation,
  children,
  position = 'top'
}: BloombergTooltipProps) {
  const [isVisible, setIsVisible] = useState(false);

  const positionClasses = {
    top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
    left: 'right-full top-1/2 -translate-y-1/2 mr-2',
    right: 'left-full top-1/2 -translate-y-1/2 ml-2'
  };

  const arrowClasses = {
    top: 'top-full left-1/2 -translate-x-1/2 border-t-terminal-surface',
    bottom: 'bottom-full left-1/2 -translate-x-1/2 border-b-terminal-surface',
    left: 'left-full top-1/2 -translate-y-1/2 border-l-terminal-surface',
    right: 'right-full top-1/2 -translate-y-1/2 border-r-terminal-surface'
  };

  const colorMap = {
    profit: 'text-profit',
    loss: 'text-loss',
    warning: 'text-accent-yellow',
    default: 'text-white'
  };

  return (
    <div 
      className="relative inline-flex"
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
    >
      {children || (
        <button className="p-0.5 rounded hover:bg-terminal-surface/50 transition-colors">
          <Info className="w-3.5 h-3.5 text-terminal-muted hover:text-accent-cyan transition-colors" />
        </button>
      )}
      
      {isVisible && (
        <div 
          className={`absolute z-50 ${positionClasses[position]} min-w-[240px] max-w-[320px]`}
        >
          {/* Tooltip Card */}
          <div className="bg-terminal-surface border border-terminal-border rounded-lg shadow-2xl shadow-black/50 overflow-hidden">
            {/* Header */}
            <div className="px-3 py-2 bg-gradient-to-r from-terminal-bg to-terminal-surface border-b border-terminal-border">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-semibold text-white">{title}</h4>
                {sparkData && sparkData.length > 0 && (
                  <Sparkline data={sparkData} width={50} height={18} />
                )}
              </div>
              {description && (
                <p className="text-[10px] text-terminal-muted mt-0.5">{description}</p>
              )}
            </div>
            
            {/* Content */}
            <div className="px-3 py-2 space-y-2">
              {/* Formula */}
              {formula && (
                <div className="p-2 bg-terminal-bg/50 rounded border border-terminal-border/50">
                  <div className="text-[10px] text-terminal-muted uppercase tracking-wider mb-1">Formula</div>
                  <code className="text-xs font-mono text-accent-cyan">{formula}</code>
                </div>
              )}
              
              {/* Data Rows */}
              {rows.length > 0 && (
                <div className="space-y-1">
                  {rows.map((row, i) => (
                    <div key={i} className="flex items-center justify-between text-xs">
                      <span className="text-terminal-muted">
                        {row.label}
                        {row.sublabel && (
                          <span className="text-[10px] ml-1">({row.sublabel})</span>
                        )}
                      </span>
                      <span className={`font-mono font-medium ${colorMap[row.color || 'default']}`}>
                        {row.value}
                      </span>
                    </div>
                  ))}
                </div>
              )}
              
              {/* Benchmark */}
              {benchmark && (
                <div className="flex items-center justify-between text-xs pt-1 border-t border-terminal-border/30">
                  <span className="text-terminal-muted">Benchmark</span>
                  <span className="font-mono text-accent-yellow">{benchmark}</span>
                </div>
              )}
              
              {/* Interpretation */}
              {interpretation && (
                <div className="pt-2 border-t border-terminal-border/30">
                  <div className="text-[10px] leading-relaxed text-terminal-muted">
                    💡 {interpretation}
                  </div>
                </div>
              )}
            </div>
          </div>
          
          {/* Arrow */}
          <div 
            className={`absolute w-0 h-0 border-4 border-transparent ${arrowClasses[position]}`} 
          />
        </div>
      )}
    </div>
  );
}

/**
 * MetricTooltip - Predefined tooltips for common financial metrics
 */
export const MetricTooltips = {
  sharpe: (value: number) => ({
    title: 'Sharpe Ratio',
    description: 'Risk-adjusted return metric',
    formula: '(Rp - Rf) / σp',
    rows: [
      { label: 'Current', value: value.toFixed(3), color: value >= 1.0 ? 'profit' as const : value >= 0.5 ? 'warning' as const : 'loss' as const },
      { label: 'Annualized', value: 'Yes', sublabel: '252 days' },
    ],
    benchmark: '≥ 1.0 (good), ≥ 2.0 (excellent)',
    interpretation: 'Higher values indicate better risk-adjusted returns. >1 is good, >2 is excellent.'
  }),
  
  sortino: (value: number) => ({
    title: 'Sortino Ratio',
    description: 'Downside risk-adjusted return',
    formula: '(Rp - Rf) / σd',
    rows: [
      { label: 'Current', value: value.toFixed(3), color: value >= 1.5 ? 'profit' as const : value >= 1.0 ? 'warning' as const : 'loss' as const },
      { label: 'Uses', value: 'Downside deviation only' },
    ],
    benchmark: '≥ 1.5 (good), ≥ 2.0 (excellent)',
    interpretation: 'Better than Sharpe for asymmetric return distributions. Ignores upside volatility.'
  }),
  
  calmar: (value: number) => ({
    title: 'Calmar Ratio',
    description: 'Return vs maximum drawdown',
    formula: 'CAGR / Max Drawdown',
    rows: [
      { label: 'Current', value: value.toFixed(3), color: value >= 1.0 ? 'profit' as const : value >= 0.5 ? 'warning' as const : 'loss' as const },
    ],
    benchmark: '≥ 1.0 (good), ≥ 3.0 (excellent)',
    interpretation: 'Higher values mean better reward relative to worst historical loss.'
  }),
  
  omega: (value: number) => ({
    title: 'Omega Ratio',
    description: 'Probability-weighted gains/losses',
    formula: '∫ gains / ∫ losses (above/below threshold)',
    rows: [
      { label: 'Current', value: value === Infinity ? '∞' : value.toFixed(3), color: value >= 1.5 ? 'profit' as const : 'warning' as const },
      { label: 'Threshold', value: '0%', sublabel: 'risk-free' },
    ],
    benchmark: '≥ 1.5 (good), ≥ 2.0 (excellent)',
    interpretation: '>1 means more wins than losses. Captures entire return distribution.'
  }),
  
  var95: (value: number) => ({
    title: 'Value at Risk (95%)',
    description: 'Maximum expected daily loss at 95% confidence',
    formula: '5th percentile of daily returns',
    rows: [
      { label: 'Daily VaR', value: `${(value * 100).toFixed(2)}%`, color: 'loss' as const },
      { label: 'Confidence', value: '95%' },
    ],
    interpretation: 'On 95% of days, losses will not exceed this amount. Does not capture tail risk.'
  }),
  
  cvar95: (value: number) => ({
    title: 'Conditional VaR (95%)',
    description: 'Expected loss in the worst 5% of cases',
    formula: 'E[loss | loss > VaR95]',
    rows: [
      { label: 'Daily CVaR', value: `${(value * 100).toFixed(2)}%`, color: 'loss' as const },
      { label: 'Also known as', value: 'Expected Shortfall' },
    ],
    interpretation: 'Average of the worst losses. Better captures tail risk than VaR.'
  }),
  
  pbo: (value: number) => ({
    title: 'Probability of Backtest Overfitting',
    description: 'Likelihood that OOS performance will be worse than IS',
    formula: 'Based on CPCV degradation distribution',
    rows: [
      { label: 'Current', value: `${(value * 100).toFixed(1)}%`, color: value <= 0.15 ? 'profit' as const : value <= 0.30 ? 'warning' as const : 'loss' as const },
      { label: 'Status', value: value <= 0.15 ? 'PASS' : 'FAIL' },
    ],
    benchmark: '≤ 15% (validated)',
    interpretation: 'Lower is better. Above 15% suggests strategy may be overfit.'
  }),
  
  dsr: (value: number) => ({
    title: 'Deflated Sharpe Ratio',
    description: 'Sharpe adjusted for multiple testing',
    formula: 'SR × √(1 - Var(SR) / SR²)',
    rows: [
      { label: 'Current', value: value.toFixed(3), color: value >= 0.5 ? 'profit' as const : 'warning' as const },
      { label: 'Trials', value: 'Adjusted' },
    ],
    benchmark: '≥ 0.5 (validated)',
    interpretation: 'Accounts for luck in backtesting many strategies. Use instead of raw Sharpe.'
  }),
  
  tstat: (value: number) => ({
    title: 'T-Statistic',
    description: 'Statistical significance of Sharpe ratio',
    formula: 'SR × √(n / 252)',
    rows: [
      { label: 'Current', value: value.toFixed(3), color: Math.abs(value) >= 2.0 ? 'profit' as const : 'warning' as const },
      { label: 'Threshold', value: '≥ 2.0 for 95% CI' },
    ],
    interpretation: 't ≥ 2.0 means <5% chance Sharpe is due to luck alone.'
  }),
  
  pvalue: (value: number) => ({
    title: 'P-Value',
    description: 'Probability returns are due to chance',
    formula: '2 × (1 - Φ(|t-stat|))',
    rows: [
      { label: 'Current', value: value < 0.001 ? '<0.001' : value.toFixed(4), color: value <= 0.05 ? 'profit' as const : 'loss' as const },
      { label: 'Status', value: value <= 0.05 ? 'Significant' : 'Not significant' },
    ],
    benchmark: '≤ 0.05 (95% confidence)',
    interpretation: 'Lower is better. <0.05 means statistically significant.'
  }),
  
  skewness: (value: number) => ({
    title: 'Skewness',
    description: 'Asymmetry of return distribution',
    formula: 'E[(X - μ)³] / σ³',
    rows: [
      { label: 'Current', value: value.toFixed(3), color: value >= 0 ? 'profit' as const : 'warning' as const },
      { label: 'Distribution', value: value > 0 ? 'Right-skewed' : value < 0 ? 'Left-skewed' : 'Symmetric' },
    ],
    interpretation: 'Positive skew = more extreme gains. Negative skew = more extreme losses.'
  }),
  
  kurtosis: (value: number) => ({
    title: 'Excess Kurtosis',
    description: 'Tail thickness vs normal distribution',
    formula: 'E[(X - μ)⁴] / σ⁴ - 3',
    rows: [
      { label: 'Current', value: value.toFixed(3) },
      { label: 'Tails', value: value > 0 ? 'Fat tails' : value < 0 ? 'Thin tails' : 'Normal' },
    ],
    interpretation: 'Positive = more extreme events than normal. Common in finance.'
  }),
};


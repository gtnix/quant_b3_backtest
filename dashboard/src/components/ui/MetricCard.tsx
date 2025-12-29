import { ReactNode } from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface MetricCardProps {
  label: string;
  value: string | number;
  change?: number;
  changeLabel?: string;
  icon?: ReactNode;
  format?: 'percent' | 'currency' | 'number' | 'ratio';
  size?: 'sm' | 'md' | 'lg';
}

export function MetricCard({ 
  label, 
  value, 
  change, 
  changeLabel,
  icon,
  format = 'number',
  size = 'md'
}: MetricCardProps) {
  const formatValue = (val: string | number): string => {
    if (typeof val === 'string') return val;
    
    switch (format) {
      case 'percent':
        return `${(val * 100).toFixed(2)}%`;
      case 'currency':
        return new Intl.NumberFormat('en-US', {
          style: 'currency',
          currency: 'USD',
          minimumFractionDigits: 0,
          maximumFractionDigits: 0,
        }).format(val);
      case 'ratio':
        return val.toFixed(3);
      default:
        return val.toLocaleString();
    }
  };

  const getChangeColor = () => {
    if (change === undefined || change === 0) return 'text-terminal-muted';
    return change > 0 ? 'text-profit' : 'text-loss';
  };

  const getChangeIcon = () => {
    if (change === undefined || change === 0) {
      return <Minus className="w-3 h-3" />;
    }
    return change > 0 
      ? <TrendingUp className="w-3 h-3" /> 
      : <TrendingDown className="w-3 h-3" />;
  };

  const sizeClasses = {
    sm: 'text-lg',
    md: 'text-2xl',
    lg: 'text-4xl',
  };

  return (
    <div className="card group hover:border-terminal-muted/50 transition-colors">
      <div className="flex items-start justify-between mb-2">
        <span className="metric-label">{label}</span>
        {icon && <div className="text-terminal-muted">{icon}</div>}
      </div>
      
      <div className={`font-mono font-bold ${sizeClasses[size]} tracking-tight`}>
        {formatValue(value)}
      </div>
      
      {change !== undefined && (
        <div className={`flex items-center gap-1 mt-2 text-xs ${getChangeColor()}`}>
          {getChangeIcon()}
          <span>{change > 0 ? '+' : ''}{(change * 100).toFixed(2)}%</span>
          {changeLabel && <span className="text-terminal-muted ml-1">{changeLabel}</span>}
        </div>
      )}
    </div>
  );
}



/**
 * Sparkline Component - Bloomberg Terminal Style
 * 
 * Ultra-compact inline chart for metric cards.
 * Shows trend at a glance without taking up space.
 */

import { useMemo } from 'react';

interface SparklineProps {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
  fillColor?: string;
  showLastPoint?: boolean;
  showZeroLine?: boolean;
  className?: string;
}

export function Sparkline({
  data,
  width = 80,
  height = 24,
  color = '#00ff88',
  fillColor,
  showLastPoint = true,
  showZeroLine = false,
  className = ''
}: SparklineProps) {
  const pathData = useMemo(() => {
    // Filter out invalid values
    const validData = data.filter(v => typeof v === 'number' && !isNaN(v) && isFinite(v));
    if (validData.length < 2) return { line: '', area: '', range: { min: 0, max: 0 }, points: [], validData: [] };
    
    const min = Math.min(...validData);
    const max = Math.max(...validData);
    const range = max - min || 1;
    
    // Add padding for the point
    const padding = 2;
    const effectiveWidth = width - padding * 2;
    const effectiveHeight = height - padding * 2;
    
    const xStep = effectiveWidth / (validData.length - 1);
    
    const points = validData.map((value, i) => {
      const x = padding + i * xStep;
      const y = padding + effectiveHeight - ((value - min) / range) * effectiveHeight;
      return { x, y };
    });
    
    // Create SVG path
    const line = points.map((p, i) => 
      `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`
    ).join(' ');
    
    // Create area path for gradient fill
    const area = `${line} L ${points[points.length - 1].x.toFixed(1)} ${height} L ${points[0].x.toFixed(1)} ${height} Z`;
    
    return { line, area, points, range: { min, max }, validData };
  }, [data, width, height]);

  // Check if we have valid data
  const validData = pathData.validData || [];
  if (validData.length < 2) {
    return (
      <div 
        className={`flex items-center justify-center text-terminal-muted text-[10px] ${className}`}
        style={{ width, height }}
      >
        —
      </div>
    );
  }

  const lastPoint = pathData.points?.[pathData.points.length - 1];
  const firstValue = validData[0];
  const lastValue = validData[validData.length - 1];
  const trend = lastValue >= firstValue ? 'up' : 'down';
  const trendColor = trend === 'up' ? '#00ff88' : '#ef4444';
  const actualColor = color || trendColor;

  return (
    <svg 
      width={width} 
      height={height} 
      className={`overflow-visible ${className}`}
      viewBox={`0 0 ${width} ${height}`}
    >
      {/* Gradient definition */}
      <defs>
        <linearGradient id={`sparkline-gradient-${data.length}`} x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor={actualColor} stopOpacity="0.3" />
          <stop offset="100%" stopColor={actualColor} stopOpacity="0" />
        </linearGradient>
      </defs>
      
      {/* Zero line */}
      {showZeroLine && (
        <line
          x1="0"
          y1={height / 2}
          x2={width}
          y2={height / 2}
          stroke="#3f3f46"
          strokeWidth="0.5"
          strokeDasharray="2,2"
        />
      )}
      
      {/* Area fill */}
      {fillColor !== 'none' && (
        <path
          d={pathData.area}
          fill={`url(#sparkline-gradient-${data.length})`}
        />
      )}
      
      {/* Line */}
      <path
        d={pathData.line}
        fill="none"
        stroke={actualColor}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      
      {/* Last point dot */}
      {showLastPoint && lastPoint && (
        <>
          <circle
            cx={lastPoint.x}
            cy={lastPoint.y}
            r="3"
            fill={actualColor}
            className="animate-pulse"
          />
          <circle
            cx={lastPoint.x}
            cy={lastPoint.y}
            r="5"
            fill={actualColor}
            opacity="0.3"
          />
        </>
      )}
    </svg>
  );
}

/**
 * SparkBar - Mini bar chart for distribution visualization
 */
interface SparkBarProps {
  data: number[];
  width?: number;
  height?: number;
  positiveColor?: string;
  negativeColor?: string;
  className?: string;
}

export function SparkBar({
  data,
  width = 60,
  height = 20,
  positiveColor = '#00ff88',
  negativeColor = '#ef4444',
  className = ''
}: SparkBarProps) {
  // Filter out invalid values
  const validData = data.filter(v => typeof v === 'number' && !isNaN(v) && isFinite(v));
  
  if (validData.length === 0) {
    return (
      <svg width={width} height={height} className={className}>
        <line x1="0" y1={height / 2} x2={width} y2={height / 2} stroke="#3f3f46" strokeWidth="0.5" />
        <text x={width / 2} y={height / 2 + 3} textAnchor="middle" fill="#71717a" fontSize="8">—</text>
      </svg>
    );
  }

  const max = Math.max(...validData.map(Math.abs));
  if (max === 0) {
    return (
      <svg width={width} height={height} className={className}>
        <line x1="0" y1={height / 2} x2={width} y2={height / 2} stroke="#3f3f46" strokeWidth="0.5" />
      </svg>
    );
  }
  
  const barWidth = Math.max(1, (width - validData.length + 1) / validData.length);

  return (
    <svg width={width} height={height} className={className}>
      {/* Center line */}
      <line
        x1="0"
        y1={height / 2}
        x2={width}
        y2={height / 2}
        stroke="#3f3f46"
        strokeWidth="0.5"
      />
      
      {validData.map((value, i) => {
        const barHeight = Math.max(0.5, (Math.abs(value) / max) * (height / 2 - 1));
        const x = i * (barWidth + 1);
        const y = value >= 0 ? height / 2 - barHeight : height / 2;
        const color = value >= 0 ? positiveColor : negativeColor;
        
        return (
          <rect
            key={i}
            x={x}
            y={y}
            width={barWidth}
            height={barHeight}
            fill={color}
            rx="1"
          />
        );
      })}
    </svg>
  );
}

/**
 * TrendIndicator - Shows trend direction with arrow
 */
interface TrendIndicatorProps {
  current: number;
  previous: number;
  size?: 'sm' | 'md' | 'lg';
  showPercent?: boolean;
}

export function TrendIndicator({ 
  current, 
  previous, 
  size = 'sm',
  showPercent = true 
}: TrendIndicatorProps) {
  const change = previous !== 0 ? ((current - previous) / Math.abs(previous)) * 100 : 0;
  const isUp = change >= 0;
  
  const sizeClasses = {
    sm: 'text-[10px]',
    md: 'text-xs',
    lg: 'text-sm'
  };
  
  const arrowSizes = {
    sm: 'w-2.5 h-2.5',
    md: 'w-3 h-3',
    lg: 'w-4 h-4'
  };

  return (
    <span className={`inline-flex items-center gap-0.5 ${sizeClasses[size]} ${isUp ? 'text-profit' : 'text-loss'}`}>
      <svg 
        className={arrowSizes[size]} 
        viewBox="0 0 12 12" 
        fill="currentColor"
        style={{ transform: isUp ? 'rotate(0deg)' : 'rotate(180deg)' }}
      >
        <path d="M6 2L10 7H2L6 2Z" />
      </svg>
      {showPercent && (
        <span className="font-mono">
          {Math.abs(change).toFixed(1)}%
        </span>
      )}
    </span>
  );
}


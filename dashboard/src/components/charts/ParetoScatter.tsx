/**
 * ParetoScatter - 2D Scatter chart for Sharpe vs MaxDD visualization
 * Simpler alternative to the 3D ParetoChart for better performance
 */

import { useMemo, useState } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface DataPoint {
  id: string;
  sharpe: number;
  maxDrawdown: number;
  cagr?: number;
  gatesPassed?: boolean;
  displayName?: string;
}

interface ParetoScatterProps {
  data: DataPoint[];
  onPointClick?: (id: string) => void;
  height?: number;
}

export function ParetoScatter({ data, onPointClick, height = 300 }: ParetoScatterProps) {
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  
  const chartData = useMemo(() => {
    return data.map(d => ({
      id: d.id,
      x: Math.abs(d.maxDrawdown) * 100, // Convert to percentage
      y: d.sharpe,
      cagr: d.cagr,
      gatesPassed: d.gatesPassed,
      displayName: d.displayName || d.id.slice(0, 12),
    }));
  }, [data]);
  
  // Find Pareto frontier
  const paretoFrontier = useMemo(() => {
    if (chartData.length === 0) return new Set<string>();
    
    const frontier = new Set<string>();
    
    for (const point of chartData) {
      let isDominated = false;
      
      for (const other of chartData) {
        if (other.id === point.id) continue;
        
        // Another point dominates if it has higher Sharpe AND lower MaxDD
        if (other.y >= point.y && other.x <= point.x && (other.y > point.y || other.x < point.x)) {
          isDominated = true;
          break;
        }
      }
      
      if (!isDominated) {
        frontier.add(point.id);
      }
    }
    
    return frontier;
  }, [chartData]);
  
  const getPointColor = (point: { id: string; gatesPassed?: boolean }) => {
    if (paretoFrontier.has(point.id)) return '#10b981'; // Emerald for Pareto
    if (point.gatesPassed) return '#3b82f6'; // Blue for validated
    return '#64748b'; // Slate for others
  };
  
  const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: Array<{ payload: typeof chartData[0] }> }) => {
    if (!active || !payload || payload.length === 0) return null;
    
    const point = payload[0].payload;
    const isPareto = paretoFrontier.has(point.id);
    
    return (
      <div className="bg-slate-900 border border-slate-700 rounded-lg p-3 shadow-xl">
        <div className="flex items-center gap-2 mb-2">
          {isPareto && (
            <span className="px-1.5 py-0.5 text-[10px] bg-emerald-500/20 text-emerald-400 rounded uppercase">
              Pareto
            </span>
          )}
          <span className="font-mono text-sm text-white">{point.displayName}</span>
        </div>
        <div className="space-y-1 text-xs">
          <div className="flex justify-between gap-4">
            <span className="text-slate-400">Sharpe:</span>
            <span className="font-mono text-emerald-400">{point.y.toFixed(3)}</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-slate-400">Max DD:</span>
            <span className="font-mono text-rose-400">{point.x.toFixed(1)}%</span>
          </div>
          {point.cagr !== undefined && (
            <div className="flex justify-between gap-4">
              <span className="text-slate-400">CAGR:</span>
              <span className="font-mono text-white">{((point.cagr || 0) * 100).toFixed(1)}%</span>
            </div>
          )}
        </div>
      </div>
    );
  };
  
  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-slate-500 text-sm">
        No data to display
      </div>
    );
  }
  
  return (
    <div style={{ width: '100%', height }}>
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 20, right: 20, bottom: 40, left: 40 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis 
            type="number" 
            dataKey="x" 
            name="Max Drawdown" 
            unit="%" 
            stroke="#64748b"
            tick={{ fill: '#64748b', fontSize: 10 }}
            label={{ value: 'Max Drawdown (%)', position: 'bottom', fill: '#64748b', fontSize: 11 }}
          />
          <YAxis 
            type="number" 
            dataKey="y" 
            name="Sharpe" 
            stroke="#64748b"
            tick={{ fill: '#64748b', fontSize: 10 }}
            label={{ value: 'Sharpe Ratio', angle: -90, position: 'left', fill: '#64748b', fontSize: 11 }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Scatter 
            data={chartData} 
            onClick={(point) => onPointClick?.(point.id)}
            onMouseEnter={(point) => setHoveredId(point.id)}
            onMouseLeave={() => setHoveredId(null)}
          >
            {chartData.map((entry) => (
              <Cell 
                key={entry.id}
                fill={getPointColor(entry)}
                opacity={hoveredId && hoveredId !== entry.id ? 0.3 : 1}
                style={{ cursor: onPointClick ? 'pointer' : 'default' }}
              />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
      
      {/* Legend */}
      <div className="flex items-center justify-center gap-6 mt-2 text-xs">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full bg-emerald-500" />
          <span className="text-slate-400">Pareto Frontier ({paretoFrontier.size})</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full bg-blue-500" />
          <span className="text-slate-400">Validated</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full bg-slate-500" />
          <span className="text-slate-400">Research</span>
        </div>
      </div>
    </div>
  );
}

export default ParetoScatter;


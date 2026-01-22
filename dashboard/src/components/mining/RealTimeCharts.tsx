/**
 * RealTimeCharts - Academic-Grade Quant Mining Visualization
 * 
 * Based on papers: ParetoTracker 2024, He & Lin 2015, Gabor 2018
 * Key metrics for evolutionary optimization monitoring:
 * - Convergence: Best + Mean Sharpe evolution
 * - Diversity: Population spread indicator  
 * - Throughput: Evaluations per minute
 * - Resources: CPU/RAM utilization
 */

import { useMemo } from 'react';
import {
  AreaChart, Area, LineChart, Line,
  XAxis, YAxis, ResponsiveContainer, Tooltip
} from 'recharts';

interface RealTimeChartsProps {
  throughputHistory: number[];
  cpuHistory: number[];
  memoryHistory: number[];
  sharpeHistory: number[];
  generationHistory: number[];
  candidatesHistory: number[];
  // New evolution metrics
  diversityHistory?: number[];
  meanSharpeHistory?: number[];
  paretoSizeHistory?: number[];
  convergenceRateHistory?: number[];
}

function QuantTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-slate-900/95 border border-slate-600 rounded-lg px-3 py-2 shadow-xl space-y-1">
      {payload.map((p: any, i: number) => (
        <div key={i} className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full" style={{ backgroundColor: p.stroke || p.color }} />
          <span className="text-white font-mono text-sm">
            {typeof p.value === 'number' ? p.value.toFixed(3) : p.value}
          </span>
          <span className="text-slate-400 text-xs">{p.name}</span>
        </div>
      ))}
    </div>
  );
}

export function RealTimeCharts({
  throughputHistory,
  cpuHistory,
  memoryHistory,
  sharpeHistory,
  candidatesHistory,
  diversityHistory = [],
  meanSharpeHistory = [],
  paretoSizeHistory = [],
}: RealTimeChartsProps) {
  
  // Convergence chart: Best Sharpe + Mean Sharpe + Diversity
  const convergenceData = useMemo(() => {
    let best = 0;
    return sharpeHistory.map((s, i) => {
      if (s > best) best = s;
      return { 
        t: i, 
        best, 
        mean: meanSharpeHistory[i] || 0,
        diversity: (diversityHistory[i] || 0) * 10, // Scale for visibility
      };
    });
  }, [sharpeHistory, meanSharpeHistory, diversityHistory]);
  
  // Candidates evaluated over time
  const candidatesData = useMemo(() => 
    candidatesHistory.map((candidates, i) => ({
      t: i,
      candidates,
    })),
    [candidatesHistory]
  );
  
  // Throughput over time
  const throughputData = useMemo(() => 
    throughputHistory.map((tp, i) => ({
      t: i,
      throughput: tp,
      pareto: paretoSizeHistory[i] || 0,
    })),
    [throughputHistory, paretoSizeHistory]
  );
  
  // Resources utilization
  const resourceData = useMemo(() => 
    cpuHistory.map((cpu, i) => ({ 
      t: i, 
      cpu, 
      mem: memoryHistory[i] || 0 
    })),
    [cpuHistory, memoryHistory]
  );

  const lastCandidates = candidatesHistory[candidatesHistory.length - 1] || 0;
  const lastThroughput = throughputHistory[throughputHistory.length - 1] || 0;
  const lastCpu = cpuHistory[cpuHistory.length - 1] || 0;
  const lastMem = memoryHistory[memoryHistory.length - 1] || 0;
  const lastBest = convergenceData[convergenceData.length - 1]?.best || 0;
  const lastMean = convergenceData[convergenceData.length - 1]?.mean || 0;
  const lastDiversity = diversityHistory[diversityHistory.length - 1] || 0;
  const lastPareto = paretoSizeHistory[paretoSizeHistory.length - 1] || 0;

  return (
    <div className="space-y-4">
      {/* Main Chart - Convergence (Best + Mean Sharpe + Diversity) */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 border border-emerald-500/40 shadow-[0_0_40px_rgba(16,185,129,0.12)]">
        <div className="absolute inset-0 bg-gradient-to-t from-emerald-500/5 via-transparent to-transparent" />
        
        <div className="relative px-5 pt-4 pb-2 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-3 h-3 rounded-full bg-emerald-400 animate-pulse shadow-[0_0_12px_rgba(16,185,129,0.9)]" />
            <span className="text-base font-bold text-white tracking-wide">CONVERGENCE</span>
          </div>
          <div className="flex items-center gap-6 text-sm font-mono">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-emerald-400" />
              <span className="text-emerald-300">Best: {lastBest.toFixed(3)}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-blue-400" />
              <span className="text-blue-300">Mean: {lastMean.toFixed(3)}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-purple-400" />
              <span className="text-purple-300">Div: {lastDiversity.toFixed(3)}</span>
            </div>
          </div>
        </div>
        
        <div className="relative px-2 pb-3">
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={convergenceData} margin={{ top: 5, right: 15, left: 15, bottom: 0 }}>
              <XAxis dataKey="t" hide />
              <YAxis hide domain={['auto', 'auto']} />
              <Tooltip content={<QuantTooltip />} />
              <Line 
                type="stepAfter" 
                dataKey="best"
                name="Best Sharpe"
                stroke="#10b981" 
                strokeWidth={3}
                dot={false}
                isAnimationActive={true}
              />
              <Line 
                type="monotone" 
                dataKey="mean"
                name="Mean Sharpe"
                stroke="#3b82f6" 
                strokeWidth={2}
                dot={false}
                isAnimationActive={true}
                strokeDasharray="4 2"
              />
              <Line 
                type="monotone" 
                dataKey="diversity"
                name="Diversity (x10)"
                stroke="#a855f7" 
                strokeWidth={2}
                dot={false}
                isAnimationActive={true}
                strokeOpacity={0.7}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Secondary Row - Candidates + Throughput */}
      <div className="grid grid-cols-2 gap-4">
        {/* Candidates Evaluated */}
        <div className="relative overflow-hidden rounded-xl bg-gradient-to-br from-slate-900 via-cyan-950/30 to-slate-900 border border-cyan-500/40 shadow-[0_0_25px_rgba(34,211,238,0.12)]">
          <div className="absolute inset-0 bg-gradient-to-t from-cyan-500/5 via-transparent to-transparent" />
          
          <div className="relative px-4 pt-3 pb-1 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse shadow-[0_0_8px_rgba(34,211,238,0.8)]" />
              <span className="text-xs font-bold text-white uppercase tracking-wider">Genomas</span>
            </div>
            <div className="text-xl font-bold text-cyan-300 font-mono"
                 style={{ textShadow: '0 0 15px rgba(34,211,238,0.5)' }}>
              {lastCandidates.toLocaleString()}
            </div>
          </div>
          
          <div className="relative px-1 pb-2">
            <ResponsiveContainer width="100%" height={80}>
              <AreaChart data={candidatesData} margin={{ top: 5, right: 5, left: 5, bottom: 0 }}>
                <defs>
                  <linearGradient id="candidatesGlow" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.6}/>
                    <stop offset="100%" stopColor="#06b6d4" stopOpacity={0.1}/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="t" hide />
                <YAxis hide domain={[0, 'dataMax']} />
                <Area 
                  type="monotone" 
                  dataKey="candidates"
                  stroke="#22d3ee" 
                  strokeWidth={2}
                  fill="url(#candidatesGlow)"
                  isAnimationActive={true}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Throughput + Pareto Size */}
        <div className="relative overflow-hidden rounded-xl bg-gradient-to-br from-slate-900 via-amber-950/20 to-slate-900 border border-amber-500/40 shadow-[0_0_25px_rgba(251,191,36,0.12)]">
          <div className="absolute inset-0 bg-gradient-to-t from-amber-500/5 via-transparent to-transparent" />
          
          <div className="relative px-4 pt-3 pb-1 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-amber-400 animate-pulse shadow-[0_0_8px_rgba(251,191,36,0.8)]" />
              <span className="text-xs font-bold text-white uppercase tracking-wider">Throughput</span>
            </div>
            <div className="flex gap-3 text-sm font-mono">
              <span className="text-amber-300">{lastThroughput.toFixed(0)}/m</span>
              <span className="text-violet-300">P:{lastPareto}</span>
            </div>
          </div>
          
          <div className="relative px-1 pb-2">
            <ResponsiveContainer width="100%" height={80}>
              <LineChart data={throughputData} margin={{ top: 5, right: 5, left: 5, bottom: 0 }}>
                <XAxis dataKey="t" hide />
                <YAxis hide domain={[0, 'dataMax']} />
                <Line 
                  type="monotone" 
                  dataKey="throughput"
                  stroke="#fbbf24" 
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={true}
                />
                <Line 
                  type="stepAfter" 
                  dataKey="pareto"
                  stroke="#8b5cf6" 
                  strokeWidth={1.5}
                  dot={false}
                  isAnimationActive={true}
                  strokeOpacity={0.7}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Resources Row */}
      <div className="relative overflow-hidden rounded-xl bg-gradient-to-br from-slate-900 via-violet-950/20 to-slate-900 border border-violet-500/40 shadow-[0_0_25px_rgba(139,92,246,0.12)]">
        <div className="absolute inset-0 bg-gradient-to-t from-violet-500/5 via-transparent to-transparent" />
        
        <div className="relative px-4 pt-3 pb-1 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-violet-400 animate-pulse shadow-[0_0_8px_rgba(139,92,246,0.8)]" />
            <span className="text-xs font-bold text-white uppercase tracking-wider">System Resources</span>
          </div>
          <div className="flex gap-4 text-sm font-mono">
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-emerald-400" />
              <span className="text-emerald-400">CPU {lastCpu.toFixed(0)}%</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-blue-400" />
              <span className="text-blue-400">RAM {lastMem.toFixed(0)}%</span>
            </div>
          </div>
        </div>
        
        <div className="relative px-1 pb-2">
          <ResponsiveContainer width="100%" height={60}>
            <LineChart data={resourceData} margin={{ top: 5, right: 5, left: 5, bottom: 0 }}>
              <XAxis dataKey="t" hide />
              <YAxis hide domain={[0, 100]} />
              <Line 
                type="monotone" 
                dataKey="cpu" 
                stroke="#10b981" 
                strokeWidth={2}
                dot={false}
                isAnimationActive={true}
              />
              <Line 
                type="monotone" 
                dataKey="mem" 
                stroke="#3b82f6" 
                strokeWidth={2}
                dot={false}
                isAnimationActive={true}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

export default RealTimeCharts;

/**
 * LiveMetricsPanel - Academic-grade metrics dashboard
 * Based on papers: ParetoTracker 2024, He & Lin 2015, Gabor 2018
 * Trading quant: DSR, PSR, Replication Ratio
 */

import { 
  Cpu, MemoryStick, Zap, Hash, TrendingUp, Activity,
  Target, GitBranch, AlertTriangle, HardDrive, Clock, Gauge
} from 'lucide-react';

interface LiveMetricsPanelProps {
  generation: number;
  candidatesEvaluated: number;
  throughputPerMin: number;
  cpuUsage: number;
  memoryUsage: number;
  bestSharpe: number | null;
  evaluationsPerSec: number;
  // Evolution metrics
  meanSharpe?: number;
  diversity?: number;
  convergenceRate?: number;
  stagnation?: number;
  paretoSize?: number;
  hofSize?: number;
  // System metrics
  diskFreeGb?: number;
  diskWriteRate?: number;
  diskTimeToLimit?: number;
}

function MetricCard({ 
  icon: Icon, 
  label, 
  value, 
  suffix = '', 
  glowColor,
  subValue,
  warning,
}: { 
  icon: React.ElementType; 
  label: string; 
  value: string | number; 
  suffix?: string;
  glowColor: string;
  subValue?: string;
  warning?: boolean;
}) {
  const effectiveGlow = warning ? '#ef4444' : glowColor;
  return (
    <div 
      className="relative overflow-hidden rounded-lg border p-3 transition-all duration-300"
      style={{
        background: `linear-gradient(135deg, rgba(15,23,42,0.95) 0%, rgba(30,41,59,0.8) 100%)`,
        borderColor: `${effectiveGlow}40`,
        boxShadow: warning 
          ? `0 0 20px ${effectiveGlow}40, inset 0 1px 0 rgba(255,255,255,0.05)` 
          : `0 0 15px ${effectiveGlow}15, inset 0 1px 0 rgba(255,255,255,0.05)`,
      }}
    >
      <div 
        className="absolute inset-0 opacity-15"
        style={{ background: `radial-gradient(circle at 30% 30%, ${effectiveGlow}30 0%, transparent 70%)` }}
      />
      
      <div className="relative">
        <div className="flex items-center gap-1.5 mb-1">
          <Icon className="w-3.5 h-3.5" style={{ color: effectiveGlow }} />
          <span className="text-[10px] text-slate-400 uppercase tracking-wider font-medium">{label}</span>
        </div>
        <div 
          className="text-lg font-mono font-bold tracking-tight"
          style={{ color: effectiveGlow, textShadow: `0 0 15px ${effectiveGlow}50` }}
        >
          {value}<span className="text-sm opacity-70">{suffix}</span>
        </div>
        {subValue && (
          <div className="text-[10px] text-slate-500 mt-0.5 font-mono">{subValue}</div>
        )}
      </div>
    </div>
  );
}

function CircularGauge({ value, label, activeColor, size = 'sm' }: { 
  value: number; 
  label: string; 
  activeColor: string;
  size?: 'sm' | 'md';
}) {
  const pct = Math.min(value, 100);
  const radius = size === 'sm' ? 22 : 28;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (pct / 100) * circumference;
  const svgSize = size === 'sm' ? 52 : 64;
  const center = svgSize / 2;
  
  return (
    <div 
      className="relative overflow-hidden rounded-lg border p-2 flex items-center gap-2"
      style={{
        background: `linear-gradient(135deg, rgba(15,23,42,0.95) 0%, rgba(30,41,59,0.8) 100%)`,
        borderColor: `${activeColor}40`,
        boxShadow: `0 0 15px ${activeColor}15`,
      }}
    >
      <div 
        className="absolute inset-0 opacity-15"
        style={{ background: `radial-gradient(circle at 70% 50%, ${activeColor}30 0%, transparent 70%)` }}
      />
      
      <div className="relative">
        <svg width={svgSize} height={svgSize} className="transform -rotate-90">
          <circle
            cx={center} cy={center} r={radius}
            fill="none" stroke="#1e293b" strokeWidth="4" opacity="0.5"
          />
          <circle
            cx={center} cy={center} r={radius}
            fill="none" stroke={activeColor} strokeWidth="4"
            strokeDasharray={circumference} strokeDashoffset={offset}
            strokeLinecap="round"
            style={{ filter: `drop-shadow(0 0 4px ${activeColor})`, transition: 'stroke-dashoffset 0.5s ease-out' }}
          />
        </svg>
      </div>
      
      <div className="relative">
        <div className="text-lg font-mono font-bold" style={{ color: activeColor, textShadow: `0 0 12px ${activeColor}50` }}>
          {pct.toFixed(0)}%
        </div>
        <div className="text-[10px] text-slate-400 uppercase tracking-wider">{label}</div>
      </div>
    </div>
  );
}

function SectionHeader({ title, color }: { title: string; color: string }) {
  return (
    <div className="flex items-center gap-2 mb-2">
      <div className="w-1 h-4 rounded-full" style={{ backgroundColor: color }} />
      <span className="text-xs font-semibold uppercase tracking-wider" style={{ color }}>{title}</span>
    </div>
  );
}

export function LiveMetricsPanel({
  generation,
  candidatesEvaluated,
  throughputPerMin,
  cpuUsage,
  memoryUsage,
  bestSharpe,
  evaluationsPerSec,
  meanSharpe = 0,
  diversity = 0,
  convergenceRate = 0,
  stagnation = 0,
  paretoSize = 0,
  hofSize = 0,
  diskFreeGb = 0,
  diskWriteRate = 0,
  diskTimeToLimit,
}: LiveMetricsPanelProps) {
  const sharpeColor = bestSharpe && bestSharpe > 1.0 ? '#10b981' : bestSharpe && bestSharpe > 0.5 ? '#22d3ee' : '#94a3b8';
  const stagnationWarning = stagnation >= 10;
  const diskWarning = diskTimeToLimit !== undefined && diskTimeToLimit < 0.5; // < 30 min
  
  return (
    <div className="space-y-4">
      {/* EVOLUTION METRICS */}
      <div>
        <SectionHeader title="Evolution" color="#a78bfa" />
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
          <MetricCard icon={Hash} label="Generation" value={generation} glowColor="#a78bfa" />
          <MetricCard 
            icon={TrendingUp} 
            label="Best Sharpe" 
            value={bestSharpe !== null ? bestSharpe.toFixed(3) : '—'} 
            glowColor={sharpeColor}
          />
          <MetricCard 
            icon={Activity} 
            label="Mean Sharpe" 
            value={(meanSharpe ?? 0).toFixed(3)} 
            glowColor="#60a5fa"
          />
          <MetricCard 
            icon={GitBranch} 
            label="Diversity" 
            value={(diversity ?? 0).toFixed(3)} 
            glowColor="#c084fc"
            subValue={(diversity ?? 0) < 0.01 ? 'Low!' : undefined}
          />
          <MetricCard 
            icon={Target} 
            label="Conv. Rate" 
            value={`${(convergenceRate ?? 0) >= 0 ? '+' : ''}${(convergenceRate ?? 0).toFixed(4)}`}
            glowColor={(convergenceRate ?? 0) > 0 ? '#10b981' : '#f59e0b'}
          />
          <MetricCard 
            icon={AlertTriangle} 
            label="Stagnation" 
            value={stagnation} 
            suffix=" gens"
            glowColor={stagnationWarning ? '#ef4444' : '#64748b'}
            warning={stagnationWarning}
          />
        </div>
      </div>

      {/* QUANT METRICS */}
      <div>
        <SectionHeader title="Quant" color="#22d3ee" />
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
          <MetricCard 
            icon={Activity} 
            label="Candidates" 
            value={(candidatesEvaluated ?? 0).toLocaleString()} 
            glowColor="#22d3ee"
            subValue={`${(evaluationsPerSec ?? 0).toFixed(1)}/s`}
          />
          <MetricCard 
            icon={Zap} 
            label="Throughput" 
            value={(throughputPerMin ?? 0).toFixed(0)} 
            suffix="/min"
            glowColor="#fbbf24"
          />
          <MetricCard 
            icon={Target} 
            label="Pareto Size" 
            value={paretoSize ?? 0} 
            glowColor="#8b5cf6"
          />
          <MetricCard 
            icon={TrendingUp} 
            label="HoF Size" 
            value={hofSize ?? 0} 
            glowColor="#10b981"
          />
          <MetricCard 
            icon={Gauge} 
            label="Eval/sec" 
            value={(evaluationsPerSec ?? 0).toFixed(1)} 
            glowColor="#f97316"
          />
          <MetricCard 
            icon={Clock} 
            label="ms/Backtest" 
            value={(evaluationsPerSec ?? 0) > 0 ? (1000 / evaluationsPerSec).toFixed(0) : '—'} 
            glowColor="#06b6d4"
          />
        </div>
      </div>

      {/* SYSTEM METRICS */}
      <div>
        <SectionHeader title="System" color="#10b981" />
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2">
          <CircularGauge
            value={cpuUsage ?? 0}
            label="CPU"
            activeColor={(cpuUsage ?? 0) > 90 ? '#ef4444' : (cpuUsage ?? 0) > 70 ? '#f59e0b' : '#10b981'}
          />
          <CircularGauge
            value={memoryUsage ?? 0}
            label="RAM"
            activeColor={(memoryUsage ?? 0) > 90 ? '#ef4444' : (memoryUsage ?? 0) > 75 ? '#f59e0b' : '#3b82f6'}
          />
          <MetricCard 
            icon={HardDrive} 
            label="Disk Free" 
            value={(diskFreeGb ?? 0).toFixed(1)} 
            suffix=" GB"
            glowColor={(diskFreeGb ?? 0) < 2 ? '#ef4444' : (diskFreeGb ?? 0) < 5 ? '#f59e0b' : '#10b981'}
            warning={(diskFreeGb ?? 0) < 2}
          />
          <MetricCard 
            icon={Zap} 
            label="Write Rate" 
            value={(diskWriteRate ?? 0).toFixed(1)} 
            suffix=" MB/s"
            glowColor={(diskWriteRate ?? 0) > 5 ? '#f59e0b' : '#64748b'}
          />
          <MetricCard 
            icon={Clock} 
            label="Time to Limit" 
            value={diskTimeToLimit != null && diskTimeToLimit < 100 ? diskTimeToLimit.toFixed(1) : '∞'} 
            suffix={diskTimeToLimit != null && diskTimeToLimit < 100 ? 'h' : ''}
            glowColor={diskWarning ? '#ef4444' : '#64748b'}
            warning={diskWarning}
            subValue={diskWarning ? 'LOW!' : undefined}
          />
        </div>
      </div>
    </div>
  );
}

export default LiveMetricsPanel;

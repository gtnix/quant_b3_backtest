/**
 * MinerControl - Perpetual Mining Orchestrator Control Panel
 * 
 * Bloomberg Terminal + Grafana + Trading Platform style dashboard
 * Real-time metrics, activity logs, and performance visualization
 */

import { useEffect, useRef, useState } from 'react';
import { 
  Play, Pause, Square, Activity, Cpu, HardDrive, 
  Trophy, Clock, Zap, Database, Gauge, Terminal,
  CheckCircle2, XCircle, AlertCircle, Wifi, WifiOff,
  RefreshCw, ChevronRight, Globe, TrendingUp, BarChart2,
  ArrowUp, ArrowDown, Minus
} from 'lucide-react';
import { useOmpStore } from '../stores/ompStore';
import type { OmpStatus, CurrentCampaign, QueuedCampaign, ActivityLogEntry } from '../stores/ompStore';
import { Sparkline } from '../components/charts/Sparkline';
import { QuickTooltip } from '../components/ui/TooltipInfo';

// =============================================================================
// STATUS INDICATOR
// =============================================================================

function StatusIndicator({ status, sseConnected }: { status: OmpStatus; sseConnected: boolean }) {
  const statusConfig: Record<OmpStatus, { color: string; pulse: boolean; label: string }> = {
    running: { color: 'bg-emerald-500', pulse: true, label: 'MINERANDO' },
    paused: { color: 'bg-amber-500', pulse: false, label: 'PAUSADO' },
    draining: { color: 'bg-orange-500', pulse: true, label: 'FINALIZANDO' },
    offline: { color: 'bg-slate-600', pulse: false, label: 'OFFLINE' },
  };
  
  const config = statusConfig[status];
  
  return (
    <div className="flex items-center gap-3">
      {/* Indicador SSE */}
      <div className={`flex items-center gap-1 text-xs ${sseConnected ? 'text-emerald-400' : 'text-rose-400'}`}>
        {sseConnected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
        <span className="font-mono">{sseConnected ? 'AO VIVO' : 'DESCONECTADO'}</span>
      </div>
      
      {/* Badge de Status */}
      <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-800 border border-slate-700 rounded">
        <span className={`w-2.5 h-2.5 rounded-full ${config.color} ${config.pulse ? 'animate-pulse' : ''}`} />
        <span className="font-mono text-sm text-white tracking-wider">{config.label}</span>
      </div>
    </div>
  );
}

// =============================================================================
// LIVE COUNTER - Animated incrementing number
// =============================================================================

function LiveCounter({ value, label, format = 'number', tooltipKey }: { 
  value: number; 
  label: string; 
  format?: 'number' | 'decimal' | 'time';
  tooltipKey?: string;
}) {
  const prevValueRef = useRef(value);
  const [displayValue, setDisplayValue] = useState(() => formatValue(value, format));
  const [isUpdating, setIsUpdating] = useState(false);
  
  function formatValue(v: number, fmt: 'number' | 'decimal' | 'time') {
    return fmt === 'decimal' 
      ? v.toFixed(2) 
      : fmt === 'time' 
        ? `${Math.floor(v / 3600)}:${String(Math.floor((v % 3600) / 60)).padStart(2, '0')}:${String(Math.floor(v % 60)).padStart(2, '0')}`
        : v.toLocaleString();
  }
  
  useEffect(() => {
    if (value !== prevValueRef.current) {
      setIsUpdating(true);
      // Small delay for smooth transition
      const timer = setTimeout(() => {
        setDisplayValue(formatValue(value, format));
        prevValueRef.current = value;
        setTimeout(() => setIsUpdating(false), 150);
      }, 50);
      return () => clearTimeout(timer);
    }
  }, [value, format]);
  
  return (
    <div className="text-center">
      <div 
        className={`font-mono text-2xl font-bold text-white tabular-nums transition-all duration-300 ease-out ${isUpdating ? 'opacity-80 scale-105' : 'opacity-100 scale-100'}`}
      >
        {displayValue}
      </div>
      <div className="text-[10px] text-slate-500 uppercase tracking-wider flex items-center justify-center gap-0.5">
        {label}
        {tooltipKey && <QuickTooltip termKey={tooltipKey as any} size="sm" />}
      </div>
    </div>
  );
}

// =============================================================================
// GAUGE COMPONENT - Circular progress for resources
// =============================================================================

function CircularGauge({ value, max = 100, label, unit = '%', size = 60, warningThreshold = 70, dangerThreshold = 85 }: {
  value: number;
  max?: number;
  label: string;
  unit?: string;
  size?: number;
  warningThreshold?: number;
  dangerThreshold?: number;
}) {
  const pct = Math.min((value / max) * 100, 100);
  const radius = (size - 8) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (pct / 100) * circumference;
  
  const color = pct >= dangerThreshold ? '#ef4444' : pct >= warningThreshold ? '#f59e0b' : '#10b981';
  
  return (
    <div className="flex flex-col items-center gap-1">
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#1e293b"
          strokeWidth="4"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="4"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          className="transition-all duration-500"
        />
      </svg>
      <div className="absolute flex flex-col items-center justify-center" style={{ width: size, height: size }}>
        <span className="font-mono text-sm font-bold text-white">{value.toFixed(1)}</span>
        <span className="text-[8px] text-slate-500">{unit}</span>
      </div>
      <span className="text-[10px] text-slate-400 uppercase tracking-wider">{label}</span>
    </div>
  );
}

// =============================================================================
// ACTIVITY LOG - Live scrolling terminal
// =============================================================================

function ActivityLog({ logs, maxHeight = 200 }: { logs: ActivityLogEntry[]; maxHeight?: number }) {
  const logRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const lastUserScrollRef = useRef<number>(0);
  
  useEffect(() => {
    if (autoScroll && logRef.current) {
      logRef.current.scrollTop = 0;
    }
  }, [logs, autoScroll]);
  
  // Reabilita auto-scroll após 30 segundos de inatividade
  useEffect(() => {
    if (!autoScroll) {
      const timer = setInterval(() => {
        if (Date.now() - lastUserScrollRef.current > 30000) {
          setAutoScroll(true);
        }
      }, 5000);
      return () => clearInterval(timer);
    }
  }, [autoScroll]);
  
  const levelColors: Record<string, string> = {
    info: 'text-blue-400',
    success: 'text-emerald-400',
    warning: 'text-amber-400',
    error: 'text-rose-400',
  };
  
  const levelIcons: Record<string, React.ReactNode> = {
    info: <Minus className="w-3 h-3" />,
    success: <CheckCircle2 className="w-3 h-3" />,
    warning: <AlertCircle className="w-3 h-3" />,
    error: <XCircle className="w-3 h-3" />,
  };
  
  return (
    <div className="bg-slate-950 border border-slate-800 rounded-lg overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2 bg-slate-900/50 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-slate-500" />
          <span className="text-xs font-medium text-slate-400">LOG DE ATIVIDADE</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-slate-600">{logs.length} entradas</span>
          <button
            onClick={() => {
              setAutoScroll(!autoScroll);
              lastUserScrollRef.current = Date.now();
            }}
            className={`text-xs px-2 py-0.5 rounded ${autoScroll ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-700 text-slate-400'}`}
            title={autoScroll ? 'Clique para pausar auto-scroll' : 'Clique para retomar (ou aguarde 30s)'}
          >
            {autoScroll ? 'AUTO' : 'PAUSADO'}
          </button>
        </div>
      </div>
      <div 
        ref={logRef}
        className="overflow-y-auto font-mono text-xs"
        style={{ maxHeight }}
        onScroll={(e) => {
          const target = e.target as HTMLDivElement;
          if (target.scrollTop > 10) {
            setAutoScroll(false);
            lastUserScrollRef.current = Date.now();
          }
        }}
      >
        {logs.length === 0 ? (
          <div className="p-4 text-center text-slate-600">Nenhuma atividade ainda</div>
        ) : (
          logs.map((log) => (
            <div 
              key={log.id} 
              className="flex items-start gap-2 px-3 py-1.5 hover:bg-slate-900/50 border-b border-slate-800/50"
            >
              <span className="text-slate-600 flex-shrink-0">
                {new Date(log.timestamp).toLocaleTimeString('en-US', { hour12: false })}
              </span>
              <span className={`flex-shrink-0 ${levelColors[log.level] || 'text-slate-400'}`}>
                {levelIcons[log.level]}
              </span>
              <span className="text-slate-300 flex-1">{log.message}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

// =============================================================================
// PERFORMANCE METRICS PANEL - Rust engine stats
// =============================================================================

function PerformancePanel() {
  const { performance, throughputHistory } = useOmpStore();
  
  if (!performance) {
    return (
      <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-4">
        <div className="flex items-center gap-2 mb-3">
          <Gauge className="w-4 h-4 text-slate-500" />
          <span className="text-xs font-medium text-slate-400">MÉTRICAS DO ENGINE RUST</span>
        </div>
        <div className="text-center py-4 text-slate-600 text-sm">Carregando...</div>
      </div>
    );
  }
  
  const current = performance.current_run;
  const historical = performance.historical;
  
  return (
    <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Gauge className="w-4 h-4 text-violet-400" />
          <span className="text-xs font-medium text-slate-400">MÉTRICAS DO ENGINE RUST</span>
        </div>
        {current && (
          <span className="text-[10px] text-slate-600 font-mono">{current.run_id}</span>
        )}
      </div>
      
      {current ? (
        <div className="space-y-4">
          {/* Throughput com Sparkline */}
          <div className="flex items-center justify-between">
            <div>
              <div className="text-2xl font-bold font-mono text-white">
                {current.throughput_genomes_per_min.toFixed(1)}
                <span className="text-sm text-slate-500 ml-1">/min</span>
              </div>
              <div className="text-[10px] text-slate-500 uppercase">Genomas Avaliados</div>
            </div>
            <Sparkline 
              data={throughputHistory} 
              width={100} 
              height={30} 
              color="#8b5cf6"
              showLastPoint={true}
            />
          </div>
          
          {/* Metrics Grid */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-lg font-bold font-mono text-emerald-400">
                {current.evaluations_per_second.toFixed(1)}
              </div>
              <div className="text-[10px] text-slate-500">EVAL/SEC</div>
            </div>
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-lg font-bold font-mono text-blue-400">
                {(current.cache_hit_rate * 100).toFixed(0)}%
              </div>
              <div className="text-[10px] text-slate-500">CACHE HITS</div>
            </div>
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-lg font-bold font-mono text-amber-400">
                {current.memory_mb}
              </div>
              <div className="text-[10px] text-slate-500">MEMORY MB</div>
            </div>
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-lg font-bold font-mono text-white">
                {current.current_generation}
              </div>
              <div className="text-[10px] text-slate-500">GENERATION</div>
            </div>
          </div>
          
          {/* Melhor Sharpe */}
          {current.best_sharpe && (
            <div className="bg-emerald-500/10 border border-emerald-500/20 rounded p-3 text-center">
              <div className="text-3xl font-bold font-mono text-emerald-400">
                {current.best_sharpe.toFixed(3)}
              </div>
              <div className="text-[10px] text-emerald-500/70 uppercase">Melhor Sharpe Ratio</div>
            </div>
          )}
        </div>
      ) : historical ? (
        <div className="space-y-4">
          {/* Throughput Histórico */}
          <div className="flex items-center justify-between">
            <div>
              <div className="text-2xl font-bold font-mono text-slate-400">
                {historical.avg_throughput_per_min.toFixed(1)}
                <span className="text-sm text-slate-500 ml-1">/min</span>
              </div>
              <div className="text-[10px] text-slate-500 uppercase">Throughput Médio (24h)</div>
            </div>
            <Sparkline 
              data={throughputHistory.length > 0 ? throughputHistory : [0, historical.avg_throughput_per_min]} 
              width={100} 
              height={30} 
              color="#64748b"
              showLastPoint={true}
            />
          </div>
          
          {/* Grid de Métricas Históricas */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-lg font-bold font-mono text-blue-400">
                {historical.candidates_1h.toLocaleString()}
              </div>
              <div className="text-[10px] text-slate-500">CANDIDATOS 1H</div>
            </div>
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-lg font-bold font-mono text-violet-400">
                {historical.candidates_24h.toLocaleString()}
              </div>
              <div className="text-[10px] text-slate-500">CANDIDATOS 24H</div>
            </div>
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-lg font-bold font-mono text-amber-400">
                {historical.promotions_24h}
              </div>
              <div className="text-[10px] text-slate-500">PROMOÇÕES 24H</div>
            </div>
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-lg font-bold font-mono text-slate-400">
                {performance.system?.memory_available_mb || '—'}
              </div>
              <div className="text-[10px] text-slate-500">MEM DISP MB</div>
            </div>
          </div>
          
          {/* Melhor Candidato 24h */}
          {historical.best_candidate_24h && (
            <div className="bg-emerald-500/10 border border-emerald-500/20 rounded p-3 text-center">
              <div className="text-2xl font-bold font-mono text-emerald-400">
                {historical.best_candidate_24h.sharpe?.toFixed(3) || '—'}
              </div>
              <div className="text-[10px] text-emerald-500/70 uppercase">Melhor Sharpe (24h)</div>
            </div>
          )}
          
          {/* Status Ocioso */}
          <div className="text-center py-2">
            <span className="px-3 py-1 text-xs bg-slate-800 text-slate-400 rounded-full">
              Em Espera
            </span>
(Content truncated due to size limit. Use page ranges or line ranges to read remaining content)
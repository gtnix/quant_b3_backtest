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

// =============================================================================
// STATUS INDICATOR
// =============================================================================

function StatusIndicator({ status, sseConnected }: { status: OmpStatus; sseConnected: boolean }) {
  const statusConfig: Record<OmpStatus, { color: string; pulse: boolean; label: string }> = {
    running: { color: 'bg-emerald-500', pulse: true, label: 'MINING' },
    paused: { color: 'bg-amber-500', pulse: false, label: 'PAUSED' },
    draining: { color: 'bg-orange-500', pulse: true, label: 'DRAINING' },
    offline: { color: 'bg-slate-600', pulse: false, label: 'OFFLINE' },
  };
  
  const config = statusConfig[status];
  
  return (
    <div className="flex items-center gap-3">
      {/* SSE Indicator */}
      <div className={`flex items-center gap-1 text-xs ${sseConnected ? 'text-emerald-400' : 'text-rose-400'}`}>
        {sseConnected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
        <span className="font-mono">{sseConnected ? 'LIVE' : 'DISCONNECTED'}</span>
      </div>
      
      {/* Status Badge */}
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

function LiveCounter({ value, label, format = 'number' }: { value: number; label: string; format?: 'number' | 'decimal' | 'time' }) {
  const displayValue = format === 'decimal' 
    ? value.toFixed(2) 
    : format === 'time' 
      ? `${Math.floor(value / 3600)}:${String(Math.floor((value % 3600) / 60)).padStart(2, '0')}:${String(Math.floor(value % 60)).padStart(2, '0')}`
      : value.toLocaleString();
  
  return (
    <div className="text-center">
      <div className="font-mono text-2xl font-bold text-white tabular-nums">{displayValue}</div>
      <div className="text-[10px] text-slate-500 uppercase tracking-wider">{label}</div>
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
  
  useEffect(() => {
    if (autoScroll && logRef.current) {
      logRef.current.scrollTop = 0;
    }
  }, [logs, autoScroll]);
  
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
          <span className="text-xs font-medium text-slate-400">ACTIVITY LOG</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-slate-600">{logs.length} entries</span>
          <button
            onClick={() => setAutoScroll(!autoScroll)}
            className={`text-xs px-2 py-0.5 rounded ${autoScroll ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-700 text-slate-400'}`}
          >
            {autoScroll ? 'AUTO' : 'PAUSED'}
          </button>
        </div>
      </div>
      <div 
        ref={logRef}
        className="overflow-y-auto font-mono text-xs"
        style={{ maxHeight }}
        onScroll={(e) => {
          const target = e.target as HTMLDivElement;
          if (target.scrollTop > 10) setAutoScroll(false);
        }}
      >
        {logs.length === 0 ? (
          <div className="p-4 text-center text-slate-600">No activity yet</div>
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
          <span className="text-xs font-medium text-slate-400">RUST ENGINE METRICS</span>
        </div>
        <div className="text-center py-4 text-slate-600 text-sm">Loading...</div>
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
          <span className="text-xs font-medium text-slate-400">RUST ENGINE METRICS</span>
        </div>
        {current && (
          <span className="text-[10px] text-slate-600 font-mono">{current.run_id}</span>
        )}
      </div>
      
      {current ? (
        <div className="space-y-4">
          {/* Throughput with Sparkline */}
          <div className="flex items-center justify-between">
            <div>
              <div className="text-2xl font-bold font-mono text-white">
                {current.throughput_genomes_per_min.toFixed(1)}
                <span className="text-sm text-slate-500 ml-1">/min</span>
              </div>
              <div className="text-[10px] text-slate-500 uppercase">Genomes Evaluated</div>
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
          
          {/* Best Sharpe */}
          {current.best_sharpe && (
            <div className="bg-emerald-500/10 border border-emerald-500/20 rounded p-3 text-center">
              <div className="text-3xl font-bold font-mono text-emerald-400">
                {current.best_sharpe.toFixed(3)}
              </div>
              <div className="text-[10px] text-emerald-500/70 uppercase">Best Sharpe Ratio</div>
            </div>
          )}
        </div>
      ) : historical ? (
        <div className="space-y-4">
          {/* Historical Throughput */}
          <div className="flex items-center justify-between">
            <div>
              <div className="text-2xl font-bold font-mono text-slate-400">
                {historical.avg_throughput_per_min.toFixed(1)}
                <span className="text-sm text-slate-500 ml-1">/min</span>
              </div>
              <div className="text-[10px] text-slate-500 uppercase">Avg Throughput (24h)</div>
            </div>
            <Sparkline 
              data={throughputHistory.length > 0 ? throughputHistory : [0, historical.avg_throughput_per_min]} 
              width={100} 
              height={30} 
              color="#64748b"
              showLastPoint={true}
            />
          </div>
          
          {/* Historical Metrics Grid */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-lg font-bold font-mono text-blue-400">
                {historical.candidates_1h.toLocaleString()}
              </div>
              <div className="text-[10px] text-slate-500">CANDIDATES 1H</div>
            </div>
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-lg font-bold font-mono text-violet-400">
                {historical.candidates_24h.toLocaleString()}
              </div>
              <div className="text-[10px] text-slate-500">CANDIDATES 24H</div>
            </div>
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-lg font-bold font-mono text-amber-400">
                {historical.promotions_24h}
              </div>
              <div className="text-[10px] text-slate-500">PROMOTIONS 24H</div>
            </div>
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-lg font-bold font-mono text-slate-400">
                {performance.system?.memory_available_mb || '—'}
              </div>
              <div className="text-[10px] text-slate-500">MEM AVAIL MB</div>
            </div>
          </div>
          
          {/* Best Candidate 24h */}
          {historical.best_candidate_24h && (
            <div className="bg-emerald-500/10 border border-emerald-500/20 rounded p-3 text-center">
              <div className="text-2xl font-bold font-mono text-emerald-400">
                {historical.best_candidate_24h.sharpe?.toFixed(3) || '—'}
              </div>
              <div className="text-[10px] text-emerald-500/70 uppercase">Best Sharpe (24h)</div>
            </div>
          )}
          
          {/* Idle Status */}
          <div className="text-center py-2">
            <span className="px-3 py-1 text-xs bg-slate-800 text-slate-400 rounded-full">
              Ready to mine
            </span>
          </div>
        </div>
      ) : (
        <div className="text-center py-4 text-slate-600 text-sm">
          Waiting for campaign to start...
        </div>
      )}
    </div>
  );
}

// =============================================================================
// CAMPAIGN CARD - Active campaign display
// =============================================================================

function ActiveCampaignCard({ campaign }: { campaign: CurrentCampaign }) {
  const elapsed = campaign.elapsedSeconds || 0;
  
  return (
    <div className="bg-emerald-500/5 border border-emerald-500/30 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <span className="text-xs font-medium text-emerald-400">ACTIVE CAMPAIGN</span>
        </div>
        <span className="text-[10px] text-slate-500 font-mono">{campaign.runId}</span>
      </div>
      
      <h3 className="text-lg font-bold text-white mb-3">{campaign.campaignName}</h3>
      
      <div className="grid grid-cols-4 gap-2">
        <div className="bg-slate-800/50 rounded p-2 text-center">
          <div className="text-lg font-bold font-mono text-white">{campaign.currentGeneration}</div>
          <div className="text-[9px] text-slate-500">GEN</div>
        </div>
        <div className="bg-slate-800/50 rounded p-2 text-center">
          <div className="text-lg font-bold font-mono text-emerald-400">
            {campaign.bestSharpe?.toFixed(2) || '—'}
          </div>
          <div className="text-[9px] text-slate-500">SHARPE</div>
        </div>
        <div className="bg-slate-800/50 rounded p-2 text-center">
          <div className="text-lg font-bold font-mono text-white">
            {campaign.candidatesEvaluated.toLocaleString()}
          </div>
          <div className="text-[9px] text-slate-500">CAND</div>
        </div>
        <div className="bg-slate-800/50 rounded p-2 text-center">
          <div className="text-lg font-bold font-mono text-blue-400">
            {Math.floor(elapsed / 60)}:{String(elapsed % 60).padStart(2, '0')}
          </div>
          <div className="text-[9px] text-slate-500">TIME</div>
        </div>
      </div>
      
      <div className="flex items-center gap-2 mt-3 text-xs text-slate-500">
        <Globe className="w-3 h-3" />
        <span>{campaign.market.toUpperCase()}</span>
      </div>
    </div>
  );
}

// =============================================================================
// QUEUE ITEM
// =============================================================================

function QueueItem({ campaign, onToggle, onRemove }: { 
  campaign: QueuedCampaign; 
  onToggle: () => void;
  onRemove: () => void;
}) {
  return (
    <div className={`flex items-center justify-between p-2 rounded border ${
      campaign.enabled 
        ? 'bg-slate-800/50 border-slate-700' 
        : 'bg-slate-900/50 border-slate-800 opacity-50'
    }`}>
      <div className="flex items-center gap-2">
        <button
          onClick={onToggle}
          className={`w-6 h-6 rounded flex items-center justify-center ${
            campaign.enabled 
              ? 'bg-emerald-500/20 text-emerald-400' 
              : 'bg-slate-700 text-slate-500'
          }`}
        >
          {campaign.enabled ? <CheckCircle2 className="w-3 h-3" /> : <XCircle className="w-3 h-3" />}
        </button>
        <div>
          <div className="text-sm text-white">{campaign.name}</div>
          <div className="text-[10px] text-slate-500">
            {campaign.market.toUpperCase()} • P{campaign.priority}
            {campaign.repeat && ' • Repeat'}
          </div>
        </div>
      </div>
      <button onClick={onRemove} className="text-slate-600 hover:text-rose-400 p-1">
        <XCircle className="w-3 h-3" />
      </button>
    </div>
  );
}

// =============================================================================
// STATS ROW - Compact metric display
// =============================================================================

function StatRow({ label, value, trend, color = 'white' }: { 
  label: string; 
  value: string | number; 
  trend?: 'up' | 'down' | 'neutral';
  color?: string;
}) {
  const colorClass = color === 'emerald' ? 'text-emerald-400' : color === 'amber' ? 'text-amber-400' : 'text-white';
  
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-slate-800/50 last:border-0">
      <span className="text-xs text-slate-500">{label}</span>
      <div className="flex items-center gap-1">
        <span className={`text-sm font-mono font-medium ${colorClass}`}>{value}</span>
        {trend && (
          <span className={trend === 'up' ? 'text-emerald-400' : trend === 'down' ? 'text-rose-400' : 'text-slate-500'}>
            {trend === 'up' ? <ArrowUp className="w-3 h-3" /> : trend === 'down' ? <ArrowDown className="w-3 h-3" /> : <Minus className="w-3 h-3" />}
          </span>
        )}
      </div>
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function MinerControl() {
  const {
    status,
    startedAt,
    loopCount,
    currentCampaign,
    resources,
    stats,
    queue,
    sseConnected,
    lastError,
    activityLog,
    start,
    stop,
    pause,
    resume,
    updateQueueItem,
    removeFromQueue,
    subscribeToUpdates,
  } = useOmpStore();
  
  // Subscribe to updates on mount
  useEffect(() => {
    const unsubscribe = subscribeToUpdates();
    return unsubscribe;
  }, [subscribeToUpdates]);
  
  const isRunning = status === 'running';
  const isPaused = status === 'paused';
  const isOffline = status === 'offline';
  
  // Calculate uptime in seconds
  const uptimeSeconds = startedAt ? Math.floor((Date.now() - new Date(startedAt).getTime()) / 1000) : 0;
  
  return (
    <div className="min-h-screen bg-slate-950 text-white">
      {/* Header Bar */}
      <div className="bg-slate-900 border-b border-slate-800 px-4 py-3">
        <div className="max-w-[1600px] mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Zap className="w-5 h-5 text-amber-400" />
              <span className="font-bold text-lg">STRATEGY MINER</span>
            </div>
            <span className="text-xs text-slate-600">Orquestrador de Mineração Perpétua</span>
          </div>
          
          <div className="flex items-center gap-4">
            <StatusIndicator status={status} sseConnected={sseConnected} />
            
            {/* Control Buttons */}
            <div className="flex items-center gap-2">
              {isOffline && (
                <button
                  onClick={start}
                  className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 rounded font-medium text-sm transition-colors"
                >
                  <Play className="w-4 h-4" />
                  START
                </button>
              )}
              
              {isRunning && (
                <>
                  <button
                    onClick={pause}
                    className="flex items-center gap-2 px-3 py-2 bg-amber-600 hover:bg-amber-500 rounded font-medium text-sm transition-colors"
                  >
                    <Pause className="w-4 h-4" />
                  </button>
                  <button
                    onClick={stop}
                    className="flex items-center gap-2 px-3 py-2 bg-rose-600 hover:bg-rose-500 rounded font-medium text-sm transition-colors"
                  >
                    <Square className="w-4 h-4" />
                  </button>
                </>
              )}
              
              {isPaused && (
                <>
                  <button
                    onClick={resume}
                    className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 rounded font-medium text-sm transition-colors"
                  >
                    <Play className="w-4 h-4" />
                    RESUME
                  </button>
                  <button
                    onClick={stop}
                    className="flex items-center gap-2 px-3 py-2 bg-rose-600 hover:bg-rose-500 rounded font-medium text-sm transition-colors"
                  >
                    <Square className="w-4 h-4" />
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
      
      {/* Error Banner */}
      {lastError && (
        <div className="bg-rose-500/10 border-b border-rose-500/30 px-4 py-2">
          <div className="max-w-[1600px] mx-auto flex items-center gap-2 text-rose-400 text-sm">
            <AlertCircle className="w-4 h-4" />
            {lastError}
          </div>
        </div>
      )}
      
      {/* Main Content */}
      <div className="max-w-[1600px] mx-auto p-4">
        {/* Top Stats Bar */}
        <div className="grid grid-cols-6 gap-3 mb-4">
          <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-3 flex items-center justify-between">
            <LiveCounter value={loopCount} label="LOOPS" />
            <RefreshCw className={`w-5 h-5 text-slate-600 ${isRunning ? 'animate-spin' : ''}`} />
          </div>
          <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-3 flex items-center justify-between">
            <LiveCounter value={uptimeSeconds} label="UPTIME" format="time" />
            <Clock className="w-5 h-5 text-slate-600" />
          </div>
          <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-3 flex items-center justify-between">
            <LiveCounter value={stats?.candidates.last24h || 0} label="CANDIDATES 24H" />
            <BarChart2 className="w-5 h-5 text-blue-500" />
          </div>
          <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-3 flex items-center justify-between">
            <LiveCounter value={stats?.promotions.last24h || 0} label="PROMOTIONS 24H" />
            <Trophy className="w-5 h-5 text-amber-500" />
          </div>
          <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-3 flex items-center justify-between">
            <LiveCounter value={stats?.promotions.total || 0} label="HALL OF FAME" />
            <Trophy className="w-5 h-5 text-emerald-500" />
          </div>
          <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-3 flex items-center justify-between">
            <LiveCounter value={stats?.throughput.candidatesPerMin || 0} label="THROUGHPUT/MIN" format="decimal" />
            <TrendingUp className="w-5 h-5 text-violet-500" />
          </div>
        </div>
        
        {/* Main Grid */}
        <div className="grid grid-cols-12 gap-4">
          {/* Left Column - Campaign + Queue */}
          <div className="col-span-4 space-y-4">
            {/* Active Campaign */}
            {currentCampaign ? (
              <ActiveCampaignCard campaign={currentCampaign} />
            ) : (
              <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6 text-center">
                <Activity className="w-8 h-8 text-slate-700 mx-auto mb-2" />
                <div className="text-slate-500 text-sm">No Active Campaign</div>
                <div className="text-slate-600 text-xs mt-1">
                  {isOffline ? 'Start mining to begin' : 'Waiting for next...'}
                </div>
              </div>
            )}
            
            {/* Queue */}
            <div className="bg-slate-900/50 border border-slate-800 rounded-lg">
              <div className="flex items-center justify-between px-3 py-2 border-b border-slate-800">
                <div className="flex items-center gap-2">
                  <Database className="w-4 h-4 text-slate-500" />
                  <span className="text-xs font-medium text-slate-400">CAMPAIGN QUEUE</span>
                </div>
                <span className="text-[10px] text-slate-600">{queue?.campaigns?.length || 0}</span>
              </div>
              <div className="p-2 space-y-1 max-h-[200px] overflow-y-auto">
                {queue?.campaigns?.length === 0 ? (
                  <div className="text-center py-4 text-slate-600 text-xs">Empty queue</div>
                ) : (
                  queue?.campaigns?.map(campaign => (
                    <QueueItem 
                      key={campaign.id} 
                      campaign={campaign}
                      onToggle={() => updateQueueItem(campaign.id, { enabled: !campaign.enabled })}
                      onRemove={() => removeFromQueue(campaign.id)}
                    />
                  ))
                )}
              </div>
            </div>
            
            {/* Stats */}
            <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-3">
              <div className="flex items-center gap-2 mb-2">
                <BarChart2 className="w-4 h-4 text-slate-500" />
                <span className="text-xs font-medium text-slate-400">STATISTICS</span>
              </div>
              <StatRow label="Candidates (7d)" value={stats?.candidates.last7d?.toLocaleString() || '0'} />
              <StatRow label="Promotions (7d)" value={stats?.promotions.last7d || 0} color="amber" />
              <StatRow label="Campaigns Done" value={stats?.campaigns.completed || 0} color="emerald" />
              <StatRow label="Campaigns Failed" value={stats?.campaigns.failed || 0} />
            </div>
          </div>
          
          {/* Center Column - Activity Log */}
          <div className="col-span-5">
            <ActivityLog logs={activityLog} maxHeight={450} />
          </div>
          
          {/* Right Column - Resources + Performance */}
          <div className="col-span-3 space-y-4">
            {/* Resources */}
            <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-4">
                <Cpu className="w-4 h-4 text-slate-500" />
                <span className="text-xs font-medium text-slate-400">SYSTEM RESOURCES</span>
              </div>
              
              <div className="flex justify-around items-start">
                <div className="relative">
                  <CircularGauge value={resources.cpuUsage} label="CPU" size={70} />
                </div>
                <div className="relative">
                  <CircularGauge value={resources.memoryUsagePct} label="MEM" size={70} />
                </div>
                <div className="relative">
                  <CircularGauge 
                    value={resources.diskFreeGb} 
                    max={100} 
                    label="DISK" 
                    unit="GB"
                    size={70}
                    warningThreshold={20}
                    dangerThreshold={10}
                  />
                </div>
              </div>
              
              <div className={`mt-4 py-2 rounded text-center text-xs ${
                resources.canStartCampaign 
                  ? 'bg-emerald-500/10 text-emerald-400' 
                  : 'bg-rose-500/10 text-rose-400'
              }`}>
                {resources.canStartCampaign ? (
                  <span className="flex items-center justify-center gap-1">
                    <CheckCircle2 className="w-3 h-3" />
                    Ready to mine
                  </span>
                ) : (
                  <span className="flex items-center justify-center gap-1">
                    <AlertCircle className="w-3 h-3" />
                    Resources constrained
                  </span>
                )}
              </div>
            </div>
            
            {/* Performance Metrics */}
            <PerformancePanel />
            
            {/* Hall of Fame Quick View */}
            <div className="bg-amber-500/5 border border-amber-500/20 rounded-lg p-3">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Trophy className="w-4 h-4 text-amber-400" />
                  <span className="text-xs font-medium text-amber-400">HALL OF FAME</span>
                </div>
                <a href="#/hall-of-fame" className="text-[10px] text-slate-500 hover:text-white flex items-center gap-1">
                  View All <ChevronRight className="w-3 h-3" />
                </a>
              </div>
              <div className="text-center py-4">
                <div className="text-4xl font-bold font-mono text-amber-400">
                  {stats?.promotions.total || 0}
                </div>
                <div className="text-[10px] text-amber-500/70 uppercase">Strategies Promoted</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default MinerControl;

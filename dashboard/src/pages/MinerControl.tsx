/**
 * MinerControl - NYC Quant Style Mining Dashboard
 * Clean, professional, minimal - inspired by trading desks
 */

import { useEffect, useState } from 'react';
import { 
  Play, Square, HardDrive, Trophy, Clock, Zap, 
  Activity, AlertTriangle, Database, 
  TrendingUp, BarChart2, CheckCircle2, XCircle
} from 'lucide-react';
import { useOmpStore } from '../stores/ompStore';
import type { QueuedCampaign, ActivityLogEntry } from '../stores/ompStore';

// =============================================================================
// HERO STATUS - The main visual indicator
// =============================================================================

function HeroStatus({ 
  status, 
  uptimeSeconds, 
  onStart, 
  onStop 
}: { 
  status: string; 
  uptimeSeconds: number; 
  onStart: () => void; 
  onStop: () => void;
}) {
  const isRunning = status === 'running';
  const isPaused = status === 'paused';
  const isActive = isRunning || isPaused;
  
  const formatTime = (secs: number) => {
    const h = Math.floor(secs / 3600);
    const m = Math.floor((secs % 3600) / 60);
    const s = secs % 60;
    return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  };

  return (
    <div className="flex flex-col items-center justify-center py-12">
      {/* Hero Circle */}
      <div className="relative mb-8">
        <div 
          className={`w-40 h-40 rounded-full flex items-center justify-center transition-all duration-500 ${
            isRunning 
              ? 'bg-emerald-500/20 ring-4 ring-emerald-500/50 shadow-[0_0_60px_rgba(16,185,129,0.3)]' 
              : isPaused
                ? 'bg-amber-500/20 ring-4 ring-amber-500/50'
                : 'bg-slate-800/50 ring-2 ring-slate-700'
          }`}
        >
          {isRunning && (
            <div className="absolute inset-0 rounded-full bg-emerald-500/10 animate-ping" />
          )}
          <div className="relative z-10 text-center">
            <div className={`text-4xl font-bold font-mono tracking-tight ${
              isRunning ? 'text-emerald-400' : isPaused ? 'text-amber-400' : 'text-slate-500'
            }`}>
              {isRunning ? 'ON' : isPaused ? 'II' : 'OFF'}
            </div>
          </div>
        </div>
      </div>

      {/* Status Text */}
      <div className="text-center mb-6">
        <div className={`text-2xl font-bold tracking-wider mb-2 ${
          isRunning ? 'text-emerald-400' : isPaused ? 'text-amber-400' : 'text-slate-500'
        }`}>
          {status.toUpperCase()}
        </div>
        {isActive && (
          <div className="font-mono text-3xl text-white tabular-nums">
            {formatTime(uptimeSeconds)}
          </div>
        )}
      </div>

      {/* Single Action Button */}
      {isActive ? (
        <button
          onClick={onStop}
          className="flex items-center gap-3 px-8 py-4 bg-rose-600 hover:bg-rose-500 rounded-xl font-bold text-lg transition-all hover:scale-105 active:scale-95"
        >
          <Square className="w-5 h-5" />
          STOP
        </button>
      ) : (
        <button
          onClick={onStart}
          className="flex items-center gap-3 px-8 py-4 bg-emerald-600 hover:bg-emerald-500 rounded-xl font-bold text-lg transition-all hover:scale-105 active:scale-95"
        >
          <Play className="w-5 h-5" />
          START
        </button>
      )}
    </div>
  );
}

// =============================================================================
// METRIC CARD - Clean stat display (same pattern as TradeBlotter)
// =============================================================================

function MetricCard({ 
  icon, 
  label, 
  value, 
  color = 'white',
  subtext
}: { 
  icon: React.ReactNode; 
  label: string; 
  value: string | number; 
  color?: 'white' | 'profit' | 'loss' | 'warning';
  subtext?: string;
}) {
  const colorClass = color === 'profit' ? 'text-emerald-400' : 
                     color === 'loss' ? 'text-rose-400' : 
                     color === 'warning' ? 'text-amber-400' : 'text-white';
  
  return (
    <div className="p-4 bg-slate-900/50 border border-slate-800 rounded-lg">
      <div className="flex items-center gap-2 text-slate-500 mb-2">
        {icon}
        <span className="text-[10px] uppercase tracking-wider">{label}</span>
      </div>
      <div className={`font-mono font-bold text-2xl ${colorClass}`}>
        {typeof value === 'number' ? value.toLocaleString() : value}
      </div>
      {subtext && (
        <div className="text-[10px] text-slate-600 mt-1">{subtext}</div>
      )}
    </div>
  );
}

// =============================================================================
// DISK MONITOR - Intelligent disk space tracking
// =============================================================================

function DiskMonitor({ 
  diskFreeGb, 
  diskTotalGb,
  writeRateMbPerSec, 
  estimatedTimeToLimitHours,
  shouldAutoStop
}: { 
  diskFreeGb: number; 
  diskTotalGb?: number;
  writeRateMbPerSec: number; 
  estimatedTimeToLimitHours: number | null;
  shouldAutoStop?: boolean;
}) {
  const total = diskTotalGb || 100;
  const usedPct = ((total - diskFreeGb) / total) * 100;
  const criticalZone = ((total - 1) / total) * 100; // 1GB threshold
  
  const formatTime = (hours: number | null) => {
    if (hours === null || hours === Infinity || isNaN(hours)) return '∞';
    if (hours < 1) return `${Math.round(hours * 60)}min`;
    if (hours < 24) return `${hours.toFixed(1)}h`;
    return `${Math.round(hours / 24)}d`;
  };

  const isWarning = diskFreeGb < 5 || (estimatedTimeToLimitHours !== null && estimatedTimeToLimitHours < 2);
  const isCritical = diskFreeGb < 2 || shouldAutoStop;

  return (
    <div className={`p-4 rounded-lg border ${
      isCritical ? 'bg-rose-500/10 border-rose-500/30' : 
      isWarning ? 'bg-amber-500/10 border-amber-500/30' : 
      'bg-slate-900/50 border-slate-800'
    }`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <HardDrive className={`w-4 h-4 ${isCritical ? 'text-rose-400' : isWarning ? 'text-amber-400' : 'text-slate-500'}`} />
          <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Disk Monitor</span>
        </div>
        {isCritical && (
          <span className="flex items-center gap-1 text-xs text-rose-400">
            <AlertTriangle className="w-3 h-3" />
            CRITICAL
          </span>
        )}
      </div>
      
      {/* Progress bar */}
      <div className="relative h-3 bg-slate-800 rounded-full overflow-hidden mb-3">
        <div 
          className={`absolute left-0 top-0 h-full transition-all duration-500 ${
            isCritical ? 'bg-rose-500' : isWarning ? 'bg-amber-500' : 'bg-emerald-500'
          }`}
          style={{ width: `${usedPct}%` }}
        />
        {/* 1GB threshold marker */}
        <div 
          className="absolute top-0 bottom-0 w-0.5 bg-rose-500/50"
          style={{ left: `${criticalZone}%` }}
        />
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-4 text-center">
        <div>
          <div className="font-mono text-lg font-bold text-white">
            {diskFreeGb.toFixed(1)} GB
          </div>
          <div className="text-[10px] text-slate-500 uppercase">Free</div>
        </div>
        <div>
          <div className={`font-mono text-lg font-bold ${writeRateMbPerSec > 1 ? 'text-amber-400' : 'text-slate-400'}`}>
            {writeRateMbPerSec.toFixed(2)} MB/s
          </div>
          <div className="text-[10px] text-slate-500 uppercase">Write Pace</div>
        </div>
        <div>
          <div className={`font-mono text-lg font-bold ${
            estimatedTimeToLimitHours !== null && estimatedTimeToLimitHours < 2 ? 'text-rose-400' : 'text-slate-400'
          }`}>
            {formatTime(estimatedTimeToLimitHours)}
          </div>
          <div className="text-[10px] text-slate-500 uppercase">To 1GB Limit</div>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// QUEUE - Simple campaign list
// =============================================================================

function QueuePanel({ 
  campaigns, 
  onToggle, 
  onRemove 
}: { 
  campaigns: QueuedCampaign[]; 
  onToggle: (id: string, enabled: boolean) => void;
  onRemove: (id: string) => void;
}) {
  return (
    <div className="bg-slate-900/50 border border-slate-800 rounded-lg overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <Database className="w-4 h-4 text-slate-500" />
          <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Queue</span>
        </div>
        <span className="text-xs text-slate-600 font-mono">{campaigns.length}</span>
      </div>
      <div className="max-h-[200px] overflow-y-auto">
        {campaigns.length === 0 ? (
          <div className="p-4 text-center text-slate-600 text-sm">Empty</div>
        ) : (
          campaigns.map(c => (
            <div 
              key={c.id} 
              className={`flex items-center justify-between px-4 py-3 border-b border-slate-800/50 ${
                c.enabled ? '' : 'opacity-50'
              }`}
            >
              <div className="flex items-center gap-3">
                <button
                  onClick={() => onToggle(c.id, !c.enabled)}
                  className={`w-5 h-5 rounded flex items-center justify-center ${
                    c.enabled ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-700 text-slate-500'
                  }`}
                >
                  {c.enabled ? <CheckCircle2 className="w-3 h-3" /> : <XCircle className="w-3 h-3" />}
                </button>
                <div>
                  <div className="text-sm text-white">{c.name}</div>
                  <div className="text-[10px] text-slate-500 font-mono">
                    {c.market.toUpperCase()} • P{c.priority}
                  </div>
                </div>
              </div>
              <button 
                onClick={() => onRemove(c.id)}
                className="text-slate-600 hover:text-rose-400 p-1"
              >
                <XCircle className="w-4 h-4" />
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

// =============================================================================
// ACTIVITY LOG - Compact terminal style
// =============================================================================

function ActivityLog({ logs }: { logs: ActivityLogEntry[] }) {
  const levelColors: Record<string, string> = {
    info: 'text-blue-400',
    success: 'text-emerald-400',
    warning: 'text-amber-400',
    error: 'text-rose-400',
  };

  return (
    <div className="bg-slate-900/50 border border-slate-800 rounded-lg overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-slate-500" />
          <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Activity</span>
        </div>
        <span className="text-xs text-slate-600 font-mono">{logs.length}</span>
      </div>
      <div className="max-h-[200px] overflow-y-auto font-mono text-xs">
        {logs.length === 0 ? (
          <div className="p-4 text-center text-slate-600">No activity</div>
        ) : (
          logs.slice(0, 20).map((log, i) => (
            <div key={log.id || i} className="flex items-start gap-2 px-4 py-2 border-b border-slate-800/50">
              <span className="text-slate-600 flex-shrink-0">
                {new Date(log.timestamp).toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' })}
              </span>
              <span className={levelColors[log.level] || 'text-slate-400'}>•</span>
              <span className="text-slate-300 flex-1 truncate">{log.message}</span>
            </div>
          ))
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
    currentCampaign,
    resources,
    stats,
    queue,
    lastError,
    activityLog,
    start,
    stop,
    updateQueueItem,
    removeFromQueue,
    subscribeToUpdates,
  } = useOmpStore();
  
  // Subscribe to updates on mount
  useEffect(() => {
    const unsubscribe = subscribeToUpdates();
    return unsubscribe;
  }, [subscribeToUpdates]);
  
  // Calculate uptime in seconds
  const uptimeSeconds = startedAt ? Math.floor((Date.now() - new Date(startedAt).getTime()) / 1000) : 0;
  
  // Update every second for live timer
  const [, setTick] = useState(0);
  useEffect(() => {
    if (status === 'running' || status === 'paused') {
      const interval = setInterval(() => setTick(t => t + 1), 1000);
      return () => clearInterval(interval);
    }
  }, [status]);

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      {/* Header */}
      <div className="border-b border-slate-800 px-6 py-4">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Zap className="w-6 h-6 text-amber-400" />
            <span className="font-bold text-xl tracking-tight">STRATEGY MINER</span>
          </div>
          
          {/* Current campaign badge */}
          {currentCampaign && (
            <div className="flex items-center gap-2 px-3 py-1.5 bg-emerald-500/10 border border-emerald-500/30 rounded-lg">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-sm font-medium text-emerald-400">{currentCampaign.campaignName}</span>
              <span className="text-xs text-slate-500">Gen {currentCampaign.currentGeneration}</span>
            </div>
          )}
        </div>
      </div>

      {/* Error Banner */}
      {lastError && (
        <div className="bg-rose-500/10 border-b border-rose-500/30 px-6 py-2">
          <div className="max-w-6xl mx-auto flex items-center gap-2 text-rose-400 text-sm">
            <AlertTriangle className="w-4 h-4" />
            {lastError}
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="max-w-6xl mx-auto p-6">
        {/* Hero Status */}
        <HeroStatus 
          status={status} 
          uptimeSeconds={uptimeSeconds} 
          onStart={start}
          onStop={stop}
        />

        {/* Metrics Row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <MetricCard 
            icon={<BarChart2 className="w-4 h-4" />}
            label="Candidates 24h"
            value={stats?.candidates?.last24h || 0}
          />
          <MetricCard 
            icon={<TrendingUp className="w-4 h-4" />}
            label="Promotions 24h"
            value={stats?.promotions?.last24h || 0}
            color={stats?.promotions?.last24h ? 'profit' : 'white'}
          />
          <MetricCard 
            icon={<Trophy className="w-4 h-4" />}
            label="Hall of Fame"
            value={stats?.promotions?.total || 0}
            color="warning"
          />
          <MetricCard 
            icon={<Clock className="w-4 h-4" />}
            label="Throughput/min"
            value={(stats?.throughput?.candidatesPerMin || 0).toFixed(1)}
            subtext={currentCampaign ? `Gen ${currentCampaign.currentGeneration}` : undefined}
          />
        </div>

        {/* Disk Monitor */}
        <div className="mb-6">
          <DiskMonitor 
            diskFreeGb={resources.diskFreeGb || 0}
            diskTotalGb={resources.diskTotalGb}
            writeRateMbPerSec={resources.writeRateMbPerSec || 0}
            estimatedTimeToLimitHours={resources.estimatedTimeToLimitHours ?? null}
            shouldAutoStop={resources.shouldAutoStop}
          />
        </div>

        {/* Queue + Activity */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <QueuePanel 
            campaigns={queue?.campaigns || []}
            onToggle={(id, enabled) => updateQueueItem(id, { enabled })}
            onRemove={removeFromQueue}
          />
          <ActivityLog logs={activityLog} />
        </div>
        
        {/* Best Sharpe (if running) */}
        {currentCampaign?.bestSharpe && (
          <div className="mt-6 p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-lg text-center">
            <div className="text-sm text-emerald-500/70 uppercase tracking-wider mb-1">Best Sharpe</div>
            <div className="text-4xl font-bold font-mono text-emerald-400">
              {currentCampaign.bestSharpe.toFixed(3)}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default MinerControl;

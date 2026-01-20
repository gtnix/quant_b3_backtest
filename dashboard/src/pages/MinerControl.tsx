/**
 * MinerControl - Simple Strategy Mining Dashboard
 * 2 buttons: Quick (15min) or Full (1h)
 */

import { useEffect, useState } from 'react';
import { 
  Square, HardDrive, Trophy, Zap, 
  Rocket, Timer, Terminal, Settings
} from 'lucide-react';
import { useOmpStore } from '../stores/ompStore';
import platformConfig from '../lib/platform';

// =============================================================================
// SIMPLE START BUTTONS
// =============================================================================

function StartButtons({ 
  onQuickStart, 
  isLoading 
}: { 
  onQuickStart: (mode: 'quick' | 'full') => void;
  isLoading: boolean;
}) {
  return (
    <div className="flex flex-col items-center gap-6 py-8">
      <div className="text-center mb-4">
        <h2 className="text-2xl font-bold text-white mb-2">Iniciar Mining</h2>
        <p className="text-slate-400 text-sm">Escolha o tempo de execução</p>
      </div>
      
      <div className="flex gap-6">
        {/* Quick Start Button */}
        <button
          onClick={() => onQuickStart('quick')}
          disabled={isLoading}
          className="group flex flex-col items-center gap-4 p-8 bg-slate-800/50 hover:bg-emerald-600/20 border-2 border-slate-700 hover:border-emerald-500 rounded-2xl transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <div className="w-20 h-20 rounded-full bg-emerald-500/20 flex items-center justify-center group-hover:bg-emerald-500/30 transition-colors">
            <Timer className="w-10 h-10 text-emerald-400" />
          </div>
          <div className="text-center">
            <div className="text-xl font-bold text-white mb-1">Rápido</div>
            <div className="text-3xl font-mono font-bold text-emerald-400">15 min</div>
            <div className="text-xs text-slate-500 mt-2">50% CPU • Exploração</div>
          </div>
        </button>

        {/* Full Start Button */}
        <button
          onClick={() => onQuickStart('full')}
          disabled={isLoading}
          className="group flex flex-col items-center gap-4 p-8 bg-slate-800/50 hover:bg-amber-600/20 border-2 border-slate-700 hover:border-amber-500 rounded-2xl transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <div className="w-20 h-20 rounded-full bg-amber-500/20 flex items-center justify-center group-hover:bg-amber-500/30 transition-colors">
            <Rocket className="w-10 h-10 text-amber-400" />
          </div>
          <div className="text-center">
            <div className="text-xl font-bold text-white mb-1">Completo</div>
            <div className="text-3xl font-mono font-bold text-amber-400">1 hora</div>
            <div className="text-xs text-slate-500 mt-2">Máximo CPU • Produção</div>
          </div>
        </button>
      </div>
    </div>
  );
}

// =============================================================================
// RUNNING STATUS
// =============================================================================

function RunningStatus({ 
  campaignName,
  elapsedSecs,
  currentGeneration,
  bestSharpe,
  isExternal,
  onStop
}: { 
  campaignName: string;
  elapsedSecs: number;
  currentGeneration: number;
  bestSharpe: number | null;
  isExternal: boolean;
  onStop: () => void;
}) {
  const formatTime = (secs: number) => {
    const h = Math.floor(secs / 3600);
    const m = Math.floor((secs % 3600) / 60);
    const s = secs % 60;
    return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  };

  return (
    <div className="flex flex-col items-center py-8">
      {/* Animated Circle */}
      <div className="relative mb-8">
        <div className="w-48 h-48 rounded-full bg-emerald-500/20 ring-4 ring-emerald-500/50 shadow-[0_0_80px_rgba(16,185,129,0.4)] flex items-center justify-center">
          <div className="absolute inset-0 rounded-full bg-emerald-500/10 animate-ping" />
          <div className="relative z-10 text-center">
            <div className="text-5xl font-bold font-mono text-emerald-400">ON</div>
          </div>
        </div>
      </div>

      {/* Campaign Name */}
      <div className="text-center mb-4">
        <div className="flex items-center justify-center gap-2 mb-2">
          {isExternal && <Terminal className="w-4 h-4 text-blue-400" />}
          <span className="text-lg font-medium text-white">{campaignName}</span>
          {isExternal && <span className="text-xs bg-blue-500/20 text-blue-400 px-2 py-0.5 rounded">CLI</span>}
        </div>
        <div className="font-mono text-4xl text-white tabular-nums">
          {formatTime(elapsedSecs)}
        </div>
      </div>

      {/* Stats */}
      <div className="flex gap-8 mb-8">
        <div className="text-center">
          <div className="text-2xl font-mono font-bold text-slate-300">{currentGeneration}</div>
          <div className="text-xs text-slate-500 uppercase">Geração</div>
        </div>
        {bestSharpe !== null && (
          <div className="text-center">
            <div className="text-2xl font-mono font-bold text-emerald-400">{bestSharpe.toFixed(2)}</div>
            <div className="text-xs text-slate-500 uppercase">Best Sharpe</div>
          </div>
        )}
      </div>

      {/* Stop Button */}
      {!isExternal && (
        <button
          onClick={onStop}
          className="flex items-center gap-3 px-8 py-4 bg-rose-600 hover:bg-rose-500 rounded-xl font-bold text-lg transition-all hover:scale-105 active:scale-95"
        >
          <Square className="w-5 h-5" />
          PARAR
        </button>
      )}
      
      {isExternal && (
        <div className="text-sm text-slate-500 bg-slate-800/50 px-4 py-2 rounded-lg">
          Iniciado via terminal. Use Ctrl+C no terminal para parar.
        </div>
      )}
    </div>
  );
}

// =============================================================================
// STATS BAR
// =============================================================================

function StatsBar({ 
  hofCount, 
  diskFreeGb,
  diskTotalGb
}: { 
  hofCount: number;
  diskFreeGb: number;
  diskTotalGb: number;
}) {
  const usedPct = diskTotalGb > 0 ? ((diskTotalGb - diskFreeGb) / diskTotalGb) * 100 : 0;
  const isLow = diskFreeGb < 5;

  return (
    <div className="grid grid-cols-2 gap-4 max-w-lg mx-auto">
      {/* Hall of Fame */}
      <div className="flex items-center gap-3 p-4 bg-slate-800/50 rounded-xl">
        <Trophy className="w-6 h-6 text-amber-400" />
        <div>
          <div className="text-2xl font-mono font-bold text-white">{hofCount}</div>
          <div className="text-xs text-slate-500">Hall of Fame</div>
        </div>
      </div>

      {/* Disk */}
      <div className="flex items-center gap-3 p-4 bg-slate-800/50 rounded-xl">
        <HardDrive className={`w-6 h-6 ${isLow ? 'text-rose-400' : 'text-slate-400'}`} />
        <div className="flex-1">
          <div className="flex justify-between items-baseline">
            <span className={`text-2xl font-mono font-bold ${isLow ? 'text-rose-400' : 'text-white'}`}>
              {diskFreeGb.toFixed(1)}
            </span>
            <span className="text-xs text-slate-500">GB livres</span>
          </div>
          <div className="h-1.5 bg-slate-700 rounded-full mt-1 overflow-hidden">
            <div 
              className={`h-full ${isLow ? 'bg-rose-500' : 'bg-emerald-500'}`}
              style={{ width: `${usedPct}%` }}
            />
          </div>
        </div>
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
    stop,
    subscribeToUpdates,
  } = useOmpStore();
  
  const [isLoading, setIsLoading] = useState(false);
  
  // Subscribe to updates on mount
  useEffect(() => {
    const unsubscribe = subscribeToUpdates();
    return unsubscribe;
  }, [subscribeToUpdates]);
  
  // Calculate uptime in seconds
  const uptimeSeconds = currentCampaign?.elapsedSecs || 
    (startedAt ? Math.floor((Date.now() - new Date(startedAt).getTime()) / 1000) : 0);
  
  // Update every second for live timer
  const [, setTick] = useState(0);
  useEffect(() => {
    if (status === 'running') {
      const interval = setInterval(() => setTick(t => t + 1), 1000);
      return () => clearInterval(interval);
    }
  }, [status]);

  // Quick start handler
  const handleQuickStart = async (mode: 'quick' | 'full') => {
    setIsLoading(true);
    try {
      const response = await fetch(`${platformConfig.config.apiBase}/omp/quick-start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode }),
      });
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || 'Failed to start');
      }
    } catch (e) {
      console.error('Quick start failed:', e);
    } finally {
      setIsLoading(false);
    }
  };

  const isRunning = status === 'running';
  const isExternal = currentCampaign?.external || false;

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      {/* Header */}
      <div className="border-b border-slate-800 px-6 py-4">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Zap className="w-6 h-6 text-amber-400" />
            <span className="font-bold text-xl tracking-tight">STRATEGY MINER</span>
          </div>
          
          {/* Status badge */}
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg ${
            isRunning 
              ? 'bg-emerald-500/10 border border-emerald-500/30' 
              : 'bg-slate-800/50 border border-slate-700'
          }`}>
            <span className={`w-2 h-2 rounded-full ${isRunning ? 'bg-emerald-500 animate-pulse' : 'bg-slate-500'}`} />
            <span className={`text-sm font-medium ${isRunning ? 'text-emerald-400' : 'text-slate-500'}`}>
              {isRunning ? (isExternal ? 'CLI Ativo' : 'Rodando') : 'Offline'}
            </span>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto p-6">
        {/* Stats Bar */}
        <div className="mb-8">
          <StatsBar 
            hofCount={stats?.promotions?.total || 0}
            diskFreeGb={resources.diskFreeGb || 0}
            diskTotalGb={resources.diskTotalGb || 100}
          />
        </div>

        {/* Main Action Area */}
        {isRunning ? (
          <RunningStatus
            campaignName={currentCampaign?.campaignName || 'Mining'}
            elapsedSecs={uptimeSeconds}
            currentGeneration={currentCampaign?.currentGeneration || 0}
            bestSharpe={currentCampaign?.bestSharpe || null}
            isExternal={isExternal}
            onStop={stop}
          />
        ) : (
          <StartButtons 
            onQuickStart={handleQuickStart}
            isLoading={isLoading}
          />
        )}

        {/* Footer hint */}
        <div className="mt-12 text-center">
          <button className="inline-flex items-center gap-2 text-sm text-slate-600 hover:text-slate-400 transition-colors">
            <Settings className="w-4 h-4" />
            Configuração avançada
          </button>
        </div>
      </div>
    </div>
  );
}

export default MinerControl;

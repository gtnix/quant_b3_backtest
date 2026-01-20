/**
 * MinerControl - Strategy Mining Dashboard
 * Dois presets: Dia (50% CPU) e Noite (100% CPU)
 * Play/Stop sem limite de tempo
 */

import { useEffect, useState } from 'react';
import { 
  Square, Play, HardDrive, Trophy, Zap, 
  Sun, Moon, Settings
} from 'lucide-react';
import { useOmpStore } from '../stores/ompStore';
import platformConfig from '../lib/platform';

type MiningMode = 'day' | 'night';

// =============================================================================
// PRESET BUTTONS
// =============================================================================

function PresetButtons({ 
  onStart, 
  isLoading 
}: { 
  onStart: (mode: MiningMode) => void;
  isLoading: boolean;
}) {
  return (
    <div className="flex flex-col items-center gap-8 py-8">
      <div className="text-center mb-4">
        <h2 className="text-2xl font-bold text-white mb-2">
          Iniciar Mineração
        </h2>
        <p className="text-slate-400 text-sm">
          BR + US • Roda até você parar
        </p>
      </div>
      
      <div className="flex gap-8">
        {/* DAY MODE - 50% CPU */}
        <button
          onClick={() => onStart('day')}
          disabled={isLoading}
          className="group flex flex-col items-center gap-4 p-8 bg-slate-800/50 backdrop-blur-sm hover:bg-cyan-600/20 border-2 border-slate-700 hover:border-cyan-400 hover:shadow-[0_0_50px_rgba(34,211,238,0.25)] rounded-2xl transition-all duration-300 hover:scale-105 hover:-translate-y-2 active:scale-98 disabled:opacity-50 disabled:cursor-not-allowed min-w-[200px]"
        >
          <div className="w-24 h-24 rounded-full bg-cyan-500/20 flex items-center justify-center group-hover:bg-cyan-500/30 transition-colors ring-4 ring-cyan-500/20 group-hover:ring-cyan-500/40">
            <Sun className="w-12 h-12 text-cyan-400" />
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-white mb-1">DIA</div>
            <div className="text-sm text-cyan-400 font-medium">50% CPU</div>
            <div className="text-xs text-slate-500 mt-2">Para usar enquanto trabalha</div>
          </div>
          <div className="flex items-center gap-2 mt-2 px-4 py-2 bg-cyan-500/20 rounded-lg group-hover:bg-cyan-500/30 transition-colors">
            <Play className="w-5 h-5 text-cyan-400" />
            <span className="font-bold text-cyan-400">INICIAR</span>
          </div>
        </button>

        {/* NIGHT MODE - 100% CPU */}
        <button
          onClick={() => onStart('night')}
          disabled={isLoading}
          className="group flex flex-col items-center gap-4 p-8 bg-slate-800/50 backdrop-blur-sm hover:bg-purple-600/20 border-2 border-slate-700 hover:border-purple-400 hover:shadow-[0_0_50px_rgba(168,85,247,0.25)] rounded-2xl transition-all duration-300 hover:scale-105 hover:-translate-y-2 active:scale-98 disabled:opacity-50 disabled:cursor-not-allowed min-w-[200px]"
        >
          <div className="w-24 h-24 rounded-full bg-purple-500/20 flex items-center justify-center group-hover:bg-purple-500/30 transition-colors ring-4 ring-purple-500/20 group-hover:ring-purple-500/40">
            <Moon className="w-12 h-12 text-purple-400" />
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-white mb-1">NOITE</div>
            <div className="text-sm text-purple-400 font-medium">100% CPU</div>
            <div className="text-xs text-slate-500 mt-2">Para deixar rodando dormindo</div>
          </div>
          <div className="flex items-center gap-2 mt-2 px-4 py-2 bg-purple-500/20 rounded-lg group-hover:bg-purple-500/30 transition-colors">
            <Play className="w-5 h-5 text-purple-400" />
            <span className="font-bold text-purple-400">INICIAR</span>
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
  mode,
  elapsedSecs,
  currentGeneration,
  bestSharpe,
  onStop
}: { 
  mode: string;
  elapsedSecs: number;
  currentGeneration: number;
  bestSharpe: number | null;
  onStop: () => void;
}) {
  const formatTime = (secs: number) => {
    const h = Math.floor(secs / 3600);
    const m = Math.floor((secs % 3600) / 60);
    const s = secs % 60;
    return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  };

  const isNight = mode === 'night' || mode === 'full';
  const color = isNight ? 'purple' : 'cyan';
  const Icon = isNight ? Moon : Sun;
  const label = isNight ? 'NOITE' : 'DIA';
  const cpuLabel = isNight ? '100% CPU' : '50% CPU';

  return (
    <div className="flex flex-col items-center py-8">
      {/* Animated Circle */}
      <div className="relative mb-8">
        <div className={`w-56 h-56 rounded-full bg-${color}-500/20 ring-4 ring-${color}-500/50 shadow-[0_0_100px_rgba(${isNight ? '168,85,247' : '34,211,238'},0.4)] flex items-center justify-center`}
             style={{ 
               boxShadow: isNight 
                 ? '0 0 100px rgba(168,85,247,0.4)' 
                 : '0 0 100px rgba(34,211,238,0.4)',
               borderColor: isNight ? 'rgba(168,85,247,0.5)' : 'rgba(34,211,238,0.5)',
               backgroundColor: isNight ? 'rgba(168,85,247,0.2)' : 'rgba(34,211,238,0.2)'
             }}>
          <div className="absolute inset-0 rounded-full animate-ping opacity-20"
               style={{ backgroundColor: isNight ? 'rgba(168,85,247,0.3)' : 'rgba(34,211,238,0.3)' }} />
          <div className="relative z-10 text-center">
            <Icon className={`w-16 h-16 mx-auto mb-2`} style={{ color: isNight ? '#a855f7' : '#22d3ee' }} />
            <div className={`text-3xl font-bold`} style={{ color: isNight ? '#a855f7' : '#22d3ee' }}>{label}</div>
            <div className="text-sm text-slate-400">{cpuLabel}</div>
          </div>
        </div>
      </div>

      {/* Timer */}
      <div className="text-center mb-6">
        <div className="font-mono text-5xl text-white tabular-nums mb-2">
          {formatTime(elapsedSecs)}
        </div>
        <div className="text-sm text-slate-500">BR + US • Rodando até parar</div>
      </div>

      {/* Stats */}
      <div className="flex gap-8 mb-8">
        <div className="text-center">
          <div className="text-3xl font-mono font-bold text-slate-300">{currentGeneration}</div>
          <div className="text-xs text-slate-500 uppercase">Geração</div>
        </div>
        {bestSharpe !== null && (
          <div className="text-center">
            <div className="text-3xl font-mono font-bold text-emerald-400">{bestSharpe.toFixed(2)}</div>
            <div className="text-xs text-slate-500 uppercase">Melhor Sharpe</div>
          </div>
        )}
      </div>

      {/* Stop Button */}
      <button
        onClick={onStop}
        className="flex items-center gap-3 px-10 py-5 bg-rose-600 hover:bg-rose-500 rounded-xl font-bold text-xl transition-all hover:scale-105 active:scale-95 shadow-lg shadow-rose-500/30"
      >
        <Square className="w-6 h-6" />
        PARAR
      </button>
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
      <div className="flex items-center gap-3 p-4 bg-slate-800/60 backdrop-blur-sm rounded-xl border border-amber-500/30 shadow-[0_0_20px_rgba(251,191,36,0.1)]">
        <Trophy className="w-7 h-7 text-amber-400" />
        <div>
          <div className="text-2xl font-mono font-bold text-white">{hofCount}</div>
          <div className="text-xs text-slate-500">Hall of Fame</div>
        </div>
      </div>

      {/* Disk */}
      <div className={`flex items-center gap-3 p-4 bg-slate-800/60 backdrop-blur-sm rounded-xl border ${isLow ? 'border-rose-500/30' : 'border-emerald-500/30'}`}>
        <HardDrive className={`w-7 h-7 ${isLow ? 'text-rose-400' : 'text-emerald-400'}`} />
        <div className="flex-1">
          <div className="flex justify-between items-baseline">
            <span className={`text-2xl font-mono font-bold ${isLow ? 'text-rose-400' : 'text-white'}`}>
              {diskFreeGb.toFixed(1)}
            </span>
            <span className="text-xs text-slate-500">GB livres</span>
          </div>
          <div className="h-1.5 bg-slate-700 rounded-full mt-1 overflow-hidden">
            <div 
              className={`h-full transition-all duration-500 ${isLow ? 'bg-rose-500' : 'bg-emerald-500'}`}
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
  const [currentMode, setCurrentMode] = useState<MiningMode>('day');
  
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

  // Start handler with mode
  const handleStart = async (mode: MiningMode) => {
    setIsLoading(true);
    setCurrentMode(mode);
    try {
      const response = await fetch(`${platformConfig.config.apiBase}/omp/quick-start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          mode: mode === 'night' ? 'full' : 'quick',
          indefinite: true,  // Roda até parar manualmente
          markets: ['BR', 'US']
        }),
      });
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || 'Failed to start');
      }
    } catch (e) {
      console.error('Start failed:', e);
    } finally {
      setIsLoading(false);
    }
  };

  const isRunning = status === 'running';

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      {/* Header */}
      <div className="border-b border-slate-800 px-6 py-4 bg-slate-900/50">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Zap className={`w-6 h-6 ${isRunning ? 'text-amber-400 animate-pulse' : 'text-amber-400'}`} />
            <span className="font-bold text-xl tracking-tight bg-gradient-to-r from-amber-400 to-orange-500 bg-clip-text text-transparent">
              STRATEGY MINER
            </span>
          </div>
          
          {/* Status badge */}
          <div className={`flex items-center gap-2 px-4 py-2 rounded-lg border ${
            isRunning 
              ? 'bg-emerald-500/10 border-emerald-500/30' 
              : 'bg-slate-800/50 border-slate-700'
          }`}>
            <span className={`w-3 h-3 rounded-full ${isRunning ? 'bg-emerald-500 animate-pulse' : 'bg-slate-500'}`} />
            <span className={`text-sm font-medium ${isRunning ? 'text-emerald-400' : 'text-slate-500'}`}>
              {isRunning ? 'Minerando' : 'Parado'}
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
            mode={currentCampaign?.mode || currentMode}
            elapsedSecs={uptimeSeconds}
            currentGeneration={currentCampaign?.currentGeneration || 0}
            bestSharpe={currentCampaign?.bestSharpe || null}
            onStop={stop}
          />
        ) : (
          <PresetButtons 
            onStart={handleStart}
            isLoading={isLoading}
          />
        )}

        {/* Footer */}
        <div className="mt-12 text-center">
          <a 
            href="#config-universe"
            onClick={(e) => {
              e.preventDefault();
              window.dispatchEvent(new CustomEvent('navigate', { detail: 'config-universe' }));
            }}
            className="inline-flex items-center gap-2 text-sm text-slate-500 hover:text-amber-400 transition-colors cursor-pointer"
          >
            <Settings className="w-4 h-4" />
            Configuração avançada
          </a>
        </div>
      </div>
    </div>
  );
}

export default MinerControl;

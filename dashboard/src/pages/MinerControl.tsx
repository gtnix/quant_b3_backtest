/**
 * MinerControl - Premium Strategy Mining Dashboard
 * Visual de alto nível com animações fluidas
 */

import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Square, HardDrive, Trophy, Zap, 
  Rocket, Timer, Terminal, Settings
} from 'lucide-react';
import { useOmpStore } from '../stores/ompStore';
import platformConfig from '../lib/platform';
import { GlassPanel } from '../components/premium/GlassPanel';
import { CountUp } from '../components/premium/CountUp';
import { PulseIndicator } from '../components/premium/PulseIndicator';

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
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="flex flex-col items-center gap-6 py-8"
    >
      <div className="text-center mb-4">
        <motion.h2 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="text-2xl font-bold text-white mb-2"
        >
          Iniciar Mining
        </motion.h2>
        <motion.p 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="text-slate-400 text-sm"
        >
          Escolha o tempo de execução
        </motion.p>
      </div>
      
      <div className="flex gap-6">
        {/* Quick Start Button */}
        <motion.button
          onClick={() => onQuickStart('quick')}
          disabled={isLoading}
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
          whileHover={{ scale: 1.05, y: -4 }}
          whileTap={{ scale: 0.98 }}
          className="group flex flex-col items-center gap-4 p-8 bg-slate-800/50 backdrop-blur-sm hover:bg-emerald-600/20 border-2 border-slate-700 hover:border-emerald-500 hover:shadow-[0_0_40px_rgba(16,185,129,0.2)] rounded-2xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <div className="w-20 h-20 rounded-full bg-emerald-500/20 flex items-center justify-center group-hover:bg-emerald-500/30 transition-colors">
            <Timer className="w-10 h-10 text-emerald-400" />
          </div>
          <div className="text-center">
            <div className="text-xl font-bold text-white mb-1">Rápido</div>
            <div className="text-3xl font-mono font-bold text-emerald-400">15 min</div>
            <div className="text-xs text-slate-500 mt-2">50% CPU • Exploração</div>
          </div>
        </motion.button>

        {/* Full Start Button */}
        <motion.button
          onClick={() => onQuickStart('full')}
          disabled={isLoading}
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3, type: 'spring', stiffness: 200 }}
          whileHover={{ scale: 1.05, y: -4 }}
          whileTap={{ scale: 0.98 }}
          className="group flex flex-col items-center gap-4 p-8 bg-slate-800/50 backdrop-blur-sm hover:bg-amber-600/20 border-2 border-slate-700 hover:border-amber-500 hover:shadow-[0_0_40px_rgba(245,158,11,0.2)] rounded-2xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <div className="w-20 h-20 rounded-full bg-amber-500/20 flex items-center justify-center group-hover:bg-amber-500/30 transition-colors">
            <Rocket className="w-10 h-10 text-amber-400" />
          </div>
          <div className="text-center">
            <div className="text-xl font-bold text-white mb-1">Completo</div>
            <div className="text-3xl font-mono font-bold text-amber-400">1 hora</div>
            <div className="text-xs text-slate-500 mt-2">Máximo CPU • Produção</div>
          </div>
        </motion.button>
      </div>
    </motion.div>
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
    <motion.div 
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="grid grid-cols-2 gap-4 max-w-lg mx-auto"
    >
      {/* Hall of Fame */}
      <GlassPanel glow="gold" className="flex items-center gap-3 p-4">
        <Trophy className="w-6 h-6 text-amber-400" />
        <div>
          <CountUp value={hofCount} decimals={0} className="text-2xl font-mono font-bold text-white" />
          <div className="text-xs text-slate-500">Hall of Fame</div>
        </div>
      </GlassPanel>

      {/* Disk */}
      <GlassPanel glow={isLow ? 'red' : 'green'} className="flex items-center gap-3 p-4">
        <HardDrive className={`w-6 h-6 ${isLow ? 'text-rose-400' : 'text-emerald-400'}`} />
        <div className="flex-1">
          <div className="flex justify-between items-baseline">
            <CountUp 
              value={diskFreeGb} 
              decimals={1} 
              suffix=" GB"
              className={`text-2xl font-mono font-bold ${isLow ? 'text-rose-400' : 'text-white'}`}
            />
          </div>
          <motion.div 
            className="h-1.5 bg-slate-700 rounded-full mt-1 overflow-hidden"
          >
            <motion.div 
              initial={{ width: 0 }}
              animate={{ width: `${usedPct}%` }}
              transition={{ duration: 0.8, ease: 'easeOut' }}
              className={`h-full ${isLow ? 'bg-rose-500' : 'bg-emerald-500'}`}
            />
          </motion.div>
        </div>
      </GlassPanel>
    </motion.div>
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
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="border-b border-slate-800 px-6 py-4 backdrop-blur-sm bg-slate-900/50"
      >
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <motion.div
              animate={{ rotate: isRunning ? 360 : 0 }}
              transition={{ duration: 2, repeat: isRunning ? Infinity : 0, ease: 'linear' }}
            >
              <Zap className="w-6 h-6 text-amber-400" />
            </motion.div>
            <span className="font-bold text-xl tracking-tight bg-gradient-to-r from-amber-400 to-orange-500 bg-clip-text text-transparent">
              STRATEGY MINER
            </span>
          </div>
          
          {/* Status badge with PulseIndicator */}
          <GlassPanel 
            glow={isRunning ? 'green' : 'none'} 
            intensity="low"
            className="flex items-center gap-2 px-3 py-1.5"
          >
            <PulseIndicator 
              status={isRunning ? 'running' : 'offline'} 
              size="sm"
            />
            <span className={`text-sm font-medium ${isRunning ? 'text-emerald-400' : 'text-slate-500'}`}>
              {isRunning ? (isExternal ? 'CLI Ativo' : 'Rodando') : 'Offline'}
            </span>
          </GlassPanel>
        </div>
      </motion.div>

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

        {/* Main Action Area with AnimatePresence */}
        <AnimatePresence mode="wait">
          {isRunning ? (
            <motion.div
              key="running"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.3 }}
            >
              <RunningStatus
                campaignName={currentCampaign?.campaignName || 'Mining'}
                elapsedSecs={uptimeSeconds}
                currentGeneration={currentCampaign?.currentGeneration || 0}
                bestSharpe={currentCampaign?.bestSharpe || null}
                isExternal={isExternal}
                onStop={stop}
              />
            </motion.div>
          ) : (
            <motion.div
              key="idle"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.3 }}
            >
              <StartButtons 
                onQuickStart={handleQuickStart}
                isLoading={isLoading}
              />
            </motion.div>
          )}
        </AnimatePresence>

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

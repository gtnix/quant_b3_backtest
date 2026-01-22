/**
 * MinerControl - Strategy Mining Dashboard
 * Dois presets: Dia (50% CPU) e Noite (100% CPU)
 * Play/Stop sem limite de tempo
 */

import { useEffect, useState } from 'react';
import { 
  Square, Play, HardDrive, Trophy, Zap, 
  Sun, Moon, Settings, FolderOpen, GitBranch, AlertTriangle,
  CheckCircle, XCircle, AlertCircle, RefreshCw, Terminal
} from 'lucide-react';
import { useOmpStore } from '../stores/ompStore';
import { useStrategyStore } from '../stores/strategyStore';
import { RealTimeCharts } from '../components/mining';
import platformConfig from '../lib/platform';

type MiningMode = 'day' | 'night';

interface PreflightCheck {
  id: string;
  name: string;
  status: 'pass' | 'fail' | 'warn';
  message?: string;
  fix?: string;
  path?: string;
  size?: string;
  age?: string;
  free?: string;
}

interface PreflightResult {
  canStart: boolean;
  checks: PreflightCheck[];
  errors: PreflightCheck[];
  warnings: PreflightCheck[];
  summary: { passed: number; failed: number; warnings: number };
}

// =============================================================================
// PREFLIGHT MODAL
// =============================================================================
function PreflightModal({ 
  result, 
  onClose, 
  onProceed,
  isLoading
}: { 
  result: PreflightResult;
  onClose: () => void;
  onProceed: () => void;
  isLoading: boolean;
}) {
  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl max-w-2xl w-full max-h-[80vh] overflow-hidden">
        {/* Header */}
        <div className={`px-6 py-4 border-b ${result.canStart ? 'border-emerald-500/30 bg-emerald-500/10' : 'border-red-500/30 bg-red-500/10'}`}>
          <div className="flex items-center gap-3">
            {result.canStart ? (
              <CheckCircle className="w-6 h-6 text-emerald-400" />
            ) : (
              <XCircle className="w-6 h-6 text-red-400" />
            )}
            <h2 className="text-xl font-bold text-white">
              {result.canStart ? 'Sistema Pronto' : 'Problemas Detectados'}
            </h2>
          </div>
          <p className="text-sm text-slate-400 mt-1">
            {result.summary.passed} verificações OK • {result.summary.failed} erros • {result.summary.warnings} avisos
          </p>
        </div>
        
        {/* Content */}
        <div className="px-6 py-4 overflow-y-auto max-h-[50vh] space-y-4">
          {/* Errors first */}
          {result.errors.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-red-400 mb-2 flex items-center gap-2">
                <XCircle className="w-4 h-4" /> Erros Críticos
              </h3>
              <div className="space-y-2">
                {result.errors.map(err => (
                  <div key={err.id} className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
                    <div className="font-medium text-red-300">{err.name}</div>
                    <div className="text-sm text-red-200/70">{err.message}</div>
                    {err.fix && (
                      <div className="mt-2 flex items-center gap-2">
                        <Terminal className="w-4 h-4 text-slate-400" />
                        <code className="text-xs bg-slate-800 px-2 py-1 rounded text-amber-300 font-mono">
                          {err.fix}
                        </code>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Warnings */}
          {result.warnings.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-amber-400 mb-2 flex items-center gap-2">
                <AlertCircle className="w-4 h-4" /> Avisos
              </h3>
              <div className="space-y-2">
                {result.warnings.map(warn => (
                  <div key={warn.id} className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-3">
                    <div className="font-medium text-amber-300">{warn.name}</div>
                    <div className="text-sm text-amber-200/70">{warn.message}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Passed checks (collapsed) */}
          {result.checks.length > 0 && (
            <details className="group">
              <summary className="text-sm font-semibold text-emerald-400 cursor-pointer flex items-center gap-2">
                <CheckCircle className="w-4 h-4" /> {result.checks.length} Verificações OK
              </summary>
              <div className="mt-2 space-y-1 pl-6">
                {result.checks.map(check => (
                  <div key={check.id} className="text-sm text-slate-400 flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-emerald-500" />
                    <span>{check.name}</span>
                    {check.size && <span className="text-xs text-slate-500">({check.size})</span>}
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>
        
        {/* Footer */}
        <div className="px-6 py-4 border-t border-slate-700 flex justify-end gap-3">
          <button
            onClick={onClose}
            className="px-4 py-2 text-slate-400 hover:text-white transition-colors"
          >
            Fechar
          </button>
          {result.canStart && (
            <button
              onClick={onProceed}
              disabled={isLoading}
              className="px-6 py-2 bg-emerald-500 hover:bg-emerald-600 text-white font-semibold rounded-lg transition-colors flex items-center gap-2 disabled:opacity-50"
            >
              {isLoading ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                <Play className="w-4 h-4" />
              )}
              Iniciar Mineração
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

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
// STAGE A/B FUNNEL
// =============================================================================

function StageFunnel({
  paretoSize,
  validatedCount,
  validatedTotal,
  hofSize
}: {
  paretoSize: number;
  validatedCount: number;
  validatedTotal: number;
  hofSize: number;
}) {
  const stageAPassRate = paretoSize > 0 ? Math.round((validatedTotal / paretoSize) * 100) : 0;
  const stageBPassRate = validatedTotal > 0 ? Math.round((validatedCount / validatedTotal) * 100) : 0;
  
  return (
    <div className="bg-slate-800/60 backdrop-blur-sm rounded-xl border border-slate-700 p-4">
      <h3 className="text-sm text-slate-400 font-medium mb-3 flex items-center gap-2">
        <GitBranch className="w-4 h-4" /> Stage A → B Funnel
      </h3>
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-sm text-slate-400">Pareto (Stage A)</span>
          <span className="text-lg font-mono text-cyan-400">{paretoSize}</span>
        </div>
        <div className="w-full bg-slate-700 rounded-full h-2">
          <div 
            className="bg-cyan-500 h-2 rounded-full transition-all duration-500"
            style={{ width: `${Math.min(100, stageAPassRate)}%` }}
          />
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm text-slate-400">Stage B Tested</span>
          <span className="text-lg font-mono text-violet-400">{validatedTotal}</span>
        </div>
        <div className="w-full bg-slate-700 rounded-full h-2">
          <div 
            className="bg-violet-500 h-2 rounded-full transition-all duration-500"
            style={{ width: `${Math.min(100, stageBPassRate)}%` }}
          />
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm text-slate-400">Validated (Pass)</span>
          <span className="text-lg font-mono text-emerald-400">{validatedCount}</span>
        </div>
        <div className="pt-2 border-t border-slate-700 flex items-center justify-between">
          <span className="text-sm text-slate-400">Hall of Fame</span>
          <span className="text-lg font-mono font-bold text-amber-400">{hofSize}</span>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// EXPERIMENT INFO
// =============================================================================

function ExperimentInfo({
  experimentId,
  artifactsPath,
  elapsedSecs,
  startedAt
}: {
  experimentId?: string;
  artifactsPath?: string;
  elapsedSecs: number;
  startedAt?: string;
}) {
  const openFolder = () => {
    if (artifactsPath) {
      // Try to open folder in system file manager
      fetch(`http://localhost:3001/api/omp/open-folder`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: artifactsPath })
      }).catch(() => {});
    }
  };
  
  return (
    <div className="bg-slate-800/60 backdrop-blur-sm rounded-xl border border-slate-700 p-4">
      <h3 className="text-sm text-slate-400 font-medium mb-3 flex items-center gap-2">
        <FolderOpen className="w-4 h-4" /> Experiment
      </h3>
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-500">ID</span>
          <span className="text-sm font-mono text-white truncate max-w-[180px]">
            {experimentId || '—'}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-500">Started</span>
          <span className="text-sm text-slate-300">
            {startedAt ? new Date(startedAt).toLocaleTimeString() : '—'}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-500">Artifacts</span>
          <button 
            onClick={openFolder}
            className="text-xs text-cyan-400 hover:text-cyan-300 font-mono truncate max-w-[180px] hover:underline"
            title={artifactsPath}
          >
            {artifactsPath || '—'}
          </button>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// MARKET BREAKDOWN
// =============================================================================

function MarketBreakdown({
  marketStats
}: {
  marketStats: Record<string, { generation: number; bestSharpe: number | null; candidates: number; hofSize: number }>;
}) {
  const markets = Object.entries(marketStats);
  if (markets.length === 0) return null;
  
  return (
    <div className="bg-slate-800/60 backdrop-blur-sm rounded-xl border border-slate-700 p-4">
      <h3 className="text-sm text-slate-400 font-medium mb-3">Markets</h3>
      <div className="space-y-3">
        {markets.map(([market, stats]) => (
          <div key={market} className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className={`text-xs font-bold px-2 py-0.5 rounded ${
                market === 'BR' ? 'bg-cyan-500/20 text-cyan-400' : 'bg-blue-500/20 text-blue-400'
              }`}>
                {market}
              </span>
              <span className="text-sm font-mono text-white">G{stats.generation}</span>
            </div>
            <div className="flex items-center gap-3 text-sm">
              <span className="text-emerald-400">{stats.bestSharpe?.toFixed(2) || '—'}</span>
              <span className="text-slate-400">{stats.candidates} bt</span>
              <span className="text-amber-400">🏆{stats.hofSize}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// =============================================================================
// STATS BAR
// =============================================================================

function StatsBar({ 
  hofCount, 
  diskFreeGb,
  diskTotalGb,
  genBR,
  genUS,
  passBR,
  passUS,
  throughput,
  pending,
  cpu,
  mem,
}: { 
  hofCount: number;
  diskFreeGb: number;
  diskTotalGb: number;
  genBR?: number;
  genUS?: number;
  passBR?: number;
  passUS?: number;
  throughput?: number;
  pending?: number;
  cpu?: number;
  mem?: number;
}) {
  const usedPct = diskTotalGb > 0 ? ((diskTotalGb - diskFreeGb) / diskTotalGb) * 100 : 0;
  const isLow = diskFreeGb < 5;

  return (
    <div className="grid grid-cols-4 gap-3 max-w-4xl mx-auto">
      {/* Hall of Fame */}
      <div className="flex items-center gap-3 p-3 bg-slate-800/60 backdrop-blur-sm rounded-xl border border-amber-500/30">
        <Trophy className="w-6 h-6 text-amber-400" />
        <div>
          <div className="text-xl font-mono font-bold text-white">{hofCount}</div>
          <div className="text-xs text-slate-500">Hall of Fame</div>
        </div>
      </div>

      {/* Markets */}
      <div className="flex items-center gap-3 p-3 bg-slate-800/60 backdrop-blur-sm rounded-xl border border-cyan-500/30">
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2">
            <span className="text-xs text-cyan-400 font-bold w-6">BR</span>
            <span className="text-sm font-mono text-white">G{genBR || 0}</span>
            <span className="text-xs text-emerald-400">{passBR || 0}%</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-blue-400 font-bold w-6">US</span>
            <span className="text-sm font-mono text-white">G{genUS || 0}</span>
            <span className="text-xs text-emerald-400">{passUS || 0}%</span>
          </div>
        </div>
      </div>

      {/* Throughput & Pending */}
      <div className="flex items-center gap-3 p-3 bg-slate-800/60 backdrop-blur-sm rounded-xl border border-violet-500/30">
        <Zap className="w-6 h-6 text-violet-400" />
        <div>
          <div className="text-xl font-mono font-bold text-white">{throughput || 0}</div>
          <div className="text-xs text-slate-500">backtests • {pending || 0} pending</div>
        </div>
      </div>

      {/* Resources */}
      <div className={`flex items-center gap-3 p-3 bg-slate-800/60 backdrop-blur-sm rounded-xl border ${isLow ? 'border-rose-500/30' : 'border-emerald-500/30'}`}>
        <HardDrive className={`w-6 h-6 ${isLow ? 'text-rose-400' : 'text-emerald-400'}`} />
        <div className="flex-1">
          <div className="flex gap-2 text-sm font-mono">
            <span className="text-emerald-400">{cpu || 0}%</span>
            <span className="text-blue-400">{mem || 0}%</span>
          </div>
          <div className="flex justify-between items-baseline">
            <span className={`text-sm font-mono ${isLow ? 'text-rose-400' : 'text-slate-300'}`}>{diskFreeGb.toFixed(1)}GB</span>
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
    throughputHistory,
    cpuHistory,
    memoryHistory,
    sharpeHistory,
    generationHistory,
    candidatesHistory,
    diversityHistory,
    meanSharpeHistory,
    paretoSizeHistory,
  } = useOmpStore();
  
  // Get selected strategies from strategy store
  const { selectedStrategies } = useStrategyStore();
  
  const [isLoading, setIsLoading] = useState(false);
  const [currentMode, setCurrentMode] = useState<MiningMode>('day');
  
  // Subscribe to updates on mount
  useEffect(() => {
    const unsubscribe = subscribeToUpdates();
    return unsubscribe;
  }, [subscribeToUpdates]);
  
  // Use marketStats from currentCampaign (single source of truth from /omp/status)
  const marketStats = currentCampaign?.marketStats || {};
  
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

  // Preflight state
  const [preflightResult, setPreflightResult] = useState<PreflightResult | null>(null);
  const [pendingMode, setPendingMode] = useState<MiningMode | null>(null);

  // Run preflight check before starting
  const handleStart = async (mode: MiningMode) => {
    setIsLoading(true);
    setPendingMode(mode);
    try {
      const response = await fetch(`${platformConfig.config.apiBase}/omp/preflight?markets=BR,US`);
      const result: PreflightResult = await response.json();
      setPreflightResult(result);
      // Always show modal for user confidence
    } catch (e) {
      console.error('Preflight failed:', e);
      setPreflightResult({
        canStart: false,
        checks: [],
        errors: [{ id: 'network', name: 'API Connection', status: 'fail', message: 'Não foi possível conectar à API' }],
        warnings: [],
        summary: { passed: 0, failed: 1, warnings: 0 }
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Actually start the mining (after preflight passes)
  const executeStart = async (mode: MiningMode) => {
    setIsLoading(true);
    setCurrentMode(mode);
    try {
      const response = await fetch(`${platformConfig.config.apiBase}/omp/quick-start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          mode: mode === 'night' ? 'full' : 'quick',
          indefinite: true,
          markets: ['BR', 'US'],
          templateSlugs: selectedStrategies.length > 0 ? selectedStrategies : undefined,
          skipPreflight: true // Already validated
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
      {/* Preflight Modal */}
      {preflightResult && (
        <PreflightModal
          result={preflightResult}
          onClose={() => {
            setPreflightResult(null);
            setPendingMode(null);
          }}
          onProceed={async () => {
            if (pendingMode) {
              await executeStart(pendingMode);
              setPreflightResult(null);
              setPendingMode(null);
            }
          }}
          isLoading={isLoading}
        />
      )}
      
      {/* Header */}
      <div className="border-b border-slate-800 px-6 py-4 bg-slate-900/50">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
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
      <div className="max-w-6xl mx-auto p-6">
        {/* Stats Bar */}
        <div className="mb-8">
          <StatsBar 
            hofCount={stats?.promotions?.total || 0}
            diskFreeGb={resources.diskFreeGb || 0}
            diskTotalGb={resources.diskTotalGb || 100}
            genBR={marketStats.BR?.generation}
            genUS={marketStats.US?.generation}
            passBR={currentCampaign?.validatedCount || 0}
            passUS={currentCampaign?.validatedTotal || 0}
            throughput={Math.round((marketStats.BR?.candidates || 0) + (marketStats.US?.candidates || 0))}
            pending={currentCampaign?.paretoSize || 0}
            cpu={resources.cpuUsage}
            mem={resources.memoryUsagePct}
          />
        </div>

        {/* Main Action Area */}
        {isRunning ? (
          <>
            <RunningStatus
              mode={currentCampaign?.mode || currentMode}
              elapsedSecs={uptimeSeconds}
              currentGeneration={currentCampaign?.currentGeneration || 0}
              bestSharpe={currentCampaign?.bestSharpe || null}
              onStop={stop}
            />
            
            {/* Wall Monitor: Funnel + Experiment + Markets */}
            <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
              <StageFunnel
                paretoSize={currentCampaign?.paretoSize || 0}
                validatedCount={currentCampaign?.validatedCount || 0}
                validatedTotal={currentCampaign?.validatedTotal || 0}
                hofSize={currentCampaign?.hofSize || 0}
              />
              <ExperimentInfo
                experimentId={currentCampaign?.experimentId}
                artifactsPath={currentCampaign?.artifactsPath}
                elapsedSecs={uptimeSeconds}
                startedAt={startedAt}
              />
              <MarketBreakdown
                marketStats={marketStats}
              />
            </div>
            
            {/* Real-time Charts */}
            <div className="mt-8">
              <RealTimeCharts
                throughputHistory={throughputHistory}
                cpuHistory={cpuHistory}
                memoryHistory={memoryHistory}
                sharpeHistory={sharpeHistory}
                generationHistory={generationHistory}
                candidatesHistory={candidatesHistory}
                diversityHistory={diversityHistory}
                meanSharpeHistory={meanSharpeHistory}
                paretoSizeHistory={paretoSizeHistory}
              />
            </div>
          </>
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

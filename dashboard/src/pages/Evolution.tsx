/**
 * Evolution Monitor - Real-time SCG Evolution Tracking
 * 
 * Shows live genetic algorithm progress with real data from SSE
 * Includes fitness charts, Pareto frontier, and performance metrics
 */

import { useState, useEffect, useMemo } from 'react';
import { GenerationChart } from '../components/charts/GenerationChart';
import { ParetoScatter } from '../components/charts/ParetoScatter';
import { Sparkline, SparkBar } from '../components/charts/Sparkline';
import { QuickTooltip } from '../components/ui/TooltipInfo';
import { useOmpStore } from '../stores/ompStore';
import { config } from '../lib/platform';
import { 
  Play, 
  Pause, 
  RotateCcw,
  Zap,
  Clock,
  Cpu,
  TrendingUp,
  Activity,
  Target,
  Database,
  RefreshCw,
  AlertCircle,
  CheckCircle2,
  Wifi,
  WifiOff,
  BarChart3,
  GitBranch,
  Layers,
  Gauge
} from 'lucide-react';

interface EvolutionState {
  status: 'idle' | 'running' | 'paused' | 'completed' | 'error';
  generation: number;
  maxGenerations: number;
  populationSize: number;
  totalEvaluations: number;
  cacheHits: number;
  elapsedTime: string;
  eta: string;
  bestSharpe: number;
  bestCagr: number;
  bestMaxDD: number;
  meanSharpe: number;
  paretoSize: number;
  campaignId?: string;
  campaignName?: string;
  runId?: string;
}

interface GenerationPoint {
  generation: number;
  bestSharpe: number;
  meanSharpe: number;
  paretoSize: number;
  bestCagr?: number;
  evaluations?: number;
}

interface ParetoPoint {
  id: string;
  sharpe: number;
  maxDrawdown: number;
  cagr?: number;
  gatesPassed?: boolean;
  displayName?: string;
}

export function Evolution() {
  const { status, currentCampaign, performance, sseConnected } = useOmpStore();
  
  const [evolutionState, setEvolutionState] = useState<EvolutionState>({
    status: 'idle',
    generation: 0,
    maxGenerations: 50,
    populationSize: 100,
    totalEvaluations: 0,
    cacheHits: 0,
    elapsedTime: '00:00:00',
    eta: '--:--:--',
    bestSharpe: 0,
    bestCagr: 0,
    bestMaxDD: 0,
    meanSharpe: 0,
    paretoSize: 0,
  });
  
  const [generationHistory, setGenerationHistory] = useState<GenerationPoint[]>([]);
  const [paretoData, setParetoData] = useState<ParetoPoint[]>([]);
  const [loading, setLoading] = useState(true);
  
  // Fetch initial data and subscribe to updates
  useEffect(() => {
    loadEvolutionState();
    loadParetoData();
    
    const interval = setInterval(() => {
      if (evolutionState.status === 'running') {
        loadEvolutionState();
        loadParetoData();
      }
    }, 5000); // Update every 5 seconds while running
    
    return () => clearInterval(interval);
  }, [evolutionState.status]);

  // Sync with OMP current campaign
  useEffect(() => {
    if (currentCampaign) {
      setEvolutionState(prev => ({
        ...prev,
        status: 'running',
        campaignId: currentCampaign.id,
        campaignName: currentCampaign.name,
        totalEvaluations: currentCampaign.candidatesEvaluated,
      }));
      
      // Parse generation info from output
      if (currentCampaign.output) {
        parseOutputForEvolution(currentCampaign.output);
      }
    } else if (status === 'running') {
      setEvolutionState(prev => ({ ...prev, status: 'idle' }));
    }
  }, [currentCampaign, status]);

  // Parse combiner output for evolution metrics
  const parseOutputForEvolution = (output: string[]) => {
    const recentLines = output.slice(-100);
    let newState: Partial<EvolutionState> = {};
    let newPoints: GenerationPoint[] = [];
    
    for (const line of recentLines) {
      // Parse generation progress
      const genMatch = line.match(/gen(?:eration)?[:\s]+(\d+)\s*(?:\/|of)\s*(\d+)/i);
      if (genMatch) {
        newState.generation = parseInt(genMatch[1]);
        newState.maxGenerations = parseInt(genMatch[2]);
      }
      
      // Parse best Sharpe
      const sharpeMatch = line.match(/best[_\s]?sharpe[:\s]+([\d.]+)/i);
      if (sharpeMatch) {
        newState.bestSharpe = parseFloat(sharpeMatch[1]);
      }
      
      // Parse mean Sharpe
      const meanSharpeMatch = line.match(/mean[_\s]?sharpe[:\s]+([\d.]+)/i);
      if (meanSharpeMatch) {
        newState.meanSharpe = parseFloat(meanSharpeMatch[1]);
      }
      
      // Parse Pareto size
      const paretoMatch = line.match(/pareto[_\s]?(?:size|frontier)[:\s]+(\d+)/i);
      if (paretoMatch) {
        newState.paretoSize = parseInt(paretoMatch[1]);
      }
      
      // Parse evaluations
      const evalMatch = line.match(/eval(?:uation)?s?[:\s]+(\d+)/i);
      if (evalMatch) {
        newState.totalEvaluations = parseInt(evalMatch[1]);
      }
      
      // Parse cache hits
      const cacheMatch = line.match(/cache[_\s]?hits?[:\s]+(\d+)/i);
      if (cacheMatch) {
        newState.cacheHits = parseInt(cacheMatch[1]);
      }
      
      // Parse CAGR
      const cagrMatch = line.match(/best[_\s]?cagr[:\s]+([\d.]+)/i);
      if (cagrMatch) {
        newState.bestCagr = parseFloat(cagrMatch[1]);
      }
      
      // Collect generation data points
      const fullGenMatch = line.match(/\[gen\s*(\d+)\].*?sharpe[:\s]*([\d.]+).*?mean[:\s]*([\d.]+)/i);
      if (fullGenMatch) {
        newPoints.push({
          generation: parseInt(fullGenMatch[1]),
          bestSharpe: parseFloat(fullGenMatch[2]),
          meanSharpe: parseFloat(fullGenMatch[3]),
          paretoSize: newState.paretoSize || 0,
        });
      }
    }
    
    setEvolutionState(prev => ({
      ...prev,
      ...newState,
      status: 'running',
    }));
    
    if (newPoints.length > 0) {
      setGenerationHistory(prev => {
        const existing = new Set(prev.map(p => p.generation));
        const newUnique = newPoints.filter(p => !existing.has(p.generation));
        return [...prev, ...newUnique].sort((a, b) => a.generation - b.generation);
      });
    }
  };

  const loadEvolutionState = async () => {
    try {
      const response = await fetch(`${config.apiBase}/evolution/state`);
      if (response.ok) {
        const data = await response.json();
        if (data.status !== 'no_data') {
          setEvolutionState(prev => ({
            ...prev,
            ...data,
          }));
          if (data.generationHistory) {
            setGenerationHistory(data.generationHistory);
          }
        }
      }
    } catch (err) {
      // Silent fail - use SSE data instead
    } finally {
      setLoading(false);
    }
  };

  const loadParetoData = async () => {
    try {
      // Load recent candidates as Pareto approximation
      const response = await fetch(`${config.apiBase}/candidates/recent?limit=100`);
      if (response.ok) {
        const data = await response.json();
        const candidates = data.candidates || [];
        
        // Map to Pareto format
        const pareto: ParetoPoint[] = candidates.map((c: any) => ({
          id: c.candidate_id,
          sharpe: c.oos_sharpe_net,
          maxDrawdown: c.max_drawdown_net,
          cagr: c.oos_cagr_net,
          gatesPassed: c.gates_passed,
          displayName: c.display_name,
        }));
        
        setParetoData(pareto);
      }
    } catch (err) {
      // Silent fail
    }
  };

  // Calculate derived metrics
  const progress = evolutionState.maxGenerations > 0 
    ? (evolutionState.generation / evolutionState.maxGenerations) * 100 
    : 0;
    
  const cacheRate = evolutionState.totalEvaluations > 0 
    ? (evolutionState.cacheHits / evolutionState.totalEvaluations) * 100 
    : 0;
    
  const evalPerSec = performance?.evalPerSec || 0;
  const genomesPerMin = evalPerSec * 60;

  // Recent fitness trend (last 10 generations)
  const fitnessSparkline = useMemo(() => {
    return generationHistory.slice(-20).map(g => g.bestSharpe);
  }, [generationHistory]);

  // Pareto count by quality
  const paretoStats = useMemo(() => {
    const validated = paretoData.filter(p => p.gatesPassed).length;
    const research = paretoData.length - validated;
    const avgSharpe = paretoData.length > 0 
      ? paretoData.reduce((a, b) => a + b.sharpe, 0) / paretoData.length 
      : 0;
    return { validated, research, avgSharpe };
  }, [paretoData]);

  const isActive = evolutionState.status === 'running' || (currentCampaign && status === 'running');

  return (
    <div className="space-y-6">
      {/* Header with Status */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-terminal-muted bg-clip-text text-transparent">
              Evolution Monitor
            </h1>
            <StatusBadge status={evolutionState.status} />
          </div>
          <p className="text-terminal-muted mt-1 flex items-center gap-2">
            <span>Track genetic algorithm progress in real-time</span>
            <QuickTooltip termKey="genetic_algorithm" />
          </p>
          {evolutionState.campaignName && (
            <div className="flex items-center gap-2 mt-2 text-sm">
              <GitBranch className="w-4 h-4 text-accent-cyan" />
              <span className="text-accent-cyan font-medium">{evolutionState.campaignName}</span>
            </div>
          )}
        </div>
        
        <div className="flex items-center gap-3">
          {/* SSE Status */}
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs ${
            sseConnected ? 'bg-profit/10 text-profit' : 'bg-loss/10 text-loss'
          }`}>
            {sseConnected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
            {sseConnected ? 'Live' : 'Offline'}
          </div>
          
          <button 
            onClick={loadEvolutionState}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-terminal-surface border border-terminal-border hover:border-terminal-muted transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh
          </button>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="card-elevated relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-profit/5 to-transparent" />
        
        <div className="relative">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-4">
              <div className={`w-3 h-3 rounded-full ${
                isActive ? 'bg-profit animate-pulse' : 'bg-terminal-muted'
              }`} />
              <span className="font-medium flex items-center gap-2">
                Generation {evolutionState.generation} 
                <span className="text-terminal-muted">of</span> 
                {evolutionState.maxGenerations}
              </span>
            </div>
            <div className="flex items-center gap-4">
              {fitnessSparkline.length > 0 && (
                <div className="flex items-center gap-2">
                  <span className="text-xs text-terminal-muted">Fitness Trend:</span>
                  <Sparkline 
                    data={fitnessSparkline} 
                    width={80} 
                    height={24}
                    color="#00ff88"
                  />
                </div>
              )}
              <span className="font-mono text-lg font-bold text-profit">
                {progress.toFixed(1)}%
              </span>
            </div>
          </div>
          
          <div className="h-4 bg-terminal-bg rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-profit via-accent-cyan to-accent-purple rounded-full transition-all duration-500 relative"
              style={{ width: `${progress}%` }}
            >
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer" />
            </div>
          </div>
          
          <div className="flex items-center justify-between mt-3 text-sm">
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-2 text-terminal-muted">
                <Clock className="w-4 h-4" />
                <span>Elapsed: <span className="font-mono text-white">{evolutionState.elapsedTime || '--:--:--'}</span></span>
              </div>
              <div className="flex items-center gap-2 text-terminal-muted">
                <Gauge className="w-4 h-4" />
                <span>ETA: <span className="font-mono text-white">{evolutionState.eta || '--:--:--'}</span></span>
              </div>
            </div>
            <div className="flex items-center gap-2 text-terminal-muted">
              <Zap className="w-4 h-4 text-accent-yellow" />
              <span className="font-mono text-accent-yellow">{genomesPerMin.toFixed(1)}</span>
              <span>genomes/min</span>
            </div>
          </div>
        </div>
      </div>

      {/* Best Metrics Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricBox 
          label="Best Sharpe" 
          value={evolutionState.bestSharpe.toFixed(3)}
          icon={<Target className="w-5 h-5" />}
          color="profit"
          delta={generationHistory.length > 1 
            ? ((evolutionState.bestSharpe - generationHistory[0].bestSharpe) / generationHistory[0].bestSharpe * 100).toFixed(1) 
            : undefined}
        />
        <MetricBox 
          label="Best CAGR" 
          value={`${(evolutionState.bestCagr * 100).toFixed(1)}%`}
          icon={<TrendingUp className="w-5 h-5" />}
          color="cyan"
        />
        <MetricBox 
          label="Mean Sharpe" 
          value={evolutionState.meanSharpe.toFixed(3)}
          icon={<Activity className="w-5 h-5" />}
          color="white"
        />
        <MetricBox 
          label="Pareto Size" 
          value={evolutionState.paretoSize.toString()}
          icon={<Layers className="w-5 h-5" />}
          color="purple"
          subtitle={`${paretoStats.validated} validated`}
        />
      </div>

      {/* Performance Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <PerformanceCard
          icon={<Cpu className="w-5 h-5" />}
          label="Total Evaluations"
          value={evolutionState.totalEvaluations.toLocaleString()}
          color="cyan"
        />
        <PerformanceCard
          icon={<Database className="w-5 h-5" />}
          label="Cache Hits"
          value={evolutionState.cacheHits.toLocaleString()}
          color="purple"
          subtitle={`${cacheRate.toFixed(0)}% hit rate`}
        />
        <PerformanceCard
          icon={<Zap className="w-5 h-5" />}
          label="Eval/Second"
          value={evalPerSec.toFixed(2)}
          color="yellow"
        />
        <PerformanceCard
          icon={<BarChart3 className="w-5 h-5" />}
          label="Population"
          value={evolutionState.populationSize.toString()}
          color="white"
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Fitness Over Generations */}
        <div className="card-elevated">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-semibold text-lg flex items-center gap-2">
              Fitness Over Generations
              <QuickTooltip termKey="fitness" />
            </h2>
            <div className="flex items-center gap-2 text-xs text-terminal-muted">
              <span className="flex items-center gap-1">
                <div className="w-3 h-0.5 bg-profit rounded" />
                Best
              </span>
              <span className="flex items-center gap-1">
                <div className="w-3 h-0.5 bg-accent-cyan rounded" />
                Mean
              </span>
            </div>
          </div>
          <div className="h-[350px]">
            {generationHistory.length > 0 ? (
              <GenerationChart data={generationHistory} />
            ) : loading ? (
              <div className="flex items-center justify-center h-full">
                <RefreshCw className="w-8 h-8 animate-spin text-terminal-muted" />
              </div>
            ) : (
              <EmptyState 
                icon={<BarChart3 className="w-12 h-12" />}
                title="No Generation Data"
                subtitle="Start a campaign to see evolution progress"
              />
            )}
          </div>
          
          {/* Generation Stats Footer */}
          {generationHistory.length > 0 && (
            <div className="grid grid-cols-4 gap-4 mt-4 pt-4 border-t border-terminal-border">
              <div className="text-center">
                <div className="text-[10px] text-terminal-muted uppercase tracking-wider">Start</div>
                <div className="font-mono text-sm">{generationHistory[0]?.bestSharpe.toFixed(2)}</div>
              </div>
              <div className="text-center">
                <div className="text-[10px] text-terminal-muted uppercase tracking-wider">Current</div>
                <div className="font-mono text-sm text-profit">
                  {generationHistory[generationHistory.length - 1]?.bestSharpe.toFixed(2)}
                </div>
              </div>
              <div className="text-center">
                <div className="text-[10px] text-terminal-muted uppercase tracking-wider">Improvement</div>
                <div className="font-mono text-sm text-profit">
                  +{((generationHistory[generationHistory.length - 1]?.bestSharpe / generationHistory[0]?.bestSharpe - 1) * 100).toFixed(1)}%
                </div>
              </div>
              <div className="text-center">
                <div className="text-[10px] text-terminal-muted uppercase tracking-wider">Avg/Gen</div>
                <div className="font-mono text-sm">
                  {((generationHistory[generationHistory.length - 1]?.bestSharpe - generationHistory[0]?.bestSharpe) / generationHistory.length).toFixed(4)}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Pareto Frontier */}
        <div className="card-elevated">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-semibold text-lg flex items-center gap-2">
              Pareto Frontier
              <QuickTooltip termKey="pareto_frontier" />
            </h2>
            <div className="flex items-center gap-2 text-xs">
              <span className="px-2 py-0.5 rounded bg-profit/20 text-profit">{paretoStats.validated} validated</span>
              <span className="px-2 py-0.5 rounded bg-accent-cyan/20 text-accent-cyan">{paretoStats.research} research</span>
            </div>
          </div>
          <div className="h-[350px]">
            {paretoData.length > 0 ? (
              <ParetoScatter data={paretoData} height={350} />
            ) : loading ? (
              <div className="flex items-center justify-center h-full">
                <RefreshCw className="w-8 h-8 animate-spin text-terminal-muted" />
              </div>
            ) : (
              <EmptyState 
                icon={<Layers className="w-12 h-12" />}
                title="No Pareto Data"
                subtitle="Candidates will appear here as they're evaluated"
              />
            )}
          </div>
          
          {/* Pareto Stats Footer */}
          {paretoData.length > 0 && (
            <div className="grid grid-cols-3 gap-4 mt-4 pt-4 border-t border-terminal-border">
              <div className="text-center">
                <div className="text-[10px] text-terminal-muted uppercase tracking-wider">Best Sharpe</div>
                <div className="font-mono text-sm text-profit">
                  {Math.max(...paretoData.map(p => p.sharpe)).toFixed(3)}
                </div>
              </div>
              <div className="text-center">
                <div className="text-[10px] text-terminal-muted uppercase tracking-wider">Avg Sharpe</div>
                <div className="font-mono text-sm">
                  {paretoStats.avgSharpe.toFixed(3)}
                </div>
              </div>
              <div className="text-center">
                <div className="text-[10px] text-terminal-muted uppercase tracking-wider">Min MaxDD</div>
                <div className="font-mono text-sm text-loss">
                  {(Math.max(...paretoData.map(p => p.maxDrawdown)) * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// HELPER COMPONENTS
// =============================================================================

function StatusBadge({ status }: { status: EvolutionState['status'] }) {
  const config = {
    idle: { bg: 'bg-terminal-muted/20', text: 'text-terminal-muted', label: 'Idle' },
    running: { bg: 'bg-profit/20', text: 'text-profit', label: 'Running' },
    paused: { bg: 'bg-accent-yellow/20', text: 'text-accent-yellow', label: 'Paused' },
    completed: { bg: 'bg-accent-cyan/20', text: 'text-accent-cyan', label: 'Completed' },
    error: { bg: 'bg-loss/20', text: 'text-loss', label: 'Error' },
  };
  
  const { bg, text, label } = config[status];
  
  return (
    <span className={`px-3 py-1 rounded-full text-xs font-medium ${bg} ${text} flex items-center gap-1.5`}>
      {status === 'running' && <div className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" />}
      {label}
    </span>
  );
}

interface MetricBoxProps {
  label: string;
  value: string;
  icon: React.ReactNode;
  color: 'profit' | 'cyan' | 'purple' | 'yellow' | 'white';
  delta?: string;
  subtitle?: string;
}

function MetricBox({ label, value, icon, color, delta, subtitle }: MetricBoxProps) {
  const colors = {
    profit: 'text-profit',
    cyan: 'text-accent-cyan',
    purple: 'text-accent-purple',
    yellow: 'text-accent-yellow',
    white: 'text-white',
  };
  
  const bgColors = {
    profit: 'bg-profit/10',
    cyan: 'bg-accent-cyan/10',
    purple: 'bg-accent-purple/10',
    yellow: 'bg-accent-yellow/10',
    white: 'bg-terminal-surface',
  };
  
  return (
    <div className="card-elevated group hover:border-terminal-muted/50 transition-colors">
      <div className="flex items-start justify-between mb-2">
        <span className="text-xs text-terminal-muted uppercase tracking-wider">{label}</span>
        <div className={`p-1.5 rounded-lg ${bgColors[color]}`}>
          <span className={colors[color]}>{icon}</span>
        </div>
      </div>
      <div className={`font-mono font-bold text-2xl ${colors[color]}`}>
        {value}
      </div>
      {delta && (
        <div className="text-xs text-profit mt-1 flex items-center gap-1">
          <TrendingUp className="w-3 h-3" />
          +{delta}% since start
        </div>
      )}
      {subtitle && (
        <div className="text-xs text-terminal-muted mt-1">{subtitle}</div>
      )}
    </div>
  );
}

interface PerformanceCardProps {
  icon: React.ReactNode;
  label: string;
  value: string;
  color: 'cyan' | 'purple' | 'yellow' | 'white';
  subtitle?: string;
}

function PerformanceCard({ icon, label, value, color, subtitle }: PerformanceCardProps) {
  const colors = {
    cyan: 'text-accent-cyan',
    purple: 'text-accent-purple',
    yellow: 'text-accent-yellow',
    white: 'text-white',
  };
  
  const bgColors = {
    cyan: 'bg-accent-cyan/10',
    purple: 'bg-accent-purple/10',
    yellow: 'bg-accent-yellow/10',
    white: 'bg-terminal-surface',
  };
  
  return (
    <div className="card flex items-center gap-4">
      <div className={`w-10 h-10 rounded-lg ${bgColors[color]} flex items-center justify-center`}>
        <span className={colors[color]}>{icon}</span>
      </div>
      <div className="flex-1">
        <div className="text-xs text-terminal-muted uppercase tracking-wider">{label}</div>
        <div className="font-mono text-lg font-bold">{value}</div>
        {subtitle && <div className="text-xs text-terminal-muted">{subtitle}</div>}
      </div>
    </div>
  );
}

function EmptyState({ icon, title, subtitle }: { icon: React.ReactNode; title: string; subtitle: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-terminal-muted">
      <div className="opacity-30 mb-4">{icon}</div>
      <p className="font-medium">{title}</p>
      <p className="text-sm mt-1">{subtitle}</p>
    </div>
  );
}

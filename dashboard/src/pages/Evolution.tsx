import { useState } from 'react';
import { MetricCard } from '../components/ui/MetricCard';
import { GenerationChart } from '../components/charts/GenerationChart';
import { ParetoChart } from '../components/charts/ParetoChart';
import { 
  Play, 
  Pause, 
  RotateCcw,
  Zap,
  Clock,
  Cpu
} from 'lucide-react';

// Mock data
const mockEvolutionStats = {
  status: 'running' as const,
  generation: 32,
  maxGenerations: 50,
  populationSize: 100,
  totalEvaluations: 3200,
  cacheHits: 456,
  elapsedTime: '00:15:32',
  eta: '00:08:45',
  bestSharpe: 1.23,
  bestCagr: 0.15,
  meanSharpe: 0.78,
  paretoSize: 18,
};

const mockGenerationData = Array.from({ length: 32 }, (_, i) => ({
  generation: i,
  bestSharpe: 0.3 + Math.random() * 0.3 + i * 0.025,
  meanSharpe: 0.2 + Math.random() * 0.2 + i * 0.015,
  paretoSize: Math.floor(8 + Math.random() * 10 + i * 0.3),
}));

const mockParetoData = Array.from({ length: 50 }, (_, i) => ({
  id: `strategy_${i.toString().padStart(3, '0')}`,
  cagr: 0.05 + Math.random() * 0.15,
  sharpe: 0.5 + Math.random() * 1.0,
  maxDrawdown: -(0.05 + Math.random() * 0.1),
  paretoRank: Math.floor(Math.random() * 3),
}));

export function Evolution() {
  const [isRunning, setIsRunning] = useState(mockEvolutionStats.status === 'running');

  const progress = (mockEvolutionStats.generation / mockEvolutionStats.maxGenerations) * 100;

  return (
    <div className="space-y-6">
      {/* Header with Controls */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Evolution Monitor</h1>
          <p className="text-terminal-muted mt-1">Track genetic algorithm progress in real-time</p>
        </div>
        <div className="flex items-center gap-3">
          <button 
            onClick={() => setIsRunning(!isRunning)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
              isRunning 
                ? 'bg-loss/10 text-loss border border-loss/30 hover:bg-loss/20' 
                : 'bg-profit/10 text-profit border border-profit/30 hover:bg-profit/20'
            }`}
          >
            {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {isRunning ? 'Pause' : 'Resume'}
          </button>
          <button className="flex items-center gap-2 px-4 py-2 rounded-lg bg-terminal-border hover:bg-terminal-muted/30 transition-all">
            <RotateCcw className="w-4 h-4" />
            Reset
          </button>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="card">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className={`w-3 h-3 rounded-full ${isRunning ? 'bg-profit animate-pulse' : 'bg-terminal-muted'}`} />
            <span className="font-medium">
              Generation {mockEvolutionStats.generation} of {mockEvolutionStats.maxGenerations}
            </span>
          </div>
          <span className="font-mono text-sm text-terminal-muted">{progress.toFixed(1)}%</span>
        </div>
        <div className="h-3 bg-terminal-bg rounded-full overflow-hidden">
          <div 
            className="h-full bg-gradient-to-r from-profit to-accent-cyan rounded-full transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>
        <div className="flex items-center justify-between mt-3 text-sm text-terminal-muted">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4" />
            <span>Elapsed: {mockEvolutionStats.elapsedTime}</span>
          </div>
          <div className="flex items-center gap-2">
            <span>ETA: {mockEvolutionStats.eta}</span>
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="Best Sharpe"
          value={mockEvolutionStats.bestSharpe}
          format="ratio"
          icon={<Zap className="w-5 h-5 text-profit" />}
        />
        <MetricCard
          label="Best CAGR"
          value={mockEvolutionStats.bestCagr}
          format="percent"
        />
        <MetricCard
          label="Mean Sharpe"
          value={mockEvolutionStats.meanSharpe}
          format="ratio"
        />
        <MetricCard
          label="Pareto Size"
          value={mockEvolutionStats.paretoSize}
        />
      </div>

      {/* Performance Stats */}
      <div className="grid grid-cols-3 gap-4">
        <div className="card flex items-center gap-4">
          <div className="w-10 h-10 rounded-lg bg-accent-cyan/10 flex items-center justify-center">
            <Cpu className="w-5 h-5 text-accent-cyan" />
          </div>
          <div>
            <div className="metric-label">Evaluations</div>
            <div className="font-mono text-lg font-bold">{mockEvolutionStats.totalEvaluations.toLocaleString()}</div>
          </div>
        </div>
        <div className="card flex items-center gap-4">
          <div className="w-10 h-10 rounded-lg bg-accent-purple/10 flex items-center justify-center">
            <Zap className="w-5 h-5 text-accent-purple" />
          </div>
          <div>
            <div className="metric-label">Cache Hits</div>
            <div className="font-mono text-lg font-bold">{mockEvolutionStats.cacheHits}</div>
          </div>
        </div>
        <div className="card flex items-center gap-4">
          <div className="w-10 h-10 rounded-lg bg-profit/10 flex items-center justify-center">
            <span className="font-mono font-bold text-profit">
              {((mockEvolutionStats.cacheHits / mockEvolutionStats.totalEvaluations) * 100).toFixed(0)}%
            </span>
          </div>
          <div>
            <div className="metric-label">Cache Rate</div>
            <div className="font-mono text-lg font-bold text-profit">Efficient</div>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Generation Progress */}
        <div className="card-elevated">
          <h2 className="font-semibold text-lg mb-4">Fitness Over Generations</h2>
          <div className="h-[350px]">
            <GenerationChart data={mockGenerationData} />
          </div>
        </div>

        {/* Pareto Frontier */}
        <div className="card-elevated">
          <h2 className="font-semibold text-lg mb-4">Pareto Frontier (CAGR vs Sharpe)</h2>
          <div className="h-[350px]">
            <ParetoChart data={mockParetoData} />
          </div>
        </div>
      </div>
    </div>
  );
}



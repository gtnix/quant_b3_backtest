import { MetricCard } from '../components/ui/MetricCard';
import { EquityChart } from '../components/charts/EquityChart';
import { GenerationChart } from '../components/charts/GenerationChart';
import { 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  Target,
  Zap,
  Award
} from 'lucide-react';

// Mock data - will be replaced with Tauri commands
const mockMetrics = {
  totalReturn: 0.2534,
  sharpeRatio: 1.23,
  maxDrawdown: -0.0797,
  winRate: 0.5432,
  totalTrades: 156,
  activeCandidates: 25,
  currentGeneration: 48,
  bestCagr: 0.15,
};

const mockEquityData = Array.from({ length: 100 }, (_, i) => ({
  time: `2024-${String(Math.floor(i / 30) + 1).padStart(2, '0')}-${String((i % 30) + 1).padStart(2, '0')}`,
  value: 100000 * (1 + Math.random() * 0.5 - 0.1 + i * 0.003),
}));

const mockGenerationData = Array.from({ length: 50 }, (_, i) => ({
  generation: i,
  bestSharpe: 0.3 + Math.random() * 0.9 + i * 0.015,
  meanSharpe: 0.2 + Math.random() * 0.3 + i * 0.008,
  paretoSize: Math.floor(10 + Math.random() * 15 + i * 0.2),
}));

export function Dashboard() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Dashboard</h1>
          <p className="text-terminal-muted mt-1">Overview of system performance and evolution status</p>
        </div>
        <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-profit/10 border border-profit/30">
          <Zap className="w-5 h-5 text-profit" />
          <span className="font-mono text-profit">SCG Active</span>
        </div>
      </div>

      {/* KPI Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Total Return"
          value={mockMetrics.totalReturn}
          format="percent"
          change={0.0234}
          changeLabel="24h"
          icon={<TrendingUp className="w-5 h-5" />}
          size="lg"
        />
        <MetricCard
          label="Sharpe Ratio"
          value={mockMetrics.sharpeRatio}
          format="ratio"
          change={0.05}
          changeLabel="vs avg"
          icon={<Target className="w-5 h-5" />}
          size="lg"
        />
        <MetricCard
          label="Max Drawdown"
          value={mockMetrics.maxDrawdown}
          format="percent"
          icon={<TrendingDown className="w-5 h-5" />}
          size="lg"
        />
        <MetricCard
          label="Win Rate"
          value={mockMetrics.winRate}
          format="percent"
          icon={<Activity className="w-5 h-5" />}
          size="lg"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Equity Curve */}
        <div className="card-elevated">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-semibold text-lg">Equity Curve</h2>
            <span className="text-xs text-terminal-muted font-mono">BEST CANDIDATE</span>
          </div>
          <div className="h-[300px]">
            <EquityChart data={mockEquityData} />
          </div>
        </div>

        {/* Evolution Progress */}
        <div className="card-elevated">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-semibold text-lg">Evolution Progress</h2>
            <span className="text-xs text-terminal-muted font-mono">
              GEN {mockMetrics.currentGeneration}/50
            </span>
          </div>
          <div className="h-[300px]">
            <GenerationChart data={mockGenerationData} />
          </div>
        </div>
      </div>

      {/* Bottom Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="card flex items-center gap-4">
          <div className="w-12 h-12 rounded-lg bg-accent-cyan/10 flex items-center justify-center">
            <Activity className="w-6 h-6 text-accent-cyan" />
          </div>
          <div>
            <div className="metric-label">Total Trades</div>
            <div className="font-mono text-xl font-bold">{mockMetrics.totalTrades}</div>
          </div>
        </div>
        
        <div className="card flex items-center gap-4">
          <div className="w-12 h-12 rounded-lg bg-accent-purple/10 flex items-center justify-center">
            <Award className="w-6 h-6 text-accent-purple" />
          </div>
          <div>
            <div className="metric-label">Hall of Fame</div>
            <div className="font-mono text-xl font-bold">{mockMetrics.activeCandidates} strategies</div>
          </div>
        </div>
        
        <div className="card flex items-center gap-4">
          <div className="w-12 h-12 rounded-lg bg-profit/10 flex items-center justify-center">
            <TrendingUp className="w-6 h-6 text-profit" />
          </div>
          <div>
            <div className="metric-label">Best CAGR</div>
            <div className="font-mono text-xl font-bold text-profit">
              {(mockMetrics.bestCagr * 100).toFixed(2)}%
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}



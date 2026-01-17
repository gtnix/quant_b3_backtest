import { useState, useEffect } from 'react';
import { MetricCard } from '../components/ui/MetricCard';
import { ReturnDistribution } from '../components/charts/ReturnDistribution';
import { DistributionFan } from '../components/charts/DistributionFan';
import { QuickTooltip } from '../components/ui/TooltipInfo';
import { useDataStore } from '../stores/dataStore';
import {
  Shuffle,
  RefreshCw,
  AlertTriangle,
  Target,
  TrendingDown,
  Activity,
  Settings,
  Play,
  BarChart3,
} from 'lucide-react';

export function MonteCarlo() {
  const [numSimulations, setNumSimulations] = useState(1000);
  const [blockSize, setBlockSize] = useState(5);
  const [activeTab, setActiveTab] = useState<'equity' | 'sharpe' | 'cagr' | 'maxdd'>('equity');

  const {
    selectedCandidate,
    monteCarloResult,
    isLoading,
    error,
    runMonteCarlo,
  } = useDataStore();

  // No candidate selected
  if (!selectedCandidate) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-6">
        <Shuffle className="w-16 h-16 text-terminal-muted" />
        <div className="text-center">
          <h2 className="text-xl font-semibold mb-2">No Candidate Selected</h2>
          <p className="text-terminal-muted">
            Select a candidate from the Candidates page to run Monte Carlo simulation.
          </p>
        </div>
      </div>
    );
  }

  const handleRunSimulation = () => {
    runMonteCarlo(selectedCandidate.candidate_id, numSimulations, blockSize);
  };

  // Loading
  if (isLoading && !monteCarloResult) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-4">
        <RefreshCw className="w-12 h-12 animate-spin text-terminal-muted" />
        <p className="text-terminal-muted">Running {numSimulations} simulations...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold inline-flex items-center">Monte Carlo Simulation<QuickTooltip termKey="monte_carlo" size="md" /></h1>
          <p className="text-terminal-muted mt-1">
            <span className="inline-flex items-center">Bootstrap<QuickTooltip termKey="bootstrap" /></span> confidence intervals for{' '}
            <span className="text-accent-cyan font-mono">
              {selectedCandidate.display_name}
            </span>
          </p>
        </div>
      </div>

      {/* Configuration */}
      <div className="card flex items-center gap-6 flex-wrap">
        <div className="flex items-center gap-2">
          <Settings className="w-4 h-4 text-terminal-muted" />
          <span className="text-sm text-terminal-muted">Parameters:</span>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-sm text-terminal-muted">Simulations:</label>
          <select
            value={numSimulations}
            onChange={(e) => setNumSimulations(Number(e.target.value))}
            className="px-2 py-1 bg-terminal-bg border border-terminal-border rounded text-sm"
          >
            <option value={100}>100</option>
            <option value={500}>500</option>
            <option value={1000}>1,000</option>
            <option value={2000}>2,000</option>
            <option value={5000}>5,000</option>
          </select>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-sm text-terminal-muted inline-flex items-center">Block Size<QuickTooltip termKey="block_size" /></label>
          <select
            value={blockSize}
            onChange={(e) => setBlockSize(Number(e.target.value))}
            className="px-2 py-1 bg-terminal-bg border border-terminal-border rounded text-sm"
          >
            <option value={1}>1 day (IID)</option>
            <option value={5}>5 days</option>
            <option value={10}>10 days</option>
            <option value={21}>21 days (1 month)</option>
          </select>
        </div>
        <button
          onClick={handleRunSimulation}
          disabled={isLoading}
          className="flex items-center gap-2 px-4 py-2 bg-profit text-black font-medium rounded-lg hover:bg-profit/90 transition-colors disabled:opacity-50"
        >
          <Play className="w-4 h-4" />
          Run Simulation
        </button>
      </div>

      {/* Results */}
      {monteCarloResult ? (
        <>
          {/* Key Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
            <MetricCard
              label="Sharpe (Median)"
              value={monteCarloResult.sharpe_distribution.p50}
              format="ratio"
              icon={<Target className="w-4 h-4" />}
            />
            <MetricCard
              label="Sharpe (P5)"
              value={monteCarloResult.sharpe_distribution.p5}
              format="ratio"
            />
            <MetricCard
              label="CAGR (Median)"
              value={monteCarloResult.cagr_distribution.p50 * 100}
              format="percent"
              icon={<Activity className="w-4 h-4 text-profit" />}
            />
            <MetricCard
              label="CAGR (P5)"
              value={monteCarloResult.cagr_distribution.p5 * 100}
              format="percent"
            />
            <MetricCard
              label="MaxDD (Median)"
              value={monteCarloResult.max_dd_distribution.p50 * 100}
              format="percent"
              icon={<TrendingDown className="w-4 h-4 text-loss" />}
            />
            <MetricCard
              label="MaxDD (P95)"
              value={monteCarloResult.max_dd_distribution.p95 * 100}
              format="percent"
            />
          </div>

          {/* Confidence Intervals */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <ConfidenceIntervalCard
              title="Sharpe Ratio"
              distribution={monteCarloResult.sharpe_distribution}
              format={(v) => v.toFixed(2)}
            />
            <ConfidenceIntervalCard
              title="CAGR"
              distribution={monteCarloResult.cagr_distribution}
              format={(v) => `${(v * 100).toFixed(1)}%`}
            />
            <ConfidenceIntervalCard
              title="Max Drawdown"
              distribution={monteCarloResult.max_dd_distribution}
              format={(v) => `${(v * 100).toFixed(1)}%`}
            />
          </div>

          {/* Tabs */}
          <div className="flex items-center gap-4 border-b border-terminal-border">
            {[
              { key: 'equity', label: 'Equity Paths', icon: BarChart3 },
              { key: 'sharpe', label: 'Sharpe Distribution', icon: Target },
              { key: 'cagr', label: 'CAGR Distribution', icon: Activity },
              { key: 'maxdd', label: 'MaxDD Distribution', icon: TrendingDown },
            ].map(({ key, label, icon: Icon }) => (
              <button
                key={key}
                onClick={() => setActiveTab(key as typeof activeTab)}
                className={`flex items-center gap-2 pb-3 px-1 font-medium transition-colors relative ${
                  activeTab === key ? 'text-profit' : 'text-terminal-muted hover:text-white'
                }`}
              >
                <Icon className="w-4 h-4" />
                {label}
                {activeTab === key && (
                  <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-profit" />
                )}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          {activeTab === 'equity' && (
            <div className="card-elevated">
              <h3 className="font-semibold text-lg mb-4">
                Equity Confidence Bands ({monteCarloResult.num_simulations} simulations)
              </h3>
              <div className="h-[400px]">
                <DistributionFan confidenceBands={monteCarloResult.confidence_bands} />
              </div>
            </div>
          )}

          {activeTab === 'sharpe' && (
            <div className="card-elevated">
              <h3 className="font-semibold text-lg mb-4">Sharpe Ratio Distribution</h3>
              <div className="h-[350px]">
                <DistributionChart distribution={monteCarloResult.sharpe_distribution} />
              </div>
            </div>
          )}

          {activeTab === 'cagr' && (
            <div className="card-elevated">
              <h3 className="font-semibold text-lg mb-4">CAGR Distribution</h3>
              <div className="h-[350px]">
                <DistributionChart distribution={monteCarloResult.cagr_distribution} multiplier={100} suffix="%" />
              </div>
            </div>
          )}

          {activeTab === 'maxdd' && (
            <div className="card-elevated">
              <h3 className="font-semibold text-lg mb-4">Max Drawdown Distribution</h3>
              <div className="h-[350px]">
                <DistributionChart distribution={monteCarloResult.max_dd_distribution} multiplier={100} suffix="%" />
              </div>
            </div>
          )}
        </>
      ) : (
        <div className="flex flex-col items-center justify-center py-16 space-y-4">
          <Shuffle className="w-12 h-12 text-terminal-muted" />
          <div className="text-center">
            <h3 className="font-semibold mb-2">No Simulation Results</h3>
            <p className="text-terminal-muted text-sm">
              Configure parameters and click "Run Simulation" to generate confidence intervals.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

// Helper Components

interface DistributionStats {
  mean: number;
  std: number;
  p5: number;
  p25: number;
  p50: number;
  p75: number;
  p95: number;
  histogram: number[];
  histogram_bins: number[];
}

function ConfidenceIntervalCard({
  title,
  distribution,
  format,
}: {
  title: string;
  distribution: DistributionStats;
  format: (v: number) => string;
}) {
  return (
    <div className="card">
      <div className="font-medium mb-3 inline-flex items-center">{title}<QuickTooltip termKey="confidence_interval" /></div>
      <div className="space-y-2 text-sm">
        <div className="flex justify-between">
          <span className="text-terminal-muted inline-flex items-center">P5<QuickTooltip termKey="percentile_p5" /> (worst case)</span>
          <span className="font-mono text-loss">{format(distribution.p5)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-terminal-muted">P25</span>
          <span className="font-mono">{format(distribution.p25)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-terminal-muted inline-flex items-center">Median<QuickTooltip termKey="percentile_p50" /> (P50)</span>
          <span className="font-mono font-bold">{format(distribution.p50)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-terminal-muted">P75</span>
          <span className="font-mono">{format(distribution.p75)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-terminal-muted inline-flex items-center">P95<QuickTooltip termKey="percentile_p95" /> (best case)</span>
          <span className="font-mono text-profit">{format(distribution.p95)}</span>
        </div>
        <div className="pt-2 border-t border-terminal-border/50">
          <div className="flex justify-between">
            <span className="text-terminal-muted">Mean ± Std</span>
            <span className="font-mono text-xs">
              {format(distribution.mean)} ± {format(distribution.std)}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

function DistributionChart({
  distribution,
  multiplier = 1,
  suffix = '',
}: {
  distribution: DistributionStats;
  multiplier?: number;
  suffix?: string;
}) {
  const data = distribution.histogram.map((count, i) => ({
    bin: distribution.histogram_bins[i] * multiplier,
    count,
    pct: (count / distribution.histogram.reduce((a, b) => a + b, 0)) * 100,
  }));

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center gap-6 mb-4 text-xs">
        <div>
          <span className="text-terminal-muted">Mean:</span>
          <span className="ml-1 font-mono">
            {(distribution.mean * multiplier).toFixed(2)}{suffix}
          </span>
        </div>
        <div>
          <span className="text-terminal-muted">Std:</span>
          <span className="ml-1 font-mono">
            {(distribution.std * multiplier).toFixed(2)}{suffix}
          </span>
        </div>
        <div>
          <span className="text-terminal-muted">95% CI:</span>
          <span className="ml-1 font-mono">
            [{(distribution.p5 * multiplier).toFixed(2)}, {(distribution.p95 * multiplier).toFixed(2)}]{suffix}
          </span>
        </div>
      </div>
      <div className="flex-1 flex items-end gap-1">
        {data.map((d, i) => (
          <div
            key={i}
            className="flex-1 bg-profit/70 rounded-t transition-all hover:bg-profit"
            style={{
              height: `${d.pct * 5}%`,
              minHeight: d.count > 0 ? '2px' : '0',
            }}
            title={`${d.bin.toFixed(2)}${suffix}: ${d.count} (${d.pct.toFixed(1)}%)`}
          />
        ))}
      </div>
      <div className="flex justify-between text-xs text-terminal-muted mt-2">
        <span>{(distribution.histogram_bins[0] * multiplier).toFixed(2)}{suffix}</span>
        <span>{(distribution.histogram_bins[distribution.histogram_bins.length - 1] * multiplier).toFixed(2)}{suffix}</span>
      </div>
    </div>
  );
}


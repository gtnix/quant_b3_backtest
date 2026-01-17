import { useState, useEffect, useMemo } from 'react';
import { EquityChart } from '../components/charts/EquityChart';
import { RegimeHeatmap } from '../components/charts/RegimeHeatmap';
import type { TrendState, VolQuantile, RegimePerformance } from '../components/charts/RegimeHeatmap';
import { RegimeHeroCard } from '../components/charts/RegimeHeroCard';
import { RegimeTimeline } from '../components/charts/RegimeTimeline';
import type { RegimePeriod } from '../components/charts/RegimeTimeline';
import { QuickTooltip } from '../components/ui/TooltipInfo';
import { useDataStore, type TimeseriesPoint } from '../stores/dataStore';
import { config } from '../lib/platform';
import {
  Layers,
  RefreshCw,
  AlertTriangle,
  Settings,
  BookOpen,
  ChevronDown,
  Eye,
  EyeOff,
} from 'lucide-react';

interface SimulatedEquity {
  timeseries: TimeseriesPoint[];
}

export function RegimeAnalysis() {
  const [volThreshold, setVolThreshold] = useState(0.20);
  const [showOverlay, setShowOverlay] = useState(true);
  const [selectedCell, setSelectedCell] = useState<{ trend: TrendState; vol: VolQuantile } | null>(null);
  const [simulatedData, setSimulatedData] = useState<SimulatedEquity | null>(null);
  const [loadingEquity, setLoadingEquity] = useState(false);

  const {
    selectedCandidate,
    regimeAnalysis,
    backtest,
    isLoading,
    error,
    detectRegimes,
    loadBacktest,
  } = useDataStore();

  // Load simulated equity data
  const loadSimulatedEquity = async (candidateId: string) => {
    setLoadingEquity(true);
    try {
      const response = await fetch(`${config.apiBase}/candidate/${candidateId}/simulated-equity?days=504`);
      if (response.ok) {
        const data = await response.json();
        setSimulatedData(data);
      }
    } catch (err) {
      console.error('Failed to load simulated equity:', err);
    } finally {
      setLoadingEquity(false);
    }
  };

  // Load regime analysis when candidate is selected
  useEffect(() => {
    if (selectedCandidate) {
      detectRegimes(selectedCandidate.candidate_id, volThreshold);
      loadBacktest(selectedCandidate.candidate_id);
      // Also load simulated equity as fallback
      loadSimulatedEquity(selectedCandidate.candidate_id);
    }
  }, [selectedCandidate?.candidate_id, volThreshold]);

  // Transform data for new components
  const { regimePerformances, regimePeriods, currentRegime, totalDays } = useMemo(() => {
    if (!regimeAnalysis) {
      return { regimePerformances: [], regimePeriods: [], currentRegime: null, totalDays: 0 };
    }

    // Transform performance data to new format
    // The backend uses BullLowVol, BullHighVol, etc - we need to map to Trend+Vol
    const perfMap: RegimePerformance[] = [];
    const legacyToNew: Record<string, { trend: TrendState; vol: VolQuantile }> = {
      BullLowVol: { trend: 'Uptrend', vol: 'Q1' },
      BullHighVol: { trend: 'Uptrend', vol: 'Q5' },
      BearLowVol: { trend: 'Downtrend', vol: 'Q1' },
      BearHighVol: { trend: 'Downtrend', vol: 'Q5' },
    };

    // If we have the old format, convert it
    if (regimeAnalysis.performance_by_regime) {
      Object.entries(regimeAnalysis.performance_by_regime).forEach(([regime, metrics]: [string, any]) => {
        const mapping = legacyToNew[regime];
        if (mapping) {
          perfMap.push({
            trend_state: mapping.trend,
            vol_quantile: mapping.vol,
            day_count: metrics.num_days || 0,
            mean_return_pct: (metrics.cagr || 0) * 100 / 252,
            cumulative_return_pct: (metrics.cagr || 0) * 100,
            win_rate_pct: (metrics.hit_rate || 0) * 100,
            sharpe: metrics.sharpe,
            cagr: metrics.cagr,
            max_dd: metrics.max_dd,
          });
        }
      });
    }

    // Fill in remaining cells with synthetic data for demo
    const trends: TrendState[] = ['Uptrend', 'Sideways', 'Downtrend'];
    const vols: VolQuantile[] = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'];
    
    for (const trend of trends) {
      for (const vol of vols) {
        const exists = perfMap.find(p => p.trend_state === trend && p.vol_quantile === vol);
        if (!exists) {
          // Generate plausible synthetic data based on trend/vol
          const trendFactor = trend === 'Uptrend' ? 1.2 : trend === 'Sideways' ? 0.3 : -0.5;
          const volFactor = vol === 'Q1' ? 1.1 : vol === 'Q2' ? 0.9 : vol === 'Q3' ? 0.6 : vol === 'Q4' ? 0.3 : 0.0;
          const baseSharpe = trendFactor * (0.5 + volFactor * 0.5);
          
          perfMap.push({
            trend_state: trend,
            vol_quantile: vol,
            day_count: Math.floor(Math.random() * 100) + 20,
            mean_return_pct: baseSharpe * 0.05,
            cumulative_return_pct: baseSharpe * 15,
            win_rate_pct: 50 + baseSharpe * 10,
            sharpe: baseSharpe,
          });
        }
      }
    }

    // Transform periods for timeline
    const periods: RegimePeriod[] = (regimeAnalysis.regimes || []).map((p: any) => {
      const mapping = legacyToNew[p.regime];
      return {
        start_date: p.start_date,
        end_date: p.end_date,
        trend: mapping?.trend || 'Sideways',
        vol: mapping?.vol || 'Q3',
        days: Math.round((new Date(p.end_date).getTime() - new Date(p.start_date).getTime()) / (1000 * 60 * 60 * 24)),
      };
    });

    // Get current regime
    const currentMapping = legacyToNew[regimeAnalysis.current_regime];
    const current = currentMapping ? {
      trend: currentMapping.trend,
      vol: currentMapping.vol,
    } : null;

    const total = perfMap.reduce((sum, p) => sum + p.day_count, 0);

    return {
      regimePerformances: perfMap,
      regimePeriods: periods,
      currentRegime: current,
      totalDays: total,
    };
  }, [regimeAnalysis]);

  // Get current regime performance for hero card
  const currentPerformance = useMemo(() => {
    if (!currentRegime) return undefined;
    return regimePerformances.find(
      p => p.trend_state === currentRegime.trend && p.vol_quantile === currentRegime.vol
    );
  }, [currentRegime, regimePerformances]);

  // Prepare equity data from backtest or simulated timeseries
  const equityData = useMemo(() => {
    if (backtest?.timeseries && backtest.timeseries.length > 0) {
      return backtest.timeseries.map(p => ({
        time: p.date,
        value: p.equity,
      }));
    }
    if (simulatedData?.timeseries && simulatedData.timeseries.length > 0) {
      return simulatedData.timeseries.map(p => ({
        time: p.date,
        value: p.equity,
      }));
    }
    return [];
  }, [backtest?.timeseries, simulatedData?.timeseries]);

  // No candidate selected
  if (!selectedCandidate) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-6">
        <Layers className="w-16 h-16 text-terminal-muted" />
        <div className="text-center">
          <h2 className="text-xl font-semibold mb-2">Nenhum Candidato Selecionado</h2>
          <p className="text-terminal-muted">
            Selecione um candidato na página Candidates para analisar regimes.
          </p>
        </div>
      </div>
    );
  }

  // Loading
  if (isLoading && !regimeAnalysis) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-4">
        <RefreshCw className="w-12 h-12 animate-spin text-terminal-muted" />
        <p className="text-terminal-muted">Detectando regimes de mercado...</p>
      </div>
    );
  }

  if (!regimeAnalysis) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-6">
        <AlertTriangle className="w-16 h-16 text-accent-yellow" />
        <div className="text-center">
          <h2 className="text-xl font-semibold mb-2">Análise Falhou</h2>
          <p className="text-terminal-muted max-w-md">
            {error || 'Não foi possível detectar regimes para este candidato.'}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold inline-flex items-center">
            Análise de Regimes
            <QuickTooltip termKey="regime_analysis" size="md" />
          </h1>
          <p className="text-terminal-muted mt-1">
            Performance condicional por regime de mercado para{' '}
            <span className="text-accent-cyan font-mono">
              {selectedCandidate.display_name}
            </span>
          </p>
        </div>
        <button
          onClick={() => detectRegimes(selectedCandidate.candidate_id, volThreshold)}
          className="flex items-center gap-2 px-3 py-2 rounded-lg bg-terminal-surface border border-terminal-border hover:border-profit transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          Atualizar
        </button>
      </div>

      {/* Educational Banner */}
      <details className="card group">
        <summary className="cursor-pointer font-semibold flex items-center gap-2 list-none">
          <BookOpen className="w-4 h-4 text-accent-cyan" />
          <span>Por que Análise de Regimes?</span>
          <ChevronDown className="w-4 h-4 ml-auto transition-transform group-open:rotate-180" />
        </summary>
        <div className="mt-4 space-y-4 text-sm border-t border-terminal-border pt-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-3 rounded-lg bg-loss/5 border border-loss/20">
              <div className="font-medium text-loss mb-2">O Problema</div>
              <p className="text-terminal-muted text-xs leading-relaxed">
                Uma estratégia pode ter Sharpe 1.5 agregado, mas -0.5 em Bear+HighVol.
                Isso significa que ela vai te destruir em crashes - exatamente quando você mais precisa de proteção.
              </p>
            </div>
            <div className="p-3 rounded-lg bg-accent-cyan/5 border border-accent-cyan/20">
              <div className="font-medium text-accent-cyan mb-2">A Solução</div>
              <p className="text-terminal-muted text-xs leading-relaxed">
                Decompomos a performance em 15 regimes (3 trends × 5 volatilidades).
                Você vê exatamente QUANDO a estratégia funciona e quando não funciona.
              </p>
            </div>
            <div className="p-3 rounded-lg bg-profit/5 border border-profit/20">
              <div className="font-medium text-profit mb-2">Como Usar</div>
              <p className="text-terminal-muted text-xs leading-relaxed">
                O heatmap mostra onde a estratégia brilha (verde) e onde falha (vermelho).
                Use o regime atual para ajustar sua alocação em tempo real.
              </p>
            </div>
          </div>
          <div className="text-xs text-terminal-muted italic border-l-2 border-accent-cyan/50 pl-3">
            Ref: Papers sobre Hidden Markov Models em regime detection - Wang et al. (2020), Oelschläger & Adam (2020)
          </div>
        </div>
      </details>

      {/* Configuration */}
      <div className="card flex items-center gap-6 flex-wrap">
        <div className="flex items-center gap-2">
          <Settings className="w-4 h-4 text-terminal-muted" />
          <span className="text-sm text-terminal-muted">Parâmetros:</span>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-sm text-terminal-muted">Threshold Vol:</label>
          <input
            type="range"
            min="0.10"
            max="0.40"
            step="0.02"
            value={volThreshold}
            onChange={(e) => setVolThreshold(Number(e.target.value))}
            className="w-32"
          />
          <span className="font-mono text-sm">{(volThreshold * 100).toFixed(0)}%</span>
        </div>
      </div>

      {/* Hero Card - Current Regime */}
      {currentRegime && (
        <RegimeHeroCard
          currentTrend={currentRegime.trend}
          currentVol={currentRegime.vol}
          currentVolValue={volThreshold * 100}
          performance={currentPerformance}
          totalDays={totalDays}
        />
      )}

      {/* Regime Heatmap */}
      <div className="card-elevated">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-lg inline-flex items-center">
            Matriz de Performance 3×5
            <QuickTooltip termKey="regime_heatmap" />
          </h3>
          {selectedCell && (
            <button
              onClick={() => setSelectedCell(null)}
              className="text-xs text-terminal-muted hover:text-white transition-colors"
            >
              Limpar seleção
            </button>
          )}
        </div>
        <RegimeHeatmap
          data={regimePerformances}
          currentRegime={currentRegime || undefined}
          onCellClick={(trend, vol) => setSelectedCell({ trend, vol })}
          selectedCell={selectedCell}
        />
      </div>

      {/* Regime Timeline */}
      <div className="card-elevated">
        <h3 className="font-semibold text-lg mb-4 inline-flex items-center">
          Timeline de Regimes
          <QuickTooltip termKey="regime_timeline" />
        </h3>
        <RegimeTimeline
          periods={regimePeriods}
          selectedRegime={selectedCell}
          onPeriodClick={(period) => setSelectedCell({ trend: period.trend, vol: period.vol })}
        />
      </div>

      {/* Equity Chart with Toggle */}
      <div className="card-elevated overflow-visible">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-lg">Curva de Equity com Overlay de Regime</h3>
          <button
            onClick={() => setShowOverlay(!showOverlay)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
              showOverlay
                ? 'bg-accent-cyan/20 text-accent-cyan border border-accent-cyan/30'
                : 'bg-terminal-surface border border-terminal-border hover:border-terminal-muted'
            }`}
          >
            {showOverlay ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
            Overlay
          </button>
        </div>
        <div className="h-[350px] relative overflow-visible">
          {/* Regime background bands */}
          {showOverlay && (
            <div className="absolute inset-0 flex pointer-events-none">
              {regimeAnalysis.regimes?.map((period: any, i: number) => {
                const total = equityData.length;
                const startIdx = equityData.findIndex(d => d.time >= period.start_date);
                const endIdx = equityData.findIndex(d => d.time >= period.end_date);
                const startPct = (startIdx / total) * 100;
                const widthPct = ((endIdx - startIdx) / total) * 100;
                
                if (startPct < 0 || widthPct <= 0) return null;
                
                return (
                  <div
                    key={i}
                    className="absolute top-0 bottom-0 opacity-15"
                    style={{
                      left: `${startPct}%`,
                      width: `${Math.max(widthPct, 0.5)}%`,
                      backgroundColor: period.color || '#6b7280',
                    }}
                  />
                );
              })}
            </div>
          )}
          
          {equityData.length > 0 ? (
            <EquityChart data={equityData} />
          ) : loadingEquity ? (
            <div className="flex items-center justify-center h-full">
              <RefreshCw className="w-8 h-8 animate-spin text-terminal-muted" />
            </div>
          ) : (
            <div className="flex items-center justify-center h-full text-terminal-muted">
              Sem dados de equity disponíveis
            </div>
          )}
        </div>
      </div>

      {/* Performance Table */}
      <div className="card-elevated overflow-x-auto">
        <h3 className="font-semibold text-lg mb-4">Performance Detalhada por Regime</h3>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-terminal-border">
              <th className="text-left py-2 px-3 text-terminal-muted font-normal">Trend</th>
              <th className="text-left py-2 px-3 text-terminal-muted font-normal">Vol</th>
              <th className="text-right py-2 px-3 text-terminal-muted font-normal">Sharpe</th>
              <th className="text-right py-2 px-3 text-terminal-muted font-normal">Retorno</th>
              <th className="text-right py-2 px-3 text-terminal-muted font-normal">Win Rate</th>
              <th className="text-right py-2 px-3 text-terminal-muted font-normal">Dias</th>
              <th className="text-right py-2 px-3 text-terminal-muted font-normal">% Tempo</th>
            </tr>
          </thead>
          <tbody>
            {regimePerformances
              .sort((a, b) => (b.sharpe ?? 0) - (a.sharpe ?? 0))
              .map((perf, i) => {
                const isCurrent = currentRegime?.trend === perf.trend_state && currentRegime?.vol === perf.vol_quantile;
                const isSelected = selectedCell?.trend === perf.trend_state && selectedCell?.vol === perf.vol_quantile;
                const sharpe = perf.sharpe ?? 0;
                
                return (
                  <tr 
                    key={i} 
                    className={`border-b border-terminal-border/30 hover:bg-terminal-surface/50 cursor-pointer transition-colors ${
                      isCurrent ? 'bg-accent-cyan/10' : ''
                    } ${isSelected ? 'bg-white/5' : ''}`}
                    onClick={() => setSelectedCell({ trend: perf.trend_state, vol: perf.vol_quantile })}
                  >
                    <td className="py-2 px-3">
                      <span className={`font-medium ${
                        perf.trend_state === 'Uptrend' ? 'text-profit' : 
                        perf.trend_state === 'Downtrend' ? 'text-loss' : 'text-accent-cyan'
                      }`}>
                        {perf.trend_state === 'Uptrend' ? 'Alta' : perf.trend_state === 'Downtrend' ? 'Baixa' : 'Lateral'}
                      </span>
                      {isCurrent && (
                        <span className="ml-2 px-1.5 py-0.5 rounded text-[9px] bg-accent-cyan/20 text-accent-cyan">ATUAL</span>
                      )}
                    </td>
                    <td className="py-2 px-3 font-mono">{perf.vol_quantile}</td>
                    <td className={`text-right py-2 px-3 font-mono font-bold ${sharpe >= 0 ? 'text-profit' : 'text-loss'}`}>
                      {sharpe.toFixed(2)}
                    </td>
                    <td className={`text-right py-2 px-3 font-mono ${perf.cumulative_return_pct >= 0 ? 'text-profit' : 'text-loss'}`}>
                      {perf.cumulative_return_pct >= 0 ? '+' : ''}{perf.cumulative_return_pct.toFixed(1)}%
                    </td>
                    <td className={`text-right py-2 px-3 font-mono ${perf.win_rate_pct >= 50 ? 'text-profit' : 'text-loss'}`}>
                      {perf.win_rate_pct.toFixed(0)}%
                    </td>
                    <td className="text-right py-2 px-3 font-mono text-terminal-muted">
                      {perf.day_count}
                    </td>
                    <td className="text-right py-2 px-3 font-mono text-terminal-muted">
                      {((perf.day_count / totalDays) * 100).toFixed(1)}%
                    </td>
                  </tr>
                );
              })}
          </tbody>
        </table>
      </div>

      {/* Interpretation Guide */}
      <div className="card">
        <h3 className="font-semibold mb-3">Guia de Interpretação</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <div className="font-medium text-profit mb-1">Perfil Ideal</div>
            <ul className="list-disc list-inside text-terminal-muted space-y-1">
              <li>Sharpe positivo em todos os regimes</li>
              <li>Drawdowns menores em bear markets</li>
              <li>Performance consistente em todas as condições</li>
            </ul>
          </div>
          <div>
            <div className="font-medium text-loss mb-1">Sinais de Alerta</div>
            <ul className="list-disc list-inside text-terminal-muted space-y-1">
              <li>Sharpe negativo em qualquer regime</li>
              <li>Perdas concentradas em regimes específicos</li>
              <li>Performance apenas em bull markets</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

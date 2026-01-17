import { useState, useEffect } from 'react';
import { MetricCard } from '../components/ui/MetricCard';
import { ReturnDistribution } from '../components/charts/ReturnDistribution';
import { MonthlyHeatmap } from '../components/charts/MonthlyHeatmap';
import { RollingMetrics } from '../components/charts/RollingMetrics';
import { VaRGauge } from '../components/charts/VaRGauge';
import { QuickTooltip } from '../components/ui/TooltipInfo';
import { useDataStore } from '../stores/dataStore';
import {
  Shield,
  TrendingDown,
  Activity,
  BarChart3,
  RefreshCw,
  AlertTriangle,
  Target,
  Zap,
  Calendar,
  ChevronDown,
} from 'lucide-react';

export function RiskAnalytics() {
  const [activeTab, setActiveTab] = useState<'var' | 'distribution' | 'rolling' | 'monthly'>('var');

  const {
    selectedCandidate,
    riskMetrics,
    isLoading,
    error,
    loadRiskMetrics,
  } = useDataStore();

  // Load risk metrics when candidate is selected
  useEffect(() => {
    if (selectedCandidate) {
      loadRiskMetrics(selectedCandidate.candidate_id);
    }
  }, [selectedCandidate?.candidate_id]);

  // Nenhum candidato selecionado
  if (!selectedCandidate) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-6">
        <Shield className="w-16 h-16 text-terminal-muted" />
        <div className="text-center">
          <h2 className="text-xl font-semibold mb-2">Nenhum Candidato Selecionado</h2>
          <p className="text-terminal-muted">
            Selecione um candidato na página de Candidatos para ver análise de risco.
          </p>
        </div>
      </div>
    );
  }

  // Estado de carregamento
  if (isLoading && !riskMetrics) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-4">
        <RefreshCw className="w-12 h-12 animate-spin text-terminal-muted" />
        <p className="text-terminal-muted">Calculando métricas de risco...</p>
      </div>
    );
  }

  if (!riskMetrics) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-6">
        <AlertTriangle className="w-16 h-16 text-accent-yellow" />
        <div className="text-center">
          <h2 className="text-xl font-semibold mb-2">Dados de Risco Indisponíveis</h2>
          <p className="text-terminal-muted max-w-md">
            {error || 'Não foi possível calcular métricas de risco para este candidato.'}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Cabeçalho */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Análise de Risco</h1>
          <p className="text-terminal-muted mt-1">
            Análise de risco padrão institucional para{' '}
            <span className="text-accent-cyan font-mono">
              {selectedCandidate.display_name}
            </span>
          </p>
        </div>
        <button
          onClick={() => loadRiskMetrics(selectedCandidate.candidate_id)}
          className="flex items-center gap-2 px-3 py-2 rounded-lg bg-terminal-surface border border-terminal-border hover:border-profit transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          Atualizar
        </button>
      </div>

      {/* Key Risk Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
        <div className="card group hover:border-terminal-muted/50 transition-colors">
          <div className="flex items-start justify-between mb-2">
            <span className="metric-label inline-flex items-center">VaR 95%<QuickTooltip termKey="var_95" /></span>
            <TrendingDown className="w-4 h-4 text-loss" />
          </div>
          <div className="font-mono font-bold text-2xl">{Math.abs(riskMetrics.var_95 * 100).toFixed(2)}%</div>
        </div>
        <div className="card group hover:border-terminal-muted/50 transition-colors">
          <div className="flex items-start justify-between mb-2">
            <span className="metric-label inline-flex items-center">CVaR 95%<QuickTooltip termKey="cvar_95" /></span>
            <TrendingDown className="w-4 h-4 text-loss" />
          </div>
          <div className="font-mono font-bold text-2xl">{Math.abs(riskMetrics.cvar_95 * 100).toFixed(2)}%</div>
        </div>
        <div className="card group hover:border-terminal-muted/50 transition-colors">
          <div className="flex items-start justify-between mb-2">
            <span className="metric-label inline-flex items-center">Sortino Ratio<QuickTooltip termKey="sortino" /></span>
            <Target className="w-4 h-4" />
          </div>
          <div className="font-mono font-bold text-2xl">{riskMetrics.sortino_ratio.toFixed(3)}</div>
        </div>
        <div className="card group hover:border-terminal-muted/50 transition-colors">
          <div className="flex items-start justify-between mb-2">
            <span className="metric-label inline-flex items-center">Calmar Ratio<QuickTooltip termKey="calmar" /></span>
            <Zap className="w-4 h-4" />
          </div>
          <div className="font-mono font-bold text-2xl">{riskMetrics.calmar_ratio.toFixed(3)}</div>
        </div>
        <div className="card group hover:border-terminal-muted/50 transition-colors">
          <div className="flex items-start justify-between mb-2">
            <span className="metric-label inline-flex items-center">Omega Ratio<QuickTooltip termKey="omega" /></span>
            <Activity className="w-4 h-4" />
          </div>
          <div className="font-mono font-bold text-2xl">{riskMetrics.omega_ratio.toFixed(3)}</div>
        </div>
        <div className="card group hover:border-terminal-muted/50 transition-colors">
          <div className="flex items-start justify-between mb-2">
            <span className="metric-label inline-flex items-center">Stability<QuickTooltip termKey="stability" /></span>
            <BarChart3 className="w-4 h-4" />
          </div>
          <div className="font-mono font-bold text-2xl">{(riskMetrics.stability_of_timeseries * 100).toFixed(2)}%</div>
        </div>
      </div>

      {/* Métricas de Risco de Cauda */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="card">
          <div className="metric-label inline-flex items-center">Assimetria<QuickTooltip termKey="skewness" /></div>
          <div className={`font-mono text-xl ${riskMetrics.skewness < 0 ? 'text-loss' : 'text-profit'}`}>
            {riskMetrics.skewness.toFixed(3)}
          </div>
          <div className="text-xs text-terminal-muted mt-1">
            {riskMetrics.skewness < -0.5 ? 'Risco de cauda negativa' : riskMetrics.skewness > 0.5 ? 'Assimetria positiva' : 'Quase simétrico'}
          </div>
        </div>
        <div className="card">
          <div className="metric-label inline-flex items-center">Curtose Excessiva<QuickTooltip termKey="kurtosis" /></div>
          <div className={`font-mono text-xl ${riskMetrics.kurtosis > 3 ? 'text-accent-yellow' : ''}`}>
            {riskMetrics.kurtosis.toFixed(3)}
          </div>
          <div className="text-xs text-terminal-muted mt-1">
            {riskMetrics.kurtosis > 3 ? 'Caudas gordas (leptocúrtica)' : 'Caudas normais'}
          </div>
        </div>
        <div className="card">
          <div className="metric-label inline-flex items-center">Razão de Cauda<QuickTooltip termKey="tail_ratio" /></div>
          <div className="font-mono text-xl">{riskMetrics.tail_ratio.toFixed(2)}</div>
          <div className="text-xs text-terminal-muted mt-1">Ganho P95 / Perda P5</div>
        </div>
        <div className="card">
          <div className="metric-label inline-flex items-center">Ganho/Dor<QuickTooltip termKey="gain_to_pain" /></div>
          <div className={`font-mono text-xl ${riskMetrics.gain_to_pain > 1 ? 'text-profit' : 'text-loss'}`}>
            {riskMetrics.gain_to_pain.toFixed(2)}
          </div>
          <div className="text-xs text-terminal-muted mt-1">Retorno total / dor total</div>
        </div>
      </div>

      {/* Análise Melhor/Pior */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="card">
          <div className="metric-label inline-flex items-center">Melhor Dia<QuickTooltip termKey="best_day" /></div>
          <div className="font-mono text-xl text-profit">+{(riskMetrics.best_day * 100).toFixed(2)}%</div>
        </div>
        <div className="card">
          <div className="metric-label inline-flex items-center">Pior Dia<QuickTooltip termKey="worst_day" /></div>
          <div className="font-mono text-xl text-loss">{(riskMetrics.worst_day * 100).toFixed(2)}%</div>
        </div>
        <div className="card">
          <div className="metric-label inline-flex items-center">Melhor Mês<QuickTooltip termKey="best_month" /></div>
          <div className="font-mono text-xl text-profit">+{riskMetrics.best_month.toFixed(2)}%</div>
        </div>
        <div className="card">
          <div className="metric-label inline-flex items-center">Pior Mês<QuickTooltip termKey="worst_month" /></div>
          <div className="font-mono text-xl text-loss">{riskMetrics.worst_month.toFixed(2)}%</div>
        </div>
      </div>

      {/* Estatísticas de Drawdown */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="card">
          <div className="metric-label inline-flex items-center">Maior Drawdown<QuickTooltip termKey="longest_dd" /></div>
          <div className="font-mono text-xl">{riskMetrics.longest_dd_days} dias</div>
        </div>
        <div className="card">
          <div className="metric-label inline-flex items-center">Duração Média DD<QuickTooltip termKey="avg_dd_duration" /></div>
          <div className="font-mono text-xl">{riskMetrics.average_dd_days.toFixed(1)} dias</div>
        </div>
        <div className="card">
          <div className="metric-label inline-flex items-center">Tempo Submerso<QuickTooltip termKey="time_underwater" /></div>
          <div className={`font-mono text-xl ${riskMetrics.time_underwater_pct > 50 ? 'text-loss' : ''}`}>
            {riskMetrics.time_underwater_pct.toFixed(1)}%
          </div>
        </div>
        <div className="card">
          <div className="metric-label inline-flex items-center">Payoff Ratio<QuickTooltip termKey="payoff_ratio" /></div>
          <div className={`font-mono text-xl ${riskMetrics.payoff_ratio > 1 ? 'text-profit' : 'text-loss'}`}>
            {riskMetrics.payoff_ratio.toFixed(2)}
          </div>
        </div>
      </div>

      {/* Abas */}
      <div className="flex items-center gap-4 border-b border-terminal-border">
        {[
          { key: 'var', label: 'Análise VaR', icon: Shield },
          { key: 'distribution', label: 'Distribuição de Retornos', icon: BarChart3 },
          { key: 'rolling', label: 'Métricas Móveis', icon: Activity },
          { key: 'monthly', label: 'Retornos Mensais', icon: Calendar },
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

      {/* Conteúdo das Abas */}
      {activeTab === 'var' && (
        <div className="card-elevated">
          <h3 className="font-semibold text-lg mb-4">Análise de Value at Risk</h3>
          <VaRGauge
            var95={riskMetrics.var_95}
            var99={riskMetrics.var_99}
            cvar95={riskMetrics.cvar_95}
            cvar99={riskMetrics.cvar_99}
          />
        </div>
      )}

      {activeTab === 'distribution' && (
        <div className="card-elevated">
          <h3 className="font-semibold text-lg mb-4">Distribuição de Retornos Diários</h3>
          <div className="h-[400px]">
            <ReturnDistribution returns={riskMetrics.daily_returns} showNormal />
          </div>
        </div>
      )}

      {activeTab === 'rolling' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="card-elevated">
            <h3 className="font-semibold text-lg mb-4 inline-flex items-center">Sharpe Móvel (252 dias)<QuickTooltip termKey="rolling_sharpe" /></h3>
            <div className="h-[300px]">
              <RollingMetrics
                data={[
                  { label: 'Sharpe', points: riskMetrics.rolling_sharpe, color: '#00ff88' },
                ]}
                showZeroLine
              />
            </div>
          </div>
          <div className="card-elevated">
            <h3 className="font-semibold text-lg mb-4 inline-flex items-center">Volatilidade Móvel (252 dias)<QuickTooltip termKey="rolling_volatility" /></h3>
            <div className="h-[300px]">
              <RollingMetrics
                data={[
                  { label: 'Volatilidade', points: riskMetrics.rolling_volatility, color: '#ff6b6b' },
                ]}
                showZeroLine={false}
              />
            </div>
          </div>
          <div className="card-elevated lg:col-span-2">
            <h3 className="font-semibold text-lg mb-4">Retornos Móveis (21 dias)</h3>
            <div className="h-[300px]">
              <RollingMetrics
                data={[
                  { label: 'Retorno 21 dias', points: riskMetrics.rolling_returns, color: '#00d4ff' },
                ]}
                showZeroLine
              />
            </div>
          </div>
        </div>
      )}

      {activeTab === 'monthly' && (
        <div className="card-elevated">
          <h3 className="font-semibold text-lg mb-4">Mapa de Calor de Retornos Mensais</h3>
          <MonthlyHeatmap data={riskMetrics.monthly_returns} />
        </div>
      )}
    </div>
  );
}


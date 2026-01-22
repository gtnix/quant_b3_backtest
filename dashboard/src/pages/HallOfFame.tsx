/**
 * HallOfFame - Display promoted strategies that passed all institutional criteria
 */

import { useEffect, useState } from 'react';
import { 
  Trophy, Shield, BarChart3, Clock, 
  GitBranch, ChevronDown, RefreshCw,
  CheckCircle2, XCircle, Filter, LineChart, Boxes,
  AlertTriangle, Info, Fingerprint, FolderOpen
} from 'lucide-react';
import { useOmpStore } from '../stores/ompStore';
import { useStrategyStore } from '../stores/strategyStore';
import { useDataStore } from '../stores/dataStore';
import type { HallOfFameEntry } from '../stores/ompStore';

// =============================================================================
// FILTER COMPONENT
// =============================================================================

interface FilterState {
  market: 'all' | 'br' | 'us';
  family: string;
  sortBy: 'sharpe' | 'date' | 'pbo' | 'dsr';
  limit: number;
}

function FilterBar({ filter, setFilter }: { filter: FilterState; setFilter: (f: FilterState) => void }) {
  const { families, fetchFamilies } = useStrategyStore();
  
  useEffect(() => {
    if (families.length === 0) fetchFamilies();
  }, [families.length, fetchFamilies]);
  
  return (
    <div className="flex flex-wrap items-center gap-4 p-4 bg-slate-800/50 rounded-xl border border-slate-700">
      <div className="flex items-center gap-2">
        <Filter className="w-4 h-4 text-slate-400" />
        <span className="text-sm text-slate-400">Filtros:</span>
      </div>
      
      {/* Filtro de Mercado */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-slate-500">Mercado:</span>
        <div className="flex rounded-lg overflow-hidden border border-slate-700">
          {(['all', 'br', 'us'] as const).map(m => (
            <button
              key={m}
              onClick={() => setFilter({ ...filter, market: m })}
              className={`px-3 py-1 text-xs font-medium transition-colors ${
                filter.market === m 
                  ? 'bg-amber-500/20 text-amber-400' 
                  : 'bg-slate-800 text-slate-400 hover:text-white'
              }`}
            >
              {m === 'all' ? 'Todos' : m.toUpperCase()}
            </button>
          ))}
        </div>
      </div>
      
      {/* Filtro de Família de Estratégia */}
      <div className="flex items-center gap-2">
        <Boxes className="w-3 h-3 text-slate-500" />
        <span className="text-xs text-slate-500">Família:</span>
        <select
          value={filter.family}
          onChange={e => setFilter({ ...filter, family: e.target.value })}
          className="px-3 py-1 text-xs bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-amber-500"
        >
          <option value="">Todas Famílias</option>
          {families.map(f => (
            <option key={f.slug} value={f.slug}>{f.name}</option>
          ))}
        </select>
      </div>
      
      {/* Ordenação */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-slate-500">Ordenar:</span>
        <select
          value={filter.sortBy}
          onChange={e => setFilter({ ...filter, sortBy: e.target.value as FilterState['sortBy'] })}
          className="px-3 py-1 text-xs bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-amber-500"
        >
          <option value="sharpe">Sharpe (Maior → Menor)</option>
          <option value="date">Data (Recente → Antigo)</option>
          <option value="pbo">PBO (Menor → Maior)</option>
          <option value="dsr">DSR (Maior → Menor)</option>
        </select>
      </div>
      
      {/* Limite */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-slate-500">Mostrar:</span>
        <select
          value={filter.limit}
          onChange={e => setFilter({ ...filter, limit: parseInt(e.target.value) })}
          className="px-3 py-1 text-xs bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-amber-500"
        >
          <option value={25}>25</option>
          <option value={50}>50</option>
          <option value={100}>100</option>
        </select>
      </div>
    </div>
  );
}

// =============================================================================
// VALIDATION BADGE COMPONENT
// =============================================================================

interface ValidationIssue {
  metric: string;
  problem: string;
  severity: 'warning' | 'error';
}

// Verifica se as métricas estão dentro de intervalos válidos (sanity check quant)
function validateMetrics(m: HallOfFameEntry['metrics']): { valid: boolean; issues: ValidationIssue[] } {
  const issues: ValidationIssue[] = [];
  
  // Sharpe ratio sanity: intervalo realista é -3 a +5 para maioria das estratégias
  if (m.oosSharpeNet != null && (m.oosSharpeNet > 10 || m.oosSharpeNet < -3)) {
    issues.push({ 
      metric: 'Sharpe', 
      problem: `Valor ${m.oosSharpeNet.toFixed(1)} está fora do intervalo realista (-3 a +5)`,
      severity: 'error'
    });
  }
  
  // PBO deve estar entre 0 e 1, tipicamente 0.05-0.50
  if (m.pbo === 0 || m.pbo == null) {
    issues.push({ 
      metric: 'PBO', 
      problem: 'Probability of Backtest Overfitting não foi calculado',
      severity: 'warning'
    });
  }
  
  // DSR deve ser positivo e tipicamente 60-80% do Sharpe bruto
  if (m.dsr === 0 || m.dsr == null) {
    issues.push({ 
      metric: 'DSR', 
      problem: 'Deflated Sharpe Ratio não foi calculado',
      severity: 'warning'
    });
  }
  
  // MaxDD deve ser negativo e entre 0 e -1
  if (m.maxDrawdownNet == null) {
    issues.push({ 
      metric: 'MaxDD', 
      problem: 'Drawdown máximo não foi registrado',
      severity: 'warning'
    });
  }
  
  return { valid: issues.length === 0, issues };
}

// Badge com tooltip explicativo para métricas incompletas
function ValidationBadge({ issues }: { issues: ValidationIssue[] }) {
  const [showTooltip, setShowTooltip] = useState(false);
  const hasErrors = issues.some(i => i.severity === 'error');
  
  if (issues.length === 0) return null;
  
  return (
    <div 
      className="relative inline-flex"
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
    >
      <span className={`flex items-center gap-1 px-2 py-0.5 text-xs rounded-full cursor-help transition-colors ${
        hasErrors 
          ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30' 
          : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
      }`}>
        <AlertTriangle className="w-3 h-3" />
        {issues.length} {issues.length === 1 ? 'alerta' : 'alertas'}
      </span>
      
      {showTooltip && (
        <div className="absolute left-0 top-full mt-2 w-72 p-3 bg-slate-800 border border-slate-600 rounded-lg shadow-xl z-50 animate-fade-in">
          <div className="flex items-center gap-2 mb-2 pb-2 border-b border-slate-700">
            <Info className="w-4 h-4 text-slate-400" />
            <span className="text-xs font-medium text-white">Métricas Incompletas</span>
          </div>
          <ul className="space-y-2">
            {issues.map((issue, i) => (
              <li key={i} className="flex items-start gap-2">
                <span className={`mt-0.5 ${issue.severity === 'error' ? 'text-rose-400' : 'text-amber-400'}`}>
                  {issue.severity === 'error' ? <XCircle className="w-3 h-3" /> : <AlertTriangle className="w-3 h-3" />}
                </span>
                <div className="flex-1">
                  <span className="text-xs font-medium text-white">{issue.metric}:</span>
                  <p className="text-xs text-slate-400 mt-0.5">{issue.problem}</p>
                </div>
              </li>
            ))}
          </ul>
          <p className="text-[10px] text-slate-500 mt-2 pt-2 border-t border-slate-700">
            Estratégias com alertas ainda podem ser válidas, mas métricas completas aumentam a confiança.
          </p>
        </div>
      )}
    </div>
  );
}

// =============================================================================
// ENTRY CARD COMPONENT
// =============================================================================

function EntryCard({ entry, rank }: { entry: HallOfFameEntry; rank: number }) {
  const [expanded, setExpanded] = useState(false);
  const [loading, setLoading] = useState(false);
  const { setSelectedCandidate } = useDataStore();
  
  const metrics = entry.metrics;
  const validation = entry.validation;
  const metricsCheck = validateMetrics(metrics);
  
  return (
    <div className="rounded-xl border border-slate-700 bg-slate-800/30 overflow-hidden hover:border-amber-500/50 transition-colors">
      {/* Main Row */}
      <div 
        className="flex items-center gap-4 p-4 cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        {/* Rank Badge */}
        <div className={`w-10 h-10 rounded-xl flex items-center justify-center font-bold text-lg ${
          rank === 1 ? 'bg-amber-500/20 text-amber-400' :
          rank === 2 ? 'bg-slate-400/20 text-slate-300' :
          rank === 3 ? 'bg-orange-500/20 text-orange-400' :
          'bg-slate-700 text-slate-400'
        }`}>
          {rank}
        </div>
        
        {/* Candidate Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-white text-sm truncate" title={entry.candidateId}>
              {entry.strategyName || entry.candidateId}
            </span>
            <span className={`px-2 py-0.5 text-xs rounded-full ${
              entry.market === 'br' 
                ? 'bg-green-500/20 text-green-400' 
                : 'bg-blue-500/20 text-blue-400'
            }`}>
              {entry.market?.toUpperCase() || 'BR'}
            </span>
            <ValidationBadge issues={metricsCheck.issues} />
          </div>
          <div className="text-xs text-slate-500 mt-0.5 font-mono">
            {entry.candidateId.slice(-12)}
          </div>
        </div>
        
        {/* Key Metrics */}
        <div className="hidden sm:flex items-center gap-6">
          <div className="text-center">
            <p className="text-lg font-bold text-emerald-400">{metrics.oosSharpeNet?.toFixed(3)}</p>
            <p className="text-xs text-slate-500">Sharpe</p>
          </div>
          <div className="text-center">
            <p className="text-lg font-bold text-white">{(metrics.pbo * 100)?.toFixed(1)}%</p>
            <p className="text-xs text-slate-500">PBO</p>
          </div>
          <div className="text-center">
            <p className="text-lg font-bold text-white">{metrics.dsr?.toFixed(2)}</p>
            <p className="text-xs text-slate-500">DSR</p>
          </div>
          <div className="text-center">
            <p className="text-lg font-bold text-rose-400">{(Math.abs(metrics.maxDrawdownNet || 0) * 100).toFixed(1)}%</p>
            <p className="text-xs text-slate-500">Max DD</p>
          </div>
        </div>
        
        {/* Expand Arrow */}
        <ChevronDown className={`w-5 h-5 text-slate-500 transition-transform ${expanded ? 'rotate-180' : ''}`} />
      </div>
      
      {/* Expanded Details */}
      {expanded && (
        <div className="border-t border-slate-700 p-4 bg-slate-900/30">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {/* Métricas */}
            <div>
              <h4 className="text-xs text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <BarChart3 className="w-3 h-3" /> Métricas
              </h4>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-500">Sharpe OOS NET</span>
                  <span className="text-emerald-400 font-mono">{metrics.oosSharpeNet?.toFixed(4)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">CAGR NET</span>
                  <span className="text-white font-mono">{((metrics.cagrNet || 0) * 100).toFixed(2)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Drawdown Máximo</span>
                  <span className="text-rose-400 font-mono">{(Math.abs(metrics.maxDrawdownNet || 0) * 100).toFixed(2)}%</span>
                </div>
              </div>
            </div>
            
            {/* Validação */}
            <div>
              <h4 className="text-xs text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <Shield className="w-3 h-3" /> Validação
              </h4>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-500">PBO</span>
                  <span className="text-white font-mono">{(metrics.pbo * 100).toFixed(2)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">DSR</span>
                  <span className="text-white font-mono">{metrics.dsr?.toFixed(3)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-500">Gates Aprovados</span>
                  {validation.gatesPassed ? (
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  ) : (
                    <XCircle className="w-4 h-4 text-rose-400" />
                  )}
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Stress</span>
                  <span className="text-white font-mono">{validation.stressPassed}/{validation.stressTotal}</span>
                </div>
              </div>
            </div>
            
            {/* Proveniência */}
            <div>
              <h4 className="text-xs text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <GitBranch className="w-3 h-3" /> Proveniência
              </h4>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-500">Git SHA</span>
                  <span className="text-white font-mono text-xs">{entry.provenance?.gitSha?.slice(0, 7) || '—'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Hash Config</span>
                  <span className="text-white font-mono text-xs">{entry.provenance?.configHash?.slice(0, 7) || '—'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Hash Dataset</span>
                  <span className="text-white font-mono text-xs">{entry.provenance?.datasetHash?.slice(0, 7) || '—'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Hash Genoma</span>
                  <span className="text-white font-mono text-xs">{entry.genomeHash?.slice(0, 7) || '—'}</span>
                </div>
              </div>
            </div>
            
            {/* Tempo */}
            <div>
              <h4 className="text-xs text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <Clock className="w-3 h-3" /> Tempo
              </h4>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-500">Promovido</span>
                  <span className="text-white text-xs">{new Date(entry.promotedAt).toLocaleDateString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">ID Execução</span>
                  <button 
                    onClick={(e) => {
                      e.stopPropagation();
                      // Navigate to Runs page - extract experiment ID from runId if it looks like scg_*
                      const experimentId = entry.runId?.startsWith('scg_') 
                        ? entry.runId 
                        : entry.campaignId?.startsWith('scg_') 
                          ? entry.campaignId 
                          : null;
                      if (experimentId) {
                        window.dispatchEvent(new CustomEvent('navigate', { detail: 'runs' }));
                      }
                    }}
                    className="text-cyan-400 hover:text-cyan-300 hover:underline font-mono text-xs flex items-center gap-1"
                    title="View run details"
                  >
                    <FolderOpen className="w-3 h-3" />
                    {entry.runId?.slice(0, 12) || '—'}
                  </button>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Campanha</span>
                  <span className="text-white text-xs truncate max-w-[100px]">{entry.campaignId?.slice(0, 8)}</span>
                </div>
              </div>
            </div>
          </div>
          
          {/* Strategy Catalog */}
          {(entry.strategyFamily || entry.strategyVariant || entry.strategyHypothesis) && (
            <div className="mt-4 pt-4 border-t border-slate-700">
              <h4 className="text-xs text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <Boxes className="w-3 h-3" /> Strategy Catalog
              </h4>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                {entry.strategyFamily && (
                  <div>
                    <span className="text-slate-500">Família:</span>
                    <span className="ml-2 text-amber-400 font-medium">{entry.strategyFamily}</span>
                  </div>
                )}
                {entry.strategyVariant && (
                  <div>
                    <span className="text-slate-500">Variante:</span>
                    <span className="ml-2 text-white">{entry.strategyVariant}</span>
                  </div>
                )}
                {entry.strategyId && (
                  <div>
                    <span className="text-slate-500">ID:</span>
                    <span className="ml-2 text-white font-mono text-xs">{entry.strategyId}</span>
                  </div>
                )}
              </div>
              {entry.strategyHypothesis && (
                <p className="text-xs text-slate-400 mt-2 italic">
                  "{entry.strategyHypothesis}"
                </p>
              )}
            </div>
          )}
          
          {/* Strategy Identity */}
          {entry.identity && (
            <div className="mt-4 pt-4 border-t border-slate-700">
              <h4 className="text-xs text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <Fingerprint className="w-3 h-3" /> Strategy Identity
              </h4>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                <div><span className="text-slate-500">Type:</span><span className="ml-2 text-white">{entry.identity.strategy_type}</span></div>
                <div><span className="text-slate-500">Market:</span><span className="ml-2 text-white">{entry.identity.market}/{entry.identity.universe}</span></div>
                <div><span className="text-slate-500">Timeframe:</span><span className="ml-2 text-white">{entry.identity.timeframe}</span></div>
              </div>
              {entry.identity.blocks && entry.identity.blocks.length > 0 && (
                <div className="mt-2">
                  <span className="text-xs text-slate-500">Blocks:</span>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {entry.identity.blocks.map((b, i) => (
                      <span key={i} className="px-1.5 py-0.5 bg-slate-700/50 rounded text-xs text-slate-300">{b.block_id}</span>
                    ))}
                  </div>
                </div>
              )}
              {entry.identity.entry_rules && entry.identity.entry_rules !== 'No explicit entry rules' && (
                <div className="mt-2 text-xs">
                  <p className="text-slate-500">Entry Rules:</p>
                  <p className="text-white">{entry.identity.entry_rules}</p>
                </div>
              )}
              {entry.identity.exit_rules && entry.identity.exit_rules !== 'No explicit exit rules (rebalance-based)' && (
                <div className="mt-2 text-xs">
                  <p className="text-slate-500">Exit Rules:</p>
                  <p className="text-white">{entry.identity.exit_rules}</p>
                </div>
              )}
            </div>
          )}
          
          {/* Notas */}
          {entry.notes && (
            <div className="mt-4 pt-4 border-t border-slate-700">
              <p className="text-xs text-slate-500">Notas: {entry.notes}</p>
            </div>
          )}
          
          {/* Ações */}
          <div className="mt-4 pt-4 border-t border-slate-700 flex justify-end gap-3">
            <button
              disabled={loading}
              onClick={async (e) => {
                e.stopPropagation();
                setLoading(true);
                try {
                  const res = await fetch(`http://localhost:3001/api/candidate/${entry.candidateId}`);
                  if (!res.ok) throw new Error('Falha ao carregar candidato');
                  const data = await res.json();
                  // Mapeia para o formato esperado pelo Backtest (snake_case)
                  const mapped = {
                    candidate_id: data.candidate_id || entry.candidateId,
                    genome_hash: data.genome_hash || entry.genomeHash,
                    display_name: data.display_name || entry.strategyName || entry.candidateId.slice(-12),
                    candidate_class: data.candidate_class || 'validated',
                    oos_sharpe_net: data.oos_sharpe_net ?? entry.metrics.oosSharpeNet,
                    oos_cagr_net: data.oos_cagr_net ?? entry.metrics.cagrNet ?? 0,
                    max_drawdown_net: data.max_drawdown_net ?? entry.metrics.maxDrawdownNet ?? 0,
                    pbo: data.pbo ?? entry.metrics.pbo ?? 0,
                    dsr: data.dsr ?? entry.metrics.dsr ?? 0,
                    gates_passed: data.gates_passed ?? entry.validation.gatesPassed ?? true,
                    stress_passed: data.stress_passed ?? entry.validation.stressPassed ?? 0,
                    stress_total: data.stress_total ?? entry.validation.stressTotal ?? 8,
                    ...data,
                  };
                  setSelectedCandidate(mapped);
                  window.dispatchEvent(new CustomEvent('navigate', { detail: 'backtest' }));
                } catch (err) {
                  console.error('Falha ao carregar candidato:', err);
                } finally {
                  setLoading(false);
                }
              }}
              className="flex items-center gap-2 px-4 py-2 bg-accent-cyan/20 text-accent-cyan rounded-lg hover:bg-accent-cyan/30 transition-colors text-sm font-medium disabled:opacity-50"
            >
              {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <LineChart className="w-4 h-4" />}
              {loading ? 'Carregando...' : 'Abrir Backtest'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function HallOfFame() {
  const { hallOfFame, hallOfFameLoading, fetchHallOfFame, stats } = useOmpStore();
  
  const [filter, setFilter] = useState<FilterState>({
    market: 'all',
    family: '',
    sortBy: 'sharpe',
    limit: 50,
  });
  
  // Fetch on mount and filter change
  useEffect(() => {
    fetchHallOfFame(filter.limit, filter.market === 'all' ? undefined : filter.market);
  }, [fetchHallOfFame, filter.limit, filter.market]);
  
  // Sort entries
  const sortedEntries = [...hallOfFame].sort((a, b) => {
    switch (filter.sortBy) {
      case 'sharpe':
        return (b.metrics.oosSharpeNet || 0) - (a.metrics.oosSharpeNet || 0);
      case 'date':
        return new Date(b.promotedAt).getTime() - new Date(a.promotedAt).getTime();
      case 'pbo':
        return (a.metrics.pbo || 1) - (b.metrics.pbo || 1);
      case 'dsr':
        return (b.metrics.dsr || 0) - (a.metrics.dsr || 0);
      default:
        return 0;
    }
  });
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        
        {/* Cabeçalho */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-white flex items-center gap-3">
              <Trophy className="w-6 h-6 text-amber-400" />
              Hall da Fama
            </h1>
            <p className="text-slate-400 text-sm mt-1">
              Estratégias de elite que passaram em todos os gates institucionais de validação
            </p>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="text-right">
              <p className="text-2xl font-bold text-amber-400">{stats?.promotions.total || hallOfFame.length}</p>
              <p className="text-xs text-slate-500">Total Promovidas</p>
            </div>
            <button
              onClick={async () => {
                try {
                  const res = await fetch('http://localhost:3001/api/omp/promote-check', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ limit: 1000 }),
                  });
                  const data = await res.json();
                  console.log('Verificação de promoção:', data);
                  if (data.promoted > 0) {
                    fetchHallOfFame(filter.limit, filter.market === 'all' ? undefined : filter.market);
                  }
                } catch (err) {
                  console.error('Verificação de promoção falhou:', err);
                }
              }}
              className="px-3 py-2 bg-amber-600 hover:bg-amber-500 text-white rounded-lg text-sm font-medium transition-colors"
            >
              Buscar Promoções
            </button>
            <button
              onClick={() => fetchHallOfFame(filter.limit, filter.market === 'all' ? undefined : filter.market)}
              disabled={hallOfFameLoading}
              className="p-2 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors"
            >
              <RefreshCw className={`w-5 h-5 text-slate-400 ${hallOfFameLoading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>
        
        {/* Filters */}
        <FilterBar filter={filter} setFilter={setFilter} />
        
        {/* Resumo de Estatísticas */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700 text-center">
            <p className="text-2xl font-bold text-emerald-400">{stats?.promotions.last24h || 0}</p>
            <p className="text-xs text-slate-500">Promovidas (24h)</p>
          </div>
          <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700 text-center">
            <p className="text-2xl font-bold text-white">{stats?.promotions.last7d || 0}</p>
            <p className="text-xs text-slate-500">Promovidas (7d)</p>
          </div>
          <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700 text-center">
            <p className="text-2xl font-bold text-white">
              {sortedEntries.length > 0 ? sortedEntries[0].metrics.oosSharpeNet?.toFixed(3) : '—'}
            </p>
            <p className="text-xs text-slate-500">Melhor Sharpe</p>
          </div>
          <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700 text-center">
            <p className="text-2xl font-bold text-white">
              {sortedEntries.length > 0 
                ? ((sortedEntries.reduce((sum, e) => sum + (e.metrics.oosSharpeNet || 0), 0) / sortedEntries.length)).toFixed(3)
                : '—'
              }
            </p>
            <p className="text-xs text-slate-500">Sharpe Médio</p>
          </div>
        </div>
        
        {/* Lista de Entradas */}
        {hallOfFameLoading ? (
          <div className="flex items-center justify-center py-12">
            <RefreshCw className="w-8 h-8 text-slate-500 animate-spin" />
          </div>
        ) : sortedEntries.length === 0 ? (
          <div className="text-center py-12 rounded-xl border border-slate-700 bg-slate-800/30">
            <Trophy className="w-12 h-12 text-slate-600 mx-auto mb-3" />
            <p className="text-slate-400">Nenhuma estratégia promovida ainda</p>
            <p className="text-sm text-slate-500 mt-1">Estratégias aparecerão aqui quando passarem em todos os gates de validação</p>
          </div>
        ) : (
          <div className="space-y-3">
            {sortedEntries.map((entry, i) => (
              <EntryCard key={entry.promotionId} entry={entry} rank={i + 1} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default HallOfFame;


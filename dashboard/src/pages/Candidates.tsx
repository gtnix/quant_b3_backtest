import { useState, useEffect } from 'react';
import { DataTable } from '../components/ui/DataTable';
import { CandidateDetail } from '../components/CandidateDetail';
import { ParetoScatter } from '../components/charts/ParetoScatter';
import { useDataStore } from '../stores/dataStore';
import type { CandidateListItem } from '../stores/dataStore';
import { platform } from '../lib/platform';
import { FolderSelector } from '../components/FolderSelector';
import { QuickTooltip } from '../components/ui/TooltipInfo';
import { 
  Search, 
  Filter, 
  Download, 
  Award,
  CheckCircle,
  XCircle,
  FolderOpen,
  RefreshCw,
  ChevronDown,
  GitCompare,
  Square,
  CheckSquare,
  X,
  TrendingUp,
  Shield,
  Target,
  Clock,
  Database,
  ScatterChart,
  Table
} from 'lucide-react';

export function Candidates() {
  const [searchQuery, setSearchQuery] = useState('');
  const [filterPbo, setFilterPbo] = useState<number | null>(null);
  const [filterClass, setFilterClass] = useState<string>('');
  const [showDetail, setShowDetail] = useState(false);
  const [viewMode, setViewMode] = useState<'table' | 'pareto'>('table');
  
  const { 
    artifactsRoot,
    candidates, 
    selectedCandidate,
    selectedRunId,
    selectedCandidateIds,
    runs,
    recentRuns,
    isLoading,
    error,
    listCandidates,
    loadCandidateDetail,
    clearSelectedCandidate,
    setArtifactsRoot,
    toggleCandidateSelection,
    clearCandidateSelection,
    fetchRecentRuns,
    loadRun,
  } = useDataStore();

  useEffect(() => {
    if (selectedRunId) {
      listCandidates(selectedRunId, {
        search: searchQuery || undefined,
        max_pbo: filterPbo ?? undefined,
        candidate_class: filterClass || undefined,
      });
    }
  }, [searchQuery, filterPbo, filterClass, selectedRunId]);

  const handleSelectFolder = async () => {
    try {
      const { open } = await import('@tauri-apps/plugin-dialog');
      const selected = await open({
        directory: true,
        multiple: false,
        title: 'Select Project Root'
      });
      if (selected && typeof selected === 'string') {
        await setArtifactsRoot(selected);
      }
    } catch (err) {
      console.error('Failed to open folder dialog:', err);
    }
  };

  const handleRowClick = async (row: Record<string, unknown>) => {
    const candidateId = row.candidate_id as string;
    await loadCandidateDetail(candidateId);
    setShowDetail(true);
  };

  const handleCloseDetail = () => {
    setShowDetail(false);
    clearSelectedCandidate();
  };

  const handleToggleSelection = (candidateId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    toggleCandidateSelection(candidateId);
  };

  const handleSelectAll = () => {
    (candidates || []).forEach(c => {
      if (!selectedCandidateIds.includes(c.candidate_id)) {
        toggleCandidateSelection(c.candidate_id);
      }
    });
  };

  const handleCompare = () => {
    window.dispatchEvent(new CustomEvent('navigate', { detail: 'comparison' }));
  };

  // Mini progress bar component
  const MiniBar = ({ value, max, color }: { value: number; max: number; color: string }) => {
    const pct = Math.min(100, Math.max(0, (value / max) * 100));
    return (
      <div className="w-12 h-1.5 bg-terminal-bg rounded-full overflow-hidden">
        <div className={`h-full ${color} transition-all`} style={{ width: `${pct}%` }} />
      </div>
    );
  };

  const columns = [
    {
      key: '_select',
      header: (
        <button onClick={handleSelectAll} className="p-1 hover:bg-terminal-surface rounded" title="Selecionar todos">
          <Square className="w-4 h-4 text-terminal-muted" />
        </button>
      ),
      width: '40px',
      render: (_: unknown, row: Record<string, unknown>) => {
        const isSelected = selectedCandidateIds.includes(row.candidate_id as string);
        return (
          <button onClick={(e) => handleToggleSelection(row.candidate_id as string, e)} className="p-1 hover:bg-terminal-surface rounded">
            {isSelected ? <CheckSquare className="w-4 h-4 text-profit" /> : <Square className="w-4 h-4 text-terminal-muted" />}
          </button>
        );
      },
    },
    {
      key: 'rank',
      header: '#',
      sortable: true,
      width: '50px',
      render: (value: unknown) => (
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent-cyan/20 to-profit/20 flex items-center justify-center font-bold text-sm">
          {String(value)}
        </div>
      ),
    },
    {
      key: 'display_name',
      header: 'Estratégia',
      sortable: true,
      render: (value: unknown, row: Record<string, unknown>) => {
        const gatesPassed = row.gates_passed as boolean;
        return (
          <div className="flex items-center gap-3">
            <div className={`w-2 h-8 rounded-full ${gatesPassed ? 'bg-profit' : 'bg-terminal-muted'}`} />
            <div>
              <div className="font-medium text-sm truncate max-w-[200px]" title={String(value)}>
                {String(value)}
              </div>
              <div className="text-xs text-terminal-muted font-mono">
                {String(row.candidate_id).substring(0, 16)}...
              </div>
            </div>
          </div>
        );
      },
    },
    {
      key: 'candidate_class',
      header: (<span className="inline-flex items-center">Status<QuickTooltip termKey="validated" /></span>),
      sortable: true,
      width: '100px',
      render: (value: unknown, row: Record<string, unknown>) => {
        const cls = String(value);
        const gatesPassed = row.gates_passed as boolean;
        if (cls === 'validated' && gatesPassed) {
          return (
            <div className="flex items-center gap-1.5 text-profit">
              <CheckCircle className="w-4 h-4" />
              <span className="text-xs font-medium">Validado</span>
            </div>
          );
        }
        return (
          <div className="flex items-center gap-1.5 text-terminal-muted">
            <Target className="w-4 h-4" />
            <span className="text-xs font-medium">Pesquisa</span>
          </div>
        );
      },
    },
    {
      key: 'oos_sharpe_net',
      header: (<span className="inline-flex items-center">Sharpe<QuickTooltip termKey="sharpe_net" /></span>),
      sortable: true,
      align: 'right' as const,
      render: (value: unknown) => {
        const v = value as number;
        const color = v >= 1.0 ? 'text-profit' : v >= 0.5 ? 'text-accent-yellow' : 'text-loss';
        const barColor = v >= 1.0 ? 'bg-profit' : v >= 0.5 ? 'bg-accent-yellow' : 'bg-loss';
        return (
          <div className="flex items-center gap-2 justify-end">
            <MiniBar value={v} max={2} color={barColor} />
            <span className={`font-mono font-bold ${color}`}>{v.toFixed(2)}</span>
          </div>
        );
      },
    },
    {
      key: 'pbo',
      header: (<span className="inline-flex items-center">PBO<QuickTooltip termKey="pbo" /></span>),
      sortable: true,
      align: 'right' as const,
      width: '90px',
      render: (value: unknown) => {
        const v = value as number;
        const color = v <= 0.10 ? 'text-profit' : v <= 0.15 ? 'text-accent-yellow' : 'text-loss';
        const barColor = v <= 0.10 ? 'bg-profit' : v <= 0.15 ? 'bg-accent-yellow' : 'bg-loss';
        return (
          <div className="flex items-center gap-2 justify-end">
            <MiniBar value={1 - v} max={1} color={barColor} />
            <span className={`font-mono ${color}`}>{(v * 100).toFixed(1)}%</span>
          </div>
        );
      },
    },
    {
      key: 'oos_cagr_net',
      header: (<span className="inline-flex items-center">CAGR<QuickTooltip termKey="cagr" /></span>),
      sortable: true,
      align: 'right' as const,
      width: '80px',
      render: (value: unknown) => {
        const v = value as number;
        const color = v >= 0.15 ? 'text-profit' : 'text-terminal-text';
        return <span className={`font-mono ${color}`}>{(v * 100).toFixed(1)}%</span>;
      },
    },
    {
      key: 'dsr',
      header: (<span className="inline-flex items-center">DSR<QuickTooltip termKey="dsr" /></span>),
      sortable: true,
      align: 'right' as const,
      width: '70px',
      render: (value: unknown) => {
        const v = value as number;
        const color = v >= 0.5 ? 'text-profit' : v >= 0.3 ? 'text-accent-yellow' : 'text-loss';
        return <span className={`font-mono ${color}`}>{v.toFixed(2)}</span>;
      },
    },
    {
      key: 'stress_passed',
      header: (<span className="inline-flex items-center">Stress<QuickTooltip termKey="stress_test" /></span>),
      sortable: true,
      align: 'center' as const,
      width: '70px',
      render: (value: unknown, row: Record<string, unknown>) => {
        const passed = value as number;
        const total = row.stress_total as number;
        const pct = total > 0 ? passed / total : 0;
        return (
          <div className="flex items-center gap-1.5 justify-center">
            <div className={`w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold ${
              pct >= 0.8 ? 'bg-profit/20 text-profit' : pct >= 0.6 ? 'bg-accent-yellow/20 text-accent-yellow' : 'bg-loss/20 text-loss'
            }`}>
              {passed}
            </div>
            <span className="text-terminal-muted text-xs">/{total}</span>
          </div>
        );
      },
    },
  ];

  // Calculate stats (with safety check for undefined candidates)
  const candidatesList = candidates || [];
  const validatedCount = candidatesList.filter(c => c.gates_passed).length;
  const lowPboCount = candidatesList.filter(c => c.pbo <= 0.10).length;
  const bestSharpe = candidatesList.length > 0 ? Math.max(...candidatesList.map(c => c.oos_sharpe_net)) : 0;
  const avgSharpe = candidatesList.length > 0 
    ? candidatesList.reduce((sum, c) => sum + c.oos_sharpe_net, 0) / candidatesList.length 
    : 0;

  // Auto-initialize in browser mode - fetch recent runs even without artifactsRoot
  useEffect(() => {
    if (!platform.isTauri && recentRuns.length === 0 && !selectedRunId) {
      fetchRecentRuns(10);
    }
  }, []);

  // In browser mode, skip the artifactsRoot requirement - we can fetch directly from Neon
  if (!artifactsRoot && platform.isTauri) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-6">
        <FolderOpen className="w-16 h-16 text-terminal-muted" />
        <div className="text-center max-w-lg">
          <h2 className="text-xl font-semibold mb-2">Nenhum Projeto Selecionado</h2>
          <p className="text-terminal-muted mb-4">Selecione uma pasta de projeto contendo artefatos SCG.</p>
          <FolderSelector 
            type="artifacts"
            label="Pasta de Artefatos"
            description="Selecione a pasta contendo o output do SCG"
            onPathChange={(path) => setArtifactsRoot(path)}
            className="mb-4"
          />
        </div>
      </div>
    );
  }

  if (!selectedRunId) {
    return (
      <div className="space-y-6 p-6">
        {/* Cabeçalho */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-3 mb-4">
            <Database className="w-8 h-8 text-accent-cyan" />
            <h1 className="text-2xl font-bold">Explorador de Candidatos</h1>
          </div>
          <p className="text-terminal-muted">Selecione uma execução para explorar candidatos de estratégia</p>
        </div>

        {/* Grid Seletor de Execuções */}
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Clock className="w-5 h-5 text-terminal-muted" />
              Execuções Recentes
            </h2>
            <button
              onClick={() => fetchRecentRuns(10)}
              className="text-sm text-terminal-muted hover:text-terminal-text flex items-center gap-1"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
              Atualizar
            </button>
          </div>

          {isLoading && recentRuns.length === 0 ? (
            <div className="flex items-center justify-center h-48">
              <RefreshCw className="w-8 h-8 animate-spin text-terminal-muted" />
            </div>
          ) : recentRuns.length === 0 ? (
            <div className="text-center py-12 bg-terminal-surface rounded-xl border border-terminal-border">
              <Award className="w-12 h-12 mx-auto mb-4 text-terminal-muted opacity-50" />
              <p className="text-terminal-muted">Nenhuma execução encontrada</p>
              <p className="text-sm text-terminal-muted mt-1">Gere estratégias a partir do Cockpit</p>
            </div>
          ) : (
            <div className="grid gap-3">
              {recentRuns.map((run) => (
                <button
                  key={run.run_id}
                  onClick={() => loadRun(run.run_id)}
                  className="w-full p-4 bg-terminal-surface rounded-xl border border-terminal-border hover:border-profit transition-all text-left group"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-accent-cyan/20 to-profit/20 flex items-center justify-center">
                        <Award className="w-6 h-6 text-accent-cyan" />
                      </div>
                      <div>
                        <div className="font-medium group-hover:text-profit transition-colors">
                          {run.campaign_name || 'Unknown Campaign'}
                        </div>
                        <div className="text-sm text-terminal-muted font-mono">
                          {run.run_id}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-6">
                      <div className="text-right">
                        <div className="text-sm text-terminal-muted">Candidatos</div>
                        <div className="font-mono font-bold text-accent-cyan">
                          {run.candidates_count}
                        </div>
                      </div>
                      {run.best_oos_sharpe_net && (
                        <div className="text-right">
                          <div className="text-sm text-terminal-muted">Melhor Sharpe</div>
                          <div className="font-mono font-bold text-profit">
                            {run.best_oos_sharpe_net.toFixed(2)}
                          </div>
                        </div>
                      )}
                      <div className="text-right">
                        <div className="text-sm text-terminal-muted">Status</div>
                        <div className={`font-medium ${run.status === 'completed' ? 'text-profit' : 'text-accent-yellow'}`}>
                          {run.status === 'completed' ? 'Concluído' : run.status}
                        </div>
                      </div>
                      <ChevronDown className="w-5 h-5 text-terminal-muted -rotate-90 group-hover:text-profit transition-colors" />
                    </div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Execuções legadas de campanhas (se disponíveis) */}
        {runs.length > 0 && (
          <div className="max-w-4xl mx-auto mt-8">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <FolderOpen className="w-5 h-5 text-terminal-muted" />
              De Artefatos Locais
            </h2>
            <div className="flex flex-wrap gap-2">
              {runs.slice(0, 5).map(run => (
                <button
                  key={run.run_id}
                  onClick={() => loadRun(run.run_id)}
                  className="px-4 py-2 bg-terminal-surface border border-terminal-border rounded-lg text-sm font-mono hover:border-profit transition-colors"
                >
                  {run.run_id.substring(0, 16)}...
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  }

  // Get the selected run info
  const selectedRunInfo = recentRuns.find(r => r.run_id === selectedRunId);

  return (
    <div className="space-y-4">
      {/* Navegação Breadcrumb */}
      <div className="flex items-center gap-2 text-sm">
        <button 
          onClick={() => {
            useDataStore.getState().clearSelectedRun?.();
            window.location.hash = '#/candidates';
          }}
          className="text-terminal-muted hover:text-white transition-colors"
        >
          Execuções
        </button>
        <ChevronDown className="w-4 h-4 text-terminal-muted -rotate-90" />
        <span className="text-accent-cyan font-mono">{selectedRunInfo?.campaign_name || selectedRunId?.slice(0, 12)}</span>
        {selectedCandidate && (
          <>
            <ChevronDown className="w-4 h-4 text-terminal-muted -rotate-90" />
            <span className="text-profit">{selectedCandidate.display_name}</span>
          </>
        )}
      </div>

      {/* Cabeçalho */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Explorador de Candidatos</h1>
          <p className="text-terminal-muted text-sm mt-1">
            <span className="font-medium">{selectedRunInfo?.campaign_name || 'Campanha'}</span>
            {' — '}
            <span className="font-mono text-accent-cyan">{selectedRunId}</span>
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button 
            onClick={() => selectedRunId && listCandidates(selectedRunId)}
            className="flex items-center gap-2 px-3 py-2 rounded-lg bg-terminal-surface border border-terminal-border hover:border-profit transition-colors text-sm"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            Atualizar
          </button>
          <button className="flex items-center gap-2 px-3 py-2 rounded-lg bg-profit/10 text-profit border border-profit/30 hover:bg-profit/20 transition-all text-sm">
            <Download className="w-4 h-4" />
            Exportar
          </button>
        </div>
      </div>

      {/* Cards de Estatísticas */}
      <div className="grid grid-cols-4 gap-3">
        <div className="bg-terminal-surface rounded-xl border border-terminal-border p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-accent-cyan/20 flex items-center justify-center">
              <Award className="w-5 h-5 text-accent-cyan" />
            </div>
            <div>
              <div className="text-2xl font-bold">{candidatesList.length}</div>
              <div className="text-xs text-terminal-muted">Total de Candidatos</div>
            </div>
          </div>
        </div>
        <div className="bg-terminal-surface rounded-xl border border-terminal-border p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-profit/20 flex items-center justify-center">
              <CheckCircle className="w-5 h-5 text-profit" />
            </div>
            <div>
              <div className="text-2xl font-bold text-profit">{validatedCount}</div>
              <div className="text-xs text-terminal-muted">Validados</div>
            </div>
          </div>
        </div>
        <div className="bg-terminal-surface rounded-xl border border-terminal-border p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-accent-yellow/20 flex items-center justify-center">
              <Shield className="w-5 h-5 text-accent-yellow" />
            </div>
            <div>
              <div className="text-2xl font-bold text-accent-yellow">{lowPboCount}</div>
              <div className="text-xs text-terminal-muted">PBO &lt; 10%</div>
            </div>
          </div>
        </div>
        <div className="bg-terminal-surface rounded-xl border border-terminal-border p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-profit/20 flex items-center justify-center">
              <TrendingUp className="w-5 h-5 text-profit" />
            </div>
            <div>
              <div className="text-2xl font-bold">{bestSharpe.toFixed(2)}</div>
              <div className="text-xs text-terminal-muted">Melhor Sharpe</div>
            </div>
          </div>
        </div>
      </div>

      {/* Barra de Seleção */}
      {selectedCandidateIds.length > 0 && (
        <div className="flex items-center gap-4 p-3 bg-accent-cyan/10 border border-accent-cyan/30 rounded-lg">
          <span className="text-sm">
            <span className="font-mono text-accent-cyan font-bold">{selectedCandidateIds.length}</span> selecionados
          </span>
          <div className="flex-1" />
          <button
            onClick={handleCompare}
            disabled={selectedCandidateIds.length < 2}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-accent-cyan text-black font-medium text-sm hover:bg-accent-cyan/90 transition-colors disabled:opacity-50"
          >
            <GitCompare className="w-4 h-4" />
            Comparar
          </button>
          <button
            onClick={clearCandidateSelection}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-terminal-surface border border-terminal-border text-sm hover:border-loss transition-colors"
          >
            <X className="w-4 h-4" />
            Limpar
          </button>
        </div>
      )}

      {error && (
        <div className="p-4 bg-loss/10 border border-loss/30 rounded-lg text-loss">{error}</div>
      )}

      {/* Filtros */}
      <div className="flex items-center gap-3 flex-wrap">
        <div className="relative flex-1 min-w-[200px] max-w-sm">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-terminal-muted" />
          <input
            type="text"
            placeholder="Buscar estratégias..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-terminal-surface border border-terminal-border rounded-lg focus:outline-none focus:border-profit text-sm"
          />
        </div>
        
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-terminal-muted" />
          <select
            value={filterPbo ?? ''}
            onChange={(e) => setFilterPbo(e.target.value ? Number(e.target.value) : null)}
            className="px-3 py-2 bg-terminal-surface border border-terminal-border rounded-lg focus:outline-none focus:border-profit text-sm cursor-pointer"
          >
            <option value="">Todo PBO</option>
            <option value="0.10">PBO ≤ 10%</option>
            <option value="0.15">PBO ≤ 15%</option>
            <option value="0.20">PBO ≤ 20%</option>
          </select>
        </div>

        <select
          value={filterClass}
          onChange={(e) => setFilterClass(e.target.value)}
          className="px-3 py-2 bg-terminal-surface border border-terminal-border rounded-lg focus:outline-none focus:border-profit text-sm cursor-pointer"
        >
          <option value="">Todo Status</option>
          <option value="validated">Validado</option>
          <option value="research">Pesquisa</option>
        </select>
        
        {/* Toggle de Modo de Visualização */}
        <div className="flex items-center bg-terminal-surface border border-terminal-border rounded-lg overflow-hidden ml-auto">
          <button
            onClick={() => setViewMode('table')}
            className={`px-3 py-2 text-sm flex items-center gap-1.5 ${viewMode === 'table' ? 'bg-profit/20 text-profit' : 'text-terminal-muted hover:text-white'}`}
          >
            <Table className="w-4 h-4" />
            Tabela
          </button>
          <button
            onClick={() => setViewMode('pareto')}
            className={`px-3 py-2 text-sm flex items-center gap-1.5 ${viewMode === 'pareto' ? 'bg-profit/20 text-profit' : 'text-terminal-muted hover:text-white'}`}
          >
            <ScatterChart className="w-4 h-4" />
            Pareto
          </button>
        </div>
      </div>

      {/* Visualização Pareto Scatter */}
      {viewMode === 'pareto' && (
        <div className="bg-terminal-surface rounded-xl border border-terminal-border p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium flex items-center gap-2">
              <ScatterChart className="w-4 h-4 text-terminal-muted" />
              Fronteira de Pareto: Sharpe vs Drawdown Máximo
            </h3>
            <span className="text-xs text-terminal-muted">{candidatesList.length} candidatos</span>
          </div>
          <ParetoScatter
            data={candidatesList.map(c => ({
              id: c.candidate_id,
              sharpe: c.oos_sharpe_net,
              maxDrawdown: c.max_drawdown_net,
              cagr: c.oos_cagr_net,
              gatesPassed: c.gates_passed,
              displayName: c.display_name,
            }))}
            onPointClick={(id) => {
              loadCandidateDetail(id);
              setShowDetail(true);
            }}
            height={400}
          />
        </div>
      )}

      {/* Visualização em Tabela */}
      {viewMode === 'table' && (
        <div className="bg-terminal-surface rounded-xl border border-terminal-border overflow-hidden">
          {isLoading && candidatesList.length === 0 ? (
            <div className="flex items-center justify-center h-64">
              <RefreshCw className="w-8 h-8 animate-spin text-terminal-muted" />
            </div>
          ) : candidatesList.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-64 text-terminal-muted">
              <Award className="w-12 h-12 mb-4 opacity-50" />
              <p>Nenhum candidato encontrado</p>
              <p className="text-sm mt-1">Tente ajustar seus filtros</p>
            </div>
          ) : (
            <DataTable
              data={candidatesList as unknown as Record<string, unknown>[]}
              columns={columns}
              maxHeight="500px"
              onRowClick={handleRowClick}
            />
          )}
        </div>
      )}

      {/* Detail Drawer */}
      {showDetail && selectedCandidate && (
        <CandidateDetail candidate={selectedCandidate} onClose={handleCloseDetail} />
      )}
    </div>
  );
}

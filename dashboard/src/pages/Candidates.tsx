import { useState, useEffect } from 'react';
import { DataTable } from '../components/ui/DataTable';
import { MetricCard } from '../components/ui/MetricCard';
import { CandidateDetail } from '../components/CandidateDetail';
import { useDataStore } from '../stores/dataStore';
import type { CandidateListItem } from '../stores/dataStore';
import { open } from '@tauri-apps/plugin-dialog';
import { 
  Search, 
  Filter, 
  Download, 
  Award,
  CheckCircle,
  XCircle,
  AlertCircle,
  FolderOpen,
  RefreshCw,
  ChevronDown,
  GitCompare,
  Square,
  CheckSquare,
  X
} from 'lucide-react';

export function Candidates() {
  const [searchQuery, setSearchQuery] = useState('');
  const [filterPbo, setFilterPbo] = useState<number | null>(null);
  const [filterClass, setFilterClass] = useState<string>('');
  const [showDetail, setShowDetail] = useState(false);
  
  const { 
    artifactsRoot,
    candidates, 
    selectedCandidate,
    selectedRunId,
    selectedCandidateIds,
    runs,
    isLoading,
    error,
    listCandidates,
    loadCandidateDetail,
    clearSelectedCandidate,
    setArtifactsRoot,
    toggleCandidateSelection,
    clearCandidateSelection,
  } = useDataStore();

  // Apply filters when they change
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
    candidates.forEach(c => {
      if (!selectedCandidateIds.includes(c.candidate_id)) {
        toggleCandidateSelection(c.candidate_id);
      }
    });
  };

  const handleCompare = () => {
    window.dispatchEvent(new CustomEvent('navigate', { detail: 'comparison' }));
  };

  const columns = [
    // Selection checkbox column
    {
      key: '_select',
      header: (
        <button
          onClick={handleSelectAll}
          className="p-1 hover:bg-terminal-surface rounded"
          title="Select all"
        >
          <Square className="w-4 h-4 text-terminal-muted" />
        </button>
      ),
      width: '40px',
      render: (_: unknown, row: Record<string, unknown>) => {
        const isSelected = selectedCandidateIds.includes(row.candidate_id as string);
        return (
          <button
            onClick={(e) => handleToggleSelection(row.candidate_id as string, e)}
            className="p-1 hover:bg-terminal-surface rounded"
          >
            {isSelected ? (
              <CheckSquare className="w-4 h-4 text-profit" />
            ) : (
              <Square className="w-4 h-4 text-terminal-muted" />
            )}
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
        <span className="font-mono text-terminal-muted">{String(value)}</span>
      ),
    },
    {
      key: 'display_name',
      header: 'Strategy',
      sortable: true,
      render: (value: unknown, row: Record<string, unknown>) => (
        <div className="max-w-md">
          <div className="font-medium text-sm truncate" title={String(value)}>
            {String(value)}
          </div>
          <div className="text-xs text-terminal-muted font-mono">
            {String(row.candidate_id).substring(0, 16)}...
          </div>
        </div>
      ),
    },
    {
      key: 'candidate_class',
      header: 'Class',
      sortable: true,
      width: '90px',
      render: (value: unknown) => {
        const cls = String(value);
        const color = cls === 'validated' ? 'bg-profit/20 text-profit' : 'bg-accent-cyan/20 text-accent-cyan';
        return (
          <span className={`px-2 py-0.5 rounded text-xs font-medium ${color}`}>
            {cls}
          </span>
        );
      },
    },
    {
      key: 'oos_sharpe_net',
      header: 'Sharpe NET',
      sortable: true,
      align: 'right' as const,
      render: (value: unknown) => {
        const v = value as number;
        const color = v >= 1.0 ? 'text-profit' : v >= 0.5 ? 'text-accent-yellow' : 'text-loss';
        return <span className={`font-mono ${color}`}>{v.toFixed(2)}</span>;
      },
    },
    {
      key: 'pbo',
      header: 'PBO',
      sortable: true,
      align: 'right' as const,
      width: '80px',
      render: (value: unknown) => {
        const v = value as number;
        const color = v <= 0.10 ? 'text-profit' : v <= 0.15 ? 'text-accent-yellow' : 'text-loss';
        return <span className={`font-mono ${color}`}>{(v * 100).toFixed(1)}%</span>;
      },
    },
    {
      key: 'oos_cagr_net',
      header: 'CAGR NET',
      sortable: true,
      align: 'right' as const,
      width: '90px',
      render: (value: unknown) => (
        <span className="font-mono">{((value as number) * 100).toFixed(1)}%</span>
      ),
    },
    {
      key: 'max_drawdown_net',
      header: 'Max DD',
      sortable: true,
      align: 'right' as const,
      width: '80px',
      render: (value: unknown) => {
        const v = Math.abs(value as number);
        return <span className="font-mono text-loss">-{(v * 100).toFixed(1)}%</span>;
      },
    },
    {
      key: 'dsr',
      header: 'DSR',
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
      header: 'Stress',
      sortable: true,
      align: 'center' as const,
      width: '70px',
      render: (value: unknown, row: Record<string, unknown>) => {
        const passed = value as number;
        const total = row.stress_total as number;
        const pct = total > 0 ? passed / total : 0;
        const color = pct >= 0.8 ? 'text-profit' : pct >= 0.6 ? 'text-accent-yellow' : 'text-loss';
        return <span className={`font-mono ${color}`}>{passed}/{total}</span>;
      },
    },
    {
      key: 'gates_passed',
      header: 'Gates',
      align: 'center' as const,
      width: '60px',
      render: (value: unknown) => {
        const passed = value as boolean;
        return passed 
          ? <CheckCircle className="w-4 h-4 text-profit mx-auto" />
          : <XCircle className="w-4 h-4 text-loss mx-auto" />;
      },
    },
  ];

  // Calculate stats
  const passedCount = candidates.filter(c => c.gates_passed && c.pbo <= 0.15).length;
  const warningCount = candidates.filter(c => c.gates_passed && c.pbo > 0.15).length;
  const bestSharpe = candidates.length > 0 
    ? Math.max(...candidates.map(c => c.oos_sharpe_net)) 
    : 0;

  // Show setup message if no artifacts root
  if (!artifactsRoot) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-6">
        <FolderOpen className="w-16 h-16 text-terminal-muted" />
        <div className="text-center">
          <h2 className="text-xl font-semibold mb-2">No Project Selected</h2>
          <p className="text-terminal-muted mb-4">
            Select a project folder containing SCG artifacts to get started.
          </p>
          <button
            onClick={handleSelectFolder}
            className="px-6 py-3 bg-profit text-black font-medium rounded-lg hover:bg-profit/90 transition-colors"
          >
            Select Project Folder
          </button>
        </div>
      </div>
    );
  }

  // Show run selector if no run selected
  if (!selectedRunId) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-6">
        <Award className="w-16 h-16 text-terminal-muted" />
        <div className="text-center">
          <h2 className="text-xl font-semibold mb-2">No Run Selected</h2>
          <p className="text-terminal-muted mb-4">
            Select a campaign and run from the Campaigns page to view candidates.
          </p>
          {runs.length > 0 && (
            <div className="mt-4">
              <p className="text-sm text-terminal-muted mb-2">Or select a run:</p>
              <div className="flex flex-wrap gap-2 justify-center">
                {runs.slice(0, 5).map(run => (
                  <button
                    key={run.run_id}
                    onClick={() => useDataStore.getState().loadRun(run.run_id)}
                    className="px-3 py-1.5 bg-terminal-surface border border-terminal-border rounded text-sm font-mono hover:border-profit transition-colors"
                  >
                    {run.run_id.substring(0, 12)}... (seed:{run.seed})
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Candidate Explorer</h1>
          <p className="text-terminal-muted mt-1">
            Run: <span className="font-mono text-accent-cyan">{selectedRunId}</span>
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button 
            onClick={() => selectedRunId && listCandidates(selectedRunId)}
            className="flex items-center gap-2 px-3 py-2 rounded-lg bg-terminal-surface border border-terminal-border hover:border-profit transition-colors"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
          <button className="flex items-center gap-2 px-4 py-2 rounded-lg bg-profit/10 text-profit border border-profit/30 hover:bg-profit/20 transition-all">
            <Download className="w-4 h-4" />
            Export CSV
          </button>
        </div>
      </div>

      {/* Selection Bar */}
      {selectedCandidateIds.length > 0 && (
        <div className="flex items-center gap-4 p-3 bg-accent-cyan/10 border border-accent-cyan/30 rounded-lg">
          <span className="text-sm">
            <span className="font-mono text-accent-cyan">{selectedCandidateIds.length}</span> candidates selected
          </span>
          <div className="flex-1" />
          <button
            onClick={handleCompare}
            disabled={selectedCandidateIds.length < 2}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-accent-cyan text-black font-medium text-sm hover:bg-accent-cyan/90 transition-colors disabled:opacity-50"
          >
            <GitCompare className="w-4 h-4" />
            Compare Selected
          </button>
          <button
            onClick={clearCandidateSelection}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-terminal-surface border border-terminal-border text-sm hover:border-loss transition-colors"
          >
            <X className="w-4 h-4" />
            Clear
          </button>
        </div>
      )}

      {/* Error message */}
      {error && (
        <div className="p-4 bg-loss/10 border border-loss/30 rounded-lg text-loss">
          {error}
        </div>
      )}

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4">
        <MetricCard
          label="Total Candidates"
          value={candidates.length}
          icon={<Award className="w-5 h-5" />}
        />
        <MetricCard
          label="Passed All Gates"
          value={passedCount}
          icon={<CheckCircle className="w-5 h-5 text-profit" />}
        />
        <MetricCard
          label="Warnings (PBO>15%)"
          value={warningCount}
          icon={<AlertCircle className="w-5 h-5 text-accent-yellow" />}
        />
        <MetricCard
          label="Best Sharpe NET"
          value={bestSharpe}
          format="ratio"
        />
      </div>

      {/* Filters */}
      <div className="flex items-center gap-4 flex-wrap">
        <div className="relative flex-1 min-w-[200px] max-w-sm">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-terminal-muted" />
          <input
            type="text"
            placeholder="Search by strategy or ID..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-terminal-bg border border-terminal-border rounded-lg focus:outline-none focus:border-profit text-sm"
          />
        </div>
        
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-terminal-muted" />
          <select
            value={filterPbo ?? ''}
            onChange={(e) => setFilterPbo(e.target.value ? Number(e.target.value) : null)}
            className="px-3 py-2 bg-terminal-bg border border-terminal-border rounded-lg focus:outline-none focus:border-profit text-sm appearance-none pr-8"
          >
            <option value="">All PBO</option>
            <option value="0.10">PBO &le; 10%</option>
            <option value="0.15">PBO &le; 15%</option>
            <option value="0.20">PBO &le; 20%</option>
          </select>
          <ChevronDown className="w-4 h-4 -ml-6 text-terminal-muted pointer-events-none" />
        </div>

        <div className="flex items-center gap-2">
          <select
            value={filterClass}
            onChange={(e) => setFilterClass(e.target.value)}
            className="px-3 py-2 bg-terminal-bg border border-terminal-border rounded-lg focus:outline-none focus:border-profit text-sm appearance-none pr-8"
          >
            <option value="">All Classes</option>
            <option value="validated">Validated</option>
            <option value="research">Research</option>
          </select>
          <ChevronDown className="w-4 h-4 -ml-6 text-terminal-muted pointer-events-none" />
        </div>
      </div>

      {/* Table */}
      <div className="card-elevated p-0 overflow-hidden">
        {isLoading && candidates.length === 0 ? (
          <div className="flex items-center justify-center h-64">
            <RefreshCw className="w-8 h-8 animate-spin text-terminal-muted" />
          </div>
        ) : candidates.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-terminal-muted">
            <Award className="w-12 h-12 mb-4 opacity-50" />
            <p>No candidates found</p>
            <p className="text-sm mt-1">Try adjusting your filters</p>
          </div>
        ) : (
          <DataTable
            data={candidates as unknown as Record<string, unknown>[]}
            columns={columns}
            maxHeight="500px"
            onRowClick={handleRowClick}
          />
        )}
      </div>

      {/* Candidate Detail Drawer */}
      {showDetail && selectedCandidate && (
        <CandidateDetail
          candidate={selectedCandidate}
          onClose={handleCloseDetail}
        />
      )}
    </div>
  );
}

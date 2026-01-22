/**
 * Runs - List of all experiments with links to details and diagnostics
 */

import { useEffect, useState } from 'react';
import { 
  FolderOpen, Clock, Trophy, ChevronRight, RefreshCw, 
  AlertTriangle, CheckCircle, Calendar
} from 'lucide-react';
import platformConfig from '../lib/platform';

interface RunEntry {
  experimentId: string;
  artifactsPath: string;
  createdAt: string;
  modifiedAt: string;
  hofCount: number;
  manifest?: {
    generations?: number;
    duration_secs?: number;
    market?: string;
    config?: string;
  };
}

export function Runs() {
  const [runs, setRuns] = useState<RunEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [runDetail, setRunDetail] = useState<any>(null);
  const [diagnostics, setDiagnostics] = useState<any>(null);

  const fetchRuns = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${platformConfig.config.apiBase}/omp/runs`);
      const data = await res.json();
      setRuns(data.runs || []);
      setError(null);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchRunDetail = async (experimentId: string) => {
    try {
      const [detailRes, diagRes] = await Promise.all([
        fetch(`${platformConfig.config.apiBase}/omp/runs/${experimentId}`),
        fetch(`${platformConfig.config.apiBase}/omp/diagnostics/${experimentId}`)
      ]);
      const detail = await detailRes.json();
      const diag = await diagRes.json();
      setRunDetail(detail);
      setDiagnostics(diag);
    } catch (e) {
      console.error('Failed to fetch run details', e);
    }
  };

  useEffect(() => {
    fetchRuns();
  }, []);

  useEffect(() => {
    if (selectedRun) {
      fetchRunDetail(selectedRun);
    } else {
      setRunDetail(null);
      setDiagnostics(null);
    }
  }, [selectedRun]);

  const formatDate = (iso: string) => {
    const d = new Date(iso);
    return d.toLocaleDateString() + ' ' + d.toLocaleTimeString();
  };

  const formatDuration = (secs: number) => {
    if (!secs) return '—';
    const h = Math.floor(secs / 3600);
    const m = Math.floor((secs % 3600) / 60);
    return h > 0 ? `${h}h ${m}m` : `${m}m`;
  };

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      {/* Header */}
      <div className="border-b border-slate-800 px-6 py-4 bg-slate-900/50">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FolderOpen className="w-6 h-6 text-cyan-400" />
            <div>
              <h1 className="text-xl font-bold text-white">Experiment Runs</h1>
              <p className="text-sm text-slate-400">Browse all mining runs and artifacts</p>
            </div>
          </div>
          <button
            onClick={fetchRuns}
            className="flex items-center gap-2 px-3 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      <div className="max-w-6xl mx-auto p-6">
        {error && (
          <div className="mb-4 p-3 bg-rose-500/10 border border-rose-500/30 rounded-lg flex items-center gap-2 text-rose-400">
            <AlertTriangle className="w-4 h-4" />
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Runs List */}
          <div className="lg:col-span-1">
            <h2 className="text-sm text-slate-400 font-medium mb-3">
              {runs.length} Experiment{runs.length !== 1 ? 's' : ''}
            </h2>
            <div className="space-y-2 max-h-[calc(100vh-200px)] overflow-y-auto">
              {runs.map((run) => (
                <button
                  key={run.experimentId}
                  onClick={() => setSelectedRun(run.experimentId)}
                  className={`w-full text-left p-3 rounded-lg border transition-all ${
                    selectedRun === run.experimentId
                      ? 'bg-cyan-500/20 border-cyan-500/50'
                      : 'bg-slate-800/60 border-slate-700 hover:border-slate-600'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-sm text-white truncate">
                      {run.experimentId}
                    </span>
                    <ChevronRight className="w-4 h-4 text-slate-500" />
                  </div>
                  <div className="flex items-center gap-4 mt-2 text-xs text-slate-400">
                    <span className="flex items-center gap-1">
                      <Calendar className="w-3 h-3" />
                      {new Date(run.createdAt).toLocaleDateString()}
                    </span>
                    <span className="flex items-center gap-1 text-amber-400">
                      <Trophy className="w-3 h-3" />
                      {run.hofCount}
                    </span>
                    {run.manifest?.generations && (
                      <span className="text-violet-400">
                        G{run.manifest.generations}
                      </span>
                    )}
                  </div>
                </button>
              ))}
              {runs.length === 0 && !loading && (
                <div className="text-center py-8 text-slate-500">
                  No experiments found
                </div>
              )}
            </div>
          </div>

          {/* Run Detail */}
          <div className="lg:col-span-2">
            {selectedRun && runDetail ? (
              <div className="space-y-4">
                {/* Header */}
                <div className="bg-slate-800/60 rounded-xl border border-slate-700 p-4">
                  <h2 className="text-lg font-bold text-white mb-2">{selectedRun}</h2>
                  <p className="text-sm text-slate-400 font-mono">{runDetail.artifactsPath}</p>
                </div>

                {/* Files */}
                <div className="bg-slate-800/60 rounded-xl border border-slate-700 p-4">
                  <h3 className="text-sm text-slate-400 font-medium mb-3">Artifacts</h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                    {runDetail.files?.map((file: string) => (
                      <div 
                        key={file}
                        className="px-2 py-1 bg-slate-700/50 rounded text-xs font-mono text-slate-300 truncate"
                        title={file}
                      >
                        {file}
                      </div>
                    ))}
                  </div>
                </div>

                {/* HoF Strategies */}
                {runDetail.hofStrategies?.length > 0 && (
                  <div className="bg-slate-800/60 rounded-xl border border-amber-500/30 p-4">
                    <h3 className="text-sm text-amber-400 font-medium mb-3 flex items-center gap-2">
                      <Trophy className="w-4 h-4" />
                      Hall of Fame ({runDetail.hofStrategies.length})
                    </h3>
                    <div className="flex flex-wrap gap-2">
                      {runDetail.hofStrategies.map((strat: string) => (
                        <span 
                          key={strat}
                          className="px-2 py-1 bg-amber-500/20 rounded text-xs font-mono text-amber-300"
                        >
                          {strat}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Diagnostics */}
                {diagnostics && (
                  <div className="bg-slate-800/60 rounded-xl border border-slate-700 p-4">
                    <h3 className="text-sm text-slate-400 font-medium mb-3">Diagnostics</h3>
                    
                    {/* Failure Breakdown */}
                    {Object.keys(diagnostics.failureBreakdown || {}).length > 0 && (
                      <div className="mb-4">
                        <h4 className="text-xs text-slate-500 mb-2">Failure Breakdown</h4>
                        <div className="space-y-1">
                          {Object.entries(diagnostics.failureBreakdown)
                            .sort((a: any, b: any) => b[1] - a[1])
                            .slice(0, 5)
                            .map(([reason, count]: [string, any]) => (
                              <div key={reason} className="flex items-center gap-2">
                                <div className="flex-1 bg-slate-700 rounded-full h-2">
                                  <div 
                                    className="bg-rose-500 h-2 rounded-full"
                                    style={{ 
                                      width: `${Math.min(100, (count / Object.values(diagnostics.failureBreakdown).reduce((a: any, b: any) => a + b, 0)) * 100)}%` 
                                    }}
                                  />
                                </div>
                                <span className="text-xs text-slate-400 w-20 truncate">{reason}</span>
                                <span className="text-xs text-rose-400 w-8 text-right">{count}</span>
                              </div>
                            ))
                          }
                        </div>
                      </div>
                    )}

                    {/* Near Misses */}
                    {diagnostics.nearMisses?.length > 0 && (
                      <div>
                        <h4 className="text-xs text-slate-500 mb-2">Near Misses (Closest to Pass)</h4>
                        <div className="space-y-1">
                          {diagnostics.nearMisses.slice(0, 5).map((nm: any, i: number) => (
                            <div key={i} className="flex items-center justify-between text-xs">
                              <span className="font-mono text-slate-400 truncate max-w-[100px]">
                                {nm.id?.slice(-8) || `#${i+1}`}
                              </span>
                              <span className="text-emerald-400">Sharpe {nm.sharpe?.toFixed(2)}</span>
                              <span className="text-rose-400">DD {(nm.drawdown * 100)?.toFixed(0)}%</span>
                              <span className="text-violet-400">{nm.reasonCount} fail</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <div className="flex items-center justify-center h-64 text-slate-500">
                Select an experiment to view details
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Runs;

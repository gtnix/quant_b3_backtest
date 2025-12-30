/**
 * HallOfFame - Display promoted strategies that passed all institutional criteria
 */

import { useEffect, useState } from 'react';
import { 
  Trophy, TrendingUp, Shield, BarChart3, Clock, 
  GitBranch, Hash, Globe, ChevronDown, RefreshCw,
  CheckCircle2, XCircle, Filter
} from 'lucide-react';
import { useOmpStore } from '../stores/ompStore';
import type { HallOfFameEntry } from '../stores/ompStore';

// =============================================================================
// FILTER COMPONENT
// =============================================================================

interface FilterState {
  market: 'all' | 'br' | 'us';
  sortBy: 'sharpe' | 'date' | 'pbo' | 'dsr';
  limit: number;
}

function FilterBar({ filter, setFilter }: { filter: FilterState; setFilter: (f: FilterState) => void }) {
  return (
    <div className="flex flex-wrap items-center gap-4 p-4 bg-slate-800/50 rounded-xl border border-slate-700">
      <div className="flex items-center gap-2">
        <Filter className="w-4 h-4 text-slate-400" />
        <span className="text-sm text-slate-400">Filters:</span>
      </div>
      
      {/* Market Filter */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-slate-500">Market:</span>
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
              {m === 'all' ? 'All' : m.toUpperCase()}
            </button>
          ))}
        </div>
      </div>
      
      {/* Sort By */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-slate-500">Sort:</span>
        <select
          value={filter.sortBy}
          onChange={e => setFilter({ ...filter, sortBy: e.target.value as FilterState['sortBy'] })}
          className="px-3 py-1 text-xs bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-amber-500"
        >
          <option value="sharpe">Sharpe (High → Low)</option>
          <option value="date">Date (Recent → Old)</option>
          <option value="pbo">PBO (Low → High)</option>
          <option value="dsr">DSR (High → Low)</option>
        </select>
      </div>
      
      {/* Limit */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-slate-500">Show:</span>
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
// ENTRY CARD COMPONENT
// =============================================================================

function EntryCard({ entry, rank }: { entry: HallOfFameEntry; rank: number }) {
  const [expanded, setExpanded] = useState(false);
  
  const metrics = entry.metrics;
  const validation = entry.validation;
  
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
            <span className="font-mono text-white text-sm truncate">{entry.candidateId}</span>
            <span className={`px-2 py-0.5 text-xs rounded-full ${
              entry.market === 'br' 
                ? 'bg-green-500/20 text-green-400' 
                : 'bg-blue-500/20 text-blue-400'
            }`}>
              {entry.market.toUpperCase()}
            </span>
          </div>
          <div className="text-xs text-slate-500 mt-0.5">
            {entry.campaignName || entry.campaignId}
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
            {/* Metrics */}
            <div>
              <h4 className="text-xs text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <BarChart3 className="w-3 h-3" /> Metrics
              </h4>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-500">OOS Sharpe NET</span>
                  <span className="text-emerald-400 font-mono">{metrics.oosSharpeNet?.toFixed(4)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">CAGR NET</span>
                  <span className="text-white font-mono">{((metrics.cagrNet || 0) * 100).toFixed(2)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Max Drawdown</span>
                  <span className="text-rose-400 font-mono">{(Math.abs(metrics.maxDrawdownNet || 0) * 100).toFixed(2)}%</span>
                </div>
              </div>
            </div>
            
            {/* Validation */}
            <div>
              <h4 className="text-xs text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <Shield className="w-3 h-3" /> Validation
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
                  <span className="text-slate-500">Gates Passed</span>
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
            
            {/* Provenance */}
            <div>
              <h4 className="text-xs text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <GitBranch className="w-3 h-3" /> Provenance
              </h4>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-500">Git SHA</span>
                  <span className="text-white font-mono text-xs">{entry.provenance.gitSha?.slice(0, 7) || '—'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Config Hash</span>
                  <span className="text-white font-mono text-xs">{entry.provenance.configHash?.slice(0, 7) || '—'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Genome Hash</span>
                  <span className="text-white font-mono text-xs">{entry.genomeHash?.slice(0, 7) || '—'}</span>
                </div>
              </div>
            </div>
            
            {/* Timing */}
            <div>
              <h4 className="text-xs text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <Clock className="w-3 h-3" /> Timing
              </h4>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-500">Promoted</span>
                  <span className="text-white text-xs">{new Date(entry.promotedAt).toLocaleDateString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Run ID</span>
                  <span className="text-white font-mono text-xs">{entry.runId?.slice(0, 8)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Campaign</span>
                  <span className="text-white text-xs truncate max-w-[100px]">{entry.campaignId?.slice(0, 8)}</span>
                </div>
              </div>
            </div>
          </div>
          
          {/* Notes */}
          {entry.notes && (
            <div className="mt-4 pt-4 border-t border-slate-700">
              <p className="text-xs text-slate-500">Notes: {entry.notes}</p>
            </div>
          )}
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
        
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-white flex items-center gap-3">
              <Trophy className="w-6 h-6 text-amber-400" />
              Hall of Fame
            </h1>
            <p className="text-slate-400 text-sm mt-1">
              Elite strategies that passed all institutional validation gates
            </p>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="text-right">
              <p className="text-2xl font-bold text-amber-400">{stats?.promotions.total || hallOfFame.length}</p>
              <p className="text-xs text-slate-500">Total Promoted</p>
            </div>
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
        
        {/* Stats Summary */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700 text-center">
            <p className="text-2xl font-bold text-emerald-400">{stats?.promotions.last24h || 0}</p>
            <p className="text-xs text-slate-500">Promoted (24h)</p>
          </div>
          <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700 text-center">
            <p className="text-2xl font-bold text-white">{stats?.promotions.last7d || 0}</p>
            <p className="text-xs text-slate-500">Promoted (7d)</p>
          </div>
          <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700 text-center">
            <p className="text-2xl font-bold text-white">
              {sortedEntries.length > 0 ? sortedEntries[0].metrics.oosSharpeNet?.toFixed(3) : '—'}
            </p>
            <p className="text-xs text-slate-500">Best Sharpe</p>
          </div>
          <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700 text-center">
            <p className="text-2xl font-bold text-white">
              {sortedEntries.length > 0 
                ? ((sortedEntries.reduce((sum, e) => sum + (e.metrics.oosSharpeNet || 0), 0) / sortedEntries.length)).toFixed(3)
                : '—'
              }
            </p>
            <p className="text-xs text-slate-500">Avg Sharpe</p>
          </div>
        </div>
        
        {/* Entries List */}
        {hallOfFameLoading ? (
          <div className="flex items-center justify-center py-12">
            <RefreshCw className="w-8 h-8 text-slate-500 animate-spin" />
          </div>
        ) : sortedEntries.length === 0 ? (
          <div className="text-center py-12 rounded-xl border border-slate-700 bg-slate-800/30">
            <Trophy className="w-12 h-12 text-slate-600 mx-auto mb-3" />
            <p className="text-slate-400">No promoted strategies yet</p>
            <p className="text-sm text-slate-500 mt-1">Strategies will appear here once they pass all validation gates</p>
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


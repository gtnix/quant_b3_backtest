import { useState, useEffect } from 'react';
import { config } from '../lib/platform';
import { QuickTooltip } from './ui/TooltipInfo';
import { 
  RefreshCw,
  TrendingUp,
  TrendingDown,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Calendar,
  BarChart3,
  Layers
} from 'lucide-react';

interface WFAFold {
  fold: number;
  is_period: { start: string; end: string; days: number };
  oos_period: { start: string; end: string; days: number };
  is_metrics: { sharpe: number; cagr: number; max_dd: number };
  oos_metrics: { sharpe: number; cagr: number; max_dd: number };
  degradation: number;
  status: 'PASS' | 'WARN' | 'FAIL';
}

interface WFAData {
  candidate_id: string;
  wfa_config: {
    method: string;
    is_ratio: number;
    oos_ratio: number;
    num_folds: number;
    min_samples: number;
  };
  folds: WFAFold[];
  summary: {
    total_folds: number;
    passed_folds: number;
    avg_degradation: number;
    consistency_score: number;
    overall_status: 'PASS' | 'FAIL';
  };
}

interface Props {
  candidateId: string;
}

export function WFAAnalysis({ candidateId }: Props) {
  const [data, setData] = useState<WFAData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadWFA();
  }, [candidateId]);

  const loadWFA = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${config.apiBase}/candidate/${candidateId}/wfa`);
      if (response.ok) {
        const result = await response.json();
        setData(result);
      } else {
        setError('Failed to load WFA data');
      }
    } catch (err) {
      setError('Connection error');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="space-y-6 animate-pulse">
        <div className="grid grid-cols-4 gap-4">
          {[1, 2, 3, 4].map(i => (
            <div key={i} className="p-4 rounded-xl bg-terminal-surface border border-terminal-border">
              <div className="h-3 w-16 bg-terminal-border rounded mb-2" />
              <div className="h-6 w-20 bg-terminal-border rounded" />
            </div>
          ))}
        </div>
        <div className="space-y-3">
          {[1, 2, 3].map(i => (
            <div key={i} className="p-4 rounded-xl bg-terminal-surface border border-terminal-border">
              <div className="h-4 w-24 bg-terminal-border rounded mb-3" />
              <div className="grid grid-cols-2 gap-4">
                <div className="h-20 bg-terminal-border/50 rounded" />
                <div className="h-20 bg-terminal-border/50 rounded" />
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (error || !data || !data.folds || !data.summary || !data.wfa_config) {
    return (
      <div className="flex items-center justify-center h-64 text-terminal-muted">
        {error || 'No WFA data'}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Summary Header */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <SummaryCard
          label={<span className="inline-flex items-center">WFA Status<QuickTooltip termKey="wfa" /></span>}
          value={data.summary.overall_status}
          isStatus
          passed={data.summary.overall_status === 'PASS'}
        />
        <SummaryCard
          label="Passed Folds"
          value={`${data.summary.passed_folds} / ${data.summary.total_folds}`}
          passed={data.summary.passed_folds >= Math.ceil(data.summary.total_folds * 0.6)}
        />
        <SummaryCard
          label={<span className="inline-flex items-center">Degradation<QuickTooltip termKey="degradation_ratio" /></span>}
          value={`${data.summary.avg_degradation.toFixed(1)}%`}
          passed={data.summary.avg_degradation < 40}
        />
        <SummaryCard
          label={<span className="inline-flex items-center">Consistency<QuickTooltip termKey="consistency_score" /></span>}
          value={`${data.summary.consistency_score}%`}
          passed={data.summary.consistency_score >= 60}
        />
      </div>

      {/* Config Info */}
      <div className="flex items-center gap-6 text-xs text-terminal-muted">
        <span className="flex items-center gap-1">
          <Layers className="w-3 h-3" />
          Method: {data.wfa_config.method}
        </span>
        <span>IS/OOS: {(data.wfa_config.is_ratio * 100).toFixed(0)}% / {(data.wfa_config.oos_ratio * 100).toFixed(0)}%</span>
        <span>Min Samples: {data.wfa_config.min_samples}</span>
      </div>

      {/* Folds Timeline */}
      <div className="space-y-3">
        <h4 className="text-sm font-medium text-terminal-muted uppercase tracking-wider">Walk-Forward Folds</h4>
        
        {data.folds.map((fold) => (
          <FoldCard key={fold.fold} fold={fold} />
        ))}
      </div>

      {/* Degradation Chart */}
      <div className="p-4 rounded-xl bg-terminal-surface border border-terminal-border">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-sm font-medium text-terminal-muted">Sharpe Degradation by Fold</h4>
          <div className="flex items-center gap-2 text-xs text-terminal-muted">
            <span>Avg: </span>
            <span className={`font-mono font-bold ${data.summary.avg_degradation < 40 ? 'text-profit' : data.summary.avg_degradation < 50 ? 'text-accent-yellow' : 'text-loss'}`}>
              {data.summary.avg_degradation.toFixed(1)}%
            </span>
          </div>
        </div>
        <div className="flex items-end gap-2 h-32">
          {data.folds.map((fold) => {
            const height = Math.min(100, Math.max(10, fold.degradation * 2.5));
            const color = fold.status === 'PASS' ? 'bg-profit' : fold.status === 'WARN' ? 'bg-accent-yellow' : 'bg-loss';
            const hoverColor = fold.status === 'PASS' ? 'hover:bg-profit/80' : fold.status === 'WARN' ? 'hover:bg-accent-yellow/80' : 'hover:bg-loss/80';
            
            return (
              <div key={fold.fold} className="flex-1 flex flex-col items-center gap-1 group relative">
                <div className="w-full flex flex-col justify-end h-24">
                  <div 
                    className={`w-full rounded-t ${color} ${hoverColor} transition-all cursor-pointer`}
                    style={{ height: `${height}%` }}
                  />
                </div>
                {/* Tooltip on hover */}
                <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                  <div className="bg-terminal-bg border border-terminal-border rounded-lg p-2 text-xs whitespace-nowrap shadow-lg">
                    <div className="font-medium mb-1">Fold {fold.fold}</div>
                    <div className="flex justify-between gap-4">
                      <span className="text-terminal-muted">IS Sharpe:</span>
                      <span className="font-mono text-profit">{fold.is_metrics.sharpe.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between gap-4">
                      <span className="text-terminal-muted">OOS Sharpe:</span>
                      <span className="font-mono text-accent-cyan">{fold.oos_metrics.sharpe.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between gap-4 border-t border-terminal-border mt-1 pt-1">
                      <span className="text-terminal-muted">Degradation:</span>
                      <span className={`font-mono font-bold ${fold.status === 'PASS' ? 'text-profit' : fold.status === 'WARN' ? 'text-accent-yellow' : 'text-loss'}`}>
                        -{fold.degradation.toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
                <span className="text-xs text-terminal-muted">F{fold.fold}</span>
                <span className="text-xs font-mono">{fold.degradation.toFixed(0)}%</span>
              </div>
            );
          })}
        </div>
        <div className="mt-4 flex items-center justify-center gap-6 text-xs">
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-profit" /> &lt;40% (Pass)</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-accent-yellow" /> 40-50% (Warn)</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-loss" /> &gt;50% (Fail)</span>
        </div>
      </div>
    </div>
  );
}

function SummaryCard({ label, value, passed, isStatus }: { label: React.ReactNode; value: string; passed: boolean; isStatus?: boolean }) {
  return (
    <div className={`p-4 rounded-xl border ${passed ? 'bg-profit/5 border-profit/30' : 'bg-loss/5 border-loss/30'}`}>
      <div className="text-xs text-terminal-muted uppercase tracking-wider mb-2">{label}</div>
      <div className="flex items-center gap-2">
        {isStatus ? (
          passed ? <CheckCircle className="w-5 h-5 text-profit" /> : <XCircle className="w-5 h-5 text-loss" />
        ) : null}
        <span className={`text-xl font-bold font-mono ${passed ? 'text-profit' : 'text-loss'}`}>
          {value}
        </span>
      </div>
    </div>
  );
}

function FoldCard({ fold }: { fold: WFAFold }) {
  const statusIcon = fold.status === 'PASS' ? (
    <CheckCircle className="w-4 h-4 text-profit" />
  ) : fold.status === 'WARN' ? (
    <AlertTriangle className="w-4 h-4 text-accent-yellow" />
  ) : (
    <XCircle className="w-4 h-4 text-loss" />
  );

  const degradationColor = fold.status === 'PASS' ? 'text-profit' : fold.status === 'WARN' ? 'text-accent-yellow' : 'text-loss';

  return (
    <div className="p-4 rounded-xl bg-terminal-surface border border-terminal-border">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          {statusIcon}
          <span className="font-medium">Fold {fold.fold}</span>
          <span className={`text-xs font-medium px-2 py-0.5 rounded ${
            fold.status === 'PASS' ? 'bg-profit/20 text-profit' : 
            fold.status === 'WARN' ? 'bg-accent-yellow/20 text-accent-yellow' : 
            'bg-loss/20 text-loss'
          }`}>
            {fold.status}
          </span>
        </div>
        <div className={`font-mono font-bold ${degradationColor}`}>
          -{fold.degradation.toFixed(1)}% degradation
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* In-Sample */}
        <div className="p-3 rounded-lg bg-terminal-bg/50">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="w-4 h-4 text-accent-cyan" />
            <span className="text-xs font-medium text-terminal-muted uppercase inline-flex items-center">In-Sample<QuickTooltip termKey="is_oos" /></span>
          </div>
          <div className="text-xs text-terminal-muted flex items-center gap-1 mb-2">
            <Calendar className="w-3 h-3" />
            {fold.is_period.start} → {fold.is_period.end} ({fold.is_period.days}d)
          </div>
          <div className="grid grid-cols-3 gap-2 text-center">
            <div>
              <div className="text-[10px] text-terminal-muted">Sharpe</div>
              <div className="font-mono font-bold text-profit">{fold.is_metrics.sharpe.toFixed(2)}</div>
            </div>
            <div>
              <div className="text-[10px] text-terminal-muted">CAGR</div>
              <div className="font-mono font-bold">{(fold.is_metrics.cagr * 100).toFixed(1)}%</div>
            </div>
            <div>
              <div className="text-[10px] text-terminal-muted">Max DD</div>
              <div className="font-mono font-bold text-loss">{(fold.is_metrics.max_dd * 100).toFixed(1)}%</div>
            </div>
          </div>
        </div>

        {/* Out-of-Sample */}
        <div className="p-3 rounded-lg bg-terminal-bg/50">
          <div className="flex items-center gap-2 mb-2">
            <TrendingDown className="w-4 h-4 text-accent-purple" />
            <span className="text-xs font-medium text-terminal-muted uppercase">Out-of-Sample</span>
          </div>
          <div className="text-xs text-terminal-muted flex items-center gap-1 mb-2">
            <Calendar className="w-3 h-3" />
            {fold.oos_period.start} → {fold.oos_period.end} ({fold.oos_period.days}d)
          </div>
          <div className="grid grid-cols-3 gap-2 text-center">
            <div>
              <div className="text-[10px] text-terminal-muted">Sharpe</div>
              <div className={`font-mono font-bold ${fold.oos_metrics.sharpe >= 0.5 ? 'text-profit' : 'text-accent-yellow'}`}>
                {fold.oos_metrics.sharpe.toFixed(2)}
              </div>
            </div>
            <div>
              <div className="text-[10px] text-terminal-muted">CAGR</div>
              <div className="font-mono font-bold">{(fold.oos_metrics.cagr * 100).toFixed(1)}%</div>
            </div>
            <div>
              <div className="text-[10px] text-terminal-muted">Max DD</div>
              <div className="font-mono font-bold text-loss">{(fold.oos_metrics.max_dd * 100).toFixed(1)}%</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}


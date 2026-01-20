import { useState, useEffect } from 'react';
import { MetricCard } from '../components/ui/MetricCard';
import {
  Shield,
  CheckCircle,
  XCircle,
  AlertTriangle,
  RefreshCw,
  FileText,
  Activity,
  TrendingUp,
  TrendingDown,
  Clock,
  Target,
  Award,
  ChevronDown,
  ChevronRight,
  BarChart3,
  Zap,
  Layers,
  GitBranch,
  PieChart,
} from 'lucide-react';

// =============================================================================
// INTERFACES
// =============================================================================

interface SanityData {
  run_id: string;
  timestamp: string;
  flags: {
    sharpe_extreme: boolean;
    volatility_zero: boolean;
    no_trades: boolean;
    null_metrics: boolean;
    lookahead_risk: boolean;
  };
  warnings: string[];
  passed: boolean;
  summary: string;
}

interface HumanReport {
  run_id: string;
  timestamp: string;
  summary: string;
  metrics: {
    generations: number;
    evaluations: number;
    duration_secs: number;
    cache_hit_rate_pct: number;
  };
  best_strategy: {
    sharpe: number;
    cagr_pct: number;
    max_drawdown_pct?: number;
    rank: number;
    note?: string;
  } | null;
  evolution_progress: {
    improvement_pct: number;
    best_sharpe: number;
    final_mean_sharpe: number;
  };
  stage_funnel: {
    stage_a_candidates: number;
    stage_b_validated: number;
    pass_rate_pct: number;
  };
  recommendation: string;
  sanity_check: boolean;
}

interface Attribution {
  run_id: string;
  timestamp: string;
  top_strategies: Array<{
    rank: number;
    sharpe: number;
    cagr_pct: number;
    genome_hash: string;
  }>;
  bottom_strategies: Array<{
    rank: number;
    sharpe: number;
    cagr_pct: number;
    genome_hash: string;
  }>;
}

interface AssetAttribution {
  run_id: string;
  timestamp: string;
  assets: Array<{
    symbol: string;
    trades: number;
    net_pnl: number;
    contribution_pct: number;
    win_rate_pct: number;
  }>;
  top_contributors: Array<unknown>;
  worst_detractors: Array<unknown>;
  diversification_score: number;
  total_assets: number;
}

interface CrosscheckResult {
  strategy_id: string;
  reported: { sharpe: number; cagr_pct: number; max_drawdown_pct: number; volatility_pct: number };
  recalculated: { sharpe: number; cagr_pct: number; max_drawdown_pct: number; volatility_pct: number };
  tolerance: { sharpe_diff: number; cagr_diff: number; within_tolerance: boolean };
  verdict: string;
}

interface AuditCrosscheck {
  run_id: string;
  timestamp: string;
  strategies_checked: number;
  all_passed: boolean;
  failures: string[];
  summary: string;
  results: CrosscheckResult[];
}

interface ValidationOverview {
  run_id: string;
  timestamp: string;
  wfa: {
    total_strategies: number;
    passed: number;
    avg_oos_sharpe: number;
    avg_is_sharpe: number;
    overfit_ratio: number;
  };
  pbo: {
    avg_pbo: number;
    below_threshold: number;
    threshold: number;
    total: number;
  };
  dsr: {
    avg_dsr: number;
    above_threshold: number;
    threshold: number;
    total: number;
  };
  stress: {
    tests_run: number;
    passed: number;
    pass_rate_pct: number;
  };
}

interface AuditMarco {
  id: number;
  name: string;
  status: string;
  checks: number;
  passed: number;
  warnings: number;
}

interface AuditMarcosData {
  run_id: string;
  timestamp: string;
  marcos: AuditMarco[];
  overall: string;
  total_checks: number;
  total_passed: number;
  total_warnings: number;
}

interface AuditRun {
  runId: string;
  path: string;
  hasAuditData: boolean;
  sanityPassed: boolean | null;
  recommendation: string | null;
  timestamp: string;
}

interface AuditDetail {
  runId: string;
  runDir: string;
  hasAuditData: boolean;
  sanity: SanityData | null;
  humanReport: HumanReport | null;
  attribution: Attribution | null;
  assetAttribution: AssetAttribution | null;
  crosscheck: AuditCrosscheck | null;
  validationOverview: ValidationOverview | null;
  auditMarcos: AuditMarcosData | null;
}

// =============================================================================
// COMPONENT
// =============================================================================

export function AuditReport() {
  const [runs, setRuns] = useState<AuditRun[]>([]);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [auditData, setAuditData] = useState<AuditDetail | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [expandedSection, setExpandedSection] = useState<string>('overview');

  // Fetch list of runs with audit data
  const fetchAuditRuns = async () => {
    try {
      setIsLoading(true);
      const response = await fetch('http://localhost:3001/api/audits');
      const data = await response.json();
      setRuns(data.runs || []);
      
      const withAudit = data.runs?.filter((r: AuditRun) => r.hasAuditData);
      if (withAudit?.length > 0 && !selectedRun) {
        setSelectedRun(withAudit[0].runId);
      }
    } catch {
      // Handle error silently
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch detailed audit data for selected run
  const fetchAuditDetail = async (runId: string) => {
    try {
      setIsLoading(true);
      const response = await fetch(`http://localhost:3001/api/audit/${runId}`);
      const data = await response.json();
      setAuditData(data);
    } catch {
      // Handle error silently
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchAuditRuns();
  }, []);

  useEffect(() => {
    if (selectedRun) {
      fetchAuditDetail(selectedRun);
    }
  }, [selectedRun]);

  // Loading
  if (isLoading && !auditData) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-4">
        <RefreshCw className="w-12 h-12 animate-spin text-terminal-muted" />
        <p className="text-terminal-muted">Loading audit reports...</p>
      </div>
    );
  }

  // No audit data available
  if (runs.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-6">
        <FileText className="w-16 h-16 text-terminal-muted" />
        <div className="text-center">
          <h2 className="text-xl font-semibold mb-2">No Audit Data Available</h2>
          <p className="text-terminal-muted">Run an SCG campaign to generate audit reports.</p>
        </div>
      </div>
    );
  }

  const sanity = auditData?.sanity;
  const report = auditData?.humanReport;
  const attribution = auditData?.attribution;
  const assetAttribution = auditData?.assetAttribution;
  const crosscheck = auditData?.crosscheck;
  const validationOverview = auditData?.validationOverview;
  const auditMarcos = auditData?.auditMarcos;

  const getRecommendationStyle = (rec: string) => {
    if (rec?.includes('APROVAR')) return 'bg-profit/20 border-profit/50 text-profit';
    if (rec?.includes('REVISAR')) return 'bg-yellow-500/20 border-yellow-500/50 text-yellow-400';
    return 'bg-loss/20 border-loss/50 text-loss';
  };

  const getRecommendationIcon = (rec: string) => {
    if (rec?.includes('APROVAR')) return <CheckCircle className="w-6 h-6" />;
    if (rec?.includes('REVISAR')) return <AlertTriangle className="w-6 h-6" />;
    return <XCircle className="w-6 h-6" />;
  };

  const getMarcoStatusColor = (status: string) => {
    if (status === 'PASS') return 'bg-profit text-white';
    if (status === 'WARN') return 'bg-yellow-500 text-black';
    return 'bg-loss text-white';
  };

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Shield className="w-7 h-7 text-primary" />
            Institutional Audit Report
          </h1>
          <p className="text-terminal-muted mt-1">
            Complete validation for external auditors
          </p>
        </div>
        
        {/* Run Selector */}
        <div className="relative">
          <select
            value={selectedRun || ''}
            onChange={(e) => setSelectedRun(e.target.value)}
            className="appearance-none bg-terminal-darker border border-terminal-border rounded-lg px-4 py-2 pr-10 font-mono text-sm min-w-[280px]"
          >
            {runs.map((run) => (
              <option key={run.runId} value={run.runId}>
                {run.runId} {run.sanityPassed ? '✓' : run.sanityPassed === false ? '✗' : ''}
              </option>
            ))}
          </select>
          <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 pointer-events-none text-terminal-muted" />
        </div>
      </div>

      {/* Audit Marcos Timeline */}
      {auditMarcos && (
        <div className="bg-terminal-darker border border-terminal-border rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Layers className="w-5 h-5 text-primary" />
            Audit Marcos
            <span className={`ml-auto px-3 py-1 rounded text-sm ${getMarcoStatusColor(auditMarcos.overall)}`}>
              {auditMarcos.overall}
            </span>
          </h3>
          
          <div className="flex items-center justify-between">
            {auditMarcos.marcos.map((marco, idx) => (
              <div key={marco.id} className="flex items-center">
                <div className="flex flex-col items-center">
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold ${getMarcoStatusColor(marco.status)}`}>
                    {marco.id}
                  </div>
                  <span className="text-xs text-terminal-muted mt-1 text-center max-w-[80px]">{marco.name}</span>
                  <span className="text-xs text-terminal-muted">{marco.passed}/{marco.checks}</span>
                </div>
                {idx < auditMarcos.marcos.length - 1 && (
                  <div className="w-12 h-0.5 bg-terminal-border mx-2" />
                )}
              </div>
            ))}
          </div>
          
          <div className="mt-4 pt-4 border-t border-terminal-border flex gap-4 text-sm text-terminal-muted">
            <span>Total Checks: {auditMarcos.total_checks}</span>
            <span className="text-profit">Passed: {auditMarcos.total_passed}</span>
            {auditMarcos.total_warnings > 0 && (
              <span className="text-yellow-400">Warnings: {auditMarcos.total_warnings}</span>
            )}
          </div>
        </div>
      )}

      {/* Recommendation Banner */}
      {report?.recommendation && (
        <div className={`flex items-center gap-4 p-4 rounded-lg border ${getRecommendationStyle(report.recommendation)}`}>
          {getRecommendationIcon(report.recommendation)}
          <div className="flex-1">
            <p className="font-semibold text-lg">{report.recommendation.split(' - ')[0]}</p>
            <p className="text-sm opacity-80">{report.recommendation.split(' - ')[1]}</p>
          </div>
          <div className="text-right text-sm opacity-70">
            <Clock className="w-4 h-4 inline mr-1" />
            {new Date(report.timestamp).toLocaleString()}
          </div>
        </div>
      )}

      {/* Summary Card */}
      {report && (
        <div className="bg-terminal-darker border border-terminal-border rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
            <FileText className="w-5 h-5 text-primary" />
            Executive Summary
          </h3>
          <p className="text-terminal-muted leading-relaxed">{report.summary}</p>
        </div>
      )}

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="Generations"
          value={report?.metrics?.generations || 0}
          icon={<Activity className="w-5 h-5 text-blue-400" />}
        />
        <MetricCard
          label="Evaluations"
          value={report?.metrics?.evaluations?.toLocaleString() || '0'}
          icon={<Zap className="w-5 h-5 text-purple-400" />}
        />
        <MetricCard
          label="Stage A"
          value={report?.stage_funnel?.stage_a_candidates || 0}
          icon={<Target className="w-5 h-5 text-yellow-400" />}
        />
        <MetricCard
          label="Stage B"
          value={report?.stage_funnel?.stage_b_validated || 0}
          icon={<Award className="w-5 h-5 text-green-400" />}
        />
      </div>

      {/* Validation Overview - WFA/PBO/DSR/Stress */}
      {validationOverview && (
        <div className="bg-terminal-darker border border-terminal-border rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <GitBranch className="w-5 h-5 text-primary" />
            Validation Overview (Anti-Overfitting)
          </h3>
          
          <div className="grid md:grid-cols-4 gap-4">
            {/* WFA */}
            <div className="bg-terminal-dark rounded-lg p-4">
              <h4 className="text-sm font-semibold text-primary mb-2">Walk-Forward Analysis</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-terminal-muted">Strategies</span>
                  <span>{validationOverview.wfa.passed}/{validationOverview.wfa.total_strategies}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-terminal-muted">Avg OOS Sharpe</span>
                  <span className="text-profit">{validationOverview.wfa.avg_oos_sharpe.toFixed(3)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-terminal-muted">Overfit Ratio</span>
                  <span className={validationOverview.wfa.overfit_ratio > 2 ? 'text-loss' : 'text-terminal-text'}>
                    {validationOverview.wfa.overfit_ratio.toFixed(2)}x
                  </span>
                </div>
              </div>
            </div>
            
            {/* PBO */}
            <div className="bg-terminal-dark rounded-lg p-4">
              <h4 className="text-sm font-semibold text-purple-400 mb-2">PBO (Backtest Overfitting)</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-terminal-muted">Average PBO</span>
                  <span className={validationOverview.pbo.avg_pbo > 0.40 ? 'text-loss' : 'text-profit'}>
                    {(validationOverview.pbo.avg_pbo * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-terminal-muted">Below Threshold</span>
                  <span>{validationOverview.pbo.below_threshold}/{validationOverview.pbo.total}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-terminal-muted">Threshold</span>
                  <span>{(validationOverview.pbo.threshold * 100).toFixed(0)}%</span>
                </div>
              </div>
            </div>
            
            {/* DSR */}
            <div className="bg-terminal-dark rounded-lg p-4">
              <h4 className="text-sm font-semibold text-blue-400 mb-2">DSR (Deflated Sharpe)</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-terminal-muted">Average DSR</span>
                  <span className={validationOverview.dsr.avg_dsr < 0.10 ? 'text-loss' : 'text-profit'}>
                    {validationOverview.dsr.avg_dsr.toFixed(3)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-terminal-muted">Above Threshold</span>
                  <span>{validationOverview.dsr.above_threshold}/{validationOverview.dsr.total}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-terminal-muted">Threshold</span>
                  <span>{validationOverview.dsr.threshold.toFixed(2)}</span>
                </div>
              </div>
            </div>
            
            {/* Stress */}
            <div className="bg-terminal-dark rounded-lg p-4">
              <h4 className="text-sm font-semibold text-yellow-400 mb-2">Stress Testing</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-terminal-muted">Tests Run</span>
                  <span>{validationOverview.stress.tests_run}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-terminal-muted">Passed</span>
                  <span className="text-profit">{validationOverview.stress.passed}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-terminal-muted">Pass Rate</span>
                  <span className={validationOverview.stress.pass_rate_pct < 80 ? 'text-loss' : 'text-profit'}>
                    {validationOverview.stress.pass_rate_pct.toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Cross-Check Section */}
      {crosscheck && (
        <div className="bg-terminal-darker border border-terminal-border rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <CheckCircle className="w-5 h-5 text-primary" />
            Cross-Check (Independent Metric Recalculation)
            <span className={`ml-auto px-2 py-1 rounded text-xs ${crosscheck.all_passed ? 'bg-profit/20 text-profit' : 'bg-loss/20 text-loss'}`}>
              {crosscheck.all_passed ? 'ALL PASSED' : `${crosscheck.failures.length} FAILURES`}
            </span>
          </h3>
          
          <p className="text-terminal-muted text-sm mb-4">{crosscheck.summary}</p>
          
          {crosscheck.results.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-terminal-border text-left">
                    <th className="pb-2 text-terminal-muted">Strategy</th>
                    <th className="pb-2 text-terminal-muted">Reported Sharpe</th>
                    <th className="pb-2 text-terminal-muted">Recalculated</th>
                    <th className="pb-2 text-terminal-muted">Diff</th>
                    <th className="pb-2 text-terminal-muted">Verdict</th>
                  </tr>
                </thead>
                <tbody>
                  {crosscheck.results.slice(0, 10).map((r) => (
                    <tr key={r.strategy_id} className="border-b border-terminal-border/50">
                      <td className="py-2 font-mono text-xs">{r.strategy_id}</td>
                      <td className="py-2">{r.reported.sharpe.toFixed(4)}</td>
                      <td className="py-2">{r.recalculated.sharpe.toFixed(4)}</td>
                      <td className={`py-2 ${r.tolerance.within_tolerance ? 'text-profit' : 'text-loss'}`}>
                        {r.tolerance.sharpe_diff.toFixed(4)}
                      </td>
                      <td className="py-2">
                        <span className={`px-2 py-0.5 rounded text-xs ${r.verdict === 'PASS' ? 'bg-profit/20 text-profit' : 'bg-loss/20 text-loss'}`}>
                          {r.verdict}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Two Column Layout - Sanity + Evolution */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Sanity Checks */}
        <div className="bg-terminal-darker border border-terminal-border rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Shield className="w-5 h-5 text-primary" />
            Sanity Checks
            {sanity?.passed ? (
              <span className="ml-auto px-2 py-1 rounded bg-profit/20 text-profit text-xs">PASSED</span>
            ) : (
              <span className="ml-auto px-2 py-1 rounded bg-loss/20 text-loss text-xs">WARNINGS</span>
            )}
          </h3>
          
          <div className="space-y-3">
            {sanity?.flags && Object.entries(sanity.flags).map(([key, value]) => (
              <div key={key} className="flex items-center justify-between py-2 border-b border-terminal-border/50 last:border-0">
                <span className="text-sm font-mono text-terminal-muted">
                  {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </span>
                {value ? (
                  <span className="flex items-center gap-1 text-loss">
                    <XCircle className="w-4 h-4" /> ALERT
                  </span>
                ) : (
                  <span className="flex items-center gap-1 text-profit">
                    <CheckCircle className="w-4 h-4" /> OK
                  </span>
                )}
              </div>
            ))}
          </div>

          {sanity?.warnings && sanity.warnings.length > 0 && (
            <div className="mt-4 space-y-2">
              <h4 className="text-sm font-semibold text-yellow-400 flex items-center gap-1">
                <AlertTriangle className="w-4 h-4" /> Warnings
              </h4>
              {sanity.warnings.map((w, i) => (
                <p key={i} className="text-sm text-terminal-muted pl-5">• {w}</p>
              ))}
            </div>
          )}
        </div>

        {/* Evolution Progress */}
        <div className="bg-terminal-darker border border-terminal-border rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-primary" />
            Evolution Progress
          </h3>
          
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-terminal-muted">Improvement</span>
              <span className={`font-mono ${(report?.evolution_progress?.improvement_pct || 0) > 0 ? 'text-profit' : 'text-loss'}`}>
                {(report?.evolution_progress?.improvement_pct || 0) > 0 ? '+' : ''}
                {report?.evolution_progress?.improvement_pct?.toFixed(1)}%
              </span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-terminal-muted">Best Sharpe</span>
              <span className="font-mono text-primary">
                {report?.evolution_progress?.best_sharpe?.toFixed(3)}
              </span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-terminal-muted">Final Mean Sharpe</span>
              <span className="font-mono">
                {report?.evolution_progress?.final_mean_sharpe?.toFixed(3)}
              </span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-terminal-muted">Pass Rate</span>
              <span className="font-mono">
                {report?.stage_funnel?.pass_rate_pct?.toFixed(1)}%
              </span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-terminal-muted">Duration</span>
              <span className="font-mono">
                {report?.metrics?.duration_secs?.toFixed(1)}s
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Best Strategy */}
      {report?.best_strategy && (
        <div className="bg-terminal-darker border border-terminal-border rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Award className="w-5 h-5 text-yellow-400" />
            Best Strategy
            {report.best_strategy.note && (
              <span className="ml-2 px-2 py-0.5 rounded bg-yellow-500/20 text-yellow-400 text-xs">
                {report.best_strategy.note}
              </span>
            )}
          </h3>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div>
              <p className="text-xs text-terminal-muted uppercase tracking-wide">Sharpe Ratio</p>
              <p className="text-2xl font-mono text-profit mt-1">{report.best_strategy.sharpe?.toFixed(3)}</p>
            </div>
            <div>
              <p className="text-xs text-terminal-muted uppercase tracking-wide">CAGR</p>
              <p className="text-2xl font-mono text-primary mt-1">{report.best_strategy.cagr_pct?.toFixed(1)}%</p>
            </div>
            {report.best_strategy.max_drawdown_pct !== undefined && (
              <div>
                <p className="text-xs text-terminal-muted uppercase tracking-wide">Max Drawdown</p>
                <p className="text-2xl font-mono text-loss mt-1">{report.best_strategy.max_drawdown_pct?.toFixed(1)}%</p>
              </div>
            )}
            <div>
              <p className="text-xs text-terminal-muted uppercase tracking-wide">Rank</p>
              <p className="text-2xl font-mono text-yellow-400 mt-1">#{report.best_strategy.rank}</p>
            </div>
          </div>
        </div>
      )}

      {/* Asset Attribution */}
      {assetAttribution && assetAttribution.assets && assetAttribution.assets.length > 0 && (
        <div className="bg-terminal-darker border border-terminal-border rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <PieChart className="w-5 h-5 text-primary" />
            Asset Attribution
            <span className="ml-auto text-sm text-terminal-muted">
              Diversification: {(assetAttribution.diversification_score * 100).toFixed(1)}%
            </span>
          </h3>
          
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-terminal-border text-left">
                  <th className="pb-2 text-terminal-muted">Symbol</th>
                  <th className="pb-2 text-terminal-muted text-right">Trades</th>
                  <th className="pb-2 text-terminal-muted text-right">Net PnL</th>
                  <th className="pb-2 text-terminal-muted text-right">Contribution</th>
                  <th className="pb-2 text-terminal-muted text-right">Win Rate</th>
                </tr>
              </thead>
              <tbody>
                {assetAttribution.assets.slice(0, 10).map((asset) => (
                  <tr key={asset.symbol} className="border-b border-terminal-border/50">
                    <td className="py-2 font-mono">{asset.symbol}</td>
                    <td className="py-2 text-right">{asset.trades}</td>
                    <td className={`py-2 text-right font-mono ${asset.net_pnl >= 0 ? 'text-profit' : 'text-loss'}`}>
                      {asset.net_pnl >= 0 ? '+' : ''}{asset.net_pnl.toFixed(2)}
                    </td>
                    <td className="py-2 text-right">
                      <div className="flex items-center justify-end gap-2">
                        <div className="w-16 h-2 bg-terminal-dark rounded overflow-hidden">
                          <div 
                            className={`h-full ${asset.contribution_pct >= 0 ? 'bg-profit' : 'bg-loss'}`}
                            style={{ width: `${Math.min(Math.abs(asset.contribution_pct), 100)}%` }}
                          />
                        </div>
                        <span className="w-12 text-right">{asset.contribution_pct.toFixed(1)}%</span>
                      </div>
                    </td>
                    <td className="py-2 text-right">{asset.win_rate_pct.toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Strategy Attribution */}
      {attribution && (
        <div className="bg-terminal-darker border border-terminal-border rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-primary" />
            Strategy Attribution
          </h3>
          
          <div className="grid md:grid-cols-2 gap-6">
            {/* Top Performers */}
            <div>
              <h4 className="text-sm font-semibold text-profit mb-3 flex items-center gap-1">
                <TrendingUp className="w-4 h-4" /> Top Performers
              </h4>
              <div className="space-y-2">
                {attribution.top_strategies?.map((s) => (
                  <div key={s.genome_hash} className="flex items-center justify-between py-2 px-3 bg-terminal-dark rounded">
                    <span className="font-mono text-xs text-terminal-muted">
                      #{s.rank} {s.genome_hash?.slice(0, 12)}...
                    </span>
                    <span className="font-mono text-profit">
                      {s.sharpe?.toFixed(3)} | {s.cagr_pct?.toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Bottom Performers */}
            <div>
              <h4 className="text-sm font-semibold text-loss mb-3 flex items-center gap-1">
                <TrendingDown className="w-4 h-4" /> Bottom Performers
              </h4>
              <div className="space-y-2">
                {attribution.bottom_strategies?.map((s) => (
                  <div key={s.genome_hash} className="flex items-center justify-between py-2 px-3 bg-terminal-dark rounded">
                    <span className="font-mono text-xs text-terminal-muted">
                      #{s.rank} {s.genome_hash?.slice(0, 12)}...
                    </span>
                    <span className="font-mono text-loss">
                      {s.sharpe?.toFixed(3)} | {s.cagr_pct?.toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Debug: Raw Data Toggle */}
      <details className="bg-terminal-darker border border-terminal-border rounded-lg">
        <summary className="p-4 cursor-pointer text-terminal-muted text-sm flex items-center gap-2">
          <ChevronRight className="w-4 h-4" />
          View Raw JSON Data
        </summary>
        <pre className="p-4 text-xs overflow-x-auto border-t border-terminal-border">
          {JSON.stringify(auditData, null, 2)}
        </pre>
      </details>
    </div>
  );
}

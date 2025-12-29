import { useState, useEffect } from 'react';
import { useDataStore } from '../stores/dataStore';
import { 
  ArrowLeft, 
  TrendingUp, 
  TrendingDown,
  Shield, 
  Target, 
  Award, 
  Clock, 
  GitBranch, 
  Layers,
  Settings,
  BarChart3,
  CheckCircle,
  XCircle,
  Copy,
  Check,
  Database,
  ChevronRight
} from 'lucide-react';

type TabId = 'overview' | 'validation' | 'parameters' | 'provenance';

export function StrategyView() {
  const [activeTab, setActiveTab] = useState<TabId>('overview');
  const [copiedField, setCopiedField] = useState<string | null>(null);
  
  const { selectedCandidate, clearSelectedCandidate } = useDataStore();

  const copyToClipboard = async (text: string, field: string) => {
    await navigator.clipboard.writeText(text);
    setCopiedField(field);
    setTimeout(() => setCopiedField(null), 2000);
  };

  if (!selectedCandidate) {
    return (
      <div className="flex flex-col items-center justify-center h-full">
        <Award className="w-16 h-16 text-terminal-muted mb-4" />
        <h2 className="text-xl font-semibold mb-2">No Strategy Selected</h2>
        <p className="text-terminal-muted">Select a strategy from Candidates to view details.</p>
      </div>
    );
  }

  const c = selectedCandidate;
  const dataSource = (c as any).data_source === 'neon' ? 'neon' : 'local';

  // Compute scores
  const sharpeScore = Math.min(100, (c.oos_sharpe_net / 2) * 100);
  const pboScore = Math.max(0, 100 - (c.pbo * 500));
  const dsrScore = (c.dsr || 0) * 100;
  const stressScore = c.stress_total > 0 ? (c.stress_passed / c.stress_total) * 100 : 0;

  const tabs = [
    { id: 'overview' as TabId, label: 'Overview', icon: BarChart3 },
    { id: 'validation' as TabId, label: 'Validation', icon: Shield },
    { id: 'parameters' as TabId, label: 'Parameters', icon: Settings },
    { id: 'provenance' as TabId, label: 'Provenance', icon: GitBranch },
  ];

  return (
    <div className="space-y-6">
      {/* Back Button & Header */}
      <div className="flex items-center gap-4">
        <button
          onClick={clearSelectedCandidate}
          className="p-2 rounded-lg bg-terminal-surface border border-terminal-border hover:border-profit transition-colors"
        >
          <ArrowLeft className="w-5 h-5" />
        </button>
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold">{c.display_name}</h1>
            {c.gates_passed && (
              <span className="px-2 py-0.5 rounded text-xs font-medium bg-profit/20 text-profit flex items-center gap-1">
                <CheckCircle className="w-3 h-3" /> Validated
              </span>
            )}
            {dataSource === 'neon' && (
              <span className="px-2 py-0.5 rounded text-xs font-medium bg-accent-cyan/10 text-accent-cyan flex items-center gap-1">
                <Database className="w-3 h-3" /> Neon
              </span>
            )}
          </div>
          <p className="text-terminal-muted font-mono text-sm">{c.candidate_id}</p>
        </div>
      </div>

      {/* Hero Stats */}
      <div className="grid grid-cols-5 gap-4">
        <HeroStat 
          icon={Award} 
          label="Rank" 
          value={`#${c.rank || 1}`} 
          color="accent-cyan" 
        />
        <HeroStat 
          icon={TrendingUp} 
          label="OOS Sharpe" 
          value={c.oos_sharpe_net.toFixed(2)} 
          color={c.oos_sharpe_net >= 1 ? 'profit' : 'default'} 
        />
        <HeroStat 
          icon={Target} 
          label="CAGR NET" 
          value={c.oos_cagr_net ? `${(c.oos_cagr_net * 100).toFixed(1)}%` : 'N/A'} 
          color="accent-cyan" 
        />
        <HeroStat 
          icon={TrendingDown} 
          label="Max Drawdown" 
          value={c.max_drawdown_net ? `-${(Math.abs(c.max_drawdown_net) * 100).toFixed(1)}%` : 'N/A'} 
          color="loss" 
        />
        <HeroStat 
          icon={Shield} 
          label="PBO" 
          value={`${(c.pbo * 100).toFixed(1)}%`} 
          color={c.pbo <= 0.15 ? 'profit' : 'loss'} 
        />
      </div>

      {/* Tabs */}
      <div className="flex gap-1 bg-terminal-surface rounded-xl p-1 border border-terminal-border">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-medium transition-all ${
              activeTab === tab.id
                ? 'bg-terminal-bg text-white shadow-sm'
                : 'text-terminal-muted hover:text-white'
            }`}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="bg-terminal-surface rounded-xl border border-terminal-border p-6">
        {activeTab === 'overview' && (
          <div className="space-y-6">
            <h3 className="text-lg font-semibold">Why This Strategy?</h3>
            <div className="grid grid-cols-2 gap-4">
              <ScoreCard 
                label="Risk-Adjusted Return" 
                value={sharpeScore} 
                detail={`${c.oos_sharpe_net.toFixed(2)} Sharpe - return per unit of risk`}
              />
              <ScoreCard 
                label="Overfitting Risk" 
                value={pboScore} 
                detail={`${(c.pbo * 100).toFixed(1)}% PBO - lower is better`}
              />
              <ScoreCard 
                label="Statistical Confidence" 
                value={dsrScore} 
                detail={`${(c.dsr || 0).toFixed(2)} DSR - accounts for multiple testing`}
              />
              <ScoreCard 
                label="Stress Resilience" 
                value={stressScore} 
                detail={`${c.stress_passed}/${c.stress_total} scenarios passed`}
              />
            </div>
            
            <h3 className="text-lg font-semibold pt-4">All Metrics</h3>
            <div className="grid grid-cols-4 gap-3">
              <MetricTile label="OOS Sharpe NET" value={c.oos_sharpe_net.toFixed(3)} />
              <MetricTile label="OOS Sharpe GROSS" value={(c.oos_sharpe_gross || c.oos_sharpe_net).toFixed(3)} />
              <MetricTile label="OOS CAGR NET" value={c.oos_cagr_net ? `${(c.oos_cagr_net * 100).toFixed(2)}%` : 'N/A'} />
              <MetricTile label="Max DD NET" value={c.max_drawdown_net ? `${(c.max_drawdown_net * 100).toFixed(2)}%` : 'N/A'} />
              <MetricTile label="PBO" value={`${(c.pbo * 100).toFixed(2)}%`} />
              <MetricTile label="DSR" value={(c.dsr || 0).toFixed(3)} />
              <MetricTile label="Turnover" value={(c.turnover_annual || 0).toFixed(2)} />
              <MetricTile label="Capacity" value={c.capacity_usd ? `$${(c.capacity_usd / 1e6).toFixed(1)}M` : 'N/A'} />
            </div>
          </div>
        )}

        {activeTab === 'validation' && (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Validation Gates</h3>
            <div className="grid grid-cols-2 gap-3">
              <ValidationGate label="Walk-Forward Analysis" passed={c.gates_passed} detail="Multi-period OOS testing" />
              <ValidationGate label="CPCV Cross-Validation" passed={c.gates_passed} detail="Combinatorial purged validation" />
              <ValidationGate label="PBO < 15%" passed={c.pbo <= 0.15} detail={`Actual: ${(c.pbo * 100).toFixed(1)}%`} />
              <ValidationGate label="DSR > 0.5" passed={(c.dsr || 0) >= 0.5} detail={`Actual: ${(c.dsr || 0).toFixed(2)}`} />
              <ValidationGate label="OOS Sharpe > 0.5" passed={c.oos_sharpe_net >= 0.5} detail={`Actual: ${c.oos_sharpe_net.toFixed(2)}`} />
              <ValidationGate label={`Stress (${c.stress_passed}/${c.stress_total})`} passed={c.stress_passed >= c.stress_total * 0.8} detail="≥80% scenarios" />
            </div>
            
            <h3 className="text-lg font-semibold pt-4">Summary</h3>
            <div className={`p-4 rounded-xl border ${c.gates_passed ? 'bg-profit/10 border-profit/30' : 'bg-loss/10 border-loss/30'}`}>
              <div className="flex items-center gap-3">
                {c.gates_passed ? (
                  <CheckCircle className="w-8 h-8 text-profit" />
                ) : (
                  <XCircle className="w-8 h-8 text-loss" />
                )}
                <div>
                  <div className={`text-lg font-bold ${c.gates_passed ? 'text-profit' : 'text-loss'}`}>
                    {c.gates_passed ? 'All Gates Passed' : 'Some Gates Failed'}
                  </div>
                  <div className="text-sm text-terminal-muted">
                    {c.gates_passed 
                      ? 'This strategy has been validated through rigorous testing.'
                      : 'Review failed gates before deploying this strategy.'}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'parameters' && (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Strategy Configuration</h3>
            {c.strategy?.pipeline && c.strategy.pipeline.length > 0 ? (
              <div className="space-y-3">
                {c.strategy.pipeline.map((block, i) => (
                  <div key={i} className="p-4 bg-terminal-bg rounded-lg border border-terminal-border">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="px-2 py-0.5 rounded text-xs font-medium bg-accent-cyan/20 text-accent-cyan uppercase">
                        {block.block_type}
                      </span>
                      <span className="font-mono text-profit">{block.block_id}</span>
                    </div>
                    {Object.keys(block.params).length > 0 && (
                      <div className="grid grid-cols-3 gap-2 text-sm">
                        {Object.entries(block.params).map(([key, value]) => (
                          <div key={key} className="flex justify-between py-1 px-2 bg-terminal-surface rounded">
                            <span className="text-terminal-muted">{key}</span>
                            <span className="font-mono">{formatParam(value)}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="p-8 text-center text-terminal-muted">
                <Layers className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p>Strategy parameters not available</p>
                <p className="text-sm mt-1">Data from Neon doesn't include full strategy config</p>
              </div>
            )}

            {c.execution && (
              <>
                <h3 className="text-lg font-semibold pt-4">Execution Config</h3>
                <div className="grid grid-cols-3 gap-3">
                  <MetricTile label="Delay Bars" value={String(c.execution.delay_bars)} />
                  <MetricTile label="Fees Tier" value={c.execution.fees?.tier ?? 'N/A'} />
                  <MetricTile label="Slippage" value={c.execution.slippage?.bps ? `${c.execution.slippage.bps} bps` : 'N/A'} />
                </div>
              </>
            )}
          </div>
        )}

        {activeTab === 'provenance' && (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Lineage & Audit Trail</h3>
            
            <div className="bg-terminal-bg rounded-lg border border-terminal-border divide-y divide-terminal-border">
              {c.provenance.campaign_name && (
                <ProvenanceRow label="Campaign" value={c.provenance.campaign_name} />
              )}
              {c.provenance.campaign_tag && (
                <ProvenanceRow label="Tag" value={c.provenance.campaign_tag} />
              )}
              {c.provenance.campaign_owner && (
                <ProvenanceRow label="Owner" value={c.provenance.campaign_owner} />
              )}
              <ProvenanceRow 
                label="Run ID" 
                value={c.provenance.run_id} 
                copyable 
                onCopy={copyToClipboard} 
                copiedField={copiedField} 
              />
              <ProvenanceRow 
                label="Candidate ID" 
                value={c.candidate_id} 
                copyable 
                onCopy={copyToClipboard} 
                copiedField={copiedField} 
              />
              <ProvenanceRow label="Seed" value={String(c.provenance.seed)} />
              {c.provenance.git_branch && (
                <ProvenanceRow label="Git Branch" value={c.provenance.git_branch} />
              )}
              {c.provenance.git_sha && (
                <ProvenanceRow 
                  label="Git SHA" 
                  value={c.provenance.git_sha} 
                  copyable 
                  onCopy={copyToClipboard} 
                  copiedField={copiedField} 
                />
              )}
              {c.provenance.duration_secs && (
                <ProvenanceRow label="Run Duration" value={formatDuration(c.provenance.duration_secs)} />
              )}
              {c.provenance.generations_completed && (
                <ProvenanceRow label="Generations" value={String(c.provenance.generations_completed)} />
              )}
              {c.provenance.total_evaluations && (
                <ProvenanceRow label="Total Evaluations" value={String(c.provenance.total_evaluations)} />
              )}
              {c.provenance.created_at && (
                <ProvenanceRow label="Created At" value={new Date(c.provenance.created_at).toLocaleString()} />
              )}
            </div>

            {/* Timeline */}
            <h3 className="text-lg font-semibold pt-4">Timeline</h3>
            <div className="relative pl-6 space-y-4">
              <div className="absolute left-2 top-2 bottom-2 w-0.5 bg-terminal-border" />
              <TimelineItem icon={GitBranch} title="Created" time={c.provenance.created_at} />
              <TimelineItem icon={Settings} title="Evaluated" time={c.provenance.created_at} />
              {c.gates_passed && (
                <TimelineItem icon={CheckCircle} title="Validated" time={c.provenance.created_at} color="profit" />
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// Components

function HeroStat({ icon: Icon, label, value, color = 'default' }: { 
  icon: React.ElementType; 
  label: string; 
  value: string; 
  color?: string;
}) {
  const colorClass = {
    profit: 'text-profit',
    loss: 'text-loss',
    'accent-cyan': 'text-accent-cyan',
    default: 'text-white'
  }[color] || 'text-white';

  return (
    <div className="bg-terminal-surface rounded-xl border border-terminal-border p-4 text-center">
      <Icon className={`w-5 h-5 mx-auto mb-2 ${colorClass}`} />
      <div className={`text-xl font-bold ${colorClass}`}>{value}</div>
      <div className="text-xs text-terminal-muted">{label}</div>
    </div>
  );
}

function ScoreCard({ label, value, detail }: { label: string; value: number; detail: string }) {
  const clamped = Math.max(0, Math.min(100, value));
  const barColor = clamped >= 70 ? 'bg-profit' : clamped >= 40 ? 'bg-accent-yellow' : 'bg-loss';
  
  return (
    <div className="p-4 bg-terminal-bg rounded-lg border border-terminal-border">
      <div className="flex items-center justify-between mb-2">
        <span className="font-medium">{label}</span>
        <span className="font-mono text-sm">{clamped.toFixed(0)}%</span>
      </div>
      <div className="h-2 bg-terminal-surface rounded-full overflow-hidden mb-2">
        <div className={`h-full ${barColor} transition-all`} style={{ width: `${clamped}%` }} />
      </div>
      <p className="text-xs text-terminal-muted">{detail}</p>
    </div>
  );
}

function ValidationGate({ label, passed, detail }: { label: string; passed: boolean; detail: string }) {
  return (
    <div className={`p-3 rounded-lg border ${passed ? 'border-profit/30 bg-profit/5' : 'border-loss/30 bg-loss/5'}`}>
      <div className="flex items-center gap-2">
        {passed ? <CheckCircle className="w-5 h-5 text-profit" /> : <XCircle className="w-5 h-5 text-loss" />}
        <span className={`font-medium ${passed ? 'text-profit' : 'text-loss'}`}>{label}</span>
      </div>
      <p className="text-xs text-terminal-muted mt-1 ml-7">{detail}</p>
    </div>
  );
}

function MetricTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="p-3 bg-terminal-bg rounded-lg text-center">
      <div className="text-xs text-terminal-muted mb-1">{label}</div>
      <div className="font-mono">{value}</div>
    </div>
  );
}

function ProvenanceRow({ label, value, copyable, onCopy, copiedField }: {
  label: string;
  value: string;
  copyable?: boolean;
  onCopy?: (v: string, f: string) => void;
  copiedField?: string | null;
}) {
  return (
    <div className="flex items-center justify-between py-3 px-4">
      <span className="text-terminal-muted">{label}</span>
      <div className="flex items-center gap-2">
        <span className="font-mono text-sm truncate max-w-[200px]" title={value}>{value}</span>
        {copyable && onCopy && (
          <button onClick={() => onCopy(value, label)} className="p-1 hover:bg-terminal-surface rounded">
            {copiedField === label ? <Check className="w-3 h-3 text-profit" /> : <Copy className="w-3 h-3 text-terminal-muted" />}
          </button>
        )}
      </div>
    </div>
  );
}

function TimelineItem({ icon: Icon, title, time, color = 'default' }: {
  icon: React.ElementType;
  title: string;
  time?: string;
  color?: string;
}) {
  const iconColor = color === 'profit' ? 'text-profit bg-profit/20' : 'text-accent-cyan bg-accent-cyan/20';
  return (
    <div className="relative flex items-center gap-3">
      <div className={`w-6 h-6 rounded-full flex items-center justify-center ${iconColor}`}>
        <Icon className="w-3 h-3" />
      </div>
      <div>
        <div className="font-medium text-sm">{title}</div>
        {time && <div className="text-xs text-terminal-muted">{new Date(time).toLocaleString()}</div>}
      </div>
    </div>
  );
}

function formatParam(value: unknown): string {
  if (typeof value === 'number') return value % 1 === 0 ? String(value) : value.toFixed(2);
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
}

function formatDuration(secs: number): string {
  const mins = Math.floor(secs / 60);
  const s = secs % 60;
  return mins > 0 ? `${mins}m ${s}s` : `${s}s`;
}


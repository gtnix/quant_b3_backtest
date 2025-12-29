import { useState } from 'react';
import { 
  X, 
  Copy, 
  Check,
  TrendingUp,
  TrendingDown,
  Shield,
  Target,
  Zap,
  Award,
  Clock,
  GitBranch,
  ChevronDown,
  ChevronRight,
  CheckCircle,
  XCircle,
  AlertTriangle,
  BarChart3,
  Layers,
  Settings,
  Database
} from 'lucide-react';
import type { CandidateDetailFull, PipelineBlock } from '../stores/dataStore';

interface CandidateDetailProps {
  candidate: CandidateDetailFull;
  onClose: () => void;
}

export function CandidateDetail({ candidate, onClose }: CandidateDetailProps) {
  const [copiedField, setCopiedField] = useState<string | null>(null);
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set(['scorecard', 'validation', 'provenance'])
  );

  const copyToClipboard = async (text: string, field: string) => {
    await navigator.clipboard.writeText(text);
    setCopiedField(field);
    setTimeout(() => setCopiedField(null), 2000);
  };

  const toggleSection = (section: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(section)) {
      newExpanded.delete(section);
    } else {
      newExpanded.add(section);
    }
    setExpandedSections(newExpanded);
  };

  // Compute scores for scorecard
  const sharpeScore = Math.min(100, (candidate.oos_sharpe_net / 2) * 100);
  const pboScore = Math.max(0, 100 - (candidate.pbo * 500)); // Lower is better
  const dsrScore = (candidate.dsr || 0) * 100;
  const stressScore = candidate.stress_total > 0 
    ? (candidate.stress_passed / candidate.stress_total) * 100 
    : 0;

  // Determine quality tier
  const getQualityTier = () => {
    const avgScore = (sharpeScore + pboScore + dsrScore + stressScore) / 4;
    if (avgScore >= 80) return { label: 'Excellent', color: 'text-profit', bg: 'bg-profit/20' };
    if (avgScore >= 60) return { label: 'Good', color: 'text-accent-cyan', bg: 'bg-accent-cyan/20' };
    if (avgScore >= 40) return { label: 'Fair', color: 'text-accent-yellow', bg: 'bg-accent-yellow/20' };
    return { label: 'Research', color: 'text-terminal-muted', bg: 'bg-terminal-surface' };
  };
  const qualityTier = getQualityTier();

  // Group pipeline blocks by type
  const blocksByType = (candidate.strategy?.pipeline || []).reduce((acc, block) => {
    const type = block.block_type;
    if (!acc[type]) acc[type] = [];
    acc[type].push(block);
    return acc;
  }, {} as Record<string, PipelineBlock[]>);

  const hasBundle = !((candidate as any).bundle_missing === true);
  const dataSource = (candidate as any).data_source === 'neon' ? 'neon' : 'local';

  const SectionHeader = ({ title, section, icon: Icon }: { title: string; section: string; icon: React.ElementType }) => (
    <button
      onClick={() => toggleSection(section)}
      className="flex items-center gap-2 w-full py-2 text-left font-semibold text-base hover:text-profit transition-colors"
    >
      {expandedSections.has(section) ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
      <Icon className="w-4 h-4 text-terminal-muted" />
      {title}
    </button>
  );

  return (
    <div className="fixed inset-0 z-50 flex">
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={onClose} />
      
      <div className="absolute right-0 top-0 bottom-0 w-full max-w-2xl bg-gradient-to-b from-terminal-bg to-terminal-surface border-l border-terminal-border overflow-hidden flex flex-col">
        {/* Hero Header */}
        <div className="relative p-6 border-b border-terminal-border bg-gradient-to-r from-terminal-surface via-terminal-bg to-terminal-surface">
          <button onClick={onClose} className="absolute right-4 top-4 p-2 hover:bg-terminal-bg rounded-lg transition-colors">
            <X className="w-5 h-5" />
          </button>
          
          <div className="flex items-start gap-4">
            {/* Rank Badge */}
            <div className="flex-shrink-0 w-16 h-16 rounded-xl bg-gradient-to-br from-accent-cyan to-profit flex items-center justify-center">
              <span className="text-2xl font-bold text-black">#{candidate.rank || 1}</span>
            </div>
            
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <span className={`px-2 py-0.5 rounded text-xs font-medium ${qualityTier.bg} ${qualityTier.color}`}>
                  {qualityTier.label}
                </span>
                {candidate.gates_passed && (
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
              <h2 className="text-xl font-bold truncate">{candidate.display_name}</h2>
              <p className="text-sm text-terminal-muted font-mono truncate">{candidate.candidate_id}</p>
            </div>
          </div>
          
          {/* Quick Stats */}
          <div className="grid grid-cols-4 gap-3 mt-4">
            <QuickStat 
              icon={TrendingUp} 
              label="Sharpe" 
              value={candidate.oos_sharpe_net.toFixed(2)} 
              color={candidate.oos_sharpe_net >= 1 ? 'profit' : 'default'} 
            />
            <QuickStat 
              icon={Target} 
              label="CAGR" 
              value={candidate.oos_cagr_net ? `${(candidate.oos_cagr_net * 100).toFixed(1)}%` : 'N/A'} 
              color="accent-cyan" 
            />
            <QuickStat 
              icon={TrendingDown} 
              label="Max DD" 
              value={candidate.max_drawdown_net ? `-${(Math.abs(candidate.max_drawdown_net) * 100).toFixed(1)}%` : 'N/A'} 
              color="loss" 
            />
            <QuickStat 
              icon={Shield} 
              label="PBO" 
              value={`${(candidate.pbo * 100).toFixed(1)}%`} 
              color={candidate.pbo <= 0.15 ? 'profit' : 'loss'} 
            />
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-4 space-y-4">
          {/* Scorecard - Why This Strategy is Good */}
          <div className="bg-terminal-surface rounded-xl border border-terminal-border p-4">
            <SectionHeader title="Why This Strategy?" section="scorecard" icon={Award} />
            {expandedSections.has('scorecard') && (
              <div className="mt-3 space-y-3">
                <ScoreBar 
                  label="Risk-Adjusted Return (Sharpe)" 
                  value={sharpeScore} 
                  detail={`${candidate.oos_sharpe_net.toFixed(2)} Sharpe ratio - measures return per unit of risk`}
                />
                <ScoreBar 
                  label="Overfitting Risk (PBO)" 
                  value={pboScore} 
                  detail={`${(candidate.pbo * 100).toFixed(1)}% probability of backtest overfitting - lower is better`}
                />
                <ScoreBar 
                  label="Statistical Confidence (DSR)" 
                  value={dsrScore} 
                  detail={`${(candidate.dsr || 0).toFixed(2)} deflated Sharpe ratio - accounts for multiple testing`}
                />
                <ScoreBar 
                  label="Stress Resilience" 
                  value={stressScore} 
                  detail={`${candidate.stress_passed}/${candidate.stress_total} stress scenarios passed`}
                />
              </div>
            )}
          </div>

          {/* Validation Gates */}
          <div className="bg-terminal-surface rounded-xl border border-terminal-border p-4">
            <SectionHeader title="Validation Gates" section="validation" icon={Shield} />
            {expandedSections.has('validation') && (
              <div className="mt-3 grid grid-cols-2 gap-2">
                <ValidationGate 
                  label="Walk-Forward Analysis" 
                  passed={candidate.gates_passed} 
                  detail="Multi-period OOS testing"
                />
                <ValidationGate 
                  label="CPCV Cross-Validation" 
                  passed={candidate.gates_passed} 
                  detail="Combinatorial purged validation"
                />
                <ValidationGate 
                  label="PBO < 15%" 
                  passed={candidate.pbo <= 0.15} 
                  detail={`Actual: ${(candidate.pbo * 100).toFixed(1)}%`}
                />
                <ValidationGate 
                  label={`Stress Tests (${candidate.stress_passed}/${candidate.stress_total})`} 
                  passed={candidate.stress_passed >= (candidate.stress_total * 0.8)} 
                  detail="Market crash scenarios"
                />
                <ValidationGate 
                  label="DSR > 0.5" 
                  passed={(candidate.dsr || 0) >= 0.5} 
                  detail={`Actual: ${(candidate.dsr || 0).toFixed(2)}`}
                />
                <ValidationGate 
                  label="OOS Sharpe > 0.5" 
                  passed={candidate.oos_sharpe_net >= 0.5} 
                  detail={`Actual: ${candidate.oos_sharpe_net.toFixed(2)}`}
                />
              </div>
            )}
          </div>

          {/* Provenance */}
          <div className="bg-terminal-surface rounded-xl border border-terminal-border p-4">
            <SectionHeader title="Provenance & Lineage" section="provenance" icon={GitBranch} />
            {expandedSections.has('provenance') && (
              <div className="mt-3 space-y-2">
                {candidate.provenance.campaign_name && (
                  <ProvenanceRow label="Campaign" value={candidate.provenance.campaign_name} />
                )}
                {candidate.provenance.campaign_tag && (
                  <ProvenanceRow label="Tag" value={candidate.provenance.campaign_tag} />
                )}
                {candidate.provenance.campaign_owner && (
                  <ProvenanceRow label="Owner" value={candidate.provenance.campaign_owner} />
                )}
                <ProvenanceRow 
                  label="Run ID" 
                  value={candidate.provenance.run_id} 
                  copyable 
                  onCopy={copyToClipboard} 
                  copiedField={copiedField} 
                />
                <ProvenanceRow label="Seed" value={String(candidate.provenance.seed)} />
                {candidate.provenance.git_sha && (
                  <ProvenanceRow 
                    label="Git SHA" 
                    value={candidate.provenance.git_sha} 
                    copyable 
                    onCopy={copyToClipboard} 
                    copiedField={copiedField} 
                  />
                )}
                {candidate.provenance.duration_secs && (
                  <ProvenanceRow 
                    label="Run Duration" 
                    value={`${Math.floor(candidate.provenance.duration_secs / 60)}m ${candidate.provenance.duration_secs % 60}s`} 
                  />
                )}
                {candidate.provenance.generations_completed && (
                  <ProvenanceRow label="Generations" value={String(candidate.provenance.generations_completed)} />
                )}
                {candidate.provenance.created_at && (
                  <ProvenanceRow label="Created" value={new Date(candidate.provenance.created_at).toLocaleString()} />
                )}
              </div>
            )}
          </div>

          {/* Strategy Pipeline - only if bundle exists */}
          {hasBundle && Object.keys(blocksByType).length > 0 && (
            <div className="bg-terminal-surface rounded-xl border border-terminal-border p-4">
              <SectionHeader title="Strategy Pipeline" section="strategy" icon={Layers} />
              {expandedSections.has('strategy') && (
                <div className="mt-3 space-y-2">
                  {Object.entries(blocksByType).map(([type, blocks]) => (
                    <div key={type} className="p-3 bg-terminal-bg rounded-lg border border-terminal-border/50">
                      <div className="text-xs font-medium text-accent-cyan uppercase mb-2">{type}</div>
                      {blocks.map((block, i) => (
                        <div key={i} className="text-sm">
                          <span className="font-mono text-profit">{block.block_id}</span>
                          {Object.keys(block.params).length > 0 && (
                            <div className="ml-4 text-xs text-terminal-muted">
                              {Object.entries(block.params).slice(0, 3).map(([k, v]) => (
                                <span key={k} className="mr-3">{k}: <span className="text-white">{formatParamValue(v)}</span></span>
                              ))}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Execution Config - only if bundle exists */}
          {hasBundle && candidate.execution && (
            <div className="bg-terminal-surface rounded-xl border border-terminal-border p-4">
              <SectionHeader title="Execution Config" section="execution" icon={Settings} />
              {expandedSections.has('execution') && (
                <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
                  <div className="flex justify-between py-1">
                    <span className="text-terminal-muted">Delay Bars</span>
                    <span className="font-mono">{candidate.execution.delay_bars}</span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-terminal-muted">Fees Tier</span>
                    <span className="font-mono">{candidate.execution.fees?.tier ?? 'N/A'}</span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-terminal-muted">Slippage</span>
                    <span className="font-mono">{candidate.execution.slippage?.bps ?? 'N/A'} bps</span>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* All Metrics */}
          <div className="bg-terminal-surface rounded-xl border border-terminal-border p-4">
            <SectionHeader title="All Metrics" section="metrics" icon={BarChart3} />
            {expandedSections.has('metrics') && (
              <div className="mt-3 grid grid-cols-3 gap-2">
                <MetricTile label="OOS Sharpe NET" value={candidate.oos_sharpe_net.toFixed(3)} />
                <MetricTile label="OOS Sharpe GROSS" value={(candidate.oos_sharpe_gross || candidate.oos_sharpe_net).toFixed(3)} />
                <MetricTile label="OOS CAGR NET" value={candidate.oos_cagr_net ? `${(candidate.oos_cagr_net * 100).toFixed(2)}%` : 'N/A'} />
                <MetricTile label="Max Drawdown NET" value={candidate.max_drawdown_net ? `${(candidate.max_drawdown_net * 100).toFixed(2)}%` : 'N/A'} />
                <MetricTile label="PBO" value={`${(candidate.pbo * 100).toFixed(2)}%`} />
                <MetricTile label="DSR" value={(candidate.dsr || 0).toFixed(3)} />
                <MetricTile label="Turnover Annual" value={(candidate.turnover_annual || 0).toFixed(2)} />
                <MetricTile label="Capacity USD" value={candidate.capacity_usd ? `$${(candidate.capacity_usd / 1e6).toFixed(1)}M` : 'N/A'} />
                <MetricTile label="Stress Passed" value={`${candidate.stress_passed}/${candidate.stress_total}`} />
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-terminal-border bg-terminal-surface">
          <div className="flex gap-2">
            <button
              onClick={() => copyToClipboard(candidate.candidate_id, 'id-footer')}
              className="flex-1 py-2 bg-terminal-bg border border-terminal-border rounded-lg hover:border-accent-cyan transition-colors flex items-center justify-center gap-2"
            >
              {copiedField === 'id-footer' ? <Check className="w-4 h-4 text-profit" /> : <Copy className="w-4 h-4" />}
              Copy ID
            </button>
            <button
              onClick={onClose}
              className="flex-1 py-2 bg-profit/10 text-profit border border-profit/30 rounded-lg hover:bg-profit/20 transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Components

function QuickStat({ icon: Icon, label, value, color }: { icon: React.ElementType; label: string; value: string; color?: string }) {
  const colorClass = {
    profit: 'text-profit',
    loss: 'text-loss',
    'accent-cyan': 'text-accent-cyan',
    default: 'text-white'
  }[color || 'default'];

  return (
    <div className="text-center">
      <Icon className={`w-4 h-4 mx-auto mb-1 ${colorClass}`} />
      <div className={`font-mono font-bold ${colorClass}`}>{value}</div>
      <div className="text-xs text-terminal-muted">{label}</div>
    </div>
  );
}

function ScoreBar({ label, value, detail }: { label: string; value: number; detail: string }) {
  const clampedValue = Math.max(0, Math.min(100, value));
  const barColor = clampedValue >= 70 ? 'bg-profit' : clampedValue >= 40 ? 'bg-accent-yellow' : 'bg-loss';
  
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm font-medium">{label}</span>
        <span className="text-sm font-mono">{clampedValue.toFixed(0)}%</span>
      </div>
      <div className="h-2 bg-terminal-bg rounded-full overflow-hidden">
        <div className={`h-full ${barColor} transition-all duration-500`} style={{ width: `${clampedValue}%` }} />
      </div>
      <p className="text-xs text-terminal-muted mt-1">{detail}</p>
    </div>
  );
}

function ValidationGate({ label, passed, detail }: { label: string; passed: boolean; detail: string }) {
  return (
    <div className={`p-2 rounded-lg border ${passed ? 'border-profit/30 bg-profit/5' : 'border-loss/30 bg-loss/5'}`}>
      <div className="flex items-center gap-2">
        {passed ? (
          <CheckCircle className="w-4 h-4 text-profit flex-shrink-0" />
        ) : (
          <XCircle className="w-4 h-4 text-loss flex-shrink-0" />
        )}
        <span className={`text-sm font-medium ${passed ? 'text-profit' : 'text-loss'}`}>{label}</span>
      </div>
      <p className="text-xs text-terminal-muted mt-1 ml-6">{detail}</p>
    </div>
  );
}

function MetricTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="p-2 bg-terminal-bg rounded-lg text-center">
      <div className="text-xs text-terminal-muted mb-1">{label}</div>
      <div className="font-mono text-sm">{value}</div>
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
    <div className="flex items-center justify-between py-1 text-sm">
      <span className="text-terminal-muted">{label}</span>
      <div className="flex items-center gap-1">
        <span className="font-mono truncate max-w-[180px]" title={value}>{value}</span>
        {copyable && onCopy && (
          <button onClick={() => onCopy(value, label)} className="p-1 hover:bg-terminal-bg rounded">
            {copiedField === label ? <Check className="w-3 h-3 text-profit" /> : <Copy className="w-3 h-3 text-terminal-muted" />}
          </button>
        )}
      </div>
    </div>
  );
}

function formatParamValue(value: unknown): string {
  if (typeof value === 'number') return value % 1 === 0 ? String(value) : value.toFixed(2);
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
}

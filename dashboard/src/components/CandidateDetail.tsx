import { useState } from 'react';
import { open } from '@tauri-apps/plugin-shell';
import { 
  X, 
  Copy, 
  Check,
  ExternalLink,
  FileCode,
  FolderOpen,
  Play,
  BarChart3,
  Shield,
  GitBranch,
  Clock,
  Layers,
  Settings,
  AlertTriangle,
  CheckCircle,
  XCircle,
  ChevronDown,
  ChevronRight,
  FileText
} from 'lucide-react';
import type { CandidateDetailFull, PipelineBlock } from '../stores/dataStore';
import { ExportModal } from './ExportModal';

interface CandidateDetailProps {
  candidate: CandidateDetailFull;
  onClose: () => void;
}

export function CandidateDetail({ candidate, onClose }: CandidateDetailProps) {
  const [copiedField, setCopiedField] = useState<string | null>(null);
  const [showExport, setShowExport] = useState(false);
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set(['strategy', 'metrics'])
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

  const openInExplorer = async (path: string) => {
    try {
      await open(path);
    } catch (e) {
      console.error('Failed to open path:', e);
    }
  };

  const runReplay = async () => {
    if (candidate.replay_script_path) {
      try {
        await open(candidate.replay_script_path);
      } catch (e) {
        console.error('Failed to run replay:', e);
      }
    }
  };

  // Group pipeline blocks by type (handle missing strategy)
  const blocksByType = (candidate.strategy?.pipeline || []).reduce((acc, block) => {
    const type = block.block_type;
    if (!acc[type]) acc[type] = [];
    acc[type].push(block);
    return acc;
  }, {} as Record<string, PipelineBlock[]>);
  
  // Check if bundle is missing
  const bundleMissing = (candidate as any).bundle_missing === true;

  const SectionHeader = ({ 
    title, 
    section, 
    icon: Icon 
  }: { 
    title: string; 
    section: string; 
    icon: React.ElementType;
  }) => (
    <button
      onClick={() => toggleSection(section)}
      className="flex items-center gap-2 w-full py-2 text-left font-semibold text-lg hover:text-profit transition-colors"
    >
      {expandedSections.has(section) ? (
        <ChevronDown className="w-5 h-5" />
      ) : (
        <ChevronRight className="w-5 h-5" />
      )}
      <Icon className="w-5 h-5 text-terminal-muted" />
      {title}
    </button>
  );

  const CopyButton = ({ value, field }: { value: string; field: string }) => (
    <button
      onClick={() => copyToClipboard(value, field)}
      className="p-1 hover:bg-terminal-surface rounded transition-colors"
      title="Copy to clipboard"
    >
      {copiedField === field ? (
        <Check className="w-4 h-4 text-profit" />
      ) : (
        <Copy className="w-4 h-4 text-terminal-muted" />
      )}
    </button>
  );

  return (
    <div className="fixed inset-0 z-50 flex">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/60"
        onClick={onClose}
      />
      
      {/* Drawer */}
      <div className="absolute right-0 top-0 bottom-0 w-full max-w-2xl bg-terminal-bg border-l border-terminal-border overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-terminal-border bg-terminal-surface">
          <div>
            <h2 className="text-xl font-bold">Candidate Detail</h2>
            <p className="text-sm text-terminal-muted font-mono">{candidate.candidate_id}</p>
          </div>
          <button 
            onClick={onClose}
            className="p-2 hover:bg-terminal-bg rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-4 space-y-6">
          {/* Bundle Missing Warning */}
          {bundleMissing && (
            <div className="p-4 bg-loss/10 rounded-lg border border-loss/30 flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-loss flex-shrink-0 mt-0.5" />
              <div>
                <div className="font-medium text-loss">Bundle Not Available</div>
                <div className="text-sm text-terminal-muted mt-1">
                  {(candidate as any).bundle_message || 'This candidate was not promoted. Strategy details are limited to CSV metrics.'}
                </div>
              </div>
            </div>
          )}
          
          {/* Display Name Banner */}
          <div className="p-4 bg-terminal-surface rounded-lg border border-terminal-border">
            <div className="text-sm text-terminal-muted mb-1">Strategy Signature</div>
            <div className="text-lg font-medium">{candidate.display_name}</div>
          </div>

          {/* Quick Actions - Only show if bundle exists */}
          {!bundleMissing && (
            <div className="flex gap-2 flex-wrap">
              <button
                onClick={() => openInExplorer(candidate.strategy_toml_path)}
                className="flex items-center gap-2 px-3 py-2 bg-terminal-surface border border-terminal-border rounded-lg hover:border-profit transition-colors text-sm"
              >
                <FileCode className="w-4 h-4" />
                Open strategy.toml
              </button>
              <button
                onClick={() => openInExplorer(candidate.bundle_path)}
                className="flex items-center gap-2 px-3 py-2 bg-terminal-surface border border-terminal-border rounded-lg hover:border-profit transition-colors text-sm"
              >
                <FolderOpen className="w-4 h-4" />
                Open Bundle
              </button>
              <button
                onClick={() => setShowExport(true)}
                className="flex items-center gap-2 px-3 py-2 bg-terminal-surface border border-terminal-border rounded-lg hover:border-accent-cyan transition-colors text-sm"
              >
                <FileText className="w-4 h-4" />
                Export Tearsheet
              </button>
              {candidate.replay_script_path && (
                <button
                  onClick={runReplay}
                  className="flex items-center gap-2 px-3 py-2 bg-profit/10 text-profit border border-profit/30 rounded-lg hover:bg-profit/20 transition-colors text-sm"
                >
                  <Play className="w-4 h-4" />
                  Run Replay
                </button>
              )}
            </div>
          )}
          
          {/* Export Modal */}
          <ExportModal
            isOpen={showExport}
            onClose={() => setShowExport(false)}
            candidateId={candidate.candidate_id}
            candidateName={candidate.display_name}
          />

          {/* Metrics Section */}
          <div>
            <SectionHeader title="Performance Metrics" section="metrics" icon={BarChart3} />
            {expandedSections.has('metrics') && (
              <div className="grid grid-cols-3 gap-3 mt-3">
                <MetricBox 
                  label="OOS Sharpe NET" 
                  value={candidate.oos_sharpe_net.toFixed(2)}
                  color={candidate.oos_sharpe_net >= 1 ? 'profit' : 'default'}
                />
                <MetricBox 
                  label="PBO" 
                  value={`${(candidate.pbo * 100).toFixed(1)}%`}
                  color={candidate.pbo <= 0.1 ? 'profit' : candidate.pbo <= 0.15 ? 'warning' : 'loss'}
                />
                <MetricBox 
                  label="DSR" 
                  value={candidate.dsr?.toFixed(2) ?? 'N/A'}
                  color={candidate.dsr && candidate.dsr >= 0.5 ? 'profit' : 'default'}
                />
                <MetricBox 
                  label="CAGR NET" 
                  value={candidate.oos_cagr_net ? `${(candidate.oos_cagr_net * 100).toFixed(1)}%` : 'N/A'}
                />
                <MetricBox 
                  label="Max DD NET" 
                  value={candidate.max_drawdown_net ? `-${(Math.abs(candidate.max_drawdown_net) * 100).toFixed(1)}%` : 'N/A'}
                  color="loss"
                />
                <MetricBox 
                  label="Turnover" 
                  value={candidate.turnover_annual?.toFixed(2) ?? 'N/A'}
                />
              </div>
            )}
          </div>

          {/* Gates & Stress */}
          <div>
            <SectionHeader title="Validation Gates" section="gates" icon={Shield} />
            {expandedSections.has('gates') && (
              <div className="mt-3 space-y-3">
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    {candidate.gates_passed ? (
                      <CheckCircle className="w-5 h-5 text-profit" />
                    ) : (
                      <XCircle className="w-5 h-5 text-loss" />
                    )}
                    <span className="font-medium">
                      Gates: {candidate.gates_passed ? 'PASSED' : 'FAILED'}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-terminal-muted">Stress:</span>
                    <span className={`font-mono ${
                      candidate.stress_passed === candidate.stress_total ? 'text-profit' : 'text-accent-yellow'
                    }`}>
                      {candidate.stress_passed}/{candidate.stress_total}
                    </span>
                  </div>
                </div>
                
                {/* Data Integrity */}
                {candidate.data_integrity && (
                  <div className="p-3 bg-terminal-surface rounded-lg border border-terminal-border">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium">Data Integrity</span>
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                        candidate.data_integrity.verdict === 'Pass' 
                          ? 'bg-profit/20 text-profit' 
                          : 'bg-loss/20 text-loss'
                      }`}>
                        {candidate.data_integrity.verdict}
                      </span>
                    </div>
                    <div className="text-sm text-terminal-muted">
                      Score: {(candidate.data_integrity.score * 100).toFixed(0)}% | 
                      Passed: {candidate.data_integrity.passed_count} | 
                      Warnings: {candidate.data_integrity.warning_count}
                    </div>
                    {candidate.data_integrity.warnings.length > 0 && (
                      <div className="mt-2 space-y-1">
                        {candidate.data_integrity.warnings.map((w, i) => (
                          <div key={i} className="flex items-start gap-2 text-xs text-accent-yellow">
                            <AlertTriangle className="w-3 h-3 mt-0.5 flex-shrink-0" />
                            <span>{w}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Strategy Blocks */}
          <div>
            <SectionHeader title="Strategy Pipeline" section="strategy" icon={Layers} />
            {expandedSections.has('strategy') && (
              <div className="mt-3 space-y-3">
                {bundleMissing ? (
                  <div className="p-3 bg-terminal-surface rounded-lg border border-terminal-border text-center text-terminal-muted">
                    <Layers className="w-8 h-8 mx-auto mb-2 opacity-50" />
                    <p>Strategy details not available</p>
                    <p className="text-xs mt-1">Promote this candidate to generate bundle with full strategy</p>
                  </div>
                ) : (
                  <>
                    {Object.entries(blocksByType).map(([type, blocks]) => (
                      <div key={type} className="p-3 bg-terminal-surface rounded-lg border border-terminal-border">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="px-2 py-0.5 rounded text-xs font-medium bg-accent-cyan/20 text-accent-cyan uppercase">
                            {type}
                          </span>
                        </div>
                        <div className="space-y-2">
                          {blocks.map((block, i) => (
                            <div key={i}>
                              <div className="font-mono text-sm text-profit">{block.block_id}</div>
                              {Object.keys(block.params).length > 0 && (
                                <div className="mt-1 grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                                  {Object.entries(block.params).map(([key, value]) => (
                                    <div key={key} className="flex justify-between">
                                      <span className="text-terminal-muted">{key}:</span>
                                      <span className="font-mono">{formatParamValue(value)}</span>
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                    
                    {/* Rebalance */}
                    {candidate.strategy?.rebalance && (
                      <div className="flex items-center gap-2 text-sm">
                        <Clock className="w-4 h-4 text-terminal-muted" />
                        <span>Rebalance:</span>
                        <span className="font-mono text-accent-cyan">
                          {candidate.strategy.rebalance.frequency}
                          {candidate.strategy.rebalance.day && ` (${candidate.strategy.rebalance.day})`}
                        </span>
                      </div>
                    )}

                    {/* Constraints */}
                    {candidate.strategy?.constraints && (
                      <div className="text-sm text-terminal-muted">
                        Max weight: {candidate.strategy.constraints.max_weight_per_asset ?? 'N/A'} | 
                        Max positions: {candidate.strategy.constraints.max_positions ?? 'N/A'}
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
          </div>

          {/* Execution Config */}
          {!bundleMissing && candidate.execution && (
            <div>
              <SectionHeader title="Execution & Costs" section="execution" icon={Settings} />
              {expandedSections.has('execution') && (
                <div className="mt-3 p-3 bg-terminal-surface rounded-lg border border-terminal-border">
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div>
                      <span className="text-terminal-muted">Delay Bars:</span>
                      <span className="ml-2 font-mono">{candidate.execution.delay_bars}</span>
                    </div>
                    <div>
                      <span className="text-terminal-muted">Fees Tier:</span>
                      <span className="ml-2 font-mono">{candidate.execution.fees?.tier ?? 'N/A'}</span>
                    </div>
                    <div>
                      <span className="text-terminal-muted">Slippage:</span>
                      <span className="ml-2 font-mono">
                        {candidate.execution.slippage?.slippage_type ?? 'N/A'}
                        {candidate.execution.slippage?.bps && ` (${candidate.execution.slippage.bps} bps)`}
                      </span>
                    </div>
                    {candidate.execution.fill_policy && (
                      <div>
                        <span className="text-terminal-muted">Max Participation:</span>
                        <span className="ml-2 font-mono">
                          {candidate.execution.fill_policy.max_participation 
                            ? `${(candidate.execution.fill_policy.max_participation * 100).toFixed(0)}%` 
                            : 'N/A'}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Provenance */}
          <div>
            <SectionHeader title="Provenance & Audit" section="provenance" icon={GitBranch} />
            {expandedSections.has('provenance') && (
              <div className="mt-3 space-y-2">
                <ProvenanceRow 
                  label="Candidate ID" 
                  value={candidate.provenance.candidate_id}
                  copyable
                  onCopy={(v, f) => copyToClipboard(v, f)}
                  copiedField={copiedField}
                />
                <ProvenanceRow 
                  label="Genome Hash" 
                  value={candidate.provenance.genome_hash}
                  copyable
                  onCopy={(v, f) => copyToClipboard(v, f)}
                  copiedField={copiedField}
                />
                <ProvenanceRow 
                  label="Run ID" 
                  value={candidate.provenance.run_id}
                  copyable
                  onCopy={(v, f) => copyToClipboard(v, f)}
                  copiedField={copiedField}
                />
                <ProvenanceRow 
                  label="Campaign ID" 
                  value={candidate.provenance.campaign_id}
                  copyable
                  onCopy={(v, f) => copyToClipboard(v, f)}
                  copiedField={copiedField}
                />
                <ProvenanceRow 
                  label="Seed" 
                  value={String(candidate.provenance.seed)}
                />
                {candidate.provenance.git_sha && (
                  <ProvenanceRow 
                    label="Git SHA" 
                    value={candidate.provenance.git_sha}
                    copyable
                    onCopy={(v, f) => copyToClipboard(v, f)}
                    copiedField={copiedField}
                  />
                )}
                {candidate.provenance.config_hash && (
                  <ProvenanceRow 
                    label="Config Hash" 
                    value={candidate.provenance.config_hash}
                    copyable
                    onCopy={(v, f) => copyToClipboard(v, f)}
                    copiedField={copiedField}
                  />
                )}
                <ProvenanceRow 
                  label="Created At" 
                  value={new Date(candidate.provenance.created_at).toLocaleString()}
                />
                {candidate.provenance.scg_version && (
                  <ProvenanceRow 
                    label="SCG Version" 
                    value={candidate.provenance.scg_version}
                  />
                )}
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-terminal-border bg-terminal-surface">
          <button
            onClick={onClose}
            className="w-full py-2 bg-terminal-bg border border-terminal-border rounded-lg hover:border-profit transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

// Helper Components

function MetricBox({ 
  label, 
  value, 
  color = 'default' 
}: { 
  label: string; 
  value: string; 
  color?: 'default' | 'profit' | 'loss' | 'warning';
}) {
  const colorClass = {
    default: 'text-white',
    profit: 'text-profit',
    loss: 'text-loss',
    warning: 'text-accent-yellow',
  }[color];

  return (
    <div className="p-3 bg-terminal-surface rounded-lg border border-terminal-border">
      <div className="text-xs text-terminal-muted mb-1">{label}</div>
      <div className={`font-mono text-lg ${colorClass}`}>{value}</div>
    </div>
  );
}

function ProvenanceRow({ 
  label, 
  value, 
  copyable = false,
  onCopy,
  copiedField
}: { 
  label: string; 
  value: string; 
  copyable?: boolean;
  onCopy?: (value: string, field: string) => void;
  copiedField?: string | null;
}) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-terminal-border/50 last:border-0">
      <span className="text-sm text-terminal-muted">{label}</span>
      <div className="flex items-center gap-2">
        <span className="font-mono text-sm truncate max-w-[200px]" title={value}>
          {value}
        </span>
        {copyable && onCopy && (
          <button
            onClick={() => onCopy(value, label)}
            className="p-1 hover:bg-terminal-surface rounded transition-colors"
            title="Copy to clipboard"
          >
            {copiedField === label ? (
              <Check className="w-3 h-3 text-profit" />
            ) : (
              <Copy className="w-3 h-3 text-terminal-muted" />
            )}
          </button>
        )}
      </div>
    </div>
  );
}

function formatParamValue(value: unknown): string {
  if (typeof value === 'number') {
    return value % 1 === 0 ? String(value) : value.toFixed(4);
  }
  if (typeof value === 'boolean') {
    return value ? 'true' : 'false';
  }
  if (typeof value === 'string') {
    return value;
  }
  return JSON.stringify(value);
}


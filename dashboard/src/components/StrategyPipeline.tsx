import { useState, useEffect } from 'react';
import { config } from '../lib/platform';
import { 
  Database, 
  Activity, 
  Zap, 
  Scale, 
  Shield,
  ChevronDown,
  ChevronRight,
  RefreshCw,
  Settings,
  Layers
} from 'lucide-react';

interface PipelineBlock {
  id: string;
  type: string;
  name: string;
  params: Record<string, any>;
  children?: PipelineBlock[];
}

interface PipelineData {
  version: string;
  genome_hash: string;
  blocks: PipelineBlock[];
  execution: {
    delay_bars: number;
    fill_policy: string;
    slippage_bps: number;
    commission_bps: number;
  };
}

interface Props {
  candidateId: string;
}

const typeIcons: Record<string, React.ComponentType<{ className?: string }>> = {
  DataLoader: Database,
  FeatureExtractor: Activity,
  Indicator: Activity,
  SignalGenerator: Zap,
  PositionSizer: Scale,
  RiskManager: Shield,
};

const typeColors: Record<string, string> = {
  DataLoader: 'from-blue-500/20 to-blue-600/10 border-blue-500/30',
  FeatureExtractor: 'from-purple-500/20 to-purple-600/10 border-purple-500/30',
  Indicator: 'from-purple-500/20 to-purple-600/10 border-purple-500/30',
  SignalGenerator: 'from-amber-500/20 to-amber-600/10 border-amber-500/30',
  PositionSizer: 'from-cyan-500/20 to-cyan-600/10 border-cyan-500/30',
  RiskManager: 'from-red-500/20 to-red-600/10 border-red-500/30',
};

const typeTextColors: Record<string, string> = {
  DataLoader: 'text-blue-400',
  FeatureExtractor: 'text-purple-400',
  Indicator: 'text-purple-400',
  SignalGenerator: 'text-amber-400',
  PositionSizer: 'text-cyan-400',
  RiskManager: 'text-red-400',
};

export function StrategyPipeline({ candidateId }: Props) {
  const [pipeline, setPipeline] = useState<PipelineData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedBlocks, setExpandedBlocks] = useState<Set<string>>(new Set(['features', 'risk']));

  useEffect(() => {
    loadPipeline();
  }, [candidateId]);

  const loadPipeline = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${config.apiBase}/candidate/${candidateId}/pipeline`);
      if (response.ok) {
        const data = await response.json();
        setPipeline(data);
      } else {
        setError('Failed to load pipeline');
      }
    } catch (err) {
      setError('Connection error');
    } finally {
      setLoading(false);
    }
  };

  const toggleExpand = (blockId: string) => {
    setExpandedBlocks(prev => {
      const next = new Set(prev);
      if (next.has(blockId)) {
        next.delete(blockId);
      } else {
        next.add(blockId);
      }
      return next;
    });
  };

  if (loading) {
    return (
      <div className="space-y-4 animate-pulse">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-lg bg-terminal-surface" />
          <div className="space-y-2">
            <div className="h-4 w-32 bg-terminal-surface rounded" />
            <div className="h-3 w-24 bg-terminal-surface rounded" />
          </div>
        </div>
        {[1, 2, 3].map(i => (
          <div key={i} className="ml-14 p-4 rounded-xl bg-terminal-surface/50 border border-terminal-border">
            <div className="h-4 w-40 bg-terminal-border rounded mb-2" />
            <div className="h-3 w-64 bg-terminal-border/50 rounded" />
          </div>
        ))}
      </div>
    );
  }

  if (error || !pipeline || !pipeline.blocks) {
    return (
      <div className="flex items-center justify-center h-64 text-terminal-muted">
        {error || 'No pipeline data'}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-gradient-to-br from-profit/20 to-profit/5">
            <Layers className="w-5 h-5 text-profit" />
          </div>
          <div>
            <h3 className="font-semibold">Strategy Pipeline</h3>
            <p className="text-xs text-terminal-muted font-mono">{pipeline.genome_hash}</p>
          </div>
        </div>
        <span className="text-xs text-terminal-muted bg-terminal-bg px-2 py-1 rounded">
          v{pipeline.version}
        </span>
      </div>

      {/* Pipeline Flow */}
      <div className="relative">
        {/* Connection Line */}
        <div className="absolute left-7 top-0 bottom-0 w-0.5 bg-gradient-to-b from-blue-500/50 via-purple-500/50 to-red-500/50" />
        
        {/* Blocks */}
        <div className="space-y-4 relative">
          {pipeline.blocks.map((block, index) => (
            <PipelineBlockComponent
              key={block.id}
              block={block}
              isFirst={index === 0}
              isLast={index === pipeline.blocks.length - 1}
              expanded={expandedBlocks.has(block.id)}
              onToggle={() => toggleExpand(block.id)}
            />
          ))}
        </div>
      </div>

      {/* Execution Config */}
      {pipeline.execution && (
        <div className="p-4 rounded-xl bg-terminal-surface border border-terminal-border">
          <div className="flex items-center gap-2 mb-3">
            <Settings className="w-4 h-4 text-terminal-muted" />
            <h4 className="text-sm font-medium text-terminal-muted">Execution Configuration</h4>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <ExecutionParam label="Delay Bars" value={`${pipeline.execution.delay_bars ?? 0}`} />
            <ExecutionParam label="Fill Policy" value={(pipeline.execution.fill_policy ?? 'unknown').replace(/_/g, ' ')} />
            <ExecutionParam label="Slippage" value={`${pipeline.execution.slippage_bps ?? 0} bps`} />
            <ExecutionParam label="Commission" value={`${pipeline.execution.commission_bps ?? 0} bps`} />
          </div>
        </div>
      )}
    </div>
  );
}

function PipelineBlockComponent({ 
  block, 
  isFirst, 
  isLast,
  expanded,
  onToggle 
}: { 
  block: PipelineBlock; 
  isFirst: boolean;
  isLast: boolean;
  expanded: boolean;
  onToggle: () => void;
}) {
  const Icon = typeIcons[block.type] || Activity;
  const colorClass = typeColors[block.type] || 'from-gray-500/20 to-gray-600/10 border-gray-500/30';
  const textColor = typeTextColors[block.type] || 'text-gray-400';
  const hasChildren = block.children && block.children.length > 0;

  return (
    <div className="relative pl-14">
      {/* Connector Node */}
      <div className={`absolute left-5 top-4 w-5 h-5 rounded-full bg-gradient-to-br ${colorClass} border flex items-center justify-center z-10`}>
        <div className={`w-2 h-2 rounded-full ${textColor.replace('text-', 'bg-')}`} />
      </div>

      {/* Block Card */}
      <div className={`p-4 rounded-xl bg-gradient-to-br ${colorClass} border transition-all`}>
        {/* Block Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Icon className={`w-5 h-5 ${textColor}`} />
            <div>
              <div className="flex items-center gap-2">
                <span className="font-medium">{block.name}</span>
                <span className="text-xs text-terminal-muted bg-terminal-bg/50 px-2 py-0.5 rounded">
                  {block.type}
                </span>
              </div>
            </div>
          </div>
          {hasChildren && (
            <button 
              onClick={onToggle}
              className="p-1 rounded hover:bg-terminal-bg/50 transition-colors"
            >
              {expanded ? (
                <ChevronDown className="w-4 h-4 text-terminal-muted" />
              ) : (
                <ChevronRight className="w-4 h-4 text-terminal-muted" />
              )}
            </button>
          )}
        </div>

        {/* Block Params */}
        {Object.keys(block.params).length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {Object.entries(block.params).map(([key, value]) => (
              <span 
                key={key}
                className="text-xs font-mono bg-terminal-bg/50 px-2 py-1 rounded"
              >
                <span className="text-terminal-muted">{key}:</span>{' '}
                <span className={textColor}>
                  {Array.isArray(value) ? value.join(', ') : String(value)}
                </span>
              </span>
            ))}
          </div>
        )}

        {/* Children */}
        {hasChildren && expanded && (
          <div className="mt-4 pl-4 border-l border-terminal-border/50 space-y-2">
            {block.children!.map((child) => (
              <div 
                key={child.id}
                className="p-2 rounded-lg bg-terminal-bg/30 text-sm"
              >
                <div className="flex items-center gap-2 mb-1">
                  <span className={`font-medium ${textColor}`}>{child.name}</span>
                </div>
                {Object.keys(child.params).length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {Object.entries(child.params).map(([key, value]) => (
                      <span key={key} className="text-xs font-mono text-terminal-muted">
                        {key}={String(value)}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ExecutionParam({ label, value }: { label: string; value: string }) {
  return (
    <div className="text-center">
      <div className="text-xs text-terminal-muted uppercase tracking-wider">{label}</div>
      <div className="font-mono font-medium mt-1">{value}</div>
    </div>
  );
}


import { useEffect, useState } from 'react';
import { useDataStore } from '../stores/dataStore';
import type { CampaignSummary, RunSummary } from '../stores/dataStore';
import { open } from '@tauri-apps/plugin-dialog';
import { 
  FolderOpen, 
  ChevronRight, 
  ChevronDown, 
  RefreshCw,
  Calendar,
  GitBranch,
  Tag,
  CheckCircle,
  Clock,
  Users,
  Award,
  TrendingUp,
  Shield
} from 'lucide-react';

export function Campaigns() {
  const [expandedCampaigns, setExpandedCampaigns] = useState<Set<string>>(new Set());
  const [selectingFolder, setSelectingFolder] = useState(false);
  
  const {
    artifactsRoot,
    siteIndex,
    campaigns,
    selectedCampaign,
    isLoading,
    error,
    setArtifactsRoot,
    loadIndex,
    loadCampaign,
    loadRun
  } = useDataStore();

  // Auto-load index when artifacts root is set
  useEffect(() => {
    if (artifactsRoot && !siteIndex) {
      loadIndex();
    }
  }, [artifactsRoot]);

  const handleSelectFolder = async () => {
    setSelectingFolder(true);
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
    setSelectingFolder(false);
  };

  const toggleCampaign = async (campaignId: string) => {
    const newExpanded = new Set(expandedCampaigns);
    if (newExpanded.has(campaignId)) {
      newExpanded.delete(campaignId);
    } else {
      newExpanded.add(campaignId);
      // Load campaign detail if not already loaded
      if (!selectedCampaign || selectedCampaign.campaign.campaign_id !== campaignId) {
        await loadCampaign(campaignId);
      }
    }
    setExpandedCampaigns(newExpanded);
  };

  const handleSelectRun = async (runId: string) => {
    await loadRun(runId);
    // Navigate to candidates page - would use router in production
    window.dispatchEvent(new CustomEvent('navigate', { detail: 'candidates' }));
  };

  // No artifacts root selected
  if (!artifactsRoot) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-6">
        <FolderOpen className="w-20 h-20 text-terminal-muted" />
        <div className="text-center max-w-md">
          <h2 className="text-2xl font-semibold mb-3">Welcome to Quant Dashboard</h2>
          <p className="text-terminal-muted mb-6">
            Select your project folder containing SCG artifacts to browse campaigns, 
            runs, and validated strategy candidates.
          </p>
          <button
            onClick={handleSelectFolder}
            disabled={selectingFolder}
            className="px-8 py-4 bg-profit text-black font-semibold rounded-lg hover:bg-profit/90 transition-colors disabled:opacity-50"
          >
            {selectingFolder ? (
              <span className="flex items-center gap-2">
                <RefreshCw className="w-5 h-5 animate-spin" />
                Loading...
              </span>
            ) : (
              'Select Project Folder'
            )}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Campaigns</h1>
          <p className="text-terminal-muted mt-1">
            Browse SCG campaigns and runs
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="px-3 py-1.5 bg-terminal-surface border border-terminal-border rounded-lg text-sm font-mono truncate max-w-xs">
            {artifactsRoot}
          </div>
          <button
            onClick={handleSelectFolder}
            className="flex items-center gap-2 px-3 py-2 rounded-lg bg-terminal-surface border border-terminal-border hover:border-profit transition-colors"
          >
            <FolderOpen className="w-4 h-4" />
            Change
          </button>
          <button
            onClick={loadIndex}
            className="flex items-center gap-2 px-3 py-2 rounded-lg bg-terminal-surface border border-terminal-border hover:border-profit transition-colors"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* Error message */}
      {error && (
        <div className="p-4 bg-loss/10 border border-loss/30 rounded-lg text-loss">
          {error}
        </div>
      )}

      {/* Index info */}
      {siteIndex && (
        <div className="flex items-center gap-4 text-sm text-terminal-muted">
          <span>Schema: {siteIndex.schema_version}</span>
          <span>•</span>
          <span>Generated: {new Date(siteIndex.generated_at).toLocaleString()}</span>
          <span>•</span>
          <span>{campaigns.length} campaigns</span>
        </div>
      )}

      {/* Loading */}
      {isLoading && campaigns.length === 0 && (
        <div className="flex items-center justify-center h-64">
          <RefreshCw className="w-8 h-8 animate-spin text-terminal-muted" />
        </div>
      )}

      {/* Campaigns List */}
      <div className="space-y-3">
        {campaigns.map((campaign) => (
          <CampaignCard
            key={campaign.campaign_id}
            campaign={campaign}
            isExpanded={expandedCampaigns.has(campaign.campaign_id)}
            runs={selectedCampaign?.campaign.campaign_id === campaign.campaign_id 
              ? selectedCampaign.runs 
              : []}
            onToggle={() => toggleCampaign(campaign.campaign_id)}
            onSelectRun={handleSelectRun}
            isLoading={isLoading && expandedCampaigns.has(campaign.campaign_id)}
          />
        ))}

        {campaigns.length === 0 && !isLoading && (
          <div className="flex flex-col items-center justify-center h-64 text-terminal-muted">
            <Award className="w-12 h-12 mb-4 opacity-50" />
            <p>No campaigns found</p>
            <p className="text-sm mt-1">Check that your artifacts folder is correct</p>
          </div>
        )}
      </div>
    </div>
  );
}

// Campaign Card Component
interface CampaignCardProps {
  campaign: CampaignSummary;
  isExpanded: boolean;
  runs: RunSummary[];
  onToggle: () => void;
  onSelectRun: (runId: string) => void;
  isLoading: boolean;
}

function CampaignCard({ 
  campaign, 
  isExpanded, 
  runs, 
  onToggle, 
  onSelectRun,
  isLoading 
}: CampaignCardProps) {
  const statusColor = {
    completed: 'bg-profit/20 text-profit',
    running: 'bg-accent-cyan/20 text-accent-cyan',
    failed: 'bg-loss/20 text-loss',
    pending: 'bg-terminal-muted/20 text-terminal-muted',
  }[campaign.status] ?? 'bg-terminal-muted/20 text-terminal-muted';

  const tagColor = {
    production: 'bg-profit/20 text-profit border-profit/30',
    blast: 'bg-accent-yellow/20 text-accent-yellow border-accent-yellow/30',
    smoke: 'bg-terminal-muted/20 text-terminal-muted border-terminal-muted/30',
  }[campaign.tag] ?? 'bg-accent-cyan/20 text-accent-cyan border-accent-cyan/30';

  return (
    <div className="card-elevated overflow-hidden">
      {/* Campaign Header */}
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between p-4 text-left hover:bg-terminal-surface/50 transition-colors"
      >
        <div className="flex items-center gap-4">
          {isExpanded ? (
            <ChevronDown className="w-5 h-5 text-terminal-muted" />
          ) : (
            <ChevronRight className="w-5 h-5 text-terminal-muted" />
          )}
          
          <div>
            <div className="flex items-center gap-3 mb-1">
              <h3 className="font-semibold text-lg">{campaign.name}</h3>
              <span className={`px-2 py-0.5 rounded text-xs font-medium ${statusColor}`}>
                {campaign.status}
              </span>
              <span className={`px-2 py-0.5 rounded text-xs font-medium border ${tagColor}`}>
                {campaign.tag}
              </span>
            </div>
            <div className="flex items-center gap-4 text-sm text-terminal-muted">
              <span className="font-mono">{campaign.campaign_id}</span>
              <span className="flex items-center gap-1">
                <Calendar className="w-3 h-3" />
                {new Date(campaign.created_at).toLocaleDateString()}
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-6 text-sm">
          <div className="text-right">
            <div className="text-terminal-muted">Runs</div>
            <div className="font-mono font-semibold">{campaign.runs_count}</div>
          </div>
        </div>
      </button>

      {/* Expanded Runs */}
      {isExpanded && (
        <div className="border-t border-terminal-border bg-terminal-bg">
          {isLoading && runs.length === 0 ? (
            <div className="flex items-center justify-center py-8">
              <RefreshCw className="w-6 h-6 animate-spin text-terminal-muted" />
            </div>
          ) : runs.length === 0 ? (
            <div className="py-8 text-center text-terminal-muted">
              No runs found
            </div>
          ) : (
            <div className="divide-y divide-terminal-border/50">
              {runs.map((run) => (
                <RunRow 
                  key={run.run_id} 
                  run={run} 
                  onSelect={() => onSelectRun(run.run_id)}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Run Row Component
interface RunRowProps {
  run: RunSummary;
  onSelect: () => void;
}

function RunRow({ run, onSelect }: RunRowProps) {
  const statusColor = run.status.includes('completed') 
    ? 'text-profit' 
    : run.status.includes('failed') 
    ? 'text-loss' 
    : 'text-accent-cyan';

  const integrityColor = {
    PASS: 'text-profit',
    FAIL: 'text-loss',
    WARN: 'text-accent-yellow',
  }[run.data_integrity_verdict ?? ''] ?? 'text-terminal-muted';

  return (
    <button
      onClick={onSelect}
      className="w-full flex items-center justify-between px-4 py-3 pl-14 hover:bg-terminal-surface/30 transition-colors text-left"
    >
      <div className="flex items-center gap-6">
        <div>
          <div className="font-mono text-sm text-accent-cyan">{run.run_id}</div>
          <div className="flex items-center gap-3 text-xs text-terminal-muted mt-1">
            <span className="flex items-center gap-1">
              <GitBranch className="w-3 h-3" />
              seed: {run.seed}
            </span>
            {run.duration_secs !== undefined && (
              <span className="flex items-center gap-1">
                <Clock className="w-3 h-3" />
                {run.duration_secs}s
              </span>
            )}
          </div>
        </div>
      </div>

      <div className="flex items-center gap-6 text-sm">
        {/* Status */}
        <div className="flex items-center gap-2">
          <CheckCircle className={`w-4 h-4 ${statusColor}`} />
          <span className={statusColor}>{run.status.replace(/"/g, '')}</span>
        </div>

        {/* Integrity */}
        {run.data_integrity_verdict && (
          <div className="flex items-center gap-2">
            <Shield className={`w-4 h-4 ${integrityColor}`} />
            <span className={integrityColor}>{run.data_integrity_verdict}</span>
          </div>
        )}

        {/* Candidates */}
        <div className="text-right min-w-[80px]">
          <div className="text-terminal-muted text-xs">Validated</div>
          <div className="font-mono font-semibold">
            {run.validated_candidates_count ?? 0}
          </div>
        </div>

        {/* Best Sharpe */}
        {run.best_oos_sharpe_net !== undefined && (
          <div className="text-right min-w-[80px]">
            <div className="text-terminal-muted text-xs">Best Sharpe</div>
            <div className="font-mono font-semibold text-profit">
              {run.best_oos_sharpe_net.toFixed(2)}
            </div>
          </div>
        )}

        <ChevronRight className="w-4 h-4 text-terminal-muted" />
      </div>
    </button>
  );
}


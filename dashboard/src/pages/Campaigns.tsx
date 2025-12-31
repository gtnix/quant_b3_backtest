/**
 * Campaigns - Campaign Timeline & Management
 * 
 * Visual timeline of all SCG campaigns with status, runs, and metrics
 * Data loaded from Neon PostgreSQL
 */

import { useEffect, useState, useMemo } from 'react';
import { config } from '../lib/platform';
import { Sparkline } from '../components/charts/Sparkline';
import { 
  FolderOpen, 
  ChevronRight, 
  ChevronDown, 
  RefreshCw,
  Calendar,
  GitBranch,
  Tag,
  CheckCircle,
  XCircle,
  Clock,
  Users,
  Award,
  TrendingUp,
  Shield,
  Activity,
  Zap,
  Database,
  Target,
  BarChart3,
  PlayCircle,
  AlertTriangle,
  Layers,
  Globe
} from 'lucide-react';

interface Campaign {
  campaign_id: string;
  name: string;
  status: 'completed' | 'running' | 'failed' | 'pending';
  tag?: string;
  market?: string;
  created_at: string;
  updated_at?: string;
  runs_count: number;
  candidates_count?: number;
  validated_count?: number;
  best_sharpe?: number;
  config_path?: string;
}

interface Run {
  run_id: string;
  campaign_id: string;
  seed: number;
  status: string;
  duration_secs?: number;
  candidates_evaluated: number;
  validated_count: number;
  best_sharpe?: number;
  best_cagr?: number;
  created_at: string;
  completed_at?: string;
}

export function Campaigns() {
  const [campaigns, setCampaigns] = useState<Campaign[]>([]);
  const [expandedCampaigns, setExpandedCampaigns] = useState<Set<string>>(new Set());
  const [campaignRuns, setCampaignRuns] = useState<Record<string, Run[]>>({});
  const [loading, setLoading] = useState(true);
  const [loadingRuns, setLoadingRuns] = useState<Set<string>>(new Set());
  const [viewMode, setViewMode] = useState<'timeline' | 'list'>('timeline');
  const [marketFilter, setMarketFilter] = useState<'all' | 'BR' | 'US'>('all');
  const [statusFilter, setStatusFilter] = useState<string>('all');

  // Load campaigns on mount
  useEffect(() => {
    loadCampaigns();
  }, []);

  const loadCampaigns = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${config.apiBase}/campaigns`);
      if (response.ok) {
        const data = await response.json();
        setCampaigns(data.campaigns || []);
      }
    } catch (err) {
      console.error('Failed to load campaigns:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadRuns = async (campaignId: string) => {
    if (campaignRuns[campaignId]) return;
    
    setLoadingRuns(prev => new Set(prev).add(campaignId));
    try {
      const response = await fetch(`${config.apiBase}/campaigns/${campaignId}/runs`);
      if (response.ok) {
        const data = await response.json();
        setCampaignRuns(prev => ({ ...prev, [campaignId]: data.runs || [] }));
      }
    } catch (err) {
      console.error('Failed to load runs:', err);
    } finally {
      setLoadingRuns(prev => {
        const next = new Set(prev);
        next.delete(campaignId);
        return next;
      });
    }
  };

  const toggleCampaign = (campaignId: string) => {
    setExpandedCampaigns(prev => {
      const next = new Set(prev);
      if (next.has(campaignId)) {
        next.delete(campaignId);
      } else {
        next.add(campaignId);
        loadRuns(campaignId);
      }
      return next;
    });
  };

  // Filter campaigns
  const filteredCampaigns = useMemo(() => {
    return campaigns.filter(c => {
      if (marketFilter !== 'all' && c.market !== marketFilter) return false;
      if (statusFilter !== 'all' && c.status !== statusFilter) return false;
      return true;
    });
  }, [campaigns, marketFilter, statusFilter]);

  // Calculate stats
  const stats = useMemo(() => {
    const total = campaigns.length;
    const completed = campaigns.filter(c => c.status === 'completed').length;
    const running = campaigns.filter(c => c.status === 'running').length;
    const failed = campaigns.filter(c => c.status === 'failed').length;
    const totalCandidates = campaigns.reduce((a, c) => a + (c.candidates_count || 0), 0);
    const totalValidated = campaigns.reduce((a, c) => a + (c.validated_count || 0), 0);
    const bestSharpe = Math.max(0, ...campaigns.map(c => c.best_sharpe || 0));
    
    return { total, completed, running, failed, totalCandidates, totalValidated, bestSharpe };
  }, [campaigns]);

  // Group campaigns by date for timeline
  const groupedByDate = useMemo(() => {
    const groups: Record<string, Campaign[]> = {};
    for (const campaign of filteredCampaigns) {
      const date = campaign.created_at.split('T')[0];
      if (!groups[date]) groups[date] = [];
      groups[date].push(campaign);
    }
    return Object.entries(groups).sort((a, b) => b[0].localeCompare(a[0]));
  }, [filteredCampaigns]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-terminal-muted bg-clip-text text-transparent">
            Campaigns
          </h1>
          <p className="text-terminal-muted mt-1">
            Campaign history and run timeline
          </p>
        </div>
        
        <div className="flex items-center gap-3">
          {/* View Mode Toggle */}
          <div className="flex rounded-lg overflow-hidden border border-terminal-border">
            <button
              onClick={() => setViewMode('timeline')}
              className={`px-3 py-2 text-sm transition-colors ${
                viewMode === 'timeline' ? 'bg-profit/20 text-profit' : 'bg-terminal-surface text-terminal-muted hover:text-white'
              }`}
            >
              <Activity className="w-4 h-4" />
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={`px-3 py-2 text-sm transition-colors border-l border-terminal-border ${
                viewMode === 'list' ? 'bg-profit/20 text-profit' : 'bg-terminal-surface text-terminal-muted hover:text-white'
              }`}
            >
              <Layers className="w-4 h-4" />
            </button>
          </div>
          
          <button
            onClick={loadCampaigns}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-terminal-surface border border-terminal-border hover:border-profit transition-colors"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* Stats Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4">
        <StatCard icon={<Layers className="w-5 h-5" />} label="Total" value={stats.total} color="white" />
        <StatCard icon={<CheckCircle className="w-5 h-5" />} label="Completed" value={stats.completed} color="profit" />
        <StatCard icon={<PlayCircle className="w-5 h-5" />} label="Running" value={stats.running} color="cyan" />
        <StatCard icon={<AlertTriangle className="w-5 h-5" />} label="Failed" value={stats.failed} color="loss" />
        <StatCard icon={<Database className="w-5 h-5" />} label="Candidates" value={stats.totalCandidates} color="white" />
        <StatCard icon={<Award className="w-5 h-5" />} label="Validated" value={stats.totalValidated} color="profit" />
        <StatCard icon={<Target className="w-5 h-5" />} label="Best Sharpe" value={stats.bestSharpe.toFixed(2)} color="profit" />
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-4">
        {/* Market Filter */}
        <div className="flex rounded-lg overflow-hidden border border-terminal-border">
          {['all', 'BR', 'US'].map((m) => (
            <button
              key={m}
              onClick={() => setMarketFilter(m as any)}
              className={`px-4 py-2 text-sm transition-colors ${
                marketFilter === m 
                  ? 'bg-profit/20 text-profit' 
                  : 'bg-terminal-surface text-terminal-muted hover:text-white'
              } ${m !== 'all' ? 'border-l border-terminal-border' : ''}`}
            >
              {m === 'all' ? '🌐 All' : m === 'BR' ? '🇧🇷 B3' : '🇺🇸 US'}
            </button>
          ))}
        </div>
        
        {/* Status Filter */}
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          className="px-4 py-2 bg-terminal-surface border border-terminal-border rounded-lg text-sm cursor-pointer"
        >
          <option value="all">All Status</option>
          <option value="completed">Completed</option>
          <option value="running">Running</option>
          <option value="failed">Failed</option>
        </select>
        
        <span className="text-sm text-terminal-muted ml-auto">
          {filteredCampaigns.length} campaigns
        </span>
      </div>

      {/* Loading */}
      {loading && campaigns.length === 0 && (
        <div className="flex items-center justify-center h-64">
          <RefreshCw className="w-8 h-8 animate-spin text-terminal-muted" />
        </div>
      )}

      {/* Timeline View */}
      {viewMode === 'timeline' && !loading && (
        <div className="space-y-8">
          {groupedByDate.map(([date, dateCampaigns]) => (
            <div key={date} className="relative">
              {/* Date Header */}
              <div className="flex items-center gap-4 mb-4">
                <div className="flex items-center gap-2 px-3 py-1.5 bg-terminal-surface border border-terminal-border rounded-lg">
                  <Calendar className="w-4 h-4 text-accent-cyan" />
                  <span className="font-medium">{formatDate(date)}</span>
                </div>
                <div className="flex-1 h-px bg-terminal-border" />
                <span className="text-sm text-terminal-muted">{dateCampaigns.length} campaigns</span>
              </div>

              {/* Timeline Items */}
              <div className="relative pl-8 border-l-2 border-terminal-border/50 space-y-4">
                {dateCampaigns.map((campaign, idx) => (
                  <TimelineCard
                    key={campaign.campaign_id}
                    campaign={campaign}
                    isExpanded={expandedCampaigns.has(campaign.campaign_id)}
                    runs={campaignRuns[campaign.campaign_id] || []}
                    isLoadingRuns={loadingRuns.has(campaign.campaign_id)}
                    onToggle={() => toggleCampaign(campaign.campaign_id)}
                    isFirst={idx === 0}
                  />
                ))}
              </div>
            </div>
          ))}

          {groupedByDate.length === 0 && (
            <EmptyState />
          )}
        </div>
      )}

      {/* List View */}
      {viewMode === 'list' && !loading && (
        <div className="space-y-3">
          {filteredCampaigns.map((campaign) => (
            <CampaignCard
              key={campaign.campaign_id}
              campaign={campaign}
              isExpanded={expandedCampaigns.has(campaign.campaign_id)}
              runs={campaignRuns[campaign.campaign_id] || []}
              isLoadingRuns={loadingRuns.has(campaign.campaign_id)}
              onToggle={() => toggleCampaign(campaign.campaign_id)}
            />
          ))}

          {filteredCampaigns.length === 0 && (
            <EmptyState />
          )}
        </div>
      )}
    </div>
  );
}

// =============================================================================
// HELPER COMPONENTS
// =============================================================================

function StatCard({ icon, label, value, color }: { 
  icon: React.ReactNode; 
  label: string; 
  value: number | string; 
  color: 'profit' | 'loss' | 'cyan' | 'white';
}) {
  const colors = {
    profit: 'text-profit',
    loss: 'text-loss',
    cyan: 'text-accent-cyan',
    white: 'text-white',
  };
  
  return (
    <div className="card p-4">
      <div className="flex items-center gap-2 mb-1">
        <span className={colors[color]}>{icon}</span>
        <span className="text-xs text-terminal-muted uppercase tracking-wider">{label}</span>
      </div>
      <div className={`font-mono text-xl font-bold ${colors[color]}`}>{value}</div>
    </div>
  );
}

function TimelineCard({ 
  campaign, 
  isExpanded, 
  runs,
  isLoadingRuns,
  onToggle,
  isFirst
}: { 
  campaign: Campaign;
  isExpanded: boolean;
  runs: Run[];
  isLoadingRuns: boolean;
  onToggle: () => void;
  isFirst: boolean;
}) {
  const statusConfig = {
    completed: { bg: 'bg-profit', icon: <CheckCircle className="w-4 h-4" /> },
    running: { bg: 'bg-accent-cyan animate-pulse', icon: <PlayCircle className="w-4 h-4" /> },
    failed: { bg: 'bg-loss', icon: <XCircle className="w-4 h-4" /> },
    pending: { bg: 'bg-terminal-muted', icon: <Clock className="w-4 h-4" /> },
  };
  
  const { bg, icon } = statusConfig[campaign.status] || statusConfig.pending;

  return (
    <div className="relative">
      {/* Timeline dot */}
      <div className={`absolute -left-[41px] w-4 h-4 rounded-full ${bg} flex items-center justify-center`}>
        <div className="w-2 h-2 bg-white rounded-full" />
      </div>
      
      {/* Card */}
      <div className="card-elevated overflow-hidden ml-2">
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
                <h3 className="font-semibold">{campaign.name}</h3>
                <StatusBadge status={campaign.status} />
                {campaign.market && <MarketBadge market={campaign.market} />}
              </div>
              <div className="flex items-center gap-4 text-xs text-terminal-muted">
                <span className="font-mono">{campaign.campaign_id.slice(0, 8)}...</span>
                <span>{formatTime(campaign.created_at)}</span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-6 text-sm">
            <MetricPill label="Runs" value={campaign.runs_count} />
            {campaign.candidates_count !== undefined && (
              <MetricPill label="Candidates" value={campaign.candidates_count} />
            )}
            {campaign.validated_count !== undefined && (
              <MetricPill label="Validated" value={campaign.validated_count} color="profit" />
            )}
            {campaign.best_sharpe != null && (
              <MetricPill label="Best Sharpe" value={campaign.best_sharpe.toFixed(2)} color="profit" />
            )}
          </div>
        </button>

        {/* Expanded Runs */}
        {isExpanded && (
          <div className="border-t border-terminal-border bg-terminal-bg">
            {isLoadingRuns ? (
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
                  <RunRow key={run.run_id} run={run} />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function CampaignCard({ 
  campaign, 
  isExpanded, 
  runs,
  isLoadingRuns,
  onToggle 
}: { 
  campaign: Campaign;
  isExpanded: boolean;
  runs: Run[];
  isLoadingRuns: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="card-elevated overflow-hidden">
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
              <StatusBadge status={campaign.status} />
              {campaign.market && <MarketBadge market={campaign.market} />}
            </div>
            <div className="flex items-center gap-4 text-sm text-terminal-muted">
              <span className="font-mono">{campaign.campaign_id}</span>
              <span className="flex items-center gap-1">
                <Calendar className="w-3 h-3" />
                {formatDateTime(campaign.created_at)}
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-6 text-sm">
          <MetricPill label="Runs" value={campaign.runs_count} />
          {campaign.candidates_count !== undefined && (
            <MetricPill label="Candidates" value={campaign.candidates_count} />
          )}
          {campaign.validated_count !== undefined && (
            <MetricPill label="Validated" value={campaign.validated_count} color="profit" />
          )}
          {campaign.best_sharpe != null && (
            <MetricPill label="Best Sharpe" value={campaign.best_sharpe.toFixed(2)} color="profit" />
          )}
        </div>
      </button>

      {/* Expanded Runs */}
      {isExpanded && (
        <div className="border-t border-terminal-border bg-terminal-bg">
          {isLoadingRuns ? (
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
                <RunRow key={run.run_id} run={run} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function RunRow({ run }: { run: Run }) {
  const statusColor = run.status.includes('completed') 
    ? 'text-profit' 
    : run.status.includes('failed') 
    ? 'text-loss' 
    : 'text-accent-cyan';

  return (
    <div className="flex items-center justify-between px-4 py-3 pl-14 hover:bg-terminal-surface/30 transition-colors">
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
                {formatDuration(run.duration_secs)}
              </span>
            )}
          </div>
        </div>
      </div>

      <div className="flex items-center gap-6 text-sm">
        {/* Status */}
        <div className="flex items-center gap-2">
          <CheckCircle className={`w-4 h-4 ${statusColor}`} />
          <span className={statusColor}>{run.status}</span>
        </div>

        {/* Candidates */}
        <div className="text-right min-w-[80px]">
          <div className="text-terminal-muted text-xs">Candidates</div>
          <div className="font-mono font-semibold">{run.candidates_evaluated}</div>
        </div>

        {/* Validated */}
        <div className="text-right min-w-[80px]">
          <div className="text-terminal-muted text-xs">Validated</div>
          <div className="font-mono font-semibold text-profit">{run.validated_count}</div>
        </div>

        {/* Best Sharpe */}
        {run.best_sharpe != null && (
          <div className="text-right min-w-[80px]">
            <div className="text-terminal-muted text-xs">Best Sharpe</div>
            <div className="font-mono font-semibold text-profit">{run.best_sharpe.toFixed(2)}</div>
          </div>
        )}

        <ChevronRight className="w-4 h-4 text-terminal-muted" />
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: Campaign['status'] }) {
  const config = {
    completed: 'bg-profit/20 text-profit',
    running: 'bg-accent-cyan/20 text-accent-cyan',
    failed: 'bg-loss/20 text-loss',
    pending: 'bg-terminal-muted/20 text-terminal-muted',
  };
  
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${config[status]}`}>
      {status}
    </span>
  );
}

function MarketBadge({ market }: { market: string }) {
  const isBR = market === 'BR' || market.toLowerCase().includes('b3');
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${
      isBR ? 'bg-green-500/20 text-green-400' : 'bg-blue-500/20 text-blue-400'
    }`}>
      {isBR ? '🇧🇷 B3' : '🇺🇸 US'}
    </span>
  );
}

function MetricPill({ label, value, color }: { label: string; value: number | string; color?: 'profit' | 'loss' }) {
  return (
    <div className="text-right">
      <div className="text-terminal-muted text-xs">{label}</div>
      <div className={`font-mono font-semibold ${color === 'profit' ? 'text-profit' : color === 'loss' ? 'text-loss' : ''}`}>
        {value}
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-terminal-muted">
      <Layers className="w-16 h-16 mb-4 opacity-30" />
      <p className="text-lg">No campaigns found</p>
      <p className="text-sm mt-1">Start a campaign to see it here</p>
    </div>
  );
}

// =============================================================================
// UTILITIES
// =============================================================================

function formatDate(dateStr: string): string {
  const date = new Date(dateStr);
  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  
  if (dateStr === today.toISOString().split('T')[0]) return 'Today';
  if (dateStr === yesterday.toISOString().split('T')[0]) return 'Yesterday';
  
  return date.toLocaleDateString('en-US', { 
    weekday: 'short', 
    month: 'short', 
    day: 'numeric' 
  });
}

function formatTime(dateStr: string): string {
  return new Date(dateStr).toLocaleTimeString('en-US', { 
    hour: '2-digit', 
    minute: '2-digit' 
  });
}

function formatDateTime(dateStr: string): string {
  return new Date(dateStr).toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
}

function formatDuration(secs: number): string {
  if (secs < 60) return `${secs}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${secs % 60}s`;
  const hours = Math.floor(secs / 3600);
  const mins = Math.floor((secs % 3600) / 60);
  return `${hours}h ${mins}m`;
}

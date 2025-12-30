/**
 * MinerControl - Perpetual Mining Orchestrator Control Panel
 * 
 * Replaces the old Cockpit with a 24/7 mining-focused interface.
 */

import { useEffect } from 'react';
import { 
  Play, Pause, Square, Activity, Cpu, HardDrive, 
  Trophy, TrendingUp, BarChart3, Clock, Zap,
  CheckCircle2, XCircle, AlertCircle, Wifi, WifiOff,
  RefreshCw, ChevronRight, Globe
} from 'lucide-react';
import { useOmpStore } from '../stores/ompStore';
import type { OmpStatus, CurrentCampaign, QueuedCampaign } from '../stores/ompStore';

// =============================================================================
// STATUS BADGE COMPONENT
// =============================================================================

function StatusBadge({ status }: { status: OmpStatus }) {
  const styles: Record<OmpStatus, { bg: string; text: string; dot: string }> = {
    running: { bg: 'bg-emerald-500/20', text: 'text-emerald-400', dot: 'bg-emerald-500 animate-pulse' },
    paused: { bg: 'bg-amber-500/20', text: 'text-amber-400', dot: 'bg-amber-500' },
    draining: { bg: 'bg-orange-500/20', text: 'text-orange-400', dot: 'bg-orange-500 animate-pulse' },
    offline: { bg: 'bg-slate-500/20', text: 'text-slate-400', dot: 'bg-slate-500' },
  };
  
  const style = styles[status];
  
  return (
    <span className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full ${style.bg} ${style.text} text-sm font-medium uppercase tracking-wider`}>
      <span className={`w-2 h-2 rounded-full ${style.dot}`} />
      {status}
    </span>
  );
}

// =============================================================================
// METRIC CARD COMPONENT
// =============================================================================

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: React.ElementType;
  trend?: 'up' | 'down' | 'neutral';
  color?: 'emerald' | 'amber' | 'rose' | 'blue' | 'violet';
}

function MetricCard({ title, value, subtitle, icon: Icon, trend, color = 'blue' }: MetricCardProps) {
  const colors = {
    emerald: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
    amber: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
    rose: 'text-rose-400 bg-rose-500/10 border-rose-500/20',
    blue: 'text-blue-400 bg-blue-500/10 border-blue-500/20',
    violet: 'text-violet-400 bg-violet-500/10 border-violet-500/20',
  };
  
  const iconColors = {
    emerald: 'text-emerald-500',
    amber: 'text-amber-500',
    rose: 'text-rose-500',
    blue: 'text-blue-500',
    violet: 'text-violet-500',
  };
  
  return (
    <div className={`rounded-xl border p-4 ${colors[color]}`}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs text-slate-400 uppercase tracking-wider mb-1">{title}</p>
          <p className="text-2xl font-bold text-white">{value}</p>
          {subtitle && <p className="text-xs text-slate-500 mt-1">{subtitle}</p>}
        </div>
        <Icon className={`w-5 h-5 ${iconColors[color]}`} />
      </div>
    </div>
  );
}

// =============================================================================
// RESOURCE GAUGE COMPONENT
// =============================================================================

interface GaugeProps {
  label: string;
  value: number;
  max?: number;
  unit?: string;
  warning?: number;
  danger?: number;
}

function ResourceGauge({ label, value, max = 100, unit = '%', warning = 70, danger = 85 }: GaugeProps) {
  const pct = Math.min((value / max) * 100, 100);
  const color = pct >= danger ? 'bg-rose-500' : pct >= warning ? 'bg-amber-500' : 'bg-emerald-500';
  
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-slate-400">{label}</span>
        <span className="text-white font-mono">{value.toFixed(1)}{unit}</span>
      </div>
      <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full transition-all duration-500`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

// =============================================================================
// CAMPAIGN CARD COMPONENT
// =============================================================================

function ActiveCampaignCard({ campaign }: { campaign: CurrentCampaign }) {
  const elapsed = campaign.elapsedSeconds || 0;
  const mins = Math.floor(elapsed / 60);
  const secs = elapsed % 60;
  
  return (
    <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <span className="text-emerald-400 text-sm font-medium">Active Campaign</span>
        </div>
        <span className="text-xs text-slate-400 font-mono">{campaign.runId}</span>
      </div>
      
      <h3 className="text-lg font-semibold text-white mb-2">{campaign.campaignName}</h3>
      
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
        <div className="text-center p-2 bg-slate-800/50 rounded-lg">
          <p className="text-xs text-slate-400">Generation</p>
          <p className="text-lg font-bold text-white">{campaign.currentGeneration}</p>
        </div>
        <div className="text-center p-2 bg-slate-800/50 rounded-lg">
          <p className="text-xs text-slate-400">Best Sharpe</p>
          <p className="text-lg font-bold text-emerald-400">{campaign.bestSharpe?.toFixed(3) || '—'}</p>
        </div>
        <div className="text-center p-2 bg-slate-800/50 rounded-lg">
          <p className="text-xs text-slate-400">Candidates</p>
          <p className="text-lg font-bold text-white">{campaign.candidatesEvaluated.toLocaleString()}</p>
        </div>
        <div className="text-center p-2 bg-slate-800/50 rounded-lg">
          <p className="text-xs text-slate-400">Elapsed</p>
          <p className="text-lg font-bold text-white font-mono">{mins}:{secs.toString().padStart(2, '0')}</p>
        </div>
      </div>
      
      <div className="flex items-center gap-2 mt-3 text-xs text-slate-400">
        <Globe className="w-3 h-3" />
        <span>{campaign.market.toUpperCase()}</span>
      </div>
    </div>
  );
}

// =============================================================================
// QUEUE ITEM COMPONENT
// =============================================================================

function QueueItem({ campaign, onToggle, onRemove }: { 
  campaign: QueuedCampaign; 
  onToggle: () => void;
  onRemove: () => void;
}) {
  return (
    <div className={`flex items-center justify-between p-3 rounded-lg border ${
      campaign.enabled 
        ? 'bg-slate-800/50 border-slate-700' 
        : 'bg-slate-900/50 border-slate-800 opacity-60'
    }`}>
      <div className="flex items-center gap-3">
        <button
          onClick={onToggle}
          className={`w-8 h-8 rounded-lg flex items-center justify-center transition-colors ${
            campaign.enabled 
              ? 'bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30' 
              : 'bg-slate-700 text-slate-500 hover:bg-slate-600'
          }`}
        >
          {campaign.enabled ? <CheckCircle2 className="w-4 h-4" /> : <XCircle className="w-4 h-4" />}
        </button>
        <div>
          <p className="text-sm font-medium text-white">{campaign.name}</p>
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <span>{campaign.market.toUpperCase()}</span>
            <span>•</span>
            <span>Priority {campaign.priority}</span>
            {campaign.repeat && (
              <>
                <span>•</span>
                <RefreshCw className="w-3 h-3" />
              </>
            )}
          </div>
        </div>
      </div>
      <button
        onClick={onRemove}
        className="text-slate-500 hover:text-rose-400 p-2 transition-colors"
      >
        <XCircle className="w-4 h-4" />
      </button>
    </div>
  );
}

// =============================================================================
// HALL OF FAME WIDGET
// =============================================================================

function HallOfFameWidget() {
  const { hallOfFame, hallOfFameLoading, fetchHallOfFame } = useOmpStore();
  
  useEffect(() => {
    fetchHallOfFame(5);
  }, [fetchHallOfFame]);
  
  const topEntries = hallOfFame.slice(0, 5);
  
  return (
    <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Trophy className="w-4 h-4 text-amber-400" />
          <span className="text-amber-400 font-medium text-sm">Hall of Fame</span>
        </div>
        <a href="#/hall-of-fame" className="text-xs text-slate-400 hover:text-white flex items-center gap-1">
          View All <ChevronRight className="w-3 h-3" />
        </a>
      </div>
      
      {hallOfFameLoading ? (
        <div className="flex items-center justify-center py-6">
          <RefreshCw className="w-5 h-5 text-slate-500 animate-spin" />
        </div>
      ) : topEntries.length === 0 ? (
        <p className="text-sm text-slate-500 text-center py-4">No promotions yet</p>
      ) : (
        <div className="space-y-2">
          {topEntries.map((entry, i) => (
            <div key={entry.promotionId} className="flex items-center justify-between p-2 bg-slate-800/50 rounded-lg">
              <div className="flex items-center gap-2">
                <span className="w-5 h-5 rounded-full bg-amber-500/20 text-amber-400 text-xs flex items-center justify-center font-medium">
                  {i + 1}
                </span>
                <span className="text-sm text-white font-mono">{entry.candidateId.slice(0, 12)}...</span>
              </div>
              <div className="text-right">
                <span className="text-sm font-bold text-emerald-400">{entry.metrics.oosSharpeNet?.toFixed(3)}</span>
                <span className="text-xs text-slate-500 ml-1">Sharpe</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function MinerControl() {
  const {
    status,
    startedAt,
    lastLoop,
    loopCount,
    queueLength,
    lastPromotion,
    currentCampaign,
    resources,
    stats,
    queue,
    sseConnected,
    lastError,
    start,
    stop,
    pause,
    resume,
    fetchQueue,
    updateQueueItem,
    removeFromQueue,
    subscribeToUpdates,
  } = useOmpStore();
  
  // Subscribe to updates on mount
  useEffect(() => {
    const unsubscribe = subscribeToUpdates();
    return unsubscribe;
  }, [subscribeToUpdates]);
  
  // Periodic queue refresh
  useEffect(() => {
    fetchQueue();
  }, [fetchQueue]);
  
  const isRunning = status === 'running';
  const isPaused = status === 'paused';
  const isOffline = status === 'offline';
  
  // Calculate uptime
  let uptimeStr = '—';
  if (startedAt) {
    const uptime = Math.floor((Date.now() - new Date(startedAt).getTime()) / 1000);
    const hrs = Math.floor(uptime / 3600);
    const mins = Math.floor((uptime % 3600) / 60);
    uptimeStr = `${hrs}h ${mins}m`;
  }
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-white flex items-center gap-3">
              <Zap className="w-6 h-6 text-amber-400" />
              Strategy Miner
            </h1>
            <p className="text-slate-400 text-sm mt-1">
              Orquestrador de Mineração Perpétua — 24/7 Strategy Discovery
            </p>
          </div>
          
          <div className="flex items-center gap-4">
            {/* SSE Status */}
            <div className={`flex items-center gap-1.5 text-xs ${sseConnected ? 'text-emerald-400' : 'text-rose-400'}`}>
              {sseConnected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
              {sseConnected ? 'Live' : 'Offline'}
            </div>
            
            <StatusBadge status={status} />
            
            {/* Control Buttons */}
            <div className="flex items-center gap-2">
              {isOffline && (
                <button
                  onClick={start}
                  className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg font-medium transition-colors"
                >
                  <Play className="w-4 h-4" />
                  Start Mining
                </button>
              )}
              
              {isRunning && (
                <>
                  <button
                    onClick={pause}
                    className="flex items-center gap-2 px-4 py-2 bg-amber-600 hover:bg-amber-500 text-white rounded-lg font-medium transition-colors"
                  >
                    <Pause className="w-4 h-4" />
                    Pause
                  </button>
                  <button
                    onClick={stop}
                    className="flex items-center gap-2 px-4 py-2 bg-rose-600 hover:bg-rose-500 text-white rounded-lg font-medium transition-colors"
                  >
                    <Square className="w-4 h-4" />
                    Stop
                  </button>
                </>
              )}
              
              {isPaused && (
                <>
                  <button
                    onClick={resume}
                    className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg font-medium transition-colors"
                  >
                    <Play className="w-4 h-4" />
                    Resume
                  </button>
                  <button
                    onClick={stop}
                    className="flex items-center gap-2 px-4 py-2 bg-rose-600 hover:bg-rose-500 text-white rounded-lg font-medium transition-colors"
                  >
                    <Square className="w-4 h-4" />
                    Stop
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
        
        {/* Error Banner */}
        {lastError && (
          <div className="flex items-center gap-2 p-3 bg-rose-500/10 border border-rose-500/30 rounded-lg text-rose-400 text-sm">
            <AlertCircle className="w-4 h-4" />
            {lastError}
          </div>
        )}
        
        {/* Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
          <MetricCard 
            title="Candidates (24h)" 
            value={stats?.candidates.last24h?.toLocaleString() || '0'} 
            icon={BarChart3}
            color="blue"
          />
          <MetricCard 
            title="Promotions (24h)" 
            value={stats?.promotions.last24h || 0} 
            icon={Trophy}
            color="amber"
          />
          <MetricCard 
            title="Throughput" 
            value={`${(stats?.throughput.candidatesPerMin || 0).toFixed(1)}/min`} 
            icon={Zap}
            color="violet"
          />
          <MetricCard 
            title="Hall of Fame" 
            value={stats?.promotions.total || 0} 
            subtitle="Total promoted"
            icon={Trophy}
            color="emerald"
          />
          <MetricCard 
            title="Uptime" 
            value={uptimeStr} 
            subtitle={`${loopCount} loops`}
            icon={Clock}
            color="blue"
          />
          <MetricCard 
            title="Queue" 
            value={queueLength} 
            subtitle="Campaigns pending"
            icon={Activity}
            color="violet"
          />
        </div>
        
        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Left Column - Active Campaign + Queue */}
          <div className="lg:col-span-2 space-y-6">
            {/* Active Campaign */}
            {currentCampaign ? (
              <ActiveCampaignCard campaign={currentCampaign} />
            ) : (
              <div className="rounded-xl border border-slate-700 bg-slate-800/30 p-6 text-center">
                <Activity className="w-8 h-8 text-slate-600 mx-auto mb-2" />
                <p className="text-slate-400">No active campaign</p>
                <p className="text-xs text-slate-500 mt-1">
                  {isOffline ? 'Start mining to begin' : 'Waiting for next campaign...'}
                </p>
              </div>
            )}
            
            {/* Campaign Queue */}
            <div className="rounded-xl border border-slate-700 bg-slate-800/30 p-4">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                  <Activity className="w-5 h-5 text-slate-400" />
                  Campaign Queue
                </h2>
                <span className="text-xs text-slate-400">{queue?.campaigns.length || 0} campaigns</span>
              </div>
              
              {queue?.campaigns.length === 0 ? (
                <p className="text-sm text-slate-500 text-center py-4">No campaigns in queue</p>
              ) : (
                <div className="space-y-2">
                  {queue?.campaigns.map(campaign => (
                    <QueueItem 
                      key={campaign.id} 
                      campaign={campaign}
                      onToggle={() => updateQueueItem(campaign.id, { enabled: !campaign.enabled })}
                      onRemove={() => removeFromQueue(campaign.id)}
                    />
                  ))}
                </div>
              )}
            </div>
          </div>
          
          {/* Right Column - Resources + Hall of Fame */}
          <div className="space-y-6">
            
            {/* Resources */}
            <div className="rounded-xl border border-slate-700 bg-slate-800/30 p-4">
              <h2 className="text-lg font-semibold text-white flex items-center gap-2 mb-4">
                <Cpu className="w-5 h-5 text-slate-400" />
                System Resources
              </h2>
              
              <div className="space-y-4">
                <ResourceGauge label="CPU Usage" value={resources.cpuUsage} />
                <ResourceGauge label="Memory" value={resources.memoryUsagePct} />
                <ResourceGauge 
                  label="Disk Free" 
                  value={resources.diskFreeGb} 
                  max={100} 
                  unit=" GB" 
                  warning={20} 
                  danger={10} 
                />
              </div>
              
              <div className={`mt-4 p-2 rounded-lg text-center text-sm ${
                resources.canStartCampaign 
                  ? 'bg-emerald-500/10 text-emerald-400' 
                  : 'bg-rose-500/10 text-rose-400'
              }`}>
                {resources.canStartCampaign ? (
                  <span className="flex items-center justify-center gap-1.5">
                    <CheckCircle2 className="w-4 h-4" />
                    Ready to start campaigns
                  </span>
                ) : (
                  <span className="flex items-center justify-center gap-1.5">
                    <AlertCircle className="w-4 h-4" />
                    Resources constrained
                  </span>
                )}
              </div>
            </div>
            
            {/* Hall of Fame Widget */}
            <HallOfFameWidget />
            
            {/* Last Activity */}
            <div className="rounded-xl border border-slate-700 bg-slate-800/30 p-4">
              <h3 className="text-sm font-medium text-slate-400 mb-3">Activity</h3>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-slate-500">Last Loop</span>
                  <span className="text-slate-300 font-mono">
                    {lastLoop ? new Date(lastLoop).toLocaleTimeString() : '—'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Last Promotion</span>
                  <span className="text-slate-300 font-mono">
                    {lastPromotion ? new Date(lastPromotion).toLocaleTimeString() : '—'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Campaigns Done</span>
                  <span className="text-slate-300">{stats?.campaigns.completed || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Campaigns Failed</span>
                  <span className="text-rose-400">{stats?.campaigns.failed || 0}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default MinerControl;


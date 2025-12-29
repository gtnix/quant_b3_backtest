import { useState, useEffect } from 'react';
import { Activity, Bell, RefreshCw, Clock } from 'lucide-react';
import { AlertsPanel } from '../AlertsPanel';
import { useDataStore } from '../../stores/dataStore';

export function Header() {
  const [time, setTime] = useState(new Date());
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [showAlerts, setShowAlerts] = useState(false);
  
  const { invalidateCache, loadIndex, selectedRunId, listCandidates } = useDataStore();

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    try {
      await loadIndex();
      if (selectedRunId) {
        await listCandidates(selectedRunId);
      }
    } catch (e) {
      console.error('Refresh failed:', e);
    }
    setIsRefreshing(false);
  };

  return (
    <>
      <header className="h-16 bg-terminal-surface border-b border-terminal-border flex items-center justify-between px-6">
        {/* Left side - Status */}
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-profit animate-pulse" />
            <span className="text-sm text-terminal-muted">System Online</span>
          </div>
          
          <div className="h-4 w-px bg-terminal-border" />
          
          <div className="flex items-center gap-2 text-sm">
            <Activity className="w-4 h-4 text-accent-cyan" />
            <span className="text-terminal-muted">NYC/Chicago Quant Platform</span>
          </div>
        </div>

        {/* Right side - Actions */}
        <div className="flex items-center gap-4">
          {/* Clock */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded bg-terminal-bg border border-terminal-border">
            <Clock className="w-4 h-4 text-terminal-muted" />
            <span className="font-mono text-sm">
              {time.toLocaleTimeString('en-US', { hour12: false })}
            </span>
          </div>

          {/* Refresh */}
          <button
            onClick={handleRefresh}
            className="p-2 rounded hover:bg-terminal-border/50 transition-colors"
            title="Refresh data"
          >
            <RefreshCw className={`w-5 h-5 text-terminal-muted ${isRefreshing ? 'animate-spin' : ''}`} />
          </button>

          {/* Notifications */}
          <button 
            onClick={() => setShowAlerts(true)}
            className="relative p-2 rounded hover:bg-terminal-border/50 transition-colors"
            title="Alerts"
          >
            <Bell className="w-5 h-5 text-terminal-muted" />
            <span className="absolute top-1 right-1 w-2 h-2 rounded-full bg-loss animate-pulse" />
          </button>
        </div>
      </header>

      {/* Alerts Panel */}
      <AlertsPanel isOpen={showAlerts} onClose={() => setShowAlerts(false)} />
    </>
  );
}

import { useState, useEffect } from 'react';
import { RefreshCw, Clock, Wifi, WifiOff } from 'lucide-react';
import { useDataStore } from '../../stores/dataStore';

interface HeaderProps {
  sseConnected?: boolean;
}

export function Header({ sseConnected = false }: HeaderProps) {
  const [time, setTime] = useState(new Date());
  const [isRefreshing, setIsRefreshing] = useState(false);
  
  const { loadIndex, selectedRunId, listCandidates } = useDataStore();

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
    <header className="h-16 bg-terminal-surface border-b border-terminal-border flex items-center justify-between px-6">
      {/* Left side - SSE Connection Status */}
      <div className="flex items-center gap-6">
        <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm ${
          sseConnected ? 'bg-profit/10 text-profit' : 'bg-loss/10 text-loss'
        }`}>
          {sseConnected ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
          <span>{sseConnected ? 'Connected' : 'Offline'}</span>
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
      </div>
    </header>
  );
}

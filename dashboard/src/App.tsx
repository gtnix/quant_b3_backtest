import { useState, useEffect, useRef } from 'react';
import { Sidebar } from './components/layout/Sidebar';
import { Header } from './components/layout/Header';
import { Dashboard } from './pages/Dashboard';
import { Evolution } from './pages/Evolution';
import { Candidates } from './pages/Candidates';
import { Backtest } from './pages/Backtest';
import { Campaigns } from './pages/Campaigns';
import { RiskAnalytics } from './pages/RiskAnalytics';
import { StrategyComparison } from './pages/StrategyComparison';
import { WalkForward } from './pages/WalkForward';
import { MonteCarlo } from './pages/MonteCarlo';
import { RegimeAnalysis } from './pages/RegimeAnalysis';
import { Cockpit } from './pages/Cockpit';
import { StrategyView } from './pages/StrategyView';
import { GlossaryOverlay } from './components/GlossaryOverlay';
import { useDataStore } from './stores/dataStore';
import { platform, features, getModeDisplay } from './lib/platform';
import { createSSEConnection, type SSEEvent } from './lib/commands';
import { Monitor, X, Wifi, WifiOff } from 'lucide-react';

// Browser Mode Banner Component with SSE status
function BrowserModeBanner({ onDismiss, sseConnected }: { onDismiss: () => void; sseConnected: boolean }) {
  const modeInfo = getModeDisplay();
  
  return (
    <div className="bg-gradient-to-r from-accent-cyan/20 via-accent-purple/10 to-accent-cyan/20 border-b border-accent-cyan/30 px-4 py-2 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <Monitor className="w-4 h-4 text-accent-cyan" />
        <span className="text-sm">
          <span className="font-semibold text-accent-cyan">{modeInfo.icon} {modeInfo.mode} Mode</span>
          <span className="text-terminal-muted ml-2">
            {modeInfo.description}
          </span>
        </span>
        {/* SSE Connection Status */}
        {features.useSSE && (
          <div className={`flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs ${
            sseConnected 
              ? 'bg-profit/20 text-profit' 
              : 'bg-loss/20 text-loss'
          }`}>
            {sseConnected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
            {sseConnected ? 'Live' : 'Offline'}
          </div>
        )}
      </div>
      <button 
        onClick={onDismiss}
        className="p-1 hover:bg-terminal-surface rounded transition-colors"
        title="Dismiss"
      >
        <X className="w-4 h-4 text-terminal-muted" />
      </button>
    </div>
  );
}

export type Page = 
  | 'cockpit'
  | 'campaigns' 
  | 'dashboard' 
  | 'evolution' 
  | 'candidates' 
  | 'strategy'
  | 'backtest' 
  | 'risk'
  | 'comparison'
  | 'walkforward'
  | 'montecarlo'
  | 'regimes';

function App() {
  const [currentPage, setCurrentPage] = useState<Page>('cockpit');
  const [showBrowserBanner, setShowBrowserBanner] = useState(features.showBrowserBanner);
  const [sseConnected, setSseConnected] = useState(false);
  const sseRef = useRef<EventSource | null>(null);
  const { startWatcher, artifactsRoot, loadIndex, invalidateCache } = useDataStore();

  // Start file watcher when artifacts root is set (Tauri mode)
  useEffect(() => {
    if (artifactsRoot && platform.isTauri) {
      startWatcher();
    }
  }, [artifactsRoot]);

  // Initialize SSE connection for browser mode
  useEffect(() => {
    if (features.useSSE && !sseRef.current) {
      sseRef.current = createSSEConnection(
        (event: SSEEvent) => {
          console.log('[App SSE]', event.type);
          
          // Handle different event types
          switch (event.type) {
            case 'connected':
              setSseConnected(true);
              break;
            case 'artifact-change':
            case 'cache-invalidated':
              // Refresh data on change
              loadIndex();
              break;
            case 'run-complete':
              // Could trigger notification or auto-navigation
              loadIndex();
              break;
          }
        },
        () => {
          setSseConnected(false);
        }
      );

      return () => {
        if (sseRef.current) {
          sseRef.current.close();
          sseRef.current = null;
        }
      };
    }
  }, []);

  // Listen for navigation events (from Campaigns page)
  useEffect(() => {
    const handleNavigate = (e: CustomEvent<string>) => {
      const validPages = ['cockpit', 'campaigns', 'dashboard', 'evolution', 'candidates', 'strategy', 'backtest', 'risk', 'comparison', 'walkforward', 'montecarlo', 'regimes'];
      if (validPages.includes(e.detail)) {
        setCurrentPage(e.detail as Page);
      }
    };
    
    window.addEventListener('navigate', handleNavigate as EventListener);
    return () => window.removeEventListener('navigate', handleNavigate as EventListener);
  }, []);

  const renderPage = () => {
    switch (currentPage) {
      case 'cockpit':
        return <Cockpit />;
      case 'campaigns':
        return <Campaigns />;
      case 'dashboard':
        return <Dashboard />;
      case 'evolution':
        return <Evolution />;
      case 'candidates':
        return <Candidates />;
      case 'strategy':
        return <StrategyView />;
      case 'backtest':
        return <Backtest />;
      case 'risk':
        return <RiskAnalytics />;
      case 'comparison':
        return <StrategyComparison />;
      case 'walkforward':
        return <WalkForward />;
      case 'montecarlo':
        return <MonteCarlo />;
      case 'regimes':
        return <RegimeAnalysis />;
      default:
        return <Cockpit />;
    }
  };

  return (
    <div className="flex h-screen bg-terminal-bg">
      <Sidebar currentPage={currentPage} onPageChange={setCurrentPage} />
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Browser Mode Banner */}
        {showBrowserBanner && (
          <BrowserModeBanner 
            onDismiss={() => setShowBrowserBanner(false)} 
            sseConnected={sseConnected}
          />
        )}
        <Header />
        <main className="flex-1 overflow-auto p-6 grid-bg">
          {renderPage()}
        </main>
      </div>
      {/* Global Glossary Overlay (activated by '?' key) */}
      <GlossaryOverlay />
    </div>
  );
}

export default App;

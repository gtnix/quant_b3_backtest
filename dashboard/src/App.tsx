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
import { MinerControl } from './pages/MinerControl';
import { HallOfFame } from './pages/HallOfFame';
import { ConfigUniverse } from './pages/ConfigUniverse';
import { ConfigTrading } from './pages/ConfigTrading';
import { ConfigBudget } from './pages/ConfigBudget';
import { ConfigGates } from './pages/ConfigGates';
import { AuditReport } from './pages/AuditReport';
import { GlossaryOverlay } from './components/GlossaryOverlay';
import { useDataStore } from './stores/dataStore';
import { platform, features } from './lib/platform';
import { createSSEConnection, type SSEEvent } from './lib/commands';

export type Page = 
  | 'miner'
  | 'hall-of-fame'
  | 'config-universe'
  | 'config-trading'
  | 'config-budget'
  | 'config-gates'
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
  | 'regimes'
  | 'audit';

function App() {
  // Default to miner control panel
  const [currentPage, setCurrentPage] = useState<Page>('miner');
  const [sseConnected, setSseConnected] = useState(false);
  const sseRef = useRef<EventSource | null>(null);
  const { startWatcher, artifactsRoot, loadIndex } = useDataStore();

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
      const validPages = [
        'miner', 'hall-of-fame', 'config-universe', 'config-trading', 'config-budget', 'config-gates',
        'cockpit', 'campaigns', 'dashboard', 'evolution', 'candidates', 'strategy', 
        'backtest', 'risk', 'comparison', 'walkforward', 'montecarlo', 'regimes', 'audit'
      ];
      if (validPages.includes(e.detail)) {
        setCurrentPage(e.detail as Page);
      }
    };
    
    window.addEventListener('navigate', handleNavigate as EventListener);
    return () => window.removeEventListener('navigate', handleNavigate as EventListener);
  }, []);

  const renderPage = () => {
    switch (currentPage) {
      case 'miner':
        return <MinerControl />;
      case 'hall-of-fame':
        return <HallOfFame />;
      case 'config-universe':
        return <ConfigUniverse />;
      case 'config-trading':
        return <ConfigTrading />;
      case 'config-budget':
        return <ConfigBudget />;
      case 'config-gates':
        return <ConfigGates />;
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
      case 'audit':
        return <AuditReport />;
      default:
        return <MinerControl />;
    }
  };

  return (
    <div className="flex h-screen bg-terminal-bg">
      <Sidebar currentPage={currentPage} onPageChange={setCurrentPage} />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header sseConnected={sseConnected} />
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

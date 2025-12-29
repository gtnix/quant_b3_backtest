import { useState, useEffect } from 'react';
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
import { GlossaryOverlay } from './components/GlossaryOverlay';
import { useDataStore } from './stores/dataStore';

export type Page = 
  | 'cockpit'
  | 'campaigns' 
  | 'dashboard' 
  | 'evolution' 
  | 'candidates' 
  | 'backtest' 
  | 'risk'
  | 'comparison'
  | 'walkforward'
  | 'montecarlo'
  | 'regimes';

function App() {
  const [currentPage, setCurrentPage] = useState<Page>('cockpit');
  const { startWatcher, artifactsRoot } = useDataStore();

  // Start file watcher when artifacts root is set
  useEffect(() => {
    if (artifactsRoot) {
      startWatcher();
    }
  }, [artifactsRoot]);

  // Listen for navigation events (from Campaigns page)
  useEffect(() => {
    const handleNavigate = (e: CustomEvent<string>) => {
      const validPages = ['cockpit', 'campaigns', 'dashboard', 'evolution', 'candidates', 'backtest', 'risk', 'comparison', 'walkforward', 'montecarlo', 'regimes'];
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

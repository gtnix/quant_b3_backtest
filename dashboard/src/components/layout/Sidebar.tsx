import { 
  LayoutDashboard, 
  TrendingUp, 
  Users, 
  LineChart,
  Settings,
  Database,
  FolderKanban,
  Shield,
  GitCompare,
  BarChart3,
  Shuffle,
  Layers,
  Gauge,
  Trophy,
  Globe,
  Cpu,
  DollarSign,
  Activity
} from 'lucide-react';
import type { Page } from '../../App';

interface SidebarProps {
  currentPage: Page;
  onPageChange: (page: Page) => void;
}

const navItems: { id: Page; label: string; icon: React.ElementType; group?: string }[] = [
  // Mining (new primary section)
  { id: 'miner', label: 'Miner Control', icon: Gauge, group: 'Mining' },
  { id: 'hall-of-fame', label: 'Hall of Fame', icon: Trophy, group: 'Mining' },
  
  // Configuration
  { id: 'config-universe', label: 'Universe', icon: Globe, group: 'Config' },
  { id: 'config-trading', label: 'Trading', icon: DollarSign, group: 'Config' },
  { id: 'config-budget', label: 'Budget', icon: Cpu, group: 'Config' },
  { id: 'config-gates', label: 'Gates', icon: Shield, group: 'Config' },
  
  // Analysis
  { id: 'backtest', label: 'Backtest', icon: LineChart, group: 'Analysis' },
  { id: 'campaigns', label: 'Campaigns', icon: FolderKanban, group: 'Analysis' },
  { id: 'candidates', label: 'Candidates', icon: Users, group: 'Analysis' },
  
  // Advanced
  { id: 'walkforward', label: 'Walk-Forward', icon: BarChart3, group: 'Advanced' },
  { id: 'montecarlo', label: 'Monte Carlo', icon: Shuffle, group: 'Advanced' },
  { id: 'regimes', label: 'Regimes', icon: Layers, group: 'Advanced' },
  { id: 'comparison', label: 'Comparison', icon: GitCompare, group: 'Advanced' },
  
  // System
  { id: 'evolution', label: 'Evolution', icon: TrendingUp, group: 'System' },
  { id: 'risk', label: 'Risk Analytics', icon: Activity, group: 'System' },
  { id: 'dashboard', label: 'Overview', icon: LayoutDashboard, group: 'System' },
  
  // Legacy (keep for compatibility)
  { id: 'cockpit', label: 'Cockpit (Legacy)', icon: Gauge, group: 'Legacy' },
];

export function Sidebar({ currentPage, onPageChange }: SidebarProps) {
  // Group items - Mining first
  const groups = ['Mining', 'Config', 'Analysis', 'Advanced', 'System'];
  
  return (
    <aside className="w-64 bg-terminal-surface border-r border-terminal-border flex flex-col">
      {/* Logo */}
      <div className="p-6 border-b border-terminal-border">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center">
            <Database className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="font-bold text-lg tracking-tight text-white">Alpha Forge</h1>
            <p className="text-xs text-terminal-muted">Strategy Miner</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 overflow-y-auto">
        {groups.map(group => {
          const items = navItems.filter(item => item.group === group);
          if (items.length === 0) return null;
          
          return (
            <div key={group} className="mb-4">
              <div className="text-xs font-semibold text-terminal-muted uppercase tracking-wider mb-2 px-4">
                {group}
              </div>
              <ul className="space-y-1">
                {items.map((item) => {
                  const Icon = item.icon;
                  const isActive = currentPage === item.id;
                  
                  // Highlight mining group differently
                  const isMining = group === 'Mining';
                  const activeColor = isMining ? 'amber' : 'profit';
                  
                  return (
                    <li key={item.id}>
                      <button
                        onClick={() => onPageChange(item.id)}
                        className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-lg transition-all duration-200 ${
                          isActive
                            ? isMining 
                              ? 'bg-amber-500/10 text-amber-400 border border-amber-500/30'
                              : 'bg-profit/10 text-profit border border-profit/30'
                            : 'text-terminal-muted hover:text-white hover:bg-terminal-border/50'
                        }`}
                      >
                        <Icon className={`w-4 h-4 ${isActive ? (isMining ? 'text-amber-400' : 'text-profit') : ''}`} />
                        <span className="text-sm font-medium">{item.label}</span>
                        {isActive && (
                          <div className={`ml-auto w-1.5 h-1.5 rounded-full ${isMining ? 'bg-amber-400' : 'bg-profit'} animate-pulse`} />
                        )}
                      </button>
                    </li>
                  );
                })}
              </ul>
            </div>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-terminal-border">
        <button 
          onClick={() => onPageChange('dashboard')}
          className="w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-terminal-muted hover:text-white hover:bg-terminal-border/50 transition-all"
        >
          <Settings className="w-4 h-4" />
          <span className="text-sm font-medium">Settings</span>
        </button>
      </div>
    </aside>
  );
}

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
  Layers
} from 'lucide-react';
import type { Page } from '../../App';

interface SidebarProps {
  currentPage: Page;
  onPageChange: (page: Page) => void;
}

const navItems: { id: Page; label: string; icon: React.ElementType; group?: string }[] = [
  // Core
  { id: 'campaigns', label: 'Campaigns', icon: FolderKanban, group: 'Core' },
  { id: 'candidates', label: 'Candidates', icon: Users, group: 'Core' },
  { id: 'backtest', label: 'Backtest', icon: LineChart, group: 'Core' },
  
  // Analytics
  { id: 'risk', label: 'Risk Analytics', icon: Shield, group: 'Analytics' },
  { id: 'comparison', label: 'Comparison', icon: GitCompare, group: 'Analytics' },
  { id: 'walkforward', label: 'Walk-Forward', icon: BarChart3, group: 'Analytics' },
  { id: 'montecarlo', label: 'Monte Carlo', icon: Shuffle, group: 'Analytics' },
  { id: 'regimes', label: 'Regimes', icon: Layers, group: 'Analytics' },
  
  // System
  { id: 'evolution', label: 'Evolution', icon: TrendingUp, group: 'System' },
  { id: 'dashboard', label: 'Overview', icon: LayoutDashboard, group: 'System' },
];

export function Sidebar({ currentPage, onPageChange }: SidebarProps) {
  // Group items
  const groups = ['Core', 'Analytics', 'System'];
  
  return (
    <aside className="w-64 bg-terminal-surface border-r border-terminal-border flex flex-col">
      {/* Logo */}
      <div className="p-6 border-b border-terminal-border">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-profit to-accent-cyan flex items-center justify-center">
            <Database className="w-6 h-6 text-terminal-bg" />
          </div>
          <div>
            <h1 className="font-bold text-lg tracking-tight">Quant B3</h1>
            <p className="text-xs text-terminal-muted">SCG Dashboard</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 overflow-y-auto">
        {groups.map(group => {
          const items = navItems.filter(item => item.group === group);
          
          return (
            <div key={group} className="mb-4">
              <div className="text-xs font-semibold text-terminal-muted uppercase tracking-wider mb-2 px-4">
                {group}
              </div>
              <ul className="space-y-1">
                {items.map((item) => {
                  const Icon = item.icon;
                  const isActive = currentPage === item.id;
                  
                  return (
                    <li key={item.id}>
                      <button
                        onClick={() => onPageChange(item.id)}
                        className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-lg transition-all duration-200 ${
                          isActive
                            ? 'bg-profit/10 text-profit border border-profit/30'
                            : 'text-terminal-muted hover:text-white hover:bg-terminal-border/50'
                        }`}
                      >
                        <Icon className={`w-4 h-4 ${isActive ? 'text-profit' : ''}`} />
                        <span className="text-sm font-medium">{item.label}</span>
                        {isActive && (
                          <div className="ml-auto w-1.5 h-1.5 rounded-full bg-profit animate-pulse" />
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
        <button className="w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-terminal-muted hover:text-white hover:bg-terminal-border/50 transition-all">
          <Settings className="w-4 h-4" />
          <span className="text-sm font-medium">Settings</span>
        </button>
      </div>
    </aside>
  );
}

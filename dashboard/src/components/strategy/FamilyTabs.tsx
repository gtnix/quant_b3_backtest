import type { StrategyFamily } from '../../stores/strategyStore';
import * as Icons from 'lucide-react';

interface FamilyTabsProps {
  families: StrategyFamily[];
  activeFamily: string | null;
  onSelect: (slug: string | null) => void;
  templateCounts: Record<string, number>;
}

const ICON_MAP: Record<string, keyof typeof Icons> = {
  Clock: 'Clock',
  TrendingUp: 'TrendingUp',
  Target: 'Target',
  GitCompare: 'GitCompare',
  Layers: 'Layers',
  Zap: 'Zap',
  RefreshCw: 'RefreshCw',
  ArrowUpRight: 'ArrowUpRight',
  Shuffle: 'Shuffle',
  BarChart3: 'BarChart3',
  Calendar: 'Calendar',
  Activity: 'Activity',
  Bell: 'Bell',
  Wallet: 'Wallet',
  Boxes: 'Boxes',
};

export function FamilyTabs({ families, activeFamily, onSelect, templateCounts }: FamilyTabsProps) {
  return (
    <div className="flex flex-wrap gap-2">
      {/* All button */}
      <button
        onClick={() => onSelect(null)}
        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2 ${
          activeFamily === null
            ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
            : 'bg-slate-800/50 text-slate-400 border border-slate-700 hover:border-slate-600 hover:text-slate-300'
        }`}
      >
        <Icons.Grid className="w-4 h-4" />
        Todas
        <span className="text-xs opacity-70">
          ({Object.values(templateCounts).reduce((a, b) => a + b, 0)})
        </span>
      </button>

      {/* Family tabs */}
      {families.map((family) => {
        const IconComponent = Icons[ICON_MAP[family.icon] || 'Layers'] as React.ElementType;
        const count = templateCounts[family.slug] || 0;
        const isActive = activeFamily === family.slug;

        return (
          <button
            key={family.slug}
            onClick={() => onSelect(family.slug)}
            className={`px-3 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2 ${
              isActive
                ? 'text-white border'
                : 'bg-slate-800/50 text-slate-400 border border-slate-700 hover:border-slate-600 hover:text-slate-300'
            }`}
            style={isActive ? {
              backgroundColor: `${family.color}20`,
              borderColor: `${family.color}50`,
              color: family.color,
            } : undefined}
            title={family.description}
          >
            <IconComponent className="w-4 h-4" />
            <span className="hidden sm:inline">{family.name}</span>
            <span className="text-xs opacity-70">({count})</span>
          </button>
        );
      })}
    </div>
  );
}





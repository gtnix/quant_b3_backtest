import { Check } from 'lucide-react';
import type { StrategyTemplate, StrategyFamily } from '../../stores/strategyStore';
import { StrategyTooltip } from './StrategyTooltip';

interface StrategyCardProps {
  strategy: StrategyTemplate;
  family?: StrategyFamily;
  selected: boolean;
  onToggle: () => void;
}

const RISK_COLORS: Record<string, string> = {
  conservative: 'bg-green-500/20 border-green-500/30 text-green-400',
  moderate: 'bg-yellow-500/20 border-yellow-500/30 text-yellow-400',
  aggressive: 'bg-orange-500/20 border-orange-500/30 text-orange-400',
  very_aggressive: 'bg-red-500/20 border-red-500/30 text-red-400',
};

const TIMEFRAME_ICONS: Record<string, string> = {
  intraday: '⚡',
  swing: '📈',
  position: '🎯',
  long_term: '🏦',
};

export function StrategyCard({ strategy, family, selected, onToggle }: StrategyCardProps) {
  const riskClass = RISK_COLORS[strategy.risk_profile] || 'bg-slate-700/50 border-slate-600 text-slate-400';
  const familyColor = family?.color || '#6366f1';

  return (
    <button
      onClick={onToggle}
      className={`group relative w-full p-4 rounded-xl border-2 transition-all duration-200 text-left ${
        selected
          ? 'bg-cyan-500/10 border-cyan-500/50 shadow-lg shadow-cyan-500/10'
          : 'bg-slate-900/50 border-slate-800 hover:border-slate-700 hover:bg-slate-800/50'
      }`}
    >
      {/* Selection indicator */}
      <div
        className={`absolute top-3 right-3 w-5 h-5 rounded-full border-2 flex items-center justify-center transition-all ${
          selected
            ? 'bg-cyan-500 border-cyan-500'
            : 'border-slate-600 group-hover:border-slate-500'
        }`}
      >
        {selected && <Check className="w-3 h-3 text-white" />}
      </div>

      {/* Header with family color bar */}
      <div
        className="absolute top-0 left-4 right-4 h-1 rounded-b"
        style={{ backgroundColor: familyColor }}
      />

      {/* Timeframe icon */}
      <div className="text-lg mb-2 mt-1">{TIMEFRAME_ICONS[strategy.timeframe]}</div>

      {/* Strategy name */}
      <h3 className="font-semibold text-white text-sm leading-tight mb-1 pr-6">
        {strategy.name}
      </h3>

      {/* Risk badge */}
      <span className={`inline-block px-2 py-0.5 text-[10px] font-medium rounded-full border ${riskClass}`}>
        {strategy.risk_profile}
      </span>

      {/* Tooltip */}
      <div className="absolute bottom-3 right-3">
        <StrategyTooltip strategy={strategy} />
      </div>

      {/* Difficulty dots */}
      <div className="flex gap-0.5 mt-2">
        {[1, 2, 3, 4, 5].map((level) => (
          <div
            key={level}
            className={`w-1.5 h-1.5 rounded-full ${
              level <= strategy.difficulty_level ? 'bg-cyan-500' : 'bg-slate-700'
            }`}
          />
        ))}
      </div>
    </button>
  );
}





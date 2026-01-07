import { Info } from 'lucide-react';
import { useState } from 'react';
import type { StrategyTemplate } from '../../stores/strategyStore';

interface StrategyTooltipProps {
  strategy: StrategyTemplate;
  className?: string;
}

const RISK_LABELS: Record<string, { label: string; color: string }> = {
  conservative: { label: 'Conservador', color: 'text-green-400' },
  moderate: { label: 'Moderado', color: 'text-yellow-400' },
  aggressive: { label: 'Agressivo', color: 'text-orange-400' },
  very_aggressive: { label: 'Muito Agressivo', color: 'text-red-400' },
};

const TIMEFRAME_LABELS: Record<string, string> = {
  intraday: 'Intraday (1 dia)',
  swing: 'Swing (2-10 dias)',
  position: 'Position (semanas)',
  long_term: 'Longo Prazo (meses)',
};

const DIFFICULTY_LABELS = ['', 'Iniciante', 'Básico', 'Intermediário', 'Avançado', 'Expert'];

export function StrategyTooltip({ strategy, className = '' }: StrategyTooltipProps) {
  const [show, setShow] = useState(false);
  const risk = RISK_LABELS[strategy.risk_profile] || { label: strategy.risk_profile, color: 'text-slate-400' };

  return (
    <div className={`relative inline-block ${className}`}>
      <button
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        onClick={(e) => { e.stopPropagation(); setShow(!show); }}
        className="p-1 hover:bg-slate-700/50 rounded transition-colors"
        aria-label="Mais informações"
      >
        <Info className="w-4 h-4 text-slate-400 hover:text-cyan-400" />
      </button>

      {show && (
        <div className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 w-72 p-4 bg-slate-900 border border-slate-700 rounded-lg shadow-xl animate-in fade-in slide-in-from-bottom-2 duration-200">
          <div className="absolute -bottom-2 left-1/2 -translate-x-1/2 border-8 border-transparent border-t-slate-700" />
          
          <h4 className="font-semibold text-white mb-2">{strategy.name}</h4>
          
          <p className="text-sm text-slate-300 mb-3 leading-relaxed">
            {strategy.tooltip_short}
          </p>
          
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <span className="text-slate-500">Timeframe:</span>
              <span className="ml-1 text-slate-300">{TIMEFRAME_LABELS[strategy.timeframe]}</span>
            </div>
            <div>
              <span className="text-slate-500">Intervalo:</span>
              <span className="ml-1 text-slate-300">{strategy.bar_interval}</span>
            </div>
            <div>
              <span className="text-slate-500">Risco:</span>
              <span className={`ml-1 ${risk.color}`}>{risk.label}</span>
            </div>
            <div>
              <span className="text-slate-500">Dificuldade:</span>
              <span className="ml-1 text-slate-300">{DIFFICULTY_LABELS[strategy.difficulty_level]}</span>
            </div>
          </div>

          {strategy.tags && strategy.tags.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1">
              {strategy.tags.slice(0, 4).map((tag) => (
                <span key={tag} className="px-1.5 py-0.5 bg-slate-800 rounded text-[10px] text-slate-400">
                  {tag}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}





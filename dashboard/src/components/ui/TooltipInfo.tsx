/**
 * TooltipInfo - Informative tooltip component with structured content
 * 
 * Provides context-aware help for UI controls with:
 * - What: One-line description
 * - Impact: How it affects generation/validation
 * - When: When to adjust
 * - Example: Practical example
 */

import React, { useState } from 'react';

// =============================================================================
// TYPES
// =============================================================================

export interface TooltipContent {
  what: string;
  impact: string;
  when: string;
  example: string;
}

interface TooltipInfoProps {
  content: TooltipContent;
  children?: React.ReactNode;
}

interface SimpleTooltipProps {
  text: string;
  children: React.ReactNode;
}

// =============================================================================
// TOOLTIP DATABASE
// =============================================================================

export const TOOLTIPS: Record<string, TooltipContent> = {
  // Compute Budget
  max_runtime: {
    what: 'Tempo máximo que o sistema vai usar para descobrir estratégias',
    impact: 'Mais tempo = mais estratégias avaliadas = maior chance de encontrar boas. Dobrar o tempo pode dobrar o número de candidatos.',
    when: 'Aumente para explorar mais, diminua para testes rápidos',
    example: '15 min é suficiente para exploração inicial. 1h para análise profunda.',
  },
  population_size: {
    what: 'Quantidade de estratégias evoluindo simultaneamente',
    impact: 'População maior = mais diversidade genética = encontra soluções em espaços maiores. Custo: mais memória e CPU.',
    when: 'Aumente se estratégias estão convergindo cedo demais',
    example: '100 para produção, 200 para exploração exaustiva',
  },
  max_generations: {
    what: 'Número máximo de ciclos evolutivos',
    impact: 'Mais gerações = mais refinamento das estratégias. Retornos decrescentes após ~50.',
    when: 'Deixe o default ou aumente se runtime permite',
    example: '50 gerações costuma ser suficiente para convergência',
  },
  workers: {
    what: 'Threads paralelas para avaliação de estratégias',
    impact: 'Mais workers = mais rápido, mas usa mais CPU/memória. Ideal: número de cores físicos.',
    when: 'Reduza se o sistema ficar lento para outras tarefas',
    example: '8 workers em CPU 8-core usa 100% da capacidade',
  },
  seeds: {
    what: 'Sementes para reprodutibilidade dos experimentos',
    impact: 'Múltiplas seeds = resultados mais robustos (menos dependência de sorte). Cada seed é um experimento independente.',
    when: 'Use 3-5 seeds para validação institucional',
    example: '3 seeds = 3 runs independentes, resultado é a média',
  },
  
  // Gates
  min_oos_sharpe: {
    what: 'Sharpe Ratio mínimo no período Out-of-Sample (fora da amostra)',
    impact: 'Gate que filtra estratégias com performance insuficiente. Sharpe OOS < 0.5 geralmente não justifica trading.',
    when: 'Ajuste baseado no benchmark. Mercados mais voláteis podem ter thresholds menores.',
    example: 'Sharpe 0.5 = 50% mais retorno que risco. 1.0 = excelente.',
  },
  max_pbo: {
    what: 'Probabilidade de Backtest Overfitting',
    impact: 'Mede a chance da estratégia ser "sortuda" vs genuinamente boa. PBO alto = provavelmente não funciona no futuro.',
    when: 'Mantenha ≤0.15 para estratégias de produção',
    example: 'PBO 0.08 = 8% chance de ser overfitting. 0.30 = preocupante.',
  },
  min_stress_passed: {
    what: 'Número mínimo de stress tests que a estratégia deve passar',
    impact: 'Testa robustez em cenários históricos extremos (2008, COVID, flash crashes). Estratégias que falham aqui podem quebrar.',
    when: 'Use 4+ para produção. 0 para exploração rápida.',
    example: '4 de 8 testes = estratégia sobrevive a maioria dos crashes',
  },
  stress_testing: {
    what: 'Simula cenários de mercado extremos',
    impact: 'Quando habilitado, cada estratégia é testada contra volatilidade 2x, gaps de preço, drawdowns prolongados.',
    when: 'Sempre habilite para produção. Desabilite apenas para testes rápidos.',
    example: 'Estratégia que passa stress testing sobreviveu a 2008 e COVID',
  },
  
  // Ranking
  ranking_institutional: {
    what: 'Ranking multi-critério ponderado (padrão institucional)',
    impact: 'Pondera Sharpe OOS (40%), PBO (25%), stress tests (20%), gates (15%). Balanceado e robusto.',
    when: 'Use como default para produção',
    example: 'Estratégia com Sharpe 1.2 e PBO 0.05 pontua mais que Sharpe 1.5 e PBO 0.25',
  },
  ranking_pareto: {
    what: 'Fronteira de Pareto (estratégias não-dominadas)',
    impact: 'Mostra estratégias que são ótimas em pelo menos uma dimensão. Nenhuma domina as outras em tudo.',
    when: 'Use quando quiser explorar trade-offs (ex: risco vs retorno)',
    example: '5 estratégias na fronteira = 5 escolhas válidas dependendo da preferência',
  },
  ranking_sharpe: {
    what: 'Ordena apenas por Sharpe Ratio OOS NET',
    impact: 'Simples mas pode premiar overfitting. Ignora PBO e stress.',
    when: 'Use para análise inicial ou quando já validou PBO separadamente',
    example: 'Top 1 por Sharpe pode ter PBO alto - verifique!',
  },
  ranking_riskadjusted: {
    what: 'Sharpe dividido por Drawdown Máximo',
    impact: 'Penaliza estratégias com quedas grandes mesmo que tenham bom Sharpe.',
    when: 'Use se drawdown é prioridade (ex: aversão a perdas)',
    example: 'Sharpe 1.0 com DD 10% > Sharpe 1.5 com DD 30%',
  },
};

// =============================================================================
// COMPONENTS
// =============================================================================

export function TooltipInfo({ content, children }: TooltipInfoProps) {
  const [isOpen, setIsOpen] = useState(false);
  
  return (
    <div className="relative inline-block">
      <button
        type="button"
        className="inline-flex items-center justify-center w-4 h-4 ml-1 text-xs text-cyan-400 hover:text-cyan-300 rounded-full border border-cyan-400/30 hover:border-cyan-400/60 transition-colors"
        onMouseEnter={() => setIsOpen(true)}
        onMouseLeave={() => setIsOpen(false)}
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Mais informações"
      >
        ?
      </button>
      
      {isOpen && (
        <div className="absolute z-50 w-80 p-4 mt-2 left-0 bg-slate-900 border border-cyan-500/30 rounded-lg shadow-xl shadow-cyan-500/10">
          <div className="space-y-3 text-sm">
            <div>
              <span className="text-cyan-400 font-mono text-xs uppercase tracking-wider">O que é</span>
              <p className="text-slate-200 mt-1">{content.what}</p>
            </div>
            
            <div>
              <span className="text-amber-400 font-mono text-xs uppercase tracking-wider">Impacto</span>
              <p className="text-slate-300 mt-1">{content.impact}</p>
            </div>
            
            <div>
              <span className="text-emerald-400 font-mono text-xs uppercase tracking-wider">Quando ajustar</span>
              <p className="text-slate-300 mt-1">{content.when}</p>
            </div>
            
            <div className="pt-2 border-t border-slate-700">
              <span className="text-slate-500 font-mono text-xs uppercase tracking-wider">Exemplo</span>
              <p className="text-slate-400 mt-1 italic">{content.example}</p>
            </div>
          </div>
          
          {/* Arrow */}
          <div className="absolute -top-2 left-4 w-4 h-4 bg-slate-900 border-l border-t border-cyan-500/30 transform rotate-45" />
        </div>
      )}
      
      {children}
    </div>
  );
}

export function SimpleTooltip({ text, children }: SimpleTooltipProps) {
  const [isOpen, setIsOpen] = useState(false);
  
  return (
    <div 
      className="relative inline-block"
      onMouseEnter={() => setIsOpen(true)}
      onMouseLeave={() => setIsOpen(false)}
    >
      {children}
      
      {isOpen && (
        <div className="absolute z-50 px-3 py-2 mt-1 left-1/2 transform -translate-x-1/2 bg-slate-800 border border-slate-600 rounded text-sm text-slate-200 whitespace-nowrap shadow-lg">
          {text}
          <div className="absolute -top-1 left-1/2 transform -translate-x-1/2 w-2 h-2 bg-slate-800 border-l border-t border-slate-600 rotate-45" />
        </div>
      )}
    </div>
  );
}

// =============================================================================
// HELPER COMPONENT
// =============================================================================

interface InfoIconProps {
  tooltipKey: keyof typeof TOOLTIPS;
}

export function InfoIcon({ tooltipKey }: InfoIconProps) {
  const content = TOOLTIPS[tooltipKey];
  if (!content) return null;
  
  return <TooltipInfo content={content} />;
}

export default TooltipInfo;


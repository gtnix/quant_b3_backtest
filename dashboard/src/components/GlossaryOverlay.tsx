/**
 * GlossaryOverlay - Quick reference for quant trading terms
 * 
 * Activated by pressing '?' key anywhere in the app.
 * Provides searchable definitions from docs/reference/glossary.md
 */

import React, { useState, useEffect, useCallback } from 'react';
import { useCockpitStore } from '../stores/cockpitStore';

// =============================================================================
// GLOSSARY DATA (extracted from docs/reference/glossary.md)
// =============================================================================

interface GlossaryTerm {
  term: string;
  definition: string;
  category: 'metrics' | 'validation' | 'execution' | 'data' | 'scg';
}

const GLOSSARY: GlossaryTerm[] = [
  // Metrics
  {
    term: 'Sharpe Ratio',
    definition: 'Retorno excedente dividido pela volatilidade. Mede retorno ajustado ao risco. > 1.0 é excelente, > 0.5 é bom.',
    category: 'metrics',
  },
  {
    term: 'CAGR',
    definition: 'Compound Annual Growth Rate. Taxa de crescimento anualizada composta. Ex: 15% CAGR = dobra em ~5 anos.',
    category: 'metrics',
  },
  {
    term: 'Max Drawdown (MDD)',
    definition: 'Maior queda do pico ao vale no período. Ex: -20% MDD = você teria perdido 20% no pior momento.',
    category: 'metrics',
  },
  {
    term: 'Calmar Ratio',
    definition: 'CAGR dividido pelo Max Drawdown. Mede retorno por unidade de pior perda. > 1.0 é bom.',
    category: 'metrics',
  },
  {
    term: 'Sortino Ratio',
    definition: 'Similar ao Sharpe mas considera apenas volatilidade negativa. Mais relevante para aversão a perdas.',
    category: 'metrics',
  },
  {
    term: 'Win Rate',
    definition: 'Percentual de trades que foram lucrativos. 50% com bom risk/reward pode ser excelente.',
    category: 'metrics',
  },
  {
    term: 'Profit Factor',
    definition: 'Soma dos ganhos dividida pela soma das perdas. > 1.5 é bom, > 2.0 é excelente.',
    category: 'metrics',
  },
  
  // Validation
  {
    term: 'PBO',
    definition: 'Probability of Backtest Overfitting. Probabilidade da estratégia ser "sortuda" vs genuinamente boa. < 0.15 é seguro.',
    category: 'validation',
  },
  {
    term: 'DSR',
    definition: 'Deflated Sharpe Ratio. Sharpe ajustado para múltiplas tentativas. Corrige o viés de seleção.',
    category: 'validation',
  },
  {
    term: 'WFA',
    definition: 'Walk-Forward Analysis. Validação rolling onde treina em janela passada e testa na próxima. Simula produção.',
    category: 'validation',
  },
  {
    term: 'IS / OOS',
    definition: 'In-Sample / Out-of-Sample. IS = dados usados para otimizar. OOS = dados nunca vistos. Performance OOS é a real.',
    category: 'validation',
  },
  {
    term: 'CPCV',
    definition: 'Combinatorial Purged Cross-Validation. Técnica avançada que gera múltiplas combinações IS/OOS para calcular PBO.',
    category: 'validation',
  },
  {
    term: 'Stress Testing',
    definition: 'Simula cenários extremos históricos (2008, COVID, flash crashes) para verificar robustez da estratégia.',
    category: 'validation',
  },
  
  // Execution
  {
    term: 'NET / GROSS',
    definition: 'NET = após custos (fees, slippage). GROSS = antes dos custos. Use sempre NET para decisões.',
    category: 'execution',
  },
  {
    term: 'Slippage',
    definition: 'Diferença entre preço esperado e preço executado. Causado por latência e liquidez.',
    category: 'execution',
  },
  {
    term: 'Delay Bars',
    definition: 'Número de barras de atraso entre sinal e execução. Simula latência real. 1 bar = conservador.',
    category: 'execution',
  },
  {
    term: 'Fill Policy',
    definition: 'Regras para execução de ordens. Inclui participação máxima no volume e ordens parciais.',
    category: 'execution',
  },
  
  // Data
  {
    term: 'Lookahead Bias',
    definition: 'Usar dados do futuro para decisões do passado. Erro fatal que invalida backtest.',
    category: 'data',
  },
  {
    term: 'Survivorship Bias',
    definition: 'Testar apenas em ativos que sobreviveram. Ignora falências/delisting. Infla resultados.',
    category: 'data',
  },
  {
    term: 'Corporate Actions',
    definition: 'Eventos como splits, dividendos, bonificações. Preços devem ser ajustados para comparação correta.',
    category: 'data',
  },
  {
    term: 'Adjusted Price',
    definition: 'Preço corrigido por corporate actions. Usado para sinais. Preço bruto usado para valuation.',
    category: 'data',
  },
  
  // SCG
  {
    term: 'SCG',
    definition: 'Strategy Combiner Generative. Sistema evolutivo que descobre e valida estratégias automaticamente.',
    category: 'scg',
  },
  {
    term: 'Genoma',
    definition: 'Representação da estratégia como genes. Inclui blocos de seleção, entrada, saída, sizing e risco.',
    category: 'scg',
  },
  {
    term: 'Pareto Front',
    definition: 'Conjunto de estratégias não-dominadas. Nenhuma é melhor que outra em todas as dimensões.',
    category: 'scg',
  },
  {
    term: 'Hall of Fame',
    definition: 'Top estratégias de todas as gerações. Preserva as melhores mesmo se população evolui para longe.',
    category: 'scg',
  },
  {
    term: 'Gates',
    definition: 'Thresholds mínimos para promoção de estratégias. Ex: Sharpe > 0.5, PBO < 0.15.',
    category: 'scg',
  },
];

// =============================================================================
// COMPONENT
// =============================================================================

export function GlossaryOverlay() {
  const { isGlossaryOpen, toggleGlossary } = useCockpitStore();
  const [search, setSearch] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  
  // Keyboard handler
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === '?' && !e.ctrlKey && !e.metaKey) {
      // Only trigger if not in an input
      const target = e.target as HTMLElement;
      if (target.tagName !== 'INPUT' && target.tagName !== 'TEXTAREA') {
        e.preventDefault();
        toggleGlossary();
      }
    }
    if (e.key === 'Escape' && isGlossaryOpen) {
      toggleGlossary();
    }
  }, [isGlossaryOpen, toggleGlossary]);
  
  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);
  
  if (!isGlossaryOpen) return null;
  
  // Filter terms
  const filteredTerms = GLOSSARY.filter((term) => {
    const matchesSearch = search === '' || 
      term.term.toLowerCase().includes(search.toLowerCase()) ||
      term.definition.toLowerCase().includes(search.toLowerCase());
    const matchesCategory = selectedCategory === null || term.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });
  
  const categories = [
    { key: 'metrics', label: 'Métricas', icon: '📊' },
    { key: 'validation', label: 'Validação', icon: '✓' },
    { key: 'execution', label: 'Execução', icon: '⚡' },
    { key: 'data', label: 'Dados', icon: '💾' },
    { key: 'scg', label: 'SCG', icon: '🧬' },
  ];
  
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
      <div className="w-full max-w-3xl max-h-[80vh] bg-slate-900 border border-cyan-500/30 rounded-lg shadow-2xl shadow-cyan-500/20 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-700">
          <div className="flex items-center gap-3">
            <span className="text-2xl">📖</span>
            <div>
              <h2 className="text-lg font-semibold text-slate-100">Glossário Quant</h2>
              <p className="text-sm text-slate-400">Pressione <kbd className="px-1.5 py-0.5 bg-slate-700 rounded text-xs">?</kbd> para abrir/fechar</p>
            </div>
          </div>
          <button
            onClick={toggleGlossary}
            className="p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        
        {/* Search & Categories */}
        <div className="px-6 py-4 border-b border-slate-800">
          <input
            type="text"
            placeholder="Buscar termo..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-cyan-500/50"
            autoFocus
          />
          
          <div className="flex gap-2 mt-3">
            <button
              onClick={() => setSelectedCategory(null)}
              className={`px-3 py-1.5 text-sm rounded-full transition-colors ${
                selectedCategory === null
                  ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                  : 'text-slate-400 hover:text-slate-300 hover:bg-slate-800'
              }`}
            >
              Todos
            </button>
            {categories.map((cat) => (
              <button
                key={cat.key}
                onClick={() => setSelectedCategory(cat.key === selectedCategory ? null : cat.key)}
                className={`px-3 py-1.5 text-sm rounded-full transition-colors ${
                  selectedCategory === cat.key
                    ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                    : 'text-slate-400 hover:text-slate-300 hover:bg-slate-800'
                }`}
              >
                {cat.icon} {cat.label}
              </button>
            ))}
          </div>
        </div>
        
        {/* Terms List */}
        <div className="overflow-y-auto max-h-[50vh] p-6">
          <div className="space-y-4">
            {filteredTerms.map((term) => (
              <div
                key={term.term}
                className="p-4 bg-slate-800/50 border border-slate-700/50 rounded-lg hover:border-cyan-500/30 transition-colors"
              >
                <div className="flex items-start justify-between">
                  <h3 className="text-base font-medium text-cyan-400">{term.term}</h3>
                  <span className="px-2 py-0.5 text-xs bg-slate-700 text-slate-400 rounded">
                    {categories.find((c) => c.key === term.category)?.label}
                  </span>
                </div>
                <p className="mt-2 text-sm text-slate-300 leading-relaxed">{term.definition}</p>
              </div>
            ))}
            
            {filteredTerms.length === 0 && (
              <div className="text-center py-8 text-slate-500">
                Nenhum termo encontrado para "{search}"
              </div>
            )}
          </div>
        </div>
        
        {/* Footer */}
        <div className="px-6 py-3 border-t border-slate-800 bg-slate-900/50">
          <p className="text-xs text-slate-500 text-center">
            Baseado em docs/reference/glossary.md · {filteredTerms.length} termos
          </p>
        </div>
      </div>
    </div>
  );
}

export default GlossaryOverlay;










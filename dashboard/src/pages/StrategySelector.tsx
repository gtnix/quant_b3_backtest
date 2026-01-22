import { useEffect, useMemo, useState } from 'react';
import { Search, X, CheckCircle2, Trash2, ChevronDown, ChevronRight, Info, Eye, EyeOff } from 'lucide-react';
import { useStrategyStore } from '../stores/strategyStore';
import { StrategyCard } from '../components/strategy/StrategyCard';
import { FamilyTabs } from '../components/strategy/FamilyTabs';
import { CatalogDropdown } from '../components/strategy/CatalogDropdown';

const TIMEFRAME_OPTIONS = [
  { value: '', label: 'Todos Timeframes' },
  { value: 'intraday', label: 'Intraday (1 dia)' },
  { value: 'swing', label: 'Swing (2-10 dias)' },
  { value: 'position', label: 'Position (semanas)' },
  { value: 'long_term', label: 'Longo Prazo (meses)' },
];

const RISK_OPTIONS = [
  { value: '', label: 'Todos Riscos' },
  { value: 'conservative', label: '🟢 Conservador' },
  { value: 'moderate', label: '🟡 Moderado' },
  { value: 'aggressive', label: '🟠 Agressivo' },
  { value: 'very_aggressive', label: '🔴 Muito Agressivo' },
];

export function StrategySelector() {
  const {
    families,
    templates,
    catalogs,
    selectedStrategies,
    activeFamily,
    filters,
    loading,
    fetchAll,
    toggleStrategy,
    selectFamily,
    setFilter,
    clearSelection,
    selectCatalog,
    getFilteredTemplates,
    getFamilyBySlug,
  } = useStrategyStore();

  // Collapsible state - all collapsed by default
  const [collapsedFamilies, setCollapsedFamilies] = useState<Set<string>>(new Set());
  const [showOnlySelected, setShowOnlySelected] = useState(false);
  const [initialized, setInitialized] = useState(false);
  
  // Toggle family collapse
  const toggleFamilyCollapse = (slug: string) => {
    setCollapsedFamilies(prev => {
      const next = new Set(prev);
      if (next.has(slug)) {
        next.delete(slug);
      } else {
        next.add(slug);
      }
      return next;
    });
  };

  useEffect(() => {
    fetchAll();
  }, [fetchAll]);
  
  // Collapse all families by default after data loads
  useEffect(() => {
    if (families.length > 0 && !initialized) {
      setCollapsedFamilies(new Set(families.map(f => f.slug)));
      setInitialized(true);
    }
  }, [families, initialized]);

  const filteredTemplates = useMemo(() => getFilteredTemplates(), [
    templates,
    activeFamily,
    filters,
  ]);

  const templateCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    families.forEach((f) => {
      counts[f.slug] = templates.filter((t) => t.family_id === f.id).length;
    });
    return counts;
  }, [families, templates]);

  const groupedTemplates = useMemo(() => {
    if (activeFamily) {
      return { [activeFamily]: filteredTemplates };
    }
    const groups: Record<string, typeof filteredTemplates> = {};
    filteredTemplates.forEach((t) => {
      const family = families.find((f) => f.id === t.family_id);
      if (family) {
        if (!groups[family.slug]) groups[family.slug] = [];
        groups[family.slug].push(t);
      }
    });
    return groups;
  }, [filteredTemplates, families, activeFamily]);

  const sortedFamilySlugs = useMemo(() => {
    return Object.keys(groupedTemplates).sort((a, b) => {
      const aFamily = families.find((f) => f.slug === a);
      const bFamily = families.find((f) => f.slug === b);
      return (aFamily?.sort_order || 0) - (bFamily?.sort_order || 0);
    });
  }, [groupedTemplates, families]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-3">
            📦 Catálogo de Estratégias
            <span className="text-sm font-normal px-2 py-1 bg-cyan-500/20 text-cyan-400 rounded">
              {templates.length} disponíveis
            </span>
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Selecione as estratégias para o algoritmo genético explorar
          </p>
        </div>

        <div className="flex items-center gap-3">
          <CatalogDropdown
            catalogs={catalogs}
            activeCatalog={null}
            onSelect={selectCatalog}
          />
        </div>
      </div>

      {/* Filters bar */}
      <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
        <div className="flex flex-wrap items-center gap-4">
          {/* Search */}
          <div className="relative flex-1 min-w-[200px]">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
            <input
              type="text"
              placeholder="Buscar estratégias..."
              value={filters.search}
              onChange={(e) => setFilter('search', e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
            />
            {filters.search && (
              <button
                onClick={() => setFilter('search', '')}
                className="absolute right-3 top-1/2 -translate-y-1/2"
              >
                <X className="w-4 h-4 text-slate-500 hover:text-white" />
              </button>
            )}
          </div>

          {/* Timeframe filter */}
          <select
            value={filters.timeframe || ''}
            onChange={(e) => setFilter('timeframe', e.target.value || null)}
            className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:outline-none focus:border-cyan-500"
          >
            {TIMEFRAME_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>

          {/* Risk filter */}
          <select
            value={filters.riskProfile || ''}
            onChange={(e) => setFilter('riskProfile', e.target.value || null)}
            className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:outline-none focus:border-cyan-500"
          >
            {RISK_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
          
          {/* Show only selected toggle */}
          <button
            onClick={() => setShowOnlySelected(!showOnlySelected)}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
              showOnlySelected 
                ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50' 
                : 'bg-slate-800 text-slate-400 border border-slate-700 hover:text-white'
            }`}
          >
            {showOnlySelected ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
            {showOnlySelected ? 'Selecionadas' : 'Todas'}
          </button>
        </div>

        {/* Family tabs */}
        <div className="mt-4">
          <FamilyTabs
            families={families}
            activeFamily={activeFamily}
            onSelect={selectFamily}
            templateCounts={templateCounts}
          />
        </div>
      </div>

      {/* Info tooltip */}
      <div className="flex items-start gap-3 p-4 bg-slate-800/30 border border-slate-700/50 rounded-lg">
        <Info className="w-5 h-5 text-cyan-400 flex-shrink-0 mt-0.5" />
        <div className="text-sm text-slate-400">
          <strong className="text-white">Dica:</strong> Selecione as estratégias que deseja incluir na geração do algoritmo genético. 
          O gerador irá explorar variações dentro das estratégias selecionadas. 
          Passe o mouse sobre o ícone <span className="text-cyan-400">ⓘ</span> de cada estratégia para ver explicações detalhadas.
        </div>
      </div>

      {/* Strategy grid by family */}
      {loading ? (
        <div className="flex items-center justify-center py-20">
          <div className="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
        </div>
      ) : (
        <div className="space-y-3">
          {sortedFamilySlugs.map((familySlug) => {
            const family = getFamilyBySlug(familySlug);
            let strategies = groupedTemplates[familySlug];
            if (!family || !strategies?.length) return null;

            // Filter to show only selected if enabled
            if (showOnlySelected) {
              strategies = strategies.filter(s => selectedStrategies.includes(s.slug));
              if (strategies.length === 0) return null;
            }

            const selectedInFamily = strategies.filter((s) =>
              selectedStrategies.includes(s.slug)
            ).length;
            
            const totalInFamily = groupedTemplates[familySlug]?.length || 0;
            const isCollapsed = collapsedFamilies.has(familySlug);

            return (
              <div key={familySlug} className="rounded-xl border border-slate-700 bg-slate-800/30 overflow-hidden">
                {/* Collapsible Family header */}
                <button
                  onClick={() => toggleFamilyCollapse(familySlug)}
                  className="w-full flex items-center justify-between p-4 hover:bg-slate-700/30 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    {isCollapsed ? (
                      <ChevronRight className="w-5 h-5 text-slate-500" />
                    ) : (
                      <ChevronDown className="w-5 h-5 text-slate-500" />
                    )}
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: family.color }}
                    />
                    <h2 className="text-lg font-semibold text-white">{family.name}</h2>
                    <span className="text-sm text-slate-500">
                      ({totalInFamily})
                    </span>
                    {selectedInFamily > 0 && (
                      <span className="px-2 py-0.5 bg-cyan-500/20 text-cyan-400 text-xs rounded-full font-medium">
                        {selectedInFamily} selecionadas
                      </span>
                    )}
                  </div>
                  <div 
                    onClick={(e) => {
                      e.stopPropagation();
                      strategies.forEach((s) => {
                        if (!selectedStrategies.includes(s.slug)) {
                          toggleStrategy(s.slug);
                        }
                      });
                    }}
                    className="text-xs text-slate-400 hover:text-cyan-400 transition-colors px-3 py-1.5 hover:bg-slate-700 rounded"
                  >
                    Selecionar todas
                  </div>
                </button>

                {/* Strategy cards grid - collapsible */}
                {!isCollapsed && (
                  <div className="p-4 pt-0 border-t border-slate-700/50">
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3 mt-4">
                      {strategies.map((strategy) => (
                        <StrategyCard
                          key={strategy.slug}
                          strategy={strategy}
                          family={family}
                          selected={selectedStrategies.includes(strategy.slug)}
                          onToggle={() => toggleStrategy(strategy.slug)}
                        />
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Selection summary bar - fixed at bottom */}
      {selectedStrategies.length > 0 && (
        <div className="fixed bottom-0 left-64 right-0 bg-slate-900/95 backdrop-blur-sm border-t border-slate-800 p-4 z-40">
          <div className="flex items-center justify-between max-w-7xl mx-auto">
            <div className="flex items-center gap-4">
              <CheckCircle2 className="w-5 h-5 text-cyan-400" />
              <div>
                <span className="text-white font-semibold">{selectedStrategies.length}</span>
                <span className="text-slate-400 ml-1">estratégias selecionadas</span>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={clearSelection}
                className="flex items-center gap-2 px-4 py-2 text-sm text-slate-400 hover:text-red-400 transition-colors"
              >
                <Trash2 className="w-4 h-4" />
                Limpar
              </button>
              <button
                onClick={() => {
                  window.dispatchEvent(new CustomEvent('navigate', { detail: 'miner' }));
                }}
                className="px-6 py-2 bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-semibold rounded-lg hover:from-cyan-400 hover:to-blue-400 transition-all shadow-lg shadow-cyan-500/20"
              >
                Aplicar e Minerar →
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Bottom padding for fixed bar */}
      {selectedStrategies.length > 0 && <div className="h-20" />}
    </div>
  );
}






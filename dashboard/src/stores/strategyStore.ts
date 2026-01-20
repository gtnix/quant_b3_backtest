import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface StrategyFamily {
  id: number;
  slug: string;
  name: string;
  description: string;
  icon: string;
  color: string;
  hypothesis: string;
  sort_order: number;
}

export interface StrategyTemplate {
  id: number;
  slug: string;
  family_id: number;
  name: string;
  description: string;
  timeframe: 'intraday' | 'swing' | 'position' | 'long_term';
  bar_interval: string;
  position_type: 'directional' | 'pair' | 'portfolio' | 'multi_strategy';
  risk_profile: 'conservative' | 'moderate' | 'aggressive' | 'very_aggressive';
  tooltip_short: string;
  tooltip_long?: string;
  difficulty_level: number;
  tags: string[];
  enabled: boolean;
  is_default: boolean;
  usage_count: number;
}

export interface StrategyCatalog {
  id: number;
  slug: string;
  name: string;
  description: string;
  icon: string;
  is_system: boolean;
  is_default: boolean;
}

interface StrategyFilters {
  timeframe: string | null;
  riskProfile: string | null;
  family: string | null;
  search: string;
}

interface StrategyState {
  families: StrategyFamily[];
  templates: StrategyTemplate[];
  catalogs: StrategyCatalog[];
  
  selectedStrategies: string[];
  activeCatalog: string | null;
  activeFamily: string | null;
  
  filters: StrategyFilters;
  
  loading: boolean;
  error: string | null;
  
  // Actions
  fetchFamilies: () => Promise<void>;
  fetchTemplates: () => Promise<void>;
  fetchCatalogs: () => Promise<void>;
  fetchAll: () => Promise<void>;
  
  selectStrategy: (slug: string) => void;
  deselectStrategy: (slug: string) => void;
  toggleStrategy: (slug: string) => void;
  selectCatalog: (slug: string) => void;
  selectFamily: (slug: string | null) => void;
  
  setFilter: (key: keyof StrategyFilters, value: string | null) => void;
  clearFilters: () => void;
  clearSelection: () => void;
  selectAll: () => void;
  
  getFilteredTemplates: () => StrategyTemplate[];
  getTemplatesByFamily: (familySlug: string) => StrategyTemplate[];
  getFamilyBySlug: (slug: string) => StrategyFamily | undefined;
  getSelectedCount: () => number;
}

// Always use explicit localhost:3001 for API calls (works in Tauri and dev)
const API_BASE = 'http://localhost:3001';

export const useStrategyStore = create<StrategyState>()(
  persist(
    (set, get) => ({
      families: [],
      templates: [],
      catalogs: [],
      selectedStrategies: [],
      activeCatalog: 'all',
      activeFamily: null,
      filters: {
        timeframe: null,
        riskProfile: null,
        family: null,
        search: '',
      },
      loading: false,
      error: null,

      fetchFamilies: async () => {
        try {
          const res = await fetch(`${API_BASE}/api/strategies/families`);
          if (!res.ok) throw new Error('Failed to fetch families');
          const data = await res.json();
          set({ families: data });
        } catch (e) {
          set({ error: (e as Error).message });
        }
      },

      fetchTemplates: async () => {
        try {
          const res = await fetch(`${API_BASE}/api/strategies`);
          if (!res.ok) throw new Error('Failed to fetch templates');
          const data = await res.json();
          set({ templates: data });
        } catch (e) {
          set({ error: (e as Error).message });
        }
      },

      fetchCatalogs: async () => {
        try {
          const res = await fetch(`${API_BASE}/api/catalogs`);
          if (!res.ok) throw new Error('Failed to fetch catalogs');
          const data = await res.json();
          set({ catalogs: data });
        } catch (e) {
          set({ error: (e as Error).message });
        }
      },

      fetchAll: async () => {
        set({ loading: true, error: null });
        try {
          await Promise.all([
            get().fetchFamilies(),
            get().fetchTemplates(),
            get().fetchCatalogs(),
          ]);
          // Auto-select all strategies on first load (if none selected)
          const { selectedStrategies, templates } = get();
          if (selectedStrategies.length === 0 && templates.length > 0) {
            set({ selectedStrategies: templates.map(t => t.slug) });
          }
        } finally {
          set({ loading: false });
        }
      },

      selectStrategy: (slug) => {
        set((state) => ({
          selectedStrategies: state.selectedStrategies.includes(slug)
            ? state.selectedStrategies
            : [...state.selectedStrategies, slug],
        }));
      },

      deselectStrategy: (slug) => {
        set((state) => ({
          selectedStrategies: state.selectedStrategies.filter((s) => s !== slug),
        }));
      },

      toggleStrategy: (slug) => {
        const { selectedStrategies } = get();
        if (selectedStrategies.includes(slug)) {
          get().deselectStrategy(slug);
        } else {
          get().selectStrategy(slug);
        }
      },

      selectCatalog: async (slug) => {
        set({ activeCatalog: slug, loading: true });
        try {
          const res = await fetch(`${API_BASE}/api/catalogs/${slug}/strategies`);
          if (!res.ok) throw new Error('Failed to fetch catalog strategies');
          const data = await res.json();
          set({ selectedStrategies: data.map((s: StrategyTemplate) => s.slug) });
        } catch (e) {
          set({ error: (e as Error).message });
        } finally {
          set({ loading: false });
        }
      },

      selectFamily: (slug) => {
        set({ activeFamily: slug });
      },

      setFilter: (key, value) => {
        set((state) => ({
          filters: { ...state.filters, [key]: value },
        }));
      },

      clearFilters: () => {
        set({
          filters: { timeframe: null, riskProfile: null, family: null, search: '' },
        });
      },

      clearSelection: () => {
        set({ selectedStrategies: [] });
      },

      selectAll: () => {
        const filtered = get().getFilteredTemplates();
        set({ selectedStrategies: filtered.map((t) => t.slug) });
      },

      getFilteredTemplates: () => {
        const { templates, filters, activeFamily } = get();
        return templates.filter((t) => {
          if (activeFamily && get().families.find(f => f.slug === activeFamily)?.id !== t.family_id) {
            return false;
          }
          if (filters.timeframe && t.timeframe !== filters.timeframe) return false;
          if (filters.riskProfile && t.risk_profile !== filters.riskProfile) return false;
          if (filters.family) {
            const family = get().families.find(f => f.slug === filters.family);
            if (family && t.family_id !== family.id) return false;
          }
          if (filters.search) {
            const search = filters.search.toLowerCase();
            return (
              t.name.toLowerCase().includes(search) ||
              t.slug.toLowerCase().includes(search) ||
              t.tooltip_short.toLowerCase().includes(search) ||
              t.tags?.some((tag) => tag.toLowerCase().includes(search))
            );
          }
          return true;
        });
      },

      getTemplatesByFamily: (familySlug) => {
        const { templates, families } = get();
        const family = families.find((f) => f.slug === familySlug);
        if (!family) return [];
        return templates.filter((t) => t.family_id === family.id);
      },

      getFamilyBySlug: (slug) => {
        return get().families.find((f) => f.slug === slug);
      },

      getSelectedCount: () => {
        return get().selectedStrategies.length;
      },
    }),
    {
      name: 'strategy-store',
      partialize: (state) => ({
        selectedStrategies: state.selectedStrategies,
        activeCatalog: state.activeCatalog,
        activeFamily: state.activeFamily,
      }),
    }
  )
);


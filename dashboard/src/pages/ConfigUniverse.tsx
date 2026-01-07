/**
 * ConfigUniverse - Parameter Universe System Configuration
 * 
 * Configure the 4 axes that control strategy generation:
 * - Robustness Profile
 * - Training Strategy
 * - Training Tech
 * - Training Model (Strategy Family)
 */

import { useEffect, useState, useMemo } from 'react';
import { 
  Globe, Save, RefreshCw, Shield, Cpu, BookOpen, 
  TrendingUp, AlertCircle, CheckCircle2, Info,
  ChevronDown, ChevronUp, Clock, Zap, List
} from 'lucide-react';
import { useOmpStore } from '../stores/ompStore';

// =============================================================================
// TYPES
// =============================================================================

interface MarketConfig {
  enabled: boolean;
  name: string;
  universe: string;
  symbols?: string[];
  calendar: string;
  currency: string;
  lot_size: number;
}

interface UniverseAxisConfig {
  robustnessProfile: string;
  trainingStrategy: string;
  trainingTech: string;
  trainingModel: string[];
  timeframeProfile: string;
}

interface StrategyRegistryEntry {
  family: string;
  variant: string;
  timeframe: string;
  hypothesis: string;
  complexity_tier: string;
  risk_profiles: string[];
}

interface StrategyRegistry {
  [key: string]: StrategyRegistryEntry;
}

interface TimeframeProfileData {
  label: string;
  holding_period: string;
  families: string[];
  strategies: string[];
}

// =============================================================================
// CONSTANTS
// =============================================================================

const ROBUSTNESS_PROFILES = {
  muito_conservador: {
    label: 'Muito Conservador',
    description: 'Capital preservation, minimal drawdowns',
    maxDD: '8%',
    kelly: '0.15',
    color: 'emerald',
  },
  conservador: {
    label: 'Conservador',
    description: 'Consistent growth, controlled drawdowns',
    maxDD: '12%',
    kelly: '0.30',
    color: 'green',
  },
  moderado: {
    label: 'Moderado',
    description: 'Balanced growth with acceptable volatility',
    maxDD: '20%',
    kelly: '0.40',
    color: 'blue',
  },
  arrojado: {
    label: 'Arrojado',
    description: 'Aggressive growth, high risk tolerance',
    maxDD: '25%',
    kelly: '0.50',
    color: 'orange',
  },
  muito_arrojado: {
    label: 'Muito Arrojado',
    description: 'Maximum risk within defensible bounds',
    maxDD: '30%',
    kelly: '0.50',
    color: 'rose',
  },
};

const TRAINING_STRATEGIES = {
  purged_kfold: {
    label: 'Purged K-Fold',
    description: 'Standard robust validation with embargo periods',
    folds: 5,
    tier: 'tier2_medium',
  },
  walk_forward: {
    label: 'Walk-Forward',
    description: 'Rolling window validation for time-series',
    folds: 8,
    tier: 'tier2_medium',
  },
  anchored: {
    label: 'Anchored',
    description: 'Fixed anchor with expanding window',
    folds: null,
    tier: 'tier1_fast',
  },
  expanding_window: {
    label: 'Expanding Window',
    description: 'Incrementally expanding training set',
    folds: 6,
    tier: 'tier2_medium',
  },
  monte_carlo: {
    label: 'Monte Carlo',
    description: 'Extensive stress testing for production',
    folds: 10,
    tier: 'tier3_slow',
  },
};

const TRAINING_TECH = {
  cpu_fast: {
    label: 'CPU Fast',
    description: 'Quick exploration for development',
    workers: 4,
    timeout: '30min',
    icon: '⚡',
  },
  cpu_parallel: {
    label: 'CPU Parallel',
    description: 'Standard production configuration',
    workers: 8,
    timeout: '2h',
    icon: '🔄',
  },
  cpu_intensive: {
    label: 'CPU Intensive',
    description: 'Heavy computation for complex strategies',
    workers: 16,
    timeout: '6h',
    icon: '💪',
  },
  distributed: {
    label: 'Distributed',
    description: 'Cluster-level for institutional strategies',
    workers: 'auto',
    timeout: '24h',
    icon: '🌐',
  },
};

const STRATEGY_FAMILIES = {
  intraday: { label: 'Intraday', tier: 'tier1_fast', holding: '1-8h' },
  swing: { label: 'Swing Trading', tier: 'tier1_fast', holding: '2-10 days' },
  position: { label: 'Position Trading', tier: 'tier2_medium', holding: 'Weeks-Months' },
  pair: { label: 'Pair Trading', tier: 'tier2_medium', holding: '5-30 days' },
  portfolio: { label: 'Portfolio Trading', tier: 'tier3_slow', holding: 'Rebalance' },
  momentum: { label: 'Momentum', tier: 'tier1_fast', holding: '1-6 months' },
  mean_reversion: { label: 'Mean Reversion', tier: 'tier1_fast', holding: '2-10 days' },
  breakout: { label: 'Breakout', tier: 'tier1_fast', holding: '3-15 days' },
  sector_rotation: { label: 'Sector Rotation', tier: 'tier2_medium', holding: '1-6 months' },
  factor: { label: 'Factor Investing', tier: 'tier3_slow', holding: 'Months-Years' },
  seasonal: { label: 'Seasonal Trading', tier: 'tier2_medium', holding: 'Days-Weeks' },
  volatility: { label: 'Volatility Trading', tier: 'tier2_medium', holding: '3-15 days' },
  event_driven: { label: 'Event-Driven', tier: 'tier2_medium', holding: '1-5 days' },
  buy_hold: { label: 'Buy & Hold', tier: 'tier1_fast', holding: 'Years' },
  multi_strategy: { label: 'Multi-Strategy', tier: 'tier3_slow', holding: 'Variable' },
};

// Default compatibility (used as fallback)
const DEFAULT_COMPATIBILITY = {
  robustnessToFamilies: {
    muito_conservador: ['position', 'portfolio', 'factor', 'seasonal', 'buy_hold'],
    conservador: ['swing', 'position', 'pair', 'portfolio', 'momentum', 'mean_reversion', 'sector_rotation', 'factor', 'seasonal', 'buy_hold', 'multi_strategy'],
    moderado: ['swing', 'position', 'pair', 'portfolio', 'momentum', 'mean_reversion', 'breakout', 'sector_rotation', 'factor', 'volatility', 'event_driven', 'buy_hold', 'multi_strategy'],
    arrojado: ['intraday', 'swing', 'position', 'pair', 'portfolio', 'momentum', 'mean_reversion', 'breakout', 'volatility', 'event_driven', 'multi_strategy'],
    muito_arrojado: ['intraday', 'swing', 'momentum', 'breakout', 'volatility', 'event_driven'],
  },
  robustnessToStrategies: {
    muito_conservador: ['purged_kfold', 'walk_forward'],
    conservador: ['purged_kfold', 'walk_forward', 'anchored'],
    moderado: ['purged_kfold', 'walk_forward', 'anchored', 'expanding_window'],
    arrojado: ['purged_kfold', 'walk_forward', 'anchored', 'expanding_window', 'monte_carlo'],
    muito_arrojado: ['purged_kfold', 'walk_forward', 'anchored', 'expanding_window', 'monte_carlo'],
  },
  techToTiers: {
    cpu_fast: ['tier1_fast'],
    cpu_parallel: ['tier1_fast', 'tier2_medium'],
    cpu_intensive: ['tier1_fast', 'tier2_medium', 'tier3_slow'],
    distributed: ['tier1_fast', 'tier2_medium', 'tier3_slow', 'tier4_very_slow'],
  },
};

interface CompatibilityMatrix {
  robustnessToFamilies: Record<string, string[]>;
  robustnessToStrategies: Record<string, string[]>;
  techToTiers: Record<string, string[]>;
  source?: string;
}

const DEFAULT_MARKETS: Record<string, MarketConfig> = {
  br: {
    enabled: true,
    name: 'B3 - Brasil',
    universe: 'ibov',
    calendar: 'b3',
    currency: 'BRL',
    lot_size: 100,
  },
  us: {
    enabled: true,
    name: 'US Equities',
    universe: 'sp500',
    calendar: 'nyse',
    currency: 'USD',
    lot_size: 1,
  },
};

const PRESET_UNIVERSES: Record<string, { label: string; description: string; symbols?: number }> = {
  ibov: { label: 'IBOV', description: 'Índice Bovespa', symbols: 90 },
  ibrx100: { label: 'IBrX 100', description: 'Brasil 100 Index', symbols: 100 },
  small: { label: 'SMLL', description: 'Small Caps Index', symbols: 100 },
  sp500: { label: 'S&P 500', description: 'Standard & Poor\'s 500', symbols: 500 },
  nasdaq100: { label: 'NASDAQ 100', description: 'NASDAQ 100 Index', symbols: 100 },
  djia: { label: 'DJIA', description: 'Dow Jones Industrial', symbols: 30 },
};

// =============================================================================
// COMPONENT
// =============================================================================

// Timeframe profile options
const TIMEFRAME_PROFILES = {
  intraday: { label: 'Intraday', holding: '1-8h', icon: '⚡' },
  swing: { label: 'Swing', holding: '2-10 days', icon: '📈' },
  position: { label: 'Position', holding: '2-12 weeks', icon: '🎯' },
  long_term: { label: 'Long-Term', holding: '3+ months', icon: '🏦' },
  adaptive: { label: 'Adaptive', holding: 'Auto-detect', icon: '🔄' },
};

export function ConfigUniverse() {
  const { config, fetchConfig } = useOmpStore();
  
  // Universe axes state
  const [universeAxes, setUniverseAxes] = useState<UniverseAxisConfig>({
    robustnessProfile: 'moderado',
    trainingStrategy: 'purged_kfold',
    trainingTech: 'cpu_parallel',
    trainingModel: ['swing', 'momentum'],
    timeframeProfile: 'swing',
  });
  
  // Markets state
  const [markets, setMarkets] = useState<Record<string, MarketConfig>>(DEFAULT_MARKETS);
  const [saving, setSaving] = useState(false);
  const [expandedSections, setExpandedSections] = useState({
    axes: true,
    markets: false,
    strategies: false,
  });
  
  // Compatibility matrix loaded from backend
  const [compatibility, setCompatibility] = useState<CompatibilityMatrix>(DEFAULT_COMPATIBILITY);
  const [matrixSource, setMatrixSource] = useState<string>('default');
  
  // Strategy registry loaded from backend
  const [strategyRegistry, setStrategyRegistry] = useState<StrategyRegistry>({});
  const [registryLoaded, setRegistryLoaded] = useState(false);

  // Fetch compatibility matrix from backend
  useEffect(() => {
    const fetchCompatibilityMatrix = async () => {
      try {
        const res = await fetch('/api/omp/universe/compatibility');
        if (res.ok) {
          const data = await res.json();
          // Transform backend format to UI format
          const transformed: CompatibilityMatrix = {
            robustnessToFamilies: data.training_model_to_robustness 
              ? Object.entries(data.training_model_to_robustness).reduce((acc, [family, profiles]) => {
                  (profiles as string[]).forEach(profile => {
                    if (!acc[profile]) acc[profile] = [];
                    acc[profile].push(family);
                  });
                  return acc;
                }, {} as Record<string, string[]>)
              : DEFAULT_COMPATIBILITY.robustnessToFamilies,
            robustnessToStrategies: data.robustness_to_training_strategy || DEFAULT_COMPATIBILITY.robustnessToStrategies,
            techToTiers: data.training_tech_to_complexity
              ? Object.entries(data.training_tech_to_complexity).reduce((acc, [tech, tiers]) => {
                  acc[tech] = (tiers as number[]).map(t => `tier${t}_${t === 1 ? 'fast' : t === 2 ? 'medium' : t === 3 ? 'slow' : 'very_slow'}`);
                  return acc;
                }, {} as Record<string, string[]>)
              : DEFAULT_COMPATIBILITY.techToTiers,
            source: data.source || 'unknown'
          };
          setCompatibility(transformed);
          setMatrixSource(data.source || 'file');
        }
      } catch (err) {
        console.warn('Failed to load compatibility matrix, using defaults');
      }
    };
    
    fetchCompatibilityMatrix();
  }, []);
  
  // Fetch strategy registry from backend
  useEffect(() => {
    const fetchStrategyRegistry = async () => {
      try {
        const res = await fetch('/api/omp/universe/strategies');
        if (res.ok) {
          const data = await res.json();
          setStrategyRegistry(data.strategies || {});
          setRegistryLoaded(true);
        }
      } catch (err) {
        console.warn('Failed to load strategy registry');
      }
    };
    
    fetchStrategyRegistry();
  }, []);
  
  // Compute filtered strategies based on current selections
  const filteredStrategies = useMemo(() => {
    if (!registryLoaded) return [];
    
    return Object.entries(strategyRegistry)
      .filter(([_, entry]) => {
        // Filter by selected families
        const familyMatch = universeAxes.trainingModel.includes(entry.family);
        // Filter by risk profile
        const riskMatch = entry.risk_profiles.includes(universeAxes.robustnessProfile);
        // Filter by timeframe if not adaptive
        const timeframeMatch = universeAxes.timeframeProfile === 'adaptive' || 
          (universeAxes.timeframeProfile === 'intraday' && entry.timeframe === '1h') ||
          (universeAxes.timeframeProfile === 'swing' && entry.timeframe === 'daily') ||
          (universeAxes.timeframeProfile === 'position' && entry.timeframe === 'daily') ||
          (universeAxes.timeframeProfile === 'long_term' && entry.timeframe === 'daily');
        
        return familyMatch && riskMatch && timeframeMatch;
      })
      .map(([id, entry]) => ({ id, ...entry }));
  }, [strategyRegistry, universeAxes.trainingModel, universeAxes.robustnessProfile, universeAxes.timeframeProfile, registryLoaded]);

  useEffect(() => {
    fetchConfig();
  }, [fetchConfig]);

  useEffect(() => {
    if (config?.markets) {
      setMarkets(prev => ({
        ...prev,
        ...(config.markets as unknown as Record<string, MarketConfig>),
      }));
    }
  }, [config]);

  // Validation helpers (use dynamic compatibility matrix)
  const isStrategyAllowed = (strategyId: string) => {
    return compatibility.robustnessToStrategies[universeAxes.robustnessProfile]?.includes(strategyId);
  };

  const isFamilyAllowed = (familyId: string) => {
    return compatibility.robustnessToFamilies[universeAxes.robustnessProfile]?.includes(familyId);
  };

  const isTechCompatible = (techId: string, familyId: string) => {
    const family = STRATEGY_FAMILIES[familyId as keyof typeof STRATEGY_FAMILIES];
    if (!family) return true;
    const allowedTiers = compatibility.techToTiers[techId];
    return allowedTiers?.includes(family.tier);
  };
  
  // Get incompatibility reason for tooltip
  const getIncompatibilityReason = (itemType: string, itemId: string): string | null => {
    if (itemType === 'strategy' && !isStrategyAllowed(itemId)) {
      return `Training strategy '${itemId}' is not compatible with robustness profile '${universeAxes.robustnessProfile}'`;
    }
    if (itemType === 'family' && !isFamilyAllowed(itemId)) {
      return `Strategy family '${itemId}' is not allowed for robustness profile '${universeAxes.robustnessProfile}'`;
    }
    if (itemType === 'family' && !isTechCompatible(universeAxes.trainingTech, itemId)) {
      return `Strategy family '${itemId}' requires higher complexity tier than '${universeAxes.trainingTech}' supports`;
    }
    return null;
  };

  const toggleFamily = (familyId: string) => {
    setUniverseAxes(prev => ({
      ...prev,
      trainingModel: prev.trainingModel.includes(familyId)
        ? prev.trainingModel.filter(f => f !== familyId)
        : [...prev.trainingModel, familyId],
    }));
  };

  const toggleMarket = (market: string) => {
    setMarkets(prev => ({
      ...prev,
      [market]: { ...prev[market], enabled: !prev[market].enabled },
    }));
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      const response = await fetch('/api/omp/config/universe', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ universe: universeAxes, markets }),
        credentials: 'same-origin',
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || 'Failed to save');
      }

      fetchConfig();
    } catch (err) {
      console.error('Failed to save universe config:', err);
    } finally {
      setSaving(false);
    }
  };

  const getColorClass = (color: string, variant: 'bg' | 'border' | 'text' = 'bg') => {
    const colors: Record<string, Record<string, string>> = {
      emerald: { bg: 'bg-emerald-500/20', border: 'border-emerald-500/50', text: 'text-emerald-400' },
      green: { bg: 'bg-green-500/20', border: 'border-green-500/50', text: 'text-green-400' },
      blue: { bg: 'bg-blue-500/20', border: 'border-blue-500/50', text: 'text-blue-400' },
      orange: { bg: 'bg-orange-500/20', border: 'border-orange-500/50', text: 'text-orange-400' },
      rose: { bg: 'bg-rose-500/20', border: 'border-rose-500/50', text: 'text-rose-400' },
    };
    return colors[color]?.[variant] || '';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white flex items-center gap-3">
              <Globe className="w-6 h-6 text-violet-400" />
              Parameter Universe System
            </h1>
            <p className="text-slate-400 text-sm mt-1">
              Configure the 4 axes that control strategy generation bounds
            </p>
          </div>

          <button
            onClick={handleSave}
            disabled={saving}
            className="flex items-center gap-2 px-4 py-2 bg-violet-600 hover:bg-violet-500 text-white rounded-lg font-medium transition-colors disabled:opacity-50"
          >
            {saving ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
            Save Configuration
          </button>
        </div>

        {/* 4 Axes Section */}
        <div className="rounded-xl border border-slate-700 bg-slate-800/50 overflow-hidden">
          <button
            onClick={() => setExpandedSections(prev => ({ ...prev, axes: !prev.axes }))}
            className="w-full flex items-center justify-between p-4 hover:bg-slate-700/30 transition-colors"
          >
            <div className="flex items-center gap-3">
              <Shield className="w-5 h-5 text-violet-400" />
              <span className="font-semibold">Universe Axes Configuration</span>
            </div>
            {expandedSections.axes ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
          </button>

          {expandedSections.axes && (
            <div className="p-6 pt-0 space-y-6">
              
              {/* Axis 1: Robustness Profile */}
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <Shield className="w-4 h-4 text-emerald-400" />
                  <h3 className="font-medium">Robustness Profile</h3>
                </div>
                <div className="grid grid-cols-2 lg:grid-cols-5 gap-2">
                  {Object.entries(ROBUSTNESS_PROFILES).map(([id, profile]) => (
                    <button
                      key={id}
                      onClick={() => setUniverseAxes(prev => ({ ...prev, robustnessProfile: id }))}
                      className={`p-3 rounded-lg border text-left transition-all ${
                        universeAxes.robustnessProfile === id
                          ? `${getColorClass(profile.color, 'bg')} ${getColorClass(profile.color, 'border')} ring-1 ring-offset-1 ring-offset-slate-900 ring-${profile.color}-400/50`
                          : 'bg-slate-800 border-slate-700 hover:border-slate-600'
                      }`}
                    >
                      <p className={`font-medium text-sm ${universeAxes.robustnessProfile === id ? getColorClass(profile.color, 'text') : 'text-white'}`}>
                        {profile.label}
                      </p>
                      <p className="text-xs text-slate-500 mt-1">Max DD: {profile.maxDD}</p>
                    </button>
                  ))}
                </div>
              </div>

              {/* Axis 2: Training Strategy */}
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <BookOpen className="w-4 h-4 text-blue-400" />
                  <h3 className="font-medium">Training Strategy</h3>
                </div>
                <div className="grid grid-cols-2 lg:grid-cols-5 gap-2">
                  {Object.entries(TRAINING_STRATEGIES).map(([id, strategy]) => {
                    const allowed = isStrategyAllowed(id);
                    return (
                      <button
                        key={id}
                        onClick={() => allowed && setUniverseAxes(prev => ({ ...prev, trainingStrategy: id }))}
                        disabled={!allowed}
                        className={`p-3 rounded-lg border text-left transition-all ${
                          !allowed
                            ? 'bg-slate-900/50 border-slate-800 opacity-40 cursor-not-allowed'
                            : universeAxes.trainingStrategy === id
                            ? 'bg-blue-500/20 border-blue-500/50 ring-1 ring-blue-400/30'
                            : 'bg-slate-800 border-slate-700 hover:border-slate-600'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <p className={`font-medium text-sm ${universeAxes.trainingStrategy === id ? 'text-blue-400' : 'text-white'}`}>
                            {strategy.label}
                          </p>
                          {!allowed && <AlertCircle className="w-3 h-3 text-slate-600" />}
                        </div>
                        <p className="text-xs text-slate-500 mt-1">
                          {strategy.folds ? `${strategy.folds} folds` : 'Single split'}
                        </p>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Axis 3: Training Tech */}
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <Cpu className="w-4 h-4 text-amber-400" />
                  <h3 className="font-medium">Training Tech</h3>
                </div>
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
                  {Object.entries(TRAINING_TECH).map(([id, tech]) => (
                    <button
                      key={id}
                      onClick={() => setUniverseAxes(prev => ({ ...prev, trainingTech: id }))}
                      className={`p-3 rounded-lg border text-left transition-all ${
                        universeAxes.trainingTech === id
                          ? 'bg-amber-500/20 border-amber-500/50 ring-1 ring-amber-400/30'
                          : 'bg-slate-800 border-slate-700 hover:border-slate-600'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <span>{tech.icon}</span>
                        <p className={`font-medium text-sm ${universeAxes.trainingTech === id ? 'text-amber-400' : 'text-white'}`}>
                          {tech.label}
                        </p>
                      </div>
                      <p className="text-xs text-slate-500 mt-1">
                        {tech.workers} workers • {tech.timeout}
                      </p>
                    </button>
                  ))}
                </div>
              </div>

              {/* Axis 4: Timeframe Profile */}
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <Clock className="w-4 h-4 text-cyan-400" />
                  <h3 className="font-medium">Timeframe Profile</h3>
                </div>
                <div className="grid grid-cols-2 lg:grid-cols-5 gap-2">
                  {Object.entries(TIMEFRAME_PROFILES).map(([id, profile]) => (
                    <button
                      key={id}
                      onClick={() => setUniverseAxes(prev => ({ ...prev, timeframeProfile: id }))}
                      className={`p-3 rounded-lg border text-left transition-all ${
                        universeAxes.timeframeProfile === id
                          ? 'bg-cyan-500/20 border-cyan-500/50 ring-1 ring-cyan-400/30'
                          : 'bg-slate-800 border-slate-700 hover:border-slate-600'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <span>{profile.icon}</span>
                        <p className={`font-medium text-sm ${universeAxes.timeframeProfile === id ? 'text-cyan-400' : 'text-white'}`}>
                          {profile.label}
                        </p>
                      </div>
                      <p className="text-xs text-slate-500 mt-1">{profile.holding}</p>
                    </button>
                  ))}
                </div>
              </div>

              {/* Axis 5: Training Model (Strategy Families) */}
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <TrendingUp className="w-4 h-4 text-violet-400" />
                  <h3 className="font-medium">Training Models (Strategy Families)</h3>
                  <span className="text-xs text-slate-500">
                    ({universeAxes.trainingModel.length} selected)
                  </span>
                </div>
                <div className="grid grid-cols-3 lg:grid-cols-5 gap-2">
                  {Object.entries(STRATEGY_FAMILIES).map(([id, family]) => {
                    const allowed = isFamilyAllowed(id);
                    const techCompatible = isTechCompatible(universeAxes.trainingTech, id);
                    const selected = universeAxes.trainingModel.includes(id);
                    const disabled = !allowed || !techCompatible;

                    return (
                      <button
                        key={id}
                        onClick={() => !disabled && toggleFamily(id)}
                        disabled={disabled}
                        className={`p-3 rounded-lg border text-left transition-all ${
                          disabled
                            ? 'bg-slate-900/50 border-slate-800 opacity-40 cursor-not-allowed'
                            : selected
                            ? 'bg-violet-500/20 border-violet-500/50 ring-1 ring-violet-400/30'
                            : 'bg-slate-800 border-slate-700 hover:border-slate-600'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <p className={`font-medium text-xs ${selected ? 'text-violet-400' : 'text-white'}`}>
                            {family.label}
                          </p>
                          {selected && <CheckCircle2 className="w-3 h-3 text-violet-400" />}
                          {disabled && <AlertCircle className="w-3 h-3 text-slate-600" />}
                        </div>
                        <p className="text-[10px] text-slate-500 mt-1">{family.holding}</p>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Compatibility Info */}
              <div className="p-3 rounded-lg bg-slate-900/50 border border-slate-700 flex items-start gap-2">
                <Info className="w-4 h-4 text-slate-500 mt-0.5 flex-shrink-0" />
                <div className="text-xs text-slate-400">
                  <p>
                    Grayed options are incompatible with the selected Robustness Profile or Training Tech.
                    The system automatically restricts parameter space based on these selections.
                  </p>
                  <p className="mt-1 text-slate-500">
                    Compatibility matrix source: <span className={matrixSource === 'file' ? 'text-green-400' : 'text-yellow-400'}>
                      {matrixSource === 'file' ? 'loaded from configs' : 'using defaults'}
                    </span>
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Available Strategies Section */}
        <div className="rounded-xl border border-slate-700 bg-slate-800/50 overflow-hidden">
          <button
            onClick={() => setExpandedSections(prev => ({ ...prev, strategies: !prev.strategies }))}
            className="w-full flex items-center justify-between p-4 hover:bg-slate-700/30 transition-colors"
          >
            <div className="flex items-center gap-3">
              <List className="w-5 h-5 text-emerald-400" />
              <span className="font-semibold">Available Strategies</span>
              <span className="text-sm text-emerald-400 font-mono">
                {filteredStrategies.length} strategies in universe
              </span>
            </div>
            {expandedSections.strategies ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
          </button>

          {expandedSections.strategies && (
            <div className="p-6 pt-0">
              {filteredStrategies.length === 0 ? (
                <div className="p-4 text-center text-slate-500">
                  <AlertCircle className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p>No strategies match the current selection.</p>
                  <p className="text-xs mt-1">Try adjusting the risk profile or family selection.</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {/* Group by family */}
                  {universeAxes.trainingModel.map(family => {
                    const familyStrategies = filteredStrategies.filter(s => s.family === family);
                    if (familyStrategies.length === 0) return null;
                    
                    return (
                      <div key={family}>
                        <h4 className="text-sm font-medium text-slate-400 mb-2 flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-violet-400"></span>
                          {STRATEGY_FAMILIES[family as keyof typeof STRATEGY_FAMILIES]?.label || family}
                          <span className="text-xs text-slate-600">({familyStrategies.length})</span>
                        </h4>
                        <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
                          {familyStrategies.slice(0, 12).map(strategy => (
                            <div
                              key={strategy.id}
                              className="p-2 rounded-lg bg-slate-900/50 border border-slate-700 text-xs"
                            >
                              <p className="font-medium text-white truncate" title={strategy.id}>
                                {strategy.variant.replace(/_/g, ' ')}
                              </p>
                              <p className="text-slate-500 mt-0.5 flex items-center gap-1">
                                <Zap className="w-3 h-3" />
                                {strategy.hypothesis}
                              </p>
                            </div>
                          ))}
                          {familyStrategies.length > 12 && (
                            <div className="p-2 rounded-lg bg-slate-900/30 border border-slate-800 text-xs flex items-center justify-center text-slate-500">
                              +{familyStrategies.length - 12} more
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
              
              {/* Registry info */}
              <div className="mt-4 p-3 rounded-lg bg-slate-900/50 border border-slate-700 flex items-start gap-2">
                <Info className="w-4 h-4 text-emerald-400 mt-0.5 flex-shrink-0" />
                <div className="text-xs text-slate-400">
                  <p>
                    <strong className="text-emerald-400">Universe-Only Mode:</strong> Only strategies registered in the strategy registry 
                    can be generated. The GA will use these as templates with controlled parameter variations.
                  </p>
                  <p className="mt-1 text-slate-500">
                    Registry status: {registryLoaded ? (
                      <span className="text-green-400">loaded ({Object.keys(strategyRegistry).length} total strategies)</span>
                    ) : (
                      <span className="text-yellow-400">loading...</span>
                    )}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Markets Section */}
        <div className="rounded-xl border border-slate-700 bg-slate-800/50 overflow-hidden">
          <button
            onClick={() => setExpandedSections(prev => ({ ...prev, markets: !prev.markets }))}
            className="w-full flex items-center justify-between p-4 hover:bg-slate-700/30 transition-colors"
          >
            <div className="flex items-center gap-3">
              <Globe className="w-5 h-5 text-blue-400" />
              <span className="font-semibold">Market Universes</span>
            </div>
            {expandedSections.markets ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
          </button>

          {expandedSections.markets && (
            <div className="p-6 pt-0 space-y-4">
              {Object.entries(markets).map(([key, market]) => (
                <div
                  key={key}
                  className={`p-4 rounded-lg border transition-colors ${
                    market.enabled
                      ? 'bg-slate-800/50 border-slate-700'
                      : 'bg-slate-900/30 border-slate-800 opacity-60'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <button
                        onClick={() => toggleMarket(key)}
                        className={`relative w-12 h-6 rounded-full transition-colors ${
                          market.enabled
                            ? key === 'br' ? 'bg-green-500' : 'bg-blue-500'
                            : 'bg-slate-600'
                        }`}
                      >
                        <span className={`absolute top-0.5 w-5 h-5 rounded-full bg-white shadow-md transition-transform ${
                          market.enabled ? 'left-6' : 'left-0.5'
                        }`} />
                      </button>
                      <div>
                        <h4 className="font-medium text-sm">{market.name}</h4>
                        <p className="text-xs text-slate-500">
                          {market.currency} • {market.calendar.toUpperCase()} • Lot: {market.lot_size}
                        </p>
                      </div>
                    </div>
                    {market.enabled && (
                      <div className="flex gap-1">
                        {Object.entries(PRESET_UNIVERSES)
                          .filter(([k]) => key === 'br' ? ['ibov', 'ibrx100', 'small'].includes(k) : ['sp500', 'nasdaq100', 'djia'].includes(k))
                          .map(([universeKey, info]) => (
                            <button
                              key={universeKey}
                              onClick={() => setMarkets(prev => ({
                                ...prev,
                                [key]: { ...prev[key], universe: universeKey },
                              }))}
                              className={`px-2 py-1 text-xs rounded transition-colors ${
                                market.universe === universeKey
                                  ? 'bg-blue-500/20 text-blue-400 border border-blue-500/50'
                                  : 'bg-slate-700 text-slate-400 border border-slate-600 hover:border-slate-500'
                              }`}
                            >
                              {info.label}
                            </button>
                          ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Summary Card */}
        <div className="p-4 rounded-lg bg-gradient-to-r from-violet-500/10 to-blue-500/10 border border-violet-500/30">
          <h4 className="font-medium text-violet-300 mb-2">Current Configuration Summary</h4>
          <div className="grid grid-cols-2 lg:grid-cols-6 gap-4 text-sm">
            <div>
              <p className="text-slate-500 text-xs">Robustness</p>
              <p className="text-white font-medium">
                {ROBUSTNESS_PROFILES[universeAxes.robustnessProfile as keyof typeof ROBUSTNESS_PROFILES]?.label}
              </p>
            </div>
            <div>
              <p className="text-slate-500 text-xs">Training Strategy</p>
              <p className="text-white font-medium">
                {TRAINING_STRATEGIES[universeAxes.trainingStrategy as keyof typeof TRAINING_STRATEGIES]?.label}
              </p>
            </div>
            <div>
              <p className="text-slate-500 text-xs">Training Tech</p>
              <p className="text-white font-medium">
                {TRAINING_TECH[universeAxes.trainingTech as keyof typeof TRAINING_TECH]?.label}
              </p>
            </div>
            <div>
              <p className="text-slate-500 text-xs">Timeframe</p>
              <p className="text-white font-medium">
                {TIMEFRAME_PROFILES[universeAxes.timeframeProfile as keyof typeof TIMEFRAME_PROFILES]?.label}
              </p>
            </div>
            <div>
              <p className="text-slate-500 text-xs">Strategy Families</p>
              <p className="text-white font-medium">
                {universeAxes.trainingModel.length} selected
              </p>
            </div>
            <div>
              <p className="text-slate-500 text-xs">Available Strategies</p>
              <p className="text-emerald-400 font-medium">
                {filteredStrategies.length} in universe
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ConfigUniverse;

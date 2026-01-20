/**
 * ConfigUniverse - Sistema de Configuração do Universo de Parâmetros
 * 
 * Configure os 5 eixos que controlam a geração de estratégias:
 * - Perfil de Robustez (tolerância a risco)
 * - Estratégia de Treino (validação)
 * - Tech de Treino (recursos computacionais)
 * - Perfil de Timeframe (horizonte de operação)
 * - Modelos de Treino (famílias de estratégia)
 */

import { useEffect, useState, useMemo } from 'react';
import { 
  Globe, Save, RefreshCw, Shield, Cpu, BookOpen, 
  TrendingUp, AlertCircle, CheckCircle2, Info,
  ChevronDown, ChevronUp, Clock, Zap, List, HelpCircle, Settings
} from 'lucide-react';
import { useOmpStore } from '../stores/ompStore';
import { StrategyConfigModal } from '../components/universe/StrategyConfigModal';

// =============================================================================
// TOOLTIPS EM PORTUGUÊS - Explicações para leigos
// =============================================================================

const UNIVERSE_TOOLTIPS = {
  // Perfis de Robustez
  robustness: {
    section: 'Define quanto risco você aceita perder antes de parar. Drawdown é a maior queda do seu patrimônio desde o pico. Perfis mais conservadores limitam perdas, mas também limitam ganhos.',
    muito_conservador: 'Para quem não tolera perdas. Máximo 8% de queda. Ideal para capital que você não pode perder. Estratégias mais lentas e seguras.',
    conservador: 'Crescimento consistente com quedas controladas. Máximo 12% de queda. Bom equilíbrio para maioria dos investidores.',
    moderado: 'Aceita volatilidade em troca de retornos maiores. Máximo 20% de queda. Padrão para traders experientes.',
    arrojado: 'Alta tolerância a risco. Máximo 25% de queda. Para capital especulativo que você pode perder.',
    muito_arrojado: 'Risco máximo dentro de limites defensáveis. Máximo 30% de queda. Apenas para capital de risco.',
  },
  // Estratégias de Treino
  training: {
    section: 'Método usado para validar se a estratégia funciona de verdade ou se é só sorte. Quanto mais rigoroso, menor chance de overfitting (decorar o passado).',
    purged_kfold: 'Divide os dados em 5 partes, treina em 4 e testa em 1, repetindo para todas. Período de embargo evita vazamento de informação. Padrão da indústria.',
    walk_forward: 'Simula exatamente como você operaria: treina no passado, testa no futuro, avança a janela. O mais realista para séries temporais.',
    anchored: 'Mantém ponto inicial fixo e expande a janela de treino. Mais simples e rápido. Bom para exploração inicial.',
    expanding_window: 'Treino cresce a cada passo, acumulando mais dados. Captura mudanças de regime do mercado.',
    monte_carlo: 'Embaralha os dados milhares de vezes para testar robustez estatística. O mais rigoroso, mas mais lento.',
  },
  // Tech de Treino
  tech: {
    section: 'Quanto poder computacional usar. Mais workers = mais rápido, mas usa mais recursos do computador.',
    cpu_fast: 'Modo rápido para desenvolvimento e testes. 4 workers paralelos, timeout de 30 min. Use para explorar ideias rapidamente.',
    cpu_parallel: 'Configuração padrão de produção. 8 workers paralelos, timeout de 2h. Bom equilíbrio entre velocidade e profundidade.',
    cpu_intensive: 'Computação pesada para estratégias complexas. 16 workers paralelos, timeout de 6h. Para validação final antes de produção.',
    distributed: 'Nível de cluster para estratégias institucionais. Workers automáticos, timeout de 24h. Para mineração profunda noturna.',
  },
  // Timeframe
  timeframe: {
    section: 'Horizonte de tempo das operações. Intraday fecha no mesmo dia, Swing segura dias, Position segura semanas.',
    intraday: 'Operações que abrem e fecham no mesmo dia. 1-8 horas de duração. Exige monitoramento constante.',
    swing: 'Operações de 2-10 dias. Captura movimentos de curto prazo. Não exige monitoramento constante.',
    position: 'Operações de 2-12 semanas. Captura tendências maiores. Menos trades, menores custos.',
    long_term: 'Operações de 3+ meses. Investimento de longo prazo. Mínimo de trades.',
    adaptive: 'Sistema detecta automaticamente o melhor timeframe para cada estratégia.',
  },
  // Famílias de Estratégia
  families: {
    section: 'Tipos de estratégia a explorar. Cada família tem uma lógica diferente de operação. Selecione múltiplas para diversificar.',
    intraday: 'Estratégias que operam dentro do dia. ORB breakout, VWAP, gaps. Exige dados de alta frequência.',
    swing: 'Captura movimentos de 2-10 dias. Médias móveis, RSI, breakouts. Popular entre traders ativos.',
    position: 'Segue tendências de semanas a meses. Menos trades, maior convicção por trade.',
    pair: 'Opera pares de ativos correlacionados. Compra um, vende outro. Market neutral (não depende da direção do mercado).',
    portfolio: 'Aloca capital entre múltiplos ativos. Risk parity, mínima variância, máximo Sharpe. Para gestão de carteira.',
    momentum: 'Compra o que está subindo, vende o que está caindo. Segue tendências de 1-12 meses.',
    mean_reversion: 'Aposta que preços voltam à média. Compra barato, vende caro. Funciona em mercados laterais.',
    breakout: 'Opera rompimentos de suporte/resistência. Donchian, expansão de volatilidade.',
    sector_rotation: 'Rotaciona entre setores baseado em ciclo econômico ou força relativa.',
    factor: 'Investimento baseado em fatores acadêmicos: valor, qualidade, momentum, baixa volatilidade.',
    seasonal: 'Explora padrões sazonais. Janeiro effect, sell in may, commodities.',
    volatility: 'Opera a volatilidade em si. VIX mean reversion, breakout de ATR.',
    event_driven: 'Opera eventos específicos. Earnings, M&A, rebalanceamentos de índice.',
    buy_hold: 'Compra e segura. Benchmark passivo para comparar estratégias ativas.',
    multi_strategy: 'Combina múltiplas estratégias com alocação dinâmica baseada em regime.',
  },
};

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


// =============================================================================
// CONSTANTS
// =============================================================================

const ROBUSTNESS_PROFILES = {
  muito_conservador: {
    label: 'Muito Conservador',
    description: 'Preservação de capital, quedas mínimas',
    maxDD: '8%',
    kelly: '0.15',
    color: 'emerald',
    tooltip: UNIVERSE_TOOLTIPS.robustness.muito_conservador,
  },
  conservador: {
    label: 'Conservador',
    description: 'Crescimento consistente, quedas controladas',
    maxDD: '12%',
    kelly: '0.30',
    color: 'green',
    tooltip: UNIVERSE_TOOLTIPS.robustness.conservador,
  },
  moderado: {
    label: 'Moderado',
    description: 'Crescimento equilibrado com volatilidade aceitável',
    maxDD: '20%',
    kelly: '0.40',
    color: 'blue',
    tooltip: UNIVERSE_TOOLTIPS.robustness.moderado,
  },
  arrojado: {
    label: 'Arrojado',
    description: 'Crescimento agressivo, alta tolerância a risco',
    maxDD: '25%',
    kelly: '0.50',
    color: 'orange',
    tooltip: UNIVERSE_TOOLTIPS.robustness.arrojado,
  },
  muito_arrojado: {
    label: 'Muito Arrojado',
    description: 'Risco máximo dentro de limites defensáveis',
    maxDD: '30%',
    kelly: '0.50',
    color: 'rose',
    tooltip: UNIVERSE_TOOLTIPS.robustness.muito_arrojado,
  },
};

const TRAINING_STRATEGIES = {
  purged_kfold: {
    label: 'Purged K-Fold',
    description: 'Validação robusta com períodos de embargo',
    folds: 5,
    tier: 'tier2_medium',
    tooltip: UNIVERSE_TOOLTIPS.training.purged_kfold,
  },
  walk_forward: {
    label: 'Walk-Forward',
    description: 'Janela deslizante para séries temporais',
    folds: 8,
    tier: 'tier2_medium',
    tooltip: UNIVERSE_TOOLTIPS.training.walk_forward,
  },
  anchored: {
    label: 'Ancorado',
    description: 'Âncora fixa com janela expansiva',
    folds: null,
    tier: 'tier1_fast',
    tooltip: UNIVERSE_TOOLTIPS.training.anchored,
  },
  expanding_window: {
    label: 'Janela Expansiva',
    description: 'Conjunto de treino cresce incrementalmente',
    folds: 6,
    tier: 'tier2_medium',
    tooltip: UNIVERSE_TOOLTIPS.training.expanding_window,
  },
  monte_carlo: {
    label: 'Monte Carlo',
    description: 'Teste de estresse extensivo para produção',
    folds: 10,
    tier: 'tier3_slow',
    tooltip: UNIVERSE_TOOLTIPS.training.monte_carlo,
  },
};

const TRAINING_TECH = {
  cpu_fast: {
    label: 'CPU Rápido',
    description: 'Exploração rápida para desenvolvimento',
    workers: 4,
    timeout: '30min',
    icon: '⚡',
    tooltip: UNIVERSE_TOOLTIPS.tech.cpu_fast,
  },
  cpu_parallel: {
    label: 'CPU Paralelo',
    description: 'Configuração padrão de produção',
    workers: 8,
    timeout: '2h',
    icon: '🔄',
    tooltip: UNIVERSE_TOOLTIPS.tech.cpu_parallel,
  },
  cpu_intensive: {
    label: 'CPU Intensivo',
    description: 'Computação pesada para estratégias complexas',
    workers: 16,
    timeout: '6h',
    icon: '💪',
    tooltip: UNIVERSE_TOOLTIPS.tech.cpu_intensive,
  },
  distributed: {
    label: 'Distribuído',
    description: 'Nível de cluster para estratégias institucionais',
    workers: 'auto',
    timeout: '24h',
    icon: '🌐',
    tooltip: UNIVERSE_TOOLTIPS.tech.distributed,
  },
};

const STRATEGY_FAMILIES = {
  intraday: { label: 'Intraday', tier: 'tier1_fast', holding: '1-8h', tooltip: UNIVERSE_TOOLTIPS.families.intraday },
  swing: { label: 'Swing Trading', tier: 'tier1_fast', holding: '2-10 dias', tooltip: UNIVERSE_TOOLTIPS.families.swing },
  position: { label: 'Position Trading', tier: 'tier2_medium', holding: 'Semanas-Meses', tooltip: UNIVERSE_TOOLTIPS.families.position },
  pair: { label: 'Pair Trading', tier: 'tier2_medium', holding: '5-30 dias', tooltip: UNIVERSE_TOOLTIPS.families.pair },
  portfolio: { label: 'Portfólio', tier: 'tier3_slow', holding: 'Rebalanceamento', tooltip: UNIVERSE_TOOLTIPS.families.portfolio },
  momentum: { label: 'Momentum', tier: 'tier1_fast', holding: '1-6 meses', tooltip: UNIVERSE_TOOLTIPS.families.momentum },
  mean_reversion: { label: 'Reversão à Média', tier: 'tier1_fast', holding: '2-10 dias', tooltip: UNIVERSE_TOOLTIPS.families.mean_reversion },
  breakout: { label: 'Rompimento', tier: 'tier1_fast', holding: '3-15 dias', tooltip: UNIVERSE_TOOLTIPS.families.breakout },
  sector_rotation: { label: 'Rotação Setorial', tier: 'tier2_medium', holding: '1-6 meses', tooltip: UNIVERSE_TOOLTIPS.families.sector_rotation },
  factor: { label: 'Fator', tier: 'tier3_slow', holding: 'Meses-Anos', tooltip: UNIVERSE_TOOLTIPS.families.factor },
  seasonal: { label: 'Sazonal', tier: 'tier2_medium', holding: 'Dias-Semanas', tooltip: UNIVERSE_TOOLTIPS.families.seasonal },
  volatility: { label: 'Volatilidade', tier: 'tier2_medium', holding: '3-15 dias', tooltip: UNIVERSE_TOOLTIPS.families.volatility },
  event_driven: { label: 'Eventos', tier: 'tier2_medium', holding: '1-5 dias', tooltip: UNIVERSE_TOOLTIPS.families.event_driven },
  buy_hold: { label: 'Buy & Hold', tier: 'tier1_fast', holding: 'Anos', tooltip: UNIVERSE_TOOLTIPS.families.buy_hold },
  multi_strategy: { label: 'Multi-Estratégia', tier: 'tier3_slow', holding: 'Variável', tooltip: UNIVERSE_TOOLTIPS.families.multi_strategy },
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

// Opções de perfil de timeframe
const TIMEFRAME_PROFILES = {
  intraday: { label: 'Intraday', holding: '1-8h', icon: '⚡', tooltip: UNIVERSE_TOOLTIPS.timeframe.intraday },
  swing: { label: 'Swing', holding: '2-10 dias', icon: '📈', tooltip: UNIVERSE_TOOLTIPS.timeframe.swing },
  position: { label: 'Position', holding: '2-12 semanas', icon: '🎯', tooltip: UNIVERSE_TOOLTIPS.timeframe.position },
  long_term: { label: 'Longo Prazo', holding: '3+ meses', icon: '🏦', tooltip: UNIVERSE_TOOLTIPS.timeframe.long_term },
  adaptive: { label: 'Adaptativo', holding: 'Auto-detectar', icon: '🔄', tooltip: UNIVERSE_TOOLTIPS.timeframe.adaptive },
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
  
  // Modal de configuração de família
  const [configModalFamily, setConfigModalFamily] = useState<string | null>(null);
  
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
        const res = await fetch('http://localhost:3001/api/omp/universe/compatibility');
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
        const res = await fetch('http://localhost:3001/api/omp/universe/strategies');
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
      const response = await fetch('http://localhost:3001/api/omp/config/universe', {
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
              Universo de Parâmetros
            </h1>
            <p className="text-slate-400 text-sm mt-1">
              Configure os 5 eixos que controlam a geração de estratégias
            </p>
          </div>

          <button
            onClick={handleSave}
            disabled={saving}
            className="flex items-center gap-2 px-4 py-2 bg-violet-600 hover:bg-violet-500 text-white rounded-lg font-medium transition-colors disabled:opacity-50"
          >
            {saving ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
            Salvar Configuração
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
              <span className="font-semibold">Configuração dos Eixos do Universo</span>
            </div>
            {expandedSections.axes ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
          </button>

          {expandedSections.axes && (
            <div className="p-6 pt-0 space-y-6">
              
              {/* Eixo 1: Perfil de Robustez */}
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <Shield className="w-4 h-4 text-emerald-400" />
                  <h3 className="font-medium">Perfil de Robustez</h3>
                  <div className="group relative">
                    <HelpCircle className="w-4 h-4 text-slate-500 hover:text-cyan-400 cursor-help" />
                    <div className="absolute left-0 top-6 w-80 p-3 bg-slate-800 border border-slate-600 rounded-lg shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50">
                      <p className="text-xs text-slate-300">{UNIVERSE_TOOLTIPS.robustness.section}</p>
                    </div>
                  </div>
                </div>
                <div className="grid grid-cols-2 lg:grid-cols-5 gap-2">
                  {Object.entries(ROBUSTNESS_PROFILES).map(([id, profile]) => (
                    <div key={id} className="group relative">
                      <button
                        onClick={() => setUniverseAxes(prev => ({ ...prev, robustnessProfile: id }))}
                        className={`w-full p-3 rounded-lg border text-left transition-all ${
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
                      {/* Tooltip */}
                      <div className="absolute left-0 bottom-full mb-2 w-72 p-3 bg-slate-800 border border-slate-600 rounded-lg shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50 pointer-events-none">
                        <p className="text-xs text-white font-medium mb-1">{profile.label}</p>
                        <p className="text-xs text-slate-300">{profile.tooltip}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Eixo 2: Estratégia de Treino */}
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <BookOpen className="w-4 h-4 text-blue-400" />
                  <h3 className="font-medium">Estratégia de Treino</h3>
                  <div className="group relative">
                    <HelpCircle className="w-4 h-4 text-slate-500 hover:text-cyan-400 cursor-help" />
                    <div className="absolute left-0 top-6 w-80 p-3 bg-slate-800 border border-slate-600 rounded-lg shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50">
                      <p className="text-xs text-slate-300">{UNIVERSE_TOOLTIPS.training.section}</p>
                    </div>
                  </div>
                </div>
                <div className="grid grid-cols-2 lg:grid-cols-5 gap-2">
                  {Object.entries(TRAINING_STRATEGIES).map(([id, strategy]) => {
                    const allowed = isStrategyAllowed(id);
                    return (
                      <div key={id} className="group relative">
                        <button
                          onClick={() => allowed && setUniverseAxes(prev => ({ ...prev, trainingStrategy: id }))}
                          disabled={!allowed}
                          className={`w-full p-3 rounded-lg border text-left transition-all ${
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
                            {strategy.folds ? `${strategy.folds} folds` : 'Divisão única'}
                          </p>
                        </button>
                        {/* Tooltip */}
                        <div className="absolute left-0 bottom-full mb-2 w-72 p-3 bg-slate-800 border border-slate-600 rounded-lg shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50 pointer-events-none">
                          <p className="text-xs text-white font-medium mb-1">{strategy.label}</p>
                          <p className="text-xs text-slate-300">{strategy.tooltip}</p>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Eixo 3: Tech de Treino */}
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <Cpu className="w-4 h-4 text-amber-400" />
                  <h3 className="font-medium">Tech de Treino</h3>
                  <div className="group relative">
                    <HelpCircle className="w-4 h-4 text-slate-500 hover:text-cyan-400 cursor-help" />
                    <div className="absolute left-0 top-6 w-80 p-3 bg-slate-800 border border-slate-600 rounded-lg shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50">
                      <p className="text-xs text-slate-300">{UNIVERSE_TOOLTIPS.tech.section}</p>
                    </div>
                  </div>
                </div>
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
                  {Object.entries(TRAINING_TECH).map(([id, tech]) => (
                    <div key={id} className="group relative">
                      <button
                        onClick={() => setUniverseAxes(prev => ({ ...prev, trainingTech: id }))}
                        className={`w-full p-3 rounded-lg border text-left transition-all ${
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
                      {/* Tooltip */}
                      <div className="absolute left-0 bottom-full mb-2 w-72 p-3 bg-slate-800 border border-slate-600 rounded-lg shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50 pointer-events-none">
                        <p className="text-xs text-white font-medium mb-1">{tech.label}</p>
                        <p className="text-xs text-slate-300">{tech.tooltip}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Eixo 4: Perfil de Timeframe */}
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <Clock className="w-4 h-4 text-cyan-400" />
                  <h3 className="font-medium">Perfil de Timeframe</h3>
                  <div className="group relative">
                    <HelpCircle className="w-4 h-4 text-slate-500 hover:text-cyan-400 cursor-help" />
                    <div className="absolute left-0 top-6 w-80 p-3 bg-slate-800 border border-slate-600 rounded-lg shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50">
                      <p className="text-xs text-slate-300">{UNIVERSE_TOOLTIPS.timeframe.section}</p>
                    </div>
                  </div>
                </div>
                <div className="grid grid-cols-2 lg:grid-cols-5 gap-2">
                  {Object.entries(TIMEFRAME_PROFILES).map(([id, profile]) => (
                    <div key={id} className="group relative">
                      <button
                        onClick={() => setUniverseAxes(prev => ({ ...prev, timeframeProfile: id }))}
                        className={`w-full p-3 rounded-lg border text-left transition-all ${
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
                      {/* Tooltip */}
                      <div className="absolute left-0 bottom-full mb-2 w-64 p-3 bg-slate-800 border border-slate-600 rounded-lg shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50 pointer-events-none">
                        <p className="text-xs text-white font-medium mb-1">{profile.label}</p>
                        <p className="text-xs text-slate-300">{profile.tooltip}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Eixo 5: Modelos de Treino (Famílias de Estratégia) */}
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <TrendingUp className="w-4 h-4 text-violet-400" />
                  <h3 className="font-medium">Famílias de Estratégia</h3>
                  <span className="text-xs text-slate-500">
                    ({universeAxes.trainingModel.length} selecionadas)
                  </span>
                  <div className="group relative">
                    <HelpCircle className="w-4 h-4 text-slate-500 hover:text-cyan-400 cursor-help" />
                    <div className="absolute left-0 top-6 w-80 p-3 bg-slate-800 border border-slate-600 rounded-lg shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50">
                      <p className="text-xs text-slate-300">{UNIVERSE_TOOLTIPS.families.section}</p>
                    </div>
                  </div>
                </div>
                <div className="grid grid-cols-3 lg:grid-cols-5 gap-2">
                  {Object.entries(STRATEGY_FAMILIES).map(([id, family]) => {
                    const allowed = isFamilyAllowed(id);
                    const techCompatible = isTechCompatible(universeAxes.trainingTech, id);
                    const selected = universeAxes.trainingModel.includes(id);
                    const disabled = !allowed || !techCompatible;

                    return (
                      <div key={id} className="group relative">
                        <button
                          onClick={() => !disabled && toggleFamily(id)}
                          disabled={disabled}
                          className={`w-full p-3 rounded-lg border text-left transition-all ${
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
                            <div className="flex items-center gap-1">
                              {selected && (
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    setConfigModalFamily(id);
                                  }}
                                  className="p-1 hover:bg-violet-500/30 rounded transition-colors"
                                  title="Configurar parâmetros"
                                >
                                  <Settings className="w-3 h-3 text-violet-400" />
                                </button>
                              )}
                              {selected && <CheckCircle2 className="w-3 h-3 text-violet-400" />}
                              {disabled && <AlertCircle className="w-3 h-3 text-slate-600" />}
                            </div>
                          </div>
                          <p className="text-[10px] text-slate-500 mt-1">{family.holding}</p>
                        </button>
                        {/* Tooltip */}
                        <div className="absolute left-0 bottom-full mb-2 w-64 p-3 bg-slate-800 border border-slate-600 rounded-lg shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50 pointer-events-none">
                          <p className="text-xs text-white font-medium mb-1">{family.label}</p>
                          <p className="text-xs text-slate-300">{family.tooltip}</p>
                          {selected && <p className="text-xs text-violet-400 mt-2">Clique no ⚙️ para configurar parâmetros</p>}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Info de Compatibilidade */}
              <div className="p-3 rounded-lg bg-slate-900/50 border border-slate-700 flex items-start gap-2">
                <Info className="w-4 h-4 text-slate-500 mt-0.5 flex-shrink-0" />
                <div className="text-xs text-slate-400">
                  <p>
                    Opções acinzentadas são incompatíveis com o Perfil de Robustez ou Tech de Treino selecionados.
                    O sistema restringe automaticamente o espaço de parâmetros baseado nessas seleções.
                  </p>
                  <p className="mt-1 text-slate-500">
                    Fonte da matriz de compatibilidade: <span className={matrixSource === 'file' ? 'text-green-400' : 'text-yellow-400'}>
                      {matrixSource === 'file' ? 'carregada dos configs' : 'usando padrões'}
                    </span>
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Seção de Estratégias Disponíveis */}
        <div className="rounded-xl border border-slate-700 bg-slate-800/50 overflow-hidden">
          <button
            onClick={() => setExpandedSections(prev => ({ ...prev, strategies: !prev.strategies }))}
            className="w-full flex items-center justify-between p-4 hover:bg-slate-700/30 transition-colors"
          >
            <div className="flex items-center gap-3">
              <List className="w-5 h-5 text-emerald-400" />
              <span className="font-semibold">Estratégias Disponíveis</span>
              <span className="text-sm text-emerald-400 font-mono">
                {filteredStrategies.length} estratégias no universo
              </span>
            </div>
            {expandedSections.strategies ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
          </button>

          {expandedSections.strategies && (
            <div className="p-6 pt-0">
              {filteredStrategies.length === 0 ? (
                <div className="p-4 text-center text-slate-500">
                  <AlertCircle className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p>Nenhuma estratégia corresponde à seleção atual.</p>
                  <p className="text-xs mt-1">Tente ajustar o perfil de risco ou a seleção de famílias.</p>
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
                              +{familyStrategies.length - 12} mais
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
              
              {/* Info do Registro */}
              <div className="mt-4 p-3 rounded-lg bg-slate-900/50 border border-slate-700 flex items-start gap-2">
                <Info className="w-4 h-4 text-emerald-400 mt-0.5 flex-shrink-0" />
                <div className="text-xs text-slate-400">
                  <p>
                    <strong className="text-emerald-400">Modo Universo:</strong> Apenas estratégias registradas no catálogo 
                    podem ser geradas. O GA usará estas como templates com variações controladas de parâmetros.
                  </p>
                  <p className="mt-1 text-slate-500">
                    Status do registro: {registryLoaded ? (
                      <span className="text-green-400">carregado ({Object.keys(strategyRegistry).length} estratégias totais)</span>
                    ) : (
                      <span className="text-yellow-400">carregando...</span>
                    )}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Seção de Mercados */}
        <div className="rounded-xl border border-slate-700 bg-slate-800/50 overflow-hidden">
          <button
            onClick={() => setExpandedSections(prev => ({ ...prev, markets: !prev.markets }))}
            className="w-full flex items-center justify-between p-4 hover:bg-slate-700/30 transition-colors"
          >
            <div className="flex items-center gap-3">
              <Globe className="w-5 h-5 text-blue-400" />
              <span className="font-semibold">Universos de Mercado</span>
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

        {/* Card de Resumo */}
        <div className="p-4 rounded-lg bg-gradient-to-r from-violet-500/10 to-blue-500/10 border border-violet-500/30">
          <h4 className="font-medium text-violet-300 mb-2">Resumo da Configuração Atual</h4>
          <div className="grid grid-cols-2 lg:grid-cols-6 gap-4 text-sm">
            <div>
              <p className="text-slate-500 text-xs">Robustez</p>
              <p className="text-white font-medium">
                {ROBUSTNESS_PROFILES[universeAxes.robustnessProfile as keyof typeof ROBUSTNESS_PROFILES]?.label}
              </p>
            </div>
            <div>
              <p className="text-slate-500 text-xs">Estratégia de Treino</p>
              <p className="text-white font-medium">
                {TRAINING_STRATEGIES[universeAxes.trainingStrategy as keyof typeof TRAINING_STRATEGIES]?.label}
              </p>
            </div>
            <div>
              <p className="text-slate-500 text-xs">Tech de Treino</p>
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
              <p className="text-slate-500 text-xs">Famílias</p>
              <p className="text-white font-medium">
                {universeAxes.trainingModel.length} selecionadas
              </p>
            </div>
            <div>
              <p className="text-slate-500 text-xs">Estratégias Disponíveis</p>
              <p className="text-emerald-400 font-medium">
                {filteredStrategies.length} no universo
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Modal de Configuração de Família */}
      <StrategyConfigModal
        isOpen={configModalFamily !== null}
        onClose={() => setConfigModalFamily(null)}
        family={configModalFamily || ''}
        familyLabel={configModalFamily ? STRATEGY_FAMILIES[configModalFamily as keyof typeof STRATEGY_FAMILIES]?.label || configModalFamily : ''}
        onSave={(params) => {
          console.log('Parâmetros salvos para', configModalFamily, params);
          // TODO: Salvar no backend
        }}
      />
    </div>
  );
}

export default ConfigUniverse;

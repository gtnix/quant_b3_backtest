/**
 * Cockpit Configuration Defaults
 * 
 * Central source of truth for all SCG run configuration.
 * UI and backend derive from these presets.
 */

// =============================================================================
// TYPES
// =============================================================================

export interface CockpitConfig {
  preset: PresetKey;
  maxRuntimeSeconds: number;
  populationSize: number;
  maxGenerations: number;
  convergenceGenerations: number;
  workers: number;
  seeds: number[];
  stressTestingEnabled: boolean;
  minOosSharpeNet: number;
  maxPbo: number;
  minStressPassed: number;
  campaignConfig?: string;
  runTag?: string;
}

export type PresetKey = 'rapid' | 'institutional' | 'exhaustive';

export interface PresetDefinition {
  key: PresetKey;
  name: string;
  description: string;
  icon: string;
  config: Omit<CockpitConfig, 'preset' | 'campaignConfig' | 'runTag'>;
}

export interface RankingMethod {
  key: RankingMethodKey;
  name: string;
  description: string;
  tooltip: string;
}

export type RankingMethodKey = 'institutional' | 'pareto' | 'sharpe' | 'riskadjusted';

// =============================================================================
// PRESETS - Time-based compute budgets
// =============================================================================

export const COCKPIT_PRESETS: Record<PresetKey, PresetDefinition> = {
  rapid: {
    key: 'rapid',
    name: 'Rápido',
    description: '3 min · Teste rápido',
    icon: '⚡',
    config: {
      maxRuntimeSeconds: 180,
      populationSize: 80,
      maxGenerations: 30,
      convergenceGenerations: 8,
      workers: 8,
      seeds: [42],
      stressTestingEnabled: true,
      minOosSharpeNet: 0.3,
      maxPbo: 0.25,
      minStressPassed: 2,
    },
  },
  institutional: {
    key: 'institutional',
    name: 'Institucional',
    description: '15 min · Produção',
    icon: '🏛️',
    config: {
      maxRuntimeSeconds: 900,
      populationSize: 100,
      maxGenerations: 50,
      convergenceGenerations: 10,
      workers: 8,
      seeds: [42, 123, 456],
      stressTestingEnabled: true,
      minOosSharpeNet: 0.5,
      maxPbo: 0.15,
      minStressPassed: 4,
    },
  },
  exhaustive: {
    key: 'exhaustive',
    name: 'Exaustivo',
    description: '1 hora · Máxima exploração',
    icon: '🔬',
    config: {
      maxRuntimeSeconds: 3600,
      populationSize: 200,
      maxGenerations: 100,
      convergenceGenerations: 15,
      workers: 16,
      seeds: [42, 123, 456, 789, 1011],
      stressTestingEnabled: true,
      minOosSharpeNet: 0.5,
      maxPbo: 0.15,
      minStressPassed: 4,
    },
  },
};

// =============================================================================
// RANKING METHODS
// =============================================================================

export const RANKING_METHODS: Record<RankingMethodKey, RankingMethod> = {
  institutional: {
    key: 'institutional',
    name: 'Institucional',
    description: 'Multi-critério ponderado',
    tooltip: 'Pondera Sharpe OOS, PBO, stress tests e consistência IS→OOS. Usado por fundos quantitativos.',
  },
  pareto: {
    key: 'pareto',
    name: 'Pareto',
    description: 'Fronteira eficiente',
    tooltip: 'Estratégias não-dominadas em Sharpe vs Drawdown. Nenhuma é melhor em tudo.',
  },
  sharpe: {
    key: 'sharpe',
    name: 'Sharpe Puro',
    description: 'Maior Sharpe OOS',
    tooltip: 'Ordena apenas por Sharpe OOS NET. Simples mas pode premiar overfitting.',
  },
  riskadjusted: {
    key: 'riskadjusted',
    name: 'Risco-Ajustado',
    description: 'Sharpe / MaxDD',
    tooltip: 'Sharpe dividido por drawdown máximo. Penaliza estratégias com quedas grandes.',
  },
};

// =============================================================================
// DEFAULT VALUES
// =============================================================================

export const DEFAULT_PRESET: PresetKey = 'institutional';

export function getDefaultConfig(): CockpitConfig {
  const preset = COCKPIT_PRESETS[DEFAULT_PRESET];
  return {
    preset: DEFAULT_PRESET,
    ...preset.config,
  };
}

// =============================================================================
// INTENSITY LEVELS (Workers/Parallelism)
// =============================================================================

export interface IntensityLevel {
  key: string;
  name: string;
  workers: number;
  description: string;
}

export const INTENSITY_LEVELS: IntensityLevel[] = [
  { key: 'low', name: 'Baixa', workers: 2, description: 'Mínimo impacto no sistema' },
  { key: 'medium', name: 'Média', workers: 4, description: 'Equilíbrio performance/recursos' },
  { key: 'high', name: 'Alta', workers: 8, description: 'Máxima velocidade' },
  { key: 'max', name: 'Máxima', workers: 16, description: 'Todos os cores' },
];

// =============================================================================
// TIME PRESETS (for slider)
// =============================================================================

export interface TimePreset {
  seconds: number;
  label: string;
  description: string;
}

export const TIME_PRESETS: TimePreset[] = [
  { seconds: 30, label: '30s', description: 'Debug rápido' },
  { seconds: 60, label: '1 min', description: 'Smoke test' },
  { seconds: 180, label: '3 min', description: 'Exploração rápida' },
  { seconds: 600, label: '10 min', description: 'Análise básica' },
  { seconds: 900, label: '15 min', description: 'Produção (recomendado)' },
  { seconds: 1800, label: '30 min', description: 'Análise profunda' },
  { seconds: 3600, label: '1 hora', description: 'Exploração completa' },
  { seconds: 7200, label: '2 horas', description: 'Overnight' },
];

// =============================================================================
// GATES CONFIGURATION
// =============================================================================

export interface GateConfig {
  key: string;
  name: string;
  description: string;
  type: 'threshold' | 'boolean';
  defaultValue: number | boolean;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  tooltip: string;
}

export const GATES_CONFIG: GateConfig[] = [
  {
    key: 'minOosSharpeNet',
    name: 'Sharpe Mínimo',
    description: 'OOS NET',
    type: 'threshold',
    defaultValue: 0.5,
    min: 0,
    max: 2,
    step: 0.1,
    tooltip: 'Sharpe Ratio mínimo no período Out-of-Sample, já descontando custos. Valores > 0.5 indicam edge consistente.',
  },
  {
    key: 'maxPbo',
    name: 'PBO Máximo',
    description: 'Anti-overfitting',
    type: 'threshold',
    defaultValue: 0.15,
    min: 0,
    max: 0.5,
    step: 0.05,
    tooltip: 'Probability of Backtest Overfitting. Mede chance da estratégia ser "sortuda". Valores < 0.15 indicam baixo risco.',
  },
  {
    key: 'minStressPassed',
    name: 'Stress Mínimo',
    description: 'Testes passados',
    type: 'threshold',
    defaultValue: 4,
    min: 0,
    max: 8,
    step: 1,
    unit: 'de 8',
    tooltip: 'Número mínimo de stress tests que a estratégia deve passar (ex: crise 2008, COVID, volatilidade extrema).',
  },
  {
    key: 'stressTestingEnabled',
    name: 'Stress Testing',
    description: 'Simula cenários extremos',
    type: 'boolean',
    defaultValue: true,
    tooltip: 'Quando habilitado, submete cada estratégia a cenários de stress históricos (crashes, alta volatilidade, etc).',
  },
];

// =============================================================================
// CONVERT TO TAURI FORMAT
// =============================================================================

export interface TauriRunConfig {
  preset: string;
  max_runtime_seconds: number;
  population_size: number;
  max_generations: number;
  convergence_generations: number;
  workers: number;
  seeds: number[];
  stress_testing_enabled: boolean;
  min_oos_sharpe_net: number;
  max_pbo: number;
  min_stress_passed: number;
  campaign_config: string | null;
  run_tag: string | null;
}

export function toTauriConfig(config: CockpitConfig): TauriRunConfig {
  return {
    preset: config.preset,
    max_runtime_seconds: config.maxRuntimeSeconds,
    population_size: config.populationSize,
    max_generations: config.maxGenerations,
    convergence_generations: config.convergenceGenerations,
    workers: config.workers,
    seeds: config.seeds,
    stress_testing_enabled: config.stressTestingEnabled,
    min_oos_sharpe_net: config.minOosSharpeNet,
    max_pbo: config.maxPbo,
    min_stress_passed: config.minStressPassed,
    campaign_config: config.campaignConfig ?? null,
    run_tag: config.runTag ?? null,
  };
}


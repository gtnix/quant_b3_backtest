/**
 * StrategyConfigModal - Modal de configuração de parâmetros por família
 * 
 * Permite ajustar os bounds de parâmetros para cada família de estratégia.
 * Inclui presets (Conservador/Padrão/Agressivo), validações cruzadas,
 * dependências entre campos e tooltips com referências.
 */

import { useState, useEffect, useMemo } from 'react';
import { X, HelpCircle, RotateCcw, Save, Shield, Target, Zap, AlertCircle } from 'lucide-react';

// =============================================================================
// TIPOS
// =============================================================================

interface ParameterConfig {
  type: 'integer' | 'float' | 'boolean' | 'enum';
  min?: number;
  max?: number;
  step?: number;
  default: number | boolean | string;
  values?: string[];
  description: string;
  tooltip: string;
  unit?: string;
  dependsOn?: { field: string; value: boolean | string };
  validates?: { rule: 'lt' | 'gt' | 'lte' | 'gte'; field: string; message: string };
}

interface FamilyParameters {
  [key: string]: ParameterConfig;
}

interface PresetValues {
  [key: string]: number | boolean | string;
}

interface FamilyPresets {
  conservador: PresetValues;
  padrao: PresetValues;
  agressivo: PresetValues;
}

interface ValidationError {
  field: string;
  message: string;
}

interface StrategyConfigModalProps {
  isOpen: boolean;
  onClose: () => void;
  family: string;
  familyLabel: string;
  initialValues?: Record<string, number | boolean | string>;
  onSave?: (params: Record<string, number | boolean | string>) => void;
}

// =============================================================================
// PRESETS POR FAMÍLIA
// =============================================================================

const FAMILY_PRESETS: Record<string, FamilyPresets> = {
  swing: {
    conservador: {
      lookback_days: 20,
      holding_days: 10,
      rsi_period: 14,
      rsi_oversold: 25,
      rsi_overbought: 75,
      stop_loss_pct: 0.08,
      take_profit_pct: 0.15,
      trend_filter_enabled: true,
    },
    padrao: {
      lookback_days: 14,
      holding_days: 7,
      rsi_period: 14,
      rsi_oversold: 30,
      rsi_overbought: 70,
      stop_loss_pct: 0.05,
      take_profit_pct: 0.10,
      trend_filter_enabled: true,
    },
    agressivo: {
      lookback_days: 10,
      holding_days: 5,
      rsi_period: 9,
      rsi_oversold: 35,
      rsi_overbought: 65,
      stop_loss_pct: 0.03,
      take_profit_pct: 0.08,
      trend_filter_enabled: false,
    },
  },
  position: {
    conservador: {
      lookback_weeks: 12,
      holding_weeks: 8,
      momentum_threshold: 0.15,
      max_positions: 5,
      stop_loss_pct: 0.15,
      rebalance_frequency: 'monthly',
    },
    padrao: {
      lookback_weeks: 8,
      holding_weeks: 4,
      momentum_threshold: 0.10,
      max_positions: 10,
      stop_loss_pct: 0.10,
      rebalance_frequency: 'monthly',
    },
    agressivo: {
      lookback_weeks: 4,
      holding_weeks: 2,
      momentum_threshold: 0.05,
      max_positions: 15,
      stop_loss_pct: 0.05,
      rebalance_frequency: 'weekly',
    },
  },
  intraday: {
    conservador: {
      orb_range_minutes: 30,
      orb_breakout_threshold: 0.008,
      vwap_deviation_bands: 2.5,
      rsi_period: 14,
      rsi_oversold: 25,
      rsi_overbought: 75,
      stop_loss_atr_mult: 2.0,
      take_profit_atr_mult: 3.0,
      max_holding_hours: 4,
    },
    padrao: {
      orb_range_minutes: 15,
      orb_breakout_threshold: 0.005,
      vwap_deviation_bands: 2.0,
      rsi_period: 14,
      rsi_oversold: 30,
      rsi_overbought: 70,
      stop_loss_atr_mult: 1.5,
      take_profit_atr_mult: 2.5,
      max_holding_hours: 6,
    },
    agressivo: {
      orb_range_minutes: 5,
      orb_breakout_threshold: 0.003,
      vwap_deviation_bands: 1.5,
      rsi_period: 7,
      rsi_oversold: 35,
      rsi_overbought: 65,
      stop_loss_atr_mult: 1.0,
      take_profit_atr_mult: 2.0,
      max_holding_hours: 8,
    },
  },
  portfolio: {
    conservador: {
      num_assets: 20,
      rebalance_frequency: 'quarterly',
      rebalance_threshold: 0.07,
      leverage_limit: 1.0,
      allow_short: false,
      max_weight: 0.15,
      max_sector_weight: 0.25,
      target_volatility: 0.08,
    },
    padrao: {
      num_assets: 10,
      rebalance_frequency: 'monthly',
      rebalance_threshold: 0.05,
      leverage_limit: 1.0,
      allow_short: false,
      max_weight: 0.25,
      max_sector_weight: 0.30,
      target_volatility: 0.12,
    },
    agressivo: {
      num_assets: 5,
      rebalance_frequency: 'weekly',
      rebalance_threshold: 0.03,
      leverage_limit: 1.5,
      allow_short: true,
      max_weight: 0.40,
      max_sector_weight: 0.50,
      target_volatility: 0.20,
    },
  },
  pair: {
    conservador: {
      cointegration_lookback: 504,
      cointegration_pvalue: 0.01,
      zscore_entry: 2.5,
      zscore_exit: 0.25,
      zscore_stop: 3.5,
      spread_lookback: 90,
      correlation_threshold: 0.90,
      max_holding_days: 60,
    },
    padrao: {
      cointegration_lookback: 252,
      cointegration_pvalue: 0.05,
      zscore_entry: 2.0,
      zscore_exit: 0.5,
      zscore_stop: 3.0,
      spread_lookback: 60,
      correlation_threshold: 0.80,
      max_holding_days: 30,
    },
    agressivo: {
      cointegration_lookback: 126,
      cointegration_pvalue: 0.10,
      zscore_entry: 1.5,
      zscore_exit: 0.75,
      zscore_stop: 2.5,
      spread_lookback: 30,
      correlation_threshold: 0.70,
      max_holding_days: 15,
    },
  },
  momentum: {
    conservador: {
      lookback_months: 12,
      skip_recent_month: true,
      holding_months: 3,
      top_percentile: 0.30,
      bottom_percentile: 0,
      trend_filter_enabled: true,
      trend_filter_ma: 200,
    },
    padrao: {
      lookback_months: 12,
      skip_recent_month: true,
      holding_months: 1,
      top_percentile: 0.20,
      bottom_percentile: 0,
      trend_filter_enabled: true,
      trend_filter_ma: 200,
    },
    agressivo: {
      lookback_months: 6,
      skip_recent_month: false,
      holding_months: 1,
      top_percentile: 0.10,
      bottom_percentile: 0.10,
      trend_filter_enabled: false,
      trend_filter_ma: 100,
    },
  },
  mean_reversion: {
    conservador: {
      bb_period: 20,
      bb_std_dev: 2.5,
      rsi_period: 14,
      rsi_oversold: 25,
      rsi_overbought: 75,
      exit_at_middle: true,
      max_holding_days: 15,
    },
    padrao: {
      bb_period: 20,
      bb_std_dev: 2.0,
      rsi_period: 14,
      rsi_oversold: 30,
      rsi_overbought: 70,
      exit_at_middle: true,
      max_holding_days: 10,
    },
    agressivo: {
      bb_period: 10,
      bb_std_dev: 1.5,
      rsi_period: 7,
      rsi_oversold: 35,
      rsi_overbought: 65,
      exit_at_middle: false,
      max_holding_days: 5,
    },
  },
  breakout: {
    conservador: {
      channel_period: 55,
      exit_channel_period: 20,
      atr_period: 14,
      atr_stop_mult: 2.5,
      atr_target_mult: 5.0,
      volume_confirmation: true,
      volume_multiplier: 2.0,
      trailing_stop: true,
      trailing_atr_mult: 3.0,
    },
    padrao: {
      channel_period: 20,
      exit_channel_period: 10,
      atr_period: 14,
      atr_stop_mult: 2.0,
      atr_target_mult: 4.0,
      volume_confirmation: true,
      volume_multiplier: 1.5,
      trailing_stop: true,
      trailing_atr_mult: 2.5,
    },
    agressivo: {
      channel_period: 10,
      exit_channel_period: 5,
      atr_period: 10,
      atr_stop_mult: 1.5,
      atr_target_mult: 3.0,
      volume_confirmation: false,
      volume_multiplier: 1.2,
      trailing_stop: false,
      trailing_atr_mult: 2.0,
    },
  },
};

// =============================================================================
// PARÂMETROS POR FAMÍLIA (com validações e dependências)
// =============================================================================

const FAMILY_PARAMETERS: Record<string, FamilyParameters> = {
  swing: {
    lookback_days: {
      type: 'integer',
      min: 5,
      max: 30,
      step: 1,
      default: 14,
      unit: 'dias',
      description: 'Lookback',
      tooltip: 'Período para calcular indicadores de entrada. 10-20 dias típico para swing trading',
    },
    holding_days: {
      type: 'integer',
      min: 2,
      max: 15,
      step: 1,
      default: 7,
      unit: 'dias',
      description: 'Holding Máximo',
      tooltip: 'Dias máximos para manter a posição. Swing trading tipicamente 2-10 dias',
    },
    rsi_period: {
      type: 'integer',
      min: 7,
      max: 21,
      step: 2,
      default: 14,
      unit: 'barras',
      description: 'Período RSI',
      tooltip: 'Período do RSI. 14 é o padrão de Wilder',
    },
    rsi_oversold: {
      type: 'integer',
      min: 20,
      max: 40,
      step: 5,
      default: 30,
      unit: '',
      description: 'RSI Sobrevendido',
      tooltip: 'Nível para considerar ativo sobrevendido',
      validates: { rule: 'lt', field: 'rsi_overbought', message: 'Deve ser menor que RSI Sobrecomprado' },
    },
    rsi_overbought: {
      type: 'integer',
      min: 60,
      max: 80,
      step: 5,
      default: 70,
      unit: '',
      description: 'RSI Sobrecomprado',
      tooltip: 'Nível para considerar ativo sobrecomprado',
      validates: { rule: 'gt', field: 'rsi_oversold', message: 'Deve ser maior que RSI Sobrevendido' },
    },
    stop_loss_pct: {
      type: 'float',
      min: 0.02,
      max: 0.15,
      step: 0.01,
      default: 0.05,
      unit: '%',
      description: 'Stop Loss',
      tooltip: 'Percentual de perda máxima por trade',
      validates: { rule: 'lt', field: 'take_profit_pct', message: 'Deve ser menor que Take Profit' },
    },
    take_profit_pct: {
      type: 'float',
      min: 0.05,
      max: 0.25,
      step: 0.01,
      default: 0.10,
      unit: '%',
      description: 'Take Profit',
      tooltip: 'Percentual de ganho alvo por trade',
      validates: { rule: 'gt', field: 'stop_loss_pct', message: 'Deve ser maior que Stop Loss' },
    },
    trend_filter_enabled: {
      type: 'boolean',
      default: true,
      description: 'Filtro de Tendência',
      tooltip: 'Só opera na direção da tendência principal (MA 50/200)',
    },
  },
  position: {
    lookback_weeks: {
      type: 'integer',
      min: 4,
      max: 24,
      step: 2,
      default: 8,
      unit: 'semanas',
      description: 'Lookback',
      tooltip: 'Período para análise de tendência. Position trading usa períodos mais longos',
    },
    holding_weeks: {
      type: 'integer',
      min: 2,
      max: 12,
      step: 1,
      default: 4,
      unit: 'semanas',
      description: 'Holding Típico',
      tooltip: 'Duração típica das posições. Position = semanas a meses',
    },
    momentum_threshold: {
      type: 'float',
      min: 0.05,
      max: 0.25,
      step: 0.05,
      default: 0.10,
      unit: '%',
      description: 'Threshold Momentum',
      tooltip: 'Retorno mínimo no período de lookback para considerar entrada',
    },
    max_positions: {
      type: 'integer',
      min: 3,
      max: 20,
      step: 1,
      default: 10,
      unit: 'posições',
      description: 'Máx Posições',
      tooltip: 'Número máximo de posições simultâneas no portfólio',
    },
    stop_loss_pct: {
      type: 'float',
      min: 0.05,
      max: 0.25,
      step: 0.01,
      default: 0.10,
      unit: '%',
      description: 'Stop Loss',
      tooltip: 'Percentual de perda máxima. Position trading usa stops mais amplos',
    },
    rebalance_frequency: {
      type: 'enum',
      values: ['weekly', 'bi_weekly', 'monthly'],
      default: 'monthly',
      description: 'Frequência Rebalanceamento',
      tooltip: 'Com que frequência revisar e ajustar posições',
    },
  },
  intraday: {
    orb_range_minutes: {
      type: 'integer',
      min: 5,
      max: 60,
      step: 5,
      default: 15,
      unit: 'min',
      description: 'Período ORB',
      tooltip: 'Janela de tempo após abertura para definir o range. 15-30min popular para ações, 5min para scalping. Fonte: Fortunly ORB Guide',
    },
    orb_breakout_threshold: {
      type: 'float',
      min: 0.001,
      max: 0.02,
      step: 0.001,
      default: 0.005,
      unit: '%',
      description: 'Threshold Breakout',
      tooltip: 'Percentual além do range para confirmar breakout. Valores menores = mais trades, mais ruído. Fonte: Fortunly',
    },
    vwap_deviation_bands: {
      type: 'float',
      min: 1.0,
      max: 3.0,
      step: 0.25,
      default: 2.0,
      unit: 'σ',
      description: 'Bandas VWAP',
      tooltip: 'Desvios padrão do VWAP. ±1σ = 68% preços, ±2σ = 95%, ±3σ = 99.7%. Fonte: VWAP deviation studies',
    },
    rsi_period: {
      type: 'integer',
      min: 5,
      max: 21,
      step: 1,
      default: 14,
      unit: 'barras',
      description: 'Período RSI',
      tooltip: 'Período do RSI. 5-9 para scalping, 14 padrão, 21+ para swing. Fonte: Wilder, 1978',
    },
    rsi_oversold: {
      type: 'integer',
      min: 15,
      max: 40,
      step: 5,
      default: 30,
      unit: '',
      description: 'RSI Sobrevendido',
      tooltip: 'Abaixo deste nível = potencial compra. Ajustar para 20 em mercados fortes',
      validates: { rule: 'lt', field: 'rsi_overbought', message: 'Deve ser menor que RSI Sobrecomprado' },
    },
    rsi_overbought: {
      type: 'integer',
      min: 60,
      max: 85,
      step: 5,
      default: 70,
      unit: '',
      description: 'RSI Sobrecomprado',
      tooltip: 'Acima deste nível = potencial venda. Ajustar para 80 em tendências fortes',
      validates: { rule: 'gt', field: 'rsi_oversold', message: 'Deve ser maior que RSI Sobrevendido' },
    },
    stop_loss_atr_mult: {
      type: 'float',
      min: 1.0,
      max: 3.0,
      step: 0.25,
      default: 1.5,
      unit: 'ATR',
      description: 'Stop Loss',
      tooltip: 'Múltiplo do ATR para stop. 1.0 = agressivo, 2.0+ = conservador',
      validates: { rule: 'lt', field: 'take_profit_atr_mult', message: 'Deve ser menor que Take Profit' },
    },
    take_profit_atr_mult: {
      type: 'float',
      min: 1.5,
      max: 5.0,
      step: 0.5,
      default: 2.5,
      unit: 'ATR',
      description: 'Take Profit',
      tooltip: 'Múltiplo do ATR para target. Manter ratio >= 1.5:1 vs stop',
      validates: { rule: 'gt', field: 'stop_loss_atr_mult', message: 'Deve ser maior que Stop Loss' },
    },
    max_holding_hours: {
      type: 'integer',
      min: 1,
      max: 8,
      step: 1,
      default: 6,
      unit: 'h',
      description: 'Holding Máximo',
      tooltip: 'Força saída antes do fechamento. Evita overnight gap risk',
    },
  },
  portfolio: {
    num_assets: {
      type: 'integer',
      min: 5,
      max: 50,
      step: 5,
      default: 10,
      unit: 'ativos',
      description: 'Número de Ativos',
      tooltip: 'Quantidade de ativos no portfólio. 10-20 oferece diversificação adequada sem diluição excessiva',
    },
    rebalance_frequency: {
      type: 'enum',
      values: ['daily', 'weekly', 'monthly', 'quarterly'],
      default: 'monthly',
      description: 'Frequência Rebalanceamento',
      tooltip: 'Frequência de rebalanceamento. Mensal é comum; diário aumenta custos de transação. Fonte: Vanguard Research',
    },
    rebalance_threshold: {
      type: 'float',
      min: 0.01,
      max: 0.10,
      step: 0.01,
      default: 0.05,
      unit: '%',
      description: 'Threshold Rebalanceamento',
      tooltip: 'Desvio mínimo do peso alvo para disparar rebalanceamento. Vanguard: não há diferença significativa entre 1-10%',
    },
    leverage_limit: {
      type: 'float',
      min: 1.0,
      max: 2.0,
      step: 0.25,
      default: 1.0,
      unit: 'x',
      description: 'Limite Alavancagem',
      tooltip: 'Alavancagem máxima permitida. 1.0 = sem alavancagem. Reg T permite até 2x. Fonte: FINRA Rule 4210',
    },
    allow_short: {
      type: 'boolean',
      default: false,
      description: 'Permitir Short',
      tooltip: 'Permite posições vendidas. Necessário para estratégias long/short ou market-neutral',
    },
    max_weight: {
      type: 'float',
      min: 0.05,
      max: 0.50,
      step: 0.05,
      default: 0.25,
      unit: '%',
      description: 'Peso Máximo/Ativo',
      tooltip: 'Peso máximo por ativo. 10-25% comum para diversificação. Evita concentração excessiva',
      validates: { rule: 'lte', field: 'max_sector_weight', message: 'Deve ser <= Peso Máximo/Setor' },
    },
    max_sector_weight: {
      type: 'float',
      min: 0.15,
      max: 0.50,
      step: 0.05,
      default: 0.30,
      unit: '%',
      description: 'Peso Máximo/Setor',
      tooltip: 'Peso máximo por setor. 25-35% típico. Evita risco setorial concentrado',
      validates: { rule: 'gte', field: 'max_weight', message: 'Deve ser >= Peso Máximo/Ativo' },
    },
    target_volatility: {
      type: 'float',
      min: 0.05,
      max: 0.25,
      step: 0.025,
      default: 0.12,
      unit: '% a.a.',
      description: 'Volatilidade Alvo',
      tooltip: 'Volatilidade alvo anualizada. 10% = conservador, 15% = moderado, 20%+ = agressivo. Fonte: Markowitz',
    },
  },
  pair: {
    cointegration_lookback: {
      type: 'integer',
      min: 60,
      max: 504,
      step: 21,
      default: 252,
      unit: 'dias',
      description: 'Lookback Cointegração',
      tooltip: 'Período para teste de cointegração. 252 = 1 ano. Fonte: QuantStart',
    },
    cointegration_pvalue: {
      type: 'float',
      min: 0.01,
      max: 0.10,
      step: 0.01,
      default: 0.05,
      unit: '',
      description: 'P-valor Cointegração',
      tooltip: 'P-valor máximo para aceitar cointegração. 0.05 = 95% confiança',
    },
    zscore_entry: {
      type: 'float',
      min: 1.5,
      max: 3.0,
      step: 0.25,
      default: 2.0,
      unit: 'σ',
      description: 'Z-Score Entrada',
      tooltip: 'Z-score para abrir posição. 2.0 comum, captura ~95% dos movimentos normais. Fonte: Gatev et al.',
      validates: { rule: 'gt', field: 'zscore_exit', message: 'Deve ser maior que Z-Score Saída' },
    },
    zscore_exit: {
      type: 'float',
      min: 0.0,
      max: 1.0,
      step: 0.25,
      default: 0.5,
      unit: 'σ',
      description: 'Z-Score Saída',
      tooltip: 'Z-score para fechar posição. 0 = reversão completa, 0.5 = saída parcial mais rápida',
      validates: { rule: 'lt', field: 'zscore_entry', message: 'Deve ser menor que Z-Score Entrada' },
    },
    zscore_stop: {
      type: 'float',
      min: 2.5,
      max: 4.0,
      step: 0.25,
      default: 3.0,
      unit: 'σ',
      description: 'Z-Score Stop',
      tooltip: 'Z-score para stop loss. Indica possível quebra da relação de cointegração',
      validates: { rule: 'gt', field: 'zscore_entry', message: 'Deve ser maior que Z-Score Entrada' },
    },
    spread_lookback: {
      type: 'integer',
      min: 30,
      max: 120,
      step: 10,
      default: 60,
      unit: 'dias',
      description: 'Lookback Spread',
      tooltip: 'Janela para cálculo de média/desvio do spread. 60 = responsivo, 120 = estável',
    },
    correlation_threshold: {
      type: 'float',
      min: 0.60,
      max: 0.95,
      step: 0.05,
      default: 0.80,
      unit: '',
      description: 'Correlação Mínima',
      tooltip: 'Correlação mínima entre ativos. 0.8+ recomendado para pares estáveis',
    },
    max_holding_days: {
      type: 'integer',
      min: 10,
      max: 90,
      step: 5,
      default: 30,
      unit: 'dias',
      description: 'Holding Máximo',
      tooltip: 'Força saída se spread não convergir. Evita capital preso indefinidamente',
    },
  },
  momentum: {
    lookback_months: {
      type: 'integer',
      min: 3,
      max: 12,
      step: 1,
      default: 12,
      unit: 'meses',
      description: 'Lookback',
      tooltip: 'Período para calcular retorno. 12m clássico (Jegadeesh-Titman), 6m mais responsivo',
    },
    skip_recent_month: {
      type: 'boolean',
      default: true,
      description: 'Pular Mês Recente',
      tooltip: 'Exclui mês mais recente. Evita reversão de curto prazo. Padrão acadêmico. Fonte: Jegadeesh-Titman 1993',
    },
    holding_months: {
      type: 'integer',
      min: 1,
      max: 6,
      step: 1,
      default: 1,
      unit: 'meses',
      description: 'Holding',
      tooltip: 'Período de holding antes de recalcular ranking. 1m = mais reativo',
    },
    top_percentile: {
      type: 'float',
      min: 0.05,
      max: 0.30,
      step: 0.05,
      default: 0.20,
      unit: '%',
      description: 'Top Percentil',
      tooltip: 'Percentual dos ativos com melhor momentum para comprar. Top 10% = concentrado, 20% = equilibrado',
    },
    bottom_percentile: {
      type: 'float',
      min: 0,
      max: 0.30,
      step: 0.05,
      default: 0,
      unit: '%',
      description: 'Bottom Percentil',
      tooltip: 'Percentual para short se habilitado. 0 = long-only',
    },
    trend_filter_enabled: {
      type: 'boolean',
      default: true,
      description: 'Filtro de Tendência',
      tooltip: 'Filtra entradas contra a tendência. Recomendado para evitar momentum crashes. Fonte: Alpha Architect',
    },
    trend_filter_ma: {
      type: 'integer',
      min: 100,
      max: 252,
      step: 50,
      default: 200,
      unit: 'dias',
      description: 'MA Filtro Tendência',
      tooltip: 'Média móvel para filtro de tendência. 200d é padrão institucional',
      dependsOn: { field: 'trend_filter_enabled', value: true },
    },
  },
  mean_reversion: {
    bb_period: {
      type: 'integer',
      min: 10,
      max: 50,
      step: 5,
      default: 20,
      unit: 'barras',
      description: 'Período Bollinger',
      tooltip: 'Período da média móvel das Bandas de Bollinger. 20 é padrão de John Bollinger',
    },
    bb_std_dev: {
      type: 'float',
      min: 1.5,
      max: 3.0,
      step: 0.25,
      default: 2.0,
      unit: 'σ',
      description: 'Desvios Bollinger',
      tooltip: 'Desvios padrão para as bandas. 2.0 captura 95% dos movimentos normais. Fonte: Bollinger 2001',
    },
    rsi_period: {
      type: 'integer',
      min: 7,
      max: 21,
      step: 2,
      default: 14,
      unit: 'barras',
      description: 'Período RSI',
      tooltip: 'Período do RSI para confirmação. 14 é padrão de Wilder',
    },
    rsi_oversold: {
      type: 'integer',
      min: 15,
      max: 40,
      step: 5,
      default: 30,
      unit: '',
      description: 'RSI Sobrevendido',
      tooltip: 'RSI abaixo = condição de sobrevenda. Combinar com BB inferior para entrada long',
      validates: { rule: 'lt', field: 'rsi_overbought', message: 'Deve ser menor que RSI Sobrecomprado' },
    },
    rsi_overbought: {
      type: 'integer',
      min: 60,
      max: 85,
      step: 5,
      default: 70,
      unit: '',
      description: 'RSI Sobrecomprado',
      tooltip: 'RSI acima = condição de sobrecompra. Combinar com BB superior para entrada short',
      validates: { rule: 'gt', field: 'rsi_oversold', message: 'Deve ser maior que RSI Sobrevendido' },
    },
    exit_at_middle: {
      type: 'boolean',
      default: true,
      description: 'Sair na Média',
      tooltip: 'Sair quando preço cruza a média móvel central das BB',
    },
    max_holding_days: {
      type: 'integer',
      min: 3,
      max: 20,
      step: 1,
      default: 10,
      unit: 'dias',
      description: 'Holding Máximo',
      tooltip: 'Holding máximo. Mean reversion deve ocorrer rapidamente; trades longos indicam falha',
    },
  },
  breakout: {
    channel_period: {
      type: 'integer',
      min: 10,
      max: 55,
      step: 5,
      default: 20,
      unit: 'barras',
      description: 'Período Canal',
      tooltip: 'Período do canal Donchian. 20 = curto prazo, 55 = Turtle Traders original. Fonte: Faith 2007',
      validates: { rule: 'gt', field: 'exit_channel_period', message: 'Deve ser maior que Período Saída' },
    },
    exit_channel_period: {
      type: 'integer',
      min: 5,
      max: 30,
      step: 5,
      default: 10,
      unit: 'barras',
      description: 'Período Saída',
      tooltip: 'Período para sinal de saída (menor que entrada para capturar reversões)',
      validates: { rule: 'lt', field: 'channel_period', message: 'Deve ser menor que Período Canal' },
    },
    atr_period: {
      type: 'integer',
      min: 10,
      max: 20,
      step: 2,
      default: 14,
      unit: 'barras',
      description: 'Período ATR',
      tooltip: 'Período do ATR para cálculo de stops',
    },
    atr_stop_mult: {
      type: 'float',
      min: 1.0,
      max: 3.0,
      step: 0.25,
      default: 2.0,
      unit: 'ATR',
      description: 'Stop Loss',
      tooltip: 'Multiplicador ATR para stop loss. 1.5-2.0 típico',
      validates: { rule: 'lt', field: 'atr_target_mult', message: 'Deve ser menor que Take Profit' },
    },
    atr_target_mult: {
      type: 'float',
      min: 2.0,
      max: 8.0,
      step: 0.5,
      default: 4.0,
      unit: 'ATR',
      description: 'Take Profit',
      tooltip: 'Multiplicador ATR para take profit. Ratio 2:1+ vs stop',
      validates: { rule: 'gt', field: 'atr_stop_mult', message: 'Deve ser maior que Stop Loss' },
    },
    volume_confirmation: {
      type: 'boolean',
      default: true,
      description: 'Confirmação Volume',
      tooltip: 'Exigir volume acima da média para confirmar breakout. Fonte: AvaTrade',
    },
    volume_multiplier: {
      type: 'float',
      min: 1.2,
      max: 2.5,
      step: 0.1,
      default: 1.5,
      unit: 'x',
      description: 'Multiplicador Volume',
      tooltip: 'Volume mínimo como múltiplo da média. 1.5 = 50% acima',
      dependsOn: { field: 'volume_confirmation', value: true },
    },
    trailing_stop: {
      type: 'boolean',
      default: true,
      description: 'Trailing Stop',
      tooltip: 'Usar trailing stop para proteger lucros em tendências',
    },
    trailing_atr_mult: {
      type: 'float',
      min: 1.5,
      max: 4.0,
      step: 0.5,
      default: 2.5,
      unit: 'ATR',
      description: 'Trailing ATR',
      tooltip: 'Distância do trailing stop em ATRs',
      dependsOn: { field: 'trailing_stop', value: true },
    },
  },
};

// Labels traduzidos para valores de enum
const ENUM_LABELS: Record<string, Record<string, string>> = {
  rebalance_frequency: {
    daily: 'Diário',
    weekly: 'Semanal',
    bi_weekly: 'Quinzenal',
    monthly: 'Mensal',
    quarterly: 'Trimestral',
  },
};

// =============================================================================
// FUNÇÕES AUXILIARES
// =============================================================================

function getInitialValues(parameters: FamilyParameters): Record<string, number | boolean | string> {
  const initial: Record<string, number | boolean | string> = {};
  Object.entries(parameters).forEach(([key, param]) => {
    initial[key] = param.default;
  });
  return initial;
}

function validateValues(
  values: Record<string, number | boolean | string>,
  parameters: FamilyParameters
): ValidationError[] {
  const errors: ValidationError[] = [];
  
  Object.entries(parameters).forEach(([key, param]) => {
    if (param.validates) {
      const currentValue = values[key] as number;
      const compareValue = values[param.validates.field] as number;
      
      let isValid = true;
      switch (param.validates.rule) {
        case 'lt':
          isValid = currentValue < compareValue;
          break;
        case 'gt':
          isValid = currentValue > compareValue;
          break;
        case 'lte':
          isValid = currentValue <= compareValue;
          break;
        case 'gte':
          isValid = currentValue >= compareValue;
          break;
      }
      
      if (!isValid) {
        errors.push({ field: key, message: param.validates.message });
      }
    }
  });
  
  return errors;
}

function isFieldDisabled(
  key: string,
  values: Record<string, number | boolean | string>,
  parameters: FamilyParameters
): boolean {
  const param = parameters[key];
  if (!param.dependsOn) return false;
  
  const dependencyValue = values[param.dependsOn.field];
  return dependencyValue !== param.dependsOn.value;
}

function formatValue(value: number | boolean | string | undefined, param: ParameterConfig): string {
  if (value === undefined || value === null) {
    return '-';
  }
  if (param.type === 'boolean') {
    return value ? 'Sim' : 'Não';
  }
  if (param.type === 'enum') {
    return value as string;
  }
  if (param.type === 'float') {
    const numVal = value as number;
    if (typeof numVal !== 'number' || isNaN(numVal)) return '-';
    // Format percentages nicely
    if (param.unit === '%' && numVal < 1) {
      return `${(numVal * 100).toFixed(1)}%`;
    }
    return numVal.toFixed(2);
  }
  return String(value);
}

// =============================================================================
// COMPONENTE
// =============================================================================

export function StrategyConfigModal({ 
  isOpen, 
  onClose, 
  family, 
  familyLabel, 
  initialValues,
  onSave 
}: StrategyConfigModalProps) {
  const parameters = FAMILY_PARAMETERS[family] || {};
  const presets = FAMILY_PRESETS[family];
  
  // Estado para os valores dos parâmetros
  const [values, setValues] = useState<Record<string, number | boolean | string>>(() => {
    if (initialValues && Object.keys(initialValues).length > 0) {
      return { ...getInitialValues(parameters), ...initialValues };
    }
    return getInitialValues(parameters);
  });
  
  // Estado para preset ativo
  const [activePreset, setActivePreset] = useState<'conservador' | 'padrao' | 'agressivo' | null>('padrao');

  // Reset values when family changes
  useEffect(() => {
    if (isOpen) {
      const newParams = FAMILY_PARAMETERS[family] || {};
      if (initialValues && Object.keys(initialValues).length > 0) {
        setValues({ ...getInitialValues(newParams), ...initialValues });
      } else {
        setValues(getInitialValues(newParams));
      }
      setActivePreset('padrao');
    }
  }, [family, isOpen, initialValues]);

  // Validação em tempo real
  const validationErrors = useMemo(() => {
    return validateValues(values, parameters);
  }, [values, parameters]);
  
  const hasErrors = validationErrors.length > 0;
  
  const getFieldError = (key: string): string | undefined => {
    const error = validationErrors.find(e => e.field === key);
    return error?.message;
  };

  const handleChange = (key: string, value: number | boolean | string) => {
    setValues(prev => ({ ...prev, [key]: value }));
    setActivePreset(null); // Clear preset when user manually changes
  };

  const handleReset = () => {
    setValues(getInitialValues(parameters));
    setActivePreset('padrao');
  };
  
  const handleApplyPreset = (presetName: 'conservador' | 'padrao' | 'agressivo') => {
    if (presets && presets[presetName]) {
      // Merge preset with defaults to ensure all fields are set
      const defaults = getInitialValues(parameters);
      setValues({ ...defaults, ...presets[presetName] });
      setActivePreset(presetName);
    }
  };

  const handleSave = () => {
    if (hasErrors) return;
    onSave?.(values);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Modal */}
      <div className="relative w-full max-w-2xl max-h-[85vh] bg-slate-900 border border-slate-700 rounded-xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-slate-700 bg-slate-800/50">
          <div>
            <h2 className="text-lg font-semibold text-white">
              Configurar Parâmetros - {familyLabel}
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Ajuste os limites de parâmetros para esta família de estratégia
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-slate-400" />
          </button>
        </div>
        
        {/* Presets Bar */}
        {presets && (
          <div className="flex items-center gap-2 px-4 py-3 border-b border-slate-700/50 bg-slate-800/30">
            <span className="text-xs text-slate-500 mr-2">Presets:</span>
            <button
              onClick={() => handleApplyPreset('conservador')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activePreset === 'conservador'
                  ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/50'
                  : 'bg-slate-800 text-slate-400 border border-slate-600 hover:border-emerald-500/50 hover:text-emerald-400'
              }`}
            >
              <Shield className="w-3.5 h-3.5" />
              Conservador
            </button>
            <button
              onClick={() => handleApplyPreset('padrao')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activePreset === 'padrao'
                  ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50'
                  : 'bg-slate-800 text-slate-400 border border-slate-600 hover:border-cyan-500/50 hover:text-cyan-400'
              }`}
            >
              <Target className="w-3.5 h-3.5" />
              Padrão
            </button>
            <button
              onClick={() => handleApplyPreset('agressivo')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activePreset === 'agressivo'
                  ? 'bg-orange-500/20 text-orange-400 border border-orange-500/50'
                  : 'bg-slate-800 text-slate-400 border border-slate-600 hover:border-orange-500/50 hover:text-orange-400'
              }`}
            >
              <Zap className="w-3.5 h-3.5" />
              Agressivo
            </button>
          </div>
        )}

        {/* Content */}
        <div className="p-4 overflow-y-auto max-h-[calc(85vh-200px)] space-y-3">
          {Object.keys(parameters).length === 0 ? (
            <div className="text-center py-8 text-slate-500">
              <p>Parâmetros não configurados para esta família.</p>
              <p className="text-xs mt-1">Usando valores padrão do sistema.</p>
            </div>
          ) : (
            Object.entries(parameters).map(([key, param]) => {
              const disabled = isFieldDisabled(key, values, parameters);
              const error = getFieldError(key);
              
              return (
                <div 
                  key={key} 
                  className={`p-3 rounded-lg border transition-all ${
                    disabled 
                      ? 'bg-slate-900/50 border-slate-800 opacity-50' 
                      : error
                      ? 'bg-red-500/5 border-red-500/30'
                      : 'bg-slate-800/50 border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <label className={`text-sm font-medium ${disabled ? 'text-slate-500' : 'text-white'}`}>
                        {param.description}
                      </label>
                      {param.unit && (
                        <span className="text-xs text-slate-500 bg-slate-700/50 px-1.5 py-0.5 rounded">
                          {param.unit}
                        </span>
                      )}
                      <div className="group relative">
                        <HelpCircle className={`w-4 h-4 cursor-help ${disabled ? 'text-slate-600' : 'text-slate-500 hover:text-cyan-400'}`} />
                        <div className="absolute left-0 bottom-full mb-2 w-80 p-3 bg-slate-700 border border-slate-600 rounded-lg shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50 pointer-events-none">
                          <p className="text-xs text-slate-200">{param.tooltip}</p>
                        </div>
                      </div>
                      {error && (
                        <div className="flex items-center gap-1 text-red-400">
                          <AlertCircle className="w-3.5 h-3.5" />
                          <span className="text-xs">{error}</span>
                        </div>
                      )}
                    </div>
                    <span className={`text-sm font-mono ${error ? 'text-red-400' : 'text-cyan-400'}`}>
                      {param.type === 'enum'
                        ? ENUM_LABELS[key]?.[values[key] as string] || values[key]
                        : formatValue(values[key], param)}
                    </span>
                  </div>

                  {/* Input baseado no tipo */}
                  {param.type === 'boolean' ? (
                    <button
                      onClick={() => !disabled && handleChange(key, !values[key])}
                      disabled={disabled}
                      className={`w-full p-2 rounded-lg border text-sm font-medium transition-colors ${
                        disabled
                          ? 'bg-slate-900 border-slate-700 text-slate-600 cursor-not-allowed'
                          : values[key]
                          ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-400'
                          : 'bg-slate-800 border-slate-600 text-slate-400 hover:border-slate-500'
                      }`}
                    >
                      {values[key] ? 'Habilitado' : 'Desabilitado'}
                    </button>
                  ) : param.type === 'enum' ? (
                    <div className="flex gap-2">
                      {param.values?.map(val => (
                        <button
                          key={val}
                          onClick={() => !disabled && handleChange(key, val)}
                          disabled={disabled}
                          className={`flex-1 p-2 rounded-lg border text-xs font-medium transition-colors ${
                            disabled
                              ? 'bg-slate-900 border-slate-700 text-slate-600 cursor-not-allowed'
                              : values[key] === val
                              ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-400'
                              : 'bg-slate-800 border-slate-600 text-slate-400 hover:border-slate-500'
                          }`}
                        >
                          {ENUM_LABELS[key]?.[val] || val}
                        </button>
                      ))}
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <input
                        type="range"
                        min={param.min}
                        max={param.max}
                        step={param.step}
                        value={values[key] as number}
                        onChange={(e) => handleChange(key, parseFloat(e.target.value))}
                        disabled={disabled}
                        className={`w-full h-2 rounded-lg appearance-none cursor-pointer ${
                          disabled 
                            ? 'bg-slate-800 accent-slate-600' 
                            : error
                            ? 'bg-red-900/30 accent-red-500'
                            : 'bg-slate-700 accent-cyan-500'
                        }`}
                      />
                      <div className="flex justify-between text-xs text-slate-500">
                        <span>{param.min}{param.unit === '%' && param.min !== undefined && param.min < 1 ? '%' : ''}</span>
                        <span className="text-slate-600">Padrão: {formatValue(param.default, param)}</span>
                        <span>{param.max}{param.unit === '%' && param.max !== undefined && param.max < 1 ? '%' : ''}</span>
                      </div>
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-4 border-t border-slate-700 bg-slate-800/50">
          <button
            onClick={handleReset}
            className="flex items-center gap-2 px-4 py-2 text-slate-400 hover:text-white transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            Restaurar Padrões
          </button>
          <div className="flex items-center gap-3">
            {hasErrors && (
              <span className="text-xs text-red-400 flex items-center gap-1">
                <AlertCircle className="w-3.5 h-3.5" />
                {validationErrors.length} erro(s) de validação
              </span>
            )}
            <button
              onClick={onClose}
              className="px-4 py-2 text-slate-400 hover:text-white transition-colors"
            >
              Cancelar
            </button>
            <button
              onClick={handleSave}
              disabled={hasErrors}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                hasErrors
                  ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
                  : 'bg-cyan-600 hover:bg-cyan-500 text-white'
              }`}
            >
              <Save className="w-4 h-4" />
              Aplicar
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default StrategyConfigModal;

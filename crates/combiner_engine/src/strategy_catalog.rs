//! Strategy Catalog - Fonte ÚNICA de templates para o GA.
//!
//! Templates definem a ESTRUTURA (quais blocos usar).
//! O GA evolui apenas os PARÂMETROS dentro dos ranges do ParamRanges.
//!
//! # Fontes de Templates
//! 
//! 1. **from_toml_dir**: Carrega de arquivos TOML (legado)
//! 2. **from_builtin**: Carrega 116 estratégias built-in baseadas no banco de dados
//!
//! Cada estratégia no DB tem um `type` que mapeia para blocos específicos:
//! - `orb_breakout` -> ORBBreakoutBlock
//! - `vwap_reversion` -> VWAPBlock (reversion mode)
//! - `donchian` -> DonchianChannelBlock
//! - etc.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use combiner_core::{BlockGene, BlockType, DataColumn, ParamRanges, ParamValue, StrategyGenome};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::Deserialize;
use thiserror::Error;
use tracing::{debug, info, warn};

/// Erros do Strategy Catalog.
#[derive(Debug, Error)]
pub enum CatalogError {
    #[error("No templates found in {0}")]
    Empty(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error in {path}: {msg}")]
    Parse { path: String, msg: String },
}

/// Bloco dentro de um template (estrutura fixa).
#[derive(Debug, Clone, Deserialize)]
pub struct TemplateBlock {
    #[serde(rename = "type")]
    pub block_type_str: String,
    pub block_id: String,
    #[serde(default)]
    pub params: HashMap<String, toml::Value>,
}

impl TemplateBlock {
    /// Converte string para BlockType.
    pub fn block_type(&self) -> BlockType {
        match self.block_type_str.as_str() {
            "selection" => BlockType::Selection,
            "entry" => BlockType::Entry,
            "exit" => BlockType::Exit,
            "sizing" => BlockType::Sizing,
            _ => BlockType::Selection,
        }
    }
}

/// Template de estratégia carregado do TOML.
#[derive(Debug, Clone)]
pub struct StrategyTemplate {
    /// Identificador único (nome do arquivo sem extensão).
    pub slug: String,
    /// Nome/descrição do template.
    pub name: String,
    /// Pipeline de blocos que compõem a estratégia.
    pub pipeline: Vec<TemplateBlock>,
    /// Caminho do arquivo fonte.
    pub source_path: String,
}

impl StrategyTemplate {
    /// Calcula colunas de dados necessárias para este template.
    pub fn required_columns(&self, param_ranges: &ParamRanges) -> HashSet<DataColumn> {
        self.pipeline
            .iter()
            .filter_map(|block| param_ranges.get_block(&block.block_id))
            .flat_map(|spec| spec.required_columns.iter().copied())
            .collect()
    }

    /// Verifica se template é compatível com colunas disponíveis.
    pub fn is_compatible(
        &self,
        available: &HashSet<DataColumn>,
        param_ranges: &ParamRanges,
    ) -> bool {
        self.required_columns(param_ranges)
            .iter()
            .all(|col| col.is_ohlcv() || available.contains(col))
    }
}

/// Catálogo de estratégias - ÚNICA fonte para geração de genomas.
#[derive(Debug, Clone)]
pub struct StrategyCatalog {
    templates: Vec<StrategyTemplate>,
}

impl Default for StrategyCatalog {
    fn default() -> Self {
        Self {
            templates: Vec::new(),
        }
    }
}

impl StrategyCatalog {
    /// Cria um catálogo vazio.
    pub fn new() -> Self {
        Self::default()
    }

    /// Cria catálogo com todos os 116 templates built-in baseados no banco de dados.
    /// 
    /// Esta é a forma RECOMENDADA de inicializar o catálogo.
    /// Cada tipo de estratégia do DB é mapeado para sua pipeline de blocos.
    pub fn from_builtin() -> Self {
        let templates = Self::create_builtin_templates();
        info!("Strategy Catalog: loaded {} built-in templates", templates.len());
        Self { templates }
    }

    /// Cria todos os templates built-in baseados no seed SQL.
    fn create_builtin_templates() -> Vec<StrategyTemplate> {
        let mut templates = Vec::with_capacity(116);

        // ==========================================
        // INTRADAY STRATEGIES (22)
        // ==========================================
        
        // ORB Breakout (3 variants)
        for (slug, name, _risk) in [
            ("orb_breakout_conservative", "ORB Breakout Conservador", "conservative"),
            ("orb_breakout_moderate", "ORB Breakout Moderado", "moderate"),
            ("orb_breakout_aggressive", "ORB Breakout Agressivo", "aggressive"),
        ] {
            templates.push(Self::make_template(slug, name, vec![
                ("selection", "momentum"),
                ("entry", "orb_breakout"),
                ("exit", "stop_loss"),
                ("sizing", "equal_weight"),
            ]));
        }

        // VWAP Mean Reversion (2 variants)
        for (slug, name) in [
            ("vwap_mean_reversion_conservative", "VWAP Reversão Conservador"),
            ("vwap_mean_reversion_moderate", "VWAP Reversão Moderado"),
        ] {
            templates.push(Self::make_template(slug, name, vec![
                ("selection", "momentum"),
                ("entry", "vwap"),
                ("exit", "stop_loss"),
                ("sizing", "equal_weight"),
            ]));
        }

        // VWAP Trend (2 variants)
        for (slug, name) in [
            ("vwap_trend_following_moderate", "VWAP Tendência Moderado"),
            ("vwap_trend_following_aggressive", "VWAP Tendência Agressivo"),
        ] {
            templates.push(Self::make_template(slug, name, vec![
                ("selection", "momentum"),
                ("entry", "vwap"),
                ("exit", "trailing_stop"),
                ("sizing", "equal_weight"),
            ]));
        }

        // RSI Intraday (2 variants)
        for (slug, name) in [
            ("intraday_mean_reversion_rsi_conservative", "RSI Intraday Conservador"),
            ("intraday_mean_reversion_rsi_moderate", "RSI Intraday Moderado"),
        ] {
            templates.push(Self::make_template(slug, name, vec![
                ("selection", "low_vol"),
                ("entry", "rsi"),
                ("exit", "stop_loss"),
                ("sizing", "equal_weight"),
            ]));
        }

        // Bollinger Intraday (2 variants)
        for (slug, name) in [
            ("intraday_mean_reversion_bb_conservative", "Bollinger Intraday Conservador"),
            ("intraday_mean_reversion_bb_moderate", "Bollinger Intraday Moderado"),
        ] {
            templates.push(Self::make_template(slug, name, vec![
                ("selection", "low_vol"),
                ("entry", "bollinger"),
                ("exit", "take_profit"),
                ("sizing", "equal_weight"),
            ]));
        }

        // MACD Intraday (2 variants)
        for (slug, name) in [
            ("intraday_momentum_macd_moderate", "MACD Intraday Moderado"),
            ("intraday_momentum_macd_aggressive", "MACD Intraday Agressivo"),
        ] {
            templates.push(Self::make_template(slug, name, vec![
                ("selection", "momentum"),
                ("entry", "macd"),
                ("exit", "trailing_stop"),
                ("sizing", "equal_weight"),
            ]));
        }

        // ADX Intraday (2 variants)
        for (slug, name) in [
            ("intraday_momentum_adx_moderate", "ADX Intraday Moderado"),
            ("intraday_momentum_adx_aggressive", "ADX Intraday Agressivo"),
        ] {
            templates.push(Self::make_template(slug, name, vec![
                ("selection", "momentum"),
                ("entry", "adx_momentum"),
                ("exit", "trailing_stop"),
                ("sizing", "equal_weight"),
            ]));
        }

        // Gap Fill (2 variants)
        for (slug, name) in [
            ("gap_fill_conservative", "Gap Fill Conservador"),
            ("gap_fill_moderate", "Gap Fill Moderado"),
        ] {
            templates.push(Self::make_template(slug, name, vec![
                ("selection", "momentum"),
                ("entry", "gap_fill"),
                ("exit", "take_profit"),
                ("sizing", "equal_weight"),
            ]));
        }

        // Gap Continuation (2 variants)
        for (slug, name) in [
            ("gap_continuation_moderate", "Gap Continuation Moderado"),
            ("gap_continuation_aggressive", "Gap Continuation Agressivo"),
        ] {
            templates.push(Self::make_template(slug, name, vec![
                ("selection", "momentum"),
                ("entry", "gap_continuation"),
                ("exit", "trailing_stop"),
                ("sizing", "equal_weight"),
            ]));
        }

        // Volume Profile (2 variants)
        templates.push(Self::make_template("volume_profile_poc_moderate", "Volume Profile POC", vec![
            ("selection", "momentum"),
            ("entry", "volume_profile"),
            ("exit", "take_profit"),
            ("sizing", "equal_weight"),
        ]));
        templates.push(Self::make_template("volume_profile_vah_val_moderate", "Volume Profile VAH/VAL", vec![
            ("selection", "momentum"),
            ("entry", "volume_profile"),
            ("exit", "stop_loss"),
            ("sizing", "equal_weight"),
        ]));

        // News Volatility (1 variant)
        templates.push(Self::make_template("news_based_volatility_moderate", "News Volatility", vec![
            ("selection", "momentum"),
            ("entry", "news_volatility"),
            ("exit", "stop_loss"),
            ("sizing", "equal_weight"),
        ]));

        // ==========================================
        // SWING TRADING STRATEGIES (12)
        // ==========================================

        // MA Crossover Swing (2 variants)
        for (slug, name) in [
            ("swing_momentum_ma_crossover_conservative", "MA Crossover Conservador"),
            ("swing_momentum_ma_crossover_moderate", "MA Crossover Moderado"),
        ] {
            templates.push(Self::make_template(slug, name, vec![
                ("selection", "momentum"),
                ("entry", "ma_crossover"),
                ("exit", "trailing_stop"),
                ("sizing", "risk_parity"),
            ]));
        }

        // MACD Swing (2 variants)
        for (slug, name) in [
            ("swing_momentum_macd_moderate", "MACD Swing Moderado"),
            ("swing_momentum_macd_aggressive", "MACD Swing Agressivo"),
        ] {
            templates.push(Self::make_template(slug, name, vec![
                ("selection", "momentum"),
                ("entry", "macd"),
                ("exit", "trailing_stop"),
                ("sizing", "risk_parity"),
            ]));
        }

        // Bollinger Swing (2 variants)
        for (slug, name) in [
            ("swing_mean_reversion_bb_conservative", "Bollinger Swing Conservador"),
            ("swing_mean_reversion_bb_moderate", "Bollinger Swing Moderado"),
        ] {
            templates.push(Self::make_template(slug, name, vec![
                ("selection", "low_vol"),
                ("entry", "bollinger"),
                ("exit", "take_profit"),
                ("sizing", "risk_parity"),
            ]));
        }

        // RSI Swing (2 variants)
        for (slug, name) in [
            ("swing_mean_reversion_rsi_conservative", "RSI Swing Conservador"),
            ("swing_mean_reversion_rsi_moderate", "RSI Swing Moderado"),
        ] {
            templates.push(Self::make_template(slug, name, vec![
                ("selection", "low_vol"),
                ("entry", "rsi"),
                ("exit", "take_profit"),
                ("sizing", "risk_parity"),
            ]));
        }

        // Channel Breakout Swing (2 variants)
        for (slug, name) in [
            ("swing_breakout_channel_moderate", "Channel Breakout Moderado"),
            ("swing_breakout_channel_aggressive", "Channel Breakout Agressivo"),
        ] {
            templates.push(Self::make_template(slug, name, vec![
                ("selection", "momentum"),
                ("entry", "channel_breakout"),
                ("exit", "trailing_stop"),
                ("sizing", "risk_parity"),
            ]));
        }

        // Volume Breakout Swing (2 variants)
        for (slug, name) in [
            ("swing_breakout_volume_moderate", "Volume Breakout Moderado"),
            ("swing_breakout_volume_aggressive", "Volume Breakout Agressivo"),
        ] {
            templates.push(Self::make_template(slug, name, vec![
                ("selection", "momentum"),
                ("entry", "volume_breakout"),
                ("exit", "trailing_stop"),
                ("sizing", "risk_parity"),
            ]));
        }

        // ==========================================
        // BREAKOUT STRATEGIES (6)
        // ==========================================

        // Donchian (3 variants)
        templates.push(Self::make_template("breakout_donchian_20d", "Donchian 20D", vec![
            ("selection", "momentum"),
            ("entry", "donchian"),
            ("exit", "trailing_stop"),
            ("sizing", "risk_parity"),
        ]));
        templates.push(Self::make_template("breakout_donchian_55d", "Donchian 55D", vec![
            ("selection", "momentum"),
            ("entry", "donchian"),
            ("exit", "trailing_stop"),
            ("sizing", "risk_parity"),
        ]));
        templates.push(Self::make_template("breakout_donchian_dual_channel", "Donchian Dual", vec![
            ("selection", "momentum"),
            ("entry", "donchian"),
            ("exit", "trailing_stop"),
            ("sizing", "risk_parity"),
        ]));

        // Volatility Expansion (3 variants)
        for (slug, name) in [
            ("breakout_volatility_expansion_conservative", "Volatility Expansion Conservador"),
            ("breakout_volatility_expansion_moderate", "Volatility Expansion Moderado"),
            ("breakout_volatility_expansion_aggressive", "Volatility Expansion Agressivo"),
        ] {
            templates.push(Self::make_template(slug, name, vec![
                ("selection", "momentum"),
                ("entry", "vol_expansion"),
                ("exit", "trailing_stop"),
                ("sizing", "risk_parity"),
            ]));
        }

        // ==========================================
        // VOLATILITY STRATEGIES (4)
        // ==========================================

        // VIX Reversion (2 variants)
        for (slug, name) in [
            ("volatility_vix_mean_reversion_conservative", "VIX Mean Reversion Conservador"),
            ("volatility_vix_mean_reversion_moderate", "VIX Mean Reversion Moderado"),
        ] {
            templates.push(Self::make_template(slug, name, vec![
                ("selection", "momentum"),
                ("entry", "vix_reversion"),
                ("exit", "take_profit"),
                ("sizing", "equal_weight"),
            ]));
        }

        // ATR Breakout
        templates.push(Self::make_template("volatility_breakout_atr_moderate", "ATR Breakout", vec![
            ("selection", "momentum"),
            ("entry", "atr_breakout"),
            ("exit", "trailing_stop"),
            ("sizing", "risk_parity"),
        ]));

        // BB Squeeze
        templates.push(Self::make_template("volatility_breakout_bb_width_moderate", "BB Width Breakout", vec![
            ("selection", "momentum"),
            ("entry", "vol_expansion"),
            ("exit", "trailing_stop"),
            ("sizing", "risk_parity"),
        ]));

        // ==========================================
        // MOMENTUM STRATEGIES (8)
        // ==========================================

        // Cross-Sectional Momentum (4 variants)
        for (slug, name, _lookback) in [
            ("momentum_cross_sectional_3m", "Cross-Sectional 3M", 63),
            ("momentum_cross_sectional_6m", "Cross-Sectional 6M", 126),
            ("momentum_cross_sectional_12m", "Cross-Sectional 12M", 252),
            ("momentum_cross_sectional_multi_period", "Cross-Sectional Multi", 126),
        ] {
            templates.push(Self::make_template(slug, name, vec![
                ("selection", "sector_rotation"),
                ("entry", "cross_sectional"),
                ("exit", "trailing_stop"),
                ("sizing", "equal_weight"),
            ]));
        }

        // Time Series Momentum (3 variants)
        for (slug, name) in [
            ("momentum_time_series_50d", "Time Series 50D"),
            ("momentum_time_series_200d", "Time Series 200D"),
        ] {
            templates.push(Self::make_template(slug, name, vec![
                ("selection", "momentum"),
                ("entry", "time_series"),
                ("exit", "trailing_stop"),
                ("sizing", "risk_parity"),
            ]));
        }

        // Dual Moving Average
        templates.push(Self::make_template("momentum_time_series_dual_ma", "Dual Moving Average", vec![
            ("selection", "momentum"),
            ("entry", "dual_ma"),
            ("exit", "trailing_stop"),
            ("sizing", "risk_parity"),
        ]));

        // Adaptive Momentum
        templates.push(Self::make_template("momentum_time_series_adaptive", "Adaptive Momentum", vec![
            ("selection", "momentum"),
            ("entry", "adaptive_momentum"),
            ("exit", "trailing_stop"),
            ("sizing", "risk_parity"),
        ]));

        // ==========================================
        // MEAN REVERSION STRATEGIES (8)
        // ==========================================

        // Bollinger Mean Reversion (3 standard variants)
        for (slug, name) in [
            ("mean_reversion_bb_conservative", "Bollinger Conservador"),
            ("mean_reversion_bb_moderate", "Bollinger Moderado"),
            ("mean_reversion_bb_aggressive", "Bollinger Agressivo"),
        ] {
            templates.push(Self::make_template(slug, name, vec![
                ("selection", "low_vol"),
                ("entry", "bollinger"),
                ("exit", "take_profit"),
                ("sizing", "risk_parity"),
            ]));
        }
        // Bollinger with Trend Filter (specialized block)
        templates.push(Self::make_template("mean_reversion_bb_trend_filtered", "Bollinger Trend Filter", vec![
            ("selection", "low_vol"),
            ("entry", "bb_filtered"),
            ("exit", "take_profit"),
            ("sizing", "risk_parity"),
        ]));

        // RSI Mean Reversion (3 standard variants)
        for (slug, name) in [
            ("mean_reversion_rsi_conservative", "RSI Conservador"),
            ("mean_reversion_rsi_moderate", "RSI Moderado"),
            ("mean_reversion_rsi_aggressive", "RSI Agressivo"),
        ] {
            templates.push(Self::make_template(slug, name, vec![
                ("selection", "low_vol"),
                ("entry", "rsi"),
                ("exit", "take_profit"),
                ("sizing", "risk_parity"),
            ]));
        }
        // RSI with Trend Filter (specialized block)
        templates.push(Self::make_template("mean_reversion_rsi_trend_filtered", "RSI Trend Filter", vec![
            ("selection", "low_vol"),
            ("entry", "rsi_filtered"),
            ("exit", "take_profit"),
            ("sizing", "risk_parity"),
        ]));

        // ==========================================
        // SECTOR ROTATION (4)
        // ==========================================

        for (slug, name) in [
            ("sector_rotation_business_cycle", "Business Cycle"),
            ("sector_rotation_business_cycle_defensive", "Business Cycle Defensivo"),
            ("sector_rotation_relative_strength_top3", "Relative Strength Top 3"),
            ("sector_rotation_relative_strength_top5", "Relative Strength Top 5"),
        ] {
            templates.push(Self::make_template(slug, name, vec![
                ("selection", "sector_rotation"),
                ("entry", "ma_crossover"),
                ("exit", "trailing_stop"),
                ("sizing", "equal_weight"),
            ]));
        }

        // ==========================================
        // FACTOR STRATEGIES (8)
        // ==========================================

        // Value Factor (using value_pe and value_pb selection)
        templates.push(Self::make_template("factor_value_pe_conservative", "Value P/E Conservador", vec![
            ("selection", "value_pe"),
            ("entry", "ma_crossover"),
            ("exit", "trailing_stop"),
            ("sizing", "risk_parity"),
        ]));
        templates.push(Self::make_template("factor_value_pb_moderate", "Value P/B Moderado", vec![
            ("selection", "value_pb"),
            ("entry", "ma_crossover"),
            ("exit", "trailing_stop"),
            ("sizing", "risk_parity"),
        ]));

        // Quality Factor
        templates.push(Self::make_template("factor_quality_roe_conservative", "Quality ROE Conservador", vec![
            ("selection", "quality_roe"),
            ("entry", "ma_crossover"),
            ("exit", "trailing_stop"),
            ("sizing", "risk_parity"),
        ]));
        templates.push(Self::make_template("factor_quality_multi_metric_moderate", "Quality Multi-Metric", vec![
            ("selection", "quality_multi"),
            ("entry", "ma_crossover"),
            ("exit", "trailing_stop"),
            ("sizing", "risk_parity"),
        ]));

        // Low Volatility Factor
        for (slug, name) in [
            ("factor_low_volatility_conservative", "Low Volatility Conservador"),
            ("factor_low_volatility_moderate", "Low Volatility Moderado"),
        ] {
            templates.push(Self::make_template(slug, name, vec![
                ("selection", "low_vol"),
                ("entry", "ma_crossover"),
                ("exit", "trailing_stop"),
                ("sizing", "risk_parity"),
            ]));
        }

        // Multi-Factor
        for (slug, name) in [
            ("factor_multi_factor_balanced", "Multi-Factor Balanced"),
            ("factor_multi_factor_aggressive", "Multi-Factor Agressivo"),
        ] {
            templates.push(Self::make_template(slug, name, vec![
                ("selection", "multi_factor"),
                ("entry", "ma_crossover"),
                ("exit", "trailing_stop"),
                ("sizing", "risk_parity"),
            ]));
        }

        // ==========================================
        // PORTFOLIO STRATEGIES (14 + 6 Position + 12 Pair + 4 Seasonal + 4 Event + 4 Buy-Hold)
        // Fill remaining with generic templates
        // ==========================================

        // Portfolio Equal Weight
        for size in [5, 10, 20, 30] {
            let slug = format!("portfolio_equal_weight_{}_assets", size);
            templates.push(Self::make_template(&slug, &format!("Equal Weight {} Ativos", size), vec![
                ("selection", "momentum"),
                ("sizing", "equal_weight"),
            ]));
        }

        // Portfolio Risk Parity
        for risk in ["conservative", "moderate", "aggressive", "leveraged"] {
            let slug = format!("portfolio_risk_parity_{}", risk);
            templates.push(Self::make_template(&slug, &format!("Risk Parity {}", risk), vec![
                ("selection", "momentum"),
                ("sizing", "risk_parity"),
            ]));
        }

        // Portfolio Min Variance / Max Sharpe
        for strat in ["min_variance_conservative", "min_variance_moderate", "min_variance_long_short",
                     "max_sharpe_conservative", "max_sharpe_moderate", "max_sharpe_aggressive"] {
            let slug = format!("portfolio_{}", strat);
            templates.push(Self::make_template(&slug, strat, vec![
                ("selection", "multi_factor"),
                ("sizing", "vol_targeting"),
            ]));
        }

        // Position Trading - Trend Following MA
        for (slug, name) in [
            ("position_trend_following_ma_conservative", "Trend Following MA Conservador"),
            ("position_trend_following_ma_moderate", "Trend Following MA Moderado"),
        ] {
            templates.push(Self::make_template(slug, name, vec![
                ("selection", "momentum"),
                ("entry", "ma_crossover"),
                ("exit", "trailing_stop"),
                ("sizing", "risk_parity"),
            ]));
        }

        // Position Trading - Trend Following ADX
        templates.push(Self::make_template("position_trend_following_adx_moderate", "Trend Following ADX", vec![
            ("selection", "momentum"),
            ("entry", "adx_momentum"),
            ("exit", "trailing_stop"),
            ("sizing", "risk_parity"),
        ]));

        // Position Trading - Value (uses value selection block)
        for (slug, name) in [
            ("position_fundamental_value_conservative", "Value Investing Conservador"),
            ("position_fundamental_value_moderate", "Value Investing Moderado"),
        ] {
            templates.push(Self::make_template(slug, name, vec![
                ("selection", "value"),
                ("entry", "ma_crossover"),
                ("exit", "trailing_stop"),
                ("sizing", "risk_parity"),
            ]));
        }

        // Position Trading - Quality (uses quality selection block)
        templates.push(Self::make_template("position_fundamental_quality_moderate", "Quality Investing", vec![
            ("selection", "quality"),
            ("entry", "ma_crossover"),
            ("exit", "trailing_stop"),
            ("sizing", "risk_parity"),
        ]));

        // Pair Trading - Cointegration (using specialized cointegration block)
        for (slug, name) in [
            ("pair_trading_cointegration_conservative", "Cointegração Conservador"),
            ("pair_trading_cointegration_moderate", "Cointegração Moderado"),
            ("pair_trading_cointegration_aggressive", "Cointegração Agressivo"),
            ("pair_trading_cointegration_short_term", "Cointegração Curto Prazo"),
        ] {
            templates.push(Self::make_template(slug, name, vec![
                ("selection", "momentum"),
                ("entry", "cointegration"),
                ("exit", "take_profit"),
                ("sizing", "equal_weight"),
            ]));
        }

        // Pair Trading - Distance (using specialized distance block)
        for (slug, name) in [
            ("pair_trading_distance_conservative", "Distance Conservador"),
            ("pair_trading_distance_moderate", "Distance Moderado"),
            ("pair_trading_distance_aggressive", "Distance Agressivo"),
            ("pair_trading_distance_short_term", "Distance Curto Prazo"),
        ] {
            templates.push(Self::make_template(slug, name, vec![
                ("selection", "momentum"),
                ("entry", "distance"),
                ("exit", "take_profit"),
                ("sizing", "equal_weight"),
            ]));
        }

        // Stat Arb Multi-Pair (using specialized multi_pair block)
        for (slug, name) in [
            ("stat_arb_multi_pair_conservative", "Multi-Pair Conservador"),
            ("stat_arb_multi_pair_moderate", "Multi-Pair Moderado"),
            ("stat_arb_multi_pair_aggressive", "Multi-Pair Agressivo"),
            ("stat_arb_multi_pair_diversified", "Multi-Pair Diversificado"),
        ] {
            templates.push(Self::make_template(slug, name, vec![
                ("selection", "momentum"),
                ("entry", "multi_pair"),
                ("exit", "take_profit"),
                ("sizing", "risk_parity"),
            ]));
        }

        // Seasonal - using specialized seasonal entry blocks
        templates.push(Self::make_template("seasonal_calendar_effects_january", "January Effect", vec![
            ("selection", "momentum"),
            ("entry", "january_effect"),
            ("exit", "time_exit"),
            ("sizing", "equal_weight"),
        ]));
        templates.push(Self::make_template("seasonal_calendar_effects_sell_in_may", "Sell in May", vec![
            ("selection", "momentum"),
            ("entry", "sell_in_may"),
            ("exit", "time_exit"),
            ("sizing", "equal_weight"),
        ]));
        templates.push(Self::make_template("seasonal_commodity_natural_gas", "Natural Gas Seasonal", vec![
            ("selection", "momentum"),
            ("entry", "natgas_seasonal"),
            ("exit", "time_exit"),
            ("sizing", "equal_weight"),
        ]));
        templates.push(Self::make_template("seasonal_commodity_grains", "Grains Seasonal", vec![
            ("selection", "momentum"),
            ("entry", "grains_seasonal"),
            ("exit", "time_exit"),
            ("sizing", "equal_weight"),
        ]));

        // Event-Driven - using specialized event entry blocks
        templates.push(Self::make_template("event_driven_earnings_pre_announcement", "Pre-Earnings", vec![
            ("selection", "momentum"),
            ("entry", "pre_earnings"),
            ("exit", "time_exit"),
            ("sizing", "equal_weight"),
        ]));
        templates.push(Self::make_template("event_driven_earnings_post_surprise", "Post-Earnings Surprise", vec![
            ("selection", "momentum"),
            ("entry", "post_earnings"),
            ("exit", "time_exit"),
            ("sizing", "equal_weight"),
        ]));
        templates.push(Self::make_template("event_driven_news_volatility", "News Volatility", vec![
            ("selection", "momentum"),
            ("entry", "news_volatility"),
            ("exit", "time_exit"),
            ("sizing", "equal_weight"),
        ]));
        templates.push(Self::make_template("event_driven_ma_arbitrage", "MA Arbitrage", vec![
            ("selection", "momentum"),
            ("entry", "ma_arb"),
            ("exit", "take_profit"),
            ("sizing", "equal_weight"),
        ]));

        // Buy and Hold - using buy_hold entry block
        templates.push(Self::make_template("buy_hold_index_ibov", "Buy Hold IBOV", vec![
            ("selection", "momentum"),
            ("entry", "buy_hold"),
            ("sizing", "equal_weight"),
        ]));
        templates.push(Self::make_template("buy_hold_index_sp500", "Buy Hold SP500", vec![
            ("selection", "momentum"),
            ("entry", "buy_hold"),
            ("sizing", "equal_weight"),
        ]));
        templates.push(Self::make_template("buy_hold_dividend_growth_conservative", "Dividend Growth Conservador", vec![
            ("selection", "dividend_growth"),
            ("entry", "buy_hold"),
            ("sizing", "equal_weight"),
        ]));
        templates.push(Self::make_template("buy_hold_dividend_growth_moderate", "Dividend Growth Moderado", vec![
            ("selection", "dividend_growth"),
            ("entry", "buy_hold"),
            ("sizing", "risk_parity"),
        ]));

        // ==========================================
        // ADVANCED STRATEGIES (Prop-Trading Level)
        // Uses new exit, sizing, and selection blocks
        // ==========================================

        // Weekly Radar Strategies (BR + US)
        templates.push(Self::make_template("radar_weekly_momentum_br", "Radar Semanal Momentum BR", vec![
            ("selection", "liquidity_filter"),
            ("entry", "cross_sectional"),
            ("exit", "time_atr_hybrid"),
            ("sizing", "vol_target_buffer"),
        ]));
        templates.push(Self::make_template("radar_weekly_momentum_us", "Radar Semanal Momentum US", vec![
            ("selection", "liquidity_filter"),
            ("entry", "cross_sectional"),
            ("exit", "time_atr_hybrid"),
            ("sizing", "vol_target_buffer"),
        ]));
        templates.push(Self::make_template("radar_weekly_lowvol_br", "Radar Semanal Low Vol BR", vec![
            ("selection", "liquidity_filter"),
            ("entry", "ma_crossover"),
            ("exit", "chandelier_atr"),
            ("sizing", "exposure_cap"),
        ]));
        templates.push(Self::make_template("radar_weekly_lowvol_us", "Radar Semanal Low Vol US", vec![
            ("selection", "liquidity_filter"),
            ("entry", "ma_crossover"),
            ("exit", "chandelier_atr"),
            ("sizing", "exposure_cap"),
        ]));

        // Regime-Aware Strategies
        templates.push(Self::make_template("regime_momentum_defensive", "Regime Momentum Defensivo", vec![
            ("selection", "regime_filter"),
            ("entry", "time_series"),
            ("exit", "volatility_spike"),
            ("sizing", "kelly_fractional"),
        ]));
        templates.push(Self::make_template("regime_trend_following", "Regime Trend Following", vec![
            ("selection", "regime_filter"),
            ("entry", "dual_ma"),
            ("exit", "trend_reversal"),
            ("sizing", "vol_target_buffer"),
        ]));

        // Drawdown-Controlled Strategies
        templates.push(Self::make_template("dd_controlled_swing_br", "Swing DD-Controlled BR", vec![
            ("selection", "momentum"),
            ("entry", "bollinger"),
            ("exit", "drawdown_limit"),
            ("sizing", "exposure_cap"),
        ]));
        templates.push(Self::make_template("dd_controlled_swing_us", "Swing DD-Controlled US", vec![
            ("selection", "momentum"),
            ("entry", "rsi"),
            ("exit", "drawdown_limit"),
            ("sizing", "exposure_cap"),
        ]));

        // Kelly-Optimized Strategies
        templates.push(Self::make_template("kelly_momentum_br", "Kelly Momentum BR", vec![
            ("selection", "liquidity_filter"),
            ("entry", "adaptive_momentum"),
            ("exit", "chandelier_atr"),
            ("sizing", "kelly_fractional"),
        ]));
        templates.push(Self::make_template("kelly_momentum_us", "Kelly Momentum US", vec![
            ("selection", "liquidity_filter"),
            ("entry", "adaptive_momentum"),
            ("exit", "chandelier_atr"),
            ("sizing", "kelly_fractional"),
        ]));

        // Volatility-Targeted Strategies
        templates.push(Self::make_template("voltarget_breakout_br", "Vol Target Breakout BR", vec![
            ("selection", "momentum"),
            ("entry", "vol_expansion"),
            ("exit", "volatility_spike"),
            ("sizing", "vol_target_buffer"),
        ]));
        templates.push(Self::make_template("voltarget_breakout_us", "Vol Target Breakout US", vec![
            ("selection", "momentum"),
            ("entry", "atr_breakout"),
            ("exit", "volatility_spike"),
            ("sizing", "vol_target_buffer"),
        ]));

        templates
    }

    /// Helper para criar um template a partir de um pipeline simples.
    fn make_template(slug: &str, name: &str, pipeline: Vec<(&str, &str)>) -> StrategyTemplate {
        let blocks: Vec<TemplateBlock> = pipeline
            .into_iter()
            .map(|(block_type, block_id)| TemplateBlock {
                block_type_str: block_type.to_string(),
                block_id: block_id.to_string(),
                params: HashMap::new(),
            })
            .collect();

        StrategyTemplate {
            slug: slug.to_string(),
            name: name.to_string(),
            pipeline: blocks,
            source_path: "builtin".to_string(),
        }
    }

    /// Carrega templates de um diretório de arquivos TOML.
    pub fn from_toml_dir(dir: &Path) -> Result<Self, CatalogError> {
        let mut templates = Vec::new();

        if !dir.exists() {
            return Err(CatalogError::Empty(dir.display().to_string()));
        }

        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map(|e| e == "toml").unwrap_or(false) {
                match Self::parse_template(&path) {
                    Ok(template) => {
                        debug!(
                            "Loaded template: {} ({} blocks)",
                            template.slug,
                            template.pipeline.len()
                        );
                        templates.push(template);
                    }
                    Err(e) => {
                        warn!("Failed to parse {}: {}", path.display(), e);
                    }
                }
            }
        }

        if templates.is_empty() {
            return Err(CatalogError::Empty(dir.display().to_string()));
        }

        info!(
            "Strategy Catalog: loaded {} templates from {}",
            templates.len(),
            dir.display()
        );
        Ok(Self { templates })
    }

    /// Parse de um único arquivo TOML de template.
    fn parse_template(path: &Path) -> Result<StrategyTemplate, CatalogError> {
        let content = std::fs::read_to_string(path)?;
        let doc: toml::Value = toml::from_str(&content).map_err(|e| CatalogError::Parse {
            path: path.display().to_string(),
            msg: e.to_string(),
        })?;

        let slug = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let name = doc
            .get("strategy")
            .and_then(|s| s.get("description"))
            .and_then(|v| v.as_str())
            .unwrap_or(&slug)
            .to_string();

        // Parse [[pipeline]] array
        let pipeline: Vec<TemplateBlock> = doc
            .get("pipeline")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| {
                        let block_type_str = v.get("type")?.as_str()?.to_string();
                        let block_id = v.get("block_id")?.as_str()?.to_string();
                        let params = v
                            .get("params")
                            .and_then(|p| p.as_table())
                            .map(|t| t.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                            .unwrap_or_default();
                        Some(TemplateBlock {
                            block_type_str,
                            block_id,
                            params,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        if pipeline.is_empty() {
            return Err(CatalogError::Parse {
                path: path.display().to_string(),
                msg: "No [[pipeline]] blocks found".into(),
            });
        }

        Ok(StrategyTemplate {
            slug,
            name,
            pipeline,
            source_path: path.display().to_string(),
        })
    }

    /// Filtra templates por colunas de dados disponíveis.
    pub fn filter_by_data(
        self,
        available: &HashSet<DataColumn>,
        param_ranges: &ParamRanges,
    ) -> Self {
        let before = self.templates.len();
        let templates: Vec<_> = self
            .templates
            .into_iter()
            .filter(|t| t.is_compatible(available, param_ranges))
            .collect();
        let after = templates.len();

        if before != after {
            info!(
                "Template filter: {}/{} templates compatible with available data",
                after, before
            );
        }

        Self { templates }
    }

    /// Número de templates disponíveis.
    pub fn len(&self) -> usize {
        self.templates.len()
    }

    /// Verifica se está vazio.
    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }

    /// Referência aos templates.
    pub fn templates(&self) -> &[StrategyTemplate] {
        &self.templates
    }

    /// Filtra o catálogo para incluir apenas os templates com os slugs especificados.
    /// Se o slice estiver vazio, retorna o catálogo completo (sem filtro).
    /// 
    /// # Exemplo
    /// ```ignore
    /// let catalog = StrategyCatalog::from_builtin();
    /// let filtered = catalog.filter_by_slugs(&["orb_breakout_conservative".to_string()]);
    /// assert_eq!(filtered.len(), 1);
    /// ```
    pub fn filter_by_slugs(&self, slugs: &[String]) -> Self {
        if slugs.is_empty() {
            return self.clone();
        }
        let filtered: Vec<StrategyTemplate> = self.templates
            .iter()
            .filter(|t| slugs.contains(&t.slug))
            .cloned()
            .collect();
        
        info!(
            "Strategy Catalog: filtered from {} to {} templates ({} slugs requested)",
            self.templates.len(),
            filtered.len(),
            slugs.len()
        );
        
        Self { templates: filtered }
    }

    /// Converte template em StrategyGenome com parâmetros RANDOMIZADOS.
    /// ESTRUTURA = fixa do template
    /// PARÂMETROS = randomizados dentro dos ranges do ParamRanges
    pub fn to_genome(
        template: &StrategyTemplate,
        rng: &mut ChaCha8Rng,
        param_ranges: &ParamRanges,
        generation: u32,
    ) -> StrategyGenome {
        let genes: Vec<BlockGene> = template
            .pipeline
            .iter()
            .map(|block| {
                let block_type = block.block_type();

                // Randomizar parâmetros dentro dos ranges
                let params: Vec<(String, ParamValue)> =
                    if let Some(spec) = param_ranges.get_block(&block.block_id) {
                        spec.params
                            .iter()
                            .map(|p| {
                                let value = Self::randomize_param(&p.default, rng);
                                (p.name.clone(), value)
                            })
                            .collect()
                    } else {
                        Vec::new()
                    };

                BlockGene::new(block_type, &block.block_id, params)
            })
            .collect();

        StrategyGenome::new(genes)
            .with_generation(generation)
            .with_template_slug(template.slug.clone())
    }

    /// Randomiza parâmetro dentro do seu range.
    fn randomize_param(template: &ParamValue, rng: &mut ChaCha8Rng) -> ParamValue {
        match template {
            ParamValue::Float {
                min, max, step, ..
            } => {
                let steps = ((*max - *min) / *step).max(1.0) as u32;
                let random_steps = rng.gen_range(0..=steps);
                let value = *min + (random_steps as f64) * *step;
                ParamValue::float(value.clamp(*min, *max), *min, *max, *step)
            }
            ParamValue::Int {
                min, max, step, ..
            } => {
                let steps = ((*max - *min) / *step).max(1) as u32;
                let random_steps = rng.gen_range(0..=steps);
                let value = *min + (random_steps as i64) * *step;
                ParamValue::int(value.clamp(*min, *max), *min, *max, *step)
            }
            ParamValue::Bool { .. } => ParamValue::bool(rng.gen_bool(0.5)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_template_block_type() {
        let block = TemplateBlock {
            block_type_str: "selection".to_string(),
            block_id: "momentum".to_string(),
            params: HashMap::new(),
        };
        assert_eq!(block.block_type(), BlockType::Selection);

        let block = TemplateBlock {
            block_type_str: "exit".to_string(),
            block_id: "stop_loss".to_string(),
            params: HashMap::new(),
        };
        assert_eq!(block.block_type(), BlockType::Exit);
    }

    #[test]
    fn test_catalog_from_dir() {
        let path = PathBuf::from("configs/strategies/");
        if path.exists() {
            let catalog = StrategyCatalog::from_toml_dir(&path);
            assert!(catalog.is_ok());
            let catalog = catalog.unwrap();
            assert!(!catalog.is_empty());
        }
    }

    #[test]
    fn test_to_genome_creates_valid_genome() {
        let template = StrategyTemplate {
            slug: "test_strategy".to_string(),
            name: "Test Strategy".to_string(),
            pipeline: vec![
                TemplateBlock {
                    block_type_str: "selection".to_string(),
                    block_id: "momentum".to_string(),
                    params: HashMap::new(),
                },
                TemplateBlock {
                    block_type_str: "sizing".to_string(),
                    block_id: "equal_weight".to_string(),
                    params: HashMap::new(),
                },
            ],
            source_path: "test.toml".to_string(),
        };

        let param_ranges = ParamRanges::new();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let genome = StrategyCatalog::to_genome(&template, &mut rng, &param_ranges, 0);

        assert_eq!(genome.template_slug, Some("test_strategy".to_string()));
        assert_eq!(genome.genes.len(), 2);
        assert!(genome.has_block_type(BlockType::Selection));
        assert!(genome.has_block_type(BlockType::Sizing));
    }

    #[test]
    fn test_builtin_catalog_has_many_templates() {
        let catalog = StrategyCatalog::from_builtin();
        
        // Should have 116 or more templates based on seed SQL
        assert!(catalog.len() >= 100, "Expected at least 100 templates, got {}", catalog.len());
        
        // Check some known templates exist
        let slugs: Vec<&str> = catalog.templates().iter().map(|t| t.slug.as_str()).collect();
        assert!(slugs.contains(&"orb_breakout_conservative"));
        assert!(slugs.contains(&"vwap_mean_reversion_moderate"));
        assert!(slugs.contains(&"breakout_donchian_20d"));
        assert!(slugs.contains(&"swing_momentum_ma_crossover_moderate"));
    }

    #[test]
    fn test_builtin_templates_have_valid_blocks() {
        let catalog = StrategyCatalog::from_builtin();
        let param_ranges = ParamRanges::new();
        
        for template in catalog.templates() {
            // Each template should have at least 1 block
            assert!(!template.pipeline.is_empty(), 
                "Template {} has empty pipeline", template.slug);
            
            // Each block should reference a valid block_id in param_ranges
            for block in &template.pipeline {
                // Selection and sizing blocks always valid
                // Entry/Exit blocks should be in param_ranges
                if block.block_type_str == "entry" || block.block_type_str == "exit" {
                    assert!(param_ranges.contains(&block.block_id),
                        "Template {} references unknown block: {}", 
                        template.slug, block.block_id);
                }
            }
        }
    }

    #[test]
    fn test_builtin_genome_generation() {
        let catalog = StrategyCatalog::from_builtin();
        let param_ranges = ParamRanges::new();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        
        // Generate genomes from all templates
        let mut genomes = Vec::new();
        for template in catalog.templates() {
            let genome = StrategyCatalog::to_genome(template, &mut rng, &param_ranges, 0);
            genomes.push(genome);
        }
        
        assert_eq!(genomes.len(), catalog.len());
        
        // All genomes should have template_slug set
        for genome in &genomes {
            assert!(genome.template_slug.is_some());
        }
    }

    #[test]
    fn test_filter_by_slugs() {
        let catalog = StrategyCatalog::from_builtin();
        let full_count = catalog.len();
        
        // Filter to specific slugs
        let slugs = vec![
            "orb_breakout_conservative".to_string(),
            "vwap_mean_reversion_moderate".to_string(),
        ];
        let filtered = catalog.filter_by_slugs(&slugs);
        assert_eq!(filtered.len(), 2);
        
        // Empty filter returns all
        let no_filter = catalog.filter_by_slugs(&[]);
        assert_eq!(no_filter.len(), full_count);
        
        // Non-existent slug returns empty
        let unknown = catalog.filter_by_slugs(&["nonexistent_slug".to_string()]);
        assert_eq!(unknown.len(), 0);
    }
}

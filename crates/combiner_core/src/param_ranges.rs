//! Parameter ranges for mutation and validation.
//!
//! Defines valid ranges and steps for all block parameters,
//! extracted from the block catalog.

use crate::genome::{BlockType, ParamValue};
use std::collections::HashMap;

/// Parameter specification with range and default.
#[derive(Debug, Clone)]
pub struct ParamSpec {
    /// Parameter name.
    pub name: String,
    /// Default value.
    pub default: ParamValue,
    /// Description.
    pub description: String,
}

impl ParamSpec {
    /// Create a float parameter spec.
    pub fn float(
        name: impl Into<String>,
        default: f64,
        min: f64,
        max: f64,
        step: f64,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            default: ParamValue::float(default, min, max, step),
            description: description.into(),
        }
    }

    /// Create an int parameter spec.
    pub fn int(
        name: impl Into<String>,
        default: i64,
        min: i64,
        max: i64,
        step: i64,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            default: ParamValue::int(default, min, max, step),
            description: description.into(),
        }
    }

    /// Create a bool parameter spec.
    pub fn bool(name: impl Into<String>, default: bool, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            default: ParamValue::bool(default),
            description: description.into(),
        }
    }
}

/// Data columns that may be required by blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataColumn {
    /// Basic OHLCV columns (always available in market data)
    Open,
    High,
    Low,
    Close,
    Volume,
    AdjClose,
    /// Fundamental data columns (require additional data sources)
    PE,           // Price/Earnings ratio
    PB,           // Price/Book ratio
    ROE,          // Return on Equity
    DebtEquity,   // Debt/Equity ratio
    DividendYield,
    MarketCap,
    RiskFreeRate, // For carry calculation
}

impl DataColumn {
    /// Check if this is a basic OHLCV column (always available).
    pub fn is_ohlcv(&self) -> bool {
        matches!(self, 
            DataColumn::Open | DataColumn::High | DataColumn::Low | 
            DataColumn::Close | DataColumn::Volume | DataColumn::AdjClose
        )
    }
}

/// Block specification with all parameters.
#[derive(Debug, Clone)]
pub struct BlockSpec {
    /// Block ID.
    pub block_id: String,
    /// Block type.
    pub block_type: BlockType,
    /// Parameter specifications.
    pub params: Vec<ParamSpec>,
    /// Description.
    pub description: String,
    /// Whether this block supports fast SoA execution.
    pub fast_supported: bool,
    /// Data columns required by this block.
    /// Blocks with only OHLCV requirements work with any dataset.
    /// Blocks with fundamental requirements need additional data.
    pub required_columns: Vec<DataColumn>,
}

impl BlockSpec {
    /// Get default parameters as a HashMap.
    pub fn default_params(&self) -> HashMap<String, ParamValue> {
        self.params
            .iter()
            .map(|p| (p.name.clone(), p.default.clone()))
            .collect()
    }
}

/// Registry of all block parameter ranges.
#[derive(Debug, Clone)]
pub struct ParamRanges {
    blocks: HashMap<String, BlockSpec>,
}

impl Default for ParamRanges {
    fn default() -> Self {
        Self::new()
    }
}

impl ParamRanges {
    /// Create a new parameter ranges registry with all built-in blocks.
    pub fn new() -> Self {
        let mut blocks = HashMap::new();

        // Selection blocks
        Self::register_selection_blocks(&mut blocks);

        // Entry blocks
        Self::register_entry_blocks(&mut blocks);

        // Exit blocks
        Self::register_exit_blocks(&mut blocks);

        // Sizing blocks
        Self::register_sizing_blocks(&mut blocks);

        Self { blocks }
    }

    fn register_selection_blocks(blocks: &mut HashMap<String, BlockSpec>) {
        // B3-calibrated: min_return range relaxed to -0.30 to +0.10
        blocks.insert(
            "momentum".into(),
            BlockSpec {
                block_id: "momentum".into(),
                block_type: BlockType::Selection,
                params: vec![
                    ParamSpec::int("lookback_days", 126, 21, 252, 21, "Lookback period in days"),
                    ParamSpec::float("top_pct", 20.0, 5.0, 50.0, 5.0, "Top % of assets to select"),
                    ParamSpec::float("min_return", 0.0, -0.30, 0.10, 0.02, "Minimum return threshold"),
                    ParamSpec::int("skip_last_days", 21, 0, 63, 7, "Days to skip at end"),
                ],
                description: "Momentum selection - ranks by 6-12 month returns".into(),
                fast_supported: true,
                required_columns: vec![DataColumn::Close, DataColumn::AdjClose],
            },
        );

        blocks.insert(
            "value".into(),
            BlockSpec {
                block_id: "value".into(),
                block_type: BlockType::Selection,
                params: vec![
                    ParamSpec::float("max_pe", 15.0, 5.0, 50.0, 5.0, "Maximum P/E ratio"),
                    ParamSpec::float("max_pb", 2.0, 0.5, 10.0, 0.5, "Maximum P/B ratio"),
                    ParamSpec::float("top_pct", 20.0, 5.0, 50.0, 5.0, "Top % of assets"),
                ],
                description: "Value selection - low P/E, P/B".into(),
                fast_supported: false,
                required_columns: vec![DataColumn::PE, DataColumn::PB],
            },
        );

        blocks.insert(
            "quality".into(),
            BlockSpec {
                block_id: "quality".into(),
                block_type: BlockType::Selection,
                params: vec![
                    ParamSpec::float("min_roe", 0.15, 0.05, 0.50, 0.05, "Minimum ROE"),
                    ParamSpec::float("max_debt_equity", 0.5, 0.1, 2.0, 0.1, "Maximum D/E ratio"),
                    ParamSpec::float("top_pct", 20.0, 5.0, 50.0, 5.0, "Top % of assets"),
                ],
                description: "Quality selection - high ROE, low debt".into(),
                fast_supported: false,
                required_columns: vec![DataColumn::ROE, DataColumn::DebtEquity],
            },
        );

        // B3-calibrated: vol tipica B3 = 25-40%, max_annualized_vol range 0.20 to 0.60
        blocks.insert(
            "low_vol".into(),
            BlockSpec {
                block_id: "low_vol".into(),
                block_type: BlockType::Selection,
                params: vec![
                    ParamSpec::float(
                        "max_annualized_vol",
                        0.30,
                        0.20,
                        0.60,
                        0.05,
                        "Maximum annualized volatility",
                    ),
                    ParamSpec::int("lookback_days", 60, 20, 252, 20, "Lookback period"),
                    ParamSpec::float("top_pct", 20.0, 5.0, 50.0, 5.0, "Top % of assets"),
                ],
                description: "Low volatility selection".into(),
                fast_supported: true,
                required_columns: vec![DataColumn::Close],
            },
        );

        // B3-calibrated: boas pagadoras DY 4-7%, max realista 8%
        blocks.insert(
            "dividend".into(),
            BlockSpec {
                block_id: "dividend".into(),
                block_type: BlockType::Selection,
                params: vec![
                    ParamSpec::float("min_yield", 0.03, 0.01, 0.08, 0.01, "Minimum dividend yield"),
                    ParamSpec::float("top_pct", 20.0, 5.0, 50.0, 5.0, "Top % of assets"),
                ],
                description: "Dividend yield selection".into(),
                fast_supported: false,
                required_columns: vec![DataColumn::DividendYield],
            },
        );

        // B3-calibrated: min_market_cap max 25B (inclui mid-caps), max_market_cap max 500B
        blocks.insert(
            "size".into(),
            BlockSpec {
                block_id: "size".into(),
                block_type: BlockType::Selection,
                params: vec![
                    ParamSpec::float(
                        "min_market_cap",
                        1_000_000_000.0,
                        100_000_000.0,
                        25_000_000_000.0,
                        1_000_000_000.0,
                        "Minimum market cap",
                    ),
                    ParamSpec::float(
                        "max_market_cap",
                        100_000_000_000.0,
                        1_000_000_000.0,
                        500_000_000_000.0,
                        10_000_000_000.0,
                        "Maximum market cap",
                    ),
                    ParamSpec::float("top_pct", 20.0, 5.0, 50.0, 5.0, "Top % of assets"),
                ],
                description: "Size selection by market cap".into(),
                fast_supported: false,
                required_columns: vec![DataColumn::MarketCap],
            },
        );

        // B3-calibrated: com Selic ~10.5%, carry > 4% e impossivel (DY - Selic)
        blocks.insert(
            "carry".into(),
            BlockSpec {
                block_id: "carry".into(),
                block_type: BlockType::Selection,
                params: vec![
                    ParamSpec::float(
                        "min_carry",
                        0.0,
                        -0.08,
                        0.04,
                        0.01,
                        "Minimum carry (yield - risk-free)",
                    ),
                    ParamSpec::float("top_pct", 20.0, 5.0, 50.0, 5.0, "Top % of assets"),
                ],
                description: "Carry selection - dividend vs risk-free".into(),
                fast_supported: false,
                required_columns: vec![DataColumn::DividendYield, DataColumn::RiskFreeRate],
            },
        );

        // ==========================================
        // ADVANCED SELECTION BLOCKS (Prop-Trading Level)
        // ==========================================

        blocks.insert(
            "liquidity_filter".into(),
            BlockSpec {
                block_id: "liquidity_filter".into(),
                block_type: BlockType::Selection,
                params: vec![
                    ParamSpec::float("min_adv_percentile", 30.0, 10.0, 70.0, 10.0, "Minimum volume percentile"),
                    ParamSpec::float("rank_by", 0.0, 0.0, 1.0, 1.0, "0=momentum, 1=low_vol"),
                    ParamSpec::float("top_pct", 20.0, 10.0, 50.0, 5.0, "Top % to select"),
                    ParamSpec::int("max_positions", 20, 5, 50, 5, "Maximum positions"),
                ],
                description: "Liquidity Filter: Volume filter + momentum/vol rank".into(),
                fast_supported: false,
                required_columns: vec![DataColumn::Close, DataColumn::Volume],
            },
        );

        blocks.insert(
            "regime_filter".into(),
            BlockSpec {
                block_id: "regime_filter".into(),
                block_type: BlockType::Selection,
                params: vec![
                    ParamSpec::int("vol_lookback", 20, 10, 60, 5, "Volatility lookback"),
                    ParamSpec::float("vol_threshold", 1.5, 1.0, 3.0, 0.5, "Vol z-score threshold"),
                    ParamSpec::float("risk_off_action", 0.0, 0.0, 1.0, 1.0, "0=no selection, 1=full"),
                    ParamSpec::float("top_pct", 30.0, 10.0, 50.0, 5.0, "Top % to select"),
                ],
                description: "Regime Filter: Adapt to vol regime".into(),
                fast_supported: false,
                required_columns: vec![DataColumn::Close],
            },
        );
    }

    fn register_entry_blocks(blocks: &mut HashMap<String, BlockSpec>) {
        // MA Crossover: slow_period max reduced to 200 to ensure compatibility with 1-year data
        blocks.insert(
            "ma_crossover".into(),
            BlockSpec {
                block_id: "ma_crossover".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("fast_period", 50, 5, 100, 5, "Fast MA period"),
                    ParamSpec::int("slow_period", 200, 50, 200, 25, "Slow MA period"),
                ],
                description: "MA Crossover entry".into(),
                fast_supported: false,
                required_columns: vec![DataColumn::Close],
            },
        );

        blocks.insert(
            "rsi".into(),
            BlockSpec {
                block_id: "rsi".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("period", 14, 5, 30, 1, "RSI period"),
                    ParamSpec::float("oversold", 30.0, 10.0, 40.0, 5.0, "Oversold threshold"),
                    ParamSpec::float("overbought", 70.0, 60.0, 90.0, 5.0, "Overbought threshold"),
                ],
                description: "RSI entry".into(),
                fast_supported: false,
                required_columns: vec![DataColumn::Close],
            },
        );

        blocks.insert(
            "macd".into(),
            BlockSpec {
                block_id: "macd".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("fast_ema", 12, 5, 20, 1, "Fast EMA period"),
                    ParamSpec::int("slow_ema", 26, 15, 50, 1, "Slow EMA period"),
                    ParamSpec::int("signal", 9, 5, 15, 1, "Signal line period"),
                ],
                description: "MACD entry".into(),
                fast_supported: false,
                required_columns: vec![DataColumn::Close],
            },
        );

        blocks.insert(
            "bollinger".into(),
            BlockSpec {
                block_id: "bollinger".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("period", 20, 10, 50, 5, "Bollinger period"),
                    ParamSpec::float("std_dev", 2.0, 1.0, 3.0, 0.5, "Standard deviation multiplier"),
                ],
                description: "Bollinger Bands entry".into(),
                fast_supported: false,
                required_columns: vec![DataColumn::Close],
            },
        );

        // B3-calibrated: Z=3 e evento rarissimo, max 2.5 para eventos mais frequentes
        blocks.insert(
            "zscore".into(),
            BlockSpec {
                block_id: "zscore".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("period", 20, 10, 60, 5, "Z-score lookback period"),
                    ParamSpec::float("threshold", 1.5, 1.0, 2.5, 0.25, "Z-score threshold"),
                ],
                description: "Z-Score mean reversion entry".into(),
                fast_supported: false,
                required_columns: vec![DataColumn::Close],
            },
        );

        // ==========================================
        // NEW BLOCKS FROM STRATEGY CATALOG (116)
        // ==========================================

        // Donchian Channel (Turtle Trading)
        // Reference: Donchian (1960), Faith (2003) "Way of the Turtle"
        blocks.insert(
            "donchian".into(),
            BlockSpec {
                block_id: "donchian".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("period", 20, 10, 100, 5, "Entry channel period"),
                    ParamSpec::int("exit_period", 10, 5, 50, 5, "Exit channel period"),
                ],
                description: "Donchian Channel breakout (Turtle Trading)".into(),
                fast_supported: true,
                required_columns: vec![DataColumn::High, DataColumn::Low, DataColumn::Close],
            },
        );

        // Opening Range Breakout
        // Reference: Crabel (1990) "Day Trading with Short Term Price Patterns"
        blocks.insert(
            "orb_breakout".into(),
            BlockSpec {
                block_id: "orb_breakout".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("or_bars", 2, 1, 10, 1, "Opening range bars"),
                    ParamSpec::float("stretch_factor", 0.0, 0.0, 2.0, 0.1, "Breakout stretch multiplier"),
                    ParamSpec::float("min_range_pct", 0.005, 0.001, 0.02, 0.001, "Minimum range filter"),
                ],
                description: "Opening Range Breakout (Crabel)".into(),
                fast_supported: true,
                required_columns: vec![DataColumn::Open, DataColumn::High, DataColumn::Low, DataColumn::Close],
            },
        );

        // VWAP (Volume-Weighted Average Price)
        // Reference: Berkowitz (1988), Almgren & Chriss (2000)
        blocks.insert(
            "vwap".into(),
            BlockSpec {
                block_id: "vwap".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("period", 20, 5, 60, 5, "VWAP lookback period"),
                    ParamSpec::float("threshold", 1.5, 0.5, 3.0, 0.25, "Z-score threshold"),
                    ParamSpec::bool("trend_mode", false, "Trend following (true) or reversion (false)"),
                ],
                description: "VWAP mean reversion or trend following".into(),
                fast_supported: true,
                required_columns: vec![DataColumn::Close, DataColumn::Volume],
            },
        );

        // Gap Fill (Mean Reversion)
        // Reference: Cooper et al. (2003) "Market States and Momentum"
        blocks.insert(
            "gap_fill".into(),
            BlockSpec {
                block_id: "gap_fill".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::float("min_gap_pct", 0.01, 0.005, 0.05, 0.005, "Minimum gap percentage"),
                    ParamSpec::float("max_gap_pct", 0.05, 0.02, 0.15, 0.01, "Maximum gap percentage"),
                ],
                description: "Gap Fill mean reversion strategy".into(),
                fast_supported: true,
                required_columns: vec![DataColumn::Open, DataColumn::Close],
            },
        );

        // Gap Continuation (Momentum)
        blocks.insert(
            "gap_continuation".into(),
            BlockSpec {
                block_id: "gap_continuation".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::float("min_gap_pct", 0.02, 0.01, 0.10, 0.01, "Minimum gap percentage"),
                    ParamSpec::bool("volume_confirm", true, "Require volume confirmation"),
                    ParamSpec::int("volume_lookback", 20, 10, 50, 5, "Volume average lookback"),
                ],
                description: "Gap Continuation momentum strategy".into(),
                fast_supported: true,
                required_columns: vec![DataColumn::Open, DataColumn::Close, DataColumn::Volume],
            },
        );

        // ATR Breakout
        // Reference: Wilder (1978) "New Concepts in Technical Trading Systems"
        blocks.insert(
            "atr_breakout".into(),
            BlockSpec {
                block_id: "atr_breakout".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("period", 14, 5, 30, 1, "ATR period"),
                    ParamSpec::float("multiplier", 2.0, 1.0, 5.0, 0.5, "ATR multiplier for bands"),
                ],
                description: "ATR Breakout volatility-adjusted signals (Wilder)".into(),
                fast_supported: true,
                required_columns: vec![DataColumn::High, DataColumn::Low, DataColumn::Close],
            },
        );

        // Volatility Expansion (Squeeze Breakout)
        // Reference: Bollinger (2002), Mandelbrot (1963)
        blocks.insert(
            "vol_expansion".into(),
            BlockSpec {
                block_id: "vol_expansion".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("period", 20, 10, 50, 5, "Bollinger period"),
                    ParamSpec::float("num_std", 2.0, 1.0, 3.0, 0.5, "Standard deviations"),
                    ParamSpec::float("squeeze_percentile", 20.0, 5.0, 40.0, 5.0, "Squeeze threshold percentile"),
                    ParamSpec::int("lookback", 126, 50, 252, 21, "Historical lookback for percentile"),
                ],
                description: "Volatility Expansion squeeze breakout".into(),
                fast_supported: false,
                required_columns: vec![DataColumn::Close],
            },
        );

        // Volume Breakout
        // Reference: Arms (1971) "Volume Cycles in the Stock Market"
        blocks.insert(
            "volume_breakout".into(),
            BlockSpec {
                block_id: "volume_breakout".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("price_period", 20, 10, 60, 5, "Price channel period"),
                    ParamSpec::int("volume_period", 20, 10, 50, 5, "Volume average period"),
                    ParamSpec::float("volume_threshold", 1.5, 1.2, 3.0, 0.1, "Volume ratio threshold"),
                ],
                description: "Volume-confirmed price breakout".into(),
                fast_supported: true,
                required_columns: vec![DataColumn::Close, DataColumn::Volume],
            },
        );

        // Volume Profile (Market Profile POC/VAH/VAL)
        // Reference: Steidlmayer (1986) "Markets and Market Logic"
        blocks.insert(
            "volume_profile".into(),
            BlockSpec {
                block_id: "volume_profile".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("period", 50, 20, 126, 10, "Profile lookback period"),
                    ParamSpec::int("num_buckets", 20, 10, 50, 5, "Price buckets for profile"),
                    ParamSpec::float("deviation_pct", 0.02, 0.01, 0.05, 0.005, "Deviation from VAH/VAL"),
                ],
                description: "Volume Profile POC/VAH/VAL trading (Market Profile)".into(),
                fast_supported: false,
                required_columns: vec![DataColumn::Close, DataColumn::Volume],
            },
        );

        // VIX Mean Reversion
        // Reference: Whaley (2000) "The Investor Fear Gauge"
        blocks.insert(
            "vix_reversion".into(),
            BlockSpec {
                block_id: "vix_reversion".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("vol_period", 20, 10, 40, 5, "Volatility calculation period"),
                    ParamSpec::int("lookback", 126, 63, 252, 21, "Historical lookback for Z-score"),
                    ParamSpec::float("threshold", 1.5, 1.0, 3.0, 0.25, "Z-score threshold"),
                ],
                description: "VIX-style volatility mean reversion (Whaley)".into(),
                fast_supported: false,
                required_columns: vec![DataColumn::Close],
            },
        );

        // ADX Momentum
        // Reference: Wilder (1978) "New Concepts in Technical Trading Systems"
        blocks.insert(
            "adx_momentum".into(),
            BlockSpec {
                block_id: "adx_momentum".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("period", 14, 7, 28, 1, "ADX/DI period"),
                    ParamSpec::float("adx_threshold", 25.0, 15.0, 40.0, 5.0, "ADX trend strength threshold"),
                ],
                description: "ADX trend-following momentum (Wilder)".into(),
                fast_supported: true,
                required_columns: vec![DataColumn::High, DataColumn::Low, DataColumn::Close],
            },
        );

        // Channel Breakout
        // Reference: Keltner (1960), Covel (2004) "Trend Following"
        blocks.insert(
            "channel_breakout".into(),
            BlockSpec {
                block_id: "channel_breakout".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("period", 20, 10, 100, 5, "Channel lookback period"),
                    ParamSpec::float("buffer_pct", 0.001, 0.0, 0.01, 0.001, "Breakout buffer percentage"),
                ],
                description: "Price channel breakout (new highs/lows)".into(),
                fast_supported: true,
                required_columns: vec![DataColumn::Close],
            },
        );

        // Adaptive Momentum
        // Reference: Kaufman (1995) "Smarter Trading"
        blocks.insert(
            "adaptive_momentum".into(),
            BlockSpec {
                block_id: "adaptive_momentum".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("min_period", 10, 5, 20, 1, "Minimum adaptive period"),
                    ParamSpec::int("max_period", 50, 30, 100, 10, "Maximum adaptive period"),
                    ParamSpec::float("momentum_threshold", 0.05, 0.02, 0.15, 0.01, "Momentum threshold"),
                ],
                description: "Adaptive momentum with efficiency ratio (Kaufman)".into(),
                fast_supported: false,
                required_columns: vec![DataColumn::Close],
            },
        );

        // Sector Rotation (Selection block registered as Entry for GA compatibility)
        blocks.insert(
            "sector_rotation".into(),
            BlockSpec {
                block_id: "sector_rotation".into(),
                block_type: BlockType::Selection,
                params: vec![
                    ParamSpec::int("lookback", 63, 21, 126, 21, "Momentum lookback days"),
                    ParamSpec::int("top_n", 3, 1, 10, 1, "Number of top sectors to select"),
                    ParamSpec::float("min_momentum", -0.10, -0.30, 0.0, 0.05, "Minimum momentum threshold"),
                ],
                description: "Sector rotation by relative strength".into(),
                fast_supported: true,
                required_columns: vec![DataColumn::Close],
            },
        );

        // Multi-Factor Selection
        // Reference: Fama & French (1993), Asness (2013)
        blocks.insert(
            "multi_factor".into(),
            BlockSpec {
                block_id: "multi_factor".into(),
                block_type: BlockType::Selection,
                params: vec![
                    ParamSpec::float("momentum_weight", 0.4, 0.0, 1.0, 0.1, "Momentum factor weight"),
                    ParamSpec::float("volatility_weight", 0.3, 0.0, 1.0, 0.1, "Low-vol factor weight"),
                    ParamSpec::float("value_weight", 0.3, 0.0, 1.0, 0.1, "Value factor weight"),
                    ParamSpec::int("momentum_period", 126, 63, 252, 21, "Momentum lookback"),
                    ParamSpec::int("vol_period", 60, 20, 126, 20, "Volatility period"),
                    ParamSpec::float("top_pct", 20.0, 5.0, 50.0, 5.0, "Top percentage to select"),
                ],
                description: "Multi-factor selection (momentum, value, low-vol)".into(),
                fast_supported: false,
                required_columns: vec![DataColumn::Close],
            },
        );

        // ============================================================
        // NEW BLOCKS - Event-Driven, Pairs, Seasonal, Momentum Variants
        // ============================================================

        // News Volatility
        // Reference: Event-driven trading literature
        blocks.insert(
            "news_volatility".into(),
            BlockSpec {
                block_id: "news_volatility".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("lookback", 20, 10, 50, 5, "Historical volatility lookback"),
                    ParamSpec::float("vol_threshold", 2.0, 1.5, 3.0, 0.25, "Volatility expansion threshold"),
                ],
                description: "Trades volatility expansion around news events".into(),
                fast_supported: true,
                required_columns: vec![DataColumn::Close],
            },
        );

        // Pre-Earnings
        // Reference: Bernard & Thomas (1990)
        blocks.insert(
            "pre_earnings".into(),
            BlockSpec {
                block_id: "pre_earnings".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("lookback", 20, 10, 30, 5, "Volatility lookback period"),
                ],
                description: "Pre-earnings positioning based on vol compression".into(),
                fast_supported: true,
                required_columns: vec![DataColumn::Close],
            },
        );

        // Post-Earnings (PEAD)
        // Reference: Bernard & Thomas (1990)
        blocks.insert(
            "post_earnings".into(),
            BlockSpec {
                block_id: "post_earnings".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::float("threshold", 0.03, 0.02, 0.05, 0.005, "Earnings surprise threshold"),
                ],
                description: "Post-earnings announcement drift (PEAD)".into(),
                fast_supported: true,
                required_columns: vec![DataColumn::Close],
            },
        );

        // Cointegration Pairs Trading
        // Reference: Gatev et al. (2006)
        blocks.insert(
            "cointegration".into(),
            BlockSpec {
                block_id: "cointegration".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("lookback", 60, 30, 90, 10, "Spread calculation lookback"),
                    ParamSpec::float("entry_threshold", 2.0, 1.5, 3.0, 0.25, "Z-score entry threshold"),
                ],
                description: "Cointegration-based pairs trading".into(),
                fast_supported: false,
                required_columns: vec![DataColumn::Close],
            },
        );

        // Cointegration Fast
        blocks.insert(
            "cointegration_fast".into(),
            BlockSpec {
                block_id: "cointegration_fast".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("lookback", 20, 10, 30, 5, "Fast spread lookback"),
                    ParamSpec::float("entry_threshold", 2.0, 1.5, 3.0, 0.25, "Z-score entry threshold"),
                ],
                description: "Fast cointegration pairs trading".into(),
                fast_supported: true,
                required_columns: vec![DataColumn::Close],
            },
        );

        // Distance Pairs Trading
        // Reference: Gatev et al. (2006)
        blocks.insert(
            "distance".into(),
            BlockSpec {
                block_id: "distance".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("lookback", 20, 10, 30, 5, "Distance calculation lookback"),
                    ParamSpec::float("threshold", 2.0, 1.5, 3.0, 0.25, "Distance threshold"),
                ],
                description: "Distance-based pairs trading".into(),
                fast_supported: true,
                required_columns: vec![DataColumn::Close],
            },
        );

        // Distance Fast
        blocks.insert(
            "distance_fast".into(),
            BlockSpec {
                block_id: "distance_fast".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("lookback", 10, 5, 15, 2, "Fast distance lookback"),
                    ParamSpec::float("threshold", 2.0, 1.0, 2.5, 0.25, "Distance threshold"),
                ],
                description: "Fast distance pairs trading".into(),
                fast_supported: true,
                required_columns: vec![DataColumn::Close],
            },
        );

        // Multi-Pair Trading
        blocks.insert(
            "multi_pair".into(),
            BlockSpec {
                block_id: "multi_pair".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("lookback", 20, 10, 30, 5, "Pair spread lookback"),
                    ParamSpec::float("threshold", 1.5, 1.0, 2.0, 0.25, "Entry threshold"),
                ],
                description: "Multi-pair trading strategy".into(),
                fast_supported: true,
                required_columns: vec![DataColumn::Close],
            },
        );

        // Multi-Pair with Dividends
        blocks.insert(
            "multi_pair_div".into(),
            BlockSpec {
                block_id: "multi_pair_div".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("lookback", 20, 10, 30, 5, "Pair spread lookback"),
                    ParamSpec::float("threshold", 1.5, 1.0, 2.0, 0.25, "Entry threshold"),
                ],
                description: "Multi-pair trading with dividend adjustment".into(),
                fast_supported: true,
                required_columns: vec![DataColumn::Close],
            },
        );

        // MA Arbitrage
        blocks.insert(
            "ma_arb".into(),
            BlockSpec {
                block_id: "ma_arb".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("fast_period", 5, 3, 10, 1, "Fast MA period"),
                    ParamSpec::int("slow_period", 20, 15, 30, 5, "Slow MA period"),
                ],
                description: "MA-based arbitrage".into(),
                fast_supported: true,
                required_columns: vec![DataColumn::Close],
            },
        );

        // January Effect
        // Reference: Thaler (1987)
        blocks.insert(
            "january_effect".into(),
            BlockSpec {
                block_id: "january_effect".into(),
                block_type: BlockType::Entry,
                params: vec![], // Calendar-based, no random params
                description: "January effect calendar anomaly".into(),
                fast_supported: true,
                required_columns: vec![DataColumn::Close],
            },
        );

        // Sell in May
        // Reference: Bouman & Jacobsen (2002)
        blocks.insert(
            "sell_in_may".into(),
            BlockSpec {
                block_id: "sell_in_may".into(),
                block_type: BlockType::Entry,
                params: vec![], // Calendar-based, no random params
                description: "Sell in May seasonal effect".into(),
                fast_supported: true,
                required_columns: vec![DataColumn::Close],
            },
        );

        // Grains Seasonal
        blocks.insert(
            "grains_seasonal".into(),
            BlockSpec {
                block_id: "grains_seasonal".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("lookback", 63, 42, 84, 7, "Seasonal lookback"),
                ],
                description: "Grains seasonal patterns".into(),
                fast_supported: true,
                required_columns: vec![DataColumn::Close],
            },
        );

        // Natural Gas Seasonal
        blocks.insert(
            "natgas_seasonal".into(),
            BlockSpec {
                block_id: "natgas_seasonal".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("lookback", 42, 21, 63, 7, "Seasonal lookback"),
                ],
                description: "Natural gas seasonal patterns".into(),
                fast_supported: true,
                required_columns: vec![DataColumn::Close],
            },
        );

        // Dual MA
        blocks.insert(
            "dual_ma".into(),
            BlockSpec {
                block_id: "dual_ma".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("fast_period", 10, 5, 20, 5, "Fast MA period"),
                    ParamSpec::int("slow_period", 30, 20, 60, 10, "Slow MA period"),
                ],
                description: "Dual moving average crossover".into(),
                fast_supported: true,
                required_columns: vec![DataColumn::Close],
            },
        );

        // Trend MA
        blocks.insert(
            "trend_ma".into(),
            BlockSpec {
                block_id: "trend_ma".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("period", 50, 20, 100, 10, "MA trend period"),
                ],
                description: "Trend following with MA filter".into(),
                fast_supported: true,
                required_columns: vec![DataColumn::Close],
            },
        );

        // Time-Series Momentum
        // Reference: Moskowitz, Ooi, Pedersen (2012)
        blocks.insert(
            "time_series".into(),
            BlockSpec {
                block_id: "time_series".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("lookback", 252, 126, 504, 63, "Momentum lookback (annual)"),
                ],
                description: "Time-series momentum (Moskowitz et al.)".into(),
                fast_supported: false,
                required_columns: vec![DataColumn::Close],
            },
        );

        // Cross-Sectional Momentum
        // Reference: Jegadeesh & Titman (1993)
        blocks.insert(
            "cross_sectional".into(),
            BlockSpec {
                block_id: "cross_sectional".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("lookback", 126, 63, 252, 21, "Momentum lookback"),
                ],
                description: "Cross-sectional momentum (Jegadeesh & Titman)".into(),
                fast_supported: false,
                required_columns: vec![DataColumn::Close],
            },
        );

        // Cross-Sectional Multi-period
        blocks.insert(
            "cross_sectional_multi".into(),
            BlockSpec {
                block_id: "cross_sectional_multi".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("lookback", 63, 21, 126, 21, "Multi-period lookback"),
                ],
                description: "Multi-period cross-sectional momentum".into(),
                fast_supported: true,
                required_columns: vec![DataColumn::Close],
            },
        );

        // Buy and Hold (baseline)
        blocks.insert(
            "buy_hold".into(),
            BlockSpec {
                block_id: "buy_hold".into(),
                block_type: BlockType::Entry,
                params: vec![], // No params - always long
                description: "Buy and hold benchmark".into(),
                fast_supported: true,
                required_columns: vec![DataColumn::Close],
            },
        );

        // RSI Filtered
        blocks.insert(
            "rsi_filtered".into(),
            BlockSpec {
                block_id: "rsi_filtered".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("period", 14, 7, 21, 7, "RSI period"),
                    ParamSpec::float("oversold", 30.0, 20.0, 40.0, 5.0, "Oversold threshold"),
                    ParamSpec::float("overbought", 70.0, 60.0, 80.0, 5.0, "Overbought threshold"),
                    ParamSpec::int("ma_period", 50, 30, 70, 10, "MA filter period"),
                ],
                description: "RSI with trend filter".into(),
                fast_supported: true,
                required_columns: vec![DataColumn::Close],
            },
        );

        // Bollinger Filtered
        blocks.insert(
            "bb_filtered".into(),
            BlockSpec {
                block_id: "bb_filtered".into(),
                block_type: BlockType::Entry,
                params: vec![
                    ParamSpec::int("period", 20, 15, 30, 5, "BB period"),
                    ParamSpec::float("std_dev", 2.0, 1.5, 2.5, 0.25, "Standard deviations"),
                ],
                description: "Bollinger Bands with volatility filter".into(),
                fast_supported: true,
                required_columns: vec![DataColumn::Close],
            },
        );
    }

    fn register_exit_blocks(blocks: &mut HashMap<String, BlockSpec>) {
        blocks.insert(
            "stop_loss".into(),
            BlockSpec {
                block_id: "stop_loss".into(),
                block_type: BlockType::Exit,
                params: vec![ParamSpec::float(
                    "threshold_pct",
                    0.10,
                    0.02,
                    0.25,
                    0.02,
                    "Stop loss percentage",
                )],
                description: "Stop-loss exit".into(),
                fast_supported: false,
                required_columns: vec![DataColumn::Close],
            },
        );

        blocks.insert(
            "take_profit".into(),
            BlockSpec {
                block_id: "take_profit".into(),
                block_type: BlockType::Exit,
                params: vec![ParamSpec::float(
                    "target_pct",
                    0.30,
                    0.05,
                    1.0,
                    0.05,
                    "Take profit percentage",
                )],
                description: "Take-profit exit".into(),
                fast_supported: false,
                required_columns: vec![DataColumn::Close],
            },
        );

        blocks.insert(
            "trailing_stop".into(),
            BlockSpec {
                block_id: "trailing_stop".into(),
                block_type: BlockType::Exit,
                params: vec![
                    ParamSpec::float(
                        "trailing_pct",
                        0.15,
                        0.03,
                        0.30,
                        0.03,
                        "Trailing stop percentage",
                    ),
                    ParamSpec::float(
                        "activation_pct",
                        0.10,
                        0.0,
                        0.30,
                        0.05,
                        "Activation gain percentage",
                    ),
                ],
                description: "Trailing stop exit".into(),
                fast_supported: false,
                required_columns: vec![DataColumn::Close, DataColumn::High],
            },
        );

        blocks.insert(
            "time_exit".into(),
            BlockSpec {
                block_id: "time_exit".into(),
                block_type: BlockType::Exit,
                params: vec![ParamSpec::int(
                    "max_days",
                    20,
                    5,
                    252,
                    5,
                    "Maximum days in position",
                )],
                description: "Time-based exit".into(),
                fast_supported: false,
                required_columns: vec![], // No price data needed, just time
            },
        );

        // ==========================================
        // ADVANCED EXIT BLOCKS (Prop-Trading Level)
        // ==========================================

        blocks.insert(
            "chandelier_atr".into(),
            BlockSpec {
                block_id: "chandelier_atr".into(),
                block_type: BlockType::Exit,
                params: vec![
                    ParamSpec::int("period", 22, 10, 50, 2, "ATR/HH lookback period"),
                    ParamSpec::float("multiplier", 3.0, 1.5, 5.0, 0.5, "ATR multiplier"),
                ],
                description: "Chandelier ATR: Trailing stop from highest high".into(),
                fast_supported: false,
                required_columns: vec![DataColumn::Close, DataColumn::High, DataColumn::Low],
            },
        );

        blocks.insert(
            "volatility_spike".into(),
            BlockSpec {
                block_id: "volatility_spike".into(),
                block_type: BlockType::Exit,
                params: vec![
                    ParamSpec::int("lookback", 20, 10, 60, 5, "Volatility lookback"),
                    ParamSpec::float("spike_threshold", 2.0, 1.5, 4.0, 0.5, "Z-score threshold"),
                    ParamSpec::float("exit_pct", 1.0, 0.5, 1.0, 0.25, "Exit percentage"),
                ],
                description: "Volatility Spike: Risk-off on vol regime change".into(),
                fast_supported: false,
                required_columns: vec![DataColumn::Close],
            },
        );

        blocks.insert(
            "time_atr_hybrid".into(),
            BlockSpec {
                block_id: "time_atr_hybrid".into(),
                block_type: BlockType::Exit,
                params: vec![
                    ParamSpec::int("max_days", 5, 1, 20, 1, "Max days in position"),
                    ParamSpec::float("atr_multiplier", 2.0, 1.0, 4.0, 0.5, "ATR multiplier"),
                    ParamSpec::int("atr_period", 14, 7, 21, 1, "ATR period"),
                ],
                description: "Time-ATR Hybrid: Weekly radar with ATR stop".into(),
                fast_supported: false,
                required_columns: vec![DataColumn::Close, DataColumn::High, DataColumn::Low],
            },
        );

        blocks.insert(
            "drawdown_limit".into(),
            BlockSpec {
                block_id: "drawdown_limit".into(),
                block_type: BlockType::Exit,
                params: vec![
                    ParamSpec::float("max_drawdown_pct", 0.15, 0.05, 0.30, 0.05, "Max drawdown %"),
                    ParamSpec::float("use_portfolio", 0.0, 0.0, 1.0, 1.0, "0=position, 1=portfolio"),
                ],
                description: "Drawdown Limit: Exit on max DD".into(),
                fast_supported: false,
                required_columns: vec![DataColumn::Close],
            },
        );

        blocks.insert(
            "trend_reversal".into(),
            BlockSpec {
                block_id: "trend_reversal".into(),
                block_type: BlockType::Exit,
                params: vec![
                    ParamSpec::int("ma_period", 20, 10, 50, 5, "MA period for slope"),
                    ParamSpec::float("slope_threshold", 0.0, -0.02, 0.02, 0.005, "Slope threshold"),
                ],
                description: "Trend Reversal: Exit on MA slope change".into(),
                fast_supported: false,
                required_columns: vec![DataColumn::Close],
            },
        );
    }

    fn register_sizing_blocks(blocks: &mut HashMap<String, BlockSpec>) {
        // ARROJADO-COMPATIBLE: Adjusted max_weight range from 0.05-0.50 to 0.10-0.40
        blocks.insert(
            "equal_weight".into(),
            BlockSpec {
                block_id: "equal_weight".into(),
                block_type: BlockType::Sizing,
                params: vec![
                    ParamSpec::float("max_weight", 0.25, 0.10, 0.40, 0.05, "Maximum weight per position"),
                    ParamSpec::float("min_weight", 0.02, 0.01, 0.10, 0.01, "Minimum weight per position"),
                    ParamSpec::int("max_positions", 15, 5, 30, 5, "Maximum number of positions"),
                ],
                description: "Equal weight sizing (1/N)".into(),
                fast_supported: true,
                required_columns: vec![], // No price data needed
            },
        );

        blocks.insert(
            "risk_parity".into(),
            BlockSpec {
                block_id: "risk_parity".into(),
                block_type: BlockType::Sizing,
                params: vec![
                    ParamSpec::float("max_weight", 0.20, 0.05, 0.50, 0.05, "Maximum weight"),
                    ParamSpec::float("min_weight", 0.02, 0.01, 0.10, 0.01, "Minimum weight"),
                    ParamSpec::int("max_positions", 20, 5, 50, 5, "Maximum positions"),
                    ParamSpec::float("fallback_vol", 0.25, 0.10, 0.50, 0.05, "Fallback volatility"),
                ],
                description: "Risk parity sizing (inverse volatility)".into(),
                fast_supported: false,
                required_columns: vec![DataColumn::Close], // Needs price for volatility
            },
        );

        blocks.insert(
            "vol_targeting".into(),
            BlockSpec {
                block_id: "vol_targeting".into(),
                block_type: BlockType::Sizing,
                params: vec![
                    ParamSpec::float("target_vol", 0.12, 0.05, 0.30, 0.02, "Target portfolio volatility"),
                    ParamSpec::float("max_weight", 0.30, 0.10, 0.50, 0.05, "Maximum weight"),
                    ParamSpec::float("min_weight", 0.02, 0.01, 0.10, 0.01, "Minimum weight"),
                    ParamSpec::float("max_leverage", 1.0, 0.5, 2.0, 0.1, "Maximum leverage"),
                    ParamSpec::float("correlation", 0.5, 0.0, 1.0, 0.1, "Assumed correlation"),
                    ParamSpec::int("max_positions", 10, 5, 30, 5, "Maximum positions"),
                ],
                description: "Volatility targeting sizing".into(),
                fast_supported: false,
                required_columns: vec![DataColumn::Close], // Needs price for volatility
            },
        );

        // ==========================================
        // ADVANCED SIZING BLOCKS (Prop-Trading Level)
        // ==========================================

        blocks.insert(
            "kelly_fractional".into(),
            BlockSpec {
                block_id: "kelly_fractional".into(),
                block_type: BlockType::Sizing,
                params: vec![
                    ParamSpec::float("kelly_fraction", 0.25, 0.10, 0.50, 0.05, "Fraction of full Kelly"),
                    ParamSpec::float("max_weight", 0.20, 0.10, 0.40, 0.05, "Maximum weight"),
                    ParamSpec::float("min_weight", 0.02, 0.01, 0.10, 0.01, "Minimum weight"),
                    ParamSpec::int("max_positions", 10, 5, 20, 5, "Maximum positions"),
                    ParamSpec::float("assumed_win_rate", 0.55, 0.40, 0.70, 0.05, "Assumed win rate"),
                    ParamSpec::float("assumed_win_loss_ratio", 1.5, 1.0, 3.0, 0.25, "Assumed W/L ratio"),
                ],
                description: "Kelly Fractional: Optimal sizing with safety cap".into(),
                fast_supported: true,
                required_columns: vec![],
            },
        );

        blocks.insert(
            "vol_target_buffer".into(),
            BlockSpec {
                block_id: "vol_target_buffer".into(),
                block_type: BlockType::Sizing,
                params: vec![
                    ParamSpec::float("target_vol", 0.12, 0.06, 0.20, 0.02, "Target portfolio vol"),
                    ParamSpec::float("cash_buffer", 0.20, 0.10, 0.40, 0.05, "Mandatory cash reserve"),
                    ParamSpec::float("max_weight", 0.15, 0.05, 0.30, 0.05, "Maximum weight"),
                    ParamSpec::float("min_weight", 0.02, 0.01, 0.10, 0.01, "Minimum weight"),
                    ParamSpec::int("max_positions", 15, 5, 30, 5, "Maximum positions"),
                    ParamSpec::float("fallback_vol", 0.25, 0.15, 0.40, 0.05, "Fallback vol"),
                ],
                description: "Vol Target Buffer: Vol targeting with cash reserve".into(),
                fast_supported: false,
                required_columns: vec![DataColumn::Close],
            },
        );

        blocks.insert(
            "exposure_cap".into(),
            BlockSpec {
                block_id: "exposure_cap".into(),
                block_type: BlockType::Sizing,
                params: vec![
                    ParamSpec::float("max_exposure", 0.80, 0.50, 1.20, 0.10, "Maximum total exposure"),
                    ParamSpec::float("max_weight", 0.20, 0.10, 0.40, 0.05, "Maximum weight"),
                    ParamSpec::float("min_weight", 0.02, 0.01, 0.10, 0.01, "Minimum weight"),
                    ParamSpec::int("max_positions", 15, 5, 30, 5, "Maximum positions"),
                ],
                description: "Exposure Cap: Hard limit on total exposure".into(),
                fast_supported: true,
                required_columns: vec![],
            },
        );
    }

    /// Get block specification by ID.
    pub fn get_block(&self, block_id: &str) -> Option<&BlockSpec> {
        self.blocks.get(block_id)
    }

    /// Get all blocks of a specific type.
    pub fn blocks_by_type(&self, block_type: BlockType) -> Vec<&BlockSpec> {
        self.blocks
            .values()
            .filter(|b| b.block_type == block_type)
            .collect()
    }

    /// Get all block IDs of a specific type.
    pub fn block_ids_by_type(&self, block_type: BlockType) -> Vec<&str> {
        self.blocks
            .values()
            .filter(|b| b.block_type == block_type)
            .map(|b| b.block_id.as_str())
            .collect()
    }

    /// Get block IDs that can run with the given available data columns.
    /// 
    /// This filters blocks based on their `required_columns` field.
    /// A block is available if ALL its required columns are present in `available`.
    /// 
    /// # Arguments
    /// * `block_type` - The type of blocks to filter
    /// * `available` - Set of data columns available in the dataset
    /// 
    /// # Example
    /// ```ignore
    /// use std::collections::HashSet;
    /// let available: HashSet<DataColumn> = [DataColumn::Open, DataColumn::Close, DataColumn::Volume]
    ///     .into_iter().collect();
    /// let usable = ranges.block_ids_for_available_data(BlockType::Selection, &available);
    /// // Returns only blocks that work with OHLCV (no fundamental data needed)
    /// ```
    pub fn block_ids_for_available_data(
        &self,
        block_type: BlockType,
        available: &std::collections::HashSet<DataColumn>,
    ) -> Vec<&str> {
        self.blocks
            .values()
            .filter(|b| {
                b.block_type == block_type &&
                b.required_columns.iter().all(|col| available.contains(col))
            })
            .map(|b| b.block_id.as_str())
            .collect()
    }

    /// Get blocks that require only OHLCV data (no fundamental data needed).
    /// 
    /// This is a convenience method for datasets that only have price/volume data.
    pub fn ohlcv_only_block_ids(&self, block_type: BlockType) -> Vec<&str> {
        self.blocks
            .values()
            .filter(|b| {
                b.block_type == block_type &&
                b.required_columns.iter().all(|col| col.is_ohlcv())
            })
            .map(|b| b.block_id.as_str())
            .collect()
    }

    /// Get the list of blocks that are unavailable due to missing data.
    /// Returns a list of (block_id, missing_columns).
    pub fn unavailable_blocks_for_data(
        &self,
        available: &std::collections::HashSet<DataColumn>,
    ) -> Vec<(&str, Vec<DataColumn>)> {
        self.blocks
            .values()
            .filter_map(|b| {
                let missing: Vec<DataColumn> = b.required_columns
                    .iter()
                    .filter(|col| !available.contains(col))
                    .copied()
                    .collect();
                if missing.is_empty() {
                    None
                } else {
                    Some((b.block_id.as_str(), missing))
                }
            })
            .collect()
    }

    /// Check if a block ID exists.
    pub fn contains(&self, block_id: &str) -> bool {
        self.blocks.contains_key(block_id)
    }

    /// Get all block IDs.
    pub fn all_block_ids(&self) -> Vec<&str> {
        self.blocks.keys().map(|s| s.as_str()).collect()
    }

    /// Apply restrictions to narrow the parameter space.
    /// 
    /// This method returns a new ParamRanges with:
    /// - Only blocks from allowed families (if specified)
    /// - Narrowed numeric ranges based on provided bounds
    /// 
    /// # Arguments
    /// * `allowed_blocks` - Optional list of block IDs to keep (filters out others)
    /// * `max_parameters` - Optional maximum number of parameters per block
    /// 
    /// # Example
    /// ```ignore
    /// let ranges = ParamRanges::new()
    ///     .with_restrictions(Some(&["momentum", "equal_weight"]), Some(5));
    /// ```
    pub fn with_restrictions(
        mut self,
        allowed_blocks: Option<&[&str]>,
        max_parameters: Option<usize>,
    ) -> Self {
        // Filter blocks if allowed list provided
        if let Some(allowed) = allowed_blocks {
            self.blocks.retain(|id, _| allowed.contains(&id.as_str()));
        }

        // Limit parameters per block if specified
        if let Some(max_params) = max_parameters {
            for block in self.blocks.values_mut() {
                if block.params.len() > max_params {
                    // Keep only the first N parameters (typically the most important)
                    block.params.truncate(max_params);
                }
            }
        }

        self
    }

    /// Apply numeric bounds to narrow parameter ranges.
    /// 
    /// # Arguments
    /// * `block_id` - The block to modify
    /// * `param_name` - The parameter to narrow
    /// * `new_min` - New minimum value (None to keep current)
    /// * `new_max` - New maximum value (None to keep current)
    /// 
    /// The resulting range is the intersection of old and new bounds.
    pub fn narrow_param_range(
        &mut self,
        block_id: &str,
        param_name: &str,
        new_min: Option<f64>,
        new_max: Option<f64>,
    ) -> bool {
        if let Some(block) = self.blocks.get_mut(block_id) {
            for param in &mut block.params {
                if param.name == param_name {
                    // Get current bounds and narrow them
                    match &mut param.default {
                        ParamValue::Float { min, max, .. } => {
                            if let Some(nm) = new_min {
                                *min = min.max(nm);
                            }
                            if let Some(nx) = new_max {
                                *max = max.min(nx);
                            }
                            return true;
                        }
                        ParamValue::Int { min, max, .. } => {
                            if let Some(nm) = new_min {
                                *min = (*min).max(nm as i64);
                            }
                            if let Some(nx) = new_max {
                                *max = (*max).min(nx as i64);
                            }
                            return true;
                        }
                        _ => {}
                    }
                }
            }
        }
        false
    }

    /// Get the total count of optimizable parameters across all blocks.
    pub fn total_parameter_count(&self) -> usize {
        self.blocks.values().map(|b| b.params.len()).sum()
    }

    /// Filter blocks based on available data columns.
    /// 
    /// This method removes blocks whose `required_columns` are not satisfied
    /// by the provided available columns. Returns a new ParamRanges instance.
    /// 
    /// # Arguments
    /// * `available` - Set of data columns available in the dataset
    /// 
    /// # Returns
    /// A tuple of (filtered_ranges, disabled_blocks) where disabled_blocks
    /// contains (block_id, missing_columns) for blocks that were removed.
    /// 
    /// # Example
    /// ```ignore
    /// use std::collections::HashSet;
    /// let available: HashSet<DataColumn> = [
    ///     DataColumn::Open, DataColumn::High, DataColumn::Low,
    ///     DataColumn::Close, DataColumn::Volume, DataColumn::AdjClose,
    /// ].into_iter().collect();
    /// 
    /// let (ranges, disabled) = ranges.with_available_data(&available);
    /// for (block, missing) in disabled {
    ///     println!("Block '{}' disabled - missing: {:?}", block, missing);
    /// }
    /// ```
    pub fn with_available_data(
        mut self,
        available: &std::collections::HashSet<DataColumn>,
    ) -> (Self, Vec<(String, Vec<DataColumn>)>) {
        let mut disabled = Vec::new();
        
        self.blocks.retain(|block_id, spec| {
            let missing: Vec<DataColumn> = spec.required_columns
                .iter()
                .filter(|col| !available.contains(col))
                .copied()
                .collect();
            
            if missing.is_empty() {
                true // Keep block
            } else {
                disabled.push((block_id.clone(), missing));
                false // Remove block
            }
        });
        
        (self, disabled)
    }

    /// Create parameter ranges with only OHLCV-compatible blocks.
    /// 
    /// This is a convenience method for datasets that only have price/volume data
    /// (no fundamentals like P/E, dividends, market cap, etc).
    /// 
    /// # Returns
    /// A tuple of (filtered_ranges, disabled_blocks) listing which blocks were removed.
    pub fn with_ohlcv_only(self) -> (Self, Vec<(String, Vec<DataColumn>)>) {
        let available: std::collections::HashSet<DataColumn> = [
            DataColumn::Open,
            DataColumn::High,
            DataColumn::Low,
            DataColumn::Close,
            DataColumn::Volume,
            DataColumn::AdjClose,
        ].into_iter().collect();
        
        self.with_available_data(&available)
    }

    /// Create parameter ranges calibrated for a specific market.
    /// 
    /// # Arguments
    /// * `market` - Market identifier: "BR" (Brazilian B3) or "US" (American S&P 500)
    /// 
    /// # Market-Specific Adjustments
    /// 
    /// **BR (Brazil - B3)**
    /// - Base configuration (Selic ~10%, vol 25-40%, DY 4-7%)
    /// - Market cap: R$500M - R$500B
    /// 
    /// **US (United States - S&P 500)**
    /// - Lower volatility: 15-45% (VIX 15-20 vs VBRA 25-35)
    /// - Lower carry: -2% to +3% (Fed Funds ~5%, DY ~1-2%)
    /// - Higher market cap: $5B - $2T (large-cap focus)
    /// - Tighter dividend yield: 0.5% - 5%
    pub fn new_for_market(market: &str) -> Self {
        let mut ranges = Self::new(); // Base BR configuration
        
        if market.to_uppercase() == "US" {
            // Volatility: US has lower volatility (VIX 15-20 vs VBRA 25-35)
            ranges.narrow_param_range("low_vol", "max_annualized_vol", Some(0.15), Some(0.45));
            
            // Carry: US has lower yields, Fed Funds ~5%
            ranges.narrow_param_range("carry", "min_carry", Some(-0.02), Some(0.03));
            
            // Market cap: US large-cap focus ($5B - $2T)
            ranges.narrow_param_range("size", "min_market_cap", Some(5_000_000_000.0), None);
            ranges.narrow_param_range("size", "max_market_cap", None, Some(2_000_000_000_000.0));
            
            // Dividend yield: US typically lower (0.5% - 5%)
            ranges.narrow_param_range("dividend", "min_yield", Some(0.005), Some(0.05));
        }
        
        ranges
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_ranges_creation() {
        let ranges = ParamRanges::new();

        // Check selection blocks
        assert!(ranges.contains("momentum"));
        assert!(ranges.contains("value"));
        assert!(ranges.contains("quality"));

        // Check entry blocks
        assert!(ranges.contains("ma_crossover"));
        assert!(ranges.contains("rsi"));

        // Check exit blocks
        assert!(ranges.contains("stop_loss"));
        assert!(ranges.contains("trailing_stop"));

        // Check sizing blocks
        assert!(ranges.contains("equal_weight"));
        assert!(ranges.contains("risk_parity"));
    }

    #[test]
    fn test_blocks_by_type() {
        let ranges = ParamRanges::new();

        let selection = ranges.blocks_by_type(BlockType::Selection);
        // 7 base + 4 factor variants + 2 advanced = varying
        assert!(selection.len() >= 9, "Expected at least 9 selection blocks, got {}", selection.len());

        let sizing = ranges.blocks_by_type(BlockType::Sizing);
        assert_eq!(sizing.len(), 6); // 3 base + 3 advanced (kelly_fractional, vol_target_buffer, exposure_cap)
        
        let exit = ranges.blocks_by_type(BlockType::Exit);
        assert_eq!(exit.len(), 9); // 4 base + 5 advanced
    }

    #[test]
    fn test_default_params() {
        let ranges = ParamRanges::new();
        let momentum = ranges.get_block("momentum").unwrap();

        let defaults = momentum.default_params();
        assert!(defaults.contains_key("lookback_days"));
        assert!(defaults.contains_key("top_pct"));
    }
}


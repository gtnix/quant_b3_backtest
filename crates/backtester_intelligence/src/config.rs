//! Configuration types for intelligence modules.

use serde::{Deserialize, Serialize};

/// Top-level intelligence configuration.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct IntelligenceConfig {
    /// Whether asset filters are enabled.
    #[serde(default)]
    pub enabled: bool,

    /// Filter combination mode.
    #[serde(default)]
    pub mode: FilterMode,

    /// Top percentile to select (e.g., 20 = top 20%).
    #[serde(default = "default_top_percentile")]
    pub top_percentile: f64,

    /// Rebalance frequency.
    #[serde(default = "default_rebalance")]
    pub rebalance_frequency: String,

    /// Individual filter configurations.
    #[serde(default)]
    pub momentum: Option<MomentumConfig>,

    #[serde(default)]
    pub value: Option<ValueConfig>,

    #[serde(default)]
    pub quality: Option<QualityConfig>,

    #[serde(default)]
    pub low_vol: Option<LowVolConfig>,

    #[serde(default)]
    pub dividend_yield: Option<DividendYieldConfig>,

    #[serde(default)]
    pub size: Option<SizeConfig>,

    #[serde(default)]
    pub carry: Option<CarryConfig>,
}

fn default_top_percentile() -> f64 {
    20.0
}
fn default_rebalance() -> String {
    "weekly".to_string()
}

/// Filter combination mode.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum FilterMode {
    /// Only assets passing ALL filters are selected.
    #[default]
    Intersection,

    /// Assets are scored by weighted combination of filters.
    Score,

    /// Filters are applied in sequence, each filtering the previous result.
    Cascade,
}

/// Base filter configuration with common fields.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct AssetFilterConfig {
    /// Whether this filter is enabled.
    #[serde(default)]
    pub enabled: bool,

    /// Weight for scoring mode (0.0 to 1.0).
    #[serde(default)]
    pub weight: f64,
}

/// Momentum filter configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MomentumConfig {
    #[serde(flatten)]
    pub base: AssetFilterConfig,

    /// Lookback period in days.
    #[serde(default = "default_momentum_lookback")]
    pub lookback_days: i32,

    /// Minimum return threshold.
    #[serde(default)]
    pub min_return: f64,

    /// Skip last N days (avoid short-term reversal).
    #[serde(default = "default_skip_days")]
    pub skip_last_days: i32,
}

fn default_momentum_lookback() -> i32 {
    126
}
fn default_skip_days() -> i32 {
    21
}

impl Default for MomentumConfig {
    fn default() -> Self {
        Self {
            base: AssetFilterConfig {
                enabled: false,
                weight: 0.25,
            },
            lookback_days: 126,
            min_return: 0.0,
            skip_last_days: 21,
        }
    }
}

/// Value filter configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ValueConfig {
    #[serde(flatten)]
    pub base: AssetFilterConfig,

    /// Maximum P/E ratio.
    #[serde(default = "default_max_pe")]
    pub max_pe: f64,

    /// Maximum P/B ratio.
    #[serde(default = "default_max_pb")]
    pub max_pb: f64,

    /// Minimum P/E (to avoid negative earnings).
    #[serde(default)]
    pub min_pe: f64,
}

fn default_max_pe() -> f64 {
    15.0
}
fn default_max_pb() -> f64 {
    2.0
}

impl Default for ValueConfig {
    fn default() -> Self {
        Self {
            base: AssetFilterConfig {
                enabled: false,
                weight: 0.20,
            },
            max_pe: 15.0,
            max_pb: 2.0,
            min_pe: 0.0,
        }
    }
}

/// Quality filter configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct QualityConfig {
    #[serde(flatten)]
    pub base: AssetFilterConfig,

    /// Minimum ROE.
    #[serde(default = "default_min_roe")]
    pub min_roe: f64,

    /// Maximum Debt/Equity ratio.
    #[serde(default = "default_max_de")]
    pub max_debt_equity: f64,

    /// Minimum profit margin.
    #[serde(default = "default_min_margin")]
    pub min_profit_margin: f64,

    /// Minimum gross margin.
    #[serde(default)]
    pub min_gross_margin: Option<f64>,
}

fn default_min_roe() -> f64 {
    0.12
}
fn default_max_de() -> f64 {
    1.0
}
fn default_min_margin() -> f64 {
    0.05
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            base: AssetFilterConfig {
                enabled: false,
                weight: 0.20,
            },
            min_roe: 0.12,
            max_debt_equity: 1.0,
            min_profit_margin: 0.05,
            min_gross_margin: None,
        }
    }
}

/// Low volatility filter configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LowVolConfig {
    #[serde(flatten)]
    pub base: AssetFilterConfig,

    /// Lookback period for volatility calculation.
    #[serde(default = "default_vol_lookback")]
    pub lookback_days: i32,

    /// Maximum annualized volatility.
    #[serde(default = "default_max_vol")]
    pub max_annualized_vol: f64,
}

fn default_vol_lookback() -> i32 {
    60
}
fn default_max_vol() -> f64 {
    0.30
}

impl Default for LowVolConfig {
    fn default() -> Self {
        Self {
            base: AssetFilterConfig {
                enabled: false,
                weight: 0.15,
            },
            lookback_days: 60,
            max_annualized_vol: 0.30,
        }
    }
}

/// Dividend yield filter configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DividendYieldConfig {
    #[serde(flatten)]
    pub base: AssetFilterConfig,

    /// Minimum dividend yield.
    #[serde(default = "default_min_dy")]
    pub min_yield: f64,

    /// Maximum dividend yield (to avoid value traps).
    #[serde(default)]
    pub max_yield: Option<f64>,
}

fn default_min_dy() -> f64 {
    0.03
}

impl Default for DividendYieldConfig {
    fn default() -> Self {
        Self {
            base: AssetFilterConfig {
                enabled: false,
                weight: 0.10,
            },
            min_yield: 0.03,
            max_yield: Some(0.15),
        }
    }
}

/// Size filter configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SizeConfig {
    #[serde(flatten)]
    pub base: AssetFilterConfig,

    /// Minimum market cap in BRL.
    #[serde(default = "default_min_cap")]
    pub min_market_cap: i64,

    /// Maximum market cap (None = no limit).
    #[serde(default)]
    pub max_market_cap: Option<i64>,
}

fn default_min_cap() -> i64 {
    5_000_000_000
} // R$ 5 bi

impl Default for SizeConfig {
    fn default() -> Self {
        Self {
            base: AssetFilterConfig {
                enabled: false,
                weight: 0.10,
            },
            min_market_cap: 5_000_000_000,
            max_market_cap: None,
        }
    }
}

/// Carry filter configuration (Technique 7).
/// Carry = dividend_yield - risk_free_rate
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CarryConfig {
    #[serde(flatten)]
    pub base: AssetFilterConfig,

    /// Minimum carry threshold (carry must be >= min_carry to pass).
    #[serde(default)]
    pub min_carry: f64,

    /// Fallback SELIC rate for BR when no historical data available (annualized).
    #[serde(default = "default_selic")]
    pub fallback_selic_br: f64,

    /// Fallback T-Bill 3M rate for US when no historical data available (annualized).
    #[serde(default = "default_tbill")]
    pub fallback_tbill_us: f64,
}

fn default_selic() -> f64 {
    0.1075
} // 10.75% SELIC Meta (Dec 2024)
fn default_tbill() -> f64 {
    0.0435
} // ~4.35% T-Bill 3M (Dec 2024)

impl Default for CarryConfig {
    fn default() -> Self {
        Self {
            base: AssetFilterConfig {
                enabled: false,
                weight: 0.15,
            },
            min_carry: 0.0,
            fallback_selic_br: 0.1075,
            fallback_tbill_us: 0.0435,
        }
    }
}

/// Risk-free rate provider configuration.
/// Controls whether to use database or fallback rates.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RiskFreeConfig {
    /// Source: "db" for database, "fallback" for static values.
    #[serde(default = "default_rf_source")]
    pub source: String,

    /// Rate type for BR market.
    #[serde(default = "default_rate_type_br")]
    pub rate_type_br: String,

    /// Rate type for US market.
    #[serde(default = "default_rate_type_us")]
    pub rate_type_us: String,

    /// Allow fallback when DB has no data for a date.
    #[serde(default = "default_allow_fallback")]
    pub allow_fallback: bool,

    /// Fallback SELIC rate (used if allow_fallback=true and no DB data).
    #[serde(default = "default_selic")]
    pub fallback_selic_br: f64,

    /// Fallback T-Bill rate.
    #[serde(default = "default_tbill")]
    pub fallback_tbill_us: f64,
}

fn default_rf_source() -> String {
    "fallback".into()
}
fn default_rate_type_br() -> String {
    "SELIC".into()
}
fn default_rate_type_us() -> String {
    "TBILL_3M".into()
}
fn default_allow_fallback() -> bool {
    true
}

impl Default for RiskFreeConfig {
    fn default() -> Self {
        Self {
            source: "fallback".into(),
            rate_type_br: "SELIC".into(),
            rate_type_us: "TBILL_3M".into(),
            allow_fallback: true,
            fallback_selic_br: 0.1075,
            fallback_tbill_us: 0.0435,
        }
    }
}

impl RiskFreeConfig {
    /// Check if using database source.
    pub fn use_db(&self) -> bool {
        self.source.to_lowercase() == "db"
    }
}

/// Configuration for fundamentals data loading (point-in-time).
/// Controls availability lag to prevent look-ahead bias.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FundamentalsConfig {
    /// Enable availability lag (data is not available immediately after quarter end).
    #[serde(default)]
    pub use_availability_lag: bool,

    /// Availability lag for BR market in days (typical: 45-90 days after quarter).
    #[serde(default = "default_lag_br")]
    pub lag_days_br: i64,

    /// Availability lag for US market in days (typical: 30-60 days after quarter).
    #[serde(default = "default_lag_us")]
    pub lag_days_us: i64,

    /// Policy when fundamentals are missing: "drop" (exclude from filters), "neutral" (score 0.5).
    #[serde(default = "default_missing_policy")]
    pub missing_policy: String,
}

fn default_lag_br() -> i64 {
    60
} // BR: ~60 days after quarter end
fn default_lag_us() -> i64 {
    45
} // US: ~45 days after quarter end
fn default_missing_policy() -> String {
    "drop".into()
}

impl Default for FundamentalsConfig {
    fn default() -> Self {
        Self {
            use_availability_lag: false, // Conservative default: no lag
            lag_days_br: 60,
            lag_days_us: 45,
            missing_policy: "drop".into(),
        }
    }
}

impl FundamentalsConfig {
    /// Get lag for a specific market in days.
    pub fn lag_days(&self, market: crate::filters::Market) -> i64 {
        if !self.use_availability_lag {
            return 0;
        }
        match market {
            crate::filters::Market::BR => self.lag_days_br,
            crate::filters::Market::US => self.lag_days_us,
        }
    }

    /// Calculate cutoff date for point-in-time query.
    pub fn cutoff_date(
        &self,
        backtest_date: chrono::NaiveDate,
        market: crate::filters::Market,
    ) -> chrono::NaiveDate {
        let lag = self.lag_days(market);
        backtest_date - chrono::Duration::days(lag)
    }
}

impl IntelligenceConfig {
    /// Parse from TOML string.
    pub fn from_toml(content: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(content)
    }

    /// Get list of enabled filters with their weights.
    pub fn enabled_filters(&self) -> Vec<(&str, f64)> {
        let mut filters = Vec::new();

        if let Some(ref m) = self.momentum {
            if m.base.enabled {
                filters.push(("momentum", m.base.weight));
            }
        }
        if let Some(ref v) = self.value {
            if v.base.enabled {
                filters.push(("value", v.base.weight));
            }
        }
        if let Some(ref q) = self.quality {
            if q.base.enabled {
                filters.push(("quality", q.base.weight));
            }
        }
        if let Some(ref l) = self.low_vol {
            if l.base.enabled {
                filters.push(("low_vol", l.base.weight));
            }
        }
        if let Some(ref d) = self.dividend_yield {
            if d.base.enabled {
                filters.push(("dividend_yield", d.base.weight));
            }
        }
        if let Some(ref s) = self.size {
            if s.base.enabled {
                filters.push(("size", s.base.weight));
            }
        }
        if let Some(ref c) = self.carry {
            if c.base.enabled {
                filters.push(("carry", c.base.weight));
            }
        }

        filters
    }

    /// Get total weight of enabled filters.
    pub fn total_weight(&self) -> f64 {
        self.enabled_filters().iter().map(|(_, w)| w).sum()
    }
}

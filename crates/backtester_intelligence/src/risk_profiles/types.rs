//! Parameter types for risk profiles.

use serde::{Deserialize, Serialize};

/// Position sizing parameters.
/// 
/// Based on Ziemba & MacLean (2011) and Vince (1992).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SizingParams {
    /// Fraction of Kelly criterion to use (0.1 to 0.5).
    /// Half-Kelly (0.5) is the academic maximum for practical use.
    /// Source: Ziemba & MacLean (2011), Thorp (2006)
    pub kelly_fraction: f64,

    /// Maximum risk per trade as percentage of capital (0.0025 to 0.025).
    /// Acts as a hard cap on position sizing.
    /// Source: Vince (1992), Chan (2021)
    pub max_risk_per_trade_pct: f64,

    /// Maximum exposure to a single asset (0.05 to 0.30).
    pub max_exposure_per_asset_pct: f64,

    /// Maximum exposure to a single sector (0.20 to 0.50).
    pub max_sector_concentration_pct: f64,

    /// Maximum number of simultaneous positions.
    pub max_positions: u32,

    /// Minimum position weight (below this, don't open).
    pub min_position_weight: f64,
}

impl Default for SizingParams {
    fn default() -> Self {
        Self {
            kelly_fraction: 0.25,
            max_risk_per_trade_pct: 0.01,
            max_exposure_per_asset_pct: 0.20,
            max_sector_concentration_pct: 0.30,
            max_positions: 20,
            min_position_weight: 0.02,
        }
    }
}

/// Stop-loss parameters.
/// 
/// ATR-based stops are preferred per Chan (2013) and Wilder (1978).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopParams {
    /// Stop-loss type: "ATR", "Percentage", "Volatility"
    pub stop_type: String,

    /// ATR period for calculation (typically 14 or 20).
    pub atr_period: u32,

    /// ATR multiplier for stop distance.
    /// BR: 2.5-4.0x, US: 2.0-3.5x
    /// Source: Wilder (1978), Chan (2013)
    pub atr_multiplier: f64,

    /// Fixed percentage stop (used if stop_type = "Percentage").
    pub fixed_stop_pct: f64,

    /// Enable trailing stop.
    pub enable_trailing: bool,

    /// Trailing stop activation threshold (gain required to activate).
    pub trailing_activation_pct: f64,

    /// Trailing stop distance as ATR multiplier.
    pub trailing_atr_multiplier: f64,
}

impl Default for StopParams {
    fn default() -> Self {
        Self {
            stop_type: "ATR".to_string(),
            atr_period: 14,
            atr_multiplier: 2.5,
            fixed_stop_pct: 0.10,
            enable_trailing: true,
            trailing_activation_pct: 0.10,
            trailing_atr_multiplier: 2.0,
        }
    }
}

/// Portfolio-level risk parameters.
/// 
/// Based on Harvey et al. (2017) for volatility targeting
/// and Chekhlov et al. (2003) for drawdown constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioRiskParams {
    /// Target portfolio volatility (annualized).
    /// BR: typically higher due to market volatility.
    /// Source: Harvey et al. (2017)
    pub volatility_target: f64,

    /// Maximum portfolio drawdown (negative value, e.g., -0.15 for 15%).
    /// Triggers circuit breaker when exceeded.
    /// Source: Chekhlov et al. (2003)
    pub max_drawdown_pct: f64,

    /// Maximum leverage allowed.
    pub max_leverage: f64,

    /// CVaR limit at 95% confidence (negative value).
    pub cvar_limit_95: f64,
}

impl Default for PortfolioRiskParams {
    fn default() -> Self {
        Self {
            volatility_target: 0.12,
            max_drawdown_pct: -0.15,
            max_leverage: 1.0,
            cvar_limit_95: -0.03,
        }
    }
}

/// Circuit breaker parameters for loss limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerParams {
    /// Daily loss limit that triggers pause (negative).
    pub daily_loss_limit_pct: f64,

    /// Weekly loss limit that triggers pause (negative).
    pub weekly_loss_limit_pct: f64,

    /// Monthly loss limit (negative).
    pub monthly_loss_limit_pct: f64,

    /// Action on drawdown breach: "CashOut", "ReduceRisk", "Alert"
    pub drawdown_action: String,

    /// Cooldown period in days after circuit breaker trips.
    pub cooldown_days: u32,
}

impl Default for CircuitBreakerParams {
    fn default() -> Self {
        Self {
            daily_loss_limit_pct: -0.02,
            weekly_loss_limit_pct: -0.05,
            monthly_loss_limit_pct: -0.10,
            drawdown_action: "ReduceRisk".to_string(),
            cooldown_days: 1,
        }
    }
}

/// Operational/microstructure parameters.
/// 
/// Calibrated separately for BR and US markets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationalParams {
    /// Minimum daily liquidity in USD.
    /// BR: $5M, US: $20M
    pub min_liquidity_usd: f64,

    /// Maximum bid-ask spread in basis points.
    /// BR: 30 bps, US: 15 bps
    pub max_spread_bps: f64,

    /// Expected slippage cost in basis points.
    pub slippage_cost_bps: f64,

    /// Commission rate as percentage.
    pub commission_rate: f64,

    /// Maximum participation rate (% of daily volume).
    pub max_participation_rate: f64,
}

impl Default for OperationalParams {
    fn default() -> Self {
        Self {
            min_liquidity_usd: 5_000_000.0,
            max_spread_bps: 30.0,
            slippage_cost_bps: 10.0,
            commission_rate: 0.001,
            max_participation_rate: 0.05,
        }
    }
}

/// Universe filter thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniverseFilterParams {
    /// Minimum market cap in local currency.
    pub min_market_cap: f64,

    /// Maximum annualized volatility to accept.
    pub max_annualized_vol: f64,

    /// Minimum dividend yield (can be negative for carry).
    pub min_dividend_yield: f64,

    /// Minimum momentum return threshold.
    pub min_momentum_return: f64,

    /// Use quantile-based thresholds instead of absolute.
    pub use_quantile_thresholds: bool,

    /// Quantile for selection (e.g., 0.20 = top 20%).
    pub top_quantile: f64,
}

impl Default for UniverseFilterParams {
    fn default() -> Self {
        Self {
            min_market_cap: 1_000_000_000.0, // R$ 1B
            max_annualized_vol: 0.50,        // 50%
            min_dividend_yield: 0.0,
            min_momentum_return: 0.0,
            use_quantile_thresholds: true,
            top_quantile: 0.20,
        }
    }
}





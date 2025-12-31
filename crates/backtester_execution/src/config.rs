//! Execution Model Configuration
//!
//! Serializable configuration types for Stage B validation with realistic
//! execution costs, slippage, and fill policies.
//!
//! ## Design Goals
//! - Conservador: defaults penalizam levemente para evitar overfitting
//! - Deterministico: mesma config = mesmos resultados
//! - Auditável: todos os parâmetros são explícitos e rastreáveis

use serde::{Deserialize, Serialize};

// =============================================================================
// EXECUTION MODEL CONFIG (Top-Level)
// =============================================================================

/// Complete execution model configuration for Stage B validation.
///
/// This is the main configuration struct that controls how orders are executed
/// during walk-forward validation. It affects the equity curve (net of costs)
/// and therefore the fitness metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionModelConfig {
    /// Delay in bars before execution (0 = execute on same bar, 1 = next bar open).
    /// Default: 1 (realistic: signal at close, trade at next open+).
    #[serde(default = "default_delay_bars")]
    pub delay_bars: u8,

    /// Slippage model configuration.
    #[serde(default)]
    pub slippage: SlippageModelConfig,

    /// Fee/cost model configuration.
    #[serde(default)]
    pub fees: FeeModelConfig,

    /// Fill policy configuration.
    #[serde(default)]
    pub fill_policy: FillPolicyConfig,

    /// Bypass all costs for debugging (NOT for production).
    /// When true, slippage=0 and fees=0 regardless of other settings.
    #[serde(default)]
    pub bypass_for_debug: bool,
}

fn default_delay_bars() -> u8 {
    1
}

impl Default for ExecutionModelConfig {
    fn default() -> Self {
        Self {
            delay_bars: 1,
            slippage: SlippageModelConfig::default(),
            fees: FeeModelConfig::default(),
            fill_policy: FillPolicyConfig::default(),
            bypass_for_debug: false,
        }
    }
}

impl ExecutionModelConfig {
    /// Create MVP configuration: conservative, deterministic, easy to audit.
    #[must_use]
    pub fn mvp() -> Self {
        Self {
            delay_bars: 1,
            slippage: SlippageModelConfig::Constant { bps: 10.0 },
            fees: FeeModelConfig::from_tier(FeeTier::B3Retail),
            fill_policy: FillPolicyConfig::default(),
            bypass_for_debug: false,
        }
    }

    /// Create zero-cost configuration for testing/debugging only.
    #[must_use]
    pub fn zero_cost() -> Self {
        Self {
            delay_bars: 0,
            slippage: SlippageModelConfig::None,
            fees: FeeModelConfig::zero(),
            fill_policy: FillPolicyConfig {
                allow_partial: true,
                max_participation: 1.0,
                gap_policy: GapPolicy::AlwaysExecute,
                reject_policy: RejectPolicy::Cancel,
            },
            bypass_for_debug: true,
        }
    }

    /// Create B3 institutional configuration.
    #[must_use]
    pub fn b3_institutional() -> Self {
        Self {
            delay_bars: 1,
            slippage: SlippageModelConfig::VolatilityAdaptive {
                base_bps: 5.0,
                vol_factor: 0.3,
                regime_multiplier: 2.0,
                regime_vol_threshold: 0.25,
            },
            fees: FeeModelConfig::from_tier(FeeTier::B3Prime),
            fill_policy: FillPolicyConfig {
                allow_partial: true,
                max_participation: 0.03,
                gap_policy: GapPolicy::SkipIfGapExceeds { threshold_pct: 5.0 },
                reject_policy: RejectPolicy::RetryNextBar { max_retries: 2 },
            },
            bypass_for_debug: false,
        }
    }

    /// Check if this config will apply any costs.
    #[must_use]
    pub fn has_costs(&self) -> bool {
        if self.bypass_for_debug {
            return false;
        }
        !matches!(self.slippage, SlippageModelConfig::None) || self.fees.has_any_cost()
    }

    /// Scale all costs by a factor (for stress testing).
    #[must_use]
    pub fn scale_costs(&self, factor: f64) -> Self {
        Self {
            delay_bars: self.delay_bars,
            slippage: self.slippage.scale_bps(factor),
            fees: self.fees.scale(factor),
            fill_policy: self.fill_policy.clone(),
            bypass_for_debug: self.bypass_for_debug,
        }
    }

    /// Add delay bars (for stress testing).
    #[must_use]
    pub fn add_delay(&self, extra_bars: u8) -> Self {
        Self {
            delay_bars: self.delay_bars.saturating_add(extra_bars),
            slippage: self.slippage.clone(),
            fees: self.fees.clone(),
            fill_policy: self.fill_policy.clone(),
            bypass_for_debug: self.bypass_for_debug,
        }
    }
}

// =============================================================================
// SLIPPAGE MODEL CONFIG
// =============================================================================

/// Slippage model configuration (serializable).
///
/// Slippage represents the difference between expected execution price
/// and actual fill price due to market impact, timing, etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SlippageModelConfig {
    /// No slippage (for testing only).
    None,

    /// Fixed basis points slippage (conservative fallback).
    /// 10 bps = 0.1% per trade.
    Constant {
        /// Slippage in basis points.
        bps: f64,
    },

    /// Slippage dependent on order size vs bar volume (market impact).
    VolumeImpact {
        /// Base slippage in bps (applied to all orders).
        base_bps: f64,
        /// Coefficient for volume impact: slippage += factor * (order_size / bar_volume).
        volume_factor: f64,
        /// Maximum participation rate in bar volume (e.g., 0.05 = 5%).
        max_participation: f64,
    },

    /// Slippage adaptive to volatility regime.
    VolatilityAdaptive {
        /// Base slippage in bps.
        base_bps: f64,
        /// Coefficient for volatility: slippage += factor * (high-low)/close.
        vol_factor: f64,
        /// Multiplier applied when annualized vol > threshold.
        regime_multiplier: f64,
        /// Annualized volatility threshold to trigger regime_multiplier.
        regime_vol_threshold: f64,
    },

    /// Spread proxy model (estimates bid-ask from price patterns).
    SpreadProxy {
        /// Base slippage in bps (fallback when spread unavailable).
        base_bps: f64,
        /// Factor to multiply estimated spread.
        spread_factor: f64,
    },
}

impl Default for SlippageModelConfig {
    fn default() -> Self {
        Self::Constant { bps: 10.0 }
    }
}

impl SlippageModelConfig {
    /// Scale all bps values by a factor.
    #[must_use]
    pub fn scale_bps(&self, factor: f64) -> Self {
        match self {
            Self::None => Self::None,
            Self::Constant { bps } => Self::Constant { bps: bps * factor },
            Self::VolumeImpact {
                base_bps,
                volume_factor,
                max_participation,
            } => Self::VolumeImpact {
                base_bps: base_bps * factor,
                volume_factor: *volume_factor,
                max_participation: *max_participation,
            },
            Self::VolatilityAdaptive {
                base_bps,
                vol_factor,
                regime_multiplier,
                regime_vol_threshold,
            } => Self::VolatilityAdaptive {
                base_bps: base_bps * factor,
                vol_factor: *vol_factor,
                regime_multiplier: *regime_multiplier,
                regime_vol_threshold: *regime_vol_threshold,
            },
            Self::SpreadProxy {
                base_bps,
                spread_factor,
            } => Self::SpreadProxy {
                base_bps: base_bps * factor,
                spread_factor: *spread_factor,
            },
        }
    }

    /// Get effective base bps for reporting.
    #[must_use]
    pub fn base_bps(&self) -> f64 {
        match self {
            Self::None => 0.0,
            Self::Constant { bps } => *bps,
            Self::VolumeImpact { base_bps, .. } => *base_bps,
            Self::VolatilityAdaptive { base_bps, .. } => *base_bps,
            Self::SpreadProxy { base_bps, .. } => *base_bps,
        }
    }

    /// Validate configuration invariants.
    pub fn validate(&self) -> Result<(), ConfigError> {
        match self {
            Self::None => Ok(()),
            Self::Constant { bps } => {
                if *bps < 0.0 {
                    return Err(ConfigError::InvalidValue("bps must be >= 0".into()));
                }
                Ok(())
            }
            Self::VolumeImpact {
                base_bps,
                volume_factor,
                max_participation,
            } => {
                if *base_bps < 0.0 {
                    return Err(ConfigError::InvalidValue("base_bps must be >= 0".into()));
                }
                if *volume_factor < 0.0 {
                    return Err(ConfigError::InvalidValue("volume_factor must be >= 0".into()));
                }
                if *max_participation <= 0.0 || *max_participation > 1.0 {
                    return Err(ConfigError::InvalidValue(
                        "max_participation must be in (0, 1]".into(),
                    ));
                }
                Ok(())
            }
            Self::VolatilityAdaptive {
                base_bps,
                vol_factor,
                regime_multiplier,
                regime_vol_threshold,
            } => {
                if *base_bps < 0.0 {
                    return Err(ConfigError::InvalidValue("base_bps must be >= 0".into()));
                }
                if *vol_factor < 0.0 {
                    return Err(ConfigError::InvalidValue("vol_factor must be >= 0".into()));
                }
                if *regime_multiplier < 1.0 {
                    return Err(ConfigError::InvalidValue(
                        "regime_multiplier must be >= 1.0".into(),
                    ));
                }
                if *regime_vol_threshold <= 0.0 {
                    return Err(ConfigError::InvalidValue(
                        "regime_vol_threshold must be > 0".into(),
                    ));
                }
                Ok(())
            }
            Self::SpreadProxy {
                base_bps,
                spread_factor,
            } => {
                if *base_bps < 0.0 {
                    return Err(ConfigError::InvalidValue("base_bps must be >= 0".into()));
                }
                if *spread_factor < 0.0 {
                    return Err(ConfigError::InvalidValue("spread_factor must be >= 0".into()));
                }
                Ok(())
            }
        }
    }
}

// =============================================================================
// FEE MODEL CONFIG
// =============================================================================

/// Fee/cost model configuration.
///
/// Models brokerage fees, exchange fees, and other execution costs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeModelConfig {
    /// Fixed cost per trade (in local currency: BRL or USD).
    #[serde(default)]
    pub fixed_per_trade: f64,

    /// Commission rate as fraction of notional (e.g., 0.001 = 0.1%).
    #[serde(default)]
    pub commission_rate: f64,

    /// Cost per share/unit (e.g., $0.005 per share).
    #[serde(default)]
    pub per_unit_cost: f64,

    /// B3 emolument rate (e.g., 0.00035 = 0.035%).
    #[serde(default)]
    pub emolument_rate: f64,

    /// Fee tier preset (used to identify the configuration).
    #[serde(default)]
    pub tier: FeeTier,
}

impl Default for FeeModelConfig {
    fn default() -> Self {
        Self::from_tier(FeeTier::B3Retail)
    }
}

impl FeeModelConfig {
    /// Create config from a fee tier preset.
    #[must_use]
    pub fn from_tier(tier: FeeTier) -> Self {
        match tier {
            FeeTier::B3Retail => Self {
                fixed_per_trade: 10.0,
                commission_rate: 0.0015,  // 0.15%
                per_unit_cost: 0.0,
                emolument_rate: 0.000_35, // 0.035%
                tier,
            },
            FeeTier::B3Prime => Self {
                fixed_per_trade: 5.0,
                commission_rate: 0.001, // 0.10%
                per_unit_cost: 0.0,
                emolument_rate: 0.000_35,
                tier,
            },
            FeeTier::USRetail => Self {
                fixed_per_trade: 1.0,
                commission_rate: 0.001,   // 0.10%
                per_unit_cost: 0.005,     // $0.005/share
                emolument_rate: 0.000_02, // SEC fee ~0.002%
                tier,
            },
            FeeTier::USPrime => Self {
                fixed_per_trade: 0.0,
                commission_rate: 0.0003,  // 0.03%
                per_unit_cost: 0.003,     // $0.003/share
                emolument_rate: 0.000_02,
                tier,
            },
            FeeTier::Custom => Self {
                fixed_per_trade: 0.0,
                commission_rate: 0.0,
                per_unit_cost: 0.0,
                emolument_rate: 0.0,
                tier,
            },
        }
    }

    /// Create zero-cost config.
    #[must_use]
    pub fn zero() -> Self {
        Self {
            fixed_per_trade: 0.0,
            commission_rate: 0.0,
            per_unit_cost: 0.0,
            emolument_rate: 0.0,
            tier: FeeTier::Custom,
        }
    }

    /// Check if this config has any costs.
    #[must_use]
    pub fn has_any_cost(&self) -> bool {
        self.fixed_per_trade > 0.0
            || self.commission_rate > 0.0
            || self.per_unit_cost > 0.0
            || self.emolument_rate > 0.0
    }

    /// Scale all fees by a factor.
    #[must_use]
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            fixed_per_trade: self.fixed_per_trade * factor,
            commission_rate: self.commission_rate * factor,
            per_unit_cost: self.per_unit_cost * factor,
            emolument_rate: self.emolument_rate * factor,
            tier: FeeTier::Custom, // Becomes custom after scaling
        }
    }

    /// Calculate total cost for a trade.
    #[must_use]
    pub fn calculate(&self, notional: f64, quantity: i64) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let qty = quantity.unsigned_abs() as f64;

        self.fixed_per_trade
            + notional * self.commission_rate
            + notional * self.emolument_rate
            + qty * self.per_unit_cost
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.fixed_per_trade < 0.0 {
            return Err(ConfigError::InvalidValue(
                "fixed_per_trade must be >= 0".into(),
            ));
        }
        if self.commission_rate < 0.0 {
            return Err(ConfigError::InvalidValue(
                "commission_rate must be >= 0".into(),
            ));
        }
        if self.per_unit_cost < 0.0 {
            return Err(ConfigError::InvalidValue("per_unit_cost must be >= 0".into()));
        }
        if self.emolument_rate < 0.0 {
            return Err(ConfigError::InvalidValue("emolument_rate must be >= 0".into()));
        }
        Ok(())
    }
}

/// Fee tier presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum FeeTier {
    /// Brazilian retail brokerage (e.g., XP, BTG).
    #[default]
    B3Retail,
    /// Brazilian prime brokerage.
    B3Prime,
    /// US retail brokerage (e.g., IBKR retail).
    USRetail,
    /// US prime brokerage.
    USPrime,
    /// Custom fee structure.
    Custom,
}

// =============================================================================
// FILL POLICY CONFIG
// =============================================================================

/// Fill policy configuration.
///
/// Controls how orders are filled when there are liquidity constraints,
/// gaps, or other execution challenges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillPolicyConfig {
    /// Allow partial fills when liquidity is insufficient.
    #[serde(default = "default_allow_partial")]
    pub allow_partial: bool,

    /// Maximum participation rate in bar volume (e.g., 0.05 = 5%).
    #[serde(default = "default_max_participation")]
    pub max_participation: f64,

    /// Policy for handling price gaps.
    #[serde(default)]
    pub gap_policy: GapPolicy,

    /// Policy for rejected orders.
    #[serde(default)]
    pub reject_policy: RejectPolicy,
}

fn default_allow_partial() -> bool {
    false
}

fn default_max_participation() -> f64 {
    0.05
}

impl Default for FillPolicyConfig {
    fn default() -> Self {
        Self {
            allow_partial: false,
            max_participation: 0.05,
            gap_policy: GapPolicy::default(),
            reject_policy: RejectPolicy::default(),
        }
    }
}

impl FillPolicyConfig {
    /// Validate configuration.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.max_participation <= 0.0 || self.max_participation > 1.0 {
            return Err(ConfigError::InvalidValue(
                "max_participation must be in (0, 1]".into(),
            ));
        }
        Ok(())
    }
}

/// Gap handling policy.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum GapPolicy {
    /// Execute at the opening price of the next bar.
    ExecuteAtOpen,
    /// Skip the order if gap exceeds threshold.
    SkipIfGapExceeds {
        /// Gap threshold as percentage (e.g., 5.0 = 5%).
        threshold_pct: f64,
    },
    /// Always execute, adjusting price as needed.
    AlwaysExecute,
}

impl Default for GapPolicy {
    fn default() -> Self {
        Self::ExecuteAtOpen
    }
}

/// Reject order handling policy.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RejectPolicy {
    /// Retry the order on the next bar.
    RetryNextBar {
        /// Maximum number of retry attempts.
        max_retries: u8,
    },
    /// Cancel the order immediately.
    Cancel,
    /// Convert to an order for the next trading day.
    ConvertToNextDay,
}

impl Default for RejectPolicy {
    fn default() -> Self {
        Self::Cancel
    }
}

// =============================================================================
// INSTITUTIONAL GATES CONFIG
// =============================================================================

/// Institutional gates configuration.
///
/// Hard constraints that candidates must pass before entering the Pareto frontier.
/// Failing a gate results in rejection, not just a penalty.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstitutionalGatesConfig {
    /// Maximum annual turnover (e.g., 12.0 = 12x portfolio per year).
    #[serde(default = "default_max_turnover")]
    pub max_turnover_annual: f64,

    /// Maximum slippage as percentage of gross PnL.
    #[serde(default = "default_max_slippage_pct")]
    pub max_slippage_pct_of_pnl: f64,

    /// Minimum capacity in USD (below this = warning, not reject).
    #[serde(default = "default_min_capacity")]
    pub min_capacity_usd: f64,

    /// Maximum average slippage in bps (warning threshold).
    #[serde(default = "default_max_avg_slippage")]
    pub max_avg_slippage_bps: f64,
}

fn default_max_turnover() -> f64 {
    12.0
}

fn default_max_slippage_pct() -> f64 {
    30.0
}

fn default_min_capacity() -> f64 {
    5_000_000.0
}

fn default_max_avg_slippage() -> f64 {
    25.0
}

impl Default for InstitutionalGatesConfig {
    fn default() -> Self {
        Self {
            max_turnover_annual: 12.0,
            max_slippage_pct_of_pnl: 30.0,
            min_capacity_usd: 5_000_000.0,
            max_avg_slippage_bps: 25.0,
        }
    }
}

// =============================================================================
// ERROR TYPES
// =============================================================================

/// Configuration validation error.
#[derive(Debug, Clone)]
pub enum ConfigError {
    /// Invalid parameter value.
    InvalidValue(String),
    /// Missing required field.
    MissingField(String),
    /// Inconsistent configuration.
    Inconsistent(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidValue(msg) => write!(f, "Invalid value: {msg}"),
            Self::MissingField(field) => write!(f, "Missing field: {field}"),
            Self::Inconsistent(msg) => write!(f, "Inconsistent config: {msg}"),
        }
    }
}

impl std::error::Error for ConfigError {}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ExecutionModelConfig::default();
        assert_eq!(config.delay_bars, 1);
        assert!(!config.bypass_for_debug);
        assert!(config.has_costs());
    }

    #[test]
    fn test_mvp_config() {
        let config = ExecutionModelConfig::mvp();
        assert_eq!(config.delay_bars, 1);
        assert!(matches!(
            config.slippage,
            SlippageModelConfig::Constant { bps } if (bps - 10.0).abs() < f64::EPSILON
        ));
        assert!(config.has_costs());
    }

    #[test]
    fn test_zero_cost_config() {
        let config = ExecutionModelConfig::zero_cost();
        assert!(!config.has_costs());
        assert!(config.bypass_for_debug);
    }

    #[test]
    fn test_scale_costs() {
        let config = ExecutionModelConfig::mvp();
        let scaled = config.scale_costs(2.0);

        match scaled.slippage {
            SlippageModelConfig::Constant { bps } => {
                assert!((bps - 20.0).abs() < f64::EPSILON);
            }
            _ => panic!("Expected Constant slippage"),
        }
    }

    #[test]
    fn test_add_delay() {
        let config = ExecutionModelConfig::mvp();
        let delayed = config.add_delay(1);
        assert_eq!(delayed.delay_bars, 2);
    }

    #[test]
    fn test_fee_tiers() {
        let b3_retail = FeeModelConfig::from_tier(FeeTier::B3Retail);
        assert!((b3_retail.fixed_per_trade - 10.0).abs() < f64::EPSILON);

        let us_prime = FeeModelConfig::from_tier(FeeTier::USPrime);
        assert!((us_prime.fixed_per_trade - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fee_calculation() {
        let fee = FeeModelConfig::from_tier(FeeTier::B3Retail);
        // 10 + 10000 * 0.0015 + 10000 * 0.00035 = 10 + 15 + 3.5 = 28.5
        let cost = fee.calculate(10_000.0, 100);
        assert!((cost - 28.5).abs() < 0.01);
    }

    #[test]
    fn test_slippage_validation() {
        let valid = SlippageModelConfig::Constant { bps: 10.0 };
        assert!(valid.validate().is_ok());

        let invalid = SlippageModelConfig::Constant { bps: -5.0 };
        assert!(invalid.validate().is_err());

        let valid_vol = SlippageModelConfig::VolatilityAdaptive {
            base_bps: 5.0,
            vol_factor: 0.3,
            regime_multiplier: 2.0,
            regime_vol_threshold: 0.25,
        };
        assert!(valid_vol.validate().is_ok());

        let invalid_vol = SlippageModelConfig::VolatilityAdaptive {
            base_bps: 5.0,
            vol_factor: 0.3,
            regime_multiplier: 0.5, // < 1.0
            regime_vol_threshold: 0.25,
        };
        assert!(invalid_vol.validate().is_err());
    }

    #[test]
    fn test_serialization() {
        let config = ExecutionModelConfig::mvp();
        let toml_str = toml::to_string(&config).expect("Failed to serialize");
        assert!(toml_str.contains("delay_bars"));

        let parsed: ExecutionModelConfig =
            toml::from_str(&toml_str).expect("Failed to deserialize");
        assert_eq!(parsed.delay_bars, config.delay_bars);
    }

    #[test]
    fn test_institutional_gates_default() {
        let gates = InstitutionalGatesConfig::default();
        assert!((gates.max_turnover_annual - 12.0).abs() < f64::EPSILON);
        assert!((gates.min_capacity_usd - 5_000_000.0).abs() < f64::EPSILON);
    }

    // =========================================================================
    // Phase 2.2: Comprehensive B3 Fee Validation
    // =========================================================================

    #[test]
    fn test_b3_retail_fee_structure() {
        let fee = FeeModelConfig::from_tier(FeeTier::B3Retail);
        
        // Verify B3 retail rates
        assert_eq!(fee.fixed_per_trade, 10.0, "B3 retail fixed fee should be R$10");
        assert!((fee.commission_rate - 0.0015).abs() < 1e-6, "B3 retail commission should be 0.15%");
        assert!((fee.emolument_rate - 0.00035).abs() < 1e-6, "B3 emolument should be 0.035%");
    }

    #[test]
    fn test_b3_prime_fee_structure() {
        let fee = FeeModelConfig::from_tier(FeeTier::B3Prime);
        
        // B3 Prime has lower costs
        assert!(fee.fixed_per_trade < 10.0, "B3 Prime should have lower fixed fee");
        assert!(fee.commission_rate < 0.0015, "B3 Prime should have lower commission");
        // Emolument is the same (exchange fee)
        assert!((fee.emolument_rate - 0.00035).abs() < 1e-6, "Emolument should be same");
    }

    #[test]
    fn test_b3_fee_calculation_known_values() {
        let fee = FeeModelConfig::from_tier(FeeTier::B3Retail);
        
        // Trade: 1000 shares of R$50 = R$50,000 notional
        // Fixed: R$10
        // Commission: R$50,000 * 0.15% = R$75
        // Emolument: R$50,000 * 0.035% = R$17.50
        // Total: R$102.50
        let cost = fee.calculate(50_000.0, 1000);
        assert!((cost - 102.50).abs() < 0.01, "B3 retail cost should be R$102.50, got {}", cost);
    }

    #[test]
    fn test_b3_fee_round_lot_enforcement() {
        // B3 requires round lots of 100 shares
        // Fees should be calculated the same way regardless
        let fee = FeeModelConfig::from_tier(FeeTier::B3Retail);
        
        let cost_100 = fee.calculate(5_000.0, 100);
        let cost_200 = fee.calculate(10_000.0, 200);
        
        // Cost for 200 shares should be roughly 2x (minus fixed portion)
        let variable_100 = cost_100 - fee.fixed_per_trade;
        let variable_200 = cost_200 - fee.fixed_per_trade;
        
        assert!((variable_200 / variable_100 - 2.0).abs() < 0.01,
            "Variable costs should scale linearly: {} / {} = {}", 
            variable_200, variable_100, variable_200 / variable_100);
    }

    #[test]
    fn test_us_retail_fee_structure() {
        let fee = FeeModelConfig::from_tier(FeeTier::USRetail);
        
        // Verify US retail has per-share cost
        assert!(fee.per_unit_cost > 0.0, "US retail should have per-share cost");
        // SEC fee is very small
        assert!(fee.emolument_rate < 0.0001, "SEC fee should be < 0.01%");
    }

    #[test]
    fn test_us_retail_fee_calculation() {
        let fee = FeeModelConfig::from_tier(FeeTier::USRetail);
        
        // Trade: 100 shares at $150 = $15,000 notional
        // Fixed: $1
        // Commission: $15,000 * 0.1% = $15
        // Per-unit: 100 * $0.005 = $0.50
        // SEC fee: $15,000 * 0.002% = $0.30
        // Total: ~$16.80
        let cost = fee.calculate(15_000.0, 100);
        assert!(cost > 15.0 && cost < 20.0, "US retail cost should be ~$16-17, got {}", cost);
    }

    #[test]
    fn test_fee_zero_notional() {
        let fee = FeeModelConfig::from_tier(FeeTier::B3Retail);
        
        let cost = fee.calculate(0.0, 0);
        // Should still have fixed cost
        assert_eq!(cost, fee.fixed_per_trade, "Zero trade should only have fixed cost");
    }

    #[test]
    fn test_fee_tier_ordering() {
        // Prime should always be cheaper than retail
        let b3_retail = FeeModelConfig::from_tier(FeeTier::B3Retail);
        let b3_prime = FeeModelConfig::from_tier(FeeTier::B3Prime);
        
        let cost_retail = b3_retail.calculate(100_000.0, 1000);
        let cost_prime = b3_prime.calculate(100_000.0, 1000);
        
        assert!(cost_prime < cost_retail, 
            "Prime {} should be cheaper than retail {}", cost_prime, cost_retail);
    }

    #[test]
    fn test_custom_fee_tier() {
        let fee = FeeModelConfig::from_tier(FeeTier::Custom);
        
        // Custom tier should have zero costs
        let cost = fee.calculate(100_000.0, 1000);
        assert_eq!(cost, 0.0, "Custom tier should have zero cost");
    }

    #[test]
    fn test_fee_config_validation() {
        let valid = FeeModelConfig::from_tier(FeeTier::B3Retail);
        assert!(valid.validate().is_ok());
        
        // Negative commission rate should fail
        let invalid = FeeModelConfig {
            commission_rate: -0.001,
            ..valid.clone()
        };
        assert!(invalid.validate().is_err(), "Negative commission should fail");
    }

    #[test]
    fn test_slippage_config_presets() {
        // Test all slippage model variants
        let configs = vec![
            SlippageModelConfig::None,
            SlippageModelConfig::Constant { bps: 10.0 },
            SlippageModelConfig::VolumeImpact {
                base_bps: 5.0,
                volume_factor: 0.5,
                max_participation: 0.1,
            },
            SlippageModelConfig::VolatilityAdaptive {
                base_bps: 5.0,
                vol_factor: 0.3,
                regime_multiplier: 2.0,
                regime_vol_threshold: 0.25,
            },
            SlippageModelConfig::SpreadProxy {
                base_bps: 5.0,
                spread_factor: 0.5,
            },
        ];
        
        for config in configs {
            assert!(config.validate().is_ok(), "Valid config should pass: {:?}", config);
        }
    }

    #[test]
    fn test_execution_model_config_bypass() {
        let mut config = ExecutionModelConfig::mvp();
        config.bypass_for_debug = true;
        
        // With bypass, effective costs should be zero
        // (This is tested at the execution level, but config should preserve the flag)
        assert!(config.bypass_for_debug);
    }
}


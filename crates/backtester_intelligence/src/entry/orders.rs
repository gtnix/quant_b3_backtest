//! Order generation module - converts target weights to buy/sell orders.
//!
//! # Features
//!
//! - BR lot sizes (multiples of 100)
//! - US lot sizes (1 share)
//! - Transaction costs per market (brokerage, exchange fees)
//! - Slippage estimation
//! - Over-allocation prevention (max_allocation_pct)
//!
//! # Performance (Milestone 6)
//!
//! All calculations use fixed-point `Price`, `Money`, and `Rate` types.
//! This provides deterministic results with ~10x faster arithmetic vs Decimal.
//!
//! # Market Impact Model Limitation
//!
//! **Current implementation uses a fixed slippage model (default: 5 bps).**

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use backtester_core::{Money, Price};
use crate::filters::Market;
use super::types::{Order, OrderSide};

/// Configuration for order generation.
///
/// # Performance (Milestone 6)
///
/// Serde uses f64 for config file compatibility. Internal calculations use
/// fixed-point types via accessor methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderGeneratorConfig {
    /// BR brokerage fee (e.g., 0.0003 for 0.03%)
    #[serde(default = "default_br_brokerage")]
    br_brokerage_f64: f64,
    
    /// BR exchange fees (emolumentos + liquidação, e.g., 0.000325 for 0.0325%)
    #[serde(default = "default_br_exchange_fee")]
    br_exchange_fee_f64: f64,
    
    /// US per-share fee (e.g., $0.005/share)
    #[serde(default = "default_us_per_share")]
    us_per_share_fee_f64: f64,
    
    /// US minimum fee percentage (e.g., 0.001 for 0.1%)
    #[serde(default = "default_us_min_pct")]
    us_min_fee_pct_f64: f64,
    
    /// Slippage estimate (e.g., 0.0005 for 0.05%)
    #[serde(default = "default_slippage")]
    slippage_f64: f64,
    
    /// BR lot size (standard is 100)
    #[serde(default = "default_br_lot")]
    pub br_lot_size: i64,
    
    /// US lot size (standard is 1)
    #[serde(default = "default_us_lot")]
    pub us_lot_size: i64,
    
    /// Maximum allocation percentage of capital (e.g., 0.99 for 99%)
    #[serde(default = "default_max_allocation")]
    max_allocation_pct_f64: f64,
}

fn default_br_brokerage() -> f64 { 0.0003 }      // 0.03%
fn default_br_exchange_fee() -> f64 { 0.000325 } // 0.0325%
fn default_us_per_share() -> f64 { 0.005 }       // $0.005/share
fn default_us_min_pct() -> f64 { 0.001 }         // 0.1%
fn default_slippage() -> f64 { 0.0005 }          // 0.05%
fn default_br_lot() -> i64 { 100 }
fn default_us_lot() -> i64 { 1 }
fn default_max_allocation() -> f64 { 0.99 }      // 99% max (1% cash buffer)

impl OrderGeneratorConfig {
    /// Combined BR fees (brokerage + exchange) as rate.
    #[inline]
    pub fn br_total_fee_rate(&self) -> f64 {
        self.br_brokerage_f64 + self.br_exchange_fee_f64
    }
    
    /// US per-share fee.
    #[inline]
    pub fn us_per_share_fee(&self) -> Money {
        Money::from_f64(self.us_per_share_fee_f64)
    }
    
    /// US minimum fee percentage.
    #[inline]
    pub fn us_min_fee_pct(&self) -> f64 {
        self.us_min_fee_pct_f64
    }
    
    /// Slippage rate.
    #[inline]
    pub fn slippage(&self) -> f64 {
        self.slippage_f64
    }
    
    /// Max allocation percentage.
    #[inline]
    pub fn max_allocation_pct(&self) -> f64 {
        self.max_allocation_pct_f64
    }
}

impl Default for OrderGeneratorConfig {
    fn default() -> Self {
        Self {
            br_brokerage_f64: default_br_brokerage(),
            br_exchange_fee_f64: default_br_exchange_fee(),
            us_per_share_fee_f64: default_us_per_share(),
            us_min_fee_pct_f64: default_us_min_pct(),
            slippage_f64: default_slippage(),
            br_lot_size: default_br_lot(),
            us_lot_size: default_us_lot(),
            max_allocation_pct_f64: default_max_allocation(),
        }
    }
}

/// Target position for order generation.
///
/// # Performance (Milestone 6)
///
/// Uses fixed-point `Price` for price field.
#[derive(Debug, Clone)]
pub struct OrderTarget {
    pub symbol: String,
    pub market: Market,
    pub target_weight: f64,
    pub price: Price,
}

/// Order generator.
///
/// # Performance (Milestone 6)
///
/// All calculations use fixed-point types for deterministic results.
#[derive(Debug, Clone)]
pub struct OrderGenerator {
    config: OrderGeneratorConfig,
}

impl OrderGenerator {
    pub fn new(config: OrderGeneratorConfig) -> Self {
        Self { config }
    }

    /// Generate orders to reach target positions.
    /// Applies max_allocation_pct constraint to prevent over-allocation.
    ///
    /// # Performance (Milestone 6)
    ///
    /// All monetary calculations use fixed-point arithmetic.
    pub fn generate_orders(
        &self,
        targets: &[OrderTarget],
        current_positions: &HashMap<String, i64>,
        capital: Money,
    ) -> (Vec<Order>, Money) {
        let max_capital = capital.mul_f64(self.config.max_allocation_pct());
        
        // Phase 1: Calculate initial target shares for all assets
        struct TargetCalc {
            symbol: String,
            market: Market,
            price: Price,
            target_shares: i64,
            current_shares: i64,
            lot_size: i64,
        }
        
        let mut calcs: Vec<TargetCalc> = targets.iter().map(|target| {
            let current_shares = current_positions.get(&target.symbol).copied().unwrap_or(0);
            let target_notional = capital.mul_f64(target.target_weight);
            let raw_shares = if target.price.is_positive() {
                target_notional.div_price(target.price)
            } else {
                0
            };
            let lot_size = match target.market {
                Market::BR => self.config.br_lot_size,
                Market::US => self.config.us_lot_size,
            };
            let target_shares = self.round_to_lot(raw_shares, lot_size);
            
            TargetCalc {
                symbol: target.symbol.clone(),
                market: target.market,
                price: target.price,
                target_shares,
                current_shares,
                lot_size,
            }
        }).collect();
        
        // Phase 2: Check for over-allocation and scale down if needed
        let total_buy_notional: Money = calcs.iter()
            .filter(|c| c.target_shares > c.current_shares)
            .map(|c| c.price.mul_shares(c.target_shares - c.current_shares))
            .sum();
        
        if total_buy_notional > max_capital {
            // Scale down all target shares proportionally
            let scale_factor = max_capital.div_money(total_buy_notional);
            for calc in &mut calcs {
                if calc.target_shares > calc.current_shares {
                    #[allow(clippy::cast_precision_loss)]
                    let scaled_shares = (calc.target_shares as f64 * scale_factor) as i64;
                    calc.target_shares = self.round_to_lot(scaled_shares, calc.lot_size);
                }
            }
        }
        
        // Phase 3: Generate orders
        let mut orders = Vec::new();
        let mut total_cost = Money::ZERO;

        for calc in calcs {
            let delta = calc.target_shares - calc.current_shares;

            if delta == 0 {
                continue;
            }

            let notional = calc.price.mul_shares(delta.abs());
            let cost = self.calculate_cost(notional, delta.abs(), calc.market);
            total_cost += cost;

            let side = if delta > 0 { OrderSide::Buy } else { OrderSide::Sell };

            orders.push(Order::new(
                calc.symbol,
                side,
                delta.abs(),
                calc.price,
                cost,
            ));
        }

        // Generate sell orders for positions not in targets
        let target_symbols: std::collections::HashSet<_> = targets.iter().map(|t| &t.symbol).collect();
        
        for (symbol, &shares) in current_positions {
            if shares > 0 && !target_symbols.contains(symbol) {
                // Need to sell entire position - but we don't have market/price info
                // This would need additional context in real implementation
                // For now, we skip - the caller should include these in targets with 0 weight
            }
        }

        (orders, total_cost)
    }

    /// Round shares to lot size (floor).
    fn round_to_lot(&self, shares: i64, lot_size: i64) -> i64 {
        if lot_size <= 1 {
            shares
        } else {
            (shares / lot_size) * lot_size
        }
    }

    /// Calculate transaction cost using fixed-point arithmetic.
    fn calculate_cost(&self, notional: Money, shares: i64, market: Market) -> Money {
        let base_cost = match market {
            Market::BR => {
                // BR: brokerage + exchange fees
                notional.mul_f64(self.config.br_total_fee_rate())
            }
            Market::US => {
                // US: per-share fee or percentage, whichever is greater
                let per_share_cost = self.config.us_per_share_fee().mul_f64(shares as f64);
                let pct_cost = notional.mul_f64(self.config.us_min_fee_pct());
                if per_share_cost > pct_cost { per_share_cost } else { pct_cost }
            }
        };

        // Add slippage
        let slippage_cost = notional.mul_f64(self.config.slippage());
        
        base_cost + slippage_cost
    }

    /// Calculate target shares for a given weight and capital.
    pub fn calculate_target_shares(
        &self,
        weight: f64,
        price: Price,
        capital: Money,
        market: Market,
    ) -> i64 {
        if !price.is_positive() {
            return 0;
        }

        let target_notional = capital.mul_f64(weight);
        let raw_shares = target_notional.div_price(price);

        let lot_size = match market {
            Market::BR => self.config.br_lot_size,
            Market::US => self.config.us_lot_size,
        };

        self.round_to_lot(raw_shares, lot_size)
    }

    /// Estimate total turnover from orders.
    pub fn calculate_turnover(&self, orders: &[Order], total_capital: Money) -> f64 {
        if !total_capital.is_positive() {
            return 0.0;
        }

        let total_traded: Money = orders.iter().map(|o| o.notional).sum();
        total_traded.div_money(total_capital)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_br_lot_rounding() {
        let generator = OrderGenerator::new(OrderGeneratorConfig::default());
        
        // 250 shares → 200 (2 lots)
        assert_eq!(generator.round_to_lot(250, 100), 200);
        
        // 99 shares → 0
        assert_eq!(generator.round_to_lot(99, 100), 0);
        
        // 300 shares → 300
        assert_eq!(generator.round_to_lot(300, 100), 300);
    }

    #[test]
    fn test_us_lot_no_rounding() {
        let generator = OrderGenerator::new(OrderGeneratorConfig::default());
        
        // US has lot size 1, no rounding needed
        assert_eq!(generator.round_to_lot(66, 1), 66);
        assert_eq!(generator.round_to_lot(123, 1), 123);
    }

    #[test]
    fn test_calculate_target_shares_br() {
        let generator = OrderGenerator::new(OrderGeneratorConfig::default());
        
        // R$ 100k capital, 10% weight, R$ 50 price
        // Target notional = R$ 10k
        // Raw shares = 200
        // BR lot = 100 → 200 shares (2 lots)
        let shares = generator.calculate_target_shares(
            0.10,
            Price::from_int(50),
            Money::from_int(100_000),
            Market::BR,
        );
        
        assert_eq!(shares, 200);
    }

    #[test]
    fn test_calculate_target_shares_us() {
        let generator = OrderGenerator::new(OrderGeneratorConfig::default());
        
        // $100k capital, 10% weight, $150 price
        // Target notional = $10k
        // Raw shares = 66.67 → 66
        // US lot = 1 → 66 shares
        let shares = generator.calculate_target_shares(
            0.10,
            Price::from_int(150),
            Money::from_int(100_000),
            Market::US,
        );
        
        assert_eq!(shares, 66);
    }

    #[test]
    fn test_generate_buy_order() {
        let generator = OrderGenerator::new(OrderGeneratorConfig::default());
        
        let targets = vec![OrderTarget {
            symbol: "PETR4".to_string(),
            market: Market::BR,
            target_weight: 0.10,
            price: Price::from_int(38),
        }];

        let (orders, total_cost) = generator.generate_orders(
            &targets,
            &HashMap::new(),
            Money::from_int(100_000),
        );

        assert_eq!(orders.len(), 1);
        assert_eq!(orders[0].side, OrderSide::Buy);
        assert_eq!(orders[0].symbol, "PETR4");
        // R$ 10k / R$ 38 = 263 → 200 (2 lots)
        assert_eq!(orders[0].shares, 200);
        assert!(total_cost.is_positive());
    }

    #[test]
    fn test_generate_sell_order_reduced_position() {
        let generator = OrderGenerator::new(OrderGeneratorConfig::default());
        
        let targets = vec![OrderTarget {
            symbol: "PETR4".to_string(),
            market: Market::BR,
            target_weight: 0.05, // Reduced from current
            price: Price::from_int(38),
        }];

        let mut current = HashMap::new();
        current.insert("PETR4".to_string(), 400); // Currently holding 400

        let (orders, _) = generator.generate_orders(
            &targets,
            &current,
            Money::from_int(100_000),
        );

        // Target: R$ 5k / R$ 38 = 131 → 100 shares
        // Current: 400
        // Delta: 100 - 400 = -300 → SELL 300
        assert_eq!(orders.len(), 1);
        assert_eq!(orders[0].side, OrderSide::Sell);
        assert_eq!(orders[0].shares, 300);
    }

    #[test]
    fn test_br_cost_calculation() {
        let generator = OrderGenerator::new(OrderGeneratorConfig::default());
        
        let notional = Money::from_int(10_000); // R$ 10k
        let cost = generator.calculate_cost(notional, 200, Market::BR);
        
        // Brokerage: 0.03% = R$ 3
        // Exchange: 0.0325% = R$ 3.25
        // Slippage: 0.05% = R$ 5
        // Total ≈ R$ 11.25
        let expected = 10_000.0 * (0.0003 + 0.000325 + 0.0005);
        assert!((cost.to_f64() - expected).abs() < 0.1);
    }

    #[test]
    fn test_us_cost_calculation() {
        let generator = OrderGenerator::new(OrderGeneratorConfig::default());
        
        let notional = Money::from_int(10_000); // $10k
        let shares = 66;
        let cost = generator.calculate_cost(notional, shares, Market::US);
        
        // Per-share: 66 * $0.005 = $0.33
        // Percentage: 0.1% of $10k = $10
        // Max($0.33, $10) = $10
        // Slippage: 0.05% of $10k = $5
        // Total ≈ $15
        assert!(cost.to_f64() > 10.0); // Should be at least $10 (min fee)
    }

    #[test]
    fn test_no_order_when_no_change() {
        let generator = OrderGenerator::new(OrderGeneratorConfig::default());
        
        let targets = vec![OrderTarget {
            symbol: "PETR4".to_string(),
            market: Market::BR,
            target_weight: 0.076, // Roughly matches 200 shares at R$38 with R$100k
            price: Price::from_int(38),
        }];

        let mut current = HashMap::new();
        current.insert("PETR4".to_string(), 200);

        let (orders, _) = generator.generate_orders(
            &targets,
            &current,
            Money::from_int(100_000),
        );

        // Target shares ≈ 200, current = 200 → no order
        assert!(orders.is_empty());
    }

    #[test]
    fn test_turnover_calculation() {
        let generator = OrderGenerator::new(OrderGeneratorConfig::default());
        
        let orders = vec![
            Order::new("A".to_string(), OrderSide::Buy, 100, Price::from_int(50), Money::from_int(1)),
            Order::new("B".to_string(), OrderSide::Sell, 200, Price::from_int(30), Money::from_int(1)),
        ];
        
        // Total traded = 100*50 + 200*30 = 5000 + 6000 = 11000
        // Capital = 100000
        // Turnover = 11%
        let turnover = generator.calculate_turnover(&orders, Money::from_int(100_000));
        assert!((turnover - 0.11).abs() < 0.001);
    }

    #[test]
    fn test_over_allocation_prevention() {
        // Create a generator with low max_allocation (50%)
        let mut config = OrderGeneratorConfig::default();
        config.max_allocation_pct_f64 = 0.50; // Only allow 50% allocation
        let generator = OrderGenerator::new(config);
        
        // Two assets each wanting 50% weight = 100% total
        // With max_allocation=50%, should scale down
        let targets = vec![
            OrderTarget {
                symbol: "A".to_string(),
                market: Market::US,
                target_weight: 0.50,
                price: Price::from_int(100),
            },
            OrderTarget {
                symbol: "B".to_string(),
                market: Market::US,
                target_weight: 0.50,
                price: Price::from_int(100),
            },
        ];

        let capital = Money::from_int(100_000);
        let (orders, _) = generator.generate_orders(&targets, &HashMap::new(), capital);

        // Calculate total buy notional
        let total_buy_notional: Money = orders.iter()
            .filter(|o| o.side == OrderSide::Buy)
            .map(|o| o.notional)
            .sum();

        // Should be <= 50% of capital (50000 + small slack for lot rounding)
        assert!(
            total_buy_notional.to_f64() <= 50_000.0 + 100.0,
            "Total buy notional {} should be <= 50% of capital",
            total_buy_notional.to_f64()
        );
    }

    #[test]
    fn test_full_allocation_respects_cash_buffer() {
        // Default config has 99% max allocation
        let generator = OrderGenerator::new(OrderGeneratorConfig::default());
        
        // One asset wanting 100% weight
        let targets = vec![OrderTarget {
            symbol: "FULL".to_string(),
            market: Market::US,
            target_weight: 1.0,
            price: Price::from_int(100),
        }];

        let capital = Money::from_int(100_000);
        let (orders, _) = generator.generate_orders(&targets, &HashMap::new(), capital);

        assert_eq!(orders.len(), 1);
        
        // Should buy floor(100000 * 0.99 / 100) = 990 shares
        // Total notional = 990 * 100 = 99000 (99% of capital)
        assert!(
            orders[0].notional.to_f64() <= 99_000.0 + 100.0,
            "Buy notional {} should be <= 99% of capital", orders[0].notional.to_f64()
        );
    }
}


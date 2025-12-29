//! # Biweekly Rebalancer
//!
//! Portfolio rebalancer that generates orders to maintain target weights.
//! Designed for biweekly (quinzenal) rebalancing schedules.

use backtester_core::{AssetId, OrderDirection, OrderEvent, OrderId, OrderType, TimeInForce};
use backtester_portfolio::Portfolio;

/// Rebalancing schedule type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebalanceSchedule {
    /// Every N days.
    Daily(u32),
    /// Every N weeks.
    Weekly(u32),
    /// Every 2 weeks (biweekly/quinzenal).
    Biweekly,
    /// Monthly.
    Monthly,
    /// Custom interval in days.
    Custom(u32),
}

impl RebalanceSchedule {
    /// Get interval in nanoseconds.
    #[must_use]
    pub fn interval_nanos(&self) -> i64 {
        const NANOS_PER_DAY: i64 = 86_400_000_000_000;
        match self {
            Self::Daily(n) => *n as i64 * NANOS_PER_DAY,
            Self::Weekly(n) => *n as i64 * 7 * NANOS_PER_DAY,
            Self::Biweekly => 14 * NANOS_PER_DAY,
            Self::Monthly => 30 * NANOS_PER_DAY,
            Self::Custom(days) => *days as i64 * NANOS_PER_DAY,
        }
    }
}

impl Default for RebalanceSchedule {
    fn default() -> Self {
        Self::Biweekly
    }
}

/// Target weight for an asset.
#[derive(Debug, Clone, Copy)]
pub struct TargetWeight {
    /// Asset identifier.
    pub asset_id: AssetId,
    /// Target weight (0.0 to 1.0).
    pub weight: f64,
}

impl TargetWeight {
    /// Create a new target weight.
    #[must_use]
    pub fn new(asset_id: AssetId, weight: f64) -> Self {
        Self {
            asset_id,
            weight: weight.clamp(0.0, 1.0),
        }
    }
}

/// Biweekly portfolio rebalancer.
/// Generates orders to maintain target portfolio weights.
#[derive(Debug, Clone)]
pub struct BiweeklyRebalancer {
    /// Rebalancing schedule.
    schedule: RebalanceSchedule,
    /// Last rebalance timestamp.
    last_rebalance: i64,
    /// Target weights by asset.
    target_weights: Vec<TargetWeight>,
    /// Tolerance band (e.g., 0.02 = 2%).
    tolerance: f64,
    /// Minimum order size (shares).
    min_order_size: i64,
    /// Next order ID.
    next_order_id: u64,
}

impl BiweeklyRebalancer {
    /// Create a new biweekly rebalancer.
    #[must_use]
    pub fn new(target_weights: Vec<TargetWeight>) -> Self {
        Self {
            schedule: RebalanceSchedule::Biweekly,
            last_rebalance: 0,
            target_weights,
            tolerance: 0.02,     // 2% tolerance band
            min_order_size: 100, // B3 round lot
            next_order_id: 1,
        }
    }

    /// Create with custom schedule.
    #[must_use]
    pub fn with_schedule(mut self, schedule: RebalanceSchedule) -> Self {
        self.schedule = schedule;
        self
    }

    /// Set tolerance band.
    #[must_use]
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance.clamp(0.001, 0.5);
        self
    }

    /// Set minimum order size.
    #[must_use]
    pub fn with_min_order_size(mut self, size: i64) -> Self {
        self.min_order_size = (size / 100) * 100; // Round to B3 lot
        self.min_order_size = self.min_order_size.max(100);
        self
    }

    /// Create equal-weight portfolio for N assets.
    #[must_use]
    pub fn equal_weight(num_assets: usize) -> Self {
        let weight = 1.0 / num_assets as f64;
        let weights = (0..num_assets)
            .map(|i| TargetWeight::new(AssetId::new(i as u16), weight))
            .collect();
        Self::new(weights)
    }

    /// Check if rebalance is due.
    #[must_use]
    pub fn should_rebalance(&self, timestamp: i64) -> bool {
        if self.last_rebalance == 0 {
            return true; // First time
        }
        let elapsed = timestamp - self.last_rebalance;
        elapsed >= self.schedule.interval_nanos()
    }

    /// Record that a rebalance occurred.
    pub fn mark_rebalanced(&mut self, timestamp: i64) {
        self.last_rebalance = timestamp;
    }

    /// Calculate current weights from portfolio.
    #[must_use]
    pub fn current_weights(&self, portfolio: &Portfolio, prices: &[f64]) -> Vec<(AssetId, f64)> {
        let nav = portfolio.nav();
        if nav <= 0.0 {
            return Vec::new();
        }

        self.target_weights
            .iter()
            .map(|tw| {
                let position = portfolio.get_position(tw.asset_id);
                let price = prices.get(tw.asset_id.as_usize()).copied().unwrap_or(0.0);
                let value = position as f64 * price;
                let weight = value / nav;
                (tw.asset_id, weight)
            })
            .collect()
    }

    /// Check if any weight deviates beyond tolerance.
    #[must_use]
    pub fn needs_rebalance(&self, portfolio: &Portfolio, prices: &[f64]) -> bool {
        let current = self.current_weights(portfolio, prices);

        for (asset_id, current_weight) in &current {
            if let Some(target) = self
                .target_weights
                .iter()
                .find(|tw| tw.asset_id == *asset_id)
            {
                let deviation = (current_weight - target.weight).abs();
                if deviation > self.tolerance {
                    return true;
                }
            }
        }
        false
    }

    /// Calculate orders needed to rebalance.
    pub fn calculate_orders(
        &mut self,
        portfolio: &Portfolio,
        prices: &[f64],
        timestamp: i64,
    ) -> Vec<OrderEvent> {
        let nav = portfolio.nav();
        if nav <= 0.0 {
            return Vec::new();
        }

        let mut orders = Vec::new();

        // First pass: calculate all sells (to free up cash)
        for target in &self.target_weights {
            let price = prices
                .get(target.asset_id.as_usize())
                .copied()
                .unwrap_or(0.0);
            if price <= 0.0 {
                continue;
            }

            let current_position = portfolio.get_position(target.asset_id);
            let current_value = current_position as f64 * price;
            let target_value = nav * target.weight;
            let diff_value = target_value - current_value;

            // Calculate shares needed
            let shares_diff = (diff_value / price) as i64;

            // Round to B3 lot size
            let rounded_shares = (shares_diff / 100) * 100;

            // Sell orders
            if rounded_shares < -self.min_order_size {
                let order = OrderEvent {
                    order_id: OrderId::new(self.next_order_id),
                    timestamp,
                    asset_id: target.asset_id,
                    direction: OrderDirection::Sell,
                    quantity: (-rounded_shares).min(current_position.abs()),
                    order_type: OrderType::Market,
                    limit_price: None,
                    stop_price: None,
                    time_in_force: TimeInForce::Day,
                };
                self.next_order_id += 1;
                orders.push(order);
            }
        }

        // Second pass: calculate all buys
        for target in &self.target_weights {
            let price = prices
                .get(target.asset_id.as_usize())
                .copied()
                .unwrap_or(0.0);
            if price <= 0.0 {
                continue;
            }

            let current_position = portfolio.get_position(target.asset_id);
            let current_value = current_position as f64 * price;
            let target_value = nav * target.weight;
            let diff_value = target_value - current_value;

            let shares_diff = (diff_value / price) as i64;
            let rounded_shares = (shares_diff / 100) * 100;

            // Buy orders
            if rounded_shares >= self.min_order_size {
                let order = OrderEvent {
                    order_id: OrderId::new(self.next_order_id),
                    timestamp,
                    asset_id: target.asset_id,
                    direction: OrderDirection::Buy,
                    quantity: rounded_shares,
                    order_type: OrderType::Market,
                    limit_price: None,
                    stop_price: None,
                    time_in_force: TimeInForce::Day,
                };
                self.next_order_id += 1;
                orders.push(order);
            }
        }

        orders
    }

    /// Get target weights.
    #[must_use]
    pub fn target_weights(&self) -> &[TargetWeight] {
        &self.target_weights
    }

    /// Set new target weights.
    pub fn set_target_weights(&mut self, weights: Vec<TargetWeight>) {
        self.target_weights = weights;
    }

    /// Get last rebalance timestamp.
    #[must_use]
    pub fn last_rebalance(&self) -> i64 {
        self.last_rebalance
    }

    /// Get schedule.
    #[must_use]
    pub fn schedule(&self) -> RebalanceSchedule {
        self.schedule
    }
}

/// Strategy wrapper that applies rebalancing.
pub struct RebalancingStrategy<S> {
    /// Inner strategy (can be no-op for pure rebalancing).
    pub inner: S,
    /// Rebalancer.
    pub rebalancer: BiweeklyRebalancer,
}

impl<S> RebalancingStrategy<S> {
    /// Create a new rebalancing strategy.
    #[must_use]
    pub fn new(inner: S, rebalancer: BiweeklyRebalancer) -> Self {
        Self { inner, rebalancer }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_weight_creates_correct_weights() {
        let rebalancer = BiweeklyRebalancer::equal_weight(4);
        assert_eq!(rebalancer.target_weights.len(), 4);

        for tw in &rebalancer.target_weights {
            assert!((tw.weight - 0.25).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn schedule_interval_nanos() {
        const NANOS_PER_DAY: i64 = 86_400_000_000_000;

        assert_eq!(RebalanceSchedule::Daily(1).interval_nanos(), NANOS_PER_DAY);
        assert_eq!(
            RebalanceSchedule::Weekly(1).interval_nanos(),
            7 * NANOS_PER_DAY
        );
        assert_eq!(
            RebalanceSchedule::Biweekly.interval_nanos(),
            14 * NANOS_PER_DAY
        );
        assert_eq!(
            RebalanceSchedule::Monthly.interval_nanos(),
            30 * NANOS_PER_DAY
        );
    }

    #[test]
    fn should_rebalance_first_time() {
        let rebalancer = BiweeklyRebalancer::equal_weight(4);
        assert!(rebalancer.should_rebalance(1000));
    }

    #[test]
    fn should_not_rebalance_too_soon() {
        let mut rebalancer = BiweeklyRebalancer::equal_weight(4);
        rebalancer.mark_rebalanced(1000);

        // 1 day later
        const NANOS_PER_DAY: i64 = 86_400_000_000_000;
        assert!(!rebalancer.should_rebalance(1000 + NANOS_PER_DAY));
    }

    #[test]
    fn should_rebalance_after_interval() {
        let mut rebalancer = BiweeklyRebalancer::equal_weight(4);
        rebalancer.mark_rebalanced(1000);

        const NANOS_PER_DAY: i64 = 86_400_000_000_000;
        // 15 days later
        assert!(rebalancer.should_rebalance(1000 + 15 * NANOS_PER_DAY));
    }

    #[test]
    fn calculate_orders_empty_portfolio() {
        let mut rebalancer = BiweeklyRebalancer::equal_weight(4);
        let portfolio = Portfolio::new(100_000.0, 4);
        let prices = vec![100.0, 50.0, 200.0, 25.0];

        let orders = rebalancer.calculate_orders(&portfolio, &prices, 1000);

        // Should generate buy orders for all assets
        assert!(!orders.is_empty());
        for order in &orders {
            assert_eq!(order.direction, OrderDirection::Buy);
            assert!(order.quantity >= 100);
            assert_eq!(order.quantity % 100, 0); // B3 round lot
        }
    }

    #[test]
    fn min_order_size_rounds_to_lot() {
        let rebalancer = BiweeklyRebalancer::equal_weight(4).with_min_order_size(150);

        assert_eq!(rebalancer.min_order_size, 100);

        let rebalancer2 = BiweeklyRebalancer::equal_weight(4).with_min_order_size(250);

        assert_eq!(rebalancer2.min_order_size, 200);
    }
}














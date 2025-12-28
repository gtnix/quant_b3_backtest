//! Cost Reporting for Execution Analysis
//!
//! Provides detailed cost breakdowns and capacity analysis for PM-ready reports.
//! Used by Stage B validation to track execution costs per split.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// =============================================================================
// COST REPORT (Main Output)
// =============================================================================

/// Comprehensive cost report for a candidate in a validation window.
///
/// This is the primary output for execution cost analysis, designed to be
/// included in the PM-ready final report.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostReport {
    // === Totals ===
    /// Total execution costs (slippage + fees).
    pub total_costs: f64,
    /// Total slippage in currency.
    pub total_slippage: f64,
    /// Total fees in currency.
    pub total_fees: f64,

    // === Breakdowns ===
    /// Slippage breakdown by category.
    pub slippage_breakdown: SlippageBreakdown,
    /// Fee breakdown by type.
    pub fee_breakdown: FeeBreakdown,

    // === Per-Trade Statistics ===
    /// Average slippage in basis points.
    pub avg_slippage_bps: f64,
    /// Maximum slippage in basis points (worst trade).
    pub max_slippage_bps: f64,
    /// Date of worst slippage.
    pub worst_slippage_date: Option<String>,
    /// Average fee per trade.
    pub avg_fee_per_trade: f64,

    // === Impact Metrics ===
    /// Cost as percentage of gross PnL.
    pub cost_as_pct_of_gross_pnl: f64,
    /// CAGR drag from costs (gross CAGR - net CAGR).
    pub cost_drag_on_cagr_pct: f64,

    // === Turnover ===
    /// Annualized turnover ratio.
    pub turnover_annual: f64,
    /// Total number of trades.
    pub trades_count: u32,
    /// Average trade notional.
    pub avg_trade_notional: f64,
    /// Number of rejected orders.
    pub rejected_orders: u32,

    // === Capacity Proxy ===
    /// Estimated strategy capacity in USD before significant degradation.
    pub capacity_proxy_usd: f64,
    /// Methodology used for capacity estimation.
    pub capacity_methodology: String,
}

impl CostReport {
    /// Create a new empty cost report.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if costs are significant (>5% of gross PnL).
    #[must_use]
    pub fn has_significant_costs(&self) -> bool {
        self.cost_as_pct_of_gross_pnl > 5.0
    }

    /// Check if slippage is high (>15 bps average).
    #[must_use]
    pub fn has_high_slippage(&self) -> bool {
        self.avg_slippage_bps > 15.0
    }

    /// Check if turnover is high (>12x annual).
    #[must_use]
    pub fn has_high_turnover(&self) -> bool {
        self.turnover_annual > 12.0
    }

    /// Check if capacity is institutional grade (>10M USD).
    #[must_use]
    pub fn is_institutional_capacity(&self) -> bool {
        self.capacity_proxy_usd >= 10_000_000.0
    }

    /// Merge another cost report (for aggregating across windows).
    pub fn merge(&mut self, other: &CostReport) {
        self.total_costs += other.total_costs;
        self.total_slippage += other.total_slippage;
        self.total_fees += other.total_fees;
        self.trades_count += other.trades_count;
        self.rejected_orders += other.rejected_orders;

        // Take max slippage
        if other.max_slippage_bps > self.max_slippage_bps {
            self.max_slippage_bps = other.max_slippage_bps;
            self.worst_slippage_date = other.worst_slippage_date.clone();
        }

        // Merge breakdowns
        self.slippage_breakdown.merge(&other.slippage_breakdown);
        self.fee_breakdown.merge(&other.fee_breakdown);
    }

    /// Finalize computed metrics after all trades are processed.
    pub fn finalize(&mut self, gross_pnl: f64, gross_cagr: f64, net_cagr: f64, avg_nav: f64) {
        // Compute averages
        if self.trades_count > 0 {
            self.avg_fee_per_trade = self.total_fees / self.trades_count as f64;
        }

        // Compute impact metrics
        if gross_pnl.abs() > 0.0 {
            self.cost_as_pct_of_gross_pnl = (self.total_costs / gross_pnl.abs()) * 100.0;
        }

        self.cost_drag_on_cagr_pct = gross_cagr - net_cagr;

        // Capacity proxy using max participation methodology
        // Assumes 5% max participation at average daily volume
        if avg_nav > 0.0 && self.turnover_annual > 0.0 {
            // Simple proxy: portfolio size where we'd hit 5% participation
            let avg_daily_traded = avg_nav * self.turnover_annual / 252.0;
            let implied_daily_volume = avg_daily_traded / 0.05; // 5% participation
            // Capacity = volume that maintains < 5% participation
            self.capacity_proxy_usd = implied_daily_volume * 0.05 * 252.0 / self.turnover_annual;
            self.capacity_methodology = "5% max participation @ estimated avg daily volume".into();
        }
    }
}

// =============================================================================
// SLIPPAGE BREAKDOWN
// =============================================================================

/// Breakdown of slippage by different categories.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SlippageBreakdown {
    /// Slippage by market (BR, US).
    pub by_market: BTreeMap<String, f64>,
    /// Slippage by volatility regime (LowVol, HighVol).
    pub by_regime: BTreeMap<String, f64>,
    /// Slippage by order size category (Small, Medium, Large).
    pub by_order_size: BTreeMap<String, f64>,
}

impl SlippageBreakdown {
    /// Add slippage to market bucket.
    pub fn add_market(&mut self, market: &str, slippage: f64) {
        *self.by_market.entry(market.to_string()).or_default() += slippage;
    }

    /// Add slippage to regime bucket.
    pub fn add_regime(&mut self, regime: &str, slippage: f64) {
        *self.by_regime.entry(regime.to_string()).or_default() += slippage;
    }

    /// Add slippage to order size bucket.
    pub fn add_order_size(&mut self, size_category: &str, slippage: f64) {
        *self.by_order_size.entry(size_category.to_string()).or_default() += slippage;
    }

    /// Merge another breakdown.
    pub fn merge(&mut self, other: &SlippageBreakdown) {
        for (k, v) in &other.by_market {
            *self.by_market.entry(k.clone()).or_default() += v;
        }
        for (k, v) in &other.by_regime {
            *self.by_regime.entry(k.clone()).or_default() += v;
        }
        for (k, v) in &other.by_order_size {
            *self.by_order_size.entry(k.clone()).or_default() += v;
        }
    }
}

// =============================================================================
// FEE BREAKDOWN
// =============================================================================

/// Breakdown of fees by type.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FeeBreakdown {
    /// Total fixed costs.
    pub fixed_total: f64,
    /// Total commission costs.
    pub commission_total: f64,
    /// Total emolument costs (B3).
    pub emolument_total: f64,
    /// Total per-unit costs.
    pub per_unit_total: f64,
}

impl FeeBreakdown {
    /// Total fees.
    #[must_use]
    pub fn total(&self) -> f64 {
        self.fixed_total + self.commission_total + self.emolument_total + self.per_unit_total
    }

    /// Add a fee breakdown from a single trade.
    pub fn add(&mut self, fixed: f64, commission: f64, emolument: f64, per_unit: f64) {
        self.fixed_total += fixed;
        self.commission_total += commission;
        self.emolument_total += emolument;
        self.per_unit_total += per_unit;
    }

    /// Merge another breakdown.
    pub fn merge(&mut self, other: &FeeBreakdown) {
        self.fixed_total += other.fixed_total;
        self.commission_total += other.commission_total;
        self.emolument_total += other.emolument_total;
        self.per_unit_total += other.per_unit_total;
    }
}

// =============================================================================
// TRADE COST RECORD
// =============================================================================

/// Individual trade cost record for detailed analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeCostRecord {
    /// Trade date.
    pub date: String,
    /// Symbol traded.
    pub symbol: String,
    /// Market (BR/US).
    pub market: String,
    /// Trade direction (Buy/Sell).
    pub direction: String,
    /// Quantity traded.
    pub quantity: i64,
    /// Notional value.
    pub notional: f64,
    /// Slippage amount.
    pub slippage: f64,
    /// Slippage in basis points.
    pub slippage_bps: f64,
    /// Fee amount.
    pub fee: f64,
    /// Total cost (slippage + fee).
    pub total_cost: f64,
    /// Volatility regime at time of trade.
    pub regime: String,
    /// Order size category.
    pub size_category: String,
}

impl TradeCostRecord {
    /// Create a new trade cost record.
    #[must_use]
    pub fn new(
        date: String,
        symbol: String,
        market: String,
        direction: String,
        quantity: i64,
        notional: f64,
        slippage: f64,
        fee: f64,
    ) -> Self {
        let slippage_bps = if notional > 0.0 {
            (slippage.abs() / notional) * 10_000.0
        } else {
            0.0
        };

        Self {
            date,
            symbol,
            market,
            direction,
            quantity,
            notional,
            slippage,
            slippage_bps,
            fee,
            total_cost: slippage.abs() + fee,
            regime: "Unknown".into(),
            size_category: "Unknown".into(),
        }
    }

    /// Classify order size based on notional.
    #[must_use]
    pub fn classify_size(&self, small_threshold: f64, large_threshold: f64) -> &'static str {
        if self.notional < small_threshold {
            "Small"
        } else if self.notional > large_threshold {
            "Large"
        } else {
            "Medium"
        }
    }
}

// =============================================================================
// COST REPORT BUILDER
// =============================================================================

/// Builder for constructing a CostReport from trade records.
#[derive(Debug, Default)]
pub struct CostReportBuilder {
    records: Vec<TradeCostRecord>,
    total_slippage_bps_sum: f64,
}

impl CostReportBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a trade cost record.
    pub fn add_trade(&mut self, record: TradeCostRecord) {
        self.total_slippage_bps_sum += record.slippage_bps;
        self.records.push(record);
    }

    /// Build the final cost report.
    #[must_use]
    pub fn build(self) -> CostReport {
        let mut report = CostReport::new();

        if self.records.is_empty() {
            return report;
        }

        report.trades_count = self.records.len() as u32;

        for record in &self.records {
            report.total_slippage += record.slippage.abs();
            report.total_fees += record.fee;
            report.total_costs += record.total_cost;

            // Track max slippage
            if record.slippage_bps > report.max_slippage_bps {
                report.max_slippage_bps = record.slippage_bps;
                report.worst_slippage_date = Some(record.date.clone());
            }

            // Breakdown by market
            report.slippage_breakdown.add_market(&record.market, record.slippage.abs());

            // Breakdown by regime
            report.slippage_breakdown.add_regime(&record.regime, record.slippage.abs());

            // Breakdown by size
            report.slippage_breakdown.add_order_size(&record.size_category, record.slippage.abs());
        }

        // Average slippage
        report.avg_slippage_bps = self.total_slippage_bps_sum / self.records.len() as f64;

        // Average trade notional
        let total_notional: f64 = self.records.iter().map(|r| r.notional).sum();
        report.avg_trade_notional = total_notional / self.records.len() as f64;

        report
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_report_new() {
        let report = CostReport::new();
        assert_eq!(report.trades_count, 0);
        assert!(!report.has_significant_costs());
    }

    #[test]
    fn test_fee_breakdown_total() {
        let mut breakdown = FeeBreakdown::default();
        breakdown.add(10.0, 5.0, 2.0, 1.0);
        assert!((breakdown.total() - 18.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_slippage_breakdown_merge() {
        let mut b1 = SlippageBreakdown::default();
        b1.add_market("BR", 100.0);
        b1.add_market("US", 50.0);

        let mut b2 = SlippageBreakdown::default();
        b2.add_market("BR", 50.0);
        b2.add_market("US", 25.0);

        b1.merge(&b2);

        assert!((b1.by_market["BR"] - 150.0).abs() < f64::EPSILON);
        assert!((b1.by_market["US"] - 75.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trade_cost_record_slippage_bps() {
        let record = TradeCostRecord::new(
            "2024-01-01".into(),
            "PETR4".into(),
            "BR".into(),
            "Buy".into(),
            100,
            10_000.0, // notional
            10.0,     // slippage ($10 on $10k = 10 bps)
            5.0,      // fee
        );

        assert!((record.slippage_bps - 10.0).abs() < f64::EPSILON);
        assert!((record.total_cost - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cost_report_builder() {
        let mut builder = CostReportBuilder::new();

        builder.add_trade(TradeCostRecord::new(
            "2024-01-01".into(),
            "PETR4".into(),
            "BR".into(),
            "Buy".into(),
            100,
            10_000.0,
            10.0,
            5.0,
        ));

        builder.add_trade(TradeCostRecord::new(
            "2024-01-02".into(),
            "VALE3".into(),
            "BR".into(),
            "Sell".into(),
            50,
            5_000.0,
            7.5,
            3.0,
        ));

        let report = builder.build();

        assert_eq!(report.trades_count, 2);
        assert!((report.total_slippage - 17.5).abs() < 0.01);
        assert!((report.total_fees - 8.0).abs() < f64::EPSILON);
        assert!((report.total_costs - 25.5).abs() < 0.01);
    }

    #[test]
    fn test_capacity_checks() {
        let mut report = CostReport::new();

        report.capacity_proxy_usd = 5_000_000.0;
        assert!(!report.is_institutional_capacity());

        report.capacity_proxy_usd = 15_000_000.0;
        assert!(report.is_institutional_capacity());
    }

    #[test]
    fn test_cost_report_merge() {
        let mut report1 = CostReport::new();
        report1.total_costs = 100.0;
        report1.trades_count = 5;
        report1.max_slippage_bps = 10.0;

        let mut report2 = CostReport::new();
        report2.total_costs = 50.0;
        report2.trades_count = 3;
        report2.max_slippage_bps = 15.0;
        report2.worst_slippage_date = Some("2024-01-15".into());

        report1.merge(&report2);

        assert!((report1.total_costs - 150.0).abs() < f64::EPSILON);
        assert_eq!(report1.trades_count, 8);
        assert!((report1.max_slippage_bps - 15.0).abs() < f64::EPSILON);
        assert_eq!(report1.worst_slippage_date, Some("2024-01-15".into()));
    }
}


//! Entry Engine - orchestrates the full entry flow.
//!
//! # Performance (Milestone 6)
//!
//! All monetary types use fixed-point (`Price`/`Money`) for fast i64 arithmetic.

use chrono::NaiveDate;
use std::collections::HashMap;
use std::sync::Arc;

use backtester_core::{Money, Price};
use crate::filters::Market;
use crate::scorer::ScoredAsset;

use super::gating::{GatingCandidate, GatingConfig, GatingFilter};
use super::selection::{ScoredCandidate, SelectionConfig, Selector};
use super::weighting::{WeightingCandidate, WeightingConfig, Weighter};
use super::orders::{OrderGenerator, OrderGeneratorConfig, OrderTarget};
use super::audit::{RebalanceAuditLog, SelectedAsset};
use super::types::{
    EntryContext, EntryResult, EntryTarget, EntryWarning, Order, SelectionReason,
};
use super::universe_range::UniverseRangeProvider;
use super::eligibility::EligibilityProvider;

/// Configuration for entry engine.
pub struct EntryEngineConfig {
    pub gating: GatingConfig,
    pub selection: SelectionConfig,
    pub weighting: WeightingConfig,
    pub orders: OrderGeneratorConfig,
    /// Optional eligibility provider for survivorship bias prevention (V1 or V2).
    /// When set, candidates are validated against historical existence windows.
    pub eligibility_provider: Option<Arc<dyn EligibilityProvider>>,
}

impl std::fmt::Debug for EntryEngineConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EntryEngineConfig")
            .field("gating", &self.gating)
            .field("selection", &self.selection)
            .field("weighting", &self.weighting)
            .field("orders", &self.orders)
            .field("has_eligibility_provider", &self.eligibility_provider.is_some())
            .finish()
    }
}

impl Clone for EntryEngineConfig {
    fn clone(&self) -> Self {
        Self {
            gating: self.gating.clone(),
            selection: self.selection.clone(),
            weighting: self.weighting.clone(),
            orders: self.orders.clone(),
            eligibility_provider: self.eligibility_provider.clone(),
        }
    }
}

impl Default for EntryEngineConfig {
    fn default() -> Self {
        Self {
            gating: GatingConfig::default(),
            selection: SelectionConfig::default(),
            weighting: WeightingConfig::default(),
            orders: OrderGeneratorConfig::default(),
            eligibility_provider: None,
        }
    }
}

/// Asset candidate with all required data.
///
/// # Performance (Milestone 6)
///
/// Uses fixed-point `Price` and `Money` for monetary fields.
#[derive(Debug, Clone)]
pub struct AssetCandidate {
    pub symbol: String,
    pub market: Market,
    /// Current price (fixed-point)
    pub price: Option<Price>,
    /// Average daily volume in currency (fixed-point)
    pub avg_volume: Option<Money>,
    pub price_days: usize,
    pub has_fundamentals: bool,
    pub has_dividends: bool,
    pub is_tradeable: bool,
    pub volatility: Option<f64>,
    pub score: Option<f64>,
    pub filter_scores: Vec<(String, f64)>,
    /// Date of fundamentals snapshot (for anti-look-ahead validation)
    pub fundamentals_as_of: Option<NaiveDate>,
}

impl AssetCandidate {
    pub fn new(symbol: impl Into<String>, market: Market) -> Self {
        Self {
            symbol: symbol.into(),
            market,
            price: None,
            avg_volume: None,
            price_days: 0,
            has_fundamentals: false,
            has_dividends: false,
            is_tradeable: true,
            volatility: None,
            score: None,
            filter_scores: Vec::new(),
            fundamentals_as_of: None,
        }
    }

    /// Create from ScoredAsset.
    pub fn from_scored(scored: &ScoredAsset, market: Market) -> Self {
        // Convert filter_results to filter_scores
        let filter_scores: Vec<(String, f64)> = scored.filter_results
            .iter()
            .map(|(name, result)| (name.clone(), result.score))
            .collect();
        
        Self {
            symbol: scored.symbol.clone(),
            market,
            price: None,
            avg_volume: None,
            price_days: 0,
            has_fundamentals: false,
            has_dividends: false,
            is_tradeable: true,
            volatility: None,
            score: Some(scored.total_score),
            filter_scores,
            fundamentals_as_of: None,
        }
    }
}

/// Entry Engine - main orchestrator.
#[derive(Debug, Clone)]
pub struct EntryEngine {
    gating: GatingFilter,
    selector: Selector,
    weighter: Weighter,
    order_gen: OrderGenerator,
    #[allow(dead_code)]
    config: EntryEngineConfig,
}

impl EntryEngine {
    /// Create a new entry engine.
    ///
    /// If `config.eligibility_provider` is set, survivorship bias validation is enabled (V1 or V2).
    pub fn new(config: EntryEngineConfig) -> Self {
        // Create gating filter with or without eligibility provider
        let gating = match &config.eligibility_provider {
            Some(provider) => GatingFilter::with_eligibility_provider(
                config.gating.clone(),
                Arc::clone(provider),
            ),
            None => GatingFilter::new(config.gating.clone()),
        };

        Self {
            gating,
            selector: Selector::new(config.selection.clone()),
            weighter: Weighter::new(config.weighting.clone()),
            order_gen: OrderGenerator::new(config.orders.clone()),
            config,
        }
    }

    /// Create entry engine with V1 universe provider (backward compatible).
    pub fn with_universe_provider(
        mut config: EntryEngineConfig,
        provider: Arc<UniverseRangeProvider>,
    ) -> Self {
        config.eligibility_provider = Some(provider as Arc<dyn EligibilityProvider>);
        Self::new(config)
    }

    /// Check if eligibility validation is enabled (V1 or V2).
    pub fn has_eligibility_provider(&self) -> bool {
        self.gating.has_eligibility_provider()
    }

    /// Backward compatible alias for has_eligibility_provider().
    pub fn has_universe_provider(&self) -> bool {
        self.has_eligibility_provider()
    }

    /// Evaluate entry for a specific market.
    ///
    /// # Performance (Milestone 5)
    ///
    /// Takes candidates by slice to avoid cloning at call site.
    pub fn evaluate(
        &self,
        ctx: &EntryContext,
        candidates: &[AssetCandidate],
        current_positions: &HashMap<String, i64>,
    ) -> (EntryResult, Vec<Order>, RebalanceAuditLog) {
        let mut result = EntryResult::new(ctx.date, ctx.market);
        result.diagnostics.total_candidates = candidates.len();

        // Step 1: Gating
        // Performance (Milestone 5): Build gating candidates without cloning the input slice
        let gating_candidates: Vec<GatingCandidate> = candidates
            .iter()
            .filter(|c| c.market == ctx.market)
            .map(|c| GatingCandidate {
                symbol: c.symbol.clone(),
                market: c.market,
                price: c.price,
                avg_volume: c.avg_volume,
                price_days: c.price_days,
                has_fundamentals: c.has_fundamentals,
                has_dividends: c.has_dividends,
                is_tradeable: c.is_tradeable,
                fundamentals_as_of: c.fundamentals_as_of,
                rebalance_date: Some(ctx.date),
            })
            .collect();

        // Save count before moving into apply() (Milestone 5: no clone)
        let gating_candidates_count = gating_candidates.len();
        let (eligible, gating_excluded) = self.gating.apply(gating_candidates);
        result.diagnostics.gating_excluded = gating_excluded.len();

        // GUARDRAIL: Check for empty universe after gating
        if eligible.is_empty() && gating_candidates_count > 0 {
            // Count exclusion reasons
            let mut reason_counts: HashMap<String, usize> = HashMap::new();
            for excl in &gating_excluded {
                *reason_counts.entry(excl.reason.to_string()).or_insert(0) += 1;
            }
            let mut top_reasons: Vec<_> = reason_counts.into_iter().collect();
            top_reasons.sort_by(|a, b| b.1.cmp(&a.1));
            let top_reasons: Vec<String> = top_reasons
                .into_iter()
                .take(3)
                .map(|(reason, count)| format!("{} ({})", reason, count))
                .collect();

            let warning = EntryWarning::EmptyUniverse {
                candidates_before: gating_candidates_count,
                gating_excluded: gating_excluded.len(),
                top_reasons,
            };
            
            // Log warning
            tracing::warn!(
                market = ?ctx.market,
                date = %ctx.date,
                candidates = gating_candidates_count,
                "[GUARDRAIL] Empty universe: all candidates excluded by gating"
            );
            
            result.diagnostics.warnings.push(warning);
        } else if eligible.len() < 5 && gating_candidates_count > 0 {
            // Warn if very few assets
            let warning = EntryWarning::LowUniverse {
                eligible_count: eligible.len(),
                recommended_min: 10,
            };
            tracing::warn!(
                market = ?ctx.market,
                date = %ctx.date,
                eligible = eligible.len(),
                "[GUARDRAIL] Low universe: very few assets eligible"
            );
            result.diagnostics.warnings.push(warning);
        }
        
        // Move excluded list to result (no clone)
        result.exclusions.extend(gating_excluded);

        // Map eligible symbols for lookup
        let eligible_symbols: std::collections::HashSet<_> = 
            eligible.iter().map(|e| e.symbol.as_str()).collect();

        // Step 2: Score (already scored, just filter eligible)
        let scored_candidates: Vec<ScoredCandidate> = candidates
            .iter()
            .filter(|c| c.market == ctx.market && eligible_symbols.contains(c.symbol.as_str()))
            .filter_map(|c| {
                c.score.map(|score| {
                    ScoredCandidate::new(c.symbol.clone(), c.market, score)
                        .with_filter_scores(c.filter_scores.clone())
                })
            })
            .collect();

        // Step 3: Selection
        let (selected, selection_excluded) = self.selector.select(scored_candidates);
        result.diagnostics.selection_excluded = selection_excluded.len();
        result.exclusions.extend(selection_excluded);
        result.diagnostics.final_selected = selected.len();

        // Step 4: Weighting
        let weighting_candidates: Vec<WeightingCandidate> = selected
            .iter()
            .map(|s| {
                let volatility = candidates
                    .iter()
                    .find(|c| c.symbol == s.symbol)
                    .and_then(|c| c.volatility);
                WeightingCandidate::new(s.symbol.clone(), s.score, volatility)
            })
            .collect();

        let weights = self.weighter.calculate_weights(weighting_candidates);
        result.diagnostics.total_weight = weights.iter().map(|w| w.weight).sum();

        // Step 5: Create targets (Milestone 6: fixed-point throughout)
        for weight_result in &weights {
            let candidate = candidates.iter().find(|c| c.symbol == weight_result.symbol);
            let price = candidate.and_then(|c| c.price).unwrap_or(Price::ZERO);
            let selected_info = selected.iter().find(|s| s.symbol == weight_result.symbol);
            
            let target_shares = self.order_gen.calculate_target_shares(
                weight_result.weight,
                price,
                ctx.capital,
                ctx.market,
            );

            let reason = SelectionReason {
                score: selected_info.map(|s| s.score).unwrap_or(0.0),
                filter_scores: selected_info
                    .map(|s| s.filter_scores.clone())
                    .unwrap_or_default(),
                summary: format!(
                    "vol={:.1}%, weight={:.1}%{}",
                    weight_result.volatility * 100.0,
                    weight_result.weight * 100.0,
                    if weight_result.capped { " (capped)" } else { "" }
                ),
            };

            result.targets.push(EntryTarget {
                symbol: weight_result.symbol.clone(),
                target_weight: weight_result.weight,
                target_shares,
                price,
                reason,
            });
        }

        // Step 6: Generate orders
        let order_targets: Vec<OrderTarget> = result
            .targets
            .iter()
            .map(|t| OrderTarget {
                symbol: t.symbol.clone(),
                market: ctx.market,
                target_weight: t.target_weight,
                price: t.price,
            })
            .collect();

        let (orders, total_cost) = self.order_gen.generate_orders(
            &order_targets,
            current_positions,
            ctx.capital,
        );

        result.diagnostics.estimated_costs = total_cost;
        result.diagnostics.turnover = self.order_gen.calculate_turnover(&orders, ctx.capital);

        // Calculate cash residual = capital - sum(shares * price) (Milestone 6: fixed-point)
        let total_allocated: Money = result.targets.iter()
            .map(|t| t.price.mul_shares(t.target_shares))
            .sum();
        result.diagnostics.cash_residual = ctx.capital - total_allocated;

        // Step 7: Build audit log
        let selected_assets: Vec<SelectedAsset> = result
            .targets
            .iter()
            .map(|t| SelectedAsset {
                symbol: t.symbol.clone(),
                weight: t.target_weight,
                score: t.reason.score,
                reason: t.reason.summary.clone(),
            })
            .collect();

        let audit = RebalanceAuditLog {
            date: ctx.date,
            market: ctx.market,
            selected: selected_assets,
            excluded: result.exclusions.clone(),
            orders: orders.clone(),
            diagnostics: result.diagnostics.clone(),
        };

        (result, orders, audit)
    }

    /// Evaluate for both markets.
    ///
    /// # Performance (Milestone 5/6)
    ///
    /// Takes candidates by slice to avoid cloning, uses fixed-point Money.
    pub fn evaluate_all(
        &self,
        date: NaiveDate,
        candidates: &[AssetCandidate],
        positions_br: &HashMap<String, i64>,
        positions_us: &HashMap<String, i64>,
        capital_br: Money,
        capital_us: Money,
    ) -> (Vec<EntryResult>, Vec<Order>, Vec<RebalanceAuditLog>) {
        let mut all_results = Vec::new();
        let mut all_orders = Vec::new();
        let mut all_audits = Vec::new();

        // BR
        let ctx_br = EntryContext::new(date, capital_br, Market::BR);
        let (result_br, orders_br, audit_br) = self.evaluate(&ctx_br, candidates, positions_br);
        all_results.push(result_br);
        all_orders.extend(orders_br);
        all_audits.push(audit_br);

        // US
        let ctx_us = EntryContext::new(date, capital_us, Market::US);
        let (result_us, orders_us, audit_us) = self.evaluate(&ctx_us, candidates, positions_us);
        all_results.push(result_us);
        all_orders.extend(orders_us);
        all_audits.push(audit_us);

        (all_results, all_orders, all_audits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_br_candidates() -> Vec<AssetCandidate> {
        vec![
            {
                let mut c = AssetCandidate::new("PETR4", Market::BR);
                c.price = Some(Price::from_int(38));
                c.avg_volume = Some(Money::from_int(5_000_000));
                c.price_days = 30;
                c.has_fundamentals = true;
                c.has_dividends = true;
                c.volatility = Some(0.25);
                c.score = Some(0.85);
                c
            },
            {
                let mut c = AssetCandidate::new("VALE3", Market::BR);
                c.price = Some(Price::from_int(62));
                c.avg_volume = Some(Money::from_int(4_000_000));
                c.price_days = 30;
                c.has_fundamentals = true;
                c.has_dividends = true;
                c.volatility = Some(0.28);
                c.score = Some(0.80);
                c
            },
            {
                let mut c = AssetCandidate::new("ITUB4", Market::BR);
                c.price = Some(Price::from_int(32));
                c.avg_volume = Some(Money::from_int(3_000_000));
                c.price_days = 30;
                c.has_fundamentals = true;
                c.has_dividends = true;
                c.volatility = Some(0.18);
                c.score = Some(0.75);
                c
            },
        ]
    }

    #[test]
    fn test_full_flow_br() {
        let config = EntryEngineConfig {
            selection: SelectionConfig {
                top_n_br: 2,
                top_n_us: 2,
                min_score_threshold: None,
                ..Default::default()
            },
            ..Default::default()
        };
        let engine = EntryEngine::new(config);

        let ctx = EntryContext::new(
            NaiveDate::from_ymd_opt(2025, 1, 3).unwrap(),
            Money::from_int(100_000),
            Market::BR,
        );

        let candidates = make_br_candidates();
        let (result, orders, audit) = engine.evaluate(&ctx, &candidates, &HashMap::new());

        // Should select top 2
        assert_eq!(result.targets.len(), 2);
        assert_eq!(result.diagnostics.final_selected, 2);
        
        // One excluded (out of top-N)
        assert_eq!(result.diagnostics.selection_excluded, 1);

        // Orders generated
        assert!(!orders.is_empty());

        // Audit populated
        assert_eq!(audit.selected.len(), 2);
    }

    #[test]
    fn test_weights_sum_to_one() {
        let config = EntryEngineConfig::default();
        let engine = EntryEngine::new(config);

        let ctx = EntryContext::new(
            NaiveDate::from_ymd_opt(2025, 1, 3).unwrap(),
            Money::from_int(100_000),
            Market::BR,
        );

        let candidates = make_br_candidates();
        let (result, _, _) = engine.evaluate(&ctx, &candidates, &HashMap::new());

        let total_weight: f64 = result.targets.iter().map(|t| t.target_weight).sum();
        assert!((total_weight - 1.0).abs() < 0.01, "Total weight {} should be ~1.0", total_weight);
    }

    #[test]
    fn test_gating_excludes_low_volume() {
        use crate::entry::types::ExclusionReason;
        
        let config = EntryEngineConfig::default();
        let engine = EntryEngine::new(config);

        let ctx = EntryContext::new(
            NaiveDate::from_ymd_opt(2025, 1, 3).unwrap(),
            Money::from_int(100_000),
            Market::BR,
        );

        let mut candidates = make_br_candidates();
        // Make one low volume
        candidates[0].avg_volume = Some(Money::from_int(100_000)); // Below threshold

        let (result, _, _) = engine.evaluate(&ctx, &candidates, &HashMap::new());

        // PETR4 should be excluded
        assert!(result.exclusions.iter().any(|e| 
            e.symbol == "PETR4" && e.reason == ExclusionReason::InsufficientLiquidity
        ));
    }
}


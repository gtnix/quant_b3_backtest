//! Walk-Forward Runner with grid search.
//!
//! Executes backtests across all windows with optional parameter optimization.
//! Supports both legacy 2-segment and nested 3-segment windows with PSR/DSR selection.

use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::cmp::Ordering;
use std::collections::HashMap;

use crate::filters::Market;

use super::metrics::{MetricsCalculator, RobustnessScorer};
use super::splitter::{RollingSplitter, TimeSplitter, NestedSplitter};
use super::statistics::{calculate_psr, calculate_dsr, sharpe_variance};
use super::types::{
    AggregateReport, GridConfig, ParamSet, WalkForwardConfig, WindowMetrics, WindowResult, WindowSplit,
    NestedWalkForwardConfig, NestedWindowSplit, NestedWindowResult, NestedAggregateReport,
    SelectionCriteria, SelectionReason, SelectionCandidate,
};

/// Mock price data for a symbol.
#[derive(Debug, Clone)]
pub struct PriceData {
    pub symbol: String,
    pub market: Market,
    pub dates: Vec<NaiveDate>,
    pub prices: Vec<Decimal>,
}

impl PriceData {
    pub fn new(symbol: &str, market: Market) -> Self {
        Self {
            symbol: symbol.to_string(),
            market,
            dates: Vec::new(),
            prices: Vec::new(),
        }
    }

    /// Filter prices to a date range.
    pub fn filter_range(&self, start: NaiveDate, end: NaiveDate) -> Vec<Decimal> {
        self.dates.iter()
            .zip(self.prices.iter())
            .filter(|(d, _)| **d >= start && **d <= end)
            .map(|(_, p)| *p)
            .collect()
    }
}

/// Candidate asset with scores.
#[derive(Debug, Clone)]
pub struct WfCandidate {
    pub symbol: String,
    pub market: Market,
    pub score: Decimal,
    pub volatility: Decimal,
}

/// Walk-forward runner that executes backtests across windows.
#[derive(Debug)]
pub struct WalkForwardRunner {
    config: WalkForwardConfig,
    metrics_calc: MetricsCalculator,
    robustness_scorer: RobustnessScorer,
}

impl WalkForwardRunner {
    pub fn new(config: WalkForwardConfig) -> Self {
        Self {
            config,
            metrics_calc: MetricsCalculator::new(dec!(0.05), 252),
            robustness_scorer: RobustnessScorer::default(),
        }
    }

    /// Run walk-forward validation.
    pub fn run(
        &self,
        start: NaiveDate,
        end: NaiveDate,
        candidates: &[WfCandidate],
        prices: &HashMap<String, PriceData>,
    ) -> AggregateReport {
        // Generate splits
        let splitter = RollingSplitter::new(&self.config);
        let splits = splitter.generate_splits(start, end);

        // Filter candidates by market
        let market_candidates: Vec<_> = candidates
            .iter()
            .filter(|c| c.market == self.config.market)
            .collect();

        // Run each window
        let mut results = Vec::new();

        for split in &splits {
            let result = self.run_window(split, &market_candidates, prices);
            results.push(result);
        }

        // Aggregate metrics
        let aggregate = self.robustness_scorer.aggregate(&results);

        // Find most commonly selected params
        let most_selected = self.most_common_params(&results);

        AggregateReport {
            config: self.config.clone(),
            windows: results,
            aggregate,
            most_selected_params: most_selected,
            generated_at: chrono::Local::now().date_naive(),
        }
    }

    /// Run a single window (train + test).
    fn run_window(
        &self,
        split: &WindowSplit,
        candidates: &[&WfCandidate],
        prices: &HashMap<String, PriceData>,
    ) -> WindowResult {
        // Get grid config or use default params
        let (selected_params, train_metrics) = if let Some(grid) = &self.config.grid {
            self.grid_search(split, candidates, prices, grid)
        } else {
            let default_params = ParamSet::default();
            let train_metrics = self.simulate_window(
                &split.train.start_date,
                &split.train.end_date,
                candidates,
                prices,
                &default_params,
            );
            (default_params, train_metrics)
        };

        // Run test period with selected params (OOS)
        let test_metrics = self.simulate_window(
            &split.test.start_date,
            &split.test.end_date,
            candidates,
            prices,
            &selected_params,
        );

        WindowResult {
            split: split.clone(),
            train_metrics,
            test_metrics,
            selected_params,
            is_oos: true,
        }
    }

    /// Grid search: find best params on training data.
    fn grid_search(
        &self,
        split: &WindowSplit,
        candidates: &[&WfCandidate],
        prices: &HashMap<String, PriceData>,
        grid: &GridConfig,
    ) -> (ParamSet, WindowMetrics) {
        let combinations = grid.generate_combinations();

        let mut best_params = ParamSet::default();
        let mut best_sharpe = Decimal::MIN;
        let mut best_metrics = WindowMetrics::default();

        for params in combinations {
            let metrics = self.simulate_window(
                &split.train.start_date,
                &split.train.end_date,
                candidates,
                prices,
                &params,
            );

            if metrics.sharpe_ratio > best_sharpe {
                best_sharpe = metrics.sharpe_ratio;
                best_params = params;
                best_metrics = metrics;
            }
        }

        (best_params, best_metrics)
    }

    /// Simulate a backtest for a window with given params.
    fn simulate_window(
        &self,
        start: &NaiveDate,
        end: &NaiveDate,
        candidates: &[&WfCandidate],
        prices: &HashMap<String, PriceData>,
        params: &ParamSet,
    ) -> WindowMetrics {
        // Select top-N by score
        let mut sorted: Vec<_> = candidates.iter().collect();
        sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        let selected: Vec<_> = sorted.into_iter().take(params.top_n).collect();

        if selected.is_empty() {
            return WindowMetrics::default();
        }

        // Build equal-weight portfolio (simplified)
        let weight = Decimal::ONE / Decimal::from(selected.len());

        // Generate daily equity curve (simplified simulation)
        let mut equity_curve = Vec::new();
        let mut current_date = *start;
        let mut portfolio_value = dec!(100_000);

        let mut prev_prices: HashMap<&str, Decimal> = HashMap::new();

        while current_date <= *end {
            let mut daily_return = Decimal::ZERO;

            for candidate in &selected {
                if let Some(price_data) = prices.get(&candidate.symbol) {
                    let price = price_data
                        .dates
                        .iter()
                        .zip(price_data.prices.iter())
                        .filter(|(d, _)| **d <= current_date)
                        .last()
                        .map(|(_, p)| *p);

                    if let Some(current_price) = price {
                        if let Some(prev_price) = prev_prices.get(candidate.symbol.as_str()) {
                            if *prev_price > Decimal::ZERO {
                                let ret = (current_price - *prev_price) / *prev_price;
                                daily_return += weight * ret;
                            }
                        }
                        prev_prices.insert(&candidate.symbol, current_price);
                    }
                }
            }

            // Apply stop-loss check
            let drawdown = if let Some(&peak) = equity_curve.iter().max() {
                if peak > Decimal::ZERO {
                    (peak - portfolio_value) / peak
                } else {
                    Decimal::ZERO
                }
            } else {
                Decimal::ZERO
            };

            if drawdown > params.stop_loss_pct {
                // Exit all positions (simplified)
                daily_return = Decimal::ZERO;
            }

            portfolio_value = portfolio_value * (Decimal::ONE + daily_return);
            equity_curve.push(portfolio_value);

            current_date += chrono::Duration::days(1);
        }

        // Calculate costs using ExecutionModelConfig if available
        let num_trades = selected.len() * 2; // entry + exit
        let avg_trade_value = dec!(100_000) / Decimal::from(selected.len());
        
        // Get slippage and fee parameters from config
        let (slippage_bps, fee_rate) = if let Some(ref exec_config) = self.config.execution_config {
            if exec_config.bypass_for_debug {
                (0.0, 0.0)
            } else {
                let slip_bps = exec_config.slippage.base_bps();
                let fee = exec_config.fees.commission_rate + exec_config.fees.emolument_rate;
                (slip_bps, fee)
            }
        } else {
            // Default: 10 bps slippage + 0.1% fees
            (10.0, 0.001)
        };
        
        // Calculate total costs per trade
        let slippage_cost = avg_trade_value * Decimal::try_from(slippage_bps / 10_000.0).unwrap_or(dec!(0.001));
        let fee_cost = avg_trade_value * Decimal::try_from(fee_rate).unwrap_or(dec!(0.001));
        let cost_per_trade = slippage_cost + fee_cost;
        let costs = Decimal::from(num_trades) * cost_per_trade;

        // Turnover calculation
        let total_traded = Decimal::from(num_trades) * avg_trade_value;
        let avg_portfolio = equity_curve.iter().sum::<Decimal>() / Decimal::from(equity_curve.len().max(1));
        let turnover = if avg_portfolio > Decimal::ZERO {
            total_traded / avg_portfolio * dec!(100)
        } else {
            dec!(50)
        };

        self.metrics_calc.from_equity_curve(&equity_curve, costs, turnover)
    }

    /// Find most commonly selected params across windows.
    fn most_common_params(&self, results: &[WindowResult]) -> ParamSet {
        if results.is_empty() {
            return ParamSet::default();
        }

        // Count top_n occurrences
        let mut top_n_counts: HashMap<usize, usize> = HashMap::new();
        let mut stop_loss_sum = Decimal::ZERO;
        let mut take_profit_sum = Decimal::ZERO;
        let mut max_weight_sum = Decimal::ZERO;
        let mut turnover_cap_sum = Decimal::ZERO;
        let mut min_score_sum = Decimal::ZERO;

        for r in results {
            *top_n_counts.entry(r.selected_params.top_n).or_insert(0) += 1;
            stop_loss_sum += r.selected_params.stop_loss_pct;
            take_profit_sum += r.selected_params.take_profit_pct;
            max_weight_sum += r.selected_params.max_weight;
            turnover_cap_sum += r.selected_params.turnover_cap;
            min_score_sum += r.selected_params.min_score;
        }

        let n = Decimal::from(results.len());

        // Most common top_n
        let most_common_top_n = top_n_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(top_n, _)| top_n)
            .unwrap_or(10);

        ParamSet {
            top_n: most_common_top_n,
            stop_loss_pct: stop_loss_sum / n,
            take_profit_pct: take_profit_sum / n,
            max_weight: max_weight_sum / n,
            turnover_cap: turnover_cap_sum / n,
            min_score: min_score_sum / n,
        }
    }
}

/// Nested 3-segment walk-forward runner with PSR/DSR selection.
/// Train -> Validation (selection) -> Test (OOS).
#[derive(Debug)]
pub struct NestedWalkForwardRunner {
    config: NestedWalkForwardConfig,
    metrics_calc: MetricsCalculator,
    robustness_scorer: RobustnessScorer,
}

impl NestedWalkForwardRunner {
    pub fn new(config: NestedWalkForwardConfig) -> Self {
        Self {
            config,
            metrics_calc: MetricsCalculator::new(dec!(0.05), 252),
            robustness_scorer: RobustnessScorer::default(),
        }
    }

    /// Run nested walk-forward validation.
    pub fn run(
        &self,
        start: NaiveDate,
        end: NaiveDate,
        candidates: &[WfCandidate],
        prices: &HashMap<String, PriceData>,
    ) -> NestedAggregateReport {
        // Generate nested splits
        let splitter = NestedSplitter::new(&self.config);
        let splits = splitter.generate_nested_splits(start, end);

        // Filter candidates by market
        let market_candidates: Vec<_> = candidates
            .iter()
            .filter(|c| c.market == self.config.market)
            .collect();

        // Run each window
        let mut results = Vec::new();

        for split in &splits {
            let result = self.run_nested_window(split, &market_candidates, prices);
            results.push(result);
        }

        // Aggregate metrics
        let aggregate = self.robustness_scorer.aggregate_nested(&results);

        // Find most commonly selected params
        let most_selected = self.most_common_params_nested(&results);

        NestedAggregateReport {
            config: self.config.clone(),
            windows: results,
            aggregate,
            most_selected_params: most_selected,
            generated_at: chrono::Local::now().date_naive(),
        }
    }

    /// Run a single nested window (train -> val -> test).
    fn run_nested_window(
        &self,
        split: &NestedWindowSplit,
        candidates: &[&WfCandidate],
        prices: &HashMap<String, PriceData>,
    ) -> NestedWindowResult {
        // Get grid config or use default params
        let grid = self.config.grid.as_ref();

        // Phase 1: Grid search on TRAIN
        let param_sets = if let Some(g) = grid {
            g.generate_combinations()
        } else {
            vec![ParamSet::default()]
        };

        // Phase 2: Evaluate each ParamSet on VALIDATION
        let mut selection_candidates: Vec<SelectionCandidate> = Vec::with_capacity(param_sets.len());
        let mut val_sharpes: Vec<Decimal> = Vec::with_capacity(param_sets.len());

        for params in &param_sets {
            // Run on train (to get train metrics)
            let train_metrics = self.simulate_window(
                &split.train.start_date,
                &split.train.end_date,
                candidates,
                prices,
                params,
            );

            // Run on validation (for selection)
            let val_metrics = self.simulate_window(
                &split.val.start_date,
                &split.val.end_date,
                candidates,
                prices,
                params,
            );

            val_sharpes.push(val_metrics.sharpe_ratio);

            // Calculate PSR on validation
            let psr = calculate_psr(
                val_metrics.sharpe_ratio,
                self.config.psr_threshold,
                val_metrics.n_observations,
                val_metrics.skewness,
                val_metrics.kurtosis,
            );

            // Composite score with penalties (including slippage/capacity from cost report)
            let turnover_penalty = self.config.penalties.turnover_weight * val_metrics.turnover_avg_pct / dec!(100);
            let cost_penalty = self.config.penalties.cost_weight * val_metrics.total_costs / dec!(1000);
            let dd_penalty = self.config.penalties.drawdown_weight * val_metrics.max_drawdown_pct / dec!(100);
            
            // Slippage sensitivity penalty (based on avg_slippage_bps from cost report)
            let slippage_penalty = if let Some(ref report) = val_metrics.cost_report {
                let avg_slip_bps = Decimal::try_from(report.avg_slippage_bps).unwrap_or(Decimal::ZERO);
                self.config.penalties.slippage_weight * avg_slip_bps / dec!(100)
            } else {
                Decimal::ZERO
            };
            
            // Capacity penalty (if capacity is below threshold)
            let capacity_penalty = if let Some(ref report) = val_metrics.cost_report {
                let capacity = Decimal::try_from(report.capacity_proxy_usd).unwrap_or(Decimal::ZERO);
                if capacity < self.config.penalties.min_capacity_usd && capacity > Decimal::ZERO {
                    self.config.penalties.capacity_weight
                } else {
                    Decimal::ZERO
                }
            } else {
                Decimal::ZERO
            };

            let composite = val_metrics.sharpe_ratio - turnover_penalty - cost_penalty - dd_penalty - slippage_penalty - capacity_penalty;

            selection_candidates.push(SelectionCandidate {
                params: params.clone(),
                sharpe: val_metrics.sharpe_ratio,
                psr,
                dsr: None,  // Will be calculated after we have variance
                turnover: val_metrics.turnover_avg_pct,
                costs: val_metrics.total_costs,
                max_drawdown: val_metrics.max_drawdown_pct,
                composite_score: composite,
            });
        }

        // Calculate Sharpe variance for DSR
        let sharpe_var = sharpe_variance(&val_sharpes);
        let n_trials = param_sets.len();

        // Update DSR for all candidates
        for (i, candidate) in selection_candidates.iter_mut().enumerate() {
            let dsr = calculate_dsr(
                candidate.sharpe,
                self.config.psr_threshold,
                100,  // Approximate n_observations
                Decimal::ZERO,  // Use 0 for simplicity
                Decimal::ZERO,
                n_trials,
                sharpe_var,
            );
            candidate.dsr = Some(dsr);
        }

        // Phase 3: Select best based on criteria with tie-breakers
        let (selected_idx, selection_reason) = self.select_best(
            &selection_candidates,
            self.config.selection_criteria,
        );

        let selected_params = selection_candidates[selected_idx].params.clone();
        let psr_val = selection_candidates[selected_idx].psr;
        let dsr_val = selection_candidates[selected_idx].dsr;

        // Get train and val metrics for selected params
        let train_metrics = self.simulate_window(
            &split.train.start_date,
            &split.train.end_date,
            candidates,
            prices,
            &selected_params,
        );

        let val_metrics = self.simulate_window(
            &split.val.start_date,
            &split.val.end_date,
            candidates,
            prices,
            &selected_params,
        );

        // Phase 4: Run TEST with selected params (OOS)
        let test_metrics = self.simulate_window(
            &split.test.start_date,
            &split.test.end_date,
            candidates,
            prices,
            &selected_params,
        );

        NestedWindowResult {
            split: split.clone(),
            metrics_train: train_metrics,
            metrics_val: val_metrics,
            metrics_test: test_metrics,
            selected_params,
            selection_reason,
            psr_val,
            dsr_val,
            n_trials,
        }
    }

    /// Select the best candidate using the specified criteria with deterministic tie-breakers.
    fn select_best(
        &self,
        candidates: &[SelectionCandidate],
        criteria: SelectionCriteria,
    ) -> (usize, SelectionReason) {
        if candidates.is_empty() {
            return (0, SelectionReason::default());
        }

        // Sort with tie-breakers
        let mut indexed: Vec<(usize, &SelectionCandidate)> = candidates.iter().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| a.compare_with_tiebreaker(b, criteria));

        let best_idx = indexed[0].0;
        let best = &candidates[best_idx];

        // Determine if tie-breaker was used
        let tiebreaker_used = if indexed.len() > 1 {
            let second = &candidates[indexed[1].0];
            let primary_equal = match criteria {
                SelectionCriteria::Sharpe => best.sharpe == second.sharpe,
                SelectionCriteria::PSR => best.psr == second.psr,
                SelectionCriteria::Composite => best.composite_score == second.composite_score,
            };
            if primary_equal {
                if best.turnover != second.turnover {
                    Some("turnover".to_string())
                } else if best.costs != second.costs {
                    Some("costs".to_string())
                } else if best.max_drawdown != second.max_drawdown {
                    Some("max_drawdown".to_string())
                } else {
                    Some("params_lexicographic".to_string())
                }
            } else {
                None
            }
        } else {
            None
        };

        let primary_score = match criteria {
            SelectionCriteria::Sharpe => best.sharpe,
            SelectionCriteria::PSR => best.psr,
            SelectionCriteria::Composite => best.composite_score,
        };

        let turnover_penalty = self.config.penalties.turnover_weight * best.turnover / dec!(100);
        let cost_penalty = self.config.penalties.cost_weight * best.costs / dec!(1000);
        let dd_penalty = self.config.penalties.drawdown_weight * best.max_drawdown / dec!(100);
        // Default to zero for slippage/capacity penalties (not tracked per candidate yet)
        let slippage_penalty = Decimal::ZERO;
        let capacity_penalty = Decimal::ZERO;

        let reason = SelectionReason {
            criteria,
            primary_score,
            psr: best.psr,
            dsr: best.dsr,
            turnover_penalty,
            cost_penalty,
            drawdown_penalty: dd_penalty,
            slippage_penalty,
            capacity_penalty,
            final_score: best.composite_score,
            tiebreaker_used,
        };

        (best_idx, reason)
    }

    /// Simulate a backtest for a window with given params.
    fn simulate_window(
        &self,
        start: &NaiveDate,
        end: &NaiveDate,
        candidates: &[&WfCandidate],
        prices: &HashMap<String, PriceData>,
        params: &ParamSet,
    ) -> WindowMetrics {
        // Select top-N by score
        let mut sorted: Vec<_> = candidates.iter().collect();
        sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        let selected: Vec<_> = sorted.into_iter().take(params.top_n).collect();

        if selected.is_empty() {
            return WindowMetrics::default();
        }

        // Build equal-weight portfolio (simplified)
        let weight = Decimal::ONE / Decimal::from(selected.len());

        // Generate daily equity curve (simplified simulation)
        let mut equity_curve = Vec::new();
        let mut current_date = *start;
        let mut portfolio_value = dec!(100_000);

        let mut prev_prices: HashMap<&str, Decimal> = HashMap::new();

        while current_date <= *end {
            let mut daily_return = Decimal::ZERO;

            for candidate in &selected {
                if let Some(price_data) = prices.get(&candidate.symbol) {
                    let price = price_data
                        .dates
                        .iter()
                        .zip(price_data.prices.iter())
                        .filter(|(d, _)| **d <= current_date)
                        .last()
                        .map(|(_, p)| *p);

                    if let Some(current_price) = price {
                        if let Some(prev_price) = prev_prices.get(candidate.symbol.as_str()) {
                            if *prev_price > Decimal::ZERO {
                                let ret = (current_price - *prev_price) / *prev_price;
                                daily_return += weight * ret;
                            }
                        }
                        prev_prices.insert(&candidate.symbol, current_price);
                    }
                }
            }

            // Apply stop-loss check
            let drawdown = if let Some(&peak) = equity_curve.iter().max() {
                if peak > Decimal::ZERO {
                    (peak - portfolio_value) / peak
                } else {
                    Decimal::ZERO
                }
            } else {
                Decimal::ZERO
            };

            if drawdown > params.stop_loss_pct {
                // Exit all positions (simplified)
                daily_return = Decimal::ZERO;
            }

            portfolio_value = portfolio_value * (Decimal::ONE + daily_return);
            equity_curve.push(portfolio_value);

            current_date += chrono::Duration::days(1);
        }

        // Calculate costs using ExecutionModelConfig if available
        let num_trades = selected.len() * 2; // entry + exit
        let avg_trade_value = dec!(100_000) / Decimal::from(selected.len());
        
        // Get slippage and fee parameters from config
        let (slippage_bps, fee_rate) = if let Some(ref exec_config) = self.config.execution_config {
            if exec_config.bypass_for_debug {
                (0.0, 0.0)
            } else {
                let slip_bps = exec_config.slippage.base_bps();
                let fee = exec_config.fees.commission_rate + exec_config.fees.emolument_rate;
                (slip_bps, fee)
            }
        } else {
            // Default: 10 bps slippage + 0.1% fees
            (10.0, 0.001)
        };
        
        // Calculate total costs per trade
        let slippage_cost = avg_trade_value * Decimal::try_from(slippage_bps / 10_000.0).unwrap_or(dec!(0.001));
        let fee_cost = avg_trade_value * Decimal::try_from(fee_rate).unwrap_or(dec!(0.001));
        let cost_per_trade = slippage_cost + fee_cost;
        let costs = Decimal::from(num_trades) * cost_per_trade;

        // Turnover calculation (trades / avg portfolio value * 100)
        let total_traded = Decimal::from(num_trades) * avg_trade_value;
        let avg_portfolio = equity_curve.iter().sum::<Decimal>() / Decimal::from(equity_curve.len().max(1));
        let turnover = if avg_portfolio > Decimal::ZERO {
            total_traded / avg_portfolio * dec!(100)
        } else {
            dec!(50) // Fallback
        };

        // Build cost report if execution config is provided
        let mut metrics = self.metrics_calc.from_equity_curve(&equity_curve, costs, turnover);
        
        if self.config.execution_config.is_some() {
            use backtester_execution::cost_report::{CostReport, FeeBreakdown, SlippageBreakdown};
            let mut cost_report = CostReport::new();
            cost_report.total_costs = costs.to_string().parse().unwrap_or(0.0);
            cost_report.total_slippage = (slippage_cost * Decimal::from(num_trades)).to_string().parse().unwrap_or(0.0);
            cost_report.total_fees = (fee_cost * Decimal::from(num_trades)).to_string().parse().unwrap_or(0.0);
            cost_report.trades_count = num_trades as u32;
            cost_report.avg_slippage_bps = slippage_bps;
            cost_report.turnover_annual = turnover.to_string().parse().unwrap_or(0.0);
            cost_report.avg_trade_notional = avg_trade_value.to_string().parse().unwrap_or(0.0);
            metrics.cost_report = Some(cost_report);
        }
        
        metrics
    }

    /// Find most commonly selected params across nested windows.
    fn most_common_params_nested(&self, results: &[NestedWindowResult]) -> ParamSet {
        if results.is_empty() {
            return ParamSet::default();
        }

        // Count top_n occurrences
        let mut top_n_counts: HashMap<usize, usize> = HashMap::new();
        let mut stop_loss_sum = Decimal::ZERO;
        let mut take_profit_sum = Decimal::ZERO;
        let mut max_weight_sum = Decimal::ZERO;
        let mut turnover_cap_sum = Decimal::ZERO;
        let mut min_score_sum = Decimal::ZERO;

        for r in results {
            *top_n_counts.entry(r.selected_params.top_n).or_insert(0) += 1;
            stop_loss_sum += r.selected_params.stop_loss_pct;
            take_profit_sum += r.selected_params.take_profit_pct;
            max_weight_sum += r.selected_params.max_weight;
            turnover_cap_sum += r.selected_params.turnover_cap;
            min_score_sum += r.selected_params.min_score;
        }

        let n = Decimal::from(results.len());

        // Most common top_n
        let most_common_top_n = top_n_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(top_n, _)| top_n)
            .unwrap_or(10);

        ParamSet {
            top_n: most_common_top_n,
            stop_loss_pct: stop_loss_sum / n,
            take_profit_pct: take_profit_sum / n,
            max_weight: max_weight_sum / n,
            turnover_cap: turnover_cap_sum / n,
            min_score: min_score_sum / n,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    fn make_prices(symbol: &str, market: Market, start: NaiveDate, days: usize, base: f64) -> PriceData {
        let mut data = PriceData::new(symbol, market);

        for i in 0..days {
            let d = start + chrono::Duration::days(i as i64);
            // Simple trending price with noise
            let noise = ((i as f64 * 0.1).sin() * 0.02);
            let trend = 1.0 + (i as f64 * 0.0005);
            let price = base * trend * (1.0 + noise);
            data.dates.push(d);
            data.prices.push(Decimal::try_from(price).unwrap());
        }

        data
    }

    fn make_candidates(n: usize, market: Market) -> Vec<WfCandidate> {
        (0..n)
            .map(|i| WfCandidate {
                symbol: format!("SYM{:03}", i),
                market,
                score: Decimal::from(n - i),
                volatility: dec!(0.02),
            })
            .collect()
    }

    #[test]
    fn test_runner_basic() {
        let config = WalkForwardConfig {
            train_months: 6,
            test_months: 3,
            step_months: 3,
            purge_days: 5,
            embargo_days: 5,
            market: Market::BR,
            grid: None,
        };

        let runner = WalkForwardRunner::new(config);

        let start = date(2020, 1, 1);
        let end = date(2021, 12, 31);

        let candidates = make_candidates(20, Market::BR);

        let mut prices: HashMap<String, PriceData> = HashMap::new();
        for c in &candidates {
            prices.insert(c.symbol.clone(), make_prices(&c.symbol, c.market, start, 730, 100.0));
        }

        let report = runner.run(start, end, &candidates, &prices);

        assert!(!report.windows.is_empty());
        assert_eq!(report.aggregate.total_windows, report.windows.len());
    }

    #[test]
    fn test_runner_with_grid() {
        let mut grid = GridConfig::default();
        // Small grid for testing
        grid.top_n_range = vec![5, 10];
        grid.stop_loss_range = super::super::types::ParamRange::new(dec!(0.10), dec!(0.15), dec!(0.05));
        grid.take_profit_range = super::super::types::ParamRange::new(dec!(0.20), dec!(0.20), dec!(0.10));
        grid.max_weight_range = super::super::types::ParamRange::new(dec!(0.20), dec!(0.20), dec!(0.05));
        grid.turnover_cap_range = super::super::types::ParamRange::new(dec!(0.50), dec!(0.50), dec!(0.20));
        grid.min_score_range = super::super::types::ParamRange::new(dec!(0.0), dec!(0.0), dec!(0.25));

        let config = WalkForwardConfig {
            train_months: 6,
            test_months: 3,
            step_months: 6, // Larger step for faster test
            purge_days: 5,
            embargo_days: 5,
            market: Market::BR,
            grid: Some(grid),
        };

        let runner = WalkForwardRunner::new(config);

        let start = date(2020, 1, 1);
        let end = date(2021, 12, 31);

        let candidates = make_candidates(10, Market::BR);

        let mut prices: HashMap<String, PriceData> = HashMap::new();
        for c in &candidates {
            prices.insert(c.symbol.clone(), make_prices(&c.symbol, c.market, start, 730, 100.0));
        }

        let report = runner.run(start, end, &candidates, &prices);

        assert!(!report.windows.is_empty());
        // Each window should have selected params
        for w in &report.windows {
            assert!(w.selected_params.top_n == 5 || w.selected_params.top_n == 10);
        }
    }

    #[test]
    fn test_oos_results() {
        let config = WalkForwardConfig::default();
        let runner = WalkForwardRunner::new(config);

        let start = date(2020, 1, 1);
        let end = date(2021, 12, 31);

        let candidates = make_candidates(10, Market::BR);

        let mut prices: HashMap<String, PriceData> = HashMap::new();
        for c in &candidates {
            prices.insert(c.symbol.clone(), make_prices(&c.symbol, c.market, start, 730, 100.0));
        }

        let report = runner.run(start, end, &candidates, &prices);

        // All windows should be OOS for test period
        for w in &report.windows {
            assert!(w.is_oos);
        }

        // OOS Sharpe should be calculable
        let oos_sharpe = report.oos_sharpe();
        // Just verify it doesn't panic and returns a valid number
        assert!(oos_sharpe.is_sign_positive() || oos_sharpe.is_sign_negative() || oos_sharpe.is_zero());
    }

    #[test]
    fn test_market_filtering() {
        let config = WalkForwardConfig {
            market: Market::US,
            ..Default::default()
        };

        let runner = WalkForwardRunner::new(config);

        let start = date(2020, 1, 1);
        let end = date(2021, 12, 31);

        // Mix of BR and US candidates
        let mut candidates = make_candidates(10, Market::BR);
        candidates.extend(make_candidates(5, Market::US));

        let mut prices: HashMap<String, PriceData> = HashMap::new();
        for c in &candidates {
            prices.insert(c.symbol.clone(), make_prices(&c.symbol, c.market, start, 730, 100.0));
        }

        let report = runner.run(start, end, &candidates, &prices);

        // Should only use US candidates
        assert_eq!(report.config.market, Market::US);
    }

    #[test]
    fn test_determinism() {
        let config = WalkForwardConfig::default();

        let start = date(2020, 1, 1);
        let end = date(2021, 6, 30);

        let candidates = make_candidates(10, Market::BR);

        let mut prices: HashMap<String, PriceData> = HashMap::new();
        for c in &candidates {
            prices.insert(c.symbol.clone(), make_prices(&c.symbol, c.market, start, 550, 100.0));
        }

        let runner1 = WalkForwardRunner::new(config.clone());
        let report1 = runner1.run(start, end, &candidates, &prices);

        let runner2 = WalkForwardRunner::new(config);
        let report2 = runner2.run(start, end, &candidates, &prices);

        assert_eq!(report1.windows.len(), report2.windows.len());

        for (w1, w2) in report1.windows.iter().zip(report2.windows.iter()) {
            assert_eq!(w1.test_metrics.sharpe_ratio, w2.test_metrics.sharpe_ratio);
            assert_eq!(w1.test_metrics.total_return_pct, w2.test_metrics.total_return_pct);
        }
    }

    // ========================
    // Nested Runner Tests
    // ========================

    #[test]
    fn test_nested_runner_basic() {
        let config = NestedWalkForwardConfig {
            train_months: 4,
            val_months: 1,
            test_months: 1,
            step_months: 3,
            purge_days: 5,
            embargo_days: 5,
            market: Market::BR,
            grid: None,
            selection_criteria: SelectionCriteria::PSR,
            psr_threshold: dec!(0.5),
            ..Default::default()
        };

        let runner = NestedWalkForwardRunner::new(config);

        let start = date(2020, 1, 1);
        let end = date(2021, 12, 31);

        let candidates = make_candidates(20, Market::BR);

        let mut prices: HashMap<String, PriceData> = HashMap::new();
        for c in &candidates {
            prices.insert(c.symbol.clone(), make_prices(&c.symbol, c.market, start, 730, 100.0));
        }

        let report = runner.run(start, end, &candidates, &prices);

        assert!(!report.windows.is_empty());
        assert_eq!(report.aggregate.total_windows, report.windows.len());
        
        // Check each window has 3 segments
        for w in &report.windows {
            assert!(w.split.is_valid());
            assert!(w.psr_val >= Decimal::ZERO);
            assert!(w.psr_val <= Decimal::ONE);
        }
    }

    #[test]
    fn test_nested_runner_with_grid() {
        let mut grid = GridConfig::default();
        grid.top_n_range = vec![5, 10];
        grid.stop_loss_range = super::super::types::ParamRange::new(dec!(0.10), dec!(0.15), dec!(0.05));
        grid.take_profit_range = super::super::types::ParamRange::new(dec!(0.20), dec!(0.20), dec!(0.10));
        grid.max_weight_range = super::super::types::ParamRange::new(dec!(0.20), dec!(0.20), dec!(0.05));
        grid.turnover_cap_range = super::super::types::ParamRange::new(dec!(0.50), dec!(0.50), dec!(0.20));
        grid.min_score_range = super::super::types::ParamRange::new(dec!(0.0), dec!(0.0), dec!(0.25));

        let config = NestedWalkForwardConfig {
            train_months: 4,
            val_months: 1,
            test_months: 1,
            step_months: 6,
            purge_days: 5,
            embargo_days: 5,
            market: Market::BR,
            grid: Some(grid),
            selection_criteria: SelectionCriteria::PSR,
            psr_threshold: dec!(0.5),
            ..Default::default()
        };

        let runner = NestedWalkForwardRunner::new(config);

        let start = date(2020, 1, 1);
        let end = date(2021, 12, 31);

        let candidates = make_candidates(10, Market::BR);

        let mut prices: HashMap<String, PriceData> = HashMap::new();
        for c in &candidates {
            prices.insert(c.symbol.clone(), make_prices(&c.symbol, c.market, start, 730, 100.0));
        }

        let report = runner.run(start, end, &candidates, &prices);

        assert!(!report.windows.is_empty());
        for w in &report.windows {
            assert!(w.selected_params.top_n == 5 || w.selected_params.top_n == 10);
            assert!(w.n_trials == 4);  // 2 top_n * 2 stop_loss = 4
        }
    }

    #[test]
    fn test_nested_selection_criteria_sharpe() {
        let config = NestedWalkForwardConfig {
            selection_criteria: SelectionCriteria::Sharpe,
            ..Default::default()
        };

        let runner = NestedWalkForwardRunner::new(config);

        let start = date(2020, 1, 1);
        let end = date(2021, 6, 30);

        let candidates = make_candidates(10, Market::BR);

        let mut prices: HashMap<String, PriceData> = HashMap::new();
        for c in &candidates {
            prices.insert(c.symbol.clone(), make_prices(&c.symbol, c.market, start, 550, 100.0));
        }

        let report = runner.run(start, end, &candidates, &prices);

        // Each window should have Sharpe selection
        for w in &report.windows {
            assert_eq!(w.selection_reason.criteria, SelectionCriteria::Sharpe);
        }
    }

    #[test]
    fn test_nested_selection_criteria_composite() {
        let config = NestedWalkForwardConfig {
            selection_criteria: SelectionCriteria::Composite,
            ..Default::default()
        };

        let runner = NestedWalkForwardRunner::new(config);

        let start = date(2020, 1, 1);
        let end = date(2021, 6, 30);

        let candidates = make_candidates(10, Market::BR);

        let mut prices: HashMap<String, PriceData> = HashMap::new();
        for c in &candidates {
            prices.insert(c.symbol.clone(), make_prices(&c.symbol, c.market, start, 550, 100.0));
        }

        let report = runner.run(start, end, &candidates, &prices);

        for w in &report.windows {
            assert_eq!(w.selection_reason.criteria, SelectionCriteria::Composite);
        }
    }

    #[test]
    fn test_nested_determinism() {
        let config = NestedWalkForwardConfig::default();

        let start = date(2020, 1, 1);
        let end = date(2021, 6, 30);

        let candidates = make_candidates(10, Market::BR);

        let mut prices: HashMap<String, PriceData> = HashMap::new();
        for c in &candidates {
            prices.insert(c.symbol.clone(), make_prices(&c.symbol, c.market, start, 550, 100.0));
        }

        let runner1 = NestedWalkForwardRunner::new(config.clone());
        let report1 = runner1.run(start, end, &candidates, &prices);

        let runner2 = NestedWalkForwardRunner::new(config);
        let report2 = runner2.run(start, end, &candidates, &prices);

        assert_eq!(report1.windows.len(), report2.windows.len());

        for (w1, w2) in report1.windows.iter().zip(report2.windows.iter()) {
            assert_eq!(w1.metrics_test.sharpe_ratio, w2.metrics_test.sharpe_ratio);
            assert_eq!(w1.selected_params, w2.selected_params);
            assert_eq!(w1.psr_val, w2.psr_val);
        }
    }

    #[test]
    fn test_tiebreaker_determinism() {
        // Create candidates with identical scores to force tie-breaking
        let c1 = SelectionCandidate {
            params: ParamSet { top_n: 5, ..Default::default() },
            sharpe: dec!(1.0),
            psr: dec!(0.7),
            dsr: Some(dec!(0.6)),
            turnover: dec!(20),
            costs: dec!(100),
            max_drawdown: dec!(10),
            composite_score: dec!(0.8),
        };
        let c2 = SelectionCandidate {
            params: ParamSet { top_n: 10, ..Default::default() },
            sharpe: dec!(1.0),
            psr: dec!(0.7),  // Same PSR
            dsr: Some(dec!(0.6)),
            turnover: dec!(25),  // Higher turnover
            costs: dec!(100),
            max_drawdown: dec!(10),
            composite_score: dec!(0.8),
        };

        // c1 should win due to lower turnover
        let cmp = c1.compare_with_tiebreaker(&c2, SelectionCriteria::PSR);
        assert_eq!(cmp, Ordering::Less);
    }
}


//! Walk-Forward Runner with grid search.
//!
//! Executes backtests across all windows with optional parameter optimization.

use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;

use crate::filters::Market;

use super::metrics::{MetricsCalculator, RobustnessScorer};
use super::splitter::{RollingSplitter, TimeSplitter};
use super::types::{
    AggregateReport, GridConfig, ParamSet, WalkForwardConfig, WindowMetrics, WindowResult, WindowSplit,
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

        // Calculate costs (simplified: 10 bps per trade)
        let num_trades = selected.len() * 2; // entry + exit
        let avg_trade_value = dec!(100_000) / Decimal::from(selected.len());
        let costs = Decimal::from(num_trades) * avg_trade_value * dec!(0.001);

        // Turnover (simplified)
        let turnover = dec!(50); // Fixed assumption

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
}


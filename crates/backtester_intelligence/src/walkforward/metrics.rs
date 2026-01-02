//! Walk-Forward metrics calculation and robustness scoring.

use rust_decimal::Decimal;
use rust_decimal::MathematicalOps;
use rust_decimal_macros::dec;

use super::types::{AggregateMetrics, WindowMetrics, WindowResult, NestedWindowResult};
use super::statistics::{calculate_skewness, calculate_kurtosis, calculate_psr, calculate_dsr};

/// Calculates metrics from equity curve and trade data.
#[derive(Debug, Clone, Default)]
pub struct MetricsCalculator {
    /// Risk-free rate for Sharpe calculation (annualized).
    pub risk_free_rate: Decimal,
    /// Trading days per year for annualization.
    pub trading_days: u32,
}

impl MetricsCalculator {
    pub fn new(risk_free_rate: Decimal, trading_days: u32) -> Self {
        Self { risk_free_rate, trading_days }
    }

    /// Calculate metrics from an equity curve (daily values).
    pub fn from_equity_curve(&self, equity: &[Decimal], costs: Decimal, turnover_pct: Decimal) -> WindowMetrics {
        if equity.len() < 2 {
            return WindowMetrics::default();
        }

        let initial = equity[0];
        let final_val = *equity.last().unwrap();

        // Total return
        let total_return = if initial > Decimal::ZERO {
            (final_val - initial) / initial * dec!(100)
        } else {
            Decimal::ZERO
        };

        // Daily returns
        let returns: Vec<Decimal> = equity
            .windows(2)
            .filter_map(|w| {
                if w[0] > Decimal::ZERO {
                    Some((w[1] - w[0]) / w[0])
                } else {
                    None
                }
            })
            .collect();

        // Volatility (daily, then annualized)
        let vol_daily = self.std_dev(&returns);
        let ann_factor = Decimal::from(self.trading_days).sqrt().unwrap_or(dec!(15.87));
        let volatility_ann = vol_daily * ann_factor * dec!(100);

        // Mean daily return
        let mean_return = if returns.is_empty() {
            Decimal::ZERO
        } else {
            returns.iter().sum::<Decimal>() / Decimal::from(returns.len())
        };

        // Sharpe ratio (annualized, clamped to [-10, 10])
        let rf_daily = self.risk_free_rate / Decimal::from(self.trading_days);
        let excess_return = mean_return - rf_daily;
        let sharpe_raw = if vol_daily > Decimal::ZERO {
            excess_return / vol_daily * ann_factor
        } else {
            Decimal::ZERO
        };
        // Clamp to realistic bounds - any value beyond [-10, 10] indicates
        // calculation error or unrealistic data (low volatility, short period)
        let sharpe = sharpe_raw.max(dec!(-10)).min(dec!(10));

        // CAGR
        let days = equity.len() as f64;
        let years = days / self.trading_days as f64;
        let cagr = if initial > Decimal::ZERO && years > 0.0 {
            let ratio = final_val / initial;
            let ratio_f64: f64 = ratio.to_string().parse().unwrap_or(1.0);
            let cagr_f64 = ratio_f64.powf(1.0 / years) - 1.0;
            Decimal::try_from(cagr_f64 * 100.0).unwrap_or(Decimal::ZERO)
        } else {
            Decimal::ZERO
        };

        // Max drawdown
        let (max_dd, dd_duration) = self.max_drawdown(equity);

        // Calculate skewness and kurtosis
        let skewness = calculate_skewness(&returns);
        let kurtosis = calculate_kurtosis(&returns);

        WindowMetrics {
            total_return_pct: total_return,
            cagr_pct: cagr,
            volatility_ann,
            sharpe_ratio: sharpe,
            max_drawdown_pct: max_dd * dec!(100),
            dd_duration_days: dd_duration,
            turnover_avg_pct: turnover_pct,
            total_costs: costs,
            hit_rate: None,
            skewness,
            kurtosis,
            n_observations: returns.len(),
            psr: None,  // Calculated separately with threshold
            dsr: None,  // Calculated separately with trials info
            cost_report: None,
        }
    }

    /// Calculate metrics with PSR/DSR included.
    pub fn from_equity_curve_with_psr(
        &self,
        equity: &[Decimal],
        costs: Decimal,
        turnover_pct: Decimal,
        psr_threshold: Decimal,
        n_trials: Option<usize>,
        sharpe_var: Option<Decimal>,
    ) -> WindowMetrics {
        let mut metrics = self.from_equity_curve(equity, costs, turnover_pct);
        
        // Calculate PSR
        let psr = calculate_psr(
            metrics.sharpe_ratio,
            psr_threshold,
            metrics.n_observations,
            metrics.skewness,
            metrics.kurtosis,
        );
        metrics.psr = Some(psr);

        // Calculate DSR if trial info is available
        if let (Some(trials), Some(var)) = (n_trials, sharpe_var) {
            let dsr = calculate_dsr(
                metrics.sharpe_ratio,
                psr_threshold,
                metrics.n_observations,
                metrics.skewness,
                metrics.kurtosis,
                trials,
                var,
            );
            metrics.dsr = Some(dsr);
        }

        metrics
    }

    /// Calculate standard deviation of a series.
    fn std_dev(&self, values: &[Decimal]) -> Decimal {
        if values.len() < 2 {
            return Decimal::ZERO;
        }

        let n = Decimal::from(values.len());
        let mean = values.iter().sum::<Decimal>() / n;
        let variance = values.iter()
            .map(|v| (*v - mean) * (*v - mean))
            .sum::<Decimal>() / n;

        variance.sqrt().unwrap_or(Decimal::ZERO)
    }

    /// Calculate max drawdown and its duration in days.
    fn max_drawdown(&self, equity: &[Decimal]) -> (Decimal, u32) {
        if equity.is_empty() {
            return (Decimal::ZERO, 0);
        }

        let mut max_dd = Decimal::ZERO;
        let mut max_duration = 0u32;
        let mut peak = equity[0];
        let mut peak_idx = 0usize;

        for (i, &val) in equity.iter().enumerate() {
            if val > peak {
                peak = val;
                peak_idx = i;
            }

            if peak > Decimal::ZERO {
                let dd = (peak - val) / peak;
                if dd > max_dd {
                    max_dd = dd;
                    max_duration = (i - peak_idx) as u32;
                }
            }
        }

        (max_dd, max_duration)
    }
}

/// Calculates robustness score from aggregate window results.
#[derive(Debug, Clone)]
pub struct RobustnessScorer {
    /// Weight for Sharpe stability in robustness score.
    pub sharpe_weight: Decimal,
    /// Weight for drawdown consistency in robustness score.
    pub drawdown_weight: Decimal,
    /// Weight for return stability in robustness score.
    pub return_weight: Decimal,
}

impl Default for RobustnessScorer {
    fn default() -> Self {
        Self {
            sharpe_weight: dec!(0.40),
            drawdown_weight: dec!(0.35),
            return_weight: dec!(0.25),
        }
    }
}

impl RobustnessScorer {
    /// Calculate aggregate metrics from window results.
    pub fn aggregate(&self, results: &[WindowResult]) -> AggregateMetrics {
        if results.is_empty() {
            return AggregateMetrics::default();
        }

        let n = Decimal::from(results.len());

        // Extract test metrics (OOS)
        let sharpes: Vec<Decimal> = results.iter().map(|r| r.test_metrics.sharpe_ratio).collect();
        let returns: Vec<Decimal> = results.iter().map(|r| r.test_metrics.total_return_pct).collect();
        let drawdowns: Vec<Decimal> = results.iter().map(|r| r.test_metrics.max_drawdown_pct).collect();
        let volatilities: Vec<Decimal> = results.iter().map(|r| r.test_metrics.volatility_ann).collect();

        // Mean/median/std for Sharpe
        let mean_sharpe = sharpes.iter().sum::<Decimal>() / n;
        let median_sharpe = self.median(&sharpes);
        let std_sharpe = self.std_dev(&sharpes);

        // Mean/median/std for returns
        let mean_return = returns.iter().sum::<Decimal>() / n;
        let median_return = self.median(&returns);
        let std_return = self.std_dev(&returns);

        // Risk stats
        let mean_drawdown = drawdowns.iter().sum::<Decimal>() / n;
        let worst_drawdown = drawdowns.iter().max().cloned().unwrap_or(Decimal::ZERO);
        let mean_volatility = volatilities.iter().sum::<Decimal>() / n;

        // Best/worst window by Sharpe
        let (best_idx, _) = sharpes.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap_or((0, &Decimal::ZERO));
        let (worst_idx, _) = sharpes.iter().enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap_or((0, &Decimal::ZERO));

        // Stability score = 1 - (std/mean) for positive mean
        let stability_score = if mean_sharpe > Decimal::ZERO {
            (Decimal::ONE - std_sharpe / mean_sharpe).max(Decimal::ZERO)
        } else {
            Decimal::ZERO
        };

        // Robustness score (weighted combination)
        let sharpe_component = if std_sharpe > Decimal::ZERO {
            (mean_sharpe / (std_sharpe + dec!(0.01))).min(dec!(2))
        } else {
            mean_sharpe.min(dec!(2))
        };

        let dd_std = self.std_dev(&drawdowns);
        let dd_component = if mean_drawdown > Decimal::ZERO {
            (dec!(1) - dd_std / mean_drawdown).max(Decimal::ZERO)
        } else {
            dec!(1)
        };

        let return_component = if mean_return > Decimal::ZERO && std_return > Decimal::ZERO {
            (mean_return / std_return / dec!(3)).min(dec!(1))
        } else {
            Decimal::ZERO
        };

        let robustness_score = self.sharpe_weight * sharpe_component
            + self.drawdown_weight * dd_component
            + self.return_weight * return_component;

        // Total months tested
        let total_months = results.iter()
            .map(|r| r.split.test.days() / 30)
            .sum::<i64>() as u32;

        AggregateMetrics {
            mean_sharpe,
            median_sharpe,
            std_sharpe,
            mean_return,
            median_return,
            std_return,
            mean_drawdown,
            worst_drawdown,
            mean_volatility,
            stability_score,
            robustness_score,
            best_window_idx: best_idx,
            worst_window_idx: worst_idx,
            total_windows: results.len(),
            total_months_tested: total_months,
            // PSR/DSR not available for legacy results
            mean_psr: Decimal::ZERO,
            median_psr: Decimal::ZERO,
            mean_dsr: None,
            median_dsr: None,
            oos_sharpe_mean: mean_sharpe,
            oos_return_mean: mean_return,
            oos_psr_mean: Decimal::ZERO,
        }
    }

    /// Calculate aggregate metrics from nested window results (3-segment).
    pub fn aggregate_nested(&self, results: &[NestedWindowResult]) -> AggregateMetrics {
        if results.is_empty() {
            return AggregateMetrics::default();
        }

        let n = Decimal::from(results.len());

        // Extract test metrics (OOS)
        let sharpes: Vec<Decimal> = results.iter().map(|r| r.metrics_test.sharpe_ratio).collect();
        let returns: Vec<Decimal> = results.iter().map(|r| r.metrics_test.total_return_pct).collect();
        let drawdowns: Vec<Decimal> = results.iter().map(|r| r.metrics_test.max_drawdown_pct).collect();
        let volatilities: Vec<Decimal> = results.iter().map(|r| r.metrics_test.volatility_ann).collect();

        // Extract PSR from validation
        let psrs: Vec<Decimal> = results.iter().map(|r| r.psr_val).collect();
        let dsrs: Vec<Decimal> = results.iter().filter_map(|r| r.dsr_val).collect();

        // Mean/median/std for Sharpe
        let mean_sharpe = sharpes.iter().sum::<Decimal>() / n;
        let median_sharpe = self.median(&sharpes);
        let std_sharpe = self.std_dev(&sharpes);

        // Mean/median/std for returns
        let mean_return = returns.iter().sum::<Decimal>() / n;
        let median_return = self.median(&returns);
        let std_return = self.std_dev(&returns);

        // Risk stats
        let mean_drawdown = drawdowns.iter().sum::<Decimal>() / n;
        let worst_drawdown = drawdowns.iter().max().cloned().unwrap_or(Decimal::ZERO);
        let mean_volatility = volatilities.iter().sum::<Decimal>() / n;

        // PSR stats
        let mean_psr = psrs.iter().sum::<Decimal>() / n;
        let median_psr = self.median(&psrs);

        // DSR stats (if available)
        let (mean_dsr, median_dsr) = if dsrs.is_empty() {
            (None, None)
        } else {
            let n_dsr = Decimal::from(dsrs.len());
            (
                Some(dsrs.iter().sum::<Decimal>() / n_dsr),
                Some(self.median(&dsrs)),
            )
        };

        // Best/worst window by Sharpe
        let (best_idx, _) = sharpes.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap_or((0, &Decimal::ZERO));
        let (worst_idx, _) = sharpes.iter().enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap_or((0, &Decimal::ZERO));

        // Stability score = 1 - (std/mean) for positive mean
        let stability_score = if mean_sharpe > Decimal::ZERO {
            (Decimal::ONE - std_sharpe / mean_sharpe).max(Decimal::ZERO)
        } else {
            Decimal::ZERO
        };

        // Robustness score (weighted combination)
        let sharpe_component = if std_sharpe > Decimal::ZERO {
            (mean_sharpe / (std_sharpe + dec!(0.01))).min(dec!(2))
        } else {
            mean_sharpe.min(dec!(2))
        };

        let dd_std = self.std_dev(&drawdowns);
        let dd_component = if mean_drawdown > Decimal::ZERO {
            (dec!(1) - dd_std / mean_drawdown).max(Decimal::ZERO)
        } else {
            dec!(1)
        };

        let return_component = if mean_return > Decimal::ZERO && std_return > Decimal::ZERO {
            (mean_return / std_return / dec!(3)).min(dec!(1))
        } else {
            Decimal::ZERO
        };

        let robustness_score = self.sharpe_weight * sharpe_component
            + self.drawdown_weight * dd_component
            + self.return_weight * return_component;

        // Total months tested
        let total_months = results.iter()
            .map(|r| r.split.test.days() / 30)
            .sum::<i64>() as u32;

        AggregateMetrics {
            mean_sharpe,
            median_sharpe,
            std_sharpe,
            mean_return,
            median_return,
            std_return,
            mean_drawdown,
            worst_drawdown,
            mean_volatility,
            stability_score,
            robustness_score,
            best_window_idx: best_idx,
            worst_window_idx: worst_idx,
            total_windows: results.len(),
            total_months_tested: total_months,
            mean_psr,
            median_psr,
            mean_dsr,
            median_dsr,
            oos_sharpe_mean: mean_sharpe,
            oos_return_mean: mean_return,
            oos_psr_mean: mean_psr,
        }
    }

    fn std_dev(&self, values: &[Decimal]) -> Decimal {
        if values.len() < 2 {
            return Decimal::ZERO;
        }

        let n = Decimal::from(values.len());
        let mean = values.iter().sum::<Decimal>() / n;
        let variance = values.iter()
            .map(|v| (*v - mean) * (*v - mean))
            .sum::<Decimal>() / n;

        variance.sqrt().unwrap_or(Decimal::ZERO)
    }

    fn median(&self, values: &[Decimal]) -> Decimal {
        if values.is_empty() {
            return Decimal::ZERO;
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / dec!(2)
        } else {
            sorted[mid]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dec_vec(vals: &[f64]) -> Vec<Decimal> {
        vals.iter().map(|v| Decimal::try_from(*v).unwrap()).collect()
    }

    #[test]
    fn test_metrics_from_equity_positive() {
        let calc = MetricsCalculator::new(dec!(0.05), 252);
        // Simple growing equity curve
        let equity = dec_vec(&[100.0, 101.0, 102.0, 103.0, 104.0, 105.0]);

        let metrics = calc.from_equity_curve(&equity, dec!(10), dec!(25));

        assert!(metrics.total_return_pct > Decimal::ZERO);
        assert!(metrics.sharpe_ratio > Decimal::ZERO);
        assert_eq!(metrics.max_drawdown_pct, Decimal::ZERO); // No drawdown in upward curve
        assert!(metrics.n_observations > 0);
        assert!(metrics.skewness.abs() < dec!(10));  // Reasonable skewness
    }

    #[test]
    fn test_metrics_with_psr() {
        let calc = MetricsCalculator::new(dec!(0.05), 252);
        let equity: Vec<Decimal> = (0..100)
            .map(|i| dec!(100) + Decimal::from(i) * dec!(0.1))
            .collect();

        let metrics = calc.from_equity_curve_with_psr(
            &equity,
            dec!(5),
            dec!(20),
            dec!(0.5),  // threshold
            Some(10),   // n_trials
            Some(dec!(0.25)),  // sharpe variance
        );

        assert!(metrics.psr.is_some());
        assert!(metrics.dsr.is_some());
        assert!(metrics.psr.unwrap() >= Decimal::ZERO);
        assert!(metrics.psr.unwrap() <= Decimal::ONE);
    }

    #[test]
    fn test_metrics_from_equity_with_drawdown() {
        let calc = MetricsCalculator::new(dec!(0.05), 252);
        // Equity curve with drawdown
        let equity = dec_vec(&[100.0, 105.0, 110.0, 100.0, 95.0, 100.0, 108.0]);

        let metrics = calc.from_equity_curve(&equity, dec!(5), dec!(30));

        assert!(metrics.max_drawdown_pct > Decimal::ZERO);
        // Max drawdown: (110-95)/110 = 13.6%
        assert!(metrics.max_drawdown_pct > dec!(13));
    }

    #[test]
    fn test_robustness_scorer_basic() {
        use chrono::NaiveDate;
        use super::super::types::{WindowSplit, WindowSpec, WindowType, ParamSet};

        let scorer = RobustnessScorer::default();

        let make_result = |sharpe: f64, ret: f64, dd: f64, idx: usize| -> WindowResult {
            WindowResult {
                split: WindowSplit {
                    train: WindowSpec::new(
                        NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(),
                        NaiveDate::from_ymd_opt(2020, 6, 30).unwrap(),
                        WindowType::Train,
                        idx,
                    ),
                    test: WindowSpec::new(
                        NaiveDate::from_ymd_opt(2020, 7, 1).unwrap(),
                        NaiveDate::from_ymd_opt(2020, 9, 30).unwrap(),
                        WindowType::Test,
                        idx,
                    ),
                    purge_days: 5,
                    embargo_days: 5,
                    index: idx,
                },
                train_metrics: WindowMetrics::default(),
                test_metrics: WindowMetrics {
                    sharpe_ratio: Decimal::try_from(sharpe).unwrap(),
                    total_return_pct: Decimal::try_from(ret).unwrap(),
                    max_drawdown_pct: Decimal::try_from(dd).unwrap(),
                    ..Default::default()
                },
                selected_params: ParamSet::default(),
                is_oos: true,
            }
        };

        let results = vec![
            make_result(1.2, 5.0, 8.0, 0),
            make_result(1.0, 4.0, 10.0, 1),
            make_result(0.8, 3.0, 12.0, 2),
            make_result(1.1, 6.0, 7.0, 3),
        ];

        let agg = scorer.aggregate(&results);

        assert!(agg.mean_sharpe > dec!(0.9));
        assert!(agg.mean_sharpe < dec!(1.2));
        assert_eq!(agg.total_windows, 4);
        assert!(agg.robustness_score > Decimal::ZERO);
    }

    #[test]
    fn test_median_calculation() {
        let scorer = RobustnessScorer::default();

        let odd = vec![dec!(1), dec!(3), dec!(2)];
        assert_eq!(scorer.median(&odd), dec!(2));

        let even = vec![dec!(1), dec!(2), dec!(3), dec!(4)];
        assert_eq!(scorer.median(&even), dec!(2.5));
    }

    #[test]
    fn test_stability_penalizes_variance() {
        use chrono::NaiveDate;
        use super::super::types::{WindowSplit, WindowSpec, WindowType, ParamSet};

        let scorer = RobustnessScorer::default();

        let make_result = |sharpe: f64, idx: usize| -> WindowResult {
            WindowResult {
                split: WindowSplit {
                    train: WindowSpec::new(
                        NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(),
                        NaiveDate::from_ymd_opt(2020, 6, 30).unwrap(),
                        WindowType::Train,
                        idx,
                    ),
                    test: WindowSpec::new(
                        NaiveDate::from_ymd_opt(2020, 7, 1).unwrap(),
                        NaiveDate::from_ymd_opt(2020, 9, 30).unwrap(),
                        WindowType::Test,
                        idx,
                    ),
                    purge_days: 5,
                    embargo_days: 5,
                    index: idx,
                },
                train_metrics: WindowMetrics::default(),
                test_metrics: WindowMetrics {
                    sharpe_ratio: Decimal::try_from(sharpe).unwrap(),
                    total_return_pct: dec!(5),
                    max_drawdown_pct: dec!(10),
                    ..Default::default()
                },
                selected_params: ParamSet::default(),
                is_oos: true,
            }
        };

        // Stable results (low variance)
        let stable = vec![
            make_result(1.0, 0),
            make_result(1.1, 1),
            make_result(0.9, 2),
            make_result(1.0, 3),
        ];
        let agg_stable = scorer.aggregate(&stable);

        // Unstable results (high variance)
        let unstable = vec![
            make_result(2.0, 0),
            make_result(-0.5, 1),
            make_result(1.5, 2),
            make_result(0.0, 3),
        ];
        let agg_unstable = scorer.aggregate(&unstable);

        assert!(
            agg_stable.stability_score > agg_unstable.stability_score,
            "Stable {} should be > unstable {}",
            agg_stable.stability_score,
            agg_unstable.stability_score
        );
    }
}


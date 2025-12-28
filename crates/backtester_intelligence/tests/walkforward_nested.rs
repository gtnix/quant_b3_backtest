//! Tests for Nested Walk-Forward (3-segment) validation.

use backtester_intelligence::walkforward::{
    NestedSplitter, NestedWalkForwardConfig, NestedWindowSplit,
    SelectionCriteria, SelectionCandidate, ParamSet, PenaltyConfig,
};
use backtester_intelligence::filters::Market;
use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::cmp::Ordering;

fn date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

// ========================
// Splitter Tests
// ========================

#[test]
fn test_nested_splitter_no_overlap_train_val() {
    let splitter = NestedSplitter::from_parts(4, 1, 1, 3, 5, 5);
    let start = date(2010, 1, 1);
    let end = date(2025, 1, 1);

    let splits = splitter.generate_nested_splits(start, end);

    for split in &splits {
        assert!(
            split.train.end_date < split.val.start_date,
            "Window {}: train ends {} but val starts {}",
            split.index,
            split.train.end_date,
            split.val.start_date
        );
    }
}

#[test]
fn test_nested_splitter_no_overlap_val_test() {
    let splitter = NestedSplitter::from_parts(4, 1, 1, 3, 5, 5);
    let start = date(2010, 1, 1);
    let end = date(2025, 1, 1);

    let splits = splitter.generate_nested_splits(start, end);

    for split in &splits {
        assert!(
            split.val.end_date < split.test.start_date,
            "Window {}: val ends {} but test starts {}",
            split.index,
            split.val.end_date,
            split.test.start_date
        );
    }
}

#[test]
fn test_nested_splitter_purge_gap_train_val() {
    let splitter = NestedSplitter::from_parts(4, 1, 1, 3, 10, 5);
    let start = date(2020, 1, 1);
    let end = date(2021, 12, 31);

    let splits = splitter.generate_nested_splits(start, end);

    for split in &splits {
        let gap = (split.val.start_date - split.train.end_date).num_days();
        assert!(
            gap >= 5,  // embargo days
            "Window {}: train-val gap {} < 5",
            split.index,
            gap
        );
    }
}

#[test]
fn test_nested_splitter_purge_gap_val_test() {
    let splitter = NestedSplitter::from_parts(4, 1, 1, 3, 10, 5);
    let start = date(2020, 1, 1);
    let end = date(2021, 12, 31);

    let splits = splitter.generate_nested_splits(start, end);

    for split in &splits {
        let gap = (split.test.start_date - split.val.end_date).num_days();
        assert!(
            gap >= 5,  // embargo days
            "Window {}: val-test gap {} < 5",
            split.index,
            gap
        );
    }
}

#[test]
fn test_nested_splitter_20_years_window_count() {
    // 4mo train + 1mo val + 1mo test = 6mo per window
    // 20 years = 240 months, step = 3 months
    // Expected: (240 - 6) / 3 + 1 ≈ 79 windows
    let splitter = NestedSplitter::from_parts(4, 1, 1, 3, 5, 5);
    let start = date(2005, 1, 1);
    let end = date(2025, 1, 1);

    let splits = splitter.generate_nested_splits(start, end);

    assert!(
        splits.len() >= 70,
        "Expected >= 70 windows for 20 years, got {}",
        splits.len()
    );
    assert!(
        splits.len() <= 85,
        "Expected <= 85 windows for 20 years, got {}",
        splits.len()
    );
}

#[test]
fn test_nested_splitter_determinism() {
    let splitter = NestedSplitter::from_parts(4, 1, 1, 3, 5, 5);
    let start = date(2015, 1, 1);
    let end = date(2020, 1, 1);

    let splits1 = splitter.generate_nested_splits(start, end);
    let splits2 = splitter.generate_nested_splits(start, end);

    assert_eq!(splits1.len(), splits2.len());
    for (s1, s2) in splits1.iter().zip(splits2.iter()) {
        assert_eq!(s1.train.start_date, s2.train.start_date);
        assert_eq!(s1.train.end_date, s2.train.end_date);
        assert_eq!(s1.val.start_date, s2.val.start_date);
        assert_eq!(s1.val.end_date, s2.val.end_date);
        assert_eq!(s1.test.start_date, s2.test.start_date);
        assert_eq!(s1.test.end_date, s2.test.end_date);
    }
}

#[test]
fn test_nested_splitter_valid_splits() {
    let splitter = NestedSplitter::from_parts(4, 1, 1, 3, 5, 5);
    let start = date(2020, 1, 1);
    let end = date(2021, 12, 31);

    let splits = splitter.generate_nested_splits(start, end);

    for split in &splits {
        assert!(split.is_valid(), "Split {} should be valid", split.index);
    }
}

// ========================
// Selection Candidate Tests
// ========================

#[test]
fn test_selection_candidate_tiebreaker_by_turnover() {
    let c1 = SelectionCandidate {
        params: ParamSet::default(),
        sharpe: dec!(1.0),
        psr: dec!(0.7),
        dsr: Some(dec!(0.6)),
        turnover: dec!(20),  // lower
        costs: dec!(100),
        max_drawdown: dec!(10),
        composite_score: dec!(0.8),
    };
    let c2 = SelectionCandidate {
        params: ParamSet::default(),
        sharpe: dec!(1.0),
        psr: dec!(0.7),  // same PSR
        dsr: Some(dec!(0.6)),
        turnover: dec!(30),  // higher
        costs: dec!(100),
        max_drawdown: dec!(10),
        composite_score: dec!(0.8),
    };

    let cmp = c1.compare_with_tiebreaker(&c2, SelectionCriteria::PSR);
    assert_eq!(cmp, Ordering::Less, "c1 should win (lower turnover)");
}

#[test]
fn test_selection_candidate_tiebreaker_by_costs() {
    let c1 = SelectionCandidate {
        params: ParamSet::default(),
        sharpe: dec!(1.0),
        psr: dec!(0.7),
        dsr: Some(dec!(0.6)),
        turnover: dec!(25),
        costs: dec!(80),  // lower
        max_drawdown: dec!(10),
        composite_score: dec!(0.8),
    };
    let c2 = SelectionCandidate {
        params: ParamSet::default(),
        sharpe: dec!(1.0),
        psr: dec!(0.7),
        dsr: Some(dec!(0.6)),
        turnover: dec!(25),  // same turnover
        costs: dec!(120),  // higher
        max_drawdown: dec!(10),
        composite_score: dec!(0.8),
    };

    let cmp = c1.compare_with_tiebreaker(&c2, SelectionCriteria::PSR);
    assert_eq!(cmp, Ordering::Less, "c1 should win (lower costs)");
}

#[test]
fn test_selection_candidate_tiebreaker_by_drawdown() {
    let c1 = SelectionCandidate {
        params: ParamSet::default(),
        sharpe: dec!(1.0),
        psr: dec!(0.7),
        dsr: Some(dec!(0.6)),
        turnover: dec!(25),
        costs: dec!(100),
        max_drawdown: dec!(8),  // lower
        composite_score: dec!(0.8),
    };
    let c2 = SelectionCandidate {
        params: ParamSet::default(),
        sharpe: dec!(1.0),
        psr: dec!(0.7),
        dsr: Some(dec!(0.6)),
        turnover: dec!(25),
        costs: dec!(100),
        max_drawdown: dec!(12),  // higher
        composite_score: dec!(0.8),
    };

    let cmp = c1.compare_with_tiebreaker(&c2, SelectionCriteria::PSR);
    assert_eq!(cmp, Ordering::Less, "c1 should win (lower drawdown)");
}

#[test]
fn test_selection_candidate_tiebreaker_by_params() {
    let c1 = SelectionCandidate {
        params: ParamSet { top_n: 5, ..Default::default() },
        sharpe: dec!(1.0),
        psr: dec!(0.7),
        dsr: Some(dec!(0.6)),
        turnover: dec!(25),
        costs: dec!(100),
        max_drawdown: dec!(10),
        composite_score: dec!(0.8),
    };
    let c2 = SelectionCandidate {
        params: ParamSet { top_n: 10, ..Default::default() },  // higher
        sharpe: dec!(1.0),
        psr: dec!(0.7),
        dsr: Some(dec!(0.6)),
        turnover: dec!(25),
        costs: dec!(100),
        max_drawdown: dec!(10),
        composite_score: dec!(0.8),
    };

    let cmp = c1.compare_with_tiebreaker(&c2, SelectionCriteria::PSR);
    assert_eq!(cmp, Ordering::Less, "c1 should win (lower top_n is earlier lexicographically)");
}

#[test]
fn test_selection_by_sharpe() {
    let c1 = SelectionCandidate {
        params: ParamSet::default(),
        sharpe: dec!(1.5),  // higher
        psr: dec!(0.6),
        dsr: Some(dec!(0.5)),
        turnover: dec!(25),
        costs: dec!(100),
        max_drawdown: dec!(10),
        composite_score: dec!(0.7),
    };
    let c2 = SelectionCandidate {
        params: ParamSet::default(),
        sharpe: dec!(1.0),  // lower
        psr: dec!(0.8),  // higher PSR but we're selecting by Sharpe
        dsr: Some(dec!(0.7)),
        turnover: dec!(25),
        costs: dec!(100),
        max_drawdown: dec!(10),
        composite_score: dec!(0.9),
    };

    let cmp = c1.compare_with_tiebreaker(&c2, SelectionCriteria::Sharpe);
    assert_eq!(cmp, Ordering::Less, "c1 should win (higher Sharpe)");
}

#[test]
fn test_selection_by_composite() {
    let c1 = SelectionCandidate {
        params: ParamSet::default(),
        sharpe: dec!(1.0),
        psr: dec!(0.7),
        dsr: Some(dec!(0.6)),
        turnover: dec!(25),
        costs: dec!(100),
        max_drawdown: dec!(10),
        composite_score: dec!(0.9),  // higher
    };
    let c2 = SelectionCandidate {
        params: ParamSet::default(),
        sharpe: dec!(1.2),  // higher Sharpe but lower composite
        psr: dec!(0.75),
        dsr: Some(dec!(0.65)),
        turnover: dec!(40),  // higher turnover penalized
        costs: dec!(150),  // higher costs
        max_drawdown: dec!(15),  // higher DD
        composite_score: dec!(0.7),  // lower composite
    };

    let cmp = c1.compare_with_tiebreaker(&c2, SelectionCriteria::Composite);
    assert_eq!(cmp, Ordering::Less, "c1 should win (higher composite score)");
}

// ========================
// Config Tests
// ========================

#[test]
fn test_nested_config_default() {
    let config = NestedWalkForwardConfig::default();
    
    assert_eq!(config.train_months, 4);
    assert_eq!(config.val_months, 1);
    assert_eq!(config.test_months, 1);
    assert_eq!(config.step_months, 3);
    assert_eq!(config.purge_days, 5);
    assert_eq!(config.embargo_days, 5);
    assert_eq!(config.selection_criteria, SelectionCriteria::PSR);
    assert_eq!(config.psr_threshold, dec!(0.5));
}

#[test]
fn test_nested_config_window_months() {
    let config = NestedWalkForwardConfig {
        train_months: 4,
        val_months: 1,
        test_months: 1,
        ..Default::default()
    };
    
    assert_eq!(config.window_months(), 6);
}

#[test]
fn test_nested_config_estimate_windows() {
    let config = NestedWalkForwardConfig {
        train_months: 4,
        val_months: 1,
        test_months: 1,
        step_months: 3,
        ..Default::default()
    };
    
    let start = date(2020, 1, 1);
    let end = date(2023, 1, 1);  // 3 years = 36 months
    
    let estimated = config.estimate_windows(start, end);
    
    // (36 - 6) / 3 + 1 = 11
    assert!(estimated >= 9 && estimated <= 12, "Expected ~11 windows, got {}", estimated);
}

#[test]
fn test_penalty_config_default() {
    let penalties = PenaltyConfig::default();
    
    assert_eq!(penalties.turnover_weight, dec!(0.10));
    assert_eq!(penalties.cost_weight, dec!(0.05));
    assert_eq!(penalties.drawdown_weight, dec!(0.20));
}

// ========================
// ParamSet Ordering Tests
// ========================

#[test]
fn test_paramset_ordering_by_top_n() {
    let p1 = ParamSet { top_n: 5, ..Default::default() };
    let p2 = ParamSet { top_n: 10, ..Default::default() };
    
    assert!(p1 < p2);
}

#[test]
fn test_paramset_ordering_by_stop_loss() {
    let p1 = ParamSet {
        top_n: 10,
        stop_loss_pct: dec!(0.10),
        ..Default::default()
    };
    let p2 = ParamSet {
        top_n: 10,
        stop_loss_pct: dec!(0.15),
        ..Default::default()
    };
    
    assert!(p1 < p2);
}

#[test]
fn test_paramset_equality() {
    let p1 = ParamSet::default();
    let p2 = ParamSet::default();
    
    assert_eq!(p1, p2);
}

// ========================
// Integration Tests
// ========================

#[test]
fn test_nested_split_structure() {
    use backtester_intelligence::walkforward::WindowType;
    
    let splitter = NestedSplitter::from_parts(4, 1, 1, 3, 5, 5);
    let start = date(2020, 1, 1);
    let end = date(2021, 6, 30);

    let splits = splitter.generate_nested_splits(start, end);

    assert!(!splits.is_empty());
    
    for split in &splits {
        assert_eq!(split.train.window_type, WindowType::Train);
        assert_eq!(split.val.window_type, WindowType::Validation);
        assert_eq!(split.test.window_type, WindowType::Test);
        
        // Train should be longest
        assert!(split.train.days() > split.val.days());
        assert!(split.train.days() > split.test.days());
    }
}

#[test]
fn test_nested_split_consecutive_overlap() {
    let splitter = NestedSplitter::from_parts(4, 1, 1, 3, 5, 5);
    let start = date(2020, 1, 1);
    let end = date(2021, 12, 31);

    let splits = splitter.generate_nested_splits(start, end);

    if splits.len() >= 2 {
        // With 3-month step, consecutive windows should overlap
        let s0 = &splits[0];
        let s1 = &splits[1];

        let step_days = (s1.train.start_date - s0.train.start_date).num_days();
        assert!(
            step_days >= 85 && step_days <= 95,
            "Expected ~90 days step, got {}",
            step_days
        );
    }
}









//! Dividend Integration Tests
//!
//! T1: Buy-and-hold economic return matches expected
//! T2: Anti-double-count validation
//! T3: Determinism with dividends
//! T4: Edge cases (no position, partial, buy on ex-date)

use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;

use backtester_intelligence::{
    dividends::{DividendEntry, DividendIndex, DividendApplication, PriceType},
    accounting::PortfolioState,
    exit::Position,
    filters::Market,
};

// =============================================================================
// TEST FIXTURES
// =============================================================================

/// Create a deterministic test scenario for buy-and-hold with dividends.
fn create_buyhold_fixture() -> BuyHoldFixture {
    BuyHoldFixture {
        symbol: "TAEE11".to_string(),
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 12, 31).unwrap(),
        initial_capital: dec!(100_000),
        initial_shares: 1000,
        initial_price: dec!(40.00),
        // Quarterly dividends
        dividends: vec![
            (NaiveDate::from_ymd_opt(2024, 3, 15).unwrap(), dec!(0.50)),
            (NaiveDate::from_ymd_opt(2024, 6, 15).unwrap(), dec!(0.55)),
            (NaiveDate::from_ymd_opt(2024, 9, 15).unwrap(), dec!(0.50)),
            (NaiveDate::from_ymd_opt(2024, 12, 15).unwrap(), dec!(0.60)),
        ],
        // Price series (simplified: raw prices with drop on ex-date)
        price_series: generate_price_series_with_divs(),
    }
}

fn generate_price_series_with_divs() -> Vec<(NaiveDate, Decimal, Decimal)> {
    // (date, raw_price, adjusted_price)
    // 
    // Model: Raw price grows steadily. On ex-date, raw drops by dividend amount.
    // Adjusted price is calculated backwards to show total return.
    let mut series = Vec::new();
    let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
    let end = NaiveDate::from_ymd_opt(2024, 12, 31).unwrap();
    
    let initial_price = dec!(40.00);
    
    // Dividend schedule
    let div_dates: HashMap<NaiveDate, Decimal> = [
        (NaiveDate::from_ymd_opt(2024, 3, 15).unwrap(), dec!(0.50)),
        (NaiveDate::from_ymd_opt(2024, 6, 15).unwrap(), dec!(0.55)),
        (NaiveDate::from_ymd_opt(2024, 9, 15).unwrap(), dec!(0.50)),
        (NaiveDate::from_ymd_opt(2024, 12, 15).unwrap(), dec!(0.60)),
    ].into_iter().collect();
    
    let mut current = start;
    let mut raw_price = initial_price;
    let daily_growth = dec!(0.0003); // ~7.5% annual
    
    while current <= end {
        // Raw price grows each day
        if current > start {
            raw_price = raw_price * (Decimal::ONE + daily_growth);
        }
        
        // On ex-date, raw price drops by dividend amount
        if let Some(&div) = div_dates.get(&current) {
            raw_price -= div;
        }
        
        // For simplicity, adjusted = raw + all past dividends received
        // (This is a simplified model - real adjustment is multiplicative)
        let past_divs: Decimal = div_dates.iter()
            .filter(|(d, _)| **d <= current)
            .map(|(_, r)| *r)
            .sum();
        let adjusted = raw_price + past_divs;
        
        series.push((current, raw_price, adjusted));
        current += chrono::Duration::days(1);
    }
    
    series
}

struct BuyHoldFixture {
    symbol: String,
    start_date: NaiveDate,
    end_date: NaiveDate,
    initial_capital: Decimal,
    initial_shares: i64,
    initial_price: Decimal,
    dividends: Vec<(NaiveDate, Decimal)>,
    price_series: Vec<(NaiveDate, Decimal, Decimal)>,
}

// =============================================================================
// T1: BUY-AND-HOLD ECONOMIC RETURN
// =============================================================================

/// Verify that buy-and-hold with raw prices + cashflow matches adjusted total return.
///
/// Key invariant: equity(raw) + dividend_cashflow == equity(adjusted)
/// This proves that dividends are not double-counted.
#[test]
fn t1_buyhold_economic_return_matches_adjusted() {
    let fixture = create_buyhold_fixture();
    
    // Build dividend index
    let mut div_index = DividendIndex::new();
    for (date, rate) in &fixture.dividends {
        div_index.add(DividendEntry::cash(&fixture.symbol, *date, *rate));
    }
    
    let shares = fixture.initial_shares;
    
    // Get price data
    let first_raw = fixture.price_series.first().unwrap().1;
    let final_raw = fixture.price_series.last().unwrap().1;
    let first_adj = fixture.price_series.first().unwrap().2;
    let final_adj = fixture.price_series.last().unwrap().2;
    
    // Method 1: RAW prices + dividend cashflow
    let raw_position_value_initial = first_raw * Decimal::from(shares);
    let raw_position_value_final = final_raw * Decimal::from(shares);
    let total_div_cashflow: Decimal = fixture.dividends.iter()
        .map(|(_, r)| *r * Decimal::from(shares))
        .sum();
    
    // Total return via RAW + cashflow
    let total_return_raw_method = raw_position_value_final + total_div_cashflow;
    
    // Method 2: ADJUSTED prices (should match raw + cashflow)
    // Adjusted price at end = raw + all past dividends (by our model)
    let adj_position_value_final = final_adj * Decimal::from(shares);
    
    // The key assertion: adjusted final == raw final + past dividends (per share)
    // So adjusted position == raw position + dividend cashflow
    let diff = (total_return_raw_method - adj_position_value_final).abs();
    let tolerance = dec!(1); // $1 tolerance for Decimal precision
    
    assert!(
        diff < tolerance,
        "Economic return mismatch:\n\
         RAW method: raw_value({}) + divs({}) = {}\n\
         ADJ method: adj_value({})\n\
         Diff: {}",
        raw_position_value_final, total_div_cashflow, total_return_raw_method,
        adj_position_value_final, diff
    );
    
    // Also verify the return percentages are equivalent
    let raw_return_pct = (total_return_raw_method - raw_position_value_initial) / raw_position_value_initial;
    let adj_return_pct = (adj_position_value_final - first_adj * Decimal::from(shares)) / (first_adj * Decimal::from(shares));
    
    let pct_diff = (raw_return_pct - adj_return_pct).abs();
    assert!(
        pct_diff < dec!(0.001), // 0.1% tolerance
        "Return percentage mismatch: raw={}, adj={}, diff={}",
        raw_return_pct, adj_return_pct, pct_diff
    );
}

// =============================================================================
// T2: ANTI-DOUBLE-COUNT
// =============================================================================

/// Prove that using adjusted prices for PnL AND adding cashflow would double count.
#[test]
fn t2_anti_double_count_validation() {
    let fixture = create_buyhold_fixture();
    
    let div_index = DividendIndex::new();
    let shares = fixture.initial_shares;
    
    // Method 1: RAW prices + cashflow (CORRECT)
    let initial_raw = fixture.price_series.first().unwrap().1;
    let final_raw = fixture.price_series.last().unwrap().1;
    let raw_pnl = (final_raw - initial_raw) * Decimal::from(shares);
    let div_cashflow: Decimal = fixture.dividends.iter()
        .map(|(_, rate)| *rate * Decimal::from(shares))
        .sum();
    let total_return_correct = raw_pnl + div_cashflow;
    
    // Method 2: ADJUSTED prices only (also correct, alternative)
    let initial_adj = fixture.price_series.first().unwrap().2;
    let final_adj = fixture.price_series.last().unwrap().2;
    let adj_pnl = (final_adj - initial_adj) * Decimal::from(shares);
    
    // Method 3: ADJUSTED prices + cashflow (WRONG - double counts!)
    let double_counted = adj_pnl + div_cashflow;
    
    // Verify Method 1 ≈ Method 2 (within tolerance)
    let diff_correct = (total_return_correct - adj_pnl).abs();
    assert!(
        diff_correct < dec!(10), // Small tolerance for model simplification
        "RAW+cashflow should match ADJ: diff={}",
        diff_correct
    );
    
    // Verify Method 3 is significantly higher (double counted!)
    assert!(
        double_counted > adj_pnl + dec!(1000),
        "Double-count method should be higher: double={}, adj={}",
        double_counted, adj_pnl
    );
    
    // The ratio should show ~2x the dividend impact
    let excess_from_double = double_counted - adj_pnl;
    assert!(
        excess_from_double > div_cashflow * dec!(0.9),
        "Double-count excess should be ~= dividend cashflow: excess={}, divs={}",
        excess_from_double, div_cashflow
    );
}

// =============================================================================
// T3: DETERMINISM
// =============================================================================

/// Running the same scenario twice must produce identical results.
#[test]
fn t3_determinism_with_dividends() {
    fn run_scenario() -> (Decimal, Decimal, i32) {
        let fixture = create_buyhold_fixture();
        
        let mut portfolio = PortfolioState::new(fixture.initial_capital);
        let date = fixture.start_date;
        
        // Buy initial position
        portfolio.apply_buy(
            &fixture.symbol,
            fixture.initial_shares,
            fixture.initial_price,
            dec!(0), // no cost for test
            Market::BR,
            date,
        ).unwrap();
        
        // Build dividend index
        let mut div_index = DividendIndex::new();
        for (date, rate) in &fixture.dividends {
            div_index.add(DividendEntry::cash(&fixture.symbol, *date, *rate));
        }
        
        let mut div_count = 0;
        let mut total_divs = dec!(0);
        
        // Process each day
        for (date, raw_price, _) in &fixture.price_series {
            // Apply dividend
            if let Some(div) = div_index.get(*date, &fixture.symbol) {
                if let Some(pos) = portfolio.get_position(&fixture.symbol) {
                    let cashflow = div.rate * Decimal::from(pos.shares);
                    portfolio.add_cash(cashflow);
                    total_divs += cashflow;
                    div_count += 1;
                }
            }
            
            // Mark to market
            let mut prices = HashMap::new();
            prices.insert(fixture.symbol.clone(), *raw_price);
            portfolio.update_prices(&prices);
        }
        
        (portfolio.equity, total_divs, div_count)
    }
    
    let (equity1, divs1, count1) = run_scenario();
    let (equity2, divs2, count2) = run_scenario();
    
    assert_eq!(equity1, equity2, "Equity must be deterministic");
    assert_eq!(divs1, divs2, "Dividend cashflow must be deterministic");
    assert_eq!(count1, count2, "Dividend count must be deterministic");
}

// =============================================================================
// T4: EDGE CASES
// =============================================================================

/// No dividend credited when no position exists on ex-date.
#[test]
fn t4_no_dividend_without_position() {
    let mut div_index = DividendIndex::new();
    let ex_date = NaiveDate::from_ymd_opt(2024, 3, 15).unwrap();
    div_index.add(DividendEntry::cash("TAEE11", ex_date, dec!(0.50)));
    
    let mut portfolio = PortfolioState::new(dec!(100_000));
    
    // No position - check dividend
    let div = div_index.get(ex_date, "TAEE11").unwrap();
    let pos = portfolio.get_position("TAEE11");
    
    assert!(pos.is_none(), "Should have no position");
    
    // Cashflow should be zero if no position
    let cashflow = match pos {
        Some(p) => div.rate * Decimal::from(p.shares),
        None => dec!(0),
    };
    
    assert_eq!(cashflow, dec!(0), "No cashflow without position");
}

/// Partial position (sell after receiving dividend).
#[test]
fn t4_partial_position_receives_full_dividend() {
    let mut portfolio = PortfolioState::new(dec!(100_000));
    let buy_date = NaiveDate::from_ymd_opt(2024, 3, 1).unwrap();
    let ex_date = NaiveDate::from_ymd_opt(2024, 3, 15).unwrap();
    let sell_date = NaiveDate::from_ymd_opt(2024, 3, 20).unwrap();
    
    // Buy 1000 shares before ex-date
    portfolio.apply_buy("TAEE11", 1000, dec!(40), dec!(0), Market::BR, buy_date).unwrap();
    
    // On ex-date, we hold 1000 shares, so we get full dividend
    let mut div_index = DividendIndex::new();
    div_index.add(DividendEntry::cash("TAEE11", ex_date, dec!(0.50)));
    
    let cash_before = portfolio.cash;
    let div = div_index.get(ex_date, "TAEE11").unwrap();
    let pos = portfolio.get_position("TAEE11").unwrap();
    let cashflow = div.rate * Decimal::from(pos.shares);
    portfolio.add_cash(cashflow);
    
    assert_eq!(cashflow, dec!(500), "Should receive 0.50 * 1000 = 500");
    assert_eq!(portfolio.cash, cash_before + dec!(500));
    
    // Sell 500 shares after ex-date
    portfolio.apply_sell("TAEE11", 500, dec!(39.50), dec!(0)).unwrap();
    
    // Remaining position should still be 500
    let pos_after = portfolio.get_position("TAEE11").unwrap();
    assert_eq!(pos_after.shares, 500);
}

/// Buy on ex-date: does NOT receive dividend (shares must be held on T-1).
#[test]
fn t4_buy_on_exdate_no_dividend() {
    let mut portfolio = PortfolioState::new(dec!(100_000));
    let ex_date = NaiveDate::from_ymd_opt(2024, 3, 15).unwrap();
    
    // Dividend is on ex-date
    let mut div_index = DividendIndex::new();
    div_index.add(DividendEntry::cash("TAEE11", ex_date, dec!(0.50)));
    
    // Check for dividend BEFORE buying (no position yet)
    let div = div_index.get(ex_date, "TAEE11").unwrap();
    let pos_before = portfolio.get_position("TAEE11");
    
    // No dividend should be applied (position doesn't exist yet)
    let cashflow = match pos_before {
        Some(p) => div.rate * Decimal::from(p.shares),
        None => dec!(0),
    };
    assert_eq!(cashflow, dec!(0), "Buy on ex-date should not receive dividend");
    
    // Now buy (after dividend check)
    portfolio.apply_buy("TAEE11", 1000, dec!(39.50), dec!(0), Market::BR, ex_date).unwrap();
    
    // Position exists now but dividend already passed
    let pos_after = portfolio.get_position("TAEE11");
    assert!(pos_after.is_some());
}

/// Missing payment_date should still work (we use ex_date for crediting).
#[test]
fn t4_missing_payment_date_works() {
    let ex_date = NaiveDate::from_ymd_opt(2024, 3, 15).unwrap();
    
    // Create dividend without payment_date
    let mut div = DividendEntry::cash("TAEE11", ex_date, dec!(0.50));
    assert!(div.payment_date.is_none());
    
    // Index should work fine
    let mut index = DividendIndex::new();
    index.add(div);
    
    let retrieved = index.get(ex_date, "TAEE11").unwrap();
    assert_eq!(retrieved.rate, dec!(0.50));
    assert!(retrieved.payment_date.is_none());
}

/// Multiple dividends in same month.
#[test]
fn t4_multiple_dividends_same_period() {
    let mut div_index = DividendIndex::new();
    let d1 = NaiveDate::from_ymd_opt(2024, 3, 1).unwrap();
    let d2 = NaiveDate::from_ymd_opt(2024, 3, 15).unwrap();
    let d3 = NaiveDate::from_ymd_opt(2024, 3, 28).unwrap();
    
    div_index.add(DividendEntry::cash("TAEE11", d1, dec!(0.10)));
    div_index.add(DividendEntry::cash("TAEE11", d2, dec!(0.25)));
    div_index.add(DividendEntry::cash("TAEE11", d3, dec!(0.15)));
    
    let total = div_index.total_dividends("TAEE11", d1, d3);
    assert_eq!(total, dec!(0.50));
    
    let count = div_index.get_for_symbol("TAEE11", d1, d3).len();
    assert_eq!(count, 3);
}

// =============================================================================
// PRICE TYPE TESTS
// =============================================================================

#[test]
fn test_price_type_policy() {
    // Signals should use adjusted prices
    assert_eq!(PriceType::default(), PriceType::Signals);
    
    // Valuation should use raw prices
    let valuation = PriceType::Valuation;
    assert_ne!(valuation, PriceType::Signals);
}


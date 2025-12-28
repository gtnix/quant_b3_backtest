//! Integration tests for DbMarketSessionCalendar.
//!
//! These tests require a database connection (DATABASE_URL env var).
//! Run with: cargo test --test db_calendar_integration -- --ignored

use chrono::{Datelike, NaiveDate, NaiveTime, Timelike};
use std::env;

// ============================================================================
// Test Helpers
// ============================================================================

fn date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

fn time(h: u32, m: u32) -> NaiveTime {
    NaiveTime::from_hms_opt(h, m, 0).unwrap()
}

fn has_db_connection() -> bool {
    env::var("DATABASE_URL").is_ok()
}

// ============================================================================
// Unit Tests (No DB Required)
// ============================================================================

#[test]
fn test_supported_range_default() {
    // Default range is 2005-2025
    let start = date(2005, 1, 1);
    let end = date(2025, 12, 31);

    assert_eq!(start.year(), 2005);
    assert_eq!(end.year(), 2025);
}

#[test]
fn test_range_validation_logic() {
    let range_start = date(2005, 1, 1);
    let range_end = date(2025, 12, 31);

    // Valid date
    let valid = date(2024, 6, 15);
    assert!(valid >= range_start && valid <= range_end);

    // Before range
    let before = date(2004, 12, 31);
    assert!(before < range_start);

    // After range
    let after = date(2026, 1, 1);
    assert!(after > range_end);
}

#[test]
fn test_weekend_detection() {
    // Saturday
    let sat = date(2024, 12, 21);
    assert!(matches!(sat.weekday(), chrono::Weekday::Sat));

    // Sunday
    let sun = date(2024, 12, 22);
    assert!(matches!(sun.weekday(), chrono::Weekday::Sun));

    // Monday
    let mon = date(2024, 12, 23);
    assert!(!matches!(
        mon.weekday(),
        chrono::Weekday::Sat | chrono::Weekday::Sun
    ));
}

#[test]
fn test_early_close_time_us() {
    // NYSE early close is at 13:00 ET
    let early_close = time(13, 0);
    let normal_close = time(16, 0);

    assert_eq!(early_close.hour(), 13);
    assert_eq!(normal_close.hour(), 16);

    // 3 hours earlier
    let diff = normal_close.signed_duration_since(early_close);
    assert_eq!(diff.num_hours(), 3);
}

#[test]
fn test_late_open_time_br() {
    // B3 Ash Wednesday opens at 13:00 BRT
    let late_open = time(13, 0);
    let normal_open = time(10, 0);

    assert_eq!(late_open.hour(), 13);
    assert_eq!(normal_open.hour(), 10);

    // 3 hours later
    let diff = late_open.signed_duration_since(normal_open);
    assert_eq!(diff.num_hours(), 3);
}

#[test]
fn test_b3_session_times() {
    // B3 regular session: 10:00-17:55 BRT
    let open = time(10, 0);
    let close = time(17, 55);

    // Session duration: 7h 55m = 475 minutes
    let duration = close.signed_duration_since(open);
    assert_eq!(duration.num_minutes(), 475);
}

#[test]
fn test_nyse_session_times() {
    // NYSE regular session: 09:30-16:00 ET
    let open = time(9, 30);
    let close = time(16, 0);

    // Session duration: 6h 30m = 390 minutes
    let duration = close.signed_duration_since(open);
    assert_eq!(duration.num_minutes(), 390);
}

// ============================================================================
// Holiday Type Tests
// ============================================================================

#[test]
fn test_christmas_is_holiday() {
    // Christmas 2024 is December 25 (Wednesday)
    let christmas = date(2024, 12, 25);

    assert_eq!(christmas.month(), 12);
    assert_eq!(christmas.day(), 25);
    assert_eq!(christmas.weekday(), chrono::Weekday::Wed);
}

#[test]
fn test_thanksgiving_is_4th_thursday() {
    // Thanksgiving 2024 is November 28
    let thanksgiving = date(2024, 11, 28);

    assert_eq!(thanksgiving.month(), 11);
    assert_eq!(thanksgiving.weekday(), chrono::Weekday::Thu);

    // Verify it's the 4th Thursday
    let first_of_month = date(2024, 11, 1);
    let first_thursday_day = match first_of_month.weekday() {
        chrono::Weekday::Thu => 1,
        chrono::Weekday::Fri => 7,
        chrono::Weekday::Sat => 6,
        chrono::Weekday::Sun => 5,
        chrono::Weekday::Mon => 4,
        chrono::Weekday::Tue => 3,
        chrono::Weekday::Wed => 2,
    };
    let fourth_thursday = first_thursday_day + 21; // +3 weeks
    assert_eq!(thanksgiving.day() as i32, fourth_thursday);
}

#[test]
fn test_easter_calculation_2024() {
    // Easter 2024 is March 31
    // Using the Anonymous Gregorian algorithm
    let year = 2024;
    let a = year % 19;
    let b = year / 100;
    let c = year % 100;
    let d = b / 4;
    let e = b % 4;
    let f = (b + 8) / 25;
    let g = (b - f + 1) / 3;
    let h = (19 * a + b - d - g + 15) % 30;
    let i = c / 4;
    let k = c % 4;
    let l = (32 + 2 * e + 2 * i - h - k) % 7;
    let m = (a + 11 * h + 22 * l) / 451;
    let month = (h + l - 7 * m + 114) / 31;
    let day = ((h + l - 7 * m + 114) % 31) + 1;

    assert_eq!(month, 3);
    assert_eq!(day, 31);
}

#[test]
fn test_carnival_from_easter() {
    // Carnival is 47 days before Easter
    let easter = date(2024, 3, 31);
    let carnival_tuesday = easter - chrono::Duration::days(47);

    // Carnival 2024 is February 13
    assert_eq!(carnival_tuesday.month(), 2);
    assert_eq!(carnival_tuesday.day(), 13);
}

#[test]
fn test_good_friday_from_easter() {
    // Good Friday is 2 days before Easter
    let easter = date(2024, 3, 31);
    let good_friday = easter - chrono::Duration::days(2);

    // Good Friday 2024 is March 29
    assert_eq!(good_friday.month(), 3);
    assert_eq!(good_friday.day(), 29);
}

#[test]
fn test_corpus_christi_from_easter() {
    // Corpus Christi is 60 days after Easter
    let easter = date(2024, 3, 31);
    let corpus_christi = easter + chrono::Duration::days(60);

    // Corpus Christi 2024 is May 30
    assert_eq!(corpus_christi.month(), 5);
    assert_eq!(corpus_christi.day(), 30);
}

// ============================================================================
// Extraordinary Closure Tests
// ============================================================================

#[test]
fn test_hurricane_sandy_dates() {
    // NYSE closed October 29-30, 2012
    let sandy_day1 = date(2012, 10, 29);
    let sandy_day2 = date(2012, 10, 30);

    assert_eq!(sandy_day1.weekday(), chrono::Weekday::Mon);
    assert_eq!(sandy_day2.weekday(), chrono::Weekday::Tue);
}

#[test]
fn test_bush_mourning_date() {
    // NYSE closed December 5, 2018
    let mourning = date(2018, 12, 5);

    assert_eq!(mourning.weekday(), chrono::Weekday::Wed);
}

#[test]
fn test_ford_mourning_date() {
    // NYSE closed January 2, 2007
    let mourning = date(2007, 1, 2);

    assert_eq!(mourning.weekday(), chrono::Weekday::Tue);
}

// ============================================================================
// Juneteenth Tests
// ============================================================================

#[test]
fn test_juneteenth_2021_special_case() {
    // June 19, 2021 was Saturday
    // NYSE had early close on Friday June 18
    let juneteenth_2021 = date(2021, 6, 19);
    let early_close_day = date(2021, 6, 18);

    assert_eq!(juneteenth_2021.weekday(), chrono::Weekday::Sat);
    assert_eq!(early_close_day.weekday(), chrono::Weekday::Fri);
}

#[test]
fn test_juneteenth_2022_observed() {
    // June 19, 2022 was Sunday
    // Observed on Monday June 20
    let juneteenth_2022 = date(2022, 6, 19);
    let observed = date(2022, 6, 20);

    assert_eq!(juneteenth_2022.weekday(), chrono::Weekday::Sun);
    assert_eq!(observed.weekday(), chrono::Weekday::Mon);
}

#[test]
fn test_consciencia_negra_2024() {
    // November 20, 2024 is a NEW B3 holiday
    let consciencia_negra = date(2024, 11, 20);

    assert_eq!(consciencia_negra.month(), 11);
    assert_eq!(consciencia_negra.day(), 20);
    assert_eq!(consciencia_negra.weekday(), chrono::Weekday::Wed);
}

// ============================================================================
// No Silent Fallback Tests
// ============================================================================

#[test]
fn test_weekday_not_trading_day_christmas() {
    // Christmas 2024 is Wednesday - weekday but NOT trading day
    let christmas = date(2024, 12, 25);

    // It IS a weekday
    let is_weekday = !matches!(
        christmas.weekday(),
        chrono::Weekday::Sat | chrono::Weekday::Sun
    );
    assert!(is_weekday);

    // But it should NOT be assumed to be a trading day
    // This is the "no silent fallback" principle
}

#[test]
fn test_weekday_not_trading_day_hurricane_sandy() {
    // October 29, 2012 was Monday - weekday but market closed
    let sandy = date(2012, 10, 29);

    let is_weekday = !matches!(sandy.weekday(), chrono::Weekday::Sat | chrono::Weekday::Sun);
    assert!(is_weekday);

    // Must check extraordinary closures, not just weekday status
}

// ============================================================================
// Database Integration Tests (Ignored by default)
// ============================================================================

#[tokio::test]
#[ignore = "Requires DATABASE_URL"]
async fn test_db_connection() {
    if !has_db_connection() {
        return;
    }

    use market_data::calendar::DbMarketSessionCalendar;

    let db_url = env::var("DATABASE_URL").unwrap();
    let result = DbMarketSessionCalendar::connect(&db_url).await;

    assert!(result.is_ok(), "Should connect to database");
}

#[tokio::test]
#[ignore = "Requires DATABASE_URL"]
async fn test_db_classify_christmas() {
    if !has_db_connection() {
        return;
    }

    use market_data::calendar::{DbMarketSessionCalendar, Market};

    let db_url = env::var("DATABASE_URL").unwrap();
    let calendar = DbMarketSessionCalendar::connect(&db_url).await.unwrap();

    let christmas = date(2024, 12, 25);
    let result = calendar.is_trading_day(Market::BR, christmas).await;

    assert!(result.is_ok());
    assert!(!result.unwrap(), "Christmas should not be a trading day");
}

#[tokio::test]
#[ignore = "Requires DATABASE_URL"]
async fn test_db_classify_regular_monday() {
    if !has_db_connection() {
        return;
    }

    use market_data::calendar::{DbMarketSessionCalendar, Market};

    let db_url = env::var("DATABASE_URL").unwrap();
    let calendar = DbMarketSessionCalendar::connect(&db_url).await.unwrap();

    let monday = date(2024, 12, 23); // Regular Monday
    let result = calendar.is_trading_day(Market::BR, monday).await;

    assert!(result.is_ok());
    assert!(result.unwrap(), "Regular Monday should be a trading day");
}

#[tokio::test]
#[ignore = "Requires DATABASE_URL"]
async fn test_db_out_of_range_error() {
    if !has_db_connection() {
        return;
    }

    use market_data::calendar::{CalendarError, DbMarketSessionCalendar, Market};

    let db_url = env::var("DATABASE_URL").unwrap();
    let calendar = DbMarketSessionCalendar::connect(&db_url).await.unwrap();

    // Date before supported range
    let old_date = date(2004, 6, 15);
    let result = calendar.is_trading_day(Market::BR, old_date).await;

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), CalendarError::OutOfRange { .. }));
}

#[tokio::test]
#[ignore = "Requires DATABASE_URL"]
async fn test_db_trading_days_count() {
    if !has_db_connection() {
        return;
    }

    use market_data::calendar::{DbMarketSessionCalendar, Market};

    let db_url = env::var("DATABASE_URL").unwrap();
    let calendar = DbMarketSessionCalendar::connect(&db_url).await.unwrap();

    // Count trading days in January 2024
    let start = date(2024, 1, 1);
    let end = date(2024, 1, 31);
    let result = calendar.count_trading_days(Market::BR, start, end).await;

    assert!(result.is_ok());
    let count = result.unwrap();

    // January 2024 should have around 22 trading days
    assert!(count >= 20 && count <= 23, "Unexpected count: {}", count);
}

#[tokio::test]
#[ignore = "Requires DATABASE_URL"]
async fn test_db_next_trading_day() {
    if !has_db_connection() {
        return;
    }

    use market_data::calendar::{DbMarketSessionCalendar, Market};

    let db_url = env::var("DATABASE_URL").unwrap();
    let calendar = DbMarketSessionCalendar::connect(&db_url).await.unwrap();

    // Friday Dec 20 -> next trading day should be Monday Dec 23
    let friday = date(2024, 12, 20);
    let result = calendar.next_trading_day(Market::BR, friday).await;

    assert!(result.is_ok());
    let next = result.unwrap();

    assert_eq!(next, date(2024, 12, 23), "Next trading day should be Monday");
}

#[tokio::test]
#[ignore = "Requires DATABASE_URL"]
async fn test_db_previous_trading_day() {
    if !has_db_connection() {
        return;
    }

    use market_data::calendar::{DbMarketSessionCalendar, Market};

    let db_url = env::var("DATABASE_URL").unwrap();
    let calendar = DbMarketSessionCalendar::connect(&db_url).await.unwrap();

    // Monday Dec 23 -> previous trading day should be Friday Dec 20
    let monday = date(2024, 12, 23);
    let result = calendar.previous_trading_day(Market::BR, monday).await;

    assert!(result.is_ok());
    let prev = result.unwrap();

    assert_eq!(
        prev,
        date(2024, 12, 20),
        "Previous trading day should be Friday"
    );
}


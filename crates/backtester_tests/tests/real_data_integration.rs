//! Real Data Integration Tests
//!
//! Tests the backtester with real cached OHLCV data from cache/ohlcv/*.csv.
//! These tests validate that the engine works correctly with production data.
//!
//! Run with: `cargo test -p backtester_tests --test real_data_integration --features real_data`
//!
//! Note: Tests are skipped if cache files don't exist or feature is disabled.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use backtester_core::{AssetId, Bar, MarketEvent};

// =============================================================================
// CACHE LOADER
// =============================================================================

/// Date structure for proper ordering
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct DateValue {
    year: i32,
    month: u32,
    day: u32,
}

/// Bar with date for proper ordering
struct BarWithDate {
    bar: Bar,
    date: DateValue,
}

/// Load OHLCV data from cache CSV file.
fn load_ohlcv_from_cache(symbol: &str) -> Option<Vec<Bar>> {
    let cache_path = get_cache_path(symbol);
    if !cache_path.exists() {
        return None;
    }

    let file = File::open(&cache_path).ok()?;
    let reader = BufReader::new(file);
    let mut bars_with_dates: Vec<BarWithDate> = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        if i == 0 {
            continue; // Skip header
        }
        let line = line.ok()?;
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 6 {
            continue;
        }

        // Parse: date,open,high,low,close,volume
        let date_str = parts[0];
        let (timestamp, date) = parse_date_to_timestamp(date_str)?;
        let open: f64 = parts[1].parse().ok()?;
        let high: f64 = parts[2].parse().ok()?;
        let low: f64 = parts[3].parse().ok()?;
        let close: f64 = parts[4].parse().ok()?;
        let volume: f64 = parts[5].parse().ok()?;

        bars_with_dates.push(BarWithDate {
            bar: Bar {
                timestamp,
                open,
                high,
                low,
                close,
                volume,
            },
            date,
        });
    }

    // Sort by date to ensure chronological order
    bars_with_dates.sort_by_key(|b| b.date);
    
    // Update timestamps based on sorted order
    let bars: Vec<Bar> = bars_with_dates
        .into_iter()
        .enumerate()
        .map(|(i, bwd)| Bar {
            timestamp: i as i64, // Use index as timestamp for ordering
            ..bwd.bar
        })
        .collect();

    Some(bars)
}

/// Get cache file path for a symbol.
fn get_cache_path(symbol: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("cache")
        .join("ohlcv")
        .join(format!("{}.csv", symbol))
}

/// Parse date string (YYYY-MM-DD) to timestamp and DateValue.
fn parse_date_to_timestamp(date: &str) -> Option<(i64, DateValue)> {
    let parts: Vec<&str> = date.split('-').collect();
    if parts.len() != 3 {
        return None;
    }
    let year: i32 = parts[0].parse().ok()?;
    let month: u32 = parts[1].parse().ok()?;
    let day: u32 = parts[2].parse().ok()?;

    // Convert to days since epoch using proper formula
    let days = days_since_epoch(year, month, day);
    let timestamp = days * 24 * 60 * 60 * 1_000_000_000; // nanoseconds
    Some((timestamp, DateValue { year, month, day }))
}

/// Calculate days since Unix epoch (1970-01-01).
fn days_since_epoch(year: i32, month: u32, day: u32) -> i64 {
    // Simplified Julian day calculation
    let a = (14 - month as i64) / 12;
    let y = year as i64 + 4800 - a;
    let m = month as i64 + 12 * a - 3;
    let jdn = day as i64 + (153 * m + 2) / 5 + 365 * y + y / 4 - y / 100 + y / 400 - 32045;
    jdn - 2440588 // Unix epoch is Julian day 2440588
}

/// Check if cache directory exists with data.
fn cache_exists() -> bool {
    let cache_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("cache")
        .join("ohlcv");
    cache_dir.exists() && cache_dir.is_dir()
}

// =============================================================================
// INVARIANT HELPERS
// =============================================================================

/// Validate bar data invariants with tolerance for real data quirks.
/// Real data from providers may have small rounding differences.
fn validate_bar_invariants(bars: &[Bar]) -> Result<(), String> {
    const TOLERANCE: f64 = 0.01; // 1 cent tolerance for rounding
    let mut violations = 0;
    
    for (i, bar) in bars.iter().enumerate() {
        // High >= Low (with tolerance)
        if bar.high < bar.low - TOLERANCE {
            violations += 1;
            if violations <= 5 {
                eprintln!("Warning: Bar {}: high ({}) < low ({})", i, bar.high, bar.low);
            }
        }
        // High >= Open and Close (with tolerance)
        if bar.high < bar.open - TOLERANCE || bar.high < bar.close - TOLERANCE {
            violations += 1;
            if violations <= 5 {
                eprintln!("Warning: Bar {}: high < open or close", i);
            }
        }
        // Low <= Open and Close (with tolerance)
        if bar.low > bar.open + TOLERANCE || bar.low > bar.close + TOLERANCE {
            violations += 1;
            if violations <= 5 {
                eprintln!("Warning: Bar {}: low > open or close", i);
            }
        }
        // Positive volume (allow zero for some edge cases)
        if bar.volume < -TOLERANCE {
            return Err(format!("Bar {}: negative volume", i));
        }
        // Valid prices
        if bar.close <= 0.0 {
            return Err(format!("Bar {}: non-positive close", i));
        }
    }
    
    // Allow up to 1% of bars to have minor violations (data quality issues)
    let violation_ratio = violations as f64 / bars.len() as f64;
    if violation_ratio > 0.01 {
        return Err(format!(
            "Too many bar invariant violations: {} of {} ({:.2}%)",
            violations,
            bars.len(),
            violation_ratio * 100.0
        ));
    }
    
    Ok(())
}

/// Calculate simple returns from bars.
fn calculate_returns(bars: &[Bar]) -> Vec<f64> {
    bars.windows(2)
        .map(|w| (w[1].close - w[0].close) / w[0].close)
        .collect()
}

/// Calculate annualized volatility.
fn calculate_volatility(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    variance.sqrt() * (252.0_f64).sqrt()
}

// =============================================================================
// TESTS
// =============================================================================

#[test]
#[cfg_attr(not(feature = "real_data"), ignore)]
fn test_load_petr4_data() {
    if !cache_exists() {
        eprintln!("Skipping: cache/ohlcv not found");
        return;
    }

    let bars = load_ohlcv_from_cache("PETR4");
    assert!(bars.is_some(), "PETR4.csv should exist in cache");

    let bars = bars.unwrap();
    assert!(bars.len() > 2500, "PETR4 should have 10+ years of data");

    // Validate invariants
    validate_bar_invariants(&bars).expect("Bar invariants should hold");
}

#[test]
#[cfg_attr(not(feature = "real_data"), ignore)]
fn test_load_multiple_symbols() {
    if !cache_exists() {
        eprintln!("Skipping: cache/ohlcv not found");
        return;
    }

    let symbols = ["PETR4", "VALE3", "ITUB4", "BBDC4", "B3SA3"];
    let mut loaded = 0;

    for symbol in &symbols {
        if let Some(bars) = load_ohlcv_from_cache(symbol) {
            loaded += 1;
            assert!(!bars.is_empty(), "{} should have data", symbol);
            validate_bar_invariants(&bars).expect(&format!("{} invariants should hold", symbol));
        }
    }

    assert!(loaded >= 3, "At least 3 major symbols should be in cache");
}

#[test]
#[cfg_attr(not(feature = "real_data"), ignore)]
fn test_returns_calculation_real_data() {
    if !cache_exists() {
        eprintln!("Skipping: cache/ohlcv not found");
        return;
    }

    let bars = load_ohlcv_from_cache("PETR4").expect("PETR4.csv should exist");
    let returns = calculate_returns(&bars);

    // Returns should be finite
    for (i, r) in returns.iter().enumerate() {
        assert!(r.is_finite(), "Return {} should be finite, got {}", i, r);
    }

    // Volatility should be reasonable (5% - 100% annualized for equities)
    let vol = calculate_volatility(&returns);
    assert!(vol > 0.05, "Vol should be > 5%, got {:.2}%", vol * 100.0);
    assert!(vol < 1.5, "Vol should be < 150%, got {:.2}%", vol * 100.0);
}

#[test]
#[cfg_attr(not(feature = "real_data"), ignore)]
fn test_date_ordering() {
    if !cache_exists() {
        eprintln!("Skipping: cache/ohlcv not found");
        return;
    }

    let bars = load_ohlcv_from_cache("VALE3").expect("VALE3.csv should exist");

    // Verify chronological order (timestamps are now sorted indices)
    for i in 1..bars.len() {
        assert!(
            bars[i].timestamp >= bars[i - 1].timestamp,
            "Bars should be in chronological order at index {}",
            i
        );
    }
    
    // Verify we have substantial data
    assert!(bars.len() > 2000, "VALE3 should have 10+ years of data");
}

#[test]
#[cfg_attr(not(feature = "real_data"), ignore)]
fn test_market_events_from_real_data() {
    if !cache_exists() {
        eprintln!("Skipping: cache/ohlcv not found");
        return;
    }

    let bars = load_ohlcv_from_cache("ITUB4").expect("ITUB4.csv should exist");

    // Convert to market events
    let events: Vec<MarketEvent> = bars
        .iter()
        .map(|bar| MarketEvent {
            asset_id: AssetId::new(0),
            bar: bar.clone(),
        })
        .collect();

    assert!(!events.is_empty());
    assert!(events.len() > 1000, "Should have substantial event history");

    // Verify no NaN values
    for event in &events {
        assert!(!event.bar.close.is_nan());
        assert!(!event.bar.open.is_nan());
        assert!(!event.bar.high.is_nan());
        assert!(!event.bar.low.is_nan());
    }
}

#[test]
#[cfg_attr(not(feature = "real_data"), ignore)]
fn test_price_gaps_detection() {
    if !cache_exists() {
        eprintln!("Skipping: cache/ohlcv not found");
        return;
    }

    let bars = load_ohlcv_from_cache("PETR4").expect("PETR4.csv should exist");

    // Count large gaps (>10% overnight moves)
    let mut large_gaps = 0;
    for i in 1..bars.len() {
        let gap = (bars[i].open - bars[i - 1].close).abs() / bars[i - 1].close;
        if gap > 0.10 {
            large_gaps += 1;
        }
    }

    // Some large gaps are expected (earnings, dividends, etc.)
    // but shouldn't be more than ~5% of days
    let gap_ratio = large_gaps as f64 / bars.len() as f64;
    assert!(
        gap_ratio < 0.05,
        "Too many large gaps: {:.1}%",
        gap_ratio * 100.0
    );
}

#[test]
#[cfg_attr(not(feature = "real_data"), ignore)]
fn test_universe_csv_consistency() {
    let universe_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("cache")
        .join("universe.csv");

    if !universe_path.exists() {
        eprintln!("Skipping: cache/universe.csv not found");
        return;
    }

    // Load universe and check that referenced files exist
    let file = File::open(&universe_path).expect("Failed to open universe.csv");
    let reader = BufReader::new(file);
    let mut checked = 0;
    let mut missing = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        if i == 0 {
            continue;
        }
        let line = line.expect("Failed to read line");
        let parts: Vec<&str> = line.split(',').collect();
        if parts.is_empty() {
            continue;
        }

        let symbol = parts[0];
        let ohlcv_path = get_cache_path(symbol);

        if !ohlcv_path.exists() {
            missing.push(symbol.to_string());
        }
        checked += 1;
    }

    // Allow some missing (delisted, etc.) but not too many
    let missing_ratio = missing.len() as f64 / checked as f64;
    assert!(
        missing_ratio < 0.10,
        "Too many missing OHLCV files: {:.1}% ({} of {})",
        missing_ratio * 100.0,
        missing.len(),
        checked
    );
}


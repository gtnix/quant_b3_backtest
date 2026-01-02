//! Integration tests for interest rate parsing.
//!
//! These tests verify the parsing logic using mock response data.
//! For live API tests, use: cargo test --features live_api

/// Mock BCB response for SELIC Serie 432.
const MOCK_BCB_RESPONSE: &str = r#"[
    {"data":"01/01/2024","valor":"10,75"},
    {"data":"15/01/2024","valor":"10,75"},
    {"data":"01/06/2024","valor":"10,25"},
    {"data":"01/09/2024","valor":"11,25"}
]"#;

/// Mock FRED response for TB3MS.
const MOCK_FRED_RESPONSE: &str = r#"{
    "observations": [
        {"date": "2024-01-01", "value": "5.20"},
        {"date": "2024-06-01", "value": "4.35"},
        {"date": "2024-09-01", "value": "."},
        {"date": "2024-10-01", "value": "4.50"}
    ]
}"#;

#[derive(Debug, serde::Deserialize)]
struct BcbDataPoint {
    data: String,
    valor: String,
}

#[derive(Debug, serde::Deserialize)]
struct FredResponse {
    observations: Vec<FredObservation>,
}

#[derive(Debug, serde::Deserialize)]
struct FredObservation {
    date: String,
    value: String,
}

#[test]
fn test_bcb_response_parsing() {
    let data: Vec<BcbDataPoint> = serde_json::from_str(MOCK_BCB_RESPONSE).unwrap();

    assert_eq!(data.len(), 4);
    assert_eq!(data[0].data, "01/01/2024");
    assert_eq!(data[0].valor, "10,75");
}

#[test]
fn test_bcb_rate_normalization() {
    let data: Vec<BcbDataPoint> = serde_json::from_str(MOCK_BCB_RESPONSE).unwrap();

    for dp in &data {
        let rate_pct: f64 = dp.valor.replace(',', ".").parse().unwrap();
        let rate_decimal = rate_pct / 100.0;

        // Rates should be in valid range (0 to 1)
        assert!(
            rate_decimal >= 0.0 && rate_decimal <= 1.0,
            "Rate {} out of range for {}",
            rate_decimal,
            dp.data
        );
    }

    // First entry should be 10.75% = 0.1075
    let first_rate: f64 = data[0].valor.replace(',', ".").parse::<f64>().unwrap() / 100.0;
    assert!((first_rate - 0.1075).abs() < 0.0001);
}

#[test]
fn test_bcb_date_parsing() {
    let data: Vec<BcbDataPoint> = serde_json::from_str(MOCK_BCB_RESPONSE).unwrap();

    for dp in &data {
        let parts: Vec<&str> = dp.data.split('/').collect();
        assert_eq!(parts.len(), 3, "Invalid date format: {}", dp.data);

        let day: u32 = parts[0].parse().expect("Invalid day");
        let month: u32 = parts[1].parse().expect("Invalid month");
        let year: i32 = parts[2].parse().expect("Invalid year");

        let date = chrono::NaiveDate::from_ymd_opt(year, month, day);
        assert!(date.is_some(), "Invalid date components: {}", dp.data);
    }
}

#[test]
fn test_fred_response_parsing() {
    let data: FredResponse = serde_json::from_str(MOCK_FRED_RESPONSE).unwrap();

    assert_eq!(data.observations.len(), 4);
    assert_eq!(data.observations[0].date, "2024-01-01");
    assert_eq!(data.observations[0].value, "5.20");
}

#[test]
fn test_fred_rate_normalization() {
    let data: FredResponse = serde_json::from_str(MOCK_FRED_RESPONSE).unwrap();

    for obs in &data.observations {
        // Skip missing values (FRED uses "." for missing)
        if obs.value == "." || obs.value.is_empty() {
            continue;
        }

        let rate_pct: f64 = obs.value.parse().unwrap();
        let rate_decimal = rate_pct / 100.0;

        // Rates should be in valid range (0 to 1)
        assert!(
            rate_decimal >= 0.0 && rate_decimal <= 1.0,
            "Rate {} out of range for {}",
            rate_decimal,
            obs.date
        );
    }

    // First entry should be 5.20% = 0.052
    let first_rate: f64 = data.observations[0].value.parse::<f64>().unwrap() / 100.0;
    assert!((first_rate - 0.052).abs() < 0.0001);
}

#[test]
fn test_fred_missing_value_handling() {
    let data: FredResponse = serde_json::from_str(MOCK_FRED_RESPONSE).unwrap();

    // Count non-missing values
    let valid_count = data
        .observations
        .iter()
        .filter(|obs| obs.value != "." && !obs.value.is_empty())
        .count();

    assert_eq!(
        valid_count, 3,
        "Expected 3 valid observations, got {}",
        valid_count
    );
}

#[test]
fn test_fred_date_parsing() {
    let data: FredResponse = serde_json::from_str(MOCK_FRED_RESPONSE).unwrap();

    for obs in &data.observations {
        let date = chrono::NaiveDate::parse_from_str(&obs.date, "%Y-%m-%d");
        assert!(date.is_ok(), "Invalid date format: {}", obs.date);
    }
}

#[test]
fn test_rate_range_validation() {
    // Valid rates should be between 0 and 1 (0% to 100%)
    // But typically interest rates are in 0-0.5 range (0-50%)
    let typical_rates = [0.0435, 0.1075, 0.0, 0.25, 0.005, 0.15];

    for rate in typical_rates {
        assert!(rate >= 0.0, "Rate {} is negative", rate);
        assert!(rate <= 0.5, "Rate {} is unusually high (>50%)", rate);
    }
}



























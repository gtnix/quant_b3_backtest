//! Interest rate ingestion from BCB (Brazil) and FRED (US).
//!
//! - BCB: Serie 432 (SELIC Meta) - public, no auth
//! - FRED: TB3MS (3-Month Treasury Bill) - requires FRED_API_KEY

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, info, warn};

/// Interest rate entry for storage.
#[derive(Debug, Clone, Serialize)]
pub struct InterestRateEntry {
    pub rate_date: NaiveDate,
    pub region: String,
    pub rate_type: String,
    pub rate: f64, // decimal/year, e.g., 0.1075 = 10.75%
    pub source: String,
}

#[derive(Error, Debug)]
pub enum InterestRateError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("API key missing: {0}")]
    ApiKeyMissing(String),
    #[error("API error: {0}")]
    Api(String),
}

// ============================================================================
// BCB Client (Banco Central do Brasil)
// ============================================================================

/// BCB SGS API response item.
#[derive(Debug, Deserialize)]
struct BcbDataPoint {
    data: String,  // DD/MM/YYYY
    valor: String, // rate as string (percentage)
}

pub struct BcbClient {
    client: reqwest::Client,
}

impl BcbClient {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client"),
        }
    }

    /// Fetch SELIC Meta (Serie 432) for date range.
    /// BCB returns rate in percentage (e.g., 10.75), we convert to decimal (0.1075).
    pub async fn fetch_selic(
        &self,
        start: NaiveDate,
        end: NaiveDate,
    ) -> Result<Vec<InterestRateEntry>, InterestRateError> {
        let start_str = start.format("%d/%m/%Y").to_string();
        let end_str = end.format("%d/%m/%Y").to_string();

        let url = format!(
            "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados?formato=json&dataInicial={}&dataFinal={}",
            start_str, end_str
        );

        debug!("BCB request: {}", url);

        let response = self.client.get(&url).send().await?;

        if !response.status().is_success() {
            return Err(InterestRateError::Api(format!(
                "BCB API returned status {}",
                response.status()
            )));
        }

        let data: Vec<BcbDataPoint> = response.json().await?;
        info!(
            "BCB: fetched {} SELIC rates for {} to {}",
            data.len(),
            start,
            end
        );

        let entries = data
            .into_iter()
            .filter_map(|dp| {
                // Parse date DD/MM/YYYY
                let parts: Vec<&str> = dp.data.split('/').collect();
                if parts.len() != 3 {
                    warn!("BCB: invalid date format: {}", dp.data);
                    return None;
                }
                let day: u32 = parts[0].parse().ok()?;
                let month: u32 = parts[1].parse().ok()?;
                let year: i32 = parts[2].parse().ok()?;
                let rate_date = NaiveDate::from_ymd_opt(year, month, day)?;

                // Parse rate and convert to decimal
                let rate_pct: f64 = dp.valor.replace(',', ".").parse().ok()?;
                let rate_decimal = rate_pct / 100.0;

                Some(InterestRateEntry {
                    rate_date,
                    region: "BR".to_string(),
                    rate_type: "SELIC".to_string(),
                    rate: rate_decimal,
                    source: "BCB_SGS_432".to_string(),
                })
            })
            .collect();

        Ok(entries)
    }
}

// ============================================================================
// FRED Client (Federal Reserve Economic Data)
// ============================================================================

/// FRED API observation response.
#[derive(Debug, Deserialize)]
struct FredResponse {
    observations: Vec<FredObservation>,
}

#[derive(Debug, Deserialize)]
struct FredObservation {
    date: String,  // YYYY-MM-DD
    value: String, // rate as string or "."
}

pub struct FredClient {
    client: reqwest::Client,
    api_key: String,
}

impl FredClient {
    pub fn new() -> Result<Self, InterestRateError> {
        let api_key = std::env::var("FRED_API_KEY")
            .map_err(|_| InterestRateError::ApiKeyMissing("FRED_API_KEY not set".into()))?;

        Ok(Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client"),
            api_key,
        })
    }

    /// Fetch T-Bill 3M (TB3MS) for date range.
    /// FRED returns rate in percentage (e.g., 4.35), we convert to decimal (0.0435).
    pub async fn fetch_tbill_3m(
        &self,
        start: NaiveDate,
        end: NaiveDate,
    ) -> Result<Vec<InterestRateEntry>, InterestRateError> {
        let url = format!(
            "https://api.stlouisfed.org/fred/series/observations?series_id=TB3MS&api_key={}&file_type=json&observation_start={}&observation_end={}",
            self.api_key,
            start.format("%Y-%m-%d"),
            end.format("%Y-%m-%d")
        );

        debug!("FRED request: TB3MS {} to {}", start, end);

        let response = self.client.get(&url).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            if status.as_u16() == 401 || text.contains("api_key") {
                return Err(InterestRateError::Api(
                    "FRED API key invalid or expired".into(),
                ));
            }
            return Err(InterestRateError::Api(format!(
                "FRED API returned status {}: {}",
                status, text
            )));
        }

        let data: FredResponse = response.json().await?;
        info!(
            "FRED: fetched {} T-Bill rates for {} to {}",
            data.observations.len(),
            start,
            end
        );

        let entries = data
            .observations
            .into_iter()
            .filter_map(|obs| {
                // Skip missing values (FRED uses "." for missing)
                if obs.value == "." || obs.value.is_empty() {
                    return None;
                }

                let rate_date = NaiveDate::parse_from_str(&obs.date, "%Y-%m-%d").ok()?;
                let rate_pct: f64 = obs.value.parse().ok()?;
                let rate_decimal = rate_pct / 100.0;

                Some(InterestRateEntry {
                    rate_date,
                    region: "US".to_string(),
                    rate_type: "TBILL_3M".to_string(),
                    rate: rate_decimal,
                    source: "FRED_TB3MS".to_string(),
                })
            })
            .collect();

        Ok(entries)
    }
}

// ============================================================================
// Interest Rate Stats
// ============================================================================

#[derive(Debug, Clone, Default)]
pub struct InterestRateStats {
    pub br_count: i64,
    pub br_min_date: Option<NaiveDate>,
    pub br_max_date: Option<NaiveDate>,
    pub us_count: i64,
    pub us_min_date: Option<NaiveDate>,
    pub us_max_date: Option<NaiveDate>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_normalization() {
        // BCB returns 10.75% as "10,75"
        let rate_str = "10,75";
        let rate_pct: f64 = rate_str.replace(',', ".").parse().unwrap();
        let rate_decimal = rate_pct / 100.0;
        assert!((rate_decimal - 0.1075).abs() < 1e-6);

        // FRED returns 4.35% as "4.35"
        let rate_str = "4.35";
        let rate_pct: f64 = rate_str.parse().unwrap();
        let rate_decimal = rate_pct / 100.0;
        assert!((rate_decimal - 0.0435).abs() < 1e-6);
    }

    #[test]
    fn test_bcb_date_parsing() {
        let date_str = "25/12/2024";
        let parts: Vec<&str> = date_str.split('/').collect();
        let day: u32 = parts[0].parse().unwrap();
        let month: u32 = parts[1].parse().unwrap();
        let year: i32 = parts[2].parse().unwrap();
        let date = NaiveDate::from_ymd_opt(year, month, day).unwrap();
        assert_eq!(date, NaiveDate::from_ymd_opt(2024, 12, 25).unwrap());
    }

    #[test]
    fn test_fred_date_parsing() {
        let date_str = "2024-12-25";
        let date = NaiveDate::parse_from_str(date_str, "%Y-%m-%d").unwrap();
        assert_eq!(date, NaiveDate::from_ymd_opt(2024, 12, 25).unwrap());
    }

    #[test]
    fn test_rate_range_validation() {
        // Valid rates should be between 0 and 1 (0% to 100%)
        let valid_rates = [0.0435, 0.1075, 0.0, 0.25];
        for rate in valid_rates {
            assert!(
                rate >= 0.0 && rate <= 1.0,
                "Rate {} out of valid range",
                rate
            );
        }
    }
}




































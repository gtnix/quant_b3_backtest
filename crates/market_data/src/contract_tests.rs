//! Provider Contract Tests - validates List -> Fetch sanity.
//!
//! Ensures tickers from /api/quote/list can be fetched via /api/quote/{ticker}.

use chrono::{DateTime, Utc};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::{debug, info, warn};

use crate::brapi::{BrapiClient, BrapiError, ReconciliationResult};
use crate::db::{Database, DbError};

// ============================================================================
// Contract Test Result
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractTestResult {
    pub test_id: String,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub total_active: usize,
    pub sample_tested: usize,
    pub successful: usize,
    pub not_found_listed: usize, // 404 for tickers that were in list (divergence)
    pub other_errors: usize,
    pub duration_secs: f64,
    pub failures: Vec<ContractFailure>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractFailure {
    pub ticker: String,
    pub error_type: String,
    pub error_message: String,
    pub was_listed: bool,
    pub timestamp: DateTime<Utc>,
}

impl ContractTestResult {
    pub fn write_manifest(&self, output_dir: &Path) -> std::io::Result<()> {
        let path = output_dir.join("provider_contract_manifest.json");
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(&path, json)?;
        info!("Contract manifest written to {}", path.display());
        Ok(())
    }
}

// ============================================================================
// Provider Contract Test
// ============================================================================

pub struct ProviderContractTest {
    client: BrapiClient,
    db: Database,
}

impl ProviderContractTest {
    pub async fn new() -> Result<Self, DbError> {
        let client = BrapiClient::new().map_err(|e| DbError::Config(e.to_string()))?;
        let db = Database::connect().await?;
        Ok(Self { client, db })
    }

    /// Run sanity check: sample ACTIVE tickers and try to fetch them.
    pub async fn run_sanity_check(
        &self,
        sample_size: usize,
    ) -> Result<ContractTestResult, DbError> {
        let started_at = Utc::now();
        let test_id = format!("contract_{}", started_at.format("%Y%m%d_%H%M%S"));

        info!("Starting contract test: {}", test_id);

        // Get all ACTIVE tickers
        let active_tickers = self.db.get_active_tickers().await?;
        let total_active = active_tickers.len();

        if total_active == 0 {
            warn!("No ACTIVE tickers in universe! Run 'universe-refresh' first.");
            return Ok(ContractTestResult {
                test_id,
                started_at,
                completed_at: Some(Utc::now()),
                total_active: 0,
                sample_tested: 0,
                successful: 0,
                not_found_listed: 0,
                other_errors: 0,
                duration_secs: 0.0,
                failures: Vec::new(),
            });
        }

        // Sample tickers
        let mut rng = rand::rng();
        let sample: Vec<String> = if sample_size >= total_active {
            active_tickers.clone()
        } else {
            active_tickers
                .choose_multiple(&mut rng, sample_size)
                .cloned()
                .collect()
        };

        info!(
            "Testing {} tickers (from {} active)",
            sample.len(),
            total_active
        );

        let mut successful = 0;
        let mut not_found_listed = 0;
        let mut other_errors = 0;
        let mut failures = Vec::new();

        for ticker in &sample {
            debug!("Testing ticker: {}", ticker);

            match self.client.fetch_quote_with_metadata(ticker).await {
                Ok(_) => {
                    successful += 1;
                    // Mark as validated
                    let _ = self.db.mark_ticker_validated(ticker).await;
                }
                Err(BrapiError::NotFound { .. }) => {
                    not_found_listed += 1;
                    warn!("404 for listed ticker: {}", ticker);

                    // Run reconciliation to determine cause
                    let reconciliation_result = match self.client.reconcile_404(ticker).await {
                        Ok(result) => result,
                        Err(_) => ReconciliationResult::ListedBut404, // Assume listed on error
                    };

                    let (decision, error_msg) = match reconciliation_result {
                        ReconciliationResult::ListedBut404 => {
                            let _ = self
                                .db
                                .mark_ticker_suspect(
                                    ticker,
                                    "LISTED_BUT_404",
                                    "Listed but returned 404",
                                )
                                .await;
                            (
                                "MARKED_SUSPECT",
                                "Ticker in list but returned 404 - marked SUSPECT",
                            )
                        }
                        ReconciliationResult::RemovedFromProvider => {
                            let _ = self
                                .db
                                .mark_ticker_inactive(ticker, "Removed from provider list")
                                .await;
                            (
                                "MARKED_INACTIVE",
                                "Ticker removed from provider list - marked INACTIVE",
                            )
                        }
                    };

                    let _ = self
                        .db
                        .log_divergence(
                            ticker,
                            "RECONCILED_404",
                            true,
                            true,
                            Some(&format!("{:?}", reconciliation_result)),
                            Some(decision),
                        )
                        .await;

                    failures.push(ContractFailure {
                        ticker: ticker.clone(),
                        error_type: "NOT_FOUND".to_string(),
                        error_message: error_msg.to_string(),
                        was_listed: true,
                        timestamp: Utc::now(),
                    });
                }
                Err(e) => {
                    other_errors += 1;
                    warn!("Error fetching {}: {}", ticker, e);

                    failures.push(ContractFailure {
                        ticker: ticker.clone(),
                        error_type: "OTHER".to_string(),
                        error_message: e.to_string(),
                        was_listed: true,
                        timestamp: Utc::now(),
                    });
                }
            }

            // Rate limit: small delay between requests
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        let completed_at = Utc::now();
        let duration = (completed_at - started_at).num_milliseconds() as f64 / 1000.0;

        info!(
            "Contract test complete: {}/{} successful, {} 404, {} other errors",
            successful,
            sample.len(),
            not_found_listed,
            other_errors
        );

        Ok(ContractTestResult {
            test_id,
            started_at,
            completed_at: Some(completed_at),
            total_active,
            sample_tested: sample.len(),
            successful,
            not_found_listed,
            other_errors,
            duration_secs: duration,
            failures,
        })
    }
}

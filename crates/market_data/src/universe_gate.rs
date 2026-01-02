//! Universe Gate - Safe Fetch validation layer.
//!
//! Ensures all API calls go through universe validation first.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use thiserror::Error;
use tracing::{debug, warn};

use crate::db::{Database, DbError};

// ============================================================================
// Gate Errors
// ============================================================================

#[derive(Error, Debug)]
pub enum GateError {
    #[error("Ticker '{ticker}' not found in provider universe")]
    NotListed { ticker: String },

    #[error("Ticker '{ticker}' is INACTIVE (reason: {reason})")]
    Inactive { ticker: String, reason: String },

    #[error("Ticker '{ticker}' is SUSPECT (needs reconciliation)")]
    Suspect { ticker: String },

    #[error("Database error: {0}")]
    Database(#[from] DbError),
}

// ============================================================================
// Validation Result
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum ValidationResult {
    /// Ticker is ACTIVE and ready for API calls
    Active,
    /// Ticker is INACTIVE but allowed (with --allow-inactive)
    InactiveAllowed,
    /// Ticker is not in the universe
    NotListed,
    /// Ticker is INACTIVE and blocked
    InactiveBlocked,
    /// Ticker is SUSPECT (404 on ACTIVE)
    Suspect,
}

// ============================================================================
// Gate Stats
// ============================================================================

#[derive(Debug, Default)]
pub struct GateStats {
    pub allowed: AtomicUsize,
    pub blocked_not_listed: AtomicUsize,
    pub blocked_inactive: AtomicUsize,
    pub blocked_suspect: AtomicUsize,
}

impl GateStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn summary(&self) -> GateStatsSummary {
        GateStatsSummary {
            allowed: self.allowed.load(Ordering::Relaxed),
            blocked_not_listed: self.blocked_not_listed.load(Ordering::Relaxed),
            blocked_inactive: self.blocked_inactive.load(Ordering::Relaxed),
            blocked_suspect: self.blocked_suspect.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct GateStatsSummary {
    pub allowed: usize,
    pub blocked_not_listed: usize,
    pub blocked_inactive: usize,
    pub blocked_suspect: usize,
}

impl GateStatsSummary {
    pub fn total_blocked(&self) -> usize {
        self.blocked_not_listed + self.blocked_inactive + self.blocked_suspect
    }
}

// ============================================================================
// Universe Gate
// ============================================================================

/// Gate that validates tickers against provider_universe before API calls.
pub struct UniverseGate {
    db: Database,
    allow_inactive: bool,
    stats: Arc<GateStats>,
}

impl UniverseGate {
    /// Create a new universe gate.
    pub async fn new(allow_inactive: bool) -> Result<Self, DbError> {
        let db = Database::connect().await?;
        Ok(Self {
            db,
            allow_inactive,
            stats: Arc::new(GateStats::new()),
        })
    }

    /// Create gate with existing database connection.
    pub fn with_db(db: Database, allow_inactive: bool) -> Self {
        Self {
            db,
            allow_inactive,
            stats: Arc::new(GateStats::new()),
        }
    }

    /// Validate a single ticker.
    pub async fn validate(&self, ticker: &str) -> Result<ValidationResult, GateError> {
        let status = self.db.get_ticker_status(ticker).await?;

        match status {
            None => {
                self.stats
                    .blocked_not_listed
                    .fetch_add(1, Ordering::Relaxed);
                debug!("BLOCKED: {} not in provider_universe", ticker);
                Ok(ValidationResult::NotListed)
            }
            Some(s) if s.status == "ACTIVE" => {
                self.stats.allowed.fetch_add(1, Ordering::Relaxed);
                Ok(ValidationResult::Active)
            }
            Some(s) if s.status == "INACTIVE" => {
                if self.allow_inactive {
                    self.stats.allowed.fetch_add(1, Ordering::Relaxed);
                    warn!("ALLOWED_INACTIVE: {} (allow_inactive=true)", ticker);
                    Ok(ValidationResult::InactiveAllowed)
                } else {
                    self.stats.blocked_inactive.fetch_add(1, Ordering::Relaxed);
                    debug!("BLOCKED: {} is INACTIVE", ticker);
                    Ok(ValidationResult::InactiveBlocked)
                }
            }
            Some(s) if s.status == "SUSPECT" => {
                self.stats.blocked_suspect.fetch_add(1, Ordering::Relaxed);
                debug!("BLOCKED: {} is SUSPECT", ticker);
                Ok(ValidationResult::Suspect)
            }
            Some(_) => {
                // Unknown status, treat as not listed
                self.stats
                    .blocked_not_listed
                    .fetch_add(1, Ordering::Relaxed);
                Ok(ValidationResult::NotListed)
            }
        }
    }

    /// Validate and return error if blocked.
    pub async fn require_active(&self, ticker: &str) -> Result<(), GateError> {
        match self.validate(ticker).await? {
            ValidationResult::Active => Ok(()),
            ValidationResult::InactiveAllowed => Ok(()),
            ValidationResult::NotListed => Err(GateError::NotListed {
                ticker: ticker.to_string(),
            }),
            ValidationResult::InactiveBlocked => Err(GateError::Inactive {
                ticker: ticker.to_string(),
                reason: "Ticker marked INACTIVE".to_string(),
            }),
            ValidationResult::Suspect => Err(GateError::Suspect {
                ticker: ticker.to_string(),
            }),
        }
    }

    /// Validate multiple tickers and return only valid ones.
    pub async fn filter_active(&self, tickers: &[&str]) -> Result<Vec<String>, GateError> {
        let mut valid = Vec::new();
        for ticker in tickers {
            match self.validate(ticker).await? {
                ValidationResult::Active | ValidationResult::InactiveAllowed => {
                    valid.push(ticker.to_string());
                }
                _ => {}
            }
        }
        Ok(valid)
    }

    /// Get gate statistics.
    pub fn stats(&self) -> GateStatsSummary {
        self.stats.summary()
    }

    /// Get database reference.
    pub fn db(&self) -> &Database {
        &self.db
    }

    /// Log blocked call for audit.
    pub async fn log_blocked(&self, ticker: &str, reason: &str) -> Result<(), DbError> {
        self.db
            .log_divergence(
                ticker,
                "BLOCKED_INVALID_TICKER",
                false,
                false,
                None,
                Some(reason),
            )
            .await
    }
}

// ============================================================================
// Batch Validator
// ============================================================================

/// Validates a batch of tickers and returns categorized results.
pub struct BatchValidationResult {
    pub active: Vec<String>,
    pub inactive: Vec<String>,
    pub not_listed: Vec<String>,
    pub suspect: Vec<String>,
}

impl BatchValidationResult {
    pub fn all_valid(&self) -> bool {
        self.not_listed.is_empty() && self.inactive.is_empty() && self.suspect.is_empty()
    }
}

impl UniverseGate {
    /// Validate a batch of tickers.
    pub async fn validate_batch(
        &self,
        tickers: &[&str],
    ) -> Result<BatchValidationResult, GateError> {
        let mut result = BatchValidationResult {
            active: Vec::new(),
            inactive: Vec::new(),
            not_listed: Vec::new(),
            suspect: Vec::new(),
        };

        for ticker in tickers {
            match self.validate(ticker).await? {
                ValidationResult::Active | ValidationResult::InactiveAllowed => {
                    result.active.push(ticker.to_string());
                }
                ValidationResult::InactiveBlocked => {
                    result.inactive.push(ticker.to_string());
                }
                ValidationResult::NotListed => {
                    result.not_listed.push(ticker.to_string());
                }
                ValidationResult::Suspect => {
                    result.suspect.push(ticker.to_string());
                }
            }
        }

        if !result.not_listed.is_empty() {
            warn!(
                "Blocked {} tickers not in universe: {:?}",
                result.not_listed.len(),
                &result.not_listed[..result.not_listed.len().min(5)]
            );
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_stats() {
        let stats = GateStats::new();
        stats.allowed.fetch_add(5, Ordering::Relaxed);
        stats.blocked_not_listed.fetch_add(2, Ordering::Relaxed);

        let summary = stats.summary();
        assert_eq!(summary.allowed, 5);
        assert_eq!(summary.blocked_not_listed, 2);
        assert_eq!(summary.total_blocked(), 2);
    }
}



























//! Timestamp Validator - Validates bar timestamps against market sessions.
//!
//! Checks if bars fall within valid trading sessions and identifies outliers.

use chrono::{DateTime, Utc};

use super::{Market, MarketSessionCalendar, Severity, TimezoneResolver};

/// Result of timestamp validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimestampValidation {
    /// Timestamp is valid for the market session
    Valid,
    /// Timestamp is outside the regular trading session
    OutsideSession {
        session_start: DateTime<Utc>,
        session_end: DateTime<Utc>,
    },
    /// Timestamp is before market opens
    BeforeMarketOpen { market_open: DateTime<Utc> },
    /// Timestamp is after market closes
    AfterMarketClose { market_close: DateTime<Utc> },
    /// Timestamp could be ambiguous due to DST transition
    DSTAmbiguous { possible_times: Vec<DateTime<Utc>> },
    /// Date is not a trading day
    NonTradingDay { reason: String },
    /// Invalid or malformed timestamp
    Invalid { reason: String },
}

impl TimestampValidation {
    /// Check if the validation passed.
    pub fn is_valid(&self) -> bool {
        matches!(self, TimestampValidation::Valid)
    }

    /// Get severity for logging.
    pub fn severity(&self) -> Severity {
        match self {
            TimestampValidation::Valid => Severity::Info,
            TimestampValidation::OutsideSession { .. } => Severity::Warn,
            TimestampValidation::BeforeMarketOpen { .. } => Severity::Error,
            TimestampValidation::AfterMarketClose { .. } => Severity::Warn,
            TimestampValidation::DSTAmbiguous { .. } => Severity::Warn,
            TimestampValidation::NonTradingDay { .. } => Severity::Warn,
            TimestampValidation::Invalid { .. } => Severity::Error,
        }
    }

    /// Format as log code.
    pub fn code(&self) -> String {
        match self {
            TimestampValidation::Valid => "VALID".to_string(),
            TimestampValidation::OutsideSession { .. } => "OUTSIDE_SESSION".to_string(),
            TimestampValidation::BeforeMarketOpen { .. } => "BEFORE_OPEN".to_string(),
            TimestampValidation::AfterMarketClose { .. } => "AFTER_CLOSE".to_string(),
            TimestampValidation::DSTAmbiguous { .. } => "DST_AMBIGUOUS".to_string(),
            TimestampValidation::NonTradingDay { reason } => format!("NON_TRADING:{}", reason),
            TimestampValidation::Invalid { reason } => format!("INVALID:{}", reason),
        }
    }
}

/// Timestamp validator for market data bars.
#[derive(Debug)]
pub struct TimestampValidator {
    calendar: MarketSessionCalendar,
    timezone_resolver: TimezoneResolver,
}

impl TimestampValidator {
    /// Create a new validator.
    pub fn new() -> Self {
        Self {
            calendar: MarketSessionCalendar::new(),
            timezone_resolver: TimezoneResolver::new(),
        }
    }

    /// Create with a custom calendar.
    pub fn with_calendar(calendar: MarketSessionCalendar) -> Self {
        Self {
            timezone_resolver: TimezoneResolver::new(),
            calendar,
        }
    }

    /// Validate an intraday bar timestamp.
    ///
    /// Checks if the timestamp falls within the valid trading session
    /// for the specified market.
    pub fn validate(&self, market: Market, timestamp: DateTime<Utc>) -> TimestampValidation {
        // Convert to local time
        let local = self.timezone_resolver.to_local(market, timestamp);
        let date = local.date_naive();
        let time = local.time();

        // Check if trading day
        if !self.calendar.is_trading_day(market, date) {
            let classification = self.calendar.classify_date(market, date);
            return TimestampValidation::NonTradingDay {
                reason: classification.closure_reason().unwrap_or_else(|| "Unknown".to_string()),
            };
        }

        // Get session
        let session = match self.calendar.get_session(market, date) {
            Some(s) => s,
            None => {
                return TimestampValidation::NonTradingDay {
                    reason: "No session info".to_string(),
                }
            }
        };

        // Calculate session times in UTC
        let session_start_utc = self.timezone_resolver
            .to_utc(market, date, session.earliest_start())
            .unwrap_or(timestamp);
        let session_end_utc = self.timezone_resolver
            .to_utc(market, date, session.latest_end())
            .unwrap_or(timestamp);

        // Check if before market open
        if time < session.earliest_start() {
            return TimestampValidation::BeforeMarketOpen {
                market_open: session_start_utc,
            };
        }

        // Check if after market close
        if time > session.latest_end() {
            return TimestampValidation::AfterMarketClose {
                market_close: session_end_utc,
            };
        }

        // Check if within regular session
        if !session.is_regular_hours(time) && !session.is_trading_hours(time) {
            return TimestampValidation::OutsideSession {
                session_start: session_start_utc,
                session_end: session_end_utc,
            };
        }

        TimestampValidation::Valid
    }

    /// Validate a batch of timestamps.
    ///
    /// Returns a list of (timestamp, validation result) pairs.
    pub fn validate_batch(
        &self,
        market: Market,
        timestamps: &[DateTime<Utc>],
    ) -> Vec<(DateTime<Utc>, TimestampValidation)> {
        timestamps
            .iter()
            .map(|&ts| (ts, self.validate(market, ts)))
            .collect()
    }

    /// Count validation results by type.
    pub fn count_validations(
        &self,
        results: &[(DateTime<Utc>, TimestampValidation)],
    ) -> ValidationCounts {
        let mut counts = ValidationCounts::default();

        for (_, validation) in results {
            match validation {
                TimestampValidation::Valid => counts.valid += 1,
                TimestampValidation::OutsideSession { .. } => counts.outside_session += 1,
                TimestampValidation::BeforeMarketOpen { .. } => counts.before_open += 1,
                TimestampValidation::AfterMarketClose { .. } => counts.after_close += 1,
                TimestampValidation::DSTAmbiguous { .. } => counts.dst_ambiguous += 1,
                TimestampValidation::NonTradingDay { .. } => counts.non_trading_day += 1,
                TimestampValidation::Invalid { .. } => counts.invalid += 1,
            }
        }

        counts
    }

    /// Filter to get only outlier timestamps.
    pub fn get_outliers<'a>(
        &self,
        results: &'a [(DateTime<Utc>, TimestampValidation)],
    ) -> Vec<(&'a DateTime<Utc>, &'a TimestampValidation)> {
        results
            .iter()
            .filter(|(_, v)| !v.is_valid())
            .map(|(ts, v)| (ts, v))
            .collect()
    }
}

impl Default for TimestampValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Counts of validation results.
#[derive(Debug, Clone, Default)]
pub struct ValidationCounts {
    pub valid: usize,
    pub outside_session: usize,
    pub before_open: usize,
    pub after_close: usize,
    pub dst_ambiguous: usize,
    pub non_trading_day: usize,
    pub invalid: usize,
}

impl ValidationCounts {
    /// Total number of validations.
    pub fn total(&self) -> usize {
        self.valid
            + self.outside_session
            + self.before_open
            + self.after_close
            + self.dst_ambiguous
            + self.non_trading_day
            + self.invalid
    }

    /// Number of outliers (non-valid).
    pub fn outliers(&self) -> usize {
        self.total() - self.valid
    }

    /// Percentage of valid timestamps.
    pub fn valid_pct(&self) -> f64 {
        if self.total() == 0 {
            100.0
        } else {
            (self.valid as f64 / self.total() as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    fn utc(y: i32, m: u32, d: u32, h: u32, min: u32) -> DateTime<Utc> {
        NaiveDate::from_ymd_opt(y, m, d)
            .unwrap()
            .and_hms_opt(h, min, 0)
            .unwrap()
            .and_utc()
    }

    #[test]
    fn test_valid_timestamp_b3() {
        let validator = TimestampValidator::new();

        // 10:00 BRT = 13:00 UTC - market open
        let result = validator.validate(Market::BR, utc(2024, 12, 23, 13, 0));
        assert!(result.is_valid());

        // 14:00 BRT = 17:00 UTC - middle of session
        let result = validator.validate(Market::BR, utc(2024, 12, 23, 17, 0));
        assert!(result.is_valid());
    }

    #[test]
    fn test_valid_timestamp_nyse() {
        let validator = TimestampValidator::new();

        // 09:30 EST = 14:30 UTC (winter) - Jan 16, 2024 (not MLK Day)
        let result = validator.validate(Market::US, utc(2024, 1, 16, 14, 30));
        assert!(result.is_valid());

        // 15:00 EST = 20:00 UTC - near close
        let result = validator.validate(Market::US, utc(2024, 1, 16, 20, 0));
        assert!(result.is_valid());
    }

    #[test]
    fn test_before_market_open() {
        let validator = TimestampValidator::new();

        // 08:00 BRT = 11:00 UTC - before B3 opens
        let result = validator.validate(Market::BR, utc(2024, 12, 23, 11, 0));
        assert!(matches!(result, TimestampValidation::BeforeMarketOpen { .. }));
    }

    #[test]
    fn test_after_market_close() {
        let validator = TimestampValidator::new();

        // 22:00 BRT = 01:00 UTC next day - after B3 closes
        // Actually this might be interpreted as the next day
        // Let's use 19:00 BRT = 22:00 UTC same day
        let result = validator.validate(Market::BR, utc(2024, 12, 23, 22, 0));
        assert!(matches!(result, TimestampValidation::AfterMarketClose { .. }));
    }

    #[test]
    fn test_non_trading_day_weekend() {
        let validator = TimestampValidator::new();

        // Saturday
        let result = validator.validate(Market::BR, utc(2024, 12, 21, 13, 0));
        assert!(matches!(result, TimestampValidation::NonTradingDay { .. }));
    }

    #[test]
    fn test_non_trading_day_holiday() {
        let validator = TimestampValidator::new();

        // Christmas
        let result = validator.validate(Market::BR, utc(2024, 12, 25, 13, 0));
        assert!(matches!(result, TimestampValidation::NonTradingDay { .. }));
    }

    #[test]
    fn test_validation_severity() {
        assert_eq!(TimestampValidation::Valid.severity(), Severity::Info);
        assert_eq!(
            TimestampValidation::BeforeMarketOpen { market_open: Utc::now() }.severity(),
            Severity::Error
        );
        assert_eq!(
            TimestampValidation::AfterMarketClose { market_close: Utc::now() }.severity(),
            Severity::Warn
        );
    }

    #[test]
    fn test_validate_batch() {
        let validator = TimestampValidator::new();

        let timestamps = vec![
            utc(2024, 12, 23, 13, 0), // Valid
            utc(2024, 12, 23, 17, 0), // Valid
            utc(2024, 12, 21, 13, 0), // Weekend
            utc(2024, 12, 25, 13, 0), // Holiday
        ];

        let results = validator.validate_batch(Market::BR, &timestamps);
        assert_eq!(results.len(), 4);

        let counts = validator.count_validations(&results);
        assert_eq!(counts.valid, 2);
        assert_eq!(counts.non_trading_day, 2);
    }

    #[test]
    fn test_get_outliers() {
        let validator = TimestampValidator::new();

        let timestamps = vec![
            utc(2024, 12, 23, 13, 0), // Valid
            utc(2024, 12, 21, 13, 0), // Weekend
        ];

        let results = validator.validate_batch(Market::BR, &timestamps);
        let outliers = validator.get_outliers(&results);

        assert_eq!(outliers.len(), 1);
    }

    #[test]
    fn test_validation_counts() {
        let counts = ValidationCounts {
            valid: 90,
            outside_session: 5,
            before_open: 2,
            after_close: 3,
            dst_ambiguous: 0,
            non_trading_day: 0,
            invalid: 0,
        };

        assert_eq!(counts.total(), 100);
        assert_eq!(counts.outliers(), 10);
        assert_eq!(counts.valid_pct(), 90.0);
    }
}


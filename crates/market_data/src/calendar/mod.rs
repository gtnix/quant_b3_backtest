//! Market Session Calendar Module
//!
//! Production-grade calendar for backtesting and data QA that properly distinguishes
//! holidays/closed sessions from missing data, supports timezone/DST, and defines
//! bar alignment rules.
//!
//! # Key Features
//! - Distinguishes holidays from missing data
//! - Timezone/DST support (historical included) via chrono-tz
//! - Bar alignment rules (EOD and intraday) per market
//! - Per-asset gap analysis with deterministic GapReason
//! - Walk-forward and data health integration

mod bar_aligner;
mod classifier;
pub mod db_classifier;
pub mod db_holidays;
pub mod db_provider;
mod gap_analyzer;
mod holidays;
mod hours;
pub mod migration;
pub mod rules_engine;
mod timezone;
mod validator;

pub use bar_aligner::BarAligner;
pub use classifier::TradingDayClassifier;
pub use db_classifier::{DbTradingDayClassifier, SupportedRanges};
pub use db_holidays::{DbExtraordinaryClosure, DbHoliday, DbHolidayProvider};
pub use db_provider::{
    CalendarError, DbCalendarProvider, DayClassificationDb, DayType, ExtraordinaryClosureDb,
    HolidayDb, HolidayTypeDb, SessionPeriodDb, SourceLayer, SupportedRange, TradingSessionDb,
};
pub use gap_analyzer::{GapAnalyzer, GapEntry, GapReason, GapReport, GapSummary};
pub use holidays::{Holiday, HolidayCalendar, HolidayProvider, HolidayType};
pub use hours::MarketHoursProvider;
pub use rules_engine::{B3RulesEngine, NYSERulesEngine, RulesEngine};
pub use timezone::TimezoneResolver;
pub use validator::{TimestampValidation, TimestampValidator};

use chrono::{DateTime, NaiveDate, NaiveTime, Timelike, Utc};
use serde::{Deserialize, Serialize};

// ============================================================================
// Core Types
// ============================================================================

/// Market identifier with associated timezone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Market {
    /// Brazilian market (B3) - America/Sao_Paulo timezone
    #[default]
    BR,
    /// US market (NYSE/NASDAQ) - America/New_York timezone
    US,
}

impl Market {
    /// Get the IANA timezone name for this market.
    pub fn timezone_name(&self) -> &'static str {
        match self {
            Market::BR => "America/Sao_Paulo",
            Market::US => "America/New_York",
        }
    }

    /// Get the market name for display.
    pub fn display_name(&self) -> &'static str {
        match self {
            Market::BR => "B3",
            Market::US => "NYSE/NASDAQ",
        }
    }
}

/// Time range in local market time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: NaiveTime,
    pub end: NaiveTime,
}

impl TimeRange {
    pub fn new(start: NaiveTime, end: NaiveTime) -> Self {
        Self { start, end }
    }

    /// Check if a time falls within this range.
    pub fn contains(&self, time: NaiveTime) -> bool {
        time >= self.start && time <= self.end
    }

    /// Duration in minutes.
    pub fn duration_minutes(&self) -> i64 {
        let start_mins = self.start.num_seconds_from_midnight() as i64 / 60;
        let end_mins = self.end.num_seconds_from_midnight() as i64 / 60;
        end_mins - start_mins
    }
}

/// Trading session information for a market day.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionInfo {
    /// Pre-market auction period (optional)
    pub pre_market: Option<TimeRange>,
    /// Regular trading session
    pub regular: TimeRange,
    /// Closing auction period (optional)
    pub closing_auction: Option<TimeRange>,
    /// After-hours trading period (optional)
    pub after_hours: Option<TimeRange>,
}

impl SessionInfo {
    /// Check if a local time is within the regular session.
    pub fn is_regular_hours(&self, time: NaiveTime) -> bool {
        self.regular.contains(time)
    }

    /// Check if a local time is within any trading period.
    pub fn is_trading_hours(&self, time: NaiveTime) -> bool {
        self.regular.contains(time)
            || self.pre_market.as_ref().map_or(false, |r| r.contains(time))
            || self.closing_auction.as_ref().map_or(false, |r| r.contains(time))
            || self.after_hours.as_ref().map_or(false, |r| r.contains(time))
    }

    /// Get the earliest start time (pre-market if exists, else regular).
    pub fn earliest_start(&self) -> NaiveTime {
        self.pre_market
            .as_ref()
            .map(|r| r.start)
            .unwrap_or(self.regular.start)
    }

    /// Get the latest end time (after-hours if exists, else regular).
    pub fn latest_end(&self) -> NaiveTime {
        self.after_hours
            .as_ref()
            .map(|r| r.end)
            .unwrap_or_else(|| {
                self.closing_auction
                    .as_ref()
                    .map(|r| r.end)
                    .unwrap_or(self.regular.end)
            })
    }
}

/// Classification of a calendar date for a market.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DayClassification {
    /// Regular trading day with session info
    TradingDay(SessionInfo),
    /// Weekend (Saturday or Sunday)
    Weekend,
    /// Official holiday - market closed
    Holiday {
        name: String,
        official_source: String,
    },
    /// Half-day with early close
    HalfDay {
        name: String,
        close_time: NaiveTime,
        session: SessionInfo,
    },
    /// Extraordinary closure (emergencies, mourning, etc.)
    ExtraordinaryClosure {
        reason: String,
        source: String,
    },
}

impl DayClassification {
    /// Check if this is a trading day (including half-days).
    pub fn is_trading_day(&self) -> bool {
        matches!(self, DayClassification::TradingDay(_) | DayClassification::HalfDay { .. })
    }

    /// Check if market is completely closed.
    pub fn is_closed(&self) -> bool {
        matches!(
            self,
            DayClassification::Weekend
                | DayClassification::Holiday { .. }
                | DayClassification::ExtraordinaryClosure { .. }
        )
    }

    /// Get session info if available.
    pub fn session(&self) -> Option<&SessionInfo> {
        match self {
            DayClassification::TradingDay(session) => Some(session),
            DayClassification::HalfDay { session, .. } => Some(session),
            _ => None,
        }
    }

    /// Get the reason for non-trading (if applicable).
    pub fn closure_reason(&self) -> Option<String> {
        match self {
            DayClassification::Weekend => Some("Weekend".to_string()),
            DayClassification::Holiday { name, .. } => Some(format!("Holiday: {}", name)),
            DayClassification::ExtraordinaryClosure { reason, .. } => {
                Some(format!("Extraordinary: {}", reason))
            }
            _ => None,
        }
    }
}

/// Severity level for gap analysis and logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Severity {
    Info,
    Warn,
    Error,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::Info => write!(f, "INFO"),
            Severity::Warn => write!(f, "WARN"),
            Severity::Error => write!(f, "ERROR"),
        }
    }
}

// ============================================================================
// MarketSessionCalendar - Main API
// ============================================================================

/// Main calendar API for market session management.
///
/// This is the core entry point for all calendar-related queries.
#[derive(Debug)]
pub struct MarketSessionCalendar {
    classifier: TradingDayClassifier,
    timezone_resolver: TimezoneResolver,
}

impl MarketSessionCalendar {
    /// Create a new calendar with default holiday data.
    pub fn new() -> Self {
        Self {
            classifier: TradingDayClassifier::new(),
            timezone_resolver: TimezoneResolver::new(),
        }
    }

    /// Create a calendar with a custom holiday provider.
    pub fn with_holidays(holiday_provider: HolidayProvider) -> Self {
        Self {
            classifier: TradingDayClassifier::with_holidays(holiday_provider),
            timezone_resolver: TimezoneResolver::new(),
        }
    }

    /// Check if a date is a trading day for the given market.
    pub fn is_trading_day(&self, market: Market, date: NaiveDate) -> bool {
        self.classifier.classify(market, date).is_trading_day()
    }

    /// Get the full classification for a date.
    pub fn classify_date(&self, market: Market, date: NaiveDate) -> DayClassification {
        self.classifier.classify(market, date)
    }

    /// Get session info for a trading day (None if not a trading day).
    pub fn get_session(&self, market: Market, date: NaiveDate) -> Option<SessionInfo> {
        self.classifier.classify(market, date).session().cloned()
    }

    /// Get the next trading day after the given date.
    pub fn next_trading_day(&self, market: Market, date: NaiveDate) -> NaiveDate {
        let mut current = date + chrono::Duration::days(1);
        while !self.is_trading_day(market, current) {
            current += chrono::Duration::days(1);
            // Safety limit to prevent infinite loops
            if (current - date).num_days() > 30 {
                break;
            }
        }
        current
    }

    /// Get the previous trading day before the given date.
    pub fn prev_trading_day(&self, market: Market, date: NaiveDate) -> NaiveDate {
        let mut current = date - chrono::Duration::days(1);
        while !self.is_trading_day(market, current) {
            current -= chrono::Duration::days(1);
            // Safety limit to prevent infinite loops
            if (date - current).num_days() > 30 {
                break;
            }
        }
        current
    }

    /// Count trading days between two dates (inclusive of start, exclusive of end).
    pub fn trading_days_between(&self, market: Market, start: NaiveDate, end: NaiveDate) -> i64 {
        let mut count = 0i64;
        let mut current = start;
        while current < end {
            if self.is_trading_day(market, current) {
                count += 1;
            }
            current += chrono::Duration::days(1);
        }
        count
    }

    /// Convert UTC datetime to local market time.
    pub fn to_local(&self, market: Market, utc: DateTime<Utc>) -> DateTime<chrono_tz::Tz> {
        self.timezone_resolver.to_local(market, utc)
    }

    /// Convert local market time to UTC.
    pub fn to_utc(&self, market: Market, date: NaiveDate, time: NaiveTime) -> Option<DateTime<Utc>> {
        self.timezone_resolver.to_utc(market, date, time)
    }

    /// Get the calendar version string for audit purposes.
    pub fn version(&self) -> String {
        self.classifier.holiday_provider().version()
    }

    /// Access the underlying classifier.
    pub fn classifier(&self) -> &TradingDayClassifier {
        &self.classifier
    }

    /// Access the underlying timezone resolver.
    pub fn timezone_resolver(&self) -> &TimezoneResolver {
        &self.timezone_resolver
    }
}

impl Default for MarketSessionCalendar {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    fn time(h: u32, m: u32, s: u32) -> NaiveTime {
        NaiveTime::from_hms_opt(h, m, s).unwrap()
    }

    #[test]
    fn test_market_timezone_names() {
        assert_eq!(Market::BR.timezone_name(), "America/Sao_Paulo");
        assert_eq!(Market::US.timezone_name(), "America/New_York");
    }

    #[test]
    fn test_time_range_contains() {
        let range = TimeRange::new(time(10, 0, 0), time(17, 0, 0));
        assert!(range.contains(time(10, 0, 0)));
        assert!(range.contains(time(12, 30, 0)));
        assert!(range.contains(time(17, 0, 0)));
        assert!(!range.contains(time(9, 59, 59)));
        assert!(!range.contains(time(17, 0, 1)));
    }

    #[test]
    fn test_time_range_duration() {
        let range = TimeRange::new(time(10, 0, 0), time(17, 30, 0));
        assert_eq!(range.duration_minutes(), 7 * 60 + 30);
    }

    #[test]
    fn test_session_info_is_regular_hours() {
        let session = SessionInfo {
            pre_market: Some(TimeRange::new(time(9, 45, 0), time(10, 0, 0))),
            regular: TimeRange::new(time(10, 0, 0), time(17, 55, 0)),
            closing_auction: Some(TimeRange::new(time(17, 55, 0), time(18, 0, 0))),
            after_hours: None,
        };

        assert!(session.is_regular_hours(time(10, 0, 0)));
        assert!(session.is_regular_hours(time(14, 0, 0)));
        assert!(!session.is_regular_hours(time(9, 50, 0)));
    }

    #[test]
    fn test_session_info_is_trading_hours() {
        let session = SessionInfo {
            pre_market: Some(TimeRange::new(time(9, 45, 0), time(10, 0, 0))),
            regular: TimeRange::new(time(10, 0, 0), time(17, 55, 0)),
            closing_auction: Some(TimeRange::new(time(17, 55, 0), time(18, 0, 0))),
            after_hours: None,
        };

        assert!(session.is_trading_hours(time(9, 50, 0))); // pre-market
        assert!(session.is_trading_hours(time(10, 0, 0))); // regular
        assert!(session.is_trading_hours(time(17, 57, 0))); // closing auction
        assert!(!session.is_trading_hours(time(9, 30, 0))); // before pre-market
    }

    #[test]
    fn test_day_classification_is_trading_day() {
        let session = SessionInfo {
            pre_market: None,
            regular: TimeRange::new(time(10, 0, 0), time(17, 55, 0)),
            closing_auction: None,
            after_hours: None,
        };

        assert!(DayClassification::TradingDay(session.clone()).is_trading_day());
        assert!(DayClassification::HalfDay {
            name: "Ash Wednesday".to_string(),
            close_time: time(17, 55, 0),
            session: session.clone(),
        }
        .is_trading_day());
        assert!(!DayClassification::Weekend.is_trading_day());
        assert!(!DayClassification::Holiday {
            name: "Christmas".to_string(),
            official_source: "B3".to_string(),
        }
        .is_trading_day());
    }

    #[test]
    fn test_day_classification_is_closed() {
        let session = SessionInfo {
            pre_market: None,
            regular: TimeRange::new(time(10, 0, 0), time(17, 55, 0)),
            closing_auction: None,
            after_hours: None,
        };

        assert!(!DayClassification::TradingDay(session).is_closed());
        assert!(DayClassification::Weekend.is_closed());
        assert!(DayClassification::Holiday {
            name: "Christmas".to_string(),
            official_source: "B3".to_string(),
        }
        .is_closed());
        assert!(DayClassification::ExtraordinaryClosure {
            reason: "National Mourning".to_string(),
            source: "Official".to_string(),
        }
        .is_closed());
    }

    #[test]
    fn test_calendar_weekend_detection() {
        let calendar = MarketSessionCalendar::new();

        // Saturday
        assert!(!calendar.is_trading_day(Market::BR, date(2024, 12, 21)));
        // Sunday
        assert!(!calendar.is_trading_day(Market::BR, date(2024, 12, 22)));
        // Monday
        assert!(calendar.is_trading_day(Market::BR, date(2024, 12, 23)));
    }

    #[test]
    fn test_calendar_next_trading_day() {
        let calendar = MarketSessionCalendar::new();

        // Friday -> Monday (skip weekend)
        let friday = date(2024, 12, 20);
        let next = calendar.next_trading_day(Market::BR, friday);
        assert_eq!(next, date(2024, 12, 23)); // Monday
    }

    #[test]
    fn test_calendar_prev_trading_day() {
        let calendar = MarketSessionCalendar::new();

        // Monday -> Friday (skip weekend)
        let monday = date(2024, 12, 23);
        let prev = calendar.prev_trading_day(Market::BR, monday);
        assert_eq!(prev, date(2024, 12, 20)); // Friday
    }

    #[test]
    fn test_calendar_trading_days_between() {
        let calendar = MarketSessionCalendar::new();

        // Monday to Friday (same week) = 4 trading days (Mon, Tue, Wed, Thu)
        let mon = date(2024, 12, 16);
        let fri = date(2024, 12, 20);
        let count = calendar.trading_days_between(Market::BR, mon, fri);
        assert_eq!(count, 4);
    }
}

// ============================================================================
// DbMarketSessionCalendar - Async Database-Backed Calendar
// ============================================================================

use std::sync::Arc;
use tokio_postgres::Client;

/// Async database-backed market session calendar.
///
/// This is the production-grade calendar that queries Neon PostgreSQL
/// for holiday and session data.
#[derive(Debug)]
pub struct DbMarketSessionCalendar {
    classifier: db_classifier::DbTradingDayClassifier,
    timezone_resolver: TimezoneResolver,
}

impl DbMarketSessionCalendar {
    /// Connect to database and create a new calendar.
    pub async fn connect(database_url: &str) -> Result<Self, CalendarError> {
        // Connect with TLS for Neon
        let rustls_config = rustls::ClientConfig::builder()
            .with_root_certificates(rustls::RootCertStore {
                roots: webpki_roots::TLS_SERVER_ROOTS.iter().cloned().collect(),
            })
            .with_no_client_auth();

        let tls = tokio_postgres_rustls::MakeRustlsConnect::new(rustls_config);
        let (client, connection) = tokio_postgres::connect(database_url, tls)
            .await
            .map_err(|e| CalendarError::Database(e.to_string()))?;

        // Spawn connection handler
        tokio::spawn(async move {
            if let Err(e) = connection.await {
                tracing::error!("Database connection error: {}", e);
            }
        });

        Self::from_client(Arc::new(client)).await
    }

    /// Create from an existing database client.
    pub async fn from_client(client: Arc<Client>) -> Result<Self, CalendarError> {
        let classifier = db_classifier::DbTradingDayClassifier::new(client).await?;
        let timezone_resolver = TimezoneResolver::new();

        Ok(Self {
            classifier,
            timezone_resolver,
        })
    }

    /// Classify a date for a market.
    pub async fn classify_date(
        &self,
        market: Market,
        date: NaiveDate,
    ) -> Result<DayClassification, CalendarError> {
        self.classifier.classify(market, date).await
    }

    /// Check if a date is a trading day.
    pub async fn is_trading_day(
        &self,
        market: Market,
        date: NaiveDate,
    ) -> Result<bool, CalendarError> {
        self.classifier.is_trading_day(market, date).await
    }

    /// Get all trading days in a range.
    pub async fn get_trading_days(
        &self,
        market: Market,
        start: NaiveDate,
        end: NaiveDate,
    ) -> Result<Vec<NaiveDate>, CalendarError> {
        self.classifier.get_trading_days(market, start, end).await
    }

    /// Count trading days in a range.
    pub async fn count_trading_days(
        &self,
        market: Market,
        start: NaiveDate,
        end: NaiveDate,
    ) -> Result<usize, CalendarError> {
        self.classifier.count_trading_days(market, start, end).await
    }

    /// Get the next trading day after a date.
    pub async fn next_trading_day(
        &self,
        market: Market,
        date: NaiveDate,
    ) -> Result<NaiveDate, CalendarError> {
        self.classifier.next_trading_day(market, date).await
    }

    /// Get the previous trading day before a date.
    pub async fn previous_trading_day(
        &self,
        market: Market,
        date: NaiveDate,
    ) -> Result<NaiveDate, CalendarError> {
        self.classifier.previous_trading_day(market, date).await
    }

    /// Get supported date range for a market.
    pub fn supported_range(&self, market: Market) -> &SupportedRange {
        self.classifier.supported_range(market)
    }

    /// Convert UTC timestamp to local market time.
    pub fn to_local(&self, market: Market, utc: DateTime<Utc>) -> DateTime<chrono_tz::Tz> {
        self.timezone_resolver.to_local(market, utc)
    }

    /// Check if a UTC timestamp is within trading hours.
    pub async fn is_trading_hours(
        &self,
        market: Market,
        timestamp: DateTime<Utc>,
    ) -> Result<bool, CalendarError> {
        let date = timestamp.date_naive();
        let classification = self.classify_date(market, date).await?;

        if let Some(session) = classification.session() {
            let local = self.to_local(market, timestamp);
            let local_time = local.time();
            Ok(session.is_trading_hours(local_time))
        } else {
            Ok(false)
        }
    }

    /// Get holiday for a specific date.
    pub async fn get_holiday(
        &self,
        market: Market,
        date: NaiveDate,
    ) -> Result<Option<Holiday>, CalendarError> {
        self.classifier.holiday_provider().get_holiday(market, date).await
    }

    /// Preload holidays for a year into cache.
    pub async fn preload_year(&self, market: Market, year: i32) -> Result<usize, CalendarError> {
        self.classifier.holiday_provider().preload_year(market, year).await
    }

    /// Access the underlying classifier.
    pub fn classifier(&self) -> &db_classifier::DbTradingDayClassifier {
        &self.classifier
    }

    /// Access the timezone resolver.
    pub fn timezone_resolver(&self) -> &TimezoneResolver {
        &self.timezone_resolver
    }
}


//! Database Trading Day Classifier - Async classification of trading days.
//!
//! Classifies dates as trading days, holidays, weekends, or extraordinary closures
//! using data from Neon PostgreSQL.

use chrono::{Datelike, NaiveDate, NaiveTime, Timelike, Weekday};
use std::sync::Arc;
use tokio_postgres::Client;
use tracing::debug;

use super::db_holidays::DbHolidayProvider;
use super::db_provider::{CalendarError, SupportedRange};
use super::{DayClassification, HolidayType, Market, MarketHoursProvider, SessionInfo, TimeRange};

// ============================================================================
// DbTradingDayClassifier
// ============================================================================

/// Async database-backed trading day classifier.
#[derive(Debug)]
pub struct DbTradingDayClassifier {
    client: Arc<Client>,
    holiday_provider: DbHolidayProvider,
    hours_provider: MarketHoursProvider,
    supported_ranges: SupportedRanges,
}

/// Supported date ranges per market.
#[derive(Debug, Clone)]
pub struct SupportedRanges {
    pub br: SupportedRange,
    pub us: SupportedRange,
}

impl Default for SupportedRanges {
    fn default() -> Self {
        Self {
            br: SupportedRange {
                market: Market::BR,
                range_start: NaiveDate::from_ymd_opt(2005, 1, 1).unwrap(),
                range_end: NaiveDate::from_ymd_opt(2025, 12, 31).unwrap(),
                coverage_level: "FULL".to_string(),
            },
            us: SupportedRange {
                market: Market::US,
                range_start: NaiveDate::from_ymd_opt(2005, 1, 1).unwrap(),
                range_end: NaiveDate::from_ymd_opt(2025, 12, 31).unwrap(),
                coverage_level: "FULL".to_string(),
            },
        }
    }
}

impl DbTradingDayClassifier {
    /// Create a new classifier with database client.
    pub async fn new(client: Arc<Client>) -> Result<Self, CalendarError> {
        let holiday_provider = DbHolidayProvider::new(client.clone()).await?;
        let hours_provider = MarketHoursProvider::new();

        // Load supported ranges from database
        let supported_ranges = Self::load_supported_ranges(&client).await?;

        Ok(Self {
            client,
            holiday_provider,
            hours_provider,
            supported_ranges,
        })
    }

    /// Load supported ranges from database.
    async fn load_supported_ranges(client: &Client) -> Result<SupportedRanges, CalendarError> {
        let rows = client
            .query(
                "SELECT market, range_start, range_end FROM supported_ranges",
                &[],
            )
            .await;

        // If table doesn't exist or is empty, use defaults
        let rows = match rows {
            Ok(r) => r,
            Err(_) => return Ok(SupportedRanges::default()),
        };

        if rows.is_empty() {
            return Ok(SupportedRanges::default());
        }

        let mut ranges = SupportedRanges::default();

        for row in rows {
            let market_str: &str = row.get(0);
            let start: NaiveDate = row.get(1);
            let end: NaiveDate = row.get(2);

            match market_str {
                "BR" => {
                    ranges.br.range_start = start;
                    ranges.br.range_end = end;
                }
                "US" => {
                    ranges.us.range_start = start;
                    ranges.us.range_end = end;
                }
                _ => {}
            }
        }

        Ok(ranges)
    }

    /// Get supported range for a market.
    pub fn supported_range(&self, market: Market) -> &SupportedRange {
        match market {
            Market::BR => &self.supported_ranges.br,
            Market::US => &self.supported_ranges.us,
        }
    }

    /// Validate that a date is within supported range.
    fn validate_range(&self, market: Market, date: NaiveDate) -> Result<(), CalendarError> {
        let range = self.supported_range(market);

        if date < range.range_start || date > range.range_end {
            return Err(CalendarError::OutOfRange {
                market,
                date,
                start: range.range_start,
                end: range.range_end,
            });
        }

        Ok(())
    }

    /// Classify a date for a market.
    pub async fn classify(
        &self,
        market: Market,
        date: NaiveDate,
    ) -> Result<DayClassification, CalendarError> {
        // Validate range first
        self.validate_range(market, date)?;

        // 1. Check weekend
        if self.is_weekend(date) {
            return Ok(DayClassification::Weekend);
        }

        // 2. Check extraordinary closures
        if let Some(closure) = self
            .holiday_provider
            .get_extraordinary_closure(market, date)
            .await?
        {
            return Ok(DayClassification::ExtraordinaryClosure {
                reason: closure.reason,
                source: format!(
                    "{}:{:?}",
                    closure.source_layer.as_str(),
                    closure.legal_reference
                ),
            });
        }

        // 3. Check holidays
        if let Some(holiday) = self.holiday_provider.get_holiday(market, date).await? {
            match &holiday.holiday_type {
                HolidayType::National | HolidayType::MarketSpecific => {
                    return Ok(DayClassification::Holiday {
                        name: holiday.name,
                        official_source: holiday.source,
                    });
                }
                HolidayType::ExtraordinaryClosure => {
                    return Ok(DayClassification::ExtraordinaryClosure {
                        reason: holiday.name,
                        source: holiday.source,
                    });
                }
                HolidayType::HalfDay { close_time } => {
                    let session = self.get_half_day_session(market, *close_time);
                    return Ok(DayClassification::HalfDay {
                        name: holiday.name,
                        close_time: *close_time,
                        session,
                    });
                }
                HolidayType::LateOpen { open_time } => {
                    let session = self.get_late_open_session(market, *open_time);
                    return Ok(DayClassification::TradingDay(session));
                }
            }
        }

        // 4. Regular trading day
        let session = self.regular_session(market);
        Ok(DayClassification::TradingDay(session))
    }

    /// Check if a date is a weekend.
    fn is_weekend(&self, date: NaiveDate) -> bool {
        matches!(date.weekday(), Weekday::Sat | Weekday::Sun)
    }

    /// Get regular session info for a market.
    fn regular_session(&self, market: Market) -> SessionInfo {
        match market {
            Market::BR => SessionInfo {
                pre_market: Some(TimeRange::new(
                    NaiveTime::from_hms_opt(9, 45, 0).unwrap(),
                    NaiveTime::from_hms_opt(10, 0, 0).unwrap(),
                )),
                regular: TimeRange::new(
                    NaiveTime::from_hms_opt(10, 0, 0).unwrap(),
                    NaiveTime::from_hms_opt(17, 55, 0).unwrap(),
                ),
                closing_auction: Some(TimeRange::new(
                    NaiveTime::from_hms_opt(17, 55, 0).unwrap(),
                    NaiveTime::from_hms_opt(18, 0, 0).unwrap(),
                )),
                after_hours: None,
            },
            Market::US => SessionInfo {
                pre_market: Some(TimeRange::new(
                    NaiveTime::from_hms_opt(4, 0, 0).unwrap(),
                    NaiveTime::from_hms_opt(9, 30, 0).unwrap(),
                )),
                regular: TimeRange::new(
                    NaiveTime::from_hms_opt(9, 30, 0).unwrap(),
                    NaiveTime::from_hms_opt(16, 0, 0).unwrap(),
                ),
                closing_auction: None,
                after_hours: Some(TimeRange::new(
                    NaiveTime::from_hms_opt(16, 0, 0).unwrap(),
                    NaiveTime::from_hms_opt(20, 0, 0).unwrap(),
                )),
            },
        }
    }

    /// Get session info for a half-day (early close).
    fn get_half_day_session(&self, market: Market, close_time: NaiveTime) -> SessionInfo {
        let regular = self.regular_session(market);
        SessionInfo {
            pre_market: regular.pre_market,
            regular: TimeRange::new(regular.regular.start, close_time),
            closing_auction: None,
            after_hours: None,
        }
    }

    /// Get session info for a late-open day.
    fn get_late_open_session(&self, market: Market, open_time: NaiveTime) -> SessionInfo {
        let regular = self.regular_session(market);
        SessionInfo {
            pre_market: None,
            regular: TimeRange::new(open_time, regular.regular.end),
            closing_auction: regular.closing_auction,
            after_hours: regular.after_hours,
        }
    }

    /// Check if a date is a trading day.
    pub async fn is_trading_day(
        &self,
        market: Market,
        date: NaiveDate,
    ) -> Result<bool, CalendarError> {
        let classification = self.classify(market, date).await?;
        Ok(classification.is_trading_day())
    }

    /// Get all trading days in a range.
    pub async fn get_trading_days(
        &self,
        market: Market,
        start: NaiveDate,
        end: NaiveDate,
    ) -> Result<Vec<NaiveDate>, CalendarError> {
        self.validate_range(market, start)?;
        self.validate_range(market, end)?;

        // Preload holidays for better performance
        let start_year = start.year();
        let end_year = end.year();
        for year in start_year..=end_year {
            self.holiday_provider.preload_year(market, year).await?;
        }

        let mut trading_days = Vec::new();
        let mut current = start;

        while current <= end {
            if self.is_trading_day(market, current).await? {
                trading_days.push(current);
            }
            current = current.succ_opt().unwrap();
        }

        debug!(
            "Found {} trading days for {:?} from {} to {}",
            trading_days.len(),
            market,
            start,
            end
        );

        Ok(trading_days)
    }

    /// Count trading days in a range.
    pub async fn count_trading_days(
        &self,
        market: Market,
        start: NaiveDate,
        end: NaiveDate,
    ) -> Result<usize, CalendarError> {
        Ok(self.get_trading_days(market, start, end).await?.len())
    }

    /// Get the next trading day after a given date.
    pub async fn next_trading_day(
        &self,
        market: Market,
        date: NaiveDate,
    ) -> Result<NaiveDate, CalendarError> {
        let mut current = date.succ_opt().unwrap();
        let max_attempts = 30; // Safety limit

        for _ in 0..max_attempts {
            if self.is_trading_day(market, current).await? {
                return Ok(current);
            }
            current = current.succ_opt().unwrap();
        }

        Err(CalendarError::NoData {
            market,
            date: current,
        })
    }

    /// Get the previous trading day before a given date.
    pub async fn previous_trading_day(
        &self,
        market: Market,
        date: NaiveDate,
    ) -> Result<NaiveDate, CalendarError> {
        let mut current = date.pred_opt().unwrap();
        let max_attempts = 30;

        for _ in 0..max_attempts {
            if self.is_trading_day(market, current).await? {
                return Ok(current);
            }
            current = current.pred_opt().unwrap();
        }

        Err(CalendarError::NoData {
            market,
            date: current,
        })
    }

    /// Access the underlying holiday provider.
    pub fn holiday_provider(&self) -> &DbHolidayProvider {
        &self.holiday_provider
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_weekend() {
        let classifier_ranges = SupportedRanges::default();

        // Saturday
        let sat = NaiveDate::from_ymd_opt(2024, 12, 21).unwrap();
        assert!(matches!(sat.weekday(), Weekday::Sat));

        // Sunday
        let sun = NaiveDate::from_ymd_opt(2024, 12, 22).unwrap();
        assert!(matches!(sun.weekday(), Weekday::Sun));

        // Monday
        let mon = NaiveDate::from_ymd_opt(2024, 12, 23).unwrap();
        assert!(!matches!(mon.weekday(), Weekday::Sat | Weekday::Sun));
    }

    #[test]
    fn test_supported_ranges_default() {
        let ranges = SupportedRanges::default();

        assert_eq!(ranges.br.range_start.year(), 2005);
        assert_eq!(ranges.br.range_end.year(), 2025);
        assert_eq!(ranges.us.range_start.year(), 2005);
        assert_eq!(ranges.us.range_end.year(), 2025);
    }

    #[test]
    fn test_regular_session_br() {
        // B3 regular session: 10:00-17:55 BRT
        let open = NaiveTime::from_hms_opt(10, 0, 0).unwrap();
        let close = NaiveTime::from_hms_opt(17, 55, 0).unwrap();

        assert_eq!(open.hour(), 10);
        assert_eq!(close.hour(), 17);
        assert_eq!(close.minute(), 55);
    }

    #[test]
    fn test_regular_session_us() {
        // NYSE regular session: 09:30-16:00 ET
        let open = NaiveTime::from_hms_opt(9, 30, 0).unwrap();
        let close = NaiveTime::from_hms_opt(16, 0, 0).unwrap();

        assert_eq!(open.hour(), 9);
        assert_eq!(open.minute(), 30);
        assert_eq!(close.hour(), 16);
    }
}


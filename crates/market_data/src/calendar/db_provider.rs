//! Database Calendar Provider - Neon PostgreSQL integration for calendar data.
//!
//! Provides persistent storage and retrieval of calendar data with full
//! provenance tracking and range validation.

use chrono::{Datelike, NaiveDate, NaiveTime};
use std::sync::Arc;
use thiserror::Error;
use tokio_postgres::Client;
use tracing::info;
use uuid::Uuid;

use super::Market;

// ============================================================================
// Error Types
// ============================================================================

/// Calendar-specific errors with explicit out-of-range handling.
#[derive(Debug, Error)]
pub enum CalendarError {
    #[error("Date {date} is outside supported range [{start}, {end}] for {market:?}")]
    OutOfRange {
        market: Market,
        date: NaiveDate,
        start: NaiveDate,
        end: NaiveDate,
    },

    #[error("No calendar data found for {market:?} on {date}")]
    NoData { market: Market, date: NaiveDate },

    #[error("No active calendar version found for {market:?}")]
    NoActiveVersion { market: Market },

    #[error("Database error: {0}")]
    Database(String),

    #[error("Configuration error: {0}")]
    Config(String),
}

impl From<tokio_postgres::Error> for CalendarError {
    fn from(e: tokio_postgres::Error) -> Self {
        CalendarError::Database(e.to_string())
    }
}

// ============================================================================
// Data Types
// ============================================================================

/// Source layer for provenance tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceLayer {
    /// Layer A: Official sources (HTML, PDF)
    AOfficial,
    /// Layer B: Rules engine generated
    BRules,
    /// Layer C: Manual patches
    CPatch,
}

impl SourceLayer {
    pub fn as_str(&self) -> &'static str {
        match self {
            SourceLayer::AOfficial => "A_OFFICIAL",
            SourceLayer::BRules => "B_RULES",
            SourceLayer::CPatch => "C_PATCH",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "A_OFFICIAL" => Some(SourceLayer::AOfficial),
            "B_RULES" => Some(SourceLayer::BRules),
            "C_PATCH" => Some(SourceLayer::CPatch),
            _ => None,
        }
    }
}

/// Day type classification from database.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DayType {
    Regular,
    HalfDay,
    LateOpen,
    Closed,
}

impl DayType {
    pub fn as_str(&self) -> &'static str {
        match self {
            DayType::Regular => "REGULAR",
            DayType::HalfDay => "HALF_DAY",
            DayType::LateOpen => "LATE_OPEN",
            DayType::Closed => "CLOSED",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "REGULAR" => Some(DayType::Regular),
            "HALF_DAY" => Some(DayType::HalfDay),
            "LATE_OPEN" => Some(DayType::LateOpen),
            "CLOSED" => Some(DayType::Closed),
            _ => None,
        }
    }
}

/// Holiday type classification from database.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HolidayTypeDb {
    National,
    MarketSpecific,
    HalfDay,
    LateOpen,
    Extraordinary,
}

impl HolidayTypeDb {
    pub fn as_str(&self) -> &'static str {
        match self {
            HolidayTypeDb::National => "NATIONAL",
            HolidayTypeDb::MarketSpecific => "MARKET_SPECIFIC",
            HolidayTypeDb::HalfDay => "HALF_DAY",
            HolidayTypeDb::LateOpen => "LATE_OPEN",
            HolidayTypeDb::Extraordinary => "EXTRAORDINARY",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "NATIONAL" => Some(HolidayTypeDb::National),
            "MARKET_SPECIFIC" => Some(HolidayTypeDb::MarketSpecific),
            "HALF_DAY" => Some(HolidayTypeDb::HalfDay),
            "LATE_OPEN" => Some(HolidayTypeDb::LateOpen),
            "EXTRAORDINARY" => Some(HolidayTypeDb::Extraordinary),
            _ => None,
        }
    }
}

/// Supported calendar range for a market.
#[derive(Debug, Clone)]
pub struct SupportedRange {
    pub market: Market,
    pub range_start: NaiveDate,
    pub range_end: NaiveDate,
    pub coverage_level: String,
}

/// Calendar version metadata.
#[derive(Debug, Clone)]
pub struct CalendarVersion {
    pub id: Uuid,
    pub market: Market,
    pub year_start: i32,
    pub year_end: i32,
    pub version_hash: String,
    pub source_id: Uuid,
    pub is_active: bool,
}

/// Trading session from database.
#[derive(Debug, Clone)]
pub struct TradingSessionDb {
    pub id: Uuid,
    pub session_date: NaiveDate,
    pub market: Market,
    pub day_type: DayType,
    pub source_layer: SourceLayer,
    pub source_id: Option<Uuid>,
}

/// Session period from database.
#[derive(Debug, Clone)]
pub struct SessionPeriodDb {
    pub id: Uuid,
    pub session_id: Uuid,
    pub period_type: String,
    pub local_open: NaiveTime,
    pub local_close: NaiveTime,
    pub utc_offset_minutes: i32,
}

/// Holiday from database.
#[derive(Debug, Clone)]
pub struct HolidayDb {
    pub id: Uuid,
    pub holiday_date: NaiveDate,
    pub market: Market,
    pub name: String,
    pub holiday_type: HolidayTypeDb,
    pub early_close_time: Option<NaiveTime>,
    pub late_open_time: Option<NaiveTime>,
    pub source_layer: SourceLayer,
    pub source_id: Option<Uuid>,
}

/// Extraordinary closure from database.
#[derive(Debug, Clone)]
pub struct ExtraordinaryClosureDb {
    pub id: Uuid,
    pub closure_date: NaiveDate,
    pub market: Market,
    pub reason: String,
    pub legal_reference: Option<String>,
    pub source_layer: SourceLayer,
}

/// Complete day classification with provenance.
#[derive(Debug, Clone)]
pub struct DayClassificationDb {
    pub date: NaiveDate,
    pub market: Market,
    pub is_trading_day: bool,
    pub day_type: DayType,
    pub holiday: Option<HolidayDb>,
    pub extraordinary_closure: Option<ExtraordinaryClosureDb>,
    pub session: Option<TradingSessionDb>,
    pub periods: Vec<SessionPeriodDb>,
    pub source_layer: SourceLayer,
}

// ============================================================================
// Database Provider
// ============================================================================

/// Database calendar provider with connection pooling and range validation.
pub struct DbCalendarProvider {
    client: Arc<Client>,
    supported_ranges: Vec<SupportedRange>,
}

impl DbCalendarProvider {
    /// Create a new provider with an existing database client.
    pub async fn new(client: Arc<Client>) -> Result<Self, CalendarError> {
        let mut provider = Self {
            client,
            supported_ranges: Vec::new(),
        };
        provider.load_supported_ranges().await?;
        Ok(provider)
    }

    /// Load supported ranges from database.
    async fn load_supported_ranges(&mut self) -> Result<(), CalendarError> {
        let rows = self
            .client
            .query(
                "SELECT market, range_start, range_end, coverage_level FROM supported_ranges",
                &[],
            )
            .await?;

        self.supported_ranges = rows
            .iter()
            .filter_map(|row| {
                let market_str: &str = row.get(0);
                let market = match market_str {
                    "BR" => Market::BR,
                    "US" => Market::US,
                    _ => return None,
                };
                Some(SupportedRange {
                    market,
                    range_start: row.get(1),
                    range_end: row.get(2),
                    coverage_level: row.get(3),
                })
            })
            .collect();

        info!(
            "Loaded {} supported ranges: {:?}",
            self.supported_ranges.len(),
            self.supported_ranges
                .iter()
                .map(|r| format!("{:?}: {} to {}", r.market, r.range_start, r.range_end))
                .collect::<Vec<_>>()
        );

        Ok(())
    }

    /// Get supported range for a market.
    pub fn get_supported_range(&self, market: Market) -> Option<&SupportedRange> {
        self.supported_ranges.iter().find(|r| r.market == market)
    }

    /// Validate that a date is within the supported range.
    /// Returns CalendarOutOfRange error if not.
    pub fn validate_range(&self, market: Market, date: NaiveDate) -> Result<(), CalendarError> {
        match self.get_supported_range(market) {
            Some(range) => {
                if date < range.range_start || date > range.range_end {
                    Err(CalendarError::OutOfRange {
                        market,
                        date,
                        start: range.range_start,
                        end: range.range_end,
                    })
                } else {
                    Ok(())
                }
            }
            None => Err(CalendarError::Config(format!(
                "No supported range configured for {:?}",
                market
            ))),
        }
    }

    /// Get the active calendar version for a market.
    pub async fn get_active_version(&self, market: Market) -> Result<CalendarVersion, CalendarError> {
        let market_str = match market {
            Market::BR => "BR",
            Market::US => "US",
        };

        let row = self
            .client
            .query_opt(
                "SELECT id, market, year_start, year_end, version_hash, source_id, is_active 
                 FROM calendar_versions 
                 WHERE market = $1 AND is_active = true 
                 LIMIT 1",
                &[&market_str],
            )
            .await?
            .ok_or(CalendarError::NoActiveVersion { market })?;

        Ok(CalendarVersion {
            id: row.get(0),
            market,
            year_start: row.get(2),
            year_end: row.get(3),
            version_hash: row.get(4),
            source_id: row.get(5),
            is_active: row.get(6),
        })
    }

    /// Get holiday for a specific date with range validation.
    pub async fn get_holiday(
        &self,
        market: Market,
        date: NaiveDate,
    ) -> Result<Option<HolidayDb>, CalendarError> {
        // First validate the range - NEVER fall back silently
        self.validate_range(market, date)?;

        let market_str = match market {
            Market::BR => "BR",
            Market::US => "US",
        };

        let row = self
            .client
            .query_opt(
                "SELECT h.id, h.holiday_date, h.market, h.name, h.holiday_type, 
                        h.early_close_time, h.late_open_time, h.source_layer, h.source_id
                 FROM holidays h
                 JOIN calendar_versions v ON h.version_id = v.id
                 WHERE h.market = $1 AND h.holiday_date = $2 AND v.is_active = true",
                &[&market_str, &date],
            )
            .await?;

        match row {
            Some(r) => {
                let holiday_type_str: String = r.get(4);
                let source_layer_str: String = r.get(7);
                
                Ok(Some(HolidayDb {
                    id: r.get(0),
                    holiday_date: r.get(1),
                    market,
                    name: r.get(3),
                    holiday_type: HolidayTypeDb::from_str(&holiday_type_str)
                        .unwrap_or(HolidayTypeDb::National),
                    early_close_time: r.get(5),
                    late_open_time: r.get(6),
                    source_layer: SourceLayer::from_str(&source_layer_str)
                        .unwrap_or(SourceLayer::BRules),
                    source_id: r.get(8),
                }))
            }
            None => Ok(None),
        }
    }

    /// Get extraordinary closure for a specific date with range validation.
    pub async fn get_extraordinary_closure(
        &self,
        market: Market,
        date: NaiveDate,
    ) -> Result<Option<ExtraordinaryClosureDb>, CalendarError> {
        self.validate_range(market, date)?;

        let market_str = match market {
            Market::BR => "BR",
            Market::US => "US",
        };

        let row = self
            .client
            .query_opt(
                "SELECT e.id, e.closure_date, e.market, e.reason, e.legal_reference, e.source_layer
                 FROM extraordinary_closures e
                 JOIN calendar_versions v ON e.version_id = v.id
                 WHERE e.market = $1 AND e.closure_date = $2 AND v.is_active = true",
                &[&market_str, &date],
            )
            .await?;

        match row {
            Some(r) => {
                let source_layer_str: String = r.get(5);
                
                Ok(Some(ExtraordinaryClosureDb {
                    id: r.get(0),
                    closure_date: r.get(1),
                    market,
                    reason: r.get(3),
                    legal_reference: r.get(4),
                    source_layer: SourceLayer::from_str(&source_layer_str)
                        .unwrap_or(SourceLayer::CPatch),
                }))
            }
            None => Ok(None),
        }
    }

    /// Get trading session for a specific date with range validation.
    pub async fn get_trading_session(
        &self,
        market: Market,
        date: NaiveDate,
    ) -> Result<Option<TradingSessionDb>, CalendarError> {
        self.validate_range(market, date)?;

        let market_str = match market {
            Market::BR => "BR",
            Market::US => "US",
        };

        let row = self
            .client
            .query_opt(
                "SELECT s.id, s.session_date, s.market, s.day_type, s.source_layer, s.source_id
                 FROM trading_sessions s
                 JOIN calendar_versions v ON s.version_id = v.id
                 WHERE s.market = $1 AND s.session_date = $2 AND v.is_active = true",
                &[&market_str, &date],
            )
            .await?;

        match row {
            Some(r) => {
                let day_type_str: String = r.get(3);
                let source_layer_str: String = r.get(4);
                
                Ok(Some(TradingSessionDb {
                    id: r.get(0),
                    session_date: r.get(1),
                    market,
                    day_type: DayType::from_str(&day_type_str).unwrap_or(DayType::Regular),
                    source_layer: SourceLayer::from_str(&source_layer_str)
                        .unwrap_or(SourceLayer::BRules),
                    source_id: r.get(5),
                }))
            }
            None => Ok(None),
        }
    }

    /// Get session periods for a trading session.
    pub async fn get_session_periods(
        &self,
        session_id: Uuid,
    ) -> Result<Vec<SessionPeriodDb>, CalendarError> {
        let rows = self
            .client
            .query(
                "SELECT id, session_id, period_type, local_open, local_close, utc_offset_minutes
                 FROM session_periods
                 WHERE session_id = $1
                 ORDER BY local_open",
                &[&session_id],
            )
            .await?;

        Ok(rows
            .iter()
            .map(|r| SessionPeriodDb {
                id: r.get(0),
                session_id: r.get(1),
                period_type: r.get(2),
                local_open: r.get(3),
                local_close: r.get(4),
                utc_offset_minutes: r.get(5),
            })
            .collect())
    }

    /// Get complete day classification with all details and provenance.
    /// This is the main entry point for calendar queries.
    /// NEVER returns a silent fallback - always explicit error or data.
    pub async fn classify_day(
        &self,
        market: Market,
        date: NaiveDate,
    ) -> Result<DayClassificationDb, CalendarError> {
        // Always validate range first - no silent fallbacks
        self.validate_range(market, date)?;

        // Check for extraordinary closure first (highest priority)
        if let Some(closure) = self.get_extraordinary_closure(market, date).await? {
            return Ok(DayClassificationDb {
                date,
                market,
                is_trading_day: false,
                day_type: DayType::Closed,
                holiday: None,
                extraordinary_closure: Some(closure.clone()),
                session: None,
                periods: Vec::new(),
                source_layer: closure.source_layer,
            });
        }

        // Check for holiday
        if let Some(holiday) = self.get_holiday(market, date).await? {
            let is_trading = matches!(
                holiday.holiday_type,
                HolidayTypeDb::HalfDay | HolidayTypeDb::LateOpen
            );
            let day_type = match holiday.holiday_type {
                HolidayTypeDb::HalfDay => DayType::HalfDay,
                HolidayTypeDb::LateOpen => DayType::LateOpen,
                _ => DayType::Closed,
            };

            // Get session if it's a partial trading day
            let (session, periods) = if is_trading {
                if let Some(s) = self.get_trading_session(market, date).await? {
                    let p = self.get_session_periods(s.id).await?;
                    (Some(s), p)
                } else {
                    (None, Vec::new())
                }
            } else {
                (None, Vec::new())
            };

            return Ok(DayClassificationDb {
                date,
                market,
                is_trading_day: is_trading,
                day_type,
                holiday: Some(holiday.clone()),
                extraordinary_closure: None,
                session,
                periods,
                source_layer: holiday.source_layer,
            });
        }

        // Check for weekend (derived, not stored)
        let weekday = date.weekday();
        if weekday == chrono::Weekday::Sat || weekday == chrono::Weekday::Sun {
            return Ok(DayClassificationDb {
                date,
                market,
                is_trading_day: false,
                day_type: DayType::Closed,
                holiday: None,
                extraordinary_closure: None,
                session: None,
                periods: Vec::new(),
                source_layer: SourceLayer::BRules, // Weekend detection is rule-based
            });
        }

        // Regular trading day - get session details
        if let Some(session) = self.get_trading_session(market, date).await? {
            let periods = self.get_session_periods(session.id).await?;
            return Ok(DayClassificationDb {
                date,
                market,
                is_trading_day: session.day_type != DayType::Closed,
                day_type: session.day_type,
                holiday: None,
                extraordinary_closure: None,
                session: Some(session.clone()),
                periods,
                source_layer: session.source_layer,
            });
        }

        // No data found for a weekday within range - this is an ERROR, not a fallback
        // We do NOT assume it's a trading day just because it's a weekday
        Err(CalendarError::NoData { market, date })
    }

    /// Count trading days in a date range for validation.
    pub async fn count_trading_days(
        &self,
        market: Market,
        start: NaiveDate,
        end: NaiveDate,
    ) -> Result<i64, CalendarError> {
        self.validate_range(market, start)?;
        self.validate_range(market, end)?;

        let market_str = match market {
            Market::BR => "BR",
            Market::US => "US",
        };

        let row = self
            .client
            .query_one(
                "SELECT COUNT(*) 
                 FROM trading_sessions s
                 JOIN calendar_versions v ON s.version_id = v.id
                 WHERE s.market = $1 
                   AND s.session_date >= $2 
                   AND s.session_date <= $3 
                   AND s.day_type != 'CLOSED'
                   AND v.is_active = true",
                &[&market_str, &start, &end],
            )
            .await?;

        Ok(row.get(0))
    }

    /// Get all holidays for a year.
    pub async fn get_holidays_for_year(
        &self,
        market: Market,
        year: i32,
    ) -> Result<Vec<HolidayDb>, CalendarError> {
        let start = NaiveDate::from_ymd_opt(year, 1, 1).unwrap();
        let end = NaiveDate::from_ymd_opt(year, 12, 31).unwrap();
        
        self.validate_range(market, start)?;

        let market_str = match market {
            Market::BR => "BR",
            Market::US => "US",
        };

        let rows = self
            .client
            .query(
                "SELECT h.id, h.holiday_date, h.market, h.name, h.holiday_type, 
                        h.early_close_time, h.late_open_time, h.source_layer, h.source_id
                 FROM holidays h
                 JOIN calendar_versions v ON h.version_id = v.id
                 WHERE h.market = $1 
                   AND h.holiday_date >= $2 
                   AND h.holiday_date <= $3 
                   AND v.is_active = true
                 ORDER BY h.holiday_date",
                &[&market_str, &start, &end],
            )
            .await?;

        Ok(rows
            .iter()
            .map(|r| {
                let holiday_type_str: String = r.get(4);
                let source_layer_str: String = r.get(7);
                
                HolidayDb {
                    id: r.get(0),
                    holiday_date: r.get(1),
                    market,
                    name: r.get(3),
                    holiday_type: HolidayTypeDb::from_str(&holiday_type_str)
                        .unwrap_or(HolidayTypeDb::National),
                    early_close_time: r.get(5),
                    late_open_time: r.get(6),
                    source_layer: SourceLayer::from_str(&source_layer_str)
                        .unwrap_or(SourceLayer::BRules),
                    source_id: r.get(8),
                }
            })
            .collect())
    }

    /// Insert a calendar version and return its ID.
    pub async fn insert_version(
        &self,
        market: Market,
        year_start: i32,
        year_end: i32,
        version_hash: &str,
        source_id: Uuid,
    ) -> Result<Uuid, CalendarError> {
        let market_str = match market {
            Market::BR => "BR",
            Market::US => "US",
        };

        // Deactivate any existing active versions for this market
        self.client
            .execute(
                "UPDATE calendar_versions SET is_active = false, valid_until = NOW() 
                 WHERE market = $1 AND is_active = true",
                &[&market_str],
            )
            .await?;

        // Insert new version
        let row = self
            .client
            .query_one(
                "INSERT INTO calendar_versions (market, year_start, year_end, version_hash, source_id, is_active)
                 VALUES ($1, $2, $3, $4, $5, true)
                 RETURNING id",
                &[&market_str, &year_start, &year_end, &version_hash, &source_id],
            )
            .await?;

        Ok(row.get(0))
    }

    /// Get source ID by source_id string.
    pub async fn get_source_id(&self, source_id: &str) -> Result<Uuid, CalendarError> {
        let row = self
            .client
            .query_one(
                "SELECT id FROM calendar_sources WHERE source_id = $1",
                &[&source_id],
            )
            .await?;

        Ok(row.get(0))
    }

    /// Insert a holiday.
    pub async fn insert_holiday(
        &self,
        version_id: Uuid,
        holiday: &HolidayDb,
    ) -> Result<Uuid, CalendarError> {
        let market_str = match holiday.market {
            Market::BR => "BR",
            Market::US => "US",
        };

        let row = self
            .client
            .query_one(
                "INSERT INTO holidays (version_id, holiday_date, market, name, holiday_type, 
                                       early_close_time, late_open_time, source_layer, source_id)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                 RETURNING id",
                &[
                    &version_id,
                    &holiday.holiday_date,
                    &market_str,
                    &holiday.name,
                    &holiday.holiday_type.as_str(),
                    &holiday.early_close_time,
                    &holiday.late_open_time,
                    &holiday.source_layer.as_str(),
                    &holiday.source_id,
                ],
            )
            .await?;

        Ok(row.get(0))
    }

    /// Insert a trading session.
    pub async fn insert_trading_session(
        &self,
        version_id: Uuid,
        session: &TradingSessionDb,
    ) -> Result<Uuid, CalendarError> {
        let market_str = match session.market {
            Market::BR => "BR",
            Market::US => "US",
        };

        let row = self
            .client
            .query_one(
                "INSERT INTO trading_sessions (version_id, session_date, market, day_type, source_layer, source_id)
                 VALUES ($1, $2, $3, $4, $5, $6)
                 RETURNING id",
                &[
                    &version_id,
                    &session.session_date,
                    &market_str,
                    &session.day_type.as_str(),
                    &session.source_layer.as_str(),
                    &session.source_id,
                ],
            )
            .await?;

        Ok(row.get(0))
    }

    /// Insert a session period.
    pub async fn insert_session_period(
        &self,
        period: &SessionPeriodDb,
    ) -> Result<Uuid, CalendarError> {
        let row = self
            .client
            .query_one(
                "INSERT INTO session_periods (session_id, period_type, local_open, local_close, utc_offset_minutes)
                 VALUES ($1, $2, $3, $4, $5)
                 RETURNING id",
                &[
                    &period.session_id,
                    &period.period_type,
                    &period.local_open,
                    &period.local_close,
                    &period.utc_offset_minutes,
                ],
            )
            .await?;

        Ok(row.get(0))
    }

    /// Insert an extraordinary closure.
    pub async fn insert_extraordinary_closure(
        &self,
        version_id: Uuid,
        closure: &ExtraordinaryClosureDb,
    ) -> Result<Uuid, CalendarError> {
        let market_str = match closure.market {
            Market::BR => "BR",
            Market::US => "US",
        };

        let row = self
            .client
            .query_one(
                "INSERT INTO extraordinary_closures (version_id, closure_date, market, reason, legal_reference, source_layer, source_id)
                 VALUES ($1, $2, $3, $4, $5, $6, $7)
                 RETURNING id",
                &[
                    &version_id,
                    &closure.closure_date,
                    &market_str,
                    &closure.reason,
                    &closure.legal_reference,
                    &closure.source_layer.as_str(),
                    &None::<Uuid>,
                ],
            )
            .await?;

        Ok(row.get(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_layer_roundtrip() {
        assert_eq!(SourceLayer::from_str("A_OFFICIAL"), Some(SourceLayer::AOfficial));
        assert_eq!(SourceLayer::from_str("B_RULES"), Some(SourceLayer::BRules));
        assert_eq!(SourceLayer::from_str("C_PATCH"), Some(SourceLayer::CPatch));
        assert_eq!(SourceLayer::AOfficial.as_str(), "A_OFFICIAL");
    }

    #[test]
    fn test_day_type_roundtrip() {
        assert_eq!(DayType::from_str("REGULAR"), Some(DayType::Regular));
        assert_eq!(DayType::from_str("HALF_DAY"), Some(DayType::HalfDay));
        assert_eq!(DayType::from_str("LATE_OPEN"), Some(DayType::LateOpen));
        assert_eq!(DayType::from_str("CLOSED"), Some(DayType::Closed));
        assert_eq!(DayType::Regular.as_str(), "REGULAR");
    }

    #[test]
    fn test_holiday_type_roundtrip() {
        assert_eq!(HolidayTypeDb::from_str("NATIONAL"), Some(HolidayTypeDb::National));
        assert_eq!(HolidayTypeDb::from_str("HALF_DAY"), Some(HolidayTypeDb::HalfDay));
        assert_eq!(HolidayTypeDb::National.as_str(), "NATIONAL");
    }
}


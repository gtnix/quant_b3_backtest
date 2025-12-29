//! Database Holiday Provider - Async Neon PostgreSQL holiday queries with caching.
//!
//! Provides holiday data from Neon database with in-memory caching for performance.

use chrono::{NaiveDate, NaiveTime};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_postgres::Client;
use tracing::{debug, info};
use uuid::Uuid;

use super::db_provider::{CalendarError, HolidayTypeDb, SourceLayer};
use super::{Holiday, HolidayType, Market};

// ============================================================================
// Database Holiday Types
// ============================================================================

/// Holiday record from database.
#[derive(Debug, Clone)]
pub struct DbHoliday {
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

impl DbHoliday {
    /// Convert to domain Holiday type.
    pub fn to_domain(&self) -> Holiday {
        let holiday_type = match self.holiday_type {
            HolidayTypeDb::National => HolidayType::National,
            HolidayTypeDb::MarketSpecific => HolidayType::MarketSpecific,
            HolidayTypeDb::HalfDay => {
                let close_time = self.early_close_time.unwrap_or_else(|| {
                    // Default early close times
                    match self.market {
                        Market::US => NaiveTime::from_hms_opt(13, 0, 0).unwrap(),
                        Market::BR => NaiveTime::from_hms_opt(14, 0, 0).unwrap(),
                    }
                });
                HolidayType::HalfDay { close_time }
            }
            HolidayTypeDb::LateOpen => {
                let open_time = self.late_open_time.unwrap_or_else(|| {
                    NaiveTime::from_hms_opt(13, 0, 0).unwrap() // Default for Ash Wednesday
                });
                HolidayType::LateOpen { open_time }
            }
            HolidayTypeDb::Extraordinary => HolidayType::ExtraordinaryClosure,
        };

        Holiday {
            date: self.holiday_date,
            market: self.market,
            name: self.name.clone(),
            holiday_type,
            source: format!("{}:{:?}", self.source_layer.as_str(), self.source_id),
        }
    }
}

// ============================================================================
// Cache Key
// ============================================================================

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct CacheKey {
    market: Market,
    date: NaiveDate,
}

// ============================================================================
// DbHolidayProvider
// ============================================================================

/// Async database-backed holiday provider with caching.
#[derive(Debug)]
pub struct DbHolidayProvider {
    client: Arc<Client>,
    /// Cache: (market, date) -> Option<DbHoliday>
    cache: RwLock<HashMap<CacheKey, Option<DbHoliday>>>,
    /// Active version IDs per market
    version_ids: RwLock<HashMap<Market, Uuid>>,
}

impl DbHolidayProvider {
    /// Create a new provider with database client.
    pub async fn new(client: Arc<Client>) -> Result<Self, CalendarError> {
        let provider = Self {
            client,
            cache: RwLock::new(HashMap::new()),
            version_ids: RwLock::new(HashMap::new()),
        };

        // Load active version IDs
        provider.load_version_ids().await?;

        Ok(provider)
    }

    /// Load active calendar version IDs for each market.
    async fn load_version_ids(&self) -> Result<(), CalendarError> {
        let rows = self
            .client
            .query(
                "SELECT market, id FROM calendar_versions WHERE is_active = true",
                &[],
            )
            .await?;

        let mut version_ids = self.version_ids.write().await;
        for row in rows {
            let market_str: &str = row.get(0);
            let id: Uuid = row.get(1);

            let market = match market_str {
                "BR" => Market::BR,
                "US" => Market::US,
                _ => continue,
            };

            version_ids.insert(market, id);
            info!("Loaded calendar version for {:?}: {}", market, id);
        }

        Ok(())
    }

    /// Get active version ID for a market.
    async fn get_version_id(&self, market: Market) -> Result<Uuid, CalendarError> {
        let version_ids = self.version_ids.read().await;
        version_ids
            .get(&market)
            .copied()
            .ok_or(CalendarError::NoActiveVersion { market })
    }

    /// Get holiday for a specific date, using cache if available.
    pub async fn get_holiday(
        &self,
        market: Market,
        date: NaiveDate,
    ) -> Result<Option<Holiday>, CalendarError> {
        let key = CacheKey { market, date };

        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(&key) {
                debug!("Cache hit for {:?} on {}", market, date);
                return Ok(cached.as_ref().map(|h| h.to_domain()));
            }
        }

        // Query database
        let db_holiday = self.query_holiday(market, date).await?;

        // Update cache
        {
            let mut cache = self.cache.write().await;
            cache.insert(key, db_holiday.clone());
        }

        Ok(db_holiday.map(|h| h.to_domain()))
    }

    /// Query holiday from database.
    async fn query_holiday(
        &self,
        market: Market,
        date: NaiveDate,
    ) -> Result<Option<DbHoliday>, CalendarError> {
        let version_id = self.get_version_id(market).await?;
        let market_str = match market {
            Market::BR => "BR",
            Market::US => "US",
        };

        let row = self
            .client
            .query_opt(
                "SELECT id, holiday_date, market, name, holiday_type, 
                        early_close_time, late_open_time, source_layer, source_id
                 FROM holidays
                 WHERE version_id = $1 AND market = $2 AND holiday_date = $3",
                &[&version_id, &market_str, &date],
            )
            .await?;

        Ok(row.map(|r| self.row_to_holiday(&r, market)))
    }

    /// Get all holidays in a date range.
    pub async fn get_holidays_range(
        &self,
        market: Market,
        start: NaiveDate,
        end: NaiveDate,
    ) -> Result<Vec<Holiday>, CalendarError> {
        let version_id = self.get_version_id(market).await?;
        let market_str = match market {
            Market::BR => "BR",
            Market::US => "US",
        };

        let rows = self
            .client
            .query(
                "SELECT id, holiday_date, market, name, holiday_type,
                        early_close_time, late_open_time, source_layer, source_id
                 FROM holidays
                 WHERE version_id = $1 AND market = $2 
                   AND holiday_date >= $3 AND holiday_date <= $4
                 ORDER BY holiday_date",
                &[&version_id, &market_str, &start, &end],
            )
            .await?;

        let mut holidays = Vec::with_capacity(rows.len());
        let mut cache = self.cache.write().await;

        for row in rows {
            let db_holiday = self.row_to_holiday(&row, market);
            let key = CacheKey {
                market,
                date: db_holiday.holiday_date,
            };

            // Update cache
            cache.insert(key, Some(db_holiday.clone()));
            holidays.push(db_holiday.to_domain());
        }

        Ok(holidays)
    }

    /// Check if a date is a holiday.
    pub async fn is_holiday(
        &self,
        market: Market,
        date: NaiveDate,
    ) -> Result<bool, CalendarError> {
        Ok(self.get_holiday(market, date).await?.is_some())
    }

    /// Preload holidays for a year into cache.
    pub async fn preload_year(&self, market: Market, year: i32) -> Result<usize, CalendarError> {
        let start = NaiveDate::from_ymd_opt(year, 1, 1).unwrap();
        let end = NaiveDate::from_ymd_opt(year, 12, 31).unwrap();

        let holidays = self.get_holidays_range(market, start, end).await?;
        let count = holidays.len();

        info!("Preloaded {} holidays for {:?} {}", count, market, year);
        Ok(count)
    }

    /// Clear cache.
    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
        info!("Holiday cache cleared");
    }

    /// Get cache statistics.
    pub async fn cache_stats(&self) -> (usize, usize) {
        let cache = self.cache.read().await;
        let total = cache.len();
        let hits = cache.values().filter(|v| v.is_some()).count();
        (total, hits)
    }

    /// Convert database row to DbHoliday.
    fn row_to_holiday(&self, row: &tokio_postgres::Row, market: Market) -> DbHoliday {
        let holiday_type_str: &str = row.get(4);
        let source_layer_str: &str = row.get(7);

        DbHoliday {
            id: row.get(0),
            holiday_date: row.get(1),
            market,
            name: row.get(3),
            holiday_type: HolidayTypeDb::from_str(holiday_type_str).unwrap_or(HolidayTypeDb::National),
            early_close_time: row.get(5),
            late_open_time: row.get(6),
            source_layer: SourceLayer::from_str(source_layer_str).unwrap_or(SourceLayer::BRules),
            source_id: row.get(8),
        }
    }
}

// ============================================================================
// Extraordinary Closures
// ============================================================================

/// Extraordinary closure record from database.
#[derive(Debug, Clone)]
pub struct DbExtraordinaryClosure {
    pub id: Uuid,
    pub closure_date: NaiveDate,
    pub market: Market,
    pub reason: String,
    pub legal_reference: Option<String>,
    pub source_layer: SourceLayer,
    pub source_id: Option<Uuid>,
}

impl DbHolidayProvider {
    /// Get extraordinary closure for a specific date.
    pub async fn get_extraordinary_closure(
        &self,
        market: Market,
        date: NaiveDate,
    ) -> Result<Option<DbExtraordinaryClosure>, CalendarError> {
        let version_id = self.get_version_id(market).await?;
        let market_str = match market {
            Market::BR => "BR",
            Market::US => "US",
        };

        let row = self
            .client
            .query_opt(
                "SELECT id, closure_date, market, reason, legal_reference, source_layer, source_id
                 FROM extraordinary_closures
                 WHERE version_id = $1 AND market = $2 AND closure_date = $3",
                &[&version_id, &market_str, &date],
            )
            .await?;

        Ok(row.map(|r| {
            let source_layer_str: &str = r.get(5);
            DbExtraordinaryClosure {
                id: r.get(0),
                closure_date: r.get(1),
                market,
                reason: r.get(3),
                legal_reference: r.get(4),
                source_layer: SourceLayer::from_str(source_layer_str).unwrap_or(SourceLayer::CPatch),
                source_id: r.get(6),
            }
        }))
    }

    /// Check if a date has an extraordinary closure.
    pub async fn is_extraordinary_closure(
        &self,
        market: Market,
        date: NaiveDate,
    ) -> Result<bool, CalendarError> {
        Ok(self.get_extraordinary_closure(market, date).await?.is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_equality() {
        let k1 = CacheKey {
            market: Market::BR,
            date: NaiveDate::from_ymd_opt(2024, 12, 25).unwrap(),
        };
        let k2 = CacheKey {
            market: Market::BR,
            date: NaiveDate::from_ymd_opt(2024, 12, 25).unwrap(),
        };
        let k3 = CacheKey {
            market: Market::US,
            date: NaiveDate::from_ymd_opt(2024, 12, 25).unwrap(),
        };

        assert_eq!(k1, k2);
        assert_ne!(k1, k3);
    }

    #[test]
    fn test_db_holiday_to_domain() {
        let db_holiday = DbHoliday {
            id: Uuid::new_v4(),
            holiday_date: NaiveDate::from_ymd_opt(2024, 12, 25).unwrap(),
            market: Market::BR,
            name: "Natal".to_string(),
            holiday_type: HolidayTypeDb::National,
            early_close_time: None,
            late_open_time: None,
            source_layer: SourceLayer::BRules,
            source_id: None,
        };

        let domain = db_holiday.to_domain();
        assert_eq!(domain.name, "Natal");
        assert_eq!(domain.market, Market::BR);
        assert!(matches!(domain.holiday_type, HolidayType::National));
    }

    #[test]
    fn test_half_day_conversion() {
        let db_holiday = DbHoliday {
            id: Uuid::new_v4(),
            holiday_date: NaiveDate::from_ymd_opt(2024, 11, 29).unwrap(),
            market: Market::US,
            name: "Day After Thanksgiving".to_string(),
            holiday_type: HolidayTypeDb::HalfDay,
            early_close_time: Some(NaiveTime::from_hms_opt(13, 0, 0).unwrap()),
            late_open_time: None,
            source_layer: SourceLayer::BRules,
            source_id: None,
        };

        let domain = db_holiday.to_domain();
        if let HolidayType::HalfDay { close_time } = domain.holiday_type {
            assert_eq!(close_time.hour(), 13);
        } else {
            panic!("Expected HalfDay type");
        }
    }
}









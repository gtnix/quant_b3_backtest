//! Holiday Provider - Pluggable holiday data source with caching and versioning.
//!
//! Provides official holiday information for B3 and NYSE markets with
//! source attribution for audit purposes.

use chrono::{Datelike, NaiveDate, NaiveTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;

use super::Market;

/// Type of holiday.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HolidayType {
    /// Full day closure - national holiday
    National,
    /// Market-specific closure
    MarketSpecific,
    /// Half-day with early close
    HalfDay { close_time: NaiveTime },
    /// Late open (e.g., Ash Wednesday in Brazil)
    LateOpen { open_time: NaiveTime },
    /// Extraordinary closure (emergencies, mourning)
    ExtraordinaryClosure,
}

/// A single holiday definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Holiday {
    pub date: NaiveDate,
    pub market: Market,
    pub name: String,
    pub holiday_type: HolidayType,
    /// Source attribution for audit (e.g., "B3:OC-149-2024")
    pub source: String,
}

impl Holiday {
    /// Create a new national holiday.
    pub fn national(date: NaiveDate, market: Market, name: &str, source: &str) -> Self {
        Self {
            date,
            market,
            name: name.to_string(),
            holiday_type: HolidayType::National,
            source: source.to_string(),
        }
    }

    /// Create a half-day holiday.
    pub fn half_day(
        date: NaiveDate,
        market: Market,
        name: &str,
        close_time: NaiveTime,
        source: &str,
    ) -> Self {
        Self {
            date,
            market,
            name: name.to_string(),
            holiday_type: HolidayType::HalfDay { close_time },
            source: source.to_string(),
        }
    }

    /// Create a late-open day.
    pub fn late_open(
        date: NaiveDate,
        market: Market,
        name: &str,
        open_time: NaiveTime,
        source: &str,
    ) -> Self {
        Self {
            date,
            market,
            name: name.to_string(),
            holiday_type: HolidayType::LateOpen { open_time },
            source: source.to_string(),
        }
    }

    /// Check if this is a full closure (no trading at all).
    pub fn is_full_closure(&self) -> bool {
        matches!(
            self.holiday_type,
            HolidayType::National | HolidayType::MarketSpecific | HolidayType::ExtraordinaryClosure
        )
    }
}

/// Versioned holiday calendar for a specific market and year.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolidayCalendar {
    pub market: Market,
    pub year: i32,
    pub holidays: Vec<Holiday>,
    /// Version timestamp
    pub version: String,
    /// SHA256 hash of source data for reproducibility
    pub source_hash: String,
    /// When this calendar was fetched/created
    pub fetched_at: chrono::DateTime<Utc>,
}

impl HolidayCalendar {
    /// Create a new holiday calendar.
    pub fn new(market: Market, year: i32, holidays: Vec<Holiday>) -> Self {
        let now = Utc::now();
        let version = now.format("%Y-%m-%dT%H:%M:%SZ").to_string();

        // Compute hash of holiday data for reproducibility
        let data = serde_json::to_string(&holidays).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(data.as_bytes());
        let hash = format!("{:x}", hasher.finalize());

        Self {
            market,
            year,
            holidays,
            version,
            source_hash: hash[..16].to_string(), // First 16 chars
            fetched_at: now,
        }
    }

    /// Get holiday for a specific date.
    pub fn get(&self, date: NaiveDate) -> Option<&Holiday> {
        self.holidays.iter().find(|h| h.date == date)
    }

    /// Check if a date is a holiday.
    pub fn is_holiday(&self, date: NaiveDate) -> bool {
        self.holidays.iter().any(|h| h.date == date && h.is_full_closure())
    }
}

/// Holiday provider with caching and multiple market support.
#[derive(Debug, Clone)]
pub struct HolidayProvider {
    calendars: Arc<HashMap<(Market, i32), HolidayCalendar>>,
    version: String,
}

impl HolidayProvider {
    /// Create a provider with default embedded holiday data.
    pub fn new() -> Self {
        let mut calendars = HashMap::new();

        // Load default calendars
        calendars.insert((Market::BR, 2024), Self::b3_2024());
        calendars.insert((Market::BR, 2025), Self::b3_2025());
        calendars.insert((Market::US, 2024), Self::nyse_2024());
        calendars.insert((Market::US, 2025), Self::nyse_2025());

        let version = format!(
            "embedded-{}",
            Utc::now().format("%Y%m%d")
        );

        Self {
            calendars: Arc::new(calendars),
            version,
        }
    }

    /// Create a provider from custom calendar data.
    pub fn from_calendars(calendars: Vec<HolidayCalendar>) -> Self {
        let mut map = HashMap::new();
        for cal in calendars {
            map.insert((cal.market, cal.year), cal);
        }

        let version = format!(
            "custom-{}",
            Utc::now().format("%Y%m%dT%H%M%S")
        );

        Self {
            calendars: Arc::new(map),
            version,
        }
    }

    /// Get calendar for a specific market and year.
    pub fn get_calendar(&self, market: Market, year: i32) -> Option<&HolidayCalendar> {
        self.calendars.get(&(market, year))
    }

    /// Get holiday for a specific date.
    pub fn get_holiday(&self, market: Market, date: NaiveDate) -> Option<&Holiday> {
        self.get_calendar(market, date.year())
            .and_then(|cal| cal.get(date))
    }

    /// Check if a date is a full-closure holiday.
    pub fn is_holiday(&self, market: Market, date: NaiveDate) -> bool {
        self.get_calendar(market, date.year())
            .map(|cal| cal.is_holiday(date))
            .unwrap_or(false)
    }

    /// Get the provider version string.
    pub fn version(&self) -> String {
        self.version.clone()
    }

    // ========================================================================
    // Default Holiday Data
    // ========================================================================

    /// B3 holidays for 2024.
    /// Source: https://www.b3.com.br/pt_br/noticias/calendario-de-feriados-2024.htm
    fn b3_2024() -> HolidayCalendar {
        let source = "B3:OC-149-2023";
        let market = Market::BR;

        fn date(m: u32, d: u32) -> NaiveDate {
            NaiveDate::from_ymd_opt(2024, m, d).unwrap()
        }
        fn time(h: u32, m: u32) -> NaiveTime {
            NaiveTime::from_hms_opt(h, m, 0).unwrap()
        }

        let holidays = vec![
            Holiday::national(date(1, 1), market, "Confraternização Universal", source),
            Holiday::national(date(2, 12), market, "Carnaval", source),
            Holiday::national(date(2, 13), market, "Carnaval", source),
            Holiday::late_open(date(2, 14), market, "Quarta-feira de Cinzas", time(13, 0), source),
            Holiday::national(date(3, 29), market, "Sexta-feira Santa", source),
            Holiday::national(date(4, 21), market, "Tiradentes", source),
            Holiday::national(date(5, 1), market, "Dia do Trabalho", source),
            Holiday::national(date(5, 30), market, "Corpus Christi", source),
            Holiday::national(date(11, 15), market, "Proclamação da República", source),
            Holiday::national(date(11, 20), market, "Consciência Negra", source),
            Holiday::national(date(12, 24), market, "Véspera de Natal", source),
            Holiday::national(date(12, 25), market, "Natal", source),
            Holiday::national(date(12, 31), market, "Véspera de Ano Novo", source),
        ];

        HolidayCalendar::new(market, 2024, holidays)
    }

    /// B3 holidays for 2025.
    /// Source: https://www.b3.com.br/pt_br/noticias/calendario-de-feriados-2025.htm
    fn b3_2025() -> HolidayCalendar {
        let source = "B3:OC-149-2024";
        let market = Market::BR;

        fn date(m: u32, d: u32) -> NaiveDate {
            NaiveDate::from_ymd_opt(2025, m, d).unwrap()
        }
        fn time(h: u32, m: u32) -> NaiveTime {
            NaiveTime::from_hms_opt(h, m, 0).unwrap()
        }

        let holidays = vec![
            Holiday::national(date(1, 1), market, "Confraternização Universal", source),
            Holiday::national(date(3, 3), market, "Carnaval", source),
            Holiday::national(date(3, 4), market, "Carnaval", source),
            Holiday::late_open(date(3, 5), market, "Quarta-feira de Cinzas", time(13, 0), source),
            Holiday::national(date(4, 18), market, "Sexta-feira Santa", source),
            Holiday::national(date(4, 21), market, "Tiradentes", source),
            Holiday::national(date(5, 1), market, "Dia do Trabalho", source),
            Holiday::national(date(6, 19), market, "Corpus Christi", source),
            Holiday::national(date(11, 20), market, "Consciência Negra", source),
            Holiday::national(date(12, 24), market, "Véspera de Natal", source),
            Holiday::national(date(12, 25), market, "Natal", source),
            Holiday::national(date(12, 31), market, "Véspera de Ano Novo", source),
        ];

        HolidayCalendar::new(market, 2025, holidays)
    }

    /// NYSE holidays for 2024.
    /// Source: https://www.nyse.com/markets/holidays-calendars
    fn nyse_2024() -> HolidayCalendar {
        let source = "NYSE:2024-calendar";
        let market = Market::US;

        fn date(m: u32, d: u32) -> NaiveDate {
            NaiveDate::from_ymd_opt(2024, m, d).unwrap()
        }
        fn time(h: u32, m: u32) -> NaiveTime {
            NaiveTime::from_hms_opt(h, m, 0).unwrap()
        }

        let holidays = vec![
            Holiday::national(date(1, 1), market, "New Year's Day", source),
            Holiday::national(date(1, 15), market, "Martin Luther King Jr. Day", source),
            Holiday::national(date(2, 19), market, "Presidents' Day", source),
            Holiday::national(date(3, 29), market, "Good Friday", source),
            Holiday::national(date(5, 27), market, "Memorial Day", source),
            Holiday::national(date(6, 19), market, "Juneteenth", source),
            Holiday::half_day(date(7, 3), market, "Independence Day Eve", time(13, 0), source),
            Holiday::national(date(7, 4), market, "Independence Day", source),
            Holiday::national(date(9, 2), market, "Labor Day", source),
            Holiday::national(date(11, 28), market, "Thanksgiving Day", source),
            Holiday::half_day(date(11, 29), market, "Day After Thanksgiving", time(13, 0), source),
            Holiday::half_day(date(12, 24), market, "Christmas Eve", time(13, 0), source),
            Holiday::national(date(12, 25), market, "Christmas Day", source),
        ];

        HolidayCalendar::new(market, 2024, holidays)
    }

    /// NYSE holidays for 2025.
    /// Source: https://www.nyse.com/markets/holidays-calendars
    fn nyse_2025() -> HolidayCalendar {
        let source = "NYSE:2025-calendar";
        let market = Market::US;

        fn date(m: u32, d: u32) -> NaiveDate {
            NaiveDate::from_ymd_opt(2025, m, d).unwrap()
        }
        fn time(h: u32, m: u32) -> NaiveTime {
            NaiveTime::from_hms_opt(h, m, 0).unwrap()
        }

        let holidays = vec![
            Holiday::national(date(1, 1), market, "New Year's Day", source),
            Holiday::national(date(1, 20), market, "Martin Luther King Jr. Day", source),
            Holiday::national(date(2, 17), market, "Presidents' Day", source),
            Holiday::national(date(4, 18), market, "Good Friday", source),
            Holiday::national(date(5, 26), market, "Memorial Day", source),
            Holiday::national(date(6, 19), market, "Juneteenth", source),
            Holiday::half_day(date(7, 3), market, "Independence Day Eve", time(13, 0), source),
            Holiday::national(date(7, 4), market, "Independence Day", source),
            Holiday::national(date(9, 1), market, "Labor Day", source),
            Holiday::national(date(11, 27), market, "Thanksgiving Day", source),
            Holiday::half_day(date(11, 28), market, "Day After Thanksgiving", time(13, 0), source),
            Holiday::half_day(date(12, 24), market, "Christmas Eve", time(13, 0), source),
            Holiday::national(date(12, 25), market, "Christmas Day", source),
        ];

        HolidayCalendar::new(market, 2025, holidays)
    }
}

impl Default for HolidayProvider {
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

    #[test]
    fn test_b3_christmas_2024() {
        let provider = HolidayProvider::new();

        assert!(provider.is_holiday(Market::BR, date(2024, 12, 25)));
        let holiday = provider.get_holiday(Market::BR, date(2024, 12, 25));
        assert!(holiday.is_some());
        assert_eq!(holiday.unwrap().name, "Natal");
    }

    #[test]
    fn test_b3_regular_day_2024() {
        let provider = HolidayProvider::new();

        // Dec 23, 2024 is a Monday - should not be a holiday
        assert!(!provider.is_holiday(Market::BR, date(2024, 12, 23)));
    }

    #[test]
    fn test_nyse_independence_day_2024() {
        let provider = HolidayProvider::new();

        assert!(provider.is_holiday(Market::US, date(2024, 7, 4)));
        let holiday = provider.get_holiday(Market::US, date(2024, 7, 4));
        assert!(holiday.is_some());
        assert_eq!(holiday.unwrap().name, "Independence Day");
    }

    #[test]
    fn test_nyse_half_day_2024() {
        let provider = HolidayProvider::new();

        // July 3 is a half-day, not a full holiday
        assert!(!provider.is_holiday(Market::US, date(2024, 7, 3)));

        let holiday = provider.get_holiday(Market::US, date(2024, 7, 3));
        assert!(holiday.is_some());
        assert!(matches!(
            holiday.unwrap().holiday_type,
            HolidayType::HalfDay { .. }
        ));
    }

    #[test]
    fn test_b3_ash_wednesday_late_open_2025() {
        let provider = HolidayProvider::new();

        // Ash Wednesday 2025 is a late-open day, not a full holiday
        assert!(!provider.is_holiday(Market::BR, date(2025, 3, 5)));

        let holiday = provider.get_holiday(Market::BR, date(2025, 3, 5));
        assert!(holiday.is_some());
        assert!(matches!(
            holiday.unwrap().holiday_type,
            HolidayType::LateOpen { .. }
        ));
    }

    #[test]
    fn test_cross_market_holiday() {
        let provider = HolidayProvider::new();

        // Brazilian Carnival (Mar 3, 2025) - B3 closed, NYSE open
        assert!(provider.is_holiday(Market::BR, date(2025, 3, 3)));
        assert!(!provider.is_holiday(Market::US, date(2025, 3, 3)));

        // US MLK Day (Jan 20, 2025) - NYSE closed, B3 open
        assert!(!provider.is_holiday(Market::BR, date(2025, 1, 20)));
        assert!(provider.is_holiday(Market::US, date(2025, 1, 20)));
    }

    #[test]
    fn test_good_friday_both_markets() {
        let provider = HolidayProvider::new();

        // Good Friday 2025 (Apr 18) - both markets closed
        assert!(provider.is_holiday(Market::BR, date(2025, 4, 18)));
        assert!(provider.is_holiday(Market::US, date(2025, 4, 18)));
    }

    #[test]
    fn test_calendar_version() {
        let provider = HolidayProvider::new();
        let version = provider.version();
        assert!(version.starts_with("embedded-"));
    }

    #[test]
    fn test_calendar_source_hash() {
        let calendar = HolidayProvider::b3_2024();
        assert!(!calendar.source_hash.is_empty());
        assert_eq!(calendar.source_hash.len(), 16);
    }
}


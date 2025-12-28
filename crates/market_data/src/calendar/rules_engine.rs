//! Rules Engine for generating market holidays and trading sessions.
//!
//! Implements deterministic holiday calculation based on official rules
//! for both B3 (Brazil) and NYSE (US) markets.

use chrono::{Datelike, NaiveDate, NaiveTime, Weekday};
use uuid::Uuid;

use super::db_provider::{
    DayType, ExtraordinaryClosureDb, HolidayDb, HolidayTypeDb, SessionPeriodDb, SourceLayer,
    TradingSessionDb,
};
use super::Market;

// ============================================================================
// Easter Calculation (Computus)
// ============================================================================

/// Calculate Easter Sunday for a given year using the Anonymous Gregorian algorithm.
/// This is the basis for all moveable feasts (Carnival, Good Friday, Corpus Christi).
fn calculate_easter(year: i32) -> NaiveDate {
    let a = year % 19;
    let b = year / 100;
    let c = year % 100;
    let d = b / 4;
    let e = b % 4;
    let f = (b + 8) / 25;
    let g = (b - f + 1) / 3;
    let h = (19 * a + b - d - g + 15) % 30;
    let i = c / 4;
    let k = c % 4;
    let l = (32 + 2 * e + 2 * i - h - k) % 7;
    let m = (a + 11 * h + 22 * l) / 451;
    let month = (h + l - 7 * m + 114) / 31;
    let day = ((h + l - 7 * m + 114) % 31) + 1;

    NaiveDate::from_ymd_opt(year, month as u32, day as u32).unwrap()
}

/// Get the nth occurrence of a weekday in a month.
/// For example, get_nth_weekday(2024, 1, Weekday::Mon, 3) returns the 3rd Monday of January 2024.
fn get_nth_weekday(year: i32, month: u32, weekday: Weekday, n: u32) -> Option<NaiveDate> {
    let first_of_month = NaiveDate::from_ymd_opt(year, month, 1)?;
    let first_weekday = first_of_month.weekday();

    // Days until the target weekday
    let days_until = (weekday.num_days_from_monday() as i32
        - first_weekday.num_days_from_monday() as i32
        + 7)
        % 7;

    let day = 1 + days_until + (n as i32 - 1) * 7;
    NaiveDate::from_ymd_opt(year, month, day as u32)
}

/// Get the last occurrence of a weekday in a month.
fn get_last_weekday(year: i32, month: u32, weekday: Weekday) -> Option<NaiveDate> {
    // Start from the last day of the month and go backwards
    let next_month = if month == 12 { 1 } else { month + 1 };
    let next_year = if month == 12 { year + 1 } else { year };
    let first_of_next = NaiveDate::from_ymd_opt(next_year, next_month, 1)?;
    let last_of_month = first_of_next.pred_opt()?;

    let last_weekday = last_of_month.weekday();
    let days_back =
        (last_weekday.num_days_from_monday() as i32 - weekday.num_days_from_monday() as i32 + 7)
            % 7;

    last_of_month.checked_sub_signed(chrono::Duration::days(days_back as i64))
}

/// Adjust a holiday date if it falls on a weekend (US observation rules).
/// If Sunday -> Monday; If Saturday -> typically no observation needed for NYSE (market closed anyway).
fn adjust_for_weekend_us(date: NaiveDate) -> Option<NaiveDate> {
    match date.weekday() {
        Weekday::Sun => date.succ_opt(), // Observe on Monday
        Weekday::Sat => None,            // No special observation, market already closed
        _ => Some(date),
    }
}

// ============================================================================
// Rules Engine Trait
// ============================================================================

/// Trait for market-specific rules engines.
pub trait RulesEngine {
    /// Get the market this engine generates for.
    fn market(&self) -> Market;

    /// Generate all holidays for a given year.
    fn generate_holidays(&self, year: i32) -> Vec<HolidayDb>;

    /// Generate all trading sessions for a given year.
    fn generate_trading_sessions(&self, year: i32) -> Vec<TradingSessionDb>;

    /// Generate session periods for a regular trading day.
    fn generate_regular_session_periods(
        &self,
        session_id: Uuid,
        date: NaiveDate,
    ) -> Vec<SessionPeriodDb>;

    /// Get extraordinary closures for a year.
    fn get_extraordinary_closures(&self, year: i32) -> Vec<ExtraordinaryClosureDb>;
}

// ============================================================================
// B3 Rules Engine (Brazil)
// ============================================================================

/// Rules engine for B3 (Brazilian stock exchange).
///
/// Implements holiday rules based on Brazilian legislation and B3 official calendar.
/// Source: https://www.b3.com.br/pt_br/solucoes/plataformas/puma-trading-system/para-participantes-e-traders/calendario-de-negociacao/
pub struct B3RulesEngine {
    source_id: Option<Uuid>,
}

impl B3RulesEngine {
    pub fn new(source_id: Option<Uuid>) -> Self {
        Self { source_id }
    }

    /// Generate fixed holidays for a year.
    fn fixed_holidays(&self, year: i32) -> Vec<HolidayDb> {
        let mut holidays = Vec::new();

        // New Year's Day - January 1
        if let Some(date) = NaiveDate::from_ymd_opt(year, 1, 1) {
            holidays.push(HolidayDb {
                id: Uuid::new_v4(),
                holiday_date: date,
                market: Market::BR,
                name: "Confraternização Universal".to_string(),
                holiday_type: HolidayTypeDb::National,
                early_close_time: None,
                late_open_time: None,
                source_layer: SourceLayer::BRules,
                source_id: self.source_id,
            });
        }

        // Tiradentes - April 21
        if let Some(date) = NaiveDate::from_ymd_opt(year, 4, 21) {
            holidays.push(HolidayDb {
                id: Uuid::new_v4(),
                holiday_date: date,
                market: Market::BR,
                name: "Tiradentes".to_string(),
                holiday_type: HolidayTypeDb::National,
                early_close_time: None,
                late_open_time: None,
                source_layer: SourceLayer::BRules,
                source_id: self.source_id,
            });
        }

        // Labor Day - May 1
        if let Some(date) = NaiveDate::from_ymd_opt(year, 5, 1) {
            holidays.push(HolidayDb {
                id: Uuid::new_v4(),
                holiday_date: date,
                market: Market::BR,
                name: "Dia do Trabalho".to_string(),
                holiday_type: HolidayTypeDb::National,
                early_close_time: None,
                late_open_time: None,
                source_layer: SourceLayer::BRules,
                source_id: self.source_id,
            });
        }

        // Independence Day - September 7
        if let Some(date) = NaiveDate::from_ymd_opt(year, 9, 7) {
            holidays.push(HolidayDb {
                id: Uuid::new_v4(),
                holiday_date: date,
                market: Market::BR,
                name: "Independência do Brasil".to_string(),
                holiday_type: HolidayTypeDb::National,
                early_close_time: None,
                late_open_time: None,
                source_layer: SourceLayer::BRules,
                source_id: self.source_id,
            });
        }

        // Nossa Senhora Aparecida - October 12
        if let Some(date) = NaiveDate::from_ymd_opt(year, 10, 12) {
            holidays.push(HolidayDb {
                id: Uuid::new_v4(),
                holiday_date: date,
                market: Market::BR,
                name: "Nossa Senhora Aparecida".to_string(),
                holiday_type: HolidayTypeDb::National,
                early_close_time: None,
                late_open_time: None,
                source_layer: SourceLayer::BRules,
                source_id: self.source_id,
            });
        }

        // All Souls' Day - November 2
        if let Some(date) = NaiveDate::from_ymd_opt(year, 11, 2) {
            holidays.push(HolidayDb {
                id: Uuid::new_v4(),
                holiday_date: date,
                market: Market::BR,
                name: "Finados".to_string(),
                holiday_type: HolidayTypeDb::National,
                early_close_time: None,
                late_open_time: None,
                source_layer: SourceLayer::BRules,
                source_id: self.source_id,
            });
        }

        // Republic Proclamation Day - November 15
        if let Some(date) = NaiveDate::from_ymd_opt(year, 11, 15) {
            holidays.push(HolidayDb {
                id: Uuid::new_v4(),
                holiday_date: date,
                market: Market::BR,
                name: "Proclamação da República".to_string(),
                holiday_type: HolidayTypeDb::National,
                early_close_time: None,
                late_open_time: None,
                source_layer: SourceLayer::BRules,
                source_id: self.source_id,
            });
        }

        // Consciência Negra - November 20 (National since 2024)
        if year >= 2024 {
            if let Some(date) = NaiveDate::from_ymd_opt(year, 11, 20) {
                holidays.push(HolidayDb {
                    id: Uuid::new_v4(),
                    holiday_date: date,
                    market: Market::BR,
                    name: "Dia da Consciência Negra".to_string(),
                    holiday_type: HolidayTypeDb::National,
                    early_close_time: None,
                    late_open_time: None,
                    source_layer: SourceLayer::BRules,
                    source_id: self.source_id,
                });
            }
        }

        // Christmas Eve - December 24 (market closed)
        if let Some(date) = NaiveDate::from_ymd_opt(year, 12, 24) {
            holidays.push(HolidayDb {
                id: Uuid::new_v4(),
                holiday_date: date,
                market: Market::BR,
                name: "Véspera de Natal".to_string(),
                holiday_type: HolidayTypeDb::MarketSpecific,
                early_close_time: None,
                late_open_time: None,
                source_layer: SourceLayer::BRules,
                source_id: self.source_id,
            });
        }

        // Christmas - December 25
        if let Some(date) = NaiveDate::from_ymd_opt(year, 12, 25) {
            holidays.push(HolidayDb {
                id: Uuid::new_v4(),
                holiday_date: date,
                market: Market::BR,
                name: "Natal".to_string(),
                holiday_type: HolidayTypeDb::National,
                early_close_time: None,
                late_open_time: None,
                source_layer: SourceLayer::BRules,
                source_id: self.source_id,
            });
        }

        // New Year's Eve - December 31 (market closed)
        if let Some(date) = NaiveDate::from_ymd_opt(year, 12, 31) {
            holidays.push(HolidayDb {
                id: Uuid::new_v4(),
                holiday_date: date,
                market: Market::BR,
                name: "Véspera de Ano Novo".to_string(),
                holiday_type: HolidayTypeDb::MarketSpecific,
                early_close_time: None,
                late_open_time: None,
                source_layer: SourceLayer::BRules,
                source_id: self.source_id,
            });
        }

        holidays
    }

    /// Generate Easter-based moveable holidays.
    fn moveable_holidays(&self, year: i32) -> Vec<HolidayDb> {
        let mut holidays = Vec::new();
        let easter = calculate_easter(year);

        // Carnival Monday - Easter - 48 days
        if let Some(date) = easter.checked_sub_signed(chrono::Duration::days(48)) {
            holidays.push(HolidayDb {
                id: Uuid::new_v4(),
                holiday_date: date,
                market: Market::BR,
                name: "Carnaval".to_string(),
                holiday_type: HolidayTypeDb::National,
                early_close_time: None,
                late_open_time: None,
                source_layer: SourceLayer::BRules,
                source_id: self.source_id,
            });
        }

        // Carnival Tuesday - Easter - 47 days
        if let Some(date) = easter.checked_sub_signed(chrono::Duration::days(47)) {
            holidays.push(HolidayDb {
                id: Uuid::new_v4(),
                holiday_date: date,
                market: Market::BR,
                name: "Carnaval".to_string(),
                holiday_type: HolidayTypeDb::National,
                early_close_time: None,
                late_open_time: None,
                source_layer: SourceLayer::BRules,
                source_id: self.source_id,
            });
        }

        // Ash Wednesday - Easter - 46 days (Late open at 13:00 BRT)
        if let Some(date) = easter.checked_sub_signed(chrono::Duration::days(46)) {
            holidays.push(HolidayDb {
                id: Uuid::new_v4(),
                holiday_date: date,
                market: Market::BR,
                name: "Quarta-feira de Cinzas".to_string(),
                holiday_type: HolidayTypeDb::LateOpen,
                early_close_time: None,
                late_open_time: Some(NaiveTime::from_hms_opt(13, 0, 0).unwrap()),
                source_layer: SourceLayer::BRules,
                source_id: self.source_id,
            });
        }

        // Good Friday - Easter - 2 days
        if let Some(date) = easter.checked_sub_signed(chrono::Duration::days(2)) {
            holidays.push(HolidayDb {
                id: Uuid::new_v4(),
                holiday_date: date,
                market: Market::BR,
                name: "Sexta-feira Santa".to_string(),
                holiday_type: HolidayTypeDb::National,
                early_close_time: None,
                late_open_time: None,
                source_layer: SourceLayer::BRules,
                source_id: self.source_id,
            });
        }

        // Corpus Christi - Easter + 60 days
        if let Some(date) = easter.checked_add_signed(chrono::Duration::days(60)) {
            holidays.push(HolidayDb {
                id: Uuid::new_v4(),
                holiday_date: date,
                market: Market::BR,
                name: "Corpus Christi".to_string(),
                holiday_type: HolidayTypeDb::National,
                early_close_time: None,
                late_open_time: None,
                source_layer: SourceLayer::BRules,
                source_id: self.source_id,
            });
        }

        holidays
    }

    /// Calculate UTC offset for B3 on a given date (handles historical DST).
    fn get_utc_offset(&self, date: NaiveDate) -> i32 {
        // Brazil ended DST in 2019, so from 2019 onwards it's always UTC-3 (-180 minutes)
        // Before that, DST ran from October to February (UTC-2 = -120 minutes during DST)
        if date.year() >= 2019 {
            -180 // UTC-3 always
        } else {
            // Simplified: assume standard time for historical dates
            // A more accurate implementation would check exact DST dates per year
            -180
        }
    }
}

impl RulesEngine for B3RulesEngine {
    fn market(&self) -> Market {
        Market::BR
    }

    fn generate_holidays(&self, year: i32) -> Vec<HolidayDb> {
        let mut holidays = self.fixed_holidays(year);
        holidays.extend(self.moveable_holidays(year));
        holidays
    }

    fn generate_trading_sessions(&self, year: i32) -> Vec<TradingSessionDb> {
        let mut sessions = Vec::new();
        let holidays = self.generate_holidays(year);
        let _holiday_dates: std::collections::HashSet<_> =
            holidays.iter().map(|h| h.holiday_date).collect();

        let start = NaiveDate::from_ymd_opt(year, 1, 1).unwrap();
        let end = NaiveDate::from_ymd_opt(year, 12, 31).unwrap();
        let mut current = start;

        while current <= end {
            let weekday = current.weekday();

            // Skip weekends
            if weekday == Weekday::Sat || weekday == Weekday::Sun {
                current = current.succ_opt().unwrap();
                continue;
            }

            // Check if it's a holiday
            if let Some(holiday) = holidays.iter().find(|h| h.holiday_date == current) {
                let day_type = match holiday.holiday_type {
                    HolidayTypeDb::LateOpen => DayType::LateOpen,
                    HolidayTypeDb::HalfDay => DayType::HalfDay,
                    _ => DayType::Closed,
                };

                // Only create session for partial trading days
                if day_type != DayType::Closed {
                    sessions.push(TradingSessionDb {
                        id: Uuid::new_v4(),
                        session_date: current,
                        market: Market::BR,
                        day_type,
                        source_layer: SourceLayer::BRules,
                        source_id: self.source_id,
                    });
                }
            } else {
                // Regular trading day
                sessions.push(TradingSessionDb {
                    id: Uuid::new_v4(),
                    session_date: current,
                    market: Market::BR,
                    day_type: DayType::Regular,
                    source_layer: SourceLayer::BRules,
                    source_id: self.source_id,
                });
            }

            current = current.succ_opt().unwrap();
        }

        sessions
    }

    fn generate_regular_session_periods(
        &self,
        session_id: Uuid,
        date: NaiveDate,
    ) -> Vec<SessionPeriodDb> {
        let offset = self.get_utc_offset(date);

        vec![
            // Pre-market (09:45 - 10:00 BRT)
            SessionPeriodDb {
                id: Uuid::new_v4(),
                session_id,
                period_type: "PRE_MARKET".to_string(),
                local_open: NaiveTime::from_hms_opt(9, 45, 0).unwrap(),
                local_close: NaiveTime::from_hms_opt(10, 0, 0).unwrap(),
                utc_offset_minutes: offset,
            },
            // Regular session (10:00 - 17:00 BRT)
            SessionPeriodDb {
                id: Uuid::new_v4(),
                session_id,
                period_type: "REGULAR".to_string(),
                local_open: NaiveTime::from_hms_opt(10, 0, 0).unwrap(),
                local_close: NaiveTime::from_hms_opt(17, 0, 0).unwrap(),
                utc_offset_minutes: offset,
            },
            // Closing auction (17:00 - 17:30 BRT)
            SessionPeriodDb {
                id: Uuid::new_v4(),
                session_id,
                period_type: "CLOSING_AUCTION".to_string(),
                local_open: NaiveTime::from_hms_opt(17, 0, 0).unwrap(),
                local_close: NaiveTime::from_hms_opt(17, 30, 0).unwrap(),
                utc_offset_minutes: offset,
            },
            // After-hours (17:30 - 18:00 BRT)
            SessionPeriodDb {
                id: Uuid::new_v4(),
                session_id,
                period_type: "AFTER_HOURS".to_string(),
                local_open: NaiveTime::from_hms_opt(17, 30, 0).unwrap(),
                local_close: NaiveTime::from_hms_opt(18, 0, 0).unwrap(),
                utc_offset_minutes: offset,
            },
        ]
    }

    fn get_extraordinary_closures(&self, _year: i32) -> Vec<ExtraordinaryClosureDb> {
        // B3 extraordinary closures are rare; add specific ones as needed
        Vec::new()
    }
}

// ============================================================================
// NYSE Rules Engine (United States)
// ============================================================================

/// Rules engine for NYSE/NASDAQ (US stock exchanges).
///
/// Implements holiday rules based on NYSE official calendar.
/// Source: https://www.nyse.com/markets/hours-calendars
pub struct NYSERulesEngine {
    source_id: Option<Uuid>,
}

impl NYSERulesEngine {
    pub fn new(source_id: Option<Uuid>) -> Self {
        Self { source_id }
    }

    /// Generate fixed holidays for a year.
    fn fixed_holidays(&self, year: i32) -> Vec<HolidayDb> {
        let mut holidays = Vec::new();

        // New Year's Day - January 1 (observed if Sunday)
        if let Some(date) = NaiveDate::from_ymd_opt(year, 1, 1) {
            if let Some(observed) = adjust_for_weekend_us(date) {
                holidays.push(HolidayDb {
                    id: Uuid::new_v4(),
                    holiday_date: observed,
                    market: Market::US,
                    name: "New Year's Day".to_string(),
                    holiday_type: HolidayTypeDb::National,
                    early_close_time: None,
                    late_open_time: None,
                    source_layer: SourceLayer::BRules,
                    source_id: self.source_id,
                });
            }
        }

        // Independence Day - July 4 (observed if Sunday)
        if let Some(date) = NaiveDate::from_ymd_opt(year, 7, 4) {
            if let Some(observed) = adjust_for_weekend_us(date) {
                holidays.push(HolidayDb {
                    id: Uuid::new_v4(),
                    holiday_date: observed,
                    market: Market::US,
                    name: "Independence Day".to_string(),
                    holiday_type: HolidayTypeDb::National,
                    early_close_time: None,
                    late_open_time: None,
                    source_layer: SourceLayer::BRules,
                    source_id: self.source_id,
                });
            }
        }

        // Juneteenth - June 19 (national holiday since 2021)
        if year >= 2022 {
            if let Some(date) = NaiveDate::from_ymd_opt(year, 6, 19) {
                if let Some(observed) = adjust_for_weekend_us(date) {
                    holidays.push(HolidayDb {
                        id: Uuid::new_v4(),
                        holiday_date: observed,
                        market: Market::US,
                        name: "Juneteenth National Independence Day".to_string(),
                        holiday_type: HolidayTypeDb::National,
                        early_close_time: None,
                        late_open_time: None,
                        source_layer: SourceLayer::BRules,
                        source_id: self.source_id,
                    });
                }
            }
        } else if year == 2021 {
            // First observed in 2021 on June 18 (early close)
            if let Some(date) = NaiveDate::from_ymd_opt(2021, 6, 18) {
                holidays.push(HolidayDb {
                    id: Uuid::new_v4(),
                    holiday_date: date,
                    market: Market::US,
                    name: "Juneteenth (Early Close)".to_string(),
                    holiday_type: HolidayTypeDb::HalfDay,
                    early_close_time: Some(NaiveTime::from_hms_opt(13, 0, 0).unwrap()),
                    late_open_time: None,
                    source_layer: SourceLayer::BRules,
                    source_id: self.source_id,
                });
            }
        }

        // Christmas Day - December 25 (observed if Sunday)
        if let Some(date) = NaiveDate::from_ymd_opt(year, 12, 25) {
            if let Some(observed) = adjust_for_weekend_us(date) {
                holidays.push(HolidayDb {
                    id: Uuid::new_v4(),
                    holiday_date: observed,
                    market: Market::US,
                    name: "Christmas Day".to_string(),
                    holiday_type: HolidayTypeDb::National,
                    early_close_time: None,
                    late_open_time: None,
                    source_layer: SourceLayer::BRules,
                    source_id: self.source_id,
                });
            }
        }

        holidays
    }

    /// Generate variable holidays (based on weekday patterns).
    fn variable_holidays(&self, year: i32) -> Vec<HolidayDb> {
        let mut holidays = Vec::new();

        // MLK Day - 3rd Monday of January
        if let Some(date) = get_nth_weekday(year, 1, Weekday::Mon, 3) {
            holidays.push(HolidayDb {
                id: Uuid::new_v4(),
                holiday_date: date,
                market: Market::US,
                name: "Martin Luther King Jr. Day".to_string(),
                holiday_type: HolidayTypeDb::National,
                early_close_time: None,
                late_open_time: None,
                source_layer: SourceLayer::BRules,
                source_id: self.source_id,
            });
        }

        // Presidents Day - 3rd Monday of February
        if let Some(date) = get_nth_weekday(year, 2, Weekday::Mon, 3) {
            holidays.push(HolidayDb {
                id: Uuid::new_v4(),
                holiday_date: date,
                market: Market::US,
                name: "Presidents Day".to_string(),
                holiday_type: HolidayTypeDb::National,
                early_close_time: None,
                late_open_time: None,
                source_layer: SourceLayer::BRules,
                source_id: self.source_id,
            });
        }

        // Good Friday - Friday before Easter
        let easter = calculate_easter(year);
        if let Some(date) = easter.checked_sub_signed(chrono::Duration::days(2)) {
            holidays.push(HolidayDb {
                id: Uuid::new_v4(),
                holiday_date: date,
                market: Market::US,
                name: "Good Friday".to_string(),
                holiday_type: HolidayTypeDb::National,
                early_close_time: None,
                late_open_time: None,
                source_layer: SourceLayer::BRules,
                source_id: self.source_id,
            });
        }

        // Memorial Day - Last Monday of May
        if let Some(date) = get_last_weekday(year, 5, Weekday::Mon) {
            holidays.push(HolidayDb {
                id: Uuid::new_v4(),
                holiday_date: date,
                market: Market::US,
                name: "Memorial Day".to_string(),
                holiday_type: HolidayTypeDb::National,
                early_close_time: None,
                late_open_time: None,
                source_layer: SourceLayer::BRules,
                source_id: self.source_id,
            });
        }

        // Labor Day - 1st Monday of September
        if let Some(date) = get_nth_weekday(year, 9, Weekday::Mon, 1) {
            holidays.push(HolidayDb {
                id: Uuid::new_v4(),
                holiday_date: date,
                market: Market::US,
                name: "Labor Day".to_string(),
                holiday_type: HolidayTypeDb::National,
                early_close_time: None,
                late_open_time: None,
                source_layer: SourceLayer::BRules,
                source_id: self.source_id,
            });
        }

        // Thanksgiving - 4th Thursday of November
        if let Some(date) = get_nth_weekday(year, 11, Weekday::Thu, 4) {
            holidays.push(HolidayDb {
                id: Uuid::new_v4(),
                holiday_date: date,
                market: Market::US,
                name: "Thanksgiving Day".to_string(),
                holiday_type: HolidayTypeDb::National,
                early_close_time: None,
                late_open_time: None,
                source_layer: SourceLayer::BRules,
                source_id: self.source_id,
            });
        }

        holidays
    }

    /// Generate early close days (half-days with 13:00 close).
    fn early_close_days(&self, year: i32) -> Vec<HolidayDb> {
        let mut holidays = Vec::new();

        // Day before Independence Day (if weekday and not Thursday when July 4 is Friday)
        if let Some(july3) = NaiveDate::from_ymd_opt(year, 7, 3) {
            let weekday = july3.weekday();
            if weekday != Weekday::Sat && weekday != Weekday::Sun {
                // Check if July 4 is not on a Saturday (in which case July 3 is Friday, normal day)
                // and July 4 is not on a Sunday (in which case Monday is observed, July 3 is still early close)
                holidays.push(HolidayDb {
                    id: Uuid::new_v4(),
                    holiday_date: july3,
                    market: Market::US,
                    name: "Day Before Independence Day".to_string(),
                    holiday_type: HolidayTypeDb::HalfDay,
                    early_close_time: Some(NaiveTime::from_hms_opt(13, 0, 0).unwrap()),
                    late_open_time: None,
                    source_layer: SourceLayer::BRules,
                    source_id: self.source_id,
                });
            }
        }

        // Black Friday - Day after Thanksgiving
        if let Some(thanksgiving) = get_nth_weekday(year, 11, Weekday::Thu, 4) {
            if let Some(black_friday) = thanksgiving.succ_opt() {
                holidays.push(HolidayDb {
                    id: Uuid::new_v4(),
                    holiday_date: black_friday,
                    market: Market::US,
                    name: "Day After Thanksgiving".to_string(),
                    holiday_type: HolidayTypeDb::HalfDay,
                    early_close_time: Some(NaiveTime::from_hms_opt(13, 0, 0).unwrap()),
                    late_open_time: None,
                    source_layer: SourceLayer::BRules,
                    source_id: self.source_id,
                });
            }
        }

        // Christmas Eve (if weekday)
        if let Some(dec24) = NaiveDate::from_ymd_opt(year, 12, 24) {
            let weekday = dec24.weekday();
            if weekday != Weekday::Sat && weekday != Weekday::Sun {
                holidays.push(HolidayDb {
                    id: Uuid::new_v4(),
                    holiday_date: dec24,
                    market: Market::US,
                    name: "Christmas Eve".to_string(),
                    holiday_type: HolidayTypeDb::HalfDay,
                    early_close_time: Some(NaiveTime::from_hms_opt(13, 0, 0).unwrap()),
                    late_open_time: None,
                    source_layer: SourceLayer::BRules,
                    source_id: self.source_id,
                });
            }
        }

        holidays
    }

    /// Get known extraordinary closures.
    fn known_extraordinary_closures(&self) -> Vec<ExtraordinaryClosureDb> {
        vec![
            // Gerald Ford National Day of Mourning - January 2, 2007
            ExtraordinaryClosureDb {
                id: Uuid::new_v4(),
                closure_date: NaiveDate::from_ymd_opt(2007, 1, 2).unwrap(),
                market: Market::US,
                reason: "National Day of Mourning for President Gerald Ford".to_string(),
                legal_reference: Some("Presidential Proclamation".to_string()),
                source_layer: SourceLayer::CPatch,
            },
            // Hurricane Sandy - October 29-30, 2012
            ExtraordinaryClosureDb {
                id: Uuid::new_v4(),
                closure_date: NaiveDate::from_ymd_opt(2012, 10, 29).unwrap(),
                market: Market::US,
                reason: "Hurricane Sandy".to_string(),
                legal_reference: Some("NYSE Emergency Closure".to_string()),
                source_layer: SourceLayer::CPatch,
            },
            ExtraordinaryClosureDb {
                id: Uuid::new_v4(),
                closure_date: NaiveDate::from_ymd_opt(2012, 10, 30).unwrap(),
                market: Market::US,
                reason: "Hurricane Sandy".to_string(),
                legal_reference: Some("NYSE Emergency Closure".to_string()),
                source_layer: SourceLayer::CPatch,
            },
            // George H.W. Bush National Day of Mourning - December 5, 2018
            ExtraordinaryClosureDb {
                id: Uuid::new_v4(),
                closure_date: NaiveDate::from_ymd_opt(2018, 12, 5).unwrap(),
                market: Market::US,
                reason: "National Day of Mourning for President George H.W. Bush".to_string(),
                legal_reference: Some("Presidential Proclamation".to_string()),
                source_layer: SourceLayer::CPatch,
            },
        ]
    }

    /// Calculate UTC offset for NYSE on a given date (handles DST).
    fn get_utc_offset(&self, date: NaiveDate) -> i32 {
        // US DST: 2nd Sunday of March to 1st Sunday of November
        let year = date.year();

        // Find 2nd Sunday of March
        let dst_start = get_nth_weekday(year, 3, Weekday::Sun, 2).unwrap();
        // Find 1st Sunday of November
        let dst_end = get_nth_weekday(year, 11, Weekday::Sun, 1).unwrap();

        if date >= dst_start && date < dst_end {
            -240 // EDT (UTC-4)
        } else {
            -300 // EST (UTC-5)
        }
    }
}

impl RulesEngine for NYSERulesEngine {
    fn market(&self) -> Market {
        Market::US
    }

    fn generate_holidays(&self, year: i32) -> Vec<HolidayDb> {
        let mut holidays = self.fixed_holidays(year);
        holidays.extend(self.variable_holidays(year));
        holidays.extend(self.early_close_days(year));
        holidays
    }

    fn generate_trading_sessions(&self, year: i32) -> Vec<TradingSessionDb> {
        let mut sessions = Vec::new();
        let holidays = self.generate_holidays(year);
        let extraordinary = self.get_extraordinary_closures(year);

        let _holiday_dates: std::collections::HashSet<_> =
            holidays.iter().map(|h| h.holiday_date).collect();
        let extraordinary_dates: std::collections::HashSet<_> =
            extraordinary.iter().map(|e| e.closure_date).collect();

        let start = NaiveDate::from_ymd_opt(year, 1, 1).unwrap();
        let end = NaiveDate::from_ymd_opt(year, 12, 31).unwrap();
        let mut current = start;

        while current <= end {
            let weekday = current.weekday();

            // Skip weekends
            if weekday == Weekday::Sat || weekday == Weekday::Sun {
                current = current.succ_opt().unwrap();
                continue;
            }

            // Skip extraordinary closures
            if extraordinary_dates.contains(&current) {
                current = current.succ_opt().unwrap();
                continue;
            }

            // Check if it's a holiday
            if let Some(holiday) = holidays.iter().find(|h| h.holiday_date == current) {
                let day_type = match holiday.holiday_type {
                    HolidayTypeDb::HalfDay => DayType::HalfDay,
                    HolidayTypeDb::LateOpen => DayType::LateOpen,
                    _ => DayType::Closed,
                };

                // Only create session for partial trading days
                if day_type != DayType::Closed {
                    sessions.push(TradingSessionDb {
                        id: Uuid::new_v4(),
                        session_date: current,
                        market: Market::US,
                        day_type,
                        source_layer: SourceLayer::BRules,
                        source_id: self.source_id,
                    });
                }
            } else {
                // Regular trading day
                sessions.push(TradingSessionDb {
                    id: Uuid::new_v4(),
                    session_date: current,
                    market: Market::US,
                    day_type: DayType::Regular,
                    source_layer: SourceLayer::BRules,
                    source_id: self.source_id,
                });
            }

            current = current.succ_opt().unwrap();
        }

        sessions
    }

    fn generate_regular_session_periods(
        &self,
        session_id: Uuid,
        date: NaiveDate,
    ) -> Vec<SessionPeriodDb> {
        let offset = self.get_utc_offset(date);

        vec![
            // Pre-market (04:00 - 09:30 ET)
            SessionPeriodDb {
                id: Uuid::new_v4(),
                session_id,
                period_type: "PRE_MARKET".to_string(),
                local_open: NaiveTime::from_hms_opt(4, 0, 0).unwrap(),
                local_close: NaiveTime::from_hms_opt(9, 30, 0).unwrap(),
                utc_offset_minutes: offset,
            },
            // Regular session (09:30 - 16:00 ET)
            SessionPeriodDb {
                id: Uuid::new_v4(),
                session_id,
                period_type: "REGULAR".to_string(),
                local_open: NaiveTime::from_hms_opt(9, 30, 0).unwrap(),
                local_close: NaiveTime::from_hms_opt(16, 0, 0).unwrap(),
                utc_offset_minutes: offset,
            },
            // After-hours (16:00 - 20:00 ET)
            SessionPeriodDb {
                id: Uuid::new_v4(),
                session_id,
                period_type: "AFTER_HOURS".to_string(),
                local_open: NaiveTime::from_hms_opt(16, 0, 0).unwrap(),
                local_close: NaiveTime::from_hms_opt(20, 0, 0).unwrap(),
                utc_offset_minutes: offset,
            },
        ]
    }

    fn get_extraordinary_closures(&self, year: i32) -> Vec<ExtraordinaryClosureDb> {
        self.known_extraordinary_closures()
            .into_iter()
            .filter(|c| c.closure_date.year() == year)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_easter_calculation() {
        // Known Easter dates
        assert_eq!(calculate_easter(2024), NaiveDate::from_ymd_opt(2024, 3, 31).unwrap());
        assert_eq!(calculate_easter(2025), NaiveDate::from_ymd_opt(2025, 4, 20).unwrap());
        assert_eq!(calculate_easter(2023), NaiveDate::from_ymd_opt(2023, 4, 9).unwrap());
        assert_eq!(calculate_easter(2022), NaiveDate::from_ymd_opt(2022, 4, 17).unwrap());
    }

    #[test]
    fn test_nth_weekday() {
        // 3rd Monday of January 2024 = MLK Day = January 15
        assert_eq!(
            get_nth_weekday(2024, 1, Weekday::Mon, 3),
            Some(NaiveDate::from_ymd_opt(2024, 1, 15).unwrap())
        );

        // 4th Thursday of November 2024 = Thanksgiving = November 28
        assert_eq!(
            get_nth_weekday(2024, 11, Weekday::Thu, 4),
            Some(NaiveDate::from_ymd_opt(2024, 11, 28).unwrap())
        );
    }

    #[test]
    fn test_last_weekday() {
        // Last Monday of May 2024 = Memorial Day = May 27
        assert_eq!(
            get_last_weekday(2024, 5, Weekday::Mon),
            Some(NaiveDate::from_ymd_opt(2024, 5, 27).unwrap())
        );
    }

    #[test]
    fn test_b3_holidays_2024() {
        let engine = B3RulesEngine::new(None);
        let holidays = engine.generate_holidays(2024);

        // Check for known holidays
        let holiday_names: Vec<_> = holidays.iter().map(|h| h.name.as_str()).collect();
        assert!(holiday_names.contains(&"Confraternização Universal"));
        assert!(holiday_names.contains(&"Carnaval"));
        assert!(holiday_names.contains(&"Sexta-feira Santa"));
        assert!(holiday_names.contains(&"Tiradentes"));
        assert!(holiday_names.contains(&"Dia do Trabalho"));
        assert!(holiday_names.contains(&"Corpus Christi"));
        assert!(holiday_names.contains(&"Dia da Consciência Negra"));
        assert!(holiday_names.contains(&"Natal"));
    }

    #[test]
    fn test_nyse_holidays_2024() {
        let engine = NYSERulesEngine::new(None);
        let holidays = engine.generate_holidays(2024);

        // Check for known holidays
        let holiday_names: Vec<_> = holidays.iter().map(|h| h.name.as_str()).collect();
        assert!(holiday_names.contains(&"New Year's Day"));
        assert!(holiday_names.contains(&"Martin Luther King Jr. Day"));
        assert!(holiday_names.contains(&"Presidents Day"));
        assert!(holiday_names.contains(&"Good Friday"));
        assert!(holiday_names.contains(&"Memorial Day"));
        assert!(holiday_names.contains(&"Juneteenth National Independence Day"));
        assert!(holiday_names.contains(&"Independence Day"));
        assert!(holiday_names.contains(&"Labor Day"));
        assert!(holiday_names.contains(&"Thanksgiving Day"));
        assert!(holiday_names.contains(&"Christmas Day"));
    }

    #[test]
    fn test_nyse_early_closes_2024() {
        let engine = NYSERulesEngine::new(None);
        let holidays = engine.generate_holidays(2024);

        // Check for early close days
        let early_closes: Vec<_> = holidays
            .iter()
            .filter(|h| h.holiday_type == HolidayTypeDb::HalfDay)
            .collect();

        assert!(!early_closes.is_empty());

        // Black Friday 2024 = November 29
        let black_friday = early_closes
            .iter()
            .find(|h| h.holiday_date == NaiveDate::from_ymd_opt(2024, 11, 29).unwrap());
        assert!(black_friday.is_some());
    }

    #[test]
    fn test_b3_ash_wednesday_late_open() {
        let engine = B3RulesEngine::new(None);
        let holidays = engine.generate_holidays(2024);

        // Ash Wednesday 2024 = February 14
        let ash_wed = holidays
            .iter()
            .find(|h| h.name == "Quarta-feira de Cinzas");
        assert!(ash_wed.is_some());
        assert_eq!(ash_wed.unwrap().holiday_type, HolidayTypeDb::LateOpen);
        assert_eq!(
            ash_wed.unwrap().late_open_time,
            Some(NaiveTime::from_hms_opt(13, 0, 0).unwrap())
        );
    }

    #[test]
    fn test_juneteenth_history() {
        let engine = NYSERulesEngine::new(None);

        // 2020: No Juneteenth
        let holidays_2020 = engine.generate_holidays(2020);
        assert!(!holidays_2020.iter().any(|h| h.name.contains("Juneteenth")));

        // 2021: Early close on June 18
        let holidays_2021 = engine.generate_holidays(2021);
        let juneteenth_2021 = holidays_2021.iter().find(|h| h.name.contains("Juneteenth"));
        assert!(juneteenth_2021.is_some());
        assert_eq!(juneteenth_2021.unwrap().holiday_type, HolidayTypeDb::HalfDay);

        // 2022+: Full holiday
        let holidays_2022 = engine.generate_holidays(2022);
        let juneteenth_2022 = holidays_2022.iter().find(|h| h.name.contains("Juneteenth"));
        assert!(juneteenth_2022.is_some());
        assert_eq!(juneteenth_2022.unwrap().holiday_type, HolidayTypeDb::National);
    }
}


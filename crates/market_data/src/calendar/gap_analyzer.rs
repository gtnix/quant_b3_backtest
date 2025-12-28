//! Gap Analyzer - Deterministic gap analysis with auditable reasons.
//!
//! Analyzes data gaps and produces deterministic, auditable `GapReason`
//! for each gap detected.

use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};

use super::{DayClassification, Market, MarketSessionCalendar, Severity};

/// Reason for a data gap (deterministic, auditable).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GapReason {
    /// Market was closed (weekend)
    Weekend,
    /// Market was closed (holiday)
    Holiday { name: String },
    /// Half-day with early close
    HalfDay { name: String },
    /// Extraordinary closure
    ExtraordinaryClosure { reason: String },
    /// Data requested before asset's IPO date
    BeforeIPO { ipo_date: NaiveDate },
    /// Data requested after asset was delisted
    AfterDelisting { delist_date: NaiveDate },
    /// Market was open but asset had no trades
    NoTrades { volume: i64 },
    /// Data missing (pipeline or data issue)
    MissingData { expected_bars: u32, found_bars: u32 },
    /// Bars outside trading hours
    OutsideTradingHours,
}

impl GapReason {
    /// Check if this gap is acceptable (not a data quality issue).
    pub fn is_acceptable(&self) -> bool {
        matches!(
            self,
            GapReason::Weekend
                | GapReason::Holiday { .. }
                | GapReason::HalfDay { .. }
                | GapReason::ExtraordinaryClosure { .. }
                | GapReason::BeforeIPO { .. }
                | GapReason::AfterDelisting { .. }
        )
    }

    /// Get the severity level for logging.
    pub fn severity(&self) -> Severity {
        match self {
            GapReason::Weekend
            | GapReason::Holiday { .. }
            | GapReason::HalfDay { .. }
            | GapReason::ExtraordinaryClosure { .. }
            | GapReason::BeforeIPO { .. }
            | GapReason::AfterDelisting { .. } => Severity::Info,
            GapReason::NoTrades { .. } => Severity::Info,
            GapReason::OutsideTradingHours => Severity::Warn,
            GapReason::MissingData { expected_bars, found_bars } => {
                if *found_bars == 0 {
                    Severity::Error
                } else if (*found_bars as f64 / *expected_bars as f64) < 0.9 {
                    Severity::Warn
                } else {
                    Severity::Info
                }
            }
        }
    }

    /// Format as a log-friendly code string.
    pub fn code(&self) -> String {
        match self {
            GapReason::Weekend => "WEEKEND".to_string(),
            GapReason::Holiday { name } => format!("HOLIDAY:{}", name),
            GapReason::HalfDay { name } => format!("HALFDAY:{}", name),
            GapReason::ExtraordinaryClosure { reason } => format!("EXTRAORDINARY:{}", reason),
            GapReason::BeforeIPO { ipo_date } => format!("BEFORE_IPO:{}", ipo_date),
            GapReason::AfterDelisting { delist_date } => format!("AFTER_DELIST:{}", delist_date),
            GapReason::NoTrades { volume } => format!("NO_TRADES:VOLUME_{}", volume),
            GapReason::MissingData { expected_bars, found_bars } => {
                format!("MISSING:EXPECTED_{}_FOUND_{}", expected_bars, found_bars)
            }
            GapReason::OutsideTradingHours => "OUTSIDE_SESSION".to_string(),
        }
    }
}

/// A single gap entry in the analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapEntry {
    pub date: NaiveDate,
    pub reason: GapReason,
    pub severity: Severity,
    pub expected_bars: Option<u32>,
    pub found_bars: Option<u32>,
}

impl GapEntry {
    /// Create a new gap entry.
    pub fn new(date: NaiveDate, reason: GapReason) -> Self {
        let severity = reason.severity();
        Self {
            date,
            reason,
            severity,
            expected_bars: None,
            found_bars: None,
        }
    }

    /// Create with bar counts.
    pub fn with_bars(date: NaiveDate, reason: GapReason, expected: u32, found: u32) -> Self {
        let severity = reason.severity();
        Self {
            date,
            reason,
            severity,
            expected_bars: Some(expected),
            found_bars: Some(found),
        }
    }

    /// Format as a log line.
    pub fn to_log(&self, market: Market, symbol: &str) -> String {
        format!(
            "{}:GAP:{:?}:{}:{}:{}",
            self.severity,
            market,
            symbol,
            self.date,
            self.reason.code()
        )
    }
}

/// Summary of gap analysis.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GapSummary {
    pub total_calendar_days: u32,
    pub total_trading_days: u32,
    pub days_with_data: u32,
    pub holidays: u32,
    pub weekends: u32,
    pub half_days: u32,
    pub extraordinary_closures: u32,
    pub no_trades_days: u32,
    pub missing_data_days: u32,
    pub before_ipo_days: u32,
    pub after_delist_days: u32,
    pub coverage_pct: f64,
}

impl GapSummary {
    /// Check if data quality is acceptable.
    pub fn is_acceptable(&self) -> bool {
        self.coverage_pct >= 0.95 && self.missing_data_days == 0
    }
}

/// Complete gap analysis report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapReport {
    pub symbol: String,
    pub market: Market,
    pub period_start: NaiveDate,
    pub period_end: NaiveDate,
    pub gaps: Vec<GapEntry>,
    pub summary: GapSummary,
    pub generated_at: DateTime<Utc>,
    pub calendar_version: String,
}

impl GapReport {
    /// Get gaps with errors only.
    pub fn errors(&self) -> Vec<&GapEntry> {
        self.gaps.iter().filter(|g| g.severity == Severity::Error).collect()
    }

    /// Get gaps with warnings.
    pub fn warnings(&self) -> Vec<&GapEntry> {
        self.gaps.iter().filter(|g| g.severity == Severity::Warn).collect()
    }

    /// Check if the data is acceptable (no errors).
    pub fn is_acceptable(&self) -> bool {
        self.errors().is_empty()
    }
}

/// Asset lifecycle information.
#[derive(Debug, Clone, Default)]
pub struct AssetLifecycle {
    pub ipo_date: Option<NaiveDate>,
    pub delist_date: Option<NaiveDate>,
}

impl AssetLifecycle {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_ipo(ipo_date: NaiveDate) -> Self {
        Self {
            ipo_date: Some(ipo_date),
            delist_date: None,
        }
    }

    pub fn with_delist(delist_date: NaiveDate) -> Self {
        Self {
            ipo_date: None,
            delist_date: Some(delist_date),
        }
    }

    pub fn is_active_on(&self, date: NaiveDate) -> bool {
        let after_ipo = self.ipo_date.map_or(true, |ipo| date >= ipo);
        let before_delist = self.delist_date.map_or(true, |delist| date <= delist);
        after_ipo && before_delist
    }
}

/// Bar data for a single day (for gap analysis input).
#[derive(Debug, Clone)]
pub struct DayBarData {
    pub date: NaiveDate,
    pub bar_count: u32,
    pub total_volume: i64,
}

/// Gap analyzer.
#[derive(Debug)]
pub struct GapAnalyzer {
    calendar: MarketSessionCalendar,
}

impl GapAnalyzer {
    /// Create a new gap analyzer.
    pub fn new() -> Self {
        Self {
            calendar: MarketSessionCalendar::new(),
        }
    }

    /// Create with a custom calendar.
    pub fn with_calendar(calendar: MarketSessionCalendar) -> Self {
        Self { calendar }
    }

    /// Analyze gaps for a symbol over a date range.
    ///
    /// # Arguments
    /// * `symbol` - The symbol being analyzed
    /// * `market` - The market for this symbol
    /// * `start` - Start date (inclusive)
    /// * `end` - End date (inclusive)
    /// * `data` - Available bar data for each day
    /// * `expected_bars_per_day` - Expected number of bars per trading day
    /// * `lifecycle` - Asset lifecycle info (IPO/delist dates)
    pub fn analyze(
        &self,
        symbol: &str,
        market: Market,
        start: NaiveDate,
        end: NaiveDate,
        data: &[DayBarData],
        expected_bars_per_day: u32,
        lifecycle: &AssetLifecycle,
    ) -> GapReport {
        let mut gaps = Vec::new();
        let mut summary = GapSummary::default();

        // Build a map of data by date
        let data_map: std::collections::HashMap<NaiveDate, &DayBarData> =
            data.iter().map(|d| (d.date, d)).collect();

        let mut current = start;
        while current <= end {
            summary.total_calendar_days += 1;

            let classification = self.calendar.classify_date(market, current);

            match &classification {
                DayClassification::Weekend => {
                    summary.weekends += 1;
                    gaps.push(GapEntry::new(current, GapReason::Weekend));
                }
                DayClassification::Holiday { name, .. } => {
                    summary.holidays += 1;
                    gaps.push(GapEntry::new(current, GapReason::Holiday { name: name.clone() }));
                }
                DayClassification::ExtraordinaryClosure { reason, .. } => {
                    summary.extraordinary_closures += 1;
                    gaps.push(GapEntry::new(
                        current,
                        GapReason::ExtraordinaryClosure { reason: reason.clone() },
                    ));
                }
                DayClassification::HalfDay { name, .. } => {
                    summary.half_days += 1;
                    summary.total_trading_days += 1;
                    // Half-days still count as trading days
                    self.analyze_trading_day(
                        current,
                        &data_map,
                        expected_bars_per_day / 2, // Approximate
                        lifecycle,
                        &mut gaps,
                        &mut summary,
                        Some(name.clone()),
                    );
                }
                DayClassification::TradingDay(_) => {
                    summary.total_trading_days += 1;
                    self.analyze_trading_day(
                        current,
                        &data_map,
                        expected_bars_per_day,
                        lifecycle,
                        &mut gaps,
                        &mut summary,
                        None,
                    );
                }
            }

            current += chrono::Duration::days(1);
        }

        // Calculate coverage
        if summary.total_trading_days > 0 {
            summary.coverage_pct =
                summary.days_with_data as f64 / summary.total_trading_days as f64 * 100.0;
        }

        GapReport {
            symbol: symbol.to_string(),
            market,
            period_start: start,
            period_end: end,
            gaps,
            summary,
            generated_at: Utc::now(),
            calendar_version: self.calendar.version(),
        }
    }

    /// Analyze a single trading day.
    fn analyze_trading_day(
        &self,
        date: NaiveDate,
        data_map: &std::collections::HashMap<NaiveDate, &DayBarData>,
        expected_bars: u32,
        lifecycle: &AssetLifecycle,
        gaps: &mut Vec<GapEntry>,
        summary: &mut GapSummary,
        half_day_name: Option<String>,
    ) {
        // Check lifecycle
        if let Some(ipo_date) = lifecycle.ipo_date {
            if date < ipo_date {
                summary.before_ipo_days += 1;
                gaps.push(GapEntry::new(date, GapReason::BeforeIPO { ipo_date }));
                return;
            }
        }

        if let Some(delist_date) = lifecycle.delist_date {
            if date > delist_date {
                summary.after_delist_days += 1;
                gaps.push(GapEntry::new(date, GapReason::AfterDelisting { delist_date }));
                return;
            }
        }

        // Check data availability
        match data_map.get(&date) {
            Some(day_data) => {
                if day_data.total_volume == 0 {
                    // Bars exist but no volume
                    summary.no_trades_days += 1;
                    gaps.push(GapEntry::new(date, GapReason::NoTrades { volume: 0 }));
                } else if day_data.bar_count < expected_bars {
                    // Partial data
                    let reason = if let Some(name) = half_day_name {
                        GapReason::HalfDay { name }
                    } else {
                        GapReason::MissingData {
                            expected_bars,
                            found_bars: day_data.bar_count,
                        }
                    };

                    if !matches!(reason, GapReason::HalfDay { .. }) {
                        summary.missing_data_days += 1;
                    }
                    gaps.push(GapEntry::with_bars(
                        date,
                        reason,
                        expected_bars,
                        day_data.bar_count,
                    ));
                } else {
                    // Complete data
                    summary.days_with_data += 1;
                }
            }
            None => {
                // No data at all
                summary.missing_data_days += 1;
                gaps.push(GapEntry::with_bars(
                    date,
                    GapReason::MissingData {
                        expected_bars,
                        found_bars: 0,
                    },
                    expected_bars,
                    0,
                ));
            }
        }
    }
}

impl Default for GapAnalyzer {
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
    fn test_gap_reason_acceptable() {
        assert!(GapReason::Weekend.is_acceptable());
        assert!(GapReason::Holiday { name: "Test".to_string() }.is_acceptable());
        assert!(GapReason::BeforeIPO { ipo_date: date(2024, 1, 1) }.is_acceptable());
        assert!(!GapReason::MissingData { expected_bars: 10, found_bars: 0 }.is_acceptable());
        assert!(!GapReason::NoTrades { volume: 0 }.is_acceptable());
    }

    #[test]
    fn test_gap_reason_severity() {
        assert_eq!(GapReason::Weekend.severity(), Severity::Info);
        assert_eq!(GapReason::Holiday { name: "Test".to_string() }.severity(), Severity::Info);
        assert_eq!(GapReason::NoTrades { volume: 0 }.severity(), Severity::Info);
        assert_eq!(
            GapReason::MissingData { expected_bars: 10, found_bars: 0 }.severity(),
            Severity::Error
        );
        assert_eq!(
            GapReason::MissingData { expected_bars: 10, found_bars: 5 }.severity(),
            Severity::Warn
        );
    }

    #[test]
    fn test_asset_lifecycle() {
        let lifecycle = AssetLifecycle {
            ipo_date: Some(date(2024, 6, 1)),
            delist_date: Some(date(2024, 12, 31)),
        };

        assert!(!lifecycle.is_active_on(date(2024, 5, 1))); // Before IPO
        assert!(lifecycle.is_active_on(date(2024, 6, 1))); // IPO date
        assert!(lifecycle.is_active_on(date(2024, 9, 1))); // During active
        assert!(lifecycle.is_active_on(date(2024, 12, 31))); // Delist date
        assert!(!lifecycle.is_active_on(date(2025, 1, 1))); // After delist
    }

    #[test]
    fn test_analyze_weekend() {
        let analyzer = GapAnalyzer::new();

        // Analyze a weekend
        let report = analyzer.analyze(
            "PETR4",
            Market::BR,
            date(2024, 12, 21), // Saturday
            date(2024, 12, 22), // Sunday
            &[],
            1,
            &AssetLifecycle::new(),
        );

        assert_eq!(report.summary.weekends, 2);
        assert_eq!(report.summary.total_trading_days, 0);
        assert!(report.is_acceptable());
    }

    #[test]
    fn test_analyze_holiday() {
        let analyzer = GapAnalyzer::new();

        // Analyze Christmas
        let report = analyzer.analyze(
            "PETR4",
            Market::BR,
            date(2024, 12, 25),
            date(2024, 12, 25),
            &[],
            1,
            &AssetLifecycle::new(),
        );

        assert_eq!(report.summary.holidays, 1);
        assert_eq!(report.gaps.len(), 1);
        assert!(matches!(report.gaps[0].reason, GapReason::Holiday { .. }));
    }

    #[test]
    fn test_analyze_missing_data() {
        let analyzer = GapAnalyzer::new();

        // Analyze a trading day with no data
        let report = analyzer.analyze(
            "PETR4",
            Market::BR,
            date(2024, 12, 23), // Monday
            date(2024, 12, 23),
            &[], // No data!
            450,
            &AssetLifecycle::new(),
        );

        assert_eq!(report.summary.missing_data_days, 1);
        assert!(!report.is_acceptable());

        let errors = report.errors();
        assert_eq!(errors.len(), 1);
    }

    #[test]
    fn test_analyze_with_data() {
        let analyzer = GapAnalyzer::new();

        let data = vec![DayBarData {
            date: date(2024, 12, 23),
            bar_count: 450,
            total_volume: 1_000_000,
        }];

        let report = analyzer.analyze(
            "PETR4",
            Market::BR,
            date(2024, 12, 23),
            date(2024, 12, 23),
            &data,
            450,
            &AssetLifecycle::new(),
        );

        assert_eq!(report.summary.days_with_data, 1);
        assert_eq!(report.summary.missing_data_days, 0);
        assert!(report.is_acceptable());
    }

    #[test]
    fn test_analyze_no_trades() {
        let analyzer = GapAnalyzer::new();

        let data = vec![DayBarData {
            date: date(2024, 12, 23),
            bar_count: 1,
            total_volume: 0, // No volume!
        }];

        let report = analyzer.analyze(
            "XYZZ4", // Illiquid stock
            Market::BR,
            date(2024, 12, 23),
            date(2024, 12, 23),
            &data,
            1,
            &AssetLifecycle::new(),
        );

        assert_eq!(report.summary.no_trades_days, 1);
        // NoTrades is INFO level, not an error
        assert!(report.is_acceptable());
    }

    #[test]
    fn test_analyze_before_ipo() {
        let analyzer = GapAnalyzer::new();

        let lifecycle = AssetLifecycle::with_ipo(date(2024, 6, 1));

        let report = analyzer.analyze(
            "NEWCO",
            Market::BR,
            date(2024, 5, 1),
            date(2024, 5, 31),
            &[],
            1,
            &lifecycle,
        );

        assert!(report.summary.before_ipo_days > 0);
        // BeforeIPO is acceptable
        assert!(report.is_acceptable());
    }

    #[test]
    fn test_gap_entry_to_log() {
        let entry = GapEntry::new(date(2024, 12, 25), GapReason::Holiday { name: "Natal".to_string() });
        let log = entry.to_log(Market::BR, "PETR4");

        assert!(log.contains("INFO"));
        assert!(log.contains("BR"));
        assert!(log.contains("PETR4"));
        assert!(log.contains("HOLIDAY:Natal"));
    }

    #[test]
    fn test_gap_summary_acceptable() {
        let good = GapSummary {
            total_trading_days: 100,
            days_with_data: 100,
            coverage_pct: 100.0,
            ..Default::default()
        };
        assert!(good.is_acceptable());

        let bad = GapSummary {
            total_trading_days: 100,
            days_with_data: 90,
            missing_data_days: 10,
            coverage_pct: 90.0,
            ..Default::default()
        };
        assert!(!bad.is_acceptable());
    }
}



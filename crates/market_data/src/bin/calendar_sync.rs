//! Calendar Sync Command - Compare and update calendar data with diff generation.
//!
//! Usage:
//!   cargo run --bin calendar_sync -- --market BR --dry-run
//!   cargo run --bin calendar_sync -- --market US --apply
//!   cargo run --bin calendar_sync -- --all --diff-only

use chrono::NaiveDate;
use clap::{Parser, ValueEnum};
use market_data::calendar::db_provider::HolidayTypeDb;
use market_data::calendar::rules_engine::{B3RulesEngine, NYSERulesEngine, RulesEngine};
use market_data::calendar::Market;
use std::collections::{HashMap, HashSet};
use std::env;
use tracing::{error, info};

/// Calendar Sync CLI
#[derive(Parser, Debug)]
#[command(name = "calendar_sync")]
#[command(about = "Sync and diff calendar data between rules engine and database")]
struct Args {
    /// Target market (or --all for both)
    #[arg(short, long, value_enum)]
    market: Option<MarketArg>,

    /// Sync all markets
    #[arg(long)]
    all: bool,

    /// Dry run - show diff without applying changes
    #[arg(long)]
    dry_run: bool,

    /// Only show diff, don't prompt for changes
    #[arg(long)]
    diff_only: bool,

    /// Apply changes without prompting
    #[arg(long)]
    apply: bool,

    /// Year range start (default: 2005)
    #[arg(long, default_value = "2005")]
    start_year: i32,

    /// Year range end (default: 2025)
    #[arg(long, default_value = "2025")]
    end_year: i32,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum MarketArg {
    BR,
    US,
}

impl From<MarketArg> for Market {
    fn from(m: MarketArg) -> Self {
        match m {
            MarketArg::BR => Market::BR,
            MarketArg::US => Market::US,
        }
    }
}

#[derive(Debug, Clone)]
struct HolidayDiff {
    date: NaiveDate,
    #[allow(dead_code)]
    diff_type: DiffType,
    expected_name: Option<String>,
    actual_name: Option<String>,
    expected_type: Option<String>,
    actual_type: Option<String>,
}

#[derive(Debug, Clone)]
enum DiffType {
    Missing,   // In rules but not in DB
    Extra,     // In DB but not in rules
    Mismatch,  // In both but different
}

impl std::fmt::Display for DiffType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiffType::Missing => write!(f, "MISSING"),
            DiffType::Extra => write!(f, "EXTRA"),
            DiffType::Mismatch => write!(f, "MISMATCH"),
        }
    }
}

#[derive(Debug, Default)]
struct SyncReport {
    market: String,
    year_range: String,
    total_expected: usize,
    total_actual: usize,
    missing: Vec<HolidayDiff>,
    extra: Vec<HolidayDiff>,
    mismatches: Vec<HolidayDiff>,
}

impl SyncReport {
    fn has_diffs(&self) -> bool {
        !self.missing.is_empty() || !self.extra.is_empty() || !self.mismatches.is_empty()
    }

    fn print(&self) {
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║  CALENDAR SYNC REPORT: {} ({})                       ║", self.market, self.year_range);
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  Expected (rules): {:>5}                                    ║", self.total_expected);
        println!("║  Actual (database): {:>4}                                    ║", self.total_actual);
        println!("╠══════════════════════════════════════════════════════════════╣");

        if !self.has_diffs() {
            println!("║  ✅ NO DIFFERENCES FOUND - Calendar is in sync!             ║");
            println!("╚══════════════════════════════════════════════════════════════╝\n");
            return;
        }

        // Missing holidays
        if !self.missing.is_empty() {
            println!("║                                                              ║");
            println!("║  🔴 MISSING ({} holidays not in database):                   ║", self.missing.len());
            println!("║  ────────────────────────────────────────────────────────    ║");
            for diff in &self.missing {
                println!("║    {} | {} ({:?})", 
                    diff.date, 
                    diff.expected_name.as_deref().unwrap_or("?"),
                    diff.expected_type.as_deref().unwrap_or("?")
                );
            }
        }

        // Extra holidays
        if !self.extra.is_empty() {
            println!("║                                                              ║");
            println!("║  🟡 EXTRA ({} holidays in DB but not in rules):              ║", self.extra.len());
            println!("║  ────────────────────────────────────────────────────────    ║");
            for diff in &self.extra {
                println!("║    {} | {} ({:?})", 
                    diff.date, 
                    diff.actual_name.as_deref().unwrap_or("?"),
                    diff.actual_type.as_deref().unwrap_or("?")
                );
            }
        }

        // Mismatches
        if !self.mismatches.is_empty() {
            println!("║                                                              ║");
            println!("║  🟠 MISMATCH ({} holidays with different data):              ║", self.mismatches.len());
            println!("║  ────────────────────────────────────────────────────────    ║");
            for diff in &self.mismatches {
                println!("║    {} | expected: {} | actual: {}", 
                    diff.date, 
                    diff.expected_name.as_deref().unwrap_or("?"),
                    diff.actual_name.as_deref().unwrap_or("?")
                );
            }
        }

        println!("║                                                              ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  SUMMARY:                                                    ║");
        println!("║    Missing:    {:>4}                                         ║", self.missing.len());
        println!("║    Extra:      {:>4}                                         ║", self.extra.len());
        println!("║    Mismatches: {:>4}                                         ║", self.mismatches.len());
        println!("║    TOTAL DIFFS: {:>3}                                         ║", 
            self.missing.len() + self.extra.len() + self.mismatches.len());
        println!("╚══════════════════════════════════════════════════════════════╝\n");
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    // Validate args
    if !args.all && args.market.is_none() {
        eprintln!("Error: Must specify --market or --all");
        std::process::exit(1);
    }

    // Load database URL
    dotenvy::dotenv().ok();
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");

    info!("Connecting to database...");

    // Connect with TLS for Neon
    let rustls_config = rustls::ClientConfig::builder()
        .with_root_certificates(rustls::RootCertStore {
            roots: webpki_roots::TLS_SERVER_ROOTS.iter().cloned().collect(),
        })
        .with_no_client_auth();

    let tls = tokio_postgres_rustls::MakeRustlsConnect::new(rustls_config);
    let (client, connection) = tokio_postgres::connect(&database_url, tls).await?;

    tokio::spawn(async move {
        if let Err(e) = connection.await {
            error!("Connection error: {}", e);
        }
    });

    info!("Connected to database");

    let markets: Vec<Market> = if args.all {
        vec![Market::BR, Market::US]
    } else {
        vec![args.market.unwrap().into()]
    };

    let mut all_reports = Vec::new();

    for market in markets {
        let report = sync_market(&client, market, args.start_year, args.end_year).await?;
        report.print();
        all_reports.push(report);
    }

    // Summary
    let total_diffs: usize = all_reports.iter()
        .map(|r| r.missing.len() + r.extra.len() + r.mismatches.len())
        .sum();

    if total_diffs == 0 {
        println!("✅ All calendars are in sync!");
    } else {
        println!("⚠️  Total differences found: {}", total_diffs);
        
        if args.diff_only || args.dry_run {
            println!("   (Dry run - no changes applied)");
        } else if args.apply {
            println!("   Applying changes...");
            // TODO: Implement apply logic
            println!("   Apply logic not yet implemented - use SQL directly for now");
        } else {
            println!("   Use --apply to apply changes, or --dry-run to skip");
        }
    }

    Ok(())
}

async fn sync_market(
    client: &tokio_postgres::Client,
    market: Market,
    start_year: i32,
    end_year: i32,
) -> Result<SyncReport, Box<dyn std::error::Error>> {
    let market_str = match market {
        Market::BR => "BR",
        Market::US => "US",
    };

    info!("Syncing {} calendar ({}-{})...", market_str, start_year, end_year);

    // Generate expected holidays from rules
    let expected: HashMap<NaiveDate, (String, HolidayTypeDb)> = match market {
        Market::BR => {
            let engine = B3RulesEngine::new(None);
            let mut map = HashMap::new();
            for year in start_year..=end_year {
                for h in engine.generate_holidays(year) {
                    map.insert(h.holiday_date, (h.name, h.holiday_type));
                }
            }
            map
        }
        Market::US => {
            let engine = NYSERulesEngine::new(None);
            let mut map = HashMap::new();
            for year in start_year..=end_year {
                for h in engine.generate_holidays(year) {
                    map.insert(h.holiday_date, (h.name, h.holiday_type));
                }
            }
            map
        }
    };

    // Fetch actual holidays from database
    let rows = client
        .query(
            "SELECT holiday_date, name, holiday_type 
             FROM holidays 
             WHERE market = $1 
               AND holiday_date >= $2 
               AND holiday_date <= $3
             ORDER BY holiday_date",
            &[
                &market_str,
                &NaiveDate::from_ymd_opt(start_year, 1, 1).unwrap(),
                &NaiveDate::from_ymd_opt(end_year, 12, 31).unwrap(),
            ],
        )
        .await?;

    let actual: HashMap<NaiveDate, (String, String)> = rows
        .iter()
        .map(|row| {
            let date: NaiveDate = row.get(0);
            let name: String = row.get(1);
            let htype: String = row.get(2);
            (date, (name, htype))
        })
        .collect();

    // Compare
    let expected_dates: HashSet<_> = expected.keys().cloned().collect();
    let actual_dates: HashSet<_> = actual.keys().cloned().collect();

    let mut report = SyncReport {
        market: market_str.to_string(),
        year_range: format!("{}-{}", start_year, end_year),
        total_expected: expected.len(),
        total_actual: actual.len(),
        ..Default::default()
    };

    // Missing: in expected but not in actual
    for date in expected_dates.difference(&actual_dates) {
        if let Some((name, htype)) = expected.get(date) {
            report.missing.push(HolidayDiff {
                date: *date,
                diff_type: DiffType::Missing,
                expected_name: Some(name.clone()),
                actual_name: None,
                expected_type: Some(format!("{:?}", htype)),
                actual_type: None,
            });
        }
    }

    // Extra: in actual but not in expected
    for date in actual_dates.difference(&expected_dates) {
        if let Some((name, htype)) = actual.get(date) {
            report.extra.push(HolidayDiff {
                date: *date,
                diff_type: DiffType::Extra,
                expected_name: None,
                actual_name: Some(name.clone()),
                expected_type: None,
                actual_type: Some(htype.clone()),
            });
        }
    }

    // Mismatches: in both but different
    for date in expected_dates.intersection(&actual_dates) {
        let (exp_name, exp_type) = expected.get(date).unwrap();
        let (act_name, act_type) = actual.get(date).unwrap();

        // Compare names (case-insensitive, ignore minor differences)
        let name_match = exp_name.to_lowercase().contains(&act_name.to_lowercase().split_whitespace().next().unwrap_or(""))
            || act_name.to_lowercase().contains(&exp_name.to_lowercase().split_whitespace().next().unwrap_or(""));

        if !name_match {
            report.mismatches.push(HolidayDiff {
                date: *date,
                diff_type: DiffType::Mismatch,
                expected_name: Some(exp_name.clone()),
                actual_name: Some(act_name.clone()),
                expected_type: Some(format!("{:?}", exp_type)),
                actual_type: Some(act_type.clone()),
            });
        }
    }

    // Sort by date
    report.missing.sort_by_key(|d| d.date);
    report.extra.sort_by_key(|d| d.date);
    report.mismatches.sort_by_key(|d| d.date);

    Ok(report)
}


//! Calendar Validator - Validate trading day counts against official sources.
//!
//! Compares database trading day counts with PREGOES.pdf (B3) and NYSE official data.
//!
//! Usage:
//!   cargo run --bin validate_calendar -- --market BR --year 2024
//!   cargo run --bin validate_calendar -- --market BR --years 2020-2025
//!   cargo run --bin validate_calendar -- --all

use clap::Parser;
use std::collections::HashMap;
use std::env;
use std::sync::Arc;
use tracing::{error, info};

/// Calendar Validator CLI
#[derive(Parser, Debug)]
#[command(name = "validate_calendar")]
#[command(about = "Validate trading day counts against official sources")]
struct Args {
    /// Market to validate (BR or US)
    #[arg(short, long)]
    market: Option<String>,

    /// Validate all markets
    #[arg(long)]
    all: bool,

    /// Specific year to validate
    #[arg(short, long)]
    year: Option<i32>,

    /// Year range (e.g., 2020-2025)
    #[arg(long)]
    years: Option<String>,

    /// Output as JSON
    #[arg(long)]
    json: bool,
}

/// Expected trading day counts from official sources.
fn get_expected_counts() -> HashMap<(&'static str, i32), i32> {
    let mut counts = HashMap::new();

    // B3 (PREGOES.pdf)
    counts.insert(("BR", 2005), 249);
    counts.insert(("BR", 2006), 248);
    counts.insert(("BR", 2007), 249);
    counts.insert(("BR", 2008), 250);
    counts.insert(("BR", 2009), 249);
    counts.insert(("BR", 2010), 249);
    counts.insert(("BR", 2011), 249);
    counts.insert(("BR", 2012), 250);
    counts.insert(("BR", 2013), 249);
    counts.insert(("BR", 2014), 249);
    counts.insert(("BR", 2015), 249);
    counts.insert(("BR", 2016), 251);
    counts.insert(("BR", 2017), 249);
    counts.insert(("BR", 2018), 247);
    counts.insert(("BR", 2019), 250);
    counts.insert(("BR", 2020), 250);
    counts.insert(("BR", 2021), 249);
    counts.insert(("BR", 2022), 249);
    counts.insert(("BR", 2023), 248);
    counts.insert(("BR", 2024), 249);
    counts.insert(("BR", 2025), 248);

    // NYSE (approximate - varies slightly by source)
    counts.insert(("US", 2005), 252);
    counts.insert(("US", 2006), 251);
    counts.insert(("US", 2007), 251);
    counts.insert(("US", 2008), 253);
    counts.insert(("US", 2009), 252);
    counts.insert(("US", 2010), 252);
    counts.insert(("US", 2011), 252);
    counts.insert(("US", 2012), 250); // Hurricane Sandy closures
    counts.insert(("US", 2013), 252);
    counts.insert(("US", 2014), 252);
    counts.insert(("US", 2015), 252);
    counts.insert(("US", 2016), 252);
    counts.insert(("US", 2017), 251);
    counts.insert(("US", 2018), 251); // Bush mourning
    counts.insert(("US", 2019), 252);
    counts.insert(("US", 2020), 253);
    counts.insert(("US", 2021), 252);
    counts.insert(("US", 2022), 251);
    counts.insert(("US", 2023), 250);
    counts.insert(("US", 2024), 252);
    counts.insert(("US", 2025), 251);

    counts
}

#[derive(Debug)]
struct ValidationResult {
    market: String,
    year: i32,
    expected: i32,
    actual: i32,
    is_valid: bool,
}

impl ValidationResult {
    fn new(market: &str, year: i32, expected: i32, actual: i32) -> Self {
        Self {
            market: market.to_string(),
            year,
            expected,
            actual,
            is_valid: expected == actual,
        }
    }
}

async fn count_trading_days(
    client: &tokio_postgres::Client,
    market: &str,
    year: i32,
) -> Result<i32, Box<dyn std::error::Error>> {
    let row = client
        .query_one(
            "SELECT COUNT(*) FROM trading_sessions 
             WHERE market = $1 
               AND day_type != 'CLOSED'
               AND EXTRACT(YEAR FROM session_date) = $2",
            &[&market, &(year as f64)],
        )
        .await?;

    Ok(row.get::<_, i64>(0) as i32)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    // Load database URL
    dotenvy::dotenv().ok();
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");

    // Connect to database
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

    let client = Arc::new(client);

    // Determine markets and years to validate
    let markets: Vec<&str> = if args.all {
        vec!["BR", "US"]
    } else if let Some(ref m) = args.market {
        vec![m.as_str()]
    } else {
        eprintln!("Error: Must specify --market or --all");
        std::process::exit(1);
    };

    let years: Vec<i32> = if let Some(y) = args.year {
        vec![y]
    } else if let Some(ref range) = args.years {
        let parts: Vec<&str> = range.split('-').collect();
        if parts.len() != 2 {
            eprintln!("Error: Invalid year range format. Use YYYY-YYYY");
            std::process::exit(1);
        }
        let start: i32 = parts[0].parse()?;
        let end: i32 = parts[1].parse()?;
        (start..=end).collect()
    } else {
        // Default to 2005-2025
        (2005..=2025).collect()
    };

    let expected_counts = get_expected_counts();
    let mut results: Vec<ValidationResult> = Vec::new();
    let mut all_valid = true;

    println!("{}", "=".repeat(60));
    println!("Calendar Validation Report");
    println!("{}", "=".repeat(60));

    for market in &markets {
        println!("\n{} Market:", market);
        println!("{}", "-".repeat(40));

        for year in &years {
            let expected = expected_counts.get(&(*market, *year)).copied();

            if expected.is_none() {
                println!("  {}: No expected data", year);
                continue;
            }

            let expected = expected.unwrap();
            let actual = count_trading_days(&client, market, *year).await?;

            let result = ValidationResult::new(market, *year, expected, actual);

            let status = if result.is_valid { "✅" } else { "❌" };
            let diff = if result.is_valid {
                String::new()
            } else {
                format!(" (diff: {:+})", actual - expected)
            };

            println!(
                "  {} {}: Expected {}, Found {}{}",
                status, year, expected, actual, diff
            );

            if !result.is_valid {
                all_valid = false;
            }

            results.push(result);
        }
    }

    println!("\n{}", "=".repeat(60));

    let valid_count = results.iter().filter(|r| r.is_valid).count();
    let total_count = results.len();

    println!("Summary: {}/{} validations passed", valid_count, total_count);

    if all_valid {
        println!("✅ All validations passed!");
        Ok(())
    } else {
        println!("❌ Some validations failed - review required!");
        std::process::exit(1);
    }
}









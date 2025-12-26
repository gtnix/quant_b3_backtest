//! Monitoring Baseline Runner
//!
//! CLI tool to run monitoring checks over historical data and generate baseline report.
//!
//! Usage:
//!   cargo run -p backtester_intelligence --bin monitoring_baseline -- --days 30
//!   
//! Requires DATABASE_URL environment variable to be set.

use backtester_intelligence::filters::Market;
use backtester_intelligence::monitoring::{
    BaselineAggregator, ContextBuilder, DailyResult,
    MonitoringConfig, MonitoringContext, MonitoringEngine,
};
use chrono::{Datelike, Duration, Utc};
use clap::Parser;
use std::path::PathBuf;
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;

/// Monitoring Baseline Runner - analyze historical data health
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Number of days to analyze
    #[arg(long, default_value = "30")]
    days: u32,

    /// Markets to analyze (comma-separated: BR,US)
    #[arg(long, default_value = "BR,US")]
    markets: String,

    /// Output JSON file path
    #[arg(long)]
    output_json: Option<PathBuf>,

    /// Output Markdown file path
    #[arg(long)]
    output_md: Option<PathBuf>,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();

    let cli = Cli::parse();

    // Setup logging
    let log_level = if cli.verbose { Level::DEBUG } else { Level::INFO };
    let subscriber = FmtSubscriber::builder()
        .with_max_level(log_level)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("Monitoring Baseline Runner");
    info!("==========================");

    // Parse markets
    let markets: Vec<Market> = cli.markets
        .split(',')
        .filter_map(|s| match s.trim().to_uppercase().as_str() {
            "BR" => Some(Market::BR),
            "US" => Some(Market::US),
            _ => None,
        })
        .collect();

    if markets.is_empty() {
        return Err("No valid markets specified".into());
    }

    info!("Markets: {:?}", markets);
    info!("Days to analyze: {}", cli.days);

    // Connect to database
    let database_url = std::env::var("DATABASE_URL")
        .map_err(|_| "DATABASE_URL not set. Please set it in .env or environment.")?;

    info!("Connecting to database...");
    let (client, connection) = connect_to_neon(&database_url).await?;

    // Spawn connection handler
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("Database connection error: {}", e);
        }
    });

    info!("Connected to Neon database");

    // Create context builder
    let context_builder = ContextBuilder::new(&client);

    // Create monitoring engine
    let config = MonitoringConfig::default();
    let mut engine = MonitoringEngine::new(config);

    // Calculate date range
    let end_date = Utc::now().date_naive();
    let start_date = end_date - Duration::days(cli.days as i64);

    info!("Analyzing period: {} to {}", start_date, end_date);

    // Create aggregator
    let mut aggregator = BaselineAggregator::new();
    aggregator.set_markets(markets.clone());

    // Run backfill
    let mut current_date = start_date;
    let mut days_processed = 0;

    while current_date <= end_date {
        // Skip weekends
        let weekday = current_date.weekday();
        if weekday == chrono::Weekday::Sat || weekday == chrono::Weekday::Sun {
            current_date = current_date + Duration::days(1);
            continue;
        }

        info!("Processing: {}", current_date);

        // Build context for this date
        match context_builder.build_data_context(current_date, &markets).await {
            Ok(data_ctx) => {
                let drift_ctx = context_builder.build_drift_context(current_date, 60).await
                    .unwrap_or_default();
                let regression_ctx = context_builder.build_regression_context(current_date).await
                    .unwrap_or_default();

                // Build full monitoring context
                let mut ctx = MonitoringContext::new(current_date);
                ctx.data = data_ctx;
                ctx.drift = drift_ctx;
                ctx.regression = regression_ctx;
                ctx.markets = markets.clone();

                // Run monitoring
                let report = engine.run_all(&ctx);

                // Create daily result
                let daily = DailyResult::from_report(current_date, &report);

                info!(
                    "  INFO: {}, WARN: {}, CRIT: {}, HALT: {} | Action: {}",
                    daily.info_count, daily.warn_count, daily.crit_count, 
                    daily.halt_count, report.action
                );

                aggregator.add(daily);
                days_processed += 1;
            }
            Err(e) => {
                warn!("Failed to build context for {}: {:?}", current_date, e);
            }
        }

        current_date = current_date + Duration::days(1);
    }

    info!("\n");
    info!("Processed {} business days", days_processed);

    // Generate report
    let report = aggregator.generate_report();

    // Output results
    println!("\n{}", "=".repeat(80));
    println!("{}", report.to_markdown());
    println!("{}", "=".repeat(80));

    // Save to files if requested
    if let Some(json_path) = cli.output_json {
        let json = serde_json::to_string_pretty(&report)?;
        std::fs::write(&json_path, json)?;
        info!("JSON report saved to: {:?}", json_path);
    }

    if let Some(md_path) = cli.output_md {
        let md = report.to_markdown();
        std::fs::write(&md_path, md)?;
        info!("Markdown report saved to: {:?}", md_path);
    }

    Ok(())
}

/// Connect to Neon PostgreSQL with TLS.
async fn connect_to_neon(
    database_url: &str,
) -> Result<
    (
        tokio_postgres::Client,
        impl std::future::Future<Output = Result<(), tokio_postgres::Error>>,
    ),
    Box<dyn std::error::Error>,
> {
    let root_store =
        rustls::RootCertStore::from_iter(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());

    let config = rustls::ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth();

    let tls = tokio_postgres_rustls::MakeRustlsConnect::new(config);

    let (client, connection) = tokio_postgres::connect(database_url, tls).await?;

    Ok((client, connection))
}


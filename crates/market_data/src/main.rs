//! Market Data Ingestion CLI
//!
//! Pipeline for ingesting B3 OHLCV data from Brapi into Neon Postgres.
//!
//! ## Environment Variables
//!
//! Required:
//! - `BRAPI_TOKEN`: API authentication token from brapi.dev
//! - `DATABASE_URL`: Neon Postgres connection string
//!
//! Optional:
//! - `BRAPI_BASE_URL`: API base URL (default: https://brapi.dev)
//! - `BRAPI_REQUESTS_PER_MINUTE`: Rate limit (default: 60)
//! - `BRAPI_MAX_RETRIES`: Max retries per request (default: 3)
//! - `BRAPI_TIMEOUT_SECS`: Request timeout (default: 30)

#![allow(dead_code)] // Campos reservados para expansão futura

mod aggregation_executor;
mod aggregation_planner;
mod audit_integrity;
mod brapi;
pub mod calendar;
mod contract_tests;
mod coverage_model;
mod db;
mod executor;
mod ingest;
mod interest_rates;
mod inventory;
mod planner;
mod probe;
mod reports;
mod universe_gate;
mod validator;

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser)]
#[command(name = "market-data")]
#[command(about = "B3 Market Data Ingestion Pipeline - brapi.dev provider")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Probe brapi.dev API capabilities (intervals, ranges, intraday support)
    Probe {
        /// Number of tickers to sample (default: 20)
        #[arg(short, long, default_value = "20")]
        sample: usize,

        /// Probe ALL tickers in universe (overrides --sample)
        #[arg(short, long)]
        full: bool,

        /// Output directory for artifacts
        #[arg(short, long, default_value = "output/probe")]
        output: PathBuf,

        /// Test specific interval/range combinations (slower)
        #[arg(long)]
        test_combos: bool,

        /// Persist discovered capabilities to database
        #[arg(long)]
        persist_db: bool,
    },

    /// Discover universe of available tickers
    DiscoverUniverse {
        /// Output CSV file
        #[arg(short, long, default_value = "output/universe.csv")]
        output: PathBuf,
    },

    /// Initialize database schema
    InitDb,

    /// Refresh universe (top N by volume)
    RefreshUniverse {
        #[arg(short, long, default_value = "150")]
        target: usize,
    },

    /// Backfill historical data (daily)
    Backfill {
        #[arg(short, long, default_value = "top_volume")]
        universe: String,

        #[arg(short, long, default_value = "max")]
        range: String,
    },

    /// Backfill intraday data
    BackfillIntraday {
        #[arg(short, long, default_value = "top_volume")]
        universe: String,

        /// Interval: 1m, 5m, 15m, 30m, 60m
        #[arg(short, long, default_value = "5m")]
        interval: String,

        /// Range for intraday (1d, 5d, 1mo)
        #[arg(short, long, default_value = "5d")]
        range: String,
    },

    /// Incremental update
    Update {
        #[arg(short, long, default_value = "top_volume")]
        universe: String,
    },

    /// Verify data integrity
    VerifyIntegrity,

    /// Show ingestion status
    Status,

    /// Generate data freshness report
    FreshnessReport {
        /// Output file
        #[arg(short, long, default_value = "output/freshness_report.csv")]
        output: PathBuf,
    },

    /// Validate brapi.dev connection and token
    ValidateConnection,

    // ========================================================================
    // Stress Ingestion Commands
    // ========================================================================
    /// Generate stress download plan (dry-run planning)
    PlanStress {
        /// Output file for the plan
        #[arg(long, default_value = "output/stress_plan.json")]
        output: PathBuf,

        /// Only include specific intervals (comma-separated: 1d,60m,5m)
        #[arg(long)]
        intervals: Option<String>,

        /// Limit to N tickers (for testing)
        #[arg(long)]
        limit: Option<usize>,

        /// Only backfill tasks (no incremental)
        #[arg(long)]
        backfill_only: bool,

        /// Only incremental tasks (no backfill)
        #[arg(long)]
        incremental_only: bool,
    },

    /// Execute full historical download
    StressDownload {
        /// Path to stress plan JSON
        #[arg(long)]
        plan: PathBuf,

        /// Output directory for artifacts
        #[arg(long, default_value = "output/stress")]
        output: PathBuf,

        /// Concurrency level
        #[arg(long, default_value = "1")]
        concurrency: usize,

        /// Max requests (budget limit)
        #[arg(long)]
        budget: Option<usize>,

        /// Abort after N consecutive failures
        #[arg(long, default_value = "50")]
        max_failures: usize,

        /// Resume from a previous run's plan
        #[arg(long)]
        resume: Option<PathBuf>,

        /// Only show what would be done, no actual requests
        #[arg(long)]
        dry_run: bool,
    },

    /// Incremental sync (update based on watermarks)
    Sync {
        /// Output directory for artifacts
        #[arg(long, default_value = "output/sync")]
        output: PathBuf,

        /// Only sync specific intervals (comma-separated)
        #[arg(long)]
        intervals: Option<String>,

        /// Max tasks per run
        #[arg(long)]
        max_tasks: Option<usize>,

        /// Concurrency level
        #[arg(long, default_value = "1")]
        concurrency: usize,
    },

    /// Generate coverage and freshness reports
    GenerateReports {
        /// Output directory for reports
        #[arg(long, default_value = "output/reports")]
        output: PathBuf,
    },

    // ========================================================================
    // Provider Universe Commands
    // ========================================================================
    /// Refresh provider universe from /api/quote/list
    UniverseRefresh {
        /// Output directory for artifacts
        #[arg(long, default_value = "output/universe")]
        output: PathBuf,

        /// Filter by asset type (stock, fund, bdr, all)
        #[arg(long, default_value = "all")]
        asset_type: String,
    },

    /// Show provider universe status
    UniverseStatus,

    /// Run provider contract tests (List -> Fetch sanity check)
    UniverseContractTest {
        /// Number of tickers to sample
        #[arg(long, default_value = "50")]
        sample: usize,

        /// Output directory for artifacts
        #[arg(long, default_value = "output/contract_test")]
        output: PathBuf,
    },

    /// Export universe snapshot
    UniverseSnapshot {
        /// Snapshot ID (or "latest")
        #[arg(long, default_value = "latest")]
        id: String,

        /// Output directory
        #[arg(long, default_value = "output/universe")]
        output: PathBuf,
    },

    // ========================================================================
    // Aggregation Intelligence Commands
    // ========================================================================
    /// Scan database and generate data inventory
    AggregateInventory {
        /// Output directory for inventory files
        #[arg(long, default_value = "output/aggregate")]
        output: PathBuf,

        /// Staleness threshold in days
        #[arg(long, default_value = "3")]
        stale_days: i64,
    },

    /// Generate aggregation plan (no API calls)
    AggregatePlan {
        /// Output directory for plan files
        #[arg(long, default_value = "output/aggregate")]
        output: PathBuf,

        /// Only generate sync plan (stale items only)
        #[arg(long)]
        sync_only: bool,
    },

    /// Execute aggregation plan
    AggregateRun {
        /// Path to aggregation plan JSON
        #[arg(long)]
        plan: PathBuf,

        /// Output directory for execution artifacts
        #[arg(long, default_value = "output/aggregate")]
        output: PathBuf,

        /// Concurrency level
        #[arg(long, default_value = "2")]
        concurrency: usize,

        /// Max requests (budget limit)
        #[arg(long)]
        budget: Option<usize>,

        /// Max consecutive failures before abort
        #[arg(long, default_value = "50")]
        max_failures: usize,
    },

    /// Quick sync: plan + run for stale data only
    AggregateSync {
        /// Output directory
        #[arg(long, default_value = "output/aggregate")]
        output: PathBuf,

        /// Concurrency level
        #[arg(long, default_value = "2")]
        concurrency: usize,

        /// Max requests
        #[arg(long)]
        budget: Option<usize>,
    },

    /// Show aggregation status (coverage summary)
    AggregateStatus,

    // ========================================================================
    // Audit Commands
    // ========================================================================
    /// Audit OHLCV data integrity (read-only, console output only)
    AuditIntegrity {
        /// Minimum cap-aware integrity to pass (default: 0.95)
        #[arg(long, default_value = "0.95")]
        min_integrity: f64,

        /// Number of outlier samples to show per interval
        #[arg(long, default_value = "10")]
        sample_outliers: usize,
    },

    // ========================================================================
    // Fundamentals Sync Commands
    // ========================================================================
    /// Sync fundamental data from Brapi API
    SyncFundamentals {
        /// Sync all active symbols
        #[arg(long)]
        all: bool,

        /// Specific symbols to sync (comma-separated)
        #[arg(long, value_delimiter = ',')]
        symbols: Option<Vec<String>>,

        /// Batch size for API requests (max 20)
        #[arg(long, default_value = "20")]
        batch_size: usize,
    },

    /// Show fundamentals sync status
    FundamentalsStatus,

    // ========================================================================
    // Interest Rates Commands
    // ========================================================================
    /// Sync interest rates from BCB (BR) and FRED (US)
    SyncInterestRates {
        /// Sync BR rates (BCB SELIC)
        #[arg(long)]
        br: bool,

        /// Sync US rates (FRED T-Bill)
        #[arg(long)]
        us: bool,

        /// Sync both BR and US
        #[arg(long)]
        all: bool,

        /// Start date (YYYY-MM-DD), default: 5 years ago
        #[arg(long)]
        start: Option<String>,

        /// End date (YYYY-MM-DD), default: today
        #[arg(long)]
        end: Option<String>,
    },

    /// Show interest rates status
    InterestRatesStatus,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();

    let cli = Cli::parse();

    let log_level = if cli.verbose {
        Level::DEBUG
    } else {
        Level::INFO
    };
    let subscriber = FmtSubscriber::builder()
        .with_max_level(log_level)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    match cli.command {
        Commands::ValidateConnection => {
            validate_connection().await?;
        }

        Commands::Probe {
            sample,
            full,
            output,
            test_combos,
            persist_db,
        } => {
            run_probe(sample, full, output, test_combos, persist_db).await?;
        }

        Commands::DiscoverUniverse { output } => {
            discover_universe(output).await?;
        }

        Commands::InitDb => {
            let db = db::Database::connect().await?;
            info!("Verifying database schema...");
            db.verify_schema().await?;
            info!("Schema verification complete");
        }

        Commands::RefreshUniverse { target } => {
            let db = db::Database::connect().await?;
            let brapi = brapi::BrapiClient::new()?;
            info!("Refreshing universe with top {} by volume", target);
            ingest::refresh_universe(&db, &brapi, target).await?;
        }

        Commands::Backfill { universe, range } => {
            let db = db::Database::connect().await?;
            let brapi = brapi::BrapiClient::new()?;
            info!(
                "Starting backfill for universe '{}' with range '{}'",
                universe, range
            );
            ingest::backfill(&db, &brapi, &universe, &range).await?;
        }

        Commands::BackfillIntraday {
            universe,
            interval,
            range,
        } => {
            let db = db::Database::connect().await?;
            let brapi = brapi::BrapiClient::new()?;
            info!(
                "Starting intraday backfill for '{}' with interval={}, range={}",
                universe, interval, range
            );
            ingest::backfill_intraday(&db, &brapi, &universe, &interval, &range).await?;
        }

        Commands::Update { universe } => {
            let db = db::Database::connect().await?;
            let brapi = brapi::BrapiClient::new()?;
            info!("Running incremental update for universe '{}'", universe);
            ingest::update_incremental(&db, &brapi, &universe).await?;
        }

        Commands::VerifyIntegrity => {
            let db = db::Database::connect().await?;
            info!("Verifying data integrity");
            ingest::verify_integrity(&db).await?;
        }

        Commands::Status => {
            let db = db::Database::connect().await?;
            ingest::show_status(&db).await?;
        }

        Commands::FreshnessReport { output } => {
            let db = db::Database::connect().await?;
            info!("Generating freshness report...");
            ingest::generate_freshness_report(&db, &output).await?;
            info!("Freshness report saved to: {}", output.display());
        }

        Commands::PlanStress {
            output,
            intervals,
            limit,
            backfill_only,
            incremental_only,
        } => {
            run_plan_stress(output, intervals, limit, backfill_only, incremental_only).await?;
        }

        Commands::StressDownload {
            plan,
            output,
            concurrency,
            budget,
            max_failures,
            resume,
            dry_run,
        } => {
            run_stress_download(
                plan,
                output,
                concurrency,
                budget,
                max_failures,
                resume,
                dry_run,
            )
            .await?;
        }

        Commands::Sync {
            output,
            intervals,
            max_tasks,
            concurrency,
        } => {
            run_sync(output, intervals, max_tasks, concurrency).await?;
        }

        Commands::GenerateReports { output } => {
            run_generate_reports(output).await?;
        }

        Commands::UniverseRefresh { output, asset_type } => {
            run_universe_refresh(output, asset_type).await?;
        }

        Commands::UniverseStatus => {
            run_universe_status().await?;
        }

        Commands::UniverseContractTest { sample, output } => {
            run_universe_contract_test(sample, output).await?;
        }

        Commands::UniverseSnapshot { id, output } => {
            run_universe_snapshot(id, output).await?;
        }

        // ====================================================================
        // Aggregation Intelligence Commands
        // ====================================================================
        Commands::AggregateInventory { output, stale_days } => {
            run_aggregate_inventory(output, stale_days).await?;
        }

        Commands::AggregatePlan { output, sync_only } => {
            run_aggregate_plan(output, sync_only).await?;
        }

        Commands::AggregateRun {
            plan,
            output,
            concurrency,
            budget,
            max_failures,
        } => {
            run_aggregate_execute(plan, output, concurrency, budget, max_failures).await?;
        }

        Commands::AggregateSync {
            output,
            concurrency,
            budget,
        } => {
            run_aggregate_sync(output, concurrency, budget).await?;
        }

        Commands::AggregateStatus => {
            run_aggregate_status().await?;
        }

        Commands::AuditIntegrity {
            min_integrity,
            sample_outliers,
        } => {
            run_audit_integrity(min_integrity, sample_outliers).await?;
        }

        Commands::SyncFundamentals {
            all,
            symbols,
            batch_size,
        } => {
            run_sync_fundamentals(all, symbols, batch_size).await?;
        }

        Commands::FundamentalsStatus => {
            run_fundamentals_status().await?;
        }

        Commands::SyncInterestRates {
            br,
            us,
            all,
            start,
            end,
        } => {
            run_sync_interest_rates(br, us, all, start, end).await?;
        }

        Commands::InterestRatesStatus => {
            run_interest_rates_status().await?;
        }
    }

    Ok(())
}

/// Validate brapi.dev connection and token.
async fn validate_connection() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== brapi.dev Connection Validation ===\n");

    // Check BRAPI_TOKEN
    match std::env::var("BRAPI_TOKEN") {
        Ok(token) => {
            let masked = if token.len() > 8 {
                format!("{}...{}", &token[..4], &token[token.len() - 4..])
            } else {
                "****".to_string()
            };
            println!("[OK] BRAPI_TOKEN is set: {}", masked);
        }
        Err(_) => {
            println!("[ERROR] BRAPI_TOKEN is NOT set");
            println!("       Get your token at: https://brapi.dev/dashboard");
            return Err("BRAPI_TOKEN not set".into());
        }
    }

    // Try to create client
    let client = match brapi::BrapiClient::new() {
        Ok(c) => {
            println!("[OK] BrapiClient initialized");
            println!("     Config: {}", c.config().redacted());
            c
        }
        Err(e) => {
            println!("[ERROR] Failed to create client: {}", e);
            return Err(e.into());
        }
    };

    // Test with public ticker (PETR4)
    println!("\nTesting API with PETR4 (public ticker, no auth required)...");
    match client.fetch_quote_with_metadata("PETR4").await {
        Ok((result, metrics)) => {
            println!("[OK] API responded successfully");
            println!("     Symbol: {}", result.symbol);
            println!("     Valid intervals: {:?}", result.valid_intervals);
            println!("     Valid ranges: {:?}", result.valid_ranges);
            println!("     Has intraday: {}", result.has_intraday());
            println!("     Latency: {}ms", metrics.duration_ms);
        }
        Err(e) => {
            println!("[ERROR] API request failed: {}", e);
            return Err(e.into());
        }
    }

    // Test pagination
    println!("\nTesting list endpoint (pagination)...");
    match client.list_stocks(10, 1).await {
        Ok((stocks, metrics)) => {
            println!("[OK] List endpoint works");
            println!("     Returned {} stocks", stocks.len());
            println!("     Latency: {}ms", metrics.duration_ms);
        }
        Err(e) => {
            println!("[ERROR] List request failed: {}", e);
            return Err(e.into());
        }
    }

    // Test authenticated request (if token works for non-test tickers)
    println!("\nTesting authenticated request (BBDC4)...");
    match client.fetch_quote_with_metadata("BBDC4").await {
        Ok((result, metrics)) => {
            println!("[OK] Authenticated request successful");
            println!("     Symbol: {}", result.symbol);
            println!("     Latency: {}ms", metrics.duration_ms);
        }
        Err(brapi::BrapiError::Unauthorized) => {
            println!("[WARN] Token may be invalid or quota exceeded");
        }
        Err(brapi::BrapiError::QuotaExceeded) => {
            println!("[WARN] API quota exceeded for current billing period");
        }
        Err(e) => {
            println!("[WARN] Request failed: {}", e);
        }
    }

    println!("\n=== Validation Complete ===\n");
    Ok(())
}

/// Run capability probe.
async fn run_probe(
    sample_size: usize,
    full: bool,
    output: PathBuf,
    _test_combos: bool,
    persist_db: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== brapi.dev Capability Probe ===\n");

    let client = brapi::BrapiClient::new()?;
    let mut probe = probe::CapabilityProbe::new(client);

    // Discover universe
    let universe_size = probe.discover_universe().await?;
    println!("Discovered {} tickers in universe\n", universe_size);

    // Run probe
    if full {
        println!("Running FULL probe (all {} tickers)...", universe_size);
        probe.probe_full().await?;
    } else {
        println!("Running sample probe ({} tickers)...", sample_size);
        probe.probe_sample(sample_size).await?;
    }

    // Generate manifest
    let manifest = probe.generate_manifest(full, if full { None } else { Some(sample_size) });

    // Print summary
    let stats = &manifest.stats;
    println!("\n=== Probe Results ===\n");
    println!(
        "Total tickers discovered: {}",
        stats.total_tickers_discovered
    );
    println!("Tickers probed:           {}", stats.tickers_probed);
    println!("Successful probes:        {}", stats.successful_probes);
    println!("Failed probes:            {}", stats.failed_probes);
    println!("---");
    println!(
        "With intraday support:    {} ({:.1}%)",
        stats.tickers_with_intraday,
        if stats.successful_probes > 0 {
            stats.tickers_with_intraday as f64 / stats.successful_probes as f64 * 100.0
        } else {
            0.0
        }
    );
    println!("Daily only:               {}", stats.tickers_daily_only);
    println!("No data:                  {}", stats.tickers_no_data);
    println!("---");
    println!(
        "Duration:                 {:.1}s",
        stats.total_duration_secs
    );

    if !stats.errors_by_type.is_empty() {
        println!("\nError breakdown:");
        for (error_type, count) in &stats.errors_by_type {
            println!("  {}: {}", error_type, count);
        }
    }

    // Persist to database if requested
    if persist_db {
        println!("\nPersisting capabilities to database...");
        let database = db::Database::connect().await?;
        let mut persisted = 0;

        for result in probe.get_successful_results() {
            if let Some(caps) = &result.capabilities {
                if let Err(e) = database.upsert_ticker_capabilities(caps).await {
                    warn!("Failed to persist {}: {}", result.symbol, e);
                } else {
                    persisted += 1;
                }
            }
        }

        println!("Persisted {} ticker capabilities to database", persisted);
    }

    // Write artifacts
    println!("\nWriting artifacts to {}...", output.display());
    std::fs::create_dir_all(&output)?;
    probe.write_artifacts(&output, &manifest)?;

    println!("\n=== Probe Complete ===\n");
    println!("Artifacts:");
    println!("  - {}/capabilities_manifest.json", output.display());
    println!("  - {}/capability_matrix.csv", output.display());
    println!("  - {}/universe.csv", output.display());
    println!("  - {}/failures.json", output.display());
    println!("  - {}/sample_payloads/", output.display());

    Ok(())
}

/// Discover universe and save to CSV.
async fn discover_universe(output: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Universe Discovery ===\n");

    let client = brapi::BrapiClient::new()?;
    let (stocks, metrics) = client.discover_universe().await?;

    println!(
        "Discovered {} tickers across {} API calls",
        stocks.len(),
        metrics.len()
    );

    // Create output directory if needed
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Write CSV
    let mut content = String::from("symbol,name,sector,type,market_cap,volume,close\n");
    for stock in &stocks {
        content.push_str(&format!(
            "{},{},{},{},{},{},{}\n",
            stock.stock,
            stock.name.as_deref().unwrap_or("").replace(',', ";"),
            stock.sector.as_deref().unwrap_or(""),
            stock.asset_type.as_deref().unwrap_or(""),
            stock.market_cap.unwrap_or(0),
            stock.volume.unwrap_or(0),
            stock.close.unwrap_or(0.0),
        ));
    }

    std::fs::write(&output, content)?;
    println!("Universe saved to: {}", output.display());

    // Print summary by type
    let mut by_type: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for stock in &stocks {
        let t = stock.asset_type.as_deref().unwrap_or("unknown");
        *by_type.entry(t.to_string()).or_insert(0) += 1;
    }
    println!("\nBreakdown by type:");
    for (t, count) in &by_type {
        println!("  {}: {}", t, count);
    }

    Ok(())
}

// ============================================================================
// Stress Ingestion Handlers
// ============================================================================

/// Generate stress download plan.
async fn run_plan_stress(
    output: PathBuf,
    intervals: Option<String>,
    limit: Option<usize>,
    backfill_only: bool,
    incremental_only: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Stress Plan Generation ===\n");

    let db = db::Database::connect().await?;

    let intervals_filter = intervals.map(|s| s.split(',').map(|i| i.trim().to_string()).collect());

    let config = planner::PlanConfig {
        intervals_filter,
        ticker_limit: limit,
        include_backfill: !incremental_only,
        include_incremental: !backfill_only,
    };

    let planner = planner::StressPlanner::with_config(&db, config);
    let plan = planner.generate_plan().await?;

    // Print summary
    println!("Plan Summary:");
    println!("  Total tickers:    {}", plan.summary.total_tickers);
    println!("  Total tasks:      {}", plan.summary.total_tasks);
    println!("  Backfill tasks:   {}", plan.summary.backfill_tasks);
    println!("  Incremental:      {}", plan.summary.incremental_tasks);
    println!("  Est. requests:    {}", plan.summary.estimated_requests);
    println!("\nTasks by interval:");
    for (interval, count) in &plan.summary.tasks_by_interval {
        println!("  {}: {}", interval, count);
    }

    // Write plan
    planner::StressPlanner::write_plan(&plan, &output)?;
    println!("\nPlan written to: {}", output.display());

    Ok(())
}

/// Execute stress download.
async fn run_stress_download(
    plan_path: PathBuf,
    output: PathBuf,
    concurrency: usize,
    budget: Option<usize>,
    max_failures: usize,
    resume: Option<PathBuf>,
    dry_run: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Stress Download Execution ===\n");

    let db = db::Database::connect().await?;
    let client = brapi::BrapiClient::new()?;

    let config = executor::ExecutorConfig {
        concurrency,
        max_requests: budget,
        max_failures,
        resume_from: resume,
        dry_run,
    };

    // Load plan
    let mut plan = if let Some(ref resume_path) = config.resume_from {
        info!("Resuming from: {}", resume_path.display());
        planner::StressPlanner::load_plan(resume_path)?
    } else {
        planner::StressPlanner::load_plan(&plan_path)?
    };

    println!("Loaded plan with {} tasks", plan.tasks.len());
    println!("Config:");
    println!("  Concurrency:   {}", config.concurrency);
    println!("  Budget:        {:?}", config.max_requests);
    println!("  Max failures:  {}", config.max_failures);
    println!("  Dry run:       {}", config.dry_run);

    // Execute
    let executor = executor::StressExecutor::new(client, db, config);
    let result = executor.execute(&mut plan, &output).await?;

    // Print results
    println!("\n=== Execution Results ===\n");
    println!("Completed tasks:  {}", result.completed_tasks);
    println!("Failed tasks:     {}", result.failed_tasks);
    println!("Skipped tasks:    {}", result.skipped_tasks);
    println!("Bars inserted:    {}", result.total_bars_inserted);
    println!("Total requests:   {}", result.total_requests);
    println!("Saved requests:   {}", result.saved_requests);
    println!("Duration:         {:.2}s", result.duration_secs);

    if !result.failures.is_empty() {
        println!("\nTop failures (first 10):");
        for f in result.failures.iter().take(10) {
            println!("  {} {} {}: {}", f.symbol, f.interval, f.range, f.error);
        }
    }

    println!("\nArtifacts written to: {}", output.display());

    Ok(())
}

/// Run incremental sync.
async fn run_sync(
    output: PathBuf,
    intervals: Option<String>,
    max_tasks: Option<usize>,
    concurrency: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Incremental Sync ===\n");

    let db = db::Database::connect().await?;
    let client = brapi::BrapiClient::new()?;

    let intervals_filter = intervals.map(|s| s.split(',').map(|i| i.trim().to_string()).collect());

    // Generate plan with incremental only
    let config = planner::PlanConfig {
        intervals_filter,
        ticker_limit: max_tasks,
        include_backfill: false,
        include_incremental: true,
    };

    let planner = planner::StressPlanner::with_config(&db, config);
    let mut plan = planner.generate_plan().await?;

    if plan.tasks.is_empty() {
        println!("All data is up to date. Nothing to sync.");
        return Ok(());
    }

    println!("Found {} incremental tasks to sync", plan.tasks.len());

    // Execute
    let exec_config = executor::ExecutorConfig {
        concurrency,
        max_requests: None,
        max_failures: 50,
        resume_from: None,
        dry_run: false,
    };

    let executor = executor::StressExecutor::new(client, db, exec_config);
    let result = executor.execute(&mut plan, &output).await?;

    println!("\n=== Sync Results ===\n");
    println!("Completed: {}", result.completed_tasks);
    println!("Failed:    {}", result.failed_tasks);
    println!("Bars:      {}", result.total_bars_inserted);
    println!("Duration:  {:.2}s", result.duration_secs);

    Ok(())
}

/// Generate all reports.
async fn run_generate_reports(output: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Report Generation ===\n");

    let db = db::Database::connect().await?;
    let generator = reports::ReportGenerator::new(&db, &output);
    generator.generate_all().await?;

    println!("\nReports written to: {}", output.display());
    println!("  - coverage_report.md");
    println!("  - freshness_report.csv");

    Ok(())
}

// ============================================================================
// Universe Commands
// ============================================================================

/// Refresh provider universe from /api/quote/list.
async fn run_universe_refresh(
    output: PathBuf,
    asset_type: String,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Provider Universe Refresh ===\n");
    println!("Source: /api/quote/list (brapi.dev)");
    println!(
        "Filter: {}",
        if asset_type == "all" {
            "none"
        } else {
            &asset_type
        }
    );

    let db = db::Database::connect().await?;
    let client = brapi::BrapiClient::new()?;

    // Get current stats before refresh
    let stats_before = db.get_universe_stats().await?;
    println!("\nBefore refresh:");
    println!("  ACTIVE:   {}", stats_before.active);
    println!("  INACTIVE: {}", stats_before.inactive);
    println!("  Total:    {}", stats_before.total);

    // Discover universe via pagination
    println!("\nFetching universe from /api/quote/list...");
    let (stocks, metrics) = client.discover_universe().await?;
    println!(
        "Discovered {} tickers across {} API calls",
        stocks.len(),
        metrics.len()
    );

    // Generate snapshot ID
    let snapshot_id = format!("snap_{}", chrono::Utc::now().format("%Y%m%d_%H%M%S"));
    println!("Snapshot ID: {}", snapshot_id);

    // Track changes
    let active_tickers: Vec<String> = stocks.iter().map(|s| s.stock.clone()).collect();

    // Batch upsert all tickers
    println!("Upserting {} tickers...", stocks.len());
    let (new_count, updated_count) = db.batch_upsert_universe(&stocks, &snapshot_id).await?;

    // Mark missing tickers as INACTIVE
    let removed_count = db
        .mark_inactive_missing(&active_tickers, &snapshot_id)
        .await?;

    // Create snapshot record
    db.create_universe_snapshot(
        &snapshot_id,
        Some(&format!(r#"{{"asset_type": "{}"}}"#, asset_type)),
        stocks.len() as i32,
        metrics.len() as i32,
        active_tickers.len() as i32,
        new_count,
        removed_count as i32,
    )
    .await?;

    // Get stats after refresh
    let stats_after = db.get_universe_stats().await?;

    println!("\n=== Refresh Complete ===\n");
    println!("Changes:");
    println!("  New tickers:     {}", new_count);
    println!("  Updated:         {}", updated_count);
    println!("  Marked INACTIVE: {}", removed_count);
    println!("\nAfter refresh:");
    println!("  ACTIVE:   {}", stats_after.active);
    println!("  INACTIVE: {}", stats_after.inactive);
    println!("  Total:    {}", stats_after.total);

    // Export artifacts
    std::fs::create_dir_all(&output)?;

    // Export universe.csv
    let universe_csv = output.join("universe.csv");
    let mut csv_content = String::from("ticker,asset_type,name,sector,status\n");
    for stock in &stocks {
        csv_content.push_str(&format!(
            "{},{},{},{},ACTIVE\n",
            stock.stock,
            stock.asset_type.as_deref().unwrap_or(""),
            stock.name.as_deref().unwrap_or("").replace(',', ";"),
            stock.sector.as_deref().unwrap_or(""),
        ));
    }
    std::fs::write(&universe_csv, csv_content)?;
    println!("\nArtifacts:");
    println!("  - {}", universe_csv.display());

    // Export snapshot manifest
    let manifest = serde_json::json!({
        "snapshot_id": snapshot_id,
        "fetched_at": chrono::Utc::now().to_rfc3339(),
        "source": "brapi.dev /api/quote/list",
        "filter": asset_type,
        "total_count": stocks.len(),
        "api_calls": metrics.len(),
        "new_count": new_count,
        "updated_count": updated_count,
        "removed_count": removed_count,
        "stats_after": {
            "active": stats_after.active,
            "inactive": stats_after.inactive,
            "total": stats_after.total
        }
    });
    let manifest_path = output.join(format!("universe_snapshot_{}.json", snapshot_id));
    std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;
    println!("  - {}", manifest_path.display());

    Ok(())
}

/// Show provider universe status.
async fn run_universe_status() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Provider Universe Status ===\n");

    let db = db::Database::connect().await?;
    let stats = db.get_universe_stats().await?;

    println!("Provider: brapi.dev");
    println!("Source:   /api/quote/list");
    println!();
    println!("┌─────────────┬─────────┐");
    println!("│ Status      │ Count   │");
    println!("├─────────────┼─────────┤");
    println!("│ ACTIVE      │ {:>7} │", stats.active);
    println!("│ INACTIVE    │ {:>7} │", stats.inactive);
    println!("│ SUSPECT     │ {:>7} │", stats.suspect);
    println!("├─────────────┼─────────┤");
    println!("│ TOTAL       │ {:>7} │", stats.total);
    println!("└─────────────┴─────────┘");

    if stats.total == 0 {
        println!("\n⚠ Universe is empty! Run 'universe-refresh' first.");
    }

    Ok(())
}

/// Run provider contract tests.
async fn run_universe_contract_test(
    sample_size: usize,
    output: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Provider Contract Test ===\n");
    println!("Testing: List -> Fetch sanity check");
    println!("Sample:  {} tickers", sample_size);

    let tester = contract_tests::ProviderContractTest::new().await?;
    let result = tester.run_sanity_check(sample_size).await?;

    // Print results
    println!("\n=== Test Results ===\n");
    println!("Active tickers in universe: {}", result.total_active);
    println!("Sample tested:              {}", result.sample_tested);
    println!("Successful fetches:         {}", result.successful);
    println!("404 errors (in list):       {}", result.not_found_listed);
    println!("Other errors:               {}", result.other_errors);
    println!();

    let success_rate = if result.sample_tested > 0 {
        result.successful as f64 / result.sample_tested as f64 * 100.0
    } else {
        0.0
    };
    println!("Success rate: {:.1}%", success_rate);

    if result.not_found_listed > 0 {
        println!(
            "\n⚠ {} tickers returned 404 despite being in the list!",
            result.not_found_listed
        );
        println!("  These have been marked as SUSPECT and logged to divergences.");
    }

    // Export artifacts
    std::fs::create_dir_all(&output)?;
    result.write_manifest(&output)?;

    println!("\nArtifacts:");
    println!("  - {}/provider_contract_manifest.json", output.display());

    if result.not_found_listed == 0 && result.other_errors == 0 {
        println!("\n✓ Contract test PASSED");
    } else {
        println!("\n✗ Contract test has issues (see divergences)");
    }

    Ok(())
}

/// Export universe snapshot.
async fn run_universe_snapshot(
    _id: String,
    output: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Universe Snapshot Export ===\n");

    let db = db::Database::connect().await?;
    let active_tickers = db.get_active_tickers().await?;

    std::fs::create_dir_all(&output)?;

    let snapshot_path = output.join("universe_active.csv");
    let mut content = String::from("ticker\n");
    for ticker in &active_tickers {
        content.push_str(ticker);
        content.push('\n');
    }
    std::fs::write(&snapshot_path, content)?;

    println!("Exported {} ACTIVE tickers to:", active_tickers.len());
    println!("  {}", snapshot_path.display());

    Ok(())
}

// ============================================================================
// Aggregation Intelligence Functions
// ============================================================================

/// Run data inventory scan.
async fn run_aggregate_inventory(
    output: PathBuf,
    stale_days: i64,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Data Inventory Scan ===\n");

    let db = db::Database::connect().await?;
    let scanner = inventory::DataInventoryScanner::with_stale_threshold(&db, stale_days);

    println!("Scanning database for data inventory...");
    let (inventory_items, summary) = scanner.scan().await?;

    // Create output directory
    std::fs::create_dir_all(&output)?;

    // Write CSV
    let csv_path = output.join("data_inventory.csv");
    inventory::DataInventoryScanner::write_csv(&inventory_items, &csv_path)?;

    // Write JSON
    let json_path = output.join("data_inventory.json");
    inventory::DataInventoryScanner::write_json(&inventory_items, &summary, &json_path)?;

    // Print summary
    println!("\n=== Inventory Summary ===\n");
    println!("Total tickers:  {}", summary.total_tickers);
    println!("Total pairs:    {}", summary.total_pairs);
    println!("Total bars:     {}", summary.total_bars);
    println!("Scan duration:  {:.2}s", summary.scan_duration_secs);

    println!("\nBy Status:");
    for (status, count) in &summary.by_status {
        println!("  {}: {}", status, count);
    }

    println!("\nBy Interval:");
    for (interval, stats) in &summary.by_interval {
        println!(
            "  {}: {} total, {:.1}% avg coverage",
            interval, stats.total, stats.coverage_avg
        );
    }

    println!("\nArtifacts:");
    println!("  - {}", csv_path.display());
    println!("  - {}", json_path.display());

    Ok(())
}

/// Generate aggregation plan.
async fn run_aggregate_plan(
    output: PathBuf,
    sync_only: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Aggregation Plan Generation ===\n");

    let db = db::Database::connect().await?;

    // First, scan inventory
    println!("Scanning current data inventory...");
    let scanner = inventory::DataInventoryScanner::new(&db);
    let (inventory_items, _) = scanner.scan().await?;

    // Generate plan
    println!("Generating aggregation plan...");
    let planner = aggregation_planner::AggregationPlanner::new(&db);
    let plan = if sync_only {
        println!("(Sync mode: only stale items)");
        planner.generate_sync_plan(&inventory_items)
    } else {
        planner.generate_plan(&inventory_items)
    };

    // Write outputs
    std::fs::create_dir_all(&output)?;

    let plan_path = output.join("aggregation_plan.json");
    plan.write_json(&plan_path)?;

    let summary_path = output.join("aggregation_plan_summary.md");
    plan.write_summary_md(&summary_path)?;

    // Print summary
    println!("\n=== Plan Summary ===\n");
    println!("Total tasks:        {}", plan.total_tasks);
    println!("Estimated requests: {}", plan.total_estimated_requests);
    println!("Skipped pairs:      {}", plan.skipped_count);

    println!("\nTasks by Reason:");
    for (reason, count) in &plan.tasks_by_reason {
        println!("  {}: {}", reason, count);
    }

    println!("\nTasks by Interval:");
    for (interval, count) in &plan.tasks_by_interval {
        println!("  {}: {}", interval, count);
    }

    println!("\nArtifacts:");
    println!("  - {}", plan_path.display());
    println!("  - {}", summary_path.display());

    if plan.total_tasks > 0 {
        println!(
            "\nRun with: market-data aggregate-run --plan {}",
            plan_path.display()
        );
    }

    Ok(())
}

/// Execute aggregation plan.
async fn run_aggregate_execute(
    plan_path: PathBuf,
    output: PathBuf,
    _concurrency: usize,
    budget: Option<usize>,
    max_failures: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Aggregation Execution ===\n");

    // Load plan
    println!("Loading plan from: {}", plan_path.display());
    let plan = aggregation_planner::AggregationPlan::load(&plan_path)?;

    println!("Plan contains {} tasks", plan.total_tasks);
    if let Some(b) = budget {
        println!("Budget limit: {} requests", b);
    }

    // Connect
    let db = db::Database::connect().await?;
    let client = brapi::BrapiClient::new()?;

    let config = aggregation_executor::AggregationExecutorConfig {
        concurrency: 1, // Sequential for DB safety
        budget,
        max_failures,
    };

    std::fs::create_dir_all(&output)?;
    let checkpoint_path = output.join("checkpoint.json");

    println!("\nExecuting...\n");
    let manifest = aggregation_executor::execute_plan(
        &db,
        &client,
        &plan,
        &plan_path.to_string_lossy(),
        Some(&checkpoint_path),
        config,
    )
    .await?;

    // Write outputs
    let manifest_path = output.join("aggregation_run_manifest.json");
    manifest.write_json(&manifest_path)?;

    let failures_path = output.join("aggregation_failures.csv");
    manifest.write_failures_csv(&failures_path)?;

    let success_path = output.join("aggregation_success.csv");
    manifest.write_success_csv(&success_path)?;

    // Print results
    println!("\n=== Execution Results ===\n");
    println!(
        "Completed:  {}/{}",
        manifest.completed_tasks, manifest.total_tasks
    );
    println!("Failed:     {}", manifest.failed_tasks);
    println!("Skipped:    {}", manifest.skipped_tasks);
    println!("Bars added: {}", manifest.total_bars_inserted);
    println!("Requests:   {}", manifest.requests_made);
    println!("Saved:      {} (idempotent skips)", manifest.saved_requests);
    println!("Duration:   {:.1}s", manifest.total_duration_secs);

    if manifest.rate_limit_hits > 0 {
        println!("Rate limit hits: {}", manifest.rate_limit_hits);
    }

    println!("\nArtifacts:");
    println!("  - {}", manifest_path.display());
    if manifest.failed_tasks > 0 {
        println!("  - {}", failures_path.display());
    }

    Ok(())
}

/// Quick sync: plan + run for stale data.
async fn run_aggregate_sync(
    output: PathBuf,
    _concurrency: usize,
    budget: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Aggregation Sync ===\n");
    println!("Scanning + planning + executing for stale data...\n");

    let db = db::Database::connect().await?;
    let client = brapi::BrapiClient::new()?;

    // Scan
    let scanner = inventory::DataInventoryScanner::new(&db);
    let (inventory_items, _) = scanner.scan().await?;

    // Plan (sync mode)
    let planner = aggregation_planner::AggregationPlanner::new(&db);
    let plan = planner.generate_sync_plan(&inventory_items);

    if plan.total_tasks == 0 {
        println!("✓ All data is fresh! Nothing to sync.");
        return Ok(());
    }

    println!("Found {} tasks to sync", plan.total_tasks);

    let config = aggregation_executor::AggregationExecutorConfig {
        concurrency: 1,
        budget,
        max_failures: 50,
    };

    std::fs::create_dir_all(&output)?;

    let manifest =
        aggregation_executor::execute_plan(&db, &client, &plan, "sync", None, config).await?;

    // Write manifest
    let manifest_path = output.join("sync_manifest.json");
    manifest.write_json(&manifest_path)?;

    println!("\n=== Sync Results ===");
    println!(
        "Completed: {}/{}",
        manifest.completed_tasks, manifest.total_tasks
    );
    println!("Bars added: {}", manifest.total_bars_inserted);

    Ok(())
}

/// Show aggregation status.
async fn run_aggregate_status() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Aggregation Status ===\n");

    let db = db::Database::connect().await?;
    let scanner = inventory::DataInventoryScanner::new(&db);
    let (_inventory_items, summary) = scanner.scan().await?;

    println!("┌────────────────────────────────────┐");
    println!("│       Data Coverage Summary        │");
    println!("├────────────────────────────────────┤");
    println!("│ Total tickers:  {:>18} │", summary.total_tickers);
    println!("│ Total pairs:    {:>18} │", summary.total_pairs);
    println!("│ Total bars:     {:>18} │", summary.total_bars);
    println!("└────────────────────────────────────┘");

    println!("\n┌──────────┬────────┬─────────┬──────────┬─────────┬──────────┐");
    println!("│ Interval │ Total  │ Empty   │ Partial  │ Complete│ Stale    │");
    println!("├──────────┼────────┼─────────┼──────────┼─────────┼──────────┤");

    for interval in &["1d", "60m", "15m", "5m", "1m"] {
        if let Some(stats) = summary.by_interval.get(*interval) {
            println!(
                "│ {:>8} │ {:>6} │ {:>7} │ {:>8} │ {:>7} │ {:>8} │",
                interval, stats.total, stats.empty, stats.partial, stats.complete, stats.stale
            );
        }
    }
    println!("└──────────┴────────┴─────────┴──────────┴─────────┴──────────┘");

    // Calculate overall coverage
    let complete_count = summary.by_status.get("COMPLETE").copied().unwrap_or(0);
    let coverage_pct = if summary.total_pairs > 0 {
        complete_count as f64 / summary.total_pairs as f64 * 100.0
    } else {
        0.0
    };

    println!(
        "\nOverall coverage: {:.1}% ({} complete pairs)",
        coverage_pct, complete_count
    );

    let stale_count = summary.by_status.get("STALE").copied().unwrap_or(0);
    if stale_count > 0 {
        println!(
            "⚠ {} pairs have stale data (run 'aggregate-sync' to update)",
            stale_count
        );
    }

    let empty_count = summary.by_status.get("EMPTY").copied().unwrap_or(0);
    if empty_count > 0 {
        println!(
            "⚠ {} pairs have no data (run 'aggregate-plan' + 'aggregate-run')",
            empty_count
        );
    }

    Ok(())
}

// ============================================================================
// Audit Integrity
// ============================================================================

/// Run OHLCV integrity audit (read-only, console output only).
async fn run_audit_integrity(
    min_integrity: f64,
    sample_outliers: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = std::time::Instant::now();

    let db = db::Database::connect().await?;

    let config = audit_integrity::AuditConfig {
        min_integrity,
        sample_outliers,
        max_hierarchy_violations_pct: 1.0,
    };

    let auditor = audit_integrity::IntegrityAuditor::new(&db, config);
    let result = auditor.run().await?;

    let duration = start.elapsed().as_secs_f64();
    auditor.print_results(&result, duration);

    // Exit with non-zero if failed
    if !result.passed {
        std::process::exit(1);
    }

    Ok(())
}

// ============================================================================
// Fundamentals Sync Functions
// ============================================================================

/// Sync fundamental data from Brapi API.
async fn run_sync_fundamentals(
    all: bool,
    symbols: Option<Vec<String>>,
    batch_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use chrono::Utc;

    println!("\n=== Fundamentals Sync ===\n");

    let db = db::Database::connect().await?;
    let client = brapi::BrapiClient::new()?;

    // Get symbols to sync
    let tickers: Vec<String> = if let Some(syms) = symbols {
        syms
    } else if all {
        println!("Fetching all active symbols...");
        db.get_active_tickers().await?
    } else {
        println!("Error: Specify --all or --symbols");
        return Err("No symbols specified".into());
    };

    if tickers.is_empty() {
        println!("No symbols to sync. Run 'universe-refresh' first.");
        return Ok(());
    }

    println!("Syncing fundamentals for {} symbols", tickers.len());
    println!("Batch size: {} (max 20)", batch_size.min(20));

    let batch_size = batch_size.min(20);
    let snapshot_date = Utc::now().date_naive();

    let mut total_synced = 0;
    let mut total_dividends = 0;
    let mut errors = 0;

    for (batch_idx, batch) in tickers.chunks(batch_size).enumerate() {
        let ticker_refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();

        print!(
            "Batch {}/{}: {} symbols... ",
            batch_idx + 1,
            (tickers.len() + batch_size - 1) / batch_size,
            batch.len()
        );

        match client.fetch_fundamentals(&ticker_refs).await {
            Ok((results, _metrics)) => {
                for quote in &results {
                    // Ensure instrument exists
                    if let Err(e) = db.ensure_instrument_exists(&quote.symbol).await {
                        warn!("Failed to ensure instrument {}: {}", quote.symbol, e);
                        continue;
                    }

                    // Build and upsert fundamental snapshot
                    let snapshot = brapi::FundamentalSnapshot::from_quote(quote, snapshot_date);
                    if let Err(e) = db.upsert_fundamental_snapshot(&snapshot).await {
                        warn!("Failed to upsert snapshot for {}: {}", quote.symbol, e);
                        errors += 1;
                        continue;
                    }
                    total_synced += 1;

                    // Upsert dividends if available
                    if let Some(ref div_data) = quote.dividends_data {
                        for cash_div in &div_data.cash_dividends {
                            if let Some(entry) =
                                brapi::DividendEntry::from_cash_dividend(&quote.symbol, cash_div)
                            {
                                if let Err(e) = db.upsert_dividend(&entry).await {
                                    warn!("Failed to upsert dividend for {}: {}", quote.symbol, e);
                                } else {
                                    total_dividends += 1;
                                }
                            }
                        }
                    }

                    // Upsert company profile if available
                    if let Some(ref profile) = quote.summary_profile {
                        if let Err(e) = db.upsert_company_profile(&quote.symbol, profile).await {
                            warn!("Failed to upsert profile for {}: {}", quote.symbol, e);
                        }
                    }
                }
                println!("OK ({} synced)", results.len());
            }
            Err(e) => {
                println!("ERROR: {}", e);
                errors += batch.len() as i32;
            }
        }

        // Rate limiting delay between batches
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }

    // Update watermark
    db.update_sync_watermark("fundamentals", total_synced, errors)
        .await?;

    println!("\n=== Sync Complete ===\n");
    println!("Fundamentals synced: {}", total_synced);
    println!("Dividends added:     {}", total_dividends);
    println!("Errors:              {}", errors);

    Ok(())
}

/// Show fundamentals sync status.
async fn run_fundamentals_status() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Fundamentals Status ===\n");

    let db = db::Database::connect().await?;

    // Check sync watermark
    let row = db.client()
        .query_opt(
            "SELECT last_sync, symbols_synced, errors, next_sync FROM sync_watermarks WHERE entity = 'fundamentals'",
            &[],
        )
        .await?;

    match row {
        Some(r) => {
            let last_sync: Option<chrono::DateTime<chrono::Utc>> = r.get(0);
            let symbols_synced: Option<i32> = r.get(1);
            let errors: Option<i32> = r.get(2);
            let next_sync: Option<chrono::DateTime<chrono::Utc>> = r.get(3);

            println!("Last sync:       {:?}", last_sync.map(|d| d.to_rfc3339()));
            println!("Symbols synced:  {:?}", symbols_synced);
            println!("Errors:          {:?}", errors);
            println!("Next sync due:   {:?}", next_sync.map(|d| d.to_rfc3339()));
        }
        None => {
            println!("No fundamentals sync has been performed yet.");
            println!("Run: market-data sync-fundamentals --all");
        }
    }

    // Count snapshots
    let snapshot_count: i64 = db
        .client()
        .query_one("SELECT COUNT(*) FROM fundamentals_snapshot", &[])
        .await?
        .get(0);

    let symbols_with_data: i64 = db
        .client()
        .query_one(
            "SELECT COUNT(DISTINCT symbol) FROM fundamentals_snapshot",
            &[],
        )
        .await?
        .get(0);

    let dividend_count: i64 = db
        .client()
        .query_one("SELECT COUNT(*) FROM dividends_history", &[])
        .await?
        .get(0);

    println!("\nData Summary:");
    println!("  Snapshot records:  {}", snapshot_count);
    println!("  Symbols with data: {}", symbols_with_data);
    println!("  Dividend records:  {}", dividend_count);

    Ok(())
}

// ============================================================================
// Interest Rates Sync Functions
// ============================================================================

/// Sync interest rates from BCB (BR) and/or FRED (US).
async fn run_sync_interest_rates(
    br: bool,
    us: bool,
    all: bool,
    start: Option<String>,
    end: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    use chrono::Duration;

    println!("\n=== Interest Rates Sync ===\n");

    let sync_br = br || all;
    let sync_us = us || all;

    if !sync_br && !sync_us {
        println!("Error: Specify --br, --us, or --all");
        return Err("No region specified".into());
    }

    let db = db::Database::connect().await?;

    // Parse dates
    let end_date = match end {
        Some(s) => chrono::NaiveDate::parse_from_str(&s, "%Y-%m-%d")?,
        None => chrono::Local::now().date_naive(),
    };
    let start_date = match start {
        Some(s) => chrono::NaiveDate::parse_from_str(&s, "%Y-%m-%d")?,
        None => end_date - Duration::days(365 * 5), // 5 years default
    };

    info!("Date range: {} to {}", start_date, end_date);

    let mut total_inserted = 0;
    let mut total_updated = 0;

    // Sync BR (BCB SELIC)
    if sync_br {
        println!("\n--- Brazil (BCB SELIC) ---");
        let bcb = interest_rates::BcbClient::new();

        match bcb.fetch_selic(start_date, end_date).await {
            Ok(entries) => {
                println!("Fetched {} SELIC rates", entries.len());

                let (inserted, updated) = db.upsert_interest_rates(&entries).await?;
                total_inserted += inserted;
                total_updated += updated;

                info!("BR: inserted={}, updated={}", inserted, updated);
            }
            Err(e) => {
                warn!("BCB fetch failed: {}", e);
                println!("ERROR: {}", e);
            }
        }
    }

    // Sync US (FRED T-Bill)
    if sync_us {
        println!("\n--- United States (FRED TB3MS) ---");

        match interest_rates::FredClient::new() {
            Ok(fred) => match fred.fetch_tbill_3m(start_date, end_date).await {
                Ok(entries) => {
                    println!("Fetched {} T-Bill rates", entries.len());

                    let (inserted, updated) = db.upsert_interest_rates(&entries).await?;
                    total_inserted += inserted;
                    total_updated += updated;

                    info!("US: inserted={}, updated={}", inserted, updated);
                }
                Err(e) => {
                    warn!("FRED fetch failed: {}", e);
                    println!("ERROR: {}", e);
                }
            },
            Err(e) => {
                println!("FRED client not available: {}", e);
                println!("Set FRED_API_KEY environment variable");
                println!("Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html");
            }
        }
    }

    println!("\n=== Sync Complete ===\n");
    println!("Total inserted: {}", total_inserted);
    println!("Total updated:  {}", total_updated);

    // Show current stats
    let stats = db.get_interest_rate_stats().await?;
    println!("\nDatabase Status:");
    println!(
        "  BR (SELIC):   {} rates ({:?} to {:?})",
        stats.br_count, stats.br_min_date, stats.br_max_date
    );
    println!(
        "  US (T-Bill):  {} rates ({:?} to {:?})",
        stats.us_count, stats.us_min_date, stats.us_max_date
    );

    Ok(())
}

/// Show interest rates status.
async fn run_interest_rates_status() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Interest Rates Status ===\n");

    let db = db::Database::connect().await?;
    let stats = db.get_interest_rate_stats().await?;

    println!("┌─────────────┬─────────┬─────────────┬─────────────┐");
    println!("│ Region      │ Count   │ Min Date    │ Max Date    │");
    println!("├─────────────┼─────────┼─────────────┼─────────────┤");
    println!(
        "│ BR (SELIC)  │ {:>7} │ {:>11} │ {:>11} │",
        stats.br_count,
        stats
            .br_min_date
            .map(|d| d.to_string())
            .unwrap_or_else(|| "-".into()),
        stats
            .br_max_date
            .map(|d| d.to_string())
            .unwrap_or_else(|| "-".into())
    );
    println!(
        "│ US (T-Bill) │ {:>7} │ {:>11} │ {:>11} │",
        stats.us_count,
        stats
            .us_min_date
            .map(|d| d.to_string())
            .unwrap_or_else(|| "-".into()),
        stats
            .us_max_date
            .map(|d| d.to_string())
            .unwrap_or_else(|| "-".into())
    );
    println!("└─────────────┴─────────┴─────────────┴─────────────┘");

    if stats.br_count == 0 && stats.us_count == 0 {
        println!("\n⚠ No interest rate data! Run:");
        println!("  market-data sync-interest-rates --all");
    }

    // Show latest rates
    let today = chrono::Local::now().date_naive();

    if stats.br_count > 0 {
        if let Ok(Some(rate)) = db.get_interest_rate_at(today, "BR", "SELIC").await {
            println!("\nLatest BR SELIC: {:.2}%", rate * 100.0);
        }
    }

    if stats.us_count > 0 {
        if let Ok(Some(rate)) = db.get_interest_rate_at(today, "US", "TBILL_3M").await {
            println!("Latest US T-Bill 3M: {:.2}%", rate * 100.0);
        }
    }

    Ok(())
}

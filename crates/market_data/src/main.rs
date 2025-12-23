//! Market Data Ingestion CLI
//! 
//! Pipeline for ingesting B3 OHLCV data from Brapi into Neon Postgres.

mod brapi;
mod db;
mod ingest;

use clap::{Parser, Subcommand};
use tracing::{info, error, Level};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser)]
#[command(name = "market-data")]
#[command(about = "B3 Market Data Ingestion Pipeline")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize database schema
    InitDb,
    /// Refresh universe (top N by volume)
    RefreshUniverse {
        #[arg(short, long, default_value = "150")]
        target: usize,
    },
    /// Backfill historical data
    Backfill {
        #[arg(short, long, default_value = "top_volume")]
        universe: String,
        #[arg(short, long, default_value = "max")]
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
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let cli = Cli::parse();
    let db = db::Database::connect().await?;
    let brapi = brapi::BrapiClient::new()?;

    match cli.command {
        Commands::InitDb => {
            info!("Database schema already initialized via Neon MCP");
            db.verify_schema().await?;
            info!("Schema verification complete");
        }
        Commands::RefreshUniverse { target } => {
            info!("Refreshing universe with top {} by volume", target);
            ingest::refresh_universe(&db, &brapi, target).await?;
        }
        Commands::Backfill { universe, range } => {
            info!("Starting backfill for universe '{}' with range '{}'", universe, range);
            ingest::backfill(&db, &brapi, &universe, &range).await?;
        }
        Commands::Update { universe } => {
            info!("Running incremental update for universe '{}'", universe);
            ingest::update_incremental(&db, &brapi, &universe).await?;
        }
        Commands::VerifyIntegrity => {
            info!("Verifying data integrity");
            ingest::verify_integrity(&db).await?;
        }
        Commands::Status => {
            ingest::show_status(&db).await?;
        }
    }

    Ok(())
}


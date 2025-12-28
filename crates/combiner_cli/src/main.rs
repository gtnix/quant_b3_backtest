//! Combiner CLI - Command-line interface for the Generative Combiner (SCG)
//!
//! Commands:
//! - `combiner run --config <path>` - Run evolution
//! - `combiner status <experiment_id>` - Check status
//! - `combiner export-top <experiment_id> --n 10` - Export top strategies

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

mod commands;

#[derive(Parser)]
#[command(name = "combiner")]
#[command(author = "SCG Team")]
#[command(version = "0.1.0")]
#[command(about = "Generative Combiner - Evolutionary strategy discovery")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run evolution experiment
    Run {
        /// Path to SCG configuration file
        #[arg(short, long)]
        config: String,

        /// Output directory
        #[arg(short, long, default_value = "output/scg")]
        output: String,

        /// Random seed for reproducibility
        #[arg(short, long)]
        seed: Option<u64>,

        /// Dry run (validate only)
        #[arg(long)]
        dry_run: bool,
    },

    /// Check experiment status
    Status {
        /// Experiment ID
        experiment_id: String,
    },

    /// Export top strategies
    ExportTop {
        /// Experiment ID
        experiment_id: String,

        /// Number of strategies to export
        #[arg(short, long, default_value = "10")]
        n: usize,

        /// Output directory for TOMLs
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Validate top strategies with Walk-Forward Analysis
    Validate {
        /// Experiment ID to validate
        experiment_id: String,

        /// Number of top strategies to validate
        #[arg(short = 'k', long, default_value = "10")]
        top_k: usize,

        /// Enable full validation (CPCV + PBO/DSR)
        #[arg(long)]
        full: bool,
    },
}

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env().add_directive("combiner=info".parse()?))
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            config,
            output,
            seed,
            dry_run,
        } => commands::run::execute(&config, &output, seed, dry_run),

        Commands::Status { experiment_id } => commands::status::execute(&experiment_id),

        Commands::ExportTop {
            experiment_id,
            n,
            output,
        } => commands::export::execute(&experiment_id, n, output.as_deref()),

        Commands::Validate {
            experiment_id,
            top_k,
            full,
        } => commands::validate::execute(&experiment_id, top_k, full),
    }
}


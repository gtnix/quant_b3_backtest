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

        /// Ultra-performance mode with SIMD optimization, batch evaluation, and
        /// parallel Stage B validation. Recommended for large experiments.
        #[arg(long)]
        ultra: bool,

        /// Number of top genomes to validate in Stage B (ultra mode only)
        #[arg(long, default_value = "10")]
        top_k: usize,

        // ========== Execution Model Overrides ==========

        /// Execution delay in bars (0=same bar, 1=next bar open)
        /// Overrides config file setting.
        #[arg(long)]
        execution_delay: Option<u8>,

        /// Slippage in basis points (e.g., 10 = 0.1%)
        /// Overrides config file setting.
        #[arg(long)]
        slippage_bps: Option<f64>,

        /// Fee tier preset: b3-retail, b3-prime, us-retail, us-prime
        /// Overrides config file setting.
        #[arg(long)]
        fee_tier: Option<String>,

        /// Enable execution stress testing for top candidates.
        #[arg(long)]
        stress_enabled: bool,

        /// Minimum stress scenarios to pass (default: 4 out of 5).
        #[arg(long)]
        min_stress_pass: Option<usize>,

        /// Bypass all costs for debugging (NOT for production).
        #[arg(long)]
        bypass_costs: bool,
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
        
        /// Include execution config parameters in exported files
        #[arg(long)]
        include_execution_config: bool,
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
        
        /// Enable stress testing during validation
        #[arg(long)]
        stress_enabled: bool,
    },
}

/// Execution model override options.
#[derive(Debug, Clone, Default)]
pub struct ExecutionOverrides {
    pub delay_bars: Option<u8>,
    pub slippage_bps: Option<f64>,
    pub fee_tier: Option<String>,
    pub stress_enabled: bool,
    pub min_stress_pass: Option<usize>,
    pub bypass_costs: bool,
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
            ultra,
            top_k,
            execution_delay,
            slippage_bps,
            fee_tier,
            stress_enabled,
            min_stress_pass,
            bypass_costs,
        } => {
            let exec_overrides = ExecutionOverrides {
                delay_bars: execution_delay,
                slippage_bps,
                fee_tier,
                stress_enabled,
                min_stress_pass,
                bypass_costs,
            };
            commands::run::execute(
                &config,
                &output,
                seed,
                dry_run,
                ultra,
                top_k,
                exec_overrides,
            )
        }

        Commands::Status { experiment_id } => commands::status::execute(&experiment_id),

        Commands::ExportTop {
            experiment_id,
            n,
            output,
            include_execution_config,
        } => commands::export::execute(&experiment_id, n, output.as_deref(), include_execution_config),

        Commands::Validate {
            experiment_id,
            top_k,
            full,
            stress_enabled,
        } => commands::validate::execute(&experiment_id, top_k, full, stress_enabled),
    }
}

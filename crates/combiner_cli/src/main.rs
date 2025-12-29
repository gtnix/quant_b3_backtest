//! Combiner CLI - Command-line interface for the Generative Combiner (SCG)
//!
//! Commands:
//! - `combiner run --config <path>` - Run evolution
//! - `combiner status <experiment_id>` - Check status
//! - `combiner export-top <experiment_id> --n 10` - Export top strategies
//! - `combiner factory ...` - Strategy Factory orchestration

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

    /// Strategy Factory - Campaign orchestration and promotion
    Factory {
        #[command(subcommand)]
        action: FactoryAction,
    },
}

/// Factory subcommands for campaign management.
#[derive(Subcommand)]
enum FactoryAction {
    /// Initialize a new campaign (creates config template)
    Init {
        /// Campaign name
        #[arg(short, long)]
        name: String,
    },

    /// Run a campaign (executes multi-seed SCG runs)
    Run {
        /// Path to campaign config file
        #[arg(short, long)]
        campaign: String,
    },

    /// Resume an interrupted campaign
    Resume {
        /// Path to campaign config file
        #[arg(short, long)]
        campaign: String,
    },

    /// Audit data integrity for a campaign (standalone)
    AuditData {
        /// Path to campaign config file
        #[arg(short, long)]
        campaign: String,

        /// Audit mode: fast or strict
        #[arg(short, long, default_value = "fast")]
        mode: String,
    },

    /// Export top N candidates with deterministic ranking
    ExportTop {
        /// Run ID to export from
        #[arg(short, long)]
        run: String,

        /// Number of top candidates to export
        #[arg(short, long, default_value = "1000")]
        top: usize,

        /// Export formats (comma-separated: json,csv)
        #[arg(short, long, default_value = "json,csv")]
        format: String,

        /// Candidate class filter: research, validated, or all
        #[arg(short, long, default_value = "research")]
        class: String,
    },

    /// List campaigns
    List {
        /// Filter by tag
        #[arg(short, long)]
        tag: Option<String>,
    },

    /// Show campaign or run details
    Show {
        /// Campaign ID or Run ID
        id: String,
    },

    /// Build site-ready JSON bundle for web consumption
    BuildSite {
        /// Campaign ID (optional, builds all if not provided)
        #[arg(short, long)]
        campaign: Option<String>,

        /// Run ID (optional, builds single run detail)
        #[arg(short, long)]
        run: Option<String>,
    },

    /// Compare candidates across multiple runs
    Compare {
        /// Comma-separated run IDs
        #[arg(short, long, value_delimiter = ',')]
        runs: Vec<String>,

        /// Number of top candidates per run
        #[arg(short, long, default_value = "5")]
        top: usize,
    },

    /// Promote candidates to next stage
    Promote {
        /// Run ID to promote from
        #[arg(short, long)]
        run: Option<String>,

        /// Campaign ID to promote from (all completed runs)
        #[arg(short, long)]
        campaign: Option<String>,

        /// Number of top candidates to promote
        #[arg(short, long, default_value = "3")]
        top: usize,

        /// Promotion stage: research, candidate, paper
        #[arg(short, long, default_value = "candidate")]
        stage: String,

        /// Force re-promotion (ignore duplicates)
        #[arg(short, long)]
        force: bool,
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
    // Initialize tracing with JSON format for structured logging
    let json_layer = std::env::var("FACTORY_JSON_LOGS").is_ok();

    if json_layer {
        tracing_subscriber::registry()
            .with(fmt::layer().json())
            .with(EnvFilter::from_default_env().add_directive("combiner=info".parse()?))
            .init();
    } else {
        tracing_subscriber::registry()
            .with(fmt::layer())
            .with(EnvFilter::from_default_env().add_directive("combiner=info".parse()?))
            .init();
    }

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

        Commands::Factory { action } => match action {
            FactoryAction::Init { name } => {
                commands::factory::execute_init(&name)
            }

            FactoryAction::Run { campaign } => {
                commands::factory::execute_run(&campaign)
            }

            FactoryAction::Resume { campaign } => {
                commands::factory::execute_resume(&campaign)
            }

            FactoryAction::List { tag } => {
                commands::factory::execute_list(tag.as_deref())
            }

            FactoryAction::Show { id } => {
                commands::factory::execute_show(&id)
            }

            FactoryAction::Compare { runs, top } => {
                commands::factory::execute_compare(&runs, top)
            }

            FactoryAction::Promote { run, campaign, top, stage, force } => {
                commands::factory::execute_promote(
                    run.as_deref(),
                    campaign.as_deref(),
                    top,
                    &stage,
                    force,
                )
            }

            FactoryAction::AuditData { campaign, mode } => {
                commands::factory::execute_audit(&campaign, &mode)
            }

            FactoryAction::ExportTop { run, top, format, class } => {
                let class_filter = commands::factory::CandidateClassFilter::from_str(&class);
                commands::factory::execute_export_top(&run, top, &format, class_filter)
            }

            FactoryAction::BuildSite { campaign, run } => {
                commands::factory::execute_build_site(campaign.as_deref(), run.as_deref())
            }
        },
    }
}


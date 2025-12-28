//! Strategy Factory - Campaign orchestration, registry, and promotion.
//!
//! Provides commands for:
//! - `factory init` - Create campaign config template
//! - `factory run` - Execute multi-seed campaigns
//! - `factory resume` - Resume interrupted campaigns
//! - `factory list` - List campaigns and runs
//! - `factory show` - Show run details
//! - `factory compare` - Compare candidates across runs
//! - `factory promote` - Promote candidates to paper trading
//! - `factory audit-data` - Standalone data integrity audit

pub mod config;
pub mod registry;

mod init;
mod list;
mod run_campaign;
mod compare;
mod promote;
mod bundle;
mod audit;

pub use init::execute_init;
pub use list::{execute_list, execute_show};
pub use run_campaign::{execute_run, execute_resume};
pub use compare::execute_compare;
pub use promote::execute_promote;
pub use audit::execute_audit;


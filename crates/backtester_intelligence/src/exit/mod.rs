//! Exit and Risk Management Module.
//!
//! This module provides:
//! - Exit policies (stop-loss, take-profit, time-based, trailing stop)
//! - Risk guards (exposure limits, turnover limits, drawdown guard)
//! - Exit engine orchestration
//! - Audit logging for exit decisions

pub mod types;
pub mod policy;
pub mod stop_loss;
pub mod take_profit;
pub mod time_exit;
pub mod trailing_stop;
pub mod risk_guard;
pub mod engine;
pub mod audit;

// Re-export main types
pub use types::{
    DrawdownAction, ExitContext, ExitDiagnostics, ExitReason, ExitResult,
    ExitTarget, Position, RiskViolation,
};

pub use policy::{ExitPolicy, ExitPolicyConfig};
pub use stop_loss::{StopLossConfig, StopLossPolicy};
pub use take_profit::{TakeProfitConfig, TakeProfitPolicy};
pub use time_exit::{TimeExitConfig, TimeExitPolicy};
pub use trailing_stop::{TrailingStopConfig, TrailingStopPolicy};
pub use risk_guard::{RiskConfig, RiskGuard};
pub use engine::{ExitEngine, ExitEngineConfig};
pub use audit::{ExitAuditLog, ExitedPosition};

/// Prelude for convenient imports.
pub mod prelude {
    pub use super::{
        DrawdownAction, ExitContext, ExitDiagnostics, ExitEngine, ExitEngineConfig,
        ExitPolicy, ExitPolicyConfig, ExitReason, ExitResult, ExitTarget, Position,
        RiskConfig, RiskGuard, RiskViolation, StopLossConfig, StopLossPolicy,
        TakeProfitConfig, TakeProfitPolicy, TimeExitConfig, TimeExitPolicy,
        TrailingStopConfig, TrailingStopPolicy,
    };
}












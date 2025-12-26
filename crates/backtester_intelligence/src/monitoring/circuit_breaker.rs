//! Circuit Breaker for Monitoring.
//!
//! Implements a state machine for trading halt conditions:
//! - Closed: Normal operation
//! - HalfOpen: Testing recovery
//! - Open: Trading halted

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

use super::config::CircuitBreakerConfig;
use super::types::{CheckCategory, CheckResult, CircuitAction, CircuitBreakerState, Severity};

/// Circuit breaker state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CircuitState {
    /// Normal operation, all systems go
    Closed,
    /// Testing recovery after trip
    HalfOpen,
    /// Trading halted, NO-TRADE mode
    Open,
}

impl Default for CircuitState {
    fn default() -> Self {
        CircuitState::Closed
    }
}

impl std::fmt::Display for CircuitState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CircuitState::Closed => write!(f, "Closed"),
            CircuitState::HalfOpen => write!(f, "HalfOpen"),
            CircuitState::Open => write!(f, "Open"),
        }
    }
}

/// Circuit breaker implementation.
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    /// Current state
    state: CircuitState,
    /// Count of critical issues in current evaluation
    crit_count: usize,
    /// Threshold for tripping
    halt_threshold: usize,
    /// Cooldown period in minutes
    cooldown_minutes: u32,
    /// Last trip timestamp
    last_trip: Option<DateTime<Utc>>,
    /// Configuration
    config: CircuitBreakerConfig,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: CircuitState::Closed,
            crit_count: 0,
            halt_threshold: config.halt_on_crit_count,
            cooldown_minutes: config.cooldown_minutes,
            last_trip: None,
            config,
        }
    }

    /// Get current state.
    pub fn state(&self) -> CircuitState {
        self.state
    }

    /// Get critical count.
    pub fn crit_count(&self) -> usize {
        self.crit_count
    }

    /// Get last trip time.
    pub fn last_trip(&self) -> Option<DateTime<Utc>> {
        self.last_trip
    }

    /// Check if in cooldown period.
    pub fn in_cooldown(&self) -> bool {
        match self.last_trip {
            Some(trip_time) => {
                let cooldown = Duration::minutes(self.cooldown_minutes as i64);
                Utc::now() < trip_time + cooldown
            }
            None => false,
        }
    }

    /// Get remaining cooldown minutes.
    pub fn cooldown_remaining(&self) -> Option<u32> {
        self.last_trip.map(|trip_time| {
            let cooldown = Duration::minutes(self.cooldown_minutes as i64);
            let end_time = trip_time + cooldown;
            let remaining = end_time - Utc::now();
            if remaining.num_minutes() > 0 {
                remaining.num_minutes() as u32
            } else {
                0
            }
        })
    }

    /// Evaluate check results and determine action.
    pub fn evaluate(&mut self, results: &[CheckResult]) -> CircuitAction {
        if !self.config.enabled {
            return CircuitAction::Continue;
        }

        // Check for auto-recovery
        if self.state == CircuitState::Open && self.config.auto_recover && !self.in_cooldown() {
            self.state = CircuitState::HalfOpen;
        }

        // Count severities
        self.crit_count = 0;
        let mut halt_count = 0;
        let mut has_data_crit = false;
        let mut has_drawdown_crit = false;
        let mut has_cost_crit = false;

        for result in results {
            match result.severity {
                Severity::Halt => halt_count += 1,
                Severity::Crit => {
                    self.crit_count += 1;
                    
                    // Check specific critical conditions
                    if result.category == CheckCategory::DataHealth {
                        if result.check_name.contains("Freshness") || 
                           result.check_name.contains("Watermark") {
                            has_data_crit = true;
                        }
                    }
                    if result.check_name.contains("Drawdown") {
                        has_drawdown_crit = true;
                    }
                    if result.check_name.contains("Cost") {
                        has_cost_crit = true;
                    }
                }
                _ => {}
            }
        }

        // Determine action based on state and conditions
        let action = self.determine_action(
            halt_count, 
            has_data_crit, 
            has_drawdown_crit, 
            has_cost_crit
        );

        // Update state based on action
        match action {
            CircuitAction::HaltWithError => {
                self.trip();
            }
            CircuitAction::FlagNoTrade => {
                if self.state == CircuitState::Closed {
                    self.state = CircuitState::HalfOpen;
                }
            }
            CircuitAction::Continue => {
                if self.state == CircuitState::HalfOpen {
                    self.state = CircuitState::Closed;
                }
            }
            _ => {}
        }

        action
    }

    /// Determine action based on conditions.
    fn determine_action(
        &self,
        halt_count: usize,
        has_data_crit: bool,
        has_drawdown_crit: bool,
        has_cost_crit: bool,
    ) -> CircuitAction {
        // Any HALT severity = immediate halt
        if halt_count > 0 {
            return CircuitAction::HaltWithError;
        }

        // Specific critical conditions that trigger immediate halt
        if self.config.halt_on_data_crit && has_data_crit {
            return CircuitAction::HaltWithError;
        }
        if self.config.halt_on_drawdown_crit && has_drawdown_crit {
            return CircuitAction::FlagNoTrade;
        }
        if self.config.halt_on_cost_crit && has_cost_crit {
            return CircuitAction::FlagNoTrade;
        }

        // Accumulation threshold
        if self.crit_count >= self.halt_threshold {
            return CircuitAction::HaltWithError;
        }

        // Any critical = flag no trade
        if self.crit_count > 0 {
            return CircuitAction::FlagNoTrade;
        }

        // Check if in problematic state
        match self.state {
            CircuitState::Open => CircuitAction::HaltWithError,
            CircuitState::HalfOpen => CircuitAction::WarnAndContinue,
            CircuitState::Closed => CircuitAction::Continue,
        }
    }

    /// Trip the circuit breaker.
    pub fn trip(&mut self) {
        self.state = CircuitState::Open;
        self.last_trip = Some(Utc::now());
    }

    /// Reset the circuit breaker.
    pub fn reset(&mut self) {
        self.state = CircuitState::Closed;
        self.crit_count = 0;
        self.last_trip = None;
    }

    /// Force close the circuit (for testing/manual intervention).
    pub fn force_close(&mut self) {
        self.state = CircuitState::Closed;
    }

    /// Get state for reporting.
    pub fn to_state(&self) -> CircuitBreakerState {
        CircuitBreakerState {
            state: self.state.to_string(),
            crit_count: self.crit_count,
            halt_threshold: self.halt_threshold,
            action: self.determine_action(0, false, false, false),
            last_trip: self.last_trip,
            cooldown_remaining_minutes: self.cooldown_remaining(),
        }
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new(CircuitBreakerConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitoring::types::CheckResult;

    fn make_pass() -> CheckResult {
        CheckResult::pass("test", CheckCategory::DataHealth)
    }

    fn make_warn() -> CheckResult {
        CheckResult::warn("test", CheckCategory::Drift, "warning")
    }

    fn make_crit() -> CheckResult {
        CheckResult::crit("test", CheckCategory::Regression, "critical")
    }

    fn make_halt() -> CheckResult {
        CheckResult::halt("test", CheckCategory::CircuitBreaker, "halt")
    }

    fn make_data_crit() -> CheckResult {
        CheckResult::crit("Freshness_BR", CheckCategory::DataHealth, "stale data")
    }

    fn make_drawdown_crit() -> CheckResult {
        CheckResult::crit("DrawdownGuardrail", CheckCategory::Regression, "high dd")
    }

    #[test]
    fn test_initial_state() {
        let cb = CircuitBreaker::default();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.crit_count(), 0);
    }

    #[test]
    fn test_all_pass() {
        let mut cb = CircuitBreaker::default();
        let results = vec![make_pass(), make_pass(), make_pass()];
        
        let action = cb.evaluate(&results);
        
        assert_eq!(action, CircuitAction::Continue);
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_warnings_only() {
        let mut cb = CircuitBreaker::default();
        let results = vec![make_pass(), make_warn(), make_warn()];
        
        let action = cb.evaluate(&results);
        
        assert_eq!(action, CircuitAction::Continue);
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_single_crit() {
        let mut cb = CircuitBreaker::default();
        let results = vec![make_pass(), make_crit()];
        
        let action = cb.evaluate(&results);
        
        assert_eq!(action, CircuitAction::FlagNoTrade);
        assert_eq!(cb.crit_count(), 1);
    }

    #[test]
    fn test_halt_severity() {
        let mut cb = CircuitBreaker::default();
        let results = vec![make_pass(), make_halt()];
        
        let action = cb.evaluate(&results);
        
        assert_eq!(action, CircuitAction::HaltWithError);
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_crit_accumulation() {
        let mut cb = CircuitBreaker::default();
        // Default threshold is 3 CRITs
        let results = vec![make_crit(), make_crit(), make_crit()];
        
        let action = cb.evaluate(&results);
        
        assert_eq!(action, CircuitAction::HaltWithError);
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_data_crit_immediate_halt() {
        let config = CircuitBreakerConfig {
            halt_on_data_crit: true,
            ..Default::default()
        };
        let mut cb = CircuitBreaker::new(config);
        let results = vec![make_pass(), make_data_crit()];
        
        let action = cb.evaluate(&results);
        
        assert_eq!(action, CircuitAction::HaltWithError);
    }

    #[test]
    fn test_drawdown_crit_flag_no_trade() {
        let config = CircuitBreakerConfig {
            halt_on_drawdown_crit: true,
            ..Default::default()
        };
        let mut cb = CircuitBreaker::new(config);
        let results = vec![make_pass(), make_drawdown_crit()];
        
        let action = cb.evaluate(&results);
        
        assert_eq!(action, CircuitAction::FlagNoTrade);
    }

    #[test]
    fn test_reset() {
        let mut cb = CircuitBreaker::default();
        cb.trip();
        assert_eq!(cb.state(), CircuitState::Open);
        
        cb.reset();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.last_trip.is_none());
    }

    #[test]
    fn test_disabled_circuit_breaker() {
        let config = CircuitBreakerConfig {
            enabled: false,
            ..Default::default()
        };
        let mut cb = CircuitBreaker::new(config);
        let results = vec![make_halt()];
        
        let action = cb.evaluate(&results);
        
        assert_eq!(action, CircuitAction::Continue);
    }

    #[test]
    fn test_to_state() {
        let cb = CircuitBreaker::default();
        let state = cb.to_state();
        
        assert_eq!(state.state, "Closed");
        assert_eq!(state.crit_count, 0);
        assert_eq!(state.halt_threshold, 3);
    }
}


//! Institutional thresholds for strategy validation.
//!
//! This module provides a single source of truth for all validation thresholds
//! used by the OMP (Orchestrator de Mineração Perpétua) system.
//!
//! These values are aligned with the OMP specification:
//! docs/especificacao_orquestrador_completa.md:458-464

use serde::{Deserialize, Serialize};

/// Institutional-grade thresholds for strategy validation and promotion.
///
/// These thresholds are used consistently across:
/// - Stage B validation (WFA)
/// - Hall of Fame criteria
/// - OMP promotion gates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstitutionalThresholds {
    /// Minimum out-of-sample Sharpe ratio for promotion
    /// OMP spec: 1.0 (very strict)
    pub min_oos_sharpe: f64,
    
    /// Maximum Probability of Backtest Overfitting
    /// OMP spec: 0.10 (10%)
    pub max_pbo: f64,
    
    /// Minimum Deflated Sharpe Ratio
    /// OMP spec: 0.8
    pub min_dsr: f64,
    
    /// Maximum allowed performance degradation (OOS vs IS)
    /// Default: 50% degradation allowed
    pub max_degradation_pct: f64,
    
    /// Minimum pass rate across validation splits
    /// Default: 50% of splits must pass
    pub min_split_pass_rate: f64,
    
    /// Maximum out-of-sample drawdown (negative value)
    /// OMP spec: -0.20 (20% max drawdown)
    pub max_oos_drawdown: f64,
}

impl Default for InstitutionalThresholds {
    fn default() -> Self {
        // Values aligned with OMP specification
        // docs/especificacao_orquestrador_completa.md:458-464
        Self {
            min_oos_sharpe: 1.0,      // OMP spec: min_oos_sharpe_net = 1.0
            max_pbo: 0.10,            // OMP spec: max_pbo = 0.10
            min_dsr: 0.8,             // OMP spec: min_dsr = 0.8
            max_degradation_pct: 50.0,
            min_split_pass_rate: 0.5,
            max_oos_drawdown: -0.20,  // OMP spec: max_drawdown_net = 0.20
        }
    }
}

impl InstitutionalThresholds {
    /// Create with research-grade thresholds (less strict, for development)
    pub fn research() -> Self {
        Self {
            min_oos_sharpe: 0.5,
            max_pbo: 0.20,
            min_dsr: 0.5,
            max_degradation_pct: 70.0,
            min_split_pass_rate: 0.4,
            max_oos_drawdown: -0.35,
        }
    }
    
    /// Create with production-grade thresholds (strictest)
    pub fn production() -> Self {
        Self::default()
    }
    
    /// Validate that thresholds are within acceptable ranges
    pub fn validate(&self) -> Result<(), String> {
        if self.min_oos_sharpe < 0.0 {
            return Err("min_oos_sharpe must be >= 0".into());
        }
        if self.max_pbo < 0.0 || self.max_pbo > 1.0 {
            return Err("max_pbo must be in [0, 1]".into());
        }
        if self.min_dsr < 0.0 || self.min_dsr > 3.0 {
            return Err("min_dsr must be in [0, 3]".into());
        }
        if self.max_degradation_pct < 0.0 || self.max_degradation_pct > 100.0 {
            return Err("max_degradation_pct must be in [0, 100]".into());
        }
        if self.min_split_pass_rate < 0.0 || self.min_split_pass_rate > 1.0 {
            return Err("min_split_pass_rate must be in [0, 1]".into());
        }
        if self.max_oos_drawdown > 0.0 {
            return Err("max_oos_drawdown must be <= 0 (negative value)".into());
        }
        Ok(())
    }
    
    /// Check if a candidate passes all thresholds
    pub fn check_candidate(
        &self,
        oos_sharpe: f64,
        pbo: f64,
        dsr: f64,
        degradation_pct: f64,
        split_pass_rate: f64,
        oos_drawdown: f64,
    ) -> (bool, Vec<String>) {
        let mut failures = Vec::new();
        
        if oos_sharpe < self.min_oos_sharpe {
            failures.push(format!(
                "OOS Sharpe {:.2} < min {:.2}",
                oos_sharpe, self.min_oos_sharpe
            ));
        }
        if pbo > self.max_pbo {
            failures.push(format!(
                "PBO {:.2} > max {:.2}",
                pbo, self.max_pbo
            ));
        }
        if dsr < self.min_dsr {
            failures.push(format!(
                "DSR {:.2} < min {:.2}",
                dsr, self.min_dsr
            ));
        }
        if degradation_pct > self.max_degradation_pct {
            failures.push(format!(
                "Degradation {:.1}% > max {:.1}%",
                degradation_pct, self.max_degradation_pct
            ));
        }
        if split_pass_rate < self.min_split_pass_rate {
            failures.push(format!(
                "Split pass rate {:.1}% < min {:.1}%",
                split_pass_rate * 100.0, self.min_split_pass_rate * 100.0
            ));
        }
        if oos_drawdown < self.max_oos_drawdown {
            failures.push(format!(
                "OOS Drawdown {:.1}% < max {:.1}%",
                oos_drawdown * 100.0, self.max_oos_drawdown * 100.0
            ));
        }
        
        (failures.is_empty(), failures)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_thresholds() {
        let t = InstitutionalThresholds::default();
        assert!(t.validate().is_ok());
        assert_eq!(t.min_oos_sharpe, 1.0);
        assert_eq!(t.max_pbo, 0.10);
        assert_eq!(t.min_dsr, 0.8);
    }
    
    #[test]
    fn test_research_thresholds() {
        let t = InstitutionalThresholds::research();
        assert!(t.validate().is_ok());
        assert!(t.min_oos_sharpe < InstitutionalThresholds::default().min_oos_sharpe);
    }
    
    #[test]
    fn test_validation_rejects_invalid() {
        let mut t = InstitutionalThresholds::default();
        t.max_pbo = 1.5;
        assert!(t.validate().is_err());
        
        t = InstitutionalThresholds::default();
        t.min_oos_sharpe = -0.5;
        assert!(t.validate().is_err());
    }
    
    #[test]
    fn test_check_candidate_pass() {
        let t = InstitutionalThresholds::default();
        let (passed, failures) = t.check_candidate(
            1.5,   // oos_sharpe > 1.0
            0.05,  // pbo < 0.10
            0.9,   // dsr > 0.8
            30.0,  // degradation < 50%
            0.6,   // pass_rate > 0.5
            -0.15, // drawdown > -0.20
        );
        assert!(passed);
        assert!(failures.is_empty());
    }
    
    #[test]
    fn test_check_candidate_fail() {
        let t = InstitutionalThresholds::default();
        let (passed, failures) = t.check_candidate(
            0.5,   // oos_sharpe < 1.0
            0.15,  // pbo > 0.10
            0.55,  // dsr < 0.8
            30.0,
            0.6,
            -0.15,
        );
        assert!(!passed);
        assert_eq!(failures.len(), 3);
    }
}


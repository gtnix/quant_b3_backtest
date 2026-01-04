//! Risk Clustering Module for Asset Classification.
//!
//! Groups assets into risk clusters based on volatility and liquidity.
//! Enables:
//! - Per-cluster parameter tuning (ATR multiplier, position sizing)
//! - Risk budget allocation by cluster
//! - Anti-concentration via cluster diversification
//!
//! Reference: Package Research v2.0 - Section (5)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::filters::Market;

/// Risk cluster categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RiskCluster {
    /// Low volatility, high liquidity (blue chip)
    LowVolHighLiq,
    /// Low volatility, medium liquidity
    LowVolMedLiq,
    /// Medium volatility, high liquidity
    MedVolHighLiq,
    /// Medium volatility, medium liquidity
    MedVolMedLiq,
    /// High volatility, high liquidity
    HighVolHighLiq,
    /// High volatility, medium liquidity (requires wider stops)
    HighVolMedLiq,
    /// High volatility, low liquidity (most risky)
    HighVolLowLiq,
}

impl RiskCluster {
    /// Get all cluster variants.
    pub fn all() -> Vec<RiskCluster> {
        vec![
            RiskCluster::LowVolHighLiq,
            RiskCluster::LowVolMedLiq,
            RiskCluster::MedVolHighLiq,
            RiskCluster::MedVolMedLiq,
            RiskCluster::HighVolHighLiq,
            RiskCluster::HighVolMedLiq,
            RiskCluster::HighVolLowLiq,
        ]
    }
    
    /// Get the risk level (0-2) for sorting/comparison.
    pub fn risk_level(&self) -> u8 {
        match self {
            RiskCluster::LowVolHighLiq => 0,
            RiskCluster::LowVolMedLiq => 1,
            RiskCluster::MedVolHighLiq => 1,
            RiskCluster::MedVolMedLiq => 2,
            RiskCluster::HighVolHighLiq => 2,
            RiskCluster::HighVolMedLiq => 3,
            RiskCluster::HighVolLowLiq => 4,
        }
    }
}

impl std::fmt::Display for RiskCluster {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RiskCluster::LowVolHighLiq => write!(f, "Low Vol / High Liq"),
            RiskCluster::LowVolMedLiq => write!(f, "Low Vol / Med Liq"),
            RiskCluster::MedVolHighLiq => write!(f, "Med Vol / High Liq"),
            RiskCluster::MedVolMedLiq => write!(f, "Med Vol / Med Liq"),
            RiskCluster::HighVolHighLiq => write!(f, "High Vol / High Liq"),
            RiskCluster::HighVolMedLiq => write!(f, "High Vol / Med Liq"),
            RiskCluster::HighVolLowLiq => write!(f, "High Vol / Low Liq"),
        }
    }
}

/// Configuration for cluster classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// Volatility thresholds (annualized)
    pub vol_low_threshold: f64,  // Below this = Low Vol
    pub vol_high_threshold: f64, // Above this = High Vol
    
    /// Liquidity thresholds (USD daily volume)
    pub liq_high_threshold_br: f64,  // Above this = High Liq (BR)
    pub liq_med_threshold_br: f64,   // Above this = Med Liq (BR)
    pub liq_high_threshold_us: f64,  // Above this = High Liq (US)
    pub liq_med_threshold_us: f64,   // Above this = Med Liq (US)
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            vol_low_threshold: 0.20,   // 20% annualized
            vol_high_threshold: 0.40,  // 40% annualized
            liq_high_threshold_br: 50_000_000.0,  // $50M daily (BR)
            liq_med_threshold_br: 10_000_000.0,   // $10M daily (BR)
            liq_high_threshold_us: 200_000_000.0, // $200M daily (US)
            liq_med_threshold_us: 50_000_000.0,   // $50M daily (US)
        }
    }
}

/// Parameters for a specific cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterParams {
    /// ATR multiplier adjustment (1.0 = base)
    pub atr_multiplier: f64,
    /// Maximum risk budget for this cluster (% of total risk)
    pub max_risk_budget_pct: f64,
    /// Maximum number of positions in this cluster
    pub max_positions: usize,
    /// Position size multiplier (1.0 = base)
    pub position_size_multiplier: f64,
}

impl Default for ClusterParams {
    fn default() -> Self {
        Self {
            atr_multiplier: 1.0,
            max_risk_budget_pct: 0.30,
            max_positions: 10,
            position_size_multiplier: 1.0,
        }
    }
}

/// Asset with data for clustering.
#[derive(Debug, Clone)]
pub struct ClusterCandidate {
    pub symbol: String,
    pub market: Market,
    pub volatility: f64,          // Annualized volatility
    pub daily_volume_usd: f64,    // Average daily volume in USD
}

impl ClusterCandidate {
    pub fn new(symbol: impl Into<String>, market: Market, volatility: f64, daily_volume_usd: f64) -> Self {
        Self {
            symbol: symbol.into(),
            market,
            volatility,
            daily_volume_usd,
        }
    }
}

/// Result of cluster classification.
#[derive(Debug, Clone)]
pub struct ClusterAssignment {
    pub symbol: String,
    pub market: Market,
    pub cluster: RiskCluster,
    pub volatility: f64,
    pub daily_volume_usd: f64,
}

/// Risk Cluster Engine for classifying and managing asset clusters.
#[derive(Debug, Clone)]
pub struct RiskClusterEngine {
    config: ClusterConfig,
    cluster_params: HashMap<RiskCluster, ClusterParams>,
}

impl RiskClusterEngine {
    pub fn new(config: ClusterConfig) -> Self {
        let cluster_params = Self::default_cluster_params();
        Self { config, cluster_params }
    }
    
    /// Create with custom cluster parameters.
    pub fn with_params(config: ClusterConfig, params: HashMap<RiskCluster, ClusterParams>) -> Self {
        Self { config, cluster_params: params }
    }
    
    /// Default parameters per cluster.
    fn default_cluster_params() -> HashMap<RiskCluster, ClusterParams> {
        let mut params = HashMap::new();
        
        // Low Vol / High Liq - safest, can use tighter stops
        params.insert(RiskCluster::LowVolHighLiq, ClusterParams {
            atr_multiplier: 0.8,
            max_risk_budget_pct: 0.40,
            max_positions: 15,
            position_size_multiplier: 1.2,
        });
        
        // Low Vol / Med Liq
        params.insert(RiskCluster::LowVolMedLiq, ClusterParams {
            atr_multiplier: 0.9,
            max_risk_budget_pct: 0.25,
            max_positions: 10,
            position_size_multiplier: 1.0,
        });
        
        // Med Vol / High Liq
        params.insert(RiskCluster::MedVolHighLiq, ClusterParams {
            atr_multiplier: 1.0,
            max_risk_budget_pct: 0.30,
            max_positions: 12,
            position_size_multiplier: 1.0,
        });
        
        // Med Vol / Med Liq
        params.insert(RiskCluster::MedVolMedLiq, ClusterParams {
            atr_multiplier: 1.1,
            max_risk_budget_pct: 0.20,
            max_positions: 8,
            position_size_multiplier: 0.9,
        });
        
        // High Vol / High Liq
        params.insert(RiskCluster::HighVolHighLiq, ClusterParams {
            atr_multiplier: 1.2,
            max_risk_budget_pct: 0.15,
            max_positions: 6,
            position_size_multiplier: 0.8,
        });
        
        // High Vol / Med Liq - requires wider stops
        params.insert(RiskCluster::HighVolMedLiq, ClusterParams {
            atr_multiplier: 1.4,
            max_risk_budget_pct: 0.10,
            max_positions: 4,
            position_size_multiplier: 0.6,
        });
        
        // High Vol / Low Liq - most risky
        params.insert(RiskCluster::HighVolLowLiq, ClusterParams {
            atr_multiplier: 1.6,
            max_risk_budget_pct: 0.05,
            max_positions: 2,
            position_size_multiplier: 0.4,
        });
        
        params
    }
    
    /// Classify a single asset into a cluster.
    pub fn classify(&self, candidate: &ClusterCandidate) -> RiskCluster {
        let vol_level = self.volatility_level(candidate.volatility);
        let liq_level = self.liquidity_level(candidate.daily_volume_usd, candidate.market);
        
        match (vol_level, liq_level) {
            (VolLevel::Low, LiqLevel::High) => RiskCluster::LowVolHighLiq,
            (VolLevel::Low, LiqLevel::Med) => RiskCluster::LowVolMedLiq,
            (VolLevel::Low, LiqLevel::Low) => RiskCluster::LowVolMedLiq, // Treat as Med
            (VolLevel::Med, LiqLevel::High) => RiskCluster::MedVolHighLiq,
            (VolLevel::Med, LiqLevel::Med) => RiskCluster::MedVolMedLiq,
            (VolLevel::Med, LiqLevel::Low) => RiskCluster::MedVolMedLiq, // Treat as Med
            (VolLevel::High, LiqLevel::High) => RiskCluster::HighVolHighLiq,
            (VolLevel::High, LiqLevel::Med) => RiskCluster::HighVolMedLiq,
            (VolLevel::High, LiqLevel::Low) => RiskCluster::HighVolLowLiq,
        }
    }
    
    /// Classify multiple assets.
    pub fn classify_all(&self, candidates: &[ClusterCandidate]) -> Vec<ClusterAssignment> {
        candidates.iter().map(|c| {
            ClusterAssignment {
                symbol: c.symbol.clone(),
                market: c.market,
                cluster: self.classify(c),
                volatility: c.volatility,
                daily_volume_usd: c.daily_volume_usd,
            }
        }).collect()
    }
    
    /// Group assignments by cluster.
    pub fn group_by_cluster(&self, assignments: &[ClusterAssignment]) -> HashMap<RiskCluster, Vec<ClusterAssignment>> {
        let mut groups: HashMap<RiskCluster, Vec<ClusterAssignment>> = HashMap::new();
        for a in assignments {
            groups.entry(a.cluster).or_default().push(a.clone());
        }
        groups
    }
    
    /// Get parameters for a cluster.
    pub fn get_params(&self, cluster: RiskCluster) -> ClusterParams {
        self.cluster_params.get(&cluster).cloned().unwrap_or_default()
    }
    
    /// Calculate cluster statistics.
    pub fn cluster_stats(&self, assignments: &[ClusterAssignment]) -> ClusterStats {
        let groups = self.group_by_cluster(assignments);
        
        let distribution: HashMap<RiskCluster, usize> = groups.iter()
            .map(|(c, v)| (*c, v.len()))
            .collect();
        
        let total = assignments.len();
        let high_risk_count = groups.iter()
            .filter(|(c, _)| c.risk_level() >= 3)
            .map(|(_, v)| v.len())
            .sum();
        
        ClusterStats {
            total_assets: total,
            high_risk_count,
            distribution,
        }
    }
    
    fn volatility_level(&self, vol: f64) -> VolLevel {
        if vol <= self.config.vol_low_threshold {
            VolLevel::Low
        } else if vol >= self.config.vol_high_threshold {
            VolLevel::High
        } else {
            VolLevel::Med
        }
    }
    
    fn liquidity_level(&self, liq: f64, market: Market) -> LiqLevel {
        let (high_thresh, med_thresh) = match market {
            Market::BR => (self.config.liq_high_threshold_br, self.config.liq_med_threshold_br),
            Market::US => (self.config.liq_high_threshold_us, self.config.liq_med_threshold_us),
        };
        
        if liq >= high_thresh {
            LiqLevel::High
        } else if liq >= med_thresh {
            LiqLevel::Med
        } else {
            LiqLevel::Low
        }
    }
}

impl Default for RiskClusterEngine {
    fn default() -> Self {
        Self::new(ClusterConfig::default())
    }
}

#[derive(Debug, Clone, Copy)]
enum VolLevel { Low, Med, High }

#[derive(Debug, Clone, Copy)]
enum LiqLevel { Low, Med, High }

/// Statistics about cluster distribution.
#[derive(Debug, Clone)]
pub struct ClusterStats {
    pub total_assets: usize,
    pub high_risk_count: usize,
    pub distribution: HashMap<RiskCluster, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_classification() {
        let engine = RiskClusterEngine::default();
        
        // Low vol, high liq (BR blue chip)
        let blue_chip = ClusterCandidate::new("ITUB4", Market::BR, 0.15, 100_000_000.0);
        assert_eq!(engine.classify(&blue_chip), RiskCluster::LowVolHighLiq);
        
        // High vol, low liq (small cap)
        let small_cap = ClusterCandidate::new("SMALL3", Market::BR, 0.50, 5_000_000.0);
        assert_eq!(engine.classify(&small_cap), RiskCluster::HighVolLowLiq);
        
        // Medium vol, medium liq
        let mid_cap = ClusterCandidate::new("MID3", Market::BR, 0.30, 20_000_000.0);
        assert_eq!(engine.classify(&mid_cap), RiskCluster::MedVolMedLiq);
    }

    #[test]
    fn test_cluster_params() {
        let engine = RiskClusterEngine::default();
        
        let safe_params = engine.get_params(RiskCluster::LowVolHighLiq);
        let risky_params = engine.get_params(RiskCluster::HighVolLowLiq);
        
        // Safer clusters should have lower ATR multiplier
        assert!(safe_params.atr_multiplier < risky_params.atr_multiplier);
        
        // Safer clusters should have higher risk budget
        assert!(safe_params.max_risk_budget_pct > risky_params.max_risk_budget_pct);
    }

    #[test]
    fn test_classify_all() {
        let engine = RiskClusterEngine::default();
        
        let candidates = vec![
            ClusterCandidate::new("PETR4", Market::BR, 0.35, 200_000_000.0),
            ClusterCandidate::new("VALE3", Market::BR, 0.30, 150_000_000.0),
            ClusterCandidate::new("SMALL3", Market::BR, 0.55, 8_000_000.0),
        ];
        
        let assignments = engine.classify_all(&candidates);
        assert_eq!(assignments.len(), 3);
        
        let stats = engine.cluster_stats(&assignments);
        assert_eq!(stats.total_assets, 3);
    }

    #[test]
    fn test_us_vs_br_thresholds() {
        let engine = RiskClusterEngine::default();
        
        // Same volume but different markets
        let br_asset = ClusterCandidate::new("PETR4", Market::BR, 0.25, 60_000_000.0);
        let us_asset = ClusterCandidate::new("AAPL", Market::US, 0.25, 60_000_000.0);
        
        let br_cluster = engine.classify(&br_asset);
        let us_cluster = engine.classify(&us_asset);
        
        // $60M is High Liq for BR but only Med Liq for US
        assert_eq!(br_cluster, RiskCluster::MedVolHighLiq);
        assert_eq!(us_cluster, RiskCluster::MedVolMedLiq);
    }
}


//! # Backtester Tests
//!
//! Integration test package for the backtester workspace.
//! This crate exists only to host integration tests.

#![deny(unsafe_code)]

use backtester_core::{MarketEvent, SignalEvent, StrategyConfig};

/// Trait for test strategies
pub trait TestStrategy: Send {
    /// Initialize strategy
    fn on_init(&mut self, _config: &StrategyConfig, _num_assets: usize) {}
    /// Process market event
    fn on_market(&mut self, _event: &MarketEvent) -> Option<SignalEvent> {
        None
    }
    /// Strategy name
    fn name(&self) -> &str;
}

/// No-operation strategy for testing infrastructure
pub struct NoOpStrategy;

impl TestStrategy for NoOpStrategy {
    fn name(&self) -> &str {
        "noop"
    }
}

/// Simple trend strategy for testing
#[allow(dead_code)]
pub struct SimpleTrendStrategy {
    short_period: usize,
    long_period: usize,
}

#[allow(dead_code)]
impl SimpleTrendStrategy {
    pub fn new(short_period: usize, long_period: usize) -> Self {
        Self {
            short_period,
            long_period,
        }
    }
}

impl TestStrategy for SimpleTrendStrategy {
    fn name(&self) -> &str {
        "simple_trend"
    }
}

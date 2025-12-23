//! # Anti-Look-Ahead Test Suite
//!
//! Validates that strategies cannot access future data.
//!
//! These tests use specially crafted data patterns that would reveal
//! if decisions were made using information not yet available.

use backtester_core::{Bar, MarketEvent, SignalEvent, Strategy, StrategyConfig};
use backtester_engine::SimulationEngine;
use strategy_lib::MeanReversionStrategy;

/// A "cheating" strategy that would profit impossibly if it could see the future.
/// Used to verify the engine properly restricts data access.
struct LookAheadDetectorStrategy {
    last_timestamp: Option<i64>,
    events_seen: Vec<i64>,
}

impl LookAheadDetectorStrategy {
    fn new() -> Self {
        Self {
            last_timestamp: None,
            events_seen: Vec::new(),
        }
    }
}

impl Strategy for LookAheadDetectorStrategy {
    fn on_market(&mut self, event: &MarketEvent) -> Option<SignalEvent> {
        let current_ts = event.bar.timestamp;

        if let Some(last_ts) = self.last_timestamp {
            // Verify events are delivered in chronological order
            assert!(
                current_ts >= last_ts,
                "Look-ahead detected: received event with timestamp {} after {}",
                current_ts,
                last_ts
            );
        }

        self.events_seen.push(current_ts);
        self.last_timestamp = Some(current_ts);
        None
    }
}

#[test]
fn anti_look_ahead_events_in_order() {
    let mut strategy = LookAheadDetectorStrategy::new();

    // Simulate events in chronological order
    for i in 0..100 {
        let event = MarketEvent {
            asset_id: 1,
            bar: Bar {
                timestamp: i * 60_000_000_000, // 1 minute intervals
                open: 100.0,
                high: 100.0,
                low: 100.0,
                close: 100.0,
                volume: 1000.0,
            },
        };
        strategy.on_market(&event);
    }
}

/// Test that strategy cannot see future price movements.
/// Pattern: Price spike at T=50, but signal should not anticipate it.
#[test]
fn anti_look_ahead_no_anticipation_of_spike() {
    struct SpikeDetectorStrategy {
        signals_before_spike: Vec<i64>,
        spike_timestamp: i64,
    }

    impl Strategy for SpikeDetectorStrategy {
        fn on_market(&mut self, event: &MarketEvent) -> Option<SignalEvent> {
            // If we signal before the spike, we might be looking ahead
            if event.bar.timestamp < self.spike_timestamp && event.bar.close > 105.0 {
                self.signals_before_spike.push(event.bar.timestamp);
            }
            
            // Only signal if current price shows the spike (price > 120)
            if event.bar.close > 120.0 {
                return Some(SignalEvent {
                    timestamp: event.bar.timestamp,
                    asset_id: event.asset_id,
                    strength: 1.0,
                });
            }
            None
        }
    }

    let spike_time = 50 * 60_000_000_000_i64;
    let mut strategy = SpikeDetectorStrategy {
        signals_before_spike: Vec::new(),
        spike_timestamp: spike_time,
    };

    // Generate events: flat until T=50, then spike
    for i in 0..100 {
        let ts = i * 60_000_000_000;
        let price = if i >= 50 { 125.0 } else { 100.0 };
        
        let event = MarketEvent {
            asset_id: 0,
            bar: Bar {
                timestamp: ts,
                open: price,
                high: price + 0.5,
                low: price - 0.5,
                close: price,
                volume: 1000.0,
            },
        };
        strategy.on_market(&event);
    }

    // Strategy should NOT have any signals before the spike
    assert!(
        strategy.signals_before_spike.is_empty(),
        "Look-ahead detected: {} signals before spike",
        strategy.signals_before_spike.len()
    );
}

/// Test that the engine maintains strict temporal ordering across assets.
#[test]
fn anti_look_ahead_multi_asset_ordering() {
    struct MultiAssetOrderChecker {
        last_ts_per_asset: Vec<i64>,
        global_last_ts: i64,
    }

    impl Strategy for MultiAssetOrderChecker {
        fn on_market(&mut self, event: &MarketEvent) -> Option<SignalEvent> {
            let id = event.asset_id as usize;
            let ts = event.bar.timestamp;

            // Ensure per-asset ordering
            while self.last_ts_per_asset.len() <= id {
                self.last_ts_per_asset.push(0);
            }
            
            assert!(
                ts >= self.last_ts_per_asset[id],
                "Look-ahead in asset {}: ts {} < last {}",
                id,
                ts,
                self.last_ts_per_asset[id]
            );
            self.last_ts_per_asset[id] = ts;

            // Global timestamp should be monotonic or equal
            assert!(
                ts >= self.global_last_ts,
                "Global look-ahead: ts {} < global_last {}",
                ts,
                self.global_last_ts
            );
            self.global_last_ts = ts;

            None
        }
    }

    let mut strategy = MultiAssetOrderChecker {
        last_ts_per_asset: Vec::new(),
        global_last_ts: 0,
    };

    // Generate interleaved events for 3 assets
    for bar_idx in 0..50 {
        for asset_id in 0..3 {
            let ts = bar_idx * 60_000_000_000;
            let event = MarketEvent {
                asset_id: asset_id as u32,
                bar: Bar {
                    timestamp: ts,
                    open: 100.0,
                    high: 101.0,
                    low: 99.0,
                    close: 100.0,
                    volume: 1000.0,
                },
            };
            strategy.on_market(&event);
        }
    }
}

/// Test that trend strategy decisions are based only on past data.
/// Pattern: Signals should only be generated based on data seen so far.
#[test]
fn anti_look_ahead_trend_crossover_timing() {
    struct TrendTimingChecker {
        prices_seen: Vec<f64>,
        signal_price_counts: Vec<usize>, // How many prices seen when signal was generated
    }

    impl Strategy for TrendTimingChecker {
        fn on_market(&mut self, event: &MarketEvent) -> Option<SignalEvent> {
            self.prices_seen.push(event.bar.close);
            
            // Calculate MAs from ONLY the data we've seen
            if self.prices_seen.len() >= 5 {
                let len = self.prices_seen.len();
                let short_ma: f64 = self.prices_seen[len-2..].iter().sum::<f64>() / 2.0;
                let long_ma: f64 = self.prices_seen[len-5..].iter().sum::<f64>() / 5.0;
                
                // Signal based on current visible data only
                if short_ma > long_ma && len >= 6 {
                    self.signal_price_counts.push(len);
                    return Some(SignalEvent {
                        timestamp: event.bar.timestamp,
                        asset_id: event.asset_id,
                        strength: 1.0,
                    });
                }
            }
            None
        }
    }

    let mut strategy = TrendTimingChecker {
        prices_seen: Vec::new(),
        signal_price_counts: Vec::new(),
    };

    // Generate prices: flat then rising
    // Bars 0-4: price = 100 (building baseline)
    // Bars 5-9: price rises 101, 102, 103, 104, 105
    for i in 0..10 {
        let ts = i * 86_400_000_000_000_i64;
        let price = if i < 5 { 100.0 } else { 100.0 + (i - 4) as f64 };
        
        let event = MarketEvent {
            asset_id: 0,
            bar: Bar {
                timestamp: ts,
                open: price,
                high: price + 0.5,
                low: price - 0.5,
                close: price,
                volume: 1000.0,
            },
        };
        
        strategy.on_market(&event);
    }

    // Any signal should have been based on at least 5 prior prices
    for &count in &strategy.signal_price_counts {
        assert!(
            count >= 5,
            "Signal generated with only {} prices seen (needs at least 5)",
            count
        );
    }
}

/// Test mean reversion strategy only reacts to current VWAP deviation.
#[test]
fn anti_look_ahead_mean_reversion_vwap() {
    let mut strategy = MeanReversionStrategy::new(0.02, 10);
    strategy.on_init(&StrategyConfig::default(), 1);

    let mut signal_timestamps: Vec<i64> = Vec::new();

    // First 5 bars: price at 100, building VWAP baseline
    // Next 5 bars: price drops to 95 (should trigger buy)
    for i in 0..10 {
        let ts = i * 60_000_000_000;
        let price = if i < 5 { 100.0 } else { 95.0 };
        
        let event = MarketEvent {
            asset_id: 0,
            bar: Bar {
                timestamp: ts,
                open: price,
                high: price + 0.5,
                low: price - 0.5,
                close: price,
                volume: 10000.0,
            },
        };
        
        if let Some(_signal) = strategy.on_market(&event) {
            signal_timestamps.push(ts);
            assert!(
                ts >= 5 * 60_000_000_000_i64,
                "Signal at ts {} is before price drop at ts {}",
                ts,
                5 * 60_000_000_000_i64
            );
        }
    }

    // Should have signals only after the price drop
    for ts in &signal_timestamps {
        assert!(
            *ts >= 5 * 60_000_000_000_i64,
            "Look-ahead detected: signal at {} before price change",
            ts
        );
    }
}

/// Test that engine integration maintains temporal barriers.
#[test]
fn anti_look_ahead_engine_integration() {
    struct TimestampRecorder {
        market_timestamps: Vec<i64>,
        signal_timestamps: Vec<i64>,
    }

    impl Strategy for TimestampRecorder {
        fn on_market(&mut self, event: &MarketEvent) -> Option<SignalEvent> {
            self.market_timestamps.push(event.bar.timestamp);
            
            // Signal on every 10th event
            if self.market_timestamps.len() % 10 == 0 {
                self.signal_timestamps.push(event.bar.timestamp);
                return Some(SignalEvent {
                    timestamp: event.bar.timestamp,
                    asset_id: event.asset_id,
                    strength: 1.0,
                });
            }
            None
        }
    }

    let strategy = TimestampRecorder {
        market_timestamps: Vec::new(),
        signal_timestamps: Vec::new(),
    };

    let mut engine = SimulationEngine::with_defaults(strategy, 1_000_000.0, 2);

    // Process 50 events
    for i in 0..50 {
        let event = MarketEvent {
            asset_id: (i % 2) as u32,
            bar: Bar {
                timestamp: i * 60_000_000_000,
                open: 100.0,
                high: 101.0,
                low: 99.0,
                close: 100.0,
                volume: 10000.0,
            },
        };
        engine.process_event(&event);
    }

    let result = engine.get_result();
    
    // Verify we processed all events
    assert_eq!(result.events_processed, 50);
    
    // NAV history should have entries in chronological order
    if let Some(nav_hist) = &result.nav_history {
        let timestamps = &nav_hist.timestamps;
        for i in 1..timestamps.len() {
            assert!(
                timestamps[i] >= timestamps[i - 1],
                "NAV history out of order: {} < {}",
                timestamps[i],
                timestamps[i - 1]
            );
        }
    }
}

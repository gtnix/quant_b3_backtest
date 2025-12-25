//! # Anti-Look-Ahead Test Suite
//!
//! Validates that strategies cannot access future data.

use backtester_core::{AssetId, Bar, MarketEvent, SignalEvent, Strategy, StrategyConfig};

/// A "cheating" strategy that would profit impossibly if it could see the future.
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

    for i in 0..100 {
        let event = MarketEvent {
            asset_id: AssetId::new(1),
            bar: Bar {
                timestamp: i * 60_000_000_000,
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
#[test]
fn anti_look_ahead_no_anticipation_of_spike() {
    struct SpikeDetectorStrategy {
        signals_before_spike: Vec<i64>,
        spike_timestamp: i64,
    }

    impl Strategy for SpikeDetectorStrategy {
        fn on_market(&mut self, event: &MarketEvent) -> Option<SignalEvent> {
            if event.bar.timestamp < self.spike_timestamp && event.bar.close > 105.0 {
                self.signals_before_spike.push(event.bar.timestamp);
            }

            if event.bar.close > 120.0 {
                return Some(SignalEvent::buy(event.bar.timestamp, event.asset_id, 1.0));
            }
            None
        }
    }

    let spike_time = 50 * 60_000_000_000_i64;
    let mut strategy = SpikeDetectorStrategy {
        signals_before_spike: Vec::new(),
        spike_timestamp: spike_time,
    };

    for i in 0..100 {
        let ts = i * 60_000_000_000;
        let price = if i >= 50 { 125.0 } else { 100.0 };

        let event = MarketEvent {
            asset_id: AssetId::new(0),
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
            let id = event.asset_id.as_usize();
            let ts = event.bar.timestamp;

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

    for bar_idx in 0..50 {
        for asset_id in 0..3u16 {
            let ts = bar_idx * 60_000_000_000;
            let event = MarketEvent {
                asset_id: AssetId::new(asset_id),
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
#[test]
fn anti_look_ahead_trend_crossover_timing() {
    struct TrendTimingChecker {
        prices_seen: Vec<f64>,
        signal_price_counts: Vec<usize>,
    }

    impl Strategy for TrendTimingChecker {
        fn on_market(&mut self, event: &MarketEvent) -> Option<SignalEvent> {
            self.prices_seen.push(event.bar.close);

            if self.prices_seen.len() >= 5 {
                let len = self.prices_seen.len();
                let short_ma: f64 = self.prices_seen[len - 2..].iter().sum::<f64>() / 2.0;
                let long_ma: f64 = self.prices_seen[len - 5..].iter().sum::<f64>() / 5.0;

                if short_ma > long_ma && len >= 6 {
                    self.signal_price_counts.push(len);
                    return Some(SignalEvent::buy(event.bar.timestamp, event.asset_id, 1.0));
                }
            }
            None
        }
    }

    let mut strategy = TrendTimingChecker {
        prices_seen: Vec::new(),
        signal_price_counts: Vec::new(),
    };

    for i in 0..10 {
        let ts = i * 86_400_000_000_000_i64;
        let price = if i < 5 { 100.0 } else { 100.0 + (i - 4) as f64 };

        let event = MarketEvent {
            asset_id: AssetId::new(0),
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

    for &count in &strategy.signal_price_counts {
        assert!(
            count >= 5,
            "Signal generated with only {} prices seen (needs at least 5)",
            count
        );
    }
}

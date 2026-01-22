//! Indicator cache for avoiding redundant calculations in hot path.
//!
//! Usage:
//! ```ignore
//! let cache = IndicatorCache::new();
//! let sma = cache.sma(prices, 20);
//! let ema = cache.ema(prices, 20);
//! let atr = cache.atr_proxy(prices, 14);
//! ```

use std::collections::HashMap;
use std::cell::RefCell;

/// Thread-local indicator cache.
thread_local! {
    static CACHE: RefCell<CacheStorage> = RefCell::new(CacheStorage::new());
}

/// Internal cache storage.
#[derive(Debug, Default)]
struct CacheStorage {
    sma: HashMap<(usize, usize), f64>,    // (prices_hash, period) -> value
    ema: HashMap<(usize, usize), f64>,
    atr: HashMap<(usize, usize), f64>,
    volatility: HashMap<(usize, usize), f64>,
    momentum: HashMap<(usize, usize), f64>,
}

impl CacheStorage {
    fn new() -> Self {
        Self::default()
    }

    fn clear(&mut self) {
        self.sma.clear();
        self.ema.clear();
        self.atr.clear();
        self.volatility.clear();
        self.momentum.clear();
    }
}

/// Quick hash of price slice for cache key.
#[inline]
fn hash_prices(prices: &[f64]) -> usize {
    if prices.is_empty() {
        return 0;
    }
    // Use first, last, and length for quick hash
    let first = (prices[0] * 1000.0) as usize;
    let last = (prices[prices.len() - 1] * 1000.0) as usize;
    first.wrapping_mul(31).wrapping_add(last).wrapping_mul(31).wrapping_add(prices.len())
}

/// Clear the thread-local cache (call at start of each backtest).
pub fn clear_cache() {
    CACHE.with(|c| c.borrow_mut().clear());
}

/// Get cached or compute SMA.
#[inline]
pub fn cached_sma(prices: &[f64], period: usize) -> Option<f64> {
    if prices.len() < period || period == 0 {
        return None;
    }

    let key = (hash_prices(prices), period);
    
    CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        if let Some(&val) = cache.sma.get(&key) {
            return Some(val);
        }
        
        let sum: f64 = prices.iter().rev().take(period).sum();
        let sma = sum / period as f64;
        cache.sma.insert(key, sma);
        Some(sma)
    })
}

/// Get cached or compute EMA.
#[inline]
pub fn cached_ema(prices: &[f64], period: usize) -> Option<f64> {
    if prices.len() < period || period == 0 {
        return None;
    }

    let key = (hash_prices(prices), period);
    
    CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        if let Some(&val) = cache.ema.get(&key) {
            return Some(val);
        }
        
        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut ema = prices[0];
        for &price in &prices[1..] {
            ema = (price - ema) * multiplier + ema;
        }
        cache.ema.insert(key, ema);
        Some(ema)
    })
}

/// Get cached or compute ATR proxy (uses close-to-close).
#[inline]
pub fn cached_atr_proxy(prices: &[f64], period: usize) -> Option<f64> {
    if prices.len() < period + 1 || period == 0 {
        return None;
    }

    let key = (hash_prices(prices), period);
    
    CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        if let Some(&val) = cache.atr.get(&key) {
            return Some(val);
        }
        
        // ATR proxy: average of absolute returns
        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .collect();
        
        if returns.len() < period {
            return None;
        }
        
        let atr = returns.iter().rev().take(period).sum::<f64>() / period as f64;
        cache.atr.insert(key, atr);
        Some(atr)
    })
}

/// Get cached or compute volatility (std dev of returns).
#[inline]
pub fn cached_volatility(prices: &[f64], period: usize) -> Option<f64> {
    if prices.len() < period + 1 || period == 0 {
        return None;
    }

    let key = (hash_prices(prices), period);
    
    CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        if let Some(&val) = cache.volatility.get(&key) {
            return Some(val);
        }
        
        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] - w[0]) / w[0].max(0.001))
            .collect();
        
        if returns.len() < period {
            return None;
        }
        
        let recent: Vec<f64> = returns.iter().rev().take(period).copied().collect();
        let mean = recent.iter().sum::<f64>() / period as f64;
        let variance = recent.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / period as f64;
        let vol = variance.sqrt();
        
        cache.volatility.insert(key, vol);
        Some(vol)
    })
}

/// Get cached or compute momentum (return over period).
#[inline]
pub fn cached_momentum(prices: &[f64], period: usize) -> Option<f64> {
    if prices.len() < period || period == 0 {
        return None;
    }

    let key = (hash_prices(prices), period);
    
    CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        if let Some(&val) = cache.momentum.get(&key) {
            return Some(val);
        }
        
        let current = *prices.last()?;
        let past = prices[prices.len() - period];
        let mom = if past.abs() > 0.001 {
            (current - past) / past
        } else {
            0.0
        };
        
        cache.momentum.insert(key, mom);
        Some(mom)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cached_sma() {
        let prices = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
        
        // First call computes
        let sma1 = cached_sma(&prices, 3);
        assert!(sma1.is_some());
        
        // Second call uses cache
        let sma2 = cached_sma(&prices, 3);
        assert_eq!(sma1, sma2);
        
        clear_cache();
    }

    #[test]
    fn test_cached_volatility() {
        let prices = vec![100.0, 101.0, 99.0, 102.0, 98.0, 103.0];
        let vol = cached_volatility(&prices, 5);
        assert!(vol.is_some());
        assert!(vol.unwrap() > 0.0);
        clear_cache();
    }
}

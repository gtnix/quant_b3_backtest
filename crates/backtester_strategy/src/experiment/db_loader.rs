//! Database Market Loader - Loads market data from Neon database.
//!
//! Provides high-performance loading of OHLCV data from Neon Postgres,
//! with OBFS caching for ultra-fast repeated loads.

use chrono::NaiveDate;
use std::path::PathBuf;
use thiserror::Error;
use tracing::{info, debug, warn};

use market_data::{
    BacktestMarketData, BacktestDataError, MarketDataCache,
    load_ohlcv_for_backtest_with_market, list_symbols_for_market, get_market_date_range,
};
use market_data::db::Database;

use super::market_data::MarketDataProvider;

/// Errors from database market loading.
#[derive(Debug, Error)]
pub enum DbLoaderError {
    #[error("Database error: {0}")]
    Database(String),
    
    #[error("No data found for market: {0}")]
    NoDataFound(String),
    
    #[error("Cache error: {0}")]
    Cache(String),
    
    #[error("Market data error: {0}")]
    MarketData(#[from] BacktestDataError),
}

/// Database market loader with OBFS caching.
///
/// Loads OHLCV data from Neon database tables:
/// - `ohlcv_daily` for BR market
/// - `ohlcv_daily_us` for US market
///
/// Uses OBFS compression (~10x) for caching to avoid repeated database queries.
pub struct DatabaseMarketLoader {
    market: String,
    cache_dir: PathBuf,
    use_cache: bool,
}

impl DatabaseMarketLoader {
    /// Create a new loader for the specified market.
    ///
    /// # Arguments
    /// * `market` - Market identifier: "BR" or "US"
    pub fn new(market: &str) -> Self {
        Self {
            market: market.to_uppercase(),
            cache_dir: PathBuf::from(".cache/market_data"),
            use_cache: true,
        }
    }
    
    /// Set custom cache directory.
    pub fn with_cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = dir.into();
        self
    }
    
    /// Disable caching (for testing or forced refresh).
    pub fn without_cache(mut self) -> Self {
        self.use_cache = false;
        self
    }
    
    /// Get the cache key for this market and date range.
    fn cache_key(&self, start: NaiveDate, end: NaiveDate) -> String {
        format!("{}_{}_to_{}", self.market, start, end)
    }
    
    /// Load market data from database or cache.
    ///
    /// Returns a MarketDataProvider ready for use in ExperimentRunner.
    /// 
    /// Note: For US market with 2M+ bars, this loads the last 5 years by default
    /// to avoid memory issues. Use load_range() for specific date ranges.
    pub async fn load(&self) -> Result<MarketDataProvider, DbLoaderError> {
        info!("Loading market data for {} from database", self.market);
        
        // Connect to database
        let db = Database::connect().await
            .map_err(|e| DbLoaderError::Database(e.to_string()))?;
        
        // Get date range for this market
        let (db_start, end) = get_market_date_range(db.client(), &self.market)
            .await?
            .ok_or_else(|| DbLoaderError::NoDataFound(self.market.clone()))?;
        
        // Limit to last 2 years by default to avoid loading millions of rows at once
        // Both BR (2.7M rows) and US (2.4M rows) have too much historical data
        // Use load_range() for specific date ranges
        let two_years_ago = end - chrono::Duration::days(2 * 365);
        let start = if two_years_ago > db_start { two_years_ago } else { db_start };
        
        info!("Market {} date range: {} to {} (db has {} to {})", 
              self.market, start, end, db_start, end);
        
        // Try cache first
        if self.use_cache {
            let cache = MarketDataCache::new(&self.cache_dir);
            
            if let Ok(data) = cache.load_with_market(&[], start, end, &self.market) {
                debug!("Cache HIT for market {}", self.market);
                return Ok(MarketDataProvider::from_backtest_data(data));
            }
        }
        
        // Cache miss - load from database
        info!("Cache MISS - loading {} data from Neon database", self.market);
        
        let data = load_ohlcv_for_backtest_with_market(
            db.client(),
            &[],  // Empty = load all symbols
            start,
            end,
            &self.market,
        ).await.map_err(|e| {
            tracing::error!("Failed to load data from database: {}", e);
            e
        })?;
        
        let summary = data.summary();
        info!(
            "Loaded {} bars for {} symbols ({} trading days) from {}",
            summary.total_bars, summary.symbols, summary.trading_days, self.market
        );
        
        // Save to cache
        if self.use_cache {
            // Ensure cache directory exists
            if let Err(e) = std::fs::create_dir_all(&self.cache_dir) {
                warn!("Failed to create cache directory: {}", e);
            } else {
                let cache = MarketDataCache::new(&self.cache_dir);
                if let Err(e) = cache.save_with_market(&[], start, end, &data, &self.market) {
                    warn!("Failed to save cache: {}", e);
                }
            }
        }
        
        Ok(MarketDataProvider::from_backtest_data(data))
    }
    
    /// Load with specific date range.
    pub async fn load_range(
        &self,
        start: NaiveDate,
        end: NaiveDate,
    ) -> Result<MarketDataProvider, DbLoaderError> {
        info!("Loading {} data for range {} to {}", self.market, start, end);
        
        let db = Database::connect().await
            .map_err(|e| DbLoaderError::Database(e.to_string()))?;
        
        // Try cache
        if self.use_cache {
            let cache = MarketDataCache::new(&self.cache_dir);
            if let Ok(data) = cache.load_with_market(&[], start, end, &self.market) {
                debug!("Cache HIT for {} range", self.market);
                return Ok(MarketDataProvider::from_backtest_data(data));
            }
        }
        
        // Load from database
        let data = load_ohlcv_for_backtest_with_market(
            db.client(),
            &[],
            start,
            end,
            &self.market,
        ).await?;
        
        // Cache it
        if self.use_cache {
            let cache = MarketDataCache::new(&self.cache_dir);
            let _ = cache.save_with_market(&[], start, end, &data, &self.market);
        }
        
        Ok(MarketDataProvider::from_backtest_data(data))
    }
    
    /// Load only the symbol list (fast operation).
    pub async fn load_symbols(&self) -> Result<Vec<String>, DbLoaderError> {
        let db = Database::connect().await
            .map_err(|e| DbLoaderError::Database(e.to_string()))?;
        
        let symbols = list_symbols_for_market(db.client(), &self.market).await?;
        
        info!("Loaded {} symbols for market {}", symbols.len(), self.market);
        Ok(symbols)
    }
    
    /// Get the market identifier.
    pub fn market(&self) -> &str {
        &self.market
    }
}

/// Synchronous wrapper for use in non-async contexts.
///
/// Uses tokio runtime to run async database operations.
pub fn load_market_data_sync(market: &str) -> Result<MarketDataProvider, DbLoaderError> {
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| DbLoaderError::Database(format!("Failed to create runtime: {}", e)))?;
    
    rt.block_on(async {
        let loader = DatabaseMarketLoader::new(market);
        loader.load().await
    })
}

/// Load market data for a specific date range (sync).
pub fn load_market_data_range_sync(
    market: &str,
    start: NaiveDate,
    end: NaiveDate,
) -> Result<MarketDataProvider, DbLoaderError> {
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| DbLoaderError::Database(format!("Failed to create runtime: {}", e)))?;
    
    rt.block_on(async {
        let loader = DatabaseMarketLoader::new(market);
        loader.load_range(start, end).await
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cache_key_generation() {
        let loader = DatabaseMarketLoader::new("US");
        let start = NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
        let end = NaiveDate::from_ymd_opt(2024, 12, 31).unwrap();
        
        let key = loader.cache_key(start, end);
        assert_eq!(key, "US_2020-01-01_to_2024-12-31");
    }
    
    #[test]
    fn test_market_normalization() {
        let loader = DatabaseMarketLoader::new("us");
        assert_eq!(loader.market(), "US");
        
        let loader = DatabaseMarketLoader::new("BR");
        assert_eq!(loader.market(), "BR");
    }
}

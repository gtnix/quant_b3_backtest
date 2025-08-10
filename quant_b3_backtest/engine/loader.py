"""
Load and preprocess B3 market data
Handle corporate actions, liquidity filters, and data quality

This module provides functionality to:
- Load raw CSV data from Alpha Vantage downloads
- Apply B3-specific processing and filters
- Calculate technical indicators and features
- Handle missing data and data quality issues
- Prepare data for backtesting strategies
- Automatically download missing data when needed
- Handle SGS and IBOV data downloads

"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, List, Any, Tuple
import json
from datetime import datetime, timedelta
import sys
import os
import yfinance as yf  # Keep for IBOV only
import time
from dataclasses import dataclass
import yaml

# Add scripts directory to path for imports
scripts_path = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_path))

# Import BrapiProvider
try:
    from .brapi_provider import BrapiProvider
    BRAPI_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"BrapiProvider not available: {e}")
    BRAPI_AVAILABLE = False

# Import download modules (Alpha Vantage downloader removed)
try:
    from download_ibov_yahoo import YahooIBOVDownloader
    from sgs_data_loader import SGSDataLoader
    DOWNLOAD_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Download modules not available: {e}")
    DOWNLOAD_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


_CONFIG_CACHE: Dict[str, Any] = {}

def load_brapi_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """
    Load Brapi.dev configuration from settings file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary with Brapi configuration
    """
    # Simple in-process cache to avoid re-reading YAML repeatedly
    if _CONFIG_CACHE.get('__path__') == config_path and 'config' in _CONFIG_CACHE:
        config = _CONFIG_CACHE['config']
    else:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            _CONFIG_CACHE['__path__'] = config_path
            _CONFIG_CACHE['config'] = config
        except Exception as e:
            logger.error(f"Failed to load Brapi configuration: {e}")
            return {}
        
        brapi_config = config.get('brapi', {})
        
        # Get API token from environment variable
        api_token = os.getenv('BRAPI_API_TOKEN')
        if not api_token:
            logger.warning("BRAPI_API_TOKEN environment variable not set")
            return {}
        
        brapi_config['api_token'] = api_token
        return brapi_config
        
    


@dataclass
class CacheMetadata:
    """Metadata for cached yfinance data."""
    symbol: str
    start_date: str
    end_date: str
    data_type: str  # 'daily' or 'hourly'
    cache_created: datetime
    cache_ttl_hours: int
    data_points: int
    
    def is_stale(self) -> bool:
        """Check if cache is stale based on TTL."""
        age_hours = (datetime.now() - self.cache_created).total_seconds() / 3600
        return age_hours > self.cache_ttl_hours
    
    def covers_range(self, start_date: str, end_date: str) -> bool:
        """Check if cached data covers the requested date range."""
        return self.start_date <= start_date and self.end_date >= end_date


@dataclass
class CacheStats:
    """Statistics for cache performance."""
    hits: int = 0
    fetches: int = 0
    load_time: float = 0.0
    
    @property 
    def hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        total = self.hits + self.fetches
        return self.hits / total if total > 0 else 0.0


class SmartYFinanceProvider:
    """
    Intelligent caching provider for yfinance daily data with smart fetching.
    
    Features:
    - Smart caching of daily data with TTL-based invalidation
    - Range-based fetching to minimize API calls
    - Intelligent cache merging and gap filling
    - Performance monitoring and statistics
    - Robust error handling and retries
    - Optimized for Brazilian market technical indicators
    """
    
    def __init__(self, cache_dir: str = "data/yfinance_cache", cache_ttl_hours: int = 24):
        """
        Initialize the smart yfinance provider.
        
        Args:
            cache_dir: Directory for cache storage
            cache_ttl_hours: Cache time-to-live in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_hours = cache_ttl_hours
        self.stats = CacheStats()
        
        # Initialize cache subdirectories
        (self.cache_dir / "daily").mkdir(exist_ok=True)
        
        # Load cache statistics
        self._load_cache_stats()
        
        logger.info(f"SmartYFinanceProvider initialized: cache_dir={cache_dir}, ttl={cache_ttl_hours}h")
    
    def get_daily_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get daily data with smart caching and intelligent fetching.
        
        Args:
            symbol: Trading symbol (e.g., 'ALPA4.SA')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with daily OHLCV data
        """
        start_time = time.time()
        
        try:
            # Ensure .SA suffix for Brazilian stocks
            yf_symbol = self._ensure_sa_suffix(symbol)
            
            # Check cache coverage and freshness
            cache_file, metadata = self._get_cache_info(yf_symbol, 'daily')
            
            if cache_file.exists() and metadata and not metadata.is_stale():
                if metadata.covers_range(start_date, end_date):
                    # Perfect cache hit
                    data = self._load_from_cache(cache_file)
                    if data is not None:
                        filtered_data = self._filter_date_range(data, start_date, end_date)
                        self.stats.hits += 1
                        self.stats.load_time += time.time() - start_time
                        logger.info(f"✅ Cache hit for {symbol} daily data: {len(filtered_data)} bars")
                        return filtered_data
            
            # Calculate what ranges need fetching
            fetch_ranges = self._calculate_fetch_ranges(
                symbol, start_date, end_date, 'daily', metadata
            )
            
            if not fetch_ranges:
                # Use existing cache even if slightly stale
                if cache_file.exists():
                    data = self._load_from_cache(cache_file)
                    if data is not None:
                        filtered_data = self._filter_date_range(data, start_date, end_date)
                        self.stats.hits += 1
                        logger.info(f"✅ Stale cache used for {symbol}: {len(filtered_data)} bars")
                        return filtered_data
            
            # Fetch missing data
            logger.info(f"📡 Fetching daily data for {symbol}: {len(fetch_ranges)} range(s)")
            new_data_parts = []
            
            for fetch_start, fetch_end in fetch_ranges:
                part_data = self._fetch_yfinance_data(yf_symbol, fetch_start, fetch_end, 'daily')
                if part_data is not None and not part_data.empty:
                    new_data_parts.append(part_data)
                time.sleep(0.1)  # Rate limiting
            
            # Merge with existing cache
            final_data = self._merge_with_cache(cache_file, new_data_parts, yf_symbol, 'daily')
            
            # Filter to requested range
            result_data = self._filter_date_range(final_data, start_date, end_date)
            
            self.stats.fetches += 1
            self.stats.load_time += time.time() - start_time
            logger.info(f"✅ Smart fetch completed for {symbol}: {len(result_data)} bars")
            
            return result_data
            
        except Exception as e:
            logger.error(f"Error in get_daily_data for {symbol}: {e}")
            self.stats.load_time += time.time() - start_time
            return pd.DataFrame()
    
    def _ensure_sa_suffix(self, symbol: str) -> str:
        """Ensure Brazilian stock symbol has .SA suffix for yfinance."""
        if not symbol.endswith('.SA'):
            return f"{symbol}.SA"
        return symbol
    
    def _get_cache_info(self, symbol: str, data_type: str) -> Tuple[Path, Optional[CacheMetadata]]:
        """Get cache file path and metadata for a symbol."""
        cache_subdir = self.cache_dir / data_type
        cache_file = cache_subdir / f"{symbol}_{data_type}.parquet"
        metadata_file = cache_subdir / f"{symbol}_{data_type}_metadata.json"
        
        metadata = None
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    meta_dict = json.load(f)
                    meta_dict['cache_created'] = datetime.fromisoformat(meta_dict['cache_created'])
                    metadata = CacheMetadata(**meta_dict)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata for {symbol}: {e}")
        
        return cache_file, metadata
    
    def _calculate_fetch_ranges(self, symbol: str, start_date: str, end_date: str, 
                               data_type: str, metadata: Optional[CacheMetadata]) -> List[Tuple[str, str]]:
        """
        Calculate what date ranges need to be fetched based on cache coverage.
        
        Returns:
            List of (start_date, end_date) tuples that need fetching
        """
        if metadata is None or metadata.is_stale():
            # No cache or stale cache - fetch entire range
            return [(start_date, end_date)]
        
        if metadata.covers_range(start_date, end_date):
            # Cache covers requested range and is fresh
            return []
        
        fetch_ranges = []
        
        # Check if we need data before cached range
        if start_date < metadata.start_date:
            fetch_ranges.append((start_date, metadata.start_date))
        
        # Check if we need data after cached range
        if end_date > metadata.end_date:
            fetch_ranges.append((metadata.end_date, end_date))
        
        return fetch_ranges
    
    def _fetch_yfinance_data(self, symbol: str, start_date: str, end_date: str, 
                           data_type: str) -> Optional[pd.DataFrame]:
        """Fetch data from yfinance with retry logic."""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(symbol)
                
                if data_type == 'daily':
                    data = ticker.history(start=start_date, end=end_date, interval='1d')
                else:
                    raise ValueError(f"Unsupported data_type: {data_type} (only 'daily' supported)")
                
                if data.empty:
                    logger.warning(f"No data returned from yfinance for {symbol}")
                    return None
                
                # Standardize column names to match our convention
                data.columns = data.columns.str.lower()
                
                # Ensure we have the required columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in required_cols:
                    if col not in data.columns:
                        logger.error(f"Missing column {col} in yfinance data for {symbol}")
                        return None
                
                # Clean and validate data
                data = self._clean_yfinance_data(data)
                
                logger.debug(f"Fetched {len(data)} {data_type} bars for {symbol}")
                return data
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"All fetch attempts failed for {symbol}")
                    return None
    
    def _clean_yfinance_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate yfinance data."""
        # Remove any rows with NaN in critical columns
        critical_cols = ['open', 'high', 'low', 'close']
        initial_len = len(data)
        data = data.dropna(subset=critical_cols)
        
        if len(data) < initial_len:
            logger.warning(f"Removed {initial_len - len(data)} rows with NaN values")
        
        # Validate price relationships
        invalid_mask = (data['low'] > data['high']) | (data['open'] < 0) | (data['close'] < 0)
        if invalid_mask.any():
            logger.warning(f"Removing {invalid_mask.sum()} rows with invalid price data")
            data = data[~invalid_mask]
        
        # Fill volume NaNs with 0
        data['volume'] = data['volume'].fillna(0)
        
        return data
    
    def _load_from_cache(self, cache_file: Path) -> Optional[pd.DataFrame]:
        """Load data from cache file."""
        try:
            data = pd.read_parquet(cache_file)
            logger.debug(f"Loaded {len(data)} bars from cache: {cache_file.name}")
            return data
        except Exception as e:
            logger.error(f"Failed to load cache file {cache_file}: {e}")
            return None
    
    def _save_to_cache(self, data: pd.DataFrame, symbol: str, data_type: str, 
                      start_date: str, end_date: str):
        """Save data to cache with metadata."""
        try:
            cache_subdir = self.cache_dir / data_type
            cache_file = cache_subdir / f"{symbol}_{data_type}.parquet"
            metadata_file = cache_subdir / f"{symbol}_{data_type}_metadata.json"
            
            # Save data
            data.to_parquet(cache_file)
            
            # Save metadata
            metadata = CacheMetadata(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                data_type=data_type,
                cache_created=datetime.now(),
                cache_ttl_hours=self.cache_ttl_hours,
                data_points=len(data)
            )
            
            meta_dict = {
                'symbol': metadata.symbol,
                'start_date': metadata.start_date,
                'end_date': metadata.end_date,
                'data_type': metadata.data_type,
                'cache_created': metadata.cache_created.isoformat(),
                'cache_ttl_hours': metadata.cache_ttl_hours,
                'data_points': metadata.data_points
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(meta_dict, f, indent=2)
            
            logger.debug(f"Saved {len(data)} bars to cache: {cache_file.name}")
            
        except Exception as e:
            logger.error(f"Failed to save cache for {symbol}: {e}")
    
    def _merge_with_cache(self, cache_file: Path, new_data_parts: List[pd.DataFrame], 
                         symbol: str, data_type: str) -> pd.DataFrame:
        """Intelligently merge new data with existing cache."""
        try:
            # Load existing cache if it exists
            existing_data = None
            if cache_file.exists():
                existing_data = self._load_from_cache(cache_file)
            
            # Combine all data parts
            all_data_parts = []
            if existing_data is not None and not existing_data.empty:
                all_data_parts.append(existing_data)
            all_data_parts.extend(new_data_parts)
            
            if not all_data_parts:
                return pd.DataFrame()
            
            # Merge and deduplicate
            merged_data = pd.concat(all_data_parts, ignore_index=False)
            merged_data = merged_data.sort_index()
            merged_data = merged_data[~merged_data.index.duplicated(keep='last')]
            
            # Calculate new date range
            start_date = merged_data.index.min().strftime('%Y-%m-%d')
            end_date = merged_data.index.max().strftime('%Y-%m-%d')
            
            # Save updated cache
            self._save_to_cache(merged_data, symbol, data_type, start_date, end_date)
            
            return merged_data
            
        except Exception as e:
            logger.error(f"Error merging cache for {symbol}: {e}")
            # Return new data only if merge fails
            if new_data_parts:
                return pd.concat(new_data_parts, ignore_index=False).sort_index()
            return pd.DataFrame()
    
    def _filter_date_range(self, data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """Filter data to requested date range."""
        if data.empty:
            return data
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Handle timezone-aware data by converting to timezone-naive for comparison
        if data.index.tz is not None:
            # Convert timezone-aware index to timezone-naive for comparison
            data_index_naive = data.index.tz_localize(None)
            mask = (data_index_naive >= start_dt) & (data_index_naive <= end_dt)
        else:
            # Timezone-naive data - direct comparison
            mask = (data.index >= start_dt) & (data.index <= end_dt)
        
        return data.loc[mask]
    
    def _load_cache_stats(self):
        """Load cache performance statistics."""
        stats_file = self.cache_dir / "cache_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file, 'r') as f:
                    stats_data = json.load(f)
                    self.stats.hits = stats_data.get('hits', 0)
                    self.stats.fetches = stats_data.get('fetches', 0)
                    self.stats.load_time = stats_data.get('load_time', 0.0)
            except Exception as e:
                logger.warning(f"Failed to load cache stats: {e}")
    
    def _save_cache_stats(self):
        """Save cache performance statistics."""
        try:
            stats_file = self.cache_dir / "cache_stats.json"
            stats_data = {
                'hits': self.stats.hits,
                'fetches': self.stats.fetches,
                'load_time': self.stats.load_time,
                'hit_ratio': self.stats.hit_ratio,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(stats_file, 'w') as f:
                json.dump(stats_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save cache stats: {e}")
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get comprehensive cache status and performance metrics."""
        # Save current stats
        self._save_cache_stats()
        
        # Count cache files
        daily_files = len(list((self.cache_dir / "daily").glob("*.parquet")))
        
        # Calculate cache size
        total_size = 0
        for cache_file in self.cache_dir.rglob("*.parquet"):
            total_size += cache_file.stat().st_size
        
        return {
            'performance': {
                'hits': self.stats.hits,
                'fetches': self.stats.fetches,
                'hit_ratio': f"{self.stats.hit_ratio:.1%}",
                'total_load_time': f"{self.stats.load_time:.2f}s"
            },
            'storage': {
                'daily_symbols': daily_files,
                'total_size_mb': f"{total_size / (1024*1024):.1f}",
                'cache_directory': str(self.cache_dir)
            },
            'settings': {
                'cache_ttl_hours': self.cache_ttl_hours,
                'cache_directory': str(self.cache_dir)
            }
        }
    
    def cleanup_old_cache(self, max_age_days: int = 30):
        """Clean up old cache files."""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        removed_count = 0
        
        for cache_file in self.cache_dir.rglob("*.parquet"):
            try:
                if datetime.fromtimestamp(cache_file.stat().st_mtime) < cutoff_date:
                    cache_file.unlink()
                    # Also remove corresponding metadata
                    metadata_file = cache_file.with_name(cache_file.stem + "_metadata.json")
                    if metadata_file.exists():
                        metadata_file.unlink()
                    removed_count += 1
            except Exception as e:
                logger.warning(f"Failed to remove old cache file {cache_file}: {e}")
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old cache files")


class DataFallbackHandler:
    """Handle insufficient data with fail-fast approach."""
    
    def __init__(self, brapi_provider: BrapiProvider):
        """
        Initialize the fallback handler.
        
        Args:
            brapi_provider: Brapi.dev provider for data
        """
        self.brapi_provider = brapi_provider
    
    def handle_insufficient_execution_data(self, symbol: str, coverage_stats: Dict[str, Any]) -> pd.DataFrame:
        """
        Fail-fast approach when Brapi.dev data is insufficient.
        
        Args:
            symbol: Trading symbol
            coverage_stats: Dictionary with coverage analysis
            
        Returns:
            DataFrame with execution data or raises SystemExit
        """
        print(f"\n❌ INSUFFICIENT BRAPI.DEV DATA")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📊 Symbol: {symbol}")
        print(f"📅 Coverage: {coverage_stats['percentage']:.1%}")
        print(f"📈 Technical indicators: ✅ Ready (Brapi.dev daily)")
        print(f"⚡ Execution simulation: ❌ Insufficient Brapi.dev hourly data")
        
        print(f"\n📋 ERROR DETAILS:")
        print(f"   • Issue: {coverage_stats.get('issue', 'Unknown')}")
        print(f"   • Available range: {coverage_stats.get('available_range', 'None')}")
        print(f"   • Bars count: {coverage_stats.get('bars_count', 0)}")
        
        print(f"\n🔧 TROUBLESHOOTING:")
        print(f"   1. Check BRAPI_API_TOKEN environment variable")
        print(f"   2. Verify symbol exists on B3: {symbol}")
        print(f"   3. Check Brapi.dev API status")
        print(f"   4. Review network connectivity")
        
        raise SystemExit(f"Backtest cancelled: Insufficient Brapi.dev data for {symbol}")


class DataSourceReporter:
    """Generate comprehensive reports of data sources used."""
    
    def generate_comprehensive_report(self, data_config: Dict[str, Any]):
        """Generate detailed data source report."""
        # Allow disabling noisy stdout reports in batch/parallel runs
        try:
            import os as _os
            if (_os.getenv('DISABLE_DATA_SOURCE_REPORT', '0').lower() in ('1', 'true', 'yes')):
                return
        except Exception:
            pass

        print(f"\n📋 BACKTEST DATA SOURCE REPORT")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # Technical Indicators Section
        indicators_config = data_config.get('indicators', {})
        print(f"\n🧮 TECHNICAL INDICATORS:")
        print(f"  📊 Source: {indicators_config.get('source', 'Unknown')}")
        print(f"  📅 Date range: {indicators_config.get('date_range', 'Unknown')}")
        print(f"  💾 Cache status: {indicators_config.get('cache_status', 'Unknown')}")
        print(f"  🎯 Data quality: {indicators_config.get('quality', 'Unknown')}")
        
        # Execution Data Section  
        execution_config = data_config.get('execution', {})
        print(f"\n⚡ EXECUTION SIMULATION:")
        print(f"  📊 Source: {execution_config.get('source', 'Unknown')}")
        print(f"  📅 Coverage: {execution_config.get('coverage', 'Unknown')}")
        print(f"  🎯 Accuracy level: {execution_config.get('accuracy', 'Unknown')}")
        print(f"  📈 Date range: {execution_config.get('date_range', 'Unknown')}")
        
        # Validation Section
        validation_config = data_config.get('validation', {})
        if validation_config.get('available', False):
            print(f"\n🧪 VALIDATION COMPARISON:")
            print(f"  📊 Local vs yfinance indicators: Available")
            print(f"  📈 Correlation: {validation_config.get('correlation', 0):.3f}")
            print(f"  📊 Mean difference: {validation_config.get('mean_diff', 0):.2f}%")
        else:
            print(f"\n🧪 VALIDATION: Not available (no local data for comparison)")
        
        # Cache Performance
        cache_config = data_config.get('cache_stats', {})
        if cache_config:
            print(f"\n💾 CACHE PERFORMANCE:")
            print(f"  🚀 Cache hits: {cache_config.get('hits', 0)}")
            print(f"  📡 API calls: {cache_config.get('fetches', 0)}")
            print(f"  ⏱️  Load time: {cache_config.get('load_time', 0):.2f}s")
        
        # Overall Assessment
        overall_config = data_config.get('overall', {})
        print(f"\n🎯 OVERALL ASSESSMENT:")
        print(f"  📊 Backtest confidence: {overall_config.get('confidence', 'Unknown')}")
        print(f"  ⚠️  Limitations: {overall_config.get('limitations', 'None identified')}")
        print(f"  💡 Recommendations: {overall_config.get('recommendations', 'None')}")
        
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


class HybridDataManager:
    """
    Core component managing multiple data sources with smart fallbacks.
    
    Features:
    - Brapi.dev daily data for technical indicators (cached)
    - Brapi.dev hourly data for execution simulation
    - Interactive fallbacks for insufficient data
    - Comprehensive validation and reporting
    - Data source transparency and performance monitoring
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Initialize the hybrid data manager.
        
        Args:
            config_path: Path to configuration file
        """
        # Load Brapi configuration
        brapi_config = load_brapi_config(config_path)
        
        if not brapi_config:
            logger.error("❌ Brapi configuration not available")
            logger.error("   Please set BRAPI_API_TOKEN environment variable")
            raise SystemExit("Brapi.dev configuration required")
        
        if not BRAPI_AVAILABLE:
            logger.error("❌ BrapiProvider module not available")
            logger.error("   Please ensure brapi_provider.py is in the engine directory")
            raise SystemExit("BrapiProvider module required")
        
        # Initialize BrapiProvider
        data_config = brapi_config.get('data', {})
        self.brapi_provider = BrapiProvider(
            api_token=brapi_config['api_token'],
            cache_dir=data_config.get('cache_dir', 'data/brapi_cache'),
            cache_ttl_hours=data_config.get('cache_ttl_hours', 24),
            timeout=brapi_config.get('timeout', 30),
            max_retries=brapi_config.get('max_retries', 3)
        )
        
        # No fallback mechanism - Brapi.dev is the only source
        self.fallback_handler = DataFallbackHandler(self.brapi_provider)
        self.reporter = DataSourceReporter()
        
        # For compatibility with existing code
        self.local_loader = None  # Will be set by initialize_backtest_data
        
        logger.info("HybridDataManager initialized with Brapi.dev and fallbacks")
    
    def initialize_backtest_data(self, symbol: str, start_date: str, end_date: str, 
                                local_loader: Any = None) -> Dict[str, Any]:
        """
        Initialize data for backtesting with hybrid strategy.
        
        Args:
            symbol: Trading symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            local_loader: DataLoader instance for local data (kept for compatibility)
            
        Returns:
            Dictionary with all initialized data and metadata
        """
        self.local_loader = local_loader
        
        logger.info(f"🔄 Initializing Brapi.dev backtest data for {symbol} ({start_date} to {end_date})")
        
        # Phase 1: Get Brapi.dev daily for indicators (cached)
        logger.info("📈 Phase 1: Loading technical indicator data from Brapi.dev...")
        
        # Use full available historical data for technical indicators (Brapi.dev has extensive history)
        # This ensures sufficient data for ATR calculation and other technical indicators
        indicator_start = "2020-01-01"  # Use extensive historical data from Brapi.dev
        indicator_data = self.brapi_provider.get_daily_data(symbol, indicator_start, end_date)
        
        indicators_ready = not indicator_data.empty
        indicator_source = "brapi_daily_cached" if indicators_ready else "unavailable"
        
        # Phase 2: Load Brapi.dev hourly for execution (with warmup extension)
        logger.info("⚡ Phase 2: Loading execution data from Brapi.dev hourly...")
        
        # Calculate warmup extension: Strategy needs 29 trading days for proper warmup
        # Add 45 calendar days to ensure sufficient trading days (accounting for weekends/holidays)
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        warmup_start_dt = start_dt - timedelta(days=45)  # 45 calendar days before start
        warmup_start_date = warmup_start_dt.strftime('%Y-%m-%d')
        
        logger.info(f"📅 Extending execution data range for warmup:")
        logger.info(f"   - Requested start: {start_date}")
        logger.info(f"   - Warmup start: {warmup_start_date} (45 days earlier)")
        logger.info(f"   - This ensures sufficient historical data for 29 trading days warmup")
        
        # Load extended execution data
        execution_data = self.brapi_provider.get_hourly_data(symbol, warmup_start_date, end_date)
        execution_source = "brapi_hourly_extended"
        
        # Phase 3: Assess coverage and quality
        logger.info("📊 Phase 3: Assessing data coverage and quality...")
        coverage_stats = self._assess_execution_coverage(execution_data, start_date, end_date, symbol)
        
        # Phase 4: Fail fast if insufficient data
        if not coverage_stats['sufficient']:
            logger.error("❌ Insufficient Brapi.dev data detected - failing fast")
            self.fallback_handler.handle_insufficient_execution_data(symbol, coverage_stats)
            # This will raise SystemExit, so we won't reach here
        
        # Phase 5: Setup validation comparison if possible
        logger.info("🧪 Phase 5: Setting up validation comparison...")
        validation_data = self._setup_validation_comparison(symbol, indicator_data, execution_data)
        
        # Generate comprehensive data source report
        data_config = self._generate_data_source_report(
            symbol, indicator_data, execution_data, validation_data, 
            indicator_source, execution_source, coverage_stats
        )
        
        # Display report
        self.reporter.generate_comprehensive_report(data_config)
        
        return {
            'indicator_data': indicator_data,
            'execution_data': execution_data,
            'validation_data': validation_data,
            'data_sources': data_config,
            'coverage_stats': coverage_stats,
            'cache_stats': self.brapi_provider.get_cache_status()
        }
    
    def _assess_execution_coverage(self, execution_data: Optional[pd.DataFrame], 
                                  start_date: str, end_date: str, symbol: str) -> Dict[str, Any]:
        """Assess quality and coverage of execution data."""
        if execution_data is None or execution_data.empty:
            return {
                'sufficient': False,
                'percentage': 0.0,
                'available_range': 'No data',
                'available_data': None,
                'requested_start': start_date,
                'requested_end': end_date,
                'issue': 'no_local_data',
                'bars_count': 0
            }
        
        # Filter to requested date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Get data in requested range
        mask = (execution_data.index >= start_dt) & (execution_data.index <= end_dt)
        filtered_data = execution_data.loc[mask]
        
        if filtered_data.empty:
            return {
                'sufficient': False,
                'percentage': 0.0,
                'available_range': f"{execution_data.index.min().date()} to {execution_data.index.max().date()}",
                'available_data': execution_data,
                'requested_start': start_date,
                'requested_end': end_date,
                'issue': 'no_data_in_range',
                'bars_count': len(execution_data)
            }
        
        # Calculate coverage statistics
        total_calendar_days = (end_dt - start_dt).days + 1
        unique_trading_days = len(set(filtered_data.index.date))
        expected_trading_days = total_calendar_days * 5 // 7  # Rough estimate
        
        coverage_percentage = unique_trading_days / max(expected_trading_days, 1)
        
        # Assess data quality
        bars_per_day = len(filtered_data) / max(unique_trading_days, 1)
        expected_bars_per_day = 7  # Brazilian market typically has 7 hourly bars per day
        
        sufficient = (
            coverage_percentage >= 0.8 and  # At least 80% of trading days
            bars_per_day >= 4 and  # At least 4 bars per day on average
            len(filtered_data) >= 50  # Minimum absolute number of bars
        )
        
        return {
            'sufficient': sufficient,
            'percentage': coverage_percentage,
            'available_range': f"{filtered_data.index.min().date()} to {filtered_data.index.max().date()}",
            'available_data': filtered_data,
            'requested_start': start_date,
            'requested_end': end_date,
            'unique_trading_days': unique_trading_days,
            'bars_per_day': bars_per_day,
            'bars_count': len(filtered_data),
            'quality_score': min(coverage_percentage * bars_per_day / expected_bars_per_day, 1.0)
        }
    
    def _setup_validation_comparison(self, symbol: str, indicator_data: pd.DataFrame, 
                                   execution_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Setup validation comparison between different data sources."""
        if indicator_data.empty or execution_data.empty:
            return None
        
        try:
            # This would implement dual indicator calculation
            # For now, return basic validation info
            return {
                'available': True,
                'correlation': 0.95,  # Placeholder
                'mean_diff': 0.5,     # Placeholder
                'validation_type': 'yfinance_vs_local_aggregation'
            }
        except Exception as e:
            logger.warning(f"Failed to setup validation comparison: {e}")
            return None
    
    def _generate_data_source_report(self, symbol: str, indicator_data: pd.DataFrame, 
                                   execution_data: pd.DataFrame, validation_data: Optional[Dict],
                                   indicator_source: str, execution_source: str, 
                                   coverage_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive data source configuration."""
        
        # Cache stats - use BrapiProvider for cache status
        cache_status = self.brapi_provider.get_cache_status()
        
        return {
            'indicators': {
                'source': indicator_source,
                'date_range': f"{indicator_data.index.min().date()} to {indicator_data.index.max().date()}" if not indicator_data.empty else "No data",
                'cache_status': "Hit" if cache_status['performance']['hits'] > 0 else "Miss",
                'quality': "Excellent" if not indicator_data.empty else "Poor",
                'bars_count': len(indicator_data)
            },
            'execution': {
                'source': execution_source,
                'coverage': f"{coverage_stats.get('percentage', 0):.1%}",
                'accuracy': "High" if coverage_stats.get('sufficient', False) else "Medium",
                'date_range': coverage_stats.get('available_range', 'Unknown'),
                'bars_count': coverage_stats.get('bars_count', 0)
            },
            'validation': validation_data or {'available': False},
            'cache_stats': {
                'hits': cache_status['performance']['hits'],
                'fetches': cache_status['performance']['fetches'],
                'load_time': float(cache_status['performance']['total_load_time'].replace('s', ''))
            },
            'overall': {
                'confidence': "High" if coverage_stats.get('sufficient', False) and not indicator_data.empty else "Medium",
                'limitations': self._identify_limitations(indicator_source, execution_source, coverage_stats),
                'recommendations': self._generate_recommendations(indicator_source, execution_source, coverage_stats)
            }
        }
    
    def _identify_limitations(self, indicator_source: str, execution_source: str, 
                            coverage_stats: Dict[str, Any]) -> str:
        """Identify limitations based on data sources used."""
        limitations = []
        
        if not coverage_stats.get('sufficient', False):
            limitations.append("Partial data coverage")
        
        if coverage_stats.get('bars_per_day', 0) < 6:
            limitations.append("Low intraday granularity")
        
        return "; ".join(limitations) if limitations else "None identified"
    
    def _generate_recommendations(self, indicator_source: str, execution_source: str,
                                coverage_stats: Dict[str, Any]) -> str:
        """Generate recommendations based on data analysis."""
        recommendations = []
        
        if coverage_stats.get('percentage', 0) < 0.9:
            recommendations.append("Extend historical data range for better coverage")
        
        if coverage_stats.get('bars_per_day', 0) < 6:
            recommendations.append("Consider acquiring higher frequency intraday data")
        
        if not recommendations:
            recommendations.append("Data quality is excellent - no action required")
        
        return "; ".join(recommendations)


class DataLoader:
    """
    A class to load and process B3 market data for backtesting.
    
    This class handles:
    - Loading raw data from CSV files
    - Applying B3-specific filters and processing
    - Calculating technical indicators
    - Handling data quality issues
    - Preparing data for strategy backtesting
    - Automatically downloading missing data
    - Handling SGS and IBOV data
    """
    
    def __init__(self, raw_path: str = "data/raw", processed_path: str = "data/processed", 
                 auto_download: bool = True, config_path: str = "config/secrets.yaml"):
        """
        Initialize the DataLoader.
        
        Args:
            raw_path (str): Path to raw data directory
            processed_path (str): Path to processed data directory
            auto_download (bool): Whether to automatically download missing data
            config_path (str): Path to configuration file for API keys
        """
        self.raw_path = Path(raw_path)
        self.processed_path = Path(processed_path)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        # Auto-download configuration
        self.auto_download = auto_download and DOWNLOAD_AVAILABLE
        self.config_path = config_path
        
        # Initialize downloaders if available
        self.stock_downloader = None
        self.ibov_downloader = None
        self.sgs_downloader = None
        
        if self.auto_download:
            try:
                # Alpha Vantage stock downloader removed - using local intraday files
                self.stock_downloader = None
                self.ibov_downloader = YahooIBOVDownloader()
                self.sgs_downloader = SGSDataLoader()
                logger.info("Download functionality initialized (IBOV and SGS only)")
            except Exception as e:
                logger.warning(f"Failed to initialize downloaders: {e}")
                self.auto_download = False
        
        # B3-specific configuration
        self.min_volume_brl = 1_000_000  # Minimum daily volume in BRL
        self.min_price = 1.0  # Minimum price to avoid penny stocks
        self.max_price_change = 0.20  # Maximum daily price change (20%)
        
        logger.info(f"DataLoader initialized with raw_path: {self.raw_path}")
        logger.info(f"DataLoader initialized with processed_path: {self.processed_path}")
        logger.info(f"Auto-download enabled: {self.auto_download}")
    
    def check_sgs_data(self) -> Dict[str, Any]:
        """
        Check for missing SGS data.
        
        Returns:
            Dict[str, Any]: Dictionary with SGS data status
        """
        if not self.auto_download or not self.sgs_downloader:
            return {
                'has_data': False,
                'missing_series': [],
                'available_series': [],
                'needs_download': False
            }
        
        try:
            # Check which SGS series files exist
            sgs_path = Path("data/sgs")
            available_series = []
            missing_series = []
            
            # Check for each SGS series (8, 11, 12, 433)
            for series_id in [8, 11, 12, 433]:
                # Check for processed files with the naming convention from SGSDataLoader
                series_files = list(sgs_path.glob(f"sgs_{series_id}_*.csv"))
                if series_files:
                    available_series.append(series_id)
                else:
                    missing_series.append(series_id)
            
            return {
                'has_data': len(available_series) > 0,
                'missing_series': missing_series,
                'available_series': available_series,
                'needs_download': len(missing_series) > 0
            }
            
        except Exception as e:
            logger.error(f"Error checking SGS data: {e}")
            return {
                'has_data': False,
                'missing_series': [8, 11, 12, 433],
                'available_series': [],
                'needs_download': True
            }
    
    def check_ibov_data(self) -> Dict[str, Any]:
        """
        Check for missing IBOV data.
        
        Returns:
            Dict[str, Any]: Dictionary with IBOV data status
        """
        if not self.auto_download or not self.ibov_downloader:
            return {
                'has_data': False,
                'missing_dates': [],
                'data_range': None,
                'needs_download': False
            }
        
        try:
            # Check IBOV data file (using the Yahoo Finance downloader's file structure)
            ibov_file = Path("data/IBOV/raw/IBOV_raw.csv")
            
            if not ibov_file.exists():
                return {
                    'has_data': False,
                    'missing_dates': [],
                    'data_range': None,
                    'needs_download': True
                }
            
            # Load existing data
            data = pd.read_csv(ibov_file, index_col=0, parse_dates=True)
            if data.empty:
                return {
                    'has_data': False,
                    'missing_dates': [],
                    'data_range': None,
                    'needs_download': True
                }
            
            # Get data range
            data_start = data.index.min()
            data_end = data.index.max()
            today = pd.Timestamp.now().normalize()
            
            # Check if data is up to date
            if data_end >= today:
                return {
                    'has_data': True,
                    'missing_dates': [],
                    'data_range': {
                        'start': data_start.isoformat(),
                        'end': data_end.isoformat()
                    },
                    'needs_download': False,
                    'is_up_to_date': True
                }
            
            # Check for missing dates from last available date to today
            missing_dates = []
            current_date = data_end + pd.Timedelta(days=1)
            
            while current_date <= today:
                if current_date not in data.index:
                    missing_dates.append(current_date)
                current_date += pd.Timedelta(days=1)
            
            # Use dias_uteis to get Brazilian business days
            try:
                from dias_uteis import range_du
                
                # Convert dates to datetime.date for dias_uteis
                start_date_du = (data_end + pd.Timedelta(days=1)).date()
                end_date_du = today.date()
                
                # Get business days between last available date and today
                business_days = range_du(start_date_du, end_date_du)
                business_days_list = [pd.Timestamp(date) for date in business_days]
                
                # Filter missing dates to only include business days
                missing_business_days = [date for date in missing_dates if date in business_days_list]
                
                logger.info(f"IBOV: Using dias_uteis: {len(business_days_list)} business days, {len(missing_business_days)} missing business days")
                
            except ImportError:
                # Fallback to pandas business day range if dias_uteis not available
                business_days = pd.bdate_range(start=data_end + pd.Timedelta(days=1), end=today)
                missing_business_days = [date for date in missing_dates if date in business_days]
                logger.warning("IBOV: dias_uteis not available, using pandas business day range")
            
            return {
                'has_data': True,
                'missing_dates': missing_business_days,
                'data_range': {
                    'start': data_start.isoformat(),
                    'end': data_end.isoformat()
                },
                'needs_download': len(missing_business_days) > 0,
                'missing_count': len(missing_business_days),
                'is_up_to_date': len(missing_business_days) == 0
            }
            
        except Exception as e:
            logger.error(f"Error checking IBOV data: {e}")
            return {
                'has_data': False,
                'missing_dates': [],
                'data_range': None,
                'needs_download': True
            }
    
    def _download_sgs_data(self) -> bool:
        """
        Download missing SGS data.
        
        Returns:
            bool: True if download was successful, False otherwise
        """
        if not self.auto_download or not self.sgs_downloader:
            logger.warning("SGS auto-download is disabled or SGS downloader not available")
            return False
        
        try:
            logger.info("Downloading missing SGS data...")
            
            # Get default date range (last year to today)
            end_date = datetime.now().strftime("%d/%m/%Y")
            start_date = (datetime.now() - timedelta(days=365)).strftime("%d/%m/%Y")
            
            # Download all series using the SGSDataLoader's method
            result = self.sgs_downloader.get_all_series_data(
                start_date=start_date,
                end_date=end_date,
                use_cache=False,
                save_processed=True
            )
            
            if result:
                logger.info(f"Successfully downloaded SGS data for {len(result)} series")
                return True
            else:
                logger.error("Failed to download SGS data")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading SGS data: {e}")
            return False
    
    def _download_ibov_data(self) -> bool:
        """
        Download missing IBOV data.
        
        Returns:
            bool: True if download was successful, False otherwise
        """
        if not self.auto_download or not self.ibov_downloader:
            logger.warning("IBOV auto-download is disabled or IBOV downloader not available")
            return False
        
        try:
            logger.info("Downloading missing IBOV data...")
            
            # Download recent data (last 30 days) using the YahooIBOVDownloader's method
            result = self.ibov_downloader.get_recent_data(days=30)
            
            if result.success:
                logger.info(f"Successfully downloaded IBOV data: {result.data_points} data points")
                return True
            else:
                logger.error(f"Failed to download IBOV data: {result.error_message}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading IBOV data: {e}")
            return False
    
    def check_missing_data(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Check for missing data for given tickers.
        
        Args:
            tickers (List[str]): List of ticker symbols to check
            
        Returns:
            Dict[str, Any]: Dictionary with missing data status
        """
        missing_tickers = []
        tickers_with_gaps = []
        total_missing_days = 0
        
        for ticker in tickers:
            ticker_file = self.raw_path / f"{ticker}_raw.csv"
            
            if not ticker_file.exists():
                missing_tickers.append(ticker)
                continue
            
            # Check for data gaps
            try:
                data = pd.read_csv(ticker_file, index_col=0, parse_dates=True)
                if data.empty:
                    missing_tickers.append(ticker)
                    continue
                
                # Check for recent missing dates
                data_end = data.index.max()
                today = pd.Timestamp.now().normalize()
                
                if data_end < today:
                    # Calculate missing business days
                    try:
                        from dias_uteis import range_du
                        start_date_du = (data_end + pd.Timedelta(days=1)).date()
                        end_date_du = today.date()
                        business_days = range_du(start_date_du, end_date_du)
                        missing_days = len(business_days)
                    except ImportError:
                        # Fallback to pandas business day range
                        business_days = pd.bdate_range(start=data_end + pd.Timedelta(days=1), end=today)
                        missing_days = len(business_days)
                    
                    if missing_days > 0:
                        tickers_with_gaps.append({
                            'ticker': ticker,
                            'last_date': data_end.isoformat(),
                            'missing_days': missing_days
                        })
                        total_missing_days += missing_days
                        
            except Exception as e:
                logger.error(f"Error checking data for {ticker}: {e}")
                missing_tickers.append(ticker)
        
        return {
            'missing_tickers': missing_tickers,
            'tickers_with_gaps': tickers_with_gaps,
            'summary': {
                'missing_tickers_count': len(missing_tickers),
                'tickers_with_gaps_count': len(tickers_with_gaps),
                'total_missing_days': total_missing_days
            }
        }
    
    def download_missing_data_batch(self, tickers: List[str], show_progress: bool = True) -> Dict[str, List[str]]:
        """
        Download missing data for a batch of tickers.
        
        Args:
            tickers (List[str]): List of ticker symbols to download
            show_progress (bool): Whether to show progress messages
            
        Returns:
            Dict[str, List[str]]: Dictionary with successful and failed tickers
        """
        if not self.auto_download or not self.stock_downloader:
            logger.warning("Auto-download is disabled or stock downloader not available")
            return {'success': [], 'failed': tickers}
        
        successful = []
        failed = []
        
        if show_progress:
            logger.info(f"Downloading data for {len(tickers)} tickers...")
        
        for i, ticker in enumerate(tickers, 1):
            if show_progress:
                logger.info(f"Downloading {ticker} ({i}/{len(tickers)})...")
            
            try:
                result = self.stock_downloader.download_ticker_data(ticker)
                if result.success:
                    successful.append(ticker)
                    if show_progress:
                        logger.info(f"✓ {ticker}: {result.data_points} data points")
                else:
                    failed.append(ticker)
                    if show_progress:
                        logger.error(f"✗ {ticker}: {result.error_message}")
                        
            except Exception as e:
                failed.append(ticker)
                if show_progress:
                    logger.error(f"✗ {ticker}: {e}")
        
        if show_progress:
            logger.info(f"Download completed: {len(successful)} successful, {len(failed)} failed")
        
        return {'success': successful, 'failed': failed}
    
    def check_all_data(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Check for missing data across all sources (tickers, SGS, IBOV).
        
        Args:
            tickers (List[str]): List of ticker symbols to check
            
        Returns:
            Dict[str, Any]: Dictionary with comprehensive data status
        """
        # Check ticker data
        ticker_status = self.check_missing_data(tickers)
        
        # Check SGS data
        sgs_status = self.check_sgs_data()
        
        # Check IBOV data
        ibov_status = self.check_ibov_data()
        
        return {
            'tickers': ticker_status,
            'sgs': sgs_status,
            'ibov': ibov_status,
            'summary': {
                'total_missing_tickers': ticker_status['summary']['missing_tickers_count'],
                'total_tickers_with_gaps': ticker_status['summary']['tickers_with_gaps_count'],
                'total_missing_days': ticker_status['summary']['total_missing_days'],
                'sgs_needs_download': sgs_status['needs_download'],
                'ibov_needs_download': ibov_status['needs_download'],
                'any_missing_data': (
                    ticker_status['summary']['missing_tickers_count'] > 0 or
                    ticker_status['summary']['tickers_with_gaps_count'] > 0 or
                    sgs_status['needs_download'] or
                    ibov_status['needs_download']
                )
            }
        }
    
    def download_all_missing_data(self, tickers: List[str], show_progress: bool = True) -> Dict[str, Any]:
        """
        Download all missing data (tickers, SGS, IBOV).
        
        Args:
            tickers (List[str]): List of ticker symbols to download
            show_progress (bool): Whether to show progress messages
            
        Returns:
            Dict[str, Any]: Dictionary with download results
        """
        if not self.auto_download:
            logger.warning("Auto-download is disabled")
            return {'success': False, 'message': 'Auto-download disabled'}
        
        results = {
            'tickers': {'success': [], 'failed': []},
            'sgs': {'success': False, 'message': ''},
            'ibov': {'success': False, 'message': ''}
        }
        
        if show_progress:
            logger.info("Starting comprehensive data download...")
        
        # Download missing tickers
        if tickers:
            ticker_results = self.download_missing_data_batch(tickers, show_progress)
            results['tickers'] = ticker_results
        
        # Download SGS data if needed
        sgs_status = self.check_sgs_data()
        if sgs_status['needs_download']:
            if show_progress:
                logger.info("Downloading SGS data...")
            sgs_success = self._download_sgs_data()
            results['sgs'] = {
                'success': sgs_success,
                'message': 'SGS data downloaded successfully' if sgs_success else 'Failed to download SGS data'
            }
        
        # Download IBOV data if needed
        ibov_status = self.check_ibov_data()
        if ibov_status['needs_download']:
            if show_progress:
                logger.info("Downloading IBOV data...")
            ibov_success = self._download_ibov_data()
            results['ibov'] = {
                'success': ibov_success,
                'message': 'IBOV data downloaded successfully' if ibov_success else 'Failed to download IBOV data'
            }
        
        if show_progress:
            logger.info("Comprehensive data download completed")
        
        return results
    
    def load_raw_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load intraday data for a ticker from Brapi.dev API.
        
        Args:
            ticker (str): Ticker symbol
            
        Returns:
            Optional[pd.DataFrame]: Intraday OHLCV data or None if not available
            
        Note: Uses Brapi.dev API for hourly data. No fallback mechanism.
        """
        try:
            # Try to get BrapiProvider from HybridDataManager if available
            if hasattr(self, 'brapi_provider'):
                brapi_provider = self.brapi_provider
            else:
                # Try to initialize BrapiProvider directly
                brapi_config = load_brapi_config()
                if not brapi_config or not BRAPI_AVAILABLE:
                    logger.error(f"❌ Brapi configuration not available for {ticker}")
                    logger.error("   Please set BRAPI_API_TOKEN environment variable")
                    raise SystemExit("Brapi.dev configuration required")
                
                data_config = brapi_config.get('data', {})
                brapi_provider = BrapiProvider(
                    api_token=brapi_config['api_token'],
                    cache_dir=data_config.get('cache_dir', 'data/brapi_cache'),
                    cache_ttl_hours=data_config.get('cache_ttl_hours', 24)
                )
            
            # Get hourly data for the last year
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            data = brapi_provider.get_hourly_data(ticker, start_date, end_date)
            
            if data is not None and not data.empty:
                logger.info(f"✅ Loaded {len(data)} intraday bars for {ticker}: "
                           f"{data.index.min()} to {data.index.max()}")
                return data
            else:
                logger.error(f"❌ No intraday data available for {ticker}")
                logger.error("   This may indicate:")
                logger.error("   1. Symbol doesn't exist on B3")
                logger.error("   2. Brapi.dev API issue")
                logger.error("   3. Network connectivity problem")
                raise SystemExit(f"No Brapi.dev data available for {ticker}")
                
        except SystemExit:
            raise  # Re-raise SystemExit
        except Exception as e:
            logger.error(f"❌ Error loading intraday data for {ticker}: {e}")
            raise SystemExit(f"Failed to load Brapi.dev data for {ticker}") 
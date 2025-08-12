"""
Brapi.dev Pro API provider for Brazilian market data.

This module provides functionality to:
- Fetch daily and hourly OHLCV data from Brapi.dev API
- Implement intelligent caching with TTL-based invalidation
- Handle Brazilian market symbols and data validation
- Provide rate limiting and retry logic
- Maintain compatibility with existing data loading interfaces

Author: Migration from yfinance to Brapi.dev
Date: 2025
"""

import pandas as pd
import numpy as np
import requests
import time
import json
import logging
import pytz
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta
import os
try:
    import fcntl  # POSIX file locking for thread/process safety
    _HAVE_FCNTL = True
except Exception:
    _HAVE_FCNTL = False
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class BrapiCacheMetadata:
    """Metadata for cached Brapi.dev data."""
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
class BrapiCacheStats:
    """Statistics for Brapi.dev cache performance."""
    hits: int = 0
    fetches: int = 0
    load_time: float = 0.0
    
    @property 
    def hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        total = self.hits + self.fetches
        return self.hits / total if total > 0 else 0.0


class BrapiProvider:
    """
    Intelligent caching provider for Brapi.dev daily and hourly data with smart fetching.
    
    Features:
    - Smart caching of daily and hourly data with TTL-based invalidation
    - Range-based fetching to minimize API calls
    - Intelligent cache merging and gap filling
    - Performance monitoring and statistics
    - Robust error handling and retries
    - Optimized for Brazilian market technical indicators
    - Rate limiting and API quota management
    """
    
    def __init__(self, api_token: str, cache_dir: str = "data/brapi_cache", 
                 cache_ttl_hours: int = 24, timeout: int = 30, max_retries: int = 3):
        """
        Initialize the smart Brapi.dev provider.
        
        Args:
            api_token: Brapi.dev API token
            cache_dir: Directory for cache storage
            cache_ttl_hours: Cache time-to-live in hours
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.api_token = api_token
        self.base_url = "https://brapi.dev/api"
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Cache configuration
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_hours = cache_ttl_hours
        self.stats = BrapiCacheStats()
        
        # Initialize cache subdirectories
        (self.cache_dir / "daily").mkdir(exist_ok=True)
        (self.cache_dir / "hourly").mkdir(exist_ok=True)
        
        # Rate limiting and persistent HTTP session (reduces TCP/TLS overhead)
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        self._session = requests.Session()
        # Persistent session with gzip and keep-alive for throughput
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_token}",
            "Accept-Encoding": "gzip",
            "Connection": "keep-alive",
            "User-Agent": "quant_b3_backtest/1.0 (BrapiProvider)"
        })
        
        # Load cache statistics
        self._load_cache_stats()
        
        logger.debug(f"BrapiProvider initialized: cache_dir={cache_dir}, ttl={cache_ttl_hours}h")

        # Batch settings based on observed API behavior and docs
        # Most stable results with up to 10 symbols per request
        self.batch_chunk_size: int = 10
        # Conservative per-request pacing (API docs mention ~5 rps typical)
        self.min_request_interval = max(self.min_request_interval, 0.2)

        # API intrinsic limits observed in practice
        self.max_hourly_lookback_days: int = 92  # ~3 months
        self.max_daily_lookback_days: int = 1825  # 5 years
    
    def get_daily_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get daily data with smart caching and intelligent fetching.
        
        Args:
            symbol: Trading symbol (e.g., 'ALPA4')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with daily OHLCV data
        """
        return self._get_data(symbol, start_date, end_date, 'daily')
    
    def get_hourly_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get hourly data with smart caching and intelligent fetching.
        
        Args:
            symbol: Trading symbol (e.g., 'ALPA4')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with hourly OHLCV data
        """
        return self._get_data(symbol, start_date, end_date, 'hourly')

    # ==========================
    # Batch quote/validation API
    # ==========================
    def fetch_quotes_batch(self, symbols: list, interval: str = '1d', range_param: str = '1mo') -> list:
        """Fetch quotes for a batch of symbols using /quote with interval/range.

        Returns a list of result objects from the API (preserves schema),
        empty list on failure.
        """
        if not symbols:
            return []
        url_base = f"{self.base_url}/quote/"
        headers = {"Authorization": f"Bearer {self.api_key}"} if hasattr(self, 'api_key') else {}
        # Use persistent session headers that already include Authorization
        results_agg: list = []
        for i in range(0, len(symbols), self.batch_chunk_size):
            chunk = symbols[i:i + self.batch_chunk_size]
            url = url_base + ",".join(chunk)
            params = {"range": range_param, "interval": interval}
            for attempt in range(self.max_retries):
                try:
                    self._rate_limit()
                    resp = self._session.get(url, params=params, timeout=self.timeout)
                    # Handle non-200s with backoff
                    if resp.status_code == 429:
                        retry_after = int(resp.headers.get('Retry-After', 1))
                        time.sleep(max(retry_after, 1))
                        continue
                    if resp.status_code != 200:
                        if attempt < self.max_retries - 1:
                            time.sleep(2 ** attempt)
                            continue
                        else:
                            logger.warning(f"/quote batch error {resp.status_code} for {chunk}")
                            break
                    try:
                        data = resp.json() or {}
                    except Exception:
                        if attempt < self.max_retries - 1:
                            time.sleep(2 ** attempt)
                            continue
                        else:
                            logger.warning(f"/quote batch JSON decode failed for {chunk}")
                            break
                    results = data.get('results', []) or []
                    results_agg.extend(results)
                    break
                except requests.exceptions.RequestException as e:
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        logger.warning(f"/quote batch request error for {chunk}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error in fetch_quotes_batch for {chunk}: {e}")
                    break
        return results_agg

    def validate_symbols_from_csv(self, csv_path: str, interval: str = '1h', range_param: str = '3mo') -> dict:
        """Validate symbols from CSV against BRAPI /quote endpoint.

        Returns dict with keys: valid_symbols, invalid_symbols, details (per-symbol object when available).
        """
        try:
            df = pd.read_csv(csv_path)
            symbols_all = [str(s).strip().upper() for s in df['symbol'].dropna().tolist() if str(s).strip()]
        except Exception as e:
            logger.error(f"Failed to read portfolio CSV {csv_path}: {e}")
            return {"valid_symbols": [], "invalid_symbols": [], "details": {}}

        results = self.fetch_quotes_batch(symbols_all, interval=interval, range_param=range_param)
        # Map symbol -> result
        details = {}
        for item in results:
            sym = str(item.get('symbol', '')).upper()
            if sym:
                details[sym] = item
        valid = [s for s in symbols_all if s in details]
        invalid = [s for s in symbols_all if s not in details]
        if invalid:
            logger.warning(f"Symbols not returned by BRAPI /quote: {','.join(invalid[:20])}{'...' if len(invalid)>20 else ''}")
        return {"valid_symbols": valid, "invalid_symbols": invalid, "details": details}

    # ==========================
    # Public OHLC data retrieval
    # ==========================
    def fetch_ohlc(self, symbol: str, interval: str, lookback_days: int, end_date: str = None) -> pd.DataFrame:
        """
        Fetch OHLC data for a symbol respecting BRAPI limits and preserving schema columns
        (open, high, low, close, volume). Index coerced from epoch seconds as UTC.

        Args:
            symbol: Ticker symbol (e.g., 'PETR4')
            interval: '1d' for daily or '1h' for hourly
            lookback_days: Desired lookback in days
            end_date: Optional end date (YYYY-MM-DD). Defaults to today.

        Returns:
            DataFrame with OHLCV, sorted by index, UTC-based timestamps (tz-naive but in UTC).
        """
        if interval not in ('1d', '1h'):
            raise ValueError("interval must be '1d' or '1h'")
        if end_date is None:
            end_date = datetime.utcnow().strftime('%Y-%m-%d')

        # Enforce API limits
        if interval == '1h' and lookback_days > self.max_hourly_lookback_days:
            logger.warning(
                "Requested hourly lookback %sd exceeds BRAPI limit ~%sd. Clamping.",
                lookback_days, self.max_hourly_lookback_days
            )
            lookback_days = self.max_hourly_lookback_days
        if interval == '1d' and lookback_days > self.max_daily_lookback_days:
            lookback_days = self.max_daily_lookback_days

        start_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=lookback_days)
        start_date = start_dt.strftime('%Y-%m-%d')

        # For BRAPI /quote, data is controlled by (range, interval). We request a single window
        # within allowed lookback. Additional historical hourly beyond ~3 months is not available.
        data_type = 'daily' if interval == '1d' else 'hourly'
        df = self._get_data(symbol, start_date, end_date, data_type)
        if df is None or df.empty:
            return pd.DataFrame()

        # Coerce index to UTC (tz-naive, consistent with pipeline, but based on UTC)
        try:
            if getattr(df.index, 'tz', None) is None:
                # Assume timestamps already represent UTC seconds
                df.index = pd.to_datetime(df.index)
        except Exception:
            pass

        df = df.sort_index()

        # Coverage audit
        expected_start = start_dt.date()
        actual_start = df.index.min().date() if len(df) else None
        if actual_start and actual_start > expected_start:
            logger.warning(
                "Coverage gap for %s %s: expected start %s, actual %s (missing ~%d days)",
                symbol, interval, expected_start, actual_start,
                (actual_start - expected_start).days
            )

        return df

    def fetch_batch(self, symbols: list, interval: str = '1d', range_param: str = '1mo') -> list:
        """Optimized bulk request wrapper preserving BRAPI schema."""
        return self.fetch_quotes_batch(symbols, interval=interval, range_param=range_param)

    # ==========================
    # Local Parquet completeness helper
    # ==========================
    def _is_data_complete(self, symbol: str, interval: str, start_date: datetime, end_date: datetime) -> bool:
        data_type = 'daily' if interval == '1d' else 'hourly'
        cache_file, metadata = self._get_cache_info(symbol, data_type)
        if metadata is None or not cache_file.exists():
            return False
        # Metadata quick pass
        try:
            meta_start = datetime.strptime(metadata.start_date, '%Y-%m-%d')
            meta_end = datetime.strptime(metadata.end_date, '%Y-%m-%d')
            if meta_start <= start_date and meta_end >= end_date:
                return True
        except Exception:
            pass
        # Fallback: precise
        df = self._load_from_cache(cache_file)
        if df is None or df.empty:
            return False
        idx_min = pd.to_datetime(df.index.min())
        idx_max = pd.to_datetime(df.index.max())
        return idx_min <= start_date and idx_max >= end_date

    # ==========================
    # Backtest integration API
    # ==========================
    def get_ohlc(self, symbol: str, interval: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Returns stitched + validated OHLC for backtest engine:
          1) Check local coverage
          2) Fetch only missing, chunked for hourly
          3) Persist
          4) Return requested window
        """
        if interval not in ('1d', '1h'):
            raise ValueError("interval must be '1d' or '1h'")
        data_type = 'daily' if interval == '1d' else 'hourly'

        # Minimum hourly lookback
        if interval == '1h':
            min_lb = timedelta(days=self.max_hourly_lookback_days)
            if (end_date - start_date) < min_lb:
                start_date = end_date - min_lb

        # Full coverage from cache
        if self._is_data_complete(symbol, interval, start_date, end_date):
            cache_file, _ = self._get_cache_info(symbol, data_type)
            df_local = self._load_from_cache(cache_file)
            if df_local is None:
                df_local = pd.DataFrame()
            m = (df_local.index >= pd.to_datetime(start_date)) & (df_local.index <= pd.to_datetime(end_date))
            return df_local.loc[m].sort_index()

        # Compute missing ranges
        cache_file, _ = self._get_cache_info(symbol, data_type)
        existing = self._load_from_cache(cache_file) if cache_file.exists() else None
        have_min = pd.to_datetime(existing.index.min()) if (existing is not None and not existing.empty) else None
        have_max = pd.to_datetime(existing.index.max()) if (existing is not None and not existing.empty) else None
        req_start = pd.to_datetime(start_date)
        req_end = pd.to_datetime(end_date)

        missing: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        if have_min is None or have_max is None:
            missing.append((req_start, req_end))
        else:
            if req_start < have_min:
                missing.append((req_start, min(have_min, req_end)))
            if req_end > have_max:
                missing.append((max(req_start, have_max), req_end))

        # Detect internal gaps for daily data and coalesce to one-shot fetch across the requested window
        if data_type == 'daily' and existing is not None and not existing.empty:
            try:
                # Normalize to date for gap analysis
                present_days = pd.to_datetime(existing.index).normalize().unique()
                present_days = pd.DatetimeIndex(sorted(present_days))
                # Business days between req_start and req_end (approx; holidays ignored)
                expected_days = pd.bdate_range(start=req_start.normalize(), end=req_end.normalize())
                # Identify missing days within requested window
                missing_day_index = expected_days.difference(present_days)
                if len(missing_day_index) > 0:
                    # One-shot backfill over the entire requested window for robustness/performance
                    missing = [(req_start, req_end)]
            except Exception:
                # If anything fails in gap detection, fall back to boundary-based fetching
                pass

        parts: List[pd.DataFrame] = []
        if existing is not None and not existing.empty:
            parts.append(existing)

        for a, b in missing:
            if a >= b:
                continue
            if interval == '1h':
                cur = a
                while cur < b:
                    chunk_end = min(cur + timedelta(days=self.max_hourly_lookback_days), b)
                    dfp = self._get_data(symbol, cur.strftime('%Y-%m-%d'), chunk_end.strftime('%Y-%m-%d'), data_type)
                    if dfp is not None and not dfp.empty:
                        parts.append(dfp)
                    cur = chunk_end
            else:
                dfp = self._get_data(symbol, a.strftime('%Y-%m-%d'), b.strftime('%Y-%m-%d'), data_type)
                if dfp is not None and not dfp.empty:
                    parts.append(dfp)

        if not parts:
            return pd.DataFrame()

        merged = pd.concat(parts, axis=0, ignore_index=False)
        merged = merged.sort_index()
        merged = merged[~merged.index.duplicated(keep='last')]

        # Persist aggregate coverage
        s = merged.index.min().strftime('%Y-%m-%d')
        e = merged.index.max().strftime('%Y-%m-%d')
        self._save_to_cache(merged, symbol, data_type, s, e)

        mask = (merged.index >= req_start) & (merged.index <= req_end)
        return merged.loc[mask]
    
    def _get_data(self, symbol: str, start_date: str, end_date: str, data_type: str) -> pd.DataFrame:
        """
        Generic method to get data with smart caching.
        
        Args:
            symbol: Trading symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            data_type: 'daily' or 'hourly'
            
        Returns:
            DataFrame with OHLCV data
        """
        start_time = time.time()
        
        try:
            # Check cache coverage and freshness
            cache_file, metadata = self._get_cache_info(symbol, data_type)
            
            if cache_file.exists() and metadata and not metadata.is_stale():
                if metadata.covers_range(start_date, end_date):
                    # Perfect cache hit
                    data = self._load_from_cache(cache_file)
                    if data is not None:
                        filtered_data = self._filter_date_range(data, start_date, end_date)
                        self.stats.hits += 1
                        self.stats.load_time += time.time() - start_time
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug("Cache hit for %s %s data: %d bars", symbol, data_type, len(filtered_data))
                        return filtered_data
            
            # Calculate what ranges need fetching
            fetch_ranges = self._calculate_fetch_ranges(
                symbol, start_date, end_date, data_type, metadata
            )
            
            if not fetch_ranges:
                # Use existing cache even if slightly stale
                if cache_file.exists():
                    data = self._load_from_cache(cache_file)
                    if data is not None:
                        filtered_data = self._filter_date_range(data, start_date, end_date)
                        self.stats.hits += 1
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug("Stale cache used for %s: %d bars", symbol, len(filtered_data))
                        return filtered_data
            
            # Fetch missing data
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Fetching %s data for %s: %d range(s)", data_type, symbol, len(fetch_ranges))
            new_data_parts = []
            
            for fetch_start, fetch_end in fetch_ranges:
                part_data = self._fetch_brapi_data(symbol, fetch_start, fetch_end, data_type)
                if part_data is not None and not part_data.empty:
                    new_data_parts.append(part_data)
                self._rate_limit()
            
            # Merge with existing cache
            final_data = self._merge_with_cache(cache_file, new_data_parts, symbol, data_type)
            
            # Filter to requested range
            result_data = self._filter_date_range(final_data, start_date, end_date)
            
            self.stats.fetches += 1
            self.stats.load_time += time.time() - start_time
            logger.debug(f"Smart fetch completed for {symbol}: {len(result_data)} bars")
            
            return result_data
            
        except Exception as e:
            logger.error(f"Error in _get_data for {symbol} ({data_type}): {e}")
            self.stats.load_time += time.time() - start_time
            return pd.DataFrame()
    
    def _fetch_brapi_data(self, symbol: str, start_date: str, end_date: str, 
                         data_type: str) -> Optional[pd.DataFrame]:
        """
        Fetch data from Brapi.dev API with retry logic.
        
        Args:
            symbol: Trading symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            data_type: 'daily' or 'hourly'
            
        Returns:
            DataFrame with OHLCV data or None if error
        """
        # Calculate range parameter for Brapi API
        range_param = self._calculate_range_parameter(start_date, end_date)
        # Clamp hourly range to provider limits (hourly typically supports up to ~3 months)
        if data_type == 'hourly' and range_param in ('6mo', '1y', '2y', '5y', 'max'):
            range_param = '3mo'
        interval_param = '1d' if data_type == 'daily' else '1h'
        
        url = f"{self.base_url}/quote/{symbol}"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        params = {
            "range": range_param,
            "interval": interval_param
        }
        
        for attempt in range(self.max_retries):
            try:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Fetching %s data for %s: %s", data_type, symbol, range_param)
                # Reuse persistent session for performance
                response = self._session.get(url, headers=headers, params=params, timeout=self.timeout)
                
                if response.status_code == 200:
                    # Some responses may be empty or not JSON (transient API/proxy issues)
                    try:
                        data = response.json()
                    except Exception:
                        if attempt < self.max_retries - 1:
                            time.sleep(2 ** attempt)
                            continue
                        else:
                            return None
                    
                    if 'results' not in data or not data['results']:
                        logger.warning(f"No results in Brapi response for {symbol}")
                        return None
                    
                    result = data['results'][0]
                    
                    if 'historicalDataPrice' not in result:
                        logger.warning(f"No historical data in Brapi response for {symbol}")
                        return None
                    
                    # Convert to DataFrame
                    df_data = []
                    for entry in result['historicalDataPrice']:
                        timestamp = entry.get('date', 0)
                        if timestamp:
                            dt = datetime.fromtimestamp(timestamp)
                            df_data.append({
                                'open': entry.get('open', 0),
                                'high': entry.get('high', 0),
                                'low': entry.get('low', 0),
                                'close': entry.get('close', 0),
                                'volume': entry.get('volume', 0)
                            })
                    
                    if not df_data:
                        logger.warning(f"No valid data points for {symbol}")
                        return None
                    
                    # Create DataFrame
                    df = pd.DataFrame(df_data)
                    if not df.empty:
                        # Create proper datetime index from timestamps
                        timestamps = [entry.get('date', 0) for entry in result['historicalDataPrice'] if entry.get('date')]
                        if timestamps:
                            # Validate timestamps - filter out invalid ones (too small or too large)
                            valid_timestamps = []
                            for ts in timestamps:
                                # Skip timestamps that are clearly invalid (before 1990 or after 2030)
                                if ts > 631152000 and ts < 1893456000:  # 1990-01-01 to 2030-01-01
                                    valid_timestamps.append(ts)
                                else:
                                    logger.warning(f"Invalid timestamp detected: {ts} for {symbol}, skipping")

                            if valid_timestamps:
                                # Convert to tz-naive UTC timestamps (canonical throughout the engine)
                                df.index = pd.to_datetime(valid_timestamps, unit='s')

                                # Filter DataFrame to only include rows with valid timestamps
                                df = df.iloc[:len(valid_timestamps)]

                                if logger.isEnabledFor(logging.DEBUG):
                                    logger.debug("Processed %d timestamps for %s", len(valid_timestamps), symbol)
                                    logger.debug("Sample timestamp: %s (hour: %d)", df.index[0], df.index[0].hour)
                            else:
                                logger.error(f"No valid timestamps found for {symbol}")
                                return None
                    
                    # Clean and validate data
                    df = self._clean_brapi_data(df)

                    # Enforce B3 session hours in UTC for hourly data
                    # BRAPI delivers UTC; B3 continuous session maps to 13:00-20:00 UTC (10:00-17:00 BRT)
                    if data_type == 'hourly':
                        try:
                            mask_session = (df.index.hour >= 13) & (df.index.hour <= 20)
                            df = df.loc[mask_session]
                        except Exception:
                            # If index is not datetime for any reason, leave as-is
                            pass
                    
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Fetched %d %s bars for %s", len(df), data_type, symbol)
                    return df
                    
                elif response.status_code == 429:
                    # Rate limit hit
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limit hit for {symbol}, waiting {retry_after}s")
                    time.sleep(retry_after)
                    continue
                    
                else:
                    logger.warning(f"Brapi API error for {symbol}: {response.status_code} - {response.text}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        return None
                        
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error for {symbol} (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return None
            except Exception as e:
                logger.error(f"Unexpected error fetching {symbol}: {e}")
                return None
        
        return None
    
    def _calculate_range_parameter(self, start_date: str, end_date: str) -> str:
        """
        Calculate the range parameter for Brapi API based on date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Range parameter string for Brapi API
        """
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        days_diff = (end_dt - start_dt).days
        
        # Map to Brapi range parameters
        if days_diff <= 1:
            return "1d"
        elif days_diff <= 5:
            return "5d"
        elif days_diff <= 30:
            return "1mo"
        elif days_diff <= 90:
            return "3mo"
        elif days_diff <= 180:
            return "6mo"
        elif days_diff <= 365:
            return "1y"
        elif days_diff <= 730:
            return "2y"
        elif days_diff <= 1825:
            return "5y"
        else:
            return "max"
    
    def _clean_brapi_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate Brapi.dev data.
        
        Args:
            data: Raw DataFrame from Brapi API
            
        Returns:
            Cleaned DataFrame
        """
        if data.empty:
            return data
        
        # Ensure we operate on a copy to avoid chained assignment warnings
        data = data.copy()
        
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
            data = data[~invalid_mask].copy()
        
        # Fill volume NaNs with 0
        data['volume'] = data['volume'].fillna(0)
        
        # Ensure volume is non-negative
        data['volume'] = data['volume'].clip(lower=0)
        
        return data
    
    def _rate_limit(self):
        """Implement rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _get_cache_info(self, symbol: str, data_type: str) -> Tuple[Path, Optional[BrapiCacheMetadata]]:
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
                    metadata = BrapiCacheMetadata(**meta_dict)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata for {symbol}: {e}")
        
        return cache_file, metadata
    
    def _calculate_fetch_ranges(self, symbol: str, start_date: str, end_date: str, 
                               data_type: str, metadata: Optional[BrapiCacheMetadata]) -> List[Tuple[str, str]]:
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
            
            # Save data with basic file locking and atomic replace
            lock_path = cache_subdir / f"{symbol}_{data_type}.lock"
            lf = None
            try:
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                lf = open(lock_path, 'w')
                if _HAVE_FCNTL:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                tmp_path = cache_file.with_suffix('.parquet.tmp')
                data.to_parquet(tmp_path)
                os.replace(tmp_path, cache_file)
            finally:
                try:
                    if _HAVE_FCNTL and lf is not None:
                        fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
                try:
                    if lf is not None:
                        lf.close()
                except Exception:
                    pass
            
            # Save metadata
            metadata = BrapiCacheMetadata(
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
        hourly_files = len(list((self.cache_dir / "hourly").glob("*.parquet")))
        
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
                'hourly_symbols': hourly_files,
                'total_size_mb': f"{total_size / (1024*1024):.1f}",
                'cache_directory': str(self.cache_dir)
            },
            'settings': {
                'cache_ttl_hours': self.cache_ttl_hours,
                'cache_directory': str(self.cache_dir),
                'api_base_url': self.base_url
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
            logger.debug(f"Cleaned up {removed_count} old cache files")
    
    def cleanup_corrupted_cache(self):
        """Clean up cache files with corrupted timestamp data."""
        try:
            cleaned_count = 0
            
            for cache_type in ['daily', 'hourly']:
                cache_subdir = self.cache_dir / cache_type
                if not cache_subdir.exists():
                    continue
                    
                for cache_file in cache_subdir.glob("*.parquet"):
                    try:
                        # Load and check the data
                        df = pd.read_parquet(cache_file)
                        if not df.empty:
                            # Check if timestamps are corrupted (before 1990)
                            min_timestamp = df.index.min()
                            if min_timestamp.year < 1990:
                                logger.warning(f"Found corrupted cache file: {cache_file.name} with invalid timestamps")
                                cache_file.unlink()
                                # Also remove metadata file
                                metadata_file = cache_file.with_suffix('.json').with_name(cache_file.stem + '_metadata.json')
                                if metadata_file.exists():
                                    metadata_file.unlink()
                                cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"Error checking cache file {cache_file.name}: {e}")
                        # If we can't read the file, it might be corrupted, so remove it
                        cache_file.unlink()
                        metadata_file = cache_file.with_suffix('.json').with_name(cache_file.stem + '_metadata.json')
                        if metadata_file.exists():
                            metadata_file.unlink()
                        cleaned_count += 1
            
            if cleaned_count > 0:
                logger.debug(f"Cleaned up {cleaned_count} corrupted cache files")
            
        except Exception as e:
            logger.error(f"Error cleaning up corrupted cache: {e}") 
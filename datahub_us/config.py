"""Configuration and constants for DataHub US."""

import os
from pathlib import Path
from datetime import date, timedelta

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "cache" / "us"
OHLCV_DIR = CACHE_DIR / "ohlcv"
METADATA_FILE = CACHE_DIR / "metadata.json"
DATA_DIR = Path(__file__).parent / "data"
UNIVERSE_FILE = DATA_DIR / "us_top100.csv"

# Ensure directories exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OHLCV_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Date ranges
BOOTSTRAP_START = date(2005, 1, 1)  # 20 years of data
BOOTSTRAP_END = date.today()

# yfinance settings
YFINANCE_DELAY_BETWEEN_SYMBOLS = 0.3  # seconds
YFINANCE_MAX_RETRIES = 3
YFINANCE_BACKOFF_FACTOR = 2.0
YFINANCE_BATCH_SIZE = 10
YFINANCE_BATCH_PAUSE = 3.0  # seconds

# Alpha Vantage settings
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"
ALPHAVANTAGE_RATE_LIMIT = 5  # requests per minute (free tier)
ALPHAVANTAGE_DAILY_LIMIT = 25  # requests per day (free tier)

# CSV schema (compatible with BR system)
CSV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
CSV_DATE_FORMAT = "%Y-%m-%d"

# QA thresholds
MAX_PRICE_CHANGE_PCT = 50.0  # Flag if daily change > 50%
MIN_VOLUME = 0
MAX_GAP_DAYS = 5  # Trading days gap threshold

# NYSE trading calendar approximation (will use proper calendar in production)
TRADING_DAYS_PER_YEAR = 252

# Logging
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


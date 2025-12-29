"""Configuration for FX data pipeline."""

from pathlib import Path
from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass
class FxConfig:
    """Configuration for FX data operations."""
    
    # Cache directory for FX data
    cache_dir: Path = field(default_factory=lambda: Path("cache/fx"))
    
    # BCB API configuration
    bcb_base_url: str = "https://api.bcb.gov.br/dados/serie/bcdata.sgs"
    bcb_series_usd_brl: int = 1  # PTAX USD/BRL venda
    bcb_series_eur_brl: int = 21619  # EUR/BRL
    
    # FRED API configuration (requires API key for high volume)
    fred_base_url: str = "https://api.stlouisfed.org/fred/series/observations"
    fred_series_eur_usd: str = "DEXUSEU"
    fred_api_key: Optional[str] = None  # Optional, works without for low volume
    
    # Date range defaults
    start_date: date = field(default_factory=lambda: date(2000, 1, 1))
    
    # Request configuration
    request_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # File format
    date_format: str = "%Y-%m-%d"
    
    def __post_init__(self):
        """Ensure cache directory exists."""
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> "FxConfig":
        """Create config from environment variables."""
        import os
        
        config = cls()
        
        if cache_dir := os.environ.get("FX_CACHE_DIR"):
            config.cache_dir = Path(cache_dir)
        
        if fred_key := os.environ.get("FRED_API_KEY"):
            config.fred_api_key = fred_key
        
        return config


# Default configuration instance
DEFAULT_CONFIG = FxConfig()






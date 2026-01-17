"""DataHub US - US Market Data Module for Backtesting.

This module provides a robust, auditable, and idempotent data pipeline
for US equity market data with 20-year historical coverage.

Providers:
    - yfinance (primary): Free, unlimited, 20+ years data
    - Alpha Vantage (fallback): Official API, limited free tier

Features:
    - Bootstrap: Download 20 years of daily OHLCV
    - Update: Incremental daily updates
    - Repair: Gap detection and filling
    - QA: OHLC validation and outlier detection
    - Reconciliation: Multi-source comparison
"""

__version__ = "0.1.0"










































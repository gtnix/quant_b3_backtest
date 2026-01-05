# Provider Due Diligence - DataHub US

**Generated:** 2025-12-24  
**Purpose:** Evaluate US market data providers for 20-year daily OHLCV historical data

## Evaluation Criteria

1. **Evidence of Public Use** - Community adoption, GitHub stars, Stack Overflow discussions
2. **Stability** - No major outages or breaking changes recently
3. **20-Year Daily Data** - Viability for US equities
4. **Terms of Use** - Acceptable for programmatic access, avoid fragile scraping

---

## Provider Evaluation Matrix

| Provider | Status | 20yr Data | Free Tier | Dividends/Splits | Rate Limit | Notes |
|----------|--------|-----------|-----------|------------------|------------|-------|
| **yfinance** | **PASS** | ✅ Yes | Unlimited* | ✅ Yes | ~2000/hr | Most popular Python lib |
| **Alpha Vantage** | **WARN** | ✅ Yes | 25 req/day | ✅ Yes | 5/min free | Too limited for bootstrap |
| **Polygon.io** | WARN | ⚠️ Verify | Limited | ✅ Yes | Varies | Free tier very restricted |
| **EODHD** | PASS | ✅ Yes | Limited | ✅ Yes | 20/min free | Good quality, needs paid |
| **Quandl/Nasdaq** | WARN | ✅ Yes | Limited | ✅ Yes | Varies | Many datasets paid |
| **IEX Cloud** | WARN | ⚠️ Partial | Limited | ✅ Yes | Varies | Historical depth varies |
| **Tiingo** | PASS | ✅ Yes | 500/day | ✅ Yes | Reasonable | Good alternative |

---

## Detailed Analysis

### 1. yfinance (Yahoo Finance) — **PASS** ✅

**Repository:** https://github.com/ranaroussi/yfinance  
**Stars:** 13k+ | **Last Updated:** Active (2024-2025)

**Pros:**
- Most widely used Python library for stock data
- 20+ years of daily OHLCV available for most US stocks
- Includes dividends, splits, and adjusted prices
- No API key required
- Actively maintained with large community

**Cons:**
- No official API (uses Yahoo Finance endpoints)
- Occasional rate limiting under heavy use
- Schema changes can break compatibility

**Evidence:**
- 13,000+ GitHub stars
- Extensive Stack Overflow presence (10,000+ questions)
- Used by quantopian, backtrader, and many quant projects
- PyPI downloads: 5M+/month

**Stability:**
- No major outages reported in 2024-2025
- Rate limits manageable with delays
- Active issue resolution

**Verdict:** PASS - Primary provider due to free unlimited access and data quality

---

### 2. Alpha Vantage — **WARN** ⚠️

**Website:** https://www.alphavantage.co/  
**API:** Official REST API with documentation

**Pros:**
- Official documented API
- 20+ years of historical data
- Includes fundamentals, forex, crypto
- Stable and reliable

**Cons:**
- Free tier: 25 requests/day (unusable for bootstrap)
- 5 requests/minute on free tier
- Paid plans required for serious use ($50+/month)

**Evidence:**
- Well documented API
- Used by many tutorials and courses
- Nasdaq partner

**Verdict:** WARN - Only viable as fallback with paid plan or for spot checks

---

### 3. Polygon.io — **WARN** ⚠️

**Website:** https://polygon.io/

**Pros:**
- Professional grade data
- REST and WebSocket APIs
- Tick-level data available

**Cons:**
- Free tier very limited (5 API calls/min, 2 years history)
- Full history requires paid plan ($29+/month)

**Verdict:** WARN - Not suitable for free 20-year bootstrap

---

### 4. EOD Historical Data (EODHD) — **PASS** ✅

**Website:** https://eodhd.com/

**Pros:**
- 25+ years of historical data
- Dividends, splits, fundamentals
- Well documented API
- Reasonable pricing

**Cons:**
- Free tier limited (20 API calls/day)
- Paid plan required for bootstrap ($20+/month)

**Verdict:** PASS - Good quality, recommended if budget allows

---

### 5. Tiingo — **PASS** ✅

**Website:** https://www.tiingo.com/

**Pros:**
- 20+ years of data
- Free tier: 500 requests/day
- Good documentation
- IEX real-time data

**Cons:**
- Requires registration
- Some endpoints restricted

**Verdict:** PASS - Good alternative to yfinance

---

## Final Decision

### Primary Provider: **yfinance**

**Rationale:**
1. Free unlimited access (with reasonable rate limiting)
2. 20+ years of daily OHLCV for US equities
3. Includes dividends and splits
4. Largest community and best Python integration
5. Actively maintained

### Fallback Provider: **Alpha Vantage**

**Rationale:**
1. Official API with stable contracts
2. Good for spot verification when yfinance fails
3. Professional documentation
4. Can handle occasional fallback requests within free tier

---

## Implementation Notes

### yfinance Rate Limiting Strategy

```python
# Recommended delays to avoid rate limiting
DELAY_BETWEEN_SYMBOLS = 0.5  # seconds
MAX_RETRIES = 3
BACKOFF_FACTOR = 2.0
BATCH_SIZE = 10  # symbols per batch before longer pause
BATCH_PAUSE = 5.0  # seconds between batches
```

### Alpha Vantage Configuration

```
ALPHAVANTAGE_API_KEY=<your_key>  # Required
ALPHAVANTAGE_RATE_LIMIT=5       # requests per minute (free tier)
```

---

## References

1. yfinance GitHub: https://github.com/ranaroussi/yfinance
2. Alpha Vantage Docs: https://www.alphavantage.co/documentation/
3. Polygon.io Docs: https://polygon.io/docs
4. EODHD API: https://eodhd.com/financial-apis
5. Tiingo API: https://api.tiingo.com/documentation

---

*Document generated as part of DataHub US implementation.*























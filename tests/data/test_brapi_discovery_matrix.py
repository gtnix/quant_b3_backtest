import os
import time
import json
import csv
from pathlib import Path
from datetime import datetime, timedelta

import pytest
import requests


BRAPI_BASE = "https://brapi.dev/api"
ART_DIR = Path("tests/artifacts")


INTRADAY_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]
DAILY_INTERVALS = ["1d", "5d", "1wk", "1mo", "3mo"]

# Per docs and practical behavior: intraday windows are short (~3 months)
INTRADAY_MAX_RANGE = "3mo"

RANGES_ORDERED = [
    "max", "10y", "5y", "2y", "1y", "6mo", "3mo", "1mo", "5d", "1d",
]


def _load_token() -> str:
    # Prefer env var
    token = os.environ.get("BRAPI_API_TOKEN", "").strip()
    if token:
        return token
    # Fallback: config/settings.yaml
    try:
        from yaml import safe_load
        cfg = safe_load(Path("config/settings.yaml").read_text()) or {}
        token = ((cfg.get("brapi") or {}).get("api_token") or "").strip()
        return token
    except Exception:
        return ""


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"} if token else {}


def _sleep_for_rate_limit():
    time.sleep(0.3)


def _fetch_available_tickers(token: str) -> list:
    # Try official endpoint for available stocks; fallback to local portfolio_full.csv
    url = f"{BRAPI_BASE}/available"
    try:
        resp = requests.get(url, headers=_headers(token), timeout=20)
        if resp.ok:
            data = resp.json() or {}
            tickers = data.get("stocks") or data.get("results") or []
            if isinstance(tickers, list) and tickers:
                # Normalize to symbols-only list when API returns objects
                if isinstance(tickers[0], dict):
                    syms = [str(t.get("symbol") or t.get("code") or "").strip().upper() for t in tickers]
                    return [s for s in syms if s]
                return [str(s).strip().upper() for s in tickers if str(s).strip()]
    except Exception:
        pass

    # Fallback: local portfolio list
    try:
        import pandas as pd
        p = Path("data/portfolio_full.csv") if Path("data/portfolio_full.csv").exists() else Path("data/portfolio.csv")
        df = pd.read_csv(p)
        col = "symbol" if "symbol" in df.columns else df.columns[0]
        return [str(s).strip().upper() for s in df[col].dropna() if str(s).strip()]
    except Exception:
        return []


def _request_quote(token: str, symbol: str, range_param: str, interval: str) -> dict | None:
    url = f"{BRAPI_BASE}/quote/{symbol}"
    params = {"range": range_param, "interval": interval}
    try:
        _sleep_for_rate_limit()
        resp = requests.get(url, headers=_headers(token), params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json() or {}
        # Treat 401/400/etc. as invalid combo or plan-restricted
        return None
    except Exception:
        return None


def _extract_earliest_date(payload: dict) -> str | None:
    try:
        results = payload.get("results") or []
        if not results:
            return None
        item = results[0]
        series = item.get("historicalDataPrice") or []
        if not series:
            return None
        # BRAPI returns POSIX seconds in 'date'
        ts = min([int(x.get("date", 0)) for x in series if int(x.get("date", 0)) > 0] or [0])
        if ts <= 0:
            return None
        return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
    except Exception:
        return None


def _valid_ranges_for_interval(interval: str) -> list:
    if interval in INTRADAY_INTERVALS:
        # Cap intraday at 3 months per request
        # Search from longer to shorter within allowed intraday set
        return ["3mo", "1mo", "5d", "1d"]
    # For non-intraday, allow full range order
    return RANGES_ORDERED


def _find_max_working_range(token: str, symbol: str, interval: str) -> tuple[str | None, str | None]:
    """Return (max_valid_range, earliest_date) for the given interval.
    Chooses the first working range from longest to shortest.
    """
    for r in _valid_ranges_for_interval(interval):
        payload = _request_quote(token, symbol, r, interval)
        if not payload:
            continue
        d0 = _extract_earliest_date(payload)
        if d0:
            return r, d0
    return None, None


@pytest.mark.stress
def test_brapi_discovery_matrix(tmp_path, request):
    # Optional opt-in to avoid heavy runs by default
    # Optional: disable by default if opt not present
    opt = None
    try:
        opt = request.config.getoption("--brapi-stress", default=False)
    except Exception:
        opt = False
    if not opt and os.environ.get("BRAPI_STRESS", "0") not in ("1", "true", "True"):  # pragma: no cover
        pytest.skip("BRAPI stress not enabled")

    token = _load_token()
    assert token, "BRAPI token not available"

    tickers = _fetch_available_tickers(token)[:]
    assert tickers, "No tickers discovered for probing"

    # Quick mode to limit number of tickers (set BRAPI_QUICK to an int)
    try:
        quick_n = int(request.config.getoption("--limit-symbols"))
    except Exception:
        try:
            quick_n = int(os.environ.get("BRAPI_QUICK", "0"))
        except Exception:
            quick_n = 0
    if quick_n > 0:
        tickers = tickers[:quick_n]

    intervals = INTRADAY_INTERVALS + DAILY_INTERVALS

    # Prepare result matrix
    rows = []
    for sym in tickers:
        row = {"symbol": sym}
        for itv in intervals:
            rng, d0 = _find_max_working_range(token, sym, itv)
            row[f"{itv}_range"] = rng or "N/A"
            row[f"{itv}_earliest"] = d0 or "N/A"
        rows.append(row)

    # Persist artifacts
    ART_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_path = ART_DIR / f"brapi_range_matrix_{ts}.csv"
    json_path = ART_DIR / f"brapi_range_matrix_{ts}.json"
    all_syms_path = ART_DIR / "all_symbols_portfolio.txt"

    # Save all symbols used in this run
    with open(all_syms_path, "w") as f:
        for s in tickers:
            f.write(f"{s}\n")

    # CSV
    fieldnames = ["symbol"] + sum([[f"{itv}_range", f"{itv}_earliest"] for itv in intervals], [])
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # JSON
    with open(json_path, "w") as f:
        json.dump({"intervals": intervals, "rows": rows}, f, indent=2)

    # Basic assertion: at least one cell has data
    assert any(r.get(f"1d_earliest") not in (None, "N/A") for r in rows), "No data discovered"



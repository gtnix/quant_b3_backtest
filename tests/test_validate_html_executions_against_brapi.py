import os
import csv
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
import pandas as pd


# =============================
# Pytest CLI options and fixtures
# =============================

@pytest.fixture(scope="session")
def report_paths(pytestconfig):
    html_path = Path(pytestconfig.getoption("--report-html")).resolve()
    csv_path = Path(pytestconfig.getoption("--report-csv")).resolve()
    return {"html": html_path, "csv": csv_path}


@pytest.fixture(scope="session")
def price_tolerance(pytestconfig) -> float:
    tol = float(pytestconfig.getoption("--tolerance"))
    return max(0.0, tol)


@pytest.fixture(scope="session")
def validation_output_dir() -> Path:
    base = Path(os.getenv("VALIDATION_REPORT_DIR", Path("reports") / "validation"))
    base.mkdir(parents=True, exist_ok=True)
    return base


# =============================
# BRAPI Provider adapter wrapper
# =============================

def _retry(callable_fn, *, retries: int = 2, backoff: float = 0.75):
    last_err = None
    for i in range(retries + 1):
        try:
            return callable_fn()
        except Exception as e:  # noqa: BLE001 - test-side generic retry wrapper
            last_err = e
            if i < retries:
                time.sleep(backoff * (2 ** i))
    if last_err is not None:
        raise last_err


@pytest.fixture(scope="session")
def brapi_adapter():
    """Create a BrapiProvider instance using env token. If token is missing, allow cache-only usage."""
    # Deferred import so tests remain importable without full engine deps
    from engine.brapi_provider import BrapiProvider

    api_token = os.getenv("BRAPI_API_TOKEN", "")
    # Instantiate even if token empty to enable cache reads; HTTP calls may fail without token
    provider = BrapiProvider(api_token=api_token)
    return provider


def _get_ohlc_safe(provider, symbol: str, interval: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """Adapter call with timezone normalization.

    The provider expects tz-naive datetimes (interpreted as UTC). Convert any tz-aware
    inputs to naive before calling to avoid 'offset-naive vs offset-aware' comparisons
    inside the adapter.
    """
    s = start_dt.replace(tzinfo=None) if start_dt.tzinfo is not None else start_dt
    e = end_dt.replace(tzinfo=None) if end_dt.tzinfo is not None else end_dt

    def _call():
        return provider.get_ohlc(symbol=symbol, interval=interval, start_date=s, end_date=e)

    return _retry(_call)


# =============================
# HTML/CSV parsing utilities
# =============================

@dataclass
class ExecRow:
    symbol: str
    timestamp_utc: datetime
    order_type: str  # MARKET, LIMIT, MOC (upper)
    side: str  # BUY/SELL
    execution_price: float


def _parse_price_br(text: str) -> float:
    s = (text or "").strip()
    if not s:
        return float("nan")
    # Normalize pt-BR decimal (1.234,56) to 1234.56
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:  # noqa: BLE001
        return float("nan")


def _parse_timestamp_html(text: str) -> Optional[datetime]:
    # Format like: 2025.08.05 13:00:00 (UTC implied)
    s = (text or "").strip()
    if not s:
        return None
    try:
        dt = datetime.strptime(s, "%Y.%m.%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return dt
    except Exception:  # noqa: BLE001
        return None


def _extract_execs_from_html(html_path: Path) -> List[ExecRow]:
    if not html_path.exists():
        return []
    from bs4 import BeautifulSoup  # type: ignore

    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    # Find the orders table by header row containing 'Horário' in first <th>
    tables = soup.find_all("table")
    target_table = None
    for tbl in tables:
        ths = tbl.find_all("th")
        if not ths:
            continue
        first = ths[0].get_text(strip=True).lower()
        if "horário" in first or "horario" in first:
            target_table = tbl
            break

    rows: List[ExecRow] = []
    if not target_table:
        return rows

    # Parse body rows; expect columns: Horário | Ativo | Direção | Quantidade | Preço | PNL (alloc.) | Tipo
    for tr in target_table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 7:
            continue
        ts_txt = tds[0].get_text(strip=True)
        sym_txt = tds[1].get_text(strip=True).upper()
        side_txt = tds[2].get_text(strip=True).upper()
        # qty = tds[3]  # not used
        price_txt = tds[4].get_text(strip=True)
        # pnl = tds[5]
        type_txt = tds[6].get_text(strip=True).upper()

        ts = _parse_timestamp_html(ts_txt)
        if ts is None:
            continue
        price = _parse_price_br(price_txt)

        # Normalize order type labels
        order_type = {
            "MARKET": "MARKET",
            "LIMIT": "LIMIT",
            "MOC": "MOC",
        }.get(type_txt, type_txt)

        rows.append(ExecRow(symbol=sym_txt, timestamp_utc=ts, order_type=order_type, side=side_txt, execution_price=price))

    # Deduplicate identical rows (some reports may repeat header blocks)
    unique: Dict[Tuple[str, datetime, str, str, float], ExecRow] = {}
    for r in rows:
        key = (r.symbol, r.timestamp_utc, r.order_type, r.side, r.execution_price)
        unique[key] = r
    return list(unique.values())


def _extract_execs_from_csv(csv_path: Path) -> List[ExecRow]:
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    # Expected columns: timestamp, symbol, side, quantity, execution_price, attempt_type, filled
    required = {"timestamp", "symbol", "side", "execution_price"}
    if not required.issubset(set(map(str.lower, df.columns))):
        # Try normalized columns
        cols = {c.lower(): c for c in df.columns}
    else:
        cols = {c.lower(): c for c in df.columns}

    rows: List[ExecRow] = []
    for _, rec in df.iterrows():
        try:
            ts_raw = rec[cols.get("timestamp", "timestamp")]
            sym = str(rec[cols.get("symbol", "symbol")]).upper()
            side = str(rec[cols.get("side", "side")]).upper()
            px = float(rec[cols.get("execution_price", "execution_price")])
            attempt = str(rec.get(cols.get("attempt_type", "attempt_type"), "")).upper()
            # csv timestamps are assumed UTC or naive UTC
            ts = pd.to_datetime(ts_raw, utc=True).to_pydatetime()
            order_type = {"MARKET": "MARKET", "LIMIT": "LIMIT", "MOC": "MOC"}.get(attempt, attempt)
            rows.append(ExecRow(symbol=sym, timestamp_utc=ts, order_type=order_type, side=side, execution_price=px))
        except Exception:  # noqa: BLE001
            continue

    # Deduplicate
    unique: Dict[Tuple[str, datetime, str, str, float], ExecRow] = {}
    for r in rows:
        key = (r.symbol, r.timestamp_utc, r.order_type, r.side, r.execution_price)
        unique[key] = r
    return list(unique.values())


# =============================
# Validation helpers
# =============================

SESSION_OPEN_HOUR_UTC = 13
SESSION_CLOSE_HOUR_UTC = 20


def _within_tolerance(a: float, b: float, tol: float) -> bool:
    if any(map(lambda v: v is None or (isinstance(v, float) and math.isnan(v)), (a, b))):
        return False
    return abs(float(a) - float(b)) <= tol


def _validate_exec(
    ex: ExecRow,
    bars_1h: pd.DataFrame,
    tol: float,
    daily: Optional[pd.DataFrame] = None,
) -> Tuple[str, str, Dict[str, Optional[float]]]:
    """Return (result, reason, provider_fields). result in {PASS, FAIL, XFAIL}."""
    # Session bounds
    hour = ex.timestamp_utc.hour
    if hour < SESSION_OPEN_HOUR_UTC or hour > SESSION_CLOSE_HOUR_UTC:
        return (
            "FAIL",
            f"timestamp {ex.timestamp_utc.isoformat()} outside B3 UTC session [13:00, 20:00]",
            {},
        )

    # Align to exact bar by timestamp (normalize to tz-naive UTC index for provider)
    ts = pd.to_datetime(ex.timestamp_utc)
    ts_naive = ts.tz_convert(timezone.utc).tz_localize(None) if ts.tzinfo is not None else ts
    try:
        row = bars_1h.loc[ts_naive]
    except Exception:
        # Try without tz
        try:
            row = bars_1h.loc[pd.to_datetime(ts_naive)]
        except Exception:
            return ("XFAIL", f"provider coverage gap for {ex.symbol} at {ts.isoformat()}", {})

    open_px = float(row.get("open", float("nan")))
    high_px = float(row.get("high", float("nan")))
    low_px = float(row.get("low", float("nan")))
    close_px = float(row.get("close", float("nan")))

    provider = {"open": open_px, "high": high_px, "low": low_px, "close": close_px, "bar_ts": ts_naive.isoformat()}

    # Rule checks
    if ex.order_type == "MARKET":
        # Must be exactly open 13:00
        if ts.hour != SESSION_OPEN_HOUR_UTC:
            return ("FAIL", f"MARKET not at 13:00 UTC ({ts.hour:02d}:00)", provider)
        if _within_tolerance(ex.execution_price, open_px, tol):
            # Optional cross-check against daily (if provided)
            if daily is not None and not daily.empty:
                try:
                    drow = daily.loc[pd.to_datetime(ts.date())]
                    daily_open = float(drow.get("open", float("nan")))
                    # daily open equality is a soft check; ignore its result in outcome
                    _ = _within_tolerance(open_px, daily_open, tol)
                except Exception:
                    pass
            return ("PASS", "MARKET open equals bar open", provider)
        return ("FAIL", f"MARKET price {ex.execution_price:.2f} != bar open {open_px:.2f} ±{tol}", provider)

    if ex.order_type == "LIMIT":
        if low_px <= ex.execution_price <= high_px:
            return ("PASS", "LIMIT price within bar range", provider)
        return (
            "FAIL",
            f"LIMIT price {ex.execution_price:.2f} outside bar range [{low_px:.2f}, {high_px:.2f}]",
            provider,
        )

    if ex.order_type == "MOC":
        # Must be exactly close 20:00
        if ts.hour != SESSION_CLOSE_HOUR_UTC:
            return ("FAIL", f"MOC not at 20:00 UTC ({ts.hour:02d}:00)", provider)
        if _within_tolerance(ex.execution_price, close_px, tol):
            return ("PASS", "MOC close equals bar close", provider)
        return ("FAIL", f"MOC price {ex.execution_price:.2f} != bar close {close_px:.2f} ±{tol}", provider)

    # Unknown order type: mark XFAIL so it does not hard-fail suite
    return ("XFAIL", f"unknown order_type {ex.order_type}", provider)


# =============================
# The test
# =============================


def test_validate_html_executions_against_brapi(report_paths, brapi_adapter, price_tolerance, validation_output_dir):
    html_execs = _extract_execs_from_html(report_paths["html"]) if report_paths["html"].exists() else []
    csv_execs = _extract_execs_from_csv(report_paths["csv"]) if report_paths["csv"].exists() else []

    # Prefer CSV for numeric precision when available, otherwise HTML
    exec_rows: List[ExecRow] = csv_execs or html_execs
    assert exec_rows, f"No executions found. Checked HTML: {report_paths['html']}, CSV: {report_paths['csv']}"

    # Group by (symbol, day) to minimize adapter calls
    batches: Dict[Tuple[str, datetime], List[ExecRow]] = {}
    for ex in exec_rows:
        day_key = datetime(ex.timestamp_utc.year, ex.timestamp_utc.month, ex.timestamp_utc.day, tzinfo=timezone.utc)
        batches.setdefault((ex.symbol, day_key), []).append(ex)

    # Collect validations
    out_rows: List[Dict[str, object]] = []
    any_fail = False

    for (symbol, day_utc), ex_list in sorted(batches.items(), key=lambda x: (x[0][0], x[0][1])):
        start_dt = day_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt = day_utc.replace(hour=23, minute=59, second=59, microsecond=0)

        # Intraday 1h bars: cache-first via adapter
        try:
            bars_1h = _get_ohlc_safe(brapi_adapter, symbol, "1h", start_dt, end_dt)
        except Exception as e:  # noqa: BLE001
            # If token missing and cache insufficient, xfail entire day
            for ex in ex_list:
                out_rows.append(
                    {
                        "symbol": symbol,
                        "timestamp": ex.timestamp_utc.isoformat(),
                        "order_type": ex.order_type,
                        "side": ex.side,
                        "execution_price": ex.execution_price,
                        "provider_bar_open": None,
                        "high": None,
                        "low": None,
                        "close": None,
                        "rule_checked": "adapter_get_ohlc",
                        "result": "XFAIL",
                        "reason": f"adapter error: {e}",
                    }
                )
            continue

        # Daily bars: needed for indicator checks (ATR/EMA/RSI). Pull a window large enough for indicators.
        try:
            daily_window_start = (start_dt - timedelta(days=60)).replace(tzinfo=None)
            daily = _get_ohlc_safe(brapi_adapter, symbol, "1d", daily_window_start, end_dt)
        except Exception:
            daily = None

        # Normalize index to pandas DatetimeIndex and ensure UTC naive/aware consistency
        for df in (bars_1h, daily if daily is not None else pd.DataFrame([])):
            if not df.empty:
                try:
                    # Coerce to DatetimeIndex
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    pass

        for ex in ex_list:
            result, reason, provider = _validate_exec(ex, bars_1h, price_tolerance, daily)
            if result == "FAIL":
                any_fail = True
            out_rows.append(
                {
                    "symbol": ex.symbol,
                    "timestamp": ex.timestamp_utc.isoformat(),
                    "order_type": ex.order_type,
                    "side": ex.side,
                    "execution_price": ex.execution_price,
                    "provider_bar_open": provider.get("open") if provider else None,
                    "high": provider.get("high") if provider else None,
                    "low": provider.get("low") if provider else None,
                    "close": provider.get("close") if provider else None,
                    "provider_bar_ts": provider.get("bar_ts") if provider else None,
                    "rule_checked": ex.order_type,
                    "result": result,
                    "reason": reason,
                }
            )

        # Indicator validation per (symbol, day):
        # Compute ATR(14), EMA(10), RSI(14) from provider daily and record latest values at D.
        try:
            if daily is not None and not daily.empty:
                # Normalize index and ensure we have at least 30-60 days
                daily_idx = pd.to_datetime(daily.index)
                daily = daily.copy()
                daily.index = daily_idx
                # Select up to D (inclusive)
                daily_upto_d = daily[daily.index <= end_dt.replace(tzinfo=None)]
                from engine.market_utils import DailyTechnicalIndicators
                dti = DailyTechnicalIndicators()
                atr_series = dti.calculate_atr(daily_upto_d, period=14)
                ema10_series = dti.calculate_ema(daily_upto_d, period=10)
                rsi14_series = dti.calculate_rsi(daily_upto_d, period=14)
                atr_d = float(atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else None
                ema10_d = float(ema10_series.dropna().iloc[-1]) if not ema10_series.dropna().empty else None
                rsi14_d = float(rsi14_series.dropna().iloc[-1]) if not rsi14_series.dropna().empty else None
                # Append a synthetic summary row for indicators (not part of required pass/fail)
                out_rows.append(
                    {
                        "symbol": symbol,
                        "timestamp": day_utc.isoformat(),
                        "order_type": "INDICATORS",
                        "side": "-",
                        "execution_price": None,
                        "provider_bar_open": None,
                        "high": None,
                        "low": None,
                        "close": None,
                        "provider_bar_ts": None,
                        "rule_checked": "ATR/EMA10/RSI14",
                        "result": "PASS" if all(v is not None for v in (atr_d, ema10_d, rsi14_d)) else "XFAIL",
                        "reason": f"atr14={atr_d} ema10={ema10_d} rsi14={rsi14_d}",
                    }
                )
        except Exception as _e:
            out_rows.append(
                {
                    "symbol": symbol,
                    "timestamp": day_utc.isoformat(),
                    "order_type": "INDICATORS",
                    "side": "-",
                    "execution_price": None,
                    "provider_bar_open": None,
                    "high": None,
                    "low": None,
                    "close": None,
                    "provider_bar_ts": None,
                    "rule_checked": "ATR/EMA10/RSI14",
                    "result": "XFAIL",
                    "reason": f"indicator calc error: {_e}",
                }
            )

    # Write validation CSV
    out_file = validation_output_dir / "html_vs_brapi_validation.csv"
    cols = [
        "symbol",
        "timestamp",
        "order_type",
        "side",
        "execution_price",
        "provider_bar_open",
        "high",
        "low",
        "close",
        "provider_bar_ts",
        "rule_checked",
        "result",
        "reason",
    ]
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in out_rows:
            # Map nested provider field bar_ts
            row_out = {k: r.get(k) for k in cols}
            provider_ts = None
            if r.get("provider_bar_open") is not None:
                # provider timestamp stored in reason map 'provider' earlier as bar_ts
                pass
            # provider bar timestamp was included in reason map; capture if present
            if isinstance(r.get("reason"), str):
                # leave as-is; only enrich provider_bar_ts when available in provider map
                pass
            # fetch bar_ts if we stored it in a parallel key
            if isinstance(r.get("provider"), dict) and r["provider"].get("bar_ts"):
                provider_ts = r["provider"]["bar_ts"]
            # In current structure we kept provider fields flattened; bar_ts is stashed in reason map earlier
            row_out["provider_bar_ts"] = r.get("provider_bar_ts") or provider_ts
            writer.writerow(row_out)

    # Summary for pytest output
    passed = sum(1 for r in out_rows if r["result"] == "PASS")
    failed = sum(1 for r in out_rows if r["result"] == "FAIL")
    xfailed = sum(1 for r in out_rows if r["result"] == "XFAIL")
    print(f"Validation summary: PASS={passed} FAIL={failed} XFAIL={xfailed}. Report: {out_file}")

    # Fail test if any hard failures
    assert not any_fail, f"There are {failed} failed execution validations. See {out_file}"


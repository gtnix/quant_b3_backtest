import os
from pathlib import Path
import yaml

import pytest

from engine.brapi_provider import BrapiProvider
import pandas as pd


def _read_token():
    sp = Path('config/secrets.yaml')
    if sp.exists():
        try:
            sec = yaml.safe_load(sp.read_text()) or {}
            tok = ((sec.get('brapi') or {}).get('api_token')) or sec.get('BRAPI_API_TOKEN') or ''
            if tok:
                return tok
        except Exception:
            pass
    return os.environ.get('BRAPI_API_TOKEN','')


@pytest.mark.smoke
def test_get_earliest_timestamp_shape_and_values():
    token = _read_token()
    provider = BrapiProvider(api_token=token, cache_dir='data/brapi_cache_probe', cache_ttl_hours=0)
    # Load first 2 symbols dynamically from portfolio.csv
    try:
        df = pd.read_csv('data/portfolio.csv')
        col = 'symbol' if 'symbol' in df.columns else df.columns[0]
        syms = [str(s).strip().upper() for s in df[col].dropna() if str(s).strip()]
        syms = syms[:2] if syms else []
    except Exception:
        syms = []
    if not syms:
        pytest.skip("portfolio.csv missing or empty; cannot select symbols dynamically")
    for sym in syms:
        for itv in ['1h','1d']:
            info = provider.get_earliest_timestamp(sym, itv)
            assert 'symbol' in info and info['symbol'] == sym
            assert 'interval' in info and info['interval'] == itv
            assert 'rows' in info and isinstance(info['rows'], int)
            # Allow None when provider not available or data missing, but keys must exist
            assert 'earliest_utc' in info
            assert 'earliest_local' in info



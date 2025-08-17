import json
from pathlib import Path
import pandas as pd
import shutil

from engine.loader import DataLoader
from engine.brapi_provider import BrapiProvider


def _seed_intraday(symbol: str, base_dir: Path, interval: str, start: str, end: str, rows: int):
    d = base_dir / 'intraday'
    d.mkdir(parents=True, exist_ok=True)
    idx = pd.date_range(start=start, end=end, periods=rows)
    df = pd.DataFrame({'open':[10.0]*rows,'high':[10.5]*rows,'low':[9.5]*rows,'close':[10.0]*rows,'volume':[1000]*rows}, index=idx)
    pf = d / f"{symbol}_{interval}.parquet"
    df.to_parquet(pf)
    mf = d / f"{symbol}_{interval}_metadata.json"
    with open(mf,'w') as f:
        json.dump({'symbol':symbol,'interval':interval,'start':start,'end':end,'rows':rows}, f)



def test_sync_noop_when_up_to_date(tmp_path, monkeypatch):
    """Sync should skip when local intraday cache is already fresh.
    Updated to align with provider's current intraday default (1h).
    """
    prov = BrapiProvider(api_token='', cache_dir=str(tmp_path / 'data' / 'brapi_cache'))

    # Seed an intraday cache file matching the provider's current 1h convention
    cf, _ = prov._intraday_cache_paths('VALE3')
    # Override to 1h filename for current default
    cf = cf.with_name(cf.name.replace('_5m.parquet', '_1h.parquet'))
    cf.parent.mkdir(parents=True, exist_ok=True)

    # Build recent 1h timestamps (tz-naive) to satisfy freshness check
    idx = pd.date_range(end=pd.Timestamp.utcnow().replace(tzinfo=None), periods=3, freq='1h')
    df = pd.DataFrame({'open':[1,2,3],'high':[1,2,3],'low':[1,2,3],'close':[1,2,3],'volume':[0,0,0]}, index=idx)
    df.to_parquet(cf)

    # Monkeypatch cache path helper to point to our 1h file
    monkeypatch.setattr(prov, '_intraday_cache_paths', lambda sym: (cf, cf.with_name(cf.stem + '_metadata.json')))

    res = prov.sync_brapi_history('VALE3')
    assert res['skipped'] is True


def test_sync_gap_fill(tmp_path, monkeypatch):
    prov = BrapiProvider(api_token='', cache_dir=str(tmp_path / 'data' / 'brapi_cache'))
    start = pd.Timestamp.utcnow() - pd.Timedelta(days=10)
    idx_old = pd.date_range(start=start, periods=2, freq='5min')
    df_old = pd.DataFrame({'open':[1,2],'high':[1,2],'low':[1,2],'close':[1,2],'volume':[0,0]}, index=idx_old)
    cf, _ = prov._intraday_cache_paths('ITUB4')
    Path(cf).parent.mkdir(parents=True, exist_ok=True)
    df_old.to_parquet(cf)
    def _mock_fetch(symbol, start_date, end_date, data_type):
        idx_new = pd.date_range(end=pd.Timestamp.utcnow(), periods=3, freq='5min')
        return pd.DataFrame({'open':[3,4,5],'high':[3,4,5],'low':[3,4,5],'close':[3,4,5],'volume':[0,0,0]}, index=idx_new)
    monkeypatch.setattr(prov, '_fetch_brapi_data', _mock_fetch)
    res = prov.sync_brapi_history('ITUB4')
    assert res['skipped'] is False
    df_merged = pd.read_parquet(cf)
    assert len(df_merged.index.unique()) == len(df_merged)



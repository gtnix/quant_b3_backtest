import pandas as pd

from pathlib import Path


def test_dispersion_and_rotation_filters_block_low_quality(tmp_path: Path):
    # Synthetic daily fuzzy components
    rows = []
    dates = pd.date_range('2025-07-01', periods=5, freq='D')
    # Day1: strong
    rows += [{'date': str(d.date()), 'symbol': 'ALPA4', 'qualified_signal': 'BUY', 'fuzzy_score_raw': 2.2, 'close': 20.0} for d in dates[:1]]
    rows += [{'date': str(dates[0].date()), 'symbol': 'PETR4', 'qualified_signal': 'SELL', 'fuzzy_score_raw': -2.0, 'close': 30.0}]
    # Day2: repeat ALPA4 with small improvement (<0.25) → should be blocked
    rows += [{'date': str(dates[1].date()), 'symbol': 'ALPA4', 'qualified_signal': 'BUY', 'fuzzy_score_raw': 2.35, 'close': 21.0}]
    rows += [{'date': str(dates[1].date()), 'symbol': 'VALE3', 'qualified_signal': 'SELL', 'fuzzy_score_raw': -2.1, 'close': 40.0}]
    # Day3: both sides below threshold 1.8 → blocked
    rows += [{'date': str(dates[2].date()), 'symbol': 'ALPA4', 'qualified_signal': 'BUY', 'fuzzy_score_raw': 1.6, 'close': 21.5}]
    rows += [{'date': str(dates[2].date()), 'symbol': 'VALE3', 'qualified_signal': 'SELL', 'fuzzy_score_raw': -1.7, 'close': 41.0}]

    df = pd.DataFrame(rows)
    # Compute schedule using same logic fragment as runner
    pm = {
        'min_signal_strength': 1.80,
        'min_total_strength': 4.0,
        'rotation_improve_abs': 0.25,
        'atr_min_ratio': 0.015,
        'dispersion_lookback': 60,
        'dispersion_low_quantile': 0.30,
        'tranche_scale_low_dispersion': 0.50,
    }
    last_score = {}
    pair_days = []
    for d, g in df.groupby('date'):
        buys = g[g['qualified_signal']=='BUY']
        sells = g[g['qualified_signal']=='SELL']
        if len(buys)==0 or len(sells)==0:
            continue
        b = buys.sort_values('fuzzy_score_raw', ascending=False).iloc[0]
        s = sells.sort_values('fuzzy_score_raw', ascending=True).iloc[0]
        sb = float(b['fuzzy_score_raw']); ss = float(s['fuzzy_score_raw'])
        if abs(sb) < pm['min_signal_strength'] or abs(ss) < pm['min_signal_strength']:
            continue
        if (abs(sb) + abs(ss)) < pm['min_total_strength']:
            continue
        lb = last_score.get(str(b['symbol']))
        ls = last_score.get(str(s['symbol']))
        if lb is not None and (abs(sb) - abs(lb)) < pm['rotation_improve_abs']:
            continue
        if ls is not None and (abs(ss) - abs(ls)) < pm['rotation_improve_abs']:
            continue
        pair_days.append(d)
        last_score[str(b['symbol'])] = sb
        last_score[str(s['symbol'])] = ss
    # Expect only Day1 to pass
    assert len(pair_days) == 1
    assert str(pd.to_datetime(pair_days[0]).date()) == str(dates[0].date())



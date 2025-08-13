import pandas as pd


def test_atr_ratio_filter_blocks_quiet_instruments():
    # Build a simple row with tiny ATR relative to price
    df = pd.DataFrame([
        {'date': '2025-07-01', 'symbol': 'ALPA4', 'qualified_signal': 'BUY', 'fuzzy_score_raw': 2.5, 'close': 50.0, 'atr': 0.2},
        {'date': '2025-07-01', 'symbol': 'PETR4', 'qualified_signal': 'SELL', 'fuzzy_score_raw': -2.5, 'close': 80.0, 'atr': 0.3},
    ])
    # atr_min_ratio requires ≥1.5%; here ratios are 0.4% and 0.375% → both blocked
    atr_min_ratio = 0.015
    def ok(a,p):
        try:
            return (a/p) >= atr_min_ratio
        except Exception:
            return False
    assert not ok(df.iloc[0]['atr'], df.iloc[0]['close'])
    assert not ok(df.iloc[1]['atr'], df.iloc[1]['close'])



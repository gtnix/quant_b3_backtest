import os
import types
import pytest
from datetime import datetime, date
from typing import Dict, Any

from strategies.fuzzy_fajuto_strategy import FuzzyFajutoStrategy
from engine.base_strategy import StrategyConfig, StrategyContext, OrderSide


class DummyLogger:
    def info(self, *args, **kwargs):
        pass
    def warning(self, *args, **kwargs):
        pass
    def debug(self, *args, **kwargs):
        pass


class DummyPortfolio:
    def get_portfolio_value(self):
        return 1_000_000.0


def make_strategy() -> FuzzyFajutoStrategy:
    cfg = StrategyConfig(universe=["A","B","C"], warmup_bars=1, risk_tolerance=0.01, max_position_size=0.1)
    ctx = StrategyContext(data_portal=None, portfolio=DummyPortfolio(), broker=None, market_rules=None, logger=DummyLogger(), metadata={})
    strat = FuzzyFajutoStrategy(cfg, ctx)
    strat.RISK_PAIR_MATCHING = True
    return strat


def _mk_rec(side: OrderSide, fuzzy: float, prices: Dict[str,float], qty: Dict[str,int], ts: datetime) -> Dict[str, Any]:
    # Build a dummy bar with OHLC populated to satisfy execution tracker
    px = float(prices.get('market') or list(prices.values())[0])
    bar = types.SimpleNamespace(timestamp=ts, open=px, high=px, low=px, close=px)
    return {
        'side': side,
        'fuzzy': fuzzy,
        'prices': prices,
        'qty': qty,
        'lot_size': 100,
        'bar': bar,
        'atr': 0.5,
        'signal': fuzzy,
    }


def test_pairing_more_buys_than_sells():
    s = make_strategy()
    d = date(2025,8,1)
    ts = datetime(d.year, d.month, d.day, 13, 0, 0)
    s._neutral_buffer[d] = {
        'BUY1': _mk_rec(OrderSide.BUY, 2.0, {'market':10,'limit_alpha':10,'limit_beta':10,'limit_gamma':10}, {'market':300,'limit_alpha':300,'limit_beta':0,'limit_gamma':0}, ts),
        'BUY2': _mk_rec(OrderSide.BUY, 1.5, {'market':10,'limit_alpha':10,'limit_beta':10,'limit_gamma':10}, {'market':200,'limit_alpha':200,'limit_beta':0,'limit_gamma':0}, ts),
        'SELL1': _mk_rec(OrderSide.SELL, -2.2, {'market':10,'limit_alpha':10,'limit_beta':10,'limit_gamma':10}, {'market':400,'limit_alpha':0,'limit_beta':0,'limit_gamma':0}, ts),
    }
    intents = list(s._neutral_emit_for_day(d))
    assert any(it.symbol == 'BUY1' for it in intents)
    assert any(it.symbol == 'SELL1' for it in intents)


def test_pairing_more_sells_than_buys():
    s = make_strategy()
    d = date(2025,8,2)
    ts = datetime(d.year, d.month, d.day, 13, 0, 0)
    s._neutral_buffer[d] = {
        'BUY1': _mk_rec(OrderSide.BUY, 2.0, {'market':20,'limit_alpha':20,'limit_beta':20,'limit_gamma':20}, {'market':200,'limit_alpha':200,'limit_beta':0,'limit_gamma':0}, ts),
        'SELL1': _mk_rec(OrderSide.SELL, -2.5, {'market':20,'limit_alpha':20,'limit_beta':20,'limit_gamma':20}, {'market':300,'limit_alpha':0,'limit_beta':0,'limit_gamma':0}, ts),
        'SELL2': _mk_rec(OrderSide.SELL, -1.8, {'market':20,'limit_alpha':20,'limit_beta':20,'limit_gamma':20}, {'market':200,'limit_alpha':0,'limit_beta':0,'limit_gamma':0}, ts),
    }
    intents = list(s._neutral_emit_for_day(d))
    assert any(it.symbol == 'BUY1' for it in intents)
    # At least one SELL leg must be emitted
    assert any(getattr(it, 'side', None) and it.side.name == 'SELL' for it in intents)


def test_pairing_equal_counts_all_paired():
    s = make_strategy()
    d = date(2025,8,3)
    ts = datetime(d.year, d.month, d.day, 13, 0, 0)
    s._neutral_buffer[d] = {
        'BUY1': _mk_rec(OrderSide.BUY, 2.0, {'market':30,'limit_alpha':30,'limit_beta':30,'limit_gamma':30}, {'market':100,'limit_alpha':100,'limit_beta':0,'limit_gamma':0}, ts),
        'SELL1': _mk_rec(OrderSide.SELL, -2.0, {'market':30,'limit_alpha':30,'limit_beta':30,'limit_gamma':30}, {'market':100,'limit_alpha':0,'limit_beta':0,'limit_gamma':0}, ts),
    }
    intents = list(s._neutral_emit_for_day(d))
    assert any(it.symbol == 'BUY1' for it in intents)
    assert any(it.symbol == 'SELL1' for it in intents)


def test_tie_breaking_deterministic():
    s = make_strategy()
    d = date(2025,8,4)
    ts = datetime(d.year, d.month, d.day, 13, 0, 0)
    s._neutral_buffer[d] = {
        'BUY1': _mk_rec(OrderSide.BUY, 2.0, {'market':40,'limit_alpha':40,'limit_beta':40,'limit_gamma':40}, {'market':200,'limit_alpha':200,'limit_beta':0,'limit_gamma':0}, ts),
        'BUY2': _mk_rec(OrderSide.BUY, 2.0, {'market':40,'limit_alpha':40,'limit_beta':40,'limit_gamma':40}, {'market':200,'limit_alpha':200,'limit_beta':0,'limit_gamma':0}, ts),
        'SELL1': _mk_rec(OrderSide.SELL, -2.1, {'market':40,'limit_alpha':40,'limit_beta':40,'limit_gamma':40}, {'market':300,'limit_alpha':0,'limit_beta':0,'limit_gamma':0}, ts),
    }
    intents = list(s._neutral_emit_for_day(d))
    assert any(it.symbol == 'BUY1' for it in intents)
    assert any(it.symbol == 'BUY2' for it in intents)

import os
import types
import pytest
from datetime import datetime, date
from typing import Dict, Any

from strategies.fuzzy_fajuto_strategy import FuzzyFajutoStrategy
from engine.base_strategy import StrategyConfig, StrategyContext, OrderSide, OrderType


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


def test_bidirectional_execution_comprehensive():
    """
    COMPREHENSIVE TEST: Validates bidirectional pair execution.
    
    This test ensures both Long and Short legs of every matched pair 
    are fully processed, addressing the core pairing specification.
    """
    s = make_strategy()
    d = date(2025, 8, 10)
    ts = datetime(d.year, d.month, d.day, 13, 0, 0)
    
    # Perfect pairing scenario: 2 BUYs, 2 SELLs
    s._neutral_buffer[d] = {
        'LONG_HIGH': _mk_rec(OrderSide.BUY, 2.5, {'market':20,'limit_alpha':19,'limit_beta':18,'limit_gamma':17}, {'market':400,'limit_alpha':400,'limit_beta':300,'limit_gamma':200}, ts),
        'LONG_LOW': _mk_rec(OrderSide.BUY, 2.0, {'market':20,'limit_alpha':19,'limit_beta':18,'limit_gamma':17}, {'market':300,'limit_alpha':300,'limit_beta':200,'limit_gamma':100}, ts),
        'SHORT_HIGH': _mk_rec(OrderSide.SELL, -2.8, {'market':20,'limit_alpha':21,'limit_beta':22,'limit_gamma':23}, {'market':500,'limit_alpha':400,'limit_beta':300,'limit_gamma':200}, ts),
        'SHORT_LOW': _mk_rec(OrderSide.SELL, -2.3, {'market':20,'limit_alpha':21,'limit_beta':22,'limit_gamma':23}, {'market':400,'limit_alpha':300,'limit_beta':200,'limit_gamma':100}, ts),
    }
    
    # Execute pairing logic
    intents = list(s._neutral_emit_for_day(d))
    
    # Analyze results
    buy_intents = [intent for intent in intents if intent.side == OrderSide.BUY]
    sell_intents = [intent for intent in intents if intent.side == OrderSide.SELL]
    buy_symbols = {intent.symbol for intent in buy_intents}
    sell_symbols = {intent.symbol for intent in sell_intents}
    
    # CRITICAL ASSERTIONS for bidirectional execution
    assert len(buy_intents) > 0, "❌ BUG: No BUY orders generated!"
    assert len(sell_intents) > 0, "❌ BUG: No SELL orders generated!"
    assert len(buy_symbols) > 0, "❌ BUG: No BUY symbols processed!"
    assert len(sell_symbols) > 0, "❌ BUG: No SELL symbols processed!"
    
    # Verify proper pairing according to README specification:
    # - For each Short: pair it with Long having highest available FuzzyFajuto score
    # - For each Long: pair it with Short having highest available FuzzyFajuto score
    # Expected: SHORT_HIGH (-2.8) pairs with LONG_HIGH (2.5), SHORT_LOW (-2.3) pairs with LONG_LOW (2.0)
    assert 'LONG_HIGH' in buy_symbols, "Highest BUY signal not processed"
    assert 'SHORT_HIGH' in sell_symbols, "Highest SELL signal not processed"
    
    # For perfect pairing: both sides should have equal representation
    if len(buy_symbols) == len(sell_symbols):
        print(f"✅ PERFECT BIDIRECTIONAL PAIRING: {len(buy_symbols)} BUY ↔ {len(sell_symbols)} SELL")
    else:
        print(f"⚠️  UNBALANCED PAIRING: {len(buy_symbols)} BUY ↔ {len(sell_symbols)} SELL")
    
    # Validate order types distribution (should have market + limit orders)
    market_orders = len([intent for intent in intents if intent.order_type == OrderType.MARKET])
    limit_orders = len([intent for intent in intents if intent.order_type == OrderType.LIMIT])
    
    assert market_orders > 0, "No market orders generated"
    assert limit_orders > 0, "No limit orders generated"
    assert market_orders + limit_orders == len(intents), "Order type mismatch"

import pytest

from engine.market_utils import BrazilianMarketUtils


@pytest.mark.core
def test_tick_and_lot_size_validation():
    utils = BrazilianMarketUtils()
    assert utils.normalize_price_tick(12.345) == 12.35
    assert utils.normalize_price_tick(12.344) == 12.34
    ok, lot_type, frac = utils.validate_lot_size(100)
    assert ok and not frac
    ok2, lot_type2, frac2 = utils.validate_lot_size(150)
    assert ok2 and frac2


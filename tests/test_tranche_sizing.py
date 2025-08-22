"""
DEPRECATED: This test has been migrated to tests/modular/test_position_sizing.py

The tranche sizing logic is now handled by the PositionSizer component.
See tests/modular/test_position_sizing.py for updated tests.
"""

import pytest
from strategies.components.position_sizing import PositionSizer


@pytest.mark.skip(reason="Migrated to modular tests - see tests/modular/test_position_sizing.py")
def test_tranche_round_lot_buy_levels():
    """DEPRECATED: Migrated to PositionSizer component tests."""
    pass


@pytest.mark.skip(reason="Migrated to modular tests - see tests/modular/test_position_sizing.py")
def test_tranche_round_lot_sell_levels():
    """DEPRECATED: Migrated to PositionSizer component tests."""
    pass


@pytest.mark.skip(reason="Migrated to modular tests - see tests/modular/test_position_sizing.py")
def test_nearest_100_rounding_policy_boundaries():
    """DEPRECATED: Migrated to PositionSizer component tests."""
    pass


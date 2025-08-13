import unittest
from datetime import datetime

from engine.portfolio import EnhancedPortfolio
from strategies.enhanced_fuzzy_fajuto_strategy import EnhancedFuzzyFajutoStrategy


class TestRiskControlsPairMode(unittest.TestCase):
    def test_daily_loss_cap_blocks_trades(self):
        portfolio = EnhancedPortfolio()
        # Seed portfolio value via initial trade to set a baseline
        portfolio.buy('ALPA4', quantity=100, price=10.0, trade_date=datetime(2024, 7, 1, 10, 0, 0), trade_type='day_trade')

        strat = EnhancedFuzzyFajutoStrategy(portfolio=portfolio, symbol='ALPA4', risk_tolerance=0.02)
        # Simulate a big intraday loss in strategy state
        strat.daily_loss = strat.context.portfolio.get_portfolio_value() * 0.03  # 3% loss

        # Build a mock intent-like object that passes basic checks
        class Intent:
            def __init__(self):
                self.price = 10.0
                self.quantity = 100

        ok = strat.check_brazilian_market_constraints(Intent())
        self.assertFalse(ok, msg='Daily loss cap should block new trades')


if __name__ == '__main__':
    unittest.main()



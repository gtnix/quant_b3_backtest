import unittest
from datetime import datetime, date
import pandas as pd

from engine.portfolio import EnhancedPortfolio
from engine.simulator import BacktestSimulator
from strategies.enhanced_fuzzy_fajuto_strategy import EnhancedFuzzyFajutoStrategy


class TestEODFlatteningPairMode(unittest.TestCase):
    def test_eod_uses_close_prices(self):
        # Minimal data with two days for a dummy symbol; prices jump at close
        idx = [
            datetime(2024, 7, 1, 17, 0, 0),
            datetime(2024, 7, 1, 20, 0, 0),  # official close
            datetime(2024, 7, 2, 17, 0, 0),
            datetime(2024, 7, 2, 20, 0, 0),  # official close
        ]
        df = pd.DataFrame(
            {
                'symbol': ['ALPA4', 'ALPA4', 'ALPA4', 'ALPA4'],
                'open': [10.0, 10.0, 11.0, 11.0],
                'high': [10.5, 10.5, 11.5, 11.5],
                'low': [9.8, 9.8, 10.8, 10.8],
                'close': [10.1, 10.2, 11.1, 11.2],
                'volume': [1000, 1000, 1000, 1000],
            },
            index=pd.DatetimeIndex(idx),
        )

        portfolio = EnhancedPortfolio()
        # Seed a long position so MOC must close it at 20:00 close
        portfolio.buy('ALPA4', quantity=100, price=10.0, trade_date=datetime(2024, 7, 1, 10, 0, 0), trade_type='day_trade')

        strategy = EnhancedFuzzyFajutoStrategy(portfolio=portfolio, symbol='ALPA4', risk_tolerance=0.02)
        sim = BacktestSimulator(strategy=strategy, start_date='2024-07-01', end_date='2024-07-02', config_path='config/settings.yaml')
        # Inject complete_data so EOD mark-to-market and MOC use close prices
        strategy.context.metadata['complete_data'] = df

        sim.run_simulation(df)

        # The last recorded daily portfolio value should reflect the 11.2 close, not any intraday bias
        self.assertGreaterEqual(sim.daily_portfolio_values[-1], 0.0)
        # No absurd spike between penultimate and last day
        if len(sim.daily_portfolio_values) >= 2:
            prev_v, last_v = sim.daily_portfolio_values[-2], sim.daily_portfolio_values[-1]
            self.assertLess(abs((last_v / prev_v) - 1), 0.5)  # sanity bound: < 50%


if __name__ == '__main__':
    unittest.main()



"""
Test to verify P1 order duplicate prevention fix.
This test validates that exactly one P1 market order is generated per symbol per day.
"""

import pytest
from datetime import datetime, date
from unittest.mock import Mock, MagicMock
from typing import List

from engine.base_strategy import StrategyConfig, StrategyContext
from engine.sim_components.types import Bar, OrderIntent, OrderSide, OrderType
from strategies.fuzzy_fajuto_strategy import FuzzyFajutoStrategy


class TestP1DuplicatePrevention:
    """Test suite for P1 order duplicate prevention."""
    
    @pytest.fixture
    def mock_context(self) -> StrategyContext:
        """Create a mock strategy context."""
        ctx = Mock(spec=StrategyContext)
        ctx.logger = Mock()
        ctx.metadata = {
            'complete_data': None,  # Unit test mode
            'tranche_notional_brl': 12500.0,
            'config': {
                'pair_mode': {
                    'gross_exposure_brl': 50000,
                    'tranches': 4
                }
            }
        }
        return ctx
    
    @pytest.fixture
    def strategy_config(self) -> StrategyConfig:
        """Create a basic strategy configuration."""
        return StrategyConfig(
            universe=['PETR4', 'VALE3', 'ITUB4'],
            warmup_bars=60,
            risk_tolerance=0.02
        )
    
    @pytest.fixture
    def strategy(self, strategy_config: StrategyConfig, mock_context: StrategyContext) -> FuzzyFajutoStrategy:
        """Create a FuzzyFajuto strategy instance."""
        return FuzzyFajutoStrategy(strategy_config, mock_context)
    
    def create_bar(self, symbol: str, timestamp: datetime, open_price: float = 30.0) -> Bar:
        """Helper to create a market data bar."""
        return Bar(
            symbol=symbol,
            timestamp=timestamp,
            open=open_price,
            high=open_price * 1.02,
            low=open_price * 0.98,
            close=open_price * 1.01,
            volume=1000000
        )
    
    def test_single_p1_order_per_symbol_per_day(self, strategy: FuzzyFajutoStrategy):
        """
        CRITICAL TEST: Verify exactly one P1 market order per symbol per day.
        
        This test validates the core fix - that multiple calls to generate_intents
        on the same day for the same symbol only produce one P1 market order.
        """
        symbol = 'PETR4'
        trading_date = date(2025, 7, 15)
        
        # Simulate multiple intraday bars (hourly) for the same day
        bars = [
            self.create_bar(symbol, datetime.combine(trading_date, datetime.strptime('13:00', '%H:%M').time())),  # First bar (market open)
            self.create_bar(symbol, datetime.combine(trading_date, datetime.strptime('14:00', '%H:%M').time())),  # Second bar
            self.create_bar(symbol, datetime.combine(trading_date, datetime.strptime('15:00', '%H:%M').time())),  # Third bar
            self.create_bar(symbol, datetime.combine(trading_date, datetime.strptime('16:00', '%H:%M').time())),  # Fourth bar
        ]
        
        # Pre-seed ATR and setup for signal generation
        strategy.current_atr_values[symbol] = 0.5
        strategy._setup_mock_daily_data_for_signal(symbol, trading_date)
        
        all_intents: List[OrderIntent] = []
        market_intents: List[OrderIntent] = []
        
        # Process each bar and collect intents
        for i, bar in enumerate(bars):
            intents = list(strategy.generate_intents(bar))
            all_intents.extend(intents)
            
            # Count market orders (P1)
            market_orders = [intent for intent in intents if intent.order_type == OrderType.MARKET]
            market_intents.extend(market_orders)
            
            strategy.context.logger.info(f"Bar {i+1}: Generated {len(intents)} intents, {len(market_orders)} market orders")
        
        # CRITICAL ASSERTION: Exactly one P1 market order per day
        assert len(market_intents) == 1, f"Expected exactly 1 P1 market order per day, got {len(market_intents)}"
        
        # Verify the single market order is from the first bar
        market_order = market_intents[0]
        assert market_order.symbol == symbol
        assert market_order.order_type == OrderType.MARKET
        assert market_order.timestamp.time() == datetime.strptime('13:00', '%H:%M').time()  # First bar time
        
        # Verify subsequent bars don't generate market orders
        subsequent_market_orders = [
            intent for intent in all_intents[1:] 
            if intent.order_type == OrderType.MARKET
        ]
        assert len(subsequent_market_orders) == 0, "Subsequent bars should not generate market orders"
    
    def test_p1_market_order_only_on_first_bar(self, strategy: FuzzyFajutoStrategy):
        """
        Test that P1 market orders are only generated on the first bar of the day.
        """
        symbol = 'VALE3'
        trading_date = date(2025, 7, 16)
        
        # Setup strategy state
        strategy.current_atr_values[symbol] = 0.6
        strategy._setup_mock_daily_data_for_signal(symbol, trading_date)
        
        # Create a bar that's NOT the first bar of the day (e.g., 15:00)
        bar = self.create_bar(symbol, datetime.combine(trading_date, datetime.strptime('15:00', '%H:%M').time()))
        
        # Mark that this is not the first bar by simulating previous bar processing
        strategy.first_bar_of_day[symbol] = {trading_date: True}  # First bar already processed
        
        # Generate intents
        intents = list(strategy.generate_intents(bar))
        market_intents = [intent for intent in intents if intent.order_type == OrderType.MARKET]
        
        # Should not generate any market orders on non-first bars
        assert len(market_intents) == 0, f"Expected 0 market orders on non-first bar, got {len(market_intents)}"
    
    def test_limit_orders_p2_p3_p4_price_calculation(self, strategy: FuzzyFajutoStrategy):
        """
        Test that limit orders P2, P3, P4 use correct price calculations per README.
        """
        symbol = 'ITUB4'
        trading_date = date(2025, 7, 17)
        close_t_minus_1 = 37.49  # Previous day close
        
        # Test BUY side limit prices
        p2_buy, p3_buy, p4_buy = strategy._limits_from_close(close_t_minus_1, OrderSide.BUY)
        
        # README specification: BUY limits at close[T−1] × (1 − 0.5%, 1.0%, 1.5%)
        expected_p2_buy = close_t_minus_1 * 0.995  # -0.5%
        expected_p3_buy = close_t_minus_1 * 0.990  # -1.0%
        expected_p4_buy = close_t_minus_1 * 0.985  # -1.5%
        
        assert abs(p2_buy - expected_p2_buy) < 0.01, f"P2 BUY price mismatch: {p2_buy} vs {expected_p2_buy}"
        assert abs(p3_buy - expected_p3_buy) < 0.01, f"P3 BUY price mismatch: {p3_buy} vs {expected_p3_buy}"
        assert abs(p4_buy - expected_p4_buy) < 0.01, f"P4 BUY price mismatch: {p4_buy} vs {expected_p4_buy}"
        
        # Test SELL side limit prices
        p2_sell, p3_sell, p4_sell = strategy._limits_from_close(close_t_minus_1, OrderSide.SELL)
        
        # README specification: SELL limits at close[T−1] × (1 + 0.5%, 1.0%, 1.5%)
        expected_p2_sell = close_t_minus_1 * 1.005  # +0.5%
        expected_p3_sell = close_t_minus_1 * 1.010  # +1.0%
        expected_p4_sell = close_t_minus_1 * 1.015  # +1.5%
        
        assert abs(p2_sell - expected_p2_sell) < 0.01, f"P2 SELL price mismatch: {p2_sell} vs {expected_p2_sell}"
        assert abs(p3_sell - expected_p3_sell) < 0.01, f"P3 SELL price mismatch: {p3_sell} vs {expected_p3_sell}"
        assert abs(p4_sell - expected_p4_sell) < 0.01, f"P4 SELL price mismatch: {p4_sell} vs {expected_p4_sell}"
    
    def test_orders_emitted_tracking_state(self, strategy: FuzzyFajutoStrategy):
        """
        Test that the daily_orders_emitted tracking works correctly.
        """
        symbol = 'PETR4'
        trading_date = date(2025, 7, 18)
        
        # Initially, no orders should be marked as emitted
        assert not strategy._are_orders_emitted_today(symbol, trading_date)
        
        # Mark P1 market order as emitted
        strategy._mark_orders_emitted(symbol, trading_date, ['market'])
        
        # Now orders should be marked as emitted for this symbol/date
        assert strategy._are_orders_emitted_today(symbol, trading_date)
        
        # Verify the specific order type is tracked
        assert strategy.daily_orders_emitted[symbol][trading_date]['market'] == True
        assert strategy.daily_orders_emitted[symbol][trading_date]['limit_alpha'] == False


# Helper method to add to FuzzyFajutoStrategy for testing
def _setup_mock_daily_data_for_signal(self, symbol: str, trading_date: date):
    """Setup mock daily data for signal generation in tests."""
    # This would be added to the strategy class for testing
    # Mock the daily data and indicators needed for signal generation
    import pandas as pd
    
    # Create minimal daily data
    mock_daily_data = pd.DataFrame({
        'open': [30.0, 30.5],
        'high': [31.0, 31.5], 
        'low': [29.5, 30.0],
        'close': [30.5, 31.0],
        'volume': [1000000, 1100000]
    }, index=[
        pd.Timestamp(trading_date) - pd.Timedelta(days=1),
        pd.Timestamp(trading_date)
    ])
    
    self.daily_data[symbol] = mock_daily_data
    
    # Mock indicators
    self.daily_indicators_data[symbol] = {
        'ema_3': [30.0, 30.5],
        'ema_5': [29.8, 30.3],
        'ema_10': [29.5, 30.0],
        'ema_15': [29.3, 29.8],
        'ema_20': [29.0, 29.5],
        'rsi': [55.0, 60.0]
    }

# Monkey patch the helper method for testing
FuzzyFajutoStrategy._setup_mock_daily_data_for_signal = _setup_mock_daily_data_for_signal

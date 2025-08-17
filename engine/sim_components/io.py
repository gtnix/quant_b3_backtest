from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from engine.base_strategy import DataPortal, OrderIntent, OrderSide, BrokerSimulation


class SimulationDataPortal(DataPortal):
    """Concrete implementation of DataPortal for backtesting simulation."""

    def __init__(self, data_loader=None):
        self.data_loader = data_loader
        self._current_data = None
        self._historical_data = {}

    def set_current_data(self, data: pd.DataFrame):
        self._current_data = data

    def get_current_price(self, symbol: str) -> Optional[float]:
        if self._current_data is not None and 'close' in self._current_data.columns:
            return float(self._current_data['close'].iloc[-1])
        return None

    def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        if self.data_loader:
            try:
                return self.data_loader.load_raw_data(symbol)
            except Exception:
                return None
        return self._historical_data.get(symbol)

    def get_market_data(self, symbol: str, timestamp: datetime) -> Optional[Dict[str, Any]]:
        if self._current_data is not None:
            try:
                if timestamp in self._current_data.index:
                    row = self._current_data.loc[timestamp]
                    return {
                        'open': float(row.get('open', 0)),
                        'high': float(row.get('high', 0)),
                        'low': float(row.get('low', 0)),
                        'close': float(row.get('close', 0)),
                        'volume': int(row.get('volume', 0)),
                        'timestamp': timestamp,
                    }
            except Exception:
                pass
        return None


class SimulationBroker(BrokerSimulation):
    """Concrete implementation of BrokerSimulation for backtesting."""

    def __init__(self, portfolio):
        self.portfolio = portfolio
        self._order_counter = 0
        self._pending_orders = {}

    def submit_order(self, intent: OrderIntent) -> str:
        self._order_counter += 1
        order_id = f"sim_order_{self._order_counter}"
        self._pending_orders[order_id] = intent
        return order_id

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        if symbol in self.portfolio.positions:
            pos = self.portfolio.positions[symbol]
            return {
                'symbol': symbol,
                'quantity': pos.quantity,
                'avg_price': pos.avg_price,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl,
            }
        return None

    def get_cash_balance(self) -> float:
        return getattr(self.portfolio, 'cash', 0.0)

    def get_portfolio_value(self) -> float:
        return self.portfolio.get_portfolio_value()

    def execute_pending_orders(self, current_prices: Dict[str, float]) -> List[str]:
        executed_orders = []
        for order_id, intent in list(self._pending_orders.items()):
            try:
                execution_price = current_prices.get(intent.symbol, intent.price)
                if execution_price:
                    if intent.side == OrderSide.BUY:
                        success = self.portfolio.buy(
                            ticker=intent.symbol,
                            quantity=intent.quantity,
                            price=execution_price,
                            trade_date=intent.timestamp,
                            trade_id=order_id,
                        )
                    else:
                        success = self.portfolio.sell(
                            ticker=intent.symbol,
                            quantity=intent.quantity,
                            price=execution_price,
                            trade_date=intent.timestamp,
                            trade_id=order_id,
                        )
                    if success:
                        executed_orders.append(order_id)
                        del self._pending_orders[order_id]
            except Exception:
                pass
        return executed_orders



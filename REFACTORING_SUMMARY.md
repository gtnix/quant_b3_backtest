# Brazilian Trade Classification Refactoring Summary

## Overview

This document summarizes the refactoring changes made to correctly implement Brazilian day trade vs swing trade classification and tax handling according to Brazilian tax law requirements.

## Key Changes Implemented

### 1. BUY Flow Refactoring

**Before**: BUY orders were immediately added to both intraday FIFO and swing inventory.

**After**: BUY orders are only added to intraday FIFO initially. Swing inventory is populated at day close rollover.

```python
# Before (incorrect)
if action == 'BUY':
    self.intraday_fifo[(day_id, ticker)].append(buy_block)
    self._add_to_swing_inventory(ticker, quantity, price, date, broker)  # Immediate

# After (correct)
if action == 'BUY':
    self.intraday_fifo[(day_id, ticker)].append(buy_block)
    # DO NOT add to swing inventory immediately - will be done at day close rollover
```

### 2. Day Close Rollover

**New Feature**: Added `rollover_day()` method to move remaining intraday quantities to swing inventory at day close.

```python
def rollover_day(self, day_id: str) -> None:
    """Rollover remaining intraday quantities to swing inventory at day close."""
    # Find all tickers with intraday FIFO for this day
    day_tickers = [ticker for (d, ticker) in self.intraday_fifo.keys() if d == day_id]
    
    for ticker in day_tickers:
        fifo_key = (day_id, ticker)
        buy_blocks = self.intraday_fifo[fifo_key]
        
        # Process remaining quantities from each buy block
        for buy_block in buy_blocks:
            if buy_block.remaining_qty > 0:
                self._add_to_swing_inventory(
                    ticker=ticker,
                    quantity=buy_block.remaining_qty,
                    price=buy_block.price,
                    date=buy_block.date,
                    broker=buy_block.broker
                )
        
        # Clear the intraday FIFO for this day/ticker
        del self.intraday_fifo[fifo_key]
```

### 3. Enhanced Swing Inventory Management

**Updated**: `_add_to_swing_inventory()` method now uses aggregated cost basis (custo médio).

```python
def _add_to_swing_inventory(self, ticker: str, quantity: int, price: float, date: datetime, broker: str = "default") -> None:
    inv = self.swing_inventory.get(ticker)
    if inv is None:
        self.swing_inventory[ticker] = SwingInventory(ticker, quantity, price, date, broker)
    else:
        # Update existing aggregated record
        total_qty = inv.quantity + quantity
        total_cost = inv.quantity * inv.avg_cost + quantity * price
        inv.avg_cost = total_cost / total_qty
        inv.quantity = total_qty
        inv.last_update = date
```

### 4. Asset Type Filtering for Exemption

**New Feature**: Added `asset_type` field to `ClassifiedTrade` and filtering for R$ 20k exemption.

```python
@dataclass
class ClassifiedTrade:
    # ... existing fields ...
    asset_type: str = "STOCK"  # 'STOCK', 'ETF', 'FII', etc. - for exemption filtering
```

**Updated**: Exemption calculation now only counts STOCK assets:

```python
# Apply swing trade exemption - ONLY for STOCK assets
if modality == 'SWING':
    # Calculate total swing sales for STOCK assets only for exemption
    stock_swing_sales = 0.0
    for trade in classified_trades:
        if (trade.date.strftime('%Y-%m') == month and 
            trade.action == 'SELL' and 
            trade.swing_trade_qty > 0 and
            trade.asset_type == 'STOCK'):
            stock_swing_sales += trade.swing_trade_value
    
    if stock_swing_sales <= exemption_limit:
        summary.exemption_applied = True
        summary.exemption_amount = stock_swing_sales
        summary.taxable_profit = 0.0
```

### 5. Fixed Profit Calculation

**Updated**: `_calculate_profit_from_audit()` method now uses correct audit steps:

```python
def _calculate_profit_from_audit(self, trade: ClassifiedTrade, trade_type: str) -> float:
    total_profit = 0.0
    
    for audit_entry in trade.classification_audit:
        if trade_type == 'day_trade' and audit_entry.get('step') == 'intraday_fifo_consumption':
            # Day trade profit calculation
            buy_price = audit_entry.get('buy_price', 0.0)
            sell_price = audit_entry.get('sell_price', 0.0)
            qty = audit_entry.get('qty', 0)
            profit = (sell_price - buy_price) * qty
            total_profit += profit
            
        elif trade_type == 'swing_trade' and audit_entry.get('step') == 'swing_fifo_consumption':
            # Swing trade profit calculation
            buy_price = audit_entry.get('buy_price', 0.0)  # avg_cost from swing inventory
            sell_price = audit_entry.get('sell_price', 0.0)
            qty = audit_entry.get('qty', 0)
            profit = (sell_price - buy_price) * qty
            total_profit += profit
    
    return total_profit
```

### 6. Enhanced Unmatched Trade Handling

**Updated**: `_consume_swing_fifo()` method now properly handles unmatched sells:

```python
if ticker not in self.swing_inventory:
    # No swing inventory available - this is an unmatched sell
    audit.append({
        'step': 'swing_fifo_consumption',
        'buy_id': f"unmatched_{ticker}",
        'sell_id': sell_trade_id,
        'qty': 0,
        'buy_price': 0.0,
        'sell_price': sell_price,
        'modality': 'SWING',
        'remaining_qty': 0,
        'reason': f"Unmatched swing trade: {sell_qty} {ticker} @ R$ {sell_price:.2f} (no swing inventory available)"
    })
```

### 7. Updated Method Signatures

**Enhanced**: Added `asset_type` parameter to buy/sell methods:

```python
def buy(self, ticker: str, quantity: int, price: float, 
        trade_date: datetime, trade_type: str = "swing_trade",
        trade_id: Optional[str] = None, description: str = "", 
        asset_type: str = "STOCK") -> bool:

def sell(self, ticker: str, quantity: int, price: float, 
         trade_date: datetime, trade_type: str = "swing_trade",
         trade_id: Optional[str] = None, description: str = "", 
         asset_type: str = "STOCK") -> bool:
```

## Data Flow Summary

### BUY Flow
1. Create `BuyBlock` and add to `intraday_fifo[(day_id, ticker)]`
2. **DO NOT** add to swing inventory immediately
3. At day close, call `rollover_day()` to move remaining quantities to swing inventory

### SELL Flow
1. Consume from `intraday_fifo` first (DT portion)
2. Consume remaining from `swing_inventory` using avg_cost (Swing portion)
3. Handle unmatched quantities with proper audit trail

### Day Close Rollover
1. Process all remaining quantities from intraday FIFO
2. Add to swing inventory using aggregated cost basis
3. Clear intraday FIFO for completed days

## Test Coverage

The refactoring includes comprehensive tests covering:

1. **Classic Day Trade**: BUY 100 @10h, SELL 100 @16h → 100 DT, 0 Swing
2. **Sell Before Buy**: SELL 50 @10h, BUY 50 @15h → Unmatched (no swing inventory)
3. **Partial DT + Swing**: BUY 100 @10h, SELL 40 @16h → 40 DT; rollover 60 to swing
4. **Custo Médio**: Multiple buys → aggregated cost basis calculation
5. **Exemption Filtering**: Only STOCK assets count for R$ 20k exemption
6. **IRRF Calculation**: 1% on positive daily DT profits only
7. **Day Close Rollover**: Proper movement of remaining quantities

## Files Modified

1. **`engine/portfolio.py`**:
   - Updated `ClassifiedTrade` dataclass with `asset_type`
   - Refactored BUY flow in `_classify_single_trade_refactored()`
   - Added `rollover_day()` method
   - Updated `_add_to_swing_inventory()` method
   - Enhanced `_consume_swing_fifo()` with unmatched handling
   - Fixed `_calculate_profit_from_audit()` method
   - Updated `aggregate_monthly()` with asset type filtering
   - Added `asset_type` parameter to buy/sell methods

2. **`engine/loss_manager.py`**:
   - Added `asset_type` parameter to `record_trade_result()`
   - Added `get_monthly_swing_sales_by_asset_type()` method

3. **`test_enhanced_classification.py`**:
   - Comprehensive test suite covering all scenarios
   - Updated tests to use new method signatures

## Compliance with Brazilian Tax Law

The refactoring ensures compliance with:

- **DT Classification**: Same ticker, buy and sell on same trading day
- **FIFO Rules**: DT pairing within same day, swing inventory across days
- **Custo Médio**: Aggregated cost basis for swing inventory
- **R$ 20k Exemption**: Only applies to STOCK assets
- **IRRF Calculation**: 1% on positive daily DT profits, 0.005% on swing sales
- **Day Close Rollover**: Proper handling of remaining intraday quantities

## Ambiguities and Edge Cases

1. **Unmatched Sells**: When selling before buying, the system marks these as unmatched rather than creating negative inventory
2. **Transaction Costs**: IRRF calculation includes transaction costs which reduce actual profits
3. **Asset Type Tracking**: Currently defaults to "STOCK" - may need enhancement for other asset types

## Next Steps

1. **Enhanced Asset Type Tracking**: Implement proper asset type detection and validation
2. **Performance Optimization**: Consider caching for frequently accessed data
3. **Audit Trail Enhancement**: Add more detailed audit information for regulatory compliance
4. **Error Handling**: Improve error handling for edge cases and invalid inputs 
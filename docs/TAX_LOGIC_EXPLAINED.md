# Brazilian Tax Logic Explained - Beginner's Guide

## Introduction

This document explains the Brazilian tax rules implemented in the quant_b3_backtest system in simple terms. Understanding these rules is crucial for accurate backtesting of Brazilian market strategies.

## Key Concepts

### 1. Two Types of Trading

**Swing Trade (Operações Comuns):**
- Buy and sell on different days
- Tax rate: 15% on monthly net profit
- Exemption: If monthly sales ≤ R$ 20,000, no tax on profit

**Day Trade (Operações Day Trade):**
- Buy and sell the same asset on the same day
- Tax rate: 20% on monthly net profit
- No exemption available

### 2. IRRF (Withholding Tax)
- **What it is:** Tax withheld by the broker
- **How it works:** Acts as a credit against your final tax bill
- **Swing trades:** 0.005% on each sale
- **Day trades:** 1% on daily net profit per asset

## Tax Calculation Step by Step

### Step 1: Record the Trade

When you sell an asset, the system first records the trade result:

```python
# From loss_manager.py:169-245
def record_trade_result(self, ticker, trade_profit, trade_date, modality, gross_sales):
    # Track monthly sales and profits
    month_key = trade_date.strftime('%Y-%m')
    
    if modality == "SWING":
        if gross_sales > 0:
            self.monthly_swing_sales[month_key] += gross_sales
        if trade_profit > 0:
            self.monthly_swing_profits[month_key] += trade_profit
    else:  # DAY
        if trade_profit > 0:
            self.monthly_day_profits[month_key] += trade_profit
```

**What this does:**
- Keeps track of how much you sold each month (for swing trades)
- Keeps track of how much profit you made each month
- Separates swing trades from day trades

### Step 2: Check for Exemption (Swing Trades Only)

For swing trades, the system checks if you qualify for the R$ 20,000 exemption:

```python
# From loss_manager.py:246-350
def calculate_taxable_amount(self, gross_profit, modality, gross_sales, month_ref):
    # Check swing trade exemption
    if modality == "SWING" and gross_sales <= self.swing_exemption_limit:
        logger.info(f"Swing trade exemption applied: sales R$ {gross_sales:,.2f} ≤ R$ {self.swing_exemption_limit:,.2f}")
        return 0.0, []  # Tax exempt, no loss consumption
```

**What this means:**
- If your monthly swing trade sales ≤ R$ 20,000 → No tax on profit
- If your monthly swing trade sales > R$ 20,000 → Tax applies
- Day trades never get this exemption

### Step 3: Apply Loss Carryforward

If you have losses from previous trades, they can offset your current profits:

```python
# From loss_manager.py:246-350
def calculate_taxable_amount(self, gross_profit, modality, gross_sales, month_ref):
    # Apply loss carryforward using FIFO with month boundary checking
    taxable = gross_profit
    audit_log = []
    fifo = self.loss_fifo[modality]
    
    while taxable > 0 and fifo:
        loss_rec = fifo[0]
        available_loss = loss_rec.amount - loss_rec.applied_amount
        
        # CRITICAL: Check month boundaries - "No Going Backwards" rule
        loss_month = loss_rec.date.replace(day=1)
        profit_month = month_ref.replace(day=1)
        
        if loss_month_date > profit_month_date:
            # Loss is from a later month - cannot apply
            break
```

**Key Rules:**
- **100% Offset:** You can use 100% of your losses to offset profits
- **FIFO:** Oldest losses are used first
- **No Going Backwards:** Losses from March can't offset profits from February
- **Per Modality:** Swing losses only offset swing profits, day losses only offset day profits

### Step 4: Calculate Capital Gains Tax

Once we know the taxable profit, we calculate the tax:

```python
# From portfolio.py:285-380
def _calculate_taxes(self, profit, trade_type, gross_sales, trade_date, gross_profit):
    # Calculate capital gains tax
    if trade_type == 'day_trade':
        capital_gains_rate = taxes['day_trade']  # 20%
    else:
        capital_gains_rate = taxes['swing_trade']  # 15%
    
    capital_gains_tax = taxable_profit * capital_gains_rate
```

**Tax Rates:**
- Swing trades: 15% of taxable profit
- Day trades: 20% of taxable profit

### Step 5: Calculate IRRF (Withholding Tax)

IRRF is calculated differently for each trade type:

```python
# From portfolio.py:285-380
def _calculate_taxes(self, profit, trade_type, gross_sales, trade_date, gross_profit):
    # Calculate IRRF withholding for this specific trade
    if trade_type == 'day_trade':
        # Day trade: IRRF 1% on daily net profit per asset (if positive)
        if gross_profit is not None and gross_profit > 0:
            irrf_withholding = gross_profit * taxes['irrf_day_rate']  # 1%
        else:
            irrf_withholding = 0.0
    else:
        # Swing trade: IRRF 0.005% on each sale (regardless of profit/loss)
        irrf_withholding = gross_sales * taxes['irrf_swing_rate']  # 0.005%
```

**IRRF Rules:**
- **Swing trades:** 0.005% on every sale (even if you lost money)
- **Day trades:** 1% on daily profit per asset (only if you made money)
- **R$ 1.00 Rule:** If swing trade IRRF ≤ R$ 1.00, it's waived

### Step 6: Calculate Final Tax Liability

The final tax you owe is calculated as:

```python
# From portfolio.py:285-380
def _calculate_taxes(self, profit, trade_type, gross_sales, trade_date, gross_profit):
    # Calculate final tax liability for this trade (capital gains tax - IRRF credit)
    final_tax_liability = max(0.0, capital_gains_tax - irrf_withholding)
    
    # Total taxes for this trade = final tax liability + IRRF withholding
    total_taxes = final_tax_liability + irrf_withholding
```

**Formula:**
- Final Tax = Capital Gains Tax - IRRF Credit
- If IRRF > Capital Gains Tax → You get a refund
- If IRRF < Capital Gains Tax → You owe the difference

## Monthly Tax Calculation

At the end of each month, the system calculates your total tax liability:

```python
# From portfolio.py:378-482
def calculate_monthly_tax_liability(self, month_ref, trade_type=None):
    # Get monthly aggregated data
    monthly_swing_sales = self.loss_manager.get_monthly_swing_sales(month_ref)
    monthly_swing_profits = self.loss_manager.get_monthly_swing_profits(month_ref)
    monthly_day_profits = self.loss_manager.get_monthly_day_profits(month_ref)
    
    # Calculate capital gains taxes
    swing_capital_gains_tax = swing_taxable_profit * taxes['swing_trade']  # 15%
    day_capital_gains_tax = day_taxable_profit * taxes['day_trade']  # 20%
    total_capital_gains_tax = swing_capital_gains_tax + day_capital_gains_tax
    
    # Calculate total IRRF credits for the month
    swing_irrf_credit = monthly_swing_sales * taxes['irrf_swing_rate']  # 0.005%
    day_irrf_credit = self.loss_manager.calculate_day_trade_irrf(month_ref)  # 1% per asset per day
    
    # Calculate final DARF liability
    final_darf_liability = max(0.0, total_capital_gains_tax - total_irrf_credit)
```

**Monthly Process:**
1. Add up all swing trade profits for the month
2. Add up all day trade profits for the month
3. Apply loss carryforward to each type
4. Calculate tax on remaining profits
5. Subtract total IRRF credits
6. This is your DARF (tax payment)

## Real-World Example

Let's say you made these trades in March 2025:

**Swing Trades:**
- Sold PETR4 for R$ 15,000 profit (monthly sales: R$ 50,000)
- Sold VALE3 for R$ 5,000 loss

**Day Trades:**
- Day traded ITUB4 for R$ 2,000 profit
- Day traded BBDC4 for R$ 1,000 loss

**Tax Calculation:**
1. **Swing Trade Tax:**
   - Monthly sales: R$ 50,000 (> R$ 20,000, so no exemption)
   - Net profit: R$ 15,000 - R$ 5,000 = R$ 10,000
   - Tax: R$ 10,000 × 15% = R$ 1,500
   - IRRF: R$ 50,000 × 0.005% = R$ 2.50

2. **Day Trade Tax:**
   - Net profit: R$ 2,000 - R$ 1,000 = R$ 1,000
   - Tax: R$ 1,000 × 20% = R$ 200
   - IRRF: R$ 2,000 × 1% = R$ 20 (calculated per asset per day)

3. **Final Tax:**
   - Total Capital Gains Tax: R$ 1,500 + R$ 200 = R$ 1,700
   - Total IRRF Credit: R$ 2.50 + R$ 20 = R$ 22.50
   - Final DARF: R$ 1,700 - R$ 22.50 = R$ 1,677.50

## Important Notes

### Loss Carryforward Rules
- Losses never expire (perpetual carryforward)
- You can't use future losses for past profits
- Swing and day trade losses are separate
- You can offset 100% of profits with losses

### IRRF Rules
- IRRF is a credit, not a deduction
- R$ 1.00 minimum rule only applies to swing trades
- Day trade IRRF is calculated per asset per day (1% on positive daily profit per asset)
- IRRF is withheld by the broker automatically
- **Important:** Day trade IRRF is now correctly calculated per asset per day (not monthly total)

### Exemption Rules
- Only applies to swing trades
- Based on monthly sales amount, not profit
- If you exceed R$ 20,000 in sales, you lose the exemption for that month
- Day trades never get this exemption

## Common Mistakes to Avoid

1. **Thinking IRRF is a deduction:** It's a credit against your final tax bill
2. **Forgetting the R$ 20,000 exemption:** Check your monthly sales carefully
3. **Mixing swing and day trade losses:** They can't offset each other
4. **Ignoring the "no going backwards" rule:** Future losses can't offset past profits
5. **Forgetting the R$ 1.00 IRRF rule:** Small swing trade IRRF amounts are waived

## Conclusion

The Brazilian tax system for individual investors is complex but the quant_b3_backtest system handles it correctly. The key is understanding that:

- **Swing trades** get a R$ 20,000 exemption but pay 15% tax
- **Day trades** pay 20% tax with no exemption
- **Losses** can offset 100% of profits and never expire
- **IRRF** is a credit, not a deduction
- **Monthly aggregation** is crucial for proper calculation
- **Day trade IRRF** is now correctly calculated per asset per day (not monthly total)

This system ensures your backtests accurately reflect the real tax implications of your trading strategies with proper Brazilian regulatory compliance. 
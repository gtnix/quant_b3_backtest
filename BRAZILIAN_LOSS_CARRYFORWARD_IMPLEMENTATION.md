# Brazilian Loss Carryforward Implementation (2025)

## Regulatory Compliance

This implementation follows Brazilian capital market regulations:
- **Brazilian Securities Commission (CVM)** - Current 2025 regulations
- **Brazilian Tax Authority (Receita Federal)** - Current 2025 regulations

## 🎯 **Core Regulatory Requirements**

### **1. Maximum 30% Loss Offset**
- Capital losses can offset a maximum of 30% of current capital gains
- This is a hard limit enforced by Brazilian tax law
- No exceptions or workarounds allowed

### **2. Capital Gains Only Restriction**
- Losses can ONLY be offset against capital gains
- Cannot offset against ordinary income or other revenue streams
- Strict separation between capital and ordinary income

### **3. Perpetual Loss Carryforward**
- Unused capital losses are carried forward indefinitely
- No time limit or expiration date
- Losses accumulate until fully utilized

### **4. CVM Compliance**
- Comprehensive audit trail requirements
- Detailed loss tracking and application history
- Regulatory reporting capabilities

## 🚀 **Implementation Details**

### **Core Method: `calculate_loss_carryforward`**

```python
def calculate_loss_carryforward(
    self, 
    current_capital_gains: float, 
    accumulated_capital_losses: Optional[float] = None
) -> Dict[str, float]:
```

**Parameters:**
- `current_capital_gains`: Total capital gains in the current period
- `accumulated_capital_losses`: Total accumulated capital losses (optional)

**Returns:**
- `taxable_gains`: Gains after loss offset (cannot be negative)
- `remaining_losses`: Losses carried forward to next period
- `loss_offset_applied`: Amount of losses actually applied
- `offset_percentage`: Percentage of gains offset (max 30%)
- `max_offset_allowed`: Maximum offset allowed under 30% rule
- `regulatory_compliance`: Compliance identifier

### **Key Algorithm**

```python
# Brazilian tax rule: Maximum 30% loss offset against capital gains
LOSS_OFFSET_PERCENTAGE = 0.30

# Calculate maximum available loss offset (30% of current gains)
max_loss_offset = current_capital_gains * LOSS_OFFSET_PERCENTAGE

# Calculate actual loss offset (limited by available losses and 30% rule)
actual_loss_offset = min(
    accumulated_capital_losses,
    max_loss_offset,
    current_capital_gains  # Cannot offset more than the gains
)

# Calculate taxable gains after loss offset
taxable_gains = current_capital_gains - actual_loss_offset

# Calculate remaining losses to carry forward
remaining_losses = accumulated_capital_losses - actual_loss_offset
```

## 📊 **Example Scenarios**

### **Scenario 1: Partial Loss Offset (30% Rule)**
```python
# Input
current_capital_gains = 1000.0
accumulated_capital_losses = 10000.0

# Result
taxable_gains = 700.0        # 1000 - 300 (30% of 1000)
remaining_losses = 9700.0    # 10000 - 300
loss_offset_applied = 300.0  # 30% of gains
offset_percentage = 30.0     # 300/1000 * 100
```

### **Scenario 2: Complete Loss Offset**
```python
# Input
current_capital_gains = 1000.0
accumulated_capital_losses = 200.0

# Result
taxable_gains = 800.0        # 1000 - 200 (all available losses)
remaining_losses = 0.0       # 200 - 200
loss_offset_applied = 200.0  # All available losses
offset_percentage = 20.0     # 200/1000 * 100
```

### **Scenario 3: Zero Gains**
```python
# Input
current_capital_gains = 0.0
accumulated_capital_losses = 5000.0

# Result
taxable_gains = 0.0          # No gains to offset
remaining_losses = 5000.0    # All losses carried forward
loss_offset_applied = 0.0    # No offset possible
offset_percentage = 0.0      # No offset percentage
```

## 🔧 **Configuration**

### **Settings.yaml Configuration**
```yaml
loss_carryforward:
  max_tracking_years: null              # null = perpetual carryforward
  global_loss_limit: null               # No limit on accumulated losses
  auto_prune_old_losses: false          # Disabled - losses never expire
  
  regulatory_compliance:
    max_offset_percentage: 0.30         # Maximum 30% loss offset
    capital_gains_only: true            # Losses can ONLY offset capital gains
    perpetual_carryforward: true        # No time limit for loss carryforward
    cvm_compliance: true                # CVM compliance
    receita_federal_compliance: true    # Receita Federal compliance
```

### **Class Constants**
```python
self.LOSS_OFFSET_PERCENTAGE = 0.30  # Maximum 30% loss offset
self.CAPITAL_GAINS_ONLY = True      # Losses can ONLY offset capital gains
self.PERPETUAL_CARRYFORWARD = True  # No time limit for loss carryforward
```

## 🧪 **Testing**

### **Comprehensive Test Coverage**
- ✅ 30% maximum offset rule validation
- ✅ Perpetual loss carryforward functionality
- ✅ Negative gains validation
- ✅ Regulatory compliance constants
- ✅ Edge cases (large losses vs small gains, etc.)
- ✅ Integration with existing portfolio management

### **Running Tests**
```bash
# Run comprehensive test suite
python3 test_loss_carryforward.py

# Run specific test
python3 -c "from engine.loss_manager import main; main()"
```

## 📈 **Performance Considerations**

### **Optimization Features**
- **Caching**: LRU cache for asset loss balance calculations
- **Lazy Loading**: Loss records loaded on demand
- **Memory Optimization**: Efficient data structures
- **Audit Trail Compression**: Compressed audit trail storage

### **Computational Efficiency**
- **O(1)**: Loss carryforward calculation
- **O(n)**: Asset-specific loss tracking (where n = number of assets)
- **O(log n)**: Cached balance calculations

## 🔍 **Audit Trail & Compliance**

### **Comprehensive Audit Trail**
```json
{
  "loss_records": [...],
  "application_history": [...],
  "summary": {
    "total_cumulative_loss": 2550.0,
    "total_losses_recorded": 4,
    "assets_with_losses": 3,
    "regulatory_compliance": "CVM_2025_BRAZILIAN_CAPITAL_MARKETS"
  },
  "export_date": "2025-01-15T10:30:00"
}
```

### **Regulatory Reporting**
- Detailed loss application history
- Per-asset loss tracking
- Temporal loss management
- Compliance validation

## 🚨 **Error Handling**

### **Input Validation**
- Capital gains cannot be negative
- Comprehensive type checking
- Meaningful error messages
- Defensive programming practices

### **Edge Cases Handled**
- Zero gains scenarios
- Very large loss vs small gain scenarios
- Very large gain vs small loss scenarios
- Perpetual carryforward edge cases

## 🔄 **Integration Points**

### **Portfolio Management**
- Seamless integration with `EnhancedPortfolio`
- Automatic loss carryforward application
- Real-time loss balance updates

### **Settlement Management**
- T+2 settlement cycle compliance
- Loss carryforward with settlement timing
- Business day calculations

### **Transaction Cost Analysis**
- Loss carryforward in cost calculations
- Tax impact analysis
- Performance attribution

## 📋 **Migration Guide**

### **From Previous Implementation**
1. **Update Configuration**: Set `max_tracking_years: null` for perpetual carryforward
2. **Update Method Calls**: Use `calculate_loss_carryforward()` for regulatory compliance
3. **Verify Constants**: Ensure regulatory constants are properly set
4. **Run Tests**: Execute comprehensive test suite

### **Backward Compatibility**
- Existing `calculate_taxable_amount()` method updated to use new logic
- All existing interfaces maintained
- Gradual migration path available

## 🎯 **Next Steps & Optimizations**

### **Immediate Enhancements**
1. **Real-time Processing**: Enable real-time loss carryforward calculations
2. **Parallel Processing**: Optimize for high-frequency trading scenarios
3. **Advanced Caching**: Implement more sophisticated caching strategies

### **Future Regulatory Updates**
1. **Dynamic Compliance**: Support for regulatory changes
2. **Multi-jurisdiction**: Extend to other markets
3. **Advanced Reporting**: Enhanced regulatory reporting capabilities

## 📞 **Support & Maintenance**

### **Code Locations**
- **Main Implementation**: `engine/loss_manager.py`
- **Configuration**: `config/settings.yaml`
- **Tests**: `tests/test_enhanced_managers.py`
- **Documentation**: This file

### **Key Methods**
- `calculate_loss_carryforward()`: Core regulatory-compliant calculation
- `calculate_taxable_amount()`: Updated to use new logic
- `record_trade_result()`: Loss recording with audit trail
- `export_audit_trail()`: Compliance reporting

---

**Compliance Status**: ✅ CVM 2025, ✅ Receita Federal, ✅ 30% Max Offset, ✅ Perpetual Carryforward

**Last Updated**: January 2025
**Version**: 2.0.0 (Brazilian Regulatory Compliance) 
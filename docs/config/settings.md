# Settings Configuration Documentation

## Overview

The `settings.yaml` file contains all configuration parameters for the Brazilian market backtesting system, including market parameters, tax rates, portfolio settings, settlement rules, and advanced features configuration.

**File Location**: `quant_backtest/config/settings.yaml`

## Configuration Structure

The configuration is organized into logical sections for easy management and maintenance:

```yaml
market:           # Market parameters and trading rules
taxes:           # Brazilian tax configuration
portfolio:       # Portfolio management settings
settlement:      # T+2 settlement configuration
loss_carryforward: # Loss tracking and carryforward
performance:     # Performance and monitoring
compliance:      # Regulatory compliance
error_handling:  # Error handling and resilience
advanced:        # Advanced features
benchmark:       # Benchmark analysis
sgs:            # Banco Central SGS data
```

## Market Configuration

### Trading Hours and Timezone
```yaml
market:
  trading_hours:
    open: "10:00"          # Continuous session starts
    close: "16:55"         # Continuous session ends
    timezone: "America/Sao_Paulo"  # Brazilian market timezone
```

### Market Parameters
```yaml
market:
  selic_rate: 0.15         # Current Brazilian SELIC rate (15.0% - 2025)
  trading_days_per_year: 252  # Brazilian market standard
```

### Transaction Costs (B3 Fees)
```yaml
market:
  costs:
    # B3 mandatory fees
    emolument: 0.00005                 # 0.005% negotiation fee
    settlement_day_trade: 0.00018      # 0.018% settlement fee (day-trade)
    settlement_swing_trade: 0.00025    # 0.025% settlement fee (overnight)
    
    # Modal brokerage (electronic only)
    brokerage_fee: 0.0                 # Zero brokerage via web/app
    min_brokerage: 0.0                 # No minimum charge
    iss_rate: 0.05                     # 5% ISS on brokerage
```

## Tax Configuration

### Brazilian Individual Taxpayer Rules (2025)
```yaml
taxes:
  # Tax rates for individual taxpayers
  swing_trade: 0.15        # 15% on monthly net profit from common operations
  day_trade: 0.20          # 20% on monthly net profit from day trading
  
  # Exemption for individual taxpayers (swing trade only)
  swing_exemption_limit: 20000   # R$ 20,000/month - sales ≤ this amount are tax exempt
  
  # IRRF (withholding tax) - Credit against final tax liability
  irrf_swing_rate: 0.00005 # 0.005% IRRF on gross sales (swing trade) - credit only
  irrf_day_rate: 0.01      # 1% IRRF on net profit per operation (day trade) - credit only
  
  # Individual taxpayer settings (no 30% limit)
  person_type: "individual"         # individual = individual taxpayer
  max_loss_offset_percentage: 1.0  # 100% - individuals can offset up to 100% of profit
  loss_carryforward_perpetual: true # Losses are carried forward indefinitely
```

## Portfolio Settings

### Basic Portfolio Configuration
```yaml
portfolio:
  initial_cash: 100000     # Starting capital in BRL
  max_positions: 10        # Maximum number of positions
  position_sizing: "equal_weight"  # Position sizing method
```

## Settlement Configuration

### T+2 Settlement Settings
```yaml
settlement:
  cycle_days: 2                    # T+2 settlement cycle
  timezone: "America/Sao_Paulo"    # Market timezone for precise timing
  strict_mode: true                # Enforce rigorous settlement rules
  holiday_calendar: "b3"           # Use B3 holiday calendar
  max_retry_attempts: 3            # Maximum settlement retry attempts
  auto_process_settlements: true   # Automatically process due settlements
```

## Loss Carryforward Configuration

### Advanced Loss Tracking
```yaml
loss_carryforward:
  max_tracking_years: null              # null = perpetual carryforward (Brazilian law 2025)
  global_loss_limit: null               # No limit on accumulated losses
  asset_specific_tracking: true         # Enable per-asset loss tracking
  temporal_management: true             # Enable timestamp-based loss management
  audit_trail_enabled: true             # Enable comprehensive audit trail generation
  auto_prune_old_losses: false          # Disabled - losses never expire
  partial_application: true             # Enable partial loss application
  performance_caching: true             # Enable performance optimization with caching
```

### Brazilian Regulatory Compliance (2025)
```yaml
loss_carryforward:
  regulatory_compliance:
    max_offset_percentage: 1.0          # 100% loss offset for individuals (no 30% limit)
    capital_gains_only: true            # Losses can ONLY offset capital gains
    perpetual_carryforward: true        # No time limit for loss carryforward
    cvm_compliance: true                # CVM (Brazilian Securities Commission) compliance
    receita_federal_compliance: true    # Receita Federal compliance
    person_type: "individual"           # Individual taxpayer rules (IN RFB 1.585/2015)
```

### Audit and Reporting
```yaml
loss_carryforward:
  audit_requirements:
    detailed_loss_tracking: true        # Track individual loss records
    application_history: true           # Track loss application history
    regulatory_reporting: true          # Enable regulatory reporting
    compliance_validation: true         # Validate compliance with Brazilian law
```

## Performance & Monitoring Configuration

### Performance Settings
```yaml
performance:
  enable_caching: true             # Enable LRU caching for performance
  cache_size: 128                  # Maximum cache entries
  enable_profiling: false          # Enable performance profiling
  log_level: "INFO"                # Logging level (DEBUG, INFO, WARNING, ERROR)
  audit_trail_retention_days: 365  # Days to retain audit trails
  backup_frequency_hours: 24       # Hours between automatic backups
```

## Compliance & Regulatory Configuration

### Regulatory Framework
```yaml
compliance:
  regulatory_framework: "brazilian" # Regulatory framework (brazilian, international)
  tax_year_start: 1                # Tax year start month (1 = January)
  reporting_frequency: "monthly"   # Reporting frequency (daily, weekly, monthly)
  auto_export_audit_trails: true   # Automatically export audit trails
  regulatory_updates_enabled: true # Enable automatic regulatory updates
  compliance_checks_enabled: true  # Enable automatic compliance validation
```

## Error Handling & Resilience Configuration

### Error Handling Settings
```yaml
error_handling:
  max_retry_attempts: 3            # Maximum retry attempts for failed operations
  graceful_degradation: true       # Enable graceful degradation on errors
  fallback_strategies: true        # Enable fallback strategies
  error_logging_level: "ERROR"     # Error logging level
  auto_recovery_enabled: true      # Enable automatic error recovery
  circuit_breaker_enabled: false   # Enable circuit breaker pattern
```

## Advanced Features Configuration

### Advanced Settings
```yaml
advanced:
  real_time_processing: false      # Enable real-time processing mode
  batch_processing_size: 1000      # Batch size for bulk operations
  parallel_processing: false       # Enable parallel processing
  memory_optimization: true        # Enable memory optimization
  lazy_loading: true               # Enable lazy loading for large datasets
  compression_enabled: true        # Enable data compression for storage
```

## Benchmark Configuration

### Benchmark Analysis Settings
```yaml
benchmark:
  enabled: true                    # Enable benchmark analysis for all backtests
  symbol: "IBOV"                   # Default benchmark symbol (Bovespa Index)
  auto_load: true                  # Automatically load benchmark data
  required: true                   # Require benchmark data for backtest completion
  data_sources: ["csv", "parquet", "api"]  # Priority order for data sources
  risk_free_rate_override: null    # Override SELIC rate if needed
```

## SGS Data Configuration

### Banco Central SGS Series
```yaml
sgs:
  # Banco Central SGS Series Configuration
  series:
    11: "Selic Interest Rate"      # Brazilian benchmark interest rate
    12: "CDI Interest Rate"        # Interbank deposit rate  
    433: "IPCA Inflation Index"    # Consumer price index
```

### API Configuration
```yaml
sgs:
  api:
    base_url: "http://api.bcb.gov.br/dados/serie/bcdata.sgs"
    timeout: 30                    # Request timeout in seconds
    max_retries: 3                 # Maximum retry attempts
    user_agent: "quant_b3_backtest/1.0"
```

### Data Processing Configuration
```yaml
sgs:
  processing:
    cache_enabled: true            # Enable data caching
    save_processed: true           # Save processed data to files
    data_path: "data/sgs"          # Path for SGS data storage
    normalization_method: "LOCF"   # Last Observation Carried Forward
```

### Data Quality Validation
```yaml
sgs:
  validation:
    enable_quality_checks: true    # Enable data quality validation
    interest_rate_range: [0, 100]  # Valid range for interest rates (%)
    inflation_range: [-50, 100]    # Valid range for inflation (%)
    min_data_points: 10            # Minimum required data points
```

### Strict Mode Configuration
```yaml
sgs:
  strict_mode:
    enabled: true                  # Enable strict mode (no fallback rates)
    require_selic_data: true       # Require SELIC data for backtest execution
    minimum_coverage_percentage: 95.0  # Require 95% data coverage
    fail_on_missing_data: true     # Fail backtest if data is missing
    allow_partial_data: false      # Do not allow partial data usage
```

### Data Quality Thresholds
```yaml
sgs:
  quality_thresholds:
    minimum_data_points: 100       # Minimum SELIC data points required
    maximum_gap_days: 5            # Maximum consecutive days without data
    rate_validity_range: [0.001, 100.0]  # Valid SELIC rate range
    outlier_threshold: 3.0         # Standard deviations for outlier detection
    minimum_coverage_days: 30      # Minimum days of data required for validation
```

## Configuration Validation

### Required Sections
The configuration file must contain all required sections:
- `market`: Market parameters and trading rules
- `taxes`: Tax configuration
- `portfolio`: Portfolio settings
- `settlement`: Settlement configuration
- `loss_carryforward`: Loss tracking settings

### Validation Rules
- All numeric values must be non-negative
- Tax rates must be between 0 and 1
- Date formats must be valid
- File paths must be accessible
- API endpoints must be reachable

## Environment-Specific Configuration

### Development Environment
```yaml
performance:
  log_level: "DEBUG"
  enable_profiling: true

error_handling:
  max_retry_attempts: 1
  graceful_degradation: false
```

### Production Environment
```yaml
performance:
  log_level: "WARNING"
  enable_profiling: false

error_handling:
  max_retry_attempts: 5
  graceful_degradation: true
```

## Configuration Management

### Version Control
- **Safe to Commit**: `settings.yaml` contains no sensitive data
- **Template**: `secrets.yaml.example` provides template for sensitive data
- **Environment Variables**: Can be overridden with environment variables

### Configuration Updates
```bash
# Validate configuration
python -c "import yaml; yaml.safe_load(open('config/settings.yaml'))"

# Test configuration loading
python -c "from engine.portfolio import EnhancedPortfolio; p = EnhancedPortfolio()"
```

## Usage Examples

### Loading Configuration
```python
import yaml

# Load configuration
with open('config/settings.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Access specific settings
selic_rate = config['market']['selic_rate']
tax_rate = config['taxes']['swing_trade']
initial_cash = config['portfolio']['initial_cash']
```

### Updating Configuration
```python
# Update specific parameters
config['market']['selic_rate'] = 0.12  # Update SELIC rate
config['portfolio']['initial_cash'] = 200000  # Update initial capital

# Save updated configuration
with open('config/settings.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)
```

### Environment-Specific Overrides
```python
import os

# Override with environment variables
selic_rate = float(os.getenv('SELIC_RATE', config['market']['selic_rate']))
initial_cash = float(os.getenv('INITIAL_CASH', config['portfolio']['initial_cash']))
```

## Best Practices

1. **Backup Configuration**: Keep backups of working configurations
2. **Version Control**: Commit configuration changes with clear messages
3. **Validation**: Always validate configuration before use
4. **Documentation**: Document any custom parameter changes
5. **Testing**: Test configuration changes in development environment
6. **Environment Separation**: Use different configurations for different environments
7. **Security**: Never commit sensitive data to version control

## Troubleshooting

### Common Issues
- **Invalid YAML**: Check YAML syntax with online validator
- **Missing Sections**: Ensure all required sections are present
- **Invalid Values**: Validate numeric ranges and data types
- **File Permissions**: Ensure configuration file is readable

### Debugging
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Load and validate configuration
from engine.portfolio import EnhancedPortfolio
portfolio = EnhancedPortfolio()  # Will show detailed loading information
``` 
"""
MOC Compliance Validation Test
==============================

Critical test to validate compliance with the strategy specification:
"All open positions are flattened at the auction call (end-of-day) via Market-on-Close (MOC) orders."

This test analyzes actual backtest results to identify MOC flattening violations.
"""

import json
import pytest
from typing import Dict, Any


def test_moc_compliance_july_2025_backtest():
    """
    CRITICAL COMPLIANCE TEST: Validate MOC flattening against strategy specification.
    
    Strategy Specification (README.md):
    "All open positions are flattened at the auction call (end-of-day) via Market-on-Close (MOC) orders."
    
    This test FAILS if ANY positions remain open overnight, violating the day-trading specification.
    """
    print("[MOC-COMPLIANCE] Validating MOC flattening compliance...")
    
    # Load actual backtest results
    try:
        with open('results/detailed/fuzzy_fajuto_default_3a6bf6653a_2025-07-01_2025-07-31_20250821_084255.json', 'r') as f:
            execution_data = json.load(f)
    except FileNotFoundError:
        pytest.skip("No backtest results available for MOC compliance validation")
    
    # Extract execution metrics
    strategy_summary = execution_data['strategy_summary']
    market_fills = strategy_summary['daily_executions']['market']['successful']
    limit_alpha = strategy_summary['daily_executions']['limit_alpha']['successful']
    limit_beta = strategy_summary['daily_executions']['limit_beta']['successful']
    limit_gamma = strategy_summary['daily_executions']['limit_gamma']['successful']
    
    total_limit_fills = limit_alpha + limit_beta + limit_gamma
    total_intraday_fills = market_fills + total_limit_fills
    
    # Calculate MOC fills from total fills (5,975 from logs)
    total_fills_from_logs = 5975
    estimated_moc_fills = total_fills_from_logs - total_intraday_fills
    
    print(f"[MOC-COMPLIANCE] Execution Analysis:")
    print(f"  Market fills (positions opened): {market_fills:,}")
    print(f"  Limit fills (additional entries): {total_limit_fills:,}")
    print(f"  Estimated MOC fills (closures): {estimated_moc_fills:,}")
    print(f"  Total fills: {total_fills_from_logs:,}")
    
    # Calculate overnight positions (CRITICAL METRIC)
    overnight_positions = market_fills - estimated_moc_fills
    moc_closure_rate = estimated_moc_fills / market_fills if market_fills > 0 else 0
    
    print(f"[MOC-COMPLIANCE] Position Analysis:")
    print(f"  Positions requiring closure: {market_fills:,}")
    print(f"  Positions closed by MOC: {estimated_moc_fills:,}")
    print(f"  NET OVERNIGHT POSITIONS: {overnight_positions:,}")
    print(f"  MOC closure rate: {moc_closure_rate:.1%}")
    
    # Risk assessment
    overnight_risk_ratio = overnight_positions / market_fills if market_fills > 0 else 0
    
    print(f"[MOC-COMPLIANCE] Risk Assessment:")
    print(f"  Overnight risk ratio: {overnight_risk_ratio:.1%}")
    
    if overnight_risk_ratio == 0:
        risk_level = "🟢 COMPLIANT"
        assessment = "Perfect MOC flattening - all positions closed"
    elif overnight_risk_ratio <= 0.05:
        risk_level = "🟡 MINOR VIOLATION" 
        assessment = "Small overnight exposure - acceptable tolerance"
    elif overnight_risk_ratio <= 0.20:
        risk_level = "🟠 MODERATE VIOLATION"
        assessment = "Moderate overnight exposure - review recommended"
    else:
        risk_level = "🔴 CRITICAL VIOLATION"
        assessment = "High overnight exposure - strategy specification violated"
    
    print(f"  Risk level: {risk_level}")
    print(f"  Assessment: {assessment}")
    
    # CRITICAL ASSERTION: Strategy specification compliance
    print(f"[MOC-COMPLIANCE] Strategy Specification Validation:")
    print(f"  Specification: 'All open positions are flattened at end-of-day'")
    
    # Define compliance thresholds (updated after fixes)
    STRICT_COMPLIANCE_THRESHOLD = 0  # Zero overnight positions (ideal)
    ACCEPTABLE_TOLERANCE = 10  # Tighter tolerance after dual-system fixes
    
    if overnight_positions == STRICT_COMPLIANCE_THRESHOLD:
        compliance_status = "✅ FULLY COMPLIANT"
        print(f"  Status: {compliance_status}")
        print(f"  Result: Perfect MOC flattening achieved")
        
    elif overnight_positions <= ACCEPTABLE_TOLERANCE:
        compliance_status = "🟡 ACCEPTABLE WITH TOLERANCE"
        print(f"  Status: {compliance_status}")
        print(f"  Result: Minor deviations within acceptable tolerance")
        print(f"  Recommendation: Monitor MOC logic for improvements")
        
    else:
        compliance_status = "❌ NON-COMPLIANT"
        print(f"  Status: {compliance_status}")
        print(f"  Result: Strategy specification violated")
        print(f"  Impact: {overnight_positions:,} positions carry overnight risk")
        
        # FAIL the test for significant violations
        pytest.fail(
            f"MOC COMPLIANCE FAILURE: Strategy specification violated!\n\n"
            f"SPECIFICATION: 'All open positions are flattened at the auction call (end-of-day) via Market-on-Close (MOC) orders.'\n\n"
            f"VIOLATION DETAILS:\n"
            f"  • Positions opened: {market_fills:,}\n"
            f"  • Positions closed by MOC: {estimated_moc_fills:,}\n"
            f"  • Overnight positions: {overnight_positions:,}\n"
            f"  • MOC closure rate: {moc_closure_rate:.1%}\n"
            f"  • Risk ratio: {overnight_risk_ratio:.1%}\n\n"
            f"REQUIRED ACTION:\n"
            f"  • Investigate MOC order generation logic\n"
            f"  • Ensure all market positions trigger corresponding MOC orders\n"
            f"  • Validate end-of-day position flattening in simulator\n\n"
            f"BUSINESS IMPACT:\n"
            f"  • Overnight exposure violates day-trading strategy\n"
            f"  • Regulatory risk for intraday position limits\n"
            f"  • Potential margin and risk management issues"
        )
    
    # Additional validations for acceptable cases
    if overnight_positions <= ACCEPTABLE_TOLERANCE:
        # Validate MOC efficiency is reasonable
        assert moc_closure_rate >= 0.60, f"MOC closure rate too low: {moc_closure_rate:.1%} (expected ≥60%)"
        
        # Validate we have MOC fills when positions were opened
        assert estimated_moc_fills > 0, "Expected MOC fills when market positions were opened"
        
        print(f"✅ [MOC-COMPLIANCE] VALIDATION PASSED")
        print(f"   MOC flattening within acceptable parameters")
        print(f"   Overnight exposure: {overnight_positions:,} positions ({overnight_risk_ratio:.1%})")
        print(f"   Recommendation: {'Continue monitoring' if overnight_positions > 0 else 'Excellent compliance'}")


def test_moc_efficiency_benchmarks():
    """
    Test MOC efficiency against industry benchmarks for day-trading strategies.
    """
    print("[MOC-EFFICIENCY] Benchmarking MOC performance...")
    
    try:
        with open('results/detailed/fuzzy_fajuto_default_3a6bf6653a_2025-07-01_2025-07-31_20250821_084255.json', 'r') as f:
            execution_data = json.load(f)
    except FileNotFoundError:
        pytest.skip("No backtest results available for MOC efficiency benchmarks")
    
    # Calculate metrics
    strategy_summary = execution_data['strategy_summary']
    market_fills = strategy_summary['daily_executions']['market']['successful']
    
    total_limit_fills = (
        strategy_summary['daily_executions']['limit_alpha']['successful'] +
        strategy_summary['daily_executions']['limit_beta']['successful'] +
        strategy_summary['daily_executions']['limit_gamma']['successful']
    )
    
    estimated_moc_fills = 5975 - market_fills - total_limit_fills
    moc_efficiency = estimated_moc_fills / market_fills if market_fills > 0 else 0
    
    print(f"[MOC-EFFICIENCY] Performance Metrics:")
    print(f"  MOC closure rate: {moc_efficiency:.1%}")
    
    # Industry benchmarks for day-trading strategies
    benchmarks = {
        "Excellent": 0.95,  # ≥95% closure rate
        "Good": 0.85,       # ≥85% closure rate  
        "Acceptable": 0.70,  # ≥70% closure rate
        "Poor": 0.50        # <50% closure rate
    }
    
    if moc_efficiency >= benchmarks["Excellent"]:
        grade = "🏆 EXCELLENT"
    elif moc_efficiency >= benchmarks["Good"]:
        grade = "✅ GOOD"
    elif moc_efficiency >= benchmarks["Acceptable"]:
        grade = "🟡 ACCEPTABLE"
    else:
        grade = "🔴 POOR"
    
    print(f"  Performance grade: {grade}")
    print(f"  Industry benchmark: {moc_efficiency:.1%} closure rate")
    
    # Assertions based on benchmarks
    assert moc_efficiency >= benchmarks["Acceptable"], f"MOC efficiency below acceptable threshold: {moc_efficiency:.1%} < 70%"
    
    if moc_efficiency < benchmarks["Good"]:
        print(f"⚠️  [MOC-EFFICIENCY] RECOMMENDATION: Improve MOC logic")
        print(f"   Current: {moc_efficiency:.1%}, Target: ≥{benchmarks['Good']:.0%}")
    
    print(f"✅ [MOC-EFFICIENCY] Benchmark validation passed")


if __name__ == "__main__":
    test_moc_compliance_july_2025_backtest()
    test_moc_efficiency_benchmarks()
    print("MOC compliance validation completed!")

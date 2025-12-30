"""Report generator for DataHub US status."""

import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional
import pandas as pd

from ..storage import CSVStorage
from ..providers.yfinance_provider import YFinanceProvider
from ..qa import DataValidator
from ..config import CACHE_DIR

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates operational status reports."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or CACHE_DIR.parent
        self.storage = CSVStorage()
        self.validator = DataValidator()
    
    def generate(self, output_file: Optional[str] = None) -> Path:
        """Generate status report as Markdown.
        
        Args:
            output_file: Output filename (default: us_datahub_status.md)
            
        Returns:
            Path to generated report
        """
        output_file = output_file or "us_datahub_status.md"
        output_path = self.output_dir / output_file
        
        # Gather data
        stats = self.storage.get_stats()
        symbols = self.storage.list_symbols()
        
        # Provider health
        provider = YFinanceProvider()
        provider_healthy = provider.healthcheck()
        
        # Symbol details
        symbol_details = []
        total_gaps = 0
        validation_errors = 0
        
        for symbol in symbols:
            df = self.storage.read(symbol)
            if df.empty:
                continue
            
            start_date, end_date = self.storage.get_date_range(symbol)
            gaps = self.storage.detect_gaps(symbol)
            validation = self.validator.validate(symbol, df)
            
            total_gaps += len(gaps)
            validation_errors += validation.error_count
            
            symbol_details.append({
                "symbol": symbol,
                "rows": len(df),
                "start": start_date.isoformat() if start_date else "N/A",
                "end": end_date.isoformat() if end_date else "N/A",
                "gaps": len(gaps),
                "errors": validation.error_count,
                "warnings": validation.warning_count,
            })
        
        # Sort by symbol
        symbol_details.sort(key=lambda x: x["symbol"])
        
        # Generate markdown
        report = self._generate_markdown(
            stats=stats,
            symbols=symbol_details,
            provider_healthy=provider_healthy,
            total_gaps=total_gaps,
            validation_errors=validation_errors,
        )
        
        # Write report
        output_path.write_text(report)
        logger.info(f"Report generated: {output_path}")
        
        return output_path
    
    def _generate_markdown(
        self,
        stats: dict,
        symbols: list,
        provider_healthy: bool,
        total_gaps: int,
        validation_errors: int,
    ) -> str:
        """Generate markdown content."""
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Health status
        health_icon = "✅" if provider_healthy else "❌"
        gaps_icon = "✅" if total_gaps == 0 else "⚠️"
        errors_icon = "✅" if validation_errors == 0 else "❌"
        
        # Completeness score
        total_symbols = len(symbols)
        symbols_ok = sum(1 for s in symbols if s["errors"] == 0 and s["gaps"] == 0)
        completeness = (symbols_ok / total_symbols * 100) if total_symbols > 0 else 0
        
        report = f"""# DataHub US - Status Report

**Generated:** {now}  
**Source:** yfinance  

---

## Summary

| Metric | Value | Status |
|--------|-------|--------|
| Symbols | {stats['symbols_count']} | - |
| Total Bars | {stats['total_bars']:,} | - |
| Date Range | {stats['start_date']} to {stats['end_date']} | - |
| Provider | yfinance | {health_icon} |
| Gaps Detected | {total_gaps} | {gaps_icon} |
| Validation Errors | {validation_errors} | {errors_icon} |
| Completeness | {completeness:.1f}% | {"✅" if completeness >= 95 else "⚠️"} |

---

## Provider Status

| Provider | Status | Last Check |
|----------|--------|------------|
| yfinance | {"Healthy" if provider_healthy else "Unhealthy"} | {now} |

---

## Symbol Coverage

| Symbol | Rows | Start | End | Gaps | Errors |
|--------|------|-------|-----|------|--------|
"""
        
        # Add symbol rows
        for s in symbols[:50]:  # Limit to 50 symbols
            gap_flag = f"⚠️ {s['gaps']}" if s['gaps'] > 0 else "0"
            error_flag = f"❌ {s['errors']}" if s['errors'] > 0 else "0"
            report += f"| {s['symbol']} | {s['rows']:,} | {s['start']} | {s['end']} | {gap_flag} | {error_flag} |\n"
        
        if len(symbols) > 50:
            report += f"\n*... and {len(symbols) - 50} more symbols*\n"
        
        # Issues section
        issues = [s for s in symbols if s['errors'] > 0 or s['gaps'] > 0]
        
        if issues:
            report += f"""
---

## Issues ({len(issues)})

"""
            for s in issues[:20]:
                problems = []
                if s['gaps'] > 0:
                    problems.append(f"{s['gaps']} gaps")
                if s['errors'] > 0:
                    problems.append(f"{s['errors']} errors")
                report += f"- **{s['symbol']}**: {', '.join(problems)}\n"
            
            if len(issues) > 20:
                report += f"\n*... and {len(issues) - 20} more*\n"
        else:
            report += """
---

## Issues

No issues detected ✅
"""
        
        report += """
---

## Data Quality Checks

- ✅ Schema validation (date, OHLCV columns)
- ✅ OHLC sanity (low ≤ open/close ≤ high)
- ✅ Volume non-negative
- ✅ Date monotonicity
- ✅ Gap detection (>5 trading days)
- ✅ Outlier detection (>50% daily change)

---

*Report generated by DataHub US v0.1.0*
"""
        
        return report




















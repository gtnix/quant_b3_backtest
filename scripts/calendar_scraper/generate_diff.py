#!/usr/bin/env python3
"""
Generate Calendar Diff - Compare scraped calendars with database and generate report.

Usage:
    python generate_diff.py --year 2025
    python generate_diff.py --year 2025 --output diff_report.md
"""

import argparse
import json
import os
import sys
from datetime import date

from b3_scraper import B3CalendarScraper
from nyse_scraper import NYSECalendarScraper


def generate_markdown_report(year: int, b3_diff, nyse_diff) -> str:
    """Generate a markdown report for the calendar differences."""
    lines = [
        f"# Calendar Update Report - {year}",
        "",
        f"Generated: {date.today()}",
        "",
    ]
    
    # B3 Section
    lines.append("## B3 (Brazil)")
    lines.append("")
    
    if not b3_diff.has_changes():
        lines.append("✅ No changes needed - calendar is in sync!")
    else:
        if b3_diff.missing_in_db:
            lines.append(f"### Missing in Database ({len(b3_diff.missing_in_db)} holidays)")
            lines.append("")
            lines.append("| Date | Name | Type |")
            lines.append("|------|------|------|")
            for h in b3_diff.missing_in_db:
                lines.append(f"| {h.date} | {h.name} | {h.holiday_type} |")
            lines.append("")
        
        if b3_diff.extra_in_db:
            lines.append(f"### Extra in Database ({len(b3_diff.extra_in_db)} holidays)")
            lines.append("")
            lines.append("| Date | Name | Type |")
            lines.append("|------|------|------|")
            for h in b3_diff.extra_in_db:
                lines.append(f"| {h['holiday_date']} | {h['name']} | {h['holiday_type']} |")
            lines.append("")
        
        if b3_diff.mismatches:
            lines.append(f"### Mismatches ({len(b3_diff.mismatches)} holidays)")
            lines.append("")
            lines.append("| Date | Expected | Actual |")
            lines.append("|------|----------|--------|")
            for scraped, db in b3_diff.mismatches:
                lines.append(f"| {scraped.date} | {scraped.name} | {db['name']} |")
            lines.append("")
    
    lines.append("")
    
    # NYSE Section
    lines.append("## NYSE (US)")
    lines.append("")
    
    if not nyse_diff.has_changes():
        lines.append("✅ No changes needed - calendar is in sync!")
    else:
        if nyse_diff.missing_in_db:
            lines.append(f"### Missing in Database ({len(nyse_diff.missing_in_db)} holidays)")
            lines.append("")
            lines.append("| Date | Name | Type | Early Close |")
            lines.append("|------|------|------|-------------|")
            for h in nyse_diff.missing_in_db:
                early = h.early_close_time or "-"
                lines.append(f"| {h.date} | {h.name} | {h.holiday_type} | {early} |")
            lines.append("")
        
        if nyse_diff.extra_in_db:
            lines.append(f"### Extra in Database ({len(nyse_diff.extra_in_db)} holidays)")
            lines.append("")
            lines.append("| Date | Name | Type |")
            lines.append("|------|------|------|")
            for h in nyse_diff.extra_in_db:
                lines.append(f"| {h['holiday_date']} | {h['name']} | {h['holiday_type']} |")
            lines.append("")
        
        if nyse_diff.mismatches:
            lines.append(f"### Mismatches ({len(nyse_diff.mismatches)} holidays)")
            lines.append("")
            lines.append("| Date | Expected | Actual |")
            lines.append("|------|----------|--------|")
            for scraped, db in nyse_diff.mismatches:
                lines.append(f"| {scraped.date} | {scraped.name} | {db['name']} |")
            lines.append("")
    
    # Summary
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    
    b3_changes = len(b3_diff.missing_in_db) + len(b3_diff.extra_in_db) + len(b3_diff.mismatches)
    nyse_changes = len(nyse_diff.missing_in_db) + len(nyse_diff.extra_in_db) + len(nyse_diff.mismatches)
    total_changes = b3_changes + nyse_changes
    
    lines.append(f"- **B3**: {b3_changes} changes")
    lines.append(f"- **NYSE**: {nyse_changes} changes")
    lines.append(f"- **Total**: {total_changes} changes")
    lines.append("")
    
    if total_changes > 0:
        lines.append("⚠️ **Review required before merging!**")
    else:
        lines.append("✅ **All calendars are in sync!**")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate Calendar Diff Report")
    parser.add_argument("--year", type=int, required=True, help="Year to compare")
    parser.add_argument("--output", type=str, help="Output file for markdown report")
    args = parser.parse_args()
    
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)
    
    # Scrape and compare B3
    print(f"Scraping B3 calendar for {args.year}...")
    b3_scraper = B3CalendarScraper()
    b3_holidays = b3_scraper.scrape_year(args.year)
    b3_diff = b3_scraper.compare_with_db(b3_holidays, db_url)
    
    # Scrape and compare NYSE
    print(f"Scraping NYSE calendar for {args.year}...")
    nyse_scraper = NYSECalendarScraper()
    nyse_holidays = nyse_scraper.scrape_year(args.year)
    nyse_diff = nyse_scraper.compare_with_db(nyse_holidays, db_url)
    
    # Generate report
    report = generate_markdown_report(args.year, b3_diff, nyse_diff)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report saved to {args.output}")
    else:
        print(report)
    
    # Set GitHub Actions output if running in CI
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            # Escape newlines for GitHub Actions
            escaped = report.replace("\n", "%0A")
            f.write(f"summary={escaped}\n")
            
            total_changes = (
                len(b3_diff.missing_in_db) + len(b3_diff.extra_in_db) + len(b3_diff.mismatches) +
                len(nyse_diff.missing_in_db) + len(nyse_diff.extra_in_db) + len(nyse_diff.mismatches)
            )
            f.write(f"has_changes={'true' if total_changes > 0 else 'false'}\n")
    
    # Exit with error if there are changes (for CI)
    total_changes = (
        len(b3_diff.missing_in_db) + len(b3_diff.extra_in_db) + len(b3_diff.mismatches) +
        len(nyse_diff.missing_in_db) + len(nyse_diff.extra_in_db) + len(nyse_diff.mismatches)
    )
    
    if total_changes > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()




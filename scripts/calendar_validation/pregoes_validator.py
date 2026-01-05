#!/usr/bin/env python3
"""
PREGOES.pdf Validator - Validate B3 trading day counts against official data.

Source: B3 PREGOES.pdf (historical trading day counts from 1968 to present)

Usage:
    python pregoes_validator.py --year 2024
    python pregoes_validator.py --years 2005-2025
    python pregoes_validator.py --all
"""

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import date

import psycopg2


# Official trading day counts from B3 PREGOES.pdf
# Format: year -> {"total": annual_total, "months": [jan, feb, ..., dec]}
PREGOES_DATA = {
    # 2005-2010
    2005: {"total": 249, "months": [20, 17, 22, 21, 21, 22, 21, 23, 21, 21, 21, 19]},
    2006: {"total": 248, "months": [21, 17, 23, 19, 22, 21, 21, 23, 20, 22, 21, 18]},
    2007: {"total": 249, "months": [22, 17, 22, 20, 22, 21, 22, 23, 20, 23, 21, 16]},
    2008: {"total": 250, "months": [22, 18, 21, 22, 21, 21, 23, 21, 22, 23, 19, 17]},
    2009: {"total": 249, "months": [21, 17, 22, 21, 21, 22, 23, 21, 22, 22, 20, 17]},
    2010: {"total": 249, "months": [20, 17, 23, 21, 21, 22, 22, 22, 22, 21, 21, 17]},
    
    # 2011-2015
    2011: {"total": 249, "months": [20, 17, 23, 20, 22, 21, 21, 23, 21, 21, 21, 19]},
    2012: {"total": 250, "months": [21, 18, 22, 20, 22, 21, 22, 23, 20, 23, 20, 18]},
    2013: {"total": 249, "months": [22, 17, 21, 22, 22, 20, 23, 22, 21, 23, 19, 17]},
    2014: {"total": 249, "months": [22, 18, 21, 22, 21, 20, 23, 21, 22, 23, 19, 17]},
    2015: {"total": 249, "months": [21, 17, 22, 21, 21, 22, 23, 21, 22, 22, 20, 17]},
    
    # 2016-2020
    2016: {"total": 251, "months": [20, 18, 22, 21, 21, 22, 21, 23, 22, 21, 21, 19]},
    2017: {"total": 249, "months": [21, 17, 23, 19, 22, 21, 21, 23, 21, 22, 21, 18]},
    2018: {"total": 247, "months": [22, 17, 22, 20, 22, 21, 22, 23, 20, 23, 20, 15]},
    2019: {"total": 250, "months": [22, 18, 21, 22, 22, 20, 23, 22, 21, 23, 19, 17]},
    2020: {"total": 250, "months": [22, 18, 22, 21, 21, 21, 23, 21, 22, 22, 20, 17]},
    
    # 2021-2025
    2021: {"total": 249, "months": [20, 17, 23, 21, 21, 22, 22, 22, 22, 21, 21, 17]},
    2022: {"total": 249, "months": [20, 17, 23, 20, 22, 21, 21, 23, 21, 21, 21, 19]},
    2023: {"total": 248, "months": [21, 17, 23, 19, 22, 21, 21, 23, 20, 22, 21, 18]},
    2024: {"total": 249, "months": [22, 18, 21, 21, 22, 20, 23, 22, 21, 23, 18, 18]},  # Consciência Negra added
    2025: {"total": 248, "months": [22, 17, 21, 22, 21, 21, 23, 21, 22, 23, 18, 17]},
}


@dataclass
class ValidationResult:
    """Result of validating a year's trading day count."""
    year: int
    expected_total: int
    actual_total: int
    expected_months: list[int]
    actual_months: list[int]
    
    @property
    def is_valid(self) -> bool:
        return self.expected_total == self.actual_total
    
    @property
    def month_diffs(self) -> list[tuple[int, int, int]]:
        """Return list of (month, expected, actual) for months with differences."""
        diffs = []
        for i, (exp, act) in enumerate(zip(self.expected_months, self.actual_months)):
            if exp != act:
                diffs.append((i + 1, exp, act))
        return diffs
    
    def __str__(self) -> str:
        status = "✅" if self.is_valid else "❌"
        line = f"{status} {self.year}: Expected {self.expected_total}, Found {self.actual_total}"
        
        if not self.is_valid:
            diffs = self.month_diffs
            if diffs:
                diff_strs = [f"M{m}({exp}→{act})" for m, exp, act in diffs]
                line += f" [{', '.join(diff_strs)}]"
        
        return line


def validate_year(db_url: str, year: int) -> ValidationResult:
    """Validate trading day count for a specific year."""
    if year not in PREGOES_DATA:
        raise ValueError(f"No PREGOES data for year {year}")
    
    expected = PREGOES_DATA[year]
    
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    # Count trading days per month from database
    cur.execute("""
        SELECT 
            EXTRACT(MONTH FROM session_date)::INT as month,
            COUNT(*) as count
        FROM trading_sessions
        WHERE market = 'BR'
          AND day_type != 'CLOSED'
          AND EXTRACT(YEAR FROM session_date) = %s
        GROUP BY month
        ORDER BY month
    """, (year,))
    
    month_counts = {row[0]: row[1] for row in cur.fetchall()}
    
    # Build actual months array
    actual_months = [month_counts.get(m, 0) for m in range(1, 13)]
    actual_total = sum(actual_months)
    
    cur.close()
    conn.close()
    
    return ValidationResult(
        year=year,
        expected_total=expected["total"],
        actual_total=actual_total,
        expected_months=expected["months"],
        actual_months=actual_months,
    )


def validate_years(db_url: str, years: list[int]) -> list[ValidationResult]:
    """Validate multiple years."""
    results = []
    for year in years:
        try:
            result = validate_year(db_url, year)
            results.append(result)
        except ValueError as e:
            print(f"Warning: {e}", file=sys.stderr)
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate B3 trading day counts")
    parser.add_argument("--year", type=int, help="Specific year to validate")
    parser.add_argument("--years", type=str, help="Year range (e.g., 2005-2025)")
    parser.add_argument("--all", action="store_true", help="Validate all available years")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)
    
    # Determine years to validate
    if args.year:
        years = [args.year]
    elif args.years:
        start, end = map(int, args.years.split("-"))
        years = list(range(start, end + 1))
    elif args.all:
        years = sorted(PREGOES_DATA.keys())
    else:
        print("ERROR: Must specify --year, --years, or --all", file=sys.stderr)
        sys.exit(1)
    
    # Validate
    results = validate_years(db_url, years)
    
    # Output
    print("=" * 60)
    print("B3 PREGOES.pdf Validation Report")
    print("=" * 60)
    
    all_valid = True
    for result in results:
        print(result)
        if not result.is_valid:
            all_valid = False
    
    print("=" * 60)
    
    valid_count = sum(1 for r in results if r.is_valid)
    total_count = len(results)
    
    print(f"Summary: {valid_count}/{total_count} years match PREGOES.pdf")
    
    if all_valid:
        print("✅ All years validated successfully!")
        sys.exit(0)
    else:
        print("❌ Some years have discrepancies - review required!")
        sys.exit(1)


if __name__ == "__main__":
    main()



























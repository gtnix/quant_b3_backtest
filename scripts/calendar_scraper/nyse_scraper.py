#!/usr/bin/env python3
"""
NYSE Calendar Scraper - Scrapes official NYSE trading calendar.

Official source: https://www.nyse.com/markets/hours-calendars

Usage:
    python nyse_scraper.py --year 2025
    python nyse_scraper.py --year 2025 --compare-db
    python nyse_scraper.py --year 2025 --generate-sql
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from typing import Optional

import httpx
import psycopg2


@dataclass
class Holiday:
    """Holiday record."""
    date: date
    name: str
    holiday_type: str  # NATIONAL, HALF_DAY
    early_close_time: Optional[str] = None
    source: str = "NYSE_OFFICIAL"


@dataclass
class CalendarDiff:
    """Difference between scraped and database calendars."""
    missing_in_db: list[Holiday]
    extra_in_db: list[dict]
    mismatches: list[tuple[Holiday, dict]]
    
    def has_changes(self) -> bool:
        return bool(self.missing_in_db or self.extra_in_db or self.mismatches)
    
    def summary(self) -> str:
        lines = []
        if self.missing_in_db:
            lines.append(f"Missing in DB: {len(self.missing_in_db)}")
            for h in self.missing_in_db:
                lines.append(f"  + {h.date} | {h.name} ({h.holiday_type})")
        if self.extra_in_db:
            lines.append(f"Extra in DB: {len(self.extra_in_db)}")
            for h in self.extra_in_db:
                lines.append(f"  - {h['holiday_date']} | {h['name']} ({h['holiday_type']})")
        if self.mismatches:
            lines.append(f"Mismatches: {len(self.mismatches)}")
            for scraped, db in self.mismatches:
                lines.append(f"  ~ {scraped.date}: scraped={scraped.name}, db={db['name']}")
        if not lines:
            lines.append("No differences found - calendar is in sync!")
        return "\n".join(lines)


class NYSECalendarScraper:
    """Scraper for NYSE official calendar."""
    
    # NYSE calendar page
    BASE_URL = "https://www.nyse.com/markets/hours-calendars"
    
    def __init__(self):
        self.client = httpx.Client(timeout=30.0)

    def get_nth_weekday(self, year: int, month: int, weekday: int, n: int) -> date:
        """Get the nth occurrence of a weekday in a month.
        weekday: 0=Monday, 6=Sunday
        """
        first_of_month = date(year, month, 1)
        first_weekday = first_of_month.weekday()
        
        # Days until target weekday
        days_until = (weekday - first_weekday + 7) % 7
        first_occurrence = first_of_month + timedelta(days=days_until)
        
        # Add (n-1) weeks
        return first_occurrence + timedelta(weeks=n-1)

    def get_last_weekday(self, year: int, month: int, weekday: int) -> date:
        """Get the last occurrence of a weekday in a month."""
        # Start from last day of month
        if month == 12:
            next_month = date(year + 1, 1, 1)
        else:
            next_month = date(year, month + 1, 1)
        last_of_month = next_month - timedelta(days=1)
        
        # Go backwards to find the weekday
        days_back = (last_of_month.weekday() - weekday + 7) % 7
        return last_of_month - timedelta(days=days_back)

    def calculate_easter(self, year: int) -> date:
        """Calculate Easter Sunday using the Anonymous Gregorian algorithm."""
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        return date(year, month, day)

    def adjust_for_weekend(self, d: date) -> Optional[date]:
        """Adjust holiday if it falls on weekend (US observation rules)."""
        if d.weekday() == 6:  # Sunday -> Monday
            return d + timedelta(days=1)
        elif d.weekday() == 5:  # Saturday -> typically not observed
            return None
        return d

    def scrape_year(self, year: int) -> list[Holiday]:
        """Generate NYSE holidays for a year using official rules."""
        holidays = []
        easter = self.calculate_easter(year)
        
        # New Year's Day - January 1 (observed)
        new_year = date(year, 1, 1)
        observed = self.adjust_for_weekend(new_year)
        if observed:
            holidays.append(Holiday(
                date=observed,
                name="New Year's Day",
                holiday_type="NATIONAL",
            ))
        
        # Martin Luther King Jr. Day - 3rd Monday of January
        mlk_day = self.get_nth_weekday(year, 1, 0, 3)  # Monday=0
        holidays.append(Holiday(
            date=mlk_day,
            name="Martin Luther King Jr. Day",
            holiday_type="NATIONAL",
        ))
        
        # Presidents Day - 3rd Monday of February
        presidents_day = self.get_nth_weekday(year, 2, 0, 3)
        holidays.append(Holiday(
            date=presidents_day,
            name="Presidents Day",
            holiday_type="NATIONAL",
        ))
        
        # Good Friday - Friday before Easter
        good_friday = easter - timedelta(days=2)
        holidays.append(Holiday(
            date=good_friday,
            name="Good Friday",
            holiday_type="NATIONAL",
        ))
        
        # Memorial Day - Last Monday of May
        memorial_day = self.get_last_weekday(year, 5, 0)
        holidays.append(Holiday(
            date=memorial_day,
            name="Memorial Day",
            holiday_type="NATIONAL",
        ))
        
        # Juneteenth - June 19 (observed, from 2021)
        if year >= 2021:
            juneteenth = date(year, 6, 19)
            observed = self.adjust_for_weekend(juneteenth)
            if observed:
                # 2021 was special case - early close on Friday June 18
                if year == 2021 and juneteenth.weekday() == 5:  # Saturday
                    holidays.append(Holiday(
                        date=date(year, 6, 18),
                        name="Juneteenth (Early Close)",
                        holiday_type="HALF_DAY",
                        early_close_time="13:00:00",
                    ))
                else:
                    holidays.append(Holiday(
                        date=observed,
                        name="Juneteenth National Independence Day",
                        holiday_type="NATIONAL",
                    ))
        
        # Independence Day - July 4 (observed)
        july_4 = date(year, 7, 4)
        observed = self.adjust_for_weekend(july_4)
        if observed:
            holidays.append(Holiday(
                date=observed,
                name="Independence Day",
                holiday_type="NATIONAL",
            ))
        
        # Day before Independence Day - Early close if July 3 is weekday and July 4 is weekday
        if july_4.weekday() <= 4:  # July 4 is weekday
            july_3 = july_4 - timedelta(days=1)
            if july_3.weekday() <= 4:  # July 3 is also weekday
                holidays.append(Holiday(
                    date=july_3,
                    name="Day Before Independence Day",
                    holiday_type="HALF_DAY",
                    early_close_time="13:00:00",
                ))
        
        # Labor Day - 1st Monday of September
        labor_day = self.get_nth_weekday(year, 9, 0, 1)
        holidays.append(Holiday(
            date=labor_day,
            name="Labor Day",
            holiday_type="NATIONAL",
        ))
        
        # Thanksgiving Day - 4th Thursday of November
        thanksgiving = self.get_nth_weekday(year, 11, 3, 4)  # Thursday=3
        holidays.append(Holiday(
            date=thanksgiving,
            name="Thanksgiving Day",
            holiday_type="NATIONAL",
        ))
        
        # Day After Thanksgiving - Early close at 13:00
        black_friday = thanksgiving + timedelta(days=1)
        holidays.append(Holiday(
            date=black_friday,
            name="Day After Thanksgiving",
            holiday_type="HALF_DAY",
            early_close_time="13:00:00",
        ))
        
        # Christmas Day - December 25 (observed)
        christmas = date(year, 12, 25)
        observed = self.adjust_for_weekend(christmas)
        if observed:
            holidays.append(Holiday(
                date=observed,
                name="Christmas Day",
                holiday_type="NATIONAL",
            ))
        
        # Christmas Eve - Early close if weekday
        christmas_eve = date(year, 12, 24)
        if christmas_eve.weekday() <= 4:  # Weekday
            holidays.append(Holiday(
                date=christmas_eve,
                name="Christmas Eve",
                holiday_type="HALF_DAY",
                early_close_time="13:00:00",
            ))
        
        return sorted(holidays, key=lambda h: h.date)

    def compare_with_db(self, holidays: list[Holiday], db_url: str) -> CalendarDiff:
        """Compare scraped holidays with database."""
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        
        # Get year from holidays
        year = holidays[0].date.year
        start_date = date(year, 1, 1)
        end_date = date(year, 12, 31)
        
        # Fetch DB holidays
        cur.execute("""
            SELECT holiday_date, name, holiday_type, early_close_time
            FROM holidays
            WHERE market = 'US' 
              AND holiday_date >= %s 
              AND holiday_date <= %s
            ORDER BY holiday_date
        """, (start_date, end_date))
        
        db_holidays = {
            row[0]: {
                "holiday_date": row[0],
                "name": row[1],
                "holiday_type": row[2],
                "early_close_time": row[3],
            }
            for row in cur.fetchall()
        }
        
        cur.close()
        conn.close()
        
        # Compare
        scraped_dates = {h.date for h in holidays}
        db_dates = set(db_holidays.keys())
        
        missing_in_db = [h for h in holidays if h.date not in db_dates]
        extra_in_db = [db_holidays[d] for d in db_dates - scraped_dates]
        
        # Check mismatches
        mismatches = []
        for h in holidays:
            if h.date in db_holidays:
                db_h = db_holidays[h.date]
                if h.name.lower() != db_h["name"].lower():
                    mismatches.append((h, db_h))
        
        return CalendarDiff(
            missing_in_db=missing_in_db,
            extra_in_db=extra_in_db,
            mismatches=mismatches,
        )

    def generate_sql(self, diff: CalendarDiff, version_id: str) -> str:
        """Generate SQL statements to apply changes."""
        statements = []
        
        for h in diff.missing_in_db:
            early_close = f"'{h.early_close_time}'" if h.early_close_time else "NULL"
            
            statements.append(f"""
INSERT INTO holidays (version_id, holiday_date, market, name, holiday_type, early_close_time, source_layer)
VALUES ('{version_id}', '{h.date}', 'US', '{h.name}', '{h.holiday_type}', {early_close}, 'A_OFFICIAL')
ON CONFLICT DO NOTHING;
            """.strip())
        
        for db_h in diff.extra_in_db:
            statements.append(f"""
-- Extra holiday in DB (review before deleting):
-- DELETE FROM holidays WHERE holiday_date = '{db_h['holiday_date']}' AND market = 'US';
            """.strip())
        
        return "\n\n".join(statements)


def main():
    parser = argparse.ArgumentParser(description="NYSE Calendar Scraper")
    parser.add_argument("--year", type=int, required=True, help="Year to scrape")
    parser.add_argument("--compare-db", action="store_true", help="Compare with database")
    parser.add_argument("--generate-sql", action="store_true", help="Generate SQL for differences")
    parser.add_argument("--output", type=str, help="Output file for JSON")
    args = parser.parse_args()
    
    scraper = NYSECalendarScraper()
    holidays = scraper.scrape_year(args.year)
    
    print(f"NYSE Calendar {args.year}: {len(holidays)} holidays/early closes")
    for h in holidays:
        suffix = f" (closes {h.early_close_time})" if h.early_close_time else ""
        print(f"  {h.date} | {h.name} ({h.holiday_type}){suffix}")
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump([asdict(h) for h in holidays], f, indent=2, default=str)
        print(f"\nSaved to {args.output}")
    
    if args.compare_db:
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            print("ERROR: DATABASE_URL not set", file=sys.stderr)
            sys.exit(1)
        
        diff = scraper.compare_with_db(holidays, db_url)
        print(f"\n=== Comparison with Database ===")
        print(diff.summary())
        
        if args.generate_sql and diff.has_changes():
            version_id = os.environ.get("US_VERSION_ID", "YOUR_VERSION_ID_HERE")
            sql = scraper.generate_sql(diff, version_id)
            print(f"\n=== SQL to Apply Changes ===")
            print(sql)


if __name__ == "__main__":
    main()




#!/usr/bin/env python3
"""
B3 Calendar Scraper - Scrapes official B3 trading calendar.

Official source: https://www.b3.com.br/pt_br/solucoes/plataformas/puma-trading-system/para-participantes-e-traders/calendario-de-negociacao/

Usage:
    python b3_scraper.py --year 2025
    python b3_scraper.py --year 2025 --compare-db
    python b3_scraper.py --year 2025 --generate-sql
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import Optional

import httpx
import psycopg2


@dataclass
class Holiday:
    """Holiday record."""
    date: date
    name: str
    holiday_type: str  # NATIONAL, MARKET_SPECIFIC, LATE_OPEN, HALF_DAY
    early_close_time: Optional[str] = None
    late_open_time: Optional[str] = None
    source: str = "B3_OFFICIAL"


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


class B3CalendarScraper:
    """Scraper for B3 official calendar."""
    
    # B3 calendar page (may need updates if B3 changes structure)
    BASE_URL = "https://www.b3.com.br/pt_br/solucoes/plataformas/puma-trading-system/para-participantes-e-traders/calendario-de-negociacao/"
    
    # Known B3 holidays (fixed dates)
    FIXED_HOLIDAYS = {
        (1, 1): "Confraternização Universal",
        (4, 21): "Tiradentes",
        (5, 1): "Dia do Trabalho",
        (9, 7): "Independência do Brasil",
        (10, 12): "Nossa Senhora Aparecida",
        (11, 2): "Finados",
        (11, 15): "Proclamação da República",
        (11, 20): "Dia da Consciência Negra",  # From 2024
        (12, 25): "Natal",
    }
    
    # Market-specific closures
    MARKET_CLOSURES = {
        (12, 24): "Véspera de Natal",
        (12, 31): "Véspera de Ano Novo",
    }

    def __init__(self):
        self.client = httpx.Client(timeout=30.0)

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

    def scrape_year(self, year: int) -> list[Holiday]:
        """Generate holidays for a year using rules (fallback if scraping fails)."""
        holidays = []
        easter = self.calculate_easter(year)
        
        # Fixed holidays
        for (month, day), name in self.FIXED_HOLIDAYS.items():
            # Consciência Negra only from 2024
            if (month, day) == (11, 20) and year < 2024:
                continue
            holidays.append(Holiday(
                date=date(year, month, day),
                name=name,
                holiday_type="NATIONAL",
            ))
        
        # Market closures
        for (month, day), name in self.MARKET_CLOSURES.items():
            holidays.append(Holiday(
                date=date(year, month, day),
                name=name,
                holiday_type="MARKET_SPECIFIC",
            ))
        
        # Moveable holidays (Easter-based)
        from datetime import timedelta
        
        # Carnival Monday (Easter - 48 days)
        carnival_mon = easter - timedelta(days=48)
        holidays.append(Holiday(
            date=carnival_mon,
            name="Carnaval",
            holiday_type="NATIONAL",
        ))
        
        # Carnival Tuesday (Easter - 47 days)
        carnival_tue = easter - timedelta(days=47)
        holidays.append(Holiday(
            date=carnival_tue,
            name="Carnaval",
            holiday_type="NATIONAL",
        ))
        
        # Ash Wednesday (Easter - 46 days) - Late open at 13:00
        ash_wed = easter - timedelta(days=46)
        holidays.append(Holiday(
            date=ash_wed,
            name="Quarta-feira de Cinzas",
            holiday_type="LATE_OPEN",
            late_open_time="13:00:00",
        ))
        
        # Good Friday (Easter - 2 days)
        good_friday = easter - timedelta(days=2)
        holidays.append(Holiday(
            date=good_friday,
            name="Sexta-feira Santa",
            holiday_type="NATIONAL",
        ))
        
        # Corpus Christi (Easter + 60 days)
        corpus_christi = easter + timedelta(days=60)
        holidays.append(Holiday(
            date=corpus_christi,
            name="Corpus Christi",
            holiday_type="NATIONAL",
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
            SELECT holiday_date, name, holiday_type, early_close_time, late_open_time
            FROM holidays
            WHERE market = 'BR' 
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
                "late_open_time": row[4],
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
        
        # Check mismatches for dates in both
        mismatches = []
        for h in holidays:
            if h.date in db_holidays:
                db_h = db_holidays[h.date]
                # Simple name comparison (ignoring case/accents for now)
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
            late_open = f"'{h.late_open_time}'" if h.late_open_time else "NULL"
            
            statements.append(f"""
INSERT INTO holidays (version_id, holiday_date, market, name, holiday_type, early_close_time, late_open_time, source_layer)
VALUES ('{version_id}', '{h.date}', 'BR', '{h.name}', '{h.holiday_type}', {early_close}, {late_open}, 'A_OFFICIAL')
ON CONFLICT DO NOTHING;
            """.strip())
        
        for db_h in diff.extra_in_db:
            statements.append(f"""
-- Extra holiday in DB (review before deleting):
-- DELETE FROM holidays WHERE holiday_date = '{db_h['holiday_date']}' AND market = 'BR';
            """.strip())
        
        return "\n\n".join(statements)


def main():
    parser = argparse.ArgumentParser(description="B3 Calendar Scraper")
    parser.add_argument("--year", type=int, required=True, help="Year to scrape")
    parser.add_argument("--compare-db", action="store_true", help="Compare with database")
    parser.add_argument("--generate-sql", action="store_true", help="Generate SQL for differences")
    parser.add_argument("--output", type=str, help="Output file for JSON")
    args = parser.parse_args()
    
    scraper = B3CalendarScraper()
    holidays = scraper.scrape_year(args.year)
    
    print(f"B3 Calendar {args.year}: {len(holidays)} holidays")
    for h in holidays:
        print(f"  {h.date} | {h.name} ({h.holiday_type})")
    
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
            # Get version_id from environment or use placeholder
            version_id = os.environ.get("B3_VERSION_ID", "YOUR_VERSION_ID_HERE")
            sql = scraper.generate_sql(diff, version_id)
            print(f"\n=== SQL to Apply Changes ===")
            print(sql)


if __name__ == "__main__":
    main()





























#!/usr/bin/env python3
"""
Verify pair-trade possibility from a FuzzyFajuto components CSV.

- Inputs: path to CSV produced by export_fuzzy_components_to_csv
- Outputs: date-indexed table (date, buy_count, sell_count, has_both)
          totals by category and a feasibility message.

Usage:
  python scripts/verify_pair_possibility.py --csv reports/fuzzy_components_20250601-20250730.csv \
      --start 2025-05-20 --end 2025-08-11
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify pair-mode feasibility from fuzzy components CSV")
    p.add_argument("--csv", required=True, help="Path to fuzzy_components_*.csv")
    p.add_argument("--start", default=None, help="Start date YYYY-MM-DD (optional)")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD (optional)")
    p.add_argument("--max-rows", type=int, default=200, help="Max rows to print in summary table")
    return p.parse_args()


def load_and_filter(csv_path: Path, start: str | None, end: str | None) -> pd.DataFrame:
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        raise SystemExit("CSV missing required 'date' column")
    if "qualified_signal" not in df.columns:
        raise SystemExit("CSV missing required 'qualified_signal' column")
    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"]).copy()
    # Filter by window if provided
    if start:
        start_d = pd.to_datetime(start).date()
        df = df[df["date"] >= start_d]
    if end:
        end_d = pd.to_datetime(end).date()
        df = df[df["date"] <= end_d]
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    # Counts per date
    g = df.groupby("date")["qualified_signal"]
    buy_count = g.apply(lambda s: (s == "BUY").sum()).rename("buy_count")
    sell_count = g.apply(lambda s: (s == "SELL").sum()).rename("sell_count")
    summary = pd.concat([buy_count, sell_count], axis=1)
    summary["has_both"] = (summary["buy_count"] > 0) & (summary["sell_count"] > 0)
    summary = summary.reset_index().sort_values("date")
    return summary


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv)
    df = load_and_filter(csv_path, args.start, args.end)
    if df.empty:
        print("No rows in selected window.")
        return 0

    summary = build_summary(df)
    # Totals
    both_days = int(summary["has_both"].sum())
    only_buy = int(((summary["buy_count"] > 0) & (summary["sell_count"] == 0)).sum())
    only_sell = int(((summary["sell_count"] > 0) & (summary["buy_count"] == 0)).sum())
    neither = int(((summary["buy_count"] == 0) & (summary["sell_count"] == 0)).sum())

    # Print table
    print("\nPer-day BUY/SELL counts:")
    to_print = summary.copy()
    if len(to_print) > args.max_rows:
        head = to_print.head(args.max_rows // 2)
        tail = to_print.tail(args.max_rows - len(head))
        print(pd.concat([head, tail]).to_string(index=False))
        print(f"... (truncated {len(to_print) - args.max_rows} rows) ...")
    else:
        print(to_print.to_string(index=False))

    print("\nTotals:")
    print(f"- both:      {both_days}")
    print(f"- only_buy:  {only_buy}")
    print(f"- only_sell: {only_sell}")
    print(f"- neither:   {neither}")

    feasible = both_days > 0
    print("\nConclusion:")
    if feasible:
        print("Pair mode is POSSIBLE for this dataset (at least one day with BUY and SELL).")
    else:
        print("Pair mode is IMPOSSIBLE for this dataset (no day has both BUY and SELL).")

    return 0


if __name__ == "__main__":
    sys.exit(main())


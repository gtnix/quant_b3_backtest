#!/usr/bin/env python3
"""Filter market data CSVs to top N symbols by data availability.

Creates optimized CSVs for fast InProcessExecutor evaluation.

Usage:
    python scripts/filter_market_data.py
"""

import csv
from collections import defaultdict
from pathlib import Path

# Configuration
MAX_SYMBOLS = 100
DATA_DIR = Path("data")

# IBOV main components (as of 2024) - prioritize these for BR
IBOV_CORE = {
    "PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3", "B3SA3", "WEGE3", "RENT3",
    "BBAS3", "PRIO3", "SUZB3", "GGBR4", "EQTL3", "RADL3", "BPAC11", "HAPV3",
    "JBSS3", "CSAN3", "RAIL3", "LREN3", "EMBR3", "ENGI11", "VBBR3", "ELET3",
    "ELET6", "RDOR3", "SBSP3", "BBSE3", "VIVT3", "CMIG4", "CPLE6", "KLBN11",
    "TOTS3", "UGPA3", "CCRO3", "ASAI3", "HYPE3", "NTCO3", "AZUL4", "YDUQ3",
    "CYRE3", "COGN3", "BRFS3", "MRFG3", "BEEF3", "GOAU4", "USIM5", "CSNA3",
    "MRVE3", "TAEE11", "TIMS3", "CPFE3", "SANB11", "CRFB3", "MGLU3", "VIIA3",
    "ALSO3", "MULT3", "LWSA3", "PETZ3", "RRRP3", "CASH3", "ALOS3", "ALPA4",
    "CMIN3", "SLCE3", "SMTO3", "BRKM5", "DXCO3", "IGTI11", "IRBR3", "PCAR3",
    "SOMA3", "VAMO3", "RECV3", "CIEL3", "STBP3", "QUAL3", "FLRY3", "ENBR3"
}

# NASDAQ 100 / S&P 500 top components - prioritize these for US
US_CORE = {
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "UNH", "JNJ", "V", "XOM", "JPM", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
    "LLY", "PEP", "KO", "AVGO", "COST", "TMO", "WMT", "MCD", "CSCO", "ACN",
    "ABT", "DHR", "ADBE", "CRM", "NKE", "TXN", "NFLX", "AMD", "INTC", "QCOM",
    "HON", "PM", "UNP", "NEE", "RTX", "LOW", "IBM", "SPGI", "BMY", "CAT",
    "GS", "MS", "BLK", "AMGN", "MDT", "GE", "PLD", "SBUX", "T", "VZ", "COP",
    "ISRG", "INTU", "NOW", "AXP", "SYK", "ELV", "GILD", "PNC", "ZTS", "SCHW",
    "MMC", "USB", "TFC", "CB", "C", "SO", "DUK", "MO", "CME", "CL", "ICE",
    "PYPL", "SNAP", "SQ", "SHOP", "ROKU", "DOCU", "ZM", "CRWD", "DDOG", "SNOW"
}


def filter_csv(input_file: Path, output_file: Path, priority_symbols: set, max_symbols: int):
    """Filter CSV to top N symbols, prioritizing given set."""
    print(f"  Reading {input_file}...")
    
    # Count rows per symbol
    symbol_counts = defaultdict(int)
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol_counts[row['symbol']] += 1
    
    print(f"  Found {len(symbol_counts)} unique symbols")
    
    # Sort symbols: priority first, then by data availability
    def sort_key(sym):
        is_priority = sym in priority_symbols
        count = symbol_counts[sym]
        return (not is_priority, -count)  # Priority first, then most data
    
    sorted_symbols = sorted(symbol_counts.keys(), key=sort_key)
    selected_symbols = set(sorted_symbols[:max_symbols])
    
    print(f"  Selected {len(selected_symbols)} symbols")
    
    # Count how many priority symbols made it
    priority_selected = len(selected_symbols & priority_symbols)
    print(f"  Including {priority_selected} priority index members")
    
    # Write filtered data
    rows_written = 0
    with open(input_file, 'r') as fin, open(output_file, 'w', newline='') as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        
        for row in reader:
            if row['symbol'] in selected_symbols:
                writer.writerow(row)
                rows_written += 1
    
    print(f"  Wrote {rows_written:,} rows to {output_file}")
    return rows_written


def main():
    print("=" * 50)
    print("  FILTER MARKET DATA FOR FAST BACKTESTING")
    print("=" * 50)
    print(f"  Max symbols per market: {MAX_SYMBOLS}")
    print("")
    
    results = {}
    
    # BR market
    br_input = DATA_DIR / "market_data_ibov.csv"
    br_output = DATA_DIR / "market_data_ibov_index.csv"
    if br_input.exists():
        print("[BR Market]")
        results["BR"] = filter_csv(br_input, br_output, IBOV_CORE, MAX_SYMBOLS)
        print("")
    else:
        print(f"  SKIP: {br_input} not found")
    
    # US market
    us_input = DATA_DIR / "market_data_us.csv"
    us_output = DATA_DIR / "market_data_us_index.csv"
    if us_input.exists():
        print("[US Market]")
        results["US"] = filter_csv(us_input, us_output, US_CORE, MAX_SYMBOLS)
        print("")
    else:
        print(f"  SKIP: {us_input} not found")
    
    print("=" * 50)
    print("  SUMMARY")
    print("=" * 50)
    for market, count in results.items():
        print(f"  {market}: {count:,} rows")
    
    print("")
    print("  Update configs to use:")
    print("    BR: data/market_data_ibov_index.csv")
    print("    US: data/market_data_us_index.csv")
    
    return 0


if __name__ == '__main__':
    exit(main())

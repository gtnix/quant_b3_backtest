#!/usr/bin/env python3
"""
Extract ticker symbols from IBRA CSV file.

This script reads the IBRA CSV file and extracts all ticker symbols
for use in data download scripts.
"""

import pandas as pd
import csv
from pathlib import Path
import re

def extract_ibra_tickers(csv_file_path: str = "../../IBRADia_08-07-25.csv") -> list:
    """
    Extract ticker symbols from IBRA CSV file.
    
    Args:
        csv_file_path (str): Path to the IBRA CSV file
        
    Returns:
        list: List of ticker symbols
    """
    tickers = []
    
    try:
        # Read the CSV file with semicolon delimiter
        with open(csv_file_path, 'r', encoding='latin-1') as file:
            # Skip the first line (header)
            next(file)
            
            # Read the second line (column headers)
            next(file)
            
            # Read the rest of the file
            for line in file:
                if line.strip():  # Skip empty lines
                    # Split by semicolon and get the first column (ticker)
                    parts = line.strip().split(';')
                    if len(parts) >= 1:
                        ticker = parts[0].strip()
                        # Only include valid tickers: 4-6 uppercase letters/numbers, start with a letter, no spaces
                        if re.match(r'^[A-Z][A-Z0-9]{3,5}$', ticker):
                            tickers.append(ticker)
    
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []
    
    return tickers

def save_tickers_to_file(tickers: list, output_file: str = "ibra_tickers.txt"):
    """
    Save tickers to a text file for easy use in other scripts.
    
    Args:
        tickers (list): List of ticker symbols
        output_file (str): Output file path
    """
    try:
        with open(output_file, 'w') as file:
            for ticker in tickers:
                file.write(f"{ticker}\n")
        print(f"Saved {len(tickers)} tickers to {output_file}")
    except Exception as e:
        print(f"Error saving tickers to file: {e}")

def main():
    """Main function to extract and save IBRA tickers."""
    print("Extracting IBRA ticker symbols...")
    
    # Extract tickers
    tickers = extract_ibra_tickers()
    
    if tickers:
        print(f"Found {len(tickers)} ticker symbols:")
        print(", ".join(tickers[:10]) + "..." if len(tickers) > 10 else ", ".join(tickers))
        
        # Save to file
        save_tickers_to_file(tickers)
        
        # Also save as Python list for easy import
        with open("ibra_tickers.py", 'w') as file:
            file.write("# IBRA Ticker Symbols\n")
            file.write("# Extracted from IBRADia_08-07-25.csv\n\n")
            file.write("IBRA_TICKERS = [\n")
            for ticker in tickers:
                file.write(f'    "{ticker}",\n')
            file.write("]\n")
        
        print(f"Saved tickers as Python list to ibra_tickers.py")
        
    else:
        print("No tickers found!")

if __name__ == "__main__":
    main() 
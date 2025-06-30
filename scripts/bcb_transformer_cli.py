#!/usr/bin/env python3
"""
BCB Daily Factor Transformer CLI

Command-line interface for the BCBDailyFactorTransformer class.
Provides easy access to transformation and validation functionality.

Usage:
    python scripts/bcb_transformer_cli.py transform --input-file data.csv --output-file transformed.csv
    python scripts/bcb_transformer_cli.py validate --input-file data.csv --output-file validated.csv
    python scripts/bcb_transformer_cli.py pipeline --input-file data.csv --output-file result.csv
    python scripts/bcb_transformer_cli.py report --input-file data.csv --output-file report.yaml
"""

import argparse
import sys
import os
import pandas as pd
from datetime import datetime
import yaml

# Add the parent directory to the path to import the module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from market_data.bcb_daily_factor_transformer import BCBDailyFactorTransformer


def load_data(input_file: str) -> pd.DataFrame:
    """
    Load data from various file formats.
    
    Args:
        input_file (str): Path to input file
        
    Returns:
        pd.DataFrame: Loaded data
        
    Raises:
        ValueError: If file format is not supported
    """
    file_ext = os.path.splitext(input_file)[1].lower()
    
    if file_ext == '.csv':
        return pd.read_csv(input_file, index_col=0, parse_dates=True)
    elif file_ext == '.parquet':
        return pd.read_parquet(input_file)
    elif file_ext == '.json':
        return pd.read_json(input_file)
    elif file_ext == '.xlsx':
        return pd.read_excel(input_file, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


def save_data(data: pd.DataFrame, output_file: str) -> None:
    """
    Save data to various file formats.
    
    Args:
        data (pd.DataFrame): Data to save
        output_file (str): Path to output file
    """
    file_ext = os.path.splitext(output_file)[1].lower()
    
    if file_ext == '.csv':
        data.to_csv(output_file)
    elif file_ext == '.parquet':
        data.to_parquet(output_file)
    elif file_ext == '.json':
        data.to_json(output_file)
    elif file_ext == '.xlsx':
        data.to_excel(output_file)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


def transform_command(args):
    """Handle transform command."""
    print(f"Loading data from {args.input_file}...")
    raw_data = load_data(args.input_file)
    
    print(f"Initializing transformer...")
    transformer = BCBDailyFactorTransformer(args.config)
    
    print(f"Transforming data using {transformer.transformation_config.method.value} method...")
    transformed_data = transformer.transform_rates(raw_data)
    
    print(f"Saving transformed data to {args.output_file}...")
    save_data(transformed_data, args.output_file)
    
    print(f"Transformation complete!")
    print(f"- Input records: {len(raw_data)}")
    print(f"- Output records: {len(transformed_data)}")
    print(f"- Method: {transformer.transformation_config.method.value}")


def validate_command(args):
    """Handle validate command."""
    print(f"Loading data from {args.input_file}...")
    input_data = load_data(args.input_file)
    
    print(f"Initializing transformer...")
    transformer = BCBDailyFactorTransformer(args.config)
    
    print(f"Validating data...")
    validated_data = transformer.validate_factors(input_data)
    
    print(f"Saving validated data to {args.output_file}...")
    save_data(validated_data, args.output_file)
    
    print(f"Validation complete!")
    print(f"- Input records: {len(input_data)}")
    print(f"- Output records: {len(validated_data)}")
    print(f"- Retention rate: {len(validated_data) / len(input_data):.2%}")


def pipeline_command(args):
    """Handle pipeline command (transform + validate)."""
    print(f"Loading data from {args.input_file}...")
    raw_data = load_data(args.input_file)
    
    print(f"Initializing transformer...")
    transformer = BCBDailyFactorTransformer(args.config)
    
    print(f"Running complete transformation and validation pipeline...")
    validated_data, report = transformer.transform_and_validate(raw_data)
    
    print(f"Saving results to {args.output_file}...")
    save_data(validated_data, args.output_file)
    
    print(f"Pipeline complete!")
    print(f"- Original records: {report.original_count}")
    print(f"- Final records: {report.validated_count}")
    print(f"- Retention rate: {report.retention_rate:.2%}")
    print(f"- Outliers removed: {report.outliers_removed}")
    print(f"- Validation methods: {', '.join(report.validation_methods_applied)}")
    
    # Save report if requested
    if args.report_file:
        print(f"Saving validation report to {args.report_file}...")
        transformer.save_validation_report(report, args.report_file)


def report_command(args):
    """Handle report command."""
    print(f"Loading data from {args.input_file}...")
    input_data = load_data(args.input_file)
    
    print(f"Initializing transformer...")
    transformer = BCBDailyFactorTransformer(args.config)
    
    print(f"Generating validation report...")
    # For report generation, we need both original and validated data
    # Since we only have input data, we'll validate it and generate report
    validated_data = transformer.validate_factors(input_data)
    report = transformer.generate_validation_report(input_data, validated_data)
    
    print(f"Saving report to {args.output_file}...")
    transformer.save_validation_report(report, args.output_file)
    
    print(f"Report generated!")
    print(f"- Original records: {report.original_count}")
    print(f"- Validated records: {report.validated_count}")
    print(f"- Retention rate: {report.retention_rate:.2%}")
    print(f"- Outliers removed: {report.outliers_removed}")
    if report.warnings:
        print(f"- Warnings: {len(report.warnings)}")


def config_command(args):
    """Handle config command - show current configuration."""
    print(f"Loading configuration from {args.config}...")
    transformer = BCBDailyFactorTransformer(args.config)
    
    print("\n=== Transformation Configuration ===")
    config = transformer.transformation_config
    print(f"Method: {config.method.value}")
    print(f"Annualization Factor: {config.annualization_factor}")
    print(f"Compound Frequency: {config.compound_frequency}")
    
    print("\n=== Validation Configuration ===")
    for i, config in enumerate(transformer.validation_configs, 1):
        print(f"Method {i}: {config.method.value}")
        if config.method.value == 'iqr':
            print(f"  - Multiplier: {config.multiplier}")
        elif config.method.value == 'zscore':
            print(f"  - Threshold: {config.threshold}")
        elif config.method.value == 'rolling_window':
            print(f"  - Window Size: {config.window_size}")
        elif config.method.value == 'bounds':
            print(f"  - Min Factor: {config.min_factor}")
            print(f"  - Max Factor: {config.max_factor}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="BCB Daily Factor Transformer CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transform raw rates to factors
  python scripts/bcb_transformer_cli.py transform --input-file raw_rates.csv --output-file factors.csv
  
  # Validate factors using configured methods
  python scripts/bcb_transformer_cli.py validate --input-file factors.csv --output-file validated.csv
  
  # Complete pipeline (transform + validate)
  python scripts/bcb_transformer_cli.py pipeline --input-file raw_rates.csv --output-file result.csv --report-file report.yaml
  
  # Generate validation report
  python scripts/bcb_transformer_cli.py report --input-file data.csv --output-file report.yaml
  
  # Show current configuration
  python scripts/bcb_transformer_cli.py config
        """
    )
    
    parser.add_argument(
        '--config',
        default='config/settings.yaml',
        help='Path to configuration file (default: config/settings.yaml)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Transform command
    transform_parser = subparsers.add_parser('transform', help='Transform raw rates to factors')
    transform_parser.add_argument('--input-file', required=True, help='Input file path')
    transform_parser.add_argument('--output-file', required=True, help='Output file path')
    transform_parser.set_defaults(func=transform_command)
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate factors using configured methods')
    validate_parser.add_argument('--input-file', required=True, help='Input file path')
    validate_parser.add_argument('--output-file', required=True, help='Output file path')
    validate_parser.set_defaults(func=validate_command)
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Complete transformation and validation pipeline')
    pipeline_parser.add_argument('--input-file', required=True, help='Input file path')
    pipeline_parser.add_argument('--output-file', required=True, help='Output file path')
    pipeline_parser.add_argument('--report-file', help='Validation report file path (optional)')
    pipeline_parser.set_defaults(func=pipeline_command)
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate validation report')
    report_parser.add_argument('--input-file', required=True, help='Input file path')
    report_parser.add_argument('--output-file', required=True, help='Report output file path')
    report_parser.set_defaults(func=report_command)
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Show current configuration')
    config_parser.set_defaults(func=config_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main() 
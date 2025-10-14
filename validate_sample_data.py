#!/usr/bin/env python3
"""
Script to fetch and validate sample data from Tiingo API.

This script demonstrates the complete data fetching and validation pipeline:
1. Fetches AAPL stock data
2. Fetches BTCUSD crypto data
3. Validates both datasets using DataValidator
4. Prints detailed reports of any issues found
"""

import logging
from datetime import datetime, timedelta
from config.api_config import TiingoConfig
from data.fetcher import TiingoClient
from data.validator import DataValidator, DataValidationError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subheader(title):
    """Print a formatted subsection header."""
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80)


def print_issue_details(issues):
    """Print detailed information about validation issues."""
    if not issues:
        print("  ✓ No issues found - data is clean!")
        return

    print(f"\n  Found {len(issues)} issue(s):\n")

    for i, issue in enumerate(issues, 1):
        issue_type = issue.get('type', 'unknown')
        message = issue.get('message', 'No message')
        count = issue.get('count', 0)

        print(f"  {i}. [{issue_type.upper()}]")
        print(f"     {message}")

        # Print additional details based on issue type
        if 'dates' in issue and issue['dates']:
            dates = issue['dates']
            print(f"     Sample dates: {', '.join(str(d) for d in dates[:5])}")
            if len(dates) > 5:
                print(f"     ... and {len(dates) - 5} more")

        if 'details' in issue:
            details = issue['details']
            if isinstance(details, list) and len(details) > 0:
                print(f"     Details: {len(details)} events detected")
                # Show first few events
                for j, detail in enumerate(details[:3], 1):
                    if isinstance(detail, dict):
                        detail_msg = detail.get('message', '')
                        print(f"       {j}. {detail_msg}")
                if len(details) > 3:
                    print(f"       ... and {len(details) - 3} more events")

        if 'size' in issue:
            print(f"     Gap size: {issue['size']} days")

        if 'severity' in issue:
            print(f"     Severity: {issue['severity']}")

        print()


def print_summary_statistics(ticker, asset_type, original_df, cleaned_df, issues, validator):
    """Print summary statistics for the validation process."""
    print_subheader(f"Summary Statistics for {ticker}")

    print(f"\n  Asset Type: {asset_type}")
    print(f"  Rows before cleaning: {len(original_df)}")
    print(f"  Rows after cleaning:  {len(cleaned_df)}")
    print(f"  Rows removed:         {len(original_df) - len(cleaned_df)}")
    print(f"  Date range:           {cleaned_df.index.min().date()} to {cleaned_df.index.max().date()}")
    print(f"  Total issues found:   {len(issues)}")

    # Issue breakdown
    issue_summary = validator.get_issues_summary()
    if issue_summary:
        print("\n  Issues by Type:")
        for issue_type, count in sorted(issue_summary.items()):
            print(f"    - {issue_type}: {count}")

    # Data quality metrics
    print("\n  Data Quality Metrics:")
    print(f"    Average volume:    {cleaned_df['volume'].mean():,.0f}")
    print(f"    Min close price:   ${cleaned_df['close'].min():.2f}")
    print(f"    Max close price:   ${cleaned_df['close'].max():.2f}")
    print(f"    Price change:      {((cleaned_df['close'].iloc[-1] / cleaned_df['close'].iloc[0]) - 1) * 100:+.2f}%")

    # Check for any remaining data issues
    has_nulls = cleaned_df.isnull().any().any()
    has_duplicates = cleaned_df.index.duplicated().any()

    print("\n  Post-Validation Checks:")
    print(f"    Contains null values:      {'❌ Yes' if has_nulls else '✓ No'}")
    print(f"    Contains duplicate dates:  {'❌ Yes' if has_duplicates else '✓ No'}")
    print(f"    Index sorted:              {'✓ Yes' if cleaned_df.index.is_monotonic_increasing else '❌ No'}")


def validate_asset(client, validator, ticker, asset_type, start_date, end_date):
    """
    Fetch and validate data for a single asset.

    Args:
        client: TiingoClient instance
        validator: DataValidator instance
        ticker: Ticker symbol
        asset_type: 'stock' or 'crypto'
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)

    Returns:
        Tuple of (original_df, cleaned_df, issues) or (None, None, None) on error
    """
    print_header(f"Validating {ticker} ({asset_type.upper()})")

    try:
        # Fetch data
        print(f"\n  Fetching data from {start_date} to {end_date}...")

        if asset_type == 'stock':
            df = client.get_stock_prices(ticker, start_date, end_date)
        elif asset_type == 'crypto':
            df = client.get_crypto_prices(ticker, start_date, end_date)
        else:
            raise ValueError(f"Unknown asset type: {asset_type}")

        if df is None or df.empty:
            print(f"  ✗ Failed to fetch data for {ticker}")
            return None, None, None

        print(f"  ✓ Fetched {len(df)} rows of data")

        # Keep original for comparison
        original_df = df.copy()

        # Validate data
        print(f"\n  Running validation pipeline...")
        cleaned_df, issues = validator.validate_all(df, ticker, asset_type)

        print(f"  ✓ Validation complete")

        # Print detailed results
        print_subheader("Validation Issues")
        print_issue_details(issues)

        # Print summary
        print_summary_statistics(ticker, asset_type, original_df, cleaned_df, issues, validator)

        return original_df, cleaned_df, issues

    except DataValidationError as e:
        print(f"  ✗ Validation error: {e}")
        logger.error(f"Validation error for {ticker}: {e}")
        return None, None, None

    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        logger.error(f"Unexpected error for {ticker}: {e}", exc_info=True)
        return None, None, None


def generate_overall_report(results):
    """Generate an overall summary report for all validated assets."""
    print_header("Overall Validation Report")

    total_assets = len(results)
    successful = sum(1 for r in results.values() if r['cleaned_df'] is not None)
    failed = total_assets - successful

    print(f"\n  Total assets validated:  {total_assets}")
    print(f"  Successful validations:  {successful}")
    print(f"  Failed validations:      {failed}")

    if successful > 0:
        print("\n  Asset Summary:")
        for ticker, result in results.items():
            if result['cleaned_df'] is not None:
                cleaned_df = result['cleaned_df']
                issues = result['issues']
                original_df = result['original_df']

                status = "✓ Clean" if len(issues) == 0 else f"⚠ {len(issues)} issue(s)"
                rows_removed = len(original_df) - len(cleaned_df)

                print(f"\n    {ticker}:")
                print(f"      Status:        {status}")
                print(f"      Rows:          {len(cleaned_df)} ({rows_removed} removed)")
                print(f"      Date range:    {cleaned_df.index.min().date()} to {cleaned_df.index.max().date()}")
                print(f"      Asset type:    {result['asset_type']}")


def main():
    """Main execution function."""
    print_header("Tiingo Data Validation Script")
    print("\nThis script fetches and validates financial data from Tiingo API")

    # Initialize configuration and clients
    try:
        config = TiingoConfig()
        client = TiingoClient(config)
        validator = DataValidator(log_level=logging.WARNING)

        print(f"\n✓ Connected to Tiingo API: {config.base_url}")

    except ValueError as e:
        print(f"\n✗ Configuration error: {e}")
        print("\nMake sure to set the TIINGO_API_KEY environment variable:")
        print("  export TIINGO_API_KEY='your_api_key_here'")
        return

    # Calculate date range (last 60 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    # Store results
    results = {}

    # Validate AAPL (stock)
    original_df, cleaned_df, issues = validate_asset(
        client, validator, 'AAPL', 'stock', start_str, end_str
    )
    results['AAPL'] = {
        'original_df': original_df,
        'cleaned_df': cleaned_df,
        'issues': issues,
        'asset_type': 'stock'
    }

    # Clear issues before next validation
    validator.clear_issues()

    # Validate BTCUSD (crypto)
    original_df, cleaned_df, issues = validate_asset(
        client, validator, 'btcusd', 'crypto', start_str, end_str
    )
    results['BTCUSD'] = {
        'original_df': original_df,
        'cleaned_df': cleaned_df,
        'issues': issues,
        'asset_type': 'crypto'
    }

    # Generate overall report
    generate_overall_report(results)

    # Final message
    print_header("Validation Complete")
    print("\n✓ All validations finished successfully!\n")


if __name__ == '__main__':
    main()

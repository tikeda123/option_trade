import os
import sys
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the path of the parent directory to sys.path
sys.path.append(parent_dir)


from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *
from option_pricing import simulate_option_prices


def process_option_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Process option data and organize it by symbol into time series

    Args:
        df: DataFrame containing option data

    Returns:
        Dict with symbol as key and corresponding time series DataFrame as value
    """
    # Convert date string to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Group data by symbol
    symbol_groups = {}

    for symbol in df['symbol'].unique():
        # Filter data for current symbol
        symbol_df = df[df['symbol'] == symbol].copy()

        # Sort by date
        symbol_df = symbol_df.sort_values('date')

        # Set date as index
        symbol_df.set_index('date', inplace=True)

        # Store in dictionary
        symbol_groups[symbol] = symbol_df

    return symbol_groups


def main():
    db = MongoDataLoader()

    # Load option data
    df = db.load_data(OPTION_TICKER)

    # Process data into time series by symbol
    symbol_timeseries = process_option_data(df)
    print(symbol_timeseries)
    exit()

    # Display total number of symbols
    print(f"\nTotal number of unique symbols: {len(symbol_timeseries)}")
    print("-" * 50)

    # Display first 40 symbols
    for i, (symbol, ts_data) in enumerate(symbol_timeseries.items()):
        if i >= 3:  # 40個表示したら終了
            break
        print(f"\nSymbol: {symbol}")
        print(f"Number of time points: {len(ts_data)}")
        print("First few rows:")
        print(ts_data)
        print("-" * 50)  # 見やすさのために区切り線を追加

if __name__ == "__main__":
    main()

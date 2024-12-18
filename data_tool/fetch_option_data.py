import sys
import os
import pandas as pd
import csv
from datetime import datetime
import time
from datetime import timezone
from typing import List, Optional
from typing import Union, Dict,Tuple
from datetime import timedelta, datetime


# Get the absolute path of the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory (A directory in this case)
parent_dir = os.path.dirname(current_dir)

# Add the parent directory's path to sys.path
sys.path.append(parent_dir)

from bybit_api.bybit_data_fetcher import BybitDataFetcher
from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *

def main():
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval_hours', type=int, default=1, choices=range(1,24),
                       help='Hour interval between data fetches (1-23). For example, 2 means fetch at 0:00, 2:00, 4:00, etc.')
    args = parser.parse_args()

    api = BybitDataFetcher()
    data_loader = MongoDataLoader()


    while True:
        # Get current time
        now = datetime.now(timezone.utc)

        # Calculate next run time based on interval_hours
        current_hour = now.hour
        next_hour = ((current_hour // args.interval_hours) + 1) * args.interval_hours
        next_run = now.replace(hour=next_hour % 24, minute=0, second=0, microsecond=0)
        if next_hour >= 24:
            next_run += timedelta(days=1)

        # Sleep until next run time
        sleep_seconds = (next_run - now).total_seconds()
        print(f"Sleeping for {sleep_seconds} seconds until {next_run}")
        time.sleep(sleep_seconds)

        base_coin = "BTC"
        # Fetch and store data
        try:
            # Get current UTC timestamp and format date string
            current_utc = int(time.time_ns() // 1_000_000_000)
            current_date = datetime.fromtimestamp(current_utc, tz=timezone.utc)
            date_str = current_date.strftime("%Y-%m-%d %H:%M:%S")
            date_str_tz = current_date.strftime("%Y-%m-%d %H:%M:%S%z")

            # Fetch and store instrument info
            instrument_info = api.fetch_instruments_info(category="option", baseCoin=base_coin)
            data_loader.insert_data(instrument_info, OPTION_SYMBOL)

            # Fetch and store ticker data
            tickers_data = api.fetch_tickers(category="option", baseCoin=base_coin)
            tickers_data["symbol_id"] = tickers_data["symbol"] + "_" + date_str
            tickers_data["date"] = date_str
            data_loader.insert_data(tickers_data, OPTION_TICKER)
            print(tickers_data)

            # Fetch and store historical volatility data for BTC and ETH
            for coin in ["BTC", "ETH"]:
                time.sleep(3)
                hv_result = api.fetch_historical_volatility_extended(
                    category="option",
                    baseCoin=coin,
                    period=7,
                    startTime="2023-01-01 00:00:00+0000",
                    endTime=date_str_tz
                )
                data_loader.insert_data(hv_result, OPTION_HV, symbol=coin, interval=7)
                print(hv_result)

        except Exception as e:
            print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()

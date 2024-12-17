"""
BybitDataFetcher module for fetching various types of market data from the Bybit API.
Supports fetching price data, open interest, funding rates, and premium index.
"""

import os
import sys
from datetime import datetime
from typing import Optional, Union, List, Dict, Any

import pandas as pd
from pybit.unified_trading import HTTP
from requests.exceptions import HTTPError

# Set up path to include parent directory for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.constants import *
from bybit_api.bybit_base_api import BybitBaseAPI, MAX_LIMIT_QUERY, CATEGORY
from trade_config import trade_config

# Interval to Bybit API interval format mapping
INTERVAL_MAPPING = {
    5: "5min",
    15: "15min",
    30: "30min",
    60: "1h",
    120: "2h",
    240: "4h",
    720: "12h",
    1440: "1d",
}

# Data type to column names mapping
COLUMN_NAMES_MAP = {
    "price": ["start_at", "open", "high", "low", "close", "volume", "turnover"],
    "oi": ["oi", "start_at"],
    "funding_rate": ["symbol", "funding_rate", "start_at"],
    "premium_index": ["start_at", "p_open", "p_high", "p_low", "p_close"],
    "instruments_info":["symbol","status","baseCoin","quoteCoin","settleCoin","optionsType","launchTime","deliveryTime","deliveryFeeRate"]
}


def convert_datetime_to_unix_time(
    dt_string: str, fmt: str = '%Y-%m-%d %H:%M:%S%z'
) -> float:
    """
    Convert a datetime string to a Unix timestamp in seconds.

    Args:
        dt_string: Datetime string to convert
        fmt: Datetime format string for parsing

    Returns:
        Unix timestamp (float)
    """
    if "T" in dt_string and "Z" in dt_string:
        fmt = '%Y-%m-%dT%H:%M:%SZ'
    return datetime.strptime(dt_string, fmt).timestamp()


def unix_time_to_datetime(unix_time: float) -> datetime:
    """
    Convert a Unix timestamp to a UTC datetime object.

    Args:
        unix_time: Unix timestamp in seconds

    Returns:
        A datetime object in UTC
    """
    return datetime.utcfromtimestamp(unix_time)


def interval_to_seconds(interval: Union[int, str]) -> int:
    """
    Convert time interval to seconds.
    Interval can be:
        - integer (minutes)
        - string with 'D' (days)
        - string with 'W' (weeks)

    Args:
        interval: The interval to convert

    Returns:
        Interval in seconds.

    Raises:
        ValueError: If the interval format is invalid.
        TypeError: If the interval is neither int nor str.
    """
    if isinstance(interval, int):
        return interval * 60
    if isinstance(interval, str):
        if interval.endswith("D"):
            return 24 * 60 * 60
        elif interval.endswith("W"):
            return 7 * 24 * 60 * 60
        raise ValueError("Invalid interval format. Must be an integer, 'D', or 'W'.")
    raise TypeError("Interval must be an integer or a string.")


class BybitDataFetcher(BybitBaseAPI):
    """
    Class for fetching market data from Bybit API.
    Inherits from BybitBaseAPI for base functionality.
    """

    def __init__(self):
        """Initialize BybitDataFetcher with parent class initialization."""
        super().__init__()

    def fetch_latest_info(self, symbol: Optional[str] = None) -> float:
        """
        Fetch the latest price for a given symbol.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')

        Returns:
            Latest price for the symbol

        Raises:
            RuntimeError: If the API request fails
        """
        symbol = symbol or trade_config.symbol
        session = HTTP(testnet=self._isTESTNET)
        try:
            result = session.get_tickers(category=CATEGORY, symbol=symbol)
            return float(result["result"]["list"][0]["lastPrice"])
        except (HTTPError, KeyError, Exception) as e:
            raise RuntimeError(f"An error occurred while fetching latest info: {e}") from e

    def _convert_interval(self, interval: Optional[int] = None) -> str:
        """
        Convert a numeric interval in minutes to a Bybit API interval string.

        Args:
            interval: Time interval in minutes

        Returns:
            The corresponding Bybit API interval string.
        """
        interval = interval or trade_config.interval
        return INTERVAL_MAPPING.get(interval, "1d")

    def _fetch_data_by_type(
        self,
        start_time: float,
        end_time: float,
        limit: int,
        data_type: str,
        symbol: Optional[str] = None,
        interval: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch specific type of market data within a time range.

        Args:
            start_time: Start time in Unix timestamp
            end_time: End time in Unix timestamp
            limit: Max number of records to fetch
            data_type: Type of data to fetch ('price', 'oi', 'funding_rate', 'premium_index')
            symbol: Trading pair symbol
            interval: Time interval in minutes

        Returns:
            List of fetched data records

        Raises:
            ValueError: If invalid data_type is specified
            RuntimeError: If API request fails
        """
        symbol = symbol or trade_config.symbol
        interval = interval or trade_config.interval

        session = HTTP(testnet=self._isTESTNET)
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)

        # Define API methods for different data types
        fetch_methods = {
            "price": lambda: session.get_kline(
                category=CATEGORY,
                symbol=symbol,
                interval="D" if interval == 1440 else interval,
                start=start_ms,
                end=end_ms,
                limit=limit,
            ),
            "oi": lambda: session.get_open_interest(
                category=CATEGORY,
                symbol=symbol,
                intervalTime=self._convert_interval(interval),
                startTime=start_ms,
                endTime=end_ms,
                limit=limit,
            ),
            "funding_rate": lambda: session.get_funding_rate_history(
                category=CATEGORY,
                symbol=symbol,
                intervalTime=self._convert_interval(interval),
                startTime=start_ms,
                endTime=end_ms,
                limit=limit,
            ),
            "premium_index": lambda: session.get_premium_index_price_kline(
                category=CATEGORY,
                symbol=symbol,
                intervalTime=interval,
                startTime=start_ms,
                endTime=end_ms,
                limit=limit,
            )
        }

        if data_type not in fetch_methods:
            raise ValueError("Invalid data type specified")

        try:
            result = fetch_methods[data_type]()
            return result["result"]["list"]
        except (HTTPError, KeyError, Exception) as e:
            raise RuntimeError(f"An error occurred while fetching {data_type} data: {e}") from e

    def _fetch_all_data_in_chunks(
        self,
        fromtime: str,
        totime: str,
        data_type: str,
        symbol: Optional[str] = None,
        interval: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch all data in chunks to handle large time ranges.

        Args:
            fromtime: Start time in datetime string format
            totime: End time in datetime string format
            data_type: Type of data to fetch
            symbol: Trading pair symbol
            interval: Time interval in minutes

        Returns:
            Combined list of all fetched data chunks
        """
        start_unix = convert_datetime_to_unix_time(fromtime)
        end_unix = convert_datetime_to_unix_time(totime)
        interval = interval or trade_config.interval
        chunk_seconds = MAX_LIMIT_QUERY * interval_to_seconds(interval)

        all_data = []
        while start_unix < end_unix:
            next_end_unix = min(start_unix + chunk_seconds, end_unix)
            data_chunk = self._fetch_data_by_type(
                start_unix, next_end_unix, MAX_LIMIT_QUERY, data_type, symbol, interval
            )
            all_data.extend(data_chunk)
            start_unix = next_end_unix
        return all_data

    def fetch_data(
        self,
        fromtime: str,
        totime: str,
        data_type: str,
        column_names: List[str],
        symbol: Optional[str] = None,
        interval: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch and process data into a pandas DataFrame.

        Args:
            fromtime: Start time in datetime string format
            totime: End time in datetime string format
            data_type: Type of data to fetch
            column_names: Column names for the DataFrame
            symbol: Trading pair symbol
            interval: Time interval in minutes

        Returns:
            Processed data in DataFrame format
        """
        all_data = self._fetch_all_data_in_chunks(fromtime, totime, data_type, symbol, interval)
        df = pd.DataFrame.from_records(all_data)
        df.columns = column_names

        # Process time columns if present
        if "start_at" in df.columns:
            df["start_at"] = df["start_at"].astype(float) / 1000
            df["date"] = pd.to_datetime(df["start_at"], unit="s")
            df = df.set_index("start_at").sort_index(ascending=True)

        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        return df

    def save_file_name(self, fromtime: str, totime: str, data_type: str) -> str:
        """
        Generate filename for saving data.

        Args:
            fromtime: Start time in datetime string format
            totime: End time in datetime string format
            data_type: Type of data

        Returns:
            Generated filename

        Raises:
            ValueError: If invalid data type specified
        """
        if data_type == "price":
            interval = trade_config.interval
        elif data_type == "oi":
            interval = interval_to_seconds(trade_config.interval)
        elif data_type == "funding_rate":
            interval = 480
        else:
            raise ValueError("Invalid data type specified")

        formatted_fromtime = fromtime[:-9].replace(":", "").replace("-", "").replace(" ", "")
        formatted_totime = totime[:-9].replace(":", "").replace("-", "").replace(" ", "")
        return f"{self._datapath}{trade_config.symbol}_{formatted_fromtime}_{formatted_totime}_{interval}_{data_type}.csv"

    def fetch_historical_data(
        self,
        fromtime: str,
        totime: str,
        data_type: str,
        symbol: Optional[str] = None,
        interval: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for specified type and time range.

        Args:
            fromtime: Start time in datetime string format
            totime: End time in datetime string format
            data_type: Type of data to fetch
            symbol: Trading pair symbol
            interval: Time interval in minutes

        Returns:
            Fetched data in DataFrame or None if failed
        """
        if data_type not in COLUMN_NAMES_MAP:
            raise ValueError("Invalid data type specified")

        try:
            return self.fetch_data(fromtime, totime, data_type, COLUMN_NAMES_MAP[data_type], symbol, interval)
        except RuntimeError as e:
            self._logger.log_system_message(f"Failed to fetch historical data: {e}")
            return None

    def fetch_historical_data_all(
        self,
        fromtime: str,
        totime: str,
        savefilename: Optional[str] = None,
        symbol: Optional[str] = None,
        interval: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch all historical price data and save to file.

        Args:
            fromtime: Start time in datetime string format
            totime: End time in datetime string format
            savefilename: Custom filename for saving data
            symbol: Trading pair symbol
            interval: Time interval in minutes

        Returns:
            Fetched price data in DataFrame or None if failed
        """
        try:
            price_data = self.fetch_historical_data(fromtime, totime, "price", symbol, interval)
            if price_data is None:
                return None

            price_data = price_data.drop_duplicates(subset=["date"], keep="first")
            savefilename = savefilename or self.save_file_name(fromtime, totime, "price")
            price_data.to_csv(savefilename)
            return price_data
        except Exception as e:
            self._logger.log_system_message(f"Failed to fetch all historical data: {e}")
            return None

    def fetch_instruments_info(self, category: str, symbol: Optional[str] = None) -> dict:
        """
        Fetch instruments information for a given category and symbol.

        Args:
            category: Category of the instruments (e.g. 'option')
            symbol: Symbol for instruments info (e.g. 'ETH-3JAN23-1250-P')

        Returns:
            The instruments info response from Bybit API

        Raises:
            RuntimeError: If API request fails
        """
        session = HTTP(testnet=self._isTESTNET)
        try:
            result = session.get_instruments_info(category=category, symbol=symbol)
            list_data = result["result"]["list"]
            df =  pd.json_normalize(list_data)
            return df
        except (HTTPError, KeyError, Exception) as e:
            raise RuntimeError(f"An error occurred while fetching instruments info: {e}") from e

def main():
    api = BybitDataFetcher()
    instruments_info = api.fetch_instruments_info(category="option", symbol="")
    print(instruments_info)

    start_time = "2024-01-01 00:00:00+0900"
    end_time = "2024-02-02 00:00:00+0900"

    _symbol = "BTCUSDT"
    _interval = 1440
    df = api.fetch_historical_data_all(start_time, end_time, symbol=_symbol, interval=_interval)
    print(df)

    latest_price = api.fetch_latest_info(_symbol)
    print(f"Latest price of {_symbol}: {latest_price}")

    _symbol = "BNBUSDT"
    _interval = 30
    df = api.fetch_historical_data_all(start_time, end_time, symbol=_symbol, interval=_interval)
    print(df)

    # Fetch instruments info example
    instruments_info = api.fetch_instruments_info(category="option", symbol="")
    print(instruments_info)

    import time
    for i in range(5):
        time.sleep(5)
        latest_price = api.fetch_latest_info(_symbol)
        print(f"Latest price of {_symbol}: {latest_price}")

if __name__ == "__main__":
    main()

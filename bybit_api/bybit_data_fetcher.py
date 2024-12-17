"""BybitDataFetcher module for fetching various types of market data from the Bybit API.
Supports fetching price data, open interest, funding rates, and premium index.
"""
import os
import sys
from datetime import datetime, timezone
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
    dt_string: str, fmt: str = '%Y-%m-%d %H:%M:%S%z') -> float:
    """
    Convert a datetime string to a Unix timestamp in seconds.
    """
    if "T" in dt_string and "Z" in dt_string:
        fmt = '%Y-%m-%dT%H:%M:%SZ'
    return datetime.strptime(dt_string, fmt).timestamp()

def unix_time_to_datetime(unix_time: float) -> datetime:
    """Convert Unix timestamp to UTC datetime object."""
    return datetime.fromtimestamp(unix_time, tz=timezone.utc)

def interval_to_seconds(interval: Union[int, str]) -> int:
    """
    Convert time interval to seconds.
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

    def fetch_tickers(self, category: str, symbol: Optional[str] = None, baseCoin: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch the latest ticker information, including best bid/ask, last price, and 24h volume.

        Args:
            category (str): The category of the instrument (e.g. "spot", "linear", "inverse", "option")
            symbol (Optional[str]): The instrument symbol (e.g. "BTCUSDT")
            baseCoin (Optional[str]): The base coin if needed (for option category without a symbol)

        Returns:
            Dict[str, Any]: The response dictionary containing ticker information.

        Raises:
            ValueError: If category="option" and neither symbol nor baseCoin is provided.
            RuntimeError: If the API request fails.
        """
        # For option category, either symbol or baseCoin must be provided
        if category == "option" and not symbol and not baseCoin:
            raise ValueError("For category='option', symbol or baseCoin must be provided.")

        session = HTTP(testnet=self._isTESTNET)
        try:
            result = session.get_tickers(category=category, symbol=symbol, baseCoin=baseCoin)
            list_data = result["result"]["list"]
            df =  pd.json_normalize(list_data)
            return df
        except (HTTPError, KeyError, Exception) as e:
            raise RuntimeError(f"An error occurred while fetching tickers: {e}") from e

    def fetch_latest_info(self, symbol: Optional[str] = None) -> float:
        """
        Fetch the latest price for a given symbol.
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
        """
        all_data = self._fetch_all_data_in_chunks(fromtime, totime, data_type, symbol, interval)
        df = pd.DataFrame.from_records(all_data)
        df.columns = column_names

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

    def fetch_instruments_info(self, category: str, symbol: Optional[str] = None, baseCoin: Optional[str] = None) -> dict:
        """
        Fetch instruments information for a given category and symbol.
        """
        session = HTTP(testnet=self._isTESTNET)
        try:
            result = session.get_instruments_info(category=category, symbol=symbol, baseCoin=baseCoin)
            list_data = result["result"]["list"]
            df =  pd.json_normalize(list_data)
            return df
        except (HTTPError, KeyError, Exception) as e:
            raise RuntimeError(f"An error occurred while fetching instruments info: {e}") from e

    def fetch_historical_volatility(
        self,
        category: str = "option",
        baseCoin: str = "ETH",
        period: int = 30,
        startTime: Optional[str] = None,
        endTime: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical volatility data from Bybit.

        Args:
            category (str): The category of the instrument, default "option".
            baseCoin (str): The base coin (e.g. "ETH", "BTC", ...). Default "ETH".
            period (int): The period in minutes for the volatility data (e.g. 30).
            startTime (Optional[str]): Start time in format 'YYYY-MM-DD HH:MM:SS+ZZ' or 'YYYY-MM-DDTHH:MM:SSZ'.
            endTime (Optional[str]): End time in format 'YYYY-MM-DD HH:MM:SS+ZZ' or 'YYYY-MM-DDTHH:MM:SSZ'.

        Returns:
            pd.DataFrame: A DataFrame containing historical volatility data with columns ['period', 'value', 'time', 'date'].

        Note:
            - If both startTime and endTime are not provided, return the most recent 1 hour worth of data.
            - If startTime/endTime are provided, they must be within a 30 day range and not exceed 2 years back.
            - The data returned is hourly.
        """
        session = HTTP(testnet=self._isTESTNET)

        # Validate input parameters
        unix_start = None
        unix_end = None
        if (startTime is None and endTime is not None) or (startTime is not None and endTime is None):
            raise ValueError("startTime and endTime must both be provided or both be omitted.")

        if startTime and endTime:
            unix_start = int(convert_datetime_to_unix_time(startTime) * 1000)
            unix_end = int(convert_datetime_to_unix_time(endTime) * 1000)
            # Check range within 30 days
            thirty_days_ms = 30 * 24 * 60 * 60 * 1000
            if (unix_end - unix_start) > thirty_days_ms:
                raise ValueError("The [endTime - startTime] range must be <= 30 days.")

            # Optionally, you can add a check for the 2-year limit if needed
            # two_years_ms = 2 * 365 * 24 * 60 * 60 * 1000
            # If needed: no explicit mention in the problem that we must enforce this strictly,
            # but we can just trust the API's constraint.
            # if (current_time_ms - unix_start) > two_years_ms:
            #     raise ValueError("The requested startTime is older than 2 years.")

        # Fetch data from API
        try:
            result = session.get_historical_volatility(
                category=category,
                baseCoin=baseCoin,
                period=period,
                startTime=unix_start,
                endTime=unix_end
            )
            data_list = result.get("result", [])

            if not data_list:
                return pd.DataFrame(columns=["period", "value", "time", "date"])

            df = pd.DataFrame(data_list)
            # Convert time (ms) to datetime
            if "time" in df.columns:
                df["time"] = df["time"].astype(float) / 1000
                df["date"] = pd.to_datetime(df["time"], unit="s", utc=True)

            return df[["period", "value", "time", "date"]].sort_values("time")
        except (HTTPError, KeyError, Exception) as e:
            raise RuntimeError(f"An error occurred while fetching historical volatility: {e}") from e

    def fetch_historical_volatility_extended(
            self,
            category: str = "option",
            baseCoin: str = "ETH",
            period: int = 30,
            startTime: Optional[str] = None,
            endTime: Optional[str] = None
    ) -> pd.DataFrame:
        """
        A method to retrieve volatility data for a period longer than 30 days.
        If the range between startTime and endTime exceeds 30 days, this method
        calls fetch_historical_volatility multiple times in 30-day increments and
        concatenates the results.

        Args:
            category (str): The category of the instrument (e.g., "option").
            baseCoin (str): The base coin (e.g., "ETH", "BTC").
            period (int): The period in minutes for the volatility data.
            startTime (Optional[str]): The start time in 'YYYY-MM-DD HH:MM:SS+ZZ' or 'YYYY-MM-DDTHH:MM:SSZ' format.
            endTime (Optional[str]): The end time in 'YYYY-MM-DD HH:MM:SS+ZZ' or 'YYYY-MM-DDTHH:MM:SSZ' format.

        Returns:
            pd.DataFrame: A DataFrame containing ['period', 'value', 'time', 'date'] columns.
        """
        # If neither startTime nor endTime is provided, directly call fetch_historical_volatility
        if startTime is None and endTime is None:
            return self.fetch_historical_volatility(category, baseCoin, period, startTime, endTime)

        # If only one of startTime or endTime is provided, raise an error
        if startTime is None or endTime is None:
            raise ValueError("Either both startTime and endTime must be provided or neither.")

        # Convert timestamps to Unix time (milliseconds)
        start_unix_ms = int(convert_datetime_to_unix_time(startTime) * 1000)
        end_unix_ms = int(convert_datetime_to_unix_time(endTime) * 1000)

        if start_unix_ms > end_unix_ms:
            raise ValueError("startTime must be earlier than endTime.")

        # 30 days in milliseconds (30 * 24 * 60 * 60 * 1000)
        thirty_days_ms = 30 * 24 * 60 * 60 * 1000
        total_diff = end_unix_ms - start_unix_ms

        # If the time range is within 30 days, just fetch once
        if total_diff <= thirty_days_ms:
            return self.fetch_historical_volatility(category, baseCoin, period, startTime, endTime)

        # If the range exceeds 30 days, fetch data in 30-day increments
        current_start = start_unix_ms
        all_dfs = []
        while current_start < end_unix_ms:
            current_end = min(current_start + thirty_days_ms, end_unix_ms)
            # Convert Unix time back to the original datetime format
            _start_str = unix_time_to_datetime(current_start / 1000.0).strftime("%Y-%m-%d %H:%M:%S%z")
            _end_str = unix_time_to_datetime(current_end / 1000.0).strftime("%Y-%m-%d %H:%M:%S%z")

            # Fetch data for each segment of up to 30 days
            df_part = self.fetch_historical_volatility(category, baseCoin, period, _start_str, _end_str)
            all_dfs.append(df_part)

            # To avoid hitting the rate limit, wait for 1 second between calls
            import time
            time.sleep(1)

            current_start = current_end

        # Concatenate all DataFrames
        result_df = pd.concat(all_dfs, ignore_index=True)

        # Remove duplicates and sort by time
        result_df = result_df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

        return result_df

def main():
    import time
    api = BybitDataFetcher()
    # Example usage of fetch_tickers method


    result = api.fetch_historical_volatility_extended(
        category="option",
        baseCoin="ETH",
        period=7,
        startTime="2024-01-15 00:00:00+0900",
        endTime="2024-06-02 00:00:00+0900"
    )
    print(result)
    exit()

    # Get list of symbols from instrument inf
    # Get current UTC timestamp
    # Iterate through symbols in order
    base_coin = "BTC"
    instrument_info = api.fetch_instruments_info(category="option", baseCoin=base_coin)
    print(instrument_info)

    current_utc = int(time.time_ns() // 1_000_000_000)
    date = datetime.fromtimestamp(current_utc, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    tickers_data = api.fetch_tickers(category="option", baseCoin=base_coin)

    tickers_data["symbol_id"] = tickers_data["symbol"] + "_" + date
    tickers_data["date"] = date
    print(tickers_data)

    exit()
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

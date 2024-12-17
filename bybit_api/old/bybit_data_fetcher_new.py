from datetime import datetime
import sys
import os
import pandas as pd

from typing import Optional, Tuple
from pybit.unified_trading import HTTP, HTTPError

# Get the absolute path of the directory where the current file is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory path
parent_dir = os.path.dirname(current_dir)
# Add the parent directory path to sys.path
sys.path.append(parent_dir)

from common.constants import *
from bybit_api.bybit_base_api import BybitBaseAPI, MAX_LIMIT_QUERY, CATEGORY


def convert_datetime_to_unix_time(dt_string: str, fmt: str = '%Y-%m-%d %H:%M:%S%z') -> float:
    """
    Converts a datetime string in a specified format to a UNIX timestamp.

    Args:
        dt_string (str): The datetime string to convert.
        fmt (str): The format of the datetime string. Defaults to '%Y-%m-%d %H:%M:%S%z'.

    Returns:
        float: The UNIX timestamp corresponding to the datetime string.
    """
    if "T" in dt_string and "Z" in dt_string:
        # If the datetime string includes "T" and "Z", use the ISO 8601 format.
        fmt = '%Y-%m-%dT%H:%M:%SZ'
    return datetime.strptime(dt_string, fmt).timestamp()


def unix_time_to_datetime(unix_time: float) -> datetime:
    """
    Converts a UNIX timestamp to a UTC datetime object.

    Args:
        unix_time (float): The UNIX timestamp to convert.

    Returns:
        datetime: The UTC datetime object corresponding to the UNIX timestamp.
    """
    return datetime.utcfromtimestamp(unix_time)


def interval_to_seconds(interval: int | str) -> int:
    """
    Converts an interval in minutes, days ('D'), or weeks ('W') to seconds.

    Args:
        interval (int or str): The interval to convert. Can be an integer representing minutes,
            or a string 'D' or 'W' for days and weeks, respectively.

    Returns:
        int: The interval in seconds.

    Raises:
        ValueError: If the interval format is invalid.
        TypeError: If the interval type is neither int nor str.
    """
    if isinstance(interval, int):
        # If the interval is an integer, assume it's in minutes and convert to seconds
        return interval * 60
    elif isinstance(interval, str):
        # If the interval is a string, check if it ends with 'D' or 'W'
        interval_seconds = 0
        if interval.endswith("D"):
            # If it ends with 'D', it represents days. Convert to seconds.
            interval_seconds = 24 * 60 * 60  # 1 day in seconds
        elif interval.endswith("W"):
            # If it ends with 'W', it represents weeks. Convert to seconds.
            interval_seconds = 7 * 24 * 60 * 60  # 1 week in seconds
        else:
            # If it's not an integer, 'D', or 'W', raise a ValueError.
            raise ValueError(
                "Invalid interval format. Interval must be an integer, 'D', or 'W'."
            )
        return interval_seconds
    else:
        # If the interval type is neither int nor str, raise a TypeError.
        raise TypeError("Interval must be an integer or a string.")


class BybitDataFetcher(BybitBaseAPI):
    def __init__(self):
        super().__init__()

    def fetch_latest_info(self) -> float:
        """Fetches the latest price of the specified symbol.

        Returns:
            float: The latest price.

        Raises:
            RuntimeError: If fetching the latest price fails.
        """
        try:
            session = HTTP(testnet=self._isTESTNET)
            result = session.get_tickers(category=CATEGORY, symbol=self._symbol)
            last_price = float(result["result"]["list"][0]["lastPrice"])
            return last_price
        except HTTPError as e:
            # If the Bybit API request fails, raise a RuntimeError with details.
            raise RuntimeError(f"ByBit API request failed: {e}") from e
        except KeyError as e:
            # If the API response format is unexpected, raise a RuntimeError.
            raise RuntimeError(f"Unexpected API response format: {e}") from e
        except Exception as e:
            # If any other error occurs during the process, raise a generic RuntimeError.
            raise RuntimeError(f"An error occurred while fetching latest info: {e}") from e

    def _convert_interval(self) -> str:
        """Converts the internal time interval to the format expected by the Bybit API.

        Returns:
            str: The time interval in the format expected by the Bybit API.
        """
        interval_mapping = {
            5: "5min",
            15: "15min",
            30: "30min",
            60: "1h",
            120: "2h",
            240: "4h",
            "D": "1d",
        }
        return interval_mapping.get(self._interval, "1d")

    def _fetch_data_by_type(
        self, start_time: float, end_time: float, limit: int, data_type: str
    ) -> list:
        """
        Fetches data of the specified type for the specified period.

        Args:
            start_time (float): Start time (Unix timestamp).
            end_time (float): End time (Unix timestamp).
            limit (int): Maximum number of records to retrieve.
            data_type (str): Data type. One of 'price', 'oi', 'funding_rate', 'premium_index'.

        Returns:
            list: A list of fetched data.

        Raises:
            ValueError: If an invalid data type is specified.
            RuntimeError: If fetching the data fails.
        """
        session = HTTP(testnet=self._isTESTNET)
        try:
            if data_type == "price":
                result = session.get_kline(
                    category=CATEGORY,
                    symbol=self._symbol,
                    interval=self._interval,
                    start=int(start_time * 1000),
                    end=int(end_time * 1000),
                    limit=limit,
                )
            elif data_type == "oi":
                result = session.get_open_interest(
                    category=CATEGORY,
                    symbol=self._symbol,
                    intervalTime=self._convert_interval(),
                    startTime=int(start_time * 1000),
                    endTime=int(end_time * 1000),
                    limit=limit,
                )
            elif data_type == "funding_rate":
                result = session.get_funding_rate_history(
                    category=CATEGORY,
                    symbol=self._symbol,
                    intervalTime=self._convert_interval(),
                    startTime=int(start_time * 1000),
                    endTime=int(end_time * 1000),
                    limit=limit,
                )
            elif data_type == "premium_index":
                result = session.get_premium_index_price_kline(
                    category=CATEGORY,
                    symbol=self._symbol,
                    intervalTime=self._interval,
                    startTime=int(start_time * 1000),
                    endTime=int(end_time * 1000),
                    limit=limit,
                )
            else:
                raise ValueError("Invalid data type specified")

            return result["result"]["list"]
        except HTTPError as e:
            # If the Bybit API request fails, raise a RuntimeError.
            raise RuntimeError(f"ByBit API request failed: {e}") from e
        except KeyError as e:
            # If the API response format is unexpected, raise a RuntimeError.
            raise RuntimeError(f"Unexpected API response format: {e}") from e
        except Exception as e:
            # If any other error occurs while fetching data, raise a generic RuntimeError.
            raise RuntimeError(f"An error occurred while fetching {data_type} data: {e}") from e

    def fetch_data(
        self, fromtime: str, totime: str, data_type: str, column_names: list, savefilename: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetches data for the specified period and optionally saves it to a CSV file.

        Args:
            fromtime (str): Start time.
            totime (str): End time.
            data_type (str): Data type. One of 'price', 'oi', 'funding_rate', 'premium_index'.
            column_names (list): Column names for the DataFrame.
            savefilename (str, optional): Save file name. Defaults to None.

        Returns:
            pd.DataFrame: The fetched data.

        Raises:
            RuntimeError: If fetching the data fails.
        """
        start_unix = convert_datetime_to_unix_time(fromtime)
        end_unix = convert_datetime_to_unix_time(totime)

        all_data = []
        while start_unix < end_unix:
            next_end_unix = min(
                start_unix + MAX_LIMIT_QUERY * interval_to_seconds(self._interval),
                end_unix,
            )
            try:
                data = self._fetch_data_by_type(start_unix, next_end_unix, MAX_LIMIT_QUERY, data_type)
            except RuntimeError as e:
                raise RuntimeError(f"Failed to fetch data: {e}") from e

            all_data.extend(data)
            start_unix = next_end_unix

        df = pd.DataFrame.from_records(all_data)
        df.columns = column_names
        df["start_at"] = df["start_at"].astype(float) / 1000
        df["date"] = pd.to_datetime(df["start_at"], unit="s")
        df["index"] = df["start_at"]
        df = df.set_index("index").sort_index(ascending=True)

        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        if savefilename:
            try:
                df.to_csv(savefilename)
            except Exception as e:
                raise RuntimeError(f"Failed to save data to CSV: {e}") from e

        return df

    def save_file_name(self, fromtime: str, totime: str, data_type: str) -> str:
        """Generates a file name for saving data based on the given parameters.

        Args:
            fromtime (str): Start time.
            totime (str): End time.
            data_type (str): Data type.

        Returns:
            str: The generated file name.
        """
        if data_type == "price":
            interval = self._interval
        elif data_type == "oi":
            interval = interval_to_seconds(self._interval)
        elif data_type == "funding_rate":
            interval = 480
        else:
            raise ValueError("Invalid data type specified")

        formatted_fromtime = fromtime[:-9].replace(":", "").replace("-", "").replace(" ", "")
        formatted_totime = totime[:-9].replace(":", "").replace("-", "").replace(" ", "")
        savefilename = (
            f"{self._symbol}_{formatted_fromtime}_{formatted_totime}_{interval}_{data_type}.csv"
        )
        return self._datapath + savefilename

    def fetch_historical_data(
        self, fromtime: str, totime: str, data_type: str, savefilename: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetches historical data for the specified period and optionally saves it to a CSV file.

        Args:
            fromtime (str): Start time.
            totime (str): End time.
            data_type (str): Data type. One of 'price', 'oi', 'funding_rate', 'premium_index'.
            savefilename (str, optional): Save file name. Defaults to None.

        Returns:
            Optional[pd.DataFrame]: The fetched historical data. Returns None if fetching fails.

        Raises:
            ValueError: If an invalid data type is specified.
        """
        column_names = {
            "price": ["start_at", "open", "high", "low", "close", "volume", "turnover"],
            "oi": ["oi", "start_at"],
            "funding_rate": ["symbol", "funding_rate", "start_at"],
            "premium_index": ["start_at", "p_open", "p_high", "p_low", "p_close"],
        }

        if data_type not in column_names:
            raise ValueError("Invalid data type specified")

        try:
            df = self.fetch_data(fromtime, totime, data_type, column_names[data_type], savefilename)
        except RuntimeError as e:
            # If fetching historical data encounters an error, log the error message and return None.
            self._logger.log_system_message(f"Failed to fetch historical data: {e}")
            return None

        return df

    def fetch_historical_data_all(
        self, fromtime: str, totime: str, savefilename: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetches all types of historical data for the specified period and optionally saves it to a CSV file.

        Args:
            fromtime (str): Start time.
            totime (str): End time.
            savefilename (str, optional): Save file name. Defaults to None.

        Returns:
            Optional[pd.DataFrame]: The fetched historical data. Returns None if fetching fails.
        """
        try:
            price_data = self.fetch_historical_data(fromtime, totime, "price", savefilename)
        except Exception as e:
            # If fetching all historical data encounters an error, log the message and return None.
            self._logger.log_system_message(f"Failed to fetch all historical data: {e}")
            return None

        if price_data is None:
            return None

        price_data = price_data.drop_duplicates(subset=["date"], keep="first")

        if savefilename is None:
            savefilename = self.save_file_name(fromtime, totime, "price")

        try:
            price_data.to_csv(savefilename)
        except Exception as e:
            raise RuntimeError(f"Failed to save data to CSV: {e}") from e

        return price_data
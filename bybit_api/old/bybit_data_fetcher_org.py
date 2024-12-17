
from datetime import datetime
import sys,os
import pandas as pd

from typing import Optional
from pybit.unified_trading import HTTP
from typing import Tuple


# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)

from common.constants import *
from bybit_api.bybit_base_api import BybitBaseAPI, MAX_LIMIT_QUERY, CATEGORY

def convert_datetime_to_unix_time(dt_string, fmt='%Y-%m-%d %H:%M:%S%z'):
    """
    Converts a datetime string in a specified format to a UNIX timestamp.

    Args:
        dt_string (str): The datetime string to convert.
        fmt (str): The format of the datetime string. Defaults to '%Y-%m-%d %H:%M:%S%z'.

    Returns:
        float: The UNIX timestamp corresponding to the datetime string.
    """
    if "T" in dt_string and "Z" in dt_string:
        fmt = '%Y-%m-%dT%H:%M:%SZ'
    return datetime.strptime(dt_string, fmt).timestamp()

def unix_time_to_datetime(unix_time):
    """
    Converts a UNIX timestamp to a UTC datetime object.

    Args:
        unix_time (float): The UNIX timestamp to convert.

    Returns:
        datetime: The UTC datetime object corresponding to the UNIX timestamp.
    """
    return datetime.utcfromtimestamp(unix_time)

def interval_to_seconds(interval):
    """
    Converts an interval in minutes, days ('D'), or weeks ('W') to seconds.

    Args:
        interval (int or str): The interval to convert. Can be an integer representing minutes, or a string 'D' or 'W' for days and weeks, respectively.

    Returns:
        int: The interval in seconds.

    Raises:
        ValueError: If the interval format is invalid.
        TypeError: If the interval type is neither int nor str.
    """
    # Check if the interval is an integer
    if isinstance(interval, int):
        # If it's an integer, assume it's in minutes and convert to seconds
        return interval * 60
    elif isinstance(interval, str):
        # Process the string format for 'D' and 'W'
        interval_seconds = 0
        if interval.endswith("D"):
            interval_seconds = 24 * 60 * 60  # 1 day in seconds
        elif interval.endswith("W"):
            interval_seconds = 7 * 24 * 60 * 60  # 1 week in seconds
        else:
            raise ValueError("Invalid interval format. Interval must be an integer, 'D', or 'W'.")
        return interval_seconds
    else:
        raise TypeError("Interval must be an integer or a string.")


class BybitDataFetcher(BybitBaseAPI):
    def __init__(self):
        super().__init__()

    def fetch_latest_info(self) -> Tuple[bool, Optional[float]]:
        try:
            session = HTTP(testnet=self._isTESTNET)
            result = session.get_tickers(category=CATEGORY, symbol=self._symbol)
            last_price = float(result["result"]["list"][0]["lastPrice"])
            return last_price
        except Exception as e:
            self._logger.log_system_message(f'ByBit HTTP Access Error: {e}')
            return None

    def _convert_interval(self):
        interval_mapping = {
            5: "5min",
            15: "15min",
            30: "30min",
            60: "1h",
            120: "2h",
            240: "4h",
            "D": "1d"
        }
        return interval_mapping.get(self._interval, "1d")

    def data_fetcher(self, start_time, end_time, limit, data_type):
        session = HTTP(testnet=self._isTESTNET)

        try:
            if data_type == 'price':
                result = session.get_kline(
                    category=CATEGORY,
                    symbol=self._symbol,
                    interval=self._interval,
                    start=start_time * 1000,
                    end=end_time * 1000,
                    limit=limit)
            elif data_type == 'oi':
                result = session.get_open_interest(
                    category=CATEGORY,
                    symbol=self._symbol,
                    intervalTime=self._convert_interval(),
                    startTime=start_time * 1000,
                    endTime=end_time * 1000,
                    limit=limit)
            elif data_type == 'funding_rate':
                result = session.get_funding_rate_history(
                    category=CATEGORY,
                    symbol=self._symbol,
                    intervalTime=self._convert_interval(),
                    startTime=start_time * 1000,
                    endTime=end_time * 1000,
                    limit=limit)
            elif data_type == 'premium_index':
                result = session.get_premium_index_price_kline(
                    category=CATEGORY,
                    symbol=self._symbol,
                    intervalTime=self._interval,
                    startTime=start_time * 1000,
                    endTime=end_time * 1000,
                    limit=limit)
            else:
                raise ValueError("Invalid data type specified")

            return True, result["result"]["list"]
        except Exception as e:
            self._logger.log_system_message(f'ByBit HTTP Access Error: {e}')
        return False, []

    def fetch_data(self, fromtime, totime, data_type, column_names, savefilename=None):
        start_unix = convert_datetime_to_unix_time(fromtime)
        end_unix = convert_datetime_to_unix_time(totime)

        all_data = []
        while start_unix < end_unix:
            next_end_unix = min(start_unix + MAX_LIMIT_QUERY * interval_to_seconds(self._interval), end_unix)
            success, data = self.data_fetcher(start_unix, next_end_unix, MAX_LIMIT_QUERY,data_type)
            if not success:
                return False, all_data

            all_data.extend(data)
            start_unix = next_end_unix

        df = pd.DataFrame.from_records(all_data)
        df.columns = column_names
        df['start_at'] = df['start_at'].astype(float) / 1000
        df['date'] = pd.to_datetime(df['start_at'], unit='s')
        df['index'] =  df['start_at']
        df = df.set_index('index').sort_index(ascending=True)

        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])

        if savefilename:
            df.to_csv(savefilename)

        return True, df

    def save_file_name(self, fromtime, totime, data_type):
        if data_type == 'price':
            interval = self._interval
        elif data_type == 'oi':
            interval = interval_to_seconds(self._interval)
        elif data_type == 'funding_rate':
            interval = 480
        else:
            raise ValueError("Invalid data type specified")

        formatted_fromtime = fromtime[:-9].replace(':', '').replace('-', '').replace(' ', '')
        formatted_totime = totime[:-9].replace(':', '').replace('-', '').replace(' ', '')
        savefilename = f"{self._symbol}_{formatted_fromtime}_{formatted_totime}_{interval}_{data_type}.csv"
        return self._datapath + savefilename

    def fetch_historical_data(self, fromtime, totime, data_type, savefilename=None):
        column_names = {
            'price': ['start_at', 'open', 'high', 'low', 'close', 'volume', 'turnover'],
            'oi': ['oi', 'start_at'],
            'funding_rate': ['symbol', 'funding_rate', 'start_at'],
            'premium_index': ['start_at', 'p_open', 'p_high', 'p_low', 'p_close']
        }

        if data_type not in column_names:
            raise ValueError("Invalid data type specified")

        success, df = self.fetch_data(fromtime, totime, data_type, column_names[data_type])

        if not success:
            return False, None

        return True, df

    def fetch_historical_data_all(self, fromtime, totime, savefilename=None):
        flag, price_data = self.fetch_historical_data(fromtime, totime, 'price')
        if not flag:
            return None

        price_data = price_data.drop_duplicates(subset=['date'], keep='first')

        if savefilename is None:
            savefilename = self.save_file_name(fromtime, totime, 'price')

        price_data.to_csv(savefilename)
        return price_data
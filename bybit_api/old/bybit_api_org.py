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

from dependency_injector.wiring import inject, Provide

from common.trading_logger import TradingLogger
from common.config_manager import ConfigManager
from common.container import Container

MAX_LIMIT_QUERY = 200

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


class BybitOnlineAPI:
    """
    BybitのUSDTパーペチュアルオンラインAPIとのインタラクションを管理するクラスです。

    Attributes:
        __logger (TradingLogger): トレーディングに関するログ情報を管理するオブジェクト。
        __config_manager (ConfigManager): 設定情報を管理するオブジェクト。
        __interval (str): データフェッチの間隔を指定する文字列。
        __symbol (str): 取引シンボル。
        __datapath (str): データ保存パス。
        __api_key (str): APIキー。
        __api_secret (str): APIシークレット。
        __isTESTNET (bool): テストネットを使用するかどうか。

    Args:
        config_manager (ConfigManager): 設定情報を管理するオブジェクト。
        trading_logger (TradingLogger): トレーディングに関するログ情報を管理するオブジェクト。
    """
    @inject
    def __init__(self, config_manager: ConfigManager = Provide[Container.config_manager],
                 trading_logger: TradingLogger = Provide[Container.trading_logger_singleton]):
        self.__logger = trading_logger
        self.__config_manager = config_manager
        self.__interval = self.__config_manager.get('ONLINE', 'INTERVAL')  # Changed to private
        self.__symbol = self.__config_manager.get('ONLINE', 'SYMBOL')  # Changed to private

        self.__datapath = parent_dir + '/' + self.__config_manager.get('DATA', 'TSTPATH')
        api_config = self.__config_manager.get('BYBIT_API')

        self.__api_key = api_config['API_KEY']  # Changed to private
        self.__api_secret = api_config['API_SECRET']  # Changed to private
        self.__logger = TradingLogger.get_instance()
        self.__isTESTNET = api_config['TESTNET']  # Changed to private

    def get_symbol(self):
        """
        現在設定されている取引シンボルを取得します。

        Returns:
            str: 取引シンボル。
        """
        return self.__symbol

    def get_interval(self):
        """
        現在設定されているデータフェッチの間隔を取得します。

        Returns:
            str: データフェッチの間隔。
        """
        return self.__interval

    def fetch_latest_info(self) -> Tuple[bool, Optional[float]]:
        """
        Fetches the latest price information for the configured symbol.

        Returns:
            Tuple[bool, Optional[float]]: A tuple containing a success flag and the last price. If an error occurs, the last price is None.
        """
        try:
            session = HTTP(testnet=self.__isTESTNET)
            result = session.get_tickers(category="linear", symbol=self.__symbol)
            last_price = float(result["result"]["list"][0]["lastPrice"])
            return True, last_price
        except Exception as e:
            self.__logger.log_system_message(f'ByBit HTTP Access Error: {e}')
            return False, None


    def __convert_interval(self):
        """
        設定された間隔をAPIが要求するフォーマットに変換します。

        Returns:
            str: APIが認識する間隔のフォーマット。対応する間隔がない場合は"1d"を返します。

        Note:
            このメソッドはプライベートメソッドです。外部から直接呼び出すことは想定されていません。
        """
        interval_mapping = {
            5: "5min",
            15: "15min",
            30: "30min",
            60: "1h",
            120: "2h",
            240: "4h",
            "D": "1d"
        }
        return interval_mapping.get(self.__interval, "1d")  # デフォルト値は "1d"

    def data_fetcher(self, start_time, end_time, limit, data_type):
        """
        指定された条件でAPIからデータをフェッチします。

        Args:
            start_time (int): データフェッチの開始時間（UNIXタイムスタンプ）。
            end_time (int): データフェッチの終了時間（UNIXタイムスタンプ）。
            limit (int): フェッチする最大レコード数。
            data_type (str): フェッチするデータのタイプ（'price', 'oi', 'funding_rate', 'premium_index'）。

        Returns:
            Tuple[bool, list]: フェッチの成功状況とフェッチされたデータのリスト。

        Raises:
            ValueError: 不正なデータタイプが指定された場合。
        """
        session = HTTP(testnet=self.__isTESTNET)

        try:
            if data_type == 'price':
                result = session.get_kline(
                    category="linear",
                    symbol=self.__symbol,
                    interval=self.__interval,
                    start=start_time * 1000,
                    end=end_time * 1000,
                    limit=limit)
            elif data_type == 'oi':
                result = session.get_open_interest(
                    category='linear',
                    symbol=self.__symbol,
                    intervalTime=self.__convert_interval(),
                    startTime=start_time * 1000,
                    endTime=end_time * 1000,
                    limit=limit)
            elif data_type == 'funding_rate':
                result = session.get_funding_rate_history(
                    category='linear',
                    symbol=self.__symbol,
                    intervalTime=self.__convert_interval(),
                    startTime=start_time * 1000,
                    endTime=end_time * 1000,
                    limit=limit)
            elif data_type == 'premium_index':
                result = session.get_premium_index_price_kline(
                    category='linear',
                    symbol=self.__symbol,
                    intervalTime=self.__interval,
                    startTime=start_time * 1000,
                    endTime=end_time * 1000,
                    limit=limit)
            else:
                raise ValueError("Invalid data type specified")


            return True, result["result"]["list"]
        except Exception as e:
            self.__logger.log_system_message(f'ByBit HTTP Access Error: {e}')
        return False, []

    def fetch_data(self, fromtime, totime, data_type, column_names, savefilename=None):
        """
        指定された期間とデータタイプに基づいてデータをフェッチし、オプションでCSVファイルに保存します。

        Args:
            fromtime (str): データの開始時間。
            totime (str): データの終了時間。
            data_type (str): データのタイプ。
            column_names (list of str): データフレームのカラム名。
            savefilename (Optional[str]): データを保存するファイル名。指定しない場合は保存されません。

        Returns:
            Tuple[bool, pandas.DataFrame]: データフェッチの成功状況とフェッチされたデータのデータフレーム。
        """
        start_unix = convert_datetime_to_unix_time(fromtime)
        end_unix = convert_datetime_to_unix_time(totime)

        all_data = []
        while start_unix < end_unix:
            next_end_unix = min(start_unix + MAX_LIMIT_QUERY * interval_to_seconds(self.__interval), end_unix)
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

        # "Unnamed: 0" という列がある場合は削除
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])

        if savefilename:
            df.to_csv(savefilename)

        return True, df


    def save_file_name(self, fromtime, totime, data_type):
        """
        ファイル名を生成します。

        Args:
            fromtime (str): データの開始時間。
            totime (str): データの終了時間。
            data_type (str): データの種類('price', 'oi', 'funding_rate'等)。

        Returns:
            str: 生成されたファイル名。

        Raises:
            ValueError: 不正なデータタイプが指定された場合。
        """
        if data_type == 'price':
            interval = self.__interval
        elif data_type == 'oi':
            interval = interval_to_seconds(self.__interval)
        elif data_type == 'funding_rate':
            interval = 480
        else:
            raise ValueError("Invalid data type specified")

        formatted_fromtime = fromtime[:-9].replace(':', '').replace('-', '').replace(' ', '')
        formatted_totime = totime[:-9].replace(':', '').replace('-', '').replace(' ', '')
        savefilename = f"{self.__symbol}_{formatted_fromtime}_{formatted_totime}_{interval}_{data_type}.csv"
        return self.__datapath + savefilename

    # ... [既存の初期化メソッドと他のメソッド]

    def fetch_historical_data(self, fromtime, totime, data_type, savefilename=None):
        """
        指定されたデータタイプに基づいて履歴データをフェッチします。このメソッドは、価格データ、オープンインタレスト、ファンディングレート、またはプレミアムインデックスデータを取得するために使用されます。

        Args:
            fromtime (str): データフェッチの開始時刻。'%Y-%m-%d %H:%M:%S%z'の形式で指定します。
            totime (str): データフェッチの終了時刻。'%Y-%m-%d %H:%M:%S%z'の形式で指定します。
            data_type (str): フェッチするデータのタイプ。'price'、'oi'（オープンインタレスト）、'funding_rate'、または'premium_index'が指定可能です。
            savefilename (Optional[str]): データを保存するファイルの名前。指定しない場合、データはファイルに保存されません。

        Returns:
            Tuple[bool, Optional[pandas.DataFrame]]: 成功フラグとフェッチされたデータを含むデータフレーム。フェッチに失敗した場合は、FalseとNoneを返します。

        Raises:
            ValueError: 指定された`data_type`がサポートされていない場合に発生します。

        Examples:
            >>> bybit_api = BybitOnlineAPI(config_manager, trading_logger)
            >>> success, data = bybit_api.fetch_historical_data("2020-01-01 00:00:00+0000", "2020-01-02 00:00:00+0000", "price")
            >>> if success:
            ...     print(data)
            ... else:
            ...     print("Data fetch failed")
        """
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
        """
        指定された期間の全てのデータタイプをフェッチし、マージしたデータを返します。

        Args:
            fromtime (str): データの開始時間。
            totime (str): データの終了時間。
            savefilename (Optional[str]): 保存するファイル名。指定しない場合は自動で生成されます。

        Returns:
            pandas.DataFrame: マージされたデータフレーム。

        Note:
            データタイプには'price', 'oi', 'funding_rate', 'premium_index'が含まれます。
        """
        # データを取得
        flag, price_data = self.fetch_historical_data(fromtime, totime, 'price')
        if not flag:
            return None

        price_data = price_data.drop_duplicates(subset=['date'], keep='first')

        # ファイルに保存
        if savefilename is None:
            savefilename = self.save_file_name(fromtime, totime, 'price')

        price_data.to_csv(savefilename)
        return price_data
    """
    def fetch_historical_data_all(self, fromtime, totime, savefilename=None):

        指定された期間の全てのデータタイプをフェッチし、マージしたデータを返します。

        Args:
            fromtime (str): データの開始時間。
            totime (str): データの終了時間。
            savefilename (Optional[str]): 保存するファイル名。指定しない場合は自動で生成されます。

        Returns:
            pandas.DataFrame: マージされたデータフレーム。

        Note:
            データタイプには'price', 'oi', 'funding_rate', 'premium_index'が含まれます。

        # データを取得
        flag, price_data = self.fetch_historical_data(fromtime, totime, 'price')
        if not flag:
            return None
        flag, funding_rate_data = self.fetch_historical_data(fromtime, totime, 'funding_rate')
        if not flag:
            return None
        flag, oi_data = self.fetch_historical_data(fromtime, totime, 'oi')
        if not flag:
            return None
        flag, premium_index = self.fetch_historical_data(fromtime, totime, 'premium_index')
        if not flag:
            return None

        # データのマージ
        merged_data = self.merge_data(price_data, funding_rate_data, oi_data,premium_index)
        merged_data = merged_data.drop_duplicates(subset=['date'], keep='first')

        # ファイルに保存
        if savefilename is None:
            savefilename = self.save_file_name(fromtime, totime, 'price')

        merged_data.to_csv(savefilename)
        return merged_data
    """
    def merge_data(self, price_data, funding_rate_data, oi_data, premium_index):
        """
        指定されたデータフレームをマージします。

        Args:
            price_data (pandas.DataFrame): 価格データ。
            funding_rate_data (pandas.DataFrame): ファンディングレートデータ。
            oi_data (pandas.DataFrame): オープンインタレストデータ。
            premium_index (pandas.DataFrame): プレミアムインデックスデータ。

        Returns:
            pandas.DataFrame: マージされたデータフレーム。
        """
        # 'date' 列を基準に各データをリセット
        price_data = price_data.reset_index()
        funding_rate_data = funding_rate_data.reset_index()
        oi_data = oi_data.reset_index()
        premium_index = premium_index.reset_index()

        # ファンディングレートのデータ補完
        price_data['funding_rate'] = price_data['date'].apply(
            lambda x: self.find_latest_before_date(funding_rate_data, x)
        )

        # プレミアムインデックスのデータ補完
        price_data['p_close'] = price_data['date'].apply(
            lambda x: self.find_latest_before_date(premium_index, x, column='p_close')
        )

        # データの結合
        merged_data = pd.merge(price_data, oi_data[['date', 'oi']], on='date', how='left')

        return merged_data

    @staticmethod
    def find_latest_before_date(data, date, column='funding_rate'):
        """
        指定された日付以前の最新のデータを検索します。

        Args:
            data (pandas.DataFrame): 検索対象のデータフレーム。
            date (datetime.datetime): 基準日付。
            column (str): データを検索する列名。

        Returns:
            Any: 指定された列の最新の値。該当するデータがない場合はNone。
        """
        filtered = data[data['date'] <= date]
        if not filtered.empty:
            return filtered.iloc[-1][column]
        return None


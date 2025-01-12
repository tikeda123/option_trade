import os
import sys
from datetime import datetime, timedelta, timezone
import pandas as pd
import time

# Add the parent directory (A directory) to the system path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from bybit_api.bybit_data_fetcher import BybitDataFetcher
from common.constants import MARKET_DATA
from mongodb.data_loader_mongo import MongoDataLoader
from trading_analysis_kit.technical_analyzer import TechnicalAnalyzer


class DataLoaderOnline(MongoDataLoader):
        """
        Loads and updates market data from the Bybit API, stores it in MongoDB,
        and converts it to technical analysis data.
        """

        def __init__(self):
                """
                Initializes the DataLoaderOnline object.
                """
                super().__init__()
                self.bybit_api = BybitDataFetcher()
                self.technical_analyzer = TechnicalAnalyzer()

        def fetch_historical_data(self, start_date: str, end_date: str, symbol: str = None, interval: int = None) -> pd.DataFrame:
                """
                Fetches historical market data for the given symbol and interval between the specified dates.

                Args:
                        start_date (str): The start date in 'YYYY-MM-DD HH:MM:SS' format.
                        end_date (str): The end date in 'YYYY-MM-DD HH:MM:SS' format.
                        symbol (str, optional): The trading symbol. Defaults to None.
                        interval (int, optional): The data interval in minutes. Defaults to None.

                Returns:
                        pd.DataFrame: A Pandas DataFrame containing the fetched historical data.
                """
                data = self.bybit_api.fetch_historical_data_all(start_date, end_date, symbol=symbol, interval=interval)
                if data is None:
                        print(f"No data fetched for {symbol} {interval}")
                        return None
                df = self.convert_marketdata(data)
                self.insert_data(df, MARKET_DATA, symbol=symbol, interval=interval)
                return df

        def convert_to_local_timezone(self, str_date:datetime) -> str:
                """
                Converts a date string to the local timezone (JST).

                Args:
                        str_date (str): The date string in 'YYYY-MM-DD HH:MM:SS' format.

                Returns:
                        str: The date string converted to JST timezone in 'YYYY-MM-DD HH:MM:SS%z' format.
                """
                dt_with_tz = str_date.replace(tzinfo=timezone(timedelta(hours=9)))
                dt_tokyo = dt_with_tz.strftime('%Y-%m-%d %H:%M:%S%z')
                return dt_tokyo

        def update_historical_data(self, symbol: str = None, interval: int = None) -> pd.DataFrame:
                """
                Updates historical data for the given symbol and interval up to the current time.

                Args:
                        symbol (str, optional): The trading symbol. Defaults to None.
                        interval (int, optional): The data interval in minutes. Defaults to None.

                Returns:
                        pd.DataFrame: A Pandas DataFrame containing the updated historical data, or None if no data exists.
                """

                now = datetime.now(timezone(timedelta(hours=9)))
                end_date = now.strftime('%Y-%m-%d %H:%M:%S%z')
                df = self.load_data(MARKET_DATA, symbol=symbol, interval=interval)
                datetime_df = df['date'].iloc[-1]
                start_date = self.convert_to_local_timezone(datetime_df)
                return self.fetch_historical_data(start_date, end_date, symbol=symbol, interval=interval)

        def convert_all_historical_data_to_tech(self, symbol: str = None, interval: int = None) -> pd.DataFrame:
                """
                Converts all historical data to technical analysis data.

                Args:
                        symbol (str, optional): The trading symbol. Defaults to None.
                        interval (int, optional): The data interval in minutes. Defaults to None.

                Returns:
                        pd.DataFrame: A Pandas DataFrame containing the calculated technical analysis data.
                """
                self.technical_analyzer.load_data_from_db(MARKET_DATA, symbol=symbol, interval=interval)
                analysis_result = self.technical_analyzer.analyze()
                self.technical_analyzer.insert_data(symbol=symbol, interval=interval)
                return analysis_result

        def convert_recent_historical_data_to_tech(self, symbol: str = None, interval: int = None) -> pd.DataFrame:
                """
                Converts recent historical data to technical analysis data.

                Args:
                        symbol (str, optional): The trading symbol. Defaults to None.
                        interval (int, optional): The data interval in minutes. Defaults to None.

                Returns:
                        pd.DataFrame: A Pandas DataFrame containing the calculated technical analysis data
                                                  for the recent historical data.
                """
                res = self.technical_analyzer.load_recent_data_from_db(MARKET_DATA, symbol=symbol, interval=interval)
                analysis_result = self.technical_analyzer.analyze(res)
                self.technical_analyzer.insert_data(symbol=symbol, interval=interval)
                return analysis_result

        def update_historical_data_if_needed(self, symbol: str = None, interval: int = None) -> pd.DataFrame:
                """
                Updates historical data if the time elapsed since the last update
                is greater than the specified interval.

                Args:
                        symbol (str, optional):
                                The trading symbol. If None, all symbols will be updated.
                        interval (int, optional):
                                The data interval in minutes. If None, the default interval will be used.

                Returns:
                        pd.DataFrame:
                                The new DataFrame if data was updated, None otherwise.
                """
                df = self.load_data(MARKET_DATA, symbol=symbol, interval=interval)
                last_update_time = df['date'].iloc[-1]

                # タイムゾーンをUTCとして設定 (変更なし)
                last_update_time = pd.to_datetime(last_update_time).tz_localize(timezone.utc)
                # 現在時刻をUTCで取得
                now_utc = datetime.now(timezone.utc)
                time_difference = now_utc - last_update_time

                if time_difference > timedelta(minutes=interval):
                        self.update_historical_data(symbol=symbol, interval=interval)
                        return self.convert_recent_historical_data_to_tech(symbol=symbol, interval=interval)
                else:
                        return None


def main():
                online_data_loader = DataLoaderOnline()
                #df = online_data_loader.fetch_historical_data('2023-04-01 00:00:00+0000', '2024-04-02 00:00:00+0000', symbol='DOGEUSDT', interval=60)

                #print(df)

                #df = online_data_loader.fetch_historical_data('2020-01-01 00:00:00+0000', '2024-07-30 00:00:00+0000', symbol='BTCUSDT', interval=15)
                #df = online_data_loader.update_historical_data(symbol='BTCUSDT', interval=15)
                #df = online_data_loader.convert_recent_historical_data_to_tech(symbol='BTCUSDT', interval=15)
                #online_data_loader.update_historical_data(symbol='BTCUSDT', interval=15)
                #df = online_data_loader.convert_recent_historical_data_to_tech(symbol='BTCUSDT', interval=15)
                #print(df)
                while(1):
                        time.sleep(5)
                        df = online_data_loader.update_historical_data_if_needed(symbol='BTCUSDT', interval=5)
                        if df is not None:
                                print(df)
                        else:
                                print('No update')

                        df = online_data_loader.update_historical_data_if_needed(symbol='BTCUSDT', interval=15)
                        if df is not None:
                                print(df)
                        else:
                                print('No update')

                        df = online_data_loader.update_historical_data_if_needed(symbol='BTCUSDT', interval=30)
                        if df is not None:
                                print(df)
                        else:
                                print('No update')

                        df = online_data_loader.update_historical_data_if_needed(symbol='BTCUSDT', interval=60)
                        if df is not None:
                                print(df)
                        else:
                                print('No update')


if __name__ == '__main__':
        main()






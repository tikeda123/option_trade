import psycopg2
from psycopg2 import extras
import pandas as pd
import decimal
import os, sys

from datetime import datetime, timedelta, timezone


# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)

from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import MARKET_DATA



class DataLoaderOnline(MongoDataLoader):
        def __init__(self,table_name=None):
                super().__init__(table_name)
                from bybit_api.bybit_data_fetcher import BybitDataFetcher

                self.bybit_online_api = BybitDataFetcher()

        def fetch_historical_data_all(self, start_date, end_date, symbol: str=None, interval: int=None):
                """
                指定された期間の全てのデータを取得する。
                """

                dfres = self.bybit_online_api.fetch_historical_data_all(start_date, end_date, symbol=symbol, interval=interval)
                df = self.convert_marketdata(dfres)
                self.insert_data(df,MARKET_DATA,symbol=symbol, interval=interval)
                return dfres

        def convert_to_local_timezone(self, str_date: str) -> str:
                """
                日本時間をUTC時間に変換する。
                """
                dt = datetime.strptime(str_date, '%Y-%m-%d %H:%M:%S')
                dt_with_tz = dt.replace(tzinfo=timezone(timedelta(hours=9))) # UTCタイムゾーンを設定
                dt_tokyo = dt_with_tz.strftime('%Y-%m-%d %H:%M:%S%z')
                return dt_tokyo

        def update_historical_data_to_now(self,symbol: str=None, interval: int=None):
                """
                指定された期間のデータを更新する。
                """
                if self.is_data_exist() == False:
                        return None

                now = datetime.now(timezone(timedelta(hours=9))) # 日本標準時 (JST) で現在時刻を取得
                end_date = now.strftime('%Y-%m-%d %H:%M:%S%z')
                df = self.load_data_from_db()

                datetime_str = df['date'].iloc[-1]
                start_date = self.convert_to_local_timezone(datetime_str)
                print(start_date, end_date)

                dfres = self.fetch_historical_data_all(start_date, end_date,symbol=symbol, interval=interval)
                return dfres

        def convert_all_historical_data_to_tech(self):
                """
                すべてのヒストリカルデータをtechデータに変換する。
                """
                from trading_analysis_kit.technical_analyzer import TechnicalAnalyzer
                ta = TechnicalAnalyzer()
                ta.load_data_from_db()
                analysis_result = ta.analyze()
                ta.import_to_db()
                return analysis_result

        def convert_historical_recent_data_to_tech(self):
                """
                指定された期間のヒストリカルデータをtechデータに変換する。
                """
                from trading_analysis_kit.technical_analyzer import TechnicalAnalyzer
                ta = TechnicalAnalyzer()
                ta.load_recent_data_from_db()
                analysis_result = ta.analyze()
                ta.import_to_db()
                return analysis_result

        def load_data_from_tech_db(self, table_name=None)->pd.DataFrame:
                """
                テクニカルデータをデータベースから取得する。
                """
                from trading_analysis_kit.technical_analyzer import TechnicalAnalyzer
                ta = TechnicalAnalyzer()
                df = ta.load_data_from_tech_db()
                return df







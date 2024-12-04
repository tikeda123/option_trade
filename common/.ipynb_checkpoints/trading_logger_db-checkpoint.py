import logging
import pandas as pd
import sys
import os

from common.trading_logger import TradingLogger
from data_loader_tran import DataLoaderTransactionDB

class TradingLoggerDB(TradingLogger):
    """
    TradingLoggerクラスを継承し、データベースへの書き込み機能を追加したクラス。

    Attributes:
        __db_loader (DataLoaderTransactionDB): データベースへの書き込み用ロガー。

    Args:
        conf (dict): ロガー設定情報を含む辞書。'VERBOSE', 'LOGPATH', 'LOGFNAME', 'LOGLVL', 'DB_CONFIG'のキーを期待します。
    """
    def __init__(self):
        super().__init__()
        from common.utils import get_config
        conf = get_config('LOG')

        self._initialized = True
        self._table_name = conf["DB_TABLE_NAME"]
        self._db_flag = conf["DB_FLAG"]
        self._db_loader = DataLoaderTransactionDB()

        if self._db_flag:
            self._db_loader.create_table_trade_log(self._table_name)

    def log_transaction(self, date: str, message: str):
        """
        トランザクションをログに記録し、CSVファイルとデータベースにも追加します。

        Args:
            date (str): トランザクションの日付。
            message (str): トランザクションのメッセージ。
        """
        serial = self._db_loader.get_next_serial(self._table_name)
        new_record = {'serial': serial, 'date': date, 'message': message}
        new_df = pd.DataFrame([new_record])  # 辞書をDataFrameに変換
        self._tradelog_df = pd.concat([self._tradelog_df, new_df], ignore_index=True)  # DataFrameを連結
        self._tradelog_df.to_csv(self._logfilename_csv, index=False)
        self.log_message(f'{date}|{message}')

        # データベースへの書き込み
        if self._db_flag:
            self._db_loader.write_db(new_df, self._table_name)
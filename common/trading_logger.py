import logging
import pandas as pd
import sys
import os

# Get the absolute path of the directory containing this file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Get the path of the parent directory

# Add the path of the parent directory to sys.path
sys.path.append(parent_dir)


class TradingLogger:
    """
    トレードログとシステムログの管理

    このクラスは、トレードの実行情報とシステムの実行状況をログファイルに記録します。
    ログはテキストファイルとして保存され、トレードログは追加でCSVファイルにも保存されます。

    Attributes:
        _instance (TradingLogger): クラスの唯一のインスタンス。
        _initialized (bool): インスタンスが初期化されているかどうか。
        __verbose (bool): 詳細ログ出力が有効かどうか。
        __loglevel (int): ログレベル。
        __logfile_trade (str): トレードログファイルのパス。
        __logfile_sys (str): システムログファイルのパス。
        __logfilename_csv (str): トレードログのCSVファイル名。
        __tradelog_df (DataFrame): トレードログを保持するDataFrame。
        __logger_trade (Logger): トレードログ用のロガー。
        __logger_sys (Logger): システムログ用のロガー。

    Args:
        conf (dict): ロガー設定情報を含む辞書。'VERBOSE', 'LOGPATH', 'LOGFNAME', 'LOGLVL'のキーを期待します。
    """
    def __init__(self):
        from common.utils import get_config
        conf = get_config('LOG')
        self._initialized = True

        # 以前と同じ設定のロード
        self._verbose = conf['VERBOSE']
        log_path = conf['LOGPATH']
        log_fname = conf['LOGFNAME']
        self._loglevel = conf['LOGLVL']
        self._logfile_trade = log_path + log_fname
        self._logfile_sys = log_path + 'system_log.log'
        self._logfilename_csv = log_path + log_fname.split('.')[0] + '.csv'

        trade_columns = ['Serial', 'Date', 'Message']
        self._tradelog_df = pd.DataFrame(columns=trade_columns)

        self.__setup_logging()

    def __setup_logging(self):
        """
        ログ設定を行います。トレードログとシステムログのためのロガーを設定し、ファイルハンドラとコンソールハンドラを追加します。
        """

        self._logger_trade = logging.getLogger('trade_logger')
        self._logger_trade.setLevel(self._loglevel)

        if not self._logger_trade.handlers:
            # Trade loggerにハンドラがまだ追加されていない場合のみ追加する
            fh_trade = logging.FileHandler(self._logfile_trade)
            fh_trade.setFormatter(logging.Formatter('%(message)s'))
            self._logger_trade.addHandler(fh_trade)

            ch_trade = logging.StreamHandler()
            ch_trade.setFormatter(logging.Formatter('%(message)s'))
            self._logger_trade.addHandler(ch_trade)

        self._logger_sys = logging.getLogger('sys_logger')
        self._logger_sys.setLevel(self._loglevel)

        if not self._logger_sys.handlers:
            # System loggerにハンドラがまだ追加されていない場合のみ追加する
            fh_sys = logging.FileHandler(self._logfile_sys)
            fh_sys.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
            self._logger_sys.addHandler(fh_sys)

            ch_sys = logging.StreamHandler()
            ch_sys.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
            self._logger_sys.addHandler(ch_sys)

    def log_message(self, msg: str):
        """
        トレードログにメッセージを記録します。

        Args:
            msg (str): 記録するメッセージ。
        """
        self._logger_trade.info(msg)

    def log_debug_message(self, msg: str):
        """
        デバッグメッセージをトレードログに記録します。

        Args:
            msg (str): 記録するデバッグメッセージ。
        """
        self._logger_trade.debug(msg)


    def log_verbose_message(self, msg: str):
        """
        詳細なメッセージを条件付きでトレードログに記録します。verbose設定が有効な場合のみ記録されます。

        Args:
            msg (str): 記録する詳細メッセージ。
        """
        if self._verbose:
            self.log_message(msg)


    def log_transaction(self, date: str, message: str):
        """
        トランザクションをログに記録し、CSVファイルにも追加します。

        Args:
            date (str): トランザクションの日付。
            message (str): トランザクションのメッセージ。
        """
        new_record = {'Serial': len(self._tradelog_df) + 1, 'Date': date, 'Message': message}
        new_df = pd.DataFrame([new_record])  # 辞書をDataFrameに変換
        self._tradelog_df = pd.concat([self._tradelog_df, new_df], ignore_index=True)  # DataFrameを連結
        self._tradelog_df.to_csv(self._logfilename_csv, index=False)
        self.log_message(f'{date}|{message}')

    def log_system_message(self, msg: str):
        """
        システムログにメッセージを記録します。

        Args:
            msg (str): 記録するメッセージ。
        """
        self._logger_sys.info(msg)


from datetime import datetime
import sys,os
import time

import pandas as pd
from typing import Optional
from pybit.unified_trading import HTTP
from typing import Tuple

# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)

from common.trading_logger import TradingLogger
from common.config_manager import ConfigManager
from common.constants import *

MAX_LIMIT_QUERY = 200
CATEGORY = "linear"
LINKED_ORDER = "ai-algo"

class BybitBaseAPI:
    def __init__(self):
        self._logger = TradingLogger()
        self._config_manager = ConfigManager()
        self._interval = self._config_manager.get('INTERVAL')
        self._symbol = self._config_manager.get('SYMBOL')
        self._datapath = parent_dir + '/' + self._config_manager.get('DATA', 'TSTPATH')
        api_config = self._config_manager.get('BYBIT_API')
        self._api_key = api_config['API_KEY']
        self._api_secret = api_config['API_SECRET']
        self._isTESTNET = api_config['TESTNET']
        self._leverage =  self._config_manager.get('ACCOUNT','LEVERAGE')
        self.set_round()

    def set_round(self):
        if self._symbol == 'BTCUSDT':
            self._ROUND_DIGIT = 3
        elif self._symbol == 'ETHUSDT':
            self._ROUND_DIGIT = 2
        else:
            self._ROUND_DIGIT = 2

    def qty_round(self, qty):
        return round(float(qty), self._ROUND_DIGIT)

    def get_symbol(self):
        return self._symbol

    def get_interval(self):
        return self._interval

    def _create_session(self):
        return HTTP(testnet=self._isTESTNET, api_key=self._api_key, api_secret=self._api_secret)

    def _api_request(self, action, **kwargs):
        session = self._create_session()
        try:
            result = action(session, **kwargs)
            return True, result
        except Exception as e:
            self._logger.log_system_message(f'ByBit HTTP Access Error: {e}')
            return False, []

    def _retry_api_request(self, action, **kwargs):
        for _ in range(MAX_TRYOUT_HTTP_REQUEST):
            flag, result = self._api_request(action, **kwargs)
            if flag:
                return flag, result
            time.sleep(MAX_TRYOUT_HTTP_REQUEST_SLEEP)
        self._logger.log_system_message(f'{action.__name__}: MAX TRY OUT Error.')
        return False, []


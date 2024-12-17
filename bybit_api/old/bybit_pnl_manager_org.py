



import sys,os
from pybit.unified_trading import HTTP
import time


# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)

from common.constants import *
from bybit_api.bybit_base_api import BybitBaseAPI, CATEGORY


class BybitPnlManager(BybitBaseAPI):
    def __init__(self):
        super().__init__()

    def get_closed_pnl(self):
        action = lambda s, **kw: s.get_closed_pnl(category=CATEGORY,
                                                  symbol=self._symbol,
                                                  **kw)

        return self._retry_api_request(action, limit=1)

    def get_pnl(self):
        time.sleep(3)
        success, data = self.get_closed_pnl()
        if success:
            return float(data['result']['list'][0]['closedPnl']),float(data['result']['list'][0]['avgExitPrice'])
        else:
            raise ValueError("PNL is not found")



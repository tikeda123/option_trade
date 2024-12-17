
import sys,os
from pybit.unified_trading import HTTP


# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)

from common.constants import *
from bybit_api.bybit_base_api import BybitBaseAPI, CATEGORY

class BybitPositionManager(BybitBaseAPI):
    def __init__(self):
        super().__init__()

    def set_switch_mode(self, mode):
        action = lambda s, **kw: s.switch_position_mode(category=CATEGORY,
                                                        symbol=self._symbol,
                                                        **kw)
        return self._retry_api_request(action, mode=mode)

    def get_position_info(self):
        session = HTTP(testnet=self._isTESTNET,
                       api_key=self._api_key,
                       api_secret=self._api_secret)
        try:
            result = session.get_positions(
                category=CATEGORY, symbol=self._symbol,
            )
            return True, result
        except Exception as e:
            self._logger.log_system_message(f'ByBit HTTP Access Error: {e}')
            return False, []

    def set_leverage(self, buy_leverage, sell_leverage):
        action = lambda s, **kw: s.set_leverage(category=CATEGORY,
                                                symbol=self._symbol, **kw)

        return self._retry_api_request(action,
                                       buyLeverage=buy_leverage,
                                       sellLeverage=sell_leverage)

    def get_positions(self):
        action = lambda s, **kw: s.get_positions(category=CATEGORY,
                                                 symbol=self._symbol, **kw)
        return self._retry_api_request(action)

    def get_my_leverage(self):
        success, data = self.get_positions()
        if success and 'result' in data and 'list' in data['result']:
            leverage = data['result']['list'][0]['leverage']
            return float(leverage)
        else:
            raise ValueError("Leverage is not found")

    def set_my_leverage(self, leverage):
        current_leverage = self.get_my_leverage()

        if current_leverage == leverage:
            return True

        success, _ = self.set_leverage(str(leverage), str(leverage))
        return success

    def get_open_position_status(self):
        success, data = self.get_positions()

        if success:
            size = data['result']['list'][0]['size']
            if float(size) > 0:
                return 'position'
            else:
                return 'No position'
        else:
            raise ValueError("Position is not found")
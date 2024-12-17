

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
from bybit_api.bybit_base_api import BybitBaseAPI, CATEGORY

class BybitOrderManager(BybitBaseAPI):
        """
        Bybitでの注文管理を行うクラス
        """
        MARKET = 'Market'
        LIMIT = 'Limit'
        BUY = 'Buy'
        SELL = 'Sell'
        GTC = 'GTC'

        def place_order(self,
                                        side: str,
                                        qty: str,
                                        orderType: str = MARKET,
                                        price: str = None,
                                        triggerPrice: str = None,
                                        triggerBy: str = None,
                                        time_in_force: str = GTC,
                                        losscut_price: str = None,
                                        triggerDirection=None,
                                        reduceOnly=None) -> dict:

                self.validate_order_params(side, orderType, price, losscut_price)

                action = lambda s, **kw: s.place_order(category=CATEGORY, symbol=self._symbol, **kw)

                if triggerPrice is not None:
                        return self._retry_api_request(action,
                                                                           side=side,
                                                                           qty=qty,
                                                                           marketUnit='quoteCoin',
                                                                           orderType=orderType,
                                                                           price=price,
                                                                           timeInForce=time_in_force,
                                                                           stopLoss=losscut_price,
                                                                           triggerPrice=triggerPrice,
                                                                           triggerBy=triggerBy,
                                                                           triggerDirection=triggerDirection,
                                                                           reduceOnly=reduceOnly)
                else:
                        return self._retry_api_request(action,
                                                                           side=side,
                                                                           qty=qty,
                                                                           marketUnit='quoteCoin',
                                                                           orderType=orderType,
                                                                           timeInForce=time_in_force,
                                                                           reduceOnly=reduceOnly)

        def validate_order_params(self, side, orderType, price, losscut_price):

                if orderType not in [self.MARKET, self.LIMIT]:
                        raise ValueError("Invalid order type specified")
                if side not in [self.BUY, self.SELL]:
                        raise ValueError("Invalid side specified")
                if orderType == self.LIMIT and price is None:
                        raise ValueError("Limit order requires an entry price")
                if losscut_price is None:
                        losscut_price = 0.0

        def api_action(self, method_name, **kwargs):
                action = lambda s, **kw: getattr(s, method_name)(category=CATEGORY, symbol=self._symbol, **kw)
                return self._retry_api_request(action, **kwargs)

        def cancel_order(self, order_id: str) -> dict:
                return self.api_action('cancel_order', orderId=order_id)

        def get_open_orders(self, orderId: str):
                return self.api_action('get_open_orders', orderId=orderId)

        def trade_entry_trigger(self,
                                                        qty,
                                                        trade_type,
                                                        entry_price,
                                                        triggerPric,
                                                        losscut_price=None):

                side, triggerDirection = (self.BUY, 2) if trade_type == ENTRY_TYPE_LONG else (self.SELL, 1)

                success, data = self.place_order(side,
                                                                                 qty,
                                                                                 orderType=self.MARKET,
                                                                                 price=entry_price,
                                                                                 triggerPrice=triggerPric,
                                                                                 triggerBy="LastPrice",
                                                                                 triggerDirection=triggerDirection,
                                                                                 losscut_price=losscut_price)
                if success:
                        return data['result']['orderId']
                else:
                        raise ValueError("Trade entry failed")

        def trade_exit(self, qty, trade_type):
                side = self.SELL if trade_type == ENTRY_TYPE_LONG else self.BUY

                success, data = self.place_order(side, qty, orderType=self.MARKET, reduceOnly=True)

                if success:
                        return True,data['result']['orderId']
                else:
                        return False, None

        def get_order_status(self, orderId: str):
                success, data = self.get_open_orders(orderId)
                if success:
                        order_list = data.get('result', {}).get('list', [])
                        if order_list:
                                order_status = order_list[0].get('orderStatus')
                                if order_status:
                                        return order_status
                                else:
                                        raise ValueError("Order status not found in the response")
                        else:
                                return None
                else:
                        raise ValueError("Failed to retrieve open orders")
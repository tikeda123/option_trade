from datetime import datetime
import sys
import os
import pandas as pd

from typing import Optional, Tuple
from pybit.unified_trading import HTTP
from requests.exceptions import HTTPError

# Get the absolute path of the directory where the current file is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory path
parent_dir = os.path.dirname(current_dir)
# Add the parent directory path to sys.path
sys.path.append(parent_dir)

from common.constants import *
from bybit_api.bybit_base_api import BybitBaseAPI, CATEGORY
from trade_config import trade_config

class BybitOrderManager(BybitBaseAPI):
        """
        This class manages order placement, cancellation, and status retrieval on Bybit.
        """
        MARKET = 'Market'
        LIMIT = 'Limit'
        BUY = 'Buy'
        SELL = 'Sell'
        GTC = 'GTC'

        def place_order(
                self,
                side: str,
                qty: str,
                orderType: str = MARKET,
                price: Optional[str] = None,
                triggerPrice: Optional[str] = None,
                triggerBy: Optional[str] = None,
                time_in_force: str = GTC,
                losscut_price: Optional[str] = None,
                triggerDirection: Optional[int] = None,
                reduceOnly: Optional[bool] = None
        ) -> str:
                """
                Places an order on Bybit.

                Args:
                        side (str): The side of the order (BUY or SELL).
                        qty (str): The quantity of the order.
                        orderType (str, optional): The type of order (MARKET or LIMIT). Defaults to MARKET.
                        price (str, optional): The price of the order (required for LIMIT orders). Defaults to None.
                        triggerPrice (str, optional): The trigger price for stop orders. Defaults to None.
                        triggerBy (str, optional): The trigger condition for stop orders. Defaults to None.
                        time_in_force (str, optional): The time in force for the order. Defaults to GTC.
                        losscut_price (str, optional): The stop loss price for the order. Defaults to None.
                        triggerDirection (int, optional): The trigger direction for stop orders. Defaults to None.
                        reduceOnly (bool, optional): Whether the order is reduce-only. Defaults to None.

                Returns:
                        str: The order ID of the placed order.

                Raises:
                        ValueError: If invalid order parameters are provided.
                        RuntimeError: If placing the order fails.
                """
                self.validate_order_params(side, orderType, price, losscut_price)

                action = lambda s, **kw: s.place_order(category=CATEGORY, symbol=trade_config.symbol, **kw)

                try:
                        if triggerPrice is not None:
                                response = self._retry_api_request(action,
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
                                response = self._retry_api_request(action,
                                                                                        side=side,
                                                                                        qty=qty,
                                                                                        marketUnit='quoteCoin',
                                                                                        orderType=orderType,
                                                                                        timeInForce=time_in_force,
                                                                                        reduceOnly=reduceOnly)

                        return response['result']['orderId']

                except HTTPError as e:
                        raise RuntimeError(f"Failed to place order: {e}") from e
                except KeyError as e:
                        raise RuntimeError(f"Unexpected API response format: {e}") from e

        def validate_order_params(self, side: str, orderType: str, price: Optional[str], losscut_price: Optional[str]) -> None:
                """Validates order parameters before placing an order.

                Args:
                        side (str): The side of the order (BUY or SELL).
                        orderType (str): The type of order (MARKET or LIMIT).
                        price (str, optional): The price of the order.
                        losscut_price (str, optional): The stop loss price for the order.

                Raises:
                        ValueError: If invalid order parameters are provided.
                """
                if orderType not in [self.MARKET, self.LIMIT]:
                        raise ValueError("Invalid order type specified")
                if side not in [self.BUY, self.SELL]:
                        raise ValueError("Invalid side specified")
                if orderType == self.LIMIT and price is None:
                        raise ValueError("Limit order requires an entry price")
                # Losscut price can be None, so no validation needed here.

        def api_action(self, method_name: str, **kwargs) -> dict:
                """
                Performs a generic API action with retries.

                This method takes the name of the Bybit API method to call and any keyword arguments
                required for that method. It then constructs a lambda function that calls the specified
                method on the Bybit API session object, passing in the provided keyword arguments.
                Finally, it calls the _retry_api_request method to execute the API action with retries.

                Args:
                        method_name (str): The name of the Bybit API method to call.
                        **kwargs: Keyword arguments to pass to the Bybit API method.

                Returns:
                        dict: The response from the Bybit API.
                """
                action = lambda s, **kw: getattr(s, method_name)(category=CATEGORY, symbol=trade_config.symbol, **kw)
                return self._retry_api_request(action, **kwargs)

        def cancel_order(self, order_id: str) -> None:
                """Cancels a previously placed order.

                Args:
                        order_id (str): The ID of the order to cancel.

                Raises:
                        RuntimeError: If canceling the order fails.
                """
                try:
                        self.api_action('cancel_order', orderId=order_id)
                except Exception as e:
                        raise RuntimeError(f"Failed to cancel order {order_id}: {e}") from e

        def get_open_orders(self, order_id: str) -> Optional[dict]:
                """Retrieves information about a specific open order.

                Args:
                        order_id (str): The ID of the order to retrieve.

                Returns:
                        Optional[dict]: A dictionary containing the order information if found, otherwise None.
                                                        The dictionary structure depends on the Bybit API response.
                """
                try:
                        response = self.api_action('get_open_orders', orderId=order_id)
                        return response.get('result', {}).get('list', [{}])[0]
                except Exception as e:
                        raise RuntimeError(f"Failed to get open order {order_id}: {e}") from e

        def trade_entry_trigger(
                self,
                qty: str,
                trade_type: str,
                entry_price: Optional[str] = None,
                triggerPrice: Optional[str] = None,
                losscut_price: Optional[str] = None
        ) -> str:
                """
                Places a trade entry trigger order.

                This function determines the appropriate order side and trigger direction based on the
                specified trade type. It then calls the `place_order` method to place a market order
                with the calculated parameters.

                Args:
                        qty (str): The quantity of the order.
                        trade_type (str): The type of trade (ENTRY_TYPE_LONG or ENTRY_TYPE_SHORT).
                        entry_price (str, optional): The entry price for the order. Defaults to None.
                        triggerPrice (str, optional): The trigger price for the order. Defaults to None.
                        losscut_price (str, optional): The stop loss price for the order. Defaults to None.

                Returns:
                        str: The order ID of the placed order.

                Raises:
                        ValueError: If an invalid trade type is specified.
                        RuntimeError: If placing the order fails.
                """
                if trade_type not in [ENTRY_TYPE_LONG, ENTRY_TYPE_SHORT]:
                        raise ValueError("Invalid trade type specified")

                side, triggerDirection = (self.BUY, 2) if trade_type == ENTRY_TYPE_LONG else (self.SELL, 1)

                try:
                        order_id = self.place_order(
                                side=side,
                                qty=qty,
                                orderType=self.MARKET,
                                price=entry_price,
                                triggerPrice=triggerPrice,
                                triggerBy="LastPrice",
                                triggerDirection=triggerDirection,
                                losscut_price=losscut_price
                        )
                        return order_id
                except Exception as e:
                        raise RuntimeError(f"Trade entry failed: {e}") from e

        def trade_exit(self, qty: str, trade_type: str) -> str:
                """Places a trade exit order.

                Args:
                        qty (str): The quantity of the order.
                        trade_type (str): The type of trade being exited (ENTRY_TYPE_LONG or ENTRY_TYPE_SHORT).

                Returns:
                        str: The order ID of the placed order.

                Raises:
                        RuntimeError: If placing the exit order fails.
                """
                side = self.SELL if trade_type == ENTRY_TYPE_LONG else self.BUY
                try:
                        order_id = self.place_order(side=side, qty=qty, orderType=self.MARKET, reduceOnly=True)
                        return order_id
                except Exception as e:
                        raise RuntimeError(f"Trade exit failed: {e}") from e

        def get_order_status(self, order_id: str) -> Optional[str]:
                """
                Retrieves the status of a specific order.

                Args:
                        order_id (str): The ID of the order.

                Returns:
                        Optional[str]: The status of the order if found, otherwise None.

                Raises:
                        RuntimeError: If retrieving the order status fails.
                """
                try:
                        order_info = self.get_open_orders(order_id)
                        if order_info:
                                return order_info.get('orderStatus')
                        else:
                                return None
                except Exception as e:
                        raise RuntimeError(f"Failed to get order status for order ID {order_id}: {e}") from e
        def set_trading_stop(
                self,
                take_profit: Optional[str] = None,
                stop_loss: Optional[str] = None,
                trailing_stop: Optional[str] = None,
                tp_trigger_by: Optional[str] = None,
                sl_trigger_by: Optional[str] = None,
                active_price: Optional[str] = None,
                tpsl_mode: str = "Full",
                tp_size: Optional[str] = None,
                sl_size: Optional[str] = None,
                tp_limit_price: Optional[str] = None,
                sl_limit_price: Optional[str] = None,
                tp_order_type: Optional[str] = None,
                sl_order_type: Optional[str] = None,
                position_idx: int = 0,
        ) -> dict:
                """Sets take profit, stop loss, or trailing stop for a position.

                Args:
                        take_profit (str, optional): Take profit price. Defaults to None.
                        stop_loss (str, optional): Stop loss price. Defaults to None.
                        trailing_stop (str, optional): Trailing stop distance. Defaults to None.
                        tp_trigger_by (str, optional): TP trigger price type. Defaults to None.
                        sl_trigger_by (str, optional): SL trigger price type. Defaults to None.
                        active_price (str, optional): Trailing stop trigger price. Defaults to None.
                        tpsl_mode (str, optional): TP/SL mode ('Full' or 'Partial'). Defaults to "Full".
                        tp_size (str, optional): TP size (for 'Partial' mode). Defaults to None.
                        sl_size (str, optional): SL size (for 'Partial' mode). Defaults to None.
                        tp_limit_price (str, optional): TP limit price (for 'Partial' mode). Defaults to None.
                        sl_limit_price (str, optional): SL limit price (for 'Partial' mode). Defaults to None.
                        tp_order_type (str, optional): TP order type ('Market' or 'Limit'). Defaults to None.
                        sl_order_type (str, optional): SL order type ('Market' or 'Limit'). Defaults to None.
                        position_idx (int, optional): Position index (0 for one-way, 1 for hedge buy, 2 for hedge sell). Defaults to 0.

                Returns:
                        dict: The API response.

                Raises:
                        RuntimeError: If setting the trading stop fails.
                """
                params = {
                        "takeProfit": take_profit,
                        "stopLoss": stop_loss,
                        "trailingStop": trailing_stop,
                        "tpTriggerBy": tp_trigger_by,
                        "slTriggerBy": sl_trigger_by,
                        "activePrice": active_price,
                        "tpslMode": tpsl_mode,
                        "tpSize": tp_size,
                        "slSize": sl_size,
                        "tpLimitPrice": tp_limit_price,
                        "slLimitPrice": sl_limit_price,
                        "tpOrderType": tp_order_type,
                        "slOrderType": sl_order_type,
                        "positionIdx": position_idx,
                }

                # Noneの値を持つキーを削除
                params = {k: v for k, v in params.items() if v is not None}


                try:
                        return self.api_action("set_trading_stop", **params)
                except Exception as e:
                        raise RuntimeError(f"Failed to set trading stop: {e}") from e



def main():
        om = BybitOrderManager()

        # Place a limit order
        order_id = om.trade_entry_trigger(
                qty='0.1', trade_type=ENTRY_TYPE_LONG, entry_price='58000', triggerPrice='57000')

        print(f"Placed order ID: {order_id}")

        status = om.get_order_status(order_id)
        print(f"Order status: {status}")

if __name__ == '__main__':
        main()
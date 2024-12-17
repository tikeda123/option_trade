import sys
import os
from typing import Optional, Tuple

# Get the absolute path of the directory where the current file is located.
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory path.
parent_dir = os.path.dirname(current_dir)
# Add the parent directory path to sys.path.
sys.path.append(parent_dir)

from common.constants import *
from bybit_api.bybit_base_api import BybitBaseAPI
from bybit_api.bybit_data_fetcher import BybitDataFetcher
from bybit_api.bybit_order_manager import BybitOrderManager
from bybit_api.bybit_position_manager import BybitPositionManager
from bybit_api.bybit_pnl_manager import BybitPnlManager

class BybitTrader(BybitBaseAPI):
        """
        This class provides a high-level interface for trading on Bybit,
        encapsulating functionality for data fetching, order management,
        position management, and profit/loss (PnL) tracking.
        """
        def __init__(self):
                """Initializes the BybitTrader with all necessary managers and sets initial leverage."""
                super().__init__()
                self.data_fetcher = BybitDataFetcher()
                self.order_manager = BybitOrderManager()
                self.position_manager = BybitPositionManager()
                self.pnl_manager = BybitPnlManager()
                self.order_id = None
                self.init_managers()

        def init_managers(self) -> None:
                """Initializes the position manager with the specified leverage."""
                try:
                        self.position_manager.set_my_leverage(self._leverage)
                except RuntimeError as e:
                        self._logger.log_system_message(f"Failed to initialize position manager: {e}")

        def get_current_price(self) -> float:
                """
                Fetches the latest price of the trading symbol.

                Returns:
                        float: The latest price of the symbol.
                """
                return self.data_fetcher.fetch_latest_info()

        def trade_entry_trigger(
                self,
                qty: float,
                trade_type: str,
                target_price: Optional[float] = None,
                trigger_price: Optional[float] = None,
                stop_loss_price: Optional[float] = None
        ) -> str:
                """
                Places a trade entry trigger order on Bybit.

                This method automatically adjusts the trigger price if it exceeds the current price.

                Args:
                        qty (float): The quantity for the trade.
                        trade_type (str): The type of trade: 'ENTRY_TYPE_LONG' for long entry,
                                                         'ENTRY_TYPE_SHORT' for short entry.
                        target_price (float, optional): The target price for taking profit. Defaults to None.
                        trigger_price (float, optional): The trigger price for the entry order. Defaults to None.
                        stop_loss_price (float, optional): The stop loss price for the trade. Defaults to None.

                Returns:
                        str: The order ID of the placed trigger order.

                Raises:
                        ValueError: If an invalid trade type is provided.
                        RuntimeError: If placing the order fails.
                """
                if trade_type not in [ENTRY_TYPE_LONG, ENTRY_TYPE_SHORT]:
                        raise ValueError("Invalid trade type specified.")

                current_price = self.get_current_price()
                trigger_price = self.adjust_trigger_price(trade_type, trigger_price, current_price)
                qty = self.qty_round(qty)

                try:
                        self.order_id = self.order_manager.trade_entry_trigger(
                                qty, trade_type, target_price, trigger_price, stop_loss_price
                        )
                        return self.order_id
                except RuntimeError as e:
                        raise RuntimeError(f"Failed to place trade entry trigger order: {e}") from e

        def adjust_trigger_price(
                self, trade_type: str, trigger_price: Optional[float], current_price: float
        ) -> float:
                """
                Adjusts the trigger price based on the current price and trade type.

                This ensures that the trigger price is set appropriately for the desired entry direction.

                Args:
                        trade_type (str): The type of trade ('ENTRY_TYPE_LONG' or 'ENTRY_TYPE_SHORT').
                        trigger_price (float, optional): The initial trigger price.
                        current_price (float): The current market price.

                Returns:
                        float: The adjusted trigger price.
                """
                if trade_type == ENTRY_TYPE_LONG and trigger_price is not None and trigger_price >= current_price:
                        return current_price * 0.9999
                elif trade_type == ENTRY_TYPE_SHORT and trigger_price is not None and trigger_price <= current_price:
                        return current_price * 1.0001
                return trigger_price if trigger_price is not None else current_price

        def get_order_status(self, order_id: str) -> Optional[str]:
                """
                Retrieves the status of a specific order.

                Args:
                        order_id (str): The ID of the order.

                Returns:
                        Optional[str]: The status of the order, or None if the order is not found.
                """
                return self.order_manager.get_order_status(order_id)

        def get_closed_pnl(self) -> Tuple[float, float]:
                """
                Retrieves the closed profit and loss (PnL) and average exit price for the most recent closed position.

                Returns:
                        Tuple[float, float]: A tuple containing the closed PnL and the average exit price.
                """
                return self.pnl_manager.get_pnl()

        def trade_exit(self, qty: float, trade_type: str) -> str:
                """
                Places a trade exit order on Bybit.

                Args:
                        qty (float): The quantity to exit.
                        trade_type (str): The type of trade being exited ('ENTRY_TYPE_LONG' or 'ENTRY_TYPE_SHORT').

                Returns:
                        str: The order ID of the placed exit order.

                Raises:
                        RuntimeError: If placing the exit order fails.
                """
                qty = self.qty_round(qty)
                try:
                        return self.order_manager.trade_exit(qty, trade_type)
                except RuntimeError as e:
                        raise RuntimeError(f"Failed to place trade exit order: {e}") from e

        def cancel_order(self, order_id: str) -> None:
                """
                Cancels a previously placed order.

                Args:
                        order_id (str): The ID of the order to cancel.
                """
                return self.order_manager.cancel_order(order_id)

        def get_open_position_status(self) -> str:
                """
                Gets the status of the current open position.

                Returns:
                        str: 'position' if there is an open position, 'No position' otherwise.
                """
                return self.position_manager.get_open_position_status()

import time

def main():
        bybit_trader = BybitTrader()
        current_price = bybit_trader.get_current_price()
        print(f"Current price: {current_price}")

        # Place a long entry trigger order
        target_price  = current_price * 0.999
        trigger_price = current_price * 0.999
        stop_loss_price = current_price * 0.8

        order_id = bybit_trader.trade_entry_trigger(
                qty=0.001, trade_type=ENTRY_TYPE_LONG, target_price=target_price, trigger_price=trigger_price, stop_loss_price=stop_loss_price)
        print(f"Long entry trigger order placed with ID: {order_id}")

        # Check the status of the order
        for _ in range(30):
                order_status = bybit_trader.get_order_status(order_id)
                if order_status ==  'Filled':
                        print("Order filled.")
                        break
                print(f"Order status: {order_status}")
                time.sleep(10)

        # Check the status of the open position
        open_position_status = bybit_trader.get_open_position_status()
        print(f"Open position status: {open_position_status}")

        for _ in range(30):
                        open_position_status = bybit_trader.get_open_position_status()
                        print(f"Open position status: {open_position_status}")
                        time.sleep(3)

        # Place a short exit order
        order_id = bybit_trader.trade_exit(qty=0.001, trade_type=ENTRY_TYPE_LONG)
        print(f"Short exit order placed with ID: {order_id}")

        # Check the status of the order
        for _ in range(30):
                order_status = bybit_trader.get_order_status(order_id)
                if order_status == 'Filled':
                        print("Order filled.")
                        break
                print(f"Order status: {order_status}")
                time.sleep(10)

        # Get the closed PnL
        closed_pnl, avg_exit_price = bybit_trader.get_closed_pnl()
        print(f"Closed PnL: {closed_pnl}, Average exit price: {avg_exit_price}")

if __name__ == "__main__":
        main()


import pandas as pd
import os, sys

from common.constants import *

class TradingStateData:
        """
        This class holds data related to the current state of the trading system.

        It stores information such as entry and exit indices, FX transaction serial number,
        order ID, Bollinger Band direction, entry price, prediction, and entry type.
        """

        def __init__(self) -> None:
                """
                Initializes the TradingStateData with default values.
                """
                self._current_index = 0  # The current index of the data being processed
                self._entry_index = 0     # The index where a trade was entered
                self._exit_index = 0      # The index where a trade was exited
                self._fx_serial = 0       # The serial number of the current FX transaction
                self._bb_direction = None  # The current direction of the Bollinger Band (UPPER or LOWER)
                self._entry_price = 0.0   # The price at which the trade was entered
                self._prediction = 0      # The prediction made for the trade (e.g., LONG or SHORT)
                self._entry_type = None    # The type of entry (e.g., LONG or SHORT)
                self._entry_counter = 0    # The number of attempted entries
                self._order_id = None      # The order ID of the current trade
                self._entry_price_garch = 0.0
                self._exit_price_garch = 0.0
                self._support_line = 0.0
                self._resistance_line = 0.0


        def reset_index(self):
                """
                Resets all indices and identifiers related to a trade.

                This method is called when a trade is finished or needs to be reset,
                clearing all data related to the previous trade.
                """
                self._current_index = 0
                self._entry_index = 0
                self._entry_type = None
                self._exit_index = 0
                self._fx_serial = 0
                self._bb_direction = None
                self._entry_price = 0.0
                self._prediction = 0
                self._entry_counter = 0
                self._order_id = None
                self._entry_price_garch = 0.0
                self._exit_price_garch = 0.0
                self._support_line = 0.0
                self._resistance_line = 0.0

        def get_support_line(self) -> float:
                """
                Gets the support line for the current trade.
                """
                return self._support_line

        def set_support_line(self, support_line: float):
                """
                Sets the support line for the current trade.
                """
                self._support_line = support_line

        def get_resistance_line(self) -> float:
                """
                Gets the resistance line for the current trade.
                """
                return self._resistance_line

        def set_resistance_line(self, resistance_line: float):
                """
                Sets the resistance line for the current trade.
                """
                self._resistance_line = resistance_line 

        def get_entry_price_garch(self) -> float:
                """
                Gets the entry price for the current trade.

                Returns:
                        float: The price at which the trade was entered.
                """
                return self._entry_price_garch

        def set_entry_price_garch(self, price: float):
                """
                Sets the entry price for the current trade.

                Args:
                        price (float): The entry price to set.
                """
                self._entry_price_garch = price

        def get_exit_price_garch(self) -> float:
                """
                Gets the exit price for the current trade.

                Returns:
                        float: The price at which the trade was exited.
                """
                return self._exit_price_garch

        def set_exit_price_garch(self, price: float):
                """
                Sets the exit price for the current trade.

                Args:
                        price (float): The exit price to set.
                """
                self._exit_price_garch = price

        def get_order_id(self) -> str:
                """
                Gets the order ID of the current trade.

                Returns:
                        str: The order ID of the current trade.
                """
                return self._order_id

        def set_order_id(self, order_id: str):
                """
                Sets the order ID of the current trade.

                Args:
                        order_id (str): The order ID to set.
                """
                self._order_id = order_id

        def get_entry_type(self) -> str:
                """
                Gets the type of entry for the current trade.

                Returns:
                        str: The type of entry (e.g., LONG or SHORT).
                """
                return self._entry_type

        def set_entry_type(self, entry_type: str):
                """
                Sets the type of entry for the current trade.

                Args:
                        entry_type (str): The type of entry to set (e.g., LONG or SHORT).
                """
                self._entry_type = entry_type

        def get_entry_price(self) -> float:
                """
                Gets the entry price for the current trade.

                Returns:
                        float: The price at which the trade was entered.
                """
                return self._entry_price

        def set_entry_price(self, price: float):
                """
                Sets the entry price for the current trade.

                Args:
                        price (float): The entry price to set.
                """
                self._entry_price = price

        def get_prediction(self) -> int:
                """
                Gets the prediction made for the current trade.

                Returns:
                        int: The prediction, typically an integer representing a trade direction (e.g., LONG or SHORT).
                """
                return self._prediction

        def set_prediction(self, prediction: int):
                """
                Sets the prediction made for the current trade.

                Args:
                        prediction (int): The prediction to set.
                """
                self._prediction = prediction

        def get_current_index(self) -> int:
                """
                Gets the current index of the data being processed.

                Returns:
                        int: The current data index.
                """
                return self._current_index

        def set_current_index(self, index: int):
                """
                Sets the current index of the data being processed.

                Args:
                        index (int): The new data index to set.
                """
                self._current_index = index

        def get_bb_direction(self) -> str:
                """
                Gets the direction of the Bollinger Band.

                Returns:
                        str: The direction of the Bollinger Band (e.g., UPPER or LOWER).
                """
                return self._bb_direction

        def set_bb_direction(self, direction: str):
                """
                Sets the direction of the Bollinger Band.

                Args:
                        direction (str): The direction of the Bollinger Band (e.g., UPPER or LOWER).
                """
                self._bb_direction = direction

        def get_entry_counter(self) -> int:
                """
                Gets the entry counter, which tracks the number of attempted entries.

                Returns:
                        int: The current entry counter value.
                """
                return self._entry_counter

        def set_entry_counter(self, counter: int):
                """
                Sets the entry counter, which tracks the number of attempted entries.

                Args:
                        counter (int): The new value for the entry counter.
                """
                self._entry_counter = counter

        def increment_entry_counter(self):
                """
                Increments the entry counter by one.
                """
                self._entry_counter += 1

        def get_entry_index(self) -> int:
                """
                Gets the index where the current trade was entered.

                Returns:
                        int: The entry index of the trade.
                """
                return self._entry_index

        def set_entry_index(self, index: int):
                """
                Sets the index where the current trade was entered.

                Args:
                        index (int): The index to set as the entry index.
                """
                self._entry_index = index

        def get_exit_index(self) -> int:
                """
                Gets the index where the current trade was exited.

                Returns:
                        int: The exit index of the trade.
                """
                return self._exit_index

        def set_exit_index(self, index: int):
                """
                Sets the index where the current trade was exited.

                Args:
                        index (int): The index to set as the exit index.
                """
                self._exit_index = index

        def get_fx_serial(self) -> int:
                """
                Gets the serial number of the current FX transaction.

                Returns:
                        int: The serial number of the FX transaction.
                """
                return self._fx_serial

        def set_fx_serial(self, serial: int):
                """
                Sets the serial number of the current FX transaction.

                Args:
                        serial (int): The serial number to set for the FX transaction.
                """
                self._fx_serial = serial









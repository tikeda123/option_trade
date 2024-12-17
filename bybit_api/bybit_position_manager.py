import sys
import os
from typing import Optional, Tuple
from pybit.unified_trading import HTTP
from requests.exceptions import HTTPError

# Get the absolute path of the directory where the current file is located.
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory path.
parent_dir = os.path.dirname(current_dir)
# Add the parent directory path to sys.path.
sys.path.append(parent_dir)

from common.constants import *
from bybit_api.bybit_base_api import BybitBaseAPI, CATEGORY
from trade_config import trade_config

class BybitPositionManager(BybitBaseAPI):
        """
        This class manages position-related operations on Bybit, such as setting leverage,
        retrieving position information, and switching position modes.
        """
        def __init__(self):
                """Initializes the BybitPositionManager by calling the constructor of its parent class."""
                super().__init__()
                self._current_leverage = None
        def set_switch_mode(self, mode: str) -> None:
                """
                Switches the position mode for the current symbol.

                Args:
                        mode (str): The desired position mode. Refer to the Bybit API documentation for valid modes.

                Raises:
                        RuntimeError: If switching the position mode fails.
                """
                action = lambda s, **kw: s.switch_position_mode(category=CATEGORY, symbol=trade_config.symbol, **kw)
                try:
                        self._retry_api_request(action, mode=mode)
                except Exception as e:
                        raise RuntimeError(f"Failed to switch position mode: {e}") from e

        def get_position_info(self) -> dict:
                """
                Retrieves detailed information about the current position for the specified symbol.

                Returns:
                        dict: A dictionary containing position information. The dictionary structure depends on the Bybit API response.

                Raises:
                        RuntimeError: If retrieving position information fails.
                """
                try:
                        session = HTTP(testnet=self._isTESTNET, api_key=self._api_key, api_secret=self._api_secret)
                        result = session.get_positions(category=CATEGORY, symbol=trade_config.symbol)
                        return result
                except HTTPError as e:
                        raise RuntimeError(f"Failed to get position info: {e}") from e
                except Exception as e:
                        raise RuntimeError(f"An error occurred while getting position info: {e}") from e

        def set_leverage(self, buy_leverage: str, sell_leverage: str) -> None:
                """
                Sets the leverage for both long (buy) and short (sell) positions for the specified symbol.

                Args:
                        buy_leverage (str): The desired leverage for long positions.
                        sell_leverage (str): The desired leverage for short positions.

                Raises:
                        RuntimeError: If setting leverage fails.
                """
                action = lambda s, **kw: s.set_leverage(category=CATEGORY, symbol=trade_config.symbol, **kw)
                try:
                        self._retry_api_request(action, buyLeverage=buy_leverage, sellLeverage=sell_leverage)
                        self._current_leverage = self.get_my_leverage()
                except Exception as e:
                        raise RuntimeError(f"Failed to set leverage: {e}") from e

        def get_positions(self) -> dict:
                """
                Retrieves a list of all open positions for the account.

                Returns:
                        dict: A dictionary containing a list of open positions. The dictionary structure
                                  depends on the Bybit API response.

                Raises:
                        RuntimeError: If retrieving positions fails.
                """
                action = lambda s, **kw: s.get_positions(category=CATEGORY, symbol=trade_config.symbol, **kw)
                try:
                        return self._retry_api_request(action)
                except Exception as e:
                        raise RuntimeError(f"Failed to get positions: {e}") from e

        def get_my_leverage(self) -> float:
                """
                Retrieves the current leverage for the specified symbol.

                Returns:
                        float: The current leverage.

                Raises:
                        RuntimeError: If retrieving leverage fails or the leverage is not found in the response.
                """
                try:
                        data = self.get_positions()
                        leverage = float(data['result']['list'][0]['leverage'])
                        return leverage
                except (KeyError, IndexError) as e:
                        raise RuntimeError("Leverage is not found in the API response.") from e
                except Exception as e:
                        raise RuntimeError(f"Failed to retrieve leverage: {e}") from e

        def set_my_leverage(self, leverage: float) -> None:
                """
                Sets the leverage for the current symbol.

                Args:
                        leverage (float): The desired leverage.

                Raises:
                        RuntimeError: If retrieving the current leverage or setting the new leverage fails.
                """
                try:
                        current_leverage = self. get_my_leverage()
                        if current_leverage  != leverage:
                                self.set_leverage(str(leverage), str(leverage))
                                self._logger.log_system_message(f"Leverage updated to {leverage}")

                except Exception as e:
                        raise RuntimeError(f"Failed to set leverage: {e}") from e

        def get_open_position_status(self) -> str:
                """
                Checks if there is an open position for the current symbol and returns its status.

                Returns:
                        str: 'position' if there is an open position, 'No position' otherwise.

                Raises:
                        RuntimeError: If retrieving position data fails or the position is not found in the response.
                """
                try:
                        data = self.get_positions()
                        size = float(data['result']['list'][0]['size'])
                        return 'position' if size > 0 else 'No position'
                except (KeyError, IndexError) as e:
                        raise RuntimeError("Position is not found in the API response.") from e
                except Exception as e:
                        raise RuntimeError(f"Failed to retrieve position status: {e}") from e

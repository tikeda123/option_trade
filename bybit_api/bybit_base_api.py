from datetime import datetime
import sys
import os
import time

import pandas as pd
from typing import Optional, Tuple, Callable
from pybit.unified_trading import HTTP
from requests.exceptions import HTTPError

# Get the absolute path of the directory where the current file is located.
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory path.
parent_dir = os.path.dirname(current_dir)
# Add the parent directory path to sys.path.
sys.path.append(parent_dir)

from trade_config import trade_config
from common.trading_logger import TradingLogger
from common.config_manager import ConfigManager
from common.constants import *
from trade_config import trade_config

MAX_LIMIT_QUERY = 200
CATEGORY = "linear"
LINKED_ORDER = "ai-algo"

class BybitBaseAPI:
        """
        This class serves as a base class for all Bybit API interactions.
        It handles initialization, configuration loading, session creation,
        and provides common methods for API requests with error handling and retries.
        """
        def __init__(self):
                """Initializes the BybitBaseAPI with configuration settings and logging."""
                self._logger = TradingLogger()
                self._config_manager = ConfigManager()
                self._datapath = parent_dir + '/' + self._config_manager.get('DATA', 'TSTPATH')
                api_config = self._config_manager.get('BYBIT_API')
                self._api_key = api_config['API_KEY']
                self._api_secret = api_config['API_SECRET']
                self._isTESTNET = api_config['TESTNET']
                self._price_round = 2
                self._current_leverage = INITIAL_LEVERAGE
                self.set_round()

        def set_round(self) -> None:
                """Sets the rounding precision for quantities based on the trading symbol."""
                symbol = trade_config.symbol
                if symbol == 'BTCUSDT':
                        self._ROUND_DIGIT = 3
                elif symbol == 'ETHUSDT':
                        self._ROUND_DIGIT = 2
                else:
                        self._ROUND_DIGIT = 2

        def qty_round(self, qty: float) -> float:
                """Rounds a quantity value to the specified number of decimal places.

                Args:
                        qty (float): The quantity value to round.

                Returns:
                        float: The rounded quantity value.
                """
                return round(float(qty), self._ROUND_DIGIT)

        def price_round(self, price: float) -> float:
                """Rounds a price value to the specified number of decimal places.

                Args:
                        price (float): The price value to round.

                Returns:
                        float: The rounded price value.
                """
                return round(float(price), self._price_round)

        def _create_session(self) -> HTTP:
                """Creates and returns a Bybit HTTP session object."""
                return HTTP(testnet=self._isTESTNET, api_key=self._api_key, api_secret=self._api_secret)

        def _api_request(self, action: Callable, **kwargs) -> dict:
                """Makes an API request to Bybit with error handling.

                This method takes a callable 'action' representing the Bybit API method to call
                and any keyword arguments required for that method. It creates a new Bybit HTTP session,
                executes the API action, and handles potential HTTP errors and unexpected response formats.

                Args:
                        action (Callable): A callable representing the Bybit API method to call (e.g., session.get_positions).
                        **kwargs: Keyword arguments to pass to the Bybit API method.

                Returns:
                        dict: The response from the Bybit API if successful.

                Raises:
                        RuntimeError: If the API request fails or the response format is unexpected.
                """
                session = self._create_session()
                try:
                        result = action(session, **kwargs)
                        return result
                except HTTPError as e:
                        raise RuntimeError(f"ByBit API request failed: {e}") from e
                except KeyError as e:
                        raise RuntimeError(f"Unexpected API response format: {e}") from e

        def _retry_api_request(self, action: Callable, **kwargs) -> dict:
                """
                Executes an API request to Bybit with retries and exponential backoff.

                This method attempts to execute the given 'action' multiple times,
                retrying on failures with an increasing delay between attempts.
                This helps to handle temporary network issues or API rate limits.

                Args:
                        action (Callable): A callable representing the Bybit API method to call.
                        **kwargs: Keyword arguments to pass to the Bybit API method.

                Returns:
                        dict: The response from the Bybit API if successful.

                Raises:
                        RuntimeError: If the maximum number of retries is reached.
                """
                for _ in range(MAX_TRYOUT_HTTP_REQUEST):
                        try:
                                result = self._api_request(action, **kwargs)
                                return result
                        except RuntimeError as e:
                                self._logger.log_system_message(f"Bybit API request failed. Retrying... Error: {e}")
                                time.sleep(MAX_TRYOUT_HTTP_REQUEST_SLEEP)

                self._logger.log_system_message(f"{action.__name__}: MAX TRY OUT Error.")
                raise RuntimeError(f"Bybit API request failed after {MAX_TRYOUT_HTTP_REQUEST} retries.")


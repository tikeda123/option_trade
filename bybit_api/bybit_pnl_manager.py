import sys
import os
import time
from typing import Tuple
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

class BybitPnlManager(BybitBaseAPI):
        """
        This class manages Profit and Loss (PnL) retrieval from Bybit.
        It provides methods to get realized PnL for closed positions.
        """
        def __init__(self):
                """Initializes the BybitPnlManager by calling the constructor of its parent class."""
                super().__init__()

        def get_closed_pnl(self) -> dict:
                """
                Retrieves the closed PnL for the specified symbol.

                This method fetches the closed Profit and Loss (PnL) data for the most recent closed position.
                It makes an API request to Bybit to get the closed PnL information and returns the raw API response.

                Returns:
                        dict: The raw API response containing the closed PnL data.

                Raises:
                        RuntimeError: If the API request fails or the response format is unexpected.
                """
                action = lambda s, **kw: s.get_closed_pnl(category=CATEGORY, symbol=trade_config.symbol, **kw)
                try:
                        return self._retry_api_request(action, limit=1)
                except HTTPError as e:
                        raise RuntimeError(f"Failed to fetch closed PnL: {e}") from e
                except KeyError as e:
                        raise RuntimeError(f"Unexpected API response format: {e}") from e

        def get_pnl(self) -> Tuple[float, float]:
                """
                Retrieves the realized PnL and average exit price for the most recent closed position.

                This method calls 'get_closed_pnl' to fetch the closed PnL data, extracts the relevant
                information from the response, and returns the realized PnL and average exit price as floats.

                Returns:
                        Tuple[float, float]: A tuple containing the realized closed PnL and the average exit price.

                Raises:
                        RuntimeError: If the API request fails, the response format is unexpected, or the PnL data is not found.
                """
                time.sleep(3)  # Wait for 3 seconds to allow for order closure and PnL update.
                try:
                        data = self.get_closed_pnl()
                        closed_pnl = float(data['result']['list'][0]['closedPnl'])
                        avg_exit_price = float(data['result']['list'][0]['avgExitPrice'])
                        return closed_pnl, avg_exit_price
                except IndexError as e:
                        raise RuntimeError("PNL data not found. No closed positions available.") from e
                except Exception as e:
                        raise RuntimeError(f"An error occurred while retrieving PnL: {e}") from e
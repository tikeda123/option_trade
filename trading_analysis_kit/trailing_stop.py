from common.config_manager import ConfigManager
from common.constants import *
from common.utils import get_config
from common.trading_logger import TradingLogger

class TrailingStopCalculator:
        """
        This class calculates the trailing stop for a trade. It manages the activation price based on the entry price,
        trailing stop percentage, and trade type (long or short). It determines if a trade should be exited
        based on the current market price and the trailing stop activation price.

        Attributes:
                trailing_rate (float): The percentage of the trailing stop.
                trailing_stop_rate (float): The calculated trailing stop percentage based on entry price and activation price.
                entry_price (float): The entry price of the trade.
                activation_price (float): The price at which the trailing stop is activated.
                current_best_price (float): The best price achieved during the trade (highest for long, lowest for short).
                trade_type (str): The type of trade, either 'ENTRY_TYPE_LONG' or 'ENTRY_TYPE_SHORT'.
                is_trailing (bool): True if trailing is active, False otherwise.
        """
        def __init__(self):
                """
                Initializes the TrailingStopCalculator with the trailing rate from the configuration.
                """
                config = get_config('ACCOUNT')
                self.trailing_rate = config['TRAILING_STOP_RATE']
                self.trailing_stop_rate = None
                self.entry_price = None
                self.activation_price = None
                self.current_best_price = None
                self.trade_type = None
                self.is_trailing = False  # 初期状態ではトレーリングは開始されていない

        def set_entry_conditions(self, entry_price, start_trailing_price, trade_type):
                """
                Sets the initial conditions for the trailing stop.

                Args:
                        entry_price (float): The entry price of the trade.
                        start_trailing_price (float): The initial price to start trailing from.
                        trade_type (str): The type of trade ('ENTRY_TYPE_LONG' or 'ENTRY_TYPE_SHORT').
                """
                self.entry_price = entry_price
                self.trailing_stop_rate = (abs(start_trailing_price - entry_price) * self.trailing_rate) / entry_price
                self.start_trailing_price = start_trailing_price
                self.trade_type = trade_type

                # Set the initial activation price based on the trade type
                if self.trade_type == ENTRY_TYPE_LONG:
                        self.activation_price = self.start_trailing_price - (self.entry_price * self.trailing_stop_rate)
                else:
                        self.activation_price = self.start_trailing_price + (self.entry_price * self.trailing_stop_rate)

                # Set the current best price to the starting trailing price
                self.current_best_price = start_trailing_price
                self.is_trailing = False  # set_entry_conditionsが呼ばれた段階ではまだトレーリングは開始されていない

        def start_trailing(self):
                """Starts trailing stop."""
                self.is_trailing = True

        def is_trailing_active(self):
                """
                Checks if the trailing stop is currently active.

                Returns:
                        bool: True if trailing is active, False otherwise.
                """
                return self.is_trailing

        def clear_status(self):
                """Clears all status except for self.trailing_rate."""
                self.trailing_stop_rate = None
                self.entry_price = None
                self.activation_price = None
                self.current_best_price = None
                self.trade_type = None
                self.is_trailing = False

        def update_price(self, current_price):
                """
                Updates the current price, recalculates the activation price, and checks if the trailing stop has been triggered.

                Args:
                        current_price (float): The current market price of the asset.

                Returns:
                        tuple: A tuple containing:
                                - bool: True if the trailing stop is triggered, False otherwise.
                                - float: The current activation price.
                """
                trade_triggered = False

                # is_trailing が True の場合のみ、有効化価格を更新
                if self.is_trailing:
                        if self.trade_type == ENTRY_TYPE_LONG:
                                # For long positions:
                                # If the current price is higher than the current best price, update the best price and recalculate the activation price
                                if current_price > self.current_best_price:
                                        self.current_best_price = current_price
                                        self.activation_price = self.current_best_price - (self.entry_price * self.trailing_stop_rate)

                                # If the current price has fallen below the activation price, trigger the trailing stop
                                if current_price <= self.activation_price:
                                        trade_triggered = True
                        else:
                                # For short positions:
                                # If the current price is lower than the current best price, update the best price and recalculate the activation price
                                if current_price < self.current_best_price:
                                        self.current_best_price = current_price
                                        self.activation_price = self.current_best_price + (self.entry_price * self.trailing_stop_rate)

                                # If the current price has risen above the activation price, trigger the trailing stop
                                if current_price >= self.activation_price:
                                        trade_triggered = True

                # Return whether the trailing stop was triggered and the current activation price
                return trade_triggered, self.activation_price

class TrailingStopAnalyzer:
                                """
                                This class analyzes trading data and applies a trailing stop strategy. It uses separate
                                TrailingStopCalculator instances for long and short trades to determine the best exit price.

                                Attributes:
                                                                __config_manager (ConfigManager): Manages configuration settings.
                                                                __logger (TradingLogger): Logs trading activity and debug messages.
                                                                __tailing_stop_duration (int): The duration for which the trailing stop is active.
                                                                __trailing_rate (float): The percentage used to calculate the trailing stop.
                                                                __long_trailing_stop (TrailingStopCalculator): Calculates the trailing stop for long trades.
                                                                __short_trailing_stop (TrailingStopCalculator): Calculates the trailing stop for short trades.
                                """

                                def __init__(self, config_manager: ConfigManager, trading_logger: TradingLogger):
                                                                """
                                                                Initializes the TrailingStopAnalyzer with configuration settings and logger.

                                                                Args:
                                                                                                config_manager (ConfigManager): The configuration manager.
                                                                                                trading_logger (TradingLogger): The trading logger.
                                                                """
                                                                self.__config_manager = config_manager
                                                                self.__logger = trading_logger
                                                                # Load the trailing stop duration and rate from the configuration
                                                                self.__tailing_stop_duration = config_manager.get('ACCOUNT', 'TRAILING_STOP_DUR')
                                                                self.__trailing_rate = config_manager.get('ACCOUNT', 'TRAILING_STOP_RATE')
                                                                # Create separate TrailingStopCalculator instances for long and short trades
                                                                self.__long_trailing_stop = TrailingStopCalculator()
                                                                self.__short_trailing_stop = TrailingStopCalculator()

                                def process_trade(self, data, index, trade_type):
                                                                """
                                                                Processes a trade and calculates the exit price based on the trailing stop strategy.

                                                                Args:
                                                                                                data (pd.DataFrame): The DataFrame containing trading data.
                                                                                                index (int): The index of the current row being processed.
                                                                                                trade_type (str): The type of trade ("long" or "short").

                                                                Returns:
                                                                                                float: The exit price calculated based on the trailing stop.
                                                                """
                                                                trade_triggered = False
                                                                entry_price = data.at[index, 'close']  # Get the entry price from the DataFrame
                                                                next_price = entry_price  # Set the initial next price to the entry price

                                                                # Select the appropriate TrailingStopCalculator based on the trade type
                                                                if trade_type == "long":
                                                                                                trailing_stop_calculator = self.__long_trailing_stop
                                                                else:
                                                                                                trailing_stop_calculator = self.__short_trailing_stop

                                                                # Process the trade over the trailing stop duration
                                                                for i in range(1, self.__tailing_stop_duration + 1):
                                                                                                if index + i < len(data):
                                                                                                                                next_price = data.at[index + i, 'close']
                                                                                                                                # Update the price in the TrailingStopCalculator and check if the trailing stop is triggered
                                                                                                                                trade_triggered, exit_price = trailing_stop_calculator.update_price(next_price)
                                                                                                                                # Log debug information about the trade processing
                                                                                                                                self.__logger.log_debug_message(f'index:{index}, i:{i}, next_price:{next_price}, trade_triggered:{trade_triggered}, exit_price:{exit_price}, trade_type:{trade_type}')
                                                                                                                                # If the trailing stop is triggered, break the loop
                                                                                                                                if trade_triggered:
                                                                                                                                                                break

                                                                # If the trailing stop was not triggered, set the exit price to the last price
                                                                if not trade_triggered:
                                                                                                exit_price = next_price

                                                                # Return the calculated exit price
                                                                return exit_price

                                def apply_trailing_stop_strategy(self, data):
                                                                """
                                                                Applies the trailing stop strategy to each row in the DataFrame.

                                                                Args:
                                                                                                data (pd.DataFrame): The DataFrame containing trading data.
                                                                """
                                                                # Iterate through each row in the DataFrame
                                                                for index in data.index:
                                                                                                # Apply the trailing stop to the current row and get the best exit price and trade type (long or short)
                                                                                                best_exit_price, is_long = self.apply_trailing_stop_to_row(data, index)

                                                                                                # Update the DataFrame with the calculated exit price and trade type
                                                                                                data.at[index, 'exit_price'] = best_exit_price
                                                                                                data.at[index, 'islong'] = is_long

                                def apply_trailing_stop_to_row(self, data, row_index):
                                                                """
                                                                Applies the trailing stop strategy to a specific row in the DataFrame.

                                                                Args:
                                                                                                data (pd.DataFrame): The DataFrame containing trading data.
                                                                                                row_index (int): The index of the row to process.

                                                                Returns:
                                                                                                tuple: A tuple containing:
                                                                                                                                - float: The best exit price calculated for the row.
                                                                                                                                - bool: True if the best exit price is for a long trade, False if it's for a short trade.

                                                                Raises:
                                                                                                IndexError: If the row index is out of bounds.
                                                                """
                                                                # Check if the row index is valid
                                                                if row_index not in data.index:
                                                                                                raise IndexError(f"Row index {row_index} is out of bounds.")

                                                                # Get the data for the current row
                                                                row = data.iloc[row_index]
                                                                # Set the entry conditions for both long and short trailing stop calculators
                                                                self.__long_trailing_stop.set_entry_conditions(row['close'], self.__trailing_rate, True)
                                                                self.__short_trailing_stop.set_entry_conditions(row['close'], self.__trailing_rate, False)

                                                                # Calculate the exit prices for both long and short trades
                                                                exit_price_long = self.process_trade(data, row_index, "long")
                                                                exit_price_short = self.process_trade(data, row_index, "short")

                                                                # Determine the best exit price and whether it's for a long or short trade
                                                                long_diff = exit_price_long - row['close']
                                                                short_diff = row['close'] - exit_price_short
                                                                is_long = long_diff > short_diff
                                                                best_exit_price = exit_price_long if is_long else exit_price_short

                                                                # Return the best exit price and the trade type
                                                                return best_exit_price, is_long




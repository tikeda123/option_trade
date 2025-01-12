import pandas as pd
import os, sys

from common.trading_logger_db import TradingLoggerDB
from common.config_manager import ConfigManager
from common.constants import *

from trading_analysis_kit.trading_state import IdleState
from trading_analysis_kit.trading_data_manager import TradingDataManager
from trading_analysis_kit.leverage_optimizer import LeverageLossCutOptimizer

class TradingContext:
        """
        This class manages the trading context, handling interactions with various trading components.

        Attributes:
                _state (IdleState): The current trading state.
                config_manager (ConfigManager): The configuration manager component.
                trading_logger (TradingLogger): The trading logger component.
                dataloader (DataLoaderDB): The data loader component.
                strategy (Any): The trading strategy context.
        """
        def __init__(self, strategy_context):
                """
                Initializes the TradingContext with the provided strategy context and initializes
                other components like the state, config manager, trading logger, and data manager.

                Args:
                        strategy_context (TradingStrategy): The trading strategy context.
                """
                self._state = IdleState()
                self.config_manager = ConfigManager()
                self.trading_logger = TradingLoggerDB()
                self.dm = TradingDataManager()
                self.strategy = strategy_context
                self.lop= LeverageLossCutOptimizer()

        def get_state(self):
                """
                Gets the current trading state.

                Returns:
                        State: The current state object.
                """
                return self._state

        def set_state(self, state):
                """
                Sets the current trading state.

                Args:
                        state (State): The new state object to set.
                """
                self._state = state

        def save_data(self):
                """
                Saves trading data to a CSV file.
                This method is currently not used as data persistence is handled elsewhere.
                """
                self.data.to_csv(FILENAME_RESULT_CSV)

        def is_first_column_less_than_second(self, index, col1, col2) -> bool:
                """
                Compares values in two columns at a specified index and returns True if the first column's value is less than the second.

                Args:
                        index (int): The index of the row in the DataFrame.
                        col1 (str): The name of the first column.
                        col2 (str): The name of the second column.

                Returns:
                        bool: True if the value in the first column is less than the value in the second column, False otherwise.
                """
                return self.dm.is_first_column_less_than_second(index, col1, col2)

        def is_first_column_greater_than_second(self, index, col1, col2) -> bool:
                """
                Compares values in two columns at a specified index and returns True if the first column's value is greater than the second.

                Args:
                        index (int): The index of the row in the DataFrame.
                        col1 (str): The name of the first column.
                        col2 (str): The name of the second column.

                Returns:
                        bool: True if the value in the first column is greater than the value in the second column, False otherwise.
                """
                return self.dm.is_first_column_greater_than_second(index, col1, col2)

        def log_transaction(self, message):
                """
                Records a transaction log with the current date and the given message.

                Args:
                        message (str): The message to log.
                """
                date = self.dm.get_current_date()
                self.trading_logger.log_transaction(date, message)

        def optimize_param(self, index: int):
                """
                Traces the volatility of the trading data at the specified index.

                Args:
                        index (int): The index of the data point to trace.
                """
                bbvi = self.dm.get_bbvi(index)
                self.lop.optimize_param(bbvi)

        def event_handle(self, index: int):
                """
                Handles an event based on the specified index. It delegates the event handling to the current state.

                Args:
                        index (int): The index of the data point where the event occurred.
                """
                self.dm.set_current_index(index)
                self._state.event_handle(self, index)

        def run_trading(self):
                """
                Executes the trading process. It analyzes data based on the provided context and generates trading events.

                Args:
                        context (TradingContext): The trading context to execute.
                """
                data = self.dm.get_raw_data()
                for index in range(len(data)):
                        self.event_handle(index)

        def load_data_from_datetime_period(self, start_datetime, end_datetime):
                """
                Loads trading data from the database for the specified period.

                Args:
                        start_datetime (datetime): The start date and time for loading data.
                        end_datetime (datetime): The end date and time for loading data.
                """
                self.dm.load_data_from_datetime_period(start_datetime, end_datetime,MARKET_DATA_TECH)

        def get_latest_data(self):
                """
                Gets the latest trading data from the database.
                This method is currently not used as data retrieval is handled elsewhere.

                Args:
                        table_name (str): The name of the database table.

                Returns:
                        pd.DataFrame: The latest data.
                """

                df = self.dm.get_latest_data()
                return df
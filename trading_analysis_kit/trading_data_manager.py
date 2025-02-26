import pandas as pd
import numpy as np
import talib as ta
import os, sys
from talib import MA_Type

# Get the absolute path of the directory containing this file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the path of the parent directory to sys.path
sys.path.append(parent_dir)

from trading_analysis_kit.trading_data import TradingStateData
from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *

class TradingDataManager:
        """
        This class manages and provides access to trading data, including both raw market data
        and derived indicators. It also handles adding analysis-specific columns to the data
        and manages the current state of trading through the TradingStateData object.
        """
        def __init__(self):
                """
                Initializes the TradingDataManager by creating instances of the data loader
                and the trading state data manager.
                """
                self.dataloader = MongoDataLoader()  # Handles loading and accessing market data
                self.state_data = TradingStateData() # Manages data related to the current trading state

                self.reset_index()  # Reset all trade-related indices

        def get_support_line(self) -> float:
                """
                Gets the support line for the current trade.
                """
                return self.state_data.get_support_line()

        def set_support_line(self, support_line: float):
                """
                Sets the support line for the current trade.
                """
                self.state_data.set_support_line(support_line)

        def get_resistance_line(self) -> float:
                """
                Gets the resistance line for the current trade.
                """
                return self.state_data.get_resistance_line()

        def set_resistance_line(self, resistance_line: float):
                """
                Sets the resistance line for the current trade.
                """
                self.state_data.set_resistance_line(resistance_line)

        def get_df_fromto(self, start_index: int, end_index: int) -> pd.DataFrame:
                """
                Retrieves a subset of the DataFrame between the specified start and end indices.

                Args:
                        start_index (int): The starting index for the slice.
                        end_index (int): The ending index for the slice.

                Returns:
                        pd.DataFrame: A slice of the DataFrame containing data between the given indices.
                """
                return self.dataloader.get_df_fromto(start_index, end_index)

        def set_df_fromto(self, start_index: int, end_index: int, col, value):
                """
                Sets a specific value to a column in the DataFrame between the specified start and end indices.

                Args:
                        start_index (int): The starting index for the range to modify.
                        end_index (int): The ending index for the range to modify.
                        col (str): The name of the column to modify.
                        value: The value to set in the specified column and index range.
                """
                self.dataloader.set_df_fromto(start_index, end_index, col, value)

        def is_first_column_less_than_second(self, column1: str, column2: str, index: int = None) -> bool:
                """
                Compares the values of two columns at a given index and returns True if the first is less than the second.

                Args:
                        column1 (str): The name of the first column.
                        column2 (str): The name of the second column.
                        index (int, optional): The index of the row to compare. If None, uses the current index.

                Returns:
                        bool: True if the value in the first column is less than the value in the second column, False otherwise.
                """
                if index is None:
                        index = self.state_data.current_index
                return self.dataloader.get_df(index, column1) < self.dataloader.get_df(index, column2)

        def is_first_column_greater_than_second(self, column1: str, column2: str, index: int = None) -> bool:
                """
                Compares the values of two columns at a given index and returns True if the first is greater than the second.

                Args:
                        column1 (str): The name of the first column.
                        column2 (str): The name of the second column.
                        index (int, optional): The index of the row to compare. If None, uses the current index.

                Returns:
                        bool: True if the value in the first column is greater than the value in the second column, False otherwise.
                """
                if index is None:
                        index = self.state_data.current_index
                return self.dataloader.get_df(index, column1) > self.dataloader.get_df(index, column2)

        def load_data_from_datetime_period(self, symbol: str, start_date: str, end_date: str):
                """
                Loads trading data for a specific symbol within a given date range from the data source.

                Args:
                        symbol (str): The trading symbol for which to load data (e.g., 'BTCUSDT').
                        start_date (str): The start date for the data in the format 'YYYY-MM-DD'.
                        end_date (str): The end date for the data in the format 'YYYY-MM-DD'.
                """
                self.dataloader.load_data_from_datetime_period(symbol, start_date, end_date)
                self.add_data_columns()  # Add custom columns for analysis after loading data

        def get_exit_price_garch(self) -> float:
                """
                Gets the exit price for the current trade.

                Returns:
                        float: The price at which the trade was exited.
                """
                return self.state_data.get_exit_price_garch()

        def set_exit_price_garch(self, price: float):
                """
                Sets the exit price for the current trade.

                Args:
                        price (float): The exit price to set.
                """
                self.state_data.set_exit_price_garch(price)

        def get_entry_price_garch(self) -> float:
                """
                Gets the entry price for the current trade.

                Returns:
                        float: The price at which the trade was entered.
                """
                return self.state_data.get_entry_price_garch()

        def set_entry_price_garch(self, price: float):
                """
                Sets the entry price for the current trade.

                Args:
                        price (float): The entry price to set.
                """
                self.state_data.set_entry_price_garch(price)

        def get_current_index(self) -> int:
                """
                Gets the current data index, representing the current point in the trading data being analyzed.

                Returns:
                        int: The current data index.
                """
                return self.state_data.get_current_index()

        def set_current_index(self, index: int):
                """
                Sets the current data index, indicating the current position in the trading data being analyzed.

                Args:
                        index (int): The index to set as the current index.
                """
                self.state_data.set_current_index(index)

        def get_entry_counter(self) -> int:
                """
                Gets the entry counter, which keeps track of the number of entry attempts.

                Returns:
                        int: The current value of the entry counter.
                """
                return self.state_data.get_entry_counter()

        def increment_entry_counter(self):
                """
                Increments the entry counter by one.
                """
                self.state_data.increment_entry_counter()

        def set_entry_counter(self, counter: int):
                """
                Sets the entry counter to a new value.

                Args:
                        counter (int): The new value to set for the entry counter.
                """
                self.state_data.set_entry_counter(counter)

        def get_entry_index(self) -> int:
                """
                Gets the index where the current trade was entered.

                Returns:
                        int: The entry index of the current trade.
                """
                return self.state_data.get_entry_index()

        def set_entry_index(self, index: int):
                """
                Sets the index where the current trade was entered.

                Args:
                        index (int): The index to set as the entry index of the current trade.
                """
                self.state_data.set_entry_index(index)

        def get_exit_index(self) -> int:
                """
                Gets the index where the current trade was exited.

                Returns:
                        int: The exit index of the current trade.
                """
                return self.state_data.get_exit_index()

        def set_exit_index(self, index: int):
                """
                Sets the index where the current trade was exited.

                Args:
                        index (int): The index to set as the exit index of the current trade.
                """
                self.state_data.set_exit_index(index)

        def get_fx_serial(self) -> int:
                """
                Gets the serial number of the current FX transaction.

                Returns:
                        int: The serial number of the current FX transaction.
                """
                return self.state_data.get_fx_serial()

        def set_fx_serial(self, serial: int):
                """
                Sets the serial number of the current FX transaction.

                Args:
                        serial (int): The serial number to set for the current FX transaction.
                """
                self.state_data.set_fx_serial(serial)

        def add_data_columns(self):
                """
                Adds custom columns to the DataFrame for trading analysis.

                These columns are used to store information calculated during analysis, such as:
                - Profit and Loss (P&L)
                - Trading State
                - Bollinger Band Direction
                - Entry and Exit Prices
                - Current and Bollinger Band Profit
                - Prediction
                - Profit Moving Average
                - Entry Type
                - Maximum and Minimum P&L
                - Exit Reason
                """
                self.dataloader.df_new_column(COLUMN_PANDL, 0.0, float)
                self.dataloader.df_new_column(COLUMN_STATE, None, str)
                self.dataloader.df_new_column(COLUMN_BB_DIRECTION, None, str)
                self.dataloader.df_new_column(COLUMN_ENTRY_PRICE, 0.0, float)
                self.dataloader.df_new_column(COLUMN_EXIT_PRICE, 0.0, float)
                self.dataloader.df_new_column(COLUMN_CURRENT_PROFIT, 0.0, float)
                self.dataloader.df_new_column(COLUMN_BB_PROFIT, 0.0, float)
                self.dataloader.df_new_column(COLUMN_PREDICTION, 0, int)
                self.dataloader.df_new_column(COLUMN_PROFIT_MA, 0.0, float)
                self.dataloader.df_new_column(COLUMN_ENTRY_TYPE,  None, str)
                self.dataloader.df_new_column(COLUMN_MAX_PANDL, 0.0, float)
                self.dataloader.df_new_column(COLUMN_MIN_PANDL, 0.0, float)
                self.dataloader.df_new_column(COLUMN_EXIT_REASON, None, str)
                self.dataloader.df_new_column(COLUMN_EXIT_TIME, None, str)
                self.dataloader.df_new_column(COLUMN_PRED_V1, 0, int)
                self.dataloader.df_new_column(COLUMN_PRED_V2, 0, int)
                self.dataloader.df_new_column(COLUMN_PRED_V3, 0, int)
                self.dataloader.df_new_column(COLUMN_PRED_V4, 0, int)
                self.dataloader.df_new_column(COLUMN_PRED_V5, 0, int)
                self.dataloader.df_new_column(COLUMN_PRED_V6, 0, int)
                self.dataloader.df_new_column(COLUMN_PRED_V7, 0, int)
                self.dataloader.df_new_column(COLUMN_PRED_V8, 0, int)
                self.dataloader.df_new_column(COLUMN_PRED_V9, 0, int)
                self.dataloader.df_new_column(COLUMN_PRED_V10, 0, int)
                self.dataloader.df_new_column(COLUMN_PRED_V11, 0, int)
                self.dataloader.df_new_column(COLUMN_PRED_V12, 0, int)
                self.dataloader.df_new_column(COLUMN_PRED_V13, 0, int)
                self.dataloader.df_new_column(COLUMN_PRED_V14, 0, int)
                self.dataloader.df_new_column(COLUMN_PRED_V15, 0, int)
                self.dataloader.df_new_column(COLUMN_PRED_V16, 0, int)
                self.dataloader.df_new_column(COLUMN_PRED_V17, 0, int)
                self.dataloader.df_new_column(COLUMN_PRED_V18, 0, int)
                self.dataloader.df_new_column(COLUMN_PRED_TARGET, 0, int)
                self.dataloader.df_new_column(COLUMN_ORDER_ID, 0.0, str)


        def get_exit_time(self, index: int = None) -> str:
                """
                Gets the exit time for a trade.

                Args:
                        index (int, optional): The index of the trade. If None, the current index is used.

                Returns:
                        str: The exit time for the trade.
                """
                return self.get_value_by_column(COLUMN_EXIT_TIME, index)

        def set_exit_time(self, exit_time: str, index: int = None):
                """
                Sets the exit time for a trade.

                Args:
                        exit_time (str): The exit time to set.
                        index (int, optional): The index of the trade. If None, the current index is used.
                """
                self.set_value_by_column(COLUMN_EXIT_TIME, exit_time, index)

        def get_order_id(self, index:int=None) -> str:
                """
                Gets the ID of the current order.

                Returns:
                        int: The ID of the current order.
                """
                if index is not None:
                        return self.get_value_by_column(COLUMN_ORDER_ID, index)
                return self.state_data.get_order_id()

        def set_order_id(self, order_id: str):
                """
                Sets the ID of the current order.

                Args:
                        order_id (int): The ID of the order to set.
                """
                self.state_data.set_order_id(order_id)
                self.set_value_by_column(COLUMN_ORDER_ID, order_id)

        def get_bbvi_median(self, start_index: int = None, end_index: int = None) -> float:
                """
                Calculates the median of the Bollinger Band Volatility Index (BBVI) within a specified range.

                Args:
                        start_index (int, optional): The starting index for the range. If None, uses the current index.
                        end_index (int, optional): The ending index for the range. If None, uses the current index + 30.

                Returns:
                        float: The median value of the BBVI within the specified range.
                """
                if end_index is None:
                        end_index = self.get_current_index()
                if start_index is None:
                        start_index = end_index - 30

                df  = self.dataloader.get_df_fromto(start_index, end_index)
                bbvi_values = df[COLUMN_BBVI].values
                return np.median(bbvi_values)

        def get_mfi(self, index: int = None) -> float:

                if index is None:
                        index = self.state_data.get_current_index()
                return self.dataloader.get_df(index, COLUMN_MFI)



        def get_bbvi(self, index: int = None) -> float:
                """
                Gets the Bollinger Band Volatility Index (BBVI) at a given index.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        float: The BBVI value.
                """
                return self.get_value_by_column(COLUMN_BBVI, index)

        def set_pred_target(self, value, index: int = None):
                """
                Sets the target prediction value for a trade.

                Args:
                        value: The target prediction value to set.
                        index (int, optional): The index of the data point. If None, the current index is used.
                """
                self.set_value_by_column(COLUMN_PRED_TARGET, value, index)

        def get_pred_target(self, index: int = None) -> int:
                """
                Gets the target prediction value for a trade.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        int: The target prediction value.
                """
                return self.get_value_by_column(COLUMN_PRED_TARGET, index)

        def set_pred_v1(self, value, index: int = None):
                """
                Sets the value of the first prediction column.

                Args:
                        value: The value to set.
                        index (int, optional): The index of the data point. If None, the current index is used.
                """
                self.set_value_by_column(COLUMN_PRED_V1, value, index)

        def get_pred_v1(self, index: int = None) -> int:
                """
                Gets the value of the first prediction column.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        int: The value of the first prediction column.
                """
                return self.get_value_by_column(COLUMN_PRED_V1, index)

        def set_pred_v2(self, value, index: int = None):
                """
                Sets the value of the second prediction column.

                Args:
                        value: The value to set.
                        index (int, optional): The index of the data point. If None, the current index is used.
                """
                self.set_value_by_column(COLUMN_PRED_V2, value, index)

        def get_pred_v2(self, index: int = None) -> int:
                """
                Gets the value of the second prediction column.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        int: The value of the second prediction column.
                """
                return self.get_value_by_column(COLUMN_PRED_V2, index)

        def set_pred_v3(self, value, index: int = None):
                """
                Sets the value of the third prediction column.

                Args:
                        value: The value to set.
                        index (int, optional): The index of the data point. If None, the current index is used.
                """
                self.set_value_by_column(COLUMN_PRED_V3, value, index)

        def get_pred_v3(self, index: int = None) -> int:
                """
                Gets the value of the third prediction column.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        int: The value of the third prediction column.
                """
                return self.get_value_by_column(COLUMN_PRED_V3, index)

        def set_pred_v4(self, value, index: int = None):
                """
                Sets the value of the fourth prediction column.

                Args:
                        value: The value to set.
                        index (int, optional): The index of the data point. If None, the current index is used.
                """
                self.set_value_by_column(COLUMN_PRED_V4, value, index)

        def get_pred_v4(self, index: int = None) -> int:
                """
                Gets the value of the fourth prediction column.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        int: The value of the fourth prediction column.
                """
                return self.get_value_by_column(COLUMN_PRED_V4, index)

        def set_pred_v5(self, value, index: int = None):
                """
                Sets the value of the fifth prediction column.

                Args:
                        value: The value to set.
                        index (int, optional): The index of the data point. If None, the current index is used.
                """
                self.set_value_by_column(COLUMN_PRED_V5, value, index)

        def get_pred_v5(self, index: int = None) -> int:
                """
                Gets the value of the fifth prediction column.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        int: The value of the fifth prediction column.
                """
                return self.get_value_by_column(COLUMN_PRED_V5, index)

        def set_pred_v6(self, value, index: int = None):
                """
                Sets the value of the sixth prediction column.

                Args:
                        value: The value to set.
                        index (int, optional): The index of the data point. If None, the current index is used.
                """
                self.set_value_by_column(COLUMN_PRED_V6, value, index)

        def get_pred_v6(self, index: int = None) -> int:
                """
                Gets the value of the sixth prediction column.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        int: The value of the sixth prediction column.
                """
                return self.get_value_by_column(COLUMN_PRED_V6, index)

        def set_pred_v7(self, value, index: int = None):
                """
                Sets the value of the seventh prediction column.

                Args:
                        value: The value to set.
                        index (int, optional): The index of the data point. If None, the current index is used.
                """
                self.set_value_by_column(COLUMN_PRED_V7, value, index)

        def get_pred_v7(self, index: int = None) -> int:
                """
                Gets the value of the seventh prediction column.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        int: The value of the seventh prediction column.
                """
                return self.get_value_by_column(COLUMN_PRED_V7, index)

        def set_pred_v8(self, value, index: int = None):
                """
                Sets the value of the eighth prediction column.

                Args:
                        value: The value to set.
                        index (int, optional): The index of the data point. If None, the current index is used.
                """
                self.set_value_by_column(COLUMN_PRED_V8, value, index)

        def get_pred_v8(self, index: int = None) -> int:
                """
                Gets the value of the eighth prediction column.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        int: The value of the eighth prediction column.
                """
                return self.get_value_by_column(COLUMN_PRED_V8, index)

        def set_pred_v9(self, value, index: int = None):
                """
                Sets the value of the ninth prediction column.

                Args:
                        value: The value to set.
                        index (int, optional): The index of the data point. If None, the current index is used.
                """
                self.set_value_by_column(COLUMN_PRED_V9, value, index)

        def get_pred_v9(self, index: int = None) -> int:
                """
                Gets the value of the ninth prediction column.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        int: The value of the ninth prediction column.
                """
                return self.get_value_by_column(COLUMN_PRED_V9, index)

        def set_pred_v10(self, value, index: int = None):
                """
                Sets the value of the tenth prediction column.

                Args:
                        value: The value to set.
                        index (int, optional): The index of the data point. If None, the current index is used.
                """
                self.set_value_by_column(COLUMN_PRED_V10, value, index)

        def get_pred_v10(self, index: int = None) -> int:
                """
                Gets the value of the tenth prediction column.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        int: The value of the tenth prediction column.
                """
                return self.get_value_by_column(COLUMN_PRED_V10, index)

        def set_pred_v11(self, value, index: int = None):
                """
                Sets the value of the eleventh prediction column.

                Args:
                        value: The value to set.
                        index (int, optional): The index of the data point. If None, the current index is used.
                """
                self.set_value_by_column(COLUMN_PRED_V11, value, index)

        def get_pred_v11(self, index: int = None) -> int:
                """
                Gets the value of the eleventh prediction column.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        int: The value of the eleventh prediction column.
                """
                return self.get_value_by_column(COLUMN_PRED_V11, index)

        def set_pred_v12(self, value, index: int = None):
                """
                Sets the value of the twelfth prediction column.

                Args:
                        value: The value to set.
                        index (int, optional): The index of the data point. If None, the current index is used.
                """
                self.set_value_by_column(COLUMN_PRED_V12, value, index)

        def get_pred_v12(self, index: int = None) -> int:
                """
                Gets the value of the twelfth prediction column.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        int: The value of the twelfth prediction column.
                """

                return self.get_value_by_column(COLUMN_PRED_V12, index)

        def set_pred_v13(self, value, index: int = None):
                """
                Sets the value of the twelfth prediction column.

                Args:
                        value: The value to set.
                        index (int, optional): The index of the data point. If None, the current index is used.
                """
                self.set_value_by_column(COLUMN_PRED_V13, value, index)

        def get_pred_v13(self, index: int = None) -> int:
                """
                Gets the value of the twelfth prediction column.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        int: The value of the twelfth prediction column.
                """
                return self.get_value_by_column(COLUMN_PRED_V13, index)

        def set_pred_v14(self, value, index: int = None):
                """
                Sets the value of the twelfth prediction column.

                Args:
                        value: The value to set.
                        index (int, optional): The index of the data point. If None, the current index is used.
                """
                self.set_value_by_column(COLUMN_PRED_V14, value, index)

        def get_pred_v14(self, index: int = None) -> int:
                """
                Gets the value of the twelfth prediction column.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        int: The value of the twelfth prediction column.
                """
                return self.get_value_by_column(COLUMN_PRED_V14, index)

        def set_pred_v15(self, value, index: int = None):
                """
                Sets the value of the twelfth prediction column.

                Args:
                        value: The value to set.
                        index (int, optional): The index of the data point. If None, the current index is used.
                """
                self.set_value_by_column(COLUMN_PRED_V15, value, index)

        def get_pred_v15(self, index: int = None) -> int:
                """
                Gets the value of the twelfth prediction column.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        int: The value of the twelfth prediction column.
                """
                return self.get_value_by_column(COLUMN_PRED_V15, index)

        def set_pred_v16(self, value, index: int = None):
                """
                Sets the value of the twelfth prediction column.

                Args:
                        value: The value to set.
                        index (int, optional): The index of the data point. If None, the current index is used.
                """
                self.set_value_by_column(COLUMN_PRED_V16, value, index)

        def get_pred_v16(self, index: int = None) -> int:
                """
                Gets the value of the twelfth prediction column.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        int: The value of the twelfth prediction column.
                """
                return self.get_value_by_column(COLUMN_PRED_V16, index)

        def set_pred_v17(self, value, index: int = None):
                """
                Sets the value of the twelfth prediction column.

                Args:
                        value: The value to set.
                        index (int, optional): The index of the data point. If None, the current index is used.
                """
                self.set_value_by_column(COLUMN_PRED_V17, value, index)

        def get_pred_v17(self, index: int = None) -> int:
                """
                Gets the value of the twelfth prediction column.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        int: The value of the twelfth prediction column.
                """
                return self.get_value_by_column(COLUMN_PRED_V17, index)

        def set_pred_v18(self, value, index: int = None):
                """
                Sets the value of the twelfth prediction column.

                Args:
                        value: The value to set.
                        index (int, optional): The index of the data point. If None, the current index is used.
                """
                self.set_value_by_column(COLUMN_PRED_V18, value, index)

        def get_pred_v18(self, index: int = None) -> int:
                """
                Gets the value of the twelfth prediction column.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        int: The value of the twelfth prediction column.
                """
                return self.get_value_by_column(COLUMN_PRED_V18, index)

        def get_raw_data(self) -> pd.DataFrame:
                """
                Gets the raw trading data DataFrame.

                Returns:
                        pd.DataFrame: The raw trading data DataFrame.
                """
                return self.dataloader.get_df_raw()

        def get_value_by_column(self, column_name: str, index: int = None) -> float:
                """
                Gets the value from a specific column at a given index.

                Args:
                        column_name (str): The name of the column.
                        index (int, optional): The row index. If None, the current index is used.

                Returns:
                        float: The value at the specified column and index.
                """
                if index is None:
                        index = self.state_data.get_current_index()
                return self.dataloader.get_df(index, column_name)

        def set_value_by_column(self, column_name: str, value, index: int = None):
                """
                Sets a value in a specific column at a given index.

                Args:
                        column_name (str): The name of the column.
                        value: The value to set.
                        index (int, optional): The row index. If None, the current index is used.
                """
                if index is None:
                        index = self.state_data.get_current_index()
                return self.dataloader.set_df(index, column_name, value)

        def get_max_pandl(self, index: int = None) -> float:
                """
                Gets the maximum profit and loss (P&L) recorded for a trade.

                Args:
                        index (int, optional): The index of the trade. If None, the current index is used.

                Returns:
                        float: The maximum P&L for the trade.
                """
                return self.get_value_by_column(COLUMN_MAX_PANDL, index)

        def set_max_pandl(self, price: float, index: int = None):
                """
                Sets the maximum profit and loss (P&L) for a trade.

                Args:
                        price (float): The maximum P&L to set.
                        index (int, optional): The index of the trade. If None, the current index is used.
                """
                self.set_value_by_column(COLUMN_MAX_PANDL, price, index)

        def get_min_pandl(self, index: int = None) -> float:
                """
                Gets the minimum profit and loss (P&L) recorded for a trade.

                Args:
                        index (int, optional): The index of the trade. If None, the current index is used.

                Returns:
                        float: The minimum P&L for the trade.
                """
                return self.get_value_by_column(COLUMN_MIN_PANDL, index)

        def set_min_pandl(self, price: float, index: int = None):
                """
                Sets the minimum profit and loss (P&L) for a trade.

                Args:
                        price (float): The minimum P&L to set.
                        index (int, optional): The index of the trade. If None, the current index is used.
                """
                self.set_value_by_column(COLUMN_MIN_PANDL, price, index)

        def get_bb_direction(self, index: int = None) -> str:
                """
                Gets the direction of the Bollinger Band at a given index.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        str: The direction of the Bollinger Band (e.g., "UPPER" or "LOWER").
                """
                if index is None:
                        return self.state_data.get_bb_direction()
                return self.get_value_by_column(COLUMN_BB_DIRECTION, index)

        def set_bb_direction(self, direction: str, index: int = None):
                """
                Sets the direction of the Bollinger Band at a given index.

                Args:
                        direction (str): The direction of the Bollinger Band (e.g., "UPPER" or "LOWER").
                        index (int, optional): The index of the data point. If None, the current index is used.
                """
                self.state_data.set_bb_direction(direction)
                self.set_value_by_column(COLUMN_BB_DIRECTION, direction, index)

        def get_bb_profit(self, index: int = None) -> float:
                """
                Gets the profit calculated using the Bollinger Band strategy at a given index.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        float: The profit from the Bollinger Band strategy.
                """
                return self.get_value_by_column(COLUMN_BB_PROFIT, index)

        def set_bb_profit(self, profit: float, index: int = None):
                """
                Sets the profit calculated using the Bollinger Band strategy at a given index.

                Args:
                        profit (float): The profit value to set.
                        index (int, optional): The index of the data point. If None, the current index is used.
                """
                self.set_value_by_column(COLUMN_BB_PROFIT, profit, index)

        def get_current_profit(self, index: int = None) -> float:
                """
                Gets the current profit for a trade.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        float: The current profit of the trade.
                """
                return self.get_value_by_column(COLUMN_CURRENT_PROFIT, index)

        def set_current_profit(self, profit: float, index: int = None):
                """
                Sets the current profit for a trade.

                Args:
                        profit (float): The current profit to set.
                        index (int, optional): The index of the data point. If None, the current index is used.
                """
                self.set_value_by_column(COLUMN_CURRENT_PROFIT, profit, index)

        def get_pandl(self, index: int = None) -> float:
                """
                Gets the profit and loss (P&L) for a trade.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        float: The P&L for the trade.
                """
                return self.get_value_by_column(COLUMN_PANDL, index)

        def set_pandl(self, pandl: float, index: int = None):
                """
                Sets the profit and loss (P&L) for a trade.

                Args:
                        pandl (float): The P&L to set.
                        index (int, optional): The index of the data point. If None, the current index is used.
                """
                self.set_value_by_column(COLUMN_PANDL, pandl, index)

        def get_high_price(self, index: int = None) -> float:
                """
                Gets the high price at a given index.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        float: The high price.
                """
                return self.get_value_by_column(COLUMN_HIGH, index)

        def get_low_price(self, index: int = None) -> float:
                """
                Gets the low price at a given index.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        float: The low price.
                """
                return self.get_value_by_column(COLUMN_LOW, index)

        def get_lower2_price(self, index: int = None) -> float:
                """
                Gets the price of the lower Bollinger Band (2nd standard deviation) at a given index.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        float: The price of the lower Bollinger Band.
                """
                return self.get_value_by_column(COLUMN_LOWER_BAND2, index)

        def get_upper2_price(self, index: int = None) -> float:
                """
                Gets the price of the upper Bollinger Band (2nd standard deviation) at a given index.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        float: The price of the upper Bollinger Band.
                """
                return self.get_value_by_column(COLUMN_UPPER_BAND2, index)

        def get_open_price(self, index: int = None) -> float:
                """
                Gets the open price at a given index.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        float: The open price.
                """
                return self.get_value_by_column(COLUMN_OPEN, index)

        def get_middle_price(self, index: int = None) -> float:
                """
                Gets the price of the middle Bollinger Band at a given index.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        float: The price of the middle Bollinger Band.
                """
                return self.get_value_by_column(COLUMN_MIDDLE_BAND, index)

        def get_ema_price(self, index: int = None) -> float:
                """
                Gets the Exponential Moving Average (EMA) price at a given index.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        float: The EMA price.
                """
                return self.get_value_by_column(COLUMN_EMA, index)

        def get_entry_type(self, index: int = None) -> str:
                """
                Gets the type of entry for a trade at a given index.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        str: The type of entry (e.g., "LONG" or "SHORT").
                """
                if index is None:
                        return self.state_data.get_entry_type()
                return self.get_value_by_column(COLUMN_ENTRY_TYPE, index)

        def set_entry_type(self, entry_type: str, index: int = None):
                """
                Sets the type of entry for a trade at a given index.

                Args:
                        entry_type (str): The type of entry to set (e.g., "LONG" or "SHORT").
                        index (int, optional): The index of the data point. If None, the current index is used.
                """
                self.state_data.set_entry_type(entry_type)
                self.set_value_by_column(COLUMN_ENTRY_TYPE, entry_type, index)

        def get_prediction(self, index: int = None) -> int:
                """
                Gets the prediction made for a trade at a given index.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        int: The prediction for the trade.
                """
                if index is None:
                        return self.state_data.get_prediction()
                return self.get_value_by_column(COLUMN_PREDICTION, index)

        def set_prediction(self, prediction: int, index: int = None):
                """
                Sets the prediction for a trade at a given index.

                Args:
                        prediction (int): The prediction to set.
                        index (int, optional): The index of the data point. If None, the current index is used.
                """
                self.state_data.set_prediction(prediction)
                self.set_value_by_column(COLUMN_PREDICTION, prediction, index)

        def get_close_price(self, index: int = None) -> float:
                """
                Gets the closing price at a given index.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        float: The closing price.
                """
                return self.get_value_by_column(COLUMN_CLOSE, index)

        def get_current_date(self, index: int = None) -> str:
                """
                Gets the date of the data point at a given index.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        str: The date of the data point.
                """
                return self.get_value_by_column(COLUMN_DATE, index)

        def get_entry_price(self, index: int = None) -> float:
                """
                Gets the entry price of a trade at a given index.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        float: The entry price of the trade.
                """
                if index is None:
                        return self.state_data.get_entry_price()
                return self.get_value_by_column(COLUMN_ENTRY_PRICE, index)

        def set_entry_price(self, price: float, index: int = None):
                """
                Sets the entry price for a trade at a given index.

                Args:
                        price (float): The entry price to set.
                        index (int, optional): The index of the data point. If None, the current index is used.
                """
                self.state_data.set_entry_price(price)
                self.set_value_by_column(COLUMN_ENTRY_PRICE, price, index)

        def get_exit_price(self, index: int = None) -> float:
                """
                Gets the exit price of a trade at a given index.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        float: The exit price of the trade.
                """
                return self.get_value_by_column(COLUMN_EXIT_PRICE, index)

        def set_exit_price(self, price: float, index: int = None):
                """
                Sets the exit price for a trade at a given index.

                Args:
                        price (float): The exit price to set.
                        index (int, optional): The index of the data point. If None, the current index is used.
                """
                self.set_value_by_column(COLUMN_EXIT_PRICE, price, index)

        def read_state(self, index: int = None) -> str:
                """
                Gets the trading state at a given index.

                Args:
                        index (int, optional): The index of the data point. If None, the current index is used.

                Returns:
                        str: The trading state (e.g., "IDLE", "ENTRY_PREPARATION", "POSITION").
                """
                return self.get_value_by_column(COLUMN_STATE, index)

        def record_state(self, state: str, index: int = None):
                """
                Records the trading state at a given index.

                Args:
                        state (str): The trading state to record.
                        index (int, optional): The index of the data point. If None, the current index is used.
                """
                self.set_value_by_column(COLUMN_STATE, state, index)

        def set_exit_reason(self, reason: str, index: int = None):
                """
                Sets the reason for exiting a trade at a given index.

                Args:
                        reason (str): The reason for exiting the trade.
                        index (int, optional): The index of the data point. If None, the current index is used.
                """
                self.set_value_by_column(COLUMN_EXIT_REASON, reason, index)

        def reset_index(self):
                """
                Resets all trade-related indices to their default values.
                """
                self.state_data.reset_index()

        def get_latest_data(self):
                """
                Gets the latest data point from the trading data.

                Returns:
                        pd.DataFrame: The latest data point.
                """
                return self.dataloader.get_latest_data()

        def get_latest_index(self):
                """
                Gets the index of the latest data point in the trading data.

                Returns:
                        int: The index of the latest data point.
                """
                return self.dataloader.get_latest_index()

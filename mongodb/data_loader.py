import sys, os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Any, Optional, Callable

# Get the absolute path of the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the parent directory's path to sys.path for module imports
sys.path.append(parent_dir)

from common.trading_logger import TradingLogger
from common.constants import COLUMN_BB_DIRECTION

class DataLoader:
        """
        Class for loading and manipulating data, particularly for trading analysis.

        Attributes:
                logger (TradingLogger): An object for managing log information.
                _df (Optional[pd.DataFrame]): The loaded raw data, stored as a Pandas DataFrame.
                                                                          Initialized as None.
        """

        def __init__(self):
                """Initializes the DataLoader object with a logger and an empty DataFrame."""
                self.logger = TradingLogger()
                self._df: Optional[pd.DataFrame] = None

        def remove_unuse_colums(self) -> None:
                """Removes the "_id" column from the DataFrame if it exists."""
                if self._df is not None and '_id' in self._df.columns:
                        self._df = self._df.drop(columns=['_id'])

        def reset_index(self) -> None:
                """Resets the index of the DataFrame."""
                if self._df is not None:
                        self._df = self._df.reset_index(drop=False)

        def get_df_raw(self) -> Optional[pd.DataFrame]:
                """
                Returns the raw DataFrame.

                Returns:
                        Optional[pd.DataFrame]: The loaded raw data as a Pandas DataFrame, or None if no data is loaded.
                """
                return self._df

        def set_df_raw(self, df: pd.DataFrame) -> None:
                """
                Sets the raw DataFrame.

                Args:
                        df (pd.DataFrame): The DataFrame to be set as the raw data.
                """
                self._df = df

        def get_df(self, index: int, column: str) -> Any:
                """
                Gets the value at the specified row index and column name in the DataFrame.

                Args:
                        index (int): The row index.
                        column (str): The column name.

                Returns:
                        Any: The value at the specified index and column.
                                 The return type depends on the data type of the column.
                """
                if self._df is not None:
                        return self._df.at[index, column]

        def set_df(self, index: int, column: str, value: Any) -> None:
                """
                Sets the value at the specified row index and column name in the DataFrame.

                Args:
                        index (int): The row index.
                        column (str): The column name.
                        value (Any): The value to be set at the specified location.
                """
                if self._df is not None:
                        self._df.at[index, column] = value

        def set_df_fromto(self, start_index: int, end_index: int, column: str, value: Any) -> None:
                """
                Sets a value for a specified column within a range of row indices in the DataFrame.

                Args:
                        start_index (int): The starting row index (inclusive).
                        end_index (int): The ending row index (inclusive).
                        column (str): The name of the column to set values for.
                        value (Any): The value to be set for the specified range.
                """
                if self._df is not None:
                        self._df.loc[start_index:end_index, column] = value

        def get_df_fromto(self, start_index: int, end_index: int) -> Optional[pd.DataFrame]:
                """
                Retrieves a subset of the DataFrame for a given range of row indices.

                Args:
                        start_index (int): The starting row index (inclusive).
                        end_index (int): The ending row index (inclusive).

                Returns:
                        Optional[pd.DataFrame]: A subset of the DataFrame for the specified range of indices,
                                                                   or None if the DataFrame is not initialized.
                """
                if  self._df['date'].isna().any():
                        print(f'************************************************************************************')
                        print("ERROR: get_df_fromto: 'date' column has NaN values")
                        self._df = self._df.dropna(subset=['date'])

                if self._df is not None:
                        return self._df.loc[start_index:end_index]

        def df_new_column(self, column: str, value: Any, dtype: Any) -> None:
                """
                Adds a new column to the DataFrame with an initial value and data type.

                Args:
                        column (str): The name of the new column.
                        value (Any): The initial value to fill the column with.
                        dtype (Any): The data type for the new column (e.g., 'int', 'float', 'str').
                """
                if self._df is not None:
                        self._df[column] = value
                        self._df[column] = self._df[column].astype(dtype)

        def is_first_column_less_than_second(self, index: int, col1: str, col2: str) -> Optional[bool]:
                """
                Checks if the value in the first column is less than the value in the second column
                at the given row index.

                Args:
                        index (int): The row index.
                        col1 (str): The name of the first column.
                        col2 (str): The name of the second column.

                Returns:
                        Optional[bool]: True if the value in the first column is less than the value in
                                                         the second column, False otherwise. Returns None if the DataFrame is not initialized.
                """
                if self._df is not None:
                        return self._df.at[index, col1] < self._df.at[index, col2]

        def is_first_column_greater_than_second(self, index: int, col1: str, col2: str) -> Optional[bool]:
                """
                Checks if the value in the first column is greater than the value in the second column
                at the given row index.

                Args:
                        index (int): The row index.
                        col1 (str): The name of the first column.
                        col2 (str): The name of the second column.

                Returns:
                        Optional[bool]: True if the value in the first column is greater than the value in
                                                        the second column, False otherwise. Returns None if the DataFrame is not initialized.
                """
                if self._df is not None:
                        return self._df.at[index, col1] > self._df.at[index, col2]

        def max_value(self, start_index: int, end_index: int, column: str) -> Optional[float]:
                """
                Finds the maximum value within a specified range of rows in a given column.

                Args:
                        start_index (int): The starting row index (inclusive).
                        end_index (int): The ending row index (inclusive).
                        column (str): The name of the column.

                Returns:
                        Optional[float]: The maximum value in the specified column and range,
                                                        or None if the DataFrame is not initialized.
                """
                if self._df is not None:
                        return self._df.loc[start_index:end_index, column].max()

        def min_value(self, start_index: int, end_index: int, column: str) -> Optional[float]:
                """
                Finds the minimum value within a specified range of rows in a given column.

                Args:
                        start_index (int): The starting row index (inclusive).
                        end_index (int): The ending row index (inclusive).
                        column (str): The name of the column.

                Returns:
                        Optional[float]: The minimum value in the specified column and range,
                                                         or None if the DataFrame is not initialized.
                """
                if self._df is not None:
                        return self._df.loc[start_index:end_index, column].min()

        def max_value_index(self, start_index: int, end_index: int, column: str) -> Optional[int]:
                """
                Finds the index of the row containing the maximum value within a specified range
                in a given column.

                Args:
                        start_index (int): The starting row index (inclusive).
                        end_index (int): The ending row index (inclusive).
                        column (str): The name of the column.

                Returns:
                        Optional[int]: The index of the row with the maximum value in the specified range,
                                                  or None if the DataFrame is not initialized.
                """
                if self._df is not None:
                        return self._df.loc[start_index:end_index, column].idxmax()

        def min_value_index(self, start_index: int, end_index: int, column: str) -> Optional[int]:
                """
                Finds the index of the row containing the minimum value within a specified range
                in a given column.

                Args:
                        start_index (int): The starting row index (inclusive).
                        end_index (int): The ending row index (inclusive).
                        column (str): The name of the column.

                Returns:
                        Optional[int]: The index of the row with the minimum value in the specified range,
                                                  or None if the DataFrame is not initialized.
                """
                if self._df is not None:
                        return self._df.loc[start_index:end_index, column].idxmin()

        def mean_value(self, start_index: int, end_index: int, column: str) -> Optional[float]:
                """
                Calculates the mean (average) value within a specified range of rows in a given column.

                Args:
                        start_index (int): The starting row index (inclusive).
                        end_index (int): The ending row index (inclusive).
                        column (str): The name of the column.

                Returns:
                        Optional[float]: The mean value within the specified range,
                                                         or None if the DataFrame is not initialized.
                """
                if self._df is not None:
                        return self._df.loc[start_index:end_index, column].mean()

        def std_value(self, start_index: int, end_index: int, column: str) -> Optional[float]:
                """
                Calculates the standard deviation of values within a specified range of rows in a given column.

                Args:
                        start_index (int): The starting row index (inclusive).
                        end_index (int): The ending row index (inclusive).
                        column (str): The name of the column.

                Returns:
                        Optional[float]: The standard deviation of values within the specified range,
                                                         or None if the DataFrame is not initialized.
                """
                if self._df is not None:
                        return self._df.loc[start_index:end_index, column].std()

        def describe(self, start_index: int, end_index: int, column: str) -> Optional[pd.Series]:
                """
                Provides descriptive statistics for a given column within a specified range of rows.

                Args:
                        start_index (int): The starting row index (inclusive).
                        end_index (int): The ending row index (inclusive).
                        column (str): The name of the column.

                Returns:
                        Optional[pd.Series]: A Pandas Series containing descriptive statistics
                                                           (count, mean, std, min, 25%, 50%, 75%, max)
                                                           for the specified range, or None if the DataFrame is not initialized.
                """
                if self._df is not None:
                        return self._df.loc[start_index:end_index, column].describe()

        def filter(self, column: str, operator: Callable, value: Any) -> Optional[pd.DataFrame]:
                """
                Filters the DataFrame based on a condition applied to a specific column.

                Args:
                        column (str): The name of the column to filter.
                        operator (Callable): The comparison operator used for filtering (e.g., `operator.gt`, `operator.lt`).
                        value (Any): The value to compare against.

                Returns:
                        Optional[pd.DataFrame]: The filtered DataFrame containing rows that satisfy the condition,
                                                                        or None if the DataFrame is not initialized.
                """
                if self._df is not None:
                        return self._df[operator(self._df[column], value)]

        def filter_and(
                self, column1: str, operator1: Callable, value1: Any, column2: str, operator2: Callable, value2: Any
        ) -> Optional[pd.DataFrame]:
                """
                Filters the DataFrame based on a combined 'AND' condition applied to two columns.

                Args:
                        column1 (str): The name of the first column to filter.
                        operator1 (Callable): The comparison operator for the first column (e.g., `operator.gt`).
                        value1 (Any): The value to compare against in the first column.
                        column2 (str): The name of the second column to filter.
                        operator2 (Callable): The comparison operator for the second column (e.g., `operator.lt`).
                        value2 (Any): The value to compare against in the second column.

                Returns:
                        Optional[pd.DataFrame]: The filtered DataFrame containing rows that satisfy both conditions,
                                                                        or None if the DataFrame is not initialized.
                """
                if self._df is not None:
                        condition1 = operator1(self._df[column1], value1)
                        condition2 = operator2(self._df[column2], value2)
                        return self._df[condition1 & condition2]

        def filter_or(
                self, column1: str, operator1: Callable, value1: Any, column2: str, operator2: Callable, value2: Any
        ) -> Optional[pd.DataFrame]:
                """
                Filters the DataFrame based on a combined 'OR' condition applied to two columns.

                Args:
                        column1 (str): The name of the first column to filter.
                        operator1 (Callable): The comparison operator for the first column (e.g., `operator.gt`).
                        value1 (Any): The value to compare against in the first column.
                        column2 (str): The name of the second column to filter.
                        operator2 (Callable): The comparison operator for the second column (e.g., `operator.lt`).
                        value2 (Any): The value to compare against in the second column.

                Returns:
                        Optional[pd.DataFrame]: The filtered DataFrame containing rows that satisfy at least one condition,
                                                                        or None if the DataFrame is not initialized.
                """
                if self._df is not None:
                        condition1 = operator1(self._df[column1], value1)
                        condition2 = operator2(self._df[column2], value2)
                        return self._df[condition1 | condition2]

        def get_latest_data(self) -> Optional[pd.DataFrame]:
                """
                Returns the latest data in the DataFrame.

                Returns:
                        Optional[pd.DataFrame]: The latest data in the DataFrame, or None if the DataFrame is not initialized.
                """
                if self._df is not None:
                        return self._df.tail(1)

        def get_latest_index(self) -> Optional[int]:
                """
                Returns the index of the latest data point in the DataFrame.

                Returns:
                        Optional[int]: The index of the latest data point, or None if the DataFrame is not initialized.
                """
                if self._df is not None:
                        return self._df.index[-1]

        def append_df (self, new_df: pd.DataFrame) -> None:
                """
                Appends a new df to the end of the DataFrame.

                Args:
                        new_row (pd.DataFrame): The new row to be appended to the DataFrame.

                Raises:
                        ValueError: If the DataFrame is not initialized or if the new_row keys don't match the DataFrame columns.
                """
                if self._df is None:
                        raise ValueError("DataFrame is not initialized. Please set a DataFrame first.")

                if set(new_df.keys()) != set(self._df.columns):
                        raise ValueError("The keys in new_row do not match the DataFrame columns.")

                self._df = pd.concat([self._df, new_df], ignore_index=True)

                #self.logger.log_system_message(f"New row appended. Current DataFrame shape: {self._df.shape}")

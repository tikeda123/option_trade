import os
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, List
from tqdm import tqdm

# Add parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.constants import MARKET_DATA
from mongodb.data_loader_mongo import MongoDataLoader

import pandas as pd
import numpy as np

from pandas.tseries.frequencies import to_offset


def resample_df(df, to_timeframe='12H', base_time='00:00:00'):
        """
        Resamples a DataFrame containing 1-hour interval data to a specified time frame, with customizable period start times.

        Parameters:
        - df (DataFrame): Input DataFrame with 1-hour interval data.
          Required columns: 'start_at', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'date' (optional), and any other columns.
        - to_timeframe (str): Target resampling time frame (e.g., '12H' for 12-hour intervals). Default is '12H'.
        - base_time (str): The time of day to align the resampling periods with (e.g., '01:00:00'). Default is '00:00:00'.

        Returns:
        - DataFrame: Resampled DataFrame with the same columns as the input, aggregated according to specified rules.
        """

        required_columns = ['start_at', 'open', 'high', 'low', 'close', 'volume', 'turnover']

        # Check if required columns exist
        for col in required_columns:
                if col not in df.columns:
                        raise ValueError(f"Column '{col}' is missing from the DataFrame.")

        # Ensure 'start_at' is datetime
        try:
                df['start_at'] = pd.to_datetime(df['start_at'])
        except Exception as e:
                raise ValueError(f"Error converting 'start_at' to datetime: {e}")

        if df['start_at'].isnull().any():
                raise ValueError("Some 'start_at' values are NaT.")

        if df.empty:
                raise ValueError("Input DataFrame is empty.")

        df = df.sort_values('start_at')
        df.set_index('start_at', inplace=True)

        # Parse base_time into Timedelta
        try:
                offset = pd.to_timedelta(base_time)
        except Exception as e:
                raise ValueError(f"Error converting 'base_time' to Timedelta: {e}")

        # Define aggregation functions
        agg_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'turnover': 'sum'
        }

        # For other columns, assign 'first' as the aggregation function
        other_cols = df.columns.difference(agg_dict.keys())
        for col in other_cols:
                agg_dict[col] = 'first'

        # Perform resampling with origin and offset
        try:
                resampled_df = df.resample(
                        to_timeframe,
                        origin='start_day',
                        offset=offset,
                        label='left',
                        closed='left'
                ).agg(agg_dict)
        except Exception as e:
                raise ValueError(f"Error during resampling: {e}")

        # Drop rows where the start of the interval is before the first data point
        resampled_df = resampled_df[resampled_df.index >= df.index[0]]

        # Drop rows with NaN in 'open' (periods with no data)
        resampled_df.dropna(subset=['open'], inplace=True)

        # Reset index to get 'start_at' back as a column
        resampled_df.reset_index(inplace=True)

        # Update 'date' column if it exists
        if 'date' in resampled_df.columns:
                resampled_df['date'] = resampled_df['start_at']

        return resampled_df


def validate_resampled_data(df_original: pd.DataFrame, df_resampled: pd.DataFrame, timeframe_hours: int):
        """
        Validates the resampled DataFrame against the original 1-hour interval DataFrame.

        Parameters:
        - df_original (DataFrame): Original 1-hour interval DataFrame with 'open', 'high', 'low', 'close'.
        - df_resampled (DataFrame): Resampled DataFrame to validate.
        - timeframe_hours (int): The number of hours for each resampled period (e.g., 12 for 12H, 4 for 4H).

        Returns:
        - None
        """
        # Ensure 'start_at' is datetime and set as index
        df_original = df_original.copy()
        df_original.set_index('start_at', inplace=True)

        # Iterate over each resampled row
        for idx, row in df_resampled.iterrows():
                start_time = row['start_at']
                # Calculate end_time based on timeframe_hours
                end_time = start_time + pd.Timedelta(hours=timeframe_hours - 1)

                # Extract the corresponding period from the original data
                df_period = df_original.loc[start_time:end_time]

                if df_period.empty:
                        print(f"{start_time} から {end_time} の間の1時間足データが存在しません。")
                        continue

                # Extract expected values
                expected_open = df_period['open'].iloc[0]
                expected_close = df_period['close'].iloc[-1]
                expected_high = df_period['high'].max()
                expected_low = df_period['low'].min()

                # Extract resampled values
                resampled_open = row['open']
                resampled_close = row['close']
                resampled_high = row['high']
                resampled_low = row['low']

                # Compare and print discrepancies
                if expected_open != resampled_open:
                        print(f"{start_time} の 'open' 値が一致しません。リサンプリング: {resampled_open}, 1時間足: {expected_open}")
                else:
                        print(f"{start_time} の 'open' 値は一致しています。")

                if expected_close != resampled_close:
                        print(f"{end_time} の 'close' 値が一致しません。リサンプリング: {resampled_close}, 1時間足: {expected_close}")
                else:
                        print(f"{end_time} の 'close' 値は一致しています。")

                if expected_high != resampled_high:
                        print(f"{start_time} の 'high' 値が一致しません。リサンプリング: {resampled_high}, 1時間足の最大値: {expected_high}")
                else:
                        print(f"{start_time} の 'high' 値は一致しています。")

                if expected_low != resampled_low:
                        print(f"{start_time} の 'low' 値が一致しません。リサンプリング: {resampled_low}, 1時間足の最小値: {expected_low}")
                else:
                        print(f"{start_time} の 'low' 値は一致しています。")


def main():
        db = MongoDataLoader()
        start_date = "2024-01-01 11:00:00"
        end_date = "2024-01-06 16:00:00"

        # Load original 1-hour data
        df_1h = db.load_data_from_datetime_period(
                start_date, end_date, coll_type=MARKET_DATA, symbol='BTCUSDT', interval=60)

        # Define timeframes to resample
        timeframes = {
                '12H': 12,
                '2H': 2
        }

        resampled_data = {}

        for tf, hours in timeframes.items():
                # Load existing resampled data from the database
                df_existing = db.load_data_from_datetime_period(
                        start_date, end_date, coll_type=MARKET_DATA, symbol='BTCUSDT', interval=hours * 60)

                # Resample the 1-hour data
                df_resampled = resample_df(df_1h, to_timeframe=tf, base_time='03:00:00')
                resampled_data[tf] = df_resampled

                # Validate the resampled data against existing data
                print(f"\n=== {tf} へのリサンプリングと検証 ===")
                validate_resampled_data(df_1h, df_resampled, timeframe_hours=hours)

                # Optionally, compare with existing data if needed
                # Here you can add code to compare df_resampled with df_existing if required

        # Display resampled data
        for tf, df_resampled in resampled_data.items():
                print(f"\nリサンプリングされたデータ ({tf}):")
                print(df_resampled)

        # Display original 1-hour data
        print("\n元の1時間足データ:")
        #print(df_1h)



if __name__ == "__main__":
        main()

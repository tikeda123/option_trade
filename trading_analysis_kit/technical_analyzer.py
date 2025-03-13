import pandas as pd
import numpy as np
import talib as ta
from typing import Optional
import os, sys
from talib import MA_Type

# Get the absolute path of the directory containing this file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the path of the parent directory to sys.path
sys.path.append(parent_dir)

from common.trading_logger_db import TradingLoggerDB
from common.config_manager import ConfigManager
from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *
from aiml.kalman_filter import apply_kalman_filter

class TechnicalAnalyzer:
        """
        This class performs technical analysis on financial data, such as stock or forex data.
        It calculates various technical indicators like RSI, Bollinger Bands, MACD, and DMI,
        adding the analysis results to the DataFrame.

        Attributes:
                __data_loader (MongoDataLoader): The data loader used to load data from the database.
                __config_manager (ConfigManager): The configuration manager used to access configuration settings.
                __data (pd.DataFrame): The DataFrame containing the financial data for analysis.
        """

        def __init__(self, df=None):
                """
                Initializes the TechnicalAnalyzer with an optional DataFrame.

                Args:
                        df (pd.DataFrame, optional): The DataFrame containing financial data. Defaults to None.
                """
                # Initialize the data loader for interacting with the database
                self.__data_loader = MongoDataLoader()
                # Initialize the configuration manager for accessing technical analysis settings
                self.__config_manager = ConfigManager()
                self.interval = int(self.__config_manager.get('TRADE_CONFIG', 'INTERVAL'))

                # If a DataFrame is provided, use it; otherwise, initialize an empty DataFrame
                if df is not None:
                        self.__data = df
                else:
                        self.__data = None

        def set_data(self, df):
                """
                Sets the DataFrame containing financial data for analysis.

                Args:
                        df (pd.DataFrame): The DataFrame containing financial data.
                """
                self.__data = df

        def get_data(self):
                """
                Returns the DataFrame containing financial data for analysis.

                Returns:
                        pd.DataFrame: The DataFrame containing financial data.
                """
                return self.__data

        def load_data_from_datetime_period(self, start_datetime, end_datetime, table_name) -> pd.DataFrame:
                """
                Loads data from the database for a specified period and table, updating the DataFrame.

                Args:
                        start_datetime (datetime): The starting datetime for data loading.
                        end_datetime (datetime): The ending datetime for data loading.
                        table_name (str): The name of the database table to load data from.

                Returns:
                        pd.DataFrame: The loaded DataFrame.
                """
                # Load data from the specified period and table using the data loader
                self.__data_loader.load_data_from_datetime_period(start_datetime, end_datetime, table_name)
                # Get the raw DataFrame from the data loader
                self.__data = self.__data_loader.get_df_raw()
                return self.__data

        def load_recent_data_from_db(self, coll_type: Optional[str] = None, nsteps: int = 100, symbol: Optional[str] = None,
                                     interval: Optional[int] = None) -> Optional[pd.DataFrame]:
                """
                Loads the most recent data from the database, updating the DataFrame.

                Args:
                        table_name (str, optional): The name of the database table. Defaults to None.

                Returns:
                        pd.DataFrame: The loaded DataFrame.
                """
                # Load recent data from the database using the data loader
                self.__data_loader.get_latest_n_records(coll_type, nsteps, symbol, interval)
                # Get the raw DataFrame from the data loader
                self.__data = self.__data_loader.get_df_raw()
                return self.__data

        def load_data_from_db(self, coll_type:str, symbol: Optional[str]=None, interval:Optional[int]=None) -> pd.DataFrame:
                """
                Loads data from the database for the specified table, updating the DataFrame.

                Args:
                        table_name (str, optional): The name of the database table. Defaults to None.

                Returns:
                        pd.DataFrame: The loaded DataFrame.
                """
                # Load market data from the database using the data loader
                self.__data_loader.load_data(coll_type, symbol, interval)
                # Get the raw DataFrame from the data loader
                self.__data = self.__data_loader.get_df_raw()
                return self.__data


        def insert_data(self, symbol: Optional[str] = None, interval: Optional[int] = None):
                """
                Imports the analysis results to the database.

                Args:
                        table_name (str, optional): The name of the database table to import data to. Defaults to None.
                """
                self.__data_loader.insert_data(self.__data, MARKET_DATA_TECH, symbol, interval)

        def analyze(self, df: Optional[ pd.DataFrame] = None, interval: Optional[int] = None) -> pd.DataFrame:
                """
                Performs technical analysis on the DataFrame, calculating various indicators.

                Returns:
                        pd.DataFrame: The DataFrame with added analysis results.
                """
                if df is not None:
                        self.__data = df

                if interval is not None:
                        self.interval = interval

                # Calculate Weighted Close Price (WCLPRICE)
                self.calculate_wclprice()
                # Calculate Relative Strength Index (RSI)
                self.calculate_rsi()
                # Calculate Bollinger Bands
                self.calculate_bollinger_bands()
                # Calculate Moving Average Convergence Divergence (MACD)
                self.calculate_macd()
                # Calculate Directional Movement Index (DMI)
                self.calculate_dmi()
                # Calculate Volume Moving Average
                self.calculate_volume_moving_average()
                # Calculate Differences between various indicators
                self.calculate_differences()
                # Calculate Simple Moving Average (SMA)
                self.calculate_sma()
                # Calculate Exponential Moving Average (EMA)
                self.calculate_ema()
                # Calculate Differences for time series analysis
                self.calculate_differences_for_time_series()
                # Calculate Aroon Oscillator
                self.calculate_aroon()
                # Calculate Accumulation/Distribution Line (AD Line)
                self.calculate_adline()
                # Calculate Money Flow Index (MFI)
                self.calculate_mfi()
                # Calculate Rate of Change (ROC)
                self.calculate_ROC()
                # Calculate Average True Range (ATR)
                self.calculate_ATR()
                # Calculate Volatility
                self.calculate_volatility(self.interval)
                # Calculate Kalman Filter on Close
                self.calculate_close_kalman()
                # Calculate Kalman Filter on MACD
                self.calculate_kalman_macd()
                # Finalize the analysis by handling any remaining data cleanup
                self.finalize_analysis()
                return self.__data

        def calculate_ATR(self):
                """
                Calculates the Average True Range (ATR) and adds it to the DataFrame.
                """
                # Get the ATR time period from the configuration manager
                timeperiod = int(self.__config_manager.get('TECHNICAL', 'ATR', 'TIMEPERIOD'))
                # Calculate ATR using the TA-Lib library
                self.__data[COLUMN_ATR] = ta.ATR(self.__data[COLUMN_HIGH], self.__data[COLUMN_LOW], self.__data[COLUMN_CLOSE], timeperiod)
                # Truncate the ATR values and add them to the DataFrame
                self._truncate_and_add_to_df(self.__data[COLUMN_ATR], COLUMN_ATR)

        def calculate_close_kalman(self):
                """
                Calculates the Kalman Filter on Close and adds it to the DataFrame.
                """
                # Calculate Kalman Filter on Close using the TA-Lib library
                self.__data[COLUMN_CLOSE_KALMAN] = apply_kalman_filter(self.__data, 'close', 1e-2, 1.0)
                # Truncate the Kalman Filter on Close values and add them to the DataFrame
                self._truncate_and_add_to_df(self.__data[COLUMN_CLOSE_KALMAN], COLUMN_CLOSE_KALMAN)

        def calculate_ROC(self):
                """
                Calculates the Rate of Change (ROC) and adds it to the DataFrame.
                """
                # Get the ROC time period from the configuration manager
                timeperiod = int(self.__config_manager.get('TECHNICAL', 'ROC', 'TIMEPERIOD'))
                # Calculate ROC using the TA-Lib library
                self.__data[COLUMN_ROC] = ta.ROC(self.__data[COLUMN_WCLPRICE], timeperiod)
                # Truncate the ROC values and add them to the DataFrame
                self._truncate_and_add_to_df(self.__data[COLUMN_ROC], COLUMN_ROC)

        def calculate_mfi(self):
                """
                Calculates the Money Flow Index (MFI) and adds it to the DataFrame.
                """
                # Get the MFI time period from the configuration manager
                timeperiod = int(self.__config_manager.get('TECHNICAL', 'MFI', 'TIMEPERIOD'))
                # Calculate MFI using the TA-Lib library
                self.__data[COLUMN_MFI] = ta.MFI(self.__data[COLUMN_HIGH], self.__data[COLUMN_LOW], self.__data[COLUMN_CLOSE], self.__data[COLUMN_VOLUME], timeperiod)
                # Truncate the MFI values and add them to the DataFrame
                self._truncate_and_add_to_df(self.__data[COLUMN_MFI], COLUMN_MFI)


        def calculate_volatility(self, interval: int = 1440):
                """
                interval: Candlestick interval in minutes (e.g., 1440=daily, 240=4h, 60=1h)
                         Assumes 365 trading days per year for bars_per_year = bars_per_day * 365
                """
                timeperiod_days = float(self.__config_manager.get('TECHNICAL', 'VOLATILITY', 'TIMEPERIOD'))
                # Using float here for flexibility, though int would also work

                # 2) Number of bars per day (divide total minutes in day by interval)
                bars_per_day = 1440 / interval  # e.g., interval=60 means 24 bars, 240 means 6 bars

                # 3) Number of bars for rolling calculation (specified days x bars per day)
                bars_for_rolling = int(timeperiod_days * bars_per_day)
                # e.g., timeperiod_days=30, interval=60(1h) → bars_for_rolling=30×24=720

                # 4) Number of bars per year (assuming 365 days) → annual_factor as √(bars_per_year)
                bars_per_year = bars_per_day * 365
                annual_factor = np.sqrt(bars_per_year)

                # 5) Calculate logarithmic returns
                self.__data['log_returns'] = np.log(
                        self.__data[COLUMN_WCLPRICE] / self.__data[COLUMN_WCLPRICE].shift(1)
                )

                # 6) Rolling standard deviation (over past bars_for_rolling bars)
                if bars_for_rolling < 2:
                        # If insufficient samples for rolling calculation, set to NaN
                        self.__data['volatility'] = np.nan
                else:
                        rolling_std = self.__data['log_returns'].rolling(bars_for_rolling).std()
                        # 7) Annualize (rolling_std × annual_factor)
                        self.__data['volatility'] = rolling_std * annual_factor

                # Round to 2 decimal places
                self._truncate_and_add_to_df(self.__data['volatility'], 'volatility')

        def calculate_wclprice(self):
                """
                Calculates the Weighted Close Price (WCLPRICE) and adds it to the DataFrame.
                """
                # Calculate WCLPRICE based on high, low, and close prices
                self.__data[COLUMN_WCLPRICE] = (self.__data[COLUMN_HIGH] + self.__data[COLUMN_LOW] + 2 * self.__data[COLUMN_CLOSE]) / 4
                # Truncate the WCLPRICE values and add them to the DataFrame
                self._truncate_and_add_to_df(self.__data[COLUMN_WCLPRICE], COLUMN_WCLPRICE)

        def finalize_analysis(self):
                """
                Finalizes the analysis by dropping rows with NaN values and resetting the DataFrame index.

                Returns:
                        pd.DataFrame: The finalized DataFrame.
                """
                # Drop rows with any NaN values
                self.__data.dropna(inplace=True)
                # Reset the index of the DataFrame
                self.__data.reset_index(drop=True, inplace=True)
                return self.__data

        def calculate_rsi(self):
                """
                Calculates the Relative Strength Index (RSI) and adds it to the DataFrame.
                """
                # Get the RSI time period from the configuration manager
                timeperiod = int(self.__config_manager.get('TECHNICAL', 'RSI', 'TIMEPERIOD'))
                # Calculate RSI using the TA-Lib library
                self.__data[COLUMN_RSI] = ta.RSI(self.__data[COLUMN_WCLPRICE], timeperiod)
                # Truncate the RSI values and add them to the DataFrame
                self._truncate_and_add_to_df(self.__data[COLUMN_RSI], COLUMN_RSI)

        def calculate_bollinger_bands(self):
                """
                Calculates Bollinger Bands and adds them to the DataFrame.

                This method calculates the upper, middle, and lower Bollinger Bands using the TA-Lib library.
                It calculates bands for multiple standard deviation multipliers (1, 2, and 3) and adds them
                as separate columns to the DataFrame.
                """
                # Get the Bollinger Band time period from the configuration manager
                timeperiod = int(self.__config_manager.get('TECHNICAL', 'BB', 'TIMEPERIOD'))

                # Calculate Bollinger Bands for each standard deviation multiplier
                for nbdev_multiplier in range(1, 4):
                        upper_band, middle_band, lower_band = ta.BBANDS(
                                self.__data[COLUMN_WCLPRICE],
                                timeperiod,
                                nbdevup=nbdev_multiplier,
                                nbdevdn=nbdev_multiplier,
                                matype=MA_Type.EMA
                        )
                        # Truncate the band values and add them to the DataFrame with appropriate column names
                        self._truncate_and_add_to_df(upper_band, f'{COLUMN_UPPER_BAND}{nbdev_multiplier}')
                        self._truncate_and_add_to_df(lower_band, f'{COLUMN_LOWER_BAND}{nbdev_multiplier}')
                        # Add the middle band only once (for nbdev_multiplier = 1)
                        if nbdev_multiplier == 1:
                                self._truncate_and_add_to_df(middle_band, COLUMN_MIDDLE_BAND)

                # Calculate Bollinger Band Width (BBVI) and add it to the DataFrame
                self.__data['bbvi'] = ((self.__data['upper2'] - self.__data['lower2']) / self.__data['middle']) * 100
                self._truncate_and_add_to_df(self.__data['bbvi'], COLUMN_BBVI)

        def calculate_macd(self):
                """
                Calculates the Moving Average Convergence Divergence (MACD) and adds it to the DataFrame.
                """
                # Get MACD parameters from the configuration manager
                fastperiod, slowperiod, signalperiod = int(self.__config_manager.get('TECHNICAL', 'MACD', 'FASTPERIOD')), int(self.__config_manager.get('TECHNICAL', 'MACD', 'SLOWPERIOD')), int(self.__config_manager.get('TECHNICAL', 'MACD', 'SIGNALPERIOD'))
                # Calculate MACD using the TA-Lib library
                macd, macdsignal, macdhist = ta.MACD(self.__data[COLUMN_WCLPRICE], fastperiod, slowperiod, signalperiod)
                # Truncate the MACD values and add them to the DataFrame
                self._truncate_and_add_to_df(macd, COLUMN_MACD)
                self._truncate_and_add_to_df(macdsignal, COLUMN_MACDSIGNAL)
                self._truncate_and_add_to_df(macdhist, COLUMN_MACDHIST)

        def calculate_kalman_macd(self):
                """
                Calculates the Kalman Filter for MACD and adds it to the DataFrame.
                """
                fastperiod, slowperiod, signalperiod = int(self.__config_manager.get('TECHNICAL', 'MACD', 'FASTPERIOD')), int(self.__config_manager.get('TECHNICAL', 'MACD', 'SLOWPERIOD')), int(self.__config_manager.get('TECHNICAL', 'MACD', 'SIGNALPERIOD'))

                macd, macdsignal, macdhist = ta.MACD(self.__data[COLUMN_CLOSE_KALMAN], fastperiod, slowperiod, signalperiod)
                self._truncate_and_add_to_df(macd, COLUMN_KALMAN_MACD)
                self._truncate_and_add_to_df(macdsignal, COLUMN_KALMAN_MACDSIGNAL)
                self._truncate_and_add_to_df(macdhist, COLUMN_KALMAN_MACDHIST)

        def calculate_dmi(self):
                """
                Calculates the Directional Movement Index (DMI) and adds it to the DataFrame.
                """
                # Get the DMI time period from the configuration manager
                timeperiod = int(self.__config_manager.get('TECHNICAL', 'DMI', 'TIMEPERIOD'))
                # Calculate DMI components (PLUS_DI, MINUS_DI, ADX, ADXR) using the TA-Lib library
                p_di = ta.PLUS_DI(self.__data[COLUMN_HIGH], self.__data[COLUMN_LOW], self.__data[COLUMN_CLOSE], timeperiod)
                m_di = ta.MINUS_DI(self.__data[COLUMN_HIGH], self.__data[COLUMN_LOW], self.__data[COLUMN_CLOSE], timeperiod)
                adx = ta.ADX(self.__data[COLUMN_HIGH], self.__data[COLUMN_LOW], self.__data[COLUMN_CLOSE], timeperiod)
                adxr = ta.ADXR(self.__data[COLUMN_HIGH], self.__data[COLUMN_LOW], self.__data[COLUMN_CLOSE], timeperiod)
                # Truncate the DMI values and add them to the DataFrame
                self._truncate_and_add_to_df(p_di, COLUMN_P_DI)
                self._truncate_and_add_to_df(m_di, COLUMN_M_DI)
                self._truncate_and_add_to_df(adx, COLUMN_ADX)
                self._truncate_and_add_to_df(adxr, COLUMN_ADXR)

        def calculate_aroon(self):
                """
                Calculates the Aroon Oscillator and adds it to the DataFrame.
                """
                # Get the Aroon time period from the configuration manager
                period = int(self.__config_manager.get('TECHNICAL', 'AROON', 'TIMEPERIOD'))
                # Calculate Aroon Oscillator using the TA-Lib library
                aroon = ta.AROONOSC(self.__data['high'], self.__data['low'], timeperiod=period)
                # Truncate the Aroon Oscillator values and add them to the DataFrame
                self._truncate_and_add_to_df(aroon, 'aroon')

        def calculate_adline(self):
                """
                Calculates the Accumulation/Distribution Line (AD Line) and adds it to the DataFrame.
                """
                # Calculate AD Line using the TA-Lib library
                ad = ta.AD(self.__data['high'], self.__data['low'], self.__data['close'], self.__data['volume'])
                # Truncate the AD Line values and add them to the DataFrame
                self._truncate_and_add_to_df(ad, COLUMN_ADLINE)

        def calculate_volume_moving_average(self):
                """
                Calculates the volume moving average and its difference, adding them to the DataFrame.
                """
                # Get the time period for volume moving average from the configuration manager
                timeperiod = int(self.__config_manager.get('TECHNICAL', 'VOLUME_MA', 'TIMEPERIOD'))
                # Calculate the volume moving average using a rolling window
                self.__data[COLUMN_VOLUME_MA] = self.__data[COLUMN_VOLUME].rolling(window=timeperiod).mean()
                # Calculate the difference of the volume moving average
                self.__data[COLUMN_VOLUME_MA_DIFF] = self.__data[COLUMN_VOLUME_MA].diff()
                # Truncate the difference values and add them to the DataFrame
                self._truncate_and_add_to_df(self.__data[COLUMN_VOLUME_MA_DIFF], COLUMN_VOLUME_MA_DIFF)

        def _truncate_and_add_to_df(self, series, column_name):
                """
                Truncates values in a series to two decimal places and adds them as a new column to the DataFrame.

                Args:
                        series (pd.Series): The series containing the values to truncate.
                        column_name (str): The name of the new column to be added to the DataFrame.
                """
                self.__data[column_name] = np.trunc(series * 100) / 100

        def calculate_differences_for_time_series(self):
                """
                Calculates various differences between technical indicators for time series analysis
                and adds them to the DataFrame.
                """
                # Calculate the difference between the upper Bollinger Band and WCLPRICE
                self.__data[COLUMN_UPPER_DIFF] = self.__data[COLUMN_UPPER_BAND2] - self.__data[COLUMN_WCLPRICE]
                # Calculate the difference between the lower Bollinger Band and WCLPRICE
                self.__data[COLUMN_LOWER_DIFF] = self.__data[COLUMN_LOWER_BAND2] - self.__data[COLUMN_WCLPRICE]
                # Calculate the difference between the middle Bollinger Band and WCLPRICE
                self.__data[COLUMN_MIDDLE_DIFF] = self.__data[COLUMN_MIDDLE_BAND] - self.__data[COLUMN_WCLPRICE]
                # Calculate the difference between the EMA and WCLPRICE
                self.__data[COLUMN_EMA_DIFF] = self.__data[COLUMN_EMA] - self.__data[COLUMN_WCLPRICE]
                # Calculate the difference between the SMA and WCLPRICE
                self.__data[COLUMN_SMA_DIFF] = self.__data[COLUMN_SMA] - self.__data[COLUMN_WCLPRICE]
                # Calculate the difference between the RSI and 70 (overbought level)
                self.__data[COLUMN_RSI_SELL] = self.__data[COLUMN_RSI] - 70
                # Calculate the difference between EMA and SMA
                self.__data['ema_sma'] = self.__data[COLUMN_EMA] - self.__data[COLUMN_SMA]
                # Calculate the difference between SMA and EMA
                self.__data['sma_ema'] = self.__data[COLUMN_SMA] - self.__data[COLUMN_EMA]
                # Create a column where values are equal to 'ema_sma' if 'ema_sma' is greater than 500, otherwise 0
                self.__data['ema_sma_500'] = np.where(self.__data['ema_sma'] > 500, self.__data['ema_sma'], 0)
                # Create a column where values are equal to 'sma_ema' if 'sma_ema' is greater than 500, otherwise 0
                self.__data['sma_ema_500'] = np.where(self.__data['sma_ema'] > 500, self.__data['sma_ema'], 0)

                # Calculate the difference between the RSI and 70 (overbought level)
                self.__data[COLUMN_RSI_SELL] = self.__data[COLUMN_RSI] - 70
                # Calculate the difference between the RSI and 30 (oversold level)
                self.__data[COLUMN_RSI_BUY] = self.__data[COLUMN_RSI] - 30
                # Calculate the difference between PLUS_DI and MINUS_DI (DMI components)
                self.__data[COLUMN_DMI_DIFF] = self.__data[COLUMN_P_DI] - self.__data[COLUMN_M_DI]
                # Calculate the difference between MACD and MACD Signal
                self.__data[COLUMN_MACD_DIFF] = self.__data[COLUMN_MACD] - self.__data[COLUMN_MACDSIGNAL]
                # Calculate the difference between the upper and lower Bollinger Bands
                self.__data[COLUMN_BOL_DIFF] = self.__data[COLUMN_UPPER_BAND2] - self.__data[COLUMN_LOWER_BAND2]

        def calculate_differences(self):
                """
                Calculates differences between Bollinger Bands, DMI, and other indicators
                over a specified period, adding them to the DataFrame.
                """
                # Get the difference period from the configuration manager
                difference = int(self.__config_manager.get('TECHNICAL', 'DIFFERENCE', 'TIMEPERIOD'))
                # Calculate the difference in the middle Bollinger Band over the specified period
                self.__data[COLUMN_MIDDLE_DIFF] = self.__data[COLUMN_MIDDLE_BAND] - self.__data[COLUMN_MIDDLE_BAND].shift(difference)
                # Calculate the difference in Bollinger Band width over the specified period
                self.__data[COLUMN_BAND_DIFF] = (self.__data[COLUMN_UPPER_BAND2] - self.__data[COLUMN_LOWER_BAND2]) - ((self.__data[COLUMN_UPPER_BAND2].shift(difference) - self.__data[COLUMN_LOWER_BAND2]).shift(difference))
                # Calculate the difference in DMI (PLUS_DI - MINUS_DI) over the specified period
                self.__data[COLUMN_DI_DIFF] = (self.__data[COLUMN_P_DI] - self.__data[COLUMN_M_DI]) - ((self.__data[COLUMN_P_DI].shift(difference) - self.__data[COLUMN_M_DI]).shift(difference))

                # Truncate the difference values and add them to the DataFrame
                self._truncate_and_add_to_df(self.__data[COLUMN_MIDDLE_DIFF], COLUMN_MIDDLE_DIFF)
                self._truncate_and_add_to_df(self.__data[COLUMN_BAND_DIFF], COLUMN_BAND_DIFF)
                self._truncate_and_add_to_df(self.__data[COLUMN_DI_DIFF], COLUMN_DI_DIFF)

        def calculate_sma(self):
                """
                Calculates the Simple Moving Average (SMA) and adds it to the DataFrame.
                """
                # Get the SMA time period from the configuration manager
                timeperiod = int(self.__config_manager.get('TECHNICAL', 'SMA', 'TIMEPERIOD'))
                # Calculate SMA using the TA-Lib library
                self.__data[COLUMN_SMA] = ta.SMA(self.__data[COLUMN_WCLPRICE], timeperiod)
                # Truncate the SMA values and add them to the DataFrame
                self._truncate_and_add_to_df(self.__data[COLUMN_SMA], COLUMN_SMA)

        def calculate_ema(self):
                """
                Calculates the Exponential Moving Average (EMA) and adds it to the DataFrame.
                """
                # Get the EMA time period from the configuration manager
                timeperiod = int(self.__config_manager.get('TECHNICAL', 'EMA', 'TIMEPERIOD'))
                # Calculate EMA using the TA-Lib library
                self.__data[COLUMN_EMA] = ta.EMA(self.__data[COLUMN_WCLPRICE], timeperiod)
                # Truncate the EMA values and add them to the DataFrame
                self._truncate_and_add_to_df(self.__data[COLUMN_EMA], COLUMN_EMA)

        def _truncate_and_add_to_df(self, series, column_name):
                """
                Truncates values in a series to two decimal places and adds them as a new column to the DataFrame.

                Args:
                        series (pd.Series): The series containing the values to truncate.
                        column_name (str): The name of the new column to be added to the DataFrame.
                """
                self.__data[column_name] = np.trunc(series * 100) / 100

        def get_data_loader(self):
                """
                Returns the data loader used by the TechnicalAnalyzer.

                Returns:
                        MongoDataLoader: The data loader.
                """
                return self.__data_loader

        def get_raw(self):
                """
                Returns the current DataFrame containing financial data and analysis results.

                Returns:
                        pd.DataFrame: The current DataFrame.
                """
                return self.__data

def calculate_ema(current_price, previous_ema, smoothing_factor=0.4):
    """
    Calculates the Exponential Moving Average (EMA) for a given price.

    Args:
        current_price (float): The current closing price.
        previous_ema (float): The EMA value from the previous period.
        smoothing_factor (float): The smoothing factor for EMA calculation. Defaults to 0.4.

    Returns:
        float: The newly calculated EMA value.

    This function calculates the EMA using the formula:
    EMA = (Current Price * Smoothing Factor) + (Previous EMA * (1 - Smoothing Factor))

    A smoothing factor of 0.4 corresponds to a 4-period EMA.
    """
    return (current_price * smoothing_factor) + (previous_ema * (1 - smoothing_factor))

def main():

        condition_list = [('BNBUSDT', 720), ('BNBUSDT', 240), ('BNBUSDT', 120), ('BNBUSDT', 60), ('BNBUSDT', 1440),('BNBUSDT', 10080),
                                       ('BTCUSDT', 720), ('BTCUSDT', 240), ('BTCUSDT', 120), ('BTCUSDT', 60), ('BTCUSDT', 1440),('BTCUSDT', 10080),
                                        ('ETHUSDT', 720), ('ETHUSDT', 240), ('ETHUSDT', 120), ('ETHUSDT', 60), ('ETHUSDT', 1440),('ETHUSDT', 10080),
                                        ('SOLUSDT', 720), ('SOLUSDT', 240), ('SOLUSDT', 120), ('SOLUSDT', 60), ('SOLUSDT', 1440),('SOLUSDT', 10080)]

        analyzer = TechnicalAnalyzer()
        for symbol, interval in condition_list:
                ts_collection = "market_data"
                print(f"Loading data for {symbol} {interval}")
                df = analyzer.load_data_from_db(ts_collection, symbol=symbol, interval=interval)
                if df is not None:
                        result = analyzer.analyze(df)
                        analyzer.insert_data(symbol=symbol, interval=interval)
                        print(f"Inserted data for {symbol} {interval}")
                        print(result)
                else:
                        print(f"No data found for {symbol} {interval}")

                #print(f"Dropping collection {ts_collection}")


if __name__ == '__main__':
        main()

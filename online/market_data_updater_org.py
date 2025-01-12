from typing import List, Tuple
import os
import sys
import time
import logging
import pandas as pd
from datetime import datetime

# Add the parent directory to the system path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from online.data_loader_online import DataLoaderOnline
from common.trading_logger_db import TradingLoggerDB
from common.constants import MARKET_DATA_TECH

# Create a logger instance
logger = TradingLoggerDB()

class MarketDataUpdater:
        """
        This class is responsible for updating market data at regular intervals.
        """
        def __init__(self, online_data_loader: DataLoaderOnline):
                """
                Initialize the MarketDataUpdater with a data loader.

                Args:
                        online_data_loader: An instance of DataLoaderOnline for fetching data.
                """
                self.online_data_loader = online_data_loader
                self.update_intervals = []

        def add_update_interval(self, symbol: str, interval: int):
                """
                Add a symbol and update interval to the updater.

                Args:
                        symbol: The trading symbol (e.g., 'BTCUSDT').
                        interval: The update interval in minutes.
                """
                self.update_intervals.append((symbol, interval))


        def update_data(self) -> List[Tuple[str, str]]:
                """
                Iterate through the symbols and intervals and update the data.
                Log messages for successful updates, skipped updates, and errors.

                Returns:
                        A list of tuples (symbol, interval) indicating which intervals were updated.
                """
                updated_intervals: List[Tuple[str, str]] = []
                for symbol, interval in self.update_intervals:
                        try:
                                df = self.online_data_loader.update_historical_data_if_needed(symbol=symbol, interval=interval)
                                if df is not None:
                                        logger.log_system_message(f"Updated data for {symbol} at {interval} minute interval.")
                                        updated_intervals.append((symbol, interval))
                                        # logger.log_system_message(f"No update needed for {symbol} at {interval} minute interval.")
                        except Exception as e:
                                logger.log_system_message(f"Error updating data for {symbol} at {interval} minute interval: {e}", logging.ERROR)
                return updated_intervals

        def is_newer(self, pdf1:pd.DataFrame, pdf2: pd.DataFrame) -> bool:
                """
                Compare two DataFrames and return True if the first DataFrame is newer than the second.

                Args:
                        pdf1: The first DataFrame.
                        pdf2: The second DataFrame.

                Returns:
                        bool: True if the first DataFrame is newer, False otherwise.
                """
                if pdf1.empty:
                        raise ValueError('pdf1 is empty')
                elif pdf2.empty:
                        raise ValueError('pdf2 is empty')

                time1 = pdf1['date'].iloc[-1]
                time2 = pdf2['date'].iloc[-1]
                return time1 > time2

def run_updater(updater: MarketDataUpdater, update_frequency: int = 5):
        """
        Run the market data updater in a loop with a specified frequency.

        Args:
                updater: An instance of MarketDataUpdater.
                update_frequency: The update frequency in seconds.
        """
        from trading_analysis_kit.simulation_strategy import SimulationStrategy
        from trading_analysis_kit.simulation_strategy_context import SimulationStrategyContext
        from trading_analysis_kit.online_bb_strategy import OnlineBollingerBandStrategy
        from mongodb.data_loader_mongo import MongoDataLoader

        mongodb  = MongoDataLoader()
        strategy_context = OnlineBollingerBandStrategy()
        context = SimulationStrategyContext(strategy_context)
        mongodb.get_latest_n_records(MARKET_DATA_TECH)
        context.load_recent_data()

        while True:
                updated_list = updater.update_data()
                interval_5_updated = any([interval == 5 for _, interval in updated_list])


                if interval_5_updated:
                        mongodb.get_latest_n_records(MARKET_DATA_TECH)
                        pdf_context = context.dm.get_latest_data()
                        pdf_db = mongodb.get_latest_data()

                        if updater.is_newer(pdf_db,pdf_context):
                                context.append_df(pdf_db.copy())
                                index = context.dm.get_latest_index()
                                context.event_handle(index)
                                context.update_roll_ai_data()
                                logger.log_verbose_message(pdf_db)

                time.sleep(update_frequency)

def main():
        """
        Initialize the data loader and updater, add update intervals, and start the updater.
        """
        online_data_loader = DataLoaderOnline()
        updater = MarketDataUpdater(online_data_loader)

        # Add update intervals for BTCUSDT
        for interval in [5, 15, 30, 60]:
                updater.add_update_interval('BTCUSDT', interval)

        # Run the updater every 5 seconds
        run_updater(updater, update_frequency=5)

if __name__ == "__main__":
        main()
from typing import List, Tuple
import os
import sys
import time
import logging
import pandas as pd
from datetime import datetime

# Add the parent directory to the system path from the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from online.data_loader_online import DataLoaderOnline
from common.trading_logger_db import TradingLoggerDB
from common.constants import MARKET_DATA_TECH
from common.utils import get_config

# Create an instance of the logger
logger = TradingLoggerDB()


def update_market_data(online_data_loader: DataLoaderOnline, symbol: str, interval: int) -> bool:
        """
        Updates market data for the specified symbol and interval.

        Args:
                online_data_loader: An instance of DataLoaderOnline for data retrieval.
                symbol: Trading symbol (e.g., 'BTCUSDT').
                interval: Update interval in minutes.

        Returns:
                bool: True if the update was successful, False otherwise.
        """
        try:
                df = online_data_loader.update_historical_data_if_needed(symbol=symbol, interval=interval)
                if df is not None:
                        logger.log_system_message(f"Updated data for {symbol} at {interval} minute interval.")
                        return True
        except Exception as e:
                logger.log_system_message(
                        f"Error updating data for {symbol} at {interval} minute interval: {e}", logging.ERROR
                )
        return False


def run_simulation(context, pdf_db):
        """
        Runs the trading strategy simulation.

        Args:
                context: An instance of SimulationStrategyContext.
                pdf_db: The latest data retrieved from the database.
        """
        context.append_df(pdf_db.copy())
        index = context.dm.get_latest_index()
        context.optimize_param(index)
        context.event_handle(index)
        context.update_roll_ai_data()
        logger.log_verbose_message(pdf_db)


def is_newer(pdf1: pd.DataFrame, pdf2: pd.DataFrame) -> bool:
        """
        Compares two DataFrames and checks if the first DataFrame is newer.

        Args:
                pdf1: The first DataFrame to compare.
                pdf2: The second DataFrame to compare.

        Returns:
                bool: True if the first DataFrame is newer, False otherwise.
        """
        if pdf1.empty:
                raise ValueError("pdf1 is empty.")
        elif pdf2.empty:
                raise ValueError("pdf2 is empty.")

        time1 = pdf1["date"].iloc[-1]
        time2 = pdf2["date"].iloc[-1]
        return time1 > time2


def run_market_data_updater(
        online_data_loader: DataLoaderOnline, update_intervals: List[Tuple[str, int]], main_interval,update_frequency: int = 5
):
        """
        Manages the updating of market data at the specified frequency.

        Args:
                online_data_loader: An instance of DataLoaderOnline for data retrieval.
                update_intervals: A list of update intervals and symbols.
                update_frequency: Update frequency in seconds.
        """
        from trading_analysis_kit.simulation_strategy_context import SimulationStrategyContext
        from trading_analysis_kit.online_bb_strategy import OnlineBollingerBandStrategy
        from mongodb.data_loader_mongo import MongoDataLoader

        mongodb = MongoDataLoader()
        strategy_context = OnlineBollingerBandStrategy()
        context = SimulationStrategyContext(strategy_context)
        mongodb.get_latest_n_records(MARKET_DATA_TECH)
        context.load_recent_data()

        while True:
                for symbol, interval in update_intervals:
                        updated = update_market_data(online_data_loader, symbol, interval)
                        if updated and interval == main_interval:
                                mongodb.get_latest_n_records(MARKET_DATA_TECH)
                                pdf_context = context.dm.get_latest_data()
                                pdf_db = mongodb.get_latest_data()

                                # Check if the data is newer
                                if is_newer(pdf_db, pdf_context):
                                        run_simulation(context, pdf_db)
                time.sleep(update_frequency)

from trade_config import trade_config

def online_main():
        """
        Initializes the data loader and updater, adds update intervals, and starts the updater.
        """
        symbol = trade_config.symbol
        main_interval = trade_config.interval

        online_data_loader = DataLoaderOnline()

        update_intervals = [
                (symbol, 30),
                (symbol, 60),
                (symbol, 120),
                (symbol, 240),
                (symbol, 720),
                (symbol, 1440)
        ]

        # Run the updater every 5 seconds
        run_market_data_updater(online_data_loader, update_intervals, main_interval,update_frequency=5)

if __name__ == "__main__":
        online_main()

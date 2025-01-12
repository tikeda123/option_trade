import os
import sys
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Get the absolute path of the directory containing this file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the path of the parent directory to sys.path
sys.path.append(parent_dir)

from trading_analysis_kit.trading_state import *
from aiml.prediction_manager import PredictionManager
from common.utils import get_config_model

class EntryStrategySIngleModel:
        """
        Implements an entry strategy based on Bollinger Bands and multiple trend prediction models.

        Attributes:
                MAX_MANAGERS (int): The maximum number of prediction models to use.
                model_trackers (dict): A dictionary storing performance trackers for each model.
                manager (dict): A dictionary storing PredictionManager instances for each model.
                model_weights (np.ndarray): An array of weights assigned to each model based on performance.
                top_model_num (int): The number of top performing models to use for the final prediction.
        """
        def __init__(self, top_model_num=1, profit_weight=0.3, hit_rate_weight=0.4, error_rate_weight=0.3):
                """
                Initializes the EntryStrategy by initializing and loading prediction models.
                """
                self.init_model()
                self.load_model()

        def init_model(self):
                """
                Initializes prediction managers for each model variant.
                """
                self.manager  = PredictionManager()
                config = get_config_model("MODEL_SHORT_TERM", "lstm_v6")  # Get the model configuration
                self.manager.initialize_model("lstm_v6", config)

        def load_model(self):
                """
                Loads pre-trained models for each variant.
                """
                self.manager.load_model()

        def update_performance(self, context, entry_index: int):
                """
                Updates the performance of each model and recalculates model weights.

                Args:
                        context: The trading context containing market data.
                        entry_index (int): The index of the current data point.
                """
                pass

        def get_strategy_performance(self):
                """
                Calculates the overall performance of the entry strategy.

                Returns:
                        dict: A dictionary containing the total profit, overall hit rate, and profit factor.
                """
                return {
                        "total_profit": 0,
                        "overall_hit_rate": 0,
                        "profit_factor": 0
                }

        def should_entry(self, context) -> bool:
                """
                Determines whether to enter a trade based on current market conditions.

                Args:
                        context: The trading context containing market data.

                Returns:
                        bool: True if entry is recommended, False otherwise.
                """
                # Placeholder for more complex entry logic
                return False

        def should_exit(self, context) -> bool:
                """
                Determines whether to exit a trade based on current market conditions.

                Args:
                        context: The trading context containing market data.

                Returns:
                        bool: True if exit is recommended, False otherwise.
                """
                # Placeholder for more complex exit logic
                return True

        def trend_prediction(self, context):
                """
                Predicts the market trend using an ensemble of the top performing models.

                Args:
                        context: The trading context containing market data.

                Returns:
                        tuple: A tuple containing:
                                - bool: True if a valid prediction is made, False otherwise.
                                - int: The predicted trend (1 for upward, 0 for downward).
                """

                # Get historical data for trend prediction
                current_index = context.dm.get_current_index()
                df = context.dm.get_df_fromto(current_index - (TIME_SERIES_PERIOD - 1), current_index)

                # Generate predictions from all models

                target_df = self.manager.create_time_series_data(df)
                rolling_pred = self.manager.predict_model(target_df, probability=True)

                set_pred_method = getattr(context.dm, f'set_pred_v1')
                set_pred_method(rolling_pred)
                pred = 1 if rolling_pred > 0.5 else 0
                context.log_transaction(f"Pred_b {rolling_pred} ,{pred}")
                return True, pred
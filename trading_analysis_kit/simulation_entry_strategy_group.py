import os
import sys
from collections import deque
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd

# Get the absolute path of the directory containing this file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the path of the parent directory to sys.path
sys.path.append(parent_dir)

from trading_analysis_kit.trading_state import *
from trading_analysis_kit.model_group_manager import ModelGroupManager
from aiml.prediction_manager import PredictionManager

class EntryStrategyGroup:
        """
        Implements an entry strategy based on Bollinger Bands and multiple trend prediction models.

        Attributes:
                short_term_model (ModelGroupManager): Manages short-term prediction models.
                middle_term_model (ModelGroupManager): Manages mid-term prediction models.
                long_term_model (ModelGroupManager): Manages long-term prediction models.
        """

        def __init__(self, top_model_num=4, profit_weight=0.3, hit_rate_weight=0.4, error_rate_weight=0.3):
                """
                Initializes the EntryStrategy by initializing and loading prediction models.
                """
                model_params = {
                        "MODEL_SHORT_TERM": {
                                "top_model_num": 2,
                                "profit_weight": 0.6,
                                "hit_rate_weight": 0.2,
                                "error_rate_weight": 0.2,
                        },
                        "MODEL_MIDDLE_TERM": {
                                "top_model_num": 2,
                                "profit_weight": 0.6,
                                "hit_rate_weight": 0.2,
                                "error_rate_weight": 0.2,
                        },
                        "MODEL_LONG_TERM": {
                                "top_model_num": 2,
                                "profit_weight": 0.6,
                                "hit_rate_weight": 0.2,
                                "error_rate_weight": 0.2,
                        }
                }

                self.short_term_model = self._create_model_group_manager("MODEL_SHORT_TERM", **model_params["MODEL_SHORT_TERM"])
                self.middle_term_model = self._create_model_group_manager("MODEL_MIDDLE_TERM", **model_params["MODEL_MIDDLE_TERM"])
        #        self.long_term_model = self._create_model_group_manager("MODEL_LONG_TERM", **model_params["MODEL_LONG_TERM"])

        def _create_model_group_manager(self, model_group: str, **kwargs) -> ModelGroupManager:
                """
                Helper method to create an instance of ModelGroupManager.

                Args:
                        model_group:str: The name of the model group.

                Returns:
                        ModelGroupManager: An instance of ModelGroupManager.
                """
                return ModelGroupManager(model_group, **kwargs)

        def update_performance(self, context, entry_index: int):
                """
                Updates the performance of each model and recalculates model weights.

                Args:
                                context: The trading context containing market data.
                                entry_index (int): The index of the current data point.
                """
                self.short_term_model.update_performance(context, entry_index)
                self.middle_term_model.update_performance(context, entry_index)
                #self.long_term_model.update_performance(context, entry_index)

        def get_strategy_performance(self) -> Tuple:
                """
                Calculates the overall performance of the entry strategy.

                Returns:
                        Tuple: A tuple containing the performance dictionaries
                                   for short, middle, and long term models.
                """
                return (
                        self.short_term_model.get_strategy_performance(),
                       self.middle_term_model.get_strategy_performance(),
                #        self.long_term_model.get_strategy_performance(),
                )

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
        def trend_prediction(self, context) -> Tuple[bool, int]:
                """
                Predicts the market trend using an ensemble of the top performing models.

                Args:
                        context: The trading context containing market data.

                Returns:
                        tuple: A tuple containing:
                                 - bool: True if a valid prediction is made, False otherwise.
                                 - int: The predicted trend (1 for upward, 0 for downward).
                """
                from trade_config import trade_config
                short_flag, short_trend = self.short_term_model.trend_prediction(context)
                mid_flag, mid_trend = self.middle_term_model.trend_prediction(context)
                print(f"mid_tred: {mid_trend}, short_trend: {short_trend}")
                #long_flag, long_trend = self.long_term_model.trend_prediction(context)

                # Define weights for short-term and mid-term trends
                short_weight = trade_config.short_model_ratio
                mid_weight = trade_config.middle_model_ratio
                long_weight = trade_config.long_model_ratio
                # Calculate weighted average of trends
                #weighted_trend = (short_trend * short_weight) + (mid_trend * mid_weight)+ (long_trend * long_weight)
                weighted_trend = (short_trend * short_weight) + (mid_trend * mid_weight)

                                # Determine final prediction
                #trend = 1 if weighted_trend > 0.5 else 0
                trend = short_trend

                return True, trend
                """
                # Determine final prediction
                if weighted_trend >  0.5:
                        return True, 1  # Upward trend
                else:
                        return True, 0  # Downward trend
                """
                """

                if bb_direction == BB_DIRECTION_UPPER:
                        if short_trend == 0:
                                return short_flag, short_trend
                        else:  # short_trend == 1
                                if mid_trend == 1:
                                        return mid_flag, mid_trend
                                else:  # mid_trend == 1
                                        return  False, mid_trend


                else:  # bb_direction == BB_DIRECTION_LOWER
                        if short_trend == 1:
                                return short_flag, short_trend
                        else:  # short_trend == 0
                                if mid_trend == 0:
                                        return mid_flag, mid_trend
                                else:
                                        return False, mid_trend

                if bb_direction == BB_DIRECTION_UPPER:
                        if short_trend == 0:
                                return short_flag, short_trend
                        else:  # short_trend == 1
                                return (long_flag, long_trend) if long_trend == 1 else (mid_flag, mid_trend)

                else:  # bb_direction == BB_DIRECTION_LOWER
                        if short_trend == 1:
                                return short_flag, short_trend
                        else:  # short_trend == 0
                                return (long_flag, long_trend) if long_trend == 0 else (mid_flag, mid_trend)
                                """

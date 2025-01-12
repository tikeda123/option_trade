import os
import sys
from collections import deque
from typing import Dict, Any

import numpy as np

# Get the absolute path of the directory containing this file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the path of the parent directory to sys.path
sys.path.append(parent_dir)

from common.utils import get_config, get_config_model
from common.constants import CONFIG_FILENAME, TIME_SERIES_PERIOD,MARKET_DATA_TECH
from trading_analysis_kit.trading_state import *
from trading_analysis_kit.simulation_entry_strategy import ModelPerformanceTracker, profit_calculator
from aiml.prediction_manager import PredictionManager
from aiml.tracing_aiml_info import TracingAimlInfo
from mongodb.data_loader_mongo import  MongoDataLoader
from aiml.aiml_comm import get_aligned_lower_timeframe_timestamp


def get_model_vlist(term_type: str) -> list:
        """
        Gets the list of model identifiers for a given term type.

        Args:
                term_type (str): The term type.

        Returns:
                list: A list of model identifiers.
        """

        type_vid = ("rolling_v","lstm_v")
        config = get_config_model(term_type)
        # tagがrolling_vまたはlstm_vで始まるものを取得
        return [tag for tag in config.keys() if tag.startswith(type_vid)]


class ModelGroupManager:
        """
        Manages a group of prediction models, tracks their performance,
        calculates weighted evaluations, and generates predictions.
        """

        def __init__(
                self,
                model_group: str,
                top_model_num=1,
                profit_weight=0.8,
                hit_rate_weight=0.1,
                error_rate_weight=0.1,
        ):
                """
                Initializes the ModelGroupManager.

                Args:
                        model_group (str): The name of the model group.
                        top_model_num (int): Number of top models to use for prediction.
                        profit_weight (float): Weight for profit in model evaluation.
                        hit_rate_weight (float): Weight for hit rate in model evaluation.
                        error_rate_weight (float): Weight for error rate in model evaluation.
                """
                from trade_config import TradeConfig
                trade_config = TradeConfig()
                self.symbol = trade_config.symbol
                self.interval= trade_config.interval
                self.model_group = model_group
                self.config = get_config_model(model_group)
                self.interval_gp = self.config["INTERVAL_GP"]
                self.manager_list = get_model_vlist(model_group)
                self.max_managers = len(self.manager_list)
                self.model_trackers = {
                        id_name: ModelPerformanceTracker() for id_name in self.manager_list
                }
                self.managers = {}
                self.model_weights = None
                self.top_model_num = top_model_num

                # Weighting parameters
                self.profit_weight = profit_weight
                self.hit_rate_weight = hit_rate_weight
                self.error_rate_weight = error_rate_weight

                self.tracing_aiml_info = TracingAimlInfo()
                self.tracing_aiml_data = self.tracing_aiml_info.create_dict(self.manager_list)
                self.data_loader = MongoDataLoader()
                self.init_models()
                self.load_models()

        def init_models(self):
                """Initializes PredictionManager for each model."""
                for id_name in self.manager_list:
                        self.managers[id_name] = PredictionManager()
                        config = self.config[id_name]
                        self.managers[id_name].initialize_model( id_name, config)

        def load_models(self):
                """Loads pre-trained models for each model."""
                for id_name in self.manager_list:
                        self.managers[id_name].load_model()

        def process_single_model_performance(
                self, context, entry_index, id_name, act, pandl
        ):
                """
                Processes performance metrics for a single model.

                Args:
                        context: The trading context.
                        entry_index: The entry index.
                        id_name: The model identifier.
                        act: Actual value.
                        pandl: Profit and Loss.

                Returns:
                        Tuple containing hit rate, average error, and average profit.
                """
                list_id = self.config[id_name]["LIST_ID"]
                get_pred_method = getattr(context.dm, f"get_pred_v{list_id}")
                model_pred = get_pred_method(entry_index)

                pred = 1 if model_pred > 0.5 else 0
                profit = profit_calculator(pred, act, pandl)
                self.model_trackers[id_name].update(pred, act, profit)

                # Get metrics
                hit_rate = self.model_trackers[id_name].get_hit_rate()
                avg_error = self.model_trackers[id_name].get_average_error()
                avg_profit = self.model_trackers[id_name].get_average_profit()

                # Update tracing_aiml_data
                self.tracing_aiml_data[f"pred_{id_name}"] = model_pred
                self.tracing_aiml_data[f"hit_rate_{id_name}"] = round(hit_rate, 4)
                self.tracing_aiml_data[f"avg_error_{id_name}"] = round(avg_error, 4)
                self.tracing_aiml_data[f"avg_profit_{id_name}"] = round(avg_profit, 4)

                # Log individual model performance
                context.log_transaction(
                        f"{id_name} - Hit Rate: {hit_rate:.4f}, Avg Error: {avg_error:.4f}, Avg Profit: {avg_profit:.4f}"
                )

                return hit_rate, avg_error, avg_profit

        def calculate_metric_weights(self, metrics, epsilon):
                """
                Calculates weights for a given metric.

                Args:
                        metrics: List of metric values.
                        epsilon: Small value to prevent division by zero.

                Returns:
                        Normalized weights as a numpy array.
                """
                total_metric = sum(metrics)
                if total_metric > epsilon:
                        weights = np.array(metrics) / total_metric
                else:
                        weights = np.ones(self.max_managers) / self.max_managers
                return weights

        def calculate_inverted_metric_weights(self, metrics, epsilon):
                """
                Calculates inverted weights for error metrics.

                Args:
                        metrics: List of error metric values.
                        epsilon: Small value to prevent division by zero.

                Returns:
                        Normalized inverted weights as a numpy array.
                """
                total_metric = sum(metrics)
                if total_metric > epsilon:
                        weights = np.max(metrics) - np.array(metrics)  # Invert error weights
                        weights /= np.sum(weights)
                else:
                        weights = np.ones(self.max_managers) / self.max_managers
                return weights

        def update_performance(self, context, entry_index: int):
                """
                Updates the performance of each model and calculates weighting parameters.

                Args:
                        context (TradingContext): The trading context.
                        entry_index (int): The entry index.
                """
                start_at = context.dm.get_current_date(entry_index)
                pred_target = context.dm.get_pred_target(entry_index)
                self.tracing_aiml_data = self.tracing_aiml_info.create_dict(self.manager_list)
                self.tracing_aiml_info.new_record(start_at, self.manager_list)

                # Get the actual value, profit/loss, etc. for the current entry
                act = context.dm.get_pred_target(entry_index)
                pandl = context.dm.get_pandl()

                self.tracing_aiml_data.update({"start_at": start_at})
                self.tracing_aiml_data.update({"pred": pred_target})
                self.tracing_aiml_data.update({"actual": act})
                self.tracing_aiml_data.update({"profit": pandl})

                hit_rates = []
                avg_errors = []
                avg_profits = []

                # Update performance metrics for each model
                for id_name in self.manager_list:
                        hit_rate, avg_error, avg_profit = self.process_single_model_performance(
                                context, entry_index, id_name, act, pandl
                        )
                        hit_rates.append(hit_rate)
                        avg_errors.append(avg_error)
                        avg_profits.append(avg_profit)

                # --- Calculate Weighting Parameters ---
                self.tracing_aiml_info.update_record_by_group(start_at, self.tracing_aiml_data)

                # Debug information: Log raw performance metrics for each model
                context.log_transaction(f"Raw hit rates: {hit_rates}")
                context.log_transaction(f"Raw average errors: {avg_errors}")
                context.log_transaction(f"Raw average profits: {avg_profits}")

                # Small value to prevent division by zero
                epsilon = 1e-10

                # Calculate weights based on profit, hit rate, and error rate.
                profit_weights = self.calculate_metric_weights(avg_profits, epsilon)
                hit_rate_weights = self.calculate_metric_weights(hit_rates, epsilon)
                error_weights = self.calculate_inverted_metric_weights(avg_errors, epsilon)

                # Apply the weighting parameters
                profit_weights *= self.profit_weight
                hit_rate_weights *= self.hit_rate_weight
                error_weights *= self.error_rate_weight

                # Calculate the final model weights
                self.model_weights = profit_weights + hit_rate_weights + error_weights
                self.model_weights /= np.sum(self.model_weights)  # Normalize

                # --- Debug Information: Log the final weight of each model ---
                context.log_transaction(f"Profit weights: {profit_weights}")
                context.log_transaction(f"Hit rate weights: {hit_rate_weights}")
                context.log_transaction(f"Error weights: {error_weights}")

                for i, weight in enumerate(self.model_weights):
                        context.log_transaction(f"Model {i+1} final weight: {weight:.4f}")

        def get_strategy_performance(self) -> dict:
                """
                Calculates the overall performance of the entry strategy.

                Returns:
                        dict: A dictionary containing the total profit,
                        overall hit rate, and profit factor.
                """
                total_profit = sum(
                        tracker.get_average_profit() for tracker in self.model_trackers.values()
                )
                overall_hit_rate = (
                        sum(tracker.get_hit_rate() for tracker in self.model_trackers.values())
                        / len(self.model_trackers)
                )
                profit_factor = (
                        sum(tracker.get_profit_factor() for tracker in self.model_trackers.values())
                        / len(self.model_trackers)
                )

                return {
                        "total_profit": total_profit,
                        "overall_hit_rate": overall_hit_rate,
                        "profit_factor": profit_factor,
                }

        def select_top_models(self, context) -> list:
                """
                Selects the top performing models. Selects all models if in the initial state.

                Args:
                        context (TradingContext): The trading context.

                Returns:
                        list: A list of indices of the selected models
                        (e.g., [0, 1, 2] for the first three models).
                """
                model_scores = []
                for i, id_name in enumerate(self.manager_list):
                        profit = self.model_trackers[id_name].get_average_profit()
                        hit_rate = self.model_trackers[id_name].get_hit_rate()
                        error_rate = self.model_trackers[id_name].get_average_error()

                        # Calculate model score
                        score = (
                                profit * self.profit_weight
                                + hit_rate * self.hit_rate_weight
                                + (1 - error_rate) * self.error_rate_weight
                        )
                        model_scores.append((i, score))

                # Sort models by score
                sorted_models = sorted(model_scores, key=lambda x: x[1], reverse=True)

                # Select all models if in initial state, otherwise select top models
                id_name = self.manager_list[0]
                if len(self.model_trackers[id_name].profits) == 0:
                        top_model_indices = list(range(self.max_managers))
                        context.log_transaction("Initial State: Selecting ALL models.")
                else:
                        top_model_indices = [
                                model_index for model_index, _ in sorted_models[: self.top_model_num]
                        ]

                        context.log_transaction(
                                f"Normal State: Selecting top {self.top_model_num} models."
                        )

                context.log_transaction(f"Selected model indices: {top_model_indices}")
                return top_model_indices

        def generate_and_update_single_model_prediction(self, context, df, id_name):
                """
                Generates and updates the prediction for a single model.

                Args:
                        context: The trading context.
                        df: The dataframe containing historical data.
                        id_name: The model identifier.

                Returns:
                        The prediction made by the model.
                """
                # Generate prediction
                target_df = self.managers[id_name].create_time_series_data(df)
                prediction = self.managers[id_name].predict_model(
                        target_df, probability=True
                )

                # Update predictions in DataManager
                list_id = self.config[id_name]["LIST_ID"]
                set_pred_method = getattr(context.dm, f"set_pred_v{list_id}")
                set_pred_method(prediction)

                return prediction

        def trend_prediction(self, context) -> tuple:
                """
                Predicts the market trend using an ensemble of the top performing models.

                Args:
                        context: The trading context containing market data.

                Returns:
                tuple: A tuple containing:
                        - bool: True if a valid prediction is made, False otherwise.
                        - int: The predicted trend (1 for upward, 0 for downward).
                """

                if self.model_weights is None:
                        self.model_weights = np.array([1 / self.max_managers] * self.max_managers)

                # Get historical data
                current_index = context.dm.get_current_index()
                date = context.dm.get_current_date(current_index)

                if self.interval_gp != self.interval:
                        mod_date = get_aligned_lower_timeframe_timestamp(date,self.interval_gp,self.interval)
                        interval_aligned = self.interval_gp
                        print(f"mod_date:{mod_date},date:{date}")
                else:
                        interval_aligned = self.interval
                        mod_date = date



                df = self.data_loader.load_data_from_point_date(mod_date,TIME_SERIES_PERIOD,MARKET_DATA_TECH,self.symbol, interval_aligned)
                #print(df)

                if  df['date'].isna().any():
                        print(f'************************************************************************************')
                        print(f"current_index:{current_index}")
                        print(df)
                        exit(0)

                # Generate predictions from all models
                rolling_pred = {}
                for i, id_name in enumerate(self.manager_list):
                        prediction = self.generate_and_update_single_model_prediction(
                                context, df, id_name
                        )
                        rolling_pred[i] = prediction

                # Select top models and calculate weighted average prediction
                top_model_indices = self.select_top_models(context)
                combined_pred = np.mean([rolling_pred[i] for i in top_model_indices])
                #prediction = 1 if combined_pred > 0.5 else 0

                #context.log_transaction(f"Pred_b {prediction}  {combined_pred}")
                return True, combined_pred

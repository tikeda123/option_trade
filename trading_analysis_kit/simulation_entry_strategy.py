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

from collections import deque
import matplotlib.pyplot as plt

class ModelPerformanceTracker:
    """
    Tracks the performance of a prediction model using a sliding window with dynamic window size.

    Attributes:
        window_size (int): The size of the sliding window for performance calculation.
        hits (deque): A deque storing binary hit/miss values for predictions within the window.
        total_predictions (deque): A deque storing the count of predictions made within the window.
        cumulative_error (deque): A deque storing the cumulative prediction error within the window.
        profits (deque): A deque storing the cumulative profits/losses within the window.
        min_window_size (int): Minimum allowed window size.
        max_window_size (int): Maximum allowed window size.
        performance_threshold (float): Threshold for standard deviation to adjust window size.
    """

    def __init__(self, window_size=60, min_window_size=30, max_window_size=100, performance_threshold=0.05):
        self.window_size = window_size
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.performance_threshold = performance_threshold  # 標準偏差の閾値
        self.hits = deque(maxlen=self.window_size)
        self.total_predictions = deque(maxlen=self.window_size)
        self.cumulative_error = deque(maxlen=self.window_size)
        self.profits = deque(maxlen=self.window_size)

    def update_deques_maxlen(self, new_window_size):
        """
        ウィンドウサイズを変更し、既存のデータを新しいサイズに合わせて保持します。

        Args:
            new_window_size (int): 新しいウィンドウサイズ。
        """
        # 新しいデックを作成
        new_hits = deque(self.hits, maxlen=new_window_size)
        new_total_predictions = deque(self.total_predictions, maxlen=new_window_size)
        new_cumulative_error = deque(self.cumulative_error, maxlen=new_window_size)
        new_profits = deque(self.profits, maxlen=new_window_size)

        # 現在のウィンドウサイズを更新
        self.window_size = new_window_size

        # デックを置き換え
        self.hits = new_hits
        self.total_predictions = new_total_predictions
        self.cumulative_error = new_cumulative_error
        self.profits = new_profits

        #logging.info(f"ウィンドウサイズを {new_window_size} に変更しました。")

    def adjust_window_size(self):
        """
        パフォーマンスの安定性に基づいてウィンドウサイズを動的に調整します。
        """
        if len(self.hits) < self.min_window_size:
            # データが少なすぎる場合は調整しない
            return

        current_hit_rate = self.get_hit_rate()
        # ヒット率の移動標準偏差を計算
        # ここではシンプルにヒット率自体の標準偏差を計算
        # より高度な方法として、ヒストリカルなヒット率を保持し計算することも可能
        # ここでは、単純化のために移動平均と現在のヒット率との差を使用
        moving_avg = sum(self.hits) / len(self.hits)
        variance = sum((hit - moving_avg) ** 2 for hit in self.hits) / len(self.hits)
        std_dev = variance ** 0.5

        #logging.debug(f"ヒット率: {current_hit_rate:.4f}, 標準偏差: {std_dev:.4f}")

        if std_dev < self.performance_threshold and self.window_size < self.max_window_size:
            # パフォーマンスが安定している場合、ウィンドウサイズを増加
            new_window_size = min(self.window_size + 10, self.max_window_size)
            self.update_deques_maxlen(new_window_size)
        elif std_dev > self.performance_threshold and self.window_size > self.min_window_size:
            # パフォーマンスが不安定な場合、ウィンドウサイズを減少
            new_window_size = max(self.window_size - 10, self.min_window_size)
            self.update_deques_maxlen(new_window_size)

    def update(self, prediction, actual, profit):
        """
        Updates the performance tracker with the latest prediction and actual value.

        Args:
            prediction: The predicted value.
            actual: The actual value.
            profit: The profit or loss value.
        """
        self.hits.append(int(prediction == actual))
        self.total_predictions.append(1)
        self.cumulative_error.append(abs(prediction - actual))
        self.profits.append(profit)

        # ウィンドウサイズの調整を試みる
        self.adjust_window_size()

    def get_hit_rate(self):
        """
        Calculates the hit rate (accuracy) within the current window.

        Returns:
            float: The hit rate, or 0 if no predictions have been made.
        """
        return sum(self.hits) / sum(self.total_predictions) if self.total_predictions else 0

    def get_average_error(self):
        """
        Calculates the average prediction error within the current window.

        Returns:
            float: The average error, or 1 (representing worst case) if no predictions have been made.
        """
        return sum(self.cumulative_error) / len(self.cumulative_error) if self.cumulative_error else 1

    def get_average_profit(self):
        """
        Calculates the average profit within the current window.

        Returns:
            float: The average profit, or 0 if no profits have been recorded.
        """
        return sum(self.profits) / len(self.profits) if self.profits else 0

    def get_profit_factor(self):
        """
        Calculates the profit factor within the current window.

        Returns:
            float: The profit factor, or infinity if no profits or losses have been recorded.
        """
        profits = sum(p for p in self.profits if p > 0)
        losses = abs(sum(p for p in self.profits if p < 0))
        return profits / losses if losses != 0 else float('inf')

    def plot_performance(self):
        """
        Plots the performance metrics over time.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.hits, label='Hits')
        plt.plot(self.total_predictions, label='Total Predictions')
        plt.plot(self.cumulative_error, label='Cumulative Error')
        plt.plot(self.profits, label='Profits')
        plt.legend()
        plt.title('Model Performance Over Time')
        plt.xlabel('Time')
        plt.ylabel('Metrics')
        plt.show()

def profit_calculator(pred, act, pandl) -> float:
    """
    Calculates the profit or loss based on the prediction and actual value.

    Args:
        pred (int): The predicted value (1 for positive, 0 for negative).
        act (int): The actual value (1 for positive, 0 for negative).
        pandl (float): The profit or loss value.

    Returns:
        float: The absolute profit if the prediction is correct, or the loss value otherwise.
    """
    if pred == act:
        abs_pandl = abs(pandl)  # 絶対値の利益を計算
        return abs_pandl
    else:
        return -1 * abs(pandl)  # 損失の場合はマイナスを返す


class EntryStrategy:
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
                self.MAX_MANAGERS = 1
                self.model_trackers = {f"model_{i}": ModelPerformanceTracker() for i in range(1, self.MAX_MANAGERS + 1)}
                self.manager = {}
                self.model_weights = None  # Initialize model weights to None
                self.top_model_num = top_model_num  # Use user-specified number of top models

                # 重み付けパラメータの初期化
                self.profit_weight = profit_weight
                self.hit_rate_weight = hit_rate_weight
                self.error_rate_weight = error_rate_weight

                self.init_model()
                self.load_model()

        def init_model(self):
                """
                Initializes prediction managers for each model variant.
                """
                for i in range(1, self.MAX_MANAGERS + 1):
                        manager_name = f"manager_rolling_v{i}"
                        model_name = f"rolling_v{i}"
                        self.manager[manager_name] = PredictionManager()
                        self.manager[manager_name].initialize_model("rolling", model_name)

        def load_model(self):
                """
                Loads pre-trained models for each variant.
                """
                for i in range(1, self.MAX_MANAGERS + 1):
                        manager_name = f"manager_rolling_v{i}"
                        self.manager[manager_name].load_model()

        def update_performance(self, context, entry_index: int):
                """
                Updates the performance of each model and recalculates model weights.

                Args:
                        context: The trading context containing market data.
                        entry_index (int): The index of the current data point.
                """

                act = context.dm.get_pred_target(entry_index)
                pandl = context.dm.get_pandl()

                hit_rates = []
                avg_errors = []
                avg_profits = []
                for i in range(self.MAX_MANAGERS):
                        get_pred_method = getattr(context.dm, f'get_pred_v{i+1}')
                        model_pred = get_pred_method(entry_index)
                        pred = 1 if model_pred > 0.5 else 0
                        profit = profit_calculator(pred, act, pandl)
                        model_name = f"model_{i+1}"
                        self.model_trackers[model_name].update(pred, act, profit)

                        hit_rates.append(self.model_trackers[model_name].get_hit_rate())
                        avg_errors.append(self.model_trackers[model_name].get_average_error())
                        avg_profits.append(self.model_trackers[model_name].get_average_profit())

                        context.log_transaction(f"{model_name} - Hit Rate: {hit_rates[-1]:.4f}, Avg Error: {avg_errors[-1]:.4f}, Avg Profit: {avg_profits[-1]:.4f}")

                # デバッグ情報の追加
                context.log_transaction(f"Raw hit rates: {hit_rates}")
                context.log_transaction(f"Raw average errors: {avg_errors}")
                context.log_transaction(f"Raw average profits: {avg_profits}")

                # 異常値の対策
                epsilon = 1e-10  # 小さな値を加えて0での除算を防ぐ
                total_profit = sum(avg_profits)
                total_hit_rate = sum(hit_rates)
                total_error = sum(avg_errors)

                if total_profit > epsilon:
                        profit_weights = np.array(avg_profits) / total_profit
                else:
                        profit_weights = np.ones(self.MAX_MANAGERS) / self.MAX_MANAGERS

                if total_hit_rate > epsilon:
                        hit_rate_weights = np.array(hit_rates) / total_hit_rate
                else:
                        hit_rate_weights = np.ones(self.MAX_MANAGERS) / self.MAX_MANAGERS

                error_weights = np.array(avg_errors)  # 修正: 逆数にしない
                error_weights = np.max(error_weights) - error_weights  # 修正: 反転させる
                error_weights /= np.sum(error_weights)

        # 重み付けパラメータを適用して model_weights を計算
                self.model_weights = (
                        profit_weights * self.profit_weight +
                        hit_rate_weights * self.hit_rate_weight +
                        error_weights * self.error_rate_weight
                )

                self.model_weights /= np.sum(self.model_weights)  # 正規化

                self.model_weights = (profit_weights + hit_rate_weights + error_weights) / 3
                self.model_weights /= np.sum(self.model_weights)

                # デバッグ情報の追加
                context.log_transaction(f"Profit weights: {profit_weights}")
                context.log_transaction(f"Hit rate weights: {hit_rate_weights}")
                context.log_transaction(f"Error weights: {error_weights}")

                for i, weight in enumerate(self.model_weights):
                        context.log_transaction(f"Model {i+1} final weight: {weight:.4f}")

        def get_strategy_performance(self):
                """
                Calculates the overall performance of the entry strategy.

                Returns:
                        dict: A dictionary containing the total profit, overall hit rate, and profit factor.
                """
                total_profit = sum(tracker.get_average_profit() for tracker in self.model_trackers.values())
                overall_hit_rate = sum(tracker.get_hit_rate() for tracker in self.model_trackers.values()) / len(self.model_trackers)
                profit_factor = sum(tracker.get_profit_factor() for tracker in self.model_trackers.values()) / len(self.model_trackers)

                return {
                        "total_profit": total_profit,
                        "overall_hit_rate": overall_hit_rate,
                        "profit_factor": profit_factor
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

        def select_top_models(self, context):
                """
                Selects the top performing models based on recent performance.
                If in the initial state, select all models regardless of top_model_num.

                Returns:
                        list: A list of indices representing the selected top models.
                """
                model_scores = []
                for i in range(self.MAX_MANAGERS):
                        profit = self.model_trackers[f"model_{i+1}"].get_average_profit()
                        hit_rate = self.model_trackers[f"model_{i+1}"].get_hit_rate()
                        error_rate = self.model_trackers[f"model_{i+1}"].get_average_error()

                         # 重み付けパラメータを適用してスコアを計算
                        score = (
                                profit * self.profit_weight +
                                hit_rate * self.hit_rate_weight +
                                (1 - error_rate) * self.error_rate_weight  # エラー率は反転
                        )


                        score = (profit + hit_rate + (1 - error_rate)) / 3  # 修正: エラー率を反転させて加算
                        model_scores.append((i, score))

                # Sort the models based on their scores in descending order
                sorted_models = sorted(model_scores, key=lambda x: x[1], reverse=True)

                # Select the top self.top_model_num models, or all models if in the initial state
                if len(self.model_trackers["model_1"].profits) == 0:  # profits にデータが1つも追加されていない場合を初期状態とする
                        top_model_indices = list(range(self.MAX_MANAGERS))  # 初期状態：全モデルを選択
                        context.log_transaction(f"Initial State: Selecting ALL models.")
                else:
                        top_model_indices = [model_index for model_index, _ in sorted_models[:self.top_model_num]]  # 通常状態：上位x個を選択
                        context.log_transaction(f"Normal State: Selecting top {self.top_model_num} models.")

                context.log_transaction(f"Selected model indices: {top_model_indices}")
                return top_model_indices

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
                if self.model_weights is None:
                        # Initialize model weights if not already initialized
                        self.model_weights = np.array([1 / self.MAX_MANAGERS] * self.MAX_MANAGERS)

                # Get historical data for trend prediction
                current_index = context.dm.get_current_index()
                df = context.dm.get_df_fromto(current_index - (TIME_SERIES_PERIOD - 1), current_index)

                pred_v1 = 0
                # Generate predictions from all models
                rolling_pred = {}
                for i in range(1, self.MAX_MANAGERS + 1):
                        manager_name = f"manager_rolling_v{i}"
                        target_df = self.manager[manager_name].create_time_series_data(df)
                        rolling_pred[i - 1] = self.manager[manager_name].predict_model(target_df, probability=True)


                        set_pred_method = getattr(context.dm, f'set_pred_v{i}')
                        set_pred_method(rolling_pred[i - 1])

                # Select top models for final prediction
                top_model_indices = self.select_top_models(context)

                # Calculate weighted average using selected models
                combined_pred = np.sum([self.model_weights[i] * rolling_pred[i] for i in top_model_indices])

                prediction = 1 if combined_pred > 0.5 else 0

                context.log_transaction(f"Pred_b {prediction}  {combined_pred}")
                return True, prediction
import numpy as np
from collections import defaultdict
import os
import sys

# Get the absolute path of the directory containing this file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the path of the parent directory to sys.path
sys.path.append(parent_dir)

from trading_analysis_kit.trading_state import *
from trading_analysis_kit.simulation_entry_strategy import EntryStrategy

class DynamicModelSelectionStrategy(EntryStrategy):
        def __init__(self):
                super().__init__()
                self.performance_history = {
                        'low': defaultdict(lambda: {'profits': [], 'hit_rates': [], 'error_rates': []}),
                        'high': defaultdict(lambda: {'profits': [], 'hit_rates': [], 'error_rates': []})
                }

                self.window_size = 100 # パフォーマンス履歴のウィンドウサイズ
                self.initial_selection_count = 0  # 初期選択のカウンター

        def is_initial_state(self, state):
                return all(len(self.performance_history[state][i]['profits']) == 0 for i in range(self.MAX_MANAGERS))

        def get_volatility_state(self, context,bbvi):
                #volatility_threshold = context.dm.get_bbvi_median()
                #volatility_threshold = 100
                #return 'low' if bbvi < volatility_threshold else 'high'
                #macdhist = context.dm.get_value_by_column(COLUMN_)
                #return 'low' if macdhist < 0 else 'high'
                bb_direction = context.dm.get_bb_direction()
                return 'low' if bb_direction == BB_DIRECTION_LOWER else 'high'

        def update_performance(self, context, entry_index: int):
                super().update_performance(context, entry_index)

                bbvi = context.dm.get_bbvi(entry_index)
                state = self.get_volatility_state(context,bbvi)


                for i, tracker in enumerate(self.model_trackers.values()):
                        profit = tracker.get_average_profit()
                        hit_rate = tracker.get_hit_rate()
                        error_rate = tracker.get_average_error()

                        self.performance_history[state][i]['profits'].append(profit)
                        self.performance_history[state][i]['hit_rates'].append(hit_rate)
                        self.performance_history[state][i]['error_rates'].append(error_rate)

                        # ウィンドウサイズを超えた古いデータを削除
                        if len(self.performance_history[state][i]['profits']) > self.window_size:
                                self.performance_history[state][i]['profits'].pop(0)
                                self.performance_history[state][i]['hit_rates'].pop(0)
                                self.performance_history[state][i]['error_rates'].pop(0)

                context.log_transaction(f"Updated performance for {state} volatility state")

        def get_top_models(self, state, n=3):
                if self.is_initial_state(state):
                        if self.initial_selection_count < self.MAX_MANAGERS:
                                # まだすべてのモデルを試していない場合、順番に選択
                                selected = list(range(self.initial_selection_count, min(self.initial_selection_count + n, self.MAX_MANAGERS)))
                                self.initial_selection_count += n
                                return [(i, 0) for i in selected]
                        else:
                                # すべてのモデルを試した後は、ランダムに選択
                                return [(i, 0) for i in np.random.choice(self.MAX_MANAGERS, n, replace=False)]
                else:
                        # 通常の状態: スコアに基づいて選択
                        model_scores = []
                        for i in range(self.MAX_MANAGERS):
                                if not self.performance_history[state][i]['profits']:
                                        score = 0
                                else:
                                        avg_profit = np.mean(self.performance_history[state][i]['profits'])
                                        avg_hit_rate = np.mean(self.performance_history[state][i]['hit_rates'])
                                        avg_error_rate = np.mean(self.performance_history[state][i]['error_rates'])
                                        score = avg_profit * 0.5 + avg_hit_rate * 0.3 - avg_error_rate * 0.2
                                model_scores.append((i, score))

                        return sorted(model_scores, key=lambda x: x[1], reverse=True)[:n]

        def calculate_weights(self, top_models):
                if all(score == 0 for _, score in top_models):
                # すべてのスコアが0の場合、均等な重みを返す
                        return [1/len(top_models)] * len(top_models)
                else:
                        total_score = sum(score for _, score in top_models)
                        return [score / total_score for _, score in top_models]


        def trend_prediction(self, context):
                current_index = context.dm.get_current_index()
                bbvi = context.dm.get_bbvi(current_index)
                state = self.get_volatility_state(context,bbvi)

                df = context.dm.get_df_fromto(current_index - (TIME_SERIES_PERIOD - 1), current_index)

                # すべてのモデルの予測を取得
                all_predictions = []
                for i in range(self.MAX_MANAGERS):
                        manager_name = f"manager_rolling_v{i + 1}"
                        target_df = self.manager[manager_name].create_time_series_data(df)
                        prediction = self.manager[manager_name].predict_model(target_df, probability=True)
                        all_predictions.append(prediction)

                         # 個々のモデルの予測をコンテキストに保存
                        set_pred_method = getattr(context.dm, f'set_pred_v{i+1}')
                        set_pred_method(prediction)

                # 現在の状態に基づいて上位3つのモデルを選択
                top_models = self.get_top_models(state)
                weights = self.calculate_weights(top_models)

                # 選択されたモデルの予測のみを使用
                selected_predictions = [all_predictions[model_index] for model_index, _ in top_models]

                combined_pred = np.sum([w * p for w, p in zip(weights, selected_predictions)])
                final_prediction = 1 if combined_pred > 0.5 else 0

                context.log_transaction(f"Volatility state: {state}")
                context.log_transaction(f"Top models: {[m[0] for m in top_models]}")
                context.log_transaction(f"Weights: {weights}")
                context.log_transaction(f"All predictions: {all_predictions}")
                context.log_transaction(f"Selected predictions: {selected_predictions}")
                context.log_transaction(f"Combined prediction: {combined_pred}")

                return True, final_prediction
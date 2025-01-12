import os, sys
from typing import Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)

from common.constants import *
from aiml.transformer_prediction_value_rolling_model import TransformerPredictionRollingModel
from aiml.aiml_comm import  *


class MagnitudePredictionModel(TransformerPredictionRollingModel):

	def set_price_change_threshold(self, threshold: float) -> None:
			self.price_change_threshold = threshold

	def _prepare_sequences(self, data):
			def sequence_generator():
				for i in range(len(data) - (TIME_SERIES_PERIOD+ 1)):
					sequence = data.iloc[i:i+TIME_SERIES_PERIOD, data.columns.get_indexer(self.feature_columns)].values
					start_price = data.iloc[i+TIME_SERIES_PERIOD-1][self.target_column[0]]
					end_price = data.iloc[i+TIME_SERIES_PERIOD][self.target_column[1]]
					price_change = (end_price - start_price) / start_price
					target = int(abs(price_change) >= self.price_change_threshold)
					yield sequence, target

			sequences = []
			targets = []

			for seq, target in sequence_generator():
				scaled_seq = self.scaler.fit_transform(seq)
				sequences.append(scaled_seq)
				targets.append(target)

			sequences = np.array(sequences)
			targets = np.array(targets)
			return sequences, targets

	def load_and_prepare_data(self, start_datetime: str, end_datetime: str, coll_type, test_size=0.2,random_state=None):
			data = self.data_loader.load_data_from_datetime_period(start_datetime, end_datetime, coll_type)
			x, y = self._prepare_sequences(data)
			return	train_test_split(x, y, test_size=test_size, random_state=random_state, shuffle=False)

	def load_and_prepare_data_mix(self, start_datetime: str, end_datetime: str, coll_type: str, test_size=0.2,random_state=None):
			data = load_data(self.data_loader, start_datetime, end_datetime, coll_type)
			x, y = self._prepare_sequences(data)
			return	train_test_split(x, y, test_size=test_size, random_state=random_state, shuffle=False)

	def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, str, np.ndarray]:
			y_pred = (self.predict(x_test) > 0.5).astype(int)
			accuracy = accuracy_score(y_test, y_pred)
			report = classification_report(y_test, y_pred, target_names=['Minor Change', 'Significant Change'])
			conf_matrix = confusion_matrix(y_test, y_pred)
			return accuracy, report, conf_matrix

	def optimize_price_change_threshold(self, data: pd.DataFrame) -> float:
			# 価格変動の計算
			price_changes = []
			for i in range(len(data) - (TIME_SERIES_PERIOD + 1)):
				start_price = data.iloc[i+TIME_SERIES_PERIOD-1][self.target_column[0]]
				end_price = data.iloc[i+TIME_SERIES_PERIOD][self.target_column[1]]
				price_change = abs((end_price - start_price) / start_price)
				price_changes.append(price_change)

			# 初期閾値として中央値を使用
			initial_threshold = np.median(price_changes)

			# バイナリサーチを使用して最適な閾値を見つける
			low, high = 0, max(price_changes)
			optimal_threshold = initial_threshold

			while low <= high:
				mid = (low + high) / 2
				significant_changes = sum(1 for change in price_changes if change >= mid)
				minor_changes = len(price_changes) - significant_changes

				if significant_changes == minor_changes:
					optimal_threshold = mid
					break
				elif significant_changes > minor_changes:
					low = mid
				else:
					high = mid

				# 十分な精度に達した場合、ループを終了
				if high - low < 0.0001:
					optimal_threshold = mid
					break

			return optimal_threshold

	def set_price_change_threshold(self, threshold: float) -> None:
		self.price_change_threshold = threshold
		print(f"Price change threshold set to: {self.price_change_threshold}")

def main():
		magnitude_model = MagnitudePredictionModel("magnitude")
		# 最適化用のデータを読み込む
		optimization_data = magnitude_model.data_loader.load_data_from_datetime_period(
			'2024-01-01 00:00:00',
			'2024-06-01 00:00:00',
			MARKET_DATA_TECH
		)

		# データの妥当性チェック
		# 最適なprice_change_thresholdの計算
		optimal_threshold = magnitude_model.optimize_price_change_threshold(optimization_data)
		print(f"Optimal price change threshold: {optimal_threshold}")

		# 新しい閾値をモデルに設定
		magnitude_model.set_price_change_threshold(optimal_threshold)

		# トレーニングデータの準備
		x_train, x_test, y_train, y_test = magnitude_model.load_and_prepare_data_mix(
			'2020-01-01 00:00:00',
			'2024-01-01 00:00:00',
			COLLECTIONS_TECH)

		# モデルのトレーニング
		#magnitude_model.train(x_train, y_train)
		magnitude_model.train_with_cross_validation(x_train, y_train)

		# モデルの評価
		accuracy, report, conf_matrix = magnitude_model.evaluate(x_test, y_test)
		print("\nMagnitude Prediction Results:")
		print(f'Accuracy: {accuracy}')
		print(report)
		print(conf_matrix)

		# モデルの保存
		magnitude_model.save_model()

		#トレーニングデータの準備
		x_train, x_test, y_train, y_test = magnitude_model.load_and_prepare_data(
			'2024-01-01 00:00:00',
			'2024-07-01 00:00:00',
			MARKET_DATA_TECH)

		accuracy, report, conf_matrix = magnitude_model.evaluate(x_test, y_test)
		print("\nMagnitude Prediction Results:")
		print(f'Accuracy: {accuracy}')
		print(report)
		print(conf_matrix)

if __name__ == '__main__':
	main()
import os
import sys
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd
import operator

# Set path to parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.constants import *
from common.utils import get_config_model
from common.trading_logger import TradingLogger
from mongodb.data_loader_mongo import MongoDataLoader
from aiml.prediction_model import PredictionModel
from aiml.lstm_prediction_rolling_model import LSTMPredictionRollingModel
from aiml.transformer_prediction_rolling_model import TransformerPredictionRollingModel
#from aiml.transformer_prediction_logsparse import TransformerPredictionRollingModel
#from aiml.transformer_triclass_rolling_model import TransformerPredictionRollingModel
#from aiml.transformer_prediction_rolling_model_atvis import TransformerPredictionRollingModel

from aiml.aiml_comm import process_timestamp_and_cyclical_features
from aiml.aiml_comm import COLLECTIONS_LOWER, COLLECTIONS_UPPER, COLLECTIONS_TECH


class ModelFactory:
        """
        Factory class to create prediction models.
        """

        @staticmethod
        def create_model(
                 id: str, config: Dict[str, Any],data_loader: MongoDataLoader, logger: TradingLogger, symbol: str = None, interval: str = None
                ) -> PredictionModel:
                """
                Creates a prediction model of the specified type.

                Args:
                        model_type (str): Type of prediction model.
                        id (str): ID of the prediction model.
                        data_loader (MongoDataLoader): Data loader instance.
                        logger (TradingLogger): Logger instance.
                        symbol (str, optional): Symbol name. Defaults to None.
                        interval (str, optional): Interval. Defaults to None.

                Returns:
                        PredictionModel: Created prediction model.

                Raises:
                        ValueError: If an unknown model type is specified.
                """
                model_map = {
                        "lstm": LSTMPredictionRollingModel,
                        "rolling": TransformerPredictionRollingModel
                }
                model_type = config["MODEL_TYPE"]
                model_class = model_map.get(model_type)
                if model_class is None:
                        raise ValueError(f"Unknown model type: {model_type}")
                return model_class(id, config, data_loader, logger, symbol, interval)


class PredictionManager:
        """
        Manages prediction models.
        Handles loading data, training models, evaluating models, making predictions, etc.
        """

        def __init__(self):
                """
                Initializes an instance of PredictionManager.
                """
                self.logger = TradingLogger()
                self.data_loader = MongoDataLoader()  # Create a DataLoader instance internally
                self.prediction_model: PredictionModel = None
                self.data: Dict[str, np.ndarray] = {}
                self.model_type = None  # type: str
                self.collection_name = None  # type: str

        def initialize_model(self, id: str, config ,symbol: str = None, interval: str = None) -> None:
                """
                Initializes a prediction model of the specified type.

                Args:
                        model_type (str): Type of prediction model.
                        id (str): ID of the prediction model.
                        symbol (str, optional): Symbol name. Defaults to None.
                        interval (str, optional): Interval. Defaults to None.
                """

                self.model_type = config["MODEL_TYPE"]
                self.prediction_model = ModelFactory.create_model(
                         id, config, self.data_loader, self.logger, symbol, interval
                )

        def set_parameters(self, **kwargs) -> None:
                """
                Sets parameters for the prediction model.

                Args:
                **kwargs: Parameters to set.
                """
                self.prediction_model.set_parameters(**kwargs)

        def load_and_prepare_data_train(
                self,
                start_datetime: str,
                end_datetime: str,
                test_size: float = 0.5,
                random_state: int = None,
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                """
                Loads and prepares training data.

                Args:
                        start_datetime (str): Start date and time of data.
                        end_datetime (str): End date and time of data.
                         test_size (float, optional): Proportion of test data. Defaults to 0.5.
                        random_state (int, optional): Random seed. Defaults to None.

                Returns:
                        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                        Training features, test features, training labels, test labels.
                """
                coll_map = {
                        "lstm": COLLECTIONS_TECH,
                        "rolling": COLLECTIONS_TECH,
                        "ts_upper": COLLECTIONS_UPPER,
                        "ts_lower": COLLECTIONS_LOWER,
                        "magnitude": COLLECTIONS_TECH,
                }
                data = self.prediction_model.load_and_prepare_data_mix(
                        start_datetime, end_datetime, coll_map[self.model_type], test_size, random_state
                )
                (
                        self.data["x_train"],
                        self.data["x_test"],
                        self.data["y_train"],
                        self.data["y_test"],
                ) = data
                return data

        def load_and_prepare_data(
                self,
                start_datetime: str,
                end_datetime: str,
                test_size: float = 0.9,
                random_state: int = None,
         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                """
                Loads and prepares data for evaluation.

                Args:
                        start_datetime (str): Start date and time of data.
                        end_datetime (str): End date and time of data.
                        test_size (float, optional): Proportion of test data. Defaults to 0.2.
                        random_state (int, optional): Random seed. Defaults to None.

                Returns:
                        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                        Training features, test features, training labels, test labels.
                """
                colltype_map = {
                        "lstm": MARKET_DATA_TECH,
                        "rolling": MARKET_DATA_TECH,
                        "ts_upper": MARKET_DATA_ML_UPPER,
                        "ts_lower": MARKET_DATA_ML_LOWER,
                        "magnitude": MARKET_DATA_TECH,
                }

                data = self.prediction_model.load_and_prepare_data(
                        start_datetime,
                        end_datetime,
                        colltype_map[self.model_type],
                        test_size=test_size,
                         random_state=random_state,
                )
                (
                        self.data["x_train"],
                        self.data["x_test"],
                        self.data["y_train"],
                        self.data["y_test"],
                ) = data
                return data

        def train_model(self, x_train: np.ndarray = None, y_train: np.ndarray = None) -> None:
                """
                Trains the prediction model.

                Args:
                        x_train (np.ndarray, optional): Training features. Defaults to None.
                        y_train (np.ndarray, optional): Training labels. Defaults to None.

                Raises:
                        ValueError: If training data is not available.
                """
                x_train = x_train if x_train is not None else self.data.get("x_train")
                y_train = y_train if y_train is not None else self.data.get("y_train")
                if x_train is None or y_train is None:
                        raise ValueError("Training data not available. Please load data first.")
                self.prediction_model.train(x_train, y_train)

        def train_with_cross_validation(self) -> List[float]:
                """
                Trains the prediction model using cross-validation.

                Returns:
                        List[float]: List of scores for each fold.

                Raises:
                        ValueError: If data is not loaded properly.
                """
                if not all(key in self.data for key in ["x_train", "x_test", "y_train", "y_test"]):
                        raise ValueError("Data not properly loaded. Please load data first.")

                x_all = np.concatenate((self.data["x_train"], self.data["x_test"]), axis=0)
                y_all = np.concatenate((self.data["y_train"], self.data["y_test"]), axis=0)
                cv_scores = self.prediction_model.train_with_cross_validation(x_all, y_all)

                for i, score in enumerate(cv_scores, 1):
                        self.logger.log_system_message(f"Fold {i}: Accuracy = {score}")
                return cv_scores

        def predict_period_model(self, date: str) -> np.ndarray:
                """
                Predicts data for the specified period.

                Args:
                date (str): Start date and time of the period to predict.

                Returns:
                        np.ndarray: Prediction results.
                """
                data = self.prediction_model.get_data_period(date, TIME_SERIES_PERIOD - 1)
                return self.prediction_model.predict(data)

        def predict_rolling_model_date(self, feature_date: str) -> int:
                """
                Performs rolling prediction for the data on the specified date and time.

                Args:
                        feature_date (str): Date and time of the data to predict.

                Returns:
                        int: Prediction result (1 for up, 0 for down).
                """
                df = self.data_loader.filter(COLUMN_START_AT, operator.eq, feature_date)

                if df.empty:
                        self.logger.log_system_message("No data found")
                        return 0

                data_frame = self.data_loader.get_df_fromto(
                        df.index[0] - (TIME_SERIES_PERIOD - 1), df.index[0]
                )

                target_df = self.create_time_series_data(data_frame)

                prediction = self.predict_model(target_df)
                return prediction
        def predict_model(self, data_point: np.ndarray, probability: bool = False) -> int:
                """
                Makes a prediction for a single data point.

                Args:
                        data_point (np.ndarray): Data point to predict.

                Returns:
                        int: Prediction result (1 for up, 0 for down).
                """
                # Get feature columns
                feature_columns = self.prediction_model.get_feature_columns()

                # If data_point is a pandas DataFrame, extract numpy array
                if hasattr(data_point, 'columns'):
                    # Convert feature_columns list to numeric indices
                    feature_indices = [data_point.columns.get_loc(col) for col in feature_columns]
                    data_point = data_point.iloc[:, feature_indices].to_numpy()

                #print(f"predict_model - data_point.shape: {data_point.shape}")  # Debug output
                if probability:
                        predict = self.prediction_model.predict_single_res(data_point)
                else:
                        predict = self.prediction_model.predict_single(data_point)
                return predict

        def evaluate_model(
                self, x_test: np.ndarray = None, y_test: np.ndarray = None
        ) -> Tuple[float, str, np.ndarray]:
                """
                Evaluates the prediction model.

                Args:
                        x_test (np.ndarray, optional): Test features. Defaults to None.
                        y_test (np.ndarray, optional): Test labels. Defaults to None.

                Returns:
                        Tuple[float, str, np.ndarray]: Accuracy, classification report, confusion matrix.

                Raises:
                        ValueError: If test data is not available.
                """
                x_test = x_test if x_test is not None else self.data.get("x_test")
                y_test = y_test if y_test is not None else self.data.get("y_test")

                if x_test is None or y_test is None:
                        raise ValueError("Test data not available. Please load data first.")

                accuracy, report, conf_matrix = self.prediction_model.evaluate(x_test, y_test)
                self.logger.log_system_message(f"Model Accuracy: {accuracy}")
                self.logger.log_system_message(f"Classification Report:\n{report}")
                self.logger.log_system_message(f"Confusion Matrix:\n{conf_matrix}")
                return accuracy, report, conf_matrix

        def predict(self, x: np.ndarray) -> np.ndarray:
                """
                Predicts data.

                Args:
                        x (np.ndarray): Data to predict.

                Returns:
                        np.ndarray: Prediction results.
                """
                return self.prediction_model.predict(x)

        def save_model(self,filename:str=None):
                """
                Saves the trained model.
                """
                self.prediction_model.save_model(filename)

        def load_model(self,filename:str=None):
                """
                Loads a saved model.
                """
                self.prediction_model.load_model(filename)

        def create_time_series_data(self, df: pd.DataFrame) -> np.ndarray:
                """
                Creates time series data from a DataFrame.

                Args:
                        df (pd.DataFrame): DataFrame containing time series data.

                Returns:
                        np.ndarray: Created time series data.
                """
                                # もし、sequnnce['date'] が存在する場合
                df_copy = df.copy()

                feature_columns = self.prediction_model.get_feature_columns()

                # Remove rows where the 'date' column is NaN
                if 'date' in feature_columns:
                        if df_copy['date'].isna().any():
                            print("DataFrame contains NaN values in 'date' column:")
                            print(df_copy[df_copy['date'].isna()])
                            raise ValueError("NaN values found in 'date' column")
                        filtered_df = process_timestamp_and_cyclical_features(df_copy)
                        sequence = filtered_df.filter(items=feature_columns)
                        if sequence['date'].count() < TIME_SERIES_PERIOD:
                                print(df_copy)
                                print(filtered_df)
                                print(sequence)
                                raise ValueError("Insufficient data for time series prediction")
                else:
                        sequence = df_copy.filter(items=feature_columns)
                return sequence.to_numpy()


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
        manager = PredictionManager()
        config = get_config_model("MODEL_SHORT_TERM", "rolling_v7")  # Get the model configuration

        manager.initialize_model("rolling_v1", config)

        # 学習済みモデルを読み込む
        #manager.load_model("rolling_BTC_ETH_SOL_BNB_60_model_macd4")
        #manager.load_model("rolling_v2_BTCUSDT_60_model_macd4")

        #manager.load_model("rolling_v4_BTCUSDT_60_model_roc2")
        #manager.load_model("rolling_v11_BTC_ETH_SOL_BNB_60_model_macd8")

        manager.load_model()
        #manager.load_model("mix_lower_mlts_model_v2")

    # 評価期間の開始日時と終了日時を指定して、データをロード
        start_datetime = "2024-04-10 00:00:00"
        end_datetime = "2025-01-01 00:00:00"

        X_train, x_test, y_train, y_test  = manager.load_and_prepare_data(start_datetime, end_datetime, test_size=0.8)
        manager.evaluate_model()

        df = manager.data_loader.load_data_from_point_date(start_datetime,TIME_SERIES_PERIOD,MARKET_DATA_TECH)
        print(df)

        res = manager.predict_model(df)
        print(res)

"""
        # 予測結果を格納するリスト
        all_predictions = []
        # 正解ラベルを格納するリスト
        all_true_labels = []

    # テストデータセットの各データポイントに対して予測を行う
        for i in range(len(x_test)):  # TIME_SERIES_PERIOD を考慮しない
                # predict_singleメソッドを使って予測
                prediction = manager.predict_model(x_test[i])  # x_test から直接データ取得

                all_predictions.append(prediction)
                # 正解ラベルを追加
                all_true_labels.append(y_test[i])

    # 正答率を計算
        accuracy = accuracy_score(all_true_labels, all_predictions)
        print(f"Accuracy: {accuracy}")

        # 適合率、再現率、F1スコアなどを含むレポートを表示
        report = classification_report(all_true_labels, all_predictions)
        print(f"Classification Report:\n{report}")

        # 混同行列を表示
        conf_matrix = confusion_matrix(all_true_labels, all_predictions)
        print(f"Confusion Matrix:\n{conf_matrix}")
"""
if __name__ == "__main__":
        main()

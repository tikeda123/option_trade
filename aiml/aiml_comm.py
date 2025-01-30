import os
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, List
from tqdm import tqdm

# Add parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.constants import MARKET_DATA_ML_UPPER, MARKET_DATA_ML_LOWER
from common.trading_logger import TradingLogger
from mongodb.data_loader_mongo import MongoDataLoader

# Lists of MongoDB collection names for different data types
COLLECTIONS_LOWER = [
                           "BTCUSDT_5_market_data_mlts_lower",
                           "BTCUSDT_15_market_data_mlts_lower",
                                "BTCUSDT_30_market_data_mlts_lower",
                                "BTCUSDT_60_market_data_mlts_lower",
                                "BTCUSDT_120_market_data_mlts_lower",
                                "BTCUSDT_240_market_data_mlts_lower",
                                "BTCUSDT_720_market_data_mlts_lower",

                           "ETHUSDT_5_market_data_mlts_lower",
                                "ETHUSDT_15_market_data_mlts_lower",
                                "ETHUSDT_30_market_data_mlts_lower",
                                "ETHUSDT_60_market_data_mlts_lower",
                                "ETHUSDT_120_market_data_mlts_lower",
                                "ETHUSDT_240_market_data_mlts_lower",
                                "ETHUSDT_720_market_data_mlts_lower",

                           "SOLUSDT_5_market_data_mlts_lower",
                                "SOLUSDT_15_market_data_mlts_lower",
                          "SOLUSDT_30_market_data_mlts_lower",
                                "SOLUSDT_60_market_data_mlts_lower",
                                "SOLUSDT_120_market_data_mlts_lower",
                                "SOLUSDT_240_market_data_mlts_lower",
                                "SOLUSDT_720_market_data_mlts_lower",

                           "BNBUSDT_5_market_data_mlts_lower",
                                "BNBUSDT_15_market_data_mlts_lower",
                                "BNBUSDT_30_market_data_mlts_lower",
                                "BNBUSDT_60_market_data_mlts_lower",
                                "BNBUSDT_120_market_data_mlts_lower",
                                "BNBUSDT_240_market_data_mlts_lower",
                                "BNBUSDT_720_market_data_mlts_lower"
]

COLLECTIONS_UPPER = [
                                "BTCUSDT_5_market_data_mlts_upper",
                                "BTCUSDT_15_market_data_mlts_upper",
                                "BTCUSDT_30_market_data_mlts_upper",
                                "BTCUSDT_60_market_data_mlts_upper",
                                "BTCUSDT_120_market_data_mlts_upper",
                                "BTCUSDT_240_market_data_mlts_upper",
                                "BTCUSDT_720_market_data_mlts_upper",

                                "ETHUSDT_5_market_data_mlts_upper",
                                "ETHUSDT_15_market_data_mlts_upper",
                                "ETHUSDT_30_market_data_mlts_upper",
                                "ETHUSDT_60_market_data_mlts_upper",
                                "ETHUSDT_120_market_data_mlts_upper",
                                "ETHUSDT_240_market_data_mlts_upper",
                                "ETHUSDT_720_market_data_mlts_upper",

                                "SOLUSDT_5_market_data_mlts_upper",
                                "SOLUSDT_15_market_data_mlts_upper",
                                "SOLUSDT_30_market_data_mlts_upper",
                                "SOLUSDT_60_market_data_mlts_upper",
                                "SOLUSDT_120_market_data_mlts_upper",
                                "SOLUSDT_240_market_data_mlts_upper",
                                "SOLUSDT_720_market_data_mlts_upper",

                                "BNBUSDT_5_market_data_mlts_upper",
                                "BNBUSDT_15_market_data_mlts_upper",
                                "BNBUSDT_30_market_data_mlts_upper",
                                "BNBUSDT_60_market_data_mlts_upper",
                                "BNBUSDT_120_market_data_mlts_upper",
                                "BNBUSDT_240_market_data_mlts_upper",
                                "BNBUSDT_720_market_data_mlts_upper"
]

"""
COLLECTIONS_TECH = [
                                "BTCUSDT_60_market_data_tech",
                                "BTCUSDT_120_market_data_tech",
                                "BTCUSDT_240_market_data_tech",
                                "BTCUSDT_720_market_data_tech",
                                "ETHUSDT_60_market_data_tech",
                                "ETHUSDT_120_market_data_tech",
                                "ETHUSDT_240_market_data_tech",
                                "ETHUSDT_720_market_data_tech",
]



COLLECTIONS_TECH = [
                                "BTCUSDT_60_market_data_tech",
                                "BTCUSDT_120_market_data_tech",
                                "BTCUSDT_240_market_data_tech",
                                "BTCUSDT_720_market_data_tech",
                                "BTCUSDT_D_market_data_tech",

                                "ETHUSDT_60_market_data_tech",
                                "ETHUSDT_120_market_data_tech",
                                "ETHUSDT_240_market_data_tech",
                                "ETHUSDT_720_market_data_tech",
                                "ETHUSDT_D_market_data_tech",

                                "BNBUSDT_60_market_data_tech",
                                "BNBUSDT_120_market_data_tech",
                                "BNBUSDT_240_market_data_tech",
                                "BNBUSDT_720_market_data_tech",
                                "BNBUSDT_D_market_data_tech",

                                "SOLUSDT_60_market_data_tech",
                                "SOLUSDT_120_market_data_tech",
                                "SOLUSDT_240_market_data_tech",
                                "SOLUSDT_720_market_data_tech",
                                "SOLUSDT_D_market_data_tech"

]
"""
COLLECTIONS_TECH = [
                                "BTCUSDT_60_market_data_tech",
                                "BTCUSDT_120_market_data_tech",
                                "BTCUSDT_240_market_data_tech",
                                "BTCUSDT_720_market_data_tech",
                                "BTCUSDT_1440_market_data_tech",
                                "BTCUSDT_4320_market_data_tech",
                                "BTCUSDT_7200_market_data_tech",
                                "BTCUSDT_10080_market_data_tech",

                                "ETHUSDT_1440_market_data_tech",
                                "ETHUSDT_4320_market_data_tech",
                                "ETHUSDT_7200_market_data_tech",
                                "ETHUSDT_10080_market_data_tech",

                                "BNBUSDT_1440_market_data_tech",
                                "BNBUSDT_4320_market_data_tech",
                                "BNBUSDT_7200_market_data_tech",
                                "BNBUSDT_10080_market_data_tech",

                                "SOLUSDT_1440_market_data_tech",
                                "SOLUSDT_4320_market_data_tech",
                                "SOLUSDT_7200_market_data_tech",
                                "SOLUSDT_10080_market_data_tech"
]

def setup_gpu():
                """
                Sets up GPU for TensorFlow if available.

                This function checks for available GPUs, prints TensorFlow version and GPU availability,
                and configures memory growth for each GPU.
                """
                print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
                print("TensorFlow version:", tf.__version__)
                print("Is GPU available: ", tf.test.is_gpu_available())

                gpus = tf.config.experimental.list_physical_devices("GPU")
                if gpus:
                                try:
                                                for gpu in gpus:
                                                                tf.config.experimental.set_memory_growth(gpu, True)
                                                print("GPU is available and configured")
                                except RuntimeError as e:
                                                print(f"Error configuring GPU: {e}")
                else:
                                print("GPU is not available, using CPU")


def configure_gpu(use_gpu: bool,logger:TradingLogger) -> None:
        """
        GPUの使用を設定する。
        """
        if not use_gpu:
            tf.config.set_visible_devices([], "GPU")
            logger.log_system_message("GPU disabled for inference.")
            return

        # GPUがある場合の設定
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.log_system_message(
                    f"GPU enabled for inference. Available GPUs: {len(gpus)}"
                )
            except RuntimeError as e:
                logger.log_error(f"GPU configuration error: {str(e)}")
        else:
            logger.log_system_message("No GPU available, using CPU instead.")

def load_data(
                db: MongoDataLoader, start_date: str, end_date: str, collections: list
) -> pd.DataFrame:
                """
                Loads and combines data from multiple MongoDB collections.

                Args:
                                db (MongoDataLoader): MongoDB data loader instance.
                                start_date (str): Start date for data loading.
                                end_date (str): End date for data loading.
                                collections (list): List of collection names to load data from.

                Returns:
                                pd.DataFrame: Combined DataFrame containing data from all specified collections.
                """
                df_root = pd.DataFrame()  # Initialize an empty DataFrame
                for collection in collections:
                                db.set_direct_collection_name(collection)  # Set the current collection in the data loader
                                df_next = db.load_data_from_datetime_period(
                                                start_date, end_date
                                )  # Load data from the current collection
                                df_root = pd.concat(
                                                [df_root, df_next], ignore_index=True
                                )  # Concatenate the loaded data to the main DataFrame
                return df_root  # Return the combined DataFrame


def train_and_evaluate(
                model: any,
                device: str,
                start_date,
                end_date,
                new_date_start,
                new_date_end,
) -> Tuple[List[float], float, str, np.ndarray]:
                """
                Trains the model using cross-validation and evaluates it on new data.

                This function first loads and prepares training data, then performs cross-validation training.
                After training, it loads and prepares a separate dataset for evaluation and evaluates the
                trained model on this new data. It prints the evaluation results and returns them.

                Args:
                                model (any): The model to train and evaluate.
                                device (str): The device to use for training (e.g., '/CPU:0' or '/GPU:0').
                                start_date: Start date for training data.
                                end_date: End date for training data.
                                new_date_start: Start date for evaluation data.
                                new_date_end: End date for evaluation data.

                Returns:
                                Tuple[List[float], float, str, np.ndarray]: Cross-validation scores, accuracy on new data,
                                                classification report, and confusion matrix.
                """
                model.load_and_prepare_data_train(
                                start_date, end_date, test_size=0.2
                )  # Load and prepare training data

                with tf.device(device):  # Specify the device for training
                                cv_scores = model.train_with_cross_validation()  # Train with cross-validation

                model.load_and_prepare_data(
                                new_date_start, new_date_end, test_size=0.8
                )  # Load and prepare evaluation data
                accuracy, report, conf_matrix = model.evaluate_model()  # Evaluate the model

                # Print the evaluation results
                print(f"Accuracy: {accuracy}")
                print(report)
                print(conf_matrix)

                return cv_scores, accuracy, report, conf_matrix  # Return the evaluation results



def permutation_feature_importance(model, X, y, metric, n_repeats=10):
        """
        Calculates permutation feature importance.

        Permutation feature importance measures the importance of a feature by randomly shuffling its values
        and measuring the decrease in model performance.

        Args:
                model: The trained model.
                X (np.ndarray): Feature data.
                y (np.ndarray): Target labels.
                metric (callable): Evaluation metric to use.
                n_repeats (int, optional): Number of times to repeat the permutation. Defaults to 10.

        Returns:
                np.ndarray: Permutation feature importance scores for each feature.
        """
        baseline_score = metric(
                y, (model.predict(X) > 0.5).astype(int).flatten()
        )
        feature_importances = []

        for column in range(X.shape[2]):
                scores = []
                for _ in range(n_repeats):
                        X_permuted = X.copy()
                        np.random.shuffle(X_permuted[:, :, column])
                        score = metric(
                                y, (model.predict(X_permuted) > 0.5).astype(int).flatten()
                        )
                        scores.append(score)
                feature_importances.append(baseline_score - np.mean(scores))

        return np.array(feature_importances)


def drop_column_feature_importance(model, X, y, metric):
                """
                 Calculates drop-column feature importance.

                Drop-column feature importance measures the importance of a feature by setting its values to zero
                and measuring the decrease in model performance.

                Args:
                                model: The trained model.
                                X (np.ndarray): Feature data.
                                y (np.ndarray): Target labels.
                                metric (callable): Evaluation metric to use.

                Returns:
                                np.ndarray: Drop-column feature importance scores for each feature.
                """
                baseline_score = metric(
                                                                                                y, (model.predict(X) > 0.5).astype(int).flatten()
                                                                                )
                feature_importances = []

                for column in range(X.shape[2]):
                                X_dropped = X.copy()
                                X_dropped[:, :, column] = 0
                                score = metric(
                                                                                y, (model.predict(X_dropped) > 0.5).astype(int).flatten()
                                                                )
                                feature_importances.append(baseline_score - score)

                return np.array(feature_importances)

def  process_timestamp_and_cyclical_features(data: pd.DataFrame) -> pd.DataFrame:
                """
                Preprocesses data by converting datetime strings to Unix timestamps and adding sine wave features for hour, day of week, and month.

                Args:
                                 data (pd.DataFrame): DataFrame to preprocess.

                Returns:
                                pd.DataFrame: Preprocessed DataFrame.
                """
                # datetime オブジェクトに変換
                data['date'] = pd.to_datetime(data['date'])

                # Unix タイムスタンプに変換
                data['date'] = data['date'].astype(np.int64) // 10**9

                                 # 時間帯を計算
                data['hour'] = (data['date'] // 3600) % 24
                data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
                data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)

                                # 曜日を計算
                data['day_of_week'] = (data['date'] // (3600 * 24)) % 7
                data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
                data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)

                                # 月を計算
                data['month'] = (pd.to_datetime(data['date'], unit='s').dt.month)
                data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
                data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

                return data

def get_aligned_lower_timeframe_timestamp(higher_timestamp: str, higher_interval: int, lower_interval: int) -> str:
    """
    Returns the corresponding timestamp for the lower timeframe based on the given timestamp for the higher timeframe.

    Args:
            higher_timestamp (str): Timestamp for the higher timeframe (e.g., "2024-01-01 14:30:00").
            higher_interval (int): Interval of the higher timeframe in minutes (e.g., 240).
            lower_interval (int): Interval of the lower timeframe in minutes (e.g., 60).

    Returns:
            str: Corresponding timestamp for the lower timeframe.

    Raises:
            ValueError: If higher_interval is less than or equal to lower_interval, or if the timestamp format is invalid.
    """
    if higher_interval  == 'D':
        higher_interval = 1440
    # Validation: higher_interval > lower_interval
    if higher_interval <= lower_interval:
        raise ValueError("higher_interval must be greater than lower_interval")

    try:
        # Convert timestamp to pandas Timestamp object
        t = pd.to_datetime(higher_timestamp)
    except Exception as e:
        raise ValueError(f"Invalid timestamp format: {higher_timestamp}") from e

    # Floor to the start time of the higher timeframe
    higher_freq = f'{higher_interval}T'  # e.g., '240T'
    t_floor_higher = t.floor(higher_freq)

    # Within the higher timeframe, floor to the lower timeframe
    lower_freq = f'{lower_interval}T'  # e.g., '60T'
    lower_timestamp = t_floor_higher.floor(lower_freq)

    # Format the timestamp as a string and return
    return lower_timestamp.strftime('%Y-%m-%d %H:%M:%S')

def main():
        test_cases = [
        # (higher_timestamp, higher_interval, lower_interval, expected_lower_timestamp)
        ("2024-03-02 14:00:00", 720, 60, "2024-03-02 12:00:00"),
        ("2024-03-02 10:00:00", 720, 60, "2024-03-02 00:00:00"),
        ("2024-03-01 00:00:00", 720, 60, "2024-03-01 00:00:00"),

        ("2024-03-02 14:00:00", 1440, 60, "2024-03-02 00:00:00"),
        ("2024-03-02 10:00:00", 1440, 60, "2024-03-02 00:00:00"),
        ("2024-03-01 00:00:00", 1440, 60, "2024-03-01 00:00:00"),

        ]

        for higher_ts, higher_int, lower_int, expected in test_cases:
                result = get_aligned_lower_timeframe_timestamp(higher_ts, higher_int, lower_int)
                assert result == expected, f"Test failed for higher timestamp {higher_ts}, higher interval {higher_int}, lower interval {lower_int}. Expected {expected}, got {result}"
                print(f"Higher timestamp: {higher_ts}, Higher interval: {higher_int} min, Lower interval: {lower_int} min -> Lower timestamp: {result} (Expected: {expected})")

        print("All test cases passed successfully.")


if __name__ == "__main__":
        main()

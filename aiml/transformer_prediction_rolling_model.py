import os
import sys
from typing import Tuple
import numpy as np
import tensorflow as tf
import pandas as pd
from typing import Dict, Any, Optional

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import joblib

from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
)

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the path of the parent directory to sys.path
sys.path.append(parent_dir)

from common.trading_logger import TradingLogger
from mongodb.data_loader_mongo import MongoDataLoader
from common.utils import get_config
from common.constants import *

from aiml.prediction_model import PredictionModel, ModelManager
from aiml.transformerblock import TransformerBlock
from aiml.model_param import ModelParam, BaseModel


class TransformerPredictionRollingModel(BaseModel, PredictionModel):
    """
    Model class that performs rolling prediction using Transformer.

    Args:
        id (str): Model ID.
        data_loader (DataLoader): Data loader instance.
        logger (TradingLogger): Logger instance.
        symbol (str, optional): Symbol name. Defaults to None.
        interval (str, optional): Data interval. Defaults to None.
        use_gpu (bool, optional): Whether to use GPU for inference. Defaults to True.

    Attributes:
        logger (TradingLogger): Instance for handling log information.
        data_loader (MongoDataLoader): Instance for handling data loading.
        config (Dict[str, Any]): Dictionary of model configuration values.
        datapath (str): Data path.
        feature_columns (list): Feature columns.
        symbol (str): Symbol name.
        interval (str): Data interval.
        filename (str): File name.
        target_column (str): Target column.
        scaler (StandardScaler): Instance used for scaling.
        table_name (str): Data table name.
        model (tf.keras.Model): Transformer model.
    """

    def __init__(
        self,
        id: str,
        config: Dict[str, Any],
        data_loader: MongoDataLoader,
        logger: TradingLogger,
        symbol: str = None,
        interval: str = None,
        use_gpu: bool = True,
    ):
        """
        Initializes the TransformerPredictionRollingModel.

        Args:
            id (str): Model ID.
            data_loader (DataLoader): Data loader instance.
            logger (TradingLogger): Logger instance.
            symbol (str, optional): Symbol name. Defaults to None.
            interval (str, optional): Data interval. Defaults to None.
            use_gpu (bool, optional): Whether to use GPU for inference. Defaults to True.
        """
        super().__init__(id, config, data_loader, logger, symbol, interval)
        self._initialize_attributes()
        self._configure_gpu(use_gpu)

    def _initialize_attributes(self):
        """
        Initializes the attributes of the model.
        """
        self.datapath = f"{parent_dir}/{self.config['DATAPATH']}"
        self.feature_columns = self.config["FEATURE_COLUMNS"]
        self.target_column = self.config["TARGET_COLUMN"]
        self.prediction_distance = self.config["PREDICTION_DISTANCE"]
        self.filename = self.config["MODLE_FILENAME"]
        self.scaler = StandardScaler()
        self.table_name = f"{self.symbol}_{self.interval}"
        self.model = None

    def set_parameters(
        self,
        time_series_period: Optional[int] = None,
        param_learning_rate: Optional[float] = None,
        param_epochs: Optional[int] = None,
        n_splits: Optional[int] = None,
        batch_size: Optional[int] = None,
        positive_threshold: Optional[float] = None,
    ) -> None:
        """
        Sets the model parameters.

        Args:
            time_series_period (Optional[int]): Period of time series data.
            param_learning_rate (Optional[float]): Learning rate.
            param_epochs (Optional[int]): Number of epochs.
            n_splits (Optional[int]): Number of splits for cross-validation.
            batch_size (Optional[int]): Batch size.
            positive_threshold (Optional[float]): Positive threshold.

        Raises:
            ValueError: If an invalid parameter value is specified.
        """
        if time_series_period is not None:
            if time_series_period <= 3:
                raise ValueError("time_series_period must be more than 3")
            self.model_param.TIME_SERIES_PERIOD = time_series_period

        if param_learning_rate is not None:
            if param_learning_rate <= 0:
                raise ValueError("param_learning_rate must be positive")
            self.model_param.ROLLING_PARAM_LEARNING_RATE = param_learning_rate

        if param_epochs is not None:
            if param_epochs <= 0:
                raise ValueError("param_epochs must be positive")
            self.model_param.PARAM_EPOCHS = param_epochs

        if n_splits is not None:
            if n_splits <= 1:
                raise ValueError("n_splits must be greater than 1")
            self.model_param.N_SPLITS = n_splits

        if batch_size is not None:
            if batch_size <= 0:
                raise ValueError("batch_size must be positive")
            self.model_param.BATCH_SIZE = batch_size

        if positive_threshold is not None:
            self.model_param.POSITIVE_THRESHOLD = positive_threshold

        self.logger.log_system_message("Model parameters updated successfully")

    def get_data_loader(self) -> MongoDataLoader:
        """
        Gets the data loader.

        Returns:
            MongoDataLoader: Data loader instance.
        """
        return self.data_loader

    def get_feature_columns(self) -> list:
        """
        Gets the feature columns used.

        Returns:
            list: List of feature columns.
        """
        return self.feature_columns

    def create_table_name(self) -> str:
        """
        Creates the table name.

        Returns:
            str: Created table name.
        """
        self.table_name = f"{self.symbol}_{self.interval}_market_data_tech"
        return self.table_name

    def load_and_prepare_data(
        self,
        start_datetime,
        end_datetime,
        coll_type,
        test_size=0.5,
        random_state=None,
    ):
        """
        Loads and prepares data from the database for training or evaluation.

        Args:
            start_datetime (str): Start date and time of the data.
            end_datetime (str): End date and time of the data.
            coll_type (str): Collection type to load data from.
            test_size (float, optional): Proportion of data to use for testing. Defaults to 0.5.
            random_state (int, optional): Random seed for data splitting. Defaults to None.

        Returns:
            tuple: A tuple containing the training and testing data.
        """
        data = self.data_loader.load_data_from_datetime_period(
            start_datetime, end_datetime, coll_type
        )

        # Prepare sequences and ensure correct shapes
        scaled_sequences, targets = self._prepare_sequences(data)
        return train_test_split(
            scaled_sequences,
            targets,
            test_size=test_size,
            random_state=random_state,
            shuffle=False,  # 時系列データなのでシャッフルしない
        )

    def prepare_data(
        self, data: pd.DataFrame, test_size=0.5, random_state=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepares the data for training or evaluation.

        Args:
            data (pd.DataFrame): Raw data to prepare.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the scaled sequences and target values.
        """
        scaled_sequences, targets = self._prepare_sequences(data)
        return train_test_split(
            scaled_sequences,
            targets,
            test_size=test_size,
            random_state=random_state,
            shuffle=False,  # 時系列データなのでシャッフルしない
        )

    def load_and_prepare_data_mix(
        self,
        start_datetime: str,
        end_datetime: str,
        coll_type: str,
        test_size=0.2,
        random_state=None,
    ):
        """
        Loads and prepares data from multiple collections for training or evaluation.

        Args:
            start_datetime (str): Start date and time of the data.
            end_datetime (str): End date and time of the data.
            coll_type (str): List of collection types to load data from.
            test_size (float, optional): Proportion of data to use for testing. Defaults to 0.2.
            random_state (int, optional): Random seed for data splitting. Defaults to None.

        Returns:
            tuple: A tuple containing the training and testing data.
        """
        from aiml.aiml_comm import load_data

        data = load_data(self.data_loader, start_datetime, end_datetime, coll_type)

        x, y = self._prepare_sequences(data)
        return train_test_split(
            x, y, test_size=test_size, random_state=random_state, shuffle=False
        )

    def _process_timestamp_and_cyclical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses data by converting datetime strings to Unix timestamps and adding sine wave features for hour, day of week, and month.

        Args:
            data (pd.DataFrame): DataFrame to preprocess.

        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        # datetime オブジェクトに変換
        data["date"] = pd.to_datetime(data["date"])

        # Unix タイムスタンプに変換
        data["date"] = data["date"].astype(np.int64) // 10**9

        # 時間帯を計算
        data["hour"] = (data["date"] // 3600) % 24
        data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
        data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)

        # 曜日を計算
        data["day_of_week"] = (data["date"] // (3600 * 24)) % 7
        data["day_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
        data["day_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)

        # 月を計算
        data["month"] = pd.to_datetime(data["date"], unit="s").dt.month
        data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
        data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)

        return data

    def _prepare_sequences(self, data):
        """
        Prepares the data for training or evaluation.

        Scales the data and creates sequences of time series data with corresponding targets.

        Args:
            data (pd.DataFrame): Raw data to prepare.

        Returns:
            tuple: A tuple containing the scaled sequences and target values.
        """
        time_period = self.model_param.TIME_SERIES_PERIOD  # Get the time series period
        forecast_horizon = self.prediction_distance  # Forecast horizon

        # Convert datetime strings to unix timestamps only if 'date' is in feature columns
        if "date" in self.feature_columns:
            data = self._process_timestamp_and_cyclical_features(data)

        def sequence_generator():
            for i in range(len(data) - (time_period + forecast_horizon)):
                sequence = data.iloc[
                    i : i + time_period, data.columns.get_indexer(self.feature_columns)
                ].values

                # 未来の価格が現在の価格を上回るかどうかで 0/1 を決定
                target = int(
                    data.iloc[i + time_period + forecast_horizon - 1][self.target_column[1]]
                    > data.iloc[i + time_period - 1][self.target_column[0]]
                )
                yield sequence, target

        sequences = []
        targets = []

        # シーケンスを作成
        for seq, target in sequence_generator():
            scaled_seq = self.scaler.fit_transform(seq)
            sequences.append(scaled_seq)
            targets.append(target)

        sequences = np.array(sequences)
        targets = np.array(targets)
        return sequences, targets

    def create_cnn_transformer_model(
        self,
        input_shape,
        num_heads=24,  # Number of attention heads
        dff=256,
        rate=0.1,
        l2_reg=0.01,
        num_transformer_blocks=6,  # Number of Transformer blocks
        num_filters=128,
        kernel_size=6,
        pool_size=4,
    ):
        """
        Creates a model combining CNN and Transformer.

        Args:
            input_shape (tuple): Shape of input data.
            num_heads (int, optional): Number of attention heads. Defaults to 24.
            dff (int, optional): Dimensionality of the feedforward network. Defaults to 256.
            rate (float, optional): Dropout rate. Defaults to 0.1.
            l2_reg (float, optional): L2 regularization coefficient. Defaults to 0.01.
            num_transformer_blocks (int, optional): Number of Transformer blocks. Defaults to 6.
            num_filters (int, optional): Number of filters in convolutional layers. Defaults to 128.
            kernel_size (int, optional): Kernel size for convolutional layers. Defaults to 6.
            pool_size (int, optional): Pool size for max pooling layers. Defaults to 4.

        Returns:
            tf.keras.Model: Created CNN-Transformer model.
        """
        inputs = Input(shape=input_shape)
        x = Conv1D(num_filters, kernel_size, activation="relu", padding="same")(inputs)
        x = MaxPooling1D(pool_size)(x)
        x = Conv1D(num_filters * 2, kernel_size, activation="relu", padding="same")(x)
        x = MaxPooling1D(pool_size)(x)

        # Transformer part
        x = LayerNormalization(epsilon=1e-6)(x)  # Add LayerNormalization
        for _ in range(num_transformer_blocks):
            x = TransformerBlock(x.shape[-1], num_heads, dff, rate, l2_reg=l2_reg)(x)

        # Output layer
        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.2)(x)
        x = Dense(160, activation="relu", kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(0.2)(x)
        x = Dense(80, activation="relu", kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(0.2)(x)
        x = Dense(40, activation="relu", kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation="relu", kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation="relu", kernel_regularizer=l2(l2_reg))(x)
        outputs = Dense(1, activation="sigmoid", kernel_regularizer=l2(l2_reg))(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def create_transformer_model(
        self,
        input_shape,
        num_heads=16,
        dff=256,
        rate=0.1,
        l2_reg=0.01,
        num_transformer_blocks=3,
    ):
        """
        Creates a Transformer model.

        Args:
            input_shape (tuple): Shape of input data.
            num_heads (int): Number of heads in the attention mechanism.
            dff (int): Dimensionality of the feedforward network.
            rate (float): Dropout rate.
            l2_reg (float): L2 regularization coefficient.

        Returns:
            tf.keras.Model: Created model.
        """
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = TransformerBlock(input_shape[1], num_heads, dff, rate, l2_reg=l2_reg)(x)

        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.2)(x)
        x = Dense(80, activation="relu", kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(0.2)(x)
        x = Dense(40, activation="relu", kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation="relu", kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation="relu", kernel_regularizer=l2(l2_reg))(x)
        x = Flatten()(x)

        outputs = Dense(1, activation="sigmoid", kernel_regularizer=l2(l2_reg))(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def train(self, x_train, y_train):
        """
        Trains the model.

        Args:
            x_train (np.array): Training data.
            y_train (np.array): Labels for training data.
        """
        epochs = self.model_param.PARAM_EPOCHS
        batch_size = self.model_param.BATCH_SIZE
        param_learning_rate = self.model_param.ROLLING_PARAM_LEARNING_RATE

        self.model = self.create_cnn_transformer_model(
            (x_train.shape[1], x_train.shape[2])
        )

        self.model.compile(
            optimizer=Adam(learning_rate=param_learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    def train_with_cross_validation(
        self, data: np.ndarray, targets: np.ndarray
    ) -> list:
        """
        Trains the model using TimeSeriesSplit for cross-validation (time series data).

        Args:
            data (np.ndarray): Dataset used for training.
            targets (np.ndarray): Target values corresponding to the dataset.

        Returns:
            list: Model performance evaluation results for the test data in each split.
        """
        epochs = self.model_param.PARAM_EPOCHS
        batch_size = self.model_param.BATCH_SIZE
        n_splits = self.model_param.N_SPLITS
        param_learning_rate = self.model_param.ROLLING_PARAM_LEARNING_RATE

        # ---- ここを KFold から TimeSeriesSplit に変更 ----
        tscv = TimeSeriesSplit(n_splits=n_splits)

        fold_no = 1
        scores = []

        for train_idx, test_idx in tscv.split(data, targets):
            # Generate the model
            self.model = self.create_cnn_transformer_model(
                (data.shape[1], data.shape[2])
            )

            # Compile the model
            self.model.compile(
                optimizer=Adam(learning_rate=param_learning_rate),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            # Train the model
            self.logger.log_system_message(f"Training for fold {fold_no} ...")
            self.model.fit(
                data[train_idx],
                targets[train_idx],
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
            )

            # Evaluate the model's performance
            score = self.model.evaluate(data[test_idx], targets[test_idx], verbose=0)
            scores.append(score)
            fold_no += 1

        return scores

    def evaluate(
        self, x_test: np.ndarray, y_test: np.ndarray
    ) -> Tuple[float, str, np.ndarray]:
        """
        Evaluates the model on the test dataset.

        Args:
            x_test (np.ndarray): Test dataset.
            y_test (np.ndarray): Correct labels for the test dataset.

        Returns:
            Tuple[float, str, np.ndarray]: Model accuracy, classification report, confusion matrix.
        """
        print(f"evaluate - x_test.shape: {x_test.shape}")  # デバッグ出力
        y_pred = (self.model.predict(x_test) > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        return accuracy, report, conf_matrix

    def predict(self, data: np.ndarray, use_gpu: bool = None) -> np.ndarray:
        """
        Makes predictions for the specified data.

        Args:
            data (np.ndarray): Data to predict for.
            use_gpu (bool, optional): Whether to use GPU for this specific prediction.
                                      If None, uses the model's default setting.

        Returns:
            np.ndarray: Prediction results.
        """
        if use_gpu is not None:
            # 一時的にGPU設定を変更
            original_devices = tf.config.get_visible_devices()
            self._configure_gpu(use_gpu)

        predictions = self.model.predict(data)

        if use_gpu is not None:
            # GPU設定を元に戻す
            tf.config.set_visible_devices(original_devices)

        return predictions

    def predict_single(self, data_point: np.ndarray, use_gpu: bool = None) -> int:
        """
        Makes a prediction for a single data point.

        Args:
            data_point (np.ndarray): Data point to predict for.
            use_gpu (bool, optional): Whether to use GPU for this specific prediction.
                                      If None, uses the model's default setting.

        Returns:
            int: Predicted class label.
        """
        prediction = self.predict_single_res(data_point, use_gpu)
        prediction = (prediction > 0.5).astype(int)
        return prediction

    def predict_single_res(self, data_point: np.ndarray, use_gpu: bool = None) -> float:
        """
        Makes a prediction for a single data point and returns the raw result.

        Args:
            data_point (np.ndarray): Data point to predict for.
            use_gpu (bool, optional): Whether to use GPU for this specific prediction.
                                      If None, uses the model's default setting.

        Returns:
            float: Raw prediction value.
        """
        scaled_data_point = self.scaler.fit_transform(data_point)
        # Reshape the data point and make prediction
        reshaped_data = scaled_data_point.reshape(1, -1, len(self.feature_columns))

        if use_gpu is not None:
            # 一時的にGPU設定を変更
            original_devices = tf.config.get_visible_devices()
            self._configure_gpu(use_gpu)

        prediction = self.model.predict(reshaped_data)

        if use_gpu is not None:
            # GPU設定を元に戻す
            tf.config.set_visible_devices(original_devices)

        return prediction[0][0]

    def save_model(self, filename=None):
        """
        Saves the trained model and scaler to files.
        """
        if filename is not None:
            self.filename = filename

        model_file_name = self.filename + ".keras"
        model_path = os.path.join(self.datapath, model_file_name)
        self.logger.log_system_message(f"Saving model to {model_path}")
        self.model.save(model_path)

        # Save the scaler
        model_scaler_file = self.filename + ".scaler"
        model_scaler_path = os.path.join(self.datapath, model_scaler_file)
        self.logger.log_system_message(f"Saving scaler to {model_scaler_path}")
        joblib.dump(self.scaler, model_scaler_path)

    def load_model(self, filename=None):
        """
        Loads the saved model and scaler.
        """
        if filename is not None:
            self.filename = filename

        self.model = ModelManager.load_model(self.filename, self.datapath, self.logger)

        # Load the scaler
        model_scaler_file = self.filename + ".scaler"
        model_scaler_path = os.path.join(self.datapath, model_scaler_file)
        self.logger.log_system_message(f"Loading scaler from {model_scaler_path}")
        self.scaler = joblib.load(model_scaler_path)

    def get_data_period(self, date: str, period: int) -> np.ndarray:
        """
        Gets data for the specified period.

        Args:
            date (str): Start date and time of the period.
            period (int): Length of the period.

        Returns:
            np.ndarray: Data for the specified period.
        """
        data = self.data_loader.load_data_from_datetime_period(
            date, period, self.table_name
        )
        if "date" in data.columns:
            data = self._process_timestamp_and_cyclical_features(data)
        return data.filter(items=self.feature_columns).to_numpy()

    def _configure_gpu(self, use_gpu: bool) -> None:
        """
        Configures GPU usage for the model.
        """
        if not use_gpu:
            tf.config.set_visible_devices([], "GPU")
            self.logger.log_system_message("GPU disabled for inference")
        else:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    self.logger.log_system_message(
                        f"GPU enabled for inference. Available GPUs: {len(gpus)}"
                    )
                except RuntimeError as e:
                    self.logger.log_error(f"GPU configuration error: {str(e)}")
            else:
                self.logger.log_system_message("No GPU available, using CPU instead")


def main():
    from aiml.aiml_comm import COLLECTIONS_TECH
    from common.utils import get_config_model

    data_loader = MongoDataLoader()
    logger = TradingLogger()
    model_id = "rolling_v1"

    config = get_config_model("MODEL_SHORT_TERM", model_id)  # Get the model configuration

    model = TransformerPredictionRollingModel(
        id=model_id,
        config=config,
        data_loader=data_loader,
        logger=logger,
        use_gpu=True
    )

    # 既存の学習済みモデルを読み込む（必要に応じて）
    model.load_model()

    """
    model.set_parameters(
        param_epochs=10,
    )
    # クロスバリデーション例
    x_train, x_test, y_train, y_test = model.load_and_prepare_data_mix(
        "2020-01-01 00:00:00",
        "2024-01-01 00:00:00",
        COLLECTIONS_TECH
    )

    cv_scores = model.train_with_cross_validation(
        np.concatenate((x_train, x_test), axis=0),
        np.concatenate((y_train, y_test), axis=0),
    )

    for i, score in enumerate(cv_scores):
        print(f"Fold {i+1}: Loss = {score[0]}, Accuracy = {score[1]}")
    """

    # 最終評価用
    x_train, x_test, y_train, y_test = model.load_and_prepare_data(
        "2024-08-01 00:00:00",
        "2025-01-05 00:00:00",
        MARKET_DATA_TECH,
        test_size=0.9,
        random_state=None,
    )

    accuracy, report, conf_matrix = model.evaluate(x_test, y_test)
    print(f"Accuracy: {accuracy}")
    print(report)
    print(conf_matrix)

    # モデルを保存
    #model.save_model()


if __name__ == "__main__":
    main()

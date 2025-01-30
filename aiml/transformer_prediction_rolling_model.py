import os
import sys
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Get the absolute path of the current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
PARENT_DIR = os.path.dirname(CURRENT_DIR)
# Add the path of the parent directory to sys.path
sys.path.append(PARENT_DIR)

from common.trading_logger import TradingLogger
from mongodb.data_loader_mongo import MongoDataLoader
from common.utils import get_config
from common.constants import MARKET_DATA_TECH, TIME_SERIES_PERIOD

from aiml.prediction_model import PredictionModel, ModelManager
from aiml.transformerblock import TransformerBlock
from aiml.model_param import ModelParam, BaseModel


class TransformerPredictionRollingModel(BaseModel, PredictionModel):
    """
    Model class that performs rolling prediction using a Transformer.

    Args:
        id (str): Model ID.
        config (Dict[str, Any]): Dictionary of model configuration values.
        data_loader (MongoDataLoader): Data loader instance.
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
        filename (str): File name for saving/loading the model.
        target_column (list): Target column(s).
        scaler (StandardScaler): Instance used for feature scaling.
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
        super().__init__(id, config, data_loader, logger, symbol, interval)
        self._initialize_attributes()
        self._configure_gpu(use_gpu)

    def _initialize_attributes(self) -> None:
        """
        Initializes the model's attributes based on provided config or defaults.
        """
        self.datapath = f"{PARENT_DIR}/{self.config['DATAPATH']}"
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
        Sets or updates the model parameters.

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
        Returns the data loader instance.

        Returns:
            MongoDataLoader: Data loader instance.
        """
        return self.data_loader

    def get_feature_columns(self) -> list:
        """
        Returns the feature columns used by the model.

        Returns:
            list: List of feature columns.
        """
        return self.feature_columns

    def create_table_name(self) -> str:
        """
        Creates or updates the table name for the model.

        Returns:
            str: Created/updated table name.
        """
        self.table_name = f"{self.symbol}_{self.interval}_market_data_tech"
        return self.table_name

    def load_and_prepare_data(
        self,
        start_datetime: str,
        end_datetime: str,
        coll_type: str,
        test_size: float = 0.5,
        random_state: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads and prepares time-series data from the database for training and testing.

        Args:
            start_datetime (str): Start date and time of the data.
            end_datetime (str): End date and time of the data.
            coll_type (str): Collection type to load data from.
            test_size (float, optional): Proportion of data to use for testing.
                                         Defaults to 0.5.
            random_state (int, optional): Random seed for data splitting.
                                          Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Arrays for
            x_train, x_test, y_train, y_test.
        """
        data = self.data_loader.load_data_from_datetime_period(
            start_datetime,
            end_datetime,
            coll_type
        )

        scaled_sequences, targets = self._prepare_sequences(data)
        return train_test_split(
            scaled_sequences,
            targets,
            test_size=test_size,
            random_state=random_state,
            shuffle=False,  # 時系列データなのでシャッフルしない
        )

    def prepare_data(
        self,
        data: pd.DataFrame,
        test_size: float = 0.5,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepares pre-loaded data for training or evaluation.

        Args:
            data (pd.DataFrame): Raw data to prepare.
            test_size (float, optional): Proportion of data to use for testing.
                                         Defaults to 0.5.
            random_state (int, optional): Random seed. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Arrays for
            x_train, x_test, y_train, y_test.
        """
        scaled_sequences, targets = self._prepare_sequences(data)
        return train_test_split(
            scaled_sequences,
            targets,
            test_size=test_size,
            random_state=random_state,
            shuffle=False,
        )

    def load_and_prepare_data_mix(
        self,
        start_datetime: str,
        end_datetime: str,
        coll_type: str,
        test_size: float = 0.2,
        random_state: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads and prepares data from multiple collections for training or evaluation.

        Args:
            start_datetime (str): Start date and time of the data.
            end_datetime (str): End date and time of the data.
            coll_type (str): List or identifier of collection types to load from.
            test_size (float, optional): Proportion of data to use for testing.
                                         Defaults to 0.2.
            random_state (int, optional): Random seed. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Arrays for
            x_train, x_test, y_train, y_test.
        """
        from aiml.aiml_comm import load_data

        data = load_data(self.data_loader, start_datetime, end_datetime, coll_type)
        x, y = self._prepare_sequences(data)
        return train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            shuffle=False
        )

    def _process_timestamp_and_cyclical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses data by converting date to Unix timestamps and adding
        cyclical features (hour, day of week, month).

        Args:
            data (pd.DataFrame): DataFrame to preprocess.

        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        # Convert to datetime
        data["date"] = pd.to_datetime(data["date"])

        # Convert to Unix timestamp
        data["date"] = data["date"].astype(np.int64) // 10**9

        # Compute hour, hour_sin, hour_cos
        data["hour"] = (data["date"] // 3600) % 24
        data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
        data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)

        # Compute day of week, day_sin, day_cos
        data["day_of_week"] = (data["date"] // (3600 * 24)) % 7
        data["day_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
        data["day_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)

        # Compute month, month_sin, month_cos
        data["month"] = pd.to_datetime(data["date"], unit="s").dt.month
        data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
        data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)

        return data

    def _prepare_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepares the data for training or evaluation by creating sequences
        of time-series data and corresponding targets.

        Args:
            data (pd.DataFrame): Raw data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays containing sequences and targets.
        """
        time_period = self.model_param.TIME_SERIES_PERIOD
        forecast_horizon = self.prediction_distance

        # Process time features if 'date' is included in feature columns
        if "date" in self.feature_columns:
            data = self._process_timestamp_and_cyclical_features(data)

        sequences = []
        targets = []

        def _sequence_generator():
            for i in range(len(data) - (time_period + forecast_horizon)):
                seq_x = data.iloc[
                    i : i + time_period,
                    data.columns.get_indexer(self.feature_columns)
                ].values

                # Set target as 1 if future price > current price
                future_price = data.iloc[i + time_period + forecast_horizon - 1][
                    self.target_column[1]
                ]
                current_price = data.iloc[i + time_period - 1][
                    self.target_column[0]
                ]
                yield seq_x, int(future_price > current_price)

        for seq, target in _sequence_generator():
            # Scale each sequence independently
            scaled_seq = self.scaler.fit_transform(seq)
            sequences.append(scaled_seq)
            targets.append(target)

        return np.array(sequences), np.array(targets)

    def create_cnn_transformer_model(
        self,
        input_shape: Tuple[int, int],
        num_heads: int = 24,
        dff: int = 256,
        rate: float = 0.1,
        l2_reg: float = 0.01,
        num_transformer_blocks: int = 6,
        num_filters: int = 128,
        kernel_size: int = 6,
        pool_size: int = 4,
    ) -> tf.keras.Model:
        """
        Creates a model that combines CNN and Transformer blocks.

        Args:
            input_shape (Tuple[int, int]): Shape of input data (timesteps, features).
            num_heads (int, optional): Number of attention heads. Defaults to 24.
            dff (int, optional): Dimensionality of feedforward network. Defaults to 256.
            rate (float, optional): Dropout rate. Defaults to 0.1.
            l2_reg (float, optional): L2 regularization. Defaults to 0.01.
            num_transformer_blocks (int, optional): Number of Transformer blocks. Defaults to 6.
            num_filters (int, optional): Number of convolutional filters. Defaults to 128.
            kernel_size (int, optional): Kernel size for Conv1D. Defaults to 6.
            pool_size (int, optional): Pool size for MaxPooling1D. Defaults to 4.

        Returns:
            tf.keras.Model: Compiled CNN-Transformer model.
        """
        inputs = Input(shape=input_shape)
        x = Conv1D(num_filters, kernel_size, activation="relu", padding="same")(inputs)
        x = MaxPooling1D(pool_size)(x)
        x = Conv1D(num_filters * 2, kernel_size, activation="relu", padding="same")(x)
        x = MaxPooling1D(pool_size)(x)

        # Transformer part
        x = LayerNormalization(epsilon=1e-6)(x)
        for _ in range(num_transformer_blocks):
            x = TransformerBlock(x.shape[-1], num_heads, dff, rate, l2_reg=l2_reg)(x)

        # Output part
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

        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def create_transformer_model(
        self,
        input_shape: Tuple[int, int],
        num_heads: int = 16,
        dff: int = 256,
        rate: float = 0.1,
        l2_reg: float = 0.01,
        num_transformer_blocks: int = 3,
    ) -> tf.keras.Model:
        """
        Creates a pure Transformer model (without CNN layers).

        Args:
            input_shape (Tuple[int, int]): Shape of input data (timesteps, features).
            num_heads (int): Number of heads in multi-head attention.
            dff (int): Dimensionality of the feedforward network.
            rate (float): Dropout rate.
            l2_reg (float): L2 regularization.
            num_transformer_blocks (int): Number of consecutive Transformer blocks.

        Returns:
            tf.keras.Model: Compiled Transformer model.
        """
        inputs = Input(shape=input_shape)
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
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Trains the model on the given dataset.

        Args:
            x_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
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
        self,
        data: np.ndarray,
        targets: np.ndarray
    ) -> list:
        """
        Trains the model using TimeSeriesSplit for cross-validation on time series data.

        Args:
            data (np.ndarray): All features for training.
            targets (np.ndarray): Corresponding labels.

        Returns:
            list: List of (loss, accuracy) tuples for each fold.
        """
        epochs = self.model_param.PARAM_EPOCHS
        batch_size = self.model_param.BATCH_SIZE
        n_splits = self.model_param.N_SPLITS
        param_learning_rate = self.model_param.ROLLING_PARAM_LEARNING_RATE

        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []

        for fold_no, (train_idx, test_idx) in enumerate(tscv.split(data, targets), 1):
            self.model = self.create_cnn_transformer_model(
                (data.shape[1], data.shape[2])
            )
            self.model.compile(
                optimizer=Adam(learning_rate=param_learning_rate),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

            self.logger.log_system_message(f"Training for fold {fold_no} ...")
            self.model.fit(
                data[train_idx],
                targets[train_idx],
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
            )

            score = self.model.evaluate(data[test_idx], targets[test_idx], verbose=0)
            scores.append(score)

        return scores

    def evaluate(
        self,
        x_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[float, str, np.ndarray]:
        """
        Evaluates the model on the test dataset.

        Args:
            x_test (np.ndarray): Test features.
            y_test (np.ndarray): Test labels.

        Returns:
            Tuple[float, str, np.ndarray]: Accuracy, classification report, confusion matrix.
        """
        self.logger.log_system_message(f"Evaluating on x_test with shape {x_test.shape}")
        y_pred = (self.model.predict(x_test) > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        return accuracy, report, conf_matrix

    def predict(self, data: np.ndarray, use_gpu: Optional[bool] = None) -> np.ndarray:
        """
        Predicts on the given data.

        Args:
            data (np.ndarray): Input features to predict on.
            use_gpu (bool, optional): Temporarily enable or disable GPU for this prediction.

        Returns:
            np.ndarray: Prediction scores or probabilities.
        """
        if use_gpu is not None:
            original_devices = tf.config.get_visible_devices()
            self._configure_gpu(use_gpu)

        predictions = self.model.predict(data)

        if use_gpu is not None:
            tf.config.set_visible_devices(original_devices)

        return predictions

    def predict_single(self, data_point: np.ndarray, use_gpu: Optional[bool] = None) -> int:
        """
        Predicts a single data point, returns 0 or 1 class.

        Args:
            data_point (np.ndarray): Single data point (timesteps, features).
            use_gpu (bool, optional): Temporarily enable/disable GPU.

        Returns:
            int: Predicted class label (0 or 1).
        """
        prediction_score = self.predict_single_res(data_point, use_gpu)
        return int(prediction_score > 0.5)

    def predict_single_res(self, data_point: np.ndarray, use_gpu: Optional[bool] = None) -> float:
        """
        Predicts a single data point, returns raw prediction score.

        Args:
            data_point (np.ndarray): Single data point (timesteps, features).
            use_gpu (bool, optional): Temporarily enable/disable GPU.

        Returns:
            float: Prediction score (0.0 ~ 1.0).
        """
        scaled_data_point = self.scaler.fit_transform(data_point)
        reshaped_data = scaled_data_point.reshape(1, -1, len(self.feature_columns))

        if use_gpu is not None:
            original_devices = tf.config.get_visible_devices()
            self._configure_gpu(use_gpu)

        prediction = self.model.predict(reshaped_data)

        if use_gpu is not None:
            tf.config.set_visible_devices(original_devices)

        return float(prediction[0][0])

    def save_model(self, filename: Optional[str] = None) -> None:
        """
        Saves the trained model and scaler.

        Args:
            filename (str, optional): Custom filename override.
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

    def load_model(self, filename: Optional[str] = None) -> None:
        """
        Loads the saved model and scaler.

        Args:
            filename (str, optional): Custom filename override.
        """
        if filename is not None:
            self.filename = filename

        self.model = ModelManager.load_model(self.filename, self.datapath, self.logger)

        # Load the scaler
        model_scaler_file = self.filename + ".scaler"
        model_scaler_path = os.path.join(self.datapath, model_scaler_file)
        self.logger.log_system_message(f"Loading scaler from {model_scaler_path}")
        self.scaler = joblib.load(model_scaler_path)

    def get_data_period(self, date: str, period: int) -> pd.DataFrame:
        """
        Loads data for a specific period starting at a given date.

        Args:
            date (str): Start date/time.
            period (int): Number of records/timesteps to load.

        Returns:
            pd.DataFrame: Data for the specified period.
        """
        data = self.data_loader.load_data_from_point_date(
            date, period, MARKET_DATA_TECH, self.symbol, self.interval
        )
        if "date" in data.columns:
            data = self._process_timestamp_and_cyclical_features(data)
        return data.filter(items=self.feature_columns)

    def _configure_gpu(self, use_gpu: bool) -> None:
        """
        Configures GPU usage.

        Args:
            use_gpu (bool): If False, hides all GPUs. If True, attempts to enable GPU.
        """
        """
        GPUの使用を設定する。
        """
        from aiml.aiml_comm import configure_gpu
        configure_gpu(use_gpu=use_gpu,logger=self.logger)


def transfomer_predictin_trend(date: str, model: TransformerPredictionRollingModel, target_column: str) -> Tuple[float, float]:
    """
    Makes a single forward-looking prediction of upward/downward trend probabilities.

    Args:
        date (str): Datetime string for the data lookup.
        model (TransformerPredictionRollingModel): Pre-initialized model.

    Returns:
        Tuple[float, float]: (Probability that price will go up [%],
                              Probability that price will go down [%])
    """
    df = model.get_data_period(date, TIME_SERIES_PERIOD)
    current_value = df.iloc[-1][target_column]
    pred = model.predict_single_res(df.to_numpy())
    up_pred = pred * 100
    down_pred = (1 - pred) * 100
    return up_pred, down_pred, current_value


def transfomer_predictin_evaluate(
    start_date: str,
    end_date: str,
    model: TransformerPredictionRollingModel
) -> None:
    """
    Evaluates the model performance on the specified date range.

    Args:
        start_date (str): Start date/time.
        end_date (str): End date/time.
        model (TransformerPredictionRollingModel): The model instance to evaluate.
    """
    x_train, x_test, y_train, y_test = model.load_and_prepare_data(
        start_date,
        end_date,
        MARKET_DATA_TECH,
        test_size=0.9,
        random_state=None,
    )
    accuracy, report, conf_matrix = model.evaluate(x_test, y_test)
    print(f"Accuracy: {accuracy:.4f}")
    print(report)
    print(conf_matrix)


def transfomer_predictin_trend_setup(
    model_id: str,
    symbol: str,
    interval: int
) -> TransformerPredictionRollingModel:
    """
    Sets up a TransformerPredictionRollingModel for trend predictions.

    Args:
        model_id (str): The model ID to load.
        symbol (str): Trading symbol (e.g., 'BTCUSDT').
        interval (int): Interval for the model (e.g., 1440 for daily).

    Returns:
        TransformerPredictionRollingModel: The initialized and loaded model.
    """
    from aiml.aiml_comm import COLLECTIONS_TECH
    from common.utils import get_config_model

    data_loader = MongoDataLoader()
    logger = TradingLogger()
    config = get_config_model("MODEL_SHORT_TERM", model_id)
    model = TransformerPredictionRollingModel(
        id=model_id,
        config=config,
        data_loader=data_loader,
        logger=logger,
        symbol=symbol,
        interval=interval,
        use_gpu=True
    )
    model.load_model()
    return model


def main() -> None:
    import datetime
    """
    Example main function demonstrating how to use the TransformerPredictionRollingModel.
    """



if __name__ == "__main__":
    main()

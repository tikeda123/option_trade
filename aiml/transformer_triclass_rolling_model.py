import os
import sys
from typing import Tuple
import numpy as np
import tensorflow as tf
import pandas as pd
from typing import Dict, Any, Optional

from sklearn.model_selection import train_test_split, KFold
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
        """
        return self.data_loader

    def get_feature_columns(self) -> list:
        """
        Gets the feature columns used.
        """
        return self.feature_columns

    def create_table_name(self) -> str:
        """
        Creates the table name.
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
        """
        data = self.data_loader.load_data_from_datetime_period(
            start_datetime, end_datetime, coll_type
        )
        print(data)
        # Prepare sequences and ensure correct shapes
        scaled_sequences, targets = self._prepare_sequences(data)
        return train_test_split(
            scaled_sequences,
            targets,
            test_size=test_size,
            random_state=random_state,
            shuffle=False,
        )

    def prepare_data(self, data: pd.DataFrame, test_size=0.5, random_state=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepares the data for training or evaluation.
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
        test_size=0.2,
        random_state=None,
    ):
        """
        Loads and prepares data from multiple collections for training or evaluation.
        """
        from aiml.aiml_comm import load_data

        data = load_data(self.data_loader, start_datetime, end_datetime, coll_type)

        x, y = self._prepare_sequences(data)
        return train_test_split(
            x, y, test_size=test_size, random_state=random_state, shuffle=False
        )

    def _process_timestamp_and_cyclical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses data by converting datetime strings to Unix timestamps
        and adding sine/cosine features (hour, day_of_week, month).
        """
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
    '''
    def _prepare_sequences(self, data):
        time_period = self.model_param.TIME_SERIES_PERIOD
        forecast_horizon = self.prediction_distance

        # ---- (1) まずは全て数値化 or 必要な列だけ抽出 ----
        # date カラムを周期特徴量に変換するのであれば、学習前に1回だけ実施:
        if 'date' in data.columns:
            data = self._process_timestamp_and_cyclical_features(data)

        all_feature_data = data[self.feature_columns].to_numpy()

        # 1回だけ fit
        self.scaler.fit(all_feature_data)  # 学習データ全体でスケールを合わせる

        sequences = []
        targets = []

        def sequence_generator():
            for i in range(len(data) - (time_period + forecast_horizon)):
                seq = data.iloc[
                    i : i + time_period,
                    data.columns.get_indexer(self.feature_columns)
                ].values

                # スライスしたシーケンスを transform のみ
                scaled_seq = self.scaler.transform(seq)

                # ターゲット列
                current_price = data.iloc[i + time_period - 1][self.target_column[0]]
                future_price = data.iloc[i + time_period + forecast_horizon - 1][self.target_column[1]]

                alpha = 0.008
                diff_ratio = (future_price - current_price) / (current_price if current_price != 0 else 1e-6)
                if diff_ratio > alpha:
                    target = 2  # 上昇
                elif diff_ratio < -alpha:
                    target = 0  # 下降
                else:
                    target = 1  # 変わらない
                yield scaled_seq, target

        for seq, target in sequence_generator():
            sequences.append(seq)
            targets.append(target)

        return np.array(sequences), np.array(targets)
    '''
    def _prepare_sequences(self, data):
        """
        Prepares the data for training or evaluation.

        **三値分類**に変更:
        - しきい値を超えれば「上昇(2)」
        - 下回れば「下降(0)」
        - それ以外は「変わらない(1)」
        """
        time_period = self.model_param.TIME_SERIES_PERIOD
        forecast_horizon = self.prediction_distance

        # Convert datetime if needed
        """
        if 'date' in self.feature_columns:
            data = self._process_timestamp_and_cyclical_features(data)
        """

        # 価格差が ±threshold を超えるかどうかで分類
        threshold = 50.0  # 例: 価格差が50USD以内なら「変わらない」に分類

        def sequence_generator():
            for i in range(len(data) - (time_period + forecast_horizon)):
                sequence = data.iloc[
                    i : i + time_period,
                    data.columns.get_indexer(self.feature_columns)
                ].values

                # 現在価格と未来価格
                current_price = data.iloc[i + time_period - 1][self.target_column[0]]
                future_price = data.iloc[i + time_period + forecast_horizon - 1][self.target_column[1]]

                alpha = 0.008
                diff_ratio = (future_price - current_price) / (current_price if current_price != 0 else 1e-6)
                if diff_ratio > alpha:
                    target = 2  # 上昇
                elif diff_ratio < -alpha:
                    target = 0  # 下降
                else:
                    target = 1  # 変わらない

                yield sequence, target

        sequences = []
        targets = []
        for seq, target in sequence_generator():
            # fit_transform() を毎回呼ぶのは本来非推奨だがサンプルなので割愛
            scaled_seq = self.scaler.fit_transform(seq)
            sequences.append(scaled_seq)
            targets.append(target)

        sequences = np.array(sequences)
        targets = np.array(targets)
        return sequences, targets


    def create_cnn_transformer_model(
        self,
        input_shape,
        num_heads=24,
        dff=256,
        rate=0.1,
        l2_reg=0.01,
        num_transformer_blocks=6,
        num_filters=128,
        kernel_size=6,
        pool_size=4,
    ):
        """
        Creates a model combining CNN and Transformer.
        (三値分類用: 出力を3, softmax)
        """
        inputs = Input(shape=input_shape)
        x = Conv1D(num_filters, kernel_size, activation="relu", padding="same")(inputs)
        x = MaxPooling1D(pool_size)(x)
        x = Conv1D(num_filters * 2, kernel_size, activation="relu", padding="same")(x)
        x = MaxPooling1D(pool_size)(x)

        x = LayerNormalization(epsilon=1e-6)(x)
        for _ in range(num_transformer_blocks):
            x = TransformerBlock(x.shape[-1], num_heads, dff, rate, l2_reg=l2_reg)(x)

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

        # 三値分類 → 出力は3ユニット + softmax
        outputs = Dense(3, activation="softmax", kernel_regularizer=l2(l2_reg))(x)

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
        (こちらを使う場合も同様に三値化する)
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

        # 三値分類 → 出力は3ユニット + softmax
        outputs = Dense(3, activation="softmax", kernel_regularizer=l2(l2_reg))(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def train(self, x_train, y_train):
        """
        Trains the model (三値分類版).
        """
        epochs = self.model_param.PARAM_EPOCHS
        batch_size = self.model_param.BATCH_SIZE
        param_learning_rate = self.model_param.ROLLING_PARAM_LEARNING_RATE

        # CNN + Transformer model
        self.model = self.create_cnn_transformer_model(
            (x_train.shape[1], x_train.shape[2])
        )

        self.model.compile(
            optimizer=Adam(learning_rate=param_learning_rate),
            loss="categorical_crossentropy",  # 三値分類
            metrics=["accuracy"],
        )

        # ラベルを one-hot ベクトルに変換
        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=3)

        self.model.fit(x_train, y_train_cat, epochs=epochs, batch_size=batch_size)

    def train_with_cross_validation(self, data: np.ndarray, targets: np.ndarray) -> list:
        """
        Trains the model using K-Fold cross-validation (三値分類版).
        """
        epochs = self.model_param.PARAM_EPOCHS
        batch_size = self.model_param.BATCH_SIZE
        n_splits = self.model_param.N_SPLITS
        param_learning_rate = self.model_param.ROLLING_PARAM_LEARNING_RATE

        kfold = KFold(n_splits=n_splits, shuffle=True)
        fold_no = 1
        scores = []

        for train_idx, test_idx in kfold.split(data, targets):
            # 新しいモデル生成
            self.model = self.create_cnn_transformer_model((data.shape[1], data.shape[2]))

            self.model.compile(
                optimizer=Adam(learning_rate=param_learning_rate),
                loss="categorical_crossentropy",  # 三値分類
                metrics=["accuracy"],
            )

            # ラベルを one-hot に
            y_train_cat = tf.keras.utils.to_categorical(targets[train_idx], 3)
            y_test_cat = tf.keras.utils.to_categorical(targets[test_idx], 3)

            self.logger.log_system_message(f"Training for fold {fold_no} ...")
            self.model.fit(data[train_idx], y_train_cat, epochs=epochs, batch_size=batch_size)

            # evaluate() は model.evaluate で返る [loss, accuracy] を使用
            score = self.model.evaluate(data[test_idx], y_test_cat, verbose=0)
            scores.append(score)
            fold_no += 1

        return scores

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, str, np.ndarray]:
        """
        Evaluates the model on the test dataset (三値分類版).
        Returns: (accuracy, classification_report, confusion_matrix)
        """
        print(f"evaluate - x_test.shape: {x_test.shape}")
        # 予測確率ベクトル (samples, 3)
        y_pred_proba = self.model.predict(x_test)
        y_pred = np.argmax(y_pred_proba, axis=1)  # 三値分類 → argmax

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        return accuracy, report, conf_matrix

    def predict(self, data: np.ndarray, use_gpu: bool = None) -> np.ndarray:
        """
        Makes predictions for the specified data (確率ベクトル).
        """
        if use_gpu is not None:
            original_devices = tf.config.get_visible_devices()
            self._configure_gpu(use_gpu)

        # shape: (batch_size, 3)
        predictions = self.model.predict(data)

        if use_gpu is not None:
            tf.config.set_visible_devices(original_devices)

        return predictions

    def predict_single(self, data_point: np.ndarray, use_gpu: bool = None) -> int:
        """
        Makes a prediction for a single data point.
        Returns a class label (0,1,2).
        """
        prediction_vector = self.predict_single_res(data_point, use_gpu)  # shape (3,)
        return int(np.argmax(prediction_vector))

    def predict_single_res(self, data_point: np.ndarray, use_gpu: bool = None) -> np.ndarray:
        """
        data_point: shape (time_steps, n_features)
        time_steps = TIME_SERIES_PERIOD に相当
        n_features = len(self.feature_columns)
        """
        # ★ 修正: 推論時には transform のみ
        scaled_data_point = self.scaler.transform(data_point)  # ← fit_transform → transform

        # shape = (1, time_steps, n_features) にリシェイプ
        reshaped_data = scaled_data_point.reshape(1, -1, len(self.feature_columns))

        if use_gpu is not None:
            original_devices = tf.config.get_visible_devices()
            self._configure_gpu(use_gpu)

        prediction = self.model.predict(reshaped_data)

        if use_gpu is not None:
            tf.config.set_visible_devices(original_devices)

        return prediction[0]  # shape (3,)



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

        model_scaler_file = self.filename + ".scaler"
        model_scaler_path = os.path.join(self.datapath, model_scaler_file)
        self.logger.log_system_message(f"Loading scaler from {model_scaler_path}")
        self.scaler = joblib.load(model_scaler_path)

    def get_data_period(self, date: str, period: int) -> np.ndarray:
        """
        Gets data for the specified period.
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
            tf.config.set_visible_devices([], 'GPU')
            self.logger.log_system_message("GPU disabled for inference")
        else:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    self.logger.log_system_message(f"GPU enabled for inference. Available GPUs: {len(gpus)}")
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

    config = get_config_model("MODEL_SHORT_TERM", model_id)

    model = TransformerPredictionRollingModel(
        id=model_id,
        config=config,
        data_loader=data_loader,
        logger=logger,
        symbol="BTCUSDT",
        interval=1440,
        use_gpu=True
    )

    # 例: epoch=20, k-fold=2など
    model.set_parameters(param_epochs=30, n_splits=2)

    # データ読み込みと分割
    x_train, x_test, y_train, y_test = model.load_and_prepare_data_mix(
        "2020-01-01 00:00:00", "2024-04-01 00:00:00", COLLECTIONS_TECH
    )

    # クロスバリデーション学習
    cv_scores = model.train_with_cross_validation(
        np.concatenate((x_train, x_test), axis=0),
        np.concatenate((y_train, y_test), axis=0),
    )

    for i, score in enumerate(cv_scores):
        # [loss, accuracy] が返る想定
        print(f"Fold {i+1}: Loss = {score[0]:.4f}, Accuracy = {score[1]:.4f}")

    # テスト評価
    accuracy, report, conf_matrix = model.evaluate(x_test, y_test)
    print(f"Accuracy: {accuracy}")
    print(report)
    print(conf_matrix)

    # モデル保存
    model.save_model()

    # 別期間で評価
    x_train2, x_test2, y_train2, y_test2 = model.load_and_prepare_data(
        "2024-04-01 00:00:00",
        "2025-01-01 00:00:00",
        MARKET_DATA_TECH,
        test_size=0.9,
        random_state=None,
    )
    accuracy2, report2, conf_matrix2 = model.evaluate(x_test2, y_test2)
    print(f"Final Accuracy: {accuracy2}")
    print(report2)
    print(conf_matrix2)



if __name__ == "__main__":
    main()

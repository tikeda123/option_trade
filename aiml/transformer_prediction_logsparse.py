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
from aiml.model_param import ModelParam, BaseModel

###############################################################################
# ここから LogSparse Attention / LogSparse TransformerBlock 実装のサンプル
###############################################################################

from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

@register_keras_serializable(package="Custom", name="LogSparseAttention")
class LogSparseAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, l2_reg=0.01, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.l2_reg = l2_reg
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.depth = d_model // num_heads

        self.wq = layers.Dense(d_model, kernel_regularizer=l2(l2_reg))
        self.wk = layers.Dense(d_model, kernel_regularizer=l2(l2_reg))
        self.wv = layers.Dense(d_model, kernel_regularizer=l2(l2_reg))
        self.dense = layers.Dense(d_model, kernel_regularizer=l2(l2_reg))

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "l2_reg": self.l2_reg,
        })
        return config

    def split_heads(self, x, batch_size):
        """
        d_model を num_heads に分割し、(batch_size, num_heads, seq_len, depth) 形状へ変換
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    # 引数を (q, k, v, training=None, mask=None) の順で受け取り、
    # Keras が内部で training, mask を混ぜるときに衝突しないようにする。
    def call(self, q, k, v, training=None, mask=None):
        batch_size = tf.shape(q)[0]

        # 全結合層を通す
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # head に分割
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # --- 以下は元の実装と同じ ---
        seq_len_q = tf.shape(q)[2]
        seq_len_k = tf.shape(k)[2]

        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # ローカル + ログスパース 風の簡易マスク例 (省略可)
        sparse_mask = tf.ones_like(scaled_attention_logits) * (-1e9)
        idx = tf.range(seq_len_k)[tf.newaxis, :]
        idx = tf.broadcast_to(idx, [seq_len_q, seq_len_k])
        row = tf.range(seq_len_q)[:, tf.newaxis]
        row = tf.broadcast_to(row, [seq_len_q, seq_len_k])
        distance = tf.abs(row - idx)

        log_thresholds = [1, 2, 4, 8, 16]
        keep_mask = tf.zeros_like(distance, dtype=tf.bool)
        for t in log_thresholds:
            keep_mask = tf.logical_or(keep_mask, tf.less_equal(distance, t))

        sparse_attention = tf.where(keep_mask, scaled_attention_logits, sparse_mask)

        attention_weights = tf.nn.softmax(sparse_attention, axis=-1)
        output = tf.matmul(attention_weights, v)

        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)
        return output

@register_keras_serializable(package="Custom", name="LogSparseTransformerBlock")
class LogSparseTransformerBlock(tf.keras.layers.Layer):
    """
    ログスパースアテンションを組み込んだ TransformerBlock
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1, l2_reg=0.01, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        self.l2_reg = l2_reg

        self.mha = LogSparseAttention(d_model, num_heads, l2_reg=l2_reg)
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='relu', kernel_regularizer=l2(l2_reg)),
            layers.Dense(d_model, kernel_regularizer=l2(l2_reg))
        ])

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "rate": self.rate,
            "l2_reg": self.l2_reg,
        })
        return config

    # call のシグネチャも「(x, training=None, mask=None)」の形にする
    def call(self, x, training=None, mask=None):
        # self-attention するときは q,k,v = x
        # かつ keyword で明示的に渡す
        attn_output = self.mha(q=x, k=x, v=x, training=training, mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2
###############################################################################
# ここまで LogSparse TransformerBlock の定義
###############################################################################

class TransformerPredictionRollingModel(BaseModel, PredictionModel):
    """
    Model class that performs rolling prediction using Transformer (LogSparse Transformer に置き換え).
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

    def _initialize_attributes(self):
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
        return self.data_loader

    def get_feature_columns(self) -> list:
        return self.feature_columns

    def create_table_name(self) -> str:
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
        data = self.data_loader.load_data_from_datetime_period(
            start_datetime, end_datetime, coll_type
        )

        scaled_sequences, targets = self._prepare_sequences(data)
        return train_test_split(
            scaled_sequences,
            targets,
            test_size=test_size,
            random_state=random_state,
            shuffle=False,
        )

    def prepare_data(self, data: pd.DataFrame, test_size=0.5, random_state=None) -> Tuple[np.ndarray, np.ndarray]:
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
        from aiml.aiml_comm import load_data
        data = load_data(self.data_loader, start_datetime, end_datetime, coll_type)
        x, y = self._prepare_sequences(data)
        return train_test_split(
            x, y, test_size=test_size, random_state=random_state, shuffle=False
        )

    def _process_timestamp_and_cyclical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data['date'] = pd.to_datetime(data['date'])
        data['date'] = data['date'].astype(np.int64) // 10**9

        data['hour'] = (data['date'] // 3600) % 24
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)

        data['day_of_week'] = (data['date'] // (3600 * 24)) % 7
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)

        data['month'] = (pd.to_datetime(data['date'], unit='s').dt.month)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

        return data

    def _prepare_sequences(self, data):
        time_period = self.model_param.TIME_SERIES_PERIOD
        forecast_horizon = self.prediction_distance

        if 'date' in self.feature_columns:
            data = self._process_timestamp_and_cyclical_features(data)

        def sequence_generator():
            for i in range(len(data) - (time_period + forecast_horizon)):
                sequence = data.iloc[
                    i : i + time_period, data.columns.get_indexer(self.feature_columns)
                ].values
                target = int(
                    data.iloc[i + time_period + forecast_horizon - 1][self.target_column[1]]
                    > data.iloc[i + time_period - 1][self.target_column[0]]
                )
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

    ########################################################################
    # CNN + LogSparseTransformerBlock を組み合わせたモデルの例
    ########################################################################
    def create_cnn_transformer_model(
        self,
        input_shape,
        num_heads=24,
        dff=256,
        rate=0.1,
        l2_reg=0.01,
        num_transformer_blocks=6,
        num_filters=144,
        kernel_size=6,
        pool_size=4,
    ):
        """
        Creates a model combining CNN and LogSparse Transformer blocks.
        """
        inputs = Input(shape=input_shape)
        x = Conv1D(num_filters, kernel_size, activation="relu", padding="same")(inputs)
        x = MaxPooling1D(pool_size)(x)
        x = Conv1D(num_filters * 2, kernel_size, activation="relu", padding="same")(x)
        x = MaxPooling1D(pool_size)(x)

        # Transformer part (LogSparse)
        x = LayerNormalization(epsilon=1e-6)(x)
        for _ in range(num_transformer_blocks):
            x = LogSparseTransformerBlock(x.shape[-1], num_heads, dff, rate, l2_reg=l2_reg)(x)

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

    ########################################################################
    # LogSparse Transformer のみのモデルを作る例（必要に応じて使用）
    ########################################################################
    def create_transformer_model(
        self,
        input_shape,
        num_heads=16,
        dff=256,
        rate=0.1,
        l2_reg=0.01,
        num_transformer_blocks=3,
    ):
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = LogSparseTransformerBlock(input_shape[1], num_heads, dff, rate, l2_reg=l2_reg)(x)

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
        epochs = self.model_param.PARAM_EPOCHS
        batch_size = self.model_param.BATCH_SIZE
        param_learning_rate = self.model_param.ROLLING_PARAM_LEARNING_RATE

        # CNN + LogSparseTransformerBlock のモデルを使用
        self.model = self.create_cnn_transformer_model((x_train.shape[1], x_train.shape[2]))

        self.model.compile(
            optimizer=Adam(learning_rate=param_learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    def train_with_cross_validation(self, data: np.ndarray, targets: np.ndarray) -> list:
        epochs = self.model_param.PARAM_EPOCHS
        batch_size = self.model_param.BATCH_SIZE
        n_splits = self.model_param.N_SPLITS
        param_learning_rate = self.model_param.ROLLING_PARAM_LEARNING_RATE

        kfold = KFold(n_splits=n_splits, shuffle=True)
        fold_no = 1
        scores = []

        for train_idx, test_idx in kfold.split(data, targets):
            self.model = self.create_cnn_transformer_model((data.shape[1], data.shape[2]))
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
                batch_size=batch_size
            )
            scores.append(self.model.evaluate(data[test_idx], targets[test_idx], verbose=0))
            fold_no += 1

        return scores

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, str, np.ndarray]:
        print(f"evaluate - x_test.shape: {x_test.shape}")
        y_pred = (self.model.predict(x_test) > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        return accuracy, report, conf_matrix

    def predict(self, data: np.ndarray, use_gpu: bool = None) -> np.ndarray:
        if use_gpu is not None:
            original_devices = tf.config.get_visible_devices()
            self._configure_gpu(use_gpu)

        predictions = self.model.predict(data)

        if use_gpu is not None:
            tf.config.set_visible_devices(original_devices)

        return predictions

    def predict_single(self, data_point: np.ndarray, use_gpu: bool = None) -> int:
        prediction = self.predict_single_res(data_point, use_gpu)
        prediction = (prediction > 0.5).astype(int)
        return prediction

    def predict_single_res(self, data_point: np.ndarray, use_gpu: bool = None) -> float:
        scaled_data_point = self.scaler.fit_transform(data_point)
        reshaped_data = scaled_data_point.reshape(1, -1, len(self.feature_columns))

        if use_gpu is not None:
            original_devices = tf.config.get_visible_devices()
            self._configure_gpu(use_gpu)

        prediction = self.model.predict(reshaped_data)

        if use_gpu is not None:
            tf.config.set_visible_devices(original_devices)

        return prediction[0][0]

    def save_model(self, filename=None):
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
        if filename is not None:
            self.filename = filename

        self.model = ModelManager.load_model(self.filename, self.datapath, self.logger)
        model_scaler_file = self.filename + ".scaler"
        model_scaler_path = os.path.join(self.datapath, model_scaler_file)
        self.logger.log_system_message(f"Loading scaler from {model_scaler_path}")
        self.scaler = joblib.load(model_scaler_path)

    def get_data_period(self, date: str, period: int) -> np.ndarray:
        data = self.data_loader.load_data_from_datetime_period(
            date, period, self.table_name
        )
        if "date" in data.columns:
            data = self._process_timestamp_and_cyclical_features(data)
        return data.filter(items=self.feature_columns).to_numpy()

    def _configure_gpu(self, use_gpu: bool) -> None:
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

    config = get_config_model("MODEL_SHORT_TERM", model_id)  # Get the model configuration



    model = TransformerPredictionRollingModel(
        id=model_id,
        config=config,
        data_loader=data_loader,
        logger=logger,
        use_gpu=True
    )

    #model.set_parameters(
    #    param_epochs=20,
    #    n_splits=2,
    #)
    # 既存モデルをロード
    model.load_model()

    """
    # ログスパースTransformerモデルを訓練したい場合は以下のように
    x_train, x_test, y_train, y_test = model.load_and_prepare_data_mix(
        "2020-01-01 00:00:00", "2024-01-01 00:00:00", COLLECTIONS_TECH
    )

    cv_scores = model.train_with_cross_validation(
        np.concatenate((x_train, x_test), axis=0),
        np.concatenate((y_train, y_test), axis=0),
    )

    for i, score in enumerate(cv_scores):
        print(f"Fold {i+1}: Accuracy = {score[1]}")

    accuracy, report, conf_matrix = model.evaluate(x_test, y_test)
    print(f"Accuracy: {accuracy}")
    print(report)
    print(conf_matrix)

    model.save_model()
    """
    # 最終評価用
    from common.constants import MARKET_DATA_TECH

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


if __name__ == "__main__":
    main()

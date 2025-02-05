import os
import sys
from typing import Any, Dict, Optional, Tuple, List

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
import matplotlib.pyplot as plt
import seaborn as sns

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
from aiml.model_param import BaseModel


######################################################################
# 1. TransformerBlock (アテンションスコア取得機能付き)
######################################################################
class TransformerBlock(tf.keras.layers.Layer):
    """
    Transformer Block that applies Multi-Head Attention + Feed Forward Network.
    MultiHeadAttention の呼び出し時に return_attention_scores=True を指定し、
    直近のアテンションスコアを self._last_attention_scores に保持します。
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        rate: float = 0.1,
        l2_reg: float = 0.01,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            kernel_regularizer=l2(l2_reg),
            bias_regularizer=l2(l2_reg),
        )
        self.feed_forward = tf.keras.Sequential(
            [
                Dense(ff_dim, activation="relu", kernel_regularizer=l2(l2_reg)),
                Dense(embed_dim, kernel_regularizer=l2(l2_reg)),
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

        # 直近のアテンションスコアを保持するための変数
        self._last_attention_scores = None

    def call(self, inputs, training: bool = False):
        # Multi-Head Attention部分 (スコアを受け取るために return_attention_scores=True)
        attn_output, attn_scores = self.att(
            query=inputs,
            value=inputs,
            key=inputs,
            return_attention_scores=True,
            training=training
        )
        self._last_attention_scores = attn_scores  # shape: (batch, num_heads, seq_len, seq_len)

        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ff_output = self.feed_forward(out1)
        ff_output = self.dropout2(ff_output, training=training)
        return self.layernorm2(out1 + ff_output)

    def get_last_attention_scores(self):
        """直近のアテンションスコアを返す"""
        return self._last_attention_scores


######################################################################
# 2. モデルクラス: TransformerPredictionRollingModel
#    evaluateにアテンション可視化の呼び出しを追加
######################################################################
class TransformerPredictionRollingModel(BaseModel, PredictionModel):
    """
    Model class that performs rolling prediction using a Transformer.
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
        start_datetime: str,
        end_datetime: str,
        coll_type: str,
        test_size: float = 0.5,
        random_state: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            shuffle=False,
        )

    def prepare_data(
        self,
        data: pd.DataFrame,
        test_size: float = 0.5,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        data["date"] = pd.to_datetime(data["date"])
        data["date"] = data["date"].astype(np.int64) // 10**9

        data["hour"] = (data["date"] // 3600) % 24
        data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
        data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)

        data["day_of_week"] = (data["date"] // (3600 * 24)) % 7
        data["day_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
        data["day_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)

        data["month"] = pd.to_datetime(data["date"], unit="s").dt.month
        data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
        data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)

        return data

    def _prepare_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        time_period = self.model_param.TIME_SERIES_PERIOD
        forecast_horizon = self.prediction_distance

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
                future_price = data.iloc[i + time_period + forecast_horizon - 1][
                    self.target_column[1]
                ]
                current_price = data.iloc[i + time_period - 1][
                    self.target_column[0]
                ]
                yield seq_x, int(future_price > current_price)

        for seq, target in _sequence_generator():
            scaled_seq = self.scaler.fit_transform(seq)
            sequences.append(scaled_seq)
            targets.append(target)

        return np.array(sequences), np.array(targets)

    ##################################################################
    # モデルの定義: CNN + Transformer Blockを組み合わせた例
    ##################################################################
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
        outputs = Dense(1, activation="sigmoid", kernel_regularizer=l2(l2_reg))(x)

        return tf.keras.Model(inputs=inputs, outputs=outputs)

    ##################################################################
    # TransformerBlock のみの単純なモデル例
    ##################################################################
    def create_transformer_model(
        self,
        input_shape: Tuple[int, int],
        num_heads: int = 16,
        dff: int = 256,
        rate: float = 0.1,
        l2_reg: float = 0.01,
        num_transformer_blocks: int = 3,
    ) -> tf.keras.Model:
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

    ##################################################################
    # モデル学習処理 (train, train_with_cross_validation)
    ##################################################################
    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
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
    ) -> List[Tuple[float, float]]:
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

    ######################################################################
    # evaluateメソッド: オプションで可視化
    ######################################################################
    def evaluate(
        self,
        x_test: np.ndarray,
        y_test: np.ndarray,
        visualize_attention: bool = False,
        sample_index_for_attention: int = 0
    ) -> Tuple[float, str, np.ndarray]:
        """
        Evaluate the model, and optionally visualize attention weights for one sample.
        """
        self.logger.log_system_message(f"Evaluating on x_test with shape {x_test.shape}")
        y_pred = (self.model.predict(x_test) > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # ----- 可視化オプション -----
        if visualize_attention:
            if sample_index_for_attention < 0 or sample_index_for_attention >= len(x_test):
                raise ValueError("sample_index_for_attention is out of range.")

            x_single = x_test[sample_index_for_attention : sample_index_for_attention + 1]
            self.visualize_attention_weights(
                x_sample=x_single,
                block_indices=None,  # Noneなら全ての TransformerBlock を対象にする
                show_feature_attention=True
            )
        # ---------------------------

        return accuracy, report, conf_matrix

    ######################################################################
    # 予測関連 (predict, predict_single, ...)
    ######################################################################
    def predict(self, data: np.ndarray, use_gpu: Optional[bool] = None) -> np.ndarray:
        if use_gpu is not None:
            original_devices = tf.config.get_visible_devices()
            self._configure_gpu(use_gpu)

        predictions = self.model.predict(data)

        if use_gpu is not None:
            tf.config.set_visible_devices(original_devices)

        return predictions

    def predict_single(self, data_point: np.ndarray, use_gpu: Optional[bool] = None) -> int:
        prediction_score = self.predict_single_res(data_point, use_gpu)
        return int(prediction_score > 0.5)

    def predict_single_res(self, data_point: np.ndarray, use_gpu: Optional[bool] = None) -> float:
        scaled_data_point = self.scaler.fit_transform(data_point)
        reshaped_data = scaled_data_point.reshape(1, -1, len(self.feature_columns))

        if use_gpu is not None:
            original_devices = tf.config.get_visible_devices()
            self._configure_gpu(use_gpu)

        prediction = self.model.predict(reshaped_data)

        if use_gpu is not None:
            tf.config.set_visible_devices(original_devices)

        return float(prediction[0][0])

    ######################################################################
    # モデル保存・読み込み
    ######################################################################
    def save_model(self, filename: Optional[str] = None) -> None:
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

    def load_model(self, filename: Optional[str] = None) -> None:
        if filename is not None:
            self.filename = filename

        self.model = ModelManager.load_model(self.filename, self.datapath, self.logger)
        model_scaler_file = self.filename + ".scaler"
        model_scaler_path = os.path.join(self.datapath, model_scaler_file)
        self.logger.log_system_message(f"Loading scaler from {model_scaler_path}")
        self.scaler = joblib.load(model_scaler_path)

    def get_data_period(self, date: str, period: int) -> pd.DataFrame:
        data = self.data_loader.load_data_from_point_date(
            date, period, MARKET_DATA_TECH, self.symbol, self.interval
        )
        if "date" in data.columns:
            data = self._process_timestamp_and_cyclical_features(data)
        return data.filter(items=self.feature_columns)

    def _configure_gpu(self, use_gpu: bool) -> None:
        from aiml.aiml_comm import configure_gpu
        configure_gpu(use_gpu=use_gpu, logger=self.logger)

    ######################################################################
    # 3. アテンション可視化メソッド (visualize_attention_weights)
    ######################################################################
    def visualize_attention_weights(
        self,
        x_sample: np.ndarray,
        block_indices: Optional[List[int]] = None,
        show_feature_attention: bool = False
    ):
        """
        Transformer部分のアテンションウェイトを可視化する例。
        """
        if x_sample.ndim != 3 or x_sample.shape[0] != 1:
            raise ValueError("x_sampleは shape=(1, time_steps, features) で1サンプルのみを与えてください。")

        # 【ポイント】_flatten_layers を使って再帰的にレイヤーを取得し、TransformerBlock を探す
        def get_transformer_blocks(model: tf.keras.Model) -> List[TransformerBlock]:
            """
            Keras の _flatten_layers を使ってすべてのサブレイヤーを探索し、
            TransformerBlock のインスタンスを返す
            """
            blocks = []
            # include_self=False → 自分自身（最上位model）を除いてフラットに展開
            for layer in model._flatten_layers(include_self=False, recursive=True):
                if isinstance(layer, TransformerBlock):
                    blocks.append(layer)
            return blocks

        # 1サンプルで推論してアテンションスコアを更新させる
        _ = self.model.predict(x_sample)

        all_blocks = get_transformer_blocks(self.model)
        if not all_blocks:
            print("TransformerBlockが見つかりませんでした。")
            return

        # block_indices が指定されなければ全ブロックを可視化
        if block_indices is None:
            block_indices = list(range(len(all_blocks)))

        for idx in block_indices:
            if idx < 0 or idx >= len(all_blocks):
                print(f"指定されたブロックインデックス {idx} は存在しません。")
                continue

            block = all_blocks[idx]
            attn_scores = block.get_last_attention_scores()  # shape: (batch, num_heads, seq_len, seq_len) or None

            if attn_scores is None:
                print(f"ブロック {idx} からアテンションスコアが取得できませんでした。")
                continue

            # batch=1前提
            attn_scores = attn_scores[0]  # shape: (num_heads, seq_len, seq_len)
            num_heads = attn_scores.shape[0]

            # ====== 時系列方向のアテンション可視化 ======
            fig, axes = plt.subplots(1, num_heads, figsize=(5 * num_heads, 4))
            if num_heads == 1:
                axes = [axes]

            fig.suptitle(f"Block {idx}: 時系列方向のアテンションマップ")
            for head_i in range(num_heads):
                ax = axes[head_i]
                sns.heatmap(attn_scores[head_i], cmap="viridis", ax=ax)
                ax.set_title(f"Head {head_i}")
                ax.set_xlabel("Time Step")
                ax.set_ylabel("Time Step")
            plt.tight_layout()
            plt.show()

            # ====== 特徴量方向(埋め込み次元)の可視化 (オプション) ======
            if show_feature_attention:
                # サブモデルを作成して、このブロックの出力を取得
                sub_model = tf.keras.Model(
                    inputs=self.model.input,
                    outputs=block.output
                )
                block_output = sub_model.predict(x_sample)  # shape: (1, seq_len, embed_dim)
                block_output = block_output[0]              # shape: (seq_len, embed_dim)

                avg_per_feature = np.mean(block_output, axis=0)  # shape: (embed_dim,)
                plt.figure(figsize=(8, 4))
                plt.bar(range(len(avg_per_feature)), avg_per_feature)
                plt.title(f"Block {idx}: 特徴量方向(埋め込み次元)の平均アクティベーション")
                plt.xlabel("Feature (Embedding Channel)")
                plt.ylabel("Activation (mean over time steps)")
                plt.show()


######################################################################
# 4. 補助関数: 推論、評価用
######################################################################
def transfomer_predictin_trend(
    date: str,
    model: TransformerPredictionRollingModel,
    target_column: str
) -> Tuple[float, float, float]:
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
    """
    Example main function demonstrating how to use the TransformerPredictionRollingModel.
    """
    pass


if __name__ == "__main__":
    main()


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
from sklearn.metrics import mean_absolute_error, mean_squared_error  # 追加

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


#####################################################################
# 追加: pinball loss の定義
#####################################################################
def pinball_loss(q, y_true, y_pred):
    """q: 分位 (0<q<1) の pinball loss"""
    e = y_true - y_pred
    return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))

def pinball_loss_25(y_true, y_pred):
    return pinball_loss(0.25, y_true, y_pred)

def pinball_loss_50(y_true, y_pred):
    return pinball_loss(0.50, y_true, y_pred)

def pinball_loss_75(y_true, y_pred):
    return pinball_loss(0.75, y_true, y_pred)


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
        # Prepare sequences and ensure correct shapes
        (scaled_sequences, y_class, y_reg) = self._prepare_sequences(data)

        # 学習用とテスト用に分割（ここで y_class と y_reg を両方分割する）
        x_train, x_test, y_class_train, y_class_test = train_test_split(
            scaled_sequences, y_class, test_size=test_size, random_state=random_state, shuffle=False
        )
        # 連続値のほうも同じインデックスで分割
        _, _, y_reg_train, y_reg_test = train_test_split(
            scaled_sequences, y_reg, test_size=test_size, random_state=random_state, shuffle=False
        )
        return x_train, x_test, y_class_train, y_class_test, y_reg_train, y_reg_test

    def prepare_data(self, data: pd.DataFrame, test_size=0.5, random_state=None):
        """
        Prepares the data for training or evaluation.
        """
        (scaled_sequences, y_class, y_reg) = self._prepare_sequences(data)

        x_train, x_test, y_class_train, y_class_test = train_test_split(
            scaled_sequences, y_class, test_size=test_size, random_state=random_state, shuffle=False
        )
        _, _, y_reg_train, y_reg_test = train_test_split(
            scaled_sequences, y_reg, test_size=test_size, random_state=random_state, shuffle=False
        )
        return x_train, x_test, y_class_train, y_class_test, y_reg_train, y_reg_test

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
        (scaled_sequences, y_class, y_reg) = self._prepare_sequences(data)

        # Train/test split
        x_train, x_test, y_class_train, y_class_test = train_test_split(
            scaled_sequences, y_class, test_size=test_size, random_state=random_state, shuffle=False
        )
        # 同じインデックスで y_reg を分割
        _, _, y_reg_train, y_reg_test = train_test_split(
            scaled_sequences, y_reg, test_size=test_size, random_state=random_state, shuffle=False
        )
        return x_train, x_test, y_class_train, y_class_test, y_reg_train, y_reg_test

    def _process_timestamp_and_cyclical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses data by converting datetime strings to Unix timestamps and adding
        sine wave features for hour, day of week, and month.
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

    def _prepare_sequences(self, data: pd.DataFrame):
        """
        Prepares the data for training or evaluation.

        Scales the data and creates sequences of time series data with corresponding
        (1) classification targets (0 or 1)
        (2) regression targets (連続値: 今回は例として将来の価格差を用いる)
        """
        time_period = self.model_param.TIME_SERIES_PERIOD  # 短期時系列の長さ
        forecast_horizon = self.prediction_distance       # 未来何ステップ先をターゲットにするか

        # 'date' が特徴量に含まれる場合は、先に周期特徴を展開
        if 'date' in self.feature_columns:
            data = self._process_timestamp_and_cyclical_features(data)

        # 今回は close価格を例示
        # self.target_column が ['close','close'] のように定義されている前提の場合もあるため、
        # [0] 過去, [1] 未来 みたいな利用を想定。
        # 必要に応じてご利用のカラムに合わせて修正してください。
        close_past_col = self.target_column[0]   # 'close'(過去参照)
        close_future_col = self.target_column[1] # 'close'(未来参照)

        sequences = []
        y_class_list = []
        y_reg_list = []

        # fit_transform の都合上、全体に一括で scaler をかけるならここでかける手もある
        # ただしサンプルでは各時系列切片ごとにスケーリングしているので注意
        # 必要に応じて各社の方針に従ってください。

        def sequence_generator():
            # データ数 - (time_period + forecast_horizon) だけループ
            for i in range(len(data) - (time_period + forecast_horizon)):
                # i から i+time_period までを 1つの時系列データに
                seq = data.iloc[i : i + time_period][self.feature_columns].values

                # バイナリ分類ラベル: 未来の価格が上昇なら1, 下降なら0
                future_price = data.iloc[i + time_period + forecast_horizon - 1][close_future_col]
                current_price = data.iloc[i + time_period - 1][close_past_col]
                class_label = 1 if future_price > current_price else 0

                # 回帰ラベル: 未来の価格変化量（ future_price - current_price ）
                reg_label = future_price - current_price

                yield seq, class_label, reg_label

        for seq, class_label, reg_label in sequence_generator():
            # 各シーケンスごとにスケーリング（本サンプルでは簡略的に fit_transform）
            scaled_seq = self.scaler.fit_transform(seq)
            sequences.append(scaled_seq)
            y_class_list.append(class_label)
            y_reg_list.append(reg_label)

        return np.array(sequences), np.array(y_class_list), np.array(y_reg_list)

    #####################################################################
    # 変更ポイント:
    # CNN + Transformer の出力を "4つ" に分割したモデルを作成する
    #####################################################################
    def create_cnn_transformer_model(
        self,
        input_shape,
        num_heads=24,  # Number of attention heads in Transformer16->24
        dff=256,
        rate=0.1,
        l2_reg=0.01,
        num_transformer_blocks=6,  # Number of Transformer blocks4->6
        num_filters=128,  # Number of filters in convolutional layers64->128
        kernel_size=6,    # Kernel size for convolutional layers3->6
        #pool_size=4,      # Pool size for max pooling layers2->4
        pool_size=2,
    ):
        """
        Creates a multi-task model (CNN + Transformer) with 4 outputs:
          - 1 for classification (sigmoid)
          - 3 for quantile regression (linear, e.g. 0.25 / 0.50 / 0.75)
        """
        # CNN part
        inputs = Input(shape=input_shape)
        x = Conv1D(num_filters, kernel_size, activation="relu", padding="same")(inputs)
        x = MaxPooling1D(pool_size)(x)
        x = Conv1D(num_filters * 2, kernel_size, activation="relu", padding="same")(x)
        x = MaxPooling1D(pool_size)(x)

        # Transformer part
        x = LayerNormalization(epsilon=1e-6)(x)  # Add LayerNormalization
        for _ in range(num_transformer_blocks):
            x = TransformerBlock(x.shape[-1], num_heads, dff, rate, l2_reg=l2_reg)(x)

        # 共通の特徴量
        shared_output = GlobalAveragePooling1D()(x)
        shared_output = Dropout(0.2)(shared_output)
        shared_output = Dense(160, activation="relu", kernel_regularizer=l2(l2_reg))(shared_output)
        shared_output = Dropout(0.2)(shared_output)
        shared_output = Dense(80, activation="relu", kernel_regularizer=l2(l2_reg))(shared_output)
        shared_output = Dropout(0.2)(shared_output)
        shared_output = Dense(40, activation="relu", kernel_regularizer=l2(l2_reg))(shared_output)
        shared_output = Dropout(0.2)(shared_output)
        shared_output = Dense(32, activation="relu", kernel_regularizer=l2(l2_reg))(shared_output)
        shared_output = Dropout(0.2)(shared_output)
        shared_output = Dense(16, activation="relu", kernel_regularizer=l2(l2_reg))(shared_output)

        # 出力層を複数に分ける（合計4つ）
        classification_output = Dense(1, activation="sigmoid", name="output_classification")(shared_output)
        quantile_25_output   = Dense(1, activation="linear", name="output_q_25")(shared_output)
        quantile_50_output   = Dense(1, activation="linear", name="output_q_50")(shared_output)
        quantile_75_output   = Dense(1, activation="linear", name="output_q_75")(shared_output)

        model = tf.keras.Model(
            inputs=inputs,
            outputs=[classification_output, quantile_25_output, quantile_50_output, quantile_75_output],
        )
        return model

    #####################################################################
    # 学習メソッドをマルチタスクに対応
    #####################################################################
    def train(self, x_train, y_class_train, y_reg_train):
        """
        Trains the model (multi-task: classification + quantile regression).
        """
        epochs = self.model_param.PARAM_EPOCHS
        batch_size = self.model_param.BATCH_SIZE
        param_learning_rate = self.model_param.ROLLING_PARAM_LEARNING_RATE

        # マルチタスク用モデルを作成
        self.model = self.create_cnn_transformer_model((x_train.shape[1], x_train.shape[2]))

        # コンパイル時に複数の損失を指定
        self.model.compile(
            optimizer=Adam(learning_rate=param_learning_rate),
            loss={
                "output_classification": tf.keras.losses.BinaryCrossentropy(),
                "output_q_25": pinball_loss_25,
                "output_q_50": pinball_loss_50,
                "output_q_75": pinball_loss_75,
            },
            loss_weights={
                "output_classification": 0.5,  # 分類を0.5
                "output_q_25": 0.1667,         # 3つの分位合計で0.5相当など適宜調整
                "output_q_50": 0.1667,
                "output_q_75": 0.1667,
            },
            metrics={
                "output_classification": "accuracy"
            }
        )

        # fit 時に出力を辞書でまとめて渡す
        self.model.fit(
            x_train,
            {
                "output_classification": y_class_train,
                "output_q_25": y_reg_train,
                "output_q_50": y_reg_train,
                "output_q_75": y_reg_train,
            },
            epochs=epochs,
            batch_size=batch_size
        )

    #####################################################################
    # K-Fold 用学習メソッド（単純化のためにマルチタスク対応版に置き換え）
    #####################################################################
    def train_with_cross_validation(self, data: np.ndarray, y_class: np.ndarray, y_reg: np.ndarray) -> list:
        """
        Trains the model using K-Fold cross-validation for multi-task outputs.
        """
        epochs = self.model_param.PARAM_EPOCHS
        batch_size = self.model_param.BATCH_SIZE
        n_splits = self.model_param.N_SPLITS
        param_learning_rate = self.model_param.ROLLING_PARAM_LEARNING_RATE

        kfold = KFold(n_splits=n_splits, shuffle=True)

        fold_no = 1
        scores = []

        for train_index, test_index in kfold.split(data):
            # マルチタスクモデル作成
            self.model = self.create_cnn_transformer_model((data.shape[1], data.shape[2]))
            self.model.compile(
                optimizer=Adam(learning_rate=param_learning_rate),
                loss={
                    "output_classification": tf.keras.losses.BinaryCrossentropy(),
                    "output_q_25": pinball_loss_25,
                    "output_q_50": pinball_loss_50,
                    "output_q_75": pinball_loss_75,
                },
                loss_weights={
                    "output_classification": 0.5,
                    "output_q_25": 0.1667,
                    "output_q_50": 0.1667,
                    "output_q_75": 0.1667,
                },
                metrics={
                    "output_classification": "accuracy"
                }
            )

            self.logger.log_system_message(f"Training for fold {fold_no} ...")

            # 学習
            self.model.fit(
                data[train_index],
                {
                    "output_classification": y_class[train_index],
                    "output_q_25": y_reg[train_index],
                    "output_q_50": y_reg[train_index],
                    "output_q_75": y_reg[train_index],
                },
                epochs=epochs,
                batch_size=batch_size
            )

            # テストセットで推論
            predictions = self.model.predict(data[test_index], verbose=0)
            # predictions は [class_pred, q25_pred, q50_pred, q75_pred] のリスト
            class_pred = (predictions[0] > 0.5).astype(int).flatten()

            # 分類タスクの accuracy
            acc = accuracy_score(y_class[test_index], class_pred)
            scores.append(acc)

            fold_no += 1

        return scores

    #####################################################################
    # 評価メソッドをマルチタスクの出力に対応させる
    # （ここでは分類タスクの結果のみを評価）
    #####################################################################
    def evaluate(self, x_test: np.ndarray, y_class_test: np.ndarray) -> Tuple[float, str, np.ndarray]:
        """
        Evaluates the classification performance on the test dataset.
        (Quantile outputs are not evaluated here, but you can extend as needed.)
        """
        # 推論 (class_pred, q25, q50, q75)
        preds = self.model.predict(x_test)
        class_pred = (preds[0] > 0.5).astype(int).flatten()

        accuracy = accuracy_score(y_class_test, class_pred)
        report = classification_report(y_class_test, class_pred)
        conf_matrix = confusion_matrix(y_class_test, class_pred)
        return accuracy, report, conf_matrix

    #####################################################################
    # 推論系もマルチタスク仕様に対応させる
    #####################################################################
    def predict(self, data: np.ndarray, use_gpu: bool = None):
        """
        Makes predictions for the specified data (multi-output).
        Returns [class_pred, q25_pred, q50_pred, q75_pred].
        """
        if use_gpu is not None:
            original_devices = tf.config.get_visible_devices()
            self._configure_gpu(use_gpu)

        predictions = self.model.predict(data)

        if use_gpu is not None:
            tf.config.set_visible_devices(original_devices)

        return predictions  # リストまたはタプル: [class, q25, q50, q75]

    def predict_single(self, data_point: np.ndarray, use_gpu: bool = None) -> int:
        """
        Makes a binary classification prediction for a single data point.
        """
        preds = self.predict_single_res(data_point, use_gpu)
        # preds は [class_pred, q25_pred, q50_pred, q75_pred]
        class_pred = (preds[0] > 0.5).astype(int)
        return int(class_pred)

    def predict_single_res(self, data_point: np.ndarray, use_gpu: bool = None):
        """
        Returns raw multi-task predictions for a single data point.
        """
        scaled_data_point = self.scaler.fit_transform(data_point)
        reshaped_data = scaled_data_point.reshape(1, -1, len(self.feature_columns))

        if use_gpu is not None:
            original_devices = tf.config.get_visible_devices()
            self._configure_gpu(use_gpu)

        predictions = self.model.predict(reshaped_data)

        if use_gpu is not None:
            tf.config.set_visible_devices(original_devices)

        # predictions は [class_pred(1D), q25(1D), q50(1D), q75(1D)] の各要素が shape=(1,1)
        # ここでは [class_val, q25_val, q50_val, q75_val] のスカラー配列を返す
        return [p[0][0] for p in predictions]

    #####################################################################
    # モデル保存・読み込み (変更なし)
    #####################################################################
    def save_model(self, filename=None):
        """
        Saves the trained model and scaler to files.
        """
        if filename is not None:
            self.filename = filename

        # Save the model
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

    # 必要があれば別のデータを再度ロードして評価するなど
    # ...



def main():
    from aiml.aiml_comm import COLLECTIONS_TECH
    from common.utils import get_config_model

    data_loader = MongoDataLoader()
    logger = TradingLogger()
    model_id = "rolling_v11"

    config = get_config_model("MODEL_MIDDLE_TERM", model_id)
    model = TransformerPredictionRollingModel(
        id=model_id,
        config=config,
        data_loader=data_loader,
        logger=logger,
        use_gpu=True,
    )

    model.load_model()
    # パラメータ設定（例）
    model.set_parameters(param_epochs=3, n_splits=2, time_series_period=10)

    # データのロード＆準備
    x_train, x_test, y_class_train, y_class_test, y_reg_train, y_reg_test = model.load_and_prepare_data_mix(
        "2020-01-01 00:00:00", "2024-08-01 00:00:00", COLLECTIONS_TECH
    )

    """
    #####################################################################
    # 1) クロスバリデーションによる汎化性能チェック
    #####################################################################
    data_all = np.concatenate([x_train, x_test], axis=0)
    class_all = np.concatenate([y_class_train, y_class_test], axis=0)
    reg_all = np.concatenate([y_reg_train, y_reg_test], axis=0)
    cv_scores = model.train_with_cross_validation(data_all, class_all, reg_all)

    print("Cross-validation results:")
    for i, acc in enumerate(cv_scores):
        print(f"  Fold {i+1}: Accuracy = {acc}")

    #####################################################################
    # 2) 最終モデルを x_train で学習＆ x_test で評価（分類タスク）
    #####################################################################
    model.train(x_train, y_class_train, y_reg_train)
    accuracy, report, conf_matrix = model.evaluate(x_test, y_class_test)
    print("=== Classification Evaluation ===")
    print(f"Accuracy: {accuracy}")
    print(report)
    print(conf_matrix)
    """

    #####################################################################
    # 3) クォンタイル回帰タスクの評価: Coverage / Pinball Loss / MAE 等
    #####################################################################
    # テストデータでの予測(4出力)
    preds = model.predict(x_test)
    class_pred = preds[0].flatten()
    q25_pred   = preds[1].flatten()
    q50_pred   = preds[2].flatten()
    q75_pred   = preds[3].flatten()

    # --- Coverage(キャリブレーション)計測 ---
    coverage_25 = np.mean(y_reg_test < q25_pred)
    coverage_50 = np.mean(y_reg_test < q50_pred)
    coverage_75 = np.mean(y_reg_test < q75_pred)

    print("\n=== Coverage (Calibration) ===")
    print(f" q=0.25 -> coverage: {coverage_25:.4f} (理想=0.25)")
    print(f" q=0.50 -> coverage: {coverage_50:.4f} (理想=0.50)")
    print(f" q=0.75 -> coverage: {coverage_75:.4f} (理想=0.75)")

    # --- ピンボールロスの計算 ---
    q25_loss = pinball_loss_25(y_reg_test, q25_pred).numpy()
    q50_loss = pinball_loss_50(y_reg_test, q50_pred).numpy()
    q75_loss = pinball_loss_75(y_reg_test, q75_pred).numpy()

    print("\n=== Pinball Loss ===")
    print(f" q=0.25 -> {q25_loss:.4f}")
    print(f" q=0.50 -> {q50_loss:.4f}")
    print(f" q=0.75 -> {q75_loss:.4f}")

    # --- 参考: q=0.50 に対する回帰指標(MAE, RMSE) ---
    mae_50 = mean_absolute_error(y_reg_test, q50_pred)
    mse_50 = mean_squared_error(y_reg_test, q50_pred)
    rmse_50 = np.sqrt(mse_50)

    print("\n=== Regression Metrics for q=0.50 (Median) ===")
    print(f" MAE : {mae_50:.4f}")
    print(f" RMSE: {rmse_50:.4f}")

    #####################################################################
    # モデル保存など
    #####################################################################
    #model.save_model()


if __name__ == "__main__":
    main()


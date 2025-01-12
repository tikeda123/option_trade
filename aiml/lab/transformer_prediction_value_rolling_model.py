import os
import sys
from typing import Tuple
import numpy as np
import tensorflow as tf
import pandas as pd
from typing import Dict, Any, Optional
import gc  # ガーベジコレクション用

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import joblib
import matplotlib.pyplot as plt

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
from common.constants import MARKET_DATA_TECH

from aiml.prediction_model import PredictionModel, ModelManager
from aiml.transformerblock import TransformerBlock
from aiml.model_param import ModelParam, BaseModel

# === Optuna インポート ===
import optuna


class TransformerPredictionRollingModel(BaseModel, PredictionModel):
    """
    BTCなどの価格を予測する回帰モデル (CNN + Transformer)。
    評価指標を追加し、予測可視化の仕組みを実装。

    【本コードの修正点】
    1. GPUメモリの動的確保 (memory growth) を有効化
    2. hyperparameter_search_optuna() 内で、Trial終了時に tf.keras.backend.clear_session() 等を呼び、
       GPUメモリを解放
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
        self._configure_gpu(use_gpu)  # GPU設定 (Memory Growth等)

    def _initialize_attributes(self):
        self.datapath = f"{parent_dir}/{self.config['DATAPATH']}"
        self.feature_columns = self.config["FEATURE_COLUMNS"]
        self.target_column = self.config["TARGET_COLUMN"]  # 例: 'close'
        self.prediction_distance = self.config["PREDICTION_DISTANCE"]
        self.filename = self.config["MODLE_FILENAME"]

        # 特徴量スケーラー & ターゲットスケーラー
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        self.table_name = f"{self.symbol}_{self.interval}"
        self.model = None

    def _configure_gpu(self, use_gpu: bool) -> None:
        """
        GPUメモリ関連の設定を行う。メモリの動的確保(Memory Growth)等。
        """
        if not use_gpu:
            tf.config.set_visible_devices([], "GPU")
            self.logger.log_system_message("GPU disabled for inference")
        else:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                try:
                    for gpu in gpus:
                        # メモリを必要時に確保し、使い終わった領域を再利用しやすくする
                        tf.config.experimental.set_memory_growth(gpu, True)
                    self.logger.log_system_message(
                        f"GPU enabled for inference. GPUs: {len(gpus)}"
                    )
                except RuntimeError as e:
                    self.logger.log_error(f"GPU configuration error: {str(e)}")
            else:
                self.logger.log_system_message("No GPU available, using CPU instead")

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

        # 回帰タスクのためpositive_thresholdは使わないが、念のため存置
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

    def prepare_data(self, data: pd.DataFrame, test_size=0.5, random_state=None):
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

        if isinstance(self.target_column, list):
            # 複数カラム指定の場合は先頭だけを利用 (例)
            target_col = self.target_column[0]
        else:
            target_col = self.target_column

        data = data.dropna(subset=self.feature_columns + [target_col])

        feature_df = data[self.feature_columns].copy()
        target_df = data[[target_col]].copy()

        features_scaled = self.scaler.fit_transform(feature_df)
        target_scaled = self.target_scaler.fit_transform(target_df)

        sequences = []
        targets = []
        total_len = len(data)

        for i in range(total_len - (time_period + forecast_horizon)):
            seq_x = features_scaled[i : i + time_period, :]
            t_index = i + time_period + forecast_horizon - 1
            seq_y = target_scaled[t_index, 0]

            sequences.append(seq_x)
            targets.append(seq_y)

        sequences = np.array(sequences)  # (サンプル数, time_period, 特徴数)
        targets = np.array(targets)      # (サンプル数, )
        return sequences, targets

    # === この部分がCNN+Transformerモデル作成 ===
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

        outputs = Dense(1, activation=None, kernel_regularizer=l2(l2_reg))(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    # === Optunaによるハイパーパラメータ探索 (GPUメモリ開放対策含む) ===
    def hyperparameter_search_optuna(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        n_trials: int = 20,
        timeout: int = 3600,
        validation_split: float = 0.2,
        max_epochs: int = 10
    ) -> Dict[str, Any]:
        """
        Optuna を使って、Transformer Block関連のハイパーパラメータを探索する。
        x_train, y_train をさらに train/val に分割して最適パラメータを見つける。
        """

        # train/val 分割
        val_size = int(len(x_train) * validation_split)
        x_train_sub, x_val_sub = x_train[:-val_size], x_train[-val_size:]
        y_train_sub, y_val_sub = y_train[:-val_size], y_train[-val_size:]

        def objective(trial):
            model = None
            try:
                # 1. ハイパーパラメータのサンプリング
                num_heads = trial.suggest_int("num_heads", 2, 32, step=2)
                dff = trial.suggest_int("dff", 64, 512, step=64)
                num_transformer_blocks = trial.suggest_int("num_blocks", 1, 4)
                l2_reg = trial.suggest_loguniform("l2_reg", 1e-5, 1e-1)
                dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.1)

                num_filters = trial.suggest_int("num_filters", 64, 256, step=64)
                kernel_size = trial.suggest_int("kernel_size", 3, 9, step=2)

                # 2. モデル構築
                model = self.create_cnn_transformer_model(
                    input_shape=(x_train.shape[1], x_train.shape[2]),
                    num_heads=num_heads,
                    dff=dff,
                    rate=dropout_rate,
                    l2_reg=l2_reg,
                    num_transformer_blocks=num_transformer_blocks,
                    num_filters=num_filters,
                    kernel_size=kernel_size,
                )

                # 3. コンパイル
                model.compile(
                    optimizer=Adam(learning_rate=self.model_param.ROLLING_PARAM_LEARNING_RATE),
                    loss="mean_squared_error",
                    metrics=["mean_absolute_error"],
                )

                # 4. 学習 (短めのエポック数)
                history = model.fit(
                    x_train_sub,
                    y_train_sub,
                    epochs=max_epochs,
                    batch_size=self.model_param.BATCH_SIZE,
                    validation_data=(x_val_sub, y_val_sub),
                    verbose=0
                )

                # 5. バリデーション損失
                val_loss = history.history["val_loss"][-1]
                return val_loss

            finally:
                # 6. GPUメモリ開放
                if model is not None:
                    del model
                tf.keras.backend.clear_session()
                gc.collect()

        # Optuna のStudyを作成し、試行開始
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        self.logger.log_system_message("Optuna hyperparameter search finished.")
        self.logger.log_system_message(f"Best trial MSE: {study.best_value}")
        self.logger.log_system_message(f"Best params: {study.best_params}")

        return study.best_params

    def train(self, x_train, y_train):
        epochs = self.model_param.PARAM_EPOCHS
        batch_size = self.model_param.BATCH_SIZE
        param_learning_rate = self.model_param.ROLLING_PARAM_LEARNING_RATE

        self.model = self.create_cnn_transformer_model(
            (x_train.shape[1], x_train.shape[2])
        )

        self.model.compile(
            optimizer=Adam(learning_rate=param_learning_rate),
            loss="mean_squared_error",
            metrics=["mean_absolute_error"],
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
            model_cv = self.create_cnn_transformer_model((data.shape[1], data.shape[2]))
            model_cv.compile(
                optimizer=Adam(learning_rate=param_learning_rate),
                loss="mean_squared_error",
                metrics=["mean_absolute_error"],
            )
            self.logger.log_system_message(f"Training for fold {fold_no} ...")
            model_cv.fit(
                data[train_idx],
                targets[train_idx],
                epochs=epochs,
                batch_size=batch_size
            )

            # 評価
            mse_mae = model_cv.evaluate(data[test_idx], targets[test_idx], verbose=0)
            scores.append(mse_mae)
            fold_no += 1

            # メモリクリア
            del model_cv
            tf.keras.backend.clear_session()
            gc.collect()

        return scores

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> dict:
        y_pred_scaled = self.model.predict(x_test)  # shape: (N,1)
        y_test_2d = y_test.reshape(-1, 1)

        y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
        y_true = self.target_scaler.inverse_transform(y_test_2d)

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }

    def plot_predictions(self, x_test: np.ndarray, y_test: np.ndarray, sample_size=300):
        y_pred_scaled = self.model.predict(x_test)
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
        y_true = self.target_scaler.inverse_transform(y_test.reshape(-1, 1))

        if sample_size > len(y_true):
            sample_size = len(y_true)

        subset_pred = y_pred[-sample_size:]
        subset_true = y_true[-sample_size:]

        plt.figure(figsize=(10, 6))
        plt.plot(range(sample_size), subset_true, label="Actual Price", marker='o')
        plt.plot(range(sample_size), subset_pred, label="Predicted Price", marker='x')
        plt.title("BTC Price Prediction vs Actual")
        plt.xlabel("Time Step (last samples)")
        plt.ylabel("BTC Price")
        plt.legend()
        plt.grid(True)
        plt.show()

    def predict(self, data: np.ndarray, use_gpu: bool = None) -> np.ndarray:
        if use_gpu is not None:
            original_devices = tf.config.get_visible_devices()
            self._configure_gpu(use_gpu)

        y_pred_scaled = self.model.predict(data)

        if use_gpu is not None:
            tf.config.set_visible_devices(original_devices)

        y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
        return y_pred

    def predict_single_res(self, data_point: np.ndarray, use_gpu: bool = None) -> float:
        scaled_data_point = self.scaler.transform(data_point)
        reshaped_data = scaled_data_point.reshape(1, -1, len(self.feature_columns))

        if use_gpu is not None:
            original_devices = tf.config.get_visible_devices()
            self._configure_gpu(use_gpu)

        y_pred_scaled = self.model.predict(reshaped_data)

        if use_gpu is not None:
            tf.config.set_visible_devices(original_devices)

        y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
        return float(y_pred[0][0])

    def predict_single(self, data_point: np.ndarray, use_gpu: bool = None) -> float:
        return self.predict_single_res(data_point, use_gpu)

    def save_model(self, filename=None):
        if filename is not None:
            self.filename = filename

        model_file_name = self.filename + ".keras"
        model_path = os.path.join(self.datapath, model_file_name)
        self.logger.log_system_message(f"Saving model to {model_path}")
        self.model.save(model_path)

        model_scaler_file = self.filename + ".scaler"
        model_scaler_path = os.path.join(self.datapath, model_scaler_file)
        self.logger.log_system_message(f"Saving feature scaler to {model_scaler_path}")
        joblib.dump(self.scaler, model_scaler_path)

        model_target_scaler_file = self.filename + ".target_scaler"
        model_target_scaler_path = os.path.join(self.datapath, model_target_scaler_file)
        self.logger.log_system_message(f"Saving target scaler to {model_target_scaler_path}")
        joblib.dump(self.target_scaler, model_target_scaler_path)

    def load_model(self, filename=None):
        if filename is not None:
            self.filename = filename

        self.model = ModelManager.load_model(self.filename, self.datapath, self.logger)

        model_scaler_file = self.filename + ".scaler"
        model_scaler_path = os.path.join(self.datapath, model_scaler_file)
        self.logger.log_system_message(f"Loading feature scaler from {model_scaler_path}")
        self.scaler = joblib.load(model_scaler_path)

        model_target_scaler_file = self.filename + ".target_scaler"
        model_target_scaler_path = os.path.join(self.datapath, model_target_scaler_file)
        self.logger.log_system_message(f"Loading target scaler from {model_target_scaler_path}")
        self.target_scaler = joblib.load(model_target_scaler_path)

    def get_data_period(self, date: str, period: int) -> np.ndarray:
        data = self.data_loader.load_data_from_datetime_period(
            date, period, self.table_name
        )
        if "date" in data.columns:
            data = self._process_timestamp_and_cyclical_features(data)
        return data.filter(items=self.feature_columns).to_numpy()


def main():
    from aiml.aiml_comm import COLLECTIONS_TECH
    from common.utils import get_config_model

    data_loader = MongoDataLoader()
    logger = TradingLogger()
    model_id = "rolling_v19"

    # コンフィグ読み込み (例)
    config = get_config_model("MODEL_LONG_TERM", model_id)
    model = TransformerPredictionRollingModel(
        id=model_id,
        config=config,
        data_loader=data_loader,
        logger=logger,
        use_gpu=True
    )

    #model.load_model()



    # 例：データ準備
    from common.constants import MARKET_DATA_TECH
    x_train, x_test, y_train, y_test = model.load_and_prepare_data(
        "2020-01-01 00:00:00",
        "2024-08-01 00:00:00",
        MARKET_DATA_TECH,
        test_size=0.3,
        random_state=None
    )


    # ハイパーパラメータ探索の前にエポック数やバッチサイズなどの基本設定をしておく
    model.set_parameters(param_epochs=20, batch_size=32, n_splits=3)

    final_model = model.create_cnn_transformer_model(
        input_shape=(x_train.shape[1], x_train.shape[2]),
        num_heads=24,
        dff=448,
        rate=0.0,
        l2_reg=9.159280150065751e-05,
        num_transformer_blocks=3,
        num_filters=256,
        kernel_size=3
    )


    """
    # ===== 1. Optunaでハイパーパラメータ探索 =====
    best_params = model.hyperparameter_search_optuna(
        x_train, y_train,
        n_trials=10,         # 試行回数(例)
        timeout=1800,        # 制限時間(秒)例
        validation_split=0.2,
        max_epochs=5         # 探索時はエポックを短めに
    )

    # ===== 2. ベストパラメータで最終学習 =====
    final_model = model.create_cnn_transformer_model(
        input_shape=(x_train.shape[1], x_train.shape[2]),
        num_heads=best_params["num_heads"],
        dff=best_params["dff"],
        rate=best_params["dropout_rate"],
        l2_reg=best_params["l2_reg"],
        num_transformer_blocks=best_params["num_blocks"],
        num_filters=best_params["num_filters"],
        kernel_size=best_params["kernel_size"]
    )

    """
    final_model.compile(
        optimizer=Adam(learning_rate=model.model_param.ROLLING_PARAM_LEARNING_RATE),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )

    # 探索終了後、本番用エポックで改めて学習
    final_model.fit(x_train, y_train, epochs=40, batch_size=model.model_param.BATCH_SIZE)
    model.model = final_model  # モデルを差し替え

    # ===== 3. テストデータで評価 =====
    metrics = model.evaluate(x_test, y_test)
    print("Test metrics after hyperparam search:", metrics)

    # ===== 4. 可視化 =====
    model.plot_predictions(x_test, y_test, sample_size=50)

    # ===== 5. 保存 =====
    model.save_model()

    # ここでさらに直近データを用いた評価なども可能
    # (省略)
    # ここでは直近のデータで評価のみ行う例
    from common.constants import MARKET_DATA_TECH
    x_train, x_test, y_train, y_test = model.load_and_prepare_data(
        "2024-08-01 00:00:00",
        "2024-12-01 00:00:00",
        MARKET_DATA_TECH
    )
    metrics = model.evaluate(x_test, y_test)
    print("Final evaluation on recent data:", metrics)

    # 予測と実際を可視化
    model.plot_predictions(x_test, y_test, sample_size=50)

if __name__ == "__main__":
    main()




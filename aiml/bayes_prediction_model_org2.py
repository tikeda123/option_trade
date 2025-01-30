import os
import sys
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import BayesianRidge

import joblib

# ===== MongoDataLoader等、外部ライブラリの読み込み ===== #
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# ここは各自の環境に合わせて修正してください
from mongodb.data_loader_mongo import MongoDataLoader
from common.utils import get_config
from common.constants import MARKET_DATA_TECH


class BayesianPricePredictor:
    """
    BayesianRidgeを用いたBTC価格予測モデルのクラス。
    """
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        interval: int = 1440*3,
        features: Optional[List[str]] = None,
        model_params: Optional[Dict[str, Any]] = None
    ):
        """
        :param symbol: 予測対象の銘柄・ペア (例: "BTCUSDT")
        :param interval: 取得するデータのタイムフレーム (分単位)。1440*3 で3日足を想定。
        :param features: 使用する特徴量のリスト
        :param model_params: BayesianRidgeのパラメータ (dict)
        """
        self.symbol = symbol
        self.interval = interval
        # ここで扱う特徴量をデフォルト定義
        self.features = features or [
            "open", "close", "high", "low", "volume",
            "rsi", "macd", "macdsignal",
            "ema", "sma",
            "upper1", "lower1", "middle",
        ]
        # モデルのパラメータ (BayesianRidge)
        default_params = {
            "max_iter": 300,
            "tol": 1e-6,
            "alpha_1": 1e-6,
            "alpha_2": 1e-6,
            "lambda_1": 1e-6,
            "lambda_2": 1e-6,
        }
        if model_params is not None:
            default_params.update(model_params)
        self.model_params = default_params

        self.data_loader = MongoDataLoader()  # MongoDBからのデータローダー（外部クラス）
        self.scaler = StandardScaler()

        self.model = None
        self.X_train_scaled = None
        self.y_train = None
        self.X_test_scaled = None
        self.y_test = None

        self.train_df = None
        self.test_df = None

        # ---- 追加: 全期間のデータを保持する変数 (predict_for_dateで使う) ----
        self.all_data = None

    def load_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        start_dateからend_dateまでのデータをMongoDBからロードし、保持する。
        """
        df = self.data_loader.load_data_from_datetime_period(
            start_date,
            end_date,
            coll_type=MARKET_DATA_TECH,
            symbol=self.symbol,
            interval=self.interval
        )
        # 後で個別予測するため、全期間のデータをクラス変数に保存しておく
        self.all_data = df.copy()
        return df

    def preprocess_data(
        self,
        df: pd.DataFrame,
        train_end_date: str = "2024-10-01",
        test_start_date: str = "2024-10-01"
    ) -> None:
        """
        データフレームを前処理し、訓練用データとテスト用データを作成する。
        テストデータが空の場合は、その後の処理でエラーにならないよう None で扱う。

        【修正ポイント】
        1) 先に train/test を分割
        2) それぞれのデータについて shift(-3) でラベルを作成
        3) NaN 行を削除してからスケーリング・セット格納
        """
        # 日付型に変換
        if "date" in df.columns and not np.issubdtype(df["date"].dtype, np.datetime64):
            df["date"] = pd.to_datetime(df["date"])

        # --- 1) 先に学習データ・テストデータに分割 ---
        train_df = df[df["date"] < train_end_date].copy()
        test_df  = df[df["date"] >= test_start_date].copy()

        # --- 2) それぞれのデータフレームで shift(-3) してラベルを作成 ---
        train_df["close_next"] = train_df["close"].shift(-3)
        test_df["close_next"]  = test_df["close"].shift(-3)

        # --- 3) shift による NaN 行などを除外 ---
        train_df.dropna(subset=["close_next"], inplace=True)
        test_df.dropna(subset=["close_next"], inplace=True)

        # --- 学習データの特徴量とラベル ---
        if not train_df.empty:
            train_data = train_df[["date"] + self.features + ["close_next"]].copy()
            X_train = train_data[self.features]
            y_train = train_data["close_next"]

            # 学習データをスケーリング
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            print("[INFO] No train data after split.")
            train_data = None
            X_train_scaled = None
            y_train = None

        # --- テストデータの特徴量とラベル ---
        if not test_df.empty:
            test_data = test_df[["date"] + self.features + ["close_next"]].copy()
            X_test = test_data[self.features]
            y_test = test_data["close_next"]
            # 学習データで fit したスケーラーを使って変換
            if X_train_scaled is not None:
                X_test_scaled = self.scaler.transform(X_test)
            else:
                # 学習データがない（イレギュラーケース）場合、fit できないので None
                X_test_scaled = None
                print("[INFO] Unable to scale test data because no train data was found.")
        else:
            print("[INFO] No test data after split.")
            test_data = None
            X_test_scaled = None
            y_test = None

        # --- クラス変数に保持 ---
        self.train_df = train_data
        self.test_df  = test_data
        self.X_train_scaled = X_train_scaled
        self.y_train = y_train
        self.X_test_scaled = X_test_scaled
        self.y_test = y_test

        # --- 学習期間とテスト期間の情報を出力 ---
        if not train_df.empty:
            print(f"Train period: {train_df['date'].min()} - {train_df['date'].max()}")
        else:
            print("Train period: None")

        if not test_df.empty:
            print(f"Test  period: {test_df['date'].min()} - {test_df['date'].max()}")
        else:
            print("Test period: None")

    def cross_validate(
        self,
        n_splits: int = 5,
        plot_last_fold: bool = True
    ) -> None:
        """
        TimeSeriesSplitによるクロスバリデーションを実施し、
        各Foldでの評価指標を表示する。
        """
        if self.X_train_scaled is None or self.y_train is None:
            print("[INFO] No training data for cross-validation.")
            return

        tscv = TimeSeriesSplit(n_splits=n_splits)
        mse_list, mae_list, r2_list = [], [], []

        last_fold_y_test_cv = None
        last_fold_y_pred_cv = None

        for fold, (cv_train_index, cv_val_index) in enumerate(tscv.split(self.X_train_scaled)):
            X_cv_train = self.X_train_scaled[cv_train_index]
            X_cv_val   = self.X_train_scaled[cv_val_index]
            y_cv_train = self.y_train.values[cv_train_index]
            y_cv_val   = self.y_train.values[cv_val_index]

            model_cv = BayesianRidge(**self.model_params)
            model_cv.fit(X_cv_train, y_cv_train)

            y_cv_pred = model_cv.predict(X_cv_val)
            mse = mean_squared_error(y_cv_val, y_cv_pred)
            mae = mean_absolute_error(y_cv_val, y_cv_pred)
            r2  = r2_score(y_cv_val, y_cv_pred)

            mse_list.append(mse)
            mae_list.append(mae)
            r2_list.append(r2)

            print(f"[Fold {fold+1}] MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

            # 最終foldの予測結果を保存して後で可視化
            if fold == (tscv.n_splits - 1):
                last_fold_y_test_cv = y_cv_val
                last_fold_y_pred_cv = y_cv_pred

        # 以下の可視化を有効にする場合はコメントアウトを外す
        """
        print("\n=== Cross Validation Result on Training Data ===")
        print(f"Average MSE: {np.mean(mse_list):.4f}")
        print(f"Average MAE: {np.mean(mae_list):.4f}")
        print(f"Average R2 : {np.mean(r2_list):.4f}")

        if plot_last_fold and (last_fold_y_test_cv is not None) and (last_fold_y_pred_cv is not None):
            plt.figure(figsize=(10, 6))
            x_values = range(len(last_fold_y_test_cv))
            plt.plot(x_values, last_fold_y_test_cv, label='Actual (val)', marker='o')
            plt.plot(x_values, last_fold_y_pred_cv, label='Predicted (val)', marker='x')
            plt.title("CV Last Fold: Actual vs Predicted")
            plt.xlabel("Index in Validation (last fold)")
            plt.ylabel("BTC Next Day Close Price")
            plt.legend()
            plt.show()
        """

    def train_final_model(self) -> None:
        """
        学習データ全体で最終モデルを再学習する。
        """
        if self.X_train_scaled is None or self.y_train is None:
            print("[INFO] No training data. Skipping final model training.")
            return

        self.model = BayesianRidge(**self.model_params)
        self.model.fit(self.X_train_scaled, self.y_train.values)
        print("\n[INFO] Final model training completed.")

    def predict_test(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        テストデータに対して予測を行い、(予測値, 標準偏差) を返す。
        テストデータがない場合は (None, None) を返す。
        """
        if self.model is None:
            raise ValueError("Model is not trained. Call train_final_model() first.")

        if self.X_test_scaled is None or self.y_test is None:
            print("[INFO] No test data available. Skipping prediction.")
            return None, None

        y_pred, y_std = self.model.predict(self.X_test_scaled, return_std=True)
        return y_pred, y_std

    def evaluate_test(self, y_pred: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
        """
        テストデータでの性能を評価し、指標を返す。
        テストデータがない場合は None を返す。
        """
        if self.X_test_scaled is None or self.y_test is None or y_pred is None:
            print("[INFO] No test data available. Skipping evaluation.")
            return {"mse": None, "mae": None, "r2": None}

        test_mse = mean_squared_error(self.y_test, y_pred)
        test_mae = mean_absolute_error(self.y_test, y_pred)
        test_r2  = r2_score(self.y_test, y_pred)

        return {
            "mse": test_mse,
            "mae": test_mae,
            "r2": test_r2
        }

    def plot_test_result(
        self,
        y_pred: Optional[np.ndarray],
        y_std: Optional[np.ndarray],
        percentile_threshold: float = 60.0,
        std_multiplier: float = 2.0
    ) -> None:
        """
        テストデータにおける実測値と予測値の可視化を行う。
        不確実性（標準偏差）を考慮した可視化を行い、thresholdを超えた部分をハイライト。
        テストデータがない場合はスキップ。
        """
        if self.test_df is None or self.y_test is None or y_pred is None or y_std is None:
            print("[INFO] No test data to plot. Skipping plot.")
            return

        test_data = self.test_df.copy()
        dates = test_data["date"]

        plt.figure(figsize=(10, 6))
        # 実測値
        plt.plot(dates, self.y_test, label='Actual (Next Day Close)', marker='o')
        # 予測値
        plt.plot(dates, y_pred, label='Predicted (Next Day Close)', marker='x')

        # 95% CI (±1.96 * std)
        y_upper = y_pred + 1.96 * y_std
        y_lower = y_pred - 1.96 * y_std
        plt.fill_between(dates, y_lower, y_upper, alpha=0.2, color='orange', label='95% CI')

        # 複数の基準で不確実性を判断
        threshold = np.percentile(y_std, percentile_threshold)
        baseline_std = np.mean(y_std)

        # 不確実性が高いと判断する条件を複数組み合わせる
        mask_high = (
            (y_std > threshold) |  # パーセンタイルベースの判断
            (y_std > baseline_std * std_multiplier)  # 平均からの乖離による判断
        )

        plt.fill_between(dates, y_lower, y_upper, where=mask_high,
                         alpha=0.3, color='red', label='High Uncertainty')

        plt.xticks(rotation=45)
        plt.title("Comparison of Actual vs Predicted (Next Day Close) with Uncertainty")
        plt.xlabel("Date")
        plt.ylabel("BTC Next Day Close Price")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def print_model_info(self) -> None:
        """
        モデル係数やハイパーパラメータなどを出力する。
        """
        if self.model is None:
            print("[INFO] Model not trained yet.")
            return

        print("\n=== Final Model Coefficients ===")
        for f, coef in zip(self.features, self.model.coef_):
            print(f"{f}: {coef:.4f}")
        print("Intercept:", self.model.intercept_)
        print("Alpha (precision of the noise):", self.model.alpha_)
        print("Lambda (precision of the weights):", self.model.lambda_)

    def save_model(self, filepath: str) -> None:
        """
        学習済みモデルをjoblib形式で保存する。
        """
        if self.model is None:
            raise ValueError("Model is not trained. Nothing to save.")
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "features": self.features
        }, filepath)
        print(f"[INFO] Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        保存済みモデルをjoblib形式から読み込む。
        """
        saved_data = joblib.load(filepath)
        self.model = saved_data["model"]
        self.scaler = saved_data["scaler"]
        self.features = saved_data["features"]
        print(f"[INFO] Model loaded from {filepath}")

    def predict_for_date(
        self,
        date_str: str,
        return_std: bool = True,
        percentile_threshold: float = 60.0,
        std_multiplier: float = 2.0
    ) -> Tuple[float, Optional[float], Optional[bool]]:
        """
        指定した日時のレコードに基づき、その翌日のclose価格を予測する。
        :param date_str: 予測したい日時 (format: "YYYY-MM-DD" 等)
        :param return_std: Trueにすると標準偏差も返す
        :param percentile_threshold: 不確実性を判断するパーセンタイルの閾値
        :param std_multiplier: 平均標準偏差の何倍を不確実とみなすか
        :return: (予測値, 標準偏差, 不確実性フラグ) または (予測値, None, None)
        """
        if self.model is None:
            raise ValueError("Model is not trained. Call train_final_model() first.")

        if self.all_data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # 日時をdatetimeに変換
        target_date = pd.to_datetime(date_str)

        # all_dataから該当する日時のレコードを取り出す
        row_data = self.all_data[self.all_data["date"] == target_date].copy()
        if row_data.empty:
            raise ValueError(f"No data found for the specified date: {date_str}")

        # 特徴量列を抜き出し
        X = row_data[self.features]
        X_scaled = self.scaler.transform(X)

        # 予測
        if return_std:
            y_pred, y_std = self.model.predict(X_scaled, return_std=True)

            # 不確実性の判定
            # 学習データ全体での予測の標準偏差を計算
            if self.X_train_scaled is not None:
                _, all_std = self.model.predict(self.X_train_scaled, return_std=True)
                threshold = np.percentile(all_std, percentile_threshold)
                baseline_std = np.mean(all_std)
                # 不確実性が高いかどうかを判定
                is_uncertain = (y_std[0] > threshold) or (y_std[0] > baseline_std * std_multiplier)
            else:
                # 学習データがない(イレギュラーケース)
                is_uncertain = None

            return y_pred[0], y_std[0], is_uncertain
        else:
            y_pred = self.model.predict(X_scaled)
            return y_pred[0], None, None


def main():
    # データ取得範囲
    start_date = "2020-01-01"
    end_date   = "2025-01-01"

    # Mongoデータローダー（環境に応じて修正）
    data_loader = MongoDataLoader()
    df = data_loader.load_data_from_datetime_period(
        start_date,
        end_date,
        coll_type=MARKET_DATA_TECH,
        symbol="BTCUSDT",
        interval=1440
    )
    print(df.tail())

    # モデルのインスタンスを作成
    model = BayesianPricePredictor(
        symbol="BTCUSDT",
        interval=1440,
        features=[
            "open", "close", "high", "low", "volume",
            "rsi", "macd", "macdsignal",
            "ema", "sma",
            "upper1", "lower1", "middle",
        ],
        model_params={
            "max_iter": 300,
            "tol": 1e-6,
            "alpha_1": 1e-6,
            "alpha_2": 1e-6,
            "lambda_1": 1e-6,
            "lambda_2": 1e-6,
        }
    )

    # データをロード (クラス内部でも保持するため)
    df = model.load_data(start_date, end_date)
    print("DataFrame columns:", df.columns)
    print(df.head())

    # 前処理 & 学習・テスト分割
    model.preprocess_data(df, train_end_date="2024-10-01", test_start_date="2024-10-01")

    # クロスバリデーション (学習データのみで実施)
    model.cross_validate(n_splits=5, plot_last_fold=True)

    # 全学習データで最終モデルを訓練
    model.train_final_model()

    # テストデータで予測 & 評価
    y_pred, y_std = model.predict_test()
    metrics = model.evaluate_test(y_pred)
    print("\n=== Final Evaluation on Test Data ===")
    print(f"Test MSE: {metrics['mse']}")
    print(f"Test MAE: {metrics['mae']}")
    print(f"Test R2 : {metrics['r2']}")

    # テストデータでの予測結果を可視化
    model.plot_test_result(y_pred, y_std, percentile_threshold=60.0)

    # モデル情報を出力
    model.print_model_info()

    # 特定の日時に対して予測する例
    target_date = "2024-12-28 00:00:00"
    try:
        pred_value, pred_std, is_uncertain = model.predict_for_date(target_date, return_std=True)
        print(f"\nPredicted next-day close for {target_date} : {pred_value:.4f} (+/- {pred_std:.4f})")
        if is_uncertain:
            print("High uncertainty")
        else:
            print("Low uncertainty")
    except ValueError as e:
        print(f"Prediction error: {e}")

    # 末尾5行を確認
    print("\nDataFrame tail:")
    print(df.tail())


if __name__ == "__main__":
    main()

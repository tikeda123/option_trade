import os
import sys
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import BayesianRidge

import tensorflow as tf
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

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mongodb.data_loader_mongo import MongoDataLoader
from common.utils import get_config
from common.constants import MARKET_DATA_TECH



def prediction_option_bayes_model(current_date:str,symbol="BTCUSDT",m_interval=1440*3):
    """
    ベイズモデルによる予測
    """
    # データ読み込み
    data_loader = MongoDataLoader()
    start_date = "2020-01-01"
    df = data_loader.load_data_from_datetime_period(
        start_date,
        current_date,
        coll_type=MARKET_DATA_TECH,
        symbol="BTCUSDT",
        interval=m_interval
    )


def main():
    #=== 1. データ読み込み (全期間) ===#
    data_loader = MongoDataLoader()
    start_date = "2020-01-01"
    end_date   = "2025-01-01"

    df = data_loader.load_data_from_datetime_period(
        start_date,
        end_date,
        coll_type=MARKET_DATA_TECH,
        symbol="BTCUSDT",
        interval=1440*3
    )

    print("DataFrame columns:", df.columns)
    print(df.head())

    #=== 2. 前処理（特徴量エンジニアリング） ===#
    features = [
        "open", "close", "high", "low", "volume",   # 価格・出来高
        "rsi", "macd", "macdsignal",                # テクニカル指標
        "ema", "sma",                               # 移動平均
        "upper1", "lower1", "middle",               # ボリンジャーバンド
    ]
    features = [f for f in features if f in df.columns]

    # 翌日の close をラベルにする
    df["close_next"] = df["close"].shift(-1)

    #=== 3. 学習期間とテスト期間を分ける ===#
    train_end_date = "2024-10-01"
    test_start_date = "2024-10-01"

    if "date" in df.columns and not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"])

    train_df = df[df["date"] < train_end_date].copy()
    test_df  = df[df["date"] >= test_start_date].copy()

    print(f"Train period: {train_df['date'].min()} - {train_df['date'].max()}")
    print(f"Test period : {test_df['date'].min()} - {test_df['date'].max()}")

    #=== 4. 特徴量・ラベルの作成 ===#
    train_data = train_df[["date"] + features + ["close_next"]].dropna().copy()
    X_train = train_data[features]
    y_train = train_data["close_next"]

    test_data = test_df[["date"] + features + ["close_next"]].dropna().copy()
    X_test = test_data[features]
    y_test = test_data["close_next"]

    #=== 5. スケーリング ===#
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    #=== 6. クロスバリデーション (TimeSeriesSplit) ===#
    tscv = TimeSeriesSplit(n_splits=5)
    mse_list, mae_list, r2_list = [], [], []

    last_fold_y_test_cv = None
    last_fold_y_pred_cv = None

    for fold, (cv_train_index, cv_val_index) in enumerate(tscv.split(X_train_scaled)):
        X_cv_train, X_cv_val = X_train_scaled[cv_train_index], X_train_scaled[cv_val_index]
        y_cv_train, y_cv_val = y_train.values[cv_train_index], y_train.values[cv_val_index]

        model_cv = BayesianRidge(
            max_iter=300,
            tol=1e-6,
            alpha_1=1e-6,
            alpha_2=1e-6,
            lambda_1=1e-6,
            lambda_2=1e-6
        )
        model_cv.fit(X_cv_train, y_cv_train)

        y_cv_pred = model_cv.predict(X_cv_val)
        mse = mean_squared_error(y_cv_val, y_cv_pred)
        mae = mean_absolute_error(y_cv_val, y_cv_pred)
        r2  = r2_score(y_cv_val, y_cv_pred)

        mse_list.append(mse)
        mae_list.append(mae)
        r2_list.append(r2)

        print(f"[Fold {fold+1}] MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

        # 最終foldの予測結果を保存
        if fold == (tscv.n_splits - 1):
            last_fold_y_test_cv = y_cv_val
            last_fold_y_pred_cv = y_cv_pred

    print("\n=== Cross Validation Result on Training Data ===")
    print(f"Average MSE: {np.mean(mse_list):.4f}")
    print(f"Average MAE: {np.mean(mae_list):.4f}")
    print(f"Average R2 : {np.mean(r2_list):.4f}")

    #=== 7. CV結果の可視化（最後のfoldのみ）===#
    if last_fold_y_test_cv is not None and last_fold_y_pred_cv is not None:
        plt.figure(figsize=(10, 6))
        x_values = range(len(last_fold_y_test_cv))
        plt.plot(x_values, last_fold_y_test_cv, label='Actual (val)', marker='o')
        plt.plot(x_values, last_fold_y_pred_cv, label='Predicted (val)', marker='x')
        plt.title("CV Last Fold: Actual vs Predicted")
        plt.xlabel("Index in Validation (last fold)")
        plt.ylabel("BTC Next Day Close Price")
        plt.legend()
        plt.show()

    #=== 8. 学習データ全体で最終モデルを再学習 ===#
    final_model = BayesianRidge(
        max_iter=300,
        tol=1e-6,
        alpha_1=1e-6,
        alpha_2=1e-6,
        lambda_1=1e-6,
        lambda_2=1e-6
    )
    final_model.fit(X_train_scaled, y_train.values)

    #=== 9. テストデータに対して予測 & 評価 (不確実性込み) ===#
    # return_std=True で予測の標準偏差を取得
    y_test_pred, y_test_std = final_model.predict(X_test_scaled, return_std=True)

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2  = r2_score(y_test, y_test_pred)

    print("\n=== Final Evaluation on Test Data ===")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test R2 : {test_r2:.4f}")

    #=== 10. テストデータでの予測結果を可視化 (不確実性表示) ===#
    plt.figure(figsize=(10, 6))
    # 実測値
    plt.plot(test_data['date'], y_test, label='Actual (Next Day Close)', marker='o')
    # 予測値
    plt.plot(test_data['date'], y_test_pred, label='Predicted (Next Day Close)', marker='x')

    # 信頼区間（95% CI程度として ±1.96 * std）
    y_upper = y_test_pred + 1.96 * y_test_std
    y_lower = y_test_pred - 1.96 * y_test_std
    plt.fill_between(test_data['date'], y_lower, y_upper, alpha=0.2, color='orange', label='95% CI')

    # 不確実性が閾値を超えた部分を赤色でハイライト
    threshold = np.percentile(y_test_std, 60)
    mask_high = (y_test_std > threshold)

    # where=mask_highで「不確実性が大きい部分」を追加で塗りつぶし
    plt.fill_between(test_data['date'], y_lower, y_upper, where=mask_high,
                     alpha=0.3, color='red', label='High Uncertainty')

    plt.xticks(rotation=45)
    plt.title("Comparison of Actual vs Predicted (Next Day Close) with Uncertainty")
    plt.xlabel("Date")
    plt.ylabel("BTC Next Day Close Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 回帰係数など
    print("\n=== Final Model Coefficients ===")
    for f, coef in zip(features, final_model.coef_):
        print(f"{f}: {coef:.4f}")
    print("Intercept:", final_model.intercept_)
    print("Alpha (precision of the noise):", final_model.alpha_)
    print("Lambda (precision of the weights):", final_model.lambda_)


if __name__ == "__main__":
    main()

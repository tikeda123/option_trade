import os
import sys
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.linear_model import LogisticRegression

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

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory to sys.path
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mongodb.data_loader_mongo import MongoDataLoader
from common.utils import get_config
from common.constants import MARKET_DATA_TECH


def main():
    #=== 1. データ読み込み ===#
    data_loader = MongoDataLoader()
    start_date = "2024-01-01"
    end_date   = "2024-12-01"

    df = data_loader.load_data_from_datetime_period(
        start_date,
        end_date,
        coll_type=MARKET_DATA_TECH,
        symbol="BTCUSDT",
        interval=1440
    )


    # df の列を確認（デバッグ用）
    print("DataFrame columns:", df.columns)
    print(df.head())

    #=== 2. 前処理（特徴量エンジニアリング） ===#
    features = [
        "open", "high", "low", "volume",   # 価格・出来高
        "rsi", "macd", "macdsignal",       # テクニカル指標
        "ema", "sma",                      # 移動平均
        "upper1", "lower1", "middle",      # ボリンジャーバンド
    ]
    # 実際に存在する列のみに絞る
    features = [f for f in features if f in df.columns]

    #----------------------------
    # yを「次のステップで価格が上昇か下降か」のラベルに変更
    #   *  y=1 (上昇) : 次のステップの終値 > 現在の終値
    #   *  y=0 (下降) : 次のステップの終値 <= 現在の終値
    #----------------------------
    # yは df["close"].shift(-1) と df["close"] を比較して作成
    df["label"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # X: 説明変数, y: 目的変数 (label) に置き換え
    X = df[features].copy()
    y = df["label"].copy()

    # shift(-1) により最終行が NaN になるため、最後の行は削除しておく
    # 欠損値の処理
    data = pd.concat([X, y], axis=1).dropna()
    X = data[features]
    y = data["label"]

    # スケーリング
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #=== 3. 時系列分割 (TimeSeriesSplit) ===#
    tscv = TimeSeriesSplit(n_splits=5)
    accuracy_list, precision_list, recall_list, f1_list = [], [], [], []

    # 後ほどグラフ描画のために、最後の fold の予測値を格納する変数
    last_fold_y_test = None
    last_fold_y_pred = None

    for fold, (train_index, test_index) in enumerate(tscv.split(X_scaled)):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]

        #=== 4. モデルの構築・学習 ===#
        # ここではロジスティック回帰を用いた二値分類
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        #=== 5. 予測 & 評価 ===#
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)

        accuracy_list.append(acc)
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)

        print(f"[Fold {fold+1}] Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

        # 最終 fold の予測値を保存しておく
        if fold == (tscv.n_splits - 1):
            last_fold_y_test = y_test
            last_fold_y_pred = y_pred

    print("\n=== Cross Validation Result (TimeSeriesSplit) ===")
    print(f"Average Accuracy : {np.mean(accuracy_list):.4f}")
    print(f"Average Precision: {np.mean(precision_list):.4f}")
    print(f"Average Recall   : {np.mean(recall_list):.4f}")
    print(f"Average F1       : {np.mean(f1_list):.4f}")

    #=== 6. グラフ描画: クロスバリデーションの最終foldでの比較 ===#
    if last_fold_y_test is not None and last_fold_y_pred is not None:
        plt.figure(figsize=(10, 6))
        x_values = range(len(last_fold_y_test))

        # Actual / Predicted を 0/1 でプロット（「上昇=1」「下降=0」）
        plt.plot(x_values, last_fold_y_test, label='Actual', marker='o')
        plt.plot(x_values, last_fold_y_pred, label='Predicted', marker='x')
        plt.title("Comparison of Actual vs Predicted (Last Fold) [0=Down, 1=Up]")
        plt.xlabel("Time index in Test (last fold)")
        plt.ylabel("Up(1)/Down(0)")
        plt.legend()
        plt.show()

    #=== 7. 最終モデルの学習 (全データ使用) ===#
    final_model = LogisticRegression(max_iter=1000)
    final_model.fit(X_scaled, y.values)

    #=== 8. 全データでの予測・レポート ===#
    final_pred = final_model.predict(X_scaled)

    print("\n=== Final Model Classification Report (All Data) ===")
    print(classification_report(y, final_pred, digits=4))

    # グラフとして可視化したい場合（0/1を時系列に沿って表示）
    plt.figure(figsize=(10, 6))
    x_values = range(len(y))
    plt.plot(x_values, y, label='Actual', marker='o')
    plt.plot(x_values, final_pred, label='Predicted', marker='x')
    plt.title("Actual vs Predicted (All Data) [0=Down, 1=Up]")
    plt.xlabel("Time Index")
    plt.ylabel("Up(1)/Down(0)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 回帰係数（重み）などを表示（参考）
    print("\n=== Final Model Coefficients ===")
    for f, coef in zip(features, final_model.coef_[0]):
        print(f"{f}: {coef:.4f}")
    print("Intercept: ", final_model.intercept_[0])


if __name__ == "__main__":
    main()

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


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """前処理（特徴量エンジニアリング）を行う関数"""
    df = df.copy()

    # 欠損値処理
    df.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)

    # 移動平均の計算
    df["ema"] = df["close"].ewm(span=20, adjust=False).mean()
    df["sma"] = df["close"].rolling(window=20).mean()

    # 欠損値処理
    df.dropna(inplace=True)

    return df


def empirical_bayes_rsi(df: pd.DataFrame, rsi_threshold: int = 30, future_days: int = 7, alpha_prior: float = 1.0, beta_prior: float = 1.0) -> Tuple[float, float]:
    """
    経験ベイズ法を用いてRSIが閾値以下の場合の価格上昇確率の事後分布を計算する関数

    Args:
        df (pd.DataFrame): データフレーム（'rsi'と'close'の列を含む）
        rsi_threshold (int): RSIの閾値
        future_days (int): 何日後の価格を予測するか
        alpha_prior (float): ベータ分布の事前分布のパラメータαの初期値
        beta_prior (float): ベータ分布の事前分布のパラメータβの初期値

    Returns:
        Tuple[float, float]: 事後分布のパラメータαとβ
    """

    # RSIが閾値以下のデータのみ抽出
    df_filtered = df[df["rsi"] <= rsi_threshold]

    # 価格変動の計算とラベル付け
    price_changes = (df_filtered["close"].shift(-future_days) - df_filtered["close"]) > 0
    price_changes = price_changes.astype(int)  # True/Falseを1/0に変換

    # 上昇と下落の回数をカウント
    n_ups = price_changes.sum()
    n_downs = len(price_changes) - n_ups

    # 経験ベイズ法によるパラメータ推定
    alpha_posterior = alpha_prior + n_ups
    beta_posterior = beta_prior + n_downs

    return alpha_posterior, beta_posterior


def main():
    # === 1. データ読み込み ===#
    data_loader = MongoDataLoader()
    start_date = "2020-01-01"
    end_date = "2024-04-01"

    df = data_loader.load_data_from_datetime_period(
        start_date,
        end_date,
        coll_type=MARKET_DATA_TECH,
        symbol="BTCUSDT",
        interval=1440
    )

    # === 2. 前処理（特徴量エンジニアリング） ===#
    df = preprocess_data(df)  # `lower2`, `upper2`, `middle` は既に存在するので、ボリンジャーバンドの計算は不要

    # === 3. 経験ベイズ法による検証 ===#
    rsi_threshold = 30
    future_days = 7
    alpha_prior = 1.0  # 事前分布のパラメータ（初期値）
    beta_prior = 1.0   # 事前分布のパラメータ（初期値）

    alpha_posterior, beta_posterior = empirical_bayes_rsi(df, rsi_threshold, future_days, alpha_prior, beta_prior)

    # 事後分布の平均（期待値）を計算
    expected_prob = alpha_posterior / (alpha_posterior + beta_posterior)

    print(f"RSI閾値: {rsi_threshold}")
    print(f"予測期間: {future_days}日後")
    print(f"事前分布のパラメータ: α={alpha_prior}, β={beta_prior}")
    print(f"事後分布のパラメータ: α={alpha_posterior}, β={beta_posterior}")
    print(f"RSIが{rsi_threshold}以下の時、{future_days}日後に価格が上昇する確率の事後分布の平均: {expected_prob:.4f}")

    # 事後分布の可視化
    x = np.linspace(0, 1, 100)
    y = beta.pdf(x, alpha_posterior, beta_posterior)
    plt.plot(x, y)
    plt.title(f"Posterior Distribution of Price Increase Probability (RSI <= {rsi_threshold})")
    plt.xlabel("Probability")
    plt.ylabel("Density")
    plt.axvline(x=expected_prob, color="red", linestyle="--", label=f"Expected Value (Mean): {expected_prob:.4f}")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    from scipy.stats import beta
    main()
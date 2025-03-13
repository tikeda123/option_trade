import os
import sys
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# ユーザー環境に合わせたパス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *


def kalman_filter_1d(data: np.ndarray, Q: float, R: float) -> np.ndarray:
    """
    1次元カルマンフィルタ（線形ガウスモデル）実装
      状態方程式: x_t = x_{t-1} + w_t
      観測方程式: y_t = x_t + v_t

    Parameters
    ----------
    data : (N,) 次元の観測データ
        N: 時系列数
    Q : プロセスノイズの分散
    R : 観測ノイズの分散

    Returns
    -------
    x_est : (N,) 次元の推定状態ベクトルの系列
    """
    N = len(data)

    # 状態推定格納用
    x_est = np.zeros(N)
    # 推定誤差分散
    P_est = np.zeros(N)

    # 初期化
    x_est[0] = data[0]     # 初期状態は最初の観測値と同じとする
    P_est[0] = 1.0         # 初期の分散は適当な大きめの値に

    for t in range(1, N):
        # ---- Predict (予測ステップ) ----
        x_pred = x_est[t-1]
        P_pred = P_est[t-1] + Q

        # ---- Update (更新ステップ) ----
        K = P_pred / (P_pred + R)  # カルマンゲイン
        x_est[t] = x_pred + K * (data[t] - x_pred)
        P_est[t] = (1 - K) * P_pred

    return x_est


def apply_kalman_filter(df: pd.DataFrame, column: str, Q: float, R: float) -> pd.Series:
    """
    指定されたカラムにカルマンフィルタを適用する

    Parameters
    ----------
    df : pd.DataFrame
        入力データフレーム
    column : str
        フィルタを適用するカラム名
    Q : float
        プロセスノイズの分散
    R : float
        観測ノイズの分散

    Returns
    -------
    pd.Series
        カルマンフィルタ適用後の系列
    """
    data = df[column].values
    filtered = kalman_filter_1d(data, Q, R)
    return pd.Series(filtered, index=df.index, name=f"{column}_kalman")


def main():
    # MongoDBからBTCの時系列データを取得
    db = MongoDataLoader()
    df = db.load_data_from_datetime_period(
        datetime(2023, 1, 1),
        datetime(2025, 1, 5),
        coll_type=MARKET_DATA_TECH,
        symbol='BTCUSDT',
        interval=1440  # 日足データ
    )

    # 利用するカラムのみ抽出（必要に応じて拡張可能）
    graph_df = df[['start_at', 'close', 'rsi', 'volatility', 'macdhist']].copy()

    # NaNや欠損がある場合の処理
    graph_df.dropna(subset=['close', 'rsi', 'volatility', 'macdhist'], inplace=True)
    graph_df.reset_index(drop=True, inplace=True)

    # 各指標に対してカルマンフィルタを適用
    # 指標ごとにノイズパラメータを調整
    filters_config = {
        'close': {'Q': 1e-2, 'R': 1.0},
        'rsi': {'Q': 1e-1, 'R': 10.0},
        'volatility': {'Q': 1e-2, 'R': 1.0},
        'macdhist': {'Q': 1e-2, 'R': 1.0}
    }

    # 各指標にフィルタを適用

    filtered_series = apply_kalman_filter(graph_df, 'close', 1e-2, 1.0)
    graph_df[f"close_kalman"] = filtered_series

    # 可視化（例：終値のみ）
    plt.figure(figsize=(12, 6))
    plt.plot(graph_df['start_at'], graph_df['close'], label='Original Close', alpha=0.5)
    plt.plot(graph_df['start_at'], graph_df['close_kalman'], label='Kalman Close', color='red')
    plt.ylabel('Close')
    plt.legend()
    plt.grid(True)
    plt.title('Kalman Filter on Close Price')
    plt.show()

if __name__ == "__main__":
    main()

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


def kalman_filter_nd(data: np.ndarray,
                     Q: np.ndarray,
                     R: np.ndarray) -> np.ndarray:
    """
    多次元カルマンフィルタ（線形ガウスモデル）実装例
      状態方程式: x_t = x_{t-1} + w_t
      観測方程式: y_t = x_t + v_t
      (A=I, H=I という単純なモデル)

    Parameters
    ----------
    data : (N, M) 次元の観測データ
        N: 時系列数, M: 観測ベクトルの次元
    Q : (M, M) 次元のプロセスノイズ共分散行列
    R : (M, M) 次元の観測ノイズ共分散行列

    Returns
    -------
    x_est : (N, M) 次元の推定状態ベクトルの系列
    """
    N, M = data.shape

    # 状態推定格納用
    x_est = np.zeros((N, M))
    # 推定誤差共分散
    P_est = np.zeros((N, M, M))

    # 初期化
    x_est[0] = data[0]            # 初期状態は最初の観測値と同じとする
    P_est[0] = np.eye(M) * 1.0    # 初期の共分散行列は適当な大きめの値に

    # 単位行列（状態遷移行列 A, 観測行列 H 用）
    I = np.eye(M)

    for t in range(1, N):
        # ---- Predict (予測ステップ) ----
        # x_pred = A x_{t-1}, ここでは A = I
        x_pred = x_est[t - 1]
        # P_pred = A P_{t-1} A^T + Q, ここでは A=I なので単純加算
        P_pred = P_est[t - 1] + Q

        # ---- Update (更新ステップ) ----
        # カルマンゲイン K = P_pred H^T (H P_pred H^T + R)^-1
        # ここでは H = I なので: K = P_pred (P_pred + R)^-1
        S = P_pred + R  # (M, M)
        K = np.dot(P_pred, np.linalg.inv(S))

        # x_est[t] = x_pred + K ( y_t - H x_pred )
        # ここでは H=I なので: x_est[t] = x_pred + K*(data[t] - x_pred)
        y_t = data[t]
        x_est[t] = x_pred + np.dot(K, (y_t - x_pred))

        # P_est[t] = (I - K H) P_pred
        # ここでは H=I なので: P_est[t] = (I - K) P_pred
        P_est[t] = np.dot((I - K), P_pred)

    return x_est


def main():
    # MongoDBからBTCの時系列データを取得
    db = MongoDataLoader()
    df = db.load_data_from_datetime_period(
        datetime(2023, 1, 1),
        datetime(2025, 1, 5),
        coll_type=MARKET_DATA_TECH,
        symbol='BTCUSDT',
        interval=1440
    )

    # 利用するカラムのみ抽出（必要に応じて拡張可能）
    graph_df = df[['start_at', 'close', 'rsi', 'volatility', 'macdhist']].copy()

    # NaNや欠損がある場合の処理（例：前の値で埋める、もしくはドロップ等）
    # 今回単純に dropna しておく
    graph_df.dropna(subset=['close', 'rsi', 'volatility', 'macdhist'], inplace=True)
    graph_df.reset_index(drop=True, inplace=True)

    # 観測ベクトルとして多次元配列にする
    # 例: (N, 4) => [ [close, rsi, vol, macdhist], [...], ... ]
    observations = graph_df[['close', 'rsi', 'volatility', 'macdhist']].values

    # --- ノイズ共分散行列を設定 ---
    # Q: プロセスノイズ, R: 観測ノイズ。大きいほどフィルタは急激な変化を抑制 or 強調
    # ここでは簡単のため対角行列で設定
    # ※指標によってスケールが違うので、指標ごとに適宜調整が必要です
    Q = np.diag([1e-2, 1e-1, 1e-2, 1e-2])  # プロセスノイズ例
    R = np.diag([1.0, 10.0, 1.0, 1.0])     # 観測ノイズ例

    # 多次元カルマンフィルタ実行
    kf_result = kalman_filter_nd(observations, Q, R)  # shape (N, 4)

    # 推定結果を DataFrame に結合
    graph_df['close_kalman'] = kf_result[:, 0]
    graph_df['rsi_kalman'] = kf_result[:, 1]
    graph_df['volatility_kalman'] = kf_result[:, 2]
    graph_df['macdhist_kalman'] = kf_result[:, 3]

    # ---- 可視化：例として4つの指標をサブプロットで比較 ----
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 10), sharex=True)

    # 1) Close
    axes[0].plot(graph_df['start_at'], graph_df['close'], label='Original Close', alpha=0.5)
    axes[0].plot(graph_df['start_at'], graph_df['close_kalman'], label='Kalman Close', color='red')
    axes[0].set_ylabel('Close')
    axes[0].legend()
    axes[0].grid(True)

    # 2) RSI
    axes[1].plot(graph_df['start_at'], graph_df['rsi'], label='Original RSI', alpha=0.5)
    axes[1].plot(graph_df['start_at'], graph_df['rsi_kalman'], label='Kalman RSI', color='red')
    axes[1].set_ylabel('RSI')
    axes[1].legend()
    axes[1].grid(True)

    # 3) Volatility
    axes[2].plot(graph_df['start_at'], graph_df['volatility'], label='Original Volatility', alpha=0.5)
    axes[2].plot(graph_df['start_at'], graph_df['volatility_kalman'], label='Kalman Volatility', color='red')
    axes[2].set_ylabel('Volatility')
    axes[2].legend()
    axes[2].grid(True)

    # 4) MACD Hist
    axes[3].plot(graph_df['start_at'], graph_df['macdhist'], label='Original MACD Hist', alpha=0.5)
    axes[3].plot(graph_df['start_at'], graph_df['macdhist_kalman'], label='Kalman MACD Hist', color='red')
    axes[3].set_ylabel('MACD Hist')
    axes[3].set_xlabel('DateTime')
    axes[3].legend()
    axes[3].grid(True)

    plt.suptitle('Multi-dimensional Kalman Filter on Close/RSI/Volatility/MACD Hist')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# ユーザ環境に合わせたパス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# MongoDBからデータ取得用モジュールと定数
from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *

def kalman_filter_nd(data: np.ndarray, Q: np.ndarray, R: np.ndarray):
    """
    多次元カルマンフィルタ（線形ガウスモデル）
      状態方程式: x_t = x_{t-1} + w_t
      観測方程式: y_t = x_t + v_t
      (A=I, H=I の単純なモデル)

    Parameters
    ----------
    data : (N, M) 次元の観測データ
    Q, R : (M, M) の共分散行列

    Returns
    -------
    x_est : (N, M) の状態推定系列
    y_pred : (N, M) の観測予測系列 (H=I なので x_pred と同一)
    """
    N, M = data.shape
    x_est = np.zeros((N, M))
    P_est = np.zeros((N, M, M))

    # 初期状態：最初の観測値を利用、初期共分散は適当な大きさ
    x_est[0] = data[0]
    P_est[0] = np.eye(M) * 1.0
    I = np.eye(M)

    # 観測予測値の保存（更新前の状態予測として）
    y_pred = np.zeros((N, M))
    y_pred[0] = x_est[0]

    for t in range(1, N):
        # --- 予測ステップ ---
        x_pred = x_est[t - 1]           # A=I のためそのまま
        P_pred = P_est[t - 1] + Q       # P_pred = P_{t-1} + Q

        # --- 更新ステップ ---
        S = P_pred + R                # イノベーション共分散
        K = np.dot(P_pred, np.linalg.inv(S))  # カルマンゲイン
        y_t = data[t]
        x_est[t] = x_pred + np.dot(K, (y_t - x_pred))
        P_est[t] = np.dot((I - K), P_pred)

        # ここでは予測値 (観測予測) として x_pred を保存
        y_pred[t] = x_pred

    return x_est, y_pred

def estimate_Q_R(data: np.ndarray, x_est: np.ndarray, y_pred: np.ndarray):
    """
    カルマンフィルタ結果からサンプル共分散により Q, R を推定する

    Parameters
    ----------
    data  : (N, M) 観測データ
    x_est : (N, M) フィルタで推定された状態系列
    y_pred: (N, M) 観測予測系列 (H=I のため x_pred と同一)

    Returns
    -------
    Q_new : (M, M) 推定されたプロセスノイズ共分散
    R_new : (M, M) 推定された観測ノイズ共分散
    """
    N, M = data.shape
    # 観測残差（イノベーション）
    e = data - y_pred  # shape (N, M)
    R_new = np.cov(e.T, bias=True)  # 1/N で割る (bias=True) により (M, M) の共分散

    # 状態差分 Δx_t = x_est[t] - x_est[t-1]
    dx = x_est[1:] - x_est[:-1]  # shape (N-1, M)
    Q_new = np.cov(dx.T, bias=True)

    return Q_new, R_new

def auto_tune_kalman_filter(data: np.ndarray,
                            Q_init: np.ndarray,
                            R_init: np.ndarray,
                            max_iter=5,
                            alpha=0.5):
    """
    Q, R を初期値から反復的に推定しながらカルマンフィルタを適用する。

    Parameters
    ----------
    data    : (N, M) 観測データ
    Q_init, R_init : (M, M) 初期の共分散行列
    max_iter: 反復回数
    alpha   : 更新時の加重平均係数 (0〜1)

    Returns
    -------
    x_est   : (N, M) 最終的な状態推定系列
    Q_est, R_est : 推定された Q, R の共分散行列
    """
    Q_est = Q_init.copy()
    R_est = R_init.copy()

    for i in range(max_iter):
        # 現在の Q, R でフィルタ適用
        x_est, y_pred = kalman_filter_nd(data, Q_est, R_est)
        # フィルタ結果から新たな Q, R を推定
        Q_new, R_new = estimate_Q_R(data, x_est, y_pred)
        # 急激な変化を抑えるために加重平均で更新
        Q_est = alpha * Q_est + (1 - alpha) * Q_new
        R_est = alpha * R_est + (1 - alpha) * R_new

    # 最終的な Q_est, R_est で再度フィルタ適用して状態推定を得る
    x_est, _ = kalman_filter_nd(data, Q_est, R_est)
    return x_est, Q_est, R_est

def main():
    # 1) MongoDBからBTCの時系列データを取得
    db = MongoDataLoader()
    df = db.load_data_from_datetime_period(
        datetime(2023, 1, 1),
        datetime(2025, 1, 5),
        coll_type=MARKET_DATA_TECH,
        symbol='BTCUSDT',
        interval=1440
    )

    # 2) 主要カラムのみ抽出 (例: start_at, close, rsi, volatility, macdhist)
    graph_df = df[['start_at', 'close', 'rsi', 'volatility', 'macdhist']].copy()
    # 欠損データの除去
    graph_df.dropna(subset=['close', 'rsi', 'volatility', 'macdhist'], inplace=True)
    graph_df.reset_index(drop=True, inplace=True)

    # 3) 観測ベクトル作成 (各行が [close, rsi, volatility, macdhist] の4次元ベクトル)
    observations = graph_df[['close', 'rsi', 'volatility', 'macdhist']].values

    # 4) 初期の Q, R の設定（ここでは対角行列として初期化）
    M = observations.shape[1]
    Q_init = np.eye(M) * 1e-3  # プロセスノイズは小さめ
    R_init = np.eye(M) * 1.0   # 観測ノイズは中程度

    # 5) 自動チューニング付きカルマンフィルタを適用
    x_est_final, Q_est, R_est = auto_tune_kalman_filter(
        data=observations,
        Q_init=Q_init,
        R_init=R_init,
        max_iter=5,    # 必要に応じて反復回数を調整
        alpha=0.5      # 加重平均係数 (0〜1)
    )

    print("Estimated Q = \n", Q_est)
    print("Estimated R = \n", R_est)

    # 6) フィルタ結果を DataFrame に追加
    graph_df['close_kalman'] = x_est_final[:, 0]
    graph_df['rsi_kalman'] = x_est_final[:, 1]
    graph_df['volatility_kalman'] = x_est_final[:, 2]
    graph_df['macdhist_kalman'] = x_est_final[:, 3]

    # 7) 可視化（各指標について元データとフィルタ結果を比較）
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 10), sharex=True)

    axes[0].plot(graph_df['start_at'], graph_df['close'],
                 label='Original Close', alpha=0.5)
    axes[0].plot(graph_df['start_at'], graph_df['close_kalman'],
                 label='Kalman Close', color='red')
    axes[0].set_ylabel('Close')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(graph_df['start_at'], graph_df['rsi'],
                 label='Original RSI', alpha=0.5)
    axes[1].plot(graph_df['start_at'], graph_df['rsi_kalman'],
                 label='Kalman RSI', color='red')
    axes[1].set_ylabel('RSI')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(graph_df['start_at'], graph_df['volatility'],
                 label='Original Volatility', alpha=0.5)
    axes[2].plot(graph_df['start_at'], graph_df['volatility_kalman'],
                 label='Kalman Volatility', color='red')
    axes[2].set_ylabel('Volatility')
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(graph_df['start_at'], graph_df['macdhist'],
                 label='Original MACD Hist', alpha=0.5)
    axes[3].plot(graph_df['start_at'], graph_df['macdhist_kalman'],
                 label='Kalman MACD Hist', color='red')
    axes[3].set_ylabel('MACD Hist')
    axes[3].set_xlabel('DateTime')
    axes[3].legend()
    axes[3].grid(True)

    plt.suptitle('Multi-dimensional Kalman Filter (Auto-tuned Q,R) on Close/RSI/Volatility/MACD Hist')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

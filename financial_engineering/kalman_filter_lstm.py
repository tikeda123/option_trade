import os
import sys
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow / Keras 関連のインポート
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
# ユーザ環境に合わせたパス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# MongoDBからデータ取得用モジュールと定数
from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *

# -----------------------------
# 1. カルマンフィルタの実装
# -----------------------------
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

    # 初期化：初期状態は最初の観測値とし、共分散は単位行列×適当な値
    x_est[0] = data[0]
    P_est[0] = np.eye(M) * 1.0

    # 単位行列（A, H用）
    I = np.eye(M)

    for t in range(1, N):
        # ---- Predict (予測ステップ) ----
        x_pred = x_est[t - 1]       # A = I なので前ステップの推定値そのまま
        P_pred = P_est[t - 1] + Q     # P_pred = P_{t-1} + Q

        # ---- Update (更新ステップ) ----
        # S = P_pred + R
        S = P_pred + R
        # カルマンゲイン K = P_pred * inv(S)
        K = np.dot(P_pred, np.linalg.inv(S))

        # 観測 y_t
        y_t = data[t]
        # 更新された状態推定
        x_est[t] = x_pred + np.dot(K, (y_t - x_pred))
        # 更新された共分散
        P_est[t] = np.dot((I - K), P_pred)

    return x_est

# -----------------------------
# 2. LSTM用データセット作成関数
# -----------------------------
def make_lstm_dataset(df: pd.DataFrame, feature_cols: List[str], label_col: str, window_size: int = 10):
    """
    時系列データから、スライドウィンドウ形式の入力データとラベルを作成する関数

    Parameters
    ----------
    df : 時系列データが日付順に並んだ DataFrame
    feature_cols : LSTM に入力する特徴量のカラム名リスト
    label_col : 予測対象のラベルのカラム名（例: 翌日の上昇/下降フラグ）
    window_size : 過去何日分のデータを1サンプルとするか

    Returns
    -------
    X : numpy.array, shape=(サンプル数, window_size, 特徴量数)
    y : numpy.array, shape=(サンプル数,)
    """
    X, y = [], []
    for i in range(len(df) - window_size):
        feat = df.iloc[i:i+window_size][feature_cols].values
        label = df.iloc[i+window_size][label_col]
        X.append(feat)
        y.append(label)
    return np.array(X), np.array(y)

# -----------------------------
# 3. メイン処理
# -----------------------------
def main():
    # --- 3.1 MongoDBからBTCの時系列データを取得 ---
    db = MongoDataLoader()
    df = db.load_data_from_datetime_period(
        datetime(2023, 1, 1),
        datetime(2025, 1, 5),
        coll_type=MARKET_DATA_TECH,
        symbol='BTCUSDT',
        interval=1440  # 日足データ
    )

    # --- 3.2 利用するカラムの抽出と前処理 ---
    # 使用するカラム: start_at, close, rsi, volatility, macdhist
    graph_df = df[['start_at', 'close', 'rsi', 'volatility', 'macdhist']].copy()
    # 欠損値の除去
    graph_df.dropna(subset=['close', 'rsi', 'volatility', 'macdhist'], inplace=True)
    graph_df.reset_index(drop=True, inplace=True)

    # --- 3.3 カルマンフィルタによる平滑化 ---
    # 観測データの作成
    observations = graph_df[['close', 'rsi', 'volatility', 'macdhist']].values

    # ノイズ共分散行列の設定（各指標のスケールに応じて調整が必要）
    Q = np.diag([1e-2, 1e-1, 1e-2, 1e-2])  # プロセスノイズ
    R = np.diag([1.0, 10.0, 1.0, 1.0])      # 観測ノイズ

    # カルマンフィルタの適用
    kf_result = kalman_filter_nd(observations, Q, R)

    # フィルタ後の結果を DataFrame に追加
    graph_df['close_kalman']      = kf_result[:, 0]
    graph_df['rsi_kalman']        = kf_result[:, 1]
    graph_df['volatility_kalman'] = kf_result[:, 2]
    graph_df['macdhist_kalman']   = kf_result[:, 3]

    # --- 3.4 翌日のclose_kalmanの上昇/下降ラベル作成 ---
    # 翌日の値が今日の値より上なら 1 (up)、そうでなければ 0 (down)
    graph_df['label'] = (graph_df['close_kalman'].shift(-1) > graph_df['close_kalman']).astype(int)
    # 最終行は翌日がないため削除
    graph_df.dropna(inplace=True)
    graph_df.reset_index(drop=True, inplace=True)

    # --- 3.5 LSTM用の入力データ作成 ---
    feature_cols = ['close_kalman', 'rsi_kalman', 'volatility_kalman', 'macdhist_kalman']
    label_col = 'label'
    window_size = 10  # 過去10日分を1サンプルとする

    X, y = make_lstm_dataset(graph_df, feature_cols, label_col, window_size)
    print("入力データの形状:", X.shape, y.shape)

    # --- 3.6 特徴量の標準化 ---
    # 時系列データを一旦2次元に変換してから標準化し、再度3次元に戻す
    from sklearn.preprocessing import StandardScaler
    num_features = X.shape[2]
    X_reshaped = X.reshape(-1, num_features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    X = X_scaled.reshape(X.shape[0], window_size, num_features)

    # --- 3.7 学習・テストデータの分割 ---
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # --- 3.8 LSTMモデルの構築 ---
    model = Sequential()
    model.add(LSTM(64, input_shape=(window_size, len(feature_cols))))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))  # 2値分類

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # --- 3.9 モデルの学習 ---
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # --- 3.10 モデルの評価 ---
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")

    # --- 3.11 学習履歴の可視化 ---
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

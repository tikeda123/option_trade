import os
import sys
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow & Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight  # 追加: クラスウェイト計算用

# ---- ユーザー環境に合わせたパス設定 ----
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# DBモジュール・定数等
from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *
from data_preprocessing import clean_option_data  # ※必要に応じて

# ======================================================================
# 1) バレー算出用の関数 (改修版): find_optimal_trades
# ======================================================================
def find_optimal_trades(prices: np.ndarray,
                        timestamps: np.ndarray,
                        drop_threshold: float = 0.01,  # 例: 1%下落に緩和
                        hold_days: int = 72) -> List[Dict[str, Any]]:
    """
    短期（最大 hold_days 時間/足 数）トレードを想定し、小さな変動をスルーするため
    一定ドロップ閾値 (drop_threshold) を組み込んだピーク・バレー検出。

    Returns:
        各トレード（entry_index等）の情報が入った辞書リスト
    """
    trades = []
    n = len(prices)
    if n == 0:
        return trades

    last_local_max = prices[0]
    i = 0

    while i < n:
        # ========== バレー探索 ==========
        valley_index = None
        valley_price = None
        valley_time = None

        while i < n:
            current_price = prices[i]
            # (last_local_max - current_price) / max(last_local_max, 1e-9)
            if (last_local_max - current_price) / max(last_local_max, 1e-9) >= drop_threshold:
                valley_index = i
                valley_price = current_price
                valley_time = timestamps[i]
                break
            else:
                if current_price > last_local_max:
                    last_local_max = current_price
                i += 1

        if valley_index is None:
            break

        # ========== Exit探索 (最大 hold_days 分) ==========
        exit_index = valley_index
        exit_price = prices[valley_index]
        exit_time = timestamps[valley_index]

        limit = min(valley_index + hold_days, n - 1)
        for j in range(valley_index + 1, limit + 1):
            if prices[j] > exit_price:
                exit_index = j
                exit_price = prices[j]
                exit_time = timestamps[j]

        profit = exit_price - valley_price
        trades.append({
            'entry_index': valley_index,
            'entry_time': valley_time,
            'entry_price': valley_price,
            'exit_index': exit_index,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'profit': profit
        })

        last_local_max = exit_price
        i = exit_index + 1

    return trades


# ======================================================================
# 2) LSTM用のデータセット作成
# ======================================================================
def create_lstm_dataset(df: pd.DataFrame,
                        feature_cols: List[str],
                        label_col: str,
                        seq_length: int = 72) -> Tuple[np.ndarray, np.ndarray]:
    """
    LSTM訓練用に、過去 seq_length 分の特徴量を1系列とし、
    その直後のラベル (label_col) を学習対象とする。

    Returns:
        X: (サンプル数, seq_length, 特徴量数) のnumpy配列
        y: (サンプル数,) のラベル配列
    """

    data = df.copy()
    # 特徴量のみ抽出
    feature_data = data[feature_cols].values
    label_data = data[label_col].values

    X_list = []
    y_list = []

    for i in range(len(data) - seq_length):
        X_list.append(feature_data[i:i+seq_length])
        # シーケンスの最後の時点のラベルを採用
        y_list.append(label_data[i+seq_length])

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y


# ======================================================================
# 3) 簡易オーバーサンプリング関数 (少数クラスの増幅)
# ======================================================================
def oversample_minority_class(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    クラス1(少数)を単純に複製して増やすオーバーサンプリング。
    ※ SMOTEなどの高度な方法を使わず、簡易実装に留めています。
    """
    X_0 = X[y == 0]
    X_1 = X[y == 1]

    n_0 = len(X_0)
    n_1 = len(X_1)

    # 少数クラスがない場合はそのまま返す
    if n_1 == 0 or n_0 == 0:
        return X, y

    # たとえば n_0/n_1 回 複製してバランスを取る（厳密には floor）
    factor = n_0 // n_1
    if factor <= 1:
        # すでにそれほど不均衡でない場合は、そのまま返す
        return X, y

    X_1_oversampled = np.concatenate([X_1] * factor, axis=0)
    y_1_oversampled = np.ones(len(X_1_oversampled), dtype=int)

    X_new = np.concatenate([X_0, X_1_oversampled], axis=0)
    y_new = np.concatenate([np.zeros(len(X_0), dtype=int), y_1_oversampled], axis=0)

    # シャッフル
    idx = np.random.permutation(len(X_new))
    X_new = X_new[idx]
    y_new = y_new[idx]
    return X_new, y_new


def main():
    # ------------------------------------------------------
    # 1. データ読み込み
    # ------------------------------------------------------
    db = MongoDataLoader()
    df = db.load_data_from_datetime_period(
        datetime(2023, 1, 1),
        datetime(2025, 2, 14),
        coll_type=MARKET_DATA_TECH,
        symbol='BTCUSDT',
        interval=60
    )

    # 必要なカラムがDataFrameに含まれていると仮定
    # [ 'start_at', 'close', 'rsi', 'macdhist', 'volume', 'wclprice', ... ]
    # df.head()等で確認必須

    # データ型変換（日時列）
    if not pd.api.types.is_datetime64_any_dtype(df['start_at']):
        df['start_at'] = pd.to_datetime(df['start_at'])

    df.sort_values('start_at', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------
    # 2. バレーを「後出しじゃんけん」で確定し、ラベルを作成
    # ------------------------------------------------------
    prices = df['close'].values
    timestamps = df['start_at'].values

    # ラベリング条件をやや緩和 (1% 下落でバレー)
    drop_threshold = 0.01
    hold_days = 72
    trades = find_optimal_trades(prices, timestamps,
                                 drop_threshold=drop_threshold,
                                 hold_days=hold_days)

    # トレード情報から「バレーindex」一覧を取得
    valley_indices = set([t['entry_index'] for t in trades])

    # dfに is_valley カラムを追加 (1:バレー, 0:それ以外)
    df['is_valley'] = 0
    for idx in valley_indices:
        if idx < len(df):
            df.at[idx, 'is_valley'] = 1

    # ------------------------------------------------------
    # 3. LSTM 用の特徴量とラベルの準備
    # ------------------------------------------------------
    # 使用する説明変数（指標＋close）
    feature_cols = [
        "close",
        "rsi",
        "macdhist",
        "volume",
        "wclprice",
        "roc",
        "mfi",
        "atr",
        "ema",
        "bbvi",
        "turnover"
    ]

    label_col = 'is_valley'  # バレー判定ラベル

    # 不要な欠損があれば除外する
    df = df.dropna(subset=feature_cols + [label_col]).reset_index(drop=True)

    # スケーリング
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # LSTM 入力作成
    seq_length = 72
    X, y = create_lstm_dataset(df, feature_cols, label_col, seq_length=seq_length)

    # ------------------------------------------------------
    # 4. 訓練データとテストデータの分割 (時系列を考慮)
    # ------------------------------------------------------
    train_size = int(len(X) * 0.7)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # ------------------------------------------------------
    # 4.1 オーバーサンプリング (少数クラスを増やす)
    # ------------------------------------------------------
    X_train_os, y_train_os = oversample_minority_class(X_train, y_train)

    # ------------------------------------------------------
    # 5. LSTMモデルの構築
    # ------------------------------------------------------
    model = Sequential()
    model.add(LSTM(64, return_sequences=False, input_shape=(seq_length, len(feature_cols))))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))  # バイナリ分類

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # ------------------------------------------------------
    # 5.1 クラスウェイトの計算 (オプション: オーバーサンプリング併用の場合は慎重に)
    # ------------------------------------------------------
    # classes = np.unique(y_train_os)
    # cw_array = class_weight.compute_class_weight(
    #     class_weight='balanced',
    #     classes=classes,
    #     y=y_train_os
    # )
    # cw_dict = dict(zip(classes, cw_array))
    #
    # print("Class Weights:", cw_dict)

    # ------------------------------------------------------
    # 6. モデルの学習
    # ------------------------------------------------------
    # ※ オーバーサンプリング後のデータ (X_train_os, y_train_os) を使用
    # ※ クラスウェイトを使うなら `class_weight=cw_dict` をfit()に渡す
    history = model.fit(
        X_train_os, y_train_os,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        shuffle=False,  # 時系列なので shuffle=False
        # class_weight=cw_dict  # ← オーバーサンプリングと併用するならコメントアウトを外す
    )

    # ------------------------------------------------------
    # 7. テストデータでの評価
    # ------------------------------------------------------
    y_pred_prob = model.predict(X_test)

    # ============================================
    # (A) 閾値を下げてみる例: 0.3にするなど
    # ============================================
    threshold = 0.3
    y_pred = (y_pred_prob > threshold).astype(int).ravel()

    # ※ デフォルトの0.5にしたい場合は以下を使う：
    # y_pred = (y_pred_prob > 0.5).astype(int).ravel()

    print(f"\n=== Classification Report (Test) with threshold={threshold} ===")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ------------------------------------------------------
    # 8. 可視化（学習曲線など）
    # ------------------------------------------------------
    plt.figure(figsize=(12, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # ------------------------------------------------------
    # テストセットでの予測結果の一部を可視化
    # ------------------------------------------------------
    test_index_offset = train_size + seq_length  # seq_length分のシフトがある
    pred_df = df.iloc[test_index_offset:].copy()
    pred_df['y_true'] = y_test
    pred_df['y_pred'] = y_pred

    plt.figure(figsize=(14, 5))
    plt.plot(pred_df['start_at'], pred_df['close'], label='Close Price', color='blue')

    # Trueバレー
    true_valley = pred_df[pred_df['y_true'] == 1]
    plt.scatter(true_valley['start_at'], true_valley['close'],
                color='green', marker='^', label='True Valley')

    # Predictedバレー
    pred_valley = pred_df[pred_df['y_pred'] == 1]
    plt.scatter(pred_valley['start_at'], pred_valley['close'],
                color='red', marker='v', label='Pred Valley')
    plt.title('Valley Detection - LSTM Prediction')
    plt.xlabel('Date')
    plt.ylabel('Scaled Close Price')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()


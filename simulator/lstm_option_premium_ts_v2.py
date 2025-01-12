import os
import sys
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *
from option_pricing import simulate_option_prices

# ------------------------------------------------------
# 追加: symbol文字列から要素を抜き出す関数
# 例: "BTC-29DEC24-86000-C" -> ("BTC", "29DEC24", 86000.0, "C")
# ------------------------------------------------------
def parse_symbol(symbol: str) -> Tuple[str, str, float, str]:
    splitted = symbol.split('-')
    ticker = splitted[0]
    expiry = splitted[1]
    strike = float(splitted[2])
    option_type = splitted[3]  # "C" or "P"
    return ticker, expiry, strike, option_type

def process_option_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Process option data and organize it by symbol into time series
    """
    df['date'] = pd.to_datetime(df['date'])
    symbol_groups = {}

    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        symbol_df = symbol_df.sort_values('date')
        symbol_df.set_index('date', inplace=True)
        symbol_groups[symbol] = symbol_df

    return symbol_groups

# 特徴量の定義を定数として上部に移動
FEATURE_COLS = [
    'ask1Price',
    'bid1Price',
    'ask1Iv',
    'bid1Iv',
    'markIv',
    'underlyingPrice',
    'delta',
    'gamma',
    'vega',
    'theta',
    'openInterest',
    'markPrice'
]

# 特徴量のインデックスを定数として定義
FEATURE_INDICES = {col: idx for idx, col in enumerate(FEATURE_COLS)}

def create_lstm_dataset(
    data: np.ndarray, window_size: int = 24
) -> Tuple[np.ndarray, np.ndarray]:
    """
    LSTM用の入力(X), 出力(y)を作成する関数
    次の1ステップで ask1Iv が上昇 or 下降を予測する

    Args:
        data: スケーリング済みの特徴量データ (shape: [サンプル数, 特徴量数])
        window_size: 時系列の窓サイズ

    Returns:
        X: LSTM入力データ (shape: [サンプル数, window_size, 特徴量数])
        y: ラベルデータ (shape: [サンプル数,])
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        # 直近 window_size ステップ分
        X.append(data[i : i + window_size])

        # ask1Ivのインデックスを定数から取得
        ask1_iv_idx = FEATURE_INDICES['ask1Iv']

        # 現在のask1Ivと次のask1Ivを比較
        current_iv = data[i + window_size - 1, ask1_iv_idx]
        next_iv = data[i + window_size, ask1_iv_idx]
        label = 1 if next_iv > current_iv else 0
        y.append(label)

    return np.array(X), np.array(y)

def main():
    db = MongoDataLoader()
    df = db.load_data(OPTION_TICKER)

    # ------------------------------------------------------
    # 追加: symbol からストライク価格を抜き出して新列 'strike' として追加
    # ------------------------------------------------------
    df['strike'] = df['symbol'].apply(lambda s: parse_symbol(s)[2])

    # ------------------------------------------------------
    # 追加: 92000～105000 の範囲にあるオプションデータのみを抽出
    # ------------------------------------------------------
    df = df[(df['strike'] >= 92000) & (df['strike'] <= 105000)]

    # ------------------------------------------------------
    # シンボルごとに時系列データへ整形
    # ------------------------------------------------------
    symbol_timeseries = process_option_data(df)

    # ------------------------------------------------------
    # 1. 時系列長が 50 ステップ未満のシンボルを排除
    # ------------------------------------------------------
    all_X, all_y = [], []
    window_size = 24

    for symbol, ts_df in symbol_timeseries.items():
        if len(ts_df) < 50:
            continue

        # 必要なカラムがすべて存在するかチェック
        missing_cols = [col for col in FEATURE_COLS if col not in ts_df.columns]
        if missing_cols:
            continue

        # 各列を float 型へ変換
        for col in FEATURE_COLS:
            ts_df[col] = ts_df[col].astype(float)

        # 特徴量をまとめて numpy 配列へ
        features_data = ts_df[FEATURE_COLS].values

        # スケーリング
        scaler = MinMaxScaler()
        features_data_scaled = scaler.fit_transform(features_data)

        # LSTM用データセット作成
        X, y = create_lstm_dataset(features_data_scaled, window_size=window_size)

        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)

    if len(all_X) == 0:
        print("有効な学習データがありません（必要なカラムがない、または時系列が短い）")
        return

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    # ------------------------------------------------------
    # 2. 学習データ (train) と テストデータ (test) に分割
    # ------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, shuffle=True, random_state=42
    )

    # ------------------------------------------------------
    # 3. LSTM モデル構築
    # ------------------------------------------------------
    model = Sequential()
    model.add(
        LSTM(
            32,
            input_shape=(X_train.shape[1], X_train.shape[2]),
            return_sequences=False
        )
    )
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # ------------------------------------------------------
    # 4. 学習
    # ------------------------------------------------------
    history = model.fit(
        X_train, y_train,
        epochs=20,        # 例
        batch_size=32,    # 例
        validation_split=0.05,
        verbose=1
    )

    # ------------------------------------------------------
    # 5. 推論 & 評価
    # ------------------------------------------------------
    y_pred_prob = model.predict(X_test)
    y_pred_class = (y_pred_prob >= 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_test, y_pred_class)
    print(f"Test Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(y_test, y_pred_class)
    print("Confusion Matrix:")
    print(cm)

    print("Classification Report:")
    print(classification_report(y_test, y_pred_class, target_names=['Down','Up']))

    # 必要に応じてモデルを保存
    # model.save('lstm_trend_model.h5')

if __name__ == "__main__":
    main()

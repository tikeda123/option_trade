import os
import sys
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit  # ★ 追加
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ユーザー環境に合わせて import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *
from option_pricing import simulate_option_prices

def parse_symbol(symbol: str):
    splitted = symbol.split('-')
    ticker = splitted[0]
    expiry = splitted[1]
    strike = float(splitted[2])
    option_type = splitted[3]  # "C" or "P"
    return ticker, expiry, strike, option_type

def process_option_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    df['date'] = pd.to_datetime(df['date'])
    symbol_groups = {}
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        symbol_df = symbol_df.sort_values('date')
        symbol_df.set_index('date', inplace=True)
        symbol_groups[symbol] = symbol_df
    return symbol_groups

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
    'markPrice',
    'indexPrice'
]

FEATURE_INDICES = {col: idx for idx, col in enumerate(FEATURE_COLS)}

def create_lstm_dataset(data: np.ndarray, window_size: int, predict_col: str):
    """
    window_size 個の連続データから学習用 X を作る。
    その直後の価格が上昇 or 下降か（0 or 1）を y として返す。
    """
    X, y = [], []
    target_idx = FEATURE_INDICES[predict_col]
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        current_val = data[i + window_size - 1, target_idx]  # 直近の値
        next_val = data[i + window_size, target_idx]         # 1足先の値
        # 次の値が大きければ1 (上昇)、そうでなければ0 (下降)
        label = 1 if next_val > current_val else 0
        y.append(label)
    return np.array(X), np.array(y)

def remove_outliers_iqr(group: pd.DataFrame, col: str = 'ask1Price', factor: float = 2.0) -> pd.DataFrame:
    group[col] = pd.to_numeric(group[col], errors='coerce')
    Q1 = group[col].quantile(0.25)
    Q3 = group[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    group.loc[group[col] < lower_bound, col] = np.nan
    return group

def fill_zeros_and_interpolate(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values('date').copy()

    columns_to_clean = ['ask1Price', 'ask1Iv', 'bid1Price', 'bid1Iv']

    for col in columns_to_clean:
        group.loc[group[col] == 0, col] = np.nan
        group = remove_outliers_iqr(group, col=col, factor=1.5)

    group.set_index('date', inplace=True)

    for col in columns_to_clean:
        group[col] = group[col].interpolate(method='time')
        group[col].fillna(method='ffill', inplace=True)
        group[col].fillna(method='bfill', inplace=True)

    group.reset_index(inplace=True)
    return group

def main():
    predict_col = "indexPrice"
    db = MongoDataLoader()
    df = db.load_data(OPTION_TICKER)

    df['date'] = pd.to_datetime(df['date'])

    # ------------------------------
    # 0→NaN 変換 → IQR による外れ値除去 → 補完
    # ------------------------------
    df = df.groupby('symbol', group_keys=False).apply(fill_zeros_and_interpolate)

    columns_to_check = ['ask1Price', 'ask1Iv', 'bid1Price', 'bid1Iv']
    before_len = len(df)
    df.dropna(subset=columns_to_check, inplace=True)
    after_len = len(df)
    print(f"IQR 外れ値除去 & 補完後の NaN 除外: {before_len - after_len} 件削除 / {before_len} 件")

    for col in columns_to_check:
        zero_count = (df[col] == 0).sum()
        nan_count = df[col].isnull().sum()
        print(f"[Check] {col} => 0 の数: {zero_count}, NaN の数: {nan_count}")

    # strike range フィルタ (任意)
    df['strike'] = df['symbol'].apply(lambda s: parse_symbol(s)[2])
    df = df[(df['strike'] >= 20000) & (df['strike'] <= 105000)]

    # シンボルごとに DataFrame を辞書化
    symbol_timeseries = process_option_data(df)

    window_size = 24
    scaler = MinMaxScaler()

    # ------------------------------
    # 全シンボル分の特徴量をまとめてスケーリング
    # ------------------------------
    all_features_list = []
    for symbol, ts_df in symbol_timeseries.items():
        if len(ts_df) < 50:
            continue
        missing_cols = [col for col in FEATURE_COLS if col not in ts_df.columns]
        if missing_cols:
            continue
        for col in FEATURE_COLS:
            ts_df[col] = ts_df[col].astype(float)
        features_data = ts_df[FEATURE_COLS].values
        all_features_list.append(features_data)

    if not all_features_list:
        print("有効な学習データがありません。")
        return

    concatenated_features = np.concatenate(all_features_list, axis=0)
    scaler.fit(concatenated_features)

    # ------------------------------
    # 各シンボルの時系列を LSTM用 X, y に変換
    # ------------------------------
    all_X, all_y = [], []
    for symbol, ts_df in symbol_timeseries.items():
        if len(ts_df) < 50:
            continue
        missing_cols = [col for col in FEATURE_COLS if col not in ts_df.columns]
        if missing_cols:
            continue

        features_data = ts_df[FEATURE_COLS].values
        features_data_scaled = scaler.transform(features_data)

        X_symbol, y_symbol = create_lstm_dataset(
            data=features_data_scaled,
            window_size=window_size,
            predict_col=predict_col
        )
        if len(X_symbol) > 0:
            all_X.append(X_symbol)
            all_y.append(y_symbol)

    if len(all_X) == 0:
        print("有効な学習データがありません。")
        return

    # すべてのシンボル分を結合
    X_all = np.concatenate(all_X, axis=0)  # shape: (N, window_size, n_features)
    y_all = np.concatenate(all_y, axis=0)  # shape: (N,)

    # ------------------------------
    # ★ TimeSeriesSplit を使った時系列交差検証 ★
    # ------------------------------
    tscv = TimeSeriesSplit(n_splits=5)

    # 各Foldの評価結果を格納するリスト
    fold_accuracies = []

    MODEL_PATH = "lstm_model_binary.h5"  # 分類モデル用にファイル名を変更

    fold_index = 1
    for train_index, test_index in tscv.split(X_all):
        print(f"\n===== Fold {fold_index} / {tscv.n_splits} =====")

        # 時系列の順番を保ったまま分割
        X_train, X_test = X_all[train_index], X_all[test_index]
        y_train, y_test = y_all[train_index], y_all[test_index]

        # すでにモデルが保存されているならロード、なければ作成
        if os.path.exists(MODEL_PATH):
            print(f"{MODEL_PATH} が存在するためモデルをロードします。")
            model = load_model(MODEL_PATH)
        else:
            print(f"{MODEL_PATH} が存在しないため、新規にモデルを構築します。")
            model = Sequential()
            model.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
            # 二値分類なので出力は 1ユニット + sigmoid
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            history = model.fit(
                X_train, y_train,
                epochs=40,
                batch_size=32,
                validation_split=0.05,
                verbose=1
            )
            model.save(MODEL_PATH)

        # 予測
        y_pred_prob = model.predict(X_test).flatten()
        # 0.5 を閾値にして上昇(1)/下降(0) の予測ラベルに変換
        y_pred = (y_pred_prob >= 0.5).astype(int)

        # 評価 (精度など)
        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)

        print(f"Fold {fold_index} Accuracy: {acc:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        fold_index += 1

    # ------------------------------
    # 交差検証の平均評価を出力
    # ------------------------------
    avg_acc = np.mean(fold_accuracies)
    print("\n===== Cross Validation Results (TimeSeriesSplit) =====")
    print(f"Average Accuracy : {avg_acc:.4f}")

    # ---------------------------------------
    # 以下、可視化などはあくまで参考例 (方向性のみで可視化する場合)
    # ---------------------------------------
    import matplotlib.pyplot as plt
    target_symbol = "BTC-28MAR25-95000-P"  # 適宜変更
    if target_symbol not in symbol_timeseries:
        print(f"{target_symbol} のデータがありません。可視化をスキップします。")
        return

    ts_df = symbol_timeseries[target_symbol].copy()
    for col in FEATURE_COLS:
        if col not in ts_df.columns:
            print(f"{target_symbol} に {col} が存在しないため可視化できません。")
            return

    features_data = ts_df[FEATURE_COLS].astype(float).values
    features_data_scaled = scaler.transform(features_data)

    X_target, y_target = create_lstm_dataset(
        data=features_data_scaled,
        window_size=window_size,
        predict_col=predict_col
    )
    if len(X_target) == 0:
        print(f"{target_symbol} の時系列が短いため可視化できません。")
        return

    # ここでは最後に作成した model を使って可視化 (交差検証の最後の fold で学習したモデル)
    y_pred_target_prob = model.predict(X_target).flatten()
    y_pred_target = (y_pred_target_prob >= 0.5).astype(int)

    # 実際の上昇(1)/下降(0) と予測値を比較 (一例: 折れ線にして可視化)
    time_index = ts_df.index[window_size:]

    plt.figure(figsize=(12, 4))
    plt.plot(time_index, y_target, label="Actual Up/Down", marker='o')
    plt.plot(time_index, y_pred_target, label="Predicted Up/Down", marker='x')
    plt.ylim(-0.2, 1.2)
    plt.title(f"Up/Down Prediction for {target_symbol} (final fold model)")
    plt.xlabel("Date")
    plt.ylabel("Trend (0=Down, 1=Up)")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()

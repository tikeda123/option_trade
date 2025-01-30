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
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---- 追加: data_preprocessing.py をインポート ----
from data_preprocessing import clean_option_data

# ユーザー環境に合わせて import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *
#from option_pricing import simulate_option_prices

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
    'markPrice'
]

FEATURE_INDICES = {col: idx for idx, col in enumerate(FEATURE_COLS)}

def create_lstm_dataset(data: np.ndarray, window_size: int, predict_col: str):
    """
    window_size 個の連続データから学習用 X を作り、
    その直後の predict_col を y として返す。
    """
    X, y = [], []
    target_idx = FEATURE_INDICES[predict_col]
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        next_val = data[i + window_size, target_idx]
        y.append(next_val)
    return np.array(X), np.array(y)

def main():
    predict_col = "ask1Price"
    db = MongoDataLoader()
    df = db.load_data(OPTION_TICKER)

    # ------------------------------
    # データ前処理: 0→NaN変換, IQR外れ値処理, 補間, dropna
    # ------------------------------
    df = clean_option_data(
        df,
        group_col='symbol',
        columns_to_clean=['ask1Price', 'ask1Iv', 'bid1Price', 'bid1Iv'],  # 必要に応じて追加
        outlier_factor=1.5,  # IQR factor
        dropna_after=True
    )

    # strike range フィルタ (例)
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
    fold_mse_real = []
    fold_rmse_real = []
    fold_mae_real = []

    MODEL_PATH = "lstm_model.h5"

    fold_index = 1
    for train_index, test_index in tscv.split(X_all):
        print(f"\n===== Fold {fold_index} / {tscv.n_splits} =====")

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
            model.add(Dense(1, activation='linear'))
            model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

            history = model.fit(
                X_train, y_train,
                epochs=20,
                batch_size=32,
                validation_split=0.05,
                verbose=1
            )
            model.save(MODEL_PATH)

        # 予測
        y_pred_scaled = model.predict(X_test).flatten()

        # スケールを元に戻す
        pred_col_idx = FEATURE_INDICES[predict_col]
        predicted_data_scaled = []
        actual_data_scaled = []

        for i in range(len(X_test)):
            last_row_pred = X_test[i, -1, :].copy()
            last_row_pred[pred_col_idx] = y_pred_scaled[i]
            predicted_data_scaled.append(last_row_pred)

            last_row_actual = X_test[i, -1, :].copy()
            last_row_actual[pred_col_idx] = y_test[i]
            actual_data_scaled.append(last_row_actual)

        predicted_data_scaled = np.array(predicted_data_scaled)
        actual_data_scaled = np.array(actual_data_scaled)

        predicted_data = scaler.inverse_transform(predicted_data_scaled)
        actual_data = scaler.inverse_transform(actual_data_scaled)

        y_pred_real = predicted_data[:, pred_col_idx]
        y_test_real = actual_data[:, pred_col_idx]

        # 評価指標
        mse_real_fold = mean_squared_error(y_test_real, y_pred_real)
        rmse_real_fold = np.sqrt(mse_real_fold)
        mae_real_fold = mean_absolute_error(y_test_real, y_pred_real)

        print(f"[Real Scale] Fold {fold_index}  MSE : {mse_real_fold:.4f}")
        print(f"[Real Scale] Fold {fold_index}  RMSE: {rmse_real_fold:.4f}")
        print(f"[Real Scale] Fold {fold_index}  MAE : {mae_real_fold:.4f}")

        fold_mse_real.append(mse_real_fold)
        fold_rmse_real.append(rmse_real_fold)
        fold_mae_real.append(mae_real_fold)

        fold_index += 1

    # 交差検証の平均評価
    avg_mse = np.mean(fold_mse_real)
    avg_rmse = np.mean(fold_rmse_real)
    avg_mae = np.mean(fold_mae_real)

    print("\n===== Cross Validation Results (TimeSeriesSplit) =====")
    print(f"Average MSE : {avg_mse:.4f}")
    print(f"Average RMSE: {avg_rmse:.4f}")
    print(f"Average MAE : {avg_mae:.4f}")

    # ---------------------------------------
    # 以下、可視化などはあくまで参考例
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

    y_pred_target_scaled = model.predict(X_target).flatten()

    pred_col_idx = FEATURE_INDICES[predict_col]
    pred_for_plot_scaled = []
    actual_for_plot_scaled = []

    for i in range(len(y_pred_target_scaled)):
        last_row_pred = X_target[i, -1, :].copy()
        last_row_pred[pred_col_idx] = y_pred_target_scaled[i]
        pred_for_plot_scaled.append(last_row_pred)

        last_row_actual = X_target[i, -1, :].copy()
        last_row_actual[pred_col_idx] = y_target[i]
        actual_for_plot_scaled.append(last_row_actual)

    pred_for_plot_scaled = np.array(pred_for_plot_scaled)
    actual_for_plot_scaled = np.array(actual_for_plot_scaled)

    pred_for_plot = scaler.inverse_transform(pred_for_plot_scaled)
    actual_for_plot = scaler.inverse_transform(actual_for_plot_scaled)
    y_pred_target_real = pred_for_plot[:, pred_col_idx]
    y_target_real = actual_for_plot[:, pred_col_idx]

    time_index = ts_df.index[window_size:]

    plt.figure(figsize=(12, 6))
    plt.plot(time_index, y_target_real, label=f'Actual {predict_col}')
    plt.plot(time_index, y_pred_target_real, label=f'Predicted {predict_col}')
    plt.title(f"Prediction vs Actual for {target_symbol} (final fold model)")
    plt.xlabel("Date")
    plt.ylabel(f"{predict_col} (actual)")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()

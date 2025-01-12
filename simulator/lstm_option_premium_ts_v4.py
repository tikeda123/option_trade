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
from sklearn.model_selection import train_test_split
# 回帰評価指標のインポート
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# --------------------------
# MongoDataLoader, constants, simulate_option_prices は
# ユーザー環境に合わせて import してください
# --------------------------
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

# ------------------------------------------------------
# IQRを使って極端に低い外れ値を NaN にする関数
# ------------------------------------------------------
def remove_outliers_iqr(group: pd.DataFrame, col: str = 'ask1Price', factor: float = 2.0) -> pd.DataFrame:
    """
    IQR(四分位範囲)を用いて、下側の外れ値だけを NaN に置き換えます。
    factor=1.5 は「Q1 - 1.5*IQR」を下限としています。
    """
    # --- 追加: 文字列を含む場合、数値に変換 ---
    group[col] = pd.to_numeric(group[col], errors='coerce')

    Q1 = group[col].quantile(0.25)
    Q3 = group[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    # group[col] < lower_bound が極端に低い外れ値とみなす
    group.loc[group[col] < lower_bound, col] = np.nan
    return group

def fill_zeros_and_interpolate(group: pd.DataFrame) -> pd.DataFrame:
    """
    シンボル単位の DataFrame について:
      1) 0 を NaN に置換
      2) IQRに基づき極端に低い外れ値を NaN に置換
      3) 時間ベースで線形補間
      4) 前後埋め（ffill, bfill）
    """
    group = group.sort_values('date').copy()

    # 対象とする列
    columns_to_clean = ['ask1Price', 'ask1Iv', 'bid1Price', 'bid1Iv']

    # 1) 0 を NaN に置換 & 2) IQR を用いて下側外れ値を NaN に
    for col in columns_to_clean:
        # 0をNaNに
        group.loc[group[col] == 0, col] = np.nan
        # IQRを用いた下側外れ値除去
        group = remove_outliers_iqr(group, col=col, factor=1.5)

    # 日付をインデックスにして時間ベースで補間するための準備
    group.set_index('date', inplace=True)

    # 3) 時間ベースで線形補完 & 4) 前後埋め
    for col in columns_to_clean:
        group[col] = group[col].interpolate(method='time')
        group[col].fillna(method='ffill', inplace=True)
        group[col].fillna(method='bfill', inplace=True)

    group.reset_index(inplace=True)
    return group

def main():
    predict_col = "ask1Price"
    db = MongoDataLoader()
    df = db.load_data(OPTION_TICKER)

    df['date'] = pd.to_datetime(df['date'])

    # ------------------------------------------------------
    # シンボルごとに 0→NaN 変換 → IQR による外れ値除去 → 補完
    # ------------------------------------------------------
    df = df.groupby('symbol', group_keys=False).apply(fill_zeros_and_interpolate)

    # ------------------------------------------------------
    # 補完後でもなお NaN が残る行は除外 (補完しきれないもの)
    # ------------------------------------------------------
    columns_to_check = ['ask1Price', 'ask1Iv', 'bid1Price', 'bid1Iv']
    before_len = len(df)
    df.dropna(subset=columns_to_check, inplace=True)
    after_len = len(df)
    print(f"IQR 外れ値除去 & 補完後もなお NaN のままだったレコードを除外しました。除外数: {before_len - after_len} / {before_len}")

    # ------------------------------------------------------
    # チェック処理: 各列ごとに0やNaNが残っていないかを確認
    # ------------------------------------------------------
    for col in columns_to_check:
        zero_count = (df[col] == 0).sum()
        nan_count = df[col].isnull().sum()
        print(f"[Check] '{col}' が 0 の残り件数: {zero_count}, NaN の残り件数: {nan_count}")

        symbols_with_zero = df[df[col] == 0]['symbol'].unique()
        symbols_with_nan = df[df[col].isnull()]['symbol'].unique()

        if len(symbols_with_zero) > 0:
            print(f"[Check] 以下のシンボルに '{col}' = 0 が残っています:")
            for sym in symbols_with_zero:
                sub_df = df[(df['symbol'] == sym) & (df[col] == 0)]
                print(sub_df[['date', 'symbol', col]])
        else:
            print(f"[Check] '{col}' = 0 は残っていません。")

        if len(symbols_with_nan) > 0:
            print(f"[Check] 以下のシンボルに NaN が残っています (列: {col}):")
            for sym in symbols_with_nan:
                sub_df = df[(df['symbol'] == sym) & (df[col].isnull())]
                print(sub_df[['date', 'symbol', col]])
        else:
            print(f"[Check] NaN は残っていません (列: {col})。")

    # ------------------------------------------------------
    # ここから下は通常の学習フロー
    # ------------------------------------------------------
    df['strike'] = df['symbol'].apply(lambda s: parse_symbol(s)[2])
    # strike range 例: 20,000～105,000
    df = df[(df['strike'] >= 20000) & (df['strike'] <= 105000)]
    symbol_timeseries = process_option_data(df)

    window_size = 24
    scaler = MinMaxScaler()
    all_X, all_y = [], []

    # スケーリング用に全シンボルの特徴量を結合
    all_features_list = []

    for symbol, ts_df in symbol_timeseries.items():
        if len(ts_df) < 50:
            continue
        missing_cols = [col for col in FEATURE_COLS if col not in ts_df.columns]
        if missing_cols:
            # 必要な列が不足しているシンボルは除外
            continue
        for col in FEATURE_COLS:
            ts_df[col] = ts_df[col].astype(float)
        features_data = ts_df[FEATURE_COLS].values
        all_features_list.append(features_data)

    if not all_features_list:
        print("有効な学習データがありません。")
        return

    # 全シンボルまとめて fit する
    concatenated_features = np.concatenate(all_features_list, axis=0)
    scaler.fit(concatenated_features)

    # シンボルごとに LSTM 入力データセットを作成
    for symbol, ts_df in symbol_timeseries.items():
        if len(ts_df) < 50:
            continue
        missing_cols = [col for col in FEATURE_COLS if col not in ts_df.columns]
        if missing_cols:
            continue

        features_data = ts_df[FEATURE_COLS].values
        features_data_scaled = scaler.transform(features_data)

        X, y = create_lstm_dataset(
            data=features_data_scaled,
            window_size=window_size,
            predict_col=predict_col
        )
        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)

    if len(all_X) == 0:
        print("有効な学習データがありません。")
        return

    # LSTM 用にまとめる
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, shuffle=True, random_state=42
    )

    MODEL_PATH = "lstm_model.h5"

    if os.path.exists(MODEL_PATH):
        print(f"{MODEL_PATH} が存在するため、モデルをロードします。")
        model = load_model(MODEL_PATH)
    else:
        print(f"{MODEL_PATH} が存在しないため、新規にモデルを構築・学習します。")
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

    # ------------------------------------------------------
    # 予測と「スケーリング後」の評価
    # ------------------------------------------------------
    y_pred_scaled = model.predict(X_test).flatten()  # スケール後の予測
    mse_scaled = mean_squared_error(y_test, y_pred_scaled)
    rmse_scaled = np.sqrt(mse_scaled)
    mae_scaled = mean_absolute_error(y_test, y_pred_scaled)

    print(f"[Scaled] Test MSE : {mse_scaled:.4f}")
    print(f"[Scaled] Test RMSE: {rmse_scaled:.4f}")
    print(f"[Scaled] Test MAE : {mae_scaled:.4f}")

    # ------------------------------------------------------
    # 逆スケールして「実際の価格スケール」での評価
    # ------------------------------------------------------
    pred_col_idx = FEATURE_INDICES[predict_col]

    # 予測値を逆変換
    predicted_data_scaled = np.zeros((len(y_pred_scaled), len(FEATURE_COLS)))
    predicted_data_scaled[:, pred_col_idx] = y_pred_scaled
    predicted_data = scaler.inverse_transform(predicted_data_scaled)
    y_pred_real = predicted_data[:, pred_col_idx]

    # テスト用実測値も逆変換
    actual_data_scaled = np.zeros((len(y_test), len(FEATURE_COLS)))
    actual_data_scaled[:, pred_col_idx] = y_test
    actual_data = scaler.inverse_transform(actual_data_scaled)
    y_test_real = actual_data[:, pred_col_idx]

    # 実価格スケールで再度誤差を算出
    mse_real = mean_squared_error(y_test_real, y_pred_real)
    rmse_real = np.sqrt(mse_real)
    mae_real = mean_absolute_error(y_test_real, y_pred_real)

    print(f"[Real Scale] Test MSE : {mse_real:.4f}")
    print(f"[Real Scale] Test RMSE: {rmse_real:.4f}")
    print(f"[Real Scale] Test MAE : {mae_real:.4f}")

    # ------------------------------------------------------
    # 可視化サンプル (指定したシンボルのみ)
    # ------------------------------------------------------
    import matplotlib.pyplot as plt
    target_symbol = "BTC-28MAR25-95000-P"  # 適宜変更
    if target_symbol not in symbol_timeseries:
        print(f"{target_symbol} のデータがありません。")
        return

    ts_df = symbol_timeseries[target_symbol].copy()
    for col in FEATURE_COLS:
        if col not in ts_df.columns:
            print(f"{target_symbol} に {col} が存在しないため可視化できません。")
            return

    # 可視化用データ作成
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
    # 可視化のため逆変換
    num_samples = len(y_pred_target_scaled)
    pred_for_plot_scaled = np.zeros((num_samples, len(FEATURE_COLS)))
    pred_for_plot_scaled[:, pred_col_idx] = y_pred_target_scaled
    pred_for_plot = scaler.inverse_transform(pred_for_plot_scaled)
    y_pred_target_real = pred_for_plot[:, pred_col_idx]

    # 実測値を逆変換
    actual_for_plot_scaled = np.zeros((len(y_target), len(FEATURE_COLS)))
    actual_for_plot_scaled[:, pred_col_idx] = y_target
    actual_for_plot = scaler.inverse_transform(actual_for_plot_scaled)
    y_target_real = actual_for_plot[:, pred_col_idx]

    time_index = ts_df.index[window_size:]

    plt.figure(figsize=(12, 6))
    plt.plot(time_index, y_target_real, label=f'Actual {predict_col}')
    plt.plot(time_index, y_pred_target_real, label=f'Predicted {predict_col}')
    plt.title(f"Prediction vs Actual for {target_symbol} ({predict_col})")
    plt.xlabel("Date")
    plt.ylabel(f"{predict_col} (actual)")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()

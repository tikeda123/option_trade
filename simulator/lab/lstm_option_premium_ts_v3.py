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
from simulator.option_pricing import simulate_option_prices

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

# 特徴量の定義
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

# ------------------------------------------------------
# 予測ターゲット列を切り替えられるように関数化
# ------------------------------------------------------
def create_lstm_dataset(
    data: np.ndarray,
    window_size: int,
    predict_col: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    LSTM用の入力(X), 出力(y)を作成する関数
    引数:
        data: スケーリング済みの特徴量データ (shape: [サンプル数, 特徴量数])
        window_size: 時系列の窓サイズ
        predict_col: 予測対象となる列名
    戻り値:
        X: LSTM入力データ (shape: [サンプル数, window_size, 特徴量数])
        y: 回帰ラベル (shape: [サンプル数,])
    """
    X, y = [], []
    # 予測ターゲット列のインデックスを取得
    target_idx = FEATURE_INDICES[predict_col]

    for i in range(len(data) - window_size):
        # 直近 window_size ステップ分を特徴量として利用
        X.append(data[i : i + window_size])
        # 次の時刻のターゲット値を取得
        next_val = data[i + window_size, target_idx]
        y.append(next_val)

    return np.array(X), np.array(y)

def main():
    # ----------------------------
    # 予測する列をユーザーが設定
    # ----------------------------
    predict_col = "ask1Price"  # 変更可能: "markPrice", "ask1Price"など

    db = MongoDataLoader()
    df = db.load_data(OPTION_TICKER)

    # ------------------------------------------------------
    # symbol からストライクなどを抜き出す (parse_symbol)
    # strikeが20000～105000のみ抽出
    # ------------------------------------------------------
    df['strike'] = df['symbol'].apply(lambda s: parse_symbol(s)[2])
    df = df[(df['strike'] >= 20000) & (df['strike'] <= 105000)]

    print(df)

    # ------------------------------------------------------
    # シンボルごとに時系列DataFrameへ
    # ------------------------------------------------------
    symbol_timeseries = process_option_data(df)

    # ------------------------------------------------------
    # 全シンボルのデータをまとめてLSTMの学習データ作成
    # ------------------------------------------------------
    all_X, all_y = [], []
    window_size = 24

    scaler = MinMaxScaler()  # 特徴量全体をスケーリングする場合

    # 先に全シンボルのデータを結合してスケーリングする場合のためのリスト
    all_features_list = []

    for symbol, ts_df in symbol_timeseries.items():
        if len(ts_df) < 50:
            # サンプルが少なすぎる場合はスキップ
            continue

        # 必要な列がそろっていない場合はスキップ
        missing_cols = [col for col in FEATURE_COLS if col not in ts_df.columns]
        if missing_cols:
            continue

        # データ型をfloatへ
        for col in FEATURE_COLS:
            ts_df[col] = ts_df[col].astype(float)

        # 全体の特徴量を一時的に保持
        features_data = ts_df[FEATURE_COLS].values
        all_features_list.append(features_data)

    if not all_features_list:
        print("有効な学習データがありません。")
        return

    # 全シンボル分まとめた配列
    concatenated_features = np.concatenate(all_features_list, axis=0)
    # スケーリング (全シンボルを通したMinMaxを想定)
    scaler.fit(concatenated_features)

    # 改めてシンボルごとにスケーリング→LSTMデータ作成
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

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    # ------------------------------------------------------
    # train_test_split で学習/テストデータを分割
    # ------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, shuffle=True, random_state=42
    )

    # ------------------------------------------------------
    # モデルの保存先を設定
    # ------------------------------------------------------
    MODEL_PATH = "lstm_model.h5"

    # ------------------------------------------------------
    # すでに学習済モデルがあればロードし、なければ学習→保存
    # ------------------------------------------------------
    if os.path.exists(MODEL_PATH):
        print(f"{MODEL_PATH} が存在するため、モデルをロードします。")
        model = load_model(MODEL_PATH)
    else:
        print(f"{MODEL_PATH} が存在しないため、新規にモデルを構築・学習します。")
        # モデル構築
        model = Sequential()
        model.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

        # モデル学習
        history = model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.05,
            verbose=1
        )

        # モデル保存
        model.save(MODEL_PATH)
        print(f"学習済モデルを {MODEL_PATH} に保存しました。")

    # ------------------------------------------------------
    # テストデータで評価（ロードしたモデルでも可）
    # ------------------------------------------------------
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Test MSE : {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE : {mae:.4f}")

    # ------------------------------------------------------
    # 【追加】特定シンボルで可視化する (実値で表示する)
    # ------------------------------------------------------
    import matplotlib.pyplot as plt

    target_symbol = "BTC-28MAR25-90000-C"  # 可視化したいsymbolを指定

    # シンボルが存在しない場合はスキップ
    if target_symbol not in symbol_timeseries:
        print(f"{target_symbol} のデータがありません。")
        return

    ts_df = symbol_timeseries[target_symbol].copy()

    # 必要なカラムが足りない場合は終了
    for col in FEATURE_COLS:
        if col not in ts_df.columns:
            print(f"{target_symbol} に {col} が存在しないため可視化できません。")
            return

    # 特徴量を再度スケーリング (※学習時と同じScalerを使用)
    features_data = ts_df[FEATURE_COLS].astype(float).values
    features_data_scaled = scaler.transform(features_data)

    # LSTM入力/ラベル作成
    X_target, y_target = create_lstm_dataset(
        data=features_data_scaled,
        window_size=window_size,
        predict_col=predict_col
    )
    if len(X_target) == 0:
        print(f"{target_symbol} の時系列が短いため可視化できません。")
        return

    # モデル推論 (スケール値のまま)
    y_pred_target_scaled = model.predict(X_target).flatten()

    # ------------------------------------------------------------------
    # スケールを逆変換（実値に戻す）ための処理：
    #   - 予測値 y_pred_target_scaled, 実測値 y_target はどちらも
    #     「predict_col のみ」の形状になっている
    #   - しかし scaler.inverse_transform() は「全特徴量列の形」を要求する
    #   - そこでゼロ埋め配列を用意し、predict_col のところにだけ値を入れて
    #     逆変換したあと、その列を取り出す
    # ------------------------------------------------------------------
    pred_col_idx = FEATURE_INDICES[predict_col]

    # 予測値の逆変換
    num_samples = len(y_pred_target_scaled)
    predicted_data_scaled = np.zeros((num_samples, len(FEATURE_COLS)))
    predicted_data_scaled[:, :] = 0.0  # ゼロ埋め
    predicted_data_scaled[:, pred_col_idx] = y_pred_target_scaled

    predicted_data = scaler.inverse_transform(predicted_data_scaled)
    y_pred_target_real = predicted_data[:, pred_col_idx]  # 実際の予測値

    # 実測値の逆変換
    num_samples = len(y_target)
    actual_data_scaled = np.zeros((num_samples, len(FEATURE_COLS)))
    actual_data_scaled[:, :] = 0.0
    actual_data_scaled[:, pred_col_idx] = y_target

    actual_data = scaler.inverse_transform(actual_data_scaled)
    y_target_real = actual_data[:, pred_col_idx]  # 実際の実測値

    # 可視化のための時系列インデックス
    time_index = ts_df.index[window_size:]  # 先頭 window_size ステップ分は予測ラベルに対応する時刻がない

    plt.figure(figsize=(12, 6))
    plt.plot(time_index, y_target_real, label=f'Actual {predict_col}')
    plt.plot(time_index, y_pred_target_real, label=f'Predicted {predict_col}')
    plt.title(f"Prediction vs Actual for {target_symbol} ({predict_col})")
    plt.xlabel("Date")
    plt.ylabel(f"{predict_col} (actual)")  # 実値であることを明示
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()

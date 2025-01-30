import os
import sys
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---- 追加: data_preprocessing.py をインポート ----
from data_preprocessing import clean_option_data

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *

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

def main():
    db = MongoDataLoader()

    start_date = datetime(2024, 12, 18)
    end_date = datetime(2025, 1, 12)

    df_op = db.load_data_from_datetime_period_option(
        start_date=start_date,
        end_date=end_date
    )

    df_op = clean_option_data(
        df_op,
        group_col='symbol',
        columns_to_clean=['ask1Price', 'ask1Iv', 'bid1Price', 'bid1Iv'],
        outlier_factor=1.5,
        dropna_after=True
    )


    df_btc = db.load_data_from_datetime_period(
        start_date=datetime(2024, 12, 18),
        end_date=datetime(2025, 1, 12),
        coll_type=MARKET_DATA,
        symbol="BTCUSDT",
        interval=60
    )
    df_op["close"] = df_btc["close"]

    # シンボルごとの時系列データを取得
    symbol_groups = process_option_data(df_op)

    # 例: 特定シンボルを指定
    target_symbol = "BTC-28MAR25-100000-C"
    symbol_data = symbol_groups[target_symbol].copy()

    print(symbol_data)

    # カラムを数値に変換（dtype: object を float に）
    numeric_cols = ['ask1Price', 'markPrice', 'delta','ask1Iv']
    for col in numeric_cols:
        if col in symbol_data.columns:
            symbol_data[col] = pd.to_numeric(symbol_data[col], errors='coerce')

    # 日時ソート
    symbol_data = symbol_data.sort_index()

    # 大きく飛んでいる日付を埋める（1日ごとにリサンプリング＆前の値で埋める）
    # 本当に補間してよいかどうかは用途次第ですが、例として forward fill
    symbol_data = symbol_data.resample('D').ffill()

    # 時系列データの基本情報を表示
    print(f"Symbol: {target_symbol}")
    print(f"Data period: {symbol_data.index.min()} to {symbol_data.index.max()}")
    print(f"Number of data points: {len(symbol_data)}\n")
    for col in numeric_cols:
        if col in symbol_data.columns:
            print(f"{col}: {symbol_data[col].describe()}\n")

    # --- プロット ---
    plt.style.use('seaborn-v0_8')

    # First plot: ask1Price, close, delta
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 左軸(ax1)に価格系データを描画
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price', color='red')
    l1 = ax1.plot(symbol_data.index, symbol_data['ask1Price'], label='Ask1 Price', color='red', alpha=0.7)
    l2 = ax1.plot(symbol_data.index, symbol_data['close'], label='BTC Price', color='green', alpha=0.7)
    ax1.tick_params(axis='y', labelcolor='red')

    # 右軸(ax2)にdeltaを描画
    ax2 = ax1.twinx()
    ax2.set_ylabel('Delta', color='blue')
    l3 = ax2.plot(symbol_data.index, symbol_data['delta'], label='Delta', color='blue', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor='blue')

    plt.title(f'ask1Price, delta, BTC Price Time Series for {target_symbol}', fontsize=14)

    # 凡例をまとめて表示
    lines = l1 + l2 + l3
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Second plot: ask1Price, close, ask1Iv
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 左軸(ax1)に価格系データを描画
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price', color='red')
    l1 = ax1.plot(symbol_data.index, symbol_data['ask1Price'], label='Ask1 Price', color='red', alpha=0.7)
    l2 = ax1.plot(symbol_data.index, symbol_data['close'], label='BTC Price', color='green', alpha=0.7)
    ax1.tick_params(axis='y', labelcolor='red')

    # 右軸(ax2)にask1Ivを描画
    ax2 = ax1.twinx()
    ax2.set_ylabel('IV', color='blue')
    l3 = ax2.plot(symbol_data.index, symbol_data['ask1Iv'], label='Ask1 IV', color='blue', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor='blue')

    plt.title(f'ask1Price, ask1Iv, BTC Price Time Series for {target_symbol}', fontsize=14)

    # 凡例をまとめて表示
    lines = l1 + l2 + l3
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

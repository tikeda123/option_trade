import os
import sys
from typing import Dict
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

    # ---- オプションデータの読み込み・クレンジング ----
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

    # ---- BTC価格データの読み込み ----
    df_btc = db.load_data_from_datetime_period(
        start_date=start_date,
        end_date=end_date,
        coll_type=MARKET_DATA,
        symbol="BTCUSDT",
        interval=60
    )
    # ここでは df_op に "close" 列として BTC価格を付加している
    df_op["close"] = df_btc["close"]

    # ---- シンボルごとの時系列データに分割 ----
    symbol_groups = process_option_data(df_op)

    # ---- 例: 特定シンボルを指定 ----
    target_symbol = "BTC-28MAR25-100000-C"
    if target_symbol not in symbol_groups:
        print(f"Symbol {target_symbol} not found in data.")
        return

    symbol_data = symbol_groups[target_symbol].copy()
    print(symbol_data)

    # ---- カラムを数値に変換（object -> float） ----
    numeric_cols = ['ask1Iv', 'ask1Price', 'vega']
    for col in numeric_cols:
        if col in symbol_data.columns:
            symbol_data[col] = pd.to_numeric(symbol_data[col], errors='coerce')

    # ---- 日時ソート & 欠損日の forward fill ----
    symbol_data = symbol_data.sort_index()
    symbol_data = symbol_data.resample('D').ffill()

    # ---- 基本統計量表示 ----
    print(f"Symbol: {target_symbol}")
    print(f"Data period: {symbol_data.index.min()} to {symbol_data.index.max()}")
    print(f"Number of data points: {len(symbol_data)}\n")
    for col in numeric_cols:
        if col in symbol_data.columns:
            print(f"{col}: {symbol_data[col].describe()}\n")

    plt.style.use('seaborn-v0_8')

    # ===============================================================
    # (1) Ask1Price と Ask1Iv を比較するグラフ
    # ===============================================================
    fig1, ax1_left = plt.subplots(figsize=(12, 6))

    # 左軸: Ask1 Price
    ax1_left.set_xlabel('Date')
    ax1_left.set_ylabel('Ask1 Price', color='red')
    line1_left = ax1_left.plot(
        symbol_data.index, symbol_data['ask1Price'],
        label='Ask1 Price', color='red', alpha=0.7
    )
    ax1_left.tick_params(axis='y', labelcolor='red')

    # 右軸: Ask1 Iv
    ax1_right = ax1_left.twinx()
    ax1_right.set_ylabel('Ask1 Iv', color='blue')
    line1_right = ax1_right.plot(
        symbol_data.index, symbol_data['ask1Iv'],
        label='Ask1 Iv', color='blue', alpha=0.7
    )
    ax1_right.tick_params(axis='y', labelcolor='blue')

    plt.title(f'(1) Ask1Price vs. Ask1Iv for {target_symbol}')

    # 凡例をまとめて表示
    lines1 = line1_left + line1_right
    labels1 = [l.get_label() for l in lines1]
    ax1_left.legend(lines1, labels1, loc='upper left')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # ===============================================================
    # (2) Ask1Iv と Vega を比較するグラフ
    # ===============================================================
    fig2, ax2_left = plt.subplots(figsize=(12, 6))

    # 左軸: Ask1 Iv
    ax2_left.set_xlabel('Date')
    ax2_left.set_ylabel('Ask1 Iv', color='green')
    line2_left = ax2_left.plot(
        symbol_data.index, symbol_data['ask1Iv'],
        label='Ask1 Iv', color='green', alpha=0.7
    )
    ax2_left.tick_params(axis='y', labelcolor='green')

    # 右軸: Vega
    ax2_right = ax2_left.twinx()
    ax2_right.set_ylabel('Vega', color='orange')
    line2_right = ax2_right.plot(
        symbol_data.index, symbol_data['vega'],
        label='Vega', color='orange', alpha=0.7
    )
    ax2_right.tick_params(axis='y', labelcolor='orange')

    plt.title(f'(2) Ask1Iv vs. Vega for {target_symbol}')

    # 凡例をまとめて表示
    lines2 = line2_left + line2_right
    labels2 = [l.get_label() for l in lines2]
    ax2_left.legend(lines2, labels2, loc='upper left')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

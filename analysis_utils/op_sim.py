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

    # シンボルごとの時系列データを取得
    symbol_groups = process_option_data(df)

    # 特定のシンボルのデータを取り出す例
    #target_symbol = df['symbol'].unique()[0]  # 最初のシンボルを例として使用

    target_symbol = "BTC-28MAR25-95000-P"

    symbol_data = symbol_groups[target_symbol]

    print(symbol_data)

    # 時系列データの基本情報を表示
    print(f"Symbol: {target_symbol}")
    print(f"Data period: {symbol_data.index.min()} to {symbol_data.index.max()}")
    print(f"Number of data points: {len(symbol_data)}")
    print("\nFeature columns:")
    for col in FEATURE_COLS:
        if col in symbol_data.columns:
            print(f"{col}: {symbol_data[col].describe()}")







    # グラフのスタイル設定
    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(15, 7))

    # ask1PriceとbidPriceの時系列プロット
    plt.plot(symbol_data.index, symbol_data['ask1Price'], label='Ask Price', color='red', alpha=0.7)
    plt.plot(symbol_data.index, symbol_data['bid1Price'], label='Bid Price', color='blue', alpha=0.7)

    # グラフの設定
    plt.title(f'Ask/Bid Price Time Series for {target_symbol}', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # x軸の日付表示を見やすく調整
    plt.xticks(rotation=45)
    plt.tight_layout()

    # グラフを表示
    plt.show()

if __name__ == "__main__":
    main()

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

    df = db.load_data_from_datetime_period(datetime(2024, 1, 1),
                                           datetime(2025, 1, 1),
                                           coll_type=MARKET_DATA_TECH,
                                           symbol='BTCUSDT',
                                           interval=1440)

    graph_df = df[['start_at', 'close', 'volatility']]
    graph_df.set_index('start_at', inplace=True)

    # Create figure and axis objects with a single subplot
    fig, ax1 = plt.subplots(figsize=(15, 7))

    # Plot close price on primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Close Price (USDT)', color=color)
    ax1.plot(graph_df.index, graph_df['close'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(30000, 120000)

    # Create second y-axis that shares x-axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Volatility', color=color)
    ax2.plot(graph_df.index, graph_df['volatility'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1)  # Set volatility range from 0 to 1

    # Add title and adjust layout
    plt.title('BTC-USDT Close Price and Volatility Over Time')
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

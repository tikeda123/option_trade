import os
import sys
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

def main():
    db = MongoDataLoader()
    df = db.load_data_from_datetime_period(
        datetime(2023, 1, 1),
        datetime(2025, 1, 1),
        coll_type=MARKET_DATA_TECH,
        symbol='BTCUSDT',
        interval=60
    )

    graph_df = df[['start_at', 'close', 'volume', 'macdhist', 'rsi', 'volatility']].copy()
    graph_df.dropna(inplace=True)
    graph_df.sort_values('start_at', inplace=True)
    graph_df.reset_index(drop=True, inplace=True)

    BLOCK_SIZE = 72
    feature_list = []
    time_index_list = []

    for i in range(0, len(graph_df) - BLOCK_SIZE + 1, BLOCK_SIZE):
        chunk = graph_df.iloc[i : i+BLOCK_SIZE]
        start_time = chunk['start_at'].iloc[0]
        end_time   = chunk['start_at'].iloc[-1]

        avg_volume      = chunk['volume'].mean()
        avg_macdhist    = chunk['macdhist'].mean()
        avg_rsi         = chunk['rsi'].mean()
        avg_volatility  = chunk['volatility'].mean()

        start_price = chunk['close'].iloc[0]
        end_price   = chunk['close'].iloc[-1]
        pct_return  = (end_price - start_price) / start_price * 100

        feature_list.append({
            'avg_volume':     avg_volume,
            'avg_macdhist':   avg_macdhist,
            'avg_rsi':        avg_rsi,
            'avg_volatility': avg_volatility,
            'pct_return':     pct_return
        })
        time_index_list.append((start_time, end_time))

    feature_df = pd.DataFrame(feature_list)
    feature_df['chunk_start'] = [t[0] for t in time_index_list]
    feature_df['chunk_end']   = [t[1] for t in time_index_list]

    # 相関ヒートマップ
    corr_cols = ['avg_volume', 'avg_macdhist', 'avg_rsi', 'avg_volatility', 'pct_return']
    corr_df = feature_df[corr_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of Features")
    plt.show()

    # (2) Pairplotで各変数ペアの散布図をまとめて表示
    sns.pairplot(
        data=feature_df[corr_cols],
        diag_kind="kde",      # 対角にKDE(カーネル密度)を表示
        corner=False,         # 上下どちらか一方だけ表示する場合は True
        plot_kws={"alpha": 0.7}
    )
    plt.suptitle("Pairplot of Key Features", y=1.02)
    plt.show()

if __name__ == "__main__":
    main()


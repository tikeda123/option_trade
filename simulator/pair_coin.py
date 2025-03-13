import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# ユーザー環境に合わせたパス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *

def analyze_rolling_correlation(btc_df, eth_df, window_hours=168):
    """
    BTCとETHの「close」値のローリング相関を算出し、時系列で可視化する関数。
    window_hours: ローリング窓のサイズ（時間単位）
                  例）168時間 = 7日分(24h * 7)
    """
    # データをマージ
    merged_df = pd.merge(btc_df, eth_df, on='start_at', suffixes=('_btc', '_eth'))

    # 時刻をDatetime型に変換し、インデックスに設定（※既にDatetime型なら不要）
    merged_df['start_at'] = pd.to_datetime(merged_df['start_at'])
    merged_df.set_index('start_at', inplace=True)

    # ローリング相関を計算
    # 例：168時間(=7日)の窓を用いて計算
    # 数値ベースのrollingの場合は .rolling(window_hours) でOK（等間隔前提）
    # 時系列ベースで"7D"のように書くことも可能（ただしデータのfreqなどの条件が必要）
    merged_df['rolling_corr'] = merged_df['close_btc'].rolling(window=window_hours).corr(merged_df['close_eth'])

    # ローリング相関をプロット
    plt.figure(figsize=(14, 7))
    plt.plot(merged_df.index, merged_df['rolling_corr'], label=f'Rolling Correlation ({window_hours}h)')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)  # 相関0の線を補助的に表示
    plt.xlabel('Date')
    plt.ylabel('Correlation')
    plt.title(f'BTC vs ETH Rolling Correlation (Window = {window_hours} hours)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    """
    MongoDBからBTCとETHの時系列データを取得し、
    ローリング相関を時系列で表示するメイン関数。
    """
    # MongoDBからBTCの時系列データを取得
    db = MongoDataLoader()
    btc_df = db.load_data_from_datetime_period(
        datetime(2024, 1, 1),
        datetime(2025, 1, 1),
        coll_type=MARKET_DATA_TECH,
        symbol='BTCUSDT',
        interval=60  # 1時間足
    )

    # MongoDBからETHの時系列データを取得
    eth_df = db.load_data_from_datetime_period(
        datetime(2024, 1, 1),
        datetime(2025, 1, 1),
        coll_type=MARKET_DATA_TECH,
        symbol='ETHUSDT',
        interval=60  # 1時間足
    )

    # 必要なカラムのみ抽出
    btc_df = btc_df[['start_at', 'close']]
    eth_df = eth_df[['start_at', 'close']]

    # ローリング相関を可視化
    # ここでは7日分(168時間)の窓で相関を計算
    analyze_rolling_correlation(btc_df, eth_df, window_hours=168)

if __name__ == '__main__':
    main()

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# statsmodels を用いたコインテグレーション検定で必要
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

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
    merged_df['rolling_corr'] = merged_df['close_btc'].rolling(window=window_hours).corr(merged_df['close_eth'])

    # ローリング相関をプロット
    plt.figure(figsize=(14, 7))
    plt.plot(merged_df.index, merged_df['rolling_corr'], label=f'Rolling Correlation ({window_hours}h)')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)  # 相関0の線
    plt.xlabel('Date')
    plt.ylabel('Correlation')
    plt.title(f'BTC vs ETH Rolling Correlation (Window = {window_hours} hours)')
    plt.legend()
    plt.grid(True)
    plt.show()


def analyze_cointegration(btc_df, eth_df):
    """
    BTCとETHの「close」値に対してエングル・グレンジャー検定（コインテグレーション分析）を行う関数。
    """
    # データをマージ
    merged_df = pd.merge(btc_df, eth_df, on='start_at', suffixes=('_btc', '_eth')).dropna()

    # コインテグレーション検定には各時系列が同じ時間軸で揃った状態の配列が必要
    # 今回は単純に close_btc, close_eth を取得
    btc_series = merged_df['close_btc'].values
    eth_series = merged_df['close_eth'].values

    # coint関数を用いてエングル・グレンジャー検定を実行
    # 戻り値: (検定統計量, p値, 使用したラグ数) のタプル
    coint_t, p_value, usedlag = coint(btc_series, eth_series)

    print("=== Engle-Granger Cointegration Test ===")
    print(f"Test Statistic (t-value): {coint_t:.4f}")
    print(f"p-value                : {p_value:.4f}")
    print(f"Used Lag               : {usedlag}")
    print("----------------------------------------")
    if p_value < 0.05:
        print("5%の有意水準で、BTCとETHの間にコインテグレーション（長期的関係）が存在すると判断できます。")
    else:
        print("5%の有意水準では、コインテグレーションは検出されませんでした。")


def main():
    """
    MongoDBからBTCとETHの時系列データを取得し、
    ローリング相関を時系列で表示＆コインテグレーションを検証するメイン関数。
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
    btc_df = btc_df[['start_at', 'close', 'sma']]
    eth_df = eth_df[['start_at', 'close', 'sma']]

    # 1) ローリング相関を可視化
    analyze_rolling_correlation(btc_df, eth_df, window_hours=168)

    # 2) コインテグレーション検定を実行
    analyze_cointegration(btc_df, eth_df)


if __name__ == '__main__':
    main()

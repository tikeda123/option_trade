import os
import sys
from typing import List, Dict, Any
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

def main():
    # MongoDBからBTCUSDTのテクニカルデータを取得
    db = MongoDataLoader()
    df_btc = db.load_data_from_datetime_period(
        datetime(2024, 12, 17, 14, 0, 0),
        datetime(2025, 3, 4, 6, 0, 0),
        coll_type=MARKET_DATA_TECH,
        symbol='BTCUSDT',
        interval=60
    )

    # df_btcの必要カラムの抽出と整列
    df_btc = df_btc[['start_at', 'volatility', 'close']]
    df_btc = df_btc.sort_values(by='start_at')
    # 'start_at'をdatetime型に変換し、共通の' date 'カラムとして扱う
    df_btc['date'] = pd.to_datetime(df_btc['start_at'])

    # CSVデータの読み込み（オプションデータ）
    df = pd.read_csv("cleaned_option_data.csv")
    # 対象シンボルでフィルタリング
    df = df[df['symbol'] == 'BTC-28MAR25-95000-P']
    # 必要なカラム（date, ask1Price, ask1Iv）を抽出し、時刻順にソート
    df = df[['date', 'ask1Price', 'ask1Iv']]
    df = df.sort_values(by='date')
    # 'date'列をdatetime型に変換
    df['date'] = pd.to_datetime(df['date'])

    # プロットの作成
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 左側のy軸にオプションのask1Priceをプロット
    line1, = ax1.plot(df['date'], df['ask1Price'], label='ask1Price', color='red')
    line2, = ax1.plot(df_btc['date'], df_btc['close'], label='close', color='orange')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Option Price")
    ax1.tick_params(axis='y')

    # 右側のy軸にオプションのask1IvとBTCUSDTのvolatilityをプロット
    ax2 = ax1.twinx()
    line3, = ax2.plot(df['date'], df['ask1Iv'], label='ask1Iv', color='blue')
    line4, = ax2.plot(df_btc['date'], df_btc['volatility'], label='volatility', color='green')
    ax2.set_ylabel("IV / Volatility")
    ax2.tick_params(axis='y')

    # 両軸のレジェンドを統合して表示
    lines = [line1, line2, line3, line4]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.title("Option Price vs IV and BTCUSDT Volatility")
    plt.show()

if __name__ == "__main__":
    main()



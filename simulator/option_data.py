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
    db = MongoDataLoader()
    df_btc = db.load_data_from_datetime_period(datetime(2024, 12, 17,14,0,0),
                                           datetime(2025, 3, 4,6,0,0),
                                           coll_type=MARKET_DATA_TECH,
                                           symbol='BTCUSDT',
                                           interval=60)

    df_btc = df_btc[['start_at',  'volatility']]
    df_btc = df_btc.sort_values(by='start_at')
    df_btc['date'] = pd.to_datetime(df_btc['start_at'])

    """
    オプションデータの可視化を行う関数
    """
# CSVデータの読み込み
    df = pd.read_csv("cleaned_option_data.csv")
# 対象シンボルでフィルタリング
    df = df[df['symbol'] == 'BTC-28MAR25-95000-P']

    # 必要なカラムの抽出と時刻順にソート
    df = df[['date', 'ask1Price', 'bid1Price']]
    df = df.sort_values(by='date')

    # 'date'列をdatetime型に変換
    df['date'] = pd.to_datetime(df['date'])


if __name__ == "__main__":
    main()

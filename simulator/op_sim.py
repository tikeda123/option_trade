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
    # MongoDBからBTCの時系列データを取得
    db = MongoDataLoader()
    df = db.load_data_from_datetime_period(datetime(2023, 1, 1),
                                           datetime(2025, 1, 1),
                                           coll_type=MARKET_DATA_TECH,
                                           symbol='BTCUSDT',
                                           interval=60)
    # 利用するカラムのみ抽出
    #graph_df = df[['start_at', 'close', 'ema', 'macdhist', 'roc', 'mfi', 'aroon', 'volatility']]
    graph_df = df[['start_at', 'close', 'volume', 'macdhist', 'rsi', 'volatility']]




if __name__ == "__main__":
    main()

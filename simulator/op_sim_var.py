import os
import sys
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# ※ 以下のTensorFlow等のインポートは元コードからの引用ですが、本アルゴリズム内では使用していません。
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit  # ★ 追加
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---- 追加: data_preprocessing.py をインポート ----
from data_preprocessing import clean_option_data

# ユーザー環境に合わせたパス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *

def main():
    # MongoDBからBTCの時系列データを取得
    db = MongoDataLoader()
    df = db.load_data_from_datetime_period(
        datetime(2024, 1, 1),
        datetime(2025, 1, 1),
        coll_type=MARKET_DATA_TECH,
        symbol='BTCUSDT',
        interval=1440  # 1440分 = 1日
    )

    # 利用するカラムのみ抽出
    graph_df = df[['start_at', 'close']].copy()

    # 日次リターンの計算（単純なパーセント変化率を利用）
    graph_df['returns'] = graph_df['close'].pct_change()
    graph_df.dropna(inplace=True)

    # 信頼水準95%の場合のVaRを計算
    # VaRは、リターン分布の下位5%に位置するリターン値
    confidence_level = 0.95
    var_value = np.quantile(graph_df['returns'], 1 - confidence_level)
    print(f"Value at Risk (VaR) at {int(confidence_level*100)}% confidence level: {var_value:.4f}")

    # CVaR (Conditional VaR) を計算
    # CVaRはVaRを下回るリターンの平均値
    cvar_value = graph_df[graph_df['returns'] <= var_value]['returns'].mean()
    print(f"Conditional Value at Risk (CVaR): {cvar_value:.4f}")

    # ヒストグラムでリターン分布、VaR、CVaRの位置を確認する
    plt.figure(figsize=(10, 6))
    sns.histplot(graph_df['returns'], kde=True, bins=50)
    plt.axvline(x=var_value, color='red', linestyle='--',
                label=f'VaR ({int(confidence_level*100)}%): {var_value:.4f}')
    plt.axvline(x=cvar_value, color='blue', linestyle='--',
                label=f'CVaR: {cvar_value:.4f}')
    plt.title('Distribution of BTC Daily Returns with VaR and CVaR')
    plt.xlabel('Daily Returns')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

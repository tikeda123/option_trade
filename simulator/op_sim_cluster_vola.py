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

# ★ クラスタリング用に追加
from sklearn.cluster import KMeans

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.constants import MARKET_DATA_TECH
from mongodb.data_loader_mongo import MongoDataLoader

def main():
    db = MongoDataLoader()

    df = db.load_data_from_datetime_period(datetime(2023, 1, 1),
                                           datetime(2025, 1, 1),
                                           coll_type=MARKET_DATA_TECH,
                                           symbol='BTCUSDT',
                                           interval=1440)

    # === 2) 必要カラムだけ抽出 & 前処理 ===
    # start_atの形式は "2024-01-01 00:00:00"
    graph_df = df[['start_at', 'volatility']].copy()
    graph_df.dropna(inplace=True)
    graph_df.sort_values('start_at', inplace=True)
    graph_df.reset_index(drop=True, inplace=True)

    # ================================
    # 1) ヒストリカルクラスタ分析 (KMeans)
    # ================================

    # ボラティリティの値を[0,1]に正規化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(graph_df[['volatility']])

    # KMeans で3つのクラスターに分割 (例)
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # 結果をデータフレームに追加
    graph_df['cluster'] = clusters

    # クラスタリング結果の可視化
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=graph_df,
                    x='start_at',
                    y='volatility',
                    hue='cluster',
                    palette='Set2')
    plt.xticks(rotation=45)
    plt.title('Volatility Clusters Over Time')
    plt.tight_layout()
    plt.show()

    # ================================
    # 2) 時間の周期でのクラスタ分析
    # ================================
    # 例として、曜日ごと、月ごとの平均ボラティリティでクラスタリング

    # 日付型に変換
    graph_df['start_at_dt'] = pd.to_datetime(graph_df['start_at'])

    # 曜日 (0=月曜日, 6=日曜日)
    graph_df['weekday'] = graph_df['start_at_dt'].dt.weekday
    # 月 (1=1月, 12=12月)
    graph_df['month'] = graph_df['start_at_dt'].dt.month

    # --- (A) 曜日ごとの平均ボラティリティでクラスタリング
    weekday_df = graph_df.groupby('weekday')['volatility'].mean().reset_index()
    weekday_df.rename(columns={'volatility': 'mean_volatility'}, inplace=True)

    # KMeansで2クラスターに分類 (例)
    kmeans_weekday = KMeans(n_clusters=2, random_state=42)
    weekday_df['cluster'] = kmeans_weekday.fit_predict(weekday_df[['mean_volatility']])

    print("▼ 曜日ごとの平均ボラティリティとクラスタ結果")
    print(weekday_df)

    # 曜日クラスタを棒グラフで可視化
    plt.figure(figsize=(8, 4))
    sns.barplot(data=weekday_df, x='weekday', y='mean_volatility', hue='cluster', palette='Set2')
    plt.title('Mean Volatility by Weekday (Clustered)')
    plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.tight_layout()
    plt.show()

    # --- (B) 月ごとの平均ボラティリティでクラスタリング
    month_df = graph_df.groupby('month')['volatility'].mean().reset_index()
    month_df.rename(columns={'volatility': 'mean_volatility'}, inplace=True)

    # KMeansで2クラスターに分類 (例)
    kmeans_month = KMeans(n_clusters=2, random_state=42)
    month_df['cluster'] = kmeans_month.fit_predict(month_df[['mean_volatility']])

    print("▼ 月ごとの平均ボラティリティとクラスタ結果")
    print(month_df)

    # 月クラスタを棒グラフで可視化
    plt.figure(figsize=(8, 4))
    sns.barplot(data=month_df, x='month', y='mean_volatility', hue='cluster', palette='Set2')
    plt.title('Mean Volatility by Month (Clustered)')
    plt.xticks(range(12), range(1, 13))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


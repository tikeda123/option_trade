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

# 追加のライブラリ
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture  # GMMをインポート

def main():
    # === 1) MongoDBから時系列データを取得 ===
    db = MongoDataLoader()
    df = db.load_data_from_datetime_period(
        datetime(2023, 1, 1),
        datetime(2025, 1, 1),
        coll_type=MARKET_DATA_TECH,
        symbol='BTCUSDT',
        interval=60
    )

    # === 2) 必要カラムだけ抽出 & 前処理 ===
    graph_df = df[['start_at', 'close', 'volume', 'macdhist', 'rsi', 'volatility']].copy()
    graph_df.dropna(inplace=True)
    graph_df.sort_values('start_at', inplace=True)
    graph_df.reset_index(drop=True, inplace=True)

    # === 3) 3日(72時間)単位でブロック分割し、各ブロックの特徴量を作成 ===
    BLOCK_SIZE = 72
    feature_list = []
    time_index_list = []  # ブロック開始〜終了時間の記録

    for i in range(0, len(graph_df) - BLOCK_SIZE + 1, BLOCK_SIZE):
        chunk = graph_df.iloc[i : i+BLOCK_SIZE]
        start_time = chunk['start_at'].iloc[0]
        end_time   = chunk['start_at'].iloc[-1]

        avg_volume      = chunk['volume'].mean()
        avg_macdhist    = chunk['macdhist'].mean()
        avg_rsi         = chunk['rsi'].mean()
        avg_volatility  = chunk['volatility'].mean()

        # 価格パフォーマンスを評価するために start_price と end_price を記録
        start_price = chunk['close'].iloc[0]
        end_price   = chunk['close'].iloc[-1]
        pct_return  = (end_price - start_price) / start_price * 100  # 3日間でのリターン率(％)

        feature_list.append({
            'avg_volume':     avg_volume,
            'avg_macdhist':   avg_macdhist,
            'avg_rsi':        avg_rsi,
            'avg_volatility': avg_volatility,
            'start_price':    start_price,
            'end_price':      end_price,
            'pct_return':     pct_return
        })
        time_index_list.append((start_time, end_time))

    feature_df = pd.DataFrame(feature_list)
    feature_df['chunk_start'] = [t[0] for t in time_index_list]
    feature_df['chunk_end']   = [t[1] for t in time_index_list]

    # === 4) クラスタリングをGaussian Mixture Models (GMM)に変更 ===
    cluster_cols = ['avg_volume', 'avg_macdhist', 'avg_rsi', 'avg_volatility']
    X = feature_df[cluster_cols].values

    # スケーリング
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # GMMによるクラスタリング (例としてn_components=4)
    gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
    gmm.fit(X_scaled)

    # 予測(クラスタ割り当て)を取得
    labels = gmm.predict(X_scaled)
    feature_df['cluster_label'] = labels

    # --- クラスタごとの特徴量平均を出力して確認 ---
    cluster_means = feature_df.groupby('cluster_label')[
        ['avg_volume', 'avg_macdhist', 'avg_rsi', 'avg_volatility']
    ].mean()
    print("\n=== Cluster Means (by cluster_label) ===")
    print(cluster_means)

    # === 5) クラスタごとのパフォーマンス指標を計算する ===
    cluster_perf = feature_df.groupby('cluster_label')['pct_return'].agg(['mean', 'median', 'std', 'count'])
    print("\n=== Cluster Performance (pct_return) ===")
    print(cluster_perf)

    # === 【Pair Plot】===
    # hue='cluster_label' とすることで、GMMのクラスタを可視化
    plt.figure(figsize=(10, 8))
    sns.pairplot(
        data=feature_df,
        vars=['avg_rsi', 'avg_macdhist', 'avg_volume', 'avg_volatility', 'pct_return'],
        hue='cluster_label',
        diag_kind='kde'
    )
    plt.suptitle("Pair Plot of Selected Features (incl. pct_return) by Cluster (GMM)", y=1.02)
    plt.show()



if __name__ == "__main__":
    main()


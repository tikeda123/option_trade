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

# 追加ライブラリ
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def main():
    # === 1) データ読み込み ===
    db = MongoDataLoader()
    df = db.load_data_from_datetime_period(
        datetime(2023, 1, 1),
        datetime(2025, 1, 1),
        coll_type=MARKET_DATA_TECH,
        symbol='BTCUSDT',
        interval=60
    )

    # === 2) 必要カラムを選択 ===
    graph_df = df[['start_at', 'close', 'volume', 'macdhist', 'rsi', 'volatility']].copy()
    # 欠損があれば落とす
    graph_df.dropna(inplace=True)

    # === 3) クラスタリング対象の特徴量を選定 (volume, macdhist, rsi, volatility) ===
    cluster_features = ['volume', 'macdhist', 'rsi', 'volatility']
    cluster_df = graph_df[cluster_features].copy()

    # === 4) スケーリング (標準化) ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_df)

    # === 5) k=4 でクラスタリング ===
    #  (エルボー法やシルエット分析により4が適切と判断した想定)
    k = 4
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(X_scaled)

    # === 6) 結果を保存 ===
    cluster_df['cluster_label'] = cluster_labels

    # === 7) クラスタ別の平均値を確認 ===
    cluster_summary = cluster_df.groupby('cluster_label').mean()
    print("\n=== Cluster Summary (mean) ===")
    print(cluster_summary)

    # === 8) クラスタラベルを「相場レジーム名」に対応付ける ===
    # 日本語の名前
    label_map = {
        0: "閑散中立",   # 低Volume, 低Volatility, RSI中立, MACDほぼ0
        1: "乱高下強気", # ボラ高, 強気MACD, RSIやや高め
        2: "安定上昇",   # Volume大きい, MACD大きくプラス, RSI高
        3: "強い下落",   # Volume大きい, MACD大きくマイナス, RSI低め
    }
    cluster_df['regime_name_jp'] = cluster_df['cluster_label'].map(label_map)

    # 英語の名前に対応する辞書
    label_map_en = {
        0: "Calm/Neutral",      # Low volume, low volatility, neutral RSI, near-zero MACD
        1: "Volatile Bullish",   # High volatility, bullish MACD, slightly high RSI
        2: "Steady Uptrend",     # Large volume, largely positive MACD, high RSI
        3: "Strong Downtrend",   # Large volume, largely negative MACD, low RSI
    }
    cluster_df['regime_name_en'] = cluster_df['cluster_label'].map(label_map_en)

    # === 9) 元の DataFrame (graph_df) へレジーム名を紐付ける (必要なら) ===
    graph_df['cluster_label'] = cluster_df['cluster_label']
    graph_df['regime_name_jp'] = cluster_df['regime_name_jp']
    graph_df['regime_name_en'] = cluster_df['regime_name_en']

    # === 10) PCA で2次元に可視化 (任意) ===
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=cluster_df['regime_name_en'],  # 英語の名前を使用
        palette='Set1',
        alpha=0.7
    )
    plt.title(f"K-Means (k={k}) Clustering - PCA 2D Plot")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Regime")
    plt.tight_layout()
    plt.show()

    # === 11) 結果の一部を表示 ===
    print("\n=== Head of final DataFrame with cluster ===")
    print(graph_df[['start_at', 'close', 'volume', 'macdhist', 'rsi', 'volatility',
                    'cluster_label', 'regime_name_jp', 'regime_name_en']].head(10))

if __name__ == "__main__":
    main()

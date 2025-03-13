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

# ★★ 追加でインポートするライブラリ ★★
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def main():
    # 1) MongoDBからBTCの時系列データを取得
    db = MongoDataLoader()
    df = db.load_data_from_datetime_period(
        datetime(2023, 1, 1),
        datetime(2025, 1, 1),
        coll_type=MARKET_DATA_TECH,
        symbol='BTCUSDT',
        interval=60
    )

    # 2) 利用するカラムのみ抽出
    #   必要に応じて 'high', 'low' があれば 'price_range' を作ってもよい
    #   ここでは volume, macdhist, rsi, volatility を使う例とする
    graph_df = df[['start_at', 'close', 'volume', 'macdhist', 'rsi', 'volatility','mfi']].copy()

    # データが欠損している行を落としておく
    graph_df.dropna(inplace=True)

    # 3) クラスタリング用の特徴量を選定
    #    （価格レベルに左右されにくい指標を例示）
    cluster_features = ['volume', 'macdhist', 'rsi', 'volatility','mfi']
    cluster_df = graph_df[cluster_features].copy()  # 解析用に抜き出し

    # 4) 標準化（スケーリング）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_df)  # numpy array になる

    # 5) クラスタ数 k を決めるためのエルボー法 (SSE を観察)
    sse = []
    K_range = range(2, 8)  # 2〜7クラスタまで試す
    for k in K_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans_temp.fit(X_scaled)
        sse.append(kmeans_temp.inertia_)

    plt.figure(figsize=(6, 4))
    plt.plot(list(K_range), sse, marker='o')
    plt.title("Elbow Method for K-Means")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Sum of Squared Distances (Inertia)")
    plt.show()

    # 6) シルエットスコアでの評価（補足的に）
    #    時間がかかる場合は、ここはコメントアウトでも可
    sil_scores = []
    for k in K_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels_temp = kmeans_temp.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels_temp)
        sil_scores.append(sil)

    plt.figure(figsize=(6, 4))
    plt.plot(list(K_range), sil_scores, marker='o', color='orange')
    plt.title("Silhouette Score for K-Means")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.show()

    # 7) ここでは仮に k=4 を選ぶ例とする（実際には上記結果を見て判断）
    best_k = 5
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(X_scaled)

    # 8) cluster_df にクラスタラベルを付ける
    cluster_df['cluster_label'] = cluster_labels

    # 9) クラスタごとの平均値などを確認
    summary_stats = cluster_df.groupby('cluster_label').mean()
    print("\n=== Cluster Summary (mean) ===")
    print(summary_stats)

    # 10) 結果の可視化例：2次元に削減してクラスタを色分け
    #     ここでは簡単に PCA で2次元化してみる
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1],
        hue=cluster_labels, palette='Set1',
        alpha=0.7
    )
    plt.title(f"K-Means (k={best_k}) Clustering - PCA 2D Plot")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Cluster")
    plt.show()

    # 参考: cluster_label をオリジナルの DataFrame (graph_df) に対応づけたい場合
    # index がずれていると困るので、一度 index を揃えておくか
    # あるいは cluster_df を結合して「同じ並び」で使うなど工夫する。
    # 例: graph_df.loc[cluster_df.index, 'cluster_label'] = cluster_df['cluster_label']

if __name__ == "__main__":
    main()


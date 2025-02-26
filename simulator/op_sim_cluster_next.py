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
from sklearn.cluster import KMeans

def main():
    # === 1) MongoDBから時系列データを取得 ===
    db = MongoDataLoader()
    df = db.load_data_from_datetime_period(
        datetime(2024, 1, 1),
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

        # 追加: 価格パフォーマンスを評価するために start_price と end_price を記録
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

    # === 4) クラスタリング (k=4) ===
    cluster_cols = ['avg_volume', 'avg_macdhist', 'avg_rsi', 'avg_volatility']
    X = feature_df[cluster_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = 4
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X_scaled)

    feature_df['cluster_label'] = labels

    # --- クラスタごとの特徴量平均を出力して確認 ---
    cluster_means = feature_df.groupby('cluster_label')[
        ['avg_volume', 'avg_macdhist', 'avg_rsi', 'avg_volatility']
    ].mean()
    print("\n=== Cluster Means (by cluster_label) ===")
    print(cluster_means)

    # === 5) クラスタごとのパフォーマンス指標を計算する ===
    # 例: リターン率の平均, 中央値, 標準偏差など
    cluster_perf = feature_df.groupby('cluster_label')['pct_return'].agg(['mean', 'median', 'std', 'count'])
    print("\n=== Cluster Performance (pct_return) ===")
    print(cluster_perf)

    # === 6) ラベルを英語に対応付ける ===
    label_map = {
        0: "Steady Uptrend",
        1: "Volatile Bullish",
        2: "Quiet & Neutral",
        3: "Strong Bearish"
    }
    feature_df['regime_name'] = feature_df['cluster_label'].map(label_map)

    # === 7) 元の1時間足にラベルを付与 ===
    graph_df['cluster_label'] = np.nan
    graph_df['regime_name']   = np.nan

    for idx, row in feature_df.iterrows():
        c_start = row['chunk_start']
        c_end   = row['chunk_end']
        c_label = row['cluster_label']
        c_name  = row['regime_name']

        cond = (graph_df['start_at'] >= c_start) & (graph_df['start_at'] <= c_end)
        graph_df.loc[cond, 'cluster_label'] = c_label
        graph_df.loc[cond, 'regime_name']   = c_name

    # === 8) 表示期間 (2024-10-01 ~ 2024-12-01) のみ抽出 ===
    start_filter = datetime(2024, 4, 1)
    end_filter   = datetime(2025, 1, 1)
    plot_df = graph_df.loc[
        (pd.to_datetime(graph_df['start_at']) >= start_filter) &
        (pd.to_datetime(graph_df['start_at']) < end_filter)
    ].copy()

    plt.figure(figsize=(12, 6))

    # クラスタ名 -> 色 の対応表（お好みで変更可）
    regime_color_map = {
        "Steady Uptrend":   "blue",
        "Volatile Bullish": "red",
        "Quiet & Neutral":  "green",
        "Strong Bearish":   "orange"
    }

    # feature_dfを時系列順にソート（chunk_start が昇順になるように）
    feature_df = feature_df.sort_values(by='chunk_start').reset_index(drop=True)

    # チャンクごとに区間を切り出して、その部分だけ plot する
    for idx, row in feature_df.iterrows():
        chunk_start = row['chunk_start']
        chunk_end   = row['chunk_end']
        regime_name = row['regime_name']

        # そのチャンクに対応する1時間足データを抜き出し
        chunk_data = plot_df[
            (plot_df['start_at'] >= chunk_start) &
            (plot_df['start_at'] <= chunk_end)
        ]

        # クラスタ名に応じた色を取得
        color = regime_color_map.get(regime_name, "gray")

        # 区間の線を描画
        plt.plot(chunk_data['start_at'], chunk_data['close'],
                 color=color, linewidth=1.5)

    plt.title("BTC Close Price (2024-10-01 to 2024-12-01) - 3-day block Clusters")
    plt.xlabel("Time")
    plt.ylabel("Close Price")
    plt.xticks(rotation=30)

    # 凡例を手動で作成 (4クラスタ分)
    import matplotlib.patches as mpatches
    legend_handles = []
    for name, col in regime_color_map.items():
        legend_handles.append(mpatches.Patch(color=col, label=name))
    plt.legend(handles=legend_handles, title="Regime")

    plt.tight_layout()
    plt.show()

    # --- 9) 前クラスタとの比較(Shift)と遷移確率ヒートマップ ---

    # 9-1) 前のブロックのクラスタをshift(1)で作成
    #      ※最初の行は前のクラスタが無いので NaN になります
    feature_df['prev_cluster_label'] = feature_df['cluster_label'].shift(1)

    # 9-2) prev_cluster_label が NaN の行は除外 (最初の行)
    temp_df = feature_df.dropna(subset=['prev_cluster_label']).copy()

    # 9-3) 集計（行が「前のクラスタ」、列が「現在のクラスタ」）
    transition_counts = temp_df.groupby(['prev_cluster_label', 'cluster_label']).size().unstack(fill_value=0)

    # 9-4) 行方向 (Fromクラスタ) による正規化 → 遷移確率行列を作る
    transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0)

    print("\n=== Transition Counts ===")
    print(transition_counts)

    print("\n=== Transition Probability Matrix ===")
    print(transition_probs)

    # 9-5) ヒートマップで可視化
    plt.figure(figsize=(6, 4))
    sns.heatmap(transition_probs, annot=True, cmap='Blues', fmt=".2f")
    plt.title("Cluster Transition Probability Matrix")
    plt.xlabel("To Cluster Label")
    plt.ylabel("From Cluster Label")
    plt.show()

    # --- デバッグ用の表示 ---
    print("\n=== feature_df (chunk-level) ===")
    print(feature_df.head(10))

    print("\n=== plot_df (2024-10-01 to 2024-12-01) with cluster_label ===")
    print(plot_df[['start_at', 'close', 'cluster_label', 'regime_name']])

if __name__ == "__main__":
    main()
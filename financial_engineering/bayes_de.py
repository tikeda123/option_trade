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

from sklearn.mixture import GaussianMixture

from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *


# 正規分布のPDFを計算するユーティリティ
def gaussian_pdf(x, mean, std):
    """
    x: 観測値
    mean: 平均
    std:  標準偏差
    """
    if std <= 0:
        # 分散が 0 以下は無効なケースなので、非常に小さい値を返す
        return 1e-12
    coeff = 1.0 / (np.sqrt(2.0 * np.pi) * std)
    exp_val = -0.5 * ((x - mean) / std)**2
    return coeff * np.exp(exp_val)


def main():
    # MongoDBからBTCの時系列データを取得
    db = MongoDataLoader()
    df = db.load_data_from_datetime_period(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2025, 1, 1),
        coll_type=MARKET_DATA_TECH,
        symbol='BTCUSDT',
        interval=1440  # 日足の例
    )

    # 利用するカラムのみ抽出
    graph_df = df[['start_at', 'close', 'macdhist']].copy()

    # 時間でソート（Mongoからの取得順が時系列順とは限らない場合に備えて）
    graph_df.sort_values('start_at', inplace=True)
    graph_df.reset_index(drop=True, inplace=True)

    # -------------------------------
    # 1. GMMで macdhist の分布を学習し、3成分に分解
    # -------------------------------
    macd_data = graph_df['macdhist'].values.reshape(-1, 1)

    # GMM を使って3つのガウス分布をフィット
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(macd_data)

    # 各ガウス成分の平均と分散(共分散)を取得
    means = gmm.means_.flatten()         # shape = (3,)
    stds = np.sqrt(gmm.covariances_.flatten())  # shape = (3,)

    # 平均値を昇順にソートして、「Down < Range < Up」と割り当て
    sorted_indices = np.argsort(means)  # 平均値が小さい順にインデックスを並べる
    down_idx = sorted_indices[0]
    range_idx = sorted_indices[1]
    up_idx   = sorted_indices[2]

    # emission_params は [Up, Down, Range] の順番で格納したいので、
    # ここでは Up, Down, Range の順序でタプルを作る
    # ただし下のベイズフィルタ部分では 0=Up, 1=Down, 2=Range としているので
    # それに合わせて配列の順番を合わせることが重要
    emission_params = [
        (means[up_idx],   stds[up_idx]),    # Up    => index 0
        (means[down_idx], stds[down_idx]),  # Down  => index 1
        (means[range_idx], stds[range_idx]) # Range => index 2
    ]

    print("=== GMM の推定結果 (平均, 標準偏差) ===")
    print("Up   (index=0):", emission_params[0])
    print("Down (index=1):", emission_params[1])
    print("Range(index=2):", emission_params[2])
    print("=====================================")

    # -------------------------------
    # 2. 状態遷移確率行列の定義 (3x3)
    #    rows: 前時点の状態, cols: 次時点の状態
    #    上昇(0), 下降(1), レンジ(2)
    # -------------------------------
    transition_matrix = np.array([
        [0.8,  0.05, 0.15],  # Up -> [Up, Down, Range]
        [0.05, 0.8,  0.15],  # Down -> [Up, Down, Range]
        [0.15, 0.15, 0.7 ]   # Range -> [Up, Down, Range]
    ])

    # -------------------------------
    # 3. 初期分布 (prior) の設定
    # -------------------------------
    prior = np.array([0.3, 0.3, 0.4])  # Up, Down, Range

    # -------------------------------
    # 4. ベイズフィルターによる逐次状態推定
    # -------------------------------
    n = len(graph_df)
    num_states = 3
    filtered_prob = np.zeros((n, num_states))

    macdhist_values = graph_df['macdhist'].values

    # t=0
    first_obs = macdhist_values[0]
    likelihoods_0 = np.array([
        gaussian_pdf(first_obs, emission_params[s][0], emission_params[s][1])
        for s in range(num_states)
    ])
    tmp = prior * likelihoods_0
    filtered_prob[0] = tmp / tmp.sum() if tmp.sum() > 0 else prior

    # t=1 以降
    for t in range(1, n):
        obs = macdhist_values[t]
        # 予測ステップ
        pred_t = filtered_prob[t-1].dot(transition_matrix)
        # 観測更新
        likelihoods_t = np.array([
            gaussian_pdf(obs, emission_params[s][0], emission_params[s][1])
            for s in range(num_states)
        ])
        tmp = pred_t * likelihoods_t
        sum_tmp = tmp.sum()
        filtered_prob[t] = tmp / sum_tmp if sum_tmp > 0 else tmp

    # -------------------------------
    # 5. 最尤推定 (argmax) を取り、状態を文字列に変換
    # -------------------------------
    # 注意: 0=Up, 1=Down, 2=Range としている
    state_map = {0: 'Up', 1: 'Down', 2: 'Range'}
    most_likely_states = np.argmax(filtered_prob, axis=1)
    state_labels = [state_map[s] for s in most_likely_states]

    # -------------------------------
    # 6. 結果を DataFrame に付与
    # -------------------------------
    graph_df['prob_up'] = filtered_prob[:, 0]
    graph_df['prob_down'] = filtered_prob[:, 1]
    graph_df['prob_range'] = filtered_prob[:, 2]
    graph_df['most_likely_state'] = state_labels

    print("\n=== 推定結果サンプル ===")
    print(graph_df.head(10))

    # -------------------------------
    # 7. 可視化
    # -------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(graph_df['start_at'], graph_df['macdhist'], label='MACD Hist', color='blue', alpha=0.5)

    # 状態に応じてカラーを分ける
    colors = {'Up':'red', 'Down':'green', 'Range':'gray'}
    for state_name in colors:
        idx = (graph_df['most_likely_state'] == state_name)
        ax.scatter(
            graph_df.loc[idx, 'start_at'],
            graph_df.loc[idx, 'macdhist'],
            color=colors[state_name],
            label=state_name,
            s=10
        )

    ax.set_title("BTC State Estimation by Bayesian Filter (GMM-based emission)")
    ax.set_xlabel("Time")
    ax.set_ylabel("MACD Hist")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()


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
    # --- 1. データの読み込み ---
    db = MongoDataLoader()
    df = db.load_data_from_datetime_period(
        start_date=datetime(2025, 2, 1),
        end_date=datetime(2025, 3, 9),
        coll_type=MARKET_DATA_TECH,
        symbol='BTCUSDT',
        interval=60
    )

    # 利用するカラム: close(価格), upper2, lower2, middle
    graph_df = df[['start_at', 'close', 'upper2', 'lower2', 'middle']].copy()
    graph_df.dropna(subset=['close', 'upper2', 'lower2', 'middle'], inplace=True)
    graph_df.reset_index(drop=True, inplace=True)

    # --- 2. 1シグマのバンドを導出 ---
    # 前提: upper2 = middle + 2*sigma, lower2 = middle - 2*sigma なので
    # sigma = (upper2 - middle) / 2（または (middle - lower2) / 2)
    graph_df['sigma'] = (graph_df['upper2'] - graph_df['middle']) / 2
    graph_df['upper1'] = graph_df['middle'] + graph_df['sigma']
    graph_df['lower1'] = graph_df['middle'] - graph_df['sigma']

    # --- 3. バンドウォークの検出と状態付与 ---
    # 定義:
    # ・初めに、2シグマのバリア（upper2 or lower2）を突破した場合にブレイクアウトとする
    # ・その後、上側ブレイクなら close が upper1 以上、下側ブレイクなら close が lower1 以下であれば
    #   ブレイクアウト状態（band walk）が継続するものとする
    # ・一度1シグマ領域内に戻った場合、ブレイクアウトカウントおよび状態をリセット
    #
    # breakout_state:
    #    1  → 上側ブレイク状態（1シグマ以上上側に残っている）
    #   -1  → 下側ブレイク状態（1シグマ以上下側に残っている）
    #    0  → ブレイクアウト状態でない（または回復）

    graph_df['band_walk'] = 0       # 各期間での連続カウント
    graph_df['breakout_state'] = 0  # 状態フラグ
    walk_count = 0
    breakout_direction = None

    for i in range(len(graph_df)):
        close_price = graph_df.loc[i, 'close']
        upper2 = graph_df.loc[i, 'upper2']
        lower2 = graph_df.loc[i, 'lower2']
        upper1 = graph_df.loc[i, 'upper1']
        lower1 = graph_df.loc[i, 'lower1']

        if walk_count == 0:
            # まだブレイクアウトしていない状態
            if close_price > upper2:
                breakout_direction = 'up'
                walk_count = 1  # 初回のブレイクアウトをカウント
                current_state = 1
            elif close_price < lower2:
                breakout_direction = 'down'
                walk_count = 1
                current_state = -1
            else:
                current_state = 0
        else:
            # 既にブレイクアウト中の場合
            if breakout_direction == 'up':
                if close_price >= upper1:
                    walk_count += 1
                    current_state = 1
                else:
                    walk_count = 0
                    breakout_direction = None
                    current_state = 0
            elif breakout_direction == 'down':
                if close_price <= lower1:
                    walk_count += 1
                    current_state = -1
                else:
                    walk_count = 0
                    breakout_direction = None
                    current_state = 0

        graph_df.loc[i, 'band_walk'] = walk_count
        graph_df.loc[i, 'breakout_state'] = current_state

    max_band_walk = graph_df['band_walk'].max()
    print(f"最大バンドウォーク連続期間: {max_band_walk}")

    # --- 4. マルコフ連鎖的状態遷移の分析 ---
    # breakout_state をもとに、現在の状態から次の状態への遷移回数・確率を算出
    graph_df['next_state'] = graph_df['breakout_state'].shift(-1)
    transition_df = graph_df.dropna(subset=['next_state']).copy()

    transition_counts = transition_df.groupby(['breakout_state', 'next_state']).size().unstack(fill_value=0)
    transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0)

    print("=== 状態遷移回数 ===")
    print(transition_counts)
    print("\n=== 状態遷移確率 ===")
    print(transition_probs)

    # 各遷移確率の抽出
    if 1 in transition_probs.index and 0 in transition_probs.columns:
        p_upper_to_inner = transition_probs.loc[1, 0]
    else:
        p_upper_to_inner = np.nan
    if 1 in transition_probs.index and 1 in transition_probs.columns:
        p_upper_to_upper = transition_probs.loc[1, 1]
    else:
        p_upper_to_upper = np.nan

    if -1 in transition_probs.index and 0 in transition_probs.columns:
        p_lower_to_inner = transition_probs.loc[-1, 0]
    else:
        p_lower_to_inner = np.nan
    if -1 in transition_probs.index and -1 in transition_probs.columns:
        p_lower_to_lower = transition_probs.loc[-1, -1]
    else:
        p_lower_to_lower = np.nan

    print(f"\nバンド上抜け状態から次足でバンド内に戻る確率: {p_upper_to_inner}")
    print(f"バンド上抜け状態が継続する(バンドウォークが続く)確率: {p_upper_to_upper}")
    print(f"\nバンド下抜け状態から次足でバンド内に戻る確率: {p_lower_to_inner}")
    print(f"バンド下抜け状態が継続する(バンドウォークが続く)確率: {p_lower_to_lower}")

    # --- 5. 分布図のプロット ---
    # breakout_state の連続セグメントの持続期間（band_walk）の分布を求める
    # まず、連続する同一状態のグループを識別するために、状態変化のインデックスを作成
    graph_df['state_change'] = (graph_df['breakout_state'] != graph_df['breakout_state'].shift()).cumsum()
    grouped = graph_df.groupby(['state_change', 'breakout_state']).size().reset_index(name='duration')

    # 上側ブレイク状態 (1) の連続期間
    up_segments = grouped[grouped['breakout_state'] == 1]['duration']
    # 下側ブレイク状態 (-1) の連続期間
    down_segments = grouped[grouped['breakout_state'] == -1]['duration']

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(up_segments, bins=range(1, int(up_segments.max()) + 2), edgecolor='black')
    plt.title('Distribution of Upward Breakout Durations (state 1)')
    plt.xlabel('連続期間（1シグマ以上の足数）')
    plt.ylabel('頻度')

    plt.subplot(1, 2, 2)
    plt.hist(down_segments, bins=range(1, int(down_segments.max()) + 2), edgecolor='black')
    plt.title('Distribution of Downward Breakout Durations (state -1)')
    plt.xlabel('連続期間（1シグマ以下の足数）')
    plt.ylabel('頻度')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


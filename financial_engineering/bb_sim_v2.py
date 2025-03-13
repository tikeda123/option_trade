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
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2025, 3, 1),
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

    # --- 3. バンドウォークの検出（連続カウント & breakout_state） ---
    # 定義:
    #   - 初回に 2シグマ(upper2 or lower2) 突破でブレイクアウト開始
    #   - その後、上側は「close >= upper1」、下側は「close <= lower1」であれば連続カウント
    #   - 1シグマ内に戻ったらカウンターをリセット
    #
    # breakout_state:
    #    1  → 上側ブレイク状態（1シグマ以上上側に残っている）
    #   -1  → 下側ブレイク状態（1シグマ以上下側に残っている）
    #    0  → 非ブレイク状態（通常 or 回復）

    graph_df['band_walk'] = 0       # 各足での連続カウント
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
                walk_count = 1
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

    # --- 5. 分布図のプロット (上抜け/下抜けの連続期間) ---
    graph_df['state_change'] = (graph_df['breakout_state'] != graph_df['breakout_state'].shift()).cumsum()
    grouped = graph_df.groupby(['state_change', 'breakout_state']).size().reset_index(name='duration')

    up_segments = grouped[grouped['breakout_state'] == 1]['duration']
    down_segments = grouped[grouped['breakout_state'] == -1]['duration']

    # 例示的にヒストグラム表示
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.hist(up_segments, bins=range(1, int(up_segments.max() + 2)), edgecolor='black')
    # plt.title('Distribution of Upward Breakout Durations (state = 1)')
    # plt.xlabel('連続期間')
    # plt.ylabel('頻度')
    #
    # plt.subplot(1, 2, 2)
    # plt.hist(down_segments, bins=range(1, int(down_segments.max() + 2)), edgecolor='black')
    # plt.title('Distribution of Downward Breakout Durations (state = -1)')
    # plt.xlabel('連続期間')
    # plt.ylabel('頻度')
    #
    # plt.tight_layout()
    # plt.show()


    # -----------------------------------------------------------------
    # 6. 2シグマ突き抜け → 5回以上連続で1シグマ維持した「バンドウォーク」の変化率を統計処理
    # -----------------------------------------------------------------
    # バンドウォーク:
    #   1) 2σを突き抜けたタイミングで開始 (start_price)
    #   2) その後、5回以上連続して1σ以上に居続けたら正式にカウント
    #   3) 1σを割り込んだ（上なら < upper1、下なら > lower1）タイミングで終了 (end_price)
    #   4) 変化率 = (end_price / start_price - 1) * 100[%]

    bandwalks_up = []   # 上抜けバンドウォークの記録
    bandwalks_down = [] # 下抜けバンドウォークの記録

    in_breakout = False
    direction = None
    consecutive = 0
    start_price = None
    start_index = None

    def finalize_walk(i_break, i_end, start_prc, end_prc, direction_str, store_list):
        """
        バンドウォーク終了処理:
          i_break:  開始インデックス
          i_end:    終了インデックス
          start_prc: 開始価格 (2σブレイク時)
          end_prc:   終了価格 (バンドウォーク終了直前の足)
          direction_str: 'up' or 'down'
          store_list:  up or down のリスト
        """
        store_list.append({
            'start_index': i_break,
            'end_index': i_end,
            'start_price': start_prc,
            'end_price': end_prc,
            'direction': direction_str
        })

    # 新たに独立したループで判定（上記の breakout_state 列に頼らず、定義通りに算出するイメージ）
    # なお、1つのループでも可能ですが、可読性のため分けています
    length_df = len(graph_df)
    for i in range(length_df):
        close_price = graph_df.loc[i, 'close']
        up2 = graph_df.loc[i, 'upper2']
        dn2 = graph_df.loc[i, 'lower2']
        up1 = graph_df.loc[i, 'upper1']
        dn1 = graph_df.loc[i, 'lower1']

        if not in_breakout:
            # まだブレイクアウト状態でない → 2σを突き抜けたか？
            if close_price > up2:
                in_breakout = True
                direction = 'up'
                consecutive = 1
                start_price = close_price
                start_index = i
            elif close_price < dn2:
                in_breakout = True
                direction = 'down'
                consecutive = 1
                start_price = close_price
                start_index = i
            else:
                # ブレイクアウトしていない
                continue
        else:
            # すでにブレイクアウト中
            if direction == 'up':
                # 上抜けウォーク中 → 1σ以上維持しているか？
                if close_price >= up1:
                    consecutive += 1
                else:
                    # 維持できなくなった → バンドウォーク終了か判定
                    # 5回以上維持していた場合のみカウント
                    if consecutive >= 5:
                        # 終了価格は一つ前の足がバンドウォークに該当
                        end_index = i - 1
                        end_price = graph_df.loc[end_index, 'close'] if end_index >= 0 else close_price
                        finalize_walk(
                            i_break=start_index,
                            i_end=end_index,
                            start_prc=start_price,
                            end_prc=end_price,
                            direction_str='up',
                            store_list=bandwalks_up
                        )
                    # リセット
                    in_breakout = False
                    direction = None
                    consecutive = 0
                    start_price = None
                    start_index = None

            elif direction == 'down':
                # 下抜けウォーク中 → 1σ以下維持しているか？
                if close_price <= dn1:
                    consecutive += 1
                else:
                    # 維持できなくなった → バンドウォーク終了か判定
                    if consecutive >= 5:
                        end_index = i - 1
                        end_price = graph_df.loc[end_index, 'close'] if end_index >= 0 else close_price
                        finalize_walk(
                            i_break=start_index,
                            i_end=end_index,
                            start_prc=start_price,
                            end_prc=end_price,
                            direction_str='down',
                            store_list=bandwalks_down
                        )
                    # リセット
                    in_breakout = False
                    direction = None
                    consecutive = 0
                    start_price = None
                    start_index = None

    # ループ終了後も break 状態が続いていれば、最終的にバンドウォークを確定できるか確認
    if in_breakout and consecutive >= 5:
        end_index = length_df - 1
        end_price = graph_df.loc[end_index, 'close']
        if direction == 'up':
            finalize_walk(
                i_break=start_index,
                i_end=end_index,
                start_prc=start_price,
                end_prc=end_price,
                direction_str='up',
                store_list=bandwalks_up
            )
        else:
            finalize_walk(
                i_break=start_index,
                i_end=end_index,
                start_prc=start_price,
                end_prc=end_price,
                direction_str='down',
                store_list=bandwalks_down
            )

    # 上抜け/下抜けのバンドウォークをDataFrame化して変化率を計算
    df_up = pd.DataFrame(bandwalks_up)
    df_down = pd.DataFrame(bandwalks_down)

    if not df_up.empty:
        df_up['change_rate(%)'] = (df_up['end_price'] / df_up['start_price'] - 1) * 100
    if not df_down.empty:
        df_down['change_rate(%)'] = (df_down['end_price'] / df_down['start_price'] - 1) * 100

    # 統計量を表示
    print("\n=== [上抜け] バンドウォーク数:", len(df_up))
    if not df_up.empty:
        print(df_up[['start_index', 'end_index', 'start_price', 'end_price', 'change_rate(%)']].head())
        print(df_up['change_rate(%)'].describe())

    print("\n=== [下抜け] バンドウォーク数:", len(df_down))
    if not df_down.empty:
        print(df_down[['start_index', 'end_index', 'start_price', 'end_price', 'change_rate(%)']].head())
        print(df_down['change_rate(%)'].describe())

    # もし上下ブレイクをまとめて集計したい場合:
    if not df_up.empty or not df_down.empty:
        df_all = pd.concat([df_up, df_down], ignore_index=True)
        print("\n=== [上下合算] バンドウォーク数:", len(df_all))
        print(df_all['change_rate(%)'].describe())

    # # 変化率のヒストグラムを描画（任意）
    # plt.figure(figsize=(10, 4))
    # if not df_up.empty:
    #     sns.histplot(df_up['change_rate(%)'], kde=True, color='blue', label='Up Breakout')
    # if not df_down.empty:
    #     sns.histplot(df_down['change_rate(%)'], kde=True, color='red', label='Down Breakout')
    # plt.title('Bandwalk Change Rate Distribution')
    # plt.xlabel('Change Rate (%)')
    # plt.ylabel('Frequency')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()


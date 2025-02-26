import os
import sys
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# 以下のTensorFlow等のインポートは元コードからの引用ですが、本アルゴリズム内では使用していません。
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

# data_preprocessing.py をインポート
from data_preprocessing import clean_option_data

# ユーザー環境に合わせたパス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *

def calculate_volume_profile(df: pd.DataFrame, bins: int = 50):
    """
    日足の high～low の価格帯に出来高を線形に分配する簡易的なボリュームプロファイルを計算する。
    df には 'high', 'low', 'volume' カラムが含まれていることを想定。

    Parameters
    ----------
    df : pd.DataFrame
        'high', 'low', 'volume' を含む時系列データ
    bins : int
        価格帯を分割するビンの数

    Returns
    -------
    bin_edges : np.ndarray
        ビンの境界値 (bins+1個)
    volume_distribution : np.ndarray
        各ビンに対応する出来高の合計 (bins個)
    """
    price_min = df['low'].min()
    price_max = df['high'].max()

    bin_edges = np.linspace(price_min, price_max, bins + 1)
    volume_distribution = np.zeros(bins)

    for _, row in df.iterrows():
        low_price = row['low']
        high_price = row['high']
        vol = row['volume']

        # low と high が同じ場合（ほぼヒゲが無いような足）は単一ビンに割り当て
        if high_price == low_price:
            idx = np.searchsorted(bin_edges, low_price, side='right') - 1
            if 0 <= idx < bins:
                volume_distribution[idx] += vol
        else:
            low_bin_idx = np.searchsorted(bin_edges, low_price, side='right') - 1
            high_bin_idx = np.searchsorted(bin_edges, high_price, side='right') - 1
            high_bin_idx = min(high_bin_idx, bins - 1)

            candle_range = high_price - low_price

            for bin_i in range(low_bin_idx, high_bin_idx + 1):
                bin_low = bin_edges[bin_i]
                bin_high = bin_edges[bin_i + 1]
                overlap_low = max(bin_low, low_price)
                overlap_high = min(bin_high, high_price)
                overlap = max(0, overlap_high - overlap_low)

                vol_frac = (overlap / candle_range) * vol
                volume_distribution[bin_i] += vol_frac

    return bin_edges, volume_distribution

def support_resistance_line(
    current_price: float,
    df: pd.DataFrame,
    bins: int = 50,
    width_pct: float = 0.05,
    fixed_side: str = "center"  # "support", "resistance", "center" のいずれかを指定
):
    """
    VolumeProfileを使った単純なサポート・レジスタンスライン抽出。
    指定した最低幅に足りなければ、fixed_sideで指定した側を基準に幅を調整する。

    Parameters
    ----------
    current_price : float
        現在価格
    df : pd.DataFrame
        'high','low','volume'カラムを含むデータフレーム
    bins : int, default=50
        ボリュームプロファイル計算用のビン数
    width_pct : float, default=0.05
        現在価格に対する最低幅の割合 (例: 0.05 -> 5%)
    fixed_side : str, default="center"
        幅調整時に固定する側。"support" または "resistance" を指定すると、その側を固定して反対側のみを調整、
        "center" の場合は従来通り中間点を中心に対称的に調整

    Returns
    -------
    support_line : float
    resistance_line : float
    """
    bin_edges, volume_distribution = calculate_volume_profile(df, bins=bins)
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 現在価格が属するビンのインデックス
    idx = np.searchsorted(bin_edges, current_price, side='right') - 1
    idx = max(0, min(idx, len(volume_distribution) - 1))

    volume_below = volume_distribution[:idx]
    volume_above = volume_distribution[idx+1:]

    if len(volume_below) == 0:
        support_line = bin_midpoints[idx]
    else:
        sup_idx = np.argmax(volume_below)
        support_line = bin_midpoints[sup_idx]

    if len(volume_above) == 0:
        resistance_line = bin_midpoints[idx]
    else:
        res_idx = np.argmax(volume_above) + (idx + 1)
        resistance_line = bin_midpoints[res_idx]

    # サポートとレジスタンスが逆転していたら修正
    if support_line > resistance_line:
        support_line, resistance_line = resistance_line, support_line

    # 指定幅の計算
    desired_diff = current_price * width_pct
    actual_diff = resistance_line - support_line

    if actual_diff < desired_diff:
        if fixed_side.lower() == "support":
            # サポートを固定し、レジスタンス側のみを調整
            resistance_line = support_line + desired_diff
            if resistance_line < current_price:
                resistance_line = current_price * 1.02
        elif fixed_side.lower() == "resistance":
            # レジスタンスを固定し、サポート側のみを調整
            support_line = resistance_line - desired_diff
            if support_line > current_price:
                support_line = current_price * 0.98
        else:
            # "center"の場合は中間点を中心に対称調整
            midpoint = (support_line + resistance_line) / 2
            half_diff = desired_diff / 2
            support_line = midpoint - half_diff
            resistance_line = midpoint + half_diff
            if support_line > current_price:
                support_line = current_price * 0.98
            if resistance_line < current_price:
                resistance_line = current_price * 1.02

    return support_line, resistance_line

def plot_volume_profile(bin_edges: np.ndarray, volume_distribution: np.ndarray):
    """
    ボリュームプロファイルを水平バーとしてプロットする。
    """
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.barh(bin_midpoints,
            volume_distribution,
            height=(bin_edges[1] - bin_edges[0]),
            color='skyblue',
            edgecolor='gray')

    ax.set_xlabel('Volume')
    ax.set_ylabel('Price')
    ax.set_title('Volume Profile')
    plt.gca().invert_yaxis()
    plt.show()

def main():
    # MongoDBからBTCの時系列データを取得
    db = MongoDataLoader()
    df = db.load_data_from_point_date(
        datetime(2025, 2, 20, 12, 0),
        nsteps=30,
        collection_name=MARKET_DATA_TECH,
        symbol='BTCUSDT',
        interval=1440
    )

    # 必要なカラムだけ抽出
    graph_df = df[['start_at', 'close', 'high', 'low', 'open', 'volume']]
    current_price = graph_df.iloc[-1]['close']

    # 幅パラメータと固定側を指定してサポート・レジスタンスを取得
    # fixed_side は "support", "resistance", または "center" を指定可能
    support_line, resistance_line = support_resistance_line(
        current_price=current_price,
        df=graph_df,
        bins=50,
        width_pct=0.05,      # 例: 現在価格の5%を最低幅として確保
        fixed_side="support" # 固定側を "support", "resistance", "center" の中から選択
    )

    print(f"現在価格: {current_price}")
    print(f"サポートライン: {support_line}")
    print(f"レジスタンスライン: {resistance_line}")

    # ボリュームプロファイルの可視化
    bin_edges, volume_distribution = calculate_volume_profile(graph_df, bins=50)
    plot_volume_profile(bin_edges, volume_distribution)

if __name__ == "__main__":
    main()

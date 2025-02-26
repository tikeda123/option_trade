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
    # 全体の最安値と最高値
    price_min = df['low'].min()
    price_max = df['high'].max()

    # ビンを等間隔に作成
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
            # ローソク足がまたぐビンの範囲を特定
            low_bin_idx = np.searchsorted(bin_edges, low_price, side='right') - 1
            high_bin_idx = np.searchsorted(bin_edges, high_price, side='right') - 1

            # high_price が最大値の場合、high_bin_idx が bins になる可能性があるため修正
            high_bin_idx = min(high_bin_idx, bins - 1)

            candle_range = high_price - low_price

            # low_bin_idx から high_bin_idx まで1つずつ処理
            for bin_i in range(low_bin_idx, high_bin_idx + 1):
                # bin_i の価格帯
                bin_low = bin_edges[bin_i]
                bin_high = bin_edges[bin_i + 1]

                # ローソク足 [low, high] とビン [bin_low, bin_high] の重なりを計算
                overlap_low = max(bin_low, low_price)
                overlap_high = min(bin_high, high_price)
                overlap = max(0, overlap_high - overlap_low)

                # ローソク足全体に対しての重み
                vol_frac = (overlap / candle_range) * vol
                volume_distribution[bin_i] += vol_frac

    return bin_edges, volume_distribution



def plot_volume_profile(bin_edges: np.ndarray, volume_distribution: np.ndarray):
    """
    ボリュームプロファイルを水平バーとしてプロットする。

    Parameters
    ----------
    bin_edges : np.ndarray
        価格帯ビンの境界値
    volume_distribution : np.ndarray
        各ビンに割り当てられた出来高
    """
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2  # ビンの中心価格
    fig, ax = plt.subplots(figsize=(7, 6))

    # 横軸を出来高、縦軸を価格にしてバーを水平に描画
    ax.barh(bin_midpoints, volume_distribution,
            height=(bin_edges[1] - bin_edges[0]),
            color='skyblue', edgecolor='gray')

    ax.set_xlabel('Volume')
    ax.set_ylabel('Price')
    ax.set_title('Volume Profile')
    plt.gca().invert_yaxis()  # 価格の高い方を上にする場合はコメントアウト
    plt.show()


def main():
    # MongoDBからBTCの時系列データを取得
    db = MongoDataLoader()
    df = db.load_data_from_datetime_period(datetime(2025, 2, 1),
                                           datetime(2025, 2, 14),
                                           coll_type=MARKET_DATA_TECH,
                                           symbol='BTCUSDT',
                                           interval=60)
    # 利用するカラムのみ抽出
    graph_df = df[['start_at', 'close', 'high', 'low', 'open', 'volume']]

    # ★ ボリュームプロファイルを計算・描画
    bin_edges, volume_distribution = calculate_volume_profile(graph_df, bins=50)
    plot_volume_profile(bin_edges, volume_distribution)


if __name__ == "__main__":
    main()

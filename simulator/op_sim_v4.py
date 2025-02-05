import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# 自己相関グラフ描画用のインポート
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# 季節分解
from statsmodels.tsa.seasonal import seasonal_decompose

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit  # ★ 追加
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---- 追加: data_preprocessing.py をインポート ----
from data_preprocessing import clean_option_data

# ユーザー環境に合わせたパスの設定
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *

def main():
    # MongoDBからデータを読み込み
    db = MongoDataLoader()
    df = db.load_data_from_datetime_period(
        datetime(2021, 1, 1),
        datetime(2024, 12, 31),
        coll_type=MARKET_DATA_TECH,
        symbol="BTCUSDT",
        interval=1440
    )

    # データの確認（必要に応じてコメントアウト）
    print(df.head())
    print(df["volatility"].head())

    # 'macdhist' カラムの欠損値がある場合は削除
    macdhist_series = df["volatility"].dropna()

    # 1. 自己相関プロット
    plt.figure(figsize=(10, 6))
    plot_acf(macdhist_series, lags=50)
    plt.title("Autocorrelation of macdhist")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.tight_layout()
    plt.show()

    # 2. 季節分解 (seasonal_decompose) による周期性の可視化
    #   - 日足の場合、週周期(7日)や月周期(30日)などを仮定。
    #   - period=7の場合の例:
    result = seasonal_decompose(macdhist_series, model='additive', period=7)
    fig = result.plot()
    fig.suptitle("Seasonal Decomposition of macdhist (period=7)")
    plt.tight_layout()
    plt.show()

    # 3. FFT(高速フーリエ変換)によるパワースペクトル解析
    fft_vals = np.fft.fft(macdhist_series.values)
    freqs = np.fft.fftfreq(len(fft_vals))
    power = np.abs(fft_vals) ** 2

    # 左右対称なので0〜ナイキスト周波数（配列半分）だけ可視化
    half = len(freqs) // 2
    freqs_pos = freqs[:half]
    power_pos = power[:half]

    plt.figure(figsize=(10, 5))
    plt.plot(freqs_pos, power_pos)
    plt.title("Power Spectrum (FFT) of macdhist")
    plt.xlabel("Frequency")
    plt.ylabel("Power")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # （参考）部分自己相関も併せて見たい場合
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plot_acf(macdhist_series, lags=50, ax=plt.gca())
    plt.title("ACF of macdhist")

    plt.subplot(1, 2, 2)
    plot_pacf(macdhist_series, lags=50, method='ywm', ax=plt.gca())
    plt.title("PACF of macdhist")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

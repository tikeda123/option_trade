import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# TensorFlowやScikit-learn関連のインポートは、本コード内では未使用のため除去していますが、
# 将来的に機械学習関連の処理を実装する予定があれば残しても構いません。
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import LSTM, Dense
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.metrics import mean_squared_error, mean_absolute_error

# 他ファイルからのインポート（パス調整は実環境に応じて行ってください）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from data_preprocessing import clean_option_data
from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import MARKET_DATA


def parse_symbol(symbol: str) -> Tuple[str, str, float, str]:
    """
    オプションのシンボル文字列を解析する関数。
    例: "BTC-28MAR25-100000-C" -> ("BTC", "28MAR25", 100000.0, "C")
    """
    splitted = symbol.split('-')
    ticker = splitted[0]
    expiry = splitted[1]
    strike = float(splitted[2])
    option_type = splitted[3]  # "C" or "P"
    return ticker, expiry, strike, option_type


def process_option_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    オプションデータをシンボルごとにグルーピングして、
    日付をインデックスにセットしたDataFrameを返します。

    Parameters
    ----------
    df : pd.DataFrame
        オプションデータを含むDataFrame

    Returns
    -------
    Dict[str, pd.DataFrame]
        キー: シンボル文字列, 値: シンボル別の時系列DataFrame
    """
    df['date'] = pd.to_datetime(df['date'])
    symbol_groups = {}
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        symbol_df.sort_values('date', inplace=True)
        symbol_df.set_index('date', inplace=True)
        symbol_groups[symbol] = symbol_df
    return symbol_groups


def plot_two_axes(
    df: pd.DataFrame,
    x_col: str,
    left_col: List[str],
    right_col: List[str],
    left_label: str,
    right_label: str,
    title: str,
    left_colors: List[str],
    right_colors: List[str],
    alpha: float = 0.7
) -> None:
    """
    一つの図に2つのY軸を持つグラフをプロットする汎用関数。

    Parameters
    ----------
    df : pd.DataFrame
        プロット対象の時系列DataFrame
    x_col : str
        X軸に使用する列名（通常はインデックスを使う場合はNoneでも良い）
    left_col : List[str]
        左Y軸にプロットする列のリスト
    right_col : List[str]
        右Y軸にプロットする列のリスト
    left_label : str
        左Y軸のラベル
    right_label : str
        右Y軸のラベル
    title : str
        グラフのタイトル
    left_colors : List[str]
        左Y軸の各系列の色のリスト
    right_colors : List[str]
        右Y軸の各系列の色のリスト
    alpha : float, optional
        プロットの透明度, デフォルトは0.7
    """
    plt.style.use('seaborn-v0_8')
    fig, ax_left = plt.subplots(figsize=(12, 6))

    # x軸は基本的にインデックスを利用するケースが多いため、
    # x_colを使用しない場合は df.index を参照するようにしています。
    x_values = df.index if x_col is None else df[x_col]

    # 左軸プロット
    ax_left.set_xlabel('Date')
    ax_left.set_ylabel(left_label, color=left_colors[0])
    lines_left = []
    for col, col_color in zip(left_col, left_colors):
        line = ax_left.plot(x_values, df[col], label=col, color=col_color, alpha=alpha)
        lines_left.extend(line)
    ax_left.tick_params(axis='y', labelcolor=left_colors[0])

    # 右軸プロット
    ax_right = ax_left.twinx()
    ax_right.set_ylabel(right_label, color=right_colors[0])
    lines_right = []
    for col, col_color in zip(right_col, right_colors):
        line = ax_right.plot(x_values, df[col], label=col, color=col_color, alpha=alpha)
        lines_right.extend(line)
    ax_right.tick_params(axis='y', labelcolor=right_colors[0])

    # タイトルや凡例など
    plt.title(title)
    lines = lines_left + lines_right
    labels = [line.get_label() for line in lines]
    ax_left.legend(lines, labels, loc='upper left')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_delta_analysis(symbol_data: pd.DataFrame, target_symbol: str) -> None:
    """
    Deltaの分析と可視化。
    Ask1Price・BTC Price（close）を左軸、Deltaを右軸にプロットする。
    """
    plot_two_axes(
        df=symbol_data,
        x_col=None,
        left_col=['ask1Price', 'close'],
        right_col=['delta'],
        left_label='Price',
        right_label='Delta',
        title=f'Ask1 Price, BTC Price, and Delta for {target_symbol}',
        left_colors=['red', 'green'],
        right_colors=['blue']
    )


def plot_gamma_analysis(symbol_data: pd.DataFrame, target_symbol: str) -> None:
    """
    Gammaの分析と可視化。
    Deltaを左軸、Gammaを右軸にプロットする。
    """
    plot_two_axes(
        df=symbol_data,
        x_col=None,
        left_col=['delta'],
        right_col=['gamma'],
        left_label='Delta',
        right_label='Gamma',
        title=f'Delta & Gamma (Separate Axes) for {target_symbol}',
        left_colors=['blue'],
        right_colors=['orange']
    )


def plot_vega_analysis(symbol_data: pd.DataFrame, target_symbol: str) -> None:
    """
    Vegaの分析と可視化。
    (1) Ask1Price vs. Ask1Iv を左右軸に
    (2) Ask1Iv vs. Vega を左右軸に
    """
    # (1)
    plot_two_axes(
        df=symbol_data,
        x_col=None,
        left_col=['ask1Price'],
        right_col=['ask1Iv'],
        left_label='Ask1 Price',
        right_label='Ask1 Iv',
        title=f'Ask1Price vs. Ask1Iv for {target_symbol}',
        left_colors=['red'],
        right_colors=['blue']
    )

    # (2)
    plot_two_axes(
        df=symbol_data,
        x_col=None,
        left_col=['ask1Iv'],
        right_col=['vega'],
        left_label='Ask1 Iv',
        right_label='Vega',
        title=f'Ask1Iv vs. Vega for {target_symbol}',
        left_colors=['green'],
        right_colors=['orange']
    )


def plot_theta_analysis(symbol_data: pd.DataFrame, target_symbol: str) -> None:
    """
    Thetaの分析と可視化。
    Ask1Price・BTC Price（close）を左軸、Thetaを右軸にプロットする。
    """
    plot_two_axes(
        df=symbol_data,
        x_col=None,
        left_col=['ask1Price', 'close'],
        right_col=['theta'],
        left_label='Price',
        right_label='Theta',
        title=f'Ask1 Price, BTC Price, and Theta for {target_symbol}',
        left_colors=['red', 'green'],
        right_colors=['blue']
    )


def main() -> None:
    """
    メイン処理。MongoDBからデータを取得し、クレンジング・可視化を行う。
    """
    db = MongoDataLoader()

    start_date = datetime(2024, 12, 18)
    end_date = datetime(2025, 1, 12)

    # ---- オプションデータの読み込み・クレンジング ----
    df_op = db.load_data_from_datetime_period_option(start_date=start_date, end_date=end_date)
    df_op = clean_option_data(
        df_op,
        group_col='symbol',
        columns_to_clean=['ask1Price', 'ask1Iv', 'bid1Price', 'bid1Iv'],
        outlier_factor=1.5,
        dropna_after=True
    )

    # ---- BTC価格データの読み込み ----
    df_btc = db.load_data_from_datetime_period(
        start_date=start_date,
        end_date=end_date,
        coll_type=MARKET_DATA,
        symbol="BTCUSDT",
        interval=60
    )
    df_op["close"] = df_btc["close"]

    # ---- シンボルごとの時系列データに分割 ----
    symbol_groups = process_option_data(df_op)

    # ---- 例: 特定シンボルを指定 ----
    target_symbol = "BTC-28MAR25-100000-C"
    if target_symbol not in symbol_groups:
        print(f"Symbol {target_symbol} not found in data.")
        return

    symbol_data = symbol_groups[target_symbol].copy()
    print(symbol_data)

    # ---- カラムを数値に変換（object -> float） ----
    numeric_cols = [
        'ask1Price', 'ask1Iv', 'bid1Price', 'bid1Iv',
        'markIv', 'delta', 'gamma', 'vega', 'theta', 'close'
    ]
    for col in numeric_cols:
        if col in symbol_data.columns:
            symbol_data[col] = pd.to_numeric(symbol_data[col], errors='coerce')

    # ---- 日時ソート & 欠損日の forward fill ----
    symbol_data.sort_index(inplace=True)
    symbol_data = symbol_data.resample('D').ffill()

    # ---- 基本統計量表示 ----
    print(f"Symbol: {target_symbol}")
    print(f"Data period: {symbol_data.index.min()} to {symbol_data.index.max()}")
    print(f"Number of data points: {len(symbol_data)}\n")
    for col in numeric_cols:
        if col in symbol_data.columns:
            print(f"{col}: {symbol_data[col].describe()}\n")

    # ---- 各ギリシャ指標の分析と可視化 ----
    plot_delta_analysis(symbol_data, target_symbol)
    plot_gamma_analysis(symbol_data, target_symbol)
    plot_vega_analysis(symbol_data, target_symbol)
    plot_theta_analysis(symbol_data, target_symbol)


if __name__ == "__main__":
    main()

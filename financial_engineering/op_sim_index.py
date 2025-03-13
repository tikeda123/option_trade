import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# ---- MongoDataLoader など、環境に依存する部分は適宜差し替えてください ----
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *


def generate_signal_generic(
    df: pd.DataFrame,
    indicator_col: str,
    method: str = 'change',
    lower_bound: float = 30,
    upper_bound: float = 70,
    shift_period: int = 1
) -> pd.Series:
    """上記と同内容: 汎用シグナル生成関数"""
    series = df[indicator_col]

    if method == 'change':
        shifted = series.shift(shift_period)
        signal = np.where(series > shifted, 1, -1)

    elif method == 'threshold':
        # 下限閾値より低ければロング, 上限閾値より高ければショート, それ以外は0
        signal = np.where(
            series < lower_bound,
            1,
            np.where(series > upper_bound, -1, 0)
        )
    else:
        raise ValueError("Unsupported method. Choose 'change' or 'threshold'.")

    return pd.Series(signal, index=df.index)


def compare_strategies(df: pd.DataFrame, signal_funcs: dict, plot: bool = True) -> pd.DataFrame:
    """
    複数のストラテジーを比較し、結果をDataFrameにまとめる。
    signal_funcs には「ストラテジー名」→「シグナル生成関数」を渡す。
    """
    df_result = df.copy()

    # 1日先の対数リターン
    df_result['log_return_1d'] = np.log(df_result['close'].shift(-1) / df_result['close'])

    # 各ストラテジーのシグナル・リターン・累積リターンを計算
    for strategy_name, func in signal_funcs.items():
        signal_col = f'signal_{strategy_name}'
        return_col = f'return_{strategy_name}'
        cumret_col = f'cumulative_return_{strategy_name}'

        # シグナル生成
        df_result[signal_col] = func(df_result)  # func は df を引数に取り、Seriesを返す関数

        # リターン計算 (シグナル × 翌日対数リターン)
        df_result[return_col] = df_result[signal_col] * df_result['log_return_1d']
        df_result[cumret_col] = df_result[return_col].cumsum()

    # 不要なNaN(最終行など)を削除
    df_result.dropna(subset=['log_return_1d'], inplace=True)

    # 日付をdatetimeに
    df_result['start_at'] = pd.to_datetime(df_result['start_at'])

    # プロット
    if plot:
        plt.figure(figsize=(12, 7))
        for strategy_name in signal_funcs.keys():
            plt.plot(
                df_result['start_at'],
                df_result[f'cumulative_return_{strategy_name}'],
                label=strategy_name
            )
        plt.xlabel('Date')
        plt.ylabel('Cumulative Log Return')
        plt.title('Strategy Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()

    return df_result


def main():
    # --- データ取得 ---
    db = MongoDataLoader()
    df = db.load_data_from_datetime_period(
        datetime(2024, 1, 1),
        datetime(2025, 3, 1),
        coll_type=MARKET_DATA_TECH,
        symbol='BTCUSDT',
        interval=1440
    )

    # 'start_at', 'close', 'macdhist', 'rsi', 'roc', 'sma', 'ema' などがある想定
    # （無い場合はコメントアウト）
    df = df[['start_at', 'close', 'macdhist', 'rsi', 'roc', 'sma','ema','mfi']].copy()

    # --- ストラテジー定義 ---
    # 例1: MACDヒストグラムの「前日より増加/減少」でシグナルを出す
    def strategy_macdhist_increase(df):
        return generate_signal_generic(
            df,
            indicator_col='macdhist',
            method='change',
        )

    def strategy_ema_increase(df):
        return generate_signal_generic(
            df,
            indicator_col='ema',
            method='change'   # 前日比
        )

    def strategy_sma_increase(df):
        return generate_signal_generic(
            df,
            indicator_col='sma',
            method='change'   # 前日比
        )

    def strategy_mfi_increase(df):
        return generate_signal_generic(
            df,
            indicator_col='mfi',
            method='change'   # 前日比
        )


    # 例2: RSI の閾値。RSI < 30 なら買い、RSI > 70 なら売り、それ以外0
    def strategy_rsi_threshold(df):
        return generate_signal_generic(
            df,
            indicator_col='rsi',
            method='threshold',
            lower_bound=45,
            upper_bound=55
        )

    # 例3: ROC (Rate of Change) の閾値を0にする(0より上→ロング、0より下→ショートとみなす)
    def strategy_roc_zero(df):
        return generate_signal_generic(
            df,
            indicator_col='roc',
            method='threshold',
            lower_bound=0,   # 0より下ならショート、
            upper_bound=0    # 0より上ならロングと見なす => ただし上記関数では上限下限の扱いが逆のため
                             # メソッドを変更する or カスタムロジックで書くほうが自然かもしれません
        )

    # 例4: 常にロング
    def strategy_always_long(df):
        return np.ones(len(df))

    # 比較したいストラテジーをまとめる
    signal_funcs = {
        "macdhist_change": strategy_macdhist_increase,
        "ema_change": strategy_ema_increase,
        "sma_change": strategy_sma_increase,
        "always_long": strategy_always_long
    }

    # --- ストラテジー比較 ---
    df_strategies = compare_strategies(
        df,
        signal_funcs=signal_funcs,
        plot=True
    )

    # 各ストラテジーの最終累積リターンを表示
    for name in signal_funcs.keys():
        final_ret = df_strategies[f'cumulative_return_{name}'].iloc[-1]
        print(f"{name} Strategy Cumulative Log Return: {final_ret:.6f}")

if __name__ == "__main__":
    main()

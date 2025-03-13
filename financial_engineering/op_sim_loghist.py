import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# ユーザー環境に合わせたパス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *

def main():
    # MongoDBからBTCの時系列データを取得
    db = MongoDataLoader()
    df = db.load_data_from_datetime_period(datetime(2021, 1, 1),
                                           datetime(2025, 1, 1),
                                           coll_type=MARKET_DATA_TECH,
                                           symbol='BTCUSDT',
                                           interval=1440)

    # closeカラムを使って前日比の対数収益率を計算
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # NaNを削除
    df = df.dropna()

    # 対数収益率のヒストグラムを描画
    plt.figure(figsize=(10, 6))
    plt.hist(df['log_returns'], bins=100, edgecolor='black')
    plt.xlabel('Log Returns')
    plt.ylabel('Frequency')
    plt.title('Histogram of BTC Log Returns (2023-2025)')
    plt.grid(True)
    plt.show()

    # 「今日買って明日売る」戦略の収益計算
    # 当日の価格で買い、翌日の価格で売るので、対数収益率は以下のように計算
    df['strategy_return'] = np.log(df['close'].shift(-1) / df['close'])

    # 最後の行は翌日のデータが存在しないためNaNとなるので除外
    df_strategy = df.dropna(subset=['strategy_return']).copy()

    # 累積収益を計算（各取引の対数収益率を累積）
    df_strategy['cumulative_return'] = df_strategy['strategy_return'].cumsum()

    # start_atカラムを日付型に変換（文字列の場合）
    df_strategy['start_at'] = pd.to_datetime(df_strategy['start_at'])

    # 時系列で累積収益をプロット
    plt.figure(figsize=(10, 6))
    plt.plot(df_strategy['start_at'], df_strategy['cumulative_return'], label='Daily Trading Strategy')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Log Return')
    plt.title('Cumulative Profit of "Buy Today, Sell Tomorrow" Strategy')
    plt.grid(True)

    # 買い持ち戦略：最初の日に買って最後の日に売る収益率を計算
    overall_return = np.log(df['close'].iloc[-1] / df['close'].iloc[0])
    # プロット上に水平線として表示（全期間の買い持ち収益）
    plt.axhline(y=overall_return, color='red', linestyle='--', label='Buy-and-Hold Return')

    plt.legend()
    plt.show()

    # 両者の比較結果を出力
    final_strategy_return = df_strategy['cumulative_return'].iloc[-1]
    print("買い持ち戦略 (最初に買って最後に売る) の対数収益率: {:.6f}".format(overall_return))
    print("「今日買って明日売る」戦略の累積対数収益率: {:.6f}".format(final_strategy_return))

if __name__ == "__main__":
    main()



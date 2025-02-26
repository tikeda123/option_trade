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
    #========== データ取得 ==========
    db = MongoDataLoader()
    df = db.load_data_from_datetime_period(
        datetime(2023, 1, 1),
        datetime(2025, 1, 1),
        coll_type=MARKET_DATA_TECH,
        symbol='BTCUSDT',
        interval=1440
    )

    # 必要なカラムに絞る
    df = df[['start_at', 'close', 'volume', 'macdhist', 'rsi', 'volatility']].copy()

    # 日時をインデックスに設定（あれば便利）
    df.set_index('start_at', inplace=True)
    df.sort_index(inplace=True)

    #========== リターンを計算する（例：対数リターン） ==========
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(inplace=True)  # 計算できない最初の行を削除

    #========== 戦略A: ボラティリティを考慮しない ==========
    # MACDヒストグラムが > 0 ならロング(+1), < 0 ならショート(-1), = 0 なら0
    def macd_signal(x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    df['position_A'] = df['macdhist'].apply(macd_signal)

    # Strategy_Aのリターン: positionを1足ずらして (エントリーは次の足から有効と仮定)、対数リターンを掛ける
    df['strategy_return_A'] = df['position_A'].shift(1) * df['log_return']

    #========== 戦略B: ボラティリティを考慮する ==========
    # 例: 同じMACD方向の判定だが、positionサイズ = k / volatility（上限・下限を設ける）
    # ボラティリティが0に近い場合や極端に小さい/大きい場合の対策にクリッピングを行う

    k = 1.0  # スケーリング用定数（調整してみてください）
    min_pos, max_pos = 0.1, 1.5  # ポジションサイズの下限・上限（例）

    def calc_position_size(vol):
        if vol == 0:
            return max_pos  # 万が一vol=0のときは最大
        raw_size = k / vol
        # クリッピング
        return np.clip(raw_size, min_pos, max_pos)

    df['vol_adjust'] = df['volatility'].apply(calc_position_size)

    # 戦略Bでは、(MACDの方向) × (ボラティリティに応じたポジションサイズ)
    df['position_B'] = df['macdhist'].apply(macd_signal) * df['vol_adjust']

    df['strategy_return_B'] = df['position_B'].shift(1) * df['log_return']

    #========== パフォーマンス集計 ==========
    df['cum_return_A'] = df['strategy_return_A'].cumsum().apply(np.exp)  # 累積対数リターン→指数変換で累積リターン
    df['cum_return_B'] = df['strategy_return_B'].cumsum().apply(np.exp)

    # シャープレシオなどを計算（例: 1期間を1日と仮定するなら年換算に要調整）
    # 今回は1時間足(= 1期間 = 1h)なので、年間にすると 24 * 365 ~= 8760 期間
    # シャープレシオ = 平均リターン / 標準偏差 * sqrt(年換算係数)
    annual_factor = np.sqrt(8760)  # 1時間足を年率換算するための例
    sharpe_A = df['strategy_return_A'].mean() / df['strategy_return_A'].std() * annual_factor
    sharpe_B = df['strategy_return_B'].mean() / df['strategy_return_B'].std() * annual_factor

    print("=== 戦略A（ボラ無調整）===")
    print(f"累積リターン: {df['cum_return_A'].iloc[-1]:.2f}")
    print(f"シャープレシオ: {sharpe_A:.2f}")

    print("=== 戦略B（ボラ調整）===")
    print(f"累積リターン: {df['cum_return_B'].iloc[-1]:.2f}")
    print(f"シャープレシオ: {sharpe_B:.2f}")

    #========== 結果を可視化 ==========
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['cum_return_A'], label='Strategy A (No Vol Adjust)')
    plt.plot(df.index, df['cum_return_B'], label='Strategy B (Vol Adjust)')
    plt.title('Cumulative Returns Comparison')
    plt.xlabel('DateTime')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


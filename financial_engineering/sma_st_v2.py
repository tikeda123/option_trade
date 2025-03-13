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
    # ==== 1. データ取得 ====
    db = MongoDataLoader()
    df = db.load_data_from_datetime_period(
        datetime(2024, 10, 1),
        datetime(2025, 1, 1),
        coll_type=MARKET_DATA_TECH,
        symbol='BTCUSDT',
        interval=1440
    )

    # 使うカラムのみ抽出
    backtest_df = df[['start_at', 'close']].copy()
    backtest_df.sort_values('start_at', inplace=True)
    backtest_df.reset_index(drop=True, inplace=True)

    # ==== 2. 移動平均・標準偏差・Zスコア を計算 ====
    ma_period = 20
    backtest_df['SMA'] = backtest_df['close'].rolling(window=ma_period).mean()
    backtest_df['STD'] = backtest_df['close'].rolling(window=ma_period).std()
    # Zスコア = (現在価格 - SMA) / STD
    backtest_df['z_score'] = (backtest_df['close'] - backtest_df['SMA']) / backtest_df['STD']

    # 計算直後は NaN の行（最初の ma_period-1 行など）を除外
    backtest_df.dropna(inplace=True)
    backtest_df.reset_index(drop=True, inplace=True)

    # ==== 3. ポジション管理用のカラム作成 ====
    backtest_df['position'] = 0
    backtest_df['days_in_position'] = 0

    # 追加：トレード記録を保存するための変数
    trades = []
    # 格納する内容のイメージ:
    # {
    #   "entry_date": ...,
    #   "exit_date": ...,
    #   "side": ...,
    #   "entry_price": ...,
    #   "exit_price": ...,
    #   "pnl_ratio": ...
    # }

    # ==== 4. ループで日々のエントリー／エグジットを管理 ====
    current_position = 0  # -1=ショート, 0=ノーポジ, +1=ロング
    days_in_pos = 0       # ポジション保有日数
    entry_price = None    # エントリー価格
    entry_date = None     # エントリーした日時

    # Zスコアに基づく閾値
    entry_threshold_z = 2.0   # |Zスコア| > 2.0でエントリー
    exit_threshold_z  = 0.5   # |Zスコア| < 0.5でエグジット
    max_hold_days     = 50    # 保有日数が10日(コメント)となっているが、実際は50に設定

    for i in range(len(backtest_df)):
        close_ = backtest_df.loc[i, 'close']
        z_     = backtest_df.loc[i, 'z_score']
        date_  = backtest_df.loc[i, 'start_at']

        # --- (A) エグジット条件を先にチェック ---
        if current_position != 0:
            days_in_pos += 1  # 連続保有日数を更新

            # ① Zスコアが ±0.5以内に戻ったらクローズ
            # ② あるいは max_hold_days を超えたらクローズ
            if (abs(z_) < exit_threshold_z) or (days_in_pos >= max_hold_days):
                exit_price = close_
                exit_date = date_
                side = current_position  # ロング(+1) or ショート(-1)

                # 損益率 pnl_ratio 計算
                if side == +1:
                    # ロングの場合: (exit_price / entry_price) - 1
                    pnl_ratio = (exit_price / entry_price) - 1.0
                else:
                    # ショートの場合: 1 - (exit_price / entry_price)
                    pnl_ratio = 1.0 - (exit_price / entry_price)

                # trades リストに保存
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "side": side,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_ratio": pnl_ratio
                })

                # ポジションをクローズ
                current_position = 0
                days_in_pos = 0
                entry_price = None
                entry_date = None

        # --- (B) ノーポジ時のみエントリー判定 ---
        if current_position == 0:
            # Zスコアが +2 超 => ショートエントリー
            if z_ > entry_threshold_z:
                current_position = -1
                days_in_pos = 1
                entry_price = close_
                entry_date = date_
            # Zスコアが -2 以下 => ロングエントリー
            elif z_ < -entry_threshold_z:
                current_position = +1
                days_in_pos = 1
                entry_price = close_
                entry_date = date_

        # (C) 当日の position 情報を DataFrame に書き込む
        backtest_df.loc[i, 'position'] = current_position
        backtest_df.loc[i, 'days_in_position'] = days_in_pos

    # ---- もし最終日までポジションが残っていた場合、強制クローズしたい場合は以下を参考に ----
    if current_position != 0 and entry_price is not None:
        close_ = backtest_df.loc[len(backtest_df)-1, 'close']
        date_  = backtest_df.loc[len(backtest_df)-1, 'start_at']
        side   = current_position

        if side == +1:
            pnl_ratio = (close_ / entry_price) - 1.0
        else:
            pnl_ratio = 1.0 - (close_ / entry_price)

        trades.append({
            "entry_date": entry_date,
            "exit_date": date_,
            "side": side,
            "entry_price": entry_price,
            "exit_price": close_,
            "pnl_ratio": pnl_ratio
        })

        current_position = 0
        days_in_pos = 0
        entry_price = None
        entry_date = None

    # ==== 5. 日次リターンの計算 ====
    backtest_df['daily_return'] = backtest_df['close'].pct_change().fillna(0)
    backtest_df['strategy_return'] = backtest_df['position'] * backtest_df['daily_return']
    backtest_df['strategy_cumprod'] = (1 + backtest_df['strategy_return']).cumprod() - 1

    # ==== 6. トレード結果の集計、勝率の算出 ====
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df['is_win'] = trades_df['pnl_ratio'] > 0
        winning_trades = trades_df['is_win'].sum()
        total_trades = len(trades_df)
        win_rate = winning_trades / total_trades * 100
        avg_pnl = trades_df['pnl_ratio'].mean() * 100  # 平均損益率 (%)
        print("=== Trade Summary (Z-score based) ===")
        print(f"Total Trades : {total_trades}")
        print(f"Winning Trades : {winning_trades}")
        print(f"Win Rate : {win_rate:.2f}%")
        print(f"Average PnL : {avg_pnl:.2f}%")
        print("=====================================")
    else:
        print("No trades were generated in this period.")

    # ==== 7. 結果を表示・可視化 ====
    print(backtest_df.head(15))  # 先頭15行
    print(backtest_df.tail(5))   # 末尾5行

    # (A) Zスコアとポジションをプロット
    plt.figure(figsize=(12,5))
    plt.title("Z-score and Position")
    plt.plot(backtest_df['start_at'], backtest_df['z_score'], label='Z-score')
    plt.plot(backtest_df['start_at'], backtest_df['position'], label='Position', color='orange')
    plt.axhline(y= 2.0, color='red', linestyle='--', label='Entry Threshold +2')
    plt.axhline(y=-2.0, color='green', linestyle='--', label='Entry Threshold -2')
    plt.axhline(y= 0, color='gray', linestyle=':')
    plt.legend()
    plt.show()

    # (B) 戦略の累積リターン
    plt.figure(figsize=(12,5))
    plt.title("Strategy Cumulative Return (Z-score based)")
    plt.plot(backtest_df['start_at'], backtest_df['strategy_cumprod'], label='Strategy Cum. Return')
    plt.axhline(0, color='gray', linestyle='--')
    plt.legend()
    plt.show()

    # ==== 8. 初期資金1000ドルの場合の最終資産を計算 ====
    final_cum_return = backtest_df['strategy_cumprod'].iloc[-1]  # 最終行の累積リターン
    initial_capital = 1000
    final_capital = initial_capital * (1 + final_cum_return)
    print("=== Final Capital Calculation ===")
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Final Capital:   ${final_capital:.2f}")
    print("=================================")

if __name__ == "__main__":
    main()

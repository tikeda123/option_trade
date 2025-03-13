import os
import sys
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime
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
        datetime(2024, 1, 1),
        datetime(2024,8, 1),
        coll_type=MARKET_DATA_TECH,
        symbol='BTCUSDT',
        interval=1440
    )

    # 必要カラムのみ抽出
    backtest_df = df[['start_at', 'close', 'kalman_macdhist']].copy()
    backtest_df.sort_values('start_at', inplace=True)
    backtest_df.reset_index(drop=True, inplace=True)

    # NaN除去
    backtest_df.dropna(subset=['close','kalman_macdhist'], inplace=True)
    backtest_df.reset_index(drop=True, inplace=True)

    # ==== 2. カラム初期化 ====
    # position: +1=ロング, -1=ショート, 0=ノーポジ
    backtest_df['position'] = 0
    backtest_df['days_in_position'] = 0

    # トレード記録用
    trades = []

    # ==== 3. 変数の初期化 ====
    current_position = 0
    entry_price = None
    entry_date = None
    days_in_pos = 0

    # ==== 4. 売買ロジック ====
    # ルックアヘッドバイアスを避けるため、i=1からスタート
    # → day i のトレード判断には前日のmacdhistを使う
    for i in range(1, len(backtest_df)):
        # 当日（i 日目）の価格・日時
        close_today = backtest_df.loc[i, 'close']
        date_today  = backtest_df.loc[i, 'start_at']

        # シグナル判定に使うのは「前日 (i-1) の macdhist」
        macd_hist_yesterday = backtest_df.loc[i-1, 'kalman_macdhist']

        # ------------ (A) エグジット判定 ------------
        if current_position == 1:  # ロング保有中
            # 前日のkalman_macdhistが負になったら (つまりシグナル変化) → 当日 close で手仕舞い＆ショートへドテン
            if macd_hist_yesterday < 0:
                exit_price = close_today
                side = current_position
                pnl_ratio = (exit_price / entry_price) - 1.0

                trades.append({
                    "entry_date": entry_date,
                    "exit_date": date_today,
                    "side": side,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_ratio": pnl_ratio
                })

                # ショートへドテン
                current_position = -1
                entry_price = close_today
                entry_date = date_today
                days_in_pos = 1

        elif current_position == -1:  # ショート保有中
            # 前日のkalman_macdhistが正になったら → 当日 close で手仕舞い＆ロングへドテン
            if macd_hist_yesterday > 0:
                exit_price = close_today
                side = current_position
                pnl_ratio = 1.0 - (exit_price / entry_price)

                trades.append({
                    "entry_date": entry_date,
                    "exit_date": date_today,
                    "side": side,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_ratio": pnl_ratio
                })

                # ロングへドテン
                current_position = 1
                entry_price = close_today
                entry_date = date_today
                days_in_pos = 1

        # ------------ (B) ポジションがない場合のエントリー ------------
        if current_position == 0:
            if macd_hist_yesterday > 0:
                # ロングエントリー
                current_position = 1
                entry_price = close_today
                entry_date = date_today
                days_in_pos = 1
            elif macd_hist_yesterday < 0:
                # ショートエントリー
                current_position = -1
                entry_price = close_today
                entry_date = date_today
                days_in_pos = 1
            else:
                # kalman_macdhistがゼロ近辺なら様子見
                pass
        else:
            # すでにポジションを持っているので日数インクリメント
            days_in_pos += 1

        # ------------ (C) DataFrame に書き込み ------------
        backtest_df.loc[i, 'position'] = current_position
        backtest_df.loc[i, 'days_in_position'] = days_in_pos

    # ------------ (D) 最終日のポジション強制クローズ（任意） ------------
    # 最後にまだポジションがあれば、最終行のcloseでクローズする
    if current_position != 0 and entry_price is not None:
        i_last = len(backtest_df) - 1
        close_last = backtest_df.loc[i_last, 'close']
        date_last = backtest_df.loc[i_last, 'start_at']
        side = current_position

        if side == 1:
            pnl_ratio = (close_last / entry_price) - 1.0
        else:
            pnl_ratio = 1.0 - (close_last / entry_price)

        trades.append({
            "entry_date": entry_date,
            "exit_date": date_last,
            "side": side,
            "entry_price": entry_price,
            "exit_price": close_last,
            "pnl_ratio": pnl_ratio
        })

        current_position = 0
        entry_price = None
        entry_date = None
        days_in_pos = 0

    # ==== 5. 日次リターン計算 (positionは「当日 close 時点のポジション」) ====
    # 日次リターン: 行 i の close / 行 i-1 の close - 1
    backtest_df['daily_return'] = backtest_df['close'].pct_change().fillna(0)

    # 実際には「前日のポジション」が「今日のリターン」を獲得するイメージのためshift(1)
    backtest_df['position_shifted'] = backtest_df['position'].shift(1).fillna(0)

    # 戦略リターン
    backtest_df['strategy_return'] = backtest_df['position_shifted'] * backtest_df['daily_return']
    backtest_df['strategy_cumprod'] = (1 + backtest_df['strategy_return']).cumprod() - 1

    # ==== 6. トレード結果の集計 ====
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df['is_win'] = trades_df['pnl_ratio'] > 0
        winning_trades = trades_df['is_win'].sum()
        total_trades = len(trades_df)
        win_rate = winning_trades / total_trades * 100
        avg_pnl = trades_df['pnl_ratio'].mean() * 100  # 平均損益率 (%)

        print("=== Trade Summary (MACD Hist based) ===")
        print(f"Total Trades : {total_trades}")
        print(f"Winning Trades : {winning_trades}")
        print(f"Win Rate : {win_rate:.2f}%")
        print(f"Average PnL : {avg_pnl:.2f}%")
        print("=====================================")
    else:
        print("No trades were generated in this period.")

    # ==== 7. 結果確認 ====
    print(backtest_df.head(15))
    print(backtest_df.tail(5))

    # (A) MACDヒストグラム & ポジション
    plt.figure(figsize=(12,5))
    plt.title("MACD Histogram and Position")
    plt.plot(backtest_df['start_at'], backtest_df['kalman_macdhist'], label='MACD Hist', color='blue')
    plt.plot(backtest_df['start_at'], backtest_df['position'], label='Position (end of day)', color='orange')
    plt.axhline(0, color='gray', linestyle='--')
    plt.legend()
    plt.show()

    # (B) 累積リターン
    plt.figure(figsize=(12,5))
    plt.title("Strategy Cumulative Return (MACD Hist based)")
    plt.plot(backtest_df['start_at'], backtest_df['strategy_cumprod'], label='Strategy Cum. Return', color='green')
    plt.axhline(0, color='gray', linestyle='--')
    plt.legend()
    plt.show()

    # ==== 8. 初期資金1000ドルの場合の最終資産 ====
    final_cum_return = backtest_df['strategy_cumprod'].iloc[-1]  # 最終行の累積リターン
    initial_capital = 1000
    final_capital = initial_capital * (1 + final_cum_return)
    print("=== Final Capital Calculation ===")
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Final Capital:   ${final_capital:.2f}")
    print("=================================")

if __name__ == "__main__":
    main()


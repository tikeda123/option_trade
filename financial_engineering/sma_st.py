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
        datetime(2024, 1, 1),
        datetime(2025, 1, 1),
        coll_type=MARKET_DATA_TECH,
        symbol='BTCUSDT',
        interval=1440
    )

    # 使うカラムのみ抽出
    backtest_df = df[['start_at', 'close']].copy()
    backtest_df.sort_values('start_at', inplace=True)
    backtest_df.reset_index(drop=True, inplace=True)

    # ==== 2. 移動平均や標準偏差、乖離率などを計算 ====
    ma_period = 20
    backtest_df['SMA'] = backtest_df['close'].rolling(window=ma_period).mean()
    backtest_df['STD'] = backtest_df['close'].rolling(window=ma_period).std()
    backtest_df['deviation_rate'] = (backtest_df['close'] - backtest_df['SMA']) / backtest_df['SMA'] * 100
    backtest_df.dropna(inplace=True)  # 移動平均が計算できない最初の19行などを除外
    backtest_df.reset_index(drop=True, inplace=True)

    # ==== 3. ポジション管理用のカラム作成 ====
    backtest_df['position'] = 0
    backtest_df['days_in_position'] = 0

    # 追加：トレード記録を保存するための変数
    trades = []  # リスト形式で「各トレードの情報」を保存
    # 例: trades.append({
    #     "entry_date": ...,
    #     "exit_date": ...,
    #     "side": ...,
    #     "entry_price": ...,
    #     "exit_price": ...,
    #     "pnl_ratio": ...  # (exit_price / entry_price - 1) あるいはその符号反転
    # })

    # ==== 4. ループで日々のエントリー／エグジットを厳密管理 ====
    current_position = 0  # -1=ショート, 0=ノーポジ, +1=ロング
    days_in_pos = 0       # ポジション保有日数
    entry_price = None    # エントリーしたときの価格
    entry_date = None     # エントリー日

    entry_threshold = 5.0   # 乖離率が +5%超でショート, -5%未満でロング
    exit_threshold = 1.0    # 絶対乖離率が 1%未満
    max_hold_days = 20      # 保有期限10日

    for i in range(len(backtest_df)):
        close_ = backtest_df.loc[i, 'close']
        sma_   = backtest_df.loc[i, 'SMA']
        dev_   = backtest_df.loc[i, 'deviation_rate']
        date_  = backtest_df.loc[i, 'start_at']  # 日付

        # --- (A) エグジット条件を先にチェック ---
        exited = False  # 当日エグジットが発生したかどうか
        if current_position != 0:
            days_in_pos += 1  # 保有継続なら日数カウントアップ

            # 乖離率が ±1%以内に戻ったらクローズ or 最大保有日数を超えたらクローズ
            if (abs(dev_) < exit_threshold) or (days_in_pos >= max_hold_days):
                # トレード終了処理
                exit_price = close_
                exit_date = date_
                side = current_position  # ロングなら+1, ショートなら-1

                # 実現損益 (pnl_ratio) 計算
                # ロングなら (exit_price / entry_price - 1)
                # ショートなら (1 - exit_price / entry_price)
                if side == +1:
                    pnl_ratio = (exit_price / entry_price) - 1.0
                else:
                    # ショートポジションは価格が下落すれば利益(+)
                    pnl_ratio = 1.0 - (exit_price / entry_price)

                # トレード情報を trades リストに追加
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "side": side,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_ratio": pnl_ratio
                })

                # ポジションを閉じる
                current_position = 0
                days_in_pos = 0
                entry_price = None
                entry_date = None
                exited = True

        # --- (B) エグジット後、あるいはノーポジの場合にエントリー条件をチェック ---
        if current_position == 0:
            # まだエグジットした日でもエントリーする可能性あり（同日反転エントリー等）
            if dev_ > entry_threshold:
                # ショートエントリー
                current_position = -1
                days_in_pos = 1
                entry_price = close_
                entry_date = date_
            elif dev_ < -entry_threshold:
                # ロングエントリー
                current_position = +1
                days_in_pos = 1
                entry_price = close_
                entry_date = date_

        # --- 本日の position 状態を記録しておく ---
        backtest_df.loc[i, 'position'] = current_position
        backtest_df.loc[i, 'days_in_position'] = days_in_pos

    # ---- もし最後までポジションが残っていたら強制クローズする (任意) ----
    # 今回はサンプルなので、明示的にやりたい場合は以下コメントアウトを外してください
    """
    if current_position != 0 and entry_price is not None:
        # 最終日の価格でクローズ
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

        # リセット
        current_position = 0
        days_in_pos = 0
        entry_price = None
        entry_date = None
    """

    # ==== 5. 日次リターン（戦略リターン） ====
    backtest_df['daily_return'] = backtest_df['close'].pct_change().fillna(0)
    backtest_df['strategy_return'] = backtest_df['position'] * backtest_df['daily_return']
    backtest_df['strategy_cumprod'] = (1 + backtest_df['strategy_return']).cumprod() - 1

    # ==== 6. トレード結果の集計・勝率の算出 ====
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        # 勝ちトレードかどうか (pnl_ratio > 0)
        trades_df['is_win'] = trades_df['pnl_ratio'] > 0
        winning_trades = trades_df['is_win'].sum()
        total_trades = len(trades_df)
        win_rate = winning_trades / total_trades * 100
        avg_pnl = trades_df['pnl_ratio'].mean() * 100  # 平均利益率(%)
        print("=== Trade Summary ===")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Average PnL: {avg_pnl:.2f}%")
        print("=====================")
    else:
        print("No trades were generated in this period.")


    # (A) 乖離率とポジションをプロット
    plt.figure(figsize=(12,5))
    plt.title("Deviation Rate (%) and Position")
    plt.plot(backtest_df['start_at'], backtest_df['deviation_rate'], label='Deviation Rate (%)')
    plt.plot(backtest_df['start_at'], backtest_df['position'], label='Position', color='orange')
    plt.axhline(y= entry_threshold, color='red', linestyle='--', label='Entry Threshold +5%')
    plt.axhline(y=-entry_threshold, color='green', linestyle='--', label='Entry Threshold -5%')
    plt.legend()
    plt.show()

    # (B) 戦略の累積リターン
    plt.figure(figsize=(12,5))
    plt.title("Strategy Cumulative Return")
    plt.plot(backtest_df['start_at'], backtest_df['strategy_cumprod'], label='Strategy Cum. Return')
    plt.axhline(0, color='gray', linestyle='--')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

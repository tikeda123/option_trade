# バックテストのロジックをリファクタリングしたもの
# backtester.py

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

LONG = 1
SHORT = -1
FLAT = 0

def compute_pnl_ratio(side: int, entry_price: float, exit_price: float) -> float:
    """
    side (1=LONG, -1=SHORT) に応じた損益率を計算する。
    """
    if side == LONG:
        return (exit_price / entry_price) - 1.0
    elif side == SHORT:
        return 1.0 - (exit_price / entry_price)
    else:
        return 0.0

def record_trade(
    trades_list: List[Dict[str, Any]],
    side: int,
    entry_date: datetime,
    exit_date: datetime,
    entry_price: float,
    exit_price: float
):
    """
    トレードの情報を trades_list に追加する。
    """
    pnl_ratio = compute_pnl_ratio(side, entry_price, exit_price)
    trades_list.append({
        "entry_date": entry_date,
        "exit_date": exit_date,
        "side": side,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl_ratio": pnl_ratio
    })

def force_close_last_position(
    df: pd.DataFrame,
    trades_list: List[Dict[str, Any]],
    position_state: Dict[str, Any],
    price_col: str
):
    """
    バックテスト最終日にポジションが残っている場合、強制的にクローズしトレードを記録する。
    """
    i_last = len(df) - 1
    price_last = df.loc[i_last, price_col]
    date_last = df.loc[i_last, 'start_at']

    record_trade(
        trades_list,
        position_state["side"],
        position_state["entry_date"],
        date_last,
        position_state["entry_price"],
        price_last
    )

    # ポジション情報のクリア
    df.loc[i_last, 'position'] = FLAT


# ================================
# ここから下がリファクタリングで追加する「ポジション管理」周りの関数例
# ================================

def init_position_state() -> Dict[str, Any]:
    """
    ポジション管理に必要な状態をまとめた辞書を初期化して返す。
    """
    return {
        "side": FLAT,
        "entry_price": None,
        "entry_date": None,
        "days_in_position": 0,
        "prev_profit": None,       # 前日の含み益
        "negative_streak": 0      # 連続で含み益が減少した日数
    }

def calc_unrealized_profit(position_state: Dict[str, Any], current_price: float) -> float:
    """
    現在のポジション状態をもとに、最新の含み益を計算して返す。
    """
    side = position_state["side"]
    entry_price = position_state["entry_price"]

    if side == LONG:
        return current_price - entry_price
    elif side == SHORT:
        return entry_price - current_price
    else:
        return 0.0

def check_forced_close(
    position_state: Dict[str, Any],
    trades: List[Dict[str, Any]],
    current_price: float,
    current_date: datetime,
    consecutive_drop_limit: int
) -> None:
    """
    連続で含み益が減少した場合にポジションを強制クローズする。
    強制クローズが発生した場合、position_state を更新する(FLATにする)。
    """
    # ポジションがない場合は何もしない
    if position_state["side"] == FLAT:
        return

    # 現在の含み益を計算
    current_profit = calc_unrealized_profit(position_state, current_price)

    prev_profit = position_state["prev_profit"]
    negative_streak = position_state["negative_streak"]

    # 前日利益と比較し、減っていたら negative_streak を+1
    if (prev_profit is not None) and (current_profit < prev_profit):
        negative_streak += 1
    else:
        negative_streak = 0

    position_state["negative_streak"] = negative_streak
    position_state["prev_profit"] = current_profit  # 今回の利益を次回比較用に保存

    # 強制クローズの判定
    if negative_streak >= consecutive_drop_limit:
        # 強制クローズ
        record_trade(
            trades,
            position_state["side"],
            position_state["entry_date"],
            current_date,
            position_state["entry_price"],
            current_price
        )
        # ポジション情報をリセット
        position_state["side"] = FLAT
        position_state["entry_price"] = None
        position_state["entry_date"] = None
        position_state["days_in_position"] = 0
        position_state["prev_profit"] = None
        position_state["negative_streak"] = 0

def check_doten(
    position_state: Dict[str, Any],
    trades: List[Dict[str, Any]],
    signal_yesterday: float,
    current_price: float,
    current_date: datetime
) -> None:
    """
    ドテン（反対シグナルが出た時に現在ポジションをクローズして逆方向にエントリー）を処理する。
    ポジションが存在しない場合は何もしない。
    """
    side = position_state["side"]
    if side == LONG and signal_yesterday < 0:
        # LONG→SHORT へのドテン
        record_trade(
            trades,
            side,
            position_state["entry_date"],
            current_date,
            position_state["entry_price"],
            current_price
        )
        # SHORT で新規建て
        position_state["side"] = SHORT
        position_state["entry_price"] = current_price
        position_state["entry_date"] = current_date
        position_state["days_in_position"] = 1
        position_state["prev_profit"] = 0
        position_state["negative_streak"] = 0
        print(f"ドテン: {current_date}")

    elif side == SHORT and signal_yesterday > 0:
        # SHORT→LONG へのドテン
        record_trade(
            trades,
            side,
            position_state["entry_date"],
            current_date,
            position_state["entry_price"],
            current_price
        )
        # LONG で新規建て
        position_state["side"] = LONG
        position_state["entry_price"] = current_price
        position_state["entry_date"] = current_date
        position_state["days_in_position"] = 1
        position_state["prev_profit"] = 0
        position_state["negative_streak"] = 0
        print(f"ドテン: {current_date}")

def check_new_entry(
    position_state: Dict[str, Any],
    signal_yesterday: float,
    current_price: float,
    current_date: datetime,
    upper2_value: float,
    lower2_value: float
) -> None:
    """
    FLAT の場合のみ、新規エントリー判定を行う。
    シグナル + ボリンジャーバンド(upper2, lower2)を参考にエントリー。
    """
    if position_state["side"] != FLAT:
        return

    # シグナルがプラスかつ close > upper2 => ロング
    if (signal_yesterday > 0) and (current_price > upper2_value):
        position_state["side"] = LONG
        position_state["entry_price"] = current_price
        position_state["entry_date"] = current_date
        position_state["days_in_position"] = 1
        position_state["prev_profit"] = 0
        position_state["negative_streak"] = 0

    # シグナルがマイナスかつ close < lower2 => ショート
    elif (signal_yesterday < 0) and (current_price < lower2_value):
        position_state["side"] = SHORT
        position_state["entry_price"] = current_price
        position_state["entry_date"] = current_date
        position_state["days_in_position"] = 1
        position_state["prev_profit"] = 0
        position_state["negative_streak"] = 0


def run_backtest(
    df: pd.DataFrame,
    price_col: str = "close",
    signal_col: str = "signal",
    consecutive_drop_limit: int = 2  # 何回連続で「利益が減少」したら強制クローズするか
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    与えられたDataFrame(必須カラム: price_col, signal_col, upper2, lower2)を使い、
    - シグナルに応じてポジションを持つ or 持たないを決定
    - 2σボリンジャーバンド(upper2, lower2)を突き抜けた場合にエントリー
    - 反対シグナルが出たらドテン(クローズ+反対ポジション)
    - 「前日の含み益より今日の含み益が小さい」という日が連続で発生したら強制クローズ
    などのロジックを実装したバックテスト関数。
    """

    # ========== 1. 前処理 ==========

    df = df.copy()
    df.sort_values("start_at", inplace=True)
    df.reset_index(drop=True, inplace=True)

    required_cols = [price_col, signal_col, "upper2", "lower2"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame.")

    df.dropna(subset=required_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 結果用列初期化
    df['position'] = FLAT
    df['days_in_position'] = 0
    df['daily_return'] = 0.0
    df['strategy_return'] = 0.0
    df['strategy_cumprod'] = 0.0

    # トレード履歴格納用
    trades: List[Dict[str, Any]] = []

    # ポジションの状態をまとめて管理
    position_state = init_position_state()

    # ========== 2. メインループ ==========

    for i in range(1, len(df)):
        current_price = df.loc[i, price_col]
        current_date  = df.loc[i, 'start_at']

        # 前日のシグナルを参照（※i番目の行には当日の価格や日時が入っている想定）
        signal_yesterday = df.loc[i-1, signal_col]

        upper2_value = df.loc[i, 'upper2']
        lower2_value = df.loc[i, 'lower2']

        # --- (A) 強制クローズ判定（連続で利益が減少しているか） ---
        check_forced_close(
            position_state,
            trades,
            current_price,
            current_date,
            consecutive_drop_limit
        )

        # --- (B) ドテン判定 ---
        check_doten(
            position_state,
            trades,
            signal_yesterday,
            current_price,
            current_date
        )

        # --- (C) 新規エントリー判定 ---
        check_new_entry(
            position_state,
            signal_yesterday,
            current_price,
            current_date,
            upper2_value,
            lower2_value
        )

        # (D) ポジション保有中なら days_in_position を更新
        if position_state["side"] != FLAT:
            # すでに新規エントリー時に days_in_position = 1 にするので、継続の場合は+1する。
            if position_state["days_in_position"] > 0:
                position_state["days_in_position"] += 1

        # (E) df に書き込み
        df.loc[i, 'position'] = position_state["side"]
        df.loc[i, 'days_in_position'] = position_state["days_in_position"]

    # ========== 3. 最終日の強制クローズ ==========
    if position_state["side"] != FLAT:
        force_close_last_position(df, trades, position_state, price_col)

    # ========== 4. リターン計算 ==========
    df['daily_return'] = df[price_col].pct_change().fillna(0)
    df['position_shifted'] = df['position'].shift(1).fillna(0)

    df['strategy_return'] = df['position_shifted'] * df['daily_return']
    df['strategy_cumprod'] = (1 + df['strategy_return']).cumprod() - 1

    # トレード履歴をDataFrame化
    trades_df = pd.DataFrame(trades)

    return df, trades_df


def analyze_trades(trades_df: pd.DataFrame) -> None:
    """
    トレード履歴をもとに勝率や平均損益率などを表示する。
    """
    if trades_df.empty:
        print("No trades were generated in this period.")
        return

    trades_df['is_win'] = trades_df['pnl_ratio'] > 0
    winning_trades = trades_df['is_win'].sum()
    total_trades = len(trades_df)

    win_rate = (winning_trades / total_trades) * 100
    avg_pnl = trades_df['pnl_ratio'].mean() * 100  # 平均損益率(%)

    print("=== Trade Summary (Bollinger 2σ & Force-Close) ===")
    print(f"Total Trades : {total_trades}")
    print(f"Winning Trades : {winning_trades}")
    print(f"Win Rate : {win_rate:.2f}%")
    print(f"Average PnL : {avg_pnl:.2f}%")
    print("=====================================")


def calculate_final_capital(df: pd.DataFrame, initial_capital: float = 1000.0) -> float:
    """
    バックテスト後の累積リターンから最終資産額を計算。
    """
    if 'strategy_cumprod' not in df.columns or df.empty:
        return initial_capital

    final_cum_return = df['strategy_cumprod'].iloc[-1]
    final_capital = initial_capital * (1 + final_cum_return)
    return final_capital

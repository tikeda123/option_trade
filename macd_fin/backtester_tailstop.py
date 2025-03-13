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
    current_side: int,
    entry_price: float,
    entry_date: datetime,
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
        current_side,
        entry_date,
        date_last,
        entry_price,
        price_last
    )

    # ポジション情報のクリア
    df.loc[i_last, 'position'] = FLAT

def run_backtest(
    df: pd.DataFrame,
    price_col: str = "close",
    signal_col: str = "signal",
    stop_loss_rate: float = None,      # 例: 0.05 (5%逆行で損切り)
    trailing_stop_rate: float = None,  # 例: 0.1  (10%逆行でトレーリングストップ)
    use_ma_stop: bool = False,         # TrueにするとMA割れ(超え)決済を実施
    ma_period: int = 20,              # MAストップに使う移動平均の期間
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    リスク管理を組み込んだ簡易バックテスト。

    Args:
        df: バックテスト対象のOHLCVなどのDataFrame
        price_col: 価格カラム名
        signal_col: 売買シグナルカラム名
        stop_loss_rate: ストップロス閾値(比率) 例: 0.05 => 5%
        trailing_stop_rate: トレーリングストップ閾値(比率) 例: 0.1 => 10%
        use_ma_stop: 移動平均線割れで手仕舞いをするかどうか
        ma_period: 移動平均線の期間

    Returns:
        (バックテスト結果を追記した DataFrame, トレード履歴 DataFrame)
    """

    # ========== 1. 前処理 ==========

    df = df.copy()
    df.sort_values("start_at", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 必要カラムの存在チェック
    for col in [price_col, signal_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame.")

    # 移動平均線ストップを使うなら移動平均を計算
    if use_ma_stop:
        ma_col = "ma_for_stop"
        df[ma_col] = df[price_col].rolling(window=ma_period).mean()
        df[ma_col].fillna(method='bfill', inplace=True)  # 初期NaNを埋める
    else:
        ma_col = None

    # NaN除去（価格やシグナルがNaNの場合は取り除く）
    df.dropna(subset=[price_col, signal_col], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 結果格納用の列を初期化
    df['position'] = FLAT
    df['days_in_position'] = 0
    df['daily_return'] = 0.0
    df['strategy_return'] = 0.0
    df['strategy_cumprod'] = 0.0

    # トレード履歴用リスト
    trades: List[Dict[str, Any]] = []

    # バックテスト用変数
    current_side = FLAT
    entry_price = None
    entry_date = None
    days_in_position = 0

    # トレーリングストップ用に、保有期間中の最高(ロング) or 最安(ショート)価格を記録
    # ロングなら highest_price_since_entry, ショートなら lowest_price_since_entry
    highest_price_since_entry = None
    lowest_price_since_entry = None

    # ========== 2. メインループ (1行目から順に見ていく) ==========
    for i in range(1, len(df)):
        price_today = df.loc[i, price_col]
        date_today  = df.loc[i, 'start_at']
        signal_yesterday = df.loc[i-1, signal_col]

        # -------------------
        # (A) すでにポジションを持っている場合：
        #     ストップロス / トレーリングストップ / MA割れ(超え) チェック
        # -------------------
        if current_side != FLAT:

            # ロングの場合
            if current_side == LONG:
                # 最高値の更新
                if highest_price_since_entry is None:
                    highest_price_since_entry = price_today
                else:
                    highest_price_since_entry = max(highest_price_since_entry, price_today)

                # 1) ストップロスチェック
                if stop_loss_rate is not None:
                    # エントリー価格から stop_loss_rate 分逆行したら決済
                    stop_price = entry_price * (1.0 - stop_loss_rate)
                    if price_today <= stop_price:
                        # 損切り
                        record_trade(trades, current_side, entry_date, date_today, entry_price, price_today)
                        # FLAT化
                        current_side = FLAT
                        entry_price = None
                        entry_date = None
                        highest_price_since_entry = None
                        lowest_price_since_entry = None
                        days_in_position = 0
                        # 当日はここで処理を終えてポジション無い状態になるので、この日の signal 判定で再エントリーがあるかも
                # 2) トレーリングストップチェック
                if current_side == LONG and trailing_stop_rate is not None and highest_price_since_entry is not None:
                    # highest_price_since_entry から trailing_stop_rate 分下落したら決済
                    trail_price = highest_price_since_entry * (1.0 - trailing_stop_rate)
                    if price_today <= trail_price:
                        # トレーリングストップ発動
                        record_trade(trades, current_side, entry_date, date_today, entry_price, price_today)
                        # FLAT化
                        current_side = FLAT
                        entry_price = None
                        entry_date = None
                        highest_price_since_entry = None
                        lowest_price_since_entry = None
                        days_in_position = 0

                # 3) 移動平均線割れチェック
                if current_side == LONG and use_ma_stop and (ma_col in df.columns):
                    ma_value = df.loc[i, ma_col]  # 今日のMA
                    if price_today < ma_value:
                        # MA割れ
                        record_trade(trades, current_side, entry_date, date_today, entry_price, price_today)
                        current_side = FLAT
                        entry_price = None
                        entry_date = None
                        highest_price_since_entry = None
                        lowest_price_since_entry = None
                        days_in_position = 0

            # ショートの場合
            elif current_side == SHORT:
                # 最安値の更新
                if lowest_price_since_entry is None:
                    lowest_price_since_entry = price_today
                else:
                    lowest_price_since_entry = min(lowest_price_since_entry, price_today)

                # 1) ストップロスチェック
                if stop_loss_rate is not None:
                    # エントリー価格から stop_loss_rate 分逆行（=上昇）したら決済
                    stop_price = entry_price * (1.0 + stop_loss_rate)
                    if price_today >= stop_price:
                        # 損切り
                        record_trade(trades, current_side, entry_date, date_today, entry_price, price_today)
                        # FLAT化
                        current_side = FLAT
                        entry_price = None
                        entry_date = None
                        highest_price_since_entry = None
                        lowest_price_since_entry = None
                        days_in_position = 0

                # 2) トレーリングストップチェック
                if current_side == SHORT and trailing_stop_rate is not None and lowest_price_since_entry is not None:
                    # lowest_price_since_entry から trailing_stop_rate 分上昇したら決済
                    trail_price = lowest_price_since_entry * (1.0 + trailing_stop_rate)
                    if price_today >= trail_price:
                        # トレーリングストップ発動
                        record_trade(trades, current_side, entry_date, date_today, entry_price, price_today)
                        current_side = FLAT
                        entry_price = None
                        entry_date = None
                        highest_price_since_entry = None
                        lowest_price_since_entry = None
                        days_in_position = 0

                # 3) 移動平均線越えチェック
                if current_side == SHORT and use_ma_stop and (ma_col in df.columns):
                    ma_value = df.loc[i, ma_col]  # 今日のMA
                    if price_today > ma_value:
                        # MA越え
                        record_trade(trades, current_side, entry_date, date_today, entry_price, price_today)
                        current_side = FLAT
                        entry_price = None
                        entry_date = None
                        highest_price_since_entry = None
                        lowest_price_since_entry = None
                        days_in_position = 0

        # -------------------
        # (B) シグナル変化によるドテン処理
        #     → シグナルが逆方向に変わったらクローズ & 反対ポジションでエントリー
        # -------------------
        if current_side == LONG:
            # ロング保有中に signal_yesterday が負ならドテン
            if signal_yesterday < 0:
                # 現ポジションをクローズ
                record_trade(trades, current_side, entry_date, date_today, entry_price, price_today)
                # ショートでドテン
                current_side = SHORT
                entry_price = price_today
                entry_date = date_today
                days_in_position = 1
                highest_price_since_entry = None
                lowest_price_since_entry = price_today

        elif current_side == SHORT:
            # ショート保有中に signal_yesterday が正ならドテン
            if signal_yesterday > 0:
                record_trade(trades, current_side, entry_date, date_today, entry_price, price_today)
                # ロングでドテン
                current_side = LONG
                entry_price = price_today
                entry_date = date_today
                days_in_position = 1
                highest_price_since_entry = price_today
                lowest_price_since_entry = None

        # -------------------
        # (C) ポジションなしの場合、新規エントリー
        # -------------------
        if current_side == FLAT:
            if signal_yesterday > 0:
                current_side = LONG
                entry_price = price_today
                entry_date = date_today
                days_in_position = 1
                highest_price_since_entry = price_today
                lowest_price_since_entry = None
            elif signal_yesterday < 0:
                current_side = SHORT
                entry_price = price_today
                entry_date = date_today
                days_in_position = 1
                highest_price_since_entry = None
                lowest_price_since_entry = price_today
            else:
                # signal == 0 なら何もしない
                pass
        else:
            # ポジション持っている場合は日数を+1
            days_in_position += 1

        # (D) 本日の終了時点のポジションと日数を DataFrame に反映
        df.loc[i, 'position'] = current_side
        df.loc[i, 'days_in_position'] = days_in_position

    # ========== 3. 最終日ポジションの強制クローズ (任意) ==========
    if current_side != FLAT and entry_price is not None:
        force_close_last_position(df, trades, current_side, entry_price, entry_date, price_col)

    # ========== 4. リターン計算 ==========
    # (positionは当日クローズ時点を指すので、翌日のリターンに対して有効)
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

    print("=== Trade Summary (MACD Hist based) ===")
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
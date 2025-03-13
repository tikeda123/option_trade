# main.py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime

from data_loader import load_market_data
from macd_fin.strategy_macd import generate_macd_signals
from backtester import run_backtest, analyze_trades, calculate_final_capital

def main():
    # ========== 1. データ取得 ==========
    df = load_market_data(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2025, 1, 10),
        coll_type="market_data_tech",  # MARKET_DATA_TECH など
        symbol='BTCUSDT',
        interval=60
    )

    # ========== 2. 戦略シグナル生成 (MACD) ==========
    # 今回はすでに df に「kalman_macdhist」列がある前提
    # それを使って MACDベースのsignal列を生成する
    df = df[['start_at', 'close', 'kalman_macdhist','lower2','upper2']].copy()  # 必要列だけに絞る
    df = generate_macd_signals(df, macdhist_col="kalman_macdhist")

    # ========== 3. バックテスト実行 ==========

    result_df, trades_df = run_backtest(
        df=df,
        price_col="close",
        signal_col="signal",
        consecutive_drop_limit=2
    )

    """
    result_df, trades_df = run_backtest(
    df=df,
    price_col="close",
    signal_col="signal",
    stop_loss_rate=0.05,
    trailing_stop_rate=0.05,
    use_ma_stop=True,
    ma_period=20
    )
    """

    # ========== 4. トレード集計 & 結果表示 ==========
    analyze_trades(trades_df)

    # バックテスト結果の一部表示
    print(result_df.head(10))
    print(result_df.tail(5))

    """
    # ========== 5. グラフ表示 (シンプルな例) ==========
    # MACDヒストグラムとポジション推移 (二軸グラフに修正)
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # 左軸: MACDヒストグラム
    color = 'blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('MACD Histogram', color=color)
    ax1.plot(result_df['start_at'], result_df['kalman_macdhist'], label='MACD Hist', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(0, color='gray', linestyle='--')

    # 右軸: ポジション
    ax2 = ax1.twinx()
    color = 'orange'
    ax2.set_ylabel('Position', color=color)
    ax2.plot(result_df['start_at'], result_df['position'], label='Position', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # タイトルと凡例
    plt.title("MACD Histogram and Position")

    # 凡例を両方の軸から取得して表示
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.show()
    """
    # 累積リターンとBTC価格（二軸グラフ）
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # 左軸: 累積リターン
    color = 'green'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Strategy Cumulative Return', color=color)
    ax1.plot(result_df['start_at'], result_df['strategy_cumprod'], label='Strategy Cum. Return', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(1, color='gray', linestyle='--')  # 初期値1.0の基準線

    # 右軸: BTC価格
    ax2 = ax1.twinx()
    color = 'blue'
    ax2.set_ylabel('BTC Price (USDT)', color=color)
    ax2.plot(result_df['start_at'], result_df['close'], label='BTC Price', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # タイトルと凡例
    plt.title("Strategy Cumulative Return vs BTC Price")

    # 凡例を両方の軸から取得して表示
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.show()

    # ========== 6. 最終資産の確認 ==========
    final_capital = calculate_final_capital(result_df, initial_capital=1000.0)
    print("=== Final Capital Calculation ===")
    print(f"Initial Capital: $1000.00")
    print(f"Final Capital:   ${final_capital:.2f}")
    print("=================================")

if __name__ == "__main__":
    main()

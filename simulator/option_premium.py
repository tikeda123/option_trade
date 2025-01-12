import os
import sys
from typing import Tuple
import numpy as np
import pandas as pd


# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the path of the parent directory to sys.path
sys.path.append(parent_dir)


from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *
from option_pricing import simulate_option_prices


def main():
    db = MongoDataLoader()
    # 日付範囲を広げて設定
    start_date = "2024-12-23 00:00:00"
    end_date = "2024-12-27 00:00:00"

    # オプションデータの取得
    df = db.load_data(OPTION_TICKER)
    print(f"Option data date range: {df.index.min()} to {df.index.max()}")

    # BTCデータの取得と日付範囲の確認
    btc_df = db.load_data_from_datetime_period(
        start_date,
        end_date,
        MARKET_DATA,
        symbol='BTCUSDT',
        interval=60
    )

    # データの存在確認とカラム名の確認
    if df is None or df.empty:
        print("Warning: No option data available")
        return

    if btc_df is None or btc_df.empty:
        print(f"Warning: No BTC data available for period {start_date} to {end_date}")
        # 利用可能なデータの日付範囲を確認
        available_btc = db.load_data(MARKET_DATA)
        if available_btc is not None and not available_btc.empty:
            print(f"Available BTC data range: {available_btc.index.min()} to {available_btc.index.max()}")
        return

    print("Option DataFrame columns:", df.columns)
    print("BTC DataFrame columns:", btc_df.columns)
    print(f"BTC data points: {len(btc_df)}")
    print(f"BTC date range: {btc_df.index.min()} to {btc_df.index.max()}")

    df = df[df['symbol'] == 'BTC-26DEC24-96500-C'].copy()

    # タイムスタンプカラムの処理（'timestamp'または'date'を使用）
    timestamp_col = 'timestamp' if 'timestamp' in df.columns else 'date'
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df.set_index(timestamp_col, inplace=True)
    df.sort_index(inplace=True)

    # BTCデータのタイムスタンプ処理
    btc_timestamp_col = 'timestamp' if 'timestamp' in btc_df.columns else 'date'
    btc_df[btc_timestamp_col] = pd.to_datetime(btc_df[btc_timestamp_col])
    btc_df.set_index(btc_timestamp_col, inplace=True)
    btc_df.sort_index(inplace=True)

    # 数値変換
    df['bid1Price'] = pd.to_numeric(df['bid1Price'], errors='coerce')
    df['ask1Price'] = pd.to_numeric(df['ask1Price'], errors='coerce')
    df['bid1Iv'] = pd.to_numeric(df['bid1Iv'], errors='coerce')
    df['ask1Iv'] = pd.to_numeric(df['ask1Iv'], errors='coerce')

    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.dates import DateFormatter

    # Set style
    sns.set_style("darkgrid")

    # Create figure with three y-axes
    fig, ax1 = plt.subplots(figsize=(15, 8))

    # Plot option prices on primary y-axis
    color = 'blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Option Price (USDT)', color=color)
    line1 = ax1.plot(df.index, df['bid1Price'], color='blue', label='Option Bid Price')
    line2 = ax1.plot(df.index, df['ask1Price'], color='green', label='Option Ask Price')
    ax1.tick_params(axis='y', labelcolor=color)

    # Create secondary y-axis for IV
    ax2 = ax1.twinx()
    color = 'red'
    ax2.set_ylabel('Implied Volatility', color=color)
    line3 = ax2.plot(df.index, df['bid1Iv'], color='red', label='Bid IV', linestyle='--')
    line4 = ax2.plot(df.index, df['ask1Iv'], color='orange', label='Ask IV', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)

    # Create third y-axis for BTC price
    ax3 = ax1.twinx()
    # オフセットを設定して、ax2とax3を分離
    ax3.spines['right'].set_position(('outward', 60))
    color = 'purple'
    ax3.set_ylabel('BTC Price (USDT)', color=color)
    line5 = ax3.plot(btc_df.index, btc_df['close'], color=color, label='BTC Price', alpha=0.7)
    ax3.tick_params(axis='y', labelcolor=color)

    # x軸の日付フォーマットを設定
    date_formatter = DateFormatter('%Y-%m-%d %H:%M')
    ax1.xaxis.set_major_formatter(date_formatter)

    # x軸のラベルを45度回転して見やすくする
    plt.xticks(rotation=45)

    # Combine legends
    lines = line1 + line2 + line3 + line4 + line5
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.title('Option Prices, Implied Volatility and BTC Price Over Time')

    # グラフの余白を調整して日付ラベルが切れないようにする
    plt.tight_layout()

    # Save the figure
    plt.savefig('option_prices_and_iv.png')

    # Show the plot
    plt.show()

    # Close the figure
    plt.close()

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print("\nBid Price:")
    print(f"Mean: {df['bid1Price'].mean():.2f} USDT")
    print(f"Min: {df['bid1Price'].min():.2f} USDT")
    print(f"Max: {df['bid1Price'].max():.2f} USDT")

    print("\nAsk Price:")
    print(f"Mean: {df['ask1Price'].mean():.2f} USDT")
    print(f"Min: {df['ask1Price'].min():.2f} USDT")
    print(f"Max: {df['ask1Price'].max():.2f} USDT")

    print("\nBid IV:")
    print(f"Mean: {df['bid1Iv'].mean()*100:.2f}%")
    print(f"Min: {df['bid1Iv'].min()*100:.2f}%")
    print(f"Max: {df['bid1Iv'].max()*100:.2f}%")

    print("\nAsk IV:")
    print(f"Mean: {df['ask1Iv'].mean()*100:.2f}%")
    print(f"Min: {df['ask1Iv'].min()*100:.2f}%")
    print(f"Max: {df['ask1Iv'].max()*100:.2f}%")
    print(df)
    pass


if __name__ == "__main__":
    main()

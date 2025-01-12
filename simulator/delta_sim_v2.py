import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import OPTION_TICKER, MARKET_DATA

def main():
    db = MongoDataLoader()

    # オプションデータを取得
    df = db.load_data(OPTION_TICKER)
    if df is None or df.empty:
        print("Warning: No option data available")
        return

    # 該当シンボルに絞る
    df = df[df['symbol'] == 'BTC-26DEC24-96500-C'].copy()
    if df.empty:
        print("Warning: No data for symbol='BTC-26DEC24-96500-C'")
        return

    # タイムスタンプ処理
    timestamp_col = 'timestamp' if 'timestamp' in df.columns else 'date'
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df.set_index(timestamp_col, inplace=True)
    df.sort_index(inplace=True)

    # ask1Price を "option_price" カラムとして扱う
    df.rename(columns={'ask1Price': 'option_price'}, inplace=True)

    # デルタとオプション価格を数値化
    df['delta'] = pd.to_numeric(df['delta'], errors='coerce')
    df['option_price'] = pd.to_numeric(df['option_price'], errors='coerce')

    # BTC データの取得
    start_date = "2024-12-23 00:00:00"
    end_date   = "2024-12-27 00:00:00"
    btc_df = db.load_data_from_datetime_period(
        start_date,
        end_date,
        MARKET_DATA,
        symbol='BTCUSDT',
        interval=60
    )
    if btc_df is None or btc_df.empty:
        print(f"Warning: No BTC data available for period {start_date} to {end_date}")
        return

    # BTC 側もタイムスタンプ処理
    btc_timestamp_col = 'timestamp' if 'timestamp' in btc_df.columns else 'date'
    btc_df[btc_timestamp_col] = pd.to_datetime(btc_df[btc_timestamp_col])
    btc_df.set_index(btc_timestamp_col, inplace=True)
    btc_df.sort_index(inplace=True)

    # BTC の価格を数値型に変換
    btc_df['close'] = pd.to_numeric(btc_df['close'], errors='coerce')

    # 2つの DataFrame を結合
    merged_df = df[['delta', 'option_price']].join(btc_df[['close']], how='inner')
    if merged_df.empty:
        print("Warning: No overlapping timestamps between option data and BTC data.")
        return

    # グラフ描画
    sns.set_style("whitegrid")

    # 1. プレミアム(オプション価格) と デルタ
    plt.figure(figsize=(10, 6))
    plt.title("Option Price and Delta over Time (BTC-26DEC24-96500-C)")

    ax1 = plt.gca()
    color1 = "blue"
    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel("Delta", color=color1)
    l1 = ax1.plot(merged_df.index, merged_df["delta"], color=color1, label="Delta")
    ax1.tick_params(axis='y', labelcolor=color1)

    color2 = "green"
    ax2 = ax1.twinx()
    ax2.set_ylabel("Option Price", color=color2)
    l2 = ax2.plot(merged_df.index, merged_df["option_price"], color=color2, label="Option Price")
    ax2.tick_params(axis='y', labelcolor=color2)

    # 凡例の設定
    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 2. BTC価格 と デルタ
    plt.figure(figsize=(10, 6))
    plt.title("BTC Price and Delta over Time (BTC-26DEC24-96500-C)")

    ax3 = plt.gca()
    color_btc = "red"
    ax3.set_xlabel("Timestamp")
    ax3.set_ylabel("BTC Price", color=color_btc)
    l3 = ax3.plot(merged_df.index, merged_df["close"], color=color_btc, label="BTC Price")
    ax3.tick_params(axis='y', labelcolor=color_btc)

    color_delta = "blue"
    ax4 = ax3.twinx()
    ax4.set_ylabel("Delta", color=color_delta)
    l4 = ax4.plot(merged_df.index, merged_df["delta"], color=color_delta, label="Delta")
    ax4.tick_params(axis='y', labelcolor=color_delta)

    lines = l3 + l4
    labels = [line.get_label() for line in lines]
    ax3.legend(lines, labels, loc="upper left")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 3. BTC価格 と プレミアム(オプション価格)
    plt.figure(figsize=(10, 6))
    plt.title("BTC Price and Option Price over Time (BTC-26DEC24-96500-C)")

    ax5 = plt.gca()
    color_btc2 = "red"
    ax5.set_xlabel("Timestamp")
    ax5.set_ylabel("BTC Price", color=color_btc2)
    l5 = ax5.plot(merged_df.index, merged_df["close"], color=color_btc2, label="BTC Price")
    ax5.tick_params(axis='y', labelcolor=color_btc2)

    color_option = "green"
    ax6 = ax5.twinx()
    ax6.set_ylabel("Option Price", color=color_option)
    l6 = ax6.plot(merged_df.index, merged_df["option_price"], color=color_option, label="Option Price")
    ax6.tick_params(axis='y', labelcolor=color_option)

    lines = l5 + l6
    labels = [line.get_label() for line in lines]
    ax5.legend(lines, labels, loc="upper left")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

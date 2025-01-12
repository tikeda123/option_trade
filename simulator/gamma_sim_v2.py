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

    # オプションデータ (ガンマを含む)
    df = db.load_data(OPTION_TICKER)
    if df is None or df.empty:
        print("Warning: No option data available")
        return

    # シンボル 'BTC-26DEC24-96500-C' に絞る
    df = df[df['symbol'] == 'BTC-26DEC24-96500-C'].copy()
    if df.empty:
        print("Warning: No data for symbol='BTC-26DEC24-96500-C'")
        return

    # タイムスタンプ処理 (オプション側)
    timestamp_col = 'timestamp' if 'timestamp' in df.columns else 'date'
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df.set_index(timestamp_col, inplace=True)
    df.sort_index(inplace=True)

    # ガンマを数値型に変換 (必要に応じて)
    df['gamma'] = pd.to_numeric(df['gamma'], errors='coerce')

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

    # タイムスタンプ処理 (BTC 側)
    btc_timestamp_col = 'timestamp' if 'timestamp' in btc_df.columns else 'date'
    btc_df[btc_timestamp_col] = pd.to_datetime(btc_df[btc_timestamp_col])
    btc_df.set_index(btc_timestamp_col, inplace=True)
    btc_df.sort_index(inplace=True)

    # BTCの価格を数値型に変換 (念のため)
    btc_df['close'] = pd.to_numeric(btc_df['close'], errors='coerce')

    # 2つのデータフレームを「同じ timestamp」で結合（inner join）
    merged_df = df[['gamma']].join(btc_df[['close']], how='inner')
    if merged_df.empty:
        print("Warning: No overlapping timestamps between option data and BTC data.")
        return

    # ===== グラフ描画 =====
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    # 左Y軸: Gamma
    ax1 = plt.gca()
    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel("Gamma", color="blue")
    ax1.plot(merged_df.index, merged_df['gamma'], color="blue", label="Gamma")
    ax1.tick_params(axis='y', labelcolor="blue")

    # 縦軸を0～1に固定したい場合は下記をコメント解除
    # ax1.set_ylim(0, 1)

    # 右Y軸: BTC Price
    ax2 = ax1.twinx()
    ax2.set_ylabel("BTC Price (close)", color="red")
    ax2.plot(merged_df.index, merged_df['close'], color="red", label="BTC Price")
    ax2.tick_params(axis='y', labelcolor="red")

    # 凡例をまとめる
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.title("Gamma and BTC Price over Time (BTC-26DEC24-96500-C)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

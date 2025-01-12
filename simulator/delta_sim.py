import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 親ディレクトリをパスに追加（MongoDataLoader 等をインポートするため）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import OPTION_TICKER, MARKET_DATA

def main():
    db = MongoDataLoader()
    start_date = "2024-12-23 00:00:00"
    end_date = "2024-12-27 00:00:00"

    # オプションデータ
    df = db.load_data(OPTION_TICKER)
    df = df[df['symbol'] == 'BTC-26DEC24-96500-C'].copy()
    timestamp_col = 'timestamp' if 'timestamp' in df.columns else 'date'
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df.sort_values(by=timestamp_col, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # BTCデータ
    btc_df = db.load_data_from_datetime_period(
        start_date, end_date, MARKET_DATA,
        symbol='BTCUSDT', interval=60
    )
    btc_timestamp_col = 'timestamp' if 'timestamp' in btc_df.columns else 'date'
    btc_df[btc_timestamp_col] = pd.to_datetime(btc_df[btc_timestamp_col])
    btc_df.sort_values(by=btc_timestamp_col, inplace=True)
    btc_df.reset_index(drop=True, inplace=True)

    # merge_asof で近い時刻を紐付け (例: 5分以内のレコードを許容)
    merged_asof = pd.merge_asof(
        df[['delta', timestamp_col]],
        btc_df[[btc_timestamp_col, 'close']],
        left_on=timestamp_col,
        right_on=btc_timestamp_col,
        direction='nearest',                       # 前後どちらに近くてもOK
        tolerance=pd.Timedelta('5min')            # 5分以内のデータを許容
    )

    # ここで merged_asof は連続した時刻 (オプション側が基準) に BTC close が紐付く
    print(merged_asof.head(20))

    # Plot
    merged_asof.set_index(timestamp_col, inplace=True)

    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel("Delta", color='blue')
    line1 = ax1.plot(merged_asof.index, merged_asof['delta'], color='blue', label='Delta')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel("BTC Price (close)", color='red')
    line2 = ax2.plot(merged_asof.index, merged_asof['close'], color='red', label='BTC Price')
    ax2.tick_params(axis='y', labelcolor='red')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.title("BTC Price and Delta over Time (using merge_asof)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

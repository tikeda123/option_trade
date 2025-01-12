import sys
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# Get the absolute path of the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from bybit_api.bybit_data_fetcher import BybitDataFetcher
from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *

def main():
    api = BybitDataFetcher()
    data_loader = MongoDataLoader()

    start_date = "2024-12-17 00:00:00+0900"
    end_date = "2024-12-18 23:00:00+0900"

    # BTCデータ読み込み
    btc_data = data_loader.load_data_from_datetime_period(
        start_date=start_date,
        end_date=end_date,
        coll_type=MARKET_DATA,
        symbol="BTCUSDT",
        interval=60
    )

    # HVデータ読み込み
    hv_data = api.fetch_historical_volatility_extended(
        baseCoin="BTC",
        period=7,
        startTime=start_date,
        endTime=end_date
    )

    # データ型変換・整形
    btc_data['start_at'] = pd.to_datetime(btc_data['start_at'])
    btc_data = btc_data.sort_values('start_at')

    hv_data['start_at'] = pd.to_datetime(hv_data['start_at'])
    hv_data['value'] = pd.to_numeric(hv_data['value'], errors='coerce')
    hv_data = hv_data.dropna(subset=['value'])
    hv_data = hv_data.drop_duplicates(subset=['start_at'])
    hv_data = hv_data.sort_values('start_at')

    # データ数・概要確認（デバッグ用、必要なければコメントアウト）
    print("BTC Data points:", len(btc_data))
    print("HV Data points:", len(hv_data))
    print(hv_data.head(10))
    print(hv_data.tail(10))

    # プロット作成
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 左軸: BTC価格
    ax1.plot(btc_data['start_at'], btc_data['close'], color='blue', label='BTC Price')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('BTC Price (USD)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # 右軸: HV
    ax2 = ax1.twinx()
    # マーカー付きでプロット（データポイントを可視化）
    ax2.plot(hv_data['start_at'], hv_data['value'], color='red', label='HV', marker='o', linestyle='-')
    ax2.set_ylabel('Historical Volatility (HV)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # 日付フォーマットの設定
    date_formatter = DateFormatter('%Y-%m-%d')
    ax1.xaxis.set_major_formatter(date_formatter)
    plt.xticks(rotation=45)

    # レジェンドの表示
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title('BTC Price vs Historical Volatility (HV)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

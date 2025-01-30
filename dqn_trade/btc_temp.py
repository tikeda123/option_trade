import pandas as pd
import numpy as np
import os
import sys


# Mongoなどの独自ライブラリに依存する部分を仮定
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import MARKET_DATA_TECH


def main():
    # ====================================================
    # 1. データロード
    # ====================================================
    data_loader = MongoDataLoader()
    df = data_loader.load_data_from_datetime_period(
        start_date="2020-01-01 00:00:00",
        end_date="2025-01-06 00:00:00",
        coll_type=MARKET_DATA_TECH,
        symbol="BTCUSDT",
        interval=1440
    )

    print(df.head())
    print(df.columns)

if __name__ == "__main__":
    main()

# data_loader.py
import sys
import os
from datetime import datetime
import pandas as pd

# ユーザー環境に合わせて import パスを設定
# 以下は例示であり、必要であれば修正してください
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *

def load_market_data(
    start_date: datetime,
    end_date: datetime,
    coll_type: str,
    symbol: str,
    interval: int
) -> pd.DataFrame:
    """
    MongoDBから指定期間・指定シンボルのデータを取得して返す。
    """
    db = MongoDataLoader()
    df = db.load_data_from_datetime_period(
        start_date,
        end_date,
        coll_type=coll_type,
        symbol=symbol,
        interval=interval
    )
    return df

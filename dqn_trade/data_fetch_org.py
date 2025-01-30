# file: data_fetcher.py

import os
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, List
from tqdm import tqdm

# Add parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import MARKET_DATA_TECH


def fetch_data():
    data_loader = MongoDataLoader()
    df = data_loader.load_data_from_datetime_period(
        start_date="2023-01-01",
        end_date="2024-01-01",
        coll_type=MARKET_DATA_TECH,
        symbol="BTCUSDT",
        interval=60
    )

    return df

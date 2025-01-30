import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# pip install arch
from arch import arch_model

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the path of the parent directory to sys.path
sys.path.append(parent_dir)

from common.trading_logger import TradingLogger
from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import MARKET_DATA_TECH,TIME_SERIES_PERIOD,PRED_TYPE_LONG,PRED_TYPE_SHORT
from common.utils import get_config_model

from aiml.garch_price_distribution_model import GarchPriceDistributionModel,judge_current_price
from aiml.prediction_manager import PredictionManager
from aiml.bayes_prediction_model import BayesianPricePredictor

def option_prediction_bayes_model(current_date):
    start_date = "2020-01-01"
    end_date   = current_date

    data_loader = MongoDataLoader()
    df = data_loader.load_data_from_datetime_period(
        start_date,
        end_date,
        coll_type=MARKET_DATA_TECH,
        symbol="BTCUSDT",
        interval=1440
    )

    model = BayesianPricePredictor(
        symbol="BTCUSDT",
        interval=1440,
        features=["close", "ema", "macdhist","rsi"],  # 例として close と ema だけを特徴量に
        model_params={
            "max_iter": 300,
            "tol": 1e-6,
            "alpha_1": 1e-6,
            "alpha_2": 1e-6,
            "lambda_1": 1e-6,
            "lambda_2": 1e-6,
        },
        target_col="close",  # 他の変数に変更可能
        shift_steps=3        # 予測期間(バー数)を変更可能
    )

    # データをクラス内部にロード
    df = model.load_data(start_date, end_date)

    # 前処理 & 学習・テスト分割
    # target_col や shift_steps はここでもオーバーライド可能
    model.preprocess_data(df, train_end_date="2024-10-01", test_start_date="2024-10-01")

    # クロスバリデーション (学習データのみで実施)
    model.cross_validate(n_splits=20, plot_last_fold=True)

    # 全学習データで最終モデルを訓練
    model.train_final_model()

def main():
    current_date = "2024-8-01 04:00:00"

if __name__ == "__main__":
    main()


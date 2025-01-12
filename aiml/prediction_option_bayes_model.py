import os
import sys
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import BayesianRidge

import joblib

# ===== MongoDataLoader等、外部ライブラリの読み込み ===== #
# プロジェクト構成によってはパス調整を行う
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mongodb.data_loader_mongo import MongoDataLoader
from common.utils import get_config
from common.constants import MARKET_DATA_TECH
from aiml.bayes_prediction_model import BayesianPricePredictor


def prediction_option_bayes_model(current_date:str,symbol="BTCUSDT",m_interval=1440):
    """
    ベイズモデルによる予測
    """
    #==== 1. パラメータ設定 & モデルインスタンス化 ====#
    start_date = "2020-01-01"
    end_date   = current_date

    model = BayesianPricePredictor(
        symbol="BTCUSDT",
        interval=m_interval,
        features=[
            "open", "close", "high", "low", "volume",
            "rsi", "macd", "macdsignal",
            "ema", "sma",
            "upper1", "lower1", "middle",
        ],
        model_params={
            "max_iter": 300,
            "tol": 1e-6,
            "alpha_1": 1e-6,
            "alpha_2": 1e-6,
            "lambda_1": 1e-6,
            "lambda_2": 1e-6,
        }
    )

    #==== 2. データ読み込み ====#
    df = model.load_data(start_date, end_date)
    #==== 3. 前処理 & 学習・テスト分割 ====#
    model.preprocess_data(df, train_end_date=end_date, test_start_date=end_date)

    #==== 4. クロスバリデーション ====#
    model.cross_validate(n_splits=5, plot_last_fold=True)

    #==== 5. 全学習データで最終モデルを訓練 ====#
    model.train_final_model()

    return model.predict_for_date(current_date, return_std=True)



def main():
    #==== 1. パラメータ設定 & モデルインスタンス化 ====#
    current_date = "2024-12-29"
    pred_value, pred_std = prediction_option_bayes_model(current_date)
    print(current_date, pred_value, pred_std)

if __name__ == "__main__":
    main()

import os
import sys
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Get the absolute path of the current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
PARENT_DIR = os.path.dirname(CURRENT_DIR)
# Add the path of the parent directory to sys.path
sys.path.append(PARENT_DIR)

from common.trading_logger import TradingLogger
from mongodb.data_loader_mongo import MongoDataLoader
from common.utils import get_config
from common.constants import MARKET_DATA_TECH, TIME_SERIES_PERIOD

from aiml.transformer_prediction_rolling_model import transfomer_predictin_trend_setup, transfomer_predictin_trend
from aiml.garch_price_distribution_model import judge_current_price, garch_volatility_forecast, garch_price_distribution_analysis, get_current_price
from aiml.bayes_prediction_model import execute_bayes_prediction_model

def execute_technical_prediction(output_file: str) -> None:
    """
    Example main function demonstrating how to use the TransformerPredictionRollingModel.
    Records predictions to a text file.
    """
    print(":::start:::")
    current_date = datetime.strptime("2025-01-15 00:00:00", "%Y-%m-%d %H:%M:%S")
    prediction_date = current_date + timedelta(days=3)
    model_target = ["ema", "macdhist", "roc", "mfi", "aroon"]
    model_id_list = ["rolling_v1", "rolling_v2", "rolling_v3", "rolling_v4", "rolling_v5"]
    model_list = []

    for model_id in model_id_list:
        model = transfomer_predictin_trend_setup(model_id, "BTCUSDT", 1440)
        model_list.append(model)

    print(f"current_date:{current_date},prediction_date:{prediction_date}")

    # Write technical predictions
    with open(output_file, "a") as f:
        f.write("\n=== Technical Analysis Predictions ===\n")
        f.write(f"Analysis Time: {current_date}\n")
        f.write(f"Prediction Target Time: {prediction_date}\n")
        f.write("-" * 50 + "\n\n")

        for i, model in enumerate(model_list):
            up_pred, down_pred, current_value = transfomer_predictin_trend(current_date, model, model_target[i])

            # Write to console
            print(f"target prediction technical indicator :{model_target[i]}")
            print(f"up prediction probability :{up_pred:.2f}%, down prediction probability:{down_pred:.2f}%")

            # Write to file
            f.write(f"Technical Indicator: {model_target[i]}\n")
            f.write(f"Up prediction probability: {up_pred:.2f}%\n")
            f.write(f"Down prediction probability: {down_pred:.2f}%\n")
            f.write(f"Current value: {current_value:.2f}\n")
            f.write("\n")

def execute_garch_price_distribution_analysis(current_date_str: str, horizon: int, interval_min: int, output_file: str) -> None:
    """
    Execute garch price distribution analysis and forecast volatility
    """
    quantiles = [0.1, 0.5, 0.9]

    current_price = get_current_price(current_date_str, interval_min)
    garch_result = garch_price_distribution_analysis(current_date_str, horizon=horizon, interval_min=interval_min)
    lower_bound = garch_result['forecast_price_quantiles'][quantiles[0]]
    median_val = garch_result['forecast_price_quantiles'][quantiles[1]]
    upper_bound = garch_result['forecast_price_quantiles'][quantiles[2]]
    returns_lower_bound = garch_result['forecast_returns_quantiles'][quantiles[0]]
    returns_median_val = garch_result['forecast_returns_quantiles'][quantiles[1]]
    returns_upper_bound = garch_result['forecast_returns_quantiles'][quantiles[2]]
    judgement_result = judge_current_price(current_price, lower_bound, upper_bound)

    volatility = garch_volatility_forecast(current_date_str, horizon=horizon, interval_min=interval_min)

    # Console output
    print("\n=== GARCHモデルによる価格分布分析結果 ===")
    print("目的: 将来の価格変動の確率分布を予測し、現在の価格水準の評価と将来のボラティリティを推定")

    # File output
    with open(output_file, "a") as f:
        f.write("\n=== GARCH Price Distribution Analysis ===\n")
        f.write(f"Analysis Time: {current_date_str}\n")
        f.write(f"Current Price: {current_price:,.2f}\n\n")

        f.write("Price Prediction Range (Confidence Intervals):\n")
        f.write(f"Lower bound (10th percentile): {lower_bound:,.2f}\n")
        f.write(f"Median price (50th percentile): {median_val:,.2f}\n")
        f.write(f"Upper bound (90th percentile): {upper_bound:,.2f}\n\n")

        f.write("Predicted Returns Range:\n")
        f.write(f"Lower bound return (10th percentile): {returns_lower_bound:.2%}\n")
        f.write(f"Median return (50th percentile): {returns_median_val:.2%}\n")
        f.write(f"Upper bound return (90th percentile): {returns_upper_bound:.2%}\n\n")

        f.write("Price Level Evaluation:\n")
        f.write(f"Current price evaluation: {judgement_result}\n\n")

        f.write("Volatility Forecast:\n")
        f.write(f"Predicted volatility: {volatility:.2%}\n")
        f.write("-" * 50 + "\n")

def execute_bayes_prediction(current_date: str, symbol: str, interval: int, output_file: str):
    current_price = get_current_price(current_date, interval)
    prediction, std_dev, is_uncertain = execute_bayes_prediction_model(current_date, symbol, interval)

    # Console output
    print("\n=== ベイズモデルによる価格予測分析結果 ===")

    # File output
    with open(output_file, "a") as f:
        f.write("\n=== Bayesian Price Prediction Analysis ===\n")
        f.write("Purpose: Predict future price trends using probabilistic approach combining past price patterns and technical indicators\n\n")

        f.write("1. Basic Information:\n")
        f.write(f"Analysis Time: {current_date}\n")
        f.write(f"Target Symbol: {symbol}\n")
        f.write(f"Time Interval: {interval} minutes\n")
        f.write(f"Current Price: {current_price:.2f}\n\n")

        f.write("2. Prediction Results:\n")
        f.write(f"Predicted Price: {prediction:.2f}\n")
        if std_dev is not None:
            f.write(f"Prediction Standard Deviation: {std_dev:.2f}\n")
            f.write(f"95% Confidence Interval: {prediction - 1.96*std_dev:.2f} to {prediction + 1.96*std_dev:.2f}\n")
            f.write(f"Predicted Change Rate: {((prediction - current_price) / current_price * 100):.2f}%\n\n")

        f.write("3. Prediction Certainty:\n")
        if is_uncertain:
            f.write("Prediction Uncertainty: High\n")
            f.write("Note: Prediction reliability might be low\n")
        else:
            f.write("Prediction Uncertainty: Low\n")
            f.write("Note: Prediction reliability is relatively high\n")
        f.write("-" * 50 + "\n")

def main() -> None:
    current_date = "2025-01-15 00:00:00"
    symbol = "BTCUSDT"
    interval = 1440

    # Create output directory if it doesn't exist
    output_dir = os.path.join(PARENT_DIR, "predictions")
    os.makedirs(output_dir, exist_ok=True)

    # Create output file with timestamp
    timestamp = datetime.strptime(current_date, "%Y-%m-%d %H:%M:%S").strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"combined_predictions_{timestamp}.txt")

    # Write header to the file
    with open(output_file, "w") as f:
        f.write("Combined Predictions Report\n")
        f.write(f"Generated at: {timestamp}\n")
        f.write(f"Analysis Date: {current_date}\n")
        f.write("=" * 50 + "\n\n")

    # Execute all predictions and write to the same file
    execute_technical_prediction(output_file)
    execute_garch_price_distribution_analysis(current_date, 3, interval, output_file)
    execute_bayes_prediction(current_date, symbol, interval, output_file)

    print(f"\nAll predictions have been saved to: {output_file}")

if __name__ == "__main__":
    main()

import os
import sys
import argparse
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timedelta, timezone

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

def get_current_utc_datetime() -> str:
    """
    Get current UTC datetime in the format 'YYYY-MM-DD 00:00:00'
    """
    # Only take the date part and append 00:00:00
    return datetime.now(timezone.utc).strftime("%Y-%m-%d") + " 00:00:00"

def parse_date_argument(date_str: Optional[str] = None) -> str:
    """
    Parse date string argument and return datetime string in the format 'YYYY-MM-DD 00:00:00'
    If no date is provided, returns current UTC datetime with time set to 00:00:00
    """
    if not date_str:
        return get_current_utc_datetime()

    try:
        # Try to parse the input date
        parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
        # Always set time to 00:00:00
        return f"{date_str} 00:00:00"
    except ValueError as e:
        print(f"Error: Invalid date format. Please use YYYY-MM-DD format. Using current UTC date instead.")
        return get_current_utc_datetime()

def execute_technical_prediction(current_date: str, output_file: str) -> None:
    """
    Example main function demonstrating how to use the TransformerPredictionRollingModel.
    Records predictions to a text file.
    """
    print(":::start:::")
    # Convert string date to datetime object before adding timedelta
    current_date_dt = datetime.strptime(current_date, "%Y-%m-%d %H:%M:%S")
    prediction_date = current_date_dt + timedelta(days=3)
    model_target = ["ema", "macdhist", "roc", "mfi", "aroon"]
    model_id_list = ["rolling_v1", "rolling_v2", "rolling_v3", "rolling_v4", "rolling_v5"]
    model_list = []

    for model_id in model_id_list:
        model = transfomer_predictin_trend_setup(model_id, "BTCUSDT", 1440)
        model_list.append(model)

    print(f"current_date:{current_date},prediction_date:{prediction_date}")

    # Write technical predictions
    with open(output_file, "a") as f:
        f.write("\n=== テクニカル分析による予測結果 ===\n")
        f.write("目的: 複数の技術指標を用いて、価格トレンドの方向性を予測します\n")
        f.write("特徴: \n")
        f.write("・各技術指標に対して独立したモデルを使用\n")
        f.write("・上昇/下降確率を算出し、トレンドの強さを評価\n")
        f.write("・複数指標の組み合わせにより、より信頼性の高い予測を実現\n\n")

        f.write(f"分析日時: {current_date}\n")
        f.write(f"予測対象日時: {prediction_date}\n")
        f.write("-" * 50 + "\n\n")

        for i, model in enumerate(model_list):
            up_pred, down_pred, current_value = transfomer_predictin_trend(current_date, model, model_target[i])

            # Write to console and file
            print(f"対象技術指標: {model_target[i]}")
            print(f"上昇確率: {up_pred:.2f}%, 下降確率: {down_pred:.2f}%")

            f.write(f"対象技術指標: {model_target[i]}\n")
            f.write(f"上昇確率: {up_pred:.2f}%\n")
            f.write(f"下降確率: {down_pred:.2f}%\n")
            f.write(f"現在の指標値: {current_value:.2f}\n")
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
        f.write("\n=== GARCHモデルによる価格分布分析結果 ===\n")
        f.write("目的: 将来の価格変動の確率分布を予測し、現在の価格水準の評価と将来のボラティリティを推定\n")
        f.write("特徴:\n")
        f.write("・価格変動の非対称性を考慮した統計モデル\n")
        f.write("・将来の価格範囲を確率的に予測\n")
        f.write("・ボラティリティの変動を考慮した分析\n\n")

        f.write("1. 基本情報:\n")
        f.write(f"分析日時: {current_date_str}\n")
        f.write(f"現在価格: {current_price:,.2f}\n\n")

        f.write("2. 価格予測範囲 (信頼区間):\n")
        f.write(f"下限価格 (10%分位点): {lower_bound:,.2f}\n")
        f.write(f"中央価格 (50%分位点): {median_val:,.2f}\n")
        f.write(f"上限価格 (90%分位点): {upper_bound:,.2f}\n\n")

        f.write("3. 予測収益率範囲:\n")
        f.write(f"下限収益率 (10%分位点): {returns_lower_bound:.2%}\n")
        f.write(f"中央収益率 (50%分位点): {returns_median_val:.2%}\n")
        f.write(f"上限収益率 (90%分位点): {returns_upper_bound:.2%}\n\n")

        f.write("4. 価格水準の評価:\n")
        f.write(f"現在価格の評価: {judgement_result}\n")
        f.write("※ この評価は予測価格範囲における現在価格の位置を示します\n\n")

        f.write("5. ボラティリティ予測:\n")
        f.write(f"予測ボラティリティ: {volatility:.2%}\n")
        f.write("※ この値は将来の価格変動の大きさを年率換算で示します\n")
        f.write("-" * 50 + "\n")

def execute_bayes_prediction(current_date: str, symbol: str, interval: int, output_file: str):
    current_price = get_current_price(current_date, interval)
    prediction, std_dev, is_uncertain = execute_bayes_prediction_model(current_date, symbol, interval)

    # Console output
    print("\n=== ベイズモデルによる価格予測分析結果 ===")

    # File output
    with open(output_file, "a") as f:
        f.write("\n=== ベイズモデルによる価格予測分析結果 ===\n")
        f.write("目的: 過去の価格パターンと技術指標を組み合わせて、確率論的アプローチで将来の価格動向を予測\n")
        f.write("特徴:\n")
        f.write("・複数の技術指標を組み合わせた確率的予測\n")
        f.write("・予測の不確実性を考慮した信頼区間の提供\n")
        f.write("・過去のパターンに基づく客観的な分析\n\n")

        f.write("1. 基本情報:\n")
        f.write(f"分析日時: {current_date}\n")
        f.write(f"対象銘柄: {symbol}\n")
        f.write(f"時間間隔: {interval}分\n")
        f.write(f"現在価格: {current_price:.2f}\n\n")

        f.write("2. 予測結果:\n")
        f.write(f"予測価格: {prediction:.2f}\n")
        if std_dev is not None:
            f.write(f"予測の標準偏差: {std_dev:.2f}\n")
            f.write(f"95%信頼区間: {prediction - 1.96*std_dev:.2f} ～ {prediction + 1.96*std_dev:.2f}\n")
            f.write(f"予測変化率: {((prediction - current_price) / current_price * 100):.2f}%\n\n")

        f.write("3. 予測の確実性:\n")
        if is_uncertain:
            f.write("予測の不確実性: 高\n")
            f.write("※ 予測値の信頼性が低い可能性があります\n")
        else:
            f.write("予測の不確実性: 低\n")
            f.write("※ 予測値の信頼性は比較的高いと考えられます\n")
        f.write("-" * 50 + "\n")

def main() -> None:
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Execute technical prediction analysis')
    parser.add_argument('--date', type=str, help='Analysis date in YYYY-MM-DD format. If not provided, current UTC date will be used.')
    args = parser.parse_args()

    # Parse the date argument
    current_date = parse_date_argument(args.date)

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
        f.write("=== 総合予測分析レポート ===\n")
        f.write(f"生成日時: {timestamp}\n")
        f.write(f"分析基準日: {current_date}\n")
        f.write("=" * 50 + "\n\n")
        f.write("本レポートの特徴:\n")
        f.write("・3つの異なる分析手法による多角的な予測\n")
        f.write("・各手法の特徴を活かした総合的な分析\n")
        f.write("・予測の不確実性を考慮した信頼性の評価\n\n")

    # Execute all predictions and write to the same file
    execute_technical_prediction(current_date, output_file)
    execute_garch_price_distribution_analysis(current_date, 3, interval, output_file)
    execute_bayes_prediction(current_date, symbol, interval, output_file)

    print(f"\n予測結果は以下のファイルに保存されました: {output_file}")

if __name__ == "__main__":
    main()

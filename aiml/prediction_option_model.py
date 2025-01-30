import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict, Any, Union
import os
import sys

# Install arch package if needed: pip install arch
from arch import arch_model

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the path of the parent directory to sys.path
sys.path.append(parent_dir)

from common.trading_logger import TradingLogger
from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import MARKET_DATA_TECH, TIME_SERIES_PERIOD, PRED_TYPE_LONG, PRED_TYPE_SHORT
from common.utils import get_config_model

from aiml.garch_price_distribution_model import garch_price_distribution_analysis, judge_current_price
from aiml.prediction_manager import PredictionManager


class MongoDataLoaderSingleton:
    """
    Singleton class for MongoDataLoader to ensure only one instance exists.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDataLoaderSingleton, cls).__new__(cls)
            cls._instance._loader = MongoDataLoader()
        return cls._instance

    def __getattr__(self, name):
        return getattr(self._instance._loader, name)


class PredictionManagerSingleton:
    """
    Singleton class for PredictionManager to ensure only one instance exists.
    """
    _instance = None

    def __new__(cls, model_name="rolling_v1"):
        if cls._instance is None:
            cls._instance = super(PredictionManagerSingleton, cls).__new__(cls)
            cls._instance._manager = PredictionManager()
            config = get_config_model("MODEL_SHORT_TERM", model_name)
            cls._instance._manager.initialize_model(model_name, config)
            cls._instance._manager.load_model()
        return cls._instance

    def __getattr__(self, name):
        return getattr(self._instance._manager, name)


# Global instances
_mongo_data_loader = MongoDataLoaderSingleton()
_prediction_manager = PredictionManagerSingleton()
_logger = TradingLogger()


def transformer_trend_prediction(current_date: str, symbol: str = "BTCUSDT", interval: int = 1440) -> Any:
    """
    Predicts trend using a transformer model.

    Parameters
    ----------
    current_date : str
        Current date for prediction.
    symbol : str, default "BTCUSDT"
        Trading symbol.
    interval : int, default 1440
        Time interval in minutes.

    Returns
    -------
    Any
        The prediction result from the transformer model.
    """
    global _mongo_data_loader, _prediction_manager, _logger

    df = _mongo_data_loader.load_data_from_point_date(
        current_date, TIME_SERIES_PERIOD, MARKET_DATA_TECH, symbol, interval
    )
    target_df = _prediction_manager.create_time_series_data(df)
    return _prediction_manager.predict_model(target_df, probability=True)


def decide_entry_exit(
    current_price: float,
    q10_price: float,
    q50_price: float,
    q90_price: float,
    r_q10: float,
    r_q50: float,
    r_q90: float,
    predicted_direction: int,
    k: float = 0.01,
    kk: float = 0.005
) -> Union[tuple[float, float], tuple[None, None]]:
    """
    Determines simple entry and exit prices based on price quantiles (q10, q50, q90),
    return quantiles (r_q10, r_q50, r_q90), and predicted direction (Long/Short).

    Parameters
    ----------
    current_price : float
        Current market price.
    q10_price : float
        Price at the 10th quantile.
    q50_price : float
        Price at the 50th quantile.
    q90_price : float
        Price at the 90th quantile.
    r_q10 : float
        Return at the 10th quantile.
    r_q50 : float
        Return at the 50th quantile.
    r_q90 : float
        Return at the 90th quantile.
    predicted_direction : int
        Predicted direction (PRED_TYPE_LONG or PRED_TYPE_SHORT).
    k : float, default 0.01
        Exit price adjustment factor.
    kk : float, default 0.005
        Entry price adjustment factor.

    Returns
    -------
    tuple[float, float] or tuple[None, None]
        Entry and exit prices, or (None, None) if no entry/exit signal is generated.
    """
    entry_price = None
    exit_price = None

    if predicted_direction == PRED_TYPE_LONG:
        # Long target
        if current_price <= q10_price:
            entry_price = current_price
            exit_price = q90_price * (1 - k)
        elif q10_price < current_price < q90_price:
            entry_price = current_price * (1 - kk)
            exit_price = q90_price * (1 - k)

    elif predicted_direction == PRED_TYPE_SHORT:
        # Short target
        if current_price >= q90_price:
            entry_price = current_price
            exit_price = q10_price * (1 + k)
        elif q10_price < current_price < q90_price:
            entry_price = current_price * (1 + kk)
            exit_price = q10_price * (1 + k)

    return entry_price, exit_price


def garch_should_entry(
    quantiles: List[float],
    current_date: str,
    current_price: float = None,
    symbol: str = "BTCUSDT",
    day_of_interval: int = 1440,
    horizon: int = 1
) -> tuple[str, Dict[str, Any]]:
    """
    Determines entry/exit signals based on GARCH price distribution.

    Parameters
    ----------
    quantiles : List[float]
        List of quantiles for GARCH analysis.
    current_date : str
        Current date for analysis.
    current_price : float, optional
        Current market price.
    symbol : str, default "BTCUSDT"
        Trading symbol.
    day_of_interval : int, default 1440
        Time interval in minutes.
    horizon : int, default 1
        GARCH forecast horizon.

    Returns
    -------
    tuple[str, Dict[str, Any]]
        Current judgement ("POTENTIALLY_UNDERVALUED", "POTENTIALLY_FAIR", "POTENTIALLY_OVERVALUED")
        and GARCH results dict.
    """
    result = garch_price_distribution_analysis(
        current_date,
        quantiles=quantiles,
        horizon=horizon
    )

    # 万が一 result が None なら、呼び出し側で None として扱う
    if result is None or ("forecast_price_quantiles" not in result) or (result["forecast_price_quantiles"] is None):
        return "UNKNOWN", {}

    # 例: For quantiles=[0.25, 0.5, 0.75]
    lower = result["forecast_price_quantiles"][quantiles[0]]  # e.g. q25
    upper = result["forecast_price_quantiles"][quantiles[2]]  # e.g. q75

    current_judgement = judge_current_price(
        current_price,
        lower_bound=lower,
        upper_bound=upper
    )
    return current_judgement, result


def garch_forecast_price(current_date: str, quantiles: List[float], horizon: int = 1) -> float:
    result = garch_price_distribution_analysis(
        current_date,
        quantiles=quantiles,
        horizon=horizon
    )
    return result


def get_current_price(current_date: str, symbol: str = "BTCUSDT", interval: int = 1440) -> float:
    """
    Retrieves the current price for a given date, symbol, and interval.

    Parameters
    ----------
    current_date : str
        The date for which to retrieve the current price.
    symbol : str, default "BTCUSDT"
        The trading symbol.
    interval : int, default 1440
        The time interval in minutes.

    Returns
    -------
    float
        The current closing price.
    """
    df = _mongo_data_loader.load_data_from_point_date(
        current_date, TIME_SERIES_PERIOD, MARKET_DATA_TECH, symbol, interval
    )
    return df.iloc[-1]["close"]


def main():
    """
    Main function for testing the prediction model.
    """
    # パラメータ設定
    horizon = 3
    quantiles = [0.1, 0.5, 0.9]
    current_date = "2024-12-15 04:00:00"

    # 現在価格の取得
    current_price = get_current_price(current_date)
    print("current_price:", current_price)


    result = garch_forecast_price(current_date, quantiles, horizon)
    
    print("result:", result)
    # GARCH 判定 (例: 需給バランス判断など)
    current_judgement, result = garch_should_entry(
        quantiles,
        current_date,
        current_price=current_price,
        horizon=horizon
    )
    print("judgement:", current_judgement)


if __name__ == "__main__":
    main()



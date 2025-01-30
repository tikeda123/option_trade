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

class MongoDataLoaderSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDataLoaderSingleton, cls).__new__(cls)
            cls._instance._loader = MongoDataLoader()
        return cls._instance

    def __getattr__(self, name):
        return getattr(self._instance._loader, name)

class PredictionManagerSingleton:
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

# インスタンスをグローバルに保持
_mongo_data_loader = MongoDataLoaderSingleton()
_prediction_manager = PredictionManagerSingleton()
_logger = TradingLogger() # Loggerはシングルトン化するか、毎回インスタンス化するかで判断してください。ここでは毎回インスタンス化としています。

def transformer_trend_prediction(current_date: str, symbol="BTCUSDT", interval=1440):
    global _mongo_data_loader, _prediction_manager, _logger

    df = _mongo_data_loader.load_data_from_point_date(current_date, TIME_SERIES_PERIOD, MARKET_DATA_TECH, symbol, interval)
    target_df = _prediction_manager.create_time_series_data(df)
    return _prediction_manager.predict_model(target_df, probability=True)


def garch_price_distribution_analysis(current_date: str, price_col="close",symbol:str="BTCUSDT",interval_min:int=1440):
    """
    GARCHモデルを用いて価格の分布を分析する。
    """

    data_loader = MongoDataLoader()
    df = data_loader.load_data_from_datetime_period(
        start_date="2020-01-01 00:00:00",
        end_date=current_date,
        coll_type=MARKET_DATA_TECH,
        symbol="BTCUSDT",
        interval=interval_min
    )

        # （念のため日付ソート & インデックス化）
    df = df.sort_values("date").reset_index(drop=True)
    df.set_index("date", inplace=True)

    model = GarchPriceDistributionModel(
        dist="t",
        quantiles=[0.25, 0.5, 0.75]
    )
    #直近を含まない200日間のデータを取り出す
    df_window = df.iloc[-200:-1]

    returns = model.prepare_data(df_window, price_col="close")
    model.fit_model(returns)
    result =  model.forecast_distribution(horizon=1)
    return result


def decide_entry_exit(
    current_price: float,
    q10_price: float,
    q50_price: float,
    q90_price: float,
    r_q10: float,
    r_q50: float,
    r_q90: float,
    predicted_direction:int,
    k:float=0.01,
    kk:float=0.005
):
    """
    GARCHモデル等で得られた価格の分位点 (q10, q50, q90) と
    リターンの分位点 (r_q10, r_q50, r_q90) から、
    予測方向 (Long / Short) に基づいてシンプルなエントリー価格・エグジット価格を決定するサンプル関数。

    Parameters
    ----------
    current_price : float
        現在のBTC価格
    q10_price : float
        予測価格のQuantile 0.1 (下位10%)
    q50_price : float
        予測価格のQuantile 0.5 (中央値)
    q90_price : float
        予測価格のQuantile 0.9 (上位10%)
    r_q10 : float
        予測リターンのQuantile 0.1
    r_q50 : float
        予測リターンのQuantile 0.5
    r_q90 : float
        予測リターンのQuantile 0.9
    predicted_direction : str
        "Long" または "Short" （二分類モデルによる予測方向）

    Returns
    -------
    entry_price : float or None
        推奨エントリー価格 (取引しない場合は None)
    exit_price : float or None
        推奨エグジット価格 (取引しない場合は None)

    Notes
    -----
    - このサンプルではストップロス（損切り）を明示的に設定していない。
      実際の運用では、Q10やQ90などの価格を参考にストップロスラインを決めるなど
      リスク管理を必ず組み込む必要がある。
    - r_q10, r_q50, r_q90 （リターンの分位点）は本サンプルでは表示用・将来的な拡張用。
      ロジック上では必ずしも使用しなくてもよい。
    - current_price と各分位点の相対関係だけで単純に判断しているため、
      実際の相場状況やトレンドが乖離している場合は要注意。
    """

    entry_price = None
    exit_price = None

    if predicted_direction == PRED_TYPE_LONG:
        # Long狙い
        if current_price <= q10_price:
            # かなり割安 -> そのまま買い
            entry_price = current_price
            exit_price = q90_price*(1-k)
        elif q10_price < current_price < q90_price:
            # 中間帯にある -> 現在価格で買い (またはQ50まで引きつける等の応用も)
            entry_price = current_price*(1-kk)
            exit_price = q90_price*(1-k)
        else:
            # current_price >= q90_price
            # 既に大きく上げている -> エントリーは見送り
            entry_price = None
            exit_price = None

    elif predicted_direction == PRED_TYPE_SHORT:
        # Short狙い
        if current_price >= q90_price:
            # かなり割高 -> そのままショート
            entry_price = current_price
            exit_price = q10_price*(1+k)
        elif q10_price < current_price < q90_price:
            # 中間帯にある -> 現在価格でショート
            entry_price = current_price*(1+kk)
            exit_price = q10_price*(1+k)
        else:
            # current_price <= q10_price
            # 既に大きく下げている -> エントリーは見送り
            entry_price = None
            exit_price = None
    else:
        # Long/Short 以外の文字列が入った場合は対象外
        entry_price = None
        exit_price = None

    return entry_price, exit_price

def garch_should_entry(current_date:str,current_price:float=None,symbol="BTCUSDT",day_of_interval=1440):
    """
    エントリー価格・エグジット価格を決定するモデル
    """

    result = garch_price_distribution_analysis(current_date, price_col="close",symbol="BTCUSDT",interval_min=day_of_interval)

    current_judgement = judge_current_price(current_price,lower_bound=result["forecast_price_quantiles"][0.25],upper_bound=result["forecast_price_quantiles"][0.75])
    #print(current_judgement)


    if current_judgement == "POTENTIALLY_OVERVALUED":
        exit_price = result["forecast_price_quantiles"][0.75]
        return current_judgement,exit_price
    elif current_judgement == "POTENTIALLY_UNDERVALUED":
        exit_price = result["forecast_price_quantiles"][0.25]
        return current_judgement,exit_price
    else:
        return current_judgement,None


def prediction_option_model(pred_type:int,current_date:str,current_price:float=None,symbol="BTCUSDT",day_of_interval=1440):
    """
    予測方向に基づいてエントリー価格・エグジット価格を決定するモデル
    """
    #current_dateは”YYYY-MM-DD 00:00:00”の形式で入力すること
    #もし、時刻が入っている場合は、00:00:00に変更すること
    # 時刻部分を00:00:00に変換
    #if " " in current_date:
    #    date_part = current_date.split(" ")[0]
    #    current_date = f"{date_part} 00:00:00"


    result = garch_price_distribution_analysis(current_date, price_col="close",symbol="BTCUSDT",interval_min=day_of_interval)

    if pred_type == PRED_TYPE_LONG:
        exit_price = result["forecast_price_quantiles"][0.75]
        print(":::exit_price:::",exit_price)
        return None,exit_price
    elif pred_type == PRED_TYPE_SHORT:
        exit_price = result["forecast_price_quantiles"][0.25]
        print(":::exit_price:::",exit_price)
        return None,exit_price
    else:
        return None,None


    entry_price, exit_price = decide_entry_exit(
        current_price,
        q10_price=result["forecast_price_quantiles"][0.25],
        q50_price=result["forecast_price_quantiles"][0.5],
        q90_price=result["forecast_price_quantiles"][0.75],
        r_q10=result["forecast_returns_quantiles"][0.25],
        r_q50=result["forecast_returns_quantiles"][0.5],
        r_q90=result["forecast_returns_quantiles"][0.75],
        predicted_direction=pred_type
    )
    return entry_price, exit_price


def main():
    current_date = "2024-8-01 04:00:00"
    entry_price, exit_price = prediction_option_model(current_date)
    print(entry_price, exit_price)

    current_date = "2024-11-10 12:00:00"
    entry_price, exit_price = prediction_option_model(current_date)
    print(entry_price, exit_price)

    current_date = "2024-12-20 00:00:00"
    entry_price, exit_price = prediction_option_model(current_date)
    print(entry_price, exit_price)


if __name__ == "__main__":
    main()


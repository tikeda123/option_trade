#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# `arch` パッケージが必要
# pip install arch
from arch import arch_model
from datetime import datetime

# -----------------------------------------------------
# 質問文にある通りのモジュールをインポート (環境依存)
# MongoDataLoader, TradingLogger, get_config, MARKET_DATA_TECH
# -----------------------------------------------------
# 下記のようにローカルの構成に合わせて import
# すでに同じ階層/親ディレクトリ等にあることを想定

# カレントディレクトリ取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.trading_logger import TradingLogger
from mongodb.data_loader_mongo import MongoDataLoader
from common.utils import get_config
from common.constants import MARKET_DATA_TECH

def fit_garch_and_forecast(
    returns_series: pd.Series,
    last_price: float,
    p=1,
    q=1,
    dist="normal",
    mean="constant",
    vol="GARCH",
    horizon=1,
    quantiles=[0.1, 0.5, 0.9]
) -> dict:
    """
    与えられたリターン系列 (returns_series) に GARCH(p, q) をフィットし、
    horizon=1 (1日先) の分位点予測 (quantiles) を返す。

    Parameters
    ----------
    returns_series : pd.Series
        対数リターン系列
    last_price : float
        モデル学習データの最後の時点の価格 (予測の基点価格)
    p, q : int
        GARCH(p, q) のパラメータ
    dist : str
        分布の指定 ('normal', 't' など)
    mean : str
        平均モデル ('constant', 'AR', 'HAR' 等)
    vol : str
        ボラティリティモデル ('GARCH', 'EGARCH', 'ARCH' 等)
    horizon : int
        何ステップ先を予測するか。ここでは1を想定
    quantiles : list
        分位点リスト

    Returns
    -------
    dict
        {
            'forecast_returns_quantiles': {0.1: x, 0.5: x, 0.9: x},
            'forecast_price_quantiles':   {0.1: x, 0.5: x, 0.9: x}
        }
    """
    # ----------------------------------------
    # GARCH モデルの学習 (フィット)
    # ----------------------------------------
    am = arch_model(
        returns_series,
        mean=mean,
        vol=vol,
        p=p,
        q=q,
        dist=dist
    )
    res = am.fit(disp="off")  # disp='off' でフィットのログ非表示

    # ----------------------------------------
    # 1ステップ先の予測
    # ----------------------------------------
    forecast_res = res.forecast(horizon=horizon)

    # 平均・分散(ボラティリティ)の取得
    mean_forecast = forecast_res.mean[f"h.{horizon}"].iloc[-1]
    var_forecast = forecast_res.variance[f"h.{horizon}"].iloc[-1]
    std_forecast = np.sqrt(var_forecast)

    # 正規分布と仮定して分位点を計算
    from scipy.stats import norm

    ret_quantiles = {}
    price_quantiles = {}
    for q_ in quantiles:
        z_q = norm.ppf(q_)
        ret_q = mean_forecast + std_forecast * z_q  # リターンの分位点
        ret_quantiles[q_] = ret_q

        # 価格の分位点 (対数リターン → 価格)
        # P_{t+1} = P_t * exp(r_{t+1})
        price_q = last_price * np.exp(ret_q)
        price_quantiles[q_] = price_q

    return {
        "forecast_returns_quantiles": ret_quantiles,
        "forecast_price_quantiles": price_quantiles
    }


def main():
    """
    メイン処理:
      - MongoDataLoader で 2020-01-01〜2024-12-01 のデータを取得
      - 2024-08-01 までを学習期間・2024-08-02〜2024-12-01 を検証期間とし、ウォークフォワードで 1日先予測
      - 実際の価格との比較グラフを描画
      - 数値的評価指標(MAE, RMSE, MAPE)を算出して表示
    """

    # =====================================================
    # 1) データ取得 (Mongo からロード)
    # =====================================================
    data_loader = MongoDataLoader()
    df = data_loader.load_data_from_datetime_period(
        start_date="2020-01-01 00:00:00",
        end_date="2024-12-01 00:00:00",
        coll_type=MARKET_DATA_TECH
    )
    # df: 少なくとも 'date' 列, 'close' 列が含まれる想定

    # 日付を index に (質問文の仕様)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    # =====================================================
    # 2) 学習・検証期間の設定
    # =====================================================
    train_end_date = "2024-08-01"
    # 学習データ: 2024-08-01 まで
    df_train = df.loc[:train_end_date].copy()
    # 検証データ: 2024-08-02 以降 (厳密には train_end_date 当日を含めず翌日から)
    df_test = df.loc[train_end_date:].iloc[1:].copy()

    # =====================================================
    # 3) ウォークフォワード (日ごとに再学習→翌日を予測)
    # =====================================================
    predictions = []
    test_dates = df_test.index.tolist()  # 検証期間の日付一覧

    for i in range(len(test_dates)):
        current_date = test_dates[i]  # 当日(予測したい日)

        # ===================================
        # 3.1) 学習データの動的拡張
        #      ＝ train_end_date 〜 current_dateの「前日」までを学習に使う
        # ===================================
        # (i=0 のときは 2024-08-02 の価格予測になるので、2024-08-01 までを学習)
        # i>0 ならウォークフォワードで 〜(current_date の前日) まで学習に含める
        prev_day = test_dates[i] - pd.Timedelta(days=1)
        df_train_dynamic = df.loc[:prev_day].copy()

        # 対数リターンを計算
        df_train_dynamic["log_return"] = np.log(df_train_dynamic["close"] / df_train_dynamic["close"].shift(1))
        df_train_dynamic.dropna(subset=["log_return"], inplace=True)

        # GARCH モデルに渡すための最終価格
        if len(df_train_dynamic) == 0:
            # 万が一、データが無い場合はスキップ
            continue
        last_price = df_train_dynamic["close"].iloc[-1]

        # ===================================
        # 3.2) GARCH(1,1) で 1日先の価格分布予想
        # ===================================
        forecast_dict = fit_garch_and_forecast(
            returns_series=df_train_dynamic["log_return"],
            last_price=last_price,
            p=1, q=1,
            dist="normal",
            mean="constant",
            vol="GARCH",
            horizon=1,
            quantiles=[0.1, 0.5, 0.9]
        )

        pred_10 = forecast_dict["forecast_price_quantiles"][0.1]
        pred_50 = forecast_dict["forecast_price_quantiles"][0.5]
        pred_90 = forecast_dict["forecast_price_quantiles"][0.9]

        predictions.append({
            "date": current_date,
            "pred_10": pred_10,
            "pred_50": pred_50,
            "pred_90": pred_90
        })

    # =====================================================
    # 4) 予測結果を DataFrame 化し、実際の価格とマージ
    # =====================================================
    df_pred = pd.DataFrame(predictions).set_index("date")
    df_plot = df_test.merge(df_pred, left_index=True, right_index=True, how="left")

    # =====================================================
    # 5) グラフ描画
    # =====================================================
    plt.figure(figsize=(12, 6))
    plt.plot(df_plot.index, df_plot["close"], label="Actual Price", color="blue")
    plt.plot(df_plot.index, df_plot["pred_10"], label="10% quantile", color="red", linestyle="--")
    plt.plot(df_plot.index, df_plot["pred_50"], label="50% quantile", color="green", linestyle="--")
    plt.plot(df_plot.index, df_plot["pred_90"], label="90% quantile", color="orange", linestyle="--")

    # 10～90% の範囲を塗りつぶす
    plt.fill_between(
        df_plot.index,
        df_plot["pred_10"],
        df_plot["pred_90"],
        color="grey",
        alpha=0.2,
        label="10%-90% Range"
    )

    plt.title("Comparison: Actual Price vs. GARCH Forecast Distribution (1-day ahead)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # =====================================================
    # 6) 数値評価 (MAE, RMSE, MAPE など)
    # =====================================================
    # ここでは予測の代表値として "pred_50" (中央値予測) を使用。
    df_plot["error_50"] = df_plot["pred_50"] - df_plot["close"]

    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(df_plot["error_50"]))

    # MSE (Mean Squared Error)
    mse = np.mean(df_plot["error_50"] ** 2)
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mse)

    # MAPE (Mean Absolute Percentage Error) [%, 0除算対策で実値 close が 0 でないもののみ]
    df_nonzero = df_plot[df_plot["close"] != 0].copy()
    mape = np.mean(np.abs((df_nonzero["pred_50"] - df_nonzero["close"]) / df_nonzero["close"])) * 100

    print("===== Forecast Accuracy Metrics (using pred_50) =====")
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"MAPE : {mape:.3f} %")



if __name__ == "__main__":
    main()
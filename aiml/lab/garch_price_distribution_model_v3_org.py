#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from arch import arch_model
from datetime import datetime

# -----------------------------------------------------
# あなたの環境依存の import (MongoDataLoader など)
# -----------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.trading_logger import TradingLogger
from mongodb.data_loader_mongo import MongoDataLoader
from common.utils import get_config
from common.constants import MARKET_DATA_TECH

# -----------------------------------------------------
# 1) GARCH/EGARCH/GJR-GARCH 用の関数（exogenous 対応）
# -----------------------------------------------------
def fit_garch_and_forecast_with_exog(
    returns_series: pd.Series,
    last_price: float,
    exog: pd.DataFrame = None,
    p=1,
    q=1,
    o=0,                  # GJR-GARCH 用 (GJR: o=1 など)
    dist="normal",
    mean="constant",
    vol="GARCH",          # 'GARCH', 'EGARCH' など
    horizon=1,
    quantiles=[0.1, 0.5, 0.9]
) -> dict:
    """
    外生変数(exog)を平均方程式に含めた GARCH (vol=GARCH), EGARCH, GJR-GARCH などで1日先予測。

    Parameters
    ----------
    returns_series : pd.Series
        対数リターン系列
    last_price : float
        モデル学習データの最後の時点の価格 (予測の基点価格)
    exog : pd.DataFrame
        外生変数 (rsi, macd, roc, atr 等) を含む DataFrame
        index は returns_series に揃えること。
    p, q, o : int
        GARCH(p,q) / GJR-GARCH(p,o,q) / EGARCH(p,o,q) の次数
    dist : str
        分布の指定 ('normal', 't' など)
    mean : str
        平均モデル ('constant', 'AR', 'HAR' 等)
    vol : str
        'GARCH', 'EGARCH', 'ARCH', ...
    horizon : int
        何ステップ先を予測するか (本例は1を想定)
    quantiles : list
        分位点リスト (例: [0.1, 0.5, 0.9])

    Returns
    -------
    dict
        {
            'forecast_returns_quantiles': {0.1: x, 0.5: x, 0.9: x},
            'forecast_price_quantiles':   {0.1: x, 0.5: x, 0.9: x}
        }
    """
    # arch_model の引数 x に外生変数を渡す
    am = arch_model(
        y=returns_series,
        x=exog,                # exogenous variables
        mean=mean,
        vol=vol,
        p=p,
        o=o,                   # GJRの場合 o=1 とか
        q=q,
        dist=dist
    )
    res = am.fit(disp="off")

    # ----------------------------------------
    # 1ステップ先の予測
    # ----------------------------------------
    forecast_res = res.forecast(horizon=horizon)

    # 平均・分散(ボラティリティ)の取得
    mean_forecast = forecast_res.mean[f"h.{horizon}"].iloc[-1]
    var_forecast = forecast_res.variance[f"h.{horizon}"].iloc[-1]
    std_forecast = np.sqrt(var_forecast)

    # 正規分布と仮定して分位点を計算 (dist='normal' 前提)
    from scipy.stats import norm

    ret_quantiles = {}
    price_quantiles = {}
    for q_ in quantiles:
        z_q = norm.ppf(q_)
        ret_q = mean_forecast + std_forecast * z_q  # リターンの分位点
        ret_quantiles[q_] = ret_q

        # 価格の分位点 (対数リターン → 価格)
        price_q = last_price * np.exp(ret_q)
        price_quantiles[q_] = price_q

    return {
        "forecast_returns_quantiles": ret_quantiles,
        "forecast_price_quantiles": price_quantiles
    }

def main():
    """
    メイン処理:
      - 1) MongoDataLoader で 2020-01-01〜2024-12-01 のデータを取得
      - 2) RSI, MACD, ROC, ATR などが df に既にある前提
      - 3) ウォークフォワードで GARCH or EGARCH or GJR-GARCH を使って 1日先予測
      - 4) 予測と実際の比較
    """
    # 1) データ取得 (Mongo からロード)
    data_loader = MongoDataLoader()
    df = data_loader.load_data_from_datetime_period(
        start_date="2020-01-01 00:00:00",
        end_date="2024-12-01 00:00:00",
        coll_type=MARKET_DATA_TECH
    )
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    # ここで df に 'rsi', 'macd', 'roc', 'atr' 等が既に含まれている想定
    # 無い場合は、別途計算して df に列を追加しておく必要があります。

    # 2) 学習・検証期間の設定
    train_end_date = "2024-10-01"
    df_train = df.loc[:train_end_date].copy()
    df_test = df.loc[train_end_date:].iloc[1:].copy()

    # 3) ウォークフォワードで予測
    predictions = []
    test_dates = df_test.index.tolist()

    # -- モデル設定 (ここを切り替えて EGARCH, GJR-GARCH を試す) --
    vol_type = "GARCH"  # 例: 'GARCH' / 'EGARCH'
    p, o, q = 1, 1, 1    # 例: GJR-GARCH -> vol='GARCH' + o=1 など

    for i in range(len(test_dates)):
        current_date = test_dates[i]  # 当日(予測したい日)
        prev_day = current_date - pd.Timedelta(days=1)
        df_train_dynamic = df.loc[:prev_day].copy()

        # 対数リターンを計算
        df_train_dynamic["log_return"] = np.log(df_train_dynamic["close"] / df_train_dynamic["close"].shift(1))
        df_train_dynamic.dropna(subset=["log_return"], inplace=True)
        if len(df_train_dynamic) == 0:
            continue

        last_price = df_train_dynamic["close"].iloc[-1]

        # 外生変数: 学習用 (RSI, MACD, ROC, ATR)
        # returns_series と同じ日時インデックスである必要がある
        # NaN があれば落とす or 補間する
        exog_train = df_train_dynamic[["rsi", "macdhist", "roc", "atr"]].copy()
        exog_train.dropna(inplace=True)

        # リターンの方も exog_train にインデックスを合わせる
        common_index = exog_train.index.intersection(df_train_dynamic.index)
        returns_series = df_train_dynamic.loc[common_index, "log_return"]
        exog_train = exog_train.loc[common_index]

        # fit & forecast
        forecast_dict = fit_garch_and_forecast_with_exog(
            returns_series=returns_series,
            last_price=last_price,
            exog=exog_train,
            p=p,
            o=o,
            q=q,
            vol=vol_type,          # EGARCH / GARCH
            dist="normal",
            mean="constant",
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

    # 4) 予測結果を DataFrame 化して可視化・評価
    df_pred = pd.DataFrame(predictions).set_index("date")
    df_plot = df_test.merge(df_pred, left_index=True, right_index=True, how="left")

    # グラフ描画
    plt.figure(figsize=(12, 6))
    plt.plot(df_plot.index, df_plot["close"], label="Actual Price", color="blue")
    plt.plot(df_plot.index, df_plot["pred_10"], label="10% quantile", color="red", linestyle="--")
    plt.plot(df_plot.index, df_plot["pred_50"], label="50% quantile", color="green", linestyle="--")
    plt.plot(df_plot.index, df_plot["pred_90"], label="90% quantile", color="orange", linestyle="--")
    plt.fill_between(
        df_plot.index,
        df_plot["pred_10"],
        df_plot["pred_90"],
        color="grey",
        alpha=0.2,
        label="10%-90% Range"
    )
    plt.title(f"Comparison: Actual Price vs. {vol_type} Forecast (with exog) (1-day ahead)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 評価指標
    df_plot["error_50"] = df_plot["pred_50"] - df_plot["close"]
    mae = np.mean(np.abs(df_plot["error_50"]))
    mse = np.mean(df_plot["error_50"] ** 2)
    rmse = np.sqrt(mse)
    df_nonzero = df_plot[df_plot["close"] != 0].copy()
    mape = np.mean(np.abs(df_nonzero["error_50"] / df_nonzero["close"])) * 100

    print(f"===== {vol_type} Forecast Accuracy (pred_50) =====")
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"MAPE : {mape:.3f} %")

    # =====================================================
    # 7) シンプルなベンチマーク比較 (ナイーブ予測)
    # =====================================================
    df_plot["benchmark_naive"] = df_plot["close"].shift(1)
    df_bench = df_plot.dropna(subset=["benchmark_naive"]).copy()
    df_bench["error_bench"] = df_bench["benchmark_naive"] - df_bench["close"]
    bench_mae = np.mean(np.abs(df_bench["error_bench"]))
    bench_mse = np.mean(df_bench["error_bench"] ** 2)
    bench_rmse = np.sqrt(bench_mse)

    df_bench_nonzero = df_bench[df_bench["close"] != 0].copy()
    bench_mape = np.mean(
        np.abs(
            (df_bench_nonzero["benchmark_naive"] - df_bench_nonzero["close"])
            / df_bench_nonzero["close"]
        )
    ) * 100

    print("\n===== Naive Benchmark (close[t] = close[t-1]) =====")
    print(f"MAE  : {bench_mae:.3f}")
    print(f"RMSE : {bench_rmse:.3f}")
    print(f"MAPE : {bench_mape:.3f} %")


if __name__ == "__main__":
    main()

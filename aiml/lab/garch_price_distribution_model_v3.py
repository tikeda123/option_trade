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
    vol_model="GARCH",
    p=1,
    o=0,  # GJR, EGARCHなどでレバレッジ項が必要な場合用 (デフォルト0)
    q=1,
    dist="normal",
    mean="constant",
    # HARCH の場合、p に [1,5,22] のようにリストを渡す
    # FIGARCH の場合、p, q は (1,1) などでOK (dは内部で推定)
    horizon=1,
    quantiles=[0.1, 0.5, 0.9]
) -> dict:
    """
    与えられたリターン系列 (returns_series) に 指定のボラティリティモデル (vol_model) をフィットし、
    horizon=1 (1日先) の分位点予測 (quantiles) を返す。

    Parameters
    ----------
    returns_series : pd.Series
        対数リターン系列
    last_price : float
        モデル学習データの最後の時点の価格 (予測の基点価格)
    vol_model : str
        ボラティリティモデル名 ('GARCH', 'EGARCH', 'GJR', 'HARCH', 'FIGARCH' など)
    p, o, q : int
        それぞれのモデルに応じた次数 (GJR, EGARCH の場合は o がレバレッジ項の次数)
        HARCH の場合、p はリスト [1,5,22] などを渡しても良い
    dist : str
        分布の指定 ('normal', 't' など)
    mean : str
        平均モデル ('constant', 'AR', 'HAR' 等)
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
    # arch_model の vol 引数に任意のモデルを指定
    #   EGARCH -> 'EGARCH'
    #   GJR-GARCH -> 'GJR'
    #   HARCH -> 'HARCH' (p は [1,5,22] のようにリストで指定)
    #   FIGARCH -> 'FIGARCH'
    # ----------------------------------------
    am = arch_model(
        returns_series,
        mean=mean,
        vol=vol_model,  # <= ここでモデルを指定
        p=p,
        o=o,            # GJR, EGARCH などレバレッジ項が必要なモデルに対応
        q=q,
        dist=dist
    )
    # フィット
    res = am.fit(disp="off")  # disp='off' でフィットのログ非表示

    # ----------------------------------------
    # 1ステップ先の予測
    # ----------------------------------------
    forecast_res = res.forecast(horizon=horizon)

    # 平均・分散(ボラティリティ)の取得
    mean_forecast = forecast_res.mean[f"h.{horizon}"].iloc[-1]
    var_forecast = forecast_res.variance[f"h.{horizon}"].iloc[-1]
    std_forecast = np.sqrt(var_forecast)

    # 正規分布(あるいは dist='t' 等)で近似し分位点を計算
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
      - MongoDataLoader で 2020-01-01〜2024-12-01 のデータを取得
      - 2024-08-01 までを学習期間・2024-08-02〜2024-12-01 を検証期間とし、ウォークフォワードで 1日先予測
      - 実際の価格との比較グラフを描画
      - 数値的評価指標(MAE, RMSE, MAPE)を算出して表示
      - シンプルなベンチマーク(ナイーブ予測)とも比較
    """
    set_interval = 4320
    day_of_interval = int(set_interval / 1440)

    # =====================================================
    # 1) データ取得 (Mongo からロード)
    # =====================================================
    data_loader = MongoDataLoader()
    df = data_loader.load_data_from_datetime_period(
        start_date="2020-01-01 00:00:00",
        end_date="2025-01-05 00:00:00",
        coll_type=MARKET_DATA_TECH,
        symbol="BTCUSDT",
        interval=set_interval
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
    # 検証データ: 2024-08-02 以降
    df_test = df.loc[train_end_date:].iloc[1:].copy()

    # =====================================================
    # 3) ウォークフォワード (日ごとに再学習→翌日を予測)
    # =====================================================
    predictions = []
    test_dates = df_test.index.tolist()  # 検証期間の日付一覧

    # ここで vol_model を指定(例: 'EGARCH', 'GJR', 'HARCH', 'FIGARCH')
    #====================================================
    # ※ 実運用では、外部設定などで切り替えることを推奨
    #====================================================
    vol_model_choice = "GARCH"  # 例: 'EGARCH' に変更してみる

    for i in range(len(test_dates)):
        current_date = test_dates[i]

        prev_day = test_dates[i] - pd.Timedelta(days=1)
        df_train_dynamic = df.loc[:prev_day].copy()

        # 対数リターン
        df_train_dynamic["log_return"] = np.log(
            df_train_dynamic["close"] / df_train_dynamic["close"].shift(1)
        )
        df_train_dynamic.dropna(subset=["log_return"], inplace=True)

        if len(df_train_dynamic) == 0:
            # 学習データが無ければスキップ
            continue

        last_price = df_train_dynamic["close"].iloc[-1]

        #=======================================
        # 3.2) 選択したモデル (例: EGARCH) で予測
        #=======================================
        forecast_dict = fit_garch_and_forecast(
            returns_series=df_train_dynamic["log_return"],
            last_price=last_price,
            vol_model=vol_model_choice,  # <= 任意のモデルを指定
            p=1,
            o=1,  # EGARCH や GJR のときは o=1 を入れてみる
            q=1,
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

    # =====================================================
    # 4) 予測結果を DataFrame 化し、実際の価格とマージ
    # =====================================================
    df_pred = pd.DataFrame(predictions).set_index("date")
    df_plot = df_test.merge(df_pred, left_index=True, right_index=True, how="left")

    # =====================================================
    # 5) グラフ描画 (予測 vs. 実測)
    # =====================================================
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

    plt.title(f"Comparison: Actual Price vs. {vol_model_choice} Forecast Distribution ({day_of_interval}-day ahead)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # =====================================================
    # 6) 数値評価 (MAE, RMSE, MAPE など)
    #    ここでは 50%タイル予測(pred_50)を使用
    # =====================================================
    df_plot["error_50"] = df_plot["pred_50"] - df_plot["close"]
    mae = np.mean(np.abs(df_plot["error_50"]))
    mse = np.mean(df_plot["error_50"] ** 2)
    rmse = np.sqrt(mse)
    df_nonzero = df_plot[df_plot["close"] != 0].copy()
    mape = np.mean(np.abs((df_nonzero["pred_50"] - df_nonzero["close"]) / df_nonzero["close"])) * 100

    print("===== Forecast Accuracy Metrics (using pred_50) =====")
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

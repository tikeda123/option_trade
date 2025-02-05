import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX

import pmdarima as pm

# TensorFlowやscikit-learn関連 (不要なら削除してOK)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ユーザー環境に合わせて
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *
from data_preprocessing import clean_option_data

def main():
    """
    ARIMA(もしくはSARIMAX)を用いてvolatilityを予測する。
    改善ポイント:
      - 対数変換で定常性を高める
      - pmdarimaのauto_arimaで(p,d,q)を自動探索
      - 週次季節性を考慮(m=7)したSARIMAXモデルを使う
    """

    # 1. MongoDBからデータを読み込み
    db = MongoDataLoader()
    df = db.load_data_from_datetime_period(
        datetime(2021, 1, 1),
        datetime(2024, 12, 31),
        coll_type=MARKET_DATA_TECH,
        symbol="BTCUSDT",
        interval=1440
    )

    # 2. 欠損処理などを行い、volatility 列を時系列として抽出
    df = df.dropna(subset=["volatility"])  # 欠損を含む行を削除
    if not isinstance(df.index, pd.DatetimeIndex):
        df.set_index("date", inplace=True)  # "date" 列をDatetimeIndexに設定

    # ボラティリティが 0 に近い値を含む場合を想定し、log(1 + x) 変換をする
    # (純粋にlog(x)でも問題なければそちらでも良い)
    volatility_series = df["volatility"].astype(float)
    transformed_series = np.log1p(volatility_series)

    # 3. 学習・テストデータに分割（例: 80% を学習、残りをテスト）
    train_size = int(len(transformed_series) * 0.8)
    train_data = transformed_series.iloc[:train_size]
    test_data = transformed_series.iloc[train_size:]

    # 4. auto_arimaで (p,d,q)(P,D,Q,m=7) を自動探索
    #    - seasonal=True, m=7 (週次季節性)
    #    - AICやBICが最小となるモデルを探す
    #    - 時間がかかる場合があるため、範囲を制限
    auto_arima_model = pm.auto_arima(
        train_data,
        start_p=1, start_q=1,
        max_p=5, max_q=5,
        d=None,              # 自動で決定
        seasonal=True,
        m=7,                 # 週次季節性を仮定
        start_P=0, start_Q=0,
        max_P=2, max_Q=2,
        trace=True,          # Trueにすると探索過程が表示される
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )
    print("Best model (p,d,q)(P,D,Q,m):", auto_arima_model.order, auto_arima_model.seasonal_order)

    # 5. SARIMAXモデルで再学習
    #    auto_arima_modelで得られた (p,d,q)(P,D,Q,m) を使う
    p, d, q = auto_arima_model.order
    P, D, Q, m = auto_arima_model.seasonal_order

    sarimax_model = SARIMAX(
        train_data,
        order=(p, d, q),
        seasonal_order=(P, D, Q, m),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    sarimax_fit = sarimax_model.fit(disp=False)
    print(sarimax_fit.summary())

    # 6. テストデータの期間を予測
    #   - 学習データの末尾～テストデータ末尾分まで一括予測
    #   - dynamic=Trueにするとウォークフォワード予測(過去予測結果も用いる)を行う
    start = len(train_data)       # 学習データ終了のindex(整数)
    end = len(train_data) + len(test_data) - 1
    forecast_log = sarimax_fit.predict(start=start, end=end, dynamic=True)

    # 予測結果を逆変換 ( expm1: log(1+x)の逆関数 )
    forecast = np.expm1(forecast_log)
    actual_test = np.expm1(test_data)

    # 7. 評価指標(MSE, MAE, RMSE)を計算
    mse = mean_squared_error(actual_test, forecast)
    mae = mean_absolute_error(actual_test, forecast)
    rmse = np.sqrt(mse)

    print(f"Test MSE  : {mse:.6f}")
    print(f"Test MAE  : {mae:.6f}")
    print(f"Test RMSE : {rmse:.6f}")

    # 8. 結果の可視化
    #   学習データ(復元前)→復元後にしてプロット or
    #   そのまま学習データを可視化→予測とテストも可視化する形でもよい
    train_data_inv = np.expm1(train_data)  # 学習データを逆変換
    test_data_inv = actual_test            # こちらは既に逆変換後

    plt.figure(figsize=(12, 6))
    plt.plot(train_data_inv.index, train_data_inv, label='Train Data (inversed)', color='blue')
    plt.plot(test_data_inv.index, test_data_inv, label='Test Data (inversed)', color='green')
    plt.plot(test_data_inv.index, forecast, label='Forecast', color='red')
    plt.title("Volatility Forecast by SARIMAX + log(1+x) transform + seasonal(m=7)")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

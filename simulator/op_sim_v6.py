import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# archパッケージ (GARCHファミリーモデル)
from arch import arch_model

# scikit-learn関連
from sklearn.metrics import mean_squared_error, mean_absolute_error

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# MongoDBなどからデータをロードするクラス・定数 (ユーザー環境に合わせて)
from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *
from data_preprocessing import clean_option_data

def main():
    """
    GJR-GARCH (TARCH) モデルを用いて "volatility" 列を予測するサンプルコード。
    発散を防ぐため、log(1+x)変換 ＋ 予測の際の無限大/極大値をクリップ。
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

    # 2. 欠損処理とvolatility列の抽出
    df = df.dropna(subset=["volatility"])
    if not isinstance(df.index, pd.DatetimeIndex):
        df.set_index("date", inplace=True)

    # ---- (A) log(1 + x) 変換 ----
    vol_original = df["volatility"].astype(float)
    vol_log = np.log1p(vol_original)  # log(1 + volatility)

    # 3. 学習データとテストデータに分割 (8:2)
    train_size = int(len(vol_log) * 0.8)
    train_data = vol_log.iloc[:train_size]
    test_data = vol_log.iloc[train_size:]

    # 4. ローリング(ウォークフォワード)予測
    history = train_data.copy()
    predictions_log = []  # logスケールでの予測値
    test_index = test_data.index

    print("Start rolling forecast...")

    # 予測結果が極端に大きい場合をクリップするための上限値 (対数スケール)
    # 必要に応じて適切な値を設定 (例: log1p(数十倍～数百倍)相当)
    LOG_CLIP_UPPER = 5.0  # expm1(5)= ~148.4 つまりボラティリティ150程度を上限

    for i in range(len(test_data)):
        # --- 4.1 モデル定義 ---
        model = arch_model(
            history,
            mean='AR',    # AR(1)平均モデル (logボラティリティに対して)
            lags=1,
            vol='GARCH',
            p=1,
            o=1,          # GJR-GARCH
            q=1,
            dist='studentst'
        )

        # --- 4.2 フィット ---
        res = model.fit(disp='off')

        # --- 4.3 1ステップ先予測(=対数スケール上の次時点平均) ---
        forecast = res.forecast(horizon=1, reindex=False)
        pred_log = forecast.mean.values[-1, 0]

        # ここでinfやNaNをチェック＆クリップ
        if not np.isfinite(pred_log):
            pred_log = 0.0  # 無限大・NaNが出た場合は何らかの適切な値に置換
        elif pred_log > LOG_CLIP_UPPER:
            pred_log = LOG_CLIP_UPPER

        predictions_log.append(pred_log)

        # --- 4.4 ローリング更新 ---
        true_log = test_data.iloc[i]
        new_series = pd.Series([true_log], index=[test_index[i]])
        history = pd.concat([history, new_series])

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(test_data)} steps...")

    print("Rolling forecast completed.")

    # 5. 予測値を逆変換し，評価指標を計算
    predictions_log_arr = np.array(predictions_log)
    predictions = np.expm1(predictions_log_arr)
    actuals = np.expm1(test_data.values)

    # さらにinfが混入していないか最終チェック＆クリップ
    predictions[~np.isfinite(predictions)] = 0.0

    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mse)

    print(f"\nEvaluation on Test Set:")
    print(f"  MSE  : {mse:.6f}")
    print(f"  MAE  : {mae:.6f}")
    print(f"  RMSE : {rmse:.6f}")

    # 6. 可視化
    train_data_inv = np.expm1(train_data)
    test_data_inv = np.expm1(test_data)

    plt.figure(figsize=(12, 6))
    plt.plot(train_data_inv.index, train_data_inv, label='Train Data (vol)', color='blue')
    plt.plot(test_data_inv.index, test_data_inv, label='Test Data (vol)', color='green')
    plt.plot(test_data_inv.index, predictions, label='GJR-GARCH Forecast (vol)', color='red')
    plt.title("Volatility Forecast by GJR-GARCH (TARCH) + log1p transform + Clipping")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

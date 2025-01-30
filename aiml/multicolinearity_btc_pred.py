import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from arch import arch_model
from scipy.stats import t, norm
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore", category=FutureWarning, module="arch")

# Mongoなどの独自ライブラリに依存する部分を仮定
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import MARKET_DATA_TECH

def main():
    # ====================================================
    # 1. データロード
    # ====================================================
    data_loader = MongoDataLoader()
    df = data_loader.load_data_from_datetime_period(
        start_date="2020-01-01 00:00:00",
        end_date="2025-01-06 00:00:00",
        coll_type=MARKET_DATA_TECH,
        symbol="BTCUSDT",
        interval=1440
    )

    print("Head of original DataFrame:")
    print(df.head())
    print("Columns of original DataFrame:")
    print(df.columns)

    # ====================================================
    # 2. VIFを計算するための前処理
    # ====================================================
    # 例: 'close', 'start_at', 'date' など目的変数 or 数値以外のカラムを除外
    #     他にも "symbol", "interval" などの文字列カラムがある場合は除外する
    # 調査対象の特徴量を選択
    feature_cols = [
        "rsi",        # 調べたい特徴量
        "macdhist",   # 調べたい特徴量
        "volume",     # 調べたい特徴量
        "wclprice",   # 調べたい特徴量
        "roc",        # 調べたい特徴量
        "sma",        # 調べたい特徴量
        "mfi",        # 調べたい特徴量
        "atr",        # 調べたい特徴量
        "ema"         # 調べたい特徴量
    ]

    # 数値データだけ抽出
    df_numeric = df[feature_cols].copy()

    # 欠損値が含まれているとエラーが出るので、VIF計算前に除去 or 補完する
    # 今回は簡単に dropna() で欠損を削除
    df_numeric = df_numeric.dropna()

    # ====================================================
    # 3. VIFを計算
    # ====================================================
    # 計算用に numpy 配列に変換
    X = df_numeric.values
    # カラム名を取得しておく
    features = df_numeric.columns

    vif_list = []
    for i in range(X.shape[1]):
        vif_val = variance_inflation_factor(X, i)
        vif_list.append(vif_val)

    vif_df = pd.DataFrame({
        "feature": features,
        "VIF": vif_list
    })

    # VIFが高いほど、多重共線性が高いと判断される
    # 一般に10を超えると多重共線性が高いとする場合が多い
    vif_df_sorted = vif_df.sort_values(by="VIF", ascending=False)

    print("\n=== VIF (Variance Inflation Factor) ===")
    print(vif_df_sorted)

if __name__ == "__main__":
    main()

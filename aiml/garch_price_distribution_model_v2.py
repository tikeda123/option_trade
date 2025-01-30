import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from arch import arch_model
import warnings
from scipy.stats import t
from scipy.stats import norm

warnings.filterwarnings("ignore", category=FutureWarning, module="arch")

# Mongoなどの独自ライブラリに依存する部分を仮定
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.trading_logger import TradingLogger
from mongodb.data_loader_mongo import MongoDataLoader
from common.utils import get_config
from common.constants import MARKET_DATA_TECH

class GarchPriceDistributionModel:
    """
    GARCH(1,1)を利用してBTC価格のリターンボラティリティを予測し、
    そこから価格の分布(クォンタイル)を推定するサンプルクラス。
    """

    def __init__(
        self,
        p=1,
        q=1,
        dist="t",  # ◆ 't' に設定することで裾の重い分布を考慮
        mean="constant",
        vol="GARCH"
    ):
        self.p = p
        self.q = q
        self.dist = dist
        self.mean = mean
        self.vol = vol
        self.model_fit = None
        self.last_price = None
        self.df_model = None

    def prepare_data(self, df: pd.DataFrame, price_col="close"):
        if "date" in df.columns:
            df = df.set_index("date")

        self.last_price = df[price_col].iloc[-1]

        df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
        df = df.dropna(subset=["log_return"])

        self.df_model = df.copy()
        return df["log_return"]

    def fit_model(self, returns: pd.Series):
        am = arch_model(
            returns,
            mean=self.mean,
            vol=self.vol,
            p=self.p,
            q=self.q,
            dist=self.dist,
            rescale=False
        )
        self.model_fit = am.fit(disp="off")

    def forecast_distribution(self, horizon=1, quantiles=[0.1, 0.5, 0.9]):
        if self.model_fit is None:
            raise ValueError("Model is not fitted yet. Please call fit_model() first.")

        forecast_res = self.model_fit.forecast(horizon=horizon)

        mean_forecast = forecast_res.mean[f"h.{horizon}"].iloc[-1]
        var_forecast = forecast_res.variance[f"h.{horizon}"].iloc[-1]
        std_forecast = np.sqrt(var_forecast)

        if self.dist == "t":
            nu = self.model_fit.params.get("nu", 10.0)
        else:
            nu = None

        ret_quantiles = {}
        price_quantiles = {}
        current_price = self.last_price

        for q in quantiles:
            if self.dist == "t":
                z_q = t.ppf(q, df=nu)
            else:
                z_q = norm.ppf(q)

            ret_q = mean_forecast + std_forecast * z_q
            ret_quantiles[q] = ret_q

            price_q = current_price * np.exp(ret_q)
            price_quantiles[q] = price_q

        return {
            "forecast_returns_quantiles": ret_quantiles,
            "forecast_price_quantiles": price_quantiles,
            "horizon": horizon
        }

def judge_current_price(
    current_price: float,
    lower_bound: float,
    upper_bound: float
):
    """現在の価格が予測範囲のどこに位置するかを即時判定"""
    if current_price < lower_bound:
        return "POTENTIALLY_UNDERVALUED"
    elif current_price > upper_bound:
        return "POTENTIALLY_OVERVALUED"
    else:
        return "POTENTIALLY_FAIR"

def judge_next_day_price(
    actual_next_price: float,
    lower_bound: float,
    upper_bound: float
):
    """次の日の実際の価格を使って判定（事後検証用）"""
    if actual_next_price < lower_bound:
        return "UNDERVALUED"
    elif actual_next_price > upper_bound:
        return "OVERVALUED"
    else:
        return "FAIR"

def evaluate_predictions(df_result: pd.DataFrame):
    """即時判断の精度を評価"""
    # 判定結果のマッピング（POTENTIALLYプレフィックスを除去して比較）
    mapping = {
        'POTENTIALLY_UNDERVALUED': 'UNDERVALUED',
        'POTENTIALLY_FAIR': 'FAIR',
        'POTENTIALLY_OVERVALUED': 'OVERVALUED'
    }

    # 予測と実際の結果を比較
    total = len(df_result)
    correct = sum(df_result.apply(lambda row: mapping[row['current_judgement']] == row['next_day_judgement'], axis=1))
    accuracy = correct / total

    # 混同行列の作成
    print("\n=== 即時判断の精度評価 ===")
    print(f"全体の的中率: {accuracy:.1%}")

    print("\n=== 詳細な分析 ===")
    confusion = {
        'UNDERVALUED': {'correct': 0, 'total': 0},
        'FAIR': {'correct': 0, 'total': 0},
        'OVERVALUED': {'correct': 0, 'total': 0}
    }

    for _, row in df_result.iterrows():
        predicted = mapping[row['current_judgement']]
        actual = row['next_day_judgement']

        confusion[predicted]['total'] += 1
        if predicted == actual:
            confusion[predicted]['correct'] += 1

    print("\n各判断の精度:")
    for judgment, stats in confusion.items():
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            print(f"{judgment}: {stats['correct']}/{stats['total']} = {accuracy:.1%}")

    # 判断の遷移分析
    print("\n=== 判断の遷移分析 ===")
    transitions = pd.crosstab(
        df_result['current_judgement'].map(mapping),
        df_result['next_day_judgement'],
        normalize='index'
    )
    print("\n即時判断が各結果に遷移した確率:")
    print(transitions.round(3))

    return {
        'overall_accuracy': accuracy,
        'confusion': confusion,
        'transitions': transitions
    }

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
        interval=1440*3
    )

    # （念のため日付ソート & インデックス化）
    df = df.sort_values("date").reset_index(drop=True)
    df.set_index("date", inplace=True)

    # ====================================================
    # 2. 判定したい期間を指定(例)
    #    start_date_judge と end_date_judge の範囲だけ
    #    judge 用にループを回す
    # ====================================================
    start_date_judge = "2024-08-01"
    end_date_judge   = "2025-01-01"

    # やり方1: まずはデータを全取得した上で、期間をマスクして
    # 「判定用インデックス」を作る
    df_for_judge = df.loc[start_date_judge : end_date_judge]

    # ====== パラメータ設定 ======
    lookback = 200  # 学習に使う直近日数
    horizon = 1     # 1日先を予測
    quantiles = [0.1, 0.5, 0.9]

    result_judgement = []

    # ====================================================
    # 3. 指定期間内で日ごとに判定を行う
    #    ただし、i - lookback < 0 を避ける
    # ====================================================
    # df_for_judge のインデックスをリスト化
    judge_indices = df_for_judge.index.tolist()

    for current_date in judge_indices:
        # current_date の「行番号」を df 全体の中で特定
        # （.get_loc() を使う）
        i = df.index.get_loc(current_date)

        # lookback分さかのぼれるか確認
        if i - lookback < 0:
            # lookback分の履歴がない場合はスキップ
            continue

        if i + horizon >= len(df):
            # horizon分先がない場合はスキップ
            break

        # 直近 lookback 日間のデータを取り出す
        df_window = df.iloc[i - lookback : i]

        # モデルを準備&学習
        model = GarchPriceDistributionModel(dist="t")
        returns = model.prepare_data(df_window, price_col="close")
        model.fit_model(returns)

        # 予測分位を取得
        result_forecast = model.forecast_distribution(horizon=horizon, quantiles=quantiles)
        forecast_price_quantiles = result_forecast["forecast_price_quantiles"]

        # 実際の horizon 日先の価格
        actual_price = df["close"].iloc[i + horizon]

        lower_10pct = forecast_price_quantiles[0.1]
        median_50pct = forecast_price_quantiles[0.5]
        upper_90pct = forecast_price_quantiles[0.9]

        # 両方の判定を実施
        current_judgement = judge_current_price(
            current_price=df["close"].iloc[i],  # 現在の価格
            lower_bound=lower_10pct,
            upper_bound=upper_90pct
        )

        next_day_judgement = judge_next_day_price(
            actual_next_price=df["close"].iloc[i + horizon],  # 次の日の価格
            lower_bound=lower_10pct,
            upper_bound=upper_90pct
        )

        next_date = df.index[i + horizon]
        result_judgement.append({
            "current_date": current_date,
            "next_date": next_date,
            "lower_10pct": lower_10pct,
            "median_50pct": median_50pct,
            "upper_90pct": upper_90pct,
            "current_price": df["close"].iloc[i],
            "next_day_price": df["close"].iloc[i + horizon],
            "current_judgement": current_judgement,    # 即時判定
            "next_day_judgement": next_day_judgement  # 事後検証
        })

    # DataFrame化
    df_result = pd.DataFrame(result_judgement)

    # 即時判定の集計
    print("\n=== 即時判定の結果 ===")
    current_counts = df_result['current_judgement'].value_counts()
    total = len(df_result)
    print(f"割安(POTENTIALLY_UNDERVALUED): {current_counts.get('POTENTIALLY_UNDERVALUED', 0)}件 ({(current_counts.get('POTENTIALLY_UNDERVALUED', 0) / total * 100):.1f}%)")
    print(f"適正(POTENTIALLY_FAIR): {current_counts.get('POTENTIALLY_FAIR', 0)}件 ({(current_counts.get('POTENTIALLY_FAIR', 0) / total * 100):.1f}%)")
    print(f"割高(POTENTIALLY_OVERVALUED): {current_counts.get('POTENTIALLY_OVERVALUED', 0)}件 ({(current_counts.get('POTENTIALLY_OVERVALUED', 0) / total * 100):.1f}%)")

    # 事後検証の集計
    print("\n=== 事後検証の結果 ===")
    next_counts = df_result['next_day_judgement'].value_counts()
    print(f"割安(UNDERVALUED): {next_counts.get('UNDERVALUED', 0)}件 ({(next_counts.get('UNDERVALUED', 0) / total * 100):.1f}%)")
    print(f"適正(FAIR): {next_counts.get('FAIR', 0)}件 ({(next_counts.get('FAIR', 0) / total * 100):.1f}%)")
    print(f"割高(OVERVALUED): {next_counts.get('OVERVALUED', 0)}件 ({(next_counts.get('OVERVALUED', 0) / total * 100):.1f}%)")

    # 予測精度の評価を実行
    evaluation_results = evaluate_predictions(df_result)

    # 例: CSV出力
    # df_result.to_csv("garch_judgement.csv", index=False)

if __name__ == "__main__":
    main()

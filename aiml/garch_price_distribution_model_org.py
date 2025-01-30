import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# pip install arch
from arch import arch_model
import warnings
# archパッケージの DataScaleWarning を無視
warnings.filterwarnings("ignore", category=FutureWarning, module="arch")  # FutureWarningやほかのwarningでも対象指定


# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the path of the parent directory to sys.path
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
        dist="normal",
        mean="constant",
        vol="GARCH"
    ):
        """
        p, q: GARCH(p, q) の p, q
        dist: 'normal', 't' など分布の指定
        mean: 'constant', 'AR', 'HAR' など平均モデルの指定
        vol : 'GARCH', 'EGARCH', 'ARCH' などボラティリティモデル
        """
        self.p = p
        self.q = q
        self.dist = dist
        self.mean = mean
        self.vol = vol
        self.model_fit = None
        self.last_price = None

    def prepare_data(self, df: pd.DataFrame, price_col="close"):
        """
        データフレーム df から対数リターン系列を作成し、学習用に返す。
        price_col: BTC価格カラム名 (例: 'close')
        """
        # 日付などがある場合は日付をindexに設定しておく(任意)
        if "date" in df.columns:
            df = df.set_index("date")

        # 価格を保持しておく（予測時に使う）
        self.last_price = df[price_col].iloc[-1]

        # 対数リターン
        df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
        df = df.dropna(subset=["log_return"])

        return df["log_return"]

    def fit_model(self, returns: pd.Series):
        """
        対数リターンにGARCHモデルをフィット
        """
        # arch_modelのインスタンス生成
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
        #print(self.model_fit.summary())

    def forecast_distribution(
        self,
        horizon=1,
        quantiles=[0.1, 0.5, 0.9]
    ):
        """
        horizon ステップ先までの条件付きボラティリティを予測し、
        そこからリターンの分布 → 価格の分布の分位点を推定して返す。

        戻り値:
          {
            'forecast_returns_quantiles': { 0.1: 〇, 0.5: 〇, ... },
            'forecast_price_quantiles':   { 0.1: 〇, 0.5: 〇, ... },
            'horizon': horizon
          }
        """
        if self.model_fit is None:
            raise ValueError("Model is not fitted yet. Please call fit_model() first.")

        # GARCH予測を取得
        # 例: horizon=1 なら1ステップ先、=5 なら5ステップ先
        forecast_res = self.model_fit.forecast(horizon=horizon)

        # 予測結果（平均, 分散）を抜き出す
        #   mean_forecast: E[r_{t+h}|情報]
        #   variance_forecast: Var(r_{t+h}|情報)
        mean_forecast = forecast_res.mean["h."+str(horizon)].iloc[-1]
        var_forecast = forecast_res.variance["h."+str(horizon)].iloc[-1]
        std_forecast = np.sqrt(var_forecast)

        # リターンが正規分布 N(mean, std^2) と仮定して quantile を計算
        #   dist='t' 等の場合はStudent-tを用いることも可能
        from scipy.stats import norm

        # horizonが1の場合の簡易例として、1ステップ先のみを想定
        # (複数ステップ先ならループで取り出すなど工夫が必要)
        ret_quantiles = {}
        price_quantiles = {}
        for q in quantiles:
            # 正規分布のq分位点（平均=mean_forecast, 標準偏差=std_forecast）
            z_q = norm.ppf(q)  # 標準正規分布のq分位
            ret_q = mean_forecast + std_forecast * z_q
            ret_quantiles[q] = ret_q

            # 価格への変換
            #   P_{t+1} = P_t * exp(r_{t+1})  (対数リターンの場合)
            price_q = self.last_price * np.exp(ret_q)
            price_quantiles[q] = price_q

        return {
            "forecast_returns_quantiles": ret_quantiles,
            "forecast_price_quantiles": price_quantiles,
            "horizon": horizon
        }

def main():
    data_loader = MongoDataLoader()
    df = data_loader.load_data_from_datetime_period(
        start_date="2020-01-01 00:00:00",
        end_date="2025-01-06 00:00:00",
        coll_type=MARKET_DATA_TECH,
        symbol="BTCUSDT",
        interval=1440
    )
    model = GarchPriceDistributionModel()
    returns = model.prepare_data(df, price_col="close")
    model.fit_model(returns)
    result = model.forecast_distribution(horizon=1, quantiles=[0.1, 0.5, 0.9])
    print(result)

if __name__ == "__main__":
    main()
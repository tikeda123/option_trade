import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# ユーザー環境に合わせたパス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *

def bayesian_support_estimation(low_prices: np.array,
                                grid_points: int = 500,
                                sigma: float = None):
    """
    ベイズ推定を用いてサポートライン S を推定する例 (正規分布モデル)。

    モデル:
      各日の low 値は N(S, sigma^2) に従うと仮定する。
      つまり、low_i ~ Normal(S, sigma^2)。
      これにより、low_i がサポートライン S を下回る場合も一定の確率で起こりうる。

    Parameters
    ----------
    low_prices : np.array
        各時点の low 価格の配列
    grid_points : int, optional
        S の候補値を求めるグリッドの数 (default=500)
    sigma : float, optional
        ノイズの標準偏差。指定がなければ low_prices の標準偏差を使用

    Returns
    -------
    S_grid : np.array
        候補となるサポートライン S のグリッド
    posterior : np.array
        各 S に対応する正規化済みの事後分布
    S_mean : float
        事後分布の平均値
    S_median : float
        事後分布の中央値
    cred_interval : tuple(float, float)
        95% 信頼区間（下限, 上限）
    """
    low_prices = np.array(low_prices)
    min_low = low_prices.min()
    max_low = low_prices.max()
    range_low = max_low - min_low

    # サポートライン S のグリッドを設定
    margin = 0.1 * range_low if range_low > 0 else 1.0
    S_min = min_low - margin
    S_max = max_low + margin
    S_grid = np.linspace(S_min, S_max, grid_points)

    # sigma が指定されていない場合、low_prices の標準偏差を使用
    if sigma is None:
        sigma = np.std(low_prices)
        # 万が一全データが一定値の場合の回避策
        if sigma <= 0:
            sigma = 1.0

    # 事前分布は一様分布
    prior = np.ones_like(S_grid)

    # 対数尤度の計算
    n = len(low_prices)
    log_likelihood = np.empty_like(S_grid)
    # 定数項: - (n/2) * log(2πσ^2)
    constant_term = -0.5 * n * np.log(2.0 * np.pi * sigma**2)

    for i, S in enumerate(S_grid):
        diff = low_prices - S
        sum_sq = np.sum(diff**2)
        # log L(S) = 定数項 - sum((low_i - S)^2) / (2σ^2)
        log_likelihood[i] = constant_term - sum_sq / (2.0 * sigma**2)

    # 事後分布は尤度 × 事前
    log_posterior = log_likelihood + np.log(prior)
    # 数値安定のため最大値を引いて exp
    log_posterior -= np.max(log_posterior)
    posterior_unnorm = np.exp(log_posterior)

    # 台形則で正規化
    area = np.trapz(posterior_unnorm, S_grid)
    posterior = posterior_unnorm / area

    # 事後平均
    S_mean = np.trapz(S_grid * posterior, S_grid)
    # CDF 計算
    dx = S_grid[1] - S_grid[0]
    cdf = np.cumsum(posterior) * dx
    # 事後中央値
    S_median = S_grid[np.searchsorted(cdf, 0.5)]
    # 95% 信頼区間
    lower_bound = S_grid[np.searchsorted(cdf, 0.025)]
    upper_bound = S_grid[np.searchsorted(cdf, 0.975)]
    cred_interval = (lower_bound, upper_bound)

    return S_grid, posterior, S_mean, S_median, cred_interval

def main():
    # MongoDBからBTCの時系列データを取得
    db = MongoDataLoader()
    df = db.load_data_from_datetime_period(datetime(2025, 1, 1),
                                           datetime(2025, 2, 1),
                                           coll_type=MARKET_DATA_TECH,
                                           symbol='BTCUSDT',
                                           interval=60)
    # 利用するカラムのみ抽出
    graph_df = df[['start_at', 'close', 'high', 'low', 'open', 'volume']]

    # サポートライン推定に用いる low 値を抽出
    low_prices = graph_df['low'].values

    # ベイズ推定によるサポートラインの推定 (正規分布モデル)
    S_grid, posterior, S_mean, S_median, cred_interval = bayesian_support_estimation(low_prices)

    print(f"推定されたサポートライン (事後平均): {S_mean:.2f}")
    print(f"推定されたサポートライン (事後中央値): {S_median:.2f}")
    print(f"95% 信頼区間: ({cred_interval[0]:.2f}, {cred_interval[1]:.2f})")

    # 事後分布のプロット
    plt.figure(figsize=(8, 5))
    plt.plot(S_grid, posterior, label='Posterior of Support Line S')
    plt.axvline(S_mean, color='r', linestyle='--', label=f'Mean: {S_mean:.2f}')
    plt.axvline(S_median, color='g', linestyle='--', label=f'Median: {S_median:.2f}')
    plt.fill_between(S_grid, posterior,
                     where=(S_grid >= cred_interval[0]) & (S_grid <= cred_interval[1]),
                     color='gray', alpha=0.3, label='95% Credible Interval')
    plt.xlabel('Support Line S')
    plt.ylabel('Posterior Density')
    plt.title('Bayesian Estimation of Support Line (Normal Likelihood)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

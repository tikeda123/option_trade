import numpy as np
import pandas as pd
from typing import Tuple

class BatesModelCalculator:
    def __init__(
        self,
        risk_free_rate: float = 0.03,
        kappa: float = 1.5,         # mean reversion speed
        theta: float = 0.04,        # long-term variance
        sigma: float = 0.3,         # vol of vol (volatility of variance)
        rho: float = -0.5,          # correlation between Brownian motions
        v0: float = 0.04,           # initial variance
        lambda_j: float = 0.1,      # jump intensity (Poisson parameter)
        mu_j: float = 0.0,          # mean of log-jump size
        sigma_j: float = 0.2        # std of log-jump size
    ):
        """
        Batesモデル（Heston + Mertonジャンプ拡散）のパラメータを保持し、
        モンテカルロ・シミュレーションでオプション価格を計算するクラス。

        Args:
            risk_free_rate (float): 無リスク金利
            kappa (float): Hestonモデルの mean reversion speed
            theta (float): Hestonモデルの long-term variance
            sigma (float): Hestonモデルの vol of vol
            rho (float): S(t)とv(t)のブラウン運動間の相関係数
            v0 (float): 初期時点の分散 v(0)
            lambda_j (float): Mertonジャンプ(ポアソン過程)の強度パラメータ
            mu_j (float): ジャンプ大きさの対数平均
            sigma_j (float): ジャンプ大きさの対数標準偏差
        """
        self.risk_free_rate = risk_free_rate
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0
        self.lambda_j = lambda_j
        self.mu_j = mu_j
        self.sigma_j = sigma_j

    def calculate_volatility(self, df: pd.DataFrame) -> float:
        """
        BTCなどの高ボラティリティ資産について、参考までに
        過去データから対数リターンの標準偏差を計算する（年率換算）。
        （実際のHestonパラメータとは異なる概念の「参考値」。）
        """
        returns = np.log(df['close'] / df['close'].shift(1)).dropna()
        # 1時間足データ → 年間化: sqrt(365*24)
        volatility = returns.std() * np.sqrt(365 * 24)
        return volatility

    def _simulate_bates_paths(
        self,
        S0: float,
        T: float,
        steps: int,
        n_sims: int
    ) -> np.ndarray:
        """
        Batesモデルに基づく価格パス S(t) をEuler-Maruyama法でシミュレート。

        Returns:
            paths (ndarray): shape = (n_sims, steps+1)
                             各シミュレーションでのS(t)の軌跡
        """
        dt = T / steps
        # 各パスの格納
        S = np.zeros((n_sims, steps + 1))
        # ボラティリティ(分散)のパス
        v = np.zeros((n_sims, steps + 1))

        # 初期値セット
        S[:, 0] = S0
        v[:, 0] = self.v0

        # 事前に乱数をまとめて生成しておく
        # z1, z2 ~ N(0,1)
        # ポアソン乱数 (ジャンプ回数)もステップ毎に生成
        rand_z1 = np.random.normal(0, 1, (n_sims, steps))
        rand_z2 = np.random.normal(0, 1, (n_sims, steps))
        poisson_rn = np.random.poisson(self.lambda_j * dt, (n_sims, steps))

        # 相関を考慮したz1, z2'
        # z2' = rho * z1 + sqrt(1-rho^2) * z2
        for t in range(steps):
            z1 = rand_z1[:, t]
            z2 = rand_z2[:, t]
            z2_ = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2

            # v(t) の更新 (Heston)
            v[:, t+1] = np.maximum(
                v[:, t]
                + self.kappa * (self.theta - v[:, t]) * dt
                + self.sigma * np.sqrt(np.maximum(v[:, t], 0)) * np.sqrt(dt) * z2_,
                0.0
            )

            # ジャンプ回数 (0以上の整数)
            jump_counts = poisson_rn[:, t]

            # ジャンプ大きさ = exp( sum of Normal(mu_j, sigma_j) )^jump_counts
            #   ただし1ステップ内でjump_counts>1の可能性がある
            #   sum of i=1..N ( mu_j + sigma_j * Z_i ) ~ Normal(N*mu_j, N*sigma_j^2)
            #   → まとめて e^( Normal(N*mu_j, sqrt(N)*sigma_j ) )
            jump_size = np.ones(n_sims)  # デフォルトでジャンプなし→1

            # ジャンプがあるシミュレーションだけ計算をする
            has_jump_idx = (jump_counts > 0)
            if np.any(has_jump_idx):
                # それぞれのパスについてジャンプ回数に応じた正規乱数を生成
                # 正規分布の合計を1回で生成するには:
                #   Normal(N*mu_j, sqrt(N)*sigma_j)
                #   N = jump_counts[i]
                # パス単位で異なるNを使う必要があるのでループ。
                # （高速化する場合はベクトル演算工夫が必要）
                for i in np.where(has_jump_idx)[0]:
                    N_i = jump_counts[i]
                    # 合計ジャンプ ~ Normal(N_i*mu_j, sqrt(N_i)*sigma_j)
                    mean_ = N_i * self.mu_j
                    std_  = np.sqrt(N_i) * self.sigma_j
                    z_jump = np.random.normal(mean_, std_)
                    jump_size[i] = np.exp(z_jump)

            # S(t) の更新
            # リスク中立ドリフト: r - 1/2 * v(t) でOK（Hestonのリスク中立測度下）
            # さらにジャンプが掛かる
            drift = (self.risk_free_rate - 0.5 * v[:, t]) * dt
            diff  = np.sqrt(np.maximum(v[:, t], 0)) * np.sqrt(dt) * z1

            S[:, t+1] = S[:, t] * np.exp(drift + diff) * jump_size

        return S

    def bates_eu_option_price_mc(
        self,
        S0: float,
        K: float,
        T: float,
        steps: int = 100,
        n_sims: int = 10_000,
        option_type: str = "call"
    ) -> float:
        """
        Batesモデルを用いてヨーロピアン・オプション価格をモンテカルロで計算

        Args:
            S0 (float): 現在のスポット価格
            K (float): 行使価格
            T (float): 満期 (年)
            steps (int): 1パスあたりのタイムステップ数
            n_sims (int): モンテカルロ試行回数
            option_type (str): "call" or "put"

        Returns:
            float: モンテカルロ推定したオプション理論価格
        """
        # パス生成
        paths = self._simulate_bates_paths(S0, T, steps, n_sims)

        # 期末価格
        S_T = paths[:, -1]
        if option_type == "call":
            payoff = np.maximum(S_T - K, 0.0)
        else:
            payoff = np.maximum(K - S_T, 0.0)

        # 割引
        price = np.exp(-self.risk_free_rate * T) * np.mean(payoff)
        return price

def simulate_option_prices(
    df: pd.DataFrame,
    strike_prices: list,
    expiry_date: pd.Timestamp,
    risk_free_rate: float = 0.03,
    option_type: str = "call"
) -> pd.DataFrame:
    """
    Batesモデルを使ったオプション価格シミュレーション例。
    （日ごとの観測に対して「残存日数を更新し、オプション理論価格をモンテカルロで計算」する）

    Args:
        df (pd.DataFrame): 'close', 'start_at' を含むDataFrame
        strike_prices (list): シミュレーションする行使価格リスト
        expiry_date (pd.Timestamp): オプションの満期日
        risk_free_rate (float): 無リスク金利
        option_type (str): "call" or "put"

    Returns:
        pd.DataFrame: 各日・各行使価格に対してオプション価格を格納したDataFrame
    """
    if option_type not in ["call", "put"]:
        raise ValueError("option_type must be either 'call' or 'put'")

    # ---- Batesモデルのパラメータは任意に調整してください ----
    bates_calc = BatesModelCalculator(
        risk_free_rate=risk_free_rate,
        kappa=1.5,      # 以下、適当な例
        theta=0.04,
        sigma=0.3,
        rho=-0.5,
        v0=0.04,
        lambda_j=0.1,   # ジャンプ強度
        mu_j=0.0,       # ジャンプ平均
        sigma_j=0.2     # ジャンプ標準偏差
    )

    results = []
    for _, row in df.iterrows():
        current_price = row['close']
        current_date = pd.to_datetime(row['start_at'])
        days_to_expiry = (expiry_date - current_date).days

        # 残存日数が0以下の場合はスキップ
        if days_to_expiry <= 0:
            continue

        T = days_to_expiry / 365.0  # 年換算

        for strike_price in strike_prices:
            # モンテカルロ計算
            option_price = bates_calc.bates_eu_option_price_mc(
                S0=current_price,
                K=strike_price,
                T=T,
                steps=100,        # タイムステップは要調整
                n_sims=5000,      # シミュレーション回数
                option_type=option_type
            )

            result = {
                'timestamp': current_date,
                'current_price': current_price,
                'strike_price': strike_price,
                'days_to_expiry': days_to_expiry,
                'option_type': option_type,
                'price': option_price
            }
            results.append(result)

    return pd.DataFrame(results)

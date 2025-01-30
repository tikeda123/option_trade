import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.optimize import brentq
from typing import Tuple, List


###############################################################################
# Black-Scholes Calculator
###############################################################################
class BlackScholesCalculator:
    def __init__(self, risk_free_rate: float = 0.03):
        """
        Initialize Black-Scholes calculator

        Args:
            risk_free_rate (float): Risk-free interest rate (annual)
        """
        self.risk_free_rate = risk_free_rate

    def calculate_option_prices(
        self,
        current_price: float,
        strike_price: float,
        volatility: float,
        days_to_expiry: int
    ) -> Tuple[float, float, float, float, float, float, float, float]:
        """
        Calculate call/put option prices and Greeks (Delta, Gamma, Vega, Theta)
        using the Black-Scholes model.

        Args:
            current_price (float): Current price of the underlying asset
            strike_price (float): Strike price of the option
            volatility (float): Annualized volatility (sigma)
            days_to_expiry (int): Number of days until expiry

        Returns:
            A tuple of 8 elements:
                (
                  call_price, put_price,
                  call_delta, put_delta,
                  gamma, vega,
                  call_theta, put_theta
                )
        """
        # Time in years
        T = days_to_expiry / 365.0
        r = self.risk_free_rate
        S = current_price
        K = strike_price
        sigma = volatility

        # T や sigma が 0 以下の場合は適宜 0 を返す
        if T <= 0 or sigma <= 0:
            return (0, 0, 0, 0, 0, 0, 0, 0)

        # d1, d2 の計算
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # ---------- オプション価格 (Call, Put) ----------
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        # Put option price (Put-Call Parity)
        put_price = call_price + K * np.exp(-r * T) - S

        # ---------- ギリシャ指標 ----------
        pdf_d1 = norm.pdf(d1)

        # Delta
        call_delta = norm.cdf(d1)
        put_delta = call_delta - 1  # = norm.cdf(d1) - 1

        # Gamma (Call/Put 共通)
        gamma = pdf_d1 / (S * sigma * np.sqrt(T))

        # Vega (Call/Put 共通)
        vega = S * np.sqrt(T) * pdf_d1

        # Theta
        call_theta = (
            - (S * pdf_d1 * sigma) / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        )
        put_theta = (
            - (S * pdf_d1 * sigma) / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        )

        return (
            call_price,
            put_price,
            call_delta,
            put_delta,
            gamma,
            vega,
            call_theta,
            put_theta
        )

    def _bs_price_for_iv(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call"
    ) -> float:
        """
        与えられた sigma での Black-Scholes オプション理論価格 (Call か Put) を返すヘルパー関数
        """
        if T <= 0 or sigma <= 0:
            return 0.0

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # "put"
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return price

    def calculate_implied_volatility(
        self,
        market_price: float,
        current_price: float,
        strike_price: float,
        days_to_expiry: int,
        option_type: str = "call",
        tol: float = 1e-8,
        max_iter: int = 100
    ) -> float:
        """
        Brent 法を用いてインプライド・ボラティリティを求める

        Args:
            market_price (float): 市場観測されたオプション価格
            current_price (float): 原資産価格
            strike_price (float): 行使価格
            days_to_expiry (int): 残存日数
            option_type (str): "call" または "put"
            tol (float): 許容誤差
            max_iter (int): 最大反復回数

        Returns:
            float: インプライド・ボラティリティ (年率)
        """
        T = days_to_expiry / 365.0
        if T <= 0:
            return 0.0

        # 探索範囲を設定（必要に応じて調整）
        low, high = 1e-9, 5.0

        def objective(sigma):
            return self._bs_price_for_iv(
                S=current_price,
                K=strike_price,
                T=T,
                r=self.risk_free_rate,
                sigma=sigma,
                option_type=option_type
            ) - market_price

        try:
            iv = brentq(objective, low, high, xtol=tol, maxiter=max_iter)
        except ValueError:
            iv = 0.0

        return iv


###############################################################################
# Simulation Function (using 'hv' column as volatility)
###############################################################################
def simulate_option_prices(
    df: pd.DataFrame,
    strike_prices: List[float],
    expiry_date: pd.Timestamp,
    risk_free_rate: float = 0.03,
    option_type: str = "call"
) -> pd.DataFrame:
    """
    Simulate option prices for given price data and strike prices

    Args:
        df (pd.DataFrame): Price data with columns:
            - 'close': Underlying asset price
            - 'start_at': Datetime or string representing the date/time
            - 'hv': Historical/Implied Vol (used as volatility each row)
        strike_prices (list): List of strike prices to simulate
        expiry_date (pd.Timestamp): Expiration date of the options
        risk_free_rate (float): Risk-free interest rate
        option_type (str): Type of option to simulate ("call" or "put")

    Returns:
        pd.DataFrame: DataFrame with columns:
            [
                'timestamp', 'current_price', 'strike_price',
                'gamma', 'vega', 'volatility',
                'days_to_expiry', 'price', 'delta', 'theta'
            ]
    """
    if option_type not in ["call", "put"]:
        raise ValueError("option_type must be either 'call' or 'put'")

    results = []
    calculator = BlackScholesCalculator(risk_free_rate=risk_free_rate)

    for _, row in df.iterrows():
        current_price = row['close']
        current_date = pd.to_datetime(row['start_at'])

        # ここで 'hv' を float に変換して使用
        # （事前に df["hv"] を to_numeric しておけばさらに安全）
        hv_value = row.get('hv', 0.0)
        try:
            volatility = float(hv_value)
        except ValueError:
            volatility = 0.0


        days_to_expiry = (expiry_date - current_date).days
        if days_to_expiry <= 0:
            # 残存日数が無い(または負)ならスキップ
            continue

        for strike_price in strike_prices:
            (
                call_price,
                put_price,
                call_delta,
                put_delta,
                gamma,
                vega,
                call_theta,
                put_theta
            ) = calculator.calculate_option_prices(
                current_price,
                strike_price,
                volatility,
                days_to_expiry
            )

            # option_type によって取り出す値を変える
            if option_type == "call":
                price = call_price
                delta = call_delta
                theta = call_theta
            else:
                price = put_price
                delta = put_delta
                theta = put_theta

            results.append({
                'timestamp': current_date,
                'current_price': current_price,
                'strike_price': strike_price,
                'gamma': gamma,
                'vega': vega,
                'volatility': volatility,
                'days_to_expiry': days_to_expiry,
                'price': price,
                'delta': delta,
                'theta': theta
            })

    return pd.DataFrame(results)


###############################################################################
# Example main() usage
###############################################################################
def main():
    # ダミーで DataFrame を用意 (本来はMongoDB等から読み込む)
    data = {
        'start_at': [
            '2024-12-17 14:00:00',
            '2024-12-17 15:00:00',
            '2024-12-17 16:00:00'
        ],
        'close': [105000, 105200, 104800],
        # hv が文字列になっているケースを想定
        'hv': ['0.55', '0.58', 'x']  # 'x' は変換エラー→0になる例
    }
    df_btc = pd.DataFrame(data)

    # hv を数値に変換（エラーは NaN 扱い→0.0 で埋める）
    df_btc["hv"] = pd.to_numeric(df_btc["hv"], errors="coerce").fillna(0.0)

    # シミュレーション条件
    strike_prices = [90000.0]
    expiry_date = pd.to_datetime("2025-03-28")  # 例

    # put オプションでシミュレーションしてみる
    df_sim_op = simulate_option_prices(
        df_btc,
        strike_prices=strike_prices,
        expiry_date=expiry_date,
        risk_free_rate=0.03,
        option_type="put"
    )

    print("Simulated Option Prices:")
    print(df_sim_op)


if __name__ == "__main__":
    main()


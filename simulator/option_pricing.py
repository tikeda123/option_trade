import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Tuple
import pandas as pd

class BlackScholesCalculator:
    def __init__(self, risk_free_rate: float = 0.03):
        """
        Initialize Black-Scholes calculator

        Args:
            risk_free_rate (float): Risk-free interest rate (annual)
        """
        self.risk_free_rate = risk_free_rate

    def calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate historical volatility from price data

        Returns:
            float: Annualized volatility based on log returns.
        """
        # Calculate daily returns (log return)
        returns = np.log(df['close'] / df['close'].shift(1))
        # 1時間足データとして 1年 = 365*24 時間 で年率化
        volatility = returns.std() * np.sqrt(365 * 24)  # Annualized from hourly data
        return volatility

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
        # Time to expiry in years
        T = days_to_expiry / 365.0
        r = self.risk_free_rate
        S = current_price
        K = strike_price
        sigma = volatility

        # T や sigma が 0 以下の場合は適宜 0 で返す
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

        # --- Delta ---
        call_delta = norm.cdf(d1)
        put_delta = call_delta - 1  # = norm.cdf(d1) - 1

        # --- Gamma (Call/Put 共通) ---
        gamma = pdf_d1 / (S * sigma * np.sqrt(T))

        # --- Vega (Call/Put 共通) ---
        vega = S * np.sqrt(T) * pdf_d1

        # --- Theta ---
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
        else:  # put
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
        # 年単位の残存期間
        T = days_to_expiry / 365.0
        if T <= 0:
            return 0.0

        # ボラティリティが 0～数百 % 程度を考慮して、探索範囲を設定
        # 必要に応じて最小値 (low) や最大値 (high) を調整
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
            # 解が得られない場合は 0 にするなど適宜対応
            iv = 0.0

        return iv


def simulate_option_prices(
    df: pd.DataFrame,
    strike_prices: list,
    expiry_date: pd.Timestamp,
    risk_free_rate: float = 0.03,
    option_type: str = "call"
) -> pd.DataFrame:
    """
    Simulate option prices for given price data and strike prices

    Args:
        df (pd.DataFrame): Price data with at least 'close' and 'start_at' columns
        strike_prices (list): List of strike prices to simulate
        expiry_date (pd.Timestamp): Expiration date of the options
        risk_free_rate (float): Risk-free interest rate
        option_type (str): Type of option to simulate ("call" or "put")

    Returns:
        pd.DataFrame: DataFrame with option prices and Greeks for the specified option type
    """
    if option_type not in ["call", "put"]:
        raise ValueError("option_type must be either 'call' or 'put'")

    results = []

    for _, row in df.iterrows():
        current_price = row['close']
        current_date = pd.to_datetime(row['start_at'])

        days_to_expiry = (expiry_date - current_date).days
        if days_to_expiry <= 0:
            continue

        calculator = BlackScholesCalculator(risk_free_rate=risk_free_rate)
        #volatility = calculator.calculate_volatility(df)
        volatility = float(0.6019)
        
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

            # シミュレーションなので "理論価格" を使っていますが、
            # もし実際の '市場価格' が別にある場合は適宜差し替えて使用します。
            if option_type == "call":
                price = call_price
                delta = call_delta
                theta = call_theta
            else:
                price = put_price
                delta = put_delta
                theta = put_theta

            result = {
                'timestamp': row['start_at'],
                'current_price': current_price,
                'strike_price': strike_price,
                'gamma': gamma,
                'vega': vega,
                'volatility': volatility,
                'days_to_expiry': days_to_expiry,
                'price': price,
                'delta': delta,
                'theta': theta
            }

            results.append(result)

    return pd.DataFrame(results)

import numpy as np
from scipy.stats import norm
from typing import Tuple
import pandas as pd

class BlackScholesCalculator:
    def __init__(self, risk_free_rate: float = 0.03, days_to_expiry: int = 30):
        """
        Initialize Black-Scholes calculator

        Args:
            risk_free_rate (float): Risk-free interest rate (annual)
            days_to_expiry (int): Days until option expiry
        """
        self.risk_free_rate = risk_free_rate
        self.days_to_expiry = days_to_expiry

    def calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate historical volatility from price data"""
        # Calculate daily returns
        returns = np.log(df['close'] / df['close'].shift(1))
        # Calculate annualized volatility
        volatility = returns.std() * np.sqrt(365 * 24)  # Annualized from hourly data
        return volatility

    def calculate_option_prices(self,
                              current_price: float,
                              strike_price: float,
                              volatility: float) -> Tuple[float, float]:
        """
        Calculate call and put option prices using Black-Scholes model

        Args:
            current_price (float): Current price of the underlying asset
            strike_price (float): Strike price of the option
            volatility (float): Annualized volatility

        Returns:
            Tuple[float, float]: Call and put option prices
        """
        T = self.days_to_expiry / 365  # Time to expiry in years

        d1 = (np.log(current_price / strike_price) +
              (self.risk_free_rate + volatility**2/2) * T) / (volatility * np.sqrt(T))
        d2 = d1 - volatility * np.sqrt(T)

        # Calculate call option price
        call_price = current_price * norm.cdf(d1) - \
                    strike_price * np.exp(-self.risk_free_rate * T) * norm.cdf(d2)

        # Calculate put option price using put-call parity
        put_price = call_price + strike_price * np.exp(-self.risk_free_rate * T) - current_price

        return call_price, put_price

def simulate_option_prices(df: pd.DataFrame,
                         strike_prices: list,
                         risk_free_rate: float = 0.03,
                         days_to_expiry: int = 30) -> pd.DataFrame:
    """
    Simulate option prices for given price data and strike prices

    Args:
        df (pd.DataFrame): Price data with OHLCV columns
        strike_prices (list): List of strike prices to simulate
        risk_free_rate (float): Risk-free interest rate
        days_to_expiry (int): Days until option expiry

    Returns:
        pd.DataFrame: DataFrame with option prices for each strike price
    """
    calculator = BlackScholesCalculator(risk_free_rate, days_to_expiry)
    volatility = calculator.calculate_volatility(df)

    results = []

    for _, row in df.iterrows():
        current_price = row['close']

        for strike_price in strike_prices:
            call_price, put_price = calculator.calculate_option_prices(
                current_price, strike_price, volatility
            )

            results.append({
                'timestamp': row['start_at'],
                'current_price': current_price,
                'strike_price': strike_price,
                'call_price': call_price,
                'put_price': put_price,
                'volatility': volatility
            })

    return pd.DataFrame(results)
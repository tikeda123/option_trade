import os
import sys
from typing import Tuple
import numpy as np


# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the path of the parent directory to sys.path
sys.path.append(parent_dir)


from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import MARKET_DATA
from option_pricing import simulate_option_prices


def main():
        start_date = "2024-08-01 00:00:00"
        end_date = "2024-09-01 00:00:00"

        # Load market data
        db = MongoDataLoader()
        df = db.load_data_from_datetime_period(start_date, end_date, MARKET_DATA, symbol="BTCUSDT", interval=60)

        # Generate strike prices (±10% around current price)
        current_price = df['close'].iloc[-1]
        strike_prices = [
            current_price * (1 + x/100)
            for x in range(-10, 11, 2)  # -10%, -8%, ..., +8%, +10%
        ]

        # Simulate option prices
        option_prices = simulate_option_prices(
            df,
            strike_prices=strike_prices,
            risk_free_rate=0.03,  # 3% annual risk-free rate
            days_to_expiry=30     # 30 days to expiry
        )

        # Print results
        print("\nOption Price Simulation Results:")
        print("================================")
        latest_prices = option_prices[option_prices['timestamp'] == option_prices['timestamp'].max()]
        for _, row in latest_prices.iterrows():
            print(f"\nStrike Price: {row['strike_price']:.2f} USDT")
            print(f"Call Option Premium: {row['call_price']:.2f} USDT")
            print(f"Put Option Premium: {row['put_price']:.2f} USDT")

        print(f"\nImplied Volatility: {latest_prices['volatility'].iloc[0]*100:.2f}%")


if __name__ == '__main__':
        main()


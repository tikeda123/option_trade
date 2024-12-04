import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the path of the parent directory to sys.path
sys.path.append(parent_dir)

from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import MARKET_DATA



class BTCPriceIVVisualizer:
    def __init__(self):
        self.plt_style()
        self.db = MongoDataLoader()

    def plt_style(self):
        sns.set_style("darkgrid")
        plt.rcParams['figure.figsize'] = [15, 10]
        plt.rcParams['font.size'] = 12

    def get_historical_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical BTC price data from MongoDB"""
        df = self.db.load_data_from_datetime_period(
            start_date,
            end_date,
            MARKET_DATA,
            symbol="BTCUSDT",
            interval=60
        )
        return df

    def calculate_rolling_volatility(self, df: pd.DataFrame, window: int = 24) -> pd.Series:
        """Calculate rolling historical volatility (24-hour window)"""
        returns = np.log(df['close'] / df['close'].shift(1))
        volatility = returns.rolling(window=window).std() * np.sqrt(365 * 24)  # Annualize from hourly data
        return volatility * 100  # Convert to percentage

    def plot_price_and_iv(self, df: pd.DataFrame):
        """Plot BTC price and IV over time"""
        # Calculate rolling volatility
        iv = self.calculate_rolling_volatility(df)

        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(15, 8))

        # Plot BTC price
        color = 'blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('BTC Price (USDT)', color=color)
        line1 = ax1.plot(df['start_at'], df['close'], color=color, label='BTC Price')
        ax1.tick_params(axis='y', labelcolor=color)

        # Create second y-axis for IV
        ax2 = ax1.twinx()
        color = 'red'
        ax2.set_ylabel('Implied Volatility (%)', color=color)
        line2 = ax2.plot(df['start_at'], iv, color=color, label='IV', alpha=0.7)
        ax2.tick_params(axis='y', labelcolor=color)

        # Add title
        plt.title('BTC Price and Implied Volatility Over Time')

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')

        # Print summary statistics
        print("\n=== Summary Statistics ===")
        print(f"\nBTC Price:")
        print(f"Start: {df['close'].iloc[0]:,.2f} USDT")
        print(f"End: {df['close'].iloc[-1]:,.2f} USDT")
        print(f"Min: {df['close'].min():,.2f} USDT")
        print(f"Max: {df['close'].max():,.2f} USDT")

        print(f"\nImplied Volatility:")
        print(f"Start: {iv.iloc[24]:,.2f}%")  # Start from 24th hour due to rolling window
        print(f"End: {iv.iloc[-1]:,.2f}%")
        print(f"Min: {iv.dropna().min():,.2f}%")
        print(f"Max: {iv.dropna().max():,.2f}%")

        # Save plot
        plt.tight_layout()
        plt.savefig('btc_price_and_iv.png')
        plt.close()

def main():
    # Get data for January 2024
    start_date = "2024-01-01 00:00:00"
    end_date = "2024-02-01 00:00:00"

    visualizer = BTCPriceIVVisualizer()
    df = visualizer.get_historical_data(start_date, end_date)

    # Create visualization
    visualizer.plot_price_and_iv(df)

if __name__ == '__main__':
    main()
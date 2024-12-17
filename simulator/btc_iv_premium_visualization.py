import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
from datetime import datetime

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the path of the parent directory to sys.path
sys.path.append(parent_dir)

from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import MARKET_DATA
from simulator.option_pricing import BlackScholesCalculator


class BTCIVPremiumVisualizer:
    def __init__(self):
        self.plt_style()
        self.db = MongoDataLoader()
        self.calculator = BlackScholesCalculator(risk_free_rate=0.03, days_to_expiry=30)

    def plt_style(self):
        sns.set_style("darkgrid")
        plt.rcParams['figure.figsize'] = [15, 10]
        plt.rcParams['font.size'] = 12

    def get_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get BTC market data from MongoDB. Only price data is available."""
        df = self.db.load_data_from_datetime_period(
            start_date,
            end_date,
            MARKET_DATA,
            symbol="BTCUSDT",
            interval=60
        )
        # IVデータが無いので、仮のIV(20%)を適用する
        df['iv'] = 0.20  # 0.20は20%に相当するIV
        return df

    def calculate_option_data(self, df: pd.DataFrame) -> pd.Series:
        """Use the provided IV (implied volatility) column to calculate ATM option prices."""
        call_prices = []
        iv_series = df['iv']

        for i, row in df.iterrows():
            current_price = row['close']
            iv = iv_series.loc[i]
            if pd.isna(iv):
                # IVがNaNの場合、直近の有効な値を使うなどのフォールバックも可能
                iv = iv_series.dropna().iloc[0]

            # ATMコールオプション価格を計算
            call_price, _ = self.calculator.calculate_option_prices(
                current_price,
                current_price,  # ATM strike price
                iv
            )
            call_prices.append(call_price)

        return pd.Series(call_prices, index=df.index)

    def plot_iv_and_premium(self, df: pd.DataFrame):
        """Plot the artificially set Implied Volatility (IV) and ATM call option premium over time"""
        # プレミアム計算
        premium_series = self.calculate_option_data(df)
        iv_series = df['iv']

        # プロット
        fig, ax1 = plt.subplots(figsize=(15, 8))

        # IVプロット
        color = 'blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Implied Volatility (%)', color=color)
        line1 = ax1.plot(df['start_at'], iv_series * 100, color=color, label='IV')  # 20%など
        ax1.tick_params(axis='y', labelcolor=color)

        # プレミアムプロット
        ax2 = ax1.twinx()
        color = 'red'
        ax2.set_ylabel('Option Premium (USDT)', color=color)
        line2 = ax2.plot(df['start_at'], premium_series, color=color, label='Premium (ATM Call)', alpha=0.7)
        ax2.tick_params(axis='y', labelcolor=color)

        # タイトル・凡例
        plt.title('BTC Implied Volatility (Fixed) and ATM Call Option Premium Over Time')
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')

        # 統計情報出力
        print("\n=== Summary Statistics ===")
        print("\nImplied Volatility (fixed 20% in this example):")
        print(f"Start: {iv_series.iloc[0]*100:.2f}%")
        print(f"End: {iv_series.iloc[-1]*100:.2f}%")
        print(f"Min: {iv_series.min()*100:.2f}%")
        print(f"Max: {iv_series.max()*100:.2f}%")

        print(f"\nATM Call Option Premium:")
        print(f"Start: {premium_series.iloc[0]:,.2f} USDT")
        print(f"End: {premium_series.iloc[-1]:,.2f} USDT")
        print(f"Min: {premium_series.min():,.2f} USDT")
        print(f"Max: {premium_series.max():,.2f} USDT")

        # プロット保存
        plt.tight_layout()
        plt.savefig('btc_iv_and_premium.png')
        plt.close()

def main():
    # 例として2024年1月のデータを取得
    start_date = "2024-01-01 00:00:00"
    end_date = "2024-02-01 00:00:00"

    visualizer = BTCIVPremiumVisualizer()
    df = visualizer.get_market_data(start_date, end_date)
    # df['iv'] は get_market_data で 0.20 に設定済み

    # グラフ生成
    visualizer.plot_iv_and_premium(df)

if __name__ == '__main__':
    main()

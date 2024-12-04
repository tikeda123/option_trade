import numpy as np
import matplotlib.pyplot as plt
from option_pricing import BlackScholesCalculator
import seaborn as sns
from typing import List, Tuple, Dict
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

class OptionVisualizer:
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

    def calculate_historical_volatility(self, df: pd.DataFrame, window: int = 24) -> pd.Series:
        """Calculate rolling historical volatility from price data"""
        returns = np.log(df['close'] / df['close'].shift(1))
        volatility = returns.rolling(window=window).std() * np.sqrt(365 * 24)
        return volatility

    def simulate_fixed_strike_premiums(self,
                                     df: pd.DataFrame,
                                     strike_prices: Dict[str, float],
                                     days_to_expiry: int = 30):
        """
        指定された行使価格でのオプションプレミアムの時間変化をシミュレーション

        Args:
            df: 価格データ
            strike_prices: 行使価格の辞書 {'label': strike_price, ...}
            days_to_expiry: 満期までの日数
        """
        # 移動平均ボラティリティを計算
        volatilities = self.calculate_historical_volatility(df)

        # 結果を格納するDataFrame
        results = pd.DataFrame()
        results['timestamp'] = df['start_at']
        results['price'] = df['close']

        calculator = BlackScholesCalculator(risk_free_rate=0.03, days_to_expiry=days_to_expiry)

        for label, strike_price in strike_prices.items():
            call_premiums = []
            put_premiums = []

            for i in range(len(df)):
                if pd.isna(volatilities.iloc[i]):
                    call_premiums.append(np.nan)
                    put_premiums.append(np.nan)
                    continue

                # 残存日数を計算（データポイントが1時間ごとなので、24で割る）
                remaining_days = days_to_expiry - (i // 24)
                if remaining_days <= 0:
                    call_premiums.append(np.nan)
                    put_premiums.append(np.nan)
                    continue

                calculator.days_to_expiry = remaining_days
                call, put = calculator.calculate_option_prices(
                    df['close'].iloc[i],
                    strike_price,
                    volatilities.iloc[i]
                )
                call_premiums.append(call)
                put_premiums.append(put)

            results[f'call_{label}'] = call_premiums
            results[f'put_{label}'] = put_premiums

        return results, strike_prices

    def print_premium_comparison(self, results: pd.DataFrame, strike_prices: dict, contract_size: float = 1.0):
        """初日と最終日のプレミアム価格を表示

        Args:
            results: シミュレーション結果のDataFrame
            strike_prices: 行使価格の辞書
            contract_size: 取引数量（BTC）
        """
        # 最初の有効なデータポイントを見つける
        first_valid_idx = results.iloc[24:48][['call_' + list(strike_prices.keys())[0],
                                             'put_' + list(strike_prices.keys())[0]]].first_valid_index()
        # 最後の有効なデータポイントを見つける
        last_valid_idx = results[['call_' + list(strike_prices.keys())[0],
                                'put_' + list(strike_prices.keys())[0]]].last_valid_index()

        if first_valid_idx is not None and last_valid_idx is not None:
            initial_data = results.loc[first_valid_idx]
            final_data = results.loc[last_valid_idx]

            print("\n=== オプションプレミアムの比較 ===")
            print(f"取引数量: {contract_size} BTC")
            print(f"\n初日 ({initial_data['timestamp']})")
            print(f"BTC価格: {initial_data['price']:,.2f} USDT")

            print("\nコールオプション:")
            for label, strike in strike_prices.items():
                premium_per_btc = initial_data[f'call_{label}']
                total_premium = premium_per_btc * contract_size
                print(f"{label} (Strike: {strike:,.2f})")
                print(f"  プレミアム/BTC: {premium_per_btc:,.2f} USDT")
                print(f"  総プレミアム: {total_premium:,.2f} USDT")

            print("\nプットオプション:")
            for label, strike in strike_prices.items():
                premium_per_btc = initial_data[f'put_{label}']
                total_premium = premium_per_btc * contract_size
                print(f"{label} (Strike: {strike:,.2f})")
                print(f"  プレミアム/BTC: {premium_per_btc:,.2f} USDT")
                print(f"  総プレミアム: {total_premium:,.2f} USDT")

            print(f"\n最終日 ({final_data['timestamp']})")
            print(f"BTC価格: {final_data['price']:,.2f} USDT")

            print("\nコールオプション:")
            for label, strike in strike_prices.items():
                premium_per_btc = final_data[f'call_{label}']
                total_premium = premium_per_btc * contract_size
                print(f"{label} (Strike: {strike:,.2f})")
                print(f"  プレミアム/BTC: {premium_per_btc:,.2f} USDT")
                print(f"  総プレミアム: {total_premium:,.2f} USDT")

            print("\nプットオプション:")
            for label, strike in strike_prices.items():
                premium_per_btc = final_data[f'put_{label}']
                total_premium = premium_per_btc * contract_size
                print(f"{label} (Strike: {strike:,.2f})")
                print(f"  プレミアム/BTC: {premium_per_btc:,.2f} USDT")
                print(f"  総プレミアム: {total_premium:,.2f} USDT")

            # 価格変化の計算
            print("\n=== プレミアム価格の変化 ===")
            print("\nコールオプション:")
            for label in strike_prices.keys():
                change_per_btc = final_data[f'call_{label}'] - initial_data[f'call_{label}']
                total_change = change_per_btc * contract_size
                change_pct = (change_per_btc / initial_data[f'call_{label}']) * 100
                print(f"{label}:")
                print(f"  変化額/BTC: {change_per_btc:+,.2f} USDT ({change_pct:+.2f}%)")
                print(f"  総変化額: {total_change:+,.2f} USDT")

            print("\nプットオプション:")
            for label in strike_prices.keys():
                change_per_btc = final_data[f'put_{label}'] - initial_data[f'put_{label}']
                total_change = change_per_btc * contract_size
                change_pct = (change_per_btc / initial_data[f'put_{label}']) * 100
                print(f"{label}:")
                print(f"  変化額/BTC: {change_per_btc:+,.2f} USDT ({change_pct:+.2f}%)")
                print(f"  総変化額: {total_change:+,.2f} USDT")

    def plot_fixed_strike_premiums(self,
                                 df: pd.DataFrame,
                                 strike_prices: Dict[str, float],
                                 days_to_expiry: int = 30,
                                 contract_size: float = 1.0):
        """固定ストライク価格でのオプションプレミアムの推移をプロット"""
        results, strike_prices = self.simulate_fixed_strike_premiums(df, strike_prices, days_to_expiry)

        # プレミアム価格の比較を表示
        self.print_premium_comparison(results, strike_prices, contract_size)

        # プロットの作成
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

        # カラーマップの作成
        colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(strike_prices)))

        # コールオプションのプロット
        for (label, strike), color in zip(strike_prices.items(), colors):
            ax1.plot(results['timestamp'], results[f'call_{label}'],
                    label=f'{label} (Strike: {strike:,.0f})', color=color)

        # BTCの価格を同じ軸に表示（コール）
        ax1_price = ax1.twinx()
        ax1_price.plot(results['timestamp'], results['price'], label='BTC Price', color='blue', linestyle='--')
        ax1_price.set_ylabel('BTC Price (USDT)', color='blue')

        # コールオプションのグラフ設定
        ax1.set_title(f'Call Option Premiums Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Premium (USDT)')

        # 凡例を結合
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_price.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        # カラーマップの作成（プット用）
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(strike_prices)))

        # プットオプションのプロット
        for (label, strike), color in zip(strike_prices.items(), colors):
            ax2.plot(results['timestamp'], results[f'put_{label}'],
                    label=f'{label} (Strike: {strike:,.0f})', color=color)

        # BTCの価格を同じ軸に表示（プット）
        ax2_price = ax2.twinx()
        ax2_price.plot(results['timestamp'], results['price'], label='BTC Price', color='blue', linestyle='--')
        ax2_price.set_ylabel('BTC Price (USDT)', color='blue')

        # プットオプションのグラフ設定
        ax2.set_title('Put Option Premiums Over Time')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Premium (USDT)')

        # 凡例を結合
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_price.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.tight_layout()
        plt.savefig('fixed_strike_premiums.png')
        plt.close()

def main():
    # 2024年1月1日から30日間のデータを取得
    start_date = "2024-01-01 00:00:00"
    end_date = "2024-02-01 00:00:00"

    visualizer = OptionVisualizer()
    df = visualizer.get_historical_data(start_date, end_date)

    # 行使価格を手動で設定
    strike_prices = {
        'Strike_45K': 45000.0,
        'Strike_50K': 50000.0,
        'Strike_55K': 55000.0,
        'Strike_60K': 60000.0,
        'Strike_65K': 65000.0,
    }

    # 取引数量を設定（例: 0.1 BTC）
    contract_size = 0.1

    # 固定ストライク価格でのオプションプレミアムの推移をプロット
    visualizer.plot_fixed_strike_premiums(df, strike_prices, days_to_expiry=30, contract_size=contract_size)

if __name__ == '__main__':
    main()
import numpy as np
import pandas as pd
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class MultiIndicatorTrendDetector:
    def __init__(self, observation_covariance=1.0, process_noise=0.01):
        """
        複数指標とカルマンフィルタを用いたトレンド検出器

        Parameters:
        -----------
        observation_covariance : float
            観測ノイズの大きさ。大きいほどフィルタリングが強くなる
        process_noise : float
            プロセスノイズの大きさ。大きいほど新しい観測値に素早く適応する
        """
        # カルマンフィルタのパラメータ
        self.transition_matrix = np.array([[1, 1], [0, 1]])  # 状態遷移行列 [位置, 速度]
        self.observation_matrix = np.array([[1, 0]])  # 観測行列
        self.initial_state_mean = np.array([0, 0])  # 初期状態 [位置, 速度]
        self.initial_state_covariance = np.array([[1, 0], [0, 1]])  # 初期共分散
        self.observation_covariance = observation_covariance  # 観測ノイズ
        self.transition_covariance = np.array([[process_noise, 0], [0, process_noise]])  # プロセスノイズ

    def prepare_crypto_indicators(self, df):
        """
        Process technical indicators provided from cryptocurrency data

        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing columns like 'close', 'rsi', 'macdhist', 'mfi', 'roc', etc.

        Returns:
        --------
        pandas.DataFrame
            DataFrame containing normalized composite indicator
        """
        # Create a copy to avoid modifying the original data
        indicators = pd.DataFrame(index=df.index)

        # Get existing technical indicators
        available_indicators = []

        if 'close' in df.columns:
            indicators['close'] = df['close']
            available_indicators.append('close')

            # Add price momentum
            indicators['price_momentum'] = df['close'].pct_change(periods=14)
            available_indicators.append('price_momentum')

        # Use ROC instead of SMA if available
        if 'roc' in df.columns:
            indicators['roc'] = df['roc']
            available_indicators.append('roc')

        # Use other existing technical indicators
        if 'rsi' in df.columns:
            indicators['rsi'] = df['rsi']
            available_indicators.append('rsi')

            # Add RSI slope (rate of change)
            indicators['rsi_slope'] = df['rsi'].diff(3)
            available_indicators.append('rsi_slope')

        if 'macdhist' in df.columns:
            indicators['macdhist'] = df['macdhist']
            available_indicators.append('macdhist')

        if 'mfi' in df.columns:
            indicators['mfi'] = df['mfi']
            available_indicators.append('mfi')

        if 'volatility' in df.columns:
            indicators['volatility'] = df['volatility']
            available_indicators.append('volatility')

        # Remove missing values
        indicators = indicators.dropna()

        # Normalize indicators
        normalized = pd.DataFrame(index=indicators.index)
        for col in available_indicators:
            # Only normalize if the indicator doesn't have all the same values
            if indicators[col].std() > 0:
                normalized[col] = (indicators[col] - indicators[col].mean()) / indicators[col].std()
            else:
                normalized[col] = 0

        # Create composite indicator
        indicators['composite_indicator'] = normalized.mean(axis=1)

        return indicators

    def apply_kalman_filter(self, composite_indicator):
        """
        カルマンフィルタを合成指標に適用

        Parameters:
        -----------
        composite_indicator : pandas.Series
            合成指標の時系列データ

        Returns:
        --------
        tuple
            (filtered_levels, filtered_trends)
            filtered_levels: 推定された価格レベル
            filtered_trends: 推定されたトレンド（速度）
        """
        kf = KalmanFilter(
            transition_matrices=self.transition_matrix,
            observation_matrices=self.observation_matrix,
            initial_state_mean=self.initial_state_mean,
            initial_state_covariance=self.initial_state_covariance,
            observation_covariance=self.observation_covariance,
            transition_covariance=self.transition_covariance
        )

        # カルマンフィルタの適用
        state_means, state_covariances = kf.filter(composite_indicator.values)

        # 位置（レベル）と速度（トレンド）を抽出
        levels = state_means[:, 0]
        trends = state_means[:, 1]

        return levels, trends

    def detect_trend(self, df, threshold=0.02):
        """
        Detect market trend

        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing technical indicators like 'close', 'rsi', etc.
        threshold : float
            Threshold for trend determination. Higher values make detection more conservative

        Returns:
        --------
        pandas.DataFrame
            DataFrame containing trend analysis results
        """
        # Prepare indicators
        indicators = self.prepare_crypto_indicators(df)
        composite = indicators['composite_indicator']

        # Apply Kalman filter
        filtered_levels, filtered_trends = self.apply_kalman_filter(composite)

        # Determine trend state
        trend_states = []
        for trend in filtered_trends:
            if trend > threshold:
                trend_states.append('Uptrend')  # 上昇トレンド in English
            elif trend < -threshold:
                trend_states.append('Downtrend')  # 下降トレンド in English
            else:
                trend_states.append('Range-bound')  # レンジ相場 in English

        # Create result DataFrame
        if 'close' in df.columns:
            price_data = df.loc[indicators.index, 'close']
        else:
            price_data = pd.Series(np.zeros(len(indicators)), index=indicators.index)

        return pd.DataFrame({
            'price': price_data.values,
            'composite_indicator': composite.values,
            'filtered_level': filtered_levels,
            'filtered_trend': filtered_trends,
            'trend_state': trend_states
        }, index=indicators.index)

    def visualize(self, results, title="Bitcoin/USDT Trend Analysis"):
        """
        Visualize the results

        Parameters:
        -----------
        results : pandas.DataFrame
            DataFrame returned from detect_trend()
        title : str
            Title for the graph
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

        # Date format setting
        date_format = mdates.DateFormatter('%Y-%m-%d')

        # Price chart and filtered level
        ax1.plot(results.index, results['price'], label='Price', alpha=0.7)
        ax1.plot(results.index, results['filtered_level'] * results['price'].mean(),
                 label='Filtered Trend', color='red', linewidth=2)

        # Background color based on trend state
        prev_state = None
        start_idx = 0

        for i in range(len(results)):
            current_state = results['trend_state'].iloc[i]

            # When state changes or at the last data point
            if prev_state is not None and (current_state != prev_state or i == len(results) - 1):
                if prev_state == 'Uptrend':
                    ax1.axvspan(results.index[start_idx], results.index[i], alpha=0.2, color='green')
                elif prev_state == 'Downtrend':
                    ax1.axvspan(results.index[start_idx], results.index[i], alpha=0.2, color='red')
                else:  # Range-bound
                    ax1.axvspan(results.index[start_idx], results.index[i], alpha=0.1, color='gray')
                start_idx = i

            # First time or when state changes
            if prev_state is None or current_state != prev_state:
                prev_state = current_state

        ax1.set_title('Price Chart and Kalman Filter Estimated Trend')
        ax1.legend(loc='upper left')
        ax1.xaxis.set_major_formatter(date_format)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel('Price')

        # Composite indicator
        ax2.plot(results.index, results['composite_indicator'], label='Composite Technical Indicator', color='purple')
        ax2.set_title('Composite Indicator Calculated from Multiple Technical Indicators')
        ax2.legend(loc='upper left')
        ax2.xaxis.set_major_formatter(date_format)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylabel('Normalized Value')

        # Filtered trend
        ax3.plot(results.index, results['filtered_trend'], label='Kalman Filter Estimated Trend', color='blue')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # Threshold lines - updated to match the new threshold value
        threshold = 0.02  # Same threshold used in detect_trend
        ax3.axhline(y=threshold, color='green', linestyle='--', alpha=0.5, label='Uptrend Threshold')
        ax3.axhline(y=-threshold, color='red', linestyle='--', alpha=0.5, label='Downtrend Threshold')

        ax3.set_title('Trend Estimation by Kalman Filter')
        ax3.legend(loc='upper left')
        ax3.xaxis.set_major_formatter(date_format)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylabel('Trend Velocity')
        ax3.set_xlabel('Date')

        # Overall title
        plt.suptitle(title, fontsize=16)

        # Adjust date labels
        fig.autofmt_xdate()

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()

        # Display trend statistics
        trend_stats = results['trend_state'].value_counts()
        print(f"Trend Statistics:")
        print(f"Uptrend: {trend_stats.get('Uptrend', 0)} periods ({trend_stats.get('Uptrend', 0)/len(results)*100:.1f}%)")
        print(f"Downtrend: {trend_stats.get('Downtrend', 0)} periods ({trend_stats.get('Downtrend', 0)/len(results)*100:.1f}%)")
        print(f"Range-bound: {trend_stats.get('Range-bound', 0)} periods ({trend_stats.get('Range-bound', 0)/len(results)*100:.1f}%)")

        # Analyze trend transitions
        transitions = []
        for i in range(1, len(results)):
            if results['trend_state'].iloc[i] != results['trend_state'].iloc[i-1]:
                transitions.append({
                    'date': results.index[i],
                    'from': results['trend_state'].iloc[i-1],
                    'to': results['trend_state'].iloc[i],
                    'price': results['price'].iloc[i]
                })

        if transitions:
            print("\nLast 5 Trend Transitions:")
            for t in transitions[-5:]:
                print(f"{t['date'].strftime('%Y-%m-%d %H:%M')}: {t['from']} → {t['to']} (Price: {t['price']:.2f})")

# Sample execution code using Bitcoin real data
def run_btc_analysis(data_df):
    """
    Run trend analysis using Bitcoin real data

    Parameters:
    -----------
    data_df : pandas.DataFrame
        Bitcoin time series data retrieved from MongoDB or similar source
    """
    # Preprocessing: Select only necessary columns and set start_at as index
    if 'start_at' in data_df.columns:
        # Convert start_at to Datetime if it's not already
        if not isinstance(data_df['start_at'].iloc[0], (pd.Timestamp, np.datetime64, datetime)):
            data_df['start_at'] = pd.to_datetime(data_df['start_at'])

        # Set as index
        df = data_df.set_index('start_at')
    else:
        df = data_df.copy()

    # Initialize trend detector
    # Note: Cryptocurrencies have high volatility, so parameter adjustment is necessary
    detector = MultiIndicatorTrendDetector(
        observation_covariance=0.5,  # Lower values make it more sensitive to observations
        process_noise=0.01  # Higher values make it more sensitive to changes
    )

    # Execute trend detection
    results = detector.detect_trend(df, threshold=0.03)  # Updated threshold is more conservative

    # Recent trend states
    print("Recent Trend States:")
    print(results[['trend_state']].tail(10))

    # Visualize results
    detector.visualize(results, title="Bitcoin/USDT Trend Analysis")

    return results

# メイン処理の例
if __name__ == "__main__":
    try:
        # MongoDB connection if available
        import os
        import sys
        from typing import List, Dict, Any
        from datetime import datetime, timedelta

        # Set paths according to user environment
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        sys.path.append(parent_dir)

        from mongodb.data_loader_mongo import MongoDataLoader
        from common.constants import *

        # Get BTC time series data from MongoDB
        db = MongoDataLoader()
        df = db.load_data_from_datetime_period(
            datetime(2024, 1, 1),
            datetime(2025, 1, 1),
            coll_type=MARKET_DATA_TECH,
            symbol='BTCUSDT',
            interval=1440
        )
        #intervalは1440で1日のデータを取得する、60で1時間のデータを取得する
        # Extract only necessary columns
        graph_df = df[['start_at', 'close', 'volume', 'macdhist', 'rsi', 'volatility', 'mfi', 'roc']]

        # Run trend analysis
        run_btc_analysis(graph_df)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please ensure MongoDB connection is properly configured and data is available.")
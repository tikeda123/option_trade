import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from scipy.linalg import inv

class ExtendedKalmanFilterTrendDetector:
    def __init__(self, observation_covariance=1.0, process_noise=0.01):
        """
        Extended Kalman Filter for trend detection using multiple indicators

        Parameters:
        -----------
        observation_covariance : float
            Observation noise magnitude. Larger values result in stronger filtering
        process_noise : float
            Process noise magnitude. Larger values adapt more quickly to new observations
        """
        # Dimension of state vector [position, velocity, acceleration]
        self.state_dim = 3

        # Initial state mean [position, velocity, acceleration]
        self.initial_state_mean = np.zeros(self.state_dim)

        # Initial state covariance
        self.initial_state_covariance = np.eye(self.state_dim)

        # Process noise covariance
        self.process_covariance = np.eye(self.state_dim) * process_noise

        # Observation noise
        self.observation_covariance = observation_covariance

    def f(self, x, dt=1.0):
        """
        Non-linear state transition function
        x[0] = position, x[1] = velocity, x[2] = acceleration

        Parameters:
        -----------
        x : numpy.ndarray
            Current state vector
        dt : float
            Time step

        Returns:
        --------
        numpy.ndarray
            Next state vector
        """
        # Non-linear state transition model
        # The acceleration component includes a damping term to model mean reversion
        damping = 0.1  # Damping factor

        next_x = np.zeros_like(x)
        next_x[0] = x[0] + x[1] * dt + 0.5 * x[2] * dt**2  # Position update
        next_x[1] = x[1] + x[2] * dt  # Velocity update
        next_x[2] = x[2] * (1 - damping)  # Acceleration with damping (mean reversion)

        return next_x

    def F_jacobian(self, x, dt=1.0):
        """
        Jacobian of the state transition function

        Parameters:
        -----------
        x : numpy.ndarray
            Current state vector
        dt : float
            Time step

        Returns:
        --------
        numpy.ndarray
            Jacobian matrix of state transition function
        """
        # Jacobian of state transition function
        F = np.array([
            [1, dt, 0.5 * dt**2],
            [0, 1, dt],
            [0, 0, 0.9]  # 0.9 is (1 - damping) from the f function
        ])
        return F

    def h(self, x):
        """
        Non-linear measurement function

        Parameters:
        -----------
        x : numpy.ndarray
            State vector

        Returns:
        --------
        numpy.ndarray
            Predicted measurement
        """
        # In this simple case, we only observe the position (first state)
        return np.array([x[0]])

    def H_jacobian(self, x):
        """
        Jacobian of the measurement function

        Parameters:
        -----------
        x : numpy.ndarray
            State vector

        Returns:
        --------
        numpy.ndarray
            Jacobian matrix of measurement function
        """
        # Jacobian of measurement function
        # We only observe the position (first state)
        H = np.array([[1, 0, 0]])
        return H

    def extended_kalman_filter(self, observations):
        """
        Apply Extended Kalman Filter to a series of observations

        Parameters:
        -----------
        observations : numpy.ndarray
            Series of observations

        Returns:
        --------
        tuple
            (filtered_states, state_covariances)
        """
        n = len(observations)

        # Initialize state and covariance
        x = self.initial_state_mean.copy()
        P = self.initial_state_covariance.copy()

        # Arrays to store results
        filtered_states = np.zeros((n, self.state_dim))
        state_covariances = np.zeros((n, self.state_dim, self.state_dim))

        for t in range(n):
            # Prediction step
            F = self.F_jacobian(x)
            x_pred = self.f(x)
            P_pred = F @ P @ F.T + self.process_covariance

            # Update step
            z = observations[t]
            H = self.H_jacobian(x_pred)
            y = z - self.h(x_pred)  # Measurement residual
            S = H @ P_pred @ H.T + self.observation_covariance  # Residual covariance
            K = P_pred @ H.T @ inv(S)  # Optimal Kalman gain

            # Update state and covariance
            x = x_pred + K @ y
            P = (np.eye(self.state_dim) - K @ H) @ P_pred

            # Store results
            filtered_states[t] = x
            state_covariances[t] = P

        return filtered_states, state_covariances

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

    def detect_trend(self, df, threshold=0.02):
        """
        Detect market trend using Extended Kalman Filter

        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing technical indicators
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

        # Apply Extended Kalman Filter
        filtered_states, state_covariances = self.extended_kalman_filter(composite.values.reshape(-1, 1))

        # Extract position (level), velocity (trend), and acceleration from the state vector
        filtered_levels = filtered_states[:, 0]
        filtered_trends = filtered_states[:, 1]
        filtered_accelerations = filtered_states[:, 2]

        # Determine trend state
        trend_states = []
        for trend, acceleration in zip(filtered_trends, filtered_accelerations):
            # Use both trend (velocity) and acceleration for more robust trend detection
            if trend > threshold and acceleration >= 0:
                trend_states.append('Uptrend')  # Strong uptrend with positive or stable acceleration
            elif trend < -threshold and acceleration <= 0:
                trend_states.append('Downtrend')  # Strong downtrend with negative or stable acceleration
            elif abs(trend) < threshold * 0.5:
                trend_states.append('Range-bound')  # Very low velocity indicates range-bound market
            elif trend > 0:
                # Positive trend but either weak or decelerating
                if acceleration < -threshold * 2:
                    trend_states.append('Slowing Uptrend')  # Significant deceleration in uptrend
                else:
                    trend_states.append('Weak Uptrend')
            else:
                # Negative trend but either weak or accelerating
                if acceleration > threshold * 2:
                    trend_states.append('Slowing Downtrend')  # Significant deceleration in downtrend
                else:
                    trend_states.append('Weak Downtrend')

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
            'filtered_acceleration': filtered_accelerations,
            'trend_state': trend_states
        }, index=indicators.index)

    def visualize(self, results, title="Bitcoin/USDT Trend Analysis with Extended Kalman Filter"):
        """
        Visualize the results

        Parameters:
        -----------
        results : pandas.DataFrame
            DataFrame returned from detect_trend()
        title : str
            Title for the graph
        """
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 16), sharex=True)

        # Date format setting
        date_format = mdates.DateFormatter('%Y-%m-%d')

        # Price chart (removed Kalman Filter line as requested)
        ax1.plot(results.index, results['price'], label='Price', color='black', linewidth=1.5)

        # Background color based on trend state with improved visibility
        prev_state = None
        start_idx = 0

        # Color mapping for different trend states
        colors = {
            'Uptrend': 'green',
            'Downtrend': 'red',
            'Range-bound': 'gray',
            'Weak Uptrend': 'lightgreen',
            'Weak Downtrend': 'lightcoral',
            'Slowing Uptrend': 'yellowgreen',
            'Slowing Downtrend': 'lightsalmon'
        }

        for i in range(len(results)):
            current_state = results['trend_state'].iloc[i]

            # When state changes or at the last data point
            if prev_state is not None and (current_state != prev_state or i == len(results) - 1):
                color = colors.get(prev_state, 'gray')
                # Increased alpha for better visibility of trend zones
                alpha = 0.3 if prev_state in ['Uptrend', 'Downtrend'] else 0.2
                ax1.axvspan(results.index[start_idx], results.index[i], alpha=alpha, color=color)
                start_idx = i

            # First time or when state changes
            if prev_state is None or current_state != prev_state:
                prev_state = current_state

        # Add legend for trend states
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[state], alpha=0.3 if state in ['Uptrend', 'Downtrend'] else 0.2,
                                label=state) for state in colors]
        ax1.legend(handles=legend_elements, loc='upper left', ncol=2)

        ax1.set_title('Price Chart with Trend State Background')
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

        # Filtered trend (velocity)
        ax3.plot(results.index, results['filtered_trend'], label='EKF Estimated Trend (Velocity)', color='blue')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # Threshold lines - updated to match the new threshold value
        threshold = 0.02  # Same threshold used in detect_trend
        ax3.axhline(y=threshold, color='green', linestyle='--', alpha=0.5, label='Uptrend Threshold')
        ax3.axhline(y=-threshold, color='red', linestyle='--', alpha=0.5, label='Downtrend Threshold')

        ax3.set_title('Trend Estimation (Velocity) by Extended Kalman Filter')
        ax3.legend(loc='upper left')
        ax3.xaxis.set_major_formatter(date_format)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylabel('Trend Velocity')

        # Filtered acceleration
        ax4.plot(results.index, results['filtered_acceleration'], label='EKF Estimated Acceleration', color='orange')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # Threshold lines for acceleration
        accel_threshold = threshold * 2
        ax4.axhline(y=accel_threshold, color='green', linestyle='--', alpha=0.5, label='Positive Acceleration')
        ax4.axhline(y=-accel_threshold, color='red', linestyle='--', alpha=0.5, label='Negative Acceleration')

        ax4.set_title('Acceleration Estimation by Extended Kalman Filter')
        ax4.legend(loc='upper left')
        ax4.xaxis.set_major_formatter(date_format)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylabel('Acceleration')
        ax4.set_xlabel('Date')

        # Overall title
        plt.suptitle(title, fontsize=16)

        # Adjust date labels
        fig.autofmt_xdate()

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()

        # Display trend statistics
        trend_stats = results['trend_state'].value_counts()
        print(f"Trend Statistics:")
        for state in sorted(trend_stats.index):
            count = trend_stats[state]
            percentage = count / len(results) * 100
            print(f"{state}: {count} periods ({percentage:.1f}%)")

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

        # Display current market state
        latest_state = results['trend_state'].iloc[-1]
        latest_trend = results['filtered_trend'].iloc[-1]
        latest_accel = results['filtered_acceleration'].iloc[-1]
        print(f"\nCurrent Market State: {latest_state}")
        print(f"Current Trend Velocity: {latest_trend:.6f}")
        print(f"Current Acceleration: {latest_accel:.6f}")

        if latest_accel > 0 and latest_trend > 0:
            print("Market Interpretation: Accelerating uptrend - bullish")
        elif latest_accel < 0 and latest_trend > 0:
            print("Market Interpretation: Decelerating uptrend - potential trend change ahead")
        elif latest_accel > 0 and latest_trend < 0:
            print("Market Interpretation: Decelerating downtrend - potential trend change ahead")
        elif latest_accel < 0 and latest_trend < 0:
            print("Market Interpretation: Accelerating downtrend - bearish")
        elif abs(latest_trend) < threshold:
            print("Market Interpretation: Range-bound market - waiting for directional move")

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
    detector = ExtendedKalmanFilterTrendDetector(
        observation_covariance=0.5,  # Lower values make it more sensitive to observations
        process_noise=0.01  # Higher values make it more sensitive to changes
    )

    # Execute trend detection
    results = detector.detect_trend(df, threshold=0.02)  # Updated threshold is more conservative

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
            datetime(2024, 12, 1),
            datetime(2025, 1, 1),
            coll_type=MARKET_DATA_TECH,
            symbol='BTCUSDT',
            interval=1440
        )

        # Extract only necessary columns
        graph_df = df[['start_at', 'close', 'volume', 'macdhist', 'rsi', 'volatility', 'mfi', 'roc']]

        # Run trend analysis
        run_btc_analysis(graph_df)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please ensure MongoDB connection is properly configured and data is available.")
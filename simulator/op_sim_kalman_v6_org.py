import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from scipy.linalg import inv

# For Ubuntu, default English fonts (e.g., DejaVu Sans) are used.
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False  # Prevent minus sign issues

class ExtendedKalmanFilterTrendDetector:
    def __init__(self, observation_covariance=1.0, process_noise=0.01, damping=0.1):
        """
        Extended Kalman Filter for trend detection using multiple indicators.
        (Example: handling prices on a log-scale and velocity as the difference in log-prices)

        Parameters:
        -----------
        observation_covariance : float
            The magnitude of observation noise. Larger values lead to smoother filter output.
        process_noise : float
            The magnitude of process noise. Larger values make the filter follow new observations more closely.
        damping : float
            Damping coefficient. Determines the fraction by which acceleration decays in the next step.
        """
        self.state_dim = 4
        self.observation_covariance = observation_covariance
        self.process_noise = process_noise
        self.damping = damping

        # Initial state
        self.initial_state_mean = np.zeros(self.state_dim)
        self.initial_state_covariance = np.eye(self.state_dim)
        # Process noise covariance matrix
        self.process_covariance = np.eye(self.state_dim) * self.process_noise

    def f(self, x, dt=1.0):
        """
        Nonlinear state transition function.
        x[0] = log_price, x[1] = log_velocity, x[2] = log_acceleration, x[3] = indicator
        """
        damping = self.damping
        log_price        = x[0]
        log_velocity     = x[1]
        log_acceleration = x[2]
        indicator        = x[3]

        next_log_price = log_price + log_velocity * dt + 0.5 * log_acceleration * (dt**2)
        next_log_velocity = log_velocity + log_acceleration * dt
        next_log_acceleration = log_acceleration * (1 - damping)
        next_indicator = indicator  # Remains unchanged

        return np.array([
            next_log_price,
            next_log_velocity,
            next_log_acceleration,
            next_indicator
        ])

    def F_jacobian(self, x, dt=1.0):
        """
        Jacobian of the state transition function (4x4)
        """
        damping = self.damping
        F = np.array([
            [1,      dt,    0.5*(dt**2), 0],
            [0,      1,     dt,          0],
            [0,      0,     1 - damping, 0],
            [0,      0,     0,           1]
        ])
        return F

    def h(self, x):
        """
        Measurement function (2-dimensional): Observing [log_price, indicator]
        """
        return np.array([x[0], x[3]])

    def H_jacobian(self, x):
        """
        Jacobian of the measurement function (2x4)
        """
        H = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ])
        return H

    def extended_kalman_filter(self, observations):
        """
        Apply the Extended Kalman Filter to the observation sequence.
        observations.shape = (N, 2) where each row is [log_price, composite_indicator]
        """
        n = len(observations)
        x = self.initial_state_mean.copy()
        P = self.initial_state_covariance.copy()

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
            y = z - self.h(x_pred)
            R = np.eye(2) * self.observation_covariance
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ inv(S)

            x = x_pred + K @ y
            P = (np.eye(self.state_dim) - K @ H) @ P_pred

            filtered_states[t] = x
            state_covariances[t] = P

        return filtered_states, state_covariances

    def prepare_crypto_indicators(self, df):
        """
        Example of constructing a composite indicator.
        """
        indicators = pd.DataFrame(index=df.index)
        available_indicators = []

        if 'close' in df.columns:
            indicators['close'] = df['close']
            available_indicators.append('close')
            indicators['price_momentum'] = df['close'].pct_change(periods=14)
            available_indicators.append('price_momentum')

        if 'roc' in df.columns:
            indicators['roc'] = df['roc']
            available_indicators.append('roc')

        if 'rsi' in df.columns:
            indicators['rsi'] = df['rsi']
            available_indicators.append('rsi')
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

        indicators.dropna(inplace=True)

        normalized = pd.DataFrame(index=indicators.index)
        for col in available_indicators:
            if indicators[col].std() > 0:
                normalized[col] = (indicators[col] - indicators[col].mean()) / indicators[col].std()
            else:
                normalized[col] = 0

        indicators['composite_indicator'] = normalized.mean(axis=1)
        return indicators

    def detect_trend(self, df, threshold=0.02):
        """
        Apply the EKF using log(price) and composite indicator, then detect trends.
        A threshold of 0.02 means an approximate 2% change in log-difference.
        """
        indicators = self.prepare_crypto_indicators(df)
        close = indicators['close']
        composite = indicators['composite_indicator']
        log_close = np.log(close)
        obs_data = np.column_stack([log_close.values, composite.values])

        # Apply Kalman filter
        filtered_states, state_covariances = self.extended_kalman_filter(obs_data)

        filtered_log_price        = filtered_states[:, 0]
        filtered_log_velocity     = filtered_states[:, 1]
        filtered_log_acceleration = filtered_states[:, 2]
        filtered_indicator        = filtered_states[:, 3]

        # Trend detection based on filtered states
        trend_states = []
        for vel, acc in zip(filtered_log_velocity, filtered_log_acceleration):
            if vel > threshold and acc >= 0:
                trend_states.append('Uptrend')
            elif vel < -threshold and acc <= 0:
                trend_states.append('Downtrend')
            elif abs(vel) < threshold * 0.5:
                trend_states.append('Range-bound')
            elif vel > 0:
                if acc < -threshold * 2:
                    trend_states.append('Slowing Uptrend')
                else:
                    trend_states.append('Weak Uptrend')
            else:
                if acc > threshold * 2:
                    trend_states.append('Slowing Downtrend')
                else:
                    trend_states.append('Weak Downtrend')

        result_df = pd.DataFrame({
            'price': close.values,
            'composite_indicator': composite.values,
            'filtered_log_price': filtered_log_price,
            'filtered_trend': filtered_log_velocity,
            'filtered_acceleration': filtered_log_acceleration,
            'trend_state': trend_states
        }, index=indicators.index)

        return result_df

    def visualize_price_with_trend(self, results, title="Price with Trend Background"):
        """
        Visualize the price chart with colored background segments indicating trend states.
        """
        colors = {
            'Uptrend': 'green',
            'Downtrend': 'red',
            'Range-bound': 'gray',
            'Weak Uptrend': 'lightgreen',
            'Weak Downtrend': 'lightcoral',
            'Slowing Uptrend': 'yellowgreen',
            'Slowing Downtrend': 'lightsalmon'
        }
        # Descriptions for legend
        color_meanings = {
            'Uptrend':          'Uptrend (log velocity > +threshold, acceleration >= 0)',
            'Downtrend':        'Downtrend (log velocity < -threshold, acceleration <= 0)',
            'Range-bound':      'Range-bound (|log velocity| < threshold*0.5)',
            'Weak Uptrend':     'Weak Uptrend (positive log velocity but below threshold)',
            'Weak Downtrend':   'Weak Downtrend (negative log velocity but above -threshold)',
            'Slowing Uptrend':  'Slowing Uptrend (log velocity > 0 with high deceleration)',
            'Slowing Downtrend':'Slowing Downtrend (log velocity < 0 with high acceleration)'
        }

        fig, ax = plt.subplots(figsize=(12, 6))
        date_format = mdates.DateFormatter('%Y-%m-%d')

        ax.plot(results.index, results['price'], label='Actual Price', color='black', linewidth=1.5)

        prev_state = None
        start_idx = 0
        for i in range(len(results)):
            state = results['trend_state'].iloc[i]
            if prev_state is not None and (state != prev_state or i == len(results) - 1):
                color = colors.get(prev_state, 'gray')
                alpha = 0.3 if prev_state in ['Uptrend', 'Downtrend'] else 0.2
                ax.axvspan(results.index[start_idx], results.index[i], color=color, alpha=alpha)
                start_idx = i
            if prev_state is None or state != prev_state:
                prev_state = state

        ax.set_title(title)
        ax.xaxis.set_major_formatter(date_format)
        ax.grid(True, alpha=0.3)

        # Create legend for color meanings
        import matplotlib.patches as mpatches
        patches_list = []
        for state, color in colors.items():
            alpha_val = 0.3 if state in ['Uptrend', 'Downtrend'] else 0.2
            label_text = f"{state}: {color_meanings[state]}"
            patch = mpatches.Patch(color=color, alpha=alpha_val, label=label_text)
            patches_list.append(patch)

        ax.legend(handles=patches_list, loc='upper left')
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

    def visualize_indicators_and_velocity(self, results, threshold=0.02):
        """
        Visualize the Composite Indicator and the estimated log velocity in separate subplots.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        date_format = mdates.DateFormatter('%Y-%m-%d')

        # Subplot 1: Composite Indicator
        ax1.plot(results.index, results['composite_indicator'],
                 label='Composite Indicator', color='purple')
        ax1.set_title('Composite Indicator')
        ax1.xaxis.set_major_formatter(date_format)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Estimated Log Velocity
        ax2.plot(results.index, results['filtered_trend'],
                 label='Estimated Log Velocity', color='blue')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.axhline(y=threshold, color='green', linestyle='--', alpha=0.5,
                    label=f'Uptrend Threshold: +{threshold*100:.1f}%')
        ax2.axhline(y=-threshold, color='red', linestyle='--', alpha=0.5,
                    label=f'Downtrend Threshold: -{threshold*100:.1f}%')
        ax2.set_title('Estimated Log Velocity (daily returns)')
        ax2.xaxis.set_major_formatter(date_format)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

def run_btc_analysis(
    data_df,
    filter_start=None,
    filter_end=None,
    display_start=None,
    display_end=None,
    threshold=0.01
):
    """
    Run the EKF analysis on BTC price data with specified filtering and display periods.

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame containing at least 'start_at' and 'close' columns along with indicators.
    filter_start, filter_end : datetime-like or None
        Period used for filtering the data for the Kalman filter.
    display_start, display_end : datetime-like or None
        Period used for visualizing the analysis results.
    threshold : float
        Trend detection threshold (in terms of log difference).
    """
    # Index the DataFrame by datetime
    if 'start_at' in data_df.columns:
        if not isinstance(data_df['start_at'].iloc[0], (pd.Timestamp, np.datetime64, datetime)):
            data_df['start_at'] = pd.to_datetime(data_df['start_at'])
        df = data_df.set_index('start_at').copy()
    else:
        df = data_df.copy()

    if filter_start is not None:
        df = df.loc[df.index >= filter_start]
    if filter_end is not None:
        df = df.loc[df.index <= filter_end]

    detector = ExtendedKalmanFilterTrendDetector(
        observation_covariance=1.0,
        process_noise=0.01,
        damping=0.1
    )

    results = detector.detect_trend(df, threshold=threshold)

    display_results = results.copy()
    if display_start is not None:
        display_results = display_results.loc[display_results.index >= display_start]
    if display_end is not None:
        display_results = display_results.loc[display_results.index <= display_end]

    print("=== Filtered Results (last 10 rows) ===")
    print(results.tail(10))

    # 1) Visualize Price Chart with Trend Background in a separate figure
    detector.visualize_price_with_trend(display_results, title="BTC/USDT Price Chart with Trend Background")

    # 2) Visualize Composite Indicator & Estimated Log Velocity in another figure
    detector.visualize_indicators_and_velocity(display_results, threshold=threshold)

    return results

if __name__ == "__main__":
    import os
    import sys
    from datetime import datetime, timedelta

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

    from mongodb.data_loader_mongo import MongoDataLoader
    from common.constants import MARKET_DATA_TECH

    db = MongoDataLoader()

    raw_df = db.load_data_from_datetime_period(
        datetime(2023, 1, 1),
        datetime(2025, 3, 10),
        coll_type=MARKET_DATA_TECH,
        symbol='BTCUSDT',
        interval=1440
    )

    filter_start_date = datetime(2023, 1, 1)
    filter_end_date   = datetime(2025, 3, 10)
    display_start_date = datetime(2025, 1, 1)
    display_end_date   = datetime(2025, 3, 10)

    results = run_btc_analysis(
        data_df=raw_df,
        filter_start=filter_start_date,
        filter_end=filter_end_date,
        display_start=display_start_date,
        display_end=display_end_date,
        threshold=0.01
    )





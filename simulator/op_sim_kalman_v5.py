import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from scipy.linalg import inv
from op_sim_kalman_v4 import ExtendedKalmanFilterTrendDetector

def investigate_trend_discrepancies(results, window_size=5):
    """
    Function to investigate discrepancies between price movements and trend detection

    Parameters:
    -----------
    results : pandas.DataFrame
        DataFrame returned from detect_trend()
    window_size : int
        Window size for calculating price changes (default: 5)

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing discrepancies
    """
    # Calculate price changes over specified window
    results['price_change'] = results['price'].pct_change(periods=window_size)
    results['price_change_pct'] = results['price_change'] / results['price'].shift(window_size) * 100

    # Identify discrepancies
    # Case 1: Price is rising but trend is detected as downward
    uptrend_but_detected_down = (results['price_change'] > 0) & (
        results['trend_state'].isin(['Downtrend', 'Weak Downtrend', 'Slowing Downtrend']))

    # Case 2: Price is falling but trend is detected as upward
    downtrend_but_detected_up = (results['price_change'] < 0) & (
        results['trend_state'].isin(['Uptrend', 'Weak Uptrend', 'Slowing Uptrend']))

    # Combine all discrepancies
    discrepancies = results[uptrend_but_detected_down | downtrend_but_detected_up].copy()

    # Add column to classify discrepancy type
    discrepancies['discrepancy_type'] = 'Unknown'
    discrepancies.loc[uptrend_but_detected_down, 'discrepancy_type'] = 'Price Rising but Detected as Downtrend'
    discrepancies.loc[downtrend_but_detected_up, 'discrepancy_type'] = 'Price Falling but Detected as Uptrend'

    # Calculate discrepancy magnitude
    discrepancies['discrepancy_magnitude'] = np.abs(results.loc[discrepancies.index, 'price_change_pct'])

    return discrepancies

def analyze_discrepancies(results, discrepancies):
    """
    Analyze characteristics of detected discrepancies

    Parameters:
    -----------
    results : pandas.DataFrame
        Original results from detect_trend()
    discrepancies : pandas.DataFrame
        DataFrame containing discrepancies
    """
    print(f"Total periods analyzed: {len(results)}")
    print(f"Total discrepancies found: {len(discrepancies)} ({len(discrepancies)/len(results)*100:.2f}%)")

    # Group by discrepancy type
    by_type = discrepancies['discrepancy_type'].value_counts()
    print("\nDiscrepancy type distribution:")
    for type_name, count in by_type.items():
        print(f"{type_name}: {count} ({count/len(results)*100:.2f}%)")

    # Statistics about discrepancy magnitude
    print("\nDiscrepancy magnitude statistics:")
    print(discrepancies['discrepancy_magnitude'].describe())

    # Analysis of filtered trend and acceleration at discrepancy points
    print("\nFiltered trend statistics at discrepancy points:")
    print(discrepancies['filtered_trend'].describe())

    print("\nFiltered acceleration statistics at discrepancy points:")
    print(discrepancies['filtered_acceleration'].describe())

    # Check for consecutive discrepancies
    consecutive_count = 0
    for i in range(1, len(results)):
        if i < len(results) and i-1 >= 0:
            if results.index[i] in discrepancies.index and results.index[i-1] in discrepancies.index:
                consecutive_count += 1

    print(f"\nNumber of consecutive discrepancies: {consecutive_count}")

def visualize_discrepancies(results, discrepancies, num_examples=3, window=10):
    """
    Visualize specific examples of discrepancies

    Parameters:
    -----------
    results : pandas.DataFrame
        Original results from detect_trend()
    discrepancies : pandas.DataFrame
        DataFrame containing discrepancies
    num_examples : int
        Number of examples to visualize (default: 3)
    window : int
        Number of periods to include before and after discrepancy (default: 10)
    """
    # Sort discrepancies by magnitude
    top_discrepancies = discrepancies.sort_values('discrepancy_magnitude', ascending=False)

    # Visualize top examples
    for i, (idx, row) in enumerate(top_discrepancies.iterrows()):
        if i >= num_examples:
            break

        try:
            # Get window around discrepancy
            idx_loc = results.index.get_loc(idx)
            start_idx = max(0, idx_loc - window)
            end_idx = min(len(results), idx_loc + window + 1)
            subset = results.iloc[start_idx:end_idx]

            # Create plots
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

            # Date format settings
            date_format = mdates.DateFormatter('%Y-%m-%d')

            # Price chart
            ax1.plot(subset.index, subset['price'], label='Price', color='black', linewidth=1.5)
            ax1.axvline(x=idx, color='red', linestyle='--', alpha=0.7)
            ax1.set_title(f'Discrepancy Example {i+1}: {row["discrepancy_type"]}')
            ax1.xaxis.set_major_formatter(date_format)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylabel('Price')

            # Composite indicator
            ax2.plot(subset.index, subset['composite_indicator'], label='Composite Indicator', color='purple')
            ax2.axvline(x=idx, color='red', linestyle='--', alpha=0.7)
            ax2.set_title('Composite Indicator')
            ax2.xaxis.set_major_formatter(date_format)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylabel('Value')

            # Filtered trend
            ax3.plot(subset.index, subset['filtered_trend'], label='Filtered Trend', color='blue')
            ax3.axvline(x=idx, color='red', linestyle='--', alpha=0.7)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            threshold = 0.02
            ax3.axhline(y=threshold, color='green', linestyle='--', alpha=0.5, label='Uptrend Threshold')
            ax3.axhline(y=-threshold, color='red', linestyle='--', alpha=0.5, label='Downtrend Threshold')
            ax3.set_title('Filtered Trend (Velocity)')
            ax3.legend(loc='upper left')
            ax3.xaxis.set_major_formatter(date_format)
            ax3.grid(True, alpha=0.3)
            ax3.set_ylabel('Trend')

            # Filtered acceleration
            ax4.plot(subset.index, subset['filtered_acceleration'], label='Filtered Acceleration', color='orange')
            ax4.axvline(x=idx, color='red', linestyle='--', alpha=0.7)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            accel_threshold = threshold * 2
            ax4.axhline(y=accel_threshold, color='green', linestyle='--', alpha=0.5, label='Positive Acceleration')
            ax4.axhline(y=-accel_threshold, color='red', linestyle='--', alpha=0.5, label='Negative Acceleration')
            ax4.set_title('Filtered Acceleration')
            ax4.legend(loc='upper left')
            ax4.xaxis.set_major_formatter(date_format)
            ax4.grid(True, alpha=0.3)
            ax4.set_ylabel('Acceleration')
            ax4.set_xlabel('Date')

            plt.tight_layout()
            plt.suptitle(f'Discrepancy Analysis (Magnitude: {row["discrepancy_magnitude"]:.2f}%)', fontsize=16)
            plt.subplots_adjust(top=0.92)

            # Output details
            print(f"\nDiscrepancy Example {i+1}:")
            print(f"Date: {idx}")
            print(f"Type: {row['discrepancy_type']}")
            print(f"Price Change: {row['price_change_pct']:.2f}%")
            print(f"Trend State: {row['trend_state']}")
            print(f"Filtered Trend: {row['filtered_trend']:.6f}")
            print(f"Filtered Acceleration: {row['filtered_acceleration']:.6f}")

            plt.show()
        except Exception as e:
            print(f"Error occurred while visualizing example {i+1}: {str(e)}")

def calculate_lag_metrics(results, window_sizes=[1, 5, 10, 20]):
    """
    Calculate lag metrics to understand if trend detection lags behind price movements

    Parameters:
    -----------
    results : pandas.DataFrame
        Results from detect_trend()
    window_sizes : list
        List of window sizes to analyze

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing lag metrics
    """
    lag_metrics = pd.DataFrame(index=window_sizes, columns=['Cross Correlation', 'Correlation Lag'])

    for window in window_sizes:
        # Calculate price changes
        price_changes = results['price'].pct_change(periods=window)

        # Calculate cross-correlation between price changes and filtered trend
        # Remove NaNs
        valid_indices = ~np.isnan(price_changes) & ~np.isnan(results['filtered_trend'])
        price_changes_valid = price_changes[valid_indices]
        filtered_trend_valid = results['filtered_trend'][valid_indices]

        if len(price_changes_valid) == 0:
            lag_metrics.loc[window, 'Cross Correlation'] = np.nan
            lag_metrics.loc[window, 'Correlation Lag'] = np.nan
            continue

        # Standardize and calculate cross-correlation
        price_norm = (price_changes_valid - price_changes_valid.mean()) / price_changes_valid.std()
        trend_norm = (filtered_trend_valid - filtered_trend_valid.mean()) / filtered_trend_valid.std()

        cross_corr = np.correlate(price_norm, trend_norm, mode='full')

        # Find lag of maximum correlation
        max_corr_idx = np.argmax(cross_corr)
        lag = max_corr_idx - len(price_changes_valid) + 1

        lag_metrics.loc[window, 'Cross Correlation'] = cross_corr[max_corr_idx]
        lag_metrics.loc[window, 'Correlation Lag'] = lag

    return lag_metrics

def suggest_improvements(results, discrepancies, lag_metrics):
    """
    Suggest improvements to trend detection algorithm based on analysis

    Parameters:
    -----------
    results : pandas.DataFrame
        Results from detect_trend()
    discrepancies : pandas.DataFrame
        DataFrame containing discrepancies
    lag_metrics : pandas.DataFrame
        DataFrame containing lag metrics
    """
    print("Improvement Suggestions:")

    # Check if lag is a significant issue
    if lag_metrics['Correlation Lag'].mean() > 2:
        print("1. Trend detection algorithm appears to lag behind price movements.")
        print("   Consider adjusting process noise parameters to make the filter more responsive.")
        print("   Current setting: process_noise=0.01, try increasing to 0.02-0.05")

    # Check if thresholds are too strict
    near_threshold = ((results['filtered_trend'].abs() > 0.015) &
                      (results['filtered_trend'].abs() < 0.025)).mean()
    if near_threshold > 0.2:  # If more than 20% of values are near threshold
        print("2. Many values are near the threshold (0.02), which may cause frequent trend state changes.")
        print("   Consider adjusting thresholds or implementing hysteresis (different thresholds for entry/exit).")

    # Check if algorithm is too sensitive to short-term changes
    if len(discrepancies) > 0 and discrepancies['discrepancy_magnitude'].mean() < 2.0:
        print("3. Most discrepancies involve small price movements (< 2%).")
        print("   Consider implementing minimum price movement threshold or longer averaging period.")

    # Check if acceleration is causing too rapid trend changes
    accel_flips = ((results['filtered_acceleration'] > 0) !=
                   (results['filtered_acceleration'].shift(1) > 0)).mean()
    if accel_flips > 0.3:  # If acceleration changes sign frequently
        print("4. Acceleration frequently changes direction, potentially causing premature trend changes.")
        print("   Consider smoothing acceleration component or reducing its weight in trend determination.")

    # Check for rapid trend state changes
    state_changes = (results['trend_state'] != results['trend_state'].shift(1)).sum()
    if state_changes / len(results) > 0.2:  # If states change more than 20% of the time
        print("5. Trend states change frequently, potentially causing false signals.")
        print("   Consider implementing minimum duration for each trend state or confirmation period.")

def run_discrepancy_analysis(results, window_size=5, num_examples=3):
    """
    Run comprehensive analysis of discrepancies between price movements and trend detection

    Parameters:
    -----------
    results : pandas.DataFrame
        Results from detect_trend()
    window_size : int
        Window size for calculating price changes
    num_examples : int
        Number of examples to visualize
    """
    print("=" * 80)
    print("Trend Detection Discrepancy Analysis")
    print("=" * 80)

    # Identify discrepancies
    discrepancies = investigate_trend_discrepancies(results, window_size=window_size)

    # Analyze discrepancies
    analyze_discrepancies(results, discrepancies)

    # Calculate lag metrics
    lag_metrics = calculate_lag_metrics(results)
    print("\nLag Metrics:")
    print(lag_metrics)

    # Suggest improvements
    suggest_improvements(results, discrepancies, lag_metrics)

    # Visualize examples
    if len(discrepancies) > 0 and num_examples > 0:
        visualize_discrepancies(results, discrepancies, num_examples=num_examples)

    return discrepancies, lag_metrics

def main():
    """
    Main function to load data, run trend detection, and analyze discrepancies
    """
    try:
        # Use sample dataset if MongoDB is not available
        try:
            # Try to connect to MongoDB
            import os
            import sys
            from datetime import datetime, timedelta

            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            sys.path.append(parent_dir)

            from mongodb.data_loader_mongo import MongoDataLoader
            from common.constants  import MARKET_DATA_TECH

            # Get BTC time series data from MongoDB
            db = MongoDataLoader()
            df = db.load_data_from_datetime_period(
                datetime(2023, 12, 1),
                datetime(2025, 1, 1),
                coll_type=MARKET_DATA_TECH,
                symbol='BTCUSDT',
                interval=1440
            )

            # Extract required columns
            graph_df = df[['start_at', 'close', 'volume', 'macdhist', 'rsi', 'volatility', 'mfi', 'roc']]
        except Exception as e:
            print(f"MongoDB connection failed: {str(e)}")
            print("Using sample data instead...")

            # Generate sample data if MongoDB fails
            import pandas as pd
            import numpy as np
            from datetime import datetime, timedelta

            # Generate sample dates
            dates = [datetime(2023, 12, 1) + timedelta(days=i) for i in range(100)]

            # Generate price series with realistic trends and volatility
            np.random.seed(42)  # For reproducibility
            price = 20000.0  # Starting price
            prices = [price]

            for i in range(1, 100):
                # Add trend
                trend = 100 * np.sin(i / 10)

                # Add random noise
                noise = np.random.normal(0, 200)

                # Update price
                price = price + trend + noise
                if price < 10000:
                    price = 10000
                prices.append(price)

            # Create sample technical indicators
            sample_data = {
                'start_at': dates,
                'close': prices,
                'volume': np.random.normal(1000000, 200000, 100),
                'macdhist': np.random.normal(0, 10, 100),
                'rsi': 50 + 20 * np.sin(np.arange(100) / 8),
                'volatility': 5 + 2 * np.sin(np.arange(100) / 15),
                'mfi': 50 + 15 * np.sin(np.arange(100) / 10),
                'roc': np.random.normal(0, 2, 100)
            }

            graph_df = pd.DataFrame(sample_data)

        # Initialize trend detector
        detector = ExtendedKalmanFilterTrendDetector(
            observation_covariance=0.5,
            process_noise=0.01
        )

        # Run trend detection
        results = detector.detect_trend(graph_df, threshold=0.02)

        # Run discrepancy analysis
        run_discrepancy_analysis(results, window_size=5, num_examples=3)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
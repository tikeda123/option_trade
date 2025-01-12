import os,sys
import pandas as pd
import logging
import matplotlib.pyplot as plt
from typing import Tuple
from collections import deque


# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the path of the parent directory to sys.path
sys.path.append(parent_dir)


from trade_config import trade_config

class LeverageLossCutOptimizer:
        def __init__(
                self,
                time_window: int = 10
        ) -> None:
                """
                Initialization method.

                Args:
                        max_loss_cut (float): Maximum loss cut value.
                        max_leverage (float): Maximum leverage value.
                        time_window (int, optional): Time window for volatility calculation. Default is 10.
                        min_loss_cut (float, optional): Minimum loss cut value. Default is 0.06.

                Raises:
                        ValueError: If max_loss_cut is less than or equal to min_loss_cut, or if max_leverage is less than or equal to 1.
                """

                max_loss_cut = trade_config.max_losscut
                min_loss_cut = trade_config.min_losscut
                max_leverage = trade_config.max_leverage

                if max_loss_cut <= min_loss_cut:
                        raise ValueError("max_loss_cut must be greater than min_loss_cut")
                if max_leverage <= 1:
                        raise ValueError("max_leverage must be greater than 1")

                self.current_loss_cut = min_loss_cut
                self.current_leverage = 1.0

                self.bbvi_window = deque(maxlen=time_window)

        def loss_cut(self) -> float:
                """
                Get the current loss cut value.

                Returns:
                        float: Current loss cut value.
                """
                return  round(self.current_loss_cut,2)

        def leverage(self) -> float:
                """
                Get the current leverage value.

                Returns:
                        float: Current leverage value.
                """
                return round(self.current_leverage, 1)


        def optimize_param(self, bbvi: float) -> Tuple[float, float]:
                """
                Process BBVI value and calculate optimal leverage and loss cut values.

                Args:
                        bbvi (float): BBVI value.

                Returns:
                        Tuple[float, float]: Optimized leverage and loss cut values.
                """
                self.bbvi_window.append(bbvi)
                self.current_leverage = self.optimize_leverage_incremental()
                self.current_loss_cut = self.optimize_loss_cut_incremental()
                return self.current_leverage, self.current_loss_cut

        def compute_normalized_volatility(self) -> float:
                """
                Calculate normalized volatility based on the current BBVI window.

                Returns:
                        float: Normalized volatility (range 0 to 1).
                """
                if len(self.bbvi_window) < 2:
                        return 0.5  # Default value

                rolling_max = max(self.bbvi_window)
                rolling_min = min(self.bbvi_window)

                if rolling_max == rolling_min:
                        return 0.5  # If volatility is zero

                norm_vol = (self.bbvi_window[-1] - rolling_min) / (rolling_max - rolling_min)
                return norm_vol

        def optimize_leverage_incremental(self) -> float:
                """
                Optimize leverage based on BBVI (for incremental processing).

                Returns:
                        float: Optimized leverage value.
                """
                max_leverage = trade_config.max_leverage
                if len(self.bbvi_window) < 2:
                        return 1.0  # Default leverage

                norm_vol = self.compute_normalized_volatility()
                leverage = 1 + (1 - norm_vol) * (max_leverage - 1)
                leverage_clipped = max(1.0, min(leverage, max_leverage))
                return leverage_clipped

        def optimize_loss_cut_incremental(self) -> float:
                """
                Optimize loss cut value based on BBVI (for incremental processing).

                Returns:
                        float: Optimized loss cut value (range 0.06 to 0.3).
                """
                max_loss_cut = trade_config.max_losscut
                min_loss_cut = trade_config.min_losscut

                if len(self.bbvi_window) < 2:
                        return trade_config.min_losscut  # Minimum loss cut value

                norm_vol = self.compute_normalized_volatility()
                losscut_range = max_loss_cut - min_loss_cut
                losscut = max_loss_cut - (norm_vol * losscut_range)
                losscut_clipped = max(min_loss_cut, min(losscut, max_loss_cut))

                return losscut_clipped

        def optimize_loss_cut(self) -> float:
                """
                Calculate loss cut value.

                Returns:
                        float: Optimized loss cut value.
                """
                losscut = self.optimize_loss_cut_incremental()
                return losscut


from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import MARKET_DATA_TECH

# Log configuration
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

COLUMN_WCLPRICE = 'wclprice'  # Change the price column name as appropriate



def main():
        try:
                # Initialize the data loader
                db = MongoDataLoader()
                logging.info("MongoDataLoader initialized.")

                # Set start and end dates for data fetching
                start_date = "2023-07-01"  # Set appropriate date range
                end_date = "2023-08-01"

                # Load actual data
                df = db.load_data_from_datetime_period(start_date, end_date, MARKET_DATA_TECH)
                logging.info(f"Data loaded from {start_date} to {end_date}.")

                # Ensure the DataFrame is sorted by start_at
                df = df.sort_values('start_at')  # Change 'start_at' column name as appropriate


                leverage_op = LeverageLossCutOptimizer()

                # Initialize lists to collect processed data
                processed_data = {
                        'start_at': [],
                        'bbvi': [],
                        'leverage': []
                }

                # Optionally, initialize plot
                plt.ion()  # Interactive mode on
                fig, ax1 = plt.subplots(figsize=(14, 7))
                ax2 = ax1.twinx()

                # Initialize plot lines
                line_bbvi, = ax1.plot([], [], color='tab:blue', label='bbvi')
                line_leverage, = ax2.plot([], [], color='tab:red', label='Leverage')

                ax1.set_xlabel('Date')
                ax1.set_ylabel('bbvi', color='tab:blue')
                ax2.set_ylabel('Leverage', color='tab:red')
                fig.suptitle('Relationship between bbvi and Optimized Leverage', fontsize=16)
                fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9), bbox_transform=ax1.transAxes)
                plt.show()

                for idx, row in df.iterrows():
                        start_at = row['start_at']  # Change timestamp column name as appropriate
                        bbvi = row['bbvi']  # Ensure 'bbvi' column exists


                        # Calculate leverage and losscut
                        leverage,losscut = leverage_op.optimize_param(bbvi)
                        # Collect processed data
                        processed_data['start_at'].append(start_at)
                        processed_data['bbvi'].append(bbvi)
                        processed_data['leverage'].append(leverage)

                        # Logging
                        logging.info(f"Processed start_at: {start_at}, bbvi: {bbvi}, leverage: {leverage}, losscut: {losscut}")

                        # Update plot
                        line_bbvi.set_xdata(processed_data['start_at'])
                        line_bbvi.set_ydata(processed_data['bbvi'])
                        line_leverage.set_xdata(processed_data['start_at'])
                        line_leverage.set_ydata(processed_data['leverage'])

                        # Adjust the x-axis to show dates properly
                        ax1.relim()
                        ax1.autoscale_view()
                        ax2.relim()
                        ax2.autoscale_view()
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                        plt.pause(0.001)  # Small pause to allow the plot to update

                # Create processed DataFrame
                processed_df = pd.DataFrame(processed_data)

                # Display the last 10 rows
                print(processed_df[['leverage']].tail(10))

                # Calculate the correlation between leverage and bbvi (excluding NaN values)
                correlation = processed_df['leverage'].corr(processed_df['bbvi'])
                print(f"Correlation between leverage and bbvi: {correlation:.4f}")
                logging.info(f"Correlation between leverage and bbvi: {correlation:.4f}")

                # Plot scatter plot of leverage vs bbvi to visualize correlation
                plt.ioff()  # Turn off interactive mode
                plt.figure(figsize=(8, 6))
                plt.scatter(processed_df['bbvi'], processed_df['leverage'], alpha=0.5)
                plt.title('Correlation between Leverage and bbvi')
                plt.xlabel('bbvi')
                plt.ylabel('Leverage')
                plt.grid(True)
                plt.show()

        except Exception as e:
                logging.error("An error occurred during execution.", exc_info=True)

if __name__ == "__main__":
        main()
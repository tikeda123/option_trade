import pandas as pd
import os, sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Get the absolute path of the directory containing this file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the path of the parent directory to sys.path
sys.path.append(parent_dir)

# Import constants and utility functions
from common.constants import *
from common.utils import  get_config
# Import modules related to trading state and strategy
from mongodb.data_loader import DataLoader

class TradeStatisticsSummarizer:
        """
        A class to summarize trade statistics from simulation results.
        """

        def __init__(self, config_key='rolling_v1'):
                """
                Initializes the TradeStatisticsSummarizer with the given configuration key.
                """
                self.config_key = config_key
                self.config = get_config(self.config_key)
                self.filename = self.make_filename()
                self.load_filepath = parent_dir + '/' + self.config['DATAPATH'] + self.filename
                self.df = pd.read_csv(self.load_filepath)
                self.filter_data = self.df[self.df[COLUMN_BB_PROFIT] != 0].copy()

        def make_filename(self):
                """
                Generates a filename for saving simulation results based on symbol and interval.
                """
                symbol = get_config("SYMBOL")
                interval = get_config("INTERVAL")
                # Create a filename incorporating the symbol and interval for easy identification
                simulation_result_filename = f'{symbol}_{interval}_simulation_result.csv'
                return simulation_result_filename

        def calculate_win_rate(self, column, filtered_data=None):
                """
                Calculates the win rate for a given prediction column.
                """
                if filtered_data is None:
                        filtered_data = self.filter_data

                filtered_data['correct_prediction'] = 0
                filtered_data.loc[((filtered_data[COLUMN_BB_PROFIT] > 0) & (filtered_data[COLUMN_ENTRY_TYPE] == ENTRY_TYPE_LONG) & (filtered_data[column] == 1)) |
                                                        ((filtered_data[COLUMN_BB_PROFIT] > 0) & (filtered_data[COLUMN_ENTRY_TYPE] == ENTRY_TYPE_SHORT) & (filtered_data[column] == 0)) |
                                                        ((filtered_data[COLUMN_BB_PROFIT] < 0) & (filtered_data[COLUMN_ENTRY_TYPE] == ENTRY_TYPE_LONG) & (filtered_data[column] == 0)) |
                                                        ((filtered_data[COLUMN_BB_PROFIT] < 0) & (filtered_data[COLUMN_ENTRY_TYPE] == ENTRY_TYPE_SHORT) & (filtered_data[column] == 1)), 'correct_prediction'] = 1

                win_rate = filtered_data['correct_prediction'].sum() / len(filtered_data)
                return win_rate

        def summarize(self):
                """
                Summarizes the trade statistics.
                """
                for column in [COLUMN_PRED_V1, COLUMN_PRED_V2, COLUMN_PRED_V3,COLUMN_PRED_V4, COLUMN_PRED_V5]:
                        win_rate = self.calculate_win_rate(column)
                        print(f"{column} win rate: {win_rate:.2f}")

                conditions = [
                        (BB_DIRECTION_UPPER, ENTRY_TYPE_LONG),
                        (BB_DIRECTION_UPPER, ENTRY_TYPE_SHORT),
                        (BB_DIRECTION_LOWER, ENTRY_TYPE_LONG),
                        (BB_DIRECTION_LOWER, ENTRY_TYPE_SHORT),
                ]

                for bb_direction, entry_type in conditions:
                        for column in [COLUMN_PRED_V1, COLUMN_PRED_V2, COLUMN_PRED_V3, COLUMN_PRED_V4, COLUMN_PRED_V5]:
                                filtered_data = self.filter_data[
                                        (self.filter_data[COLUMN_BB_DIRECTION] == bb_direction) &
                                        (self.filter_data[COLUMN_ENTRY_TYPE] == entry_type)
                                ].copy()

                                if len(filtered_data) > 0:
                                        win_rate = self.calculate_win_rate(column, filtered_data)
                                        print(f"BB Direction: {bb_direction}, Entry Type: {entry_type}, {column} win rate: {win_rate:.2f}")
                                else:
                                        print(f"BB Direction: {bb_direction}, Entry Type: {entry_type}, {column} win rate: No data available")

        def plot_normal_distribution(self, column_to_plot, win_condition_column, win_or_lose, bb_direction, entry_type):
                """
                Creates a normal distribution plot for the specified column and conditions.

                Args:
                        column_to_plot (str): The name of the column to plot.
                        win_condition_column (str): The column to use for win condition (COLUMN_PRED_V1, COLUMN_PRED_V2, or COLUMN_PRED_V3).
                        win_or_lose (str): Whether to filter for wins ('win') or losses ('lose').
                        bb_direction (str): The BB direction (BB_DIRECTION_UPPER or BB_DIRECTION_LOWER).
                        entry_type (str): The entry type (ENTRY_TYPE_LONG or ENTRY_TYPE_SHORT).
                """
                if win_or_lose not in ['win', 'lose']:
                        raise ValueError("win_or_lose should be either 'win' or 'lose'.")

                win_condition = 1 if win_or_lose == 'win' else 0

                filtered_data = self.filter_data[
                        (self.filter_data[win_condition_column] == win_condition) &
                        (self.filter_data[COLUMN_BB_DIRECTION] == bb_direction) &
                        (self.filter_data[COLUMN_ENTRY_TYPE] == entry_type)
                ]

                if len(filtered_data) > 0:
                        data = filtered_data[column_to_plot]
                        # Fit a normal distribution to the data
                        mu, std = norm.fit(data)

                        # Plot the histogram
                        plt.hist(data, bins=25, density=True, alpha=0.6, color='g')

                        # Plot the normal distribution curve
                        xmin, xmax = plt.xlim()
                        x = np.linspace(xmin, xmax, 100)
                        p = norm.pdf(x, mu, std)
                        plt.plot(x, p, 'k', linewidth=2)
                        title = f"Normal Distribution of {column_to_plot}\n({win_condition_column}: {win_or_lose}, BB: {bb_direction}, Entry: {entry_type})"
                        plt.title(title)

                        plt.show()
                else:
                        print(f"No data found for the specified conditions ({win_condition_column}: {win_or_lose}, BB: {bb_direction}, Entry: {entry_type})")


        def get_start_times_by_condition(self, win_condition_column, win_or_lose, bb_direction, entry_type):
                """
                Retrieves all start times (COLUMN_START_AT values) that match the specified conditions.

                Args:
                        win_condition_column (str): The column to use for win condition
                                        (e.g., COLUMN_PRED_V1, COLUMN_PRED_V2, or COLUMN_PRED_V3).
                        win_or_lose (str): Whether to filter for wins ('win') or losses ('lose').
                        bb_direction (str): The BB direction (BB_DIRECTION_UPPER or BB_DIRECTION_LOWER).
                        entry_type (str): The entry type (ENTRY_TYPE_LONG or ENTRY_TYPE_SHORT).

                Returns:
                        pandas.Series: A Series containing the COLUMN_START_AT values matching the conditions.
                """
                if win_or_lose not in ['win', 'lose']:
                        raise ValueError("win_or_lose should be either 'win' or 'lose'.")

                win_condition = 1 if win_or_lose == 'win' else 0

                filtered_data = self.filter_data[
                        (self.filter_data[win_condition_column] == win_condition) &
                        (self.filter_data[COLUMN_BB_DIRECTION] == bb_direction) &
                        (self.filter_data[COLUMN_ENTRY_TYPE] == entry_type)
                ]

                return filtered_data[COLUMN_START_AT]

        def plot_value_around_time(self, column_to_plot, column_start_at):
                """
                Plots the values of a specified column within a time window around a given start time,
                along with the values of COLUMN_CLOSE in a different color (red) with adjusted scale.

                Args:
                column_to_plot (str): The name of the column to plot the values from.
                column_start_at (str): The start time around which to plot the values.
                """
                try:
                        target_row_index = self.df[self.df[COLUMN_START_AT] == column_start_at].index[0]
                except IndexError:
                        print(f"Error: No data found for column_start_at = {column_start_at}")
                        return

                start_step = max(0, target_row_index - 10)
                end_step = min(len(self.df) - 1, target_row_index + 11)

                plot_data = self.df.iloc[start_step:end_step]
                time_values = plot_data[COLUMN_START_AT]
                plot_values = plot_data[column_to_plot]
                close_values = plot_data[COLUMN_CLOSE]  # Get COLUMN_CLOSE values

                # Create the plot with two y-axes
                fig, ax1 = plt.subplots(figsize=(10, 6))

                # Plot the main column
                ax1.plot(time_values, plot_values, label=column_to_plot, color='b')
                ax1.set_xlabel("Time")
                ax1.set_ylabel(column_to_plot, color='b')
                ax1.tick_params(axis='y', labelcolor='b')

                # Create a second y-axis
                ax2 = ax1.twinx()

                # Plot COLUMN_CLOSE on the second y-axis
                ax2.plot(time_values, close_values, label=COLUMN_CLOSE, color='r')
                ax2.set_ylabel(COLUMN_CLOSE, color='r')
                ax2.tick_params(axis='y', labelcolor='r')

                # Highlight the target start time
                plt.axvline(x=column_start_at, color='gray', linestyle='--', label='Target Start Time')

                plt.title(f"Values of {column_to_plot} and {COLUMN_CLOSE} Around {column_start_at}")
                plt.xticks(rotation=45)
                fig.legend(loc="upper left") # Show legend for both lines
                plt.tight_layout()
                plt.show()

def main():
        """
        Main function to execute the trade statistics summarization.
        """
        summarizer = TradeStatisticsSummarizer()

        summarizer.summarize()
        """
        summarizer.plot_normal_distribution(
                column_to_plot='rsi',
                win_condition_column=COLUMN_PRED_V1,
                win_or_lose='lose',
                bb_direction=BB_DIRECTION_UPPER,
                entry_type=ENTRY_TYPE_LONG
        )

        start_times = summarizer.get_start_times_by_condition(
                win_condition_column=COLUMN_PRED_V3,
                win_or_lose='win',
                bb_direction=BB_DIRECTION_LOWER,
                entry_type=ENTRY_TYPE_LONG
        )
        print(start_times)

        for i in range(len(start_times)):
                summarizer.plot_value_around_time('roc', start_times.iloc[i])
        """
if __name__ == '__main__':
        main()
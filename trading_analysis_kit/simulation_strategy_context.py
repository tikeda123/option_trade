# Import necessary libraries
import pandas as pd
from typing import Optional
import os, sys
from typing import Tuple
from datetime import datetime

# Get the absolute path of the directory containing b.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Get the path of directory A

# Add the path of directory A to sys.path
sys.path.append(parent_dir)

# Import modules for managing common settings and constants
from common.constants import *

# Import classes for technical analysis, simulation strategies, and trading context
from fxtransaction import FXTransaction
from trading_analysis_kit.trading_context import TradingContext
from trade_config import trade_config



class SimulationStrategyContext(TradingContext):
        """
        Manages the context for simulation strategy.

        This class provides various methods for simulating FX trading.
        It includes functionalities for entry, exit, profit calculation, and stop-loss decisions.

        Attributes:
                fx_transaction (FXTransaction): Object managing FX transactions
                entry_manager (BollingerBand_EntryStrategy): Object managing entry strategy
                losscut (float): Stop-loss threshold
                leverage (float): Leverage ratio
                init_amount (float): Initial investment amount
                ptc (float): Per-trade commission rate
        """
        def __init__(self,strategy):
                """
                Initializes the class instance.

                Calls the parent class constructor, resets the index,
                and initializes the FX transaction class instance.

                Args:
                        strategy (SimulationStrategy): Instance of the simulation strategy class.
                """
                super().__init__(strategy)  # Initialize the parent TradingContext

                # Initialize FX transaction and entry strategy objects
                self.fx_transaction = FXTransaction()

                self.init_amount = trade_config.init_amount
                self.ptc = trade_config.ptc

                # Generate the filename for saving simulation results
                self.make_filename()

        def make_filename(self):
                """
                Generates a filename for saving simulation results based on symbol and interval.
                """
                symbol = trade_config.symbol
                interval = trade_config.interval
                # Create a filename incorporating the symbol and interval for easy identification
                self.simulation_result_filename = f'{symbol}_{interval}_simulation_result.csv'

        def save_simulation_result(self,context):
                """
                Saves the simulation results to a CSV file.

                Args:
                context (TradingContext): The trading context containing the simulation data.
                """
                # Check if the simulation result file should be saved
                save_filepath = parent_dir + '/' + self.config_manager.get('LOG','LOGPATH') + self.simulation_result_filename

                # Get the raw data from the data manager
                df = context.dm.get_raw_data()

                # Save the data to a CSV file without the index
                df.to_csv(save_filepath, index=False)

        def record_max_min_pandl_to_entrypoint(self):
                """
                Records the maximum and minimum profit and loss from entry point to exit.
                This helps in analyzing the trade's performance over its lifetime.
                """
                entry_index = self.dm.get_entry_index()
                exit_index = self.dm.get_exit_index()

                # Get the data between entry and exit points
                df = self.dm.get_df_fromto(entry_index+1, exit_index)

                # Find the minimum and maximum P&L within this period
                min_pandl = df[COLUMN_MIN_PANDL].min()
                max_pandl = df[COLUMN_MAX_PANDL].max()

                # Record these values at the entry point for later analysis
                self.dm.set_min_pandl(min_pandl, entry_index)
                self.dm.set_max_pandl(max_pandl, entry_index)

        def record_entry_exit_price(self):
                """
                Records the entry price, exit price, and Bollinger Band direction for each trade.
                This information is crucial for post-trade analysis and strategy refinement.
                """
                entry_index = self.dm.get_entry_index()
                exit_index = self.dm.get_exit_index()

                # Record the entry price for the entire duration of the trade
                self.dm.set_df_fromto(entry_index, exit_index, COLUMN_ENTRY_PRICE, self.dm.get_entry_price())

                # Record the exit price (will be the same for all rows in this range)
                self.dm.set_df_fromto(entry_index, exit_index, COLUMN_EXIT_PRICE, self.dm.get_exit_price())

                # Record the Bollinger Band direction at entry
                self.dm.set_df_fromto(entry_index, exit_index, COLUMN_BB_DIRECTION, self.dm.get_bb_direction())


        def calculate_current_profit(self,current_price=None) -> float:
                """
                Calculates the current profit or loss for the open position.

                Args:
                        current_price (float, optional): The current market price. If not provided, uses the closing price.

                Returns:
                        float: The calculated profit or loss.
                """
                if current_price is None:
                        current_price = self.dm.get_close_price()

                entry_price = self.dm.get_entry_price()
                pred_type = self.dm.get_prediction()
                leverage = self.lop.leverage()

                # Calculate the quantity based on leverage and initial amount
                qty = self.init_amount * leverage / entry_price
                buymnt = (qty * entry_price)  # Total buy amount
                selmnt = (qty * current_price)  # Total sell amount at current price

                # Calculate fees
                buy_fee = self.init_amount * self.ptc * leverage
                sel_fee = self.init_amount * self.ptc * leverage

                # Calculate profit based on position type (LONG or SHORT)
                if pred_type == PRED_TYPE_LONG:
                        current_profit = selmnt - buymnt - (buy_fee + sel_fee)
                else:  # SHORT position
                        current_profit = buymnt - selmnt - (buy_fee + sel_fee)

                return current_profit

        def is_profit_triggered(self,triggered_profit)->Tuple[bool, float]:
                """
                Checks if the profit target has been reached.

                Args:
                        triggered_profit (float): The profit target to check against.

                Returns:
                        tuple: (bool: Whether the profit target was reached, float: The price at which it was reached)
                """
                entry_type = self.dm.get_entry_type()

                # Determine the price to check based on the entry type
                if entry_type == ENTRY_TYPE_LONG:
                        profit_price = self.dm.get_high_price()  # For long positions, check the high price
                else:
                        profit_price = self.dm.get_low_price()   # For short positions, check the low price

                # Calculate the current profit at this price
                profit = self.calculate_current_profit(profit_price)

                if profit > triggered_profit:
                        # If profit target is reached, calculate the exact price at which it was triggered
                        calculate_profit_price = self.calculate_profit_triggered_price(triggered_profit)
                        return True, calculate_profit_price

                return False, profit_price

        def calculate_profit_triggered_price(self, profit) -> float:
                """
                Calculates the exact price at which a specified profit would be realized.

                Args:
                        profit (float): The target profit amount.

                Returns:
                        float: The price at which the specified profit would be realized.
                """
                entry_price = self.dm.get_entry_price()
                pred_type = self.dm.get_prediction()
                leverage = self.lop.leverage()

                # Calculate quantity and fees
                qty = self.init_amount * leverage/ entry_price
                buymnt = qty * entry_price
                buy_fee = self.init_amount * self.ptc * leverage
                sell_fee = (self.init_amount + profit) * self.ptc * leverage

                # Calculate the trigger price based on position type
                if pred_type == PRED_TYPE_LONG:
                # For long positions, add fees and profit to the buy amount
                        profit_price = (buymnt + buy_fee + sell_fee + profit) / qty
                else:
                        # For short positions, subtract fees and profit from the buy amount
                        profit_price = (buymnt - buy_fee - sell_fee - profit) / qty

                return profit_price


        def is_losscut_triggered(self)->Tuple[bool, float]:
                """
                Checks if the stop-loss (loss-cut) condition has been triggered for the current position.

                This method calculates whether the current loss has exceeded the predefined loss-cut threshold.
                It uses different price points (low for long positions, high for short positions) to determine
                 the worst-case scenario for the current market conditions.

                Returns:
                        tuple: (bool: Whether the loss-cut has been triggered, float: The amount of loss at trigger point)
                 """
                # Get the entry type (long or short) for the current position
                entry_type = self.dm.get_entry_type()
                losscut_price = None
                losscut = self.lop.loss_cut()

                # Determine the price to use for loss-cut calculation based on position type
                if entry_type == ENTRY_TYPE_LONG:
                        # For long positions, use the low price (worst case for longs)
                        losscut_price = self.dm.get_low_price()
                else:
                        # For short positions, use the high price (worst case for shorts)
                        losscut_price = self.dm.get_high_price()

                # Calculate the maximum allowable loss based on initial investment and loss-cut percentage
                loss_cut_pandl = losscut * self.init_amount * -1  # Negative value represents a loss

                # Calculate the current profit/loss at the loss-cut price
                pandl = self.calculate_current_profit(losscut_price)

                # Check if the current loss exceeds the maximum allowable loss
                if pandl < loss_cut_pandl:
                        # Loss-cut triggered
                        return True, loss_cut_pandl
                # Loss-cut not triggered
                return False, pandl

        def change_to_idle_state(self):
                """
                Transitions the trading system to the idle state.

                In the idle state, the system is not actively in a trade and is waiting for new opportunities.
                This method resets the current index in the data manager and records the state transition.
                """
                # Reset the current index in the data manager
                self.dm.reset_index()
                # Record the transition to the idle state
                self.record_state_and_transition(STATE_IDLE)

        def change_to_position_state(self):
                """
                Transitions the trading system to the position state.

                The position state indicates that the system has entered a new trade. This method
                records the entry index and price, and updates the system state.
                """
                # Get the current index as the entry point for the new position
                #entry_index = self.dm.get_current_index()
                # Use the current closing price as the entry price
                #entry_price = self.dm.get_close_price()

                # Record the entry index and price in the data manager
                #self.dm.set_entry_index(entry_index)
                #self.dm.set_entry_price(entry_price)

                # Record the transition to the position state
                self.record_state_and_transition(STATE_POSITION)

        def change_to_entrypreparation_state(self):
                """
                Transitions the trading system to the entry preparation state.

                In the entry preparation state, the system is getting ready to enter a new trade.
                This could involve finalizing calculations, checking conditions, or preparing orders.
                """
                # Record the transition to the entry preparation state
                self.record_state_and_transition(STATE_ENTRY_PREPARATION)

        def change_to_exitpreparation_state(self):
                """
                Transitions the trading system to the entry preparation state.

                In the entry preparation state, the system is getting ready to enter a new trade.
                This could involve finalizing calculations, checking conditions, or preparing orders.
                """
                # Record the transition to the entry preparation state
                self.record_state_and_transition(STATE_EXIT_PREPARATION)

        def record_state_and_transition(self, state: str):
                """
                Records the current state and handles the state transition.

                This method logs the state transition and triggers any necessary actions associated
                with the new state.

                Args:
                        state (str): The new state to transition to.
                """
                # Log the state transition
                self.log_transaction(f'Transitioning to {state}')
                # Handle the state transition using the state machine
                self._state.handle_request(self, state)

        def set_current_max_min_pandl(self):
                """
                Calculates and sets the current maximum and minimum profit/loss based on high and low prices.
                This helps in tracking the potential range of outcomes for the current position.

                Returns:
                        tuple: (Maximum P&L, Minimum P&L)
                """
                # Calculate P&L at high and low prices
                h_pandl = self.calculate_current_profit(self.dm.get_high_price())
                l_pandl = self.calculate_current_profit(self.dm.get_low_price())

                # Determine max and min P&L
                max_pandl = max(h_pandl, l_pandl)
                min_pandl = min(h_pandl, l_pandl)

                # Record these values in the data manager
                self.dm.set_max_pandl(max_pandl)
                self.dm.set_min_pandl(min_pandl)

                return max_pandl, min_pandl

        def load_recent_data(self) -> Optional[pd.DataFrame]:
                """
                Fetches the current historical snapshot for the trading system.

                This method retrieves the historical data for the current time period and updates the data manager.
                """
                self.dm.dataloader.get_latest_n_records( MARKET_DATA_TECH)
                self.dm.add_data_columns()

                df = self.dm.get_raw_data()
                #if self.dm.dataloader.is_collection_exists(ROLLING_AI_DATA):
                #        self.dm.dataloader.drop_collection(ROLLING_AI_DATA)

                self.dm.dataloader.insert_data(df, ROLLING_AI_DATA)
                return self.dm.get_raw_data()

        def load_data_for_offline(self, start_date: datetime)-> Optional[pd.DataFrame]:
                """
                Loads historical data for the specified start date.

                Args:
                        start_date (datetime): The start date for loading historical data.

                Returns:
                        pd.DataFrame: The historical data for the specified start date.
                """
                self.dm.dataloader.load_data_from_point_date(start_date, 100,MARKET_DATA_TECH)
                self.dm.add_data_columns()
                df = self.dm.get_raw_data()

                if self.dm.dataloader.is_collection_exists(ROLLING_AI_DATA):
                        self.dm.dataloader.drop_collection(ROLLING_AI_DATA)

                self.dm.dataloader.insert_data(df, ROLLING_AI_DATA)
                return self.dm.get_raw_data()

        def append_df(self, new_df: pd.DataFrame):
                """
                Appends a new DataFrame to the existing data manager DataFrame.

                Args:
                        new_df (pd.DataFrame): The new DataFrame to append.
                """
                self.new_add_columns(new_df,COLUMN_PANDL, 0.0, float)
                self.new_add_columns(new_df,COLUMN_STATE, None, str)
                self.new_add_columns(new_df,COLUMN_BB_DIRECTION, None, str)
                self.new_add_columns(new_df,COLUMN_ENTRY_PRICE, 0.0, float)
                self.new_add_columns(new_df,COLUMN_EXIT_PRICE, 0.0, float)
                self.new_add_columns(new_df,COLUMN_CURRENT_PROFIT, 0.0, float)
                self.new_add_columns(new_df,COLUMN_BB_PROFIT, 0.0, float)
                self.new_add_columns(new_df,COLUMN_PREDICTION, 0, int)
                self.new_add_columns(new_df,COLUMN_PROFIT_MA, 0.0, float)
                self.new_add_columns(new_df,COLUMN_ENTRY_TYPE,  None, str)
                self.new_add_columns(new_df,COLUMN_MAX_PANDL, 0.0, float)
                self.new_add_columns(new_df,COLUMN_MIN_PANDL, 0.0, float)
                self.new_add_columns(new_df,COLUMN_EXIT_REASON, None, str)
                self.new_add_columns(new_df,COLUMN_EXIT_TIME, None, str)
                self.new_add_columns(new_df,COLUMN_PRED_V1, 0, int)
                self.new_add_columns(new_df,COLUMN_PRED_V2, 0, int)
                self.new_add_columns(new_df,COLUMN_PRED_V3, 0, int)
                self.new_add_columns(new_df,COLUMN_PRED_V4, 0, int)
                self.new_add_columns(new_df,COLUMN_PRED_V5, 0, int)
                self.new_add_columns(new_df,COLUMN_PRED_V6, 0, int)
                self.new_add_columns(new_df,COLUMN_PRED_V7, 0, int)
                self.new_add_columns(new_df,COLUMN_PRED_V8, 0, int)
                self.new_add_columns(new_df,COLUMN_PRED_V9, 0, int)
                self.new_add_columns(new_df,COLUMN_PRED_V10, 0, int)
                self.new_add_columns(new_df,COLUMN_PRED_V11, 0, int)
                self.new_add_columns(new_df,COLUMN_PRED_V12, 0, int)
                self.new_add_columns(new_df,COLUMN_PRED_V13, 0, int)
                self.new_add_columns(new_df,COLUMN_PRED_V14, 0, int)
                self.new_add_columns(new_df,COLUMN_PRED_V15, 0, int)
                self.new_add_columns(new_df,COLUMN_PRED_V16, 0, int)
                self.new_add_columns(new_df,COLUMN_PRED_V17, 0, int)
                self.new_add_columns(new_df,COLUMN_PRED_V18, 0, int)
                self.new_add_columns(new_df,COLUMN_PRED_TARGET, 0, int)
                self.new_add_columns(new_df,COLUMN_ORDER_ID, None, str)
                self.dm.dataloader.append_df(new_df)

        def new_add_columns(self, new_df: pd.DataFrame, column: str, value: float, dtype: str):
                """
                Adds new columns to the existing data manager DataFrame.

                Args:
                        new_df (pd.DataFrame): The new DataFrame to add.
                """
                new_df.loc[:, column] = value  # valueはすでにfloat型と想定
                new_df.loc[:, column] = new_df[column].astype(dtype)

        def update_roll_ai_data(self):
                """
                Updates the rolling AI data collection with the latest data.
                """
                self.dm.dataloader.insert_data(self.dm.get_raw_data().copy(), ROLLING_AI_DATA)

        def print_win_lose(self):
                """
                Displays win rates and plots the balance over time.
                This provides a visual summary of the trading strategy's performance.
                """
                self.fx_transaction.display_all_win_rates()
                self.fx_transaction.plot_balance_over_time()

# テスト用のコード
def main():
        # 設定ファイルのパスを指定
        from trading_analysis_kit.simulation_stepwise_profits import SimulationStepwiseProfits
        from trading_analysis_kit.simulation_strategy import SimulationStrategy
        strategy_context = SimulationStrategy()
        #strategy_context = SimulationStepwiseProfits()


        context = SimulationStrategyContext(strategy_context)
        context.load_data_from_datetime_period('2023-12-01 00:00:00', '2024-01-01- 00:00:00')
        context.run_trading()
        context.print_win_lose()
        context.save_simulation_result(context)

if __name__ == "__main__":
        main()
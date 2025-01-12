import pandas as pd
import os, sys

# Get the absolute path of the directory containing this file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the path of the parent directory to sys.path
sys.path.append(parent_dir)

# Import constants and utility functions
from common.constants import *
# Import modules related to trading state and strategy
from trading_analysis_kit.trading_state import *
from trading_analysis_kit.trading_strategy import TradingStrategy
from trading_analysis_kit.simulation_entry_strategy import EntryStrategy
from trading_analysis_kit.simulation_strategy_context import SimulationStrategyContext
from trading_analysis_kit.dynamic_modelselection_strategy import DynamicModelSelectionStrategy
from trading_analysis_kit.simulation_entry_strategy_singlemodel import EntryStrategySIngleModel
from trading_analysis_kit.simulation_entry_strategy_group import EntryStrategyGroup
from trade_config import trade_config

class SimulationStrategy(TradingStrategy):
        """
        A simulation class that implements a trading strategy.
        It handles decisions for trade entry and exit, and manages state transitions.
        """

        def __init__(self):
                """
                Initializes the simulation strategy by loading configuration settings for the account.
                """
                #self.__entry_strategy = EntryStrategy()
                #self.__entry_strategy = DynamicModelSelectionStrategy()
                #self.__entry_strategy = EntryStrategySIngleModel()
                self.__entry_strategy = EntryStrategyGroup()

        def should_entry(self, context, index: int) -> bool:
                """
                Determines whether to enter a trade.

                Args:
                        context (TradingContext): The trading context object.
                        index (int): The current data index.

                Returns:
                        bool: True if a trade should be entered, False otherwise.

                This method checks if the closing price crosses the upper or lower Bollinger Band.
                If it does, it sets the Bollinger Band direction accordingly and returns True,
                indicating that a trade entry is recommended. Otherwise, it returns False.
                """
                if context.dm.get_current_index() < TIME_SERIES_PERIOD:
                        return False

                # Check if the closing price crosses the upper Bollinger Band
                if context.is_first_column_greater_than_second(COLUMN_CLOSE, COLUMN_UPPER_BAND2, index):
                        context.log_transaction(f'upper band cross')
                        if trade_config.entry_enabled  == False:
                                context.log_transaction(f'entry disabled')
                                return False

                        context.dm.set_bb_direction(BB_DIRECTION_UPPER)
                        return True
                # Check if the closing price crosses the lower Bollinger Band
                elif context.is_first_column_less_than_second(COLUMN_CLOSE, COLUMN_LOWER_BAND2, index):
                        context.log_transaction(f'lower band cross')
                        if trade_config.entry_enabled  == False:
                                context.log_transaction(f'entry disabled')
                                return False

                        context.dm.set_bb_direction(BB_DIRECTION_LOWER)
                        return True

                # No entry condition met
                return False

        def Idle_event_execute(self, context: SimulationStrategyContext)->None:
                """
                Executes an event in the idle state, transitioning to the entry preparation state.

                Args:
                        context (TradingContext): The trading context object.
                """
                # Check if enough data points are available for analysis
                #if context.dm.get_current_index() < TIME_SERIES_PERIOD:
                #        return

                # Get trend prediction from the entry manager
                flag, pred = self.__entry_strategy.trend_prediction(context)

                if flag ==False:
                        # Transition to the entry preparation state
                        return

                context.dm.set_prediction(pred)
                self.trade_entry(context)
                # Transition to the position state
                context.change_to_position_state()
                return

        def EntryPreparation_event_execute(self, context: SimulationStrategyContext)->None:
                """
                Executes an event when the counter exceeds the threshold in the entry preparation state.
                It makes an entry decision based on trend prediction and if favorable, enters a trade
                and transitions to the position state. Otherwise, it returns to the idle state.

                Args:
                        context (TradingContext): The trading context object.
                """
                pass

        def PositionState_event_exit_execute(self, context: SimulationStrategyContext)->None:
                """
                Executes an exit event in the position state.
                This method is called when the exit conditions are met.
                Currently, it doesn't perform any specific action.

                Args:
                        context (TradingContext): The trading context object.
                """
                pass

        def PositionState_event_continue_execute(self, context):
                """
                Executes a continue event in the position state.
                It checks for a loss-cut trigger and exits the position if necessary.
                If no loss-cut is triggered, it calculates and records the current profit and loss.

                Args:
                        context (TradingContext): The trading context object.
                """

                #flag,price = context.is_profit_triggered(200)

                #if flag:
                #        self._handle_position_exit(context, price)
                #        return

                # Calculate the current profit and loss
                pandl = self.calculate_current_pandl(context)
                # Set the profit and loss in the data manager
                context.dm.set_pandl(pandl)
                # Log the current profit and loss
                context.log_transaction(f'continue Position state pandl:{pandl}')

                # Get the Bollinger Band direction and prediction
                bb_direction = context.dm.get_bb_direction()
                pred = context.dm.get_prediction()

                # Define exit conditions based on Bollinger Band direction and prediction
                position_state_dict = {
                        (BB_DIRECTION_UPPER, PRED_TYPE_LONG): ['less_than', COLUMN_MIDDLE_BAND],
                        (BB_DIRECTION_UPPER, PRED_TYPE_SHORT): ['less_than', COLUMN_LOWER_BAND1],
                        (BB_DIRECTION_LOWER, PRED_TYPE_LONG): ['greater_than', COLUMN_UPPER_BAND1],
                        (BB_DIRECTION_LOWER, PRED_TYPE_SHORT): ['greater_than', COLUMN_MIDDLE_BAND]
                }

                # Get the exit condition for the current direction and prediction
                condition = position_state_dict.get((bb_direction, pred))
                # If no condition is found, continue holding the position
                if condition is None:
                        raise ValueError(ERROR_MESSAGE_BB_DIRECTION.format(bb_direction=bb_direction))

                # Extract the operator and column for the exit condition
                operator, column = condition

                # Check if the exit condition is met


                if operator == 'less_than':
                        if context.is_first_column_less_than_second(COLUMN_CLOSE, column, context.dm.get_current_index()):
                                        self._handle_position_exit(context, context.dm.get_close_price())
                                        return

                elif operator == 'greater_than':
                        if context.is_first_column_greater_than_second(COLUMN_CLOSE, column, context.dm.get_current_index()):
                                        self._handle_position_exit(context, context.dm.get_close_price())
                                        return


                # No exit condition met, continue holding the position


        def ExitPreparationState_event_exit_execute(self, context: SimulationStrategyContext)->None:
                """
                Executes an exit event in the position state.
                This method is called when the exit conditions are met.
                Currently, it doesn't perform any specific action.

                Args:
                        context (TradingContext): The trading context object.
                """
                pass

        def ExitPreparationState_event_continue_execute(self, context):
                """
                Executes a continue event in the position state.
                It checks for a loss-cut trigger and exits the position if necessary.
                If no loss-cut is triggered, it calculates and records the current profit and loss.

                Args:
                        context (TradingContext): The trading context object.
                """
                # Calculate the current profit and loss
                pandl = self.calculate_current_pandl(context)
                # Set the profit and loss in the data manager
                context.dm.set_pandl(pandl)
                # Log the current profit and loss
                context.log_transaction(f'continue Position state pandl:{pandl}')




        def should_exit_position(self, context:SimulationStrategyContext, index: int) -> bool:
                """
                Determines whether to exit the current position.

                Args:
                        context (TradingContext): The trading context object.
                        index (int): The current data index.

                Returns:
                        bool: True if the position should be exited, False otherwise.

                This method checks for loss-cut triggers and specific exit conditions based on the Bollinger Band direction and prediction.
                If any of these conditions are met, it calls _handle_position_exit to exit the trade and returns True.
                """
                # Update current maximum and minimum profit and loss
                context.set_current_max_min_pandl()
                # Check if a loss-cut is triggered
                is_losscut_triggered, exit_price = self.is_losscut_triggered(context)

                # If loss-cut is triggered, exit the position
                if is_losscut_triggered:
                        self._handle_position_exit(context, exit_price, losscut=True)
                        return True

                return False


        def should_exit_preparataion(self, context, index: int) -> bool:
                """
                Determines whether to exit the current position.

                Args:
                        context (TradingContext): The trading context object.
                        index (int): The current data index.

                Returns:
                        bool: True if the position should be exited, False otherwise.

                This method checks for loss-cut triggers and specific exit conditions based on the Bollinger Band direction and prediction.
                If any of these conditions are met, it calls _handle_position_exit to exit the trade and returns True.
                """
                # Update current maximum and minimum profit and loss
                context.set_current_max_min_pandl()
                # Check if a loss-cut is triggered
                is_losscut_triggered, exit_price = self.is_losscut_triggered(context)

                # If loss-cut is triggered, exit the position
                if is_losscut_triggered:
                        self._handle_position_exit(context, exit_price, losscut=True)
                        return True


                bb_direction = context.dm.get_bb_direction()
                pred = context.dm.get_prediction()

                # Define exit conditions based on Bollinger Band direction and prediction
                position_state_dict = {
                        (BB_DIRECTION_UPPER, PRED_TYPE_LONG): ['less_than', COLUMN_MIDDLE_BAND],
                        (BB_DIRECTION_UPPER, PRED_TYPE_SHORT): ['less_than', COLUMN_LOWER_BAND1],
                        (BB_DIRECTION_LOWER, PRED_TYPE_LONG): ['greater_than', COLUMN_UPPER_BAND1],
                        (BB_DIRECTION_LOWER, PRED_TYPE_SHORT): ['greater_than', COLUMN_MIDDLE_BAND]
                }

                # Get the exit condition for the current direction and prediction
                condition = position_state_dict.get((bb_direction, pred))
                # If no condition is found, continue holding the position
                if condition is None:
                        raise ValueError(ERROR_MESSAGE_BB_DIRECTION.format(bb_direction=bb_direction))

                # Extract the operator and column for the exit condition
                operator, column = condition

                # Check if the exit condition is met
                if operator == 'less_than':
                        if context.is_first_column_less_than_second(COLUMN_CLOSE, column, index):
                                self._handle_position_exit(context, context.dm.get_close_price())
                                return True
                elif operator == 'greater_than':
                        if context.is_first_column_greater_than_second(COLUMN_CLOSE, column, index):
                                self._handle_position_exit(context, context.dm.get_close_price())
                                return True

                # No exit condition met, continue holding the position
                return False



        def _handle_position_exit(self, context, exit_price, losscut=False):
                """
                Handles the process of exiting a position, recording the exit details and transitioning to the idle state.

                Args:
                        context (TradingContext): The trading context object.
                        exit_price (float): The exit price.
                        losscut (bool, optional): True if the exit is due to a loss-cut, False otherwise. Defaults to False.
                """
                # Determine the reason for exit
                if losscut:
                        context.log_transaction(f'losscut price: {exit_price}')
                        losscut = "losscut"
                        reason = EXIT_REASON_LOSSCUT
                else:
                        reason = EXIT_REASON_NORMAL
                        losscut = None
                # Execute the trade exit
                self.trade_exit(context, exit_price, losscut=losscut)
                # Calculate the profit from the trade
                profit = context.calculate_current_profit(exit_price)
                # Record the profit in the data manager
                context.dm.set_bb_profit(profit, context.dm.get_entry_index())
                exit_time = context.dm.get_current_date()
                context.dm.set_exit_time(exit_time, context.dm.get_entry_index())

                # Update profit and loss, exit reason, and record entry/exit prices
                context.dm.set_pandl(profit)
                context.dm.set_exit_reason(reason)
                context.dm.set_exit_reason(reason, context.dm.get_entry_index())
                context.record_entry_exit_price()

                # Get the prediction and set the prediction target
                pred = context.dm.get_prediction()
                if profit > 0:
                        pred_target = pred
                else:
                        pred_target = 1 -  pred

                #for i in range(0,7):
                #        context.dm.set_pred_target(pred_target,  context.dm.get_entry_index()+i)
                context.dm.set_pred_target(pred_target,  context.dm.get_entry_index())

                self.__entry_strategy.update_performance(context, context.dm.get_entry_index())

                # Record maximum and minimum profit and loss to the entry point
                context.record_max_min_pandl_to_entrypoint()
                # Transition to the idle state
                context.change_to_idle_state()

        def trade_entry(self, context):
                """
                Executes a trade entry, determining the entry type based on the Bollinger Band direction and prediction.
                It records the trade details and updates the transaction serial number in the data manager.

                Args:
                        context (TradingContext): The trading context object.
                """
                # Get the current date, prediction, Bollinger Band direction, and entry price
                date = context.dm.get_current_date()
                pred = context.dm.get_prediction()
                bb_direction = context.dm.get_bb_direction()
                entry_price = context.dm.get_close_price()
                leverage = context.lop.leverage()
                losscut = context.lop.loss_cut()

                # Determine the entry type (long or short) based on the prediction
                entry_type = ENTRY_TYPE_LONG if pred == 1 else ENTRY_TYPE_SHORT

                # Record the entry type and price in the data manager
                context.dm.set_entry_type(entry_type)
                context.dm.set_entry_price(entry_price)
                context.dm.set_entry_index(context.dm.get_current_index())

                # Execute the trade entry and get the transaction serial number
                serial = context.fx_transaction.trade_entry(entry_type, pred, entry_price, date, bb_direction, leverage, losscut)
                # Set the transaction serial number in the data manager
                context.dm.set_fx_serial(serial)

        def trade_exit(self, context, exit_price, losscut=None):
                """
                Executes a trade exit at the specified price and calculates the profit or loss.

                Args:
                        context (TradingContext): The trading context object.
                        exit_price (float): The exit price.
                        losscut (str, optional): Indicates whether the exit is due to a loss-cut. Defaults to None.

                Returns:
                        float: The profit or loss from the executed trade.
                """
                # Get the transaction serial number and current date
                serial = context.dm.get_fx_serial()
                date = context.dm.get_current_date()

                # Record the exit index and price in the data manager
                context.dm.set_exit_index(context.dm.get_current_index())
                context.dm.set_exit_price(exit_price)

                # Execute the trade exit and return the profit or loss
                return context.fx_transaction.trade_exit(serial, exit_price, date, losscut=losscut)

        def is_losscut_triggered(self, context)-> tuple[bool, float]:
                """
                Determines if a loss-cut has been triggered.

                Args:
                        context (TradingContext): The trading context object.

                Returns:
                        tuple: A tuple containing a boolean value indicating whether the loss-cut was triggered
                               and the price at which it was triggered.

                This method checks the current price against the loss-cut threshold based on the entry type (long or short).
                If the loss-cut is triggered, it returns True and the corresponding price. Otherwise, it returns False.
                """
                # Get the transaction serial number and entry type
                serial = context.dm.get_fx_serial()
                entry_type = context.dm.get_entry_type()
                losscut_price = None
                losscut = context.lop.loss_cut()
                leverage = context.lop.leverage()

                # Determine the loss-cut price based on the entry type
                if entry_type == ENTRY_TYPE_LONG:
                        losscut_price = context.dm.get_low_price()
                else:
                        losscut_price = context.dm.get_high_price()

                # Check if the loss-cut is triggered and return the result
                return context.fx_transaction.is_losscut_triggered(serial, losscut_price)

        def calculate_current_pandl(self, context, exit_price=None):
                """
                Calculates the current profit and loss.

                Args:
                        context (TradingContext): The trading context object.
                        exit_price (float, optional): The exit price to calculate profit/loss against. Defaults to None.

                Returns:
                        float: The current profit and loss.

                This method calculates the profit or loss based on either the current price or a specified exit price.
                """
                # Get the transaction serial number
                serial = context.dm.get_fx_serial()

                # If an exit price is specified, use it for calculation
                if exit_price is not None:
                        pandl = context.fx_transaction.get_pandl(serial, exit_price)
                        return pandl

                # If no exit price is specified, use the current closing price
                current_price = context.dm.get_close_price()
                pandl = context.fx_transaction.get_pandl(serial, current_price)
                return pandl

        def should_hold_position(self, context):
                """
                Determines whether to continue holding the current position.
                Currently, it always returns False, indicating not to hold the position.

                Args:
                        context (TradingContext): The trading context object.

                Returns:
                        bool: False, indicating not to hold the position.
                """
                # This method is currently not used and always returns False
                return False
                # Placeholder for potential future logic to determine if a position should be held
                pandl = self.calculate_current_pandl(context)

                if pandl < 0:
                        return False

                trend_prediction = context.dm.get_prediction()
                rolling_pred = self.__entry_strategy.predict_trend_rolling(context)

                if rolling_pred != trend_prediction:
                        return False

                return True

        def show_win_lose(self, context):
                """
                Displays win-loss statistics for the trades.

                Args:
                        context (TradingContext): The trading context object.

                This method uses the fx_transaction object to display the win rates and plot the balance over time.
                """
                context.fx_transaction.display_all_win_rates()
                context.fx_transaction.plot_balance_over_time()




import pandas as pd
import os, sys
import time

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
from trading_analysis_kit.simulation_strategy_context import SimulationStrategyContext
from trading_analysis_kit.simulation_entry_strategy_group import EntryStrategyGroup
from bybit_api.bybit_trader import BybitTrader
from trade_config import trade_config

class OnlineBollingerBandStrategy(TradingStrategy):
        """
        A simulation class that implements a trading strategy.
        It handles decisions for trade entry and exit, and manages state transitions.
        """

        def __init__(self):
                """
                Initializes the simulation strategy by loading configuration settings for the account.
                """
                self.__entry_strategy = EntryStrategyGroup()
                self.__online_api = BybitTrader()


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

                # Check if the closing price crosses the upper Bollinger Band
                if context.is_first_column_greater_than_second(COLUMN_CLOSE, COLUMN_UPPER_BAND2, index):
                        context.log_transaction(f'upper band cross')
                        if trade_config.entry_enabled == False:
                                context.log_transaction(f'entry disabled')
                                return False

                        context.dm.set_bb_direction(BB_DIRECTION_UPPER)
                        return True
                # Check if the closing price crosses the lower Bollinger Band
                elif context.is_first_column_less_than_second(COLUMN_CLOSE, COLUMN_LOWER_BAND2, index):
                        context.log_transaction(f'lower band cross')
                        if trade_config.entry_enabled == False:
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
                # Get trend prediction from the entry manager
                flag, pred = self.__entry_strategy.trend_prediction(context)

                if flag ==False:
                        # Transition to the entry preparation state
                        return

                context.dm.set_prediction(pred)
                self.trade_entry(context)
                # Transition to the position state
                context.change_to_entrypreparation_state()
                return

        def EntryPreparation_event_execute(self, context: SimulationStrategyContext)->None:
                """
                Executes an event when the counter exceeds the threshold in the entry preparation state.
                It makes an entry decision based on trend prediction and if favorable, enters a trade
                and transitions to the position state. Otherwise, it returns to the idle state.

                Args:
                        context (TradingContext): The trading context object.
                """
                serial = context.dm.get_fx_serial()
                date = context.dm.get_current_date()
                orderId = context.dm.get_order_id()


                position_status = self.__online_api.get_open_position_status()

                if position_status ==  "position":
                        context.log_transaction(f'Check Online Position status: {position_status}')
                        context.change_to_position_state()
                        return

                order_staus = self.__online_api.get_order_status(orderId)

                if order_staus != "Filled":
                        retcode  = self.__online_api.cancel_order(orderId)
                        context.log_transaction(f'cancel order retcode  : {retcode}')
                        context.fx_transaction.trade_cancel(serial,date)
                        context.log_transaction(f'Canceled order: {orderId},order_staus: {order_staus}')
                        context.change_to_idle_state()
                        return


                if self.is_losscut_triggered(context):
                        self._handle_position_exit(context, losscut=True)
                        return

                context.change_to_idle_state()
                return


        def PositionState_event_exit_execute(self, context: SimulationStrategyContext)->None:
                """
                Executes an exit event in the position state.
                This method is called when the exit conditions are met.
                Currently, it doesn't perform any specific action.

                Args:
                        context (TradingContext): The trading context object.
                """
                pass

        def PositionState_event_continue_execute(self, context: SimulationStrategyContext)->None:
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

                if operator == 'less_than':
                        if context.is_first_column_less_than_second(COLUMN_CLOSE, column, context.dm.get_current_index()):
                                        self._handle_position_exit(context)
                                        return

                elif operator == 'greater_than':
                        if context.is_first_column_greater_than_second(COLUMN_CLOSE, column, context.dm.get_current_index()):
                                        self._handle_position_exit(context,)
                                        return


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
                pass



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
                if  self.is_losscut_triggered(context):
                        self._handle_position_exit(context, losscut=True)
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
                return True


        def _handle_position_exit(self, context:SimulationStrategyContext, losscut=False):
                """
                Handles the process of exiting a position, recording the exit details and transitioning to the idle state.

                Args:
                        context (TradingContext): The trading context object.
                        exit_price (float): The exit price.
                        losscut (bool, optional): True if the exit is due to a loss-cut, False otherwise. Defaults to False.
                """
                # Determine the reason for exit
                if losscut:
                        reason = EXIT_REASON_LOSSCUT
                else:
                        reason = EXIT_REASON_NORMAL

                # Execute the trade exit
                pandl = self.online_trade_exit(context, losscut)

                context.dm.set_bb_profit(pandl, context.dm.get_entry_index())
                context.dm.set_exit_reason(reason)
                context.dm.set_exit_reason(reason, context.dm.get_entry_index())
                context.record_entry_exit_price()

                # Get the prediction and set the prediction target
                pred = context.dm.get_prediction()
                if pandl  > 0:
                        pred_target = pred
                else:
                        pred_target = 1 -  pred

                context.dm.set_pred_target(pred_target,  context.dm.get_entry_index())

                self.__entry_strategy.update_performance(context, context.dm.get_entry_index())

                # Record maximum and minimum profit and loss to the entry point
                context.record_max_min_pandl_to_entrypoint()
                # Transition to the idle state
                context.change_to_idle_state()
                return

        def trade_entry(self, context:SimulationStrategyContext):
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
                serial = context.fx_transaction.trade_entry(entry_type, pred, entry_price, date, bb_direction,leverage, losscut)
                # Set the transaction serial number in the data manager
                context.dm.set_fx_serial(serial)
                self.online_trade_entry(context, serial, entry_type, entry_price,leverage)

        def online_trade_entry(self, context:SimulationStrategyContext,serial,entry_type, entry_price,leverage):
                """
                Executes a trade entry, determining the entry type based on the Bollinger Band direction and prediction.
                It records the trade details and updates the transaction serial number in the data manager.

                Args:
                        context (TradingContext): The trading context object.
                        serial (int): The transaction serial number.
                        entry_type (str): The entry type (long or short).
                """

                qty = context.fx_transaction.get_qty(serial)
                losscut_price = context.fx_transaction.get_losscut_price(serial)
                self.__online_api.position_manager.set_my_leverage(leverage)
                orderId = self.__online_api.trade_entry_trigger(qty,
                                                                                                entry_type,
                                                                                                target_price=entry_price,
                                                                                                stop_loss_price=losscut_price)
                context.dm.set_order_id(orderId)

        def online_trade_exit(self, context:SimulationStrategyContext, losscut=False)->float:
                """
                Executes a trade exit, recording the exit details and updating the transaction serial number in the data manager.

                Args:
                        context (TradingContext): The trading context object.
                        losscut (bool, optional): True if the exit is due to a loss-cut, False otherwise. Defaults to None.

                Returns:
                        float: The profit and loss from the trade.
                """
                serial = context.dm.get_fx_serial()
                date = context.dm.get_current_date()
                trade_tpye = context.dm.get_entry_type()
                qty = context.fx_transaction.get_qty(serial)

                if losscut ==  False:
                        losscut_str = "normal"
                        print(f"trade_exit: {qty}, {trade_tpye}")
                        self.__online_api.trade_exit(qty, trade_tpye)
                else:
                        losscut_str = "losscut"

                time.sleep(10)
                pandl,exit_price = self.__online_api.get_closed_pnl()
                # Record the exit index and price in the data manager
                context.dm.set_exit_index(context.dm.get_current_index())
                context.dm.set_exit_price(exit_price)
                context.dm.set_bb_profit(pandl, context.dm.get_entry_index())
                context.dm.set_pandl(pandl)
                context.fx_transaction.trade_exit(serial, exit_price, date, pandl=pandl,losscut=losscut_str)
                context.log_transaction(f'Trade Exit price: {exit_price}, PnL: {pandl}')
                return pandl


        def is_losscut_triggered(self, context:SimulationStrategyContext) -> bool:
                """
                Checks if a loss-cut trigger is met or if there's no open position.

                Args:
                        context (TradingContext): The trading context object.

                Returns:
                        bool: True if a loss-cut is triggered or there's no position, False if there's an open position without losscut.
                """
                position_status = self.__online_api.get_open_position_status()
                context.log_transaction(f'Check Online Position status: {position_status}')

                # If there's an open position, no losscut
                if position_status != "No position":
                        return False
                return True


        def calculate_current_pandl(self, context:SimulationStrategyContext, exit_price=None):
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

        def show_win_lose(self, context):
                """
                Displays win-loss statistics for the trades.

                Args:
                        context (TradingContext): The trading context object.

                This method uses the fx_transaction object to display the win rates and plot the balance over time.
                """
                context.fx_transaction.display_all_win_rates()
                context.fx_transaction.plot_balance_over_time()




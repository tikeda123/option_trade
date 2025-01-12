import pandas as pd
import os, sys

# Get the absolute path of the directory containing this file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the path of the parent directory to sys.path
sys.path.append(parent_dir)

from trading_analysis_kit.trading_state import *
from trading_analysis_kit.trading_strategy import TradingStrategy
from trading_analysis_kit.simulation_strategy_context import SimulationStrategyContext

class MLDataCreationStrategy(TradingStrategy):
        """
        MLDataCreationStrategy is a class that implements a trading strategy
        specifically designed for creating machine learning training data.

        This class inherits from the TradingStrategy base class and overrides
        its methods to implement a specific strategy based on Bollinger Bands
        and other technical indicators.

        The strategy manages different states of trading:
        1. Idle state: Waiting for entry conditions
        2. Entry preparation: Setting up for a new trade
        3. Position state: Managing an open position

        The class provides methods to handle events in each of these states,
        as well as a method to decide when to exit a position.

        This strategy is designed to work within a simulation context, allowing
        for backtesting and data generation for machine learning models.
        """

        def should_entry(self, context, index: int) -> bool:
                """
                Determines whether to enter a trade based on the closing price crossing Bollinger Bands.

                Args:
                        context (TradingContext): The trading context object, containing market data and indicators.
                        index (int): The current data index.

                Returns:
                        bool: True if a trade should be entered, False otherwise.

                This method checks if the current closing price has crossed either the upper or lower Bollinger Band.
                If it has, it sets the Bollinger Band direction accordingly and signals to enter a trade.
                """
                # Check if closing price crossed above the upper Bollinger Band
                if context.is_first_column_greater_than_second(COLUMN_CLOSE, COLUMN_UPPER_BAND2, index):
                        context.dm.set_bb_direction(BB_DIRECTION_UPPER)
                        context.log_transaction("cross upper band")
                        return True
                # Check if closing price crossed below the lower Bollinger Band
                elif context.is_first_column_less_than_second(COLUMN_CLOSE, COLUMN_LOWER_BAND2, index):
                        context.dm.set_bb_direction(BB_DIRECTION_LOWER)
                        context.log_transaction("cross lower band")
                        return True

                # No crossing detected, no trade entry
                return False

        def Idle_event_execute(self, context : SimulationStrategyContext):
                """
                Execute actions for the idle state.
                This method is called when the trading system is in an idle state, waiting for entry conditions.

                Args:
                        context (SimulationStrategyContext): The context object containing trading data, indicators, and methods.

                Actions:
                        1. Record the current state as IDLE.
                        2. Change the state to entry preparation.
                """
                if context.dm.get_current_index() < TIME_SERIES_PERIOD:
                        return

                context.dm.record_state(STATE_IDLE)

                # Set the entry index and price based on the current market data
                context.dm.set_entry_index(context.dm.get_current_index())
                context.dm.set_entry_price(context.dm.get_close_price())

                # Set prediction to LONG regardless of the Bollinger Band direction
                # This might be a potential bug or intentional strategy choice
                context.dm.set_prediction(PRED_TYPE_LONG)
                # Record the state transition and move to the position state
                context.change_to_position_state()


        def EntryPreparation_event_execute(self, context: SimulationStrategyContext):
                """
                Execute actions for the entry preparation state.
                This method is called when the system is preparing to enter a trade, after entry conditions have been met.

                Args:
                        context (SimulationStrategyContext): The context object containing trading data, indicators, and methods.
                """
                pass

        def PositionState_event_exit_execute(self, context: SimulationStrategyContext):
                """
                Execute actions for exiting a position.
                This method is called when the system decides to exit a trade, based on the should_exit_position logic.

                Args:
                        context (SimulationStrategyContext): The context object containing trading data, indicators, and methods.
                """
                # Record the state as POSITION
                context.dm.record_state(STATE_POSITION)
                self._handle_position_exit(context)


        def PositionState_event_continue_execute(self, context : SimulationStrategyContext):
                """
                Execute actions for continuing a position.
                This method is called when the system decides to maintain the current position,
                based on the should_exit_position logic.

                Args:
                        context (SimulationStrategyContext): The context object containing trading data, indicators, and methods.
                """
                # Record the current state as POSITION
                context.dm.record_state(STATE_POSITION)
                context.dm.increment_entry_counter()
                self._handle_position_continue(context)

        def _handle_position_exit(self, context):
                context.set_current_max_min_pandl()

                # Set exit index and price based on the current market data
                context.dm.set_exit_index(context.dm.get_current_index())
                context.dm.set_exit_price(context.dm.get_close_price())

                # Calculate and record the profit for the trade
                profit = context.calculate_current_profit(context.dm.get_close_price())
                context.dm.set_bb_profit(profit, context.dm.get_entry_index())

                # Record entry and exit prices for analysis
                context.record_entry_exit_price()
                # Record maximum and minimum profit and loss relative to the entry point
                context.record_max_min_pandl_to_entrypoint()
                # Transition back to the idle state
                context.change_to_idle_state()


        def _handle_position_continue(self, context):
                context.set_current_max_min_pandl()

                # Calculate and set the current profit in the context
                current_profit = context.calculate_current_profit()
                context.dm.set_current_profit(current_profit)
                context.log_transaction("Current Profit: " + str(current_profit))


        def should_exit_position(self, context, index: int) -> bool:
                """
                Determines whether to exit the current position based on predefined exit conditions.

                Args:
                        context (TradingContext): The trading context object, containing market data and indicators.
                        index (int): The current data index.

                Returns:
                        bool: True if the position should be exited, False otherwise.

                This method defines exit conditions based on the current Bollinger Band direction and the prediction.
                If the exit conditions are met, it signals to exit the trade.
                """

                bb_direction = context.dm.get_bb_direction()
                pred = context.dm.get_prediction()

                # Define exit conditions based on Bollinger Band direction and prediction
                position_state_dict = {
                        (BB_DIRECTION_UPPER, PRED_TYPE_LONG): ['less_than', COLUMN_MIDDLE_BAND],
                        (BB_DIRECTION_LOWER, PRED_TYPE_LONG): ['greater_than', COLUMN_MIDDLE_BAND],
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
                                context.log_transaction("Exit condition met: close price less than middle band")
                                return True


                elif operator == 'greater_than':
                        if context.is_first_column_greater_than_second(COLUMN_CLOSE, column, context.dm.get_current_index()):
                                context.log_transaction("Exit condition met: close price greater than middle band")
                                return True

                return False


        def should_exit_preparataion (self, context, index: int) -> bool:
                """
                Determines whether to exit the current position based on predefined exit conditions.

                Args:
                        context (TradingContext): The trading context object, containing market data and indicators.
                        index (int): The current data index.

                Returns:
                        bool: True if the position should be exited, False otherwise.

                This method defines exit conditions based on the current Bollinger Band direction and the prediction.
                If the exit conditions are met, it signals to exit the trade.
                """
                return True



        def ExitPreparationState_event_exit_execute(self, context: SimulationStrategyContext)->None:
                """
                Executes an exit event in the position state.
                This method is called when the exit conditions are met.
                Currently, it doesn't perform any specific action.

                Args:
                        context (TradingContext): The trading context object.
                """
                # Record the state as POSITION
                context.dm.record_state(STATE_EXIT_PREPARATION)
                self._handle_position_exit(context)


        def ExitPreparationState_event_continue_execute(self, context)->None:
                """
                Executes a continue event in the position state.
                It checks for a loss-cut trigger and exits the position if necessary.
                If no loss-cut is triggered, it calculates and records the current profit and loss.

                Args:
                        context (TradingContext): The trading context object.
                """
                context.dm.record_state(STATE_EXIT_PREPARATION)
                self._handle_position_continue(context)
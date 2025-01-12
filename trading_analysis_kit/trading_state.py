import pandas as pd
import os, sys

from common.constants import *

class State:
        """
        Abstract state class. All concrete state classes should inherit from this class.

        This class provides a base for defining how to handle events based on the current state.
        """

        def handle_request(self, context, event):
                """
                Handles an event based on the current state. This method should be implemented by each subclass.

                Args:
                        context: The current context object.
                        event: The event to be handled.

                Raises:
                        NotImplementedError: This method is an abstract method and must be implemented in subclasses.
                """
                raise NotImplementedError("Each state must implement 'handle_request' method.")

        def invalid_event(self, event):
                """
                This method is called when an invalid event occurs for the current state.

                Args:
                        event: The invalid event.
                """
                print(f"Error: '{event}' event is invalid in the current state.")

class IdleState(State):
        """
        Idle State:
        This state represents the system waiting for a trading opportunity.
        It waits until specific conditions are met and then transitions to the next state.
        """

        def handle_request(self, context, event):
                """
                Handles state transitions based on the received event.

                Args:
                        context: The current context object.
                        event: The event to be handled.

                If the event is EVENT_ENTER_PREPARATION, it transitions to the EntryPreparationState.
                Other events are treated as invalid.
                """
                if event == EVENT_ENTER_PREPARATION:
                        context.log_transaction("Transitioning to Entry Preparation State.")
                        context.set_state(EntryPreparationState())
                elif event == EVENT_POSITION:
                        context.log_transaction("Transitioning to Position State.")
                        context.set_state(PositionState())
                elif event == EVENT_IDLE:
                        context.log_transaction(LOG_TRANSITION_TO_IDLE_FROM_POSITION)
                        context.set_state(IdleState())
                else:
                        self.invalid_event(event)

        def event_handle(self, context, index: int):
                """
                Handles specific events that occur when certain conditions are met in the idle state.

                Args:
                        context: The current context object.
                        index: The index of the data point where the event occurred.
                """
                # Record the current state as IDLE
                context.dm.record_state(STATE_IDLE)
                # If the entry condition is met, execute the Idle event in the strategy
                if context.strategy.should_entry(context, index):
                        context.strategy.Idle_event_execute(context)
                else:
                        # Log an empty message if no entry condition is met
                        context.log_transaction(" ")

class EntryPreparationState(State):
        """
        Entry Preparation State:
        This state represents the system preparing for a trade entry.
        It waits until the conditions for trade entry are met and then transitions to the PositionState.
        """

        def handle_request(self, context, event):
                """
                Handles state transitions based on the received event.

                Args:
                        context: The current context object.
                        event: The event to be handled.

                If the event is EVENT_POSITION, it transitions to the PositionState.
                If the event is EVENT_IDLE, it transitions back to the IdleState.
                Other events are treated as invalid.
                """
                if event == EVENT_POSITION:
                        context.log_transaction( "Entry event occurred. Transitioning to Position State.")
                        context.set_state(PositionState())
                elif event == EVENT_IDLE:
                        context.log_transaction("Transitioning to Idle State.")
                        context.set_state(IdleState())
                else:
                        self.invalid_event(event)

        def event_handle(self, context, index: int):
                """
                Handles events in the entry preparation state, delegating to the strategy for execution.

                Args:
                        context: The current context object.
                        index: The index of the data point where the event occurred.
                """
                # Record the current state as ENTRY_PREPARATION
                context.dm.record_state(STATE_ENTRY_PREPARATION)
                # Execute the EntryPreparation event in the strategy
                context.strategy.EntryPreparation_event_execute(context)

class PositionState(State):
        """
        Position State:
        This state represents the system having an open position in a trade.
        It manages the position and handles events related to exiting the position.
        """

        def handle_request(self, context, event):
                """
                Handles state transitions based on the received event.

                Args:
                        context: The current context object.
                        event: The event to be handled.

                If the event is EVENT_IDLE, it transitions to the IdleState.
                Other events are treated as invalid.
                """
                if event == EVENT_IDLE:
                        context.log_transaction(LOG_TRANSITION_TO_IDLE_FROM_POSITION)
                        context.set_state(IdleState())
                elif event == EVENT_EXIT_PREPARATION:
                        context.log_transaction("Transitioning to Exit Preparation State.")
                        context.set_state(ExitPreparationState())
                else:
                        self.invalid_event(event)

        def event_handle(self, context, index: int):
                """
                Handles events in the position state, deciding whether to exit or continue the position.

                Args:
                        context: The current context object.
                        index: The index of the data point where the event occurred.
                """
                # Record the current state as POSITION
                context.dm.record_state(STATE_POSITION)
                # Check if the exit condition is met based on the strategy
                if context.strategy.should_exit_position(context, index):
                        # Execute the PositionState exit event in the strategy
                        context.strategy.PositionState_event_exit_execute(context)
                else:
                        # Execute the PositionState continue event in the strategy
                        context.strategy.PositionState_event_continue_execute(context)


class ExitPreparationState(State):
        """
        Exit  State:
        This state represents the system having an open position in a trade.
        It manages the position and handles events related to exiting the position.
        """

        def handle_request(self, context, event):
                """
                Handles state transitions based on the received event.

                Args:
                        context: The current context object.
                        event: The event to be handled.

                If the event is EVENT_IDLE, it transitions to the IdleState.
                Other events are treated as invalid.
                """
                if event == EVENT_IDLE:
                        context.log_transaction(LOG_TRANSITION_TO_IDLE_FROM_POSITION)
                        context.set_state(IdleState())
                else:
                        self.invalid_event(event)

        def event_handle(self, context, index: int):
                """
                Handles events in the position state, deciding whether to exit or continue the position.

                Args:
                        context: The current context object.
                        index: The index of the data point where the event occurred.
                """
                # Record the current state as POSITION
                context.dm.record_state(STATE_EXIT_PREPARATION)
                # Check if the exit condition is met based on the strategy
                if context.strategy.should_exit_preparataion(context, index):
                        # Execute the PositionState exit event in the strategy
                        context.strategy.ExitPreparationState_event_exit_execute(context)
                else:
                        # Execute the PositionState continue event in the strategy
                        context.strategy.ExitPreparationState_event_continue_execute(context)


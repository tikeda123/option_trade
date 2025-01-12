from abc import ABC, abstractmethod

# Strategy Interface
class TradingStrategy(ABC):
        """
        Abstract base class for trading strategies. To implement a concrete trading strategy,
        you must inherit from this class and override all abstract methods.
        Each method defines the actions to be taken at different stages or states of a trade.
        """

        @abstractmethod
        def Idle_event_execute(self, context):
                """
                Method executed when the trade is in the idle state.

                Args:
                        context: Object providing the execution context of the trade.
                """
                pass

        @abstractmethod
        def EntryPreparation_event_execute(self, context):
                """
                Method executed when the counter exceeds the set threshold in the entry preparation state.

                Args:
                        context: Object providing the execution context of the trade.
                """
                pass

        @abstractmethod
        def PositionState_event_exit_execute(self, context):
                """
                Method executed when the exit conditions are met while holding a position.

                Args:
                        context: Object providing the execution context of the trade.
                """
                pass

        @abstractmethod
        def PositionState_event_continue_execute(self, context):
                """
                Method executed when the conditions for continuing the position are met while holding a position.

                Args:
                        context: Object providing the execution context of the trade.
                """
                pass

        def ExitPreparationState_event_exit_execute(self, context):
                """
                Method executed when the exit conditions are met while holding a position.

                Args:
                        context: Object providing the execution context of the trade.
                """
                pass

        @abstractmethod
        def ExitPreparationState_event_continue_execute(self, context):
                """
                Method executed when the conditions for continuing the position are met while holding a position.

                Args:
                        context: Object providing the execution context of the trade.
                """
                pass


        @abstractmethod
        def should_entry(self, context, index: int) -> bool:
                """
                Abstract method to determine whether to enter a trade.

                Args:
                        context: Object providing the execution context of the trade.
                        index: The current data index.

                Returns:
                        bool: True if should enter the trade, False otherwise.
                """
                raise NotImplementedError

        @abstractmethod
        def should_exit_position(self, context, index: int) -> bool:
                """
                Abstract method to determine whether to exit a position.

                Args:
                        context: Object providing the execution context of the trade.
                        index: The current data index.

                Returns:
                        bool: True if should exit the position, False otherwise.
                """
                raise NotImplementedError

        @abstractmethod
        def should_exit_preparataion(self, context, index: int) -> bool:
                """
                Abstract method to determine whether to exit a position.

                Args:
                        context: Object providing the execution context of the trade.
                        index: The current data index.

                Returns:
                        bool: True if should exit the position, False otherwise.
                """
                raise NotImplementedError




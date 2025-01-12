import pandas as pd
import os,sys

# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)

from common.constants import *
from common.utils import get_config

from trading_analysis_kit.trading_state import *
from trading_analysis_kit.trading_strategy import TradingStrategy
#from trading_analysis_kit.simulation_strategy_context import SimulationStrategyContext

class SimulationStrategy(TradingStrategy):
    """
    トレーディング戦略を実装するシミュレーションクラスです。
    トレードのエントリーとエグジットの判断、状態遷移の管理を行います。
    """
    def __init__(self):
        self.__config = get_config("ACCOUNT")
        self.__entry_rate= self.__config["ENTRY_RATE"]

    def Idel_event_execute(self, context):
        """
        Executes an event in the idle state by increasing the entry counter
        and transitioning to the entry preparation state.

        Args:
            context (TradingContext): The trading context object.
        """
        if context.dm.get_current_index() < 10:
            return

        context.dm.record_state(STATE_IDLE)
        context.change_to_entrypreparation_state()


    def EntryPreparation_event_execute(self, context):
        """
        エントリー準備状態でカウンターが閾値を超えた場合のイベントを実行します。
        エントリー判断を行い、エントリーする場合はポジション状態に遷移し、トレードのエントリーと
        エグジットラインの決定を行います。エントリーしない場合はアイドル状態に戻ります。

        Args:
            context (TradingContext): トレーディングコンテキストオブジェクト。
        """
        context.dm.record_state(STATE_ENTRY_PREPARATION)

        if context.entry_manager.should_entry(context):

            self.trade_entry(context)
            context.change_to_position_state()
            return

        context.change_to_idle_state()
        return

    def PositionState_event_exit_execute(self, context):
        """
        ポジション状態でのエグジットイベントを実行します。
        ロスカットがトリガーされた場合は、ロスカット価格でポジションを終了し、
        そうでない場合は現在の価格でポジションを終了します。
        その後、状態をアイドル状態に変更します。

        Args:
            context (TradingContext): トレーディングコンテキストオブジェクト。
        """
        is_losscut_triggered, exit_price = self.is_losscut_triggered(context)

        if is_losscut_triggered:
            self._handle_position_exit(context, exit_price, losscut=True)
        else:
            if self.should_hold_position(context):
                pandl = self.calculate_current_pandl(context)
                context.dm.set_pandl(pandl)
                context.log_transaction(f'continue Position state pandl:{pandl}')
                return

        self._handle_position_exit(context, context.dm.get_close_price())

    def PositionState_event_continue_execute(self, context):
        """
        ポジション状態での継続イベントを実行します。
        ロスカットの判断を行い、必要に応じてポジションを終了しアイドル状態に遷移します。
        ロスカットがトリガーされない場合は、現在の損益を計算し、ログに記録します。

        Args:
            context (TradingContext): トレーディングコンテキストオブジェクト。
        """
        is_losscut_triggered, exit_price = self.is_losscut_triggered(context)

        if is_losscut_triggered:
            self._handle_position_exit(context, exit_price, losscut=True)
            return

        is_profit_triggered,profit_triggered_price = context.is_profit_triggered()

        if is_profit_triggered:
            context.log_transaction(f'profit triggered price: {profit_triggered_price}')
            self._handle_position_exit(context, profit_triggered_price)
            return

        pandl = self.calculate_current_pandl(context)
        context.dm.set_pandl(pandl)
        context.log_transaction(f'continue Position state pandl:{pandl}')


    def _handle_position_exit(self, context, exit_price, losscut=False):
        """
        ポジションの終了処理を行います。

        Args:
            context (TradingContext): トレーディングコンテキストオブジェクト。
            exit_price (float, optional): ポジション終了価格。デフォルトはNone。
            losscut (bool, optional): ロスカットによる終了かどうか。デフォルトはFalse。
        """

        if losscut:
            context.log_transaction(f'losscut price: {exit_price}')
            self.trade_exit(context, exit_price, losscut="losscut")
        else:
            self.trade_exit(context, exit_price)

        context.change_to_idle_state()

    def trade_entry(self, context):
        """
        トレードのエントリーを実行します。

        Args:
            context (TradingContext): トレーディングコンテキストオブジェクト。
            pred (int): 予測結果（1または0）。

        ボリンジャーバンドの方向と予測結果に基づいてエントリータイプを決定し、
        トレードエントリーを実行します。
        """
        entry_price = context.dm.get_entry_price()
        date = context.dm.get_current_date()
        pred = context.dm.get_prediction()
        bb_direction = context.dm.get_bb_direction()
        entry_type = ENTRY_TYPE_LONG if pred == 1 else ENTRY_TYPE_SHORT
        context.dm.set_entry_type(entry_type)

        # トレードエントリーを実行し、トランザクションシリアル番号を取得
        serial = context.fx_transaction.trade_entry(entry_type, pred, entry_price, date, bb_direction)
        # 取得したトランザクションシリアル番号をコンテキストに設定
        context.dm.set_fx_serial(serial)

    def trade_exit(self, context, exit_price,losscut=None):
        """
        トレードのエグジットを実行します。

        Args:
            context (TradingContext): トレーディングコンテキストオブジェクト。
            exit_price (float): エグジットする価格。

        Returns:
            float: 実行されたトレードの損益。

        指定された価格でトレードのエグジットを実行し、損益を計算します。
        """
        serial = context.dm.get_fx_serial()
        date = context.dm.get_current_date()
        context.dm.set_exit_index(context.dm.get_current_index())
        context.dm.set_exit_price(exit_price)
        return context.fx_transaction.trade_exit(serial,exit_price, date, losscut=losscut)

    def is_losscut_triggered(self, context):
        """
        損切りがトリガーされたかどうかを判断します。

        Args:
            context (TradingContext): トレーディングコンテキストオブジェクト。

        Returns:
            bool, float: 損切りがトリガーされたかの真偽値と、損切りがトリガーされた場合の価格。

        現在の価格をもとに損切りがトリガーされたかどうかを判断し、
        トリガーされた場合はその価格を返します。
        """
        serial = context.dm.get_fx_serial()
        entry_type = context.dm.get_entry_type()
        losscut_price = None

        if entry_type == ENTRY_TYPE_LONG:
            losscut_price = context.dm.get_low_price()
        else:
            losscut_price = context.dm.get_high_price()

        return context.fx_transaction.is_losscut_triggered(serial, losscut_price)

    def calculate_current_pandl(self, context):
        """
        現在の損益を計算します。

        Args:
            context (TradingContext): トレーディングコンテキストオブジェクト。

        Returns:
            float: 現在の損益。

        現在の価格とエントリー価格をもとに損益を計算します。
        """
        serial = context.dm.get_fx_serial()
        current_price = context.dm.get_close_price()
        pandl = context.fx_transaction.get_pandl(serial, current_price)
        return pandl

    def decide_on_position_exit(self, context, index: int):
        bb_direction = context.dm.get_bb_direction()
        pred = context.dm.get_prediction()

        position_state_dict = {
            (BB_DIRECTION_UPPER, PRED_TYPE_LONG): ['less_than', COLUMN_MIDDLE_BAND],
            (BB_DIRECTION_UPPER, PRED_TYPE_SHORT): ['less_than', COLUMN_MIDDLE_BAND],
            (BB_DIRECTION_LOWER, PRED_TYPE_LONG): ['greater_than', COLUMN_MIDDLE_BAND],
            (BB_DIRECTION_LOWER, PRED_TYPE_SHORT): ['greater_than', COLUMN_MIDDLE_BAND]
        }

        condition = position_state_dict.get((bb_direction, pred))
        if condition is None:
            return 'PositionState_event_continue_execute'

        operator, column = condition

        if operator == 'less_than':
            if context.is_first_column_less_than_second(COLUMN_CLOSE, column,index):
                return 'PositionState_event_exit_execute'
        elif operator == 'greater_than':
            if context.is_first_column_greater_than_second(COLUMN_CLOSE, column,index):
                return 'PositionState_event_exit_execute'

        return 'PositionState_event_continue_execute'

    def should_hold_position(self, context):
        """
        ポジションを保持すべきかどうかを判断します。

        Args:
            context (TradingContext): トレーディングコンテキストオブジェクト。

        Returns:
            bool: ポジションを保持すべきかどうかの真偽値。

        ポジションを保持すべきかどうかを判断します。
        """
        return False
        trend_prediction = context.dm.get_prediction()
        rolling_pred = context.entry_manager.predict_trend_rolling(context)

        if rolling_pred != trend_prediction:
            return False

        return True

    def show_win_lose(self,context):
        """
        勝敗の統計情報を表示します。

        Args:
            context (TradingContext): トレーディングコンテキストオブジェクト。

        トレードの勝敗に関する統計情報を表示します。
        """
        context.fx_transaction.display_all_win_rates()
        context.fx_transaction.plot_balance_over_time()




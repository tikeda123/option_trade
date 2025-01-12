import pandas as pd
import os,sys

from common.constants import *
from common.utils import get_config

from trading_analysis_kit.trading_state import *
from trading_analysis_kit.trading_strategy import TradingStrategy
from bybit_api.bybit_trader import BybitTrader

class OnlineStrategy(TradingStrategy):
        """
        トレーディング戦略を実装するシミュレーションクラスです。
        トレードのエントリーとエグジットの判断、状態遷移の管理を行います。
        """
        def __init__(self):
                self.__config = get_config("ACCOUNT")
                self.__entry_rate= self.__config["ENTRY_RATE"]
                self.__online_api = BybitTrader()


        def Idel_event_execute(self, context):
                """
                Executes an event in the idle state by increasing the entry counter
                and transitioning to the entry preparation state.

                Args:
                        context (TradingContext): The trading context object.
                """
                if context.get_current_index() < 1:
                        return

                current_price = self.__online_api.get_current_price()
                ema_price = context.get_ema_price(context.get_current_index())
                trend_prediction = context.prediction_trend()

                # Determine the adjusted entry price based on trend prediction
                entry_price = self.calculate_adjusted_entry_price(ema_price, current_price, trend_prediction)

                self.trade_entry(context, trend_prediction, entry_price)
                context.change_to_entrypreparation_state()


        def calculate_adjusted_entry_price(self, ema_price, current_price, trend_prediction):
                """
                Calculates the adjusted entry price based on the EMA and the current price.

                Args:
                        ema_price (float): EMA price from the previous day.
                        current_price (float): Today's current price.
                        trend_prediction (int): Prediction of the market trend (1 for upward, other for downward).

                Returns:
                        float: The adjusted entry price.
                """
                adjustment_factor = 1 - self.__entry_rate if trend_prediction == 1 else 1 + self.__entry_rate
                ideal_price = ema_price * adjustment_factor

                return current_price if (trend_prediction == 1 and ideal_price > current_price) or \
                                                                (trend_prediction != 1 and ideal_price < current_price) else ideal_price


        def EntryPreparation_execute(self, context):
                """
                エントリー準備状態でカウンターが閾値を超えた場合のイベントを実行します。
                エントリー判断を行い、エントリーする場合はポジション状態に遷移し、トレードのエントリーと
                エグジットラインの決定を行います。エントリーしない場合はアイドル状態に戻ります。

                Args:
                        context (TradingContext): トレーディングコンテキストオブジェクト。
                """
                serial = context.get_fx_serial()
                date = context.get_current_date()
                orderId = context.get_order_id()

                order_staus = self.__online_api.get_order_status(orderId)

                if order_staus != "Filled":
                        self.__online_api.cancel_order(orderId)
                        context.fx_transaction.trade_cancel(serial,date)
                        context.log_transaction(f'Canceled order: {orderId},order_staus: {order_staus}')
                        context.change_to_idle_state()
                        return

                position_status = self.__online_api.get_open_position_status()
                context.log_transaction(f'Active Postion: {orderId},order_staus: {order_staus}')

                if position_status == "No position":
                        pandl,exit_price = self.__online_api.get_closed_pnl()
                        context.fx_transaction.trade_exit(serial,exit_price, date, pandl=pandl,losscut=exit_price)
                        context.log_transaction(f'losscut price: {exit_price}, PnL: {pandl}')
                        context.set_pandl(pandl)
                        context.change_to_idle_state()
                        return

                context.change_to_position_state()


        def PositionState_event_exit_execute(self, context):
                """
                ポジション状態でのエグジットイベントを実行します。ロスカットがトリガーされた場合は、
                ロスカット価格でポジションを終了し、そうでない場合は現在の価格でポジションを終了します。
                その後、状態をアイドル状態に変更します。

                Args:
                        context (TradingContext): トレーディングコンテキストオブジェクト。
                """
                serial = context.get_fx_serial()
                date = context.get_current_date()

                position_status = self.__online_api.get_open_position_status()

                if position_status == "No position":
                        pandl,exit_price = self.__online_api.get_closed_pnl()
                        context.fx_transaction.trade_exit(serial,exit_price, date, pandl=pandl,losscut=exit_price)
                        context.set_pandl(pandl)
                        context.log_transaction(f'losscut price: {exit_price}, PnL: {pandl}')
                        context.change_to_idle_state()
                        return

                if False == self.should_hold_position(context):
                        self.online_trade_exit(context)
                        context.log_transaction(f'Exit Position state')
                        context.change_to_idle_state()
                        return

                pandl = self.calculate_current_pandl(context)
                context.set_pandl(pandl)
                context.log_transaction(f'continue Position state PnL: {pandl}')

        def PositionState_event_continue_execute(self, context):
                """
                ポジション状態での継続イベントを実行します。ロスカットの判断を行い、必要に応じて
                ポジションを終了しアイドル状態に遷移します。

                Args:
                        context (TradingContext): トレーディングコンテキストオブジェクト。
                """
                pass
        def should_hold_position(self, context):
                """
                ポジションを保持すべきかどうかを判断します。

                Args:
                        context (TradingContext): トレーディングコンテキストオブジェクト。

                Returns:
                        bool: ポジションを保持すべきかどうかの真偽値。

                ポジションを保持すべきかどうかを判断します。
                """
                trend_prediction = context.prediction_trend()
                entry_type = context.get_entry_type()
                entry_price = context.get_ema_price()
                current_ema_price = context.get_ema_price(context.get_current_index())

                if trend_prediction == 1 and entry_type == ENTRY_TYPE_LONG and entry_price < current_ema_price:
                        return True
                elif trend_prediction == 0 and entry_type == ENTRY_TYPE_SHORT and entry_price > current_ema_price:
                        return True

                return False

        def trade_entry(self, context, pred,entry_price):
                """
                トレードのエントリーを実行します。

                Args:
                        context (TradingContext): トレーディングコンテキストオブジェクト。
                        pred (int): 予測結果（1または0）。

                予測結果に基づいてエントリータイプを決定し、
                トレードエントリーを実行します。
                """
                # 現在のBollinger Bandsの方向を取得
                if pred == 1:
                        entry_type = ENTRY_TYPE_LONG
                else:
                        entry_type = ENTRY_TYPE_SHORT

                context.set_entry_index(context.get_current_index())
                context.set_entry_type(entry_type)
                context.set_entry_price(entry_price)
                date = context.get_current_date()

                # トレードエントリーを実行し、トランザクションシリアル番号を取得
                serial = context.fx_transaction.trade_entry(entry_type, pred, entry_price, date, "upper")
                # 取得したトランザクションシリアル番号をコンテキストに設定
                context.set_fx_serial(serial)
                self.online_trade_entry(context, serial, entry_type, entry_price)

        def online_trade_entry(self, context,serial,entry_type, entry_price):
                """
                オンラインでトレードをエントリーします。

                Args:
                        context (TradingContext): トレーディングコンテキストオブジェクト。
                """
                qty = context.fx_transaction.get_qty(serial)
                losscut_price = context.fx_transaction.get_losscut_price(serial)

                orderId = self.__online_api.trade_entry_trigger(qty,
                                                                                                entry_type,
                                                                                                entry_price,
                                                                                                entry_price, # triggerPrice
                                                                                                losscut_price)
                context.set_order_id(orderId)


        def online_trade_exit(self, context):
                """
                オンラインでトレードをエグジットします。

                Args:
                        context (TradingContext): トレーディングコンテキストオブジェクト。
                """
                serial = context.get_fx_serial()
                date = context.get_current_date()
                trade_tpye = context.get_entry_type()
                qty = context.fx_transaction.get_qty(serial)

                self.__online_api.trade_exit(qty, trade_tpye)
                pandl,exit_price = self.__online_api.get_closed_pnl()
                context.fx_transaction.trade_exit(serial,exit_price,date,pandl=pandl)
                context.log_transaction(f'Trade Exit price: {exit_price}, PnL: {pandl}')
                context.set_pandl(pandl)
                return


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
                serial = context.get_fx_serial()
                entry_type = context.get_entry_type()
                losscut_price = None

                if entry_type == ENTRY_TYPE_LONG:
                        losscut_price = context.get_low_price()
                else:
                        losscut_price = context.get_high_price()

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
                serial = context.get_fx_serial()
                current_price = context.get_current_price()
                pandl = context.fx_transaction.get_pandl(serial, current_price)
                return pandl





#
# Python Script with Long Short Class
# for Event-Based Backtesting
#
# Python for Algorithmic Trading
# (c) Toshihiko Ikeda
# The Python Quants GmbH
#
#from pybit.spot import HTTP
from pybit import usdt_perpetual

from datetime import datetime, timezone
import time

from typing import Tuple
import json
import pandas as pd
import traceback
import math
import traceback
import asyncio

from trading_def import *
from trading_logger import *


def datetime_to_utime(time):
    '''
        デフォルトはUTCで計算
        文字列からunix timeへの変換
    '''
    dt = datetime.strptime(time, '%Y-%m-%d %H:%M:%S%z')
    return dt.timestamp()


def datetime_to_utime_ISO8601(time):
    '''
        デフォルトはUTCで計算
        文字列からunix timeへの変換
    '''
    dt = datetime.strptime(time, '%Y-%m-%dT%H:%M:%SZ')
    return dt.timestamp()


def utime_to_datetime(time):
    # unix timeから文字列からへの変換
    # UTC 時間を表すタイムゾーン情報
    dt = datetime.utcfromtimestamp(time)
    return dt


class BybitOnlineAPI:
    '''
        ByBITからオンラインでデリバティブUSDT無限のトレード情報を取ってくる処理
    '''

    def __init__(self, var_conf: dict):
        conf = var_conf[const.SET_BYBIT_API]
        self.__sym = var_conf[const.SET_ONLINE][const.SET_SYMBOL]
        self.__url = conf[const.SET_URL]
        self.__api_key = conf[const.SET_API_KEY]
        self.__api_secret = conf[const.SET_API_SECRET]

    def latest_info_sym(self) -> Tuple[bool, list]:
        '''
            Get the latest information for symbol.
            戻り値:
            Parameter	    Type	Comment
            id	            string	Latest data ID
            symbol	        string	Symbol
            price	        number	Execution price
            qty	            number	Order quantity in cryptocurrency
            side	        string	Side of taker order
            time	        string	UTC time
            trade_time_ms	number	Millisecond timestamp
            is_block_trade	boolean	Is block trade
            (https://bybit-exchange.github.io/docs/futuresV2/linear/#t-publictradingrecords)
        '''
        try:
            self.__session = usdt_perpetual.HTTP(endpoint=self.__url)
            result = self.__session.latest_information_for_symbol(symbol=self.__sym)[
                const.RESULT][0]

        except Exception as e:
            tdlg().syslm(f'ByBit HTTP Access Error:{e}')
            return False, 0

        return True, result

    def sub_mark_price_kline(self, interval: int = 1, limit: int = 1, fromtime: str = None) -> Tuple[bool, list]:
        '''
            ByBITからオンラインでデリバティブUSDT無限のマーク価格を取得
            lm:何個データを取得するか（通常は１個、初期は３０ぐらい）
            inter:何分足のデータを取得するか
            fromtime:
            bool:Falseの場合、Noneで設定
            戻り値  bool
            戻り値　list
            Parameter       Type	Comment
            symbol	        string	Symbol
            period	        string	Data recording period. 5min, 15min, 30min, 1h, 4h, 1d
            open_time	    integer	Start timestamp point for result, in seconds
            open	        string	Starting price
            high	        string	Maximum price
            low	            string	Minimum price
            close	        string	Closing price
            (https://bybit-exchange.github.io/docs/futuresV2/linear/#t-markpricekline)
        '''

        if fromtime is None:
            # デフォルトは現在時刻で１分単位でデータ取得のため、秒、ミリ秒単位は切り捨て:UTCでリクエストを投げる。
            tt = datetime.now(timezone.utc).replace(
                second=0, microsecond=0)
            fromtime = datetime_to_utime(str(tt))
            fromtime = fromtime - (const.SixtySec*interval*limit)

        try:
            self.__session = usdt_perpetual.HTTP(endpoint=self.__url)
            result = self.__session.query_mark_price_kline(
                symbol=self.__sym, interval=interval, limit=limit, from_time=fromtime)[const.RESULT]
        except Exception as e:
            tdlg().syslm(f'ByBit HTTP Access Error:{e}')
            return False, 0

        return True, result

    def mark_price_kline(self, interval: int = None, limit: int = None, fromtime: str = None) -> list:
        '''
            ByBITからオンラインでデリバティブUSDT無限のマーク価格を取得
            lm:何個データを取得するか（通常は１個、初期は３０ぐらい）
            inter:何分足のデータを取得するか
            fromtime:
            戻り値　list
            Parameter       Type	Comment
            symbol	        string	Symbol
            period	        string	Data recording period. 5min, 15min, 30min, 1h, 4h, 1d
            open_time	    integer	Start timestamp point for result, in seconds
            open	        string	Starting price
            high	        string	Maximum price
            low	            string	Minimum price
            close	        string	Closing price
            (https://bybit-exchange.github.io/docs/futuresV2/linear/#t-markpricekline)
        '''
        for _ in range(const.MAX_TRYOUT_HTTP):
            flag, res = self.sub_mark_price_kline(interval, limit, fromtime)
            if flag == True:
                return res
            time.sleep(const.SLEEP_HTTP)
        else:
            tdlg().syslm('mark_price_kline:MAX TRY OUT Error.')
            exit(-1)

    def fetch_historical_price_data(self, inter, fromtime, totime, savefilename=None, path=None):
        '''
            ByBITへヒストリカルデータを取得し、csvファイルとしてデータを保存
            sym:BTCUSDT,ETHUSDTなど
            inter:15,30,60など
            fromtime:"2022-01-01 00:00:00+00:00"のようなUTC形式[いつから]
            totime:"2022-01-01 00:00:00+00:00"のようなUTC形式[いつまで]
        '''

        # savefilename作成　’BYBIT_BTCUSDT_2022-08-01_2022-12-28_15m.csv’のような形式
        if savefilename == None:
            savefilename = const.BYBIT+'_' + self.__sym + '_' + \
                fromtime.split(' ')[0] + '_'+totime.split(' ')[0] + \
                '_'+f'{inter}m'+const.FILE_IDF_CSV

        # ファイル保存のためのディレクトリがあれば指定する。
        if path != None:
            savefilename = path + savefilename

        tt = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        u_now = datetime_to_utime(str(tt))  # 現在の時刻をutimeに変換（秒）（秒）
        u_fromtime = datetime_to_utime(fromtime)  # unixtimeに変換（秒）
        u_totime = datetime_to_utime(totime)  # unixtimeに変換（秒）

        # 現在の時刻と比べ大きかったらFalseとして返答
        if u_fromtime > u_now or u_totime > u_now:
            return False

        diff = u_totime - u_fromtime
        numrecord = diff/(inter*60)

        # (query_mark_price_klineの最大取得レコード)で何回データをFetchすれば良いか
        rec_cnt = math.ceil(numrecord/const.MAXLIME_PQUARY)

        df_root = []
        for i in range(int(rec_cnt)):
            time.sleep(0.5)
            # interval*60秒*200(query_mark_price_klineの最大取得レコード)
            ftime = u_fromtime + inter*60*const.MAXLIME_PQUARY*i

            try:
                self.__session = usdt_perpetual.HTTP(endpoint=self.__url)
                result = self.__session.query_mark_price_kline(
                    symbol=self.__sym, interval=inter, from_time=ftime, limit=const.MAXLIME_PQUARY)[const.RESULT]
            except Exception as e:
                tdlg().syslm(f'ByBit HTTP Access Error:{e}')
                raise Exception

            # ByBitからデータを取得できない可能性もあり。
            if len(result) == 0:
                return False

            if i == 0:
                df_root = pd.DataFrame.from_records(result)
            else:
                df = pd.DataFrame.from_records(result)
                df_root = pd.concat([df_root, df], axis=0)
            tdlg().syslm(df_root.tail())

        df_root[const.COL_DATE] = pd.to_datetime(
            (df_root[const.COL_START_AT]), unit='s')
        df_root = df_root.set_index(
            const.COL_START_AT)  # 'start_at'のColumnをに変換
        print(df_root)
        df_root.to_csv(savefilename)
        return True

    def sub_set_leverage(self, buy_lv: int, sell_lv: int) -> bool:
        '''
            symbolごとのLeverage設定
            buy_lv:買いのレバレッジ値
            sell_lv:売りのレバレッジ値
            戻り値：設定結果（成功・失敗などの情報）
        '''
        try:
            self.__session = usdt_perpetual.HTTP(
                endpoint=self.__url, api_key=self.__api_key, api_secret=self.__api_secret)
            res = self.__session.set_leverage(
                symbol=self.__sym, buy_leverage=buy_lv, sell_leverage=sell_lv)
            return True
        except Exception as e:
            # Leverageが変更なしの場合はLeverage not modified (ErrCode: 34036) となるが無視できるエラー
            #print(f'ByBit HTTP Access Error:{e}')
            if 'leverage not modified' in e.message:
                tdlg().syslm(
                    f'Leverage not modified (ErrCode: 34036):ignore error')
                return True

            tdlg().syslm(f'ByBit HTTP Access Error:{e}')
            return False

    def set_leverage(self, buy_lv: int, sell_lv: int) -> None:
        '''
            symbolごとのLeverage設定
            buy_lv:買いのレバレッジ値
            sell_lv:売りのレバレッジ値
            戻り値：設定結果（成功・失敗などの情報）
        '''
        for _ in range(const.MAX_TRYOUT_HTTP):
            flag = self.sub_set_leverage(buy_lv, sell_lv)
            if flag == True:
                return
            time.sleep(const.SLEEP_HTTP)
        else:
            tdlg().syslm('set_leverage:MAX TRY OUT Error.')
            exit(-1)

    def trading_fee(self) -> list:
        '''
            Query Trading Fee Rate
            戻り値
            Parameter	Type	Comment
            predicted_funding_rate	number	Predicted funding rate
            predicted_funding_fee	number	Predicted funding fee
        '''
        try:

            self.__session = usdt_perpetual.HTTP(
                endpoint=self.__url, api_key=self.__api_key, api_secret=self.__api_secret)
            result = self.__session.query_trading_fee_rate(symbol=self.__sym)[
                const.RESULT]

        except Exception as e:
            tdlg().syslm(f'ByBit HTTP Access Error:{e}')
            exit(-1)

        return result

    def set_loss_cut(self, side: str, loss_price: float) -> None:
        '''
            set stop_loss to current position
            side: 'Buy' or 'Sell'
            loss_price:損切りターゲット価格
        '''
        try:
            self.__session = usdt_perpetual.HTTP(
                endpoint=self.__url, api_key=self.__api_key, api_secret=self.__api_secret)
            res = self.__session.set_trading_stop(
                symbol=self.__sym, side=side, sl_size=loss_price)
        except Exception as e:
            tdlg().syslm(f'ByBit HTTP Access Error:{e}')
            exit(-1)

        return res

    def sub_get_pandl(self) -> Tuple[bool, list]:
        '''
            Get user's closed profit and loss records. The results are ordered in descending order
            (the first item is the latest).
            戻り値: bool
            戻り値: list
            Parameter	        Type	Comment
            id                  number	PositionID
            user_id             number	UserID
            symbol	            string	Symbol
            order_id	        string	Order ID of the closing order
            side	            string	Side of the closing order
            qty	number	        Order   qty
            order_price	        number	Order price
            order_type	        string	Order type
            exec_type	        string	Exec type
            closed_size	        number	Closed size
            cum_entry_value	    number	Closed position value
            avg_entry_price	    number	Average entry price
            cum_exit_value	    number	Cumulative trading value of position closing orders
            avg_exit_price	    number	Average exit price
            closed_pnl	        number	Closed Profit and Loss
            fill_count	        number	The number of fills in a single order
            leverage	        number	In Isolated Margin mode, the value is set by the user. In Cross Margin mode, the value is the max leverage at current risk level
            created_at	        number	Creation time (when the order_status was Created)
            (https://bybit-exchange.github.io/docs/futuresV2/linear/#t-closedprofitandloss)
        '''
        try:
            self.__session = usdt_perpetual.HTTP(
                endpoint=self.__url, api_key=self.__api_key, api_secret=self.__api_secret)
            res = self.__session.closed_profit_and_loss(
                symbol=self.__sym)
        except Exception as e:
            tdlg().syslm(f'ByBit HTTP Access Error:{e}')
            return False, 0

        return True, res[const.RESULT][const.COL_DATA]

    def get_pandl(self):
        '''
            Get user's closed profit and loss records. The results are ordered in descending order
            (the first item is the latest).
            戻り値: list
            Parameter	        Type	Comment
            id                  number	PositionID
            user_id             number	UserID
            symbol	            string	Symbol
            order_id	        string	Order ID of the closing order
            side	            string	Side of the closing order
            qty	number	        Order   qty
            order_price	        number	Order price
            order_type	        string	Order type
            exec_type	        string	Exec type
            closed_size	        number	Closed size
            cum_entry_value	    number	Closed position value
            avg_entry_price	    number	Average entry price
            cum_exit_value	    number	Cumulative trading value of position closing orders
            avg_exit_price	    number	Average exit price
            closed_pnl	        number	Closed Profit and Loss
            fill_count	        number	The number of fills in a single order
            leverage	        number	In Isolated Margin mode, the value is set by the user. In Cross Margin mode, the value is the max leverage at current risk level
            created_at	        number	Creation time (when the order_status was Created)
            (https://bybit-exchange.github.io/docs/futuresV2/linear/#t-closedprofitandloss)
        '''
        for i in range(const.MAX_TIMEOUT_CNT):
            # Bybit側の実現損益データに反映されるのに時差があるため、最初だけ、少し調整する必要がある。
            if i == 0:
                tdlg().syslm(
                    f'Please wait {const.SLEEP_PNL}sec for getting P&L.')
                time.sleep(const.SLEEP_PNL)

            flag, res = self.sub_get_pandl()
            if flag == True:
                return res
            time.sleep(const.SLEEP_HTTP)
        else:
            tdlg().syslm('get_pandl:MAX TRY OUT Error.')
            exit(-1)

    def get_pandl_orderid(self, order_id):
        '''
            Get user's closed profit and loss records. The results are ordered in descending order
            (the first item is the latest).
            戻り値: list
            Parameter	        Type	Comment
            id                  number	PositionID
            user_id             number	UserID
            symbol	            string	Symbol
            order_id	        string	Order ID of the closing order
            side	            string	Side of the closing order
            qty	number	        Order   qty
            order_price	        number	Order price
            order_type	        string	Order type
            exec_type	        string	Exec type
            closed_size	        number	Closed size
            cum_entry_value	    number	Closed position value
            avg_entry_price	    number	Average entry price
            cum_exit_value	    number	Cumulative trading value of position closing orders
            avg_exit_price	    number	Average exit price
            closed_pnl	        number	Closed Profit and Loss
            fill_count	        number	The number of fills in a single order
            leverage	        number	In Isolated Margin mode, the value is set by the user. In Cross Margin mode, the value is the max leverage at current risk level
            created_at	        number	Creation time (when the order_status was Created)
            (https://bybit-exchange.github.io/docs/futuresV2/linear/#t-closedprofitandloss)
        '''
        for i in range(const.MAX_TIMEOUT_CNT):
            # Bybit側の実現損益データに反映されるのに時差があるため、最初だけ、少し調整する必要がある。
            if i == 0:
                tdlg().syslm(
                    f'Please wait {const.SLEEP_PNL}sec for getting P&L.')
                time.sleep(const.SLEEP_PNL)

            flag, res = self.sub_get_pandl()
            if flag == True:
                for j in range(len(res)):
                    if res[j][const.COL_ORDER_ID] == order_id:
                        return res[j]
            time.sleep(const.SLEEP_HTTP)
        else:
            tdlg().syslm('get_pandl_orderid:MAX TRY OUT Error.')
            exit(-1)

    def get_pandl_for_maker(self, qty):
        '''
            Get user's closed profit and loss records. The results are ordered in descending order
            (the first item is the latest).
            引き数: qty :エントリー時に購入した商品のサイズ;このサイズ分売却した利益の総和を求める。指値によるClose Positon_makerは分割して
            されて取引させる可能性がある。
            戻り値: pandl:利益
            戻り値: list[0]の情報
            Parameter	        Type	Comment
            id                  number	PositionID
            user_id             number	UserID
            symbol	            string	Symbol
            order_id	        string	Order ID of the closing order
            side	            string	Side of the closing order
            qty	                number	Order   qty
            order_price	        number	Order price
            order_type	        string	Order type
            exec_type	        string	Exec type
            closed_size	        number	Closed size
            cum_entry_value	    number	Closed position value
            avg_entry_price	    number	Average entry price
            cum_exit_value	    number	Cumulative trading value of position closing orders
            avg_exit_price	    number	Average exit price
            closed_pnl	        number	Closed Profit and Loss
            fill_count	        number	The number of fills in a single order
            leverage	        number	In Isolated Margin mode, the value is set by the user. In Cross Margin mode, the value is the max leverage at current risk level
            created_at	        number	Creation time (when the order_status was Created)
            (https://bybit-exchange.github.io/docs/futuresV2/linear/#t-closedprofitandloss)
        '''
        for i in range(const.MAX_TIMEOUT_CNT):
            # Bybit側の実現損益データに反映されるのに時差があるため、最初だけ、少し調整する必要がある。
            if i == 0:
                tdlg().syslm(
                    f'Please wait {const.SLEEP_PNL}sec for getting P&L.')
                time.sleep(const.SLEEP_PNL)

            flag, res = self.sub_get_pandl()
            if flag == True:
                total_size = qty
                tm_size = 0
                pandl = 0
                for j in range(len(res)):
                    cl_size = res[j][const.COL_CLOSED_SIZE]
                    tm_size += cl_size
                    pandl += res[j][const.COL_CLOSED_PNL]

                    if total_size == tm_size:
                        return pandl, res[0]

            time.sleep(const.SLEEP_HTTP)
        else:
            tdlg().syslm('get_pandl_orderid:MAX TRY OUT Error.')
            exit(-1)

    def latest_price(self) -> float:
        '''
            Get the latest price data for symbol.
            float:latest price data
        '''

        for _ in range(const.MAX_TRYOUT_HTTP):
            flag, res = self.latest_info_sym()
            if flag == True:
                return float(res[const.COL_LAST_PRICE])
            time.sleep(const.SLEEP_HTTP)
        else:
            tdlg().syslm('latest_price:MAX TRY OUT Error.')
            exit(-1)

    def sub_place_order(self,
                        side: str,
                        qty: float,
                        otype: str,
                        entry_price: float,
                        time_in_force: bool,
                        reduceflg: bool,
                        losscut_price: float
                        ) -> Tuple[bool, list]:
        '''
            参入注文
            引数
            Parameter	    Type	Comment
            side            str     "Buy" or "Sell"
            oype	        string	"Limit"=指値、"Market"成行
            qty	    	    number	Order quantity in cryptocurrency
            entry_price     number	Order price. Required if you make limit price order
            time_in_force   string	Time in force:"GoodTillCancel"を設定
            reduce_only     bool	What is a reduce-only order? True means your position
                                    can only reduce in size if this order is triggered.
                                    When reduce_only is true, take profit/stop loss cannot be set
            stop_loss       float   損切り価格
            戻り値：注文情報
            Parameter	    Type	Comment
            order_id	    string	Order ID
            user_id	        number	UserID
            symbol	        string	Symbol
            side	        string	Side
            order_type	    string	Order type
            price	        number	Order price
            qty	            number	Order quantity in USD
            time_in_force	string	Time in force
            order_status	string	Order status
            last_exec_price	number	Last execution price
            cum_exec_qty	number	Cumulative qty of trading
            cum_exec_value	number	Cumulative value of trading
            cum_exec_fee	number	Cumulative trading fees
            reduce_only	    bool	true means close order, false means open position
            close_on_trigger	bool	Is close on trigger order
            order_link_id	string	Unique user-set order ID. Maximum length of 36 characters
            created_time	string	Creation time (when the order_status was Created)
            updated_time	string	Update time
            take_profit	    number	Take profit price
            stop_loss	    number	Stop loss price
            tp_trigger_by	string	Take profit trigger price type, default: LastPrice
            sl_trigger_by	string	Stop loss trigger price type, default: LastPrice
            position_idx	integer	Position idx, used to identify positions in different position modes:
                                    0-One-Way Mode
                                    1-Buy side of both side mode
                                    2-Sell side of both side mode
            (https://bybit-exchange.github.io/docs/futuresV2/linear/#t-placeactive)

        '''

        try:
            self.__session = usdt_perpetual.HTTP(
                endpoint=self.__url, api_key=self.__api_key, api_secret=self.__api_secret)
            res = self.__session.place_active_order(
                symbol=self.__sym,
                side=side,
                order_type=otype,
                qty=qty,
                price=entry_price,
                time_in_force=time_in_force,
                reduce_only=reduceflg,
                close_on_trigger=False,
                stop_loss=None  # testnetのパルス対策本番は外す。（loss_cut不可）
                # stop_loss=losscut_price
            )
        except Exception as e:
            tdlg().syslm(f'ByBit HTTP Access Error:{e}')
            return False, None

        return True, res[const.COL_RESULT]

    def place_order(self,
                    side: str,
                    qty: float,
                    otype: str = 'Market',
                    entry_price: float = None,
                    time_in_force: str = "GoodTillCancel",
                    reduceflg: bool = False,
                    losscut_price: float = None) -> dict:
        '''
            参入注文
            side(必須):"Buy" or "Sell"
            qty(必須):symの量
            price:sideが"Limit"の場合は必須
            reduceflg:True/False Trueの場合は手持ちの通貨使う
            戻り値：注文情報
            Parameter	    Type	Comment
            order_id	    string	Order ID
            user_id	        number	UserID
            symbol	        string	Symbol
            side	        string	Side
            order_type	    string	Order type
            price	        number	Order price
            qty	            number	Order quantity in USD
            time_in_force	string	Time in force
            order_status	string	Order status
            last_exec_price	number	Last execution price
            cum_exec_qty	number	Cumulative qty of trading
            cum_exec_value	number	Cumulative value of trading
            cum_exec_fee	number	Cumulative trading fees
            reduce_only	    bool	true means close order, false means open position
            close_on_trigger	bool	Is close on trigger order
            order_link_id	string	Unique user-set order ID. Maximum length of 36 characters
            created_time	string	Creation time (when the order_status was Created)
            updated_time	string	Update time
            take_profit	    number	Take profit price
            stop_loss	    number	Stop loss price
            tp_trigger_by	string	Take profit trigger price type, default: LastPrice
            sl_trigger_by	string	Stop loss trigger price type, default: LastPrice
            position_idx	integer	Position idx, used to identify positions in different position modes:
                                    0-One-Way Mode
                                    1-Buy side of both side mode
                                    2-Sell side of both side mode
            (https://bybit-exchange.github.io/docs/futuresV2/linear/#t-placeactive)
        '''
        for _ in range(const.MAX_TRYOUT_HTTP):
            flag, res = self.sub_place_order(
                side, qty, otype, entry_price, time_in_force, reduceflg, losscut_price)
            if flag == True:
                return res
            time.sleep(const.SLEEP_HTTP)
        else:
            tdlg().syslm('place_order:MAX TRY OUT Error.')
            exit(-1)

    def trade_entry(self, side: str, equity: float, leverage: float, loss_cut: float) -> dict:
        '''
            参入注文（成行注文）
            sym(必須):"BTCUSDT","ETHUSDT"など
            side(必須):"Buy" or "Sell"
            qty(必須):symの量
            price:sideが"Limit"の場合は必須
            戻り値：注文情報
            result[0]:"Buy",resutl[1]:"Sell"
            Parameter	        Type	Comment
            user_id	            number	UserID
            symbol	            string	Symbol
            side	            string	Side
            size	            number	Position qty
            position_value	    number	Position value
            entry_price	        number	Average opening price
            liq_price	        number	Liquidation price
            bust_price	        number	Bust price
            leverage	        number	In Isolated Margin mode, the value is set by the user. In Cross Margin mode, the value is the max leverage at current risk level
            auto_add_margin	    number	Whether or not auto-margin replenishment is enabled
            is_isolated	bool	true means isolated margin mode; false means cross margin mode
            position_margin	    number	Position margin
            occ_closing_fee	    number	Pre-occupancy closing fee
            realised_pnl	    number	Today's realised Profit and Loss
            cum_realised_pnl	number	Cumulative realised Profit and Loss
            free_qty	        number	Qty which can be closed. (If you have a long position, free_qty is negative. vice versa)
            tp_sl_mode	        string	Stop loss and take profit mode
            deleverage_indicator	number	Deleverage indicator level (1,2,3,4,5)
            unrealised_pnl	    number	unrealised pnl
            risk_id	            integer	Risk ID
            take_profit	        number	Take profit price
            stop_loss	        number	Stop loss price
            trailing_stop	    number	Trailing stop (the distance from the current price)
            position_idx	    integer	Position idx, used to identify positions in different position modes:
                                        0-One-Way Mode
                                        1-Buy side of both side mode
                                        2-Sell side of both side mode
            mode	            string	Position Mode. MergedSingle: One-Way Mode; BothSide: Hedge Mode
            (https://bybit-exchange.github.io/docs/futuresV2/linear/#t-myposition)
        '''
        # レバレッジの設定
        try:
            self.set_leverage(leverage, leverage)  # leverageの設定
            entry_price = self.latest_price()  # 現在の価格を取得
            qty = round((equity*leverage)/entry_price, 3)  # 小数点第3位までを扱う
            loss_eq = equity*loss_cut

            if side == 'Buy':
                # Stop lossされる価格の下限を計算
                loss_price = round(
                    entry_price - (loss_eq/qty), 2)  # 小数点第２位までを扱う
            else:  # SHORT ENTRYの場合
                loss_price = round(
                    entry_price + (loss_eq/qty), 2)  # 小数点第２位までを扱う

            self.place_order(
                side, qty, losscut_price=loss_price)  # オーダー処理(成行注文）

            # 注文が成立した時のPostion情報を取得
            res = self.get_position()

        except:
            traceback.print_exc()
            exit(-1)

        # 'Sell'の場合はget_positionのres indexは1,'Buy’の場合は0
        index = 0 if side == 'Buy' else 1
        return res[index]

    def sub_trade_entry_maker(self, side: str, qty: float, equity: float, loss_cut: float, entry_price: float) -> dict:
        '''
            参入注文（成行注文）
            sym(必須):"BTCUSDT","ETHUSDT"など
            side(必須):"Buy" or "Sell"
            qty(必須):symの量
            price:sideが"Limit"の場合は必須
            戻り値：注文情報
            Parameter	    Type	Comment
            order_id	    string	Order ID
            user_id	        number	UserID
            symbol	        string	Symbol
            side	        string	Side
            order_type	    string	Order type
            price	        number	Order price
            qty	            number	Order quantity in USD
            time_in_force	string	Time in force
            order_status	string	Order status
            last_exec_price	number	Last execution price
            cum_exec_qty	number	Cumulative qty of trading
            cum_exec_value	number	Cumulative value of trading
            cum_exec_fee	number	Cumulative trading fees
            reduce_only	    bool	true means close order, false means open position
            close_on_trigger	bool	Is close on trigger order
            order_link_id	string	Unique user-set order ID. Maximum length of 36 characters
            created_time	string	Creation time (when the order_status was Created)
            updated_time	string	Update time
            take_profit	    number	Take profit price
            stop_loss	    number	Stop loss price
            tp_trigger_by	string	Take profit trigger price type, default: LastPrice
            sl_trigger_by	string	Stop loss trigger price type, default: LastPrice
            position_idx	integer	Position idx, used to identify positions in different position modes:
                                    0-One-Way Mode
                                    1-Buy side of both side mode
                                    2-Sell side of both side mode
            (https://bybit-exchange.github.io/docs/futuresV2/linear/#t-placeactive)
        '''
        loss_eq = equity*loss_cut

        if side == 'Buy':
            # Stop lossされる価格の下限を計算
            loss_price = round(entry_price - (loss_eq/qty), 2)  # 小数点第２位までを扱う
        else:  # SHORT ENTRYの場合
            loss_price = round(entry_price + (loss_eq/qty), 2)  # 小数点第２位までを扱う

        res = self.place_order(side, qty, otype='Limit', entry_price=entry_price,
                               losscut_price=loss_price, time_in_force="PostOnly")  # オーダー処理(指値注文）

        return res

    """
    def trade_entry_maker(self, side: str, equity: float, leverage: float, loss_cut: float, entry_price=None) -> Tuple[bool, list]:
        '''
            参入注文（成行注文）
            sym(必須):"BTCUSDT","ETHUSDT"など
            side(必須):"Buy" or "Sell"
            qty(必須):symの量
            price:sideが"Limit"の場合は必須
            戻り値：注文情報
            result[0]:"Buy",resutl[1]:"Sell"
            Parameter	        Type	Comment
            user_id	            number	UserID
            symbol	            string	Symbol
            side	            string	Side
            size	            number	Position qty
            position_value	    number	Position value
            entry_price	        number	Average opening price
            liq_price	        number	Liquidation price
            bust_price	        number	Bust price
            leverage	        number	In Isolated Margin mode, the value is set by the user. In Cross Margin mode, the value is the max leverage at current risk level
            auto_add_margin	    number	Whether or not auto-margin replenishment is enabled
            is_isolated	bool	true means isolated margin mode; false means cross margin mode
            position_margin	    number	Position margin
            occ_closing_fee	    number	Pre-occupancy closing fee
            realised_pnl	    number	Today's realised Profit and Loss
            cum_realised_pnl	number	Cumulative realised Profit and Loss
            free_qty	        number	Qty which can be closed. (If you have a long position, free_qty is negative. vice versa)
            tp_sl_mode	        string	Stop loss and take profit mode
            deleverage_indicator	number	Deleverage indicator level (1,2,3,4,5)
            unrealised_pnl	    number	unrealised pnl
            risk_id	            integer	Risk ID
            take_profit	        number	Take profit price
            stop_loss	        number	Stop loss price
            trailing_stop	    number	Trailing stop (the distance from the current price)
            position_idx	    integer	Position idx, used to identify positions in different position modes:
                                        0-One-Way Mode
                                        1-Buy side of both side mode
                                        2-Sell side of both side mode
            mode	            string	Position Mode. MergedSingle: One-Way Mode; BothSide: Hedge Mode
            (https://bybit-exchange.github.io/docs/futuresV2/linear/#t-myposition)
        '''
        if entry_price is None:
            entry_price = self.latest_price()

        # レバレッジの設定
        self.set_leverage(leverage, leverage)  # leverageの設定
        total_qty = round((equity*leverage)/entry_price, 3)  # 小数点第3位までを扱う
        req_qty = total_qty

        for rt in range(const.MAX_TRYOUT_15):
            # 注文が部分的に契約した場合、総量を削減してあげる必要がある。

            orres = self.sub_trade_entry_maker(
                side, req_qty, equity, loss_cut, entry_price)
            order_id = orres['order_id']

            tdlg().syslm(f'order_id:{order_id}:{rt}')

            for wc in range(const.MAX_TRYOUT_15):
                # 注意:wait outの時点でPositionが入っている可能性があるので、このタイミングでsleepを入れるのが重要
                time.sleep(const.SLEEP_TIME_5)
                res = self.query_active_order(order_id)
                order_status = res['order_status']
                tdlg().syslm(f'order_status:{order_status}:{wc}')

                if order_status == 'Cancelled':
                    tdlg().syslm(f'Cancel this order.but try it')
                    break

                # 注文が部分的な契約と場合は契約は総量を調整する必要がある。
                if order_status == 'PartiallyFilled':
                    pos_res = self.get_position()
                    index = 0 if side == 'Buy' else 1
                    part_qty = pos_res[index][const.COL_SIZE]
                    tdlg().syslm(
                        f'req_qty:{req_qty},total_qty:{total_qty},part_qty:{part_qty}')
                    req_qty = total_qty - part_qty
                    req_qty = round(req_qty, 3)
                    tdlg().syslm(f'request,req_aty:{req_qty}')

                if order_status == 'Filled':
                    tdlg().syslm(f'Success entry this trade!!!')
                    pos_res = self.get_position()
                    index = 0 if side == 'Buy' else 1
                    return True, pos_res[index]
            else:
                # 注意:wait outの時点でPositionが入っている可能性がある
                tdlg().syslm(f'wait time out. cancel this order.')
                self.cancel_all_active_orders()

            time.sleep(const.SLEEP_TIME_5)

        pos_res = self.get_position()
        index = 0 if side == 'Buy' else 1
        part_qty = pos_res[index][const.COL_SIZE]

        # 注文が部分的な契約と場合は契約(保有しているポジションを処理)
        if part_qty != 0:
            self.close_position_maker()
        return False, 0
    """

    def trade_entry_maker(self, side: str, equity: float, leverage: float, loss_cut: float, entry_price=None) -> Tuple[bool, list]:
        '''
            参入注文（成行注文）
            sym(必須):"BTCUSDT","ETHUSDT"など
            side(必須):"Buy" or "Sell"
            qty(必須):symの量
            price:sideが"Limit"の場合は必須
            戻り値：注文情報
            result[0]:"Buy",resutl[1]:"Sell"
            Parameter	        Type	Comment
            user_id	            number	UserID
            symbol	            string	Symbol
            side	            string	Side
            size	            number	Position qty
            position_value	    number	Position value
            entry_price	        number	Average opening price
            liq_price	        number	Liquidation price
            bust_price	        number	Bust price
            leverage	        number	In Isolated Margin mode, the value is set by the user. In Cross Margin mode, the value is the max leverage at current risk level
            auto_add_margin	    number	Whether or not auto-margin replenishment is enabled
            is_isolated	bool	true means isolated margin mode; false means cross margin mode
            position_margin	    number	Position margin
            occ_closing_fee	    number	Pre-occupancy closing fee
            realised_pnl	    number	Today's realised Profit and Loss
            cum_realised_pnl	number	Cumulative realised Profit and Loss
            free_qty	        number	Qty which can be closed. (If you have a long position, free_qty is negative. vice versa)
            tp_sl_mode	        string	Stop loss and take profit mode
            deleverage_indicator	number	Deleverage indicator level (1,2,3,4,5)
            unrealised_pnl	    number	unrealised pnl
            risk_id	            integer	Risk ID
            take_profit	        number	Take profit price
            stop_loss	        number	Stop loss price
            trailing_stop	    number	Trailing stop (the distance from the current price)
            position_idx	    integer	Position idx, used to identify positions in different position modes:
                                        0-One-Way Mode
                                        1-Buy side of both side mode
                                        2-Sell side of both side mode
            mode	            string	Position Mode. MergedSingle: One-Way Mode; BothSide: Hedge Mode
            (https://bybit-exchange.github.io/docs/futuresV2/linear/#t-myposition)
        '''
        if entry_price is None:
            entry_price = self.latest_price()

        # レバレッジの設定
        self.set_leverage(leverage, leverage)  # leverageの設定
        total_qty = round((equity*leverage)/entry_price, 3)  # 小数点第3位までを扱う
        req_qty = total_qty

        orres = self.sub_trade_entry_maker(
            side, req_qty, equity, loss_cut, entry_price)
        order_id = orres['order_id']

        tdlg().syslm(f'order_id:{order_id}')

        for wc in range(const.MAX_TRYOUT_15):
            # 注意:wait outの時点でPositionが入っている可能性があるので、このタイミングでsleepを入れるのが重要
            time.sleep(const.SLEEP_TIME_5)
            res = self.query_active_order(order_id)
            order_status = res['order_status']
            tdlg().syslm(f'order_status:{order_status}:{wc}')

            if order_status == 'Cancelled':
                tdlg().syslm(f'Cancel this order.but try it')
                break

            # 注文が部分的な契約と場合は契約は総量を調整する必要がある。
            if order_status == 'PartiallyFilled':
                pos_res = self.get_position()
                index = 0 if side == 'Buy' else 1
                part_qty = pos_res[index][const.COL_SIZE]
                tdlg().syslm(
                    f'req_qty:{req_qty},total_qty:{total_qty},part_qty:{part_qty}')
                req_qty = total_qty - part_qty
                req_qty = round(req_qty, 3)
                tdlg().syslm(f'request,req_aty:{req_qty}')

            if order_status == 'Filled':
                tdlg().syslm(f'Success entry this trade!!!')
                pos_res = self.get_position()
                index = 0 if side == 'Buy' else 1
                return True, pos_res[index]
        else:
            # 注意:wait outの時点でPositionが入っている可能性がある
            tdlg().syslm(f'wait time out. cancel this order.')
            self.cancel_all_active_orders()

        pos_res = self.get_position()
        index = 0 if side == 'Buy' else 1
        part_qty = pos_res[index][const.COL_SIZE]

        # 注文が部分的な契約と場合は契約(保有しているポジションを処理)
        if part_qty != 0:
            self.close_position_maker()
        return False, 0

    def challenge_entry(self, target_price, threshold, entry_timeout_min, side: str, equity: float, leverage: float, loss_cut: float) -> Tuple[bool, list]:
        '''
            なるべく有利な条件でエントリーできるような処理
            直近の価格が閾値以内だったら、直近の価格でエントリーし、閾値を外れる価格であったら、閾値でエントリーする。
            target_entry : 目標エントリー価格
            threshold  : 許容範囲
            entry_timeout_min : エントリーチャレンジできる時間（分）
        '''

        current_time = int(time.time())
        end_time = current_time + 60*entry_timeout_min  # チャレンジ時間の設定

        while True:
            current_time = int(time.time())
            if current_time > end_time:
                tdlg().syslm("challenge retry timeout,cancel this entry.")
                return False, 0

            current_price = self.latest_price()
            # 閾値価格を算出
            threshold_price = target_price + target_price * \
                (threshold if side == const.COL_BUY else -threshold)

            threshold_price = round(threshold_price, 3)

            print(
                f'current_price:{current_price},threshold_price:{threshold_price},target_price:{target_price}')

            # 直近の価格が閾値価格内であるかどうか
            if (side == const.COL_BUY and current_price < threshold_price) or (side == const.COL_SELL and current_price > threshold_price):
                entry_price = current_price
            else:
                entry_price = threshold_price

            success, res = self.trade_entry_maker(
                side, equity, leverage, loss_cut, entry_price)
            if success:
                return True, res

    def sub_query_active_order(self, order_id: str) -> Tuple[bool, list]:
        '''
            Get my active order list.
            戻り値
            Parameter	        Type	Comment
            order_id	        string	Order ID
            user_id	            number	UserID
            symbol	            string	Symbol
            side	            string	Side
            order_type	        string	Order type
            price	            number	Order price
            qty	                number	Order quantity in cryptocurrency
            time_in_force	    string	Time in force
            order_status	    string	Order status
            last_exec_price	    number	Last execution price
            cum_exec_qty	    number	Cumulative qty of trading
            cum_exec_value	    number	Cumulative value of trading
            cum_exec_fee	    number	Cumulative trading fees
            reduce_only	        bool	true means close order, false means open position
            close_on_trigger	bool	Is close on trigger order
            order_link_id	    string	Unique user-set order ID. Maximum length of 36 characters
            created_time	    string	Creation time (when the order_status was Created)
            updated_time	    string	Update time
            take_profit	        number	Take profit price
            stop_loss	        number	Stop loss price
            tp_trigger_by	    string	Take profit trigger price type, default: LastPrice
            sl_trigger_by	    string	Stop loss trigger price type, default: LastPrice
            (https://bybit-exchange.github.io/docs/futuresV2/linear/#t-queryactive)
        '''
        try:
            self.__session = usdt_perpetual.HTTP(
                endpoint=self.__url, api_key=self.__api_key, api_secret=self.__api_secret)
            res = self.__session.query_active_order(
                symbol=self.__sym, order_id=order_id)
        except Exception as e:
            tdlg().syslm(f'ByBit HTTP Access Error:{e}')
            return False, 0

        return True, res[const.COL_RESULT]

    def query_active_order(self, order_id: str = None) -> dict:
        '''
            Get my active order list.
            戻り値
            Parameter	        Type	Comment
            order_id	        string	Order ID
            user_id	            number	UserID
            symbol	            string	Symbol
            side	            string	Side
            order_type	        string	Order type
            price	            number	Order price
            qty	                number	Order quantity in cryptocurrency
            time_in_force	    string	Time in force
            order_status	    string	Order status
            last_exec_price	    number	Last execution price
            cum_exec_qty	    number	Cumulative qty of trading
            cum_exec_value	    number	Cumulative value of trading
            cum_exec_fee	    number	Cumulative trading fees
            reduce_only	        bool	true means close order, false means open position
            close_on_trigger	bool	Is close on trigger order
            order_link_id	    string	Unique user-set order ID. Maximum length of 36 characters
            created_time	    string	Creation time (when the order_status was Created)
            updated_time	    string	Update time
            take_profit	        number	Take profit price
            stop_loss	        number	Stop loss price
            tp_trigger_by	    string	Take profit trigger price type, default: LastPrice
            sl_trigger_by	    string	Stop loss trigger price type, default: LastPrice
            (https://bybit-exchange.github.io/docs/futuresV2/linear/#t-queryactive)
        '''
        for _ in range(const.MAX_TRYOUT_HTTP):
            flag, res = self.sub_query_active_order(order_id)
            if flag == True:
                return res
            time.sleep(const.SLEEP_HTTP)
        else:
            tdlg().syslm('query_active_order:MAX TRY OUT Error.')
            exit(-1)

    def sub_cancel_all_active_orders(self) -> Tuple[bool, dict]:
        '''
            Get my active order list.
            Parameter	Type	Comment
            order_id	string	Order ID
        '''
        try:
            self.__session = usdt_perpetual.HTTP(
                endpoint=self.__url, api_key=self.__api_key, api_secret=self.__api_secret)
            res = self.__session.cancel_all_active_orders(symbol=self.__sym)
        except Exception as e:
            tdlg().syslm(f'ByBit HTTP Access Error:{e}')
            return False, 0

        return True, res[const.COL_RESULT]

    def cancel_all_active_orders(self) -> dict:
        '''
            Get my active order list.
            Parameter	Type	Comment
            order_id	string	Order ID
        '''
        for _ in range(const.MAX_TRYOUT_HTTP):
            flag, res = self.sub_cancel_all_active_orders()
            if flag == True:
                return res
            time.sleep(const.SLEEP_HTTP)
        else:
            tdlg().syslm('cancel_all_active_orders:MAX TRY OUT Error.')
            exit(-1)

    def sub_get_position(self) -> Tuple[bool, dict]:
        '''
            現在保有しているPostion情報
            sym(必須):"BTCUSDT","ETHUSDT"など
            戻り値:Postion情報
            result[0]:"Buy",resutl[1]:"Sell"
            Parameter	        Type	Comment
            user_id	            number	UserID
            symbol	            string	Symbol
            side	            string	Side
            size	            number	Position qty
            position_value	    number	Position value
            entry_price	        number	Average opening price
            liq_price	        number	Liquidation price
            bust_price	        number	Bust price
            leverage	        number	In Isolated Margin mode, the value is set by the user. In Cross Margin mode, the value is the max leverage at current risk level
            auto_add_margin	    number	Whether or not auto-margin replenishment is enabled
            is_isolated	bool	true means isolated margin mode; false means cross margin mode
            position_margin	    number	Position margin
            occ_closing_fee	    number	Pre-occupancy closing fee
            realised_pnl	    number	Today's realised Profit and Loss
            cum_realised_pnl	number	Cumulative realised Profit and Loss
            free_qty	        number	Qty which can be closed. (If you have a long position, free_qty is negative. vice versa)
            tp_sl_mode	        string	Stop loss and take profit mode
            deleverage_indicator	number	Deleverage indicator level (1,2,3,4,5)
            unrealised_pnl	    number	unrealised pnl
            risk_id	            integer	Risk ID
            take_profit	        number	Take profit price
            stop_loss	        number	Stop loss price
            trailing_stop	    number	Trailing stop (the distance from the current price)
            position_idx	    integer	Position idx, used to identify positions in different position modes:
                                        0-One-Way Mode
                                        1-Buy side of both side mode
                                        2-Sell side of both side mode
            mode	            string	Position Mode. MergedSingle: One-Way Mode; BothSide: Hedge Mode
            (https://bybit-exchange.github.io/docs/futuresV2/linear/#t-myposition)
        '''
        try:
            self.__session = usdt_perpetual.HTTP(
                endpoint=self.__url, api_key=self.__api_key, api_secret=self.__api_secret)
            res = self.__session.my_position(symbol=self.__sym)
        except Exception as e:
            tdlg().syslm(f'ByBit HTTP Access Error:{e}')
            return False, 0

        return True, res[const.COL_RESULT]

    def get_position(self) -> dict:
        '''
            現在保有しているPostion情報
            戻り値:Postion情報
            戻り値:Postion情報
            result[0]:"Buy",resutl[1]:"Sell"
            Parameter	        Type	Comment
            user_id	            number	UserID
            symbol	            string	Symbol
            side	            string	Side
            size	            number	Position qty
            position_value	    number	Position value
            entry_price	        number	Average opening price
            liq_price	        number	Liquidation price
            bust_price	        number	Bust price
            leverage	        number	In Isolated Margin mode, the value is set by the user. In Cross Margin mode, the value is the max leverage at current risk level
            auto_add_margin	    number	Whether or not auto-margin replenishment is enabled
            is_isolated	bool	true means isolated margin mode; false means cross margin mode
            position_margin	    number	Position margin
            occ_closing_fee	    number	Pre-occupancy closing fee
            realised_pnl	    number	Today's realised Profit and Loss
            cum_realised_pnl	number	Cumulative realised Profit and Loss
            free_qty	        number	Qty which can be closed. (If you have a long position, free_qty is negative. vice versa)
            tp_sl_mode	        string	Stop loss and take profit mode
            deleverage_indicator	number	Deleverage indicator level (1,2,3,4,5)
            unrealised_pnl	    number	unrealised pnl
            risk_id	            integer	Risk ID
            take_profit	        number	Take profit price
            stop_loss	        number	Stop loss price
            trailing_stop	    number	Trailing stop (the distance from the current price)
            position_idx	    integer	Position idx, used to identify positions in different position modes:
                                        0-One-Way Mode
                                        1-Buy side of both side mode
                                        2-Sell side of both side mode
            mode	            string	Position Mode. MergedSingle: One-Way Mode; BothSide: Hedge Mode
            (https://bybit-exchange.github.io/docs/futuresV2/linear/#t-myposition)
        '''
        for _ in range(const.MAX_TRYOUT_HTTP):
            flag, res = self.sub_get_position()
            if flag == True:
                return res
            time.sleep(const.SLEEP_HTTP)
        else:
            tdlg().syslm('get_position:MAX TRY OUT Error.')
            exit(-1)

    def have_position(self) -> Tuple[bool, dict]:
        '''
            現在を保有しているかどうか？
            sym(必須):"BTCUSDT","ETHUSDT"など
            戻り値:True(保有）/False(保有していない)+Postion情報
            戻り値:Postion情報
            result[0]:"Buy",resutl[1]:"Sell"
            Parameter	        Type	Comment
            user_id	            number	UserID
            symbol	            string	Symbol
            side	            string	Side
            size	            number	Position qty
            position_value	    number	Position value
            entry_price	        number	Average opening price
            liq_price	        number	Liquidation price
            bust_price	        number	Bust price
            leverage	        number	In Isolated Margin mode, the value is set by the user. In Cross Margin mode, the value is the max leverage at current risk level
            auto_add_margin	    number	Whether or not auto-margin replenishment is enabled
            is_isolated	bool	true means isolated margin mode; false means cross margin mode
            position_margin	    number	Position margin
            occ_closing_fee	    number	Pre-occupancy closing fee
            realised_pnl	    number	Today's realised Profit and Loss
            cum_realised_pnl	number	Cumulative realised Profit and Loss
            free_qty	        number	Qty which can be closed. (If you have a long position, free_qty is negative. vice versa)
            tp_sl_mode	        string	Stop loss and take profit mode
            deleverage_indicator	number	Deleverage indicator level (1,2,3,4,5)
            unrealised_pnl	    number	unrealised pnl
            risk_id	            integer	Risk ID
            take_profit	        number	Take profit price
            stop_loss	        number	Stop loss price
            trailing_stop	    number	Trailing stop (the distance from the current price)
            position_idx	    integer	Position idx, used to identify positions in different position modes:
                                        0-One-Way Mode
                                        1-Buy side of both side mode
                                        2-Sell side of both side mode
            mode	            string	Position Mode. MergedSingle: One-Way Mode; BothSide: Hedge Mode
            (https://bybit-exchange.github.io/docs/futuresV2/linear/#t-myposition)
        '''
        res = self.get_position()
        for i in range(len(res)):
            qty = res[i][const.COL_SIZE]
            if qty > 0:
                return True, res[i]
        else:
            return False, 0

    def sub_close_position(self) -> Tuple[bool, dict]:
        '''
            手持ちに持っている全てのポジションを解消する。
            成行注文
             Long(Buy)/Short(Sell)のいずれかがポジションとして入っていることが前提
            戻り値：注文情報
            Parameter	    Type	Comment
            order_id	    string	Order ID
            user_id	        number	UserID
            symbol	        string	Symbol
            side	        string	Side
            order_type	    string	Order type
            price	        number	Order price
            qty	            number	Order quantity in USD
            time_in_force	string	Time in force
            order_status	string	Order status
            last_exec_price	number	Last execution price
            cum_exec_qty	number	Cumulative qty of trading
            cum_exec_value	number	Cumulative value of trading
            cum_exec_fee	number	Cumulative trading fees
            reduce_only	    bool	true means close order, false means open position
            close_on_trigger	bool	Is close on trigger order
            order_link_id	string	Unique user-set order ID. Maximum length of 36 characters
            created_time	string	Creation time (when the order_status was Created)
            updated_time	string	Update time
            take_profit	    number	Take profit price
            stop_loss	    number	Stop loss price
            tp_trigger_by	string	Take profit trigger price type, default: LastPrice
            sl_trigger_by	string	Stop loss trigger price type, default: LastPrice
            position_idx	integer	Position idx, used to identify positions in different position modes:
                                    0-One-Way Mode
                                    1-Buy side of both side mode
                                    2-Sell side of both side mode
            (https://bybit-exchange.github.io/docs/futuresV2/linear/#t-placeactive)
        '''
        try:
            res = self.get_position()

            for i in range(len(res)):
                # i==0:Buy側のポジション情報、i==1がSell側
                ops = res[i]
                qty = ops[const.COL_SIZE]
                pos_side = ops[const.COL_SIDE]
                side = const.COL_SELL if pos_side == const.COL_BUY else const.COL_BUY

                if qty != 0:
                    self.__session = usdt_perpetual.HTTP(
                        endpoint=self.__url, api_key=self.__api_key, api_secret=self.__api_secret)

                    res = self.__session.place_active_order(
                        symbol=self.__sym,
                        side=side,
                        order_type="Market",
                        qty=qty,
                        time_in_force="GoodTillCancel",
                        reduce_only=True,
                        close_on_trigger=False
                    )
                    return True, res[const.COL_RESULT]
        except Exception as e:
            tdlg().syslm(f'ByBit HTTP Access Error:{e}')
            return False, 0

        tdlg().syslm(f'maybe no exist my position')
        return False, 0

    def close_position(self) -> dict:
        '''
            手持ちに持っている全てのポジションを解消する。
            成行注文
             Long(Buy)/Short(Sell)のいずれかがポジションとして入っていることが前提
            戻り値：注文情報
            Parameter	    Type	Comment
            order_id	    string	Order ID
            user_id	        number	UserID
            symbol	        string	Symbol
            side	        string	Side
            order_type	    string	Order type
            price	        number	Order price
            qty	            number	Order quantity in USD
            time_in_force	string	Time in force
            order_status	string	Order status
            last_exec_price	number	Last execution price
            cum_exec_qty	number	Cumulative qty of trading
            cum_exec_value	number	Cumulative value of trading
            cum_exec_fee	number	Cumulative trading fees
            reduce_only	    bool	true means close order, false means open position
            close_on_trigger	bool	Is close on trigger order
            order_link_id	string	Unique user-set order ID. Maximum length of 36 characters
            created_time	string	Creation time (when the order_status was Created)
            updated_time	string	Update time
            take_profit	    number	Take profit price
            stop_loss	    number	Stop loss price
            tp_trigger_by	string	Take profit trigger price type, default: LastPrice
            sl_trigger_by	string	Stop loss trigger price type, default: LastPrice
            position_idx	integer	Position idx, used to identify positions in different position modes:
                                    0-One-Way Mode
                                    1-Buy side of both side mode
                                    2-Sell side of both side mode
            (https://bybit-exchange.github.io/docs/futuresV2/linear/#t-placeactive)
        '''
        for _ in range(const.MAX_TRYOUT_HTTP):
            flag, res = self.sub_close_position()
            if flag == True:
                return res
            time.sleep(const.SLEEP_HTTP)
        else:
            tdlg().syslm('close_position:MAX TRY OUT Error.')
            exit(-1)

    def sub_close_position_maker(self) -> Tuple[bool, dict]:
        '''
            手持ちに持っているポジションを解消する。
            指値注文
             Long(Buy)/Short(Sell)のいずれかがポジションとして入っていることが前提
            戻り値：注文情報
            Parameter	    Type	Comment
            order_id	    string	Order ID
            user_id	        number	UserID
            symbol	        string	Symbol
            side	        string	Side
            order_type	    string	Order type
            price	        number	Order price
            qty	            number	Order quantity in USD
            time_in_force	string	Time in force
            order_status	string	Order status
            last_exec_price	number	Last execution price
            cum_exec_qty	number	Cumulative qty of trading
            cum_exec_value	number	Cumulative value of trading
            cum_exec_fee	number	Cumulative trading fees
            reduce_only	    bool	true means close order, false means open position
            close_on_trigger	bool	Is close on trigger order
            order_link_id	string	Unique user-set order ID. Maximum length of 36 characters
            created_time	string	Creation time (when the order_status was Created)
            updated_time	string	Update time
            take_profit	    number	Take profit price
            stop_loss	    number	Stop loss price
            tp_trigger_by	string	Take profit trigger price type, default: LastPrice
            sl_trigger_by	string	Stop loss trigger price type, default: LastPrice
            position_idx	integer	Position idx, used to identify positions in different position modes:
                                    0-One-Way Mode
                                    1-Buy side of both side mode
                                    2-Sell side of both side mode
            (https://bybit-exchange.github.io/docs/futuresV2/linear/#t-placeactive)
        '''
        res = self.get_position()
        entry_price = self.latest_price()  # 現在の価格を取得
        for i in range(len(res)):
            # i==0:Buy側のポジション情報、i==1がSell側
            ops = res[i]
            qty = round(ops[const.COL_SIZE], 3)  # 小数点第3位までを扱う
            pos_side = ops[const.COL_SIDE]
            side = const.COL_SELL if pos_side == const.COL_BUY else const.COL_BUY

            if qty != 0:
                order_res = self.place_order(side, qty, otype=const.LIMIT,
                                             entry_price=entry_price, time_in_force=const.POSTONLY, reduceflg=True)
                return True, order_res

        tdlg().syslm('No my position. do not need to sell.')
        return False, 0

    def close_position_maker(self) -> dict:
        '''
            手持ちに持っているポジションを解消する。
            指値注文
             Long(Buy)/Short(Sell)のいずれかがポジションとして入っていることが前提
            戻り値：注文情報
            Parameter	    Type	Comment
            order_id	    string	Order ID
            user_id	        number	UserID
            symbol	        string	Symbol
            side	        string	Side
            order_type	    string	Order type
            price	        number	Order price
            qty	            number	Order quantity in USD
            time_in_force	string	Time in force
            order_status	string	Order status
            last_exec_price	number	Last execution price
            cum_exec_qty	number	Cumulative qty of trading
            cum_exec_value	number	Cumulative value of trading
            cum_exec_fee	number	Cumulative trading fees
            reduce_only	    bool	true means close order, false means open position
            close_on_trigger	bool	Is close on trigger order
            order_link_id	string	Unique user-set order ID. Maximum length of 36 characters
            created_time	string	Creation time (when the order_status was Created)
            updated_time	string	Update time
            take_profit	    number	Take profit price
            stop_loss	    number	Stop loss price
            tp_trigger_by	string	Take profit trigger price type, default: LastPrice
            sl_trigger_by	string	Stop loss trigger price type, default: LastPrice
            position_idx	integer	Position idx, used to identify positions in different position modes:
                                    0-One-Way Mode
                                    1-Buy side of both side mode
                                    2-Sell side of both side mode
            (https://bybit-exchange.github.io/docs/futuresV2/linear/#t-placeactive)
        '''
        for rt in range(const.MAX_TRYOUT_15):
            flag, orres = self.sub_close_position_maker()

            if flag == True:
                order_id = orres['order_id']
                tdlg().syslm(f'order_id:{order_id}:{rt}')

                for wc in range(const.MAX_TRYOUT_10):
                    time.sleep(const.SLEEP_TIME_5)
                    res = self.query_active_order(order_id)
                    order_status = res['order_status']
                    tdlg().syslm(f'order_status:{order_status}:{wc}')

                    if order_status == 'Cancelled':
                        tdlg().syslm(f'Cancel this order. But try it')
                        break

                    if order_status == 'Filled':
                        tdlg().syslm(f'Success close position !!!')
                        return True, res

                else:
                    tdlg().syslm(f'wait time out. cancel this order.')
                    self.cancel_all_active_orders()
            else:
                tdlg().syslm(f'Error Order.But retry it.')
                time.sleep(const.SLEEP_TIME_5)

        else:
            tdlg().syslm(f'Try out close position.')
            return False, None


if __name__ == '__main__':

    """
    #Trading Data download(csv) from bybit
    json_file = open('aitrading_settings_BTC_15m.json', 'r')
    var_conf = json.load(json_file)
    lg = TradingLogger(var_conf)
    byapi = BybitOnlineAPI(var_conf)

    byapi.fetch_historical_price_data(
        15, "2022-07-01 00:00:00+00:00", "2023-01-20 00:00:00+00:00")

    # def place_order(self, sym, side, qty, otype='Market', entry_price=None, reduceflg=False, losscut_price=None):
    """

"""
    class wait_sample:
        async def wait_a(self) -> None:
        await asyncio.sleep(12)
        print('finish a')

    async def wait_b(self) -> None:
        await asyncio.sleep(15)
        print('finish b')

    async def wait_c(self) -> None:
        while True:
            await asyncio.sleep(5)
            print('finish c')
            await asyncio.sleep(2)
            print('finish c+')


async def last_price():
    for _ in range(30):
        await asyncio.sleep(5)
        res = byapi.latest_price()
        print(res)

w = wait_sample()

loop = asyncio.get_event_loop()

asyncio.ensure_future(last_price())
asyncio.ensure_future(w.wait_c())
loop.run_forever()

    # sleep(5)

    # print(f'last_price:{last_price},Entry_price:{res["price"]}')
    # print(dict['qty'])
    #flag, res = byapi.place_order(sym, "Buy", 0.181, last_price, reduceflg=True)
    # print('---------------------------------------------------------------')
    #res = byapi.close_position(sym)
    #order_id = res[const.COL_ORDER_ID]
    # print(res)

    #res = byapi.get_pandl_orderid(sym, order_id)
    # print(res)
    #flag, res = byapi.get_position(sym)
    # print(res)
    # print(len(res))

    '''
    loss_price = last_price - (last_price*0.05)
    flag, res = byapi.place_order(sym, "Buy", 0.01, last_price, loss_price)
    print(res)
    '''
    #flag, result = byapi.trading_fee("BTCUSDT")
    #dic_sym = result
    # print(dic_sym)
 """

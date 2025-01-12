import pandas as pd
import os, sys

from common.trading_logger_db import TradingLoggerDB
from common.config_manager import ConfigManager
from fxaccount import FXAccount
from mongodb.data_loader_mongo import MongoDataLoader
from trade_config import trade_config
from common.constants import TRANSACTION_DATA,ENTRY_TYPE_LONG,ENTRY_TYPE_SHORT


class FXTransactionDataFrame:
        """
        This class manages the FX transaction data. It handles the
        initialization of transaction logs, adding new transaction data,
        saving log files, checking the existence of specific serial numbers,
        getting the next serial number, and retrieving and setting transaction data.
        """

        def __init__(self):
                """
                Initializes the instance of FXTransactionDataFrame.
                """
                self.__logger = TradingLoggerDB()
                self.__config_manager = ConfigManager()
                self.__data_loader = MongoDataLoader()

                self.__fxcol_names = [
                        'serial',
                        'init_equity',
                        'equity',
                        'leverage',
                        'contract',
                        'qty',
                        'entry_price',
                        'losscut_price',
                        'exit_price',
                        'limit_price',
                        'pl',
                        'pred',
                        'tradetype',
                        'stage',
                        'losscut',
                        'entrytime',
                        'exittime',
                        'direction',
                        'startup_flag'
                ]
                # Generate an empty dataframe
                self.__fxtrn = pd.DataFrame(columns=self.__fxcol_names)
                self.__contract = trade_config.contract
                self.table_name = f"{self.__contract}" + "_fxtransaction"
                self.startup_flag = True

        def get_startup_flag(self):
                if self.startup_flag:
                        self.startup_flag = False
                        return 1
                else:
                        return 0

        def _create_filename(self):
                """
                Generates the full file name of the log file.

                Returns:
                        str: Generated file name.
                """
                log_dir = self.__config_manager.get('LOG', 'LOGPATH')
                file_idf = self.__config_manager.get('LOG', 'FILE_IDF_FX')
                self.__contract = trade_config.contract

                return os.path.join(log_dir, f"{self.__contract}{file_idf}")

        def _check_and_remove_file(self, filename):
                """
                Deletes the specified file if it exists.

                Args:
                        filename (str): The path of the file to be deleted.

                Returns:
                        bool: Whether the file operation was successful.
                """
                try:
                        if os.path.isfile(filename):
                                os.remove(filename)
                except OSError as e:
                        self.__logger.log_system_message(f'File Operation Error: {filename}, {e}')
                        return False
                return True

        def _initialize_fxtranzaction_log(self):
                """
                Initializes the FX transaction log file.

                Returns:
                        bool: Whether the initialization of the log file was successful.
                """
                filename = self._create_filename()
                return self._check_and_remove_file(filename)

        def _new_fxtrn_dataframe(self, serial, init_equity, equity, leverage, contract,
                                                         qty, entry_price, losscut_price, exit_price, limit_price, pl,
                                                         pred, tradetype, stage, losscut, losscut_rate,entry_time, exit_time, direction):
                """
                Adds new FX transaction data to the DataFrame.

                Args:
                        The respective arguments correspond to the columns of the transaction data.
                """
                # Define new record as dictionary
                new_record = {
                        'serial': serial,
                        'init_equity': init_equity,
                        'equity': equity,
                        'leverage': leverage,
                        'contract': contract,
                        'qty': qty,
                        'entry_price': entry_price,
                        'losscut_price': losscut_price,
                        'exit_price': exit_price,
                        'limit_price': limit_price,
                        'pl': pl,
                        'pred': pred,
                        'tradetype': tradetype,
                        'stage': stage,
                        'losscut': losscut,
                        'losscut_rate': losscut_rate,
                        'entrytime': entry_time,
                        'exittime': exit_time,
                        'direction': direction,
                        'startup_flag': self.get_startup_flag()
                }
                # Specify data type of DataFrame
                dtypes = {
                        'serial': 'int64',
                        'init_equity': 'float64',
                        'equity': 'float64',
                        'leverage': 'float64',
                        'contract': 'object',
                        'qty': 'float64',
                        'entry_price': 'float64',
                        'losscut_price': 'float64',
                        'exit_price': 'float64',
                        'limit_price': 'float64',
                        'pl': 'float64',
                        'pred': 'int64',
                        'tradetype': 'object',
                        'stage': 'object',
                        'losscut': 'int64',
                        'losscut_rate': 'float64',
                        'entrytime': 'datetime64[ns]',
                        'exittime': 'datetime64[ns]',
                        'direction': 'object',
                        'startup_flag': 'int64'
                }

                # Convert new record to DataFrame
                new_record_df = pd.DataFrame([new_record])

                # Specify the data type of each column
                for column, dtype in dtypes.items():
                        new_record_df[column] = new_record_df[column].astype(dtype)

                # Combine the DataFrame of the new record with the current DataFrame

                if not self.__fxtrn.empty:
                        self.__fxtrn = pd.concat([self.__fxtrn, new_record_df], ignore_index=True)
                else:
                        self.__fxtrn = new_record_df

                new_record_df['exittime'] = pd.to_datetime(new_record_df['exittime'], errors='coerce')
                new_record_df['exittime'] = new_record_df['exittime'].apply(
                        lambda x: x.to_pydatetime() if pd.notnull(x) else None)

                self.__data_loader.insert_data(new_record_df, coll_type=TRANSACTION_DATA)

        def _update_fxtrn_dataframe(self, serial):
                """
                Updates the transaction data with the specified serial number.

                Args:
                        serial (int): Serial number.
                """
                df = self.__fxtrn[self.__fxtrn['serial'] == serial]
                self.__data_loader.update_data_by_serial(serial, df, coll_type=TRANSACTION_DATA)

        def save_fxtrn_log(self) -> bool:
                """
                Save the FX transaction log to a file.

                Returns:
                        bool: Whether the log was saved successfully.
                """
                filename = self._create_filename()
                if not self._check_and_remove_file(filename):
                        return False
                try:
                        self.__fxtrn.to_csv(filename)
                except OSError as e:
                        self.__logger.log_system_message(f'File Write Error: {filename}, {e}')
                        return False
                return True

        def does_serial_exist(self, serial: int) -> bool:
                """
                Returns whether the specified serial exists in the record.

                Args:
                        serial (int): Serial number to check.

                Returns:
                        bool: Whether the specified serial number exists.
                """
                return serial in self.__fxtrn['serial'].values

        def get_next_serial(self) -> int:
                """
                Returns the next available serial number.

                Returns:
                        int: The next serial number.
                """
                return self.__data_loader.get_next_serial(coll_type=TRANSACTION_DATA)

        def get_fxtrn_dataframe(self):
                """
                Returns the DataFrame for FX transactions.

                Returns:
                        pd.DataFrame: DataFrame that stores FX transaction data.
                """
                return self.__fxtrn

        def set_fd(self, serial: int, col: str, valume):
                """
                Sets a value for a specific column of transaction data with the specified serial number.

                Args:
                        serial (int): Serial number.
                        col (str): Column name to set the value for.
                        value: Value to set.
                """
                # print(f'serial:{serial},col:{col},valume:{valume}')
                self.__fxtrn.loc[self.__fxtrn['serial'] == serial, col] = valume

        def get_fd(self, serial: int, col: str):
                """
                Gets the value of a specific column from the transaction data with the specified serial number.

                Args:
                        serial (int): Serial number.
                        col (str): The name of the column from which to get the value.

                Returns:
                        The retrieved value.
                """
                value = self.__fxtrn.loc[self.__fxtrn['serial'] == serial, col]
                if not value.empty:
                        return value.iloc[0]
                return None


class FXTransaction:
        """
        This class manages FX transactions. It performs a series of operations related to FX transactions,
        such as trade entry and exit processing, profit and loss calculation, loss cut confirmation,
        and winning percentage calculation.
        """

        def __init__(self):
                """
                Initializes the instance of FXTransaction.
                """
                self.__config_manager = ConfigManager()
                self.__logger = TradingLoggerDB()
                self.__fxac = FXAccount()
                self.__fxtrn = FXTransactionDataFrame()
                self.set_round()

        def get_contract(self) -> str:
                """Getter method to get the type of trading contract.

                Returns:
                        str: The type of trading contract.
                """
                return trade_config.contract

        def get_init_equity(self) -> float:
                """Getter method to get the initial capital.

                Returns:
                        float: Initial capital.
                """
                return trade_config.init_equity

        def get_max_leverage(self) -> int:
                """Getter method to get leverage.

                Returns:
                        int: Leverage.
                """
                return trade_config.max_leverage

        def get_ptc(self) -> float:
                """Getter method to get the percentage of transaction cost.

                Returns:
                        float: Percentage of transaction cost.
                """
                return trade_config.ptc

        def set_round(self):
                if trade_config.symbol  == 'BTCUSDT':
                        self.__ROUND_DIGIT = 3
                elif trade_config.symbol == 'ETHUSDT':
                        self.__ROUND_DIGIT = 2
                else:
                        self.__ROUND_DIGIT = 2

        def get_qty(self, serial):
                return self.__fxtrn.get_fd(serial, 'qty')

        def get_losscut_price(self, serial):
                return self.__fxtrn.get_fd(serial, 'losscut_price')

        def trade_entry(self, tradetype: str, pred: int, entry_price: float, entry_time: str,
                                        direction: str,leverage:float,losscut_rate:float) -> int:
                """
                Make a trade entry. Returns the new trade number (serial) as a return value.

                Args:
                        tradetype (str): Trade type ('LONG' or 'SHORT').
                        pred (int): Forecast value (1: Bullish forecast, 0: Bearish forecast).
                        entry_price (float): Entry price.
                        entry_time (str): Entry time.
                        direction (str): Trading direction.

                Returns:
                        int: New trade number.
                """
                init_equity = trade_config.init_equity
                contract = trade_config.contract
                init_equity = self.__fxac.withdraw(entry_time, init_equity)
                equity = init_equity

                # Event where cash flow is exhausted
                if equity == 0:
                        self.__logger.log_message(f'out of cashflow:{equity:.2f}')
                        exit(0)

                qty = (equity * leverage) / entry_price
                qty = round(qty, self.__ROUND_DIGIT)

                if tradetype == 'LONG':
                        # Calculate the lower limit of the price at which forced liquidation will occur
                        limit_price = entry_price - (equity / qty)
                else:  # In case of SHORT ENTRY
                        limit_price = entry_price + (equity / qty)

                # Get trade number
                serial = self.__fxtrn.get_next_serial()
                losscut_price = self.calculate_losscut_price(entry_price, tradetype, leverage, losscut_rate)
                exit_time = None
                # Add a new record to the data frame
                self.__fxtrn._new_fxtrn_dataframe(serial, init_equity, equity, leverage, contract,
                                                                                qty, entry_price, losscut_price, 0, limit_price, 0,
                                                                                pred, tradetype, 0, 0,losscut_rate, entry_time, exit_time, direction)
                # Output to log
                self.__logger.log_transaction(entry_time,
                                                                        f'Entry: {direction}, {tradetype}, pred:{pred}, entry_price{entry_price}')
                self.__fxtrn.save_fxtrn_log()
                return serial

        def get_pandl(self, serial: int, exit_price: float) -> float:
                """
                Calculates the profit and loss of the specified trade number.

                Args:
                        serial (int): Trade number.
                        exit_price (float): Exit price.

                Returns:
                        float: Calculated profit and loss.
                """
                # If the specified serial does not exist in the record, return 0
                if not self.__fxtrn.does_serial_exist(serial):
                        return 0
                ptc = trade_config.ptc
                tradetype = self.__fxtrn.get_fd(serial, 'tradetype')
                qty = self.__fxtrn.get_fd(serial, 'qty')
                entry_price = self.__fxtrn.get_fd(serial, 'entry_price')
                equity = self.__fxtrn.get_fd(serial, 'equity')
                leverage = self.__fxtrn.get_fd(serial, 'leverage')

                buymnt = (qty * entry_price)  # Amount at the time of purchase
                selmnt = (qty * exit_price)
                buy_fee = equity * ptc * leverage
                sel_fee = equity * ptc * leverage

                if tradetype == "LONG":  # Profit calculation for LONG
                        return selmnt - buymnt - (buy_fee + sel_fee)  # Profit P&L
                else:  # Profit calculation for SHORT
                        return buymnt - selmnt - (buy_fee + sel_fee)  # Profit P&L

        def trade_exit(self, serial: int, exit_price: float, time: str, pandl=None,
                                losscut=None) -> bool:
                """
                Exits the trade.

                Args:
                        serial (int): Trade number.
                        exit_price (float): Exit price.
                        time (str): Exit time.
                        losscut (float, optional): Loss cut price.

                Returns:
                        bool: True if the process ends normally, False otherwise.
                """
                # If the specified serial does not exist in the record, return 0
                if not self.__fxtrn.does_serial_exist(serial):
                        self.__logger.log_message(f'trade_exit:no exit record: {serial}')
                        raise ValueError(f'trade_exit:no exit record: {serial}')

                equity = self.__fxtrn.get_fd(serial, 'equity')
                tradetype = self.__fxtrn.get_fd(serial, 'tradetype')
                leverage = self.__fxtrn.get_fd(serial, 'leverage')

                if pandl is None:
                        pandl = self.get_pandl(serial, exit_price)

                equity += pandl

                # Margin is cleared for each trade. Return to cash account.
                self.__fxac.deposit(time, equity)

                # 'LONG_ENTRY -> 'LONG_EXIT' string conversion
                tradetype = tradetype.split('_')[0] + '_' + 'EXIT'
                # Add new record to data frame
                self.__fxtrn.set_fd(serial, 'equity', equity)
                self.__fxtrn.set_fd(serial, 'exit_price', exit_price)
                self.__fxtrn.set_fd(serial, 'pl', pandl)
                self.__fxtrn.set_fd(serial, 'tradetype', tradetype)
                self.__fxtrn.set_fd(serial, 'exittime', time)

                if losscut is not None:
                        self.__fxtrn.set_fd(serial, 'losscut', losscut)
                # Output to log
                entry_price = self.__fxtrn.get_fd(serial, 'entry_price')
                self.__fxtrn._update_fxtrn_dataframe(serial)
                self.__logger.log_transaction(time,f'Exit: {serial}, {tradetype}, {losscut}, Entry_price:{entry_price}, Exit_price:{exit_price},P%L: {pandl:.2f}')

                # Save to log file
                self.__fxtrn.save_fxtrn_log()
                self.__fxac.print_performance(time, leverage)
                return pandl

        def trade_cancel(self, serial: int, date: str):
                """
                Cancel the specified trade.

                Args:
                        serial (int): Serial number of the trade to cancel.
                """
                # Return 0 if the specified serial does not exist in the record
                if not self.__fxtrn.does_serial_exist(serial):
                        self.__logger.log_message(f'trade_cancel:no exit record: {serial}')
                        raise ValueError(f'trade_cancel:no exit record: {serial}')

                equity = self.__fxtrn.get_fd(serial, 'equity')
                tradetype = self.__fxtrn.get_fd(serial, 'tradetype')
                pandl = 0
                equity += pandl

                # Margin is cleared for each trade. Return to cash account.
                self.__fxac.deposit(date, equity)

                # 'LONG_ENTRY -> 'LONG_EXIT' string conversion
                tradetype = tradetype.split('_')[0] + '_' + 'CANCEL'
                # Add new record to data frame
                self.__fxtrn.set_fd(serial, 'equity', equity)
                self.__fxtrn.set_fd(serial, 'exit_price', 0)
                self.__fxtrn.set_fd(serial, 'pl', pandl)
                self.__fxtrn.set_fd(serial, 'tradetype', tradetype)
                self.__fxtrn.set_fd(serial, 'exittime', date)

                # Output to log
                self.__fxtrn._update_fxtrn_dataframe(serial)
                self.__logger.log_transaction(date, f'Cancel: {serial}, {tradetype}')

                # Save to log file
                self.__fxtrn.save_fxtrn_log()
                self.__fxac.print_performance(date)
                return

        def check_losscut(self, serial: int, current_price: float) -> tuple[bool, float]:
                """
                Check the loss cut conditions for the specified trade based on the current price.

                Returns whether the conditions are such that a loss cut should be triggered (whether the
                current price exceeds the loss cut price) and the set loss cut price.

                Args:
                        serial (int): Serial number of the trade to check.
                        current_price (float): Current market price.

                Returns:
                        (bool, float): Boolean value of whether loss cut should be triggered and loss cut price.
                                                   True if loss cut should be triggered, False otherwise.
                """
                if not self.__fxtrn.does_serial_exist(serial):
                        self.__logger.log_message(f'check_losscut:no exit record: {serial}')
                        raise ValueError(f'check_losscut:no exit record: {serial}')

                losscut_price = self.__fxtrn.get_fd(serial, 'losscut_price')
                tradetype = self.__fxtrn.get_fd(serial, 'tradetype')

                if tradetype == 'LONG' and losscut_price <= current_price:
                        return False, losscut_price
                elif tradetype == 'SHORT' and losscut_price >= current_price:
                        return False, losscut_price
                else:
                        return True, losscut_price

        def is_losscut_triggered(self, serial: int, current_price: float) -> tuple[bool, float]:
                """
                Checks if a loss cut is triggered at the current price for the specified trade.

                Args:
                        serial (int): Trade number.
                        current_price (float): Current price.

                Returns:
                        (bool, float): Boolean value of whether a loss cut is triggered and the loss cut price.
                """

                losscut_rate = self.__fxtrn.get_fd(serial, 'losscut_rate')
                leverage = self.__fxtrn.get_fd(serial, 'leverage')


                # If the specified serial does not exist in the record
                if not self.__fxtrn.does_serial_exist(serial):
                        self.__logger.log_message(f'is_losscut_triggered:no exit record: {serial}')
                        # Fatal error handling
                        raise ValueError(f'is_losscut_triggered:no exit record: {serial}')


                #losscut_price = self.__fxtrn.get_fd(serial, 'losscut_price')
                tradetype = self.__fxtrn.get_fd(serial, 'tradetype')
                if tradetype == ENTRY_TYPE_LONG:
                        losscut_price = self.calculate_losscut_price(self.__fxtrn.get_fd(serial, 'entry_price'), ENTRY_TYPE_LONG, leverage, losscut_rate)
                        is_triggered = current_price <= losscut_price
                        return is_triggered, losscut_price
                else:
                        losscut_price = self.calculate_losscut_price(self.__fxtrn.get_fd(serial, 'entry_price'), ENTRY_TYPE_SHORT, leverage, losscut_rate)
                        is_triggered = current_price >= losscut_price
                        return is_triggered, losscut_price

        def calculate_losscut_price(self, entry_price: float, tradetype: str,leverage:float,losscut_rate: float) -> float:
                """
                Calculates the loss cut price considering leverage.

                Args:
                        entry_price (float): Entry price.
                        tradetype (str): Trade type ('LONG' or 'SHORT').

                Returns:
                        float: Calculated loss cut price.
                """
                init_equity = trade_config.init_equity
                total_amount = trade_config.total_amount

                init_equity = min(init_equity, total_amount)
                # Calculate the transaction volume using leverage considering the fee
                leveraged_amount = init_equity * leverage

                # Calculate the allowable loss amount
                allowable_loss = init_equity * losscut_rate

                # Calculate the allowable loss amount per Bitcoin
                loss_per_btc = allowable_loss / (leveraged_amount / entry_price)

                # Calculate the loss cut price including the fee
                if tradetype == 'LONG':
                        return entry_price - loss_per_btc
                else:  # In case of SHORT
                        return entry_price + loss_per_btc

        def plot_balance_over_time(self):
                """
                Displays a graph of changes in account balance over time.
                """
                # Plot the change in balance
                self.__fxac.plot_balance_over_time()

        def calculate_win_rate(self, direction, tradetype_exit):
                """
                Calculates and displays the win rate based on the specified direction and trade type.

                Args:
                        direction (str): Trade direction ('lower' or 'upper').
                        tradetype_exit (str): Exit trade type ('LONG_EXIT' or 'SHORT_EXIT').
                """
                # Filter based on the specified direction and trade type
                filtered_trades = self.__fxtrn.get_fxtrn_dataframe()[
                        (self.__fxtrn.get_fxtrn_dataframe()['direction'] == direction) &
                        (self.__fxtrn.get_fxtrn_dataframe()['tradetype'] == tradetype_exit)
                ]

                # Number of winning trades
                wins = filtered_trades[filtered_trades['pl'] > 0].shape[0]

                # Number of losing trades
                losses = filtered_trades[filtered_trades['pl'] < 0].shape[0]

                # Total number of trades
                total_trades = filtered_trades.shape[0]

                # Calculate the win rate (if the total number of trades is 0, the win rate is also 0)
                win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

                # Average pl when you win
                average_win_pl = filtered_trades[filtered_trades['pl'] > 0]['pl'].mean() if wins > 0 else 0

                # Average pl when you lose
                average_loss_pl = filtered_trades[filtered_trades['pl'] < 0]['pl'].mean() if losses > 0 else 0

                self.__logger.log_verbose_message(
                        f"Direction: {direction}, TradeType: {tradetype_exit}, Win Rate: {win_rate:.2f}%, Average Win PL: {average_win_pl:.2f}, Average Loss PL: {average_loss_pl:.2f}")

        def calculate_overall_stats(self):
                """
                        Calculates and displays the overall win rate and average profit/loss for all trades.
                """
                # Get all trades
                all_trades = self.__fxtrn.get_fxtrn_dataframe()

                # Number of winning trades
                wins = all_trades[all_trades['pl'] > 0].shape[0]

                # Number of losing trades
                losses = all_trades[all_trades['pl'] < 0].shape[0]

                # Total number of trades
                total_trades = all_trades.shape[0]

                # Calculate the overall win rate
                overall_win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

                # Average profit when winning
                average_win_pl = all_trades[all_trades['pl'] > 0]['pl'].mean() if wins > 0 else 0

                # Average loss when losing
                average_loss_pl = all_trades[all_trades['pl'] < 0]['pl'].mean() if losses > 0 else 0

                # Overall average profit/loss
                overall_average_pl = all_trades['pl'].mean()

                self.__logger.log_verbose_message(
                        f"Overall Statistics:\n"
                        f"Total Trades: {total_trades}\n"
                        f"Win Rate: {overall_win_rate:.2f}%\n"
                        f"Average Win: {average_win_pl:.2f}\n"
                        f"Average Loss: {average_loss_pl:.2f}\n"
                        f"Overall Average P/L: {overall_average_pl:.2f}"
                )


        def display_all_win_rates(self):
                """
                Calculates and displays the win rate for all directions and trade types.
                """
                # Combination of direction and trade type
                directions = ['lower', 'upper']
                tradetype_exits = ['LONG_EXIT', 'SHORT_EXIT']

                # Loop through all combinations
                for direction in directions:
                        for tradetype_exit in tradetype_exits:
                                # Calculate and display the win rate
                                self.calculate_win_rate(direction, tradetype_exit)

                self.calculate_overall_stats()


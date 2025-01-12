import pandas as pd
import os
import matplotlib.pyplot as plt

from common.config_manager import ConfigManager
from common.trading_logger_db import TradingLoggerDB
from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import ACCOUNT_DATA
from trade_config import trade_config

class FXAccount:
        """
        This class manages an FX account. It supports depositing and withdrawing from the account,
        managing transaction records, and displaying a graph of changes in account balance over time.
        """

        def __init__(self):
                """
                Initializes an instance of FXAccount.
                """
                self.__config_manager = ConfigManager()
                self.__logger = TradingLoggerDB()
                self.__data_loader = MongoDataLoader()

                self.__log_dir = self.__config_manager.get('LOG', 'LOGPATH')
                self.__file_idf = self.__config_manager.get('LOG', 'FILE_IDF_AC')
                self.startup_flag = True
                self.initialize_db_log()

        def get_contract(self) -> str:
                """Getter method to get the type of trading contract.

                Returns:
                        str: The type of trading contract.
                """
                return trade_config.contract

        def get_init_amount(self) -> float:
                """Getter method to get the initial balance.

                Returns:
                        float: The initial balance.
                """
                return trade_config.init_amount

        def get_amount(self) -> float:
                """Getter method to get the current balance.

                Returns:
                        float: The current balance.
                """
                return trade_config.amount


        def get_startup_flag(self):
                """
                Getter method to get the startup flag.
                """
                if self.startup_flag:
                        self.startup_flag = False
                        return 1
                else:
                        return 0

        def initialize_db_log(self):
                """
                Initializes the transaction log in the database.
                """
                self.trn_df = None
                self.table_name = f"{trade_config.contract}" + "_account"

        def generate_filename(self) -> str:
                """
                Generates and returns the full path to the log file.
                Returns:
                        str: The full path to the log file.
                """
                contract = trade_config.contract
                return os.path.join(self.__log_dir, f"{contract}{self.__file_idf}")

        def update_transaction_log(self, date, cash_in, cash_out) -> None:
                """Adds a new record to the transaction log.

                Args:
                        date (str): The transaction date.
                        cash_in (float): The amount deposited.
                        cash_out (float): The amount withdrawn.
                """
                serial = self.__data_loader.get_next_serial(ACCOUNT_DATA)

                new_record = {
                        'serial': serial,
                        'date': date,
                        'cash_in': cash_in,
                        'cash_out': cash_out,
                        'amount': trade_config.amount,
                        'total_assets':trade_config.amount + cash_out,
                        'startup_flag': self.get_startup_flag()
                }

                if self.trn_df is not None:
                        self.trn_df = pd.concat([self.trn_df, pd.DataFrame([new_record])], ignore_index=True)
                else:
                        self.trn_df = pd.DataFrame([new_record])

                self.__data_loader.insert_data(pd.DataFrame([new_record]), coll_type=ACCOUNT_DATA)

        def save_log(self) -> bool:
                """Saves the transaction log to a file.

                Returns:
                        bool: True if the save was successful, False otherwise.
                """
                try:
                        self.trn_df.to_csv(self.generate_filename())
                except OSError as e:
                        self.__logger.log_system_message(f'File Open Error: {self.generate_filename()}, {e}')
                        return False
                return True

        def initialize_log(self) -> bool:
                """Initializes the log file. Deletes the file if it exists.

                Returns:
                        bool: True if the initialization was successful, False otherwise.
                """
                try:
                        filename = self.generate_filename()
                        if os.path.isfile(filename):
                                os.remove(filename)
                except OSError as e:
                        self.__logger.log_system_message(f'File Open Error: {filename}, {e}')
                        return False
                return True

        def withdraw(self, date, cash) -> float:
                """Withdraws the specified amount from the account.

                Args:
                        date (str): The withdrawal date.
                        cash (float): The amount to withdraw.

                Returns:
                        float: The amount actually withdrawn.
                """
                amount = trade_config.amount
                cash_out = min(cash, amount)
                total_amount  = amount
                amount -= cash_out
                trade_config.amount = amount
                trade_config.total_amount  = total_amount
                self.update_transaction_log(date, 0, cash_out)
                self.save_log()
                return cash_out

        def deposit(self, date, cash) -> float:
                """Deposits the specified amount into the account.

                Args:
                        date (str): The deposit date.
                        cash (float): The amount to deposit.

                Returns:
                        float: The amount actually deposited.
                """
                amount = trade_config.amount
                amount += cash
                trade_config.amount = amount
                trade_config.total_amount = amount
                self.update_transaction_log(date, cash, 0)
                self.save_log()
                return cash

        def print_balance(self):
                '''Displays the current account balance.'''
                amount = trade_config.amount
                self.__logger.log_message(f'Current balance: {amount:.2f}')

        def print_performance(self, time: str,leverage: float):
                '''Displays account performance.'''
                contract = trade_config.contract
                amount = trade_config.amount
                init_amount = trade_config.init_amount

                date = self.trn_df['date'].iloc[-1] if not self.trn_df.empty else 'N/A'
                self.__logger.log_transaction(time, '=' * 55)
                self.__logger.log_transaction(time, f'{date}: {contract} ,leverage: {leverage:.2f}')
                self.__logger.log_transaction(time, f'Initial balance [$]: {init_amount:.2f}')
                self.__logger.log_transaction(time, f'Final balance   [$]: {amount:.2f}')
                perf = ((amount - init_amount) / init_amount * 100)
                self.__logger.log_transaction(time, f'Net Performance [%]: {perf:.2f}')
                self.__logger.log_transaction(time, '=' * 55)

        def plot_balance_over_time(self):
                '''Displays a graph of account balance changes over time.'''
                if self.trn_df.empty:
                        self.__logger.log_message("No transaction records found.")
                        return

                plt.figure(figsize=(10, 6))
                plt.plot(self.trn_df['date'], self.trn_df['amount'] + self.trn_df['cash_out'], marker='o', linestyle='-',
                                 color='blue')
                plt.title('Account Balance Over Time')
                plt.xlabel('Date')
                plt.ylabel('Balance [$]')
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
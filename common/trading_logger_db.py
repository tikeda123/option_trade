import logging
import os, sys
import pandas as pd
from typing import Dict, Any

# Get the absolute path of the directory containing this file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Get the path of the parent directory

# Add the path of the parent directory to sys.path
sys.path.append(parent_dir)


from common.trading_logger import TradingLogger
from common.constants import TRADING_LOG
from mongodb.data_loader_mongo import MongoDataLoader
from common.utils import get_config

class TradingLoggerDB(TradingLogger):
        """
        Extended TradingLogger class with database writing capabilities.
        """

        def __init__(self):
                super().__init__()
                self._initialize_from_config()

        def _initialize_from_config(self):
                """
                Initialize logger settings from configuration.
                """
                conf = get_config('LOG')
                self._table_name = conf["DB_TABLE_NAME"]
                self._db_flag = conf["DB_FLAG"]
                self._verbose = conf['VERBOSE']
                self._db_loader = MongoDataLoader()

        def log_transaction(self, date: str, message: str) -> int:
                """
                Log a new transaction to CSV and database if enabled.

                Args:
                        date (str): Transaction date.
                        message (str): Transaction message.

                Returns:
                        int: Serial number of the logged transaction.
                """
                if not self._verbose:
                        return 0

                serial = self._db_loader.get_next_serial(TRADING_LOG)
                new_record = self._create_record(serial, date, message)
                self._update_dataframe(new_record)
                self.log_message(f'{date}|{message}')

                if self._db_flag:
                        self._db_loader.insert_data(pd.DataFrame([new_record]), coll_type=TRADING_LOG)

                return serial

        def log_transaction_update(self, serial: int, date: str, message: str):
                """
                Update an existing transaction log in CSV and database if enabled.

                Args:
                        serial (int): Serial number of the transaction to update.
                        date (str): Updated transaction date.
                        message (str): Updated transaction message.
                """
                new_record = self._create_record(serial, date, message)
                self._update_dataframe(new_record)
                self.log_message(f'{date}|{message}')

                if self._db_flag:
                        self._db_loader.update_data_by_serial(serial, pd.DataFrame([new_record]), coll_type=TRADING_LOG)

        def _create_record(self, serial: int, date: str, message: str) -> Dict[str, Any]:
                """
                Create a record dictionary for a transaction.

                Args:
                        serial (int): Serial number of the transaction.
                        date (str): Transaction date.
                        message (str): Transaction message.

                Returns:
                        Dict[str, Any]: Record dictionary.
                """
                return {'serial': serial, 'date': date, 'message': message}

        def _update_dataframe(self, new_record: Dict[str, Any]):
                """
                Update the internal DataFrame with a new record and save it.

                Args:
                        new_record (Dict[str, Any]): New record to add to the DataFrame.
                """
                new_df = pd.DataFrame([new_record])
                self._tradelog_df = pd.concat([self._tradelog_df, new_df], ignore_index=True)
                self._save_dataframe()

        def _save_dataframe(self):
                """
                Save the internal DataFrame to a CSV file.
                """
                self._tradelog_df.to_csv(self._logfilename_csv, index=False)

def main():
        """
        Main function to demonstrate the usage of TradingLoggerDB.
        """
        logger = TradingLoggerDB()
        serial = logger.log_transaction('2020-01-01', 'test message 11')
        logger.log_transaction('2020-01-01', 'test message 22')
        logger.log_transaction('2020-01-01', 'test message 32')
        logger.log_transaction('2020-01-01', 'test message 42')

        print(f'serial: {serial}')
        logger.log_transaction_update(serial, '2020-01-01', 'test message 52')

if __name__ == '__main__':
        main()
import logging
import pandas as pd
import sys
import os

# Get the absolute path of the directory containing this file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Get the path of the parent directory

# Add the path of the parent directory to sys.path
sys.path.append(parent_dir)


class TradingLogger:
        """
        Manages trade logs and system logs.

        This class records trade execution information and system operation status to log files.
        Logs are saved as text files, and trade logs are additionally saved in CSV format.

        Attributes:
                _initialized (bool): Whether the instance has been initialized.
                __verbose (bool): Whether verbose logging is enabled.
                __loglevel (int): The logging level.
                __logfile_trade (str): Path to the trade log file.
                __logfile_sys (str): Path to the system log file.
                __logfilename_csv (str): CSV filename for trade logs.
                __tradelog_df (DataFrame): DataFrame holding the trade logs.
                __logger_trade (Logger): Logger for trade logs.
                __logger_sys (Logger): Logger for system logs.

        Args:
                conf (dict): Dictionary containing logger configuration information. Expects keys 'VERBOSE', 'LOGPATH', 'LOGFNAME', 'LOGLVL'.
        """
        def __init__(self):
                from common.utils import get_config
                conf = get_config('LOG')
                self._initialized = True

                # Load the same settings as before
                self.__verbose = conf['VERBOSE']
                log_path = f"{parent_dir}/{conf['LOGPATH']}"
                log_fname = conf['LOGFNAME']
                self.__loglevel = conf['LOGLVL']
                self.__logfile_trade = log_path + log_fname
                self.__logfile_sys = log_path + 'system_log.log'
                self.__logfilename_csv = log_path + log_fname.split('.')[0] + '.csv'

                trade_columns = ['Serial', 'Date', 'Message']
                self.__tradelog_df = pd.DataFrame(columns=trade_columns)

                self.__setup_logging()

        def __setup_logging(self):
                """
                Sets up logging. Configures loggers for trade logs and system logs, and adds file handlers and console handlers.
                """

                self.__logger_trade = logging.getLogger('trade_logger')
                self.__logger_trade.setLevel(self.__loglevel)

                if not self.__logger_trade.handlers:
                        # Only add handlers if they haven't been added to the Trade logger yet
                        fh_trade = logging.FileHandler(self.__logfile_trade)
                        fh_trade.setFormatter(logging.Formatter('%(message)s'))
                        self.__logger_trade.addHandler(fh_trade)

                        ch_trade = logging.StreamHandler()
                        ch_trade.setFormatter(logging.Formatter('%(message)s'))
                        self.__logger_trade.addHandler(ch_trade)

                self.__logger_sys = logging.getLogger('sys_logger')
                self.__logger_sys.setLevel(self.__loglevel)

                if not self.__logger_sys.handlers:
                        # Only add handlers if they haven't been added to the System logger yet
                        fh_sys = logging.FileHandler(self.__logfile_sys)
                        fh_sys.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
                        self.__logger_sys.addHandler(fh_sys)

                        ch_sys = logging.StreamHandler()
                        ch_sys.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
                        self.__logger_sys.addHandler(ch_sys)

        def log_message(self, msg: str):
                """
                Logs a message to the trade log.

                Args:
                        msg (str): The message to log.
                """
                self.__logger_trade.info(msg)

        def log_debug_message(self, msg: str):
                """
                Logs a debug message to the trade log.

                Args:
                        msg (str): The debug message to log.
                """
                self.__logger_trade.debug(msg)


        def log_verbose_message(self, msg: str):
                """
                Logs a verbose message to the trade log conditionally. Only logs if the verbose setting is enabled.

                Args:
                        msg (str): The verbose message to log.
                """
                if self.__verbose:
                        self.log_message(msg)


        def log_transaction(self, date: str, message: str):
                """
                Logs a transaction and adds it to the CSV file.

                Args:
                        date (str): The date of the transaction.
                        message (str): The message of the transaction.
                """
                new_record = {'Serial': len(self.__tradelog_df) + 1, 'Date': date, 'Message': message}
                new_df = pd.DataFrame([new_record])  # Convert dictionary to DataFrame
                self.__tradelog_df = pd.concat([self.__tradelog_df, new_df], ignore_index=True)  # Concatenate DataFrames
                self.__tradelog_df.to_csv(self.__logfilename_csv, index=False)
                self.log_message(f'{date}|{message}')

        def log_system_message(self, msg: str):
                """
                Logs a message to the system log.

                Args:
                        msg (str): The message to log.
                """
                self.__logger_sys.info(msg)
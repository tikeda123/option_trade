import sys
import os
import pandas as pd
import csv
from datetime import datetime

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from typing import List, Optional
from typing import Union, Dict,Tuple
from datetime import timedelta, datetime

# Get the absolute path of the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory (A directory in this case)
parent_dir = os.path.dirname(current_dir)

# Add the parent directory's path to sys.path
sys.path.append(parent_dir)

# Import modules
from common.trading_logger import TradingLogger
from common.config_manager import ConfigManager
from mongodb.data_loader import DataLoader  # Import the DataLoader class from data_loader.py
from common.constants import *

class MongoDataLoader(DataLoader):
        """
        Class for loading and interacting with data from a MongoDB database.
        """

        def __init__(self):

                # Call the constructor of the parent class
                super().__init__()
                # Initialize logger, config manager, symbol, interval, etc.
                self.logger = TradingLogger()
                self.config = ConfigManager()
                self.symbol = self.config.get('TRADE_CONFIG', 'SYMBOL')
                self.interval = self.config.get('TRADE_CONFIG', 'INTERVAL')
                # Read MongoDB connection information from the configuration file
                self.host = self.config.get('MONGODB', 'HOST')
                self.port = int(self.config.get('MONGODB', 'PORT'))
                self.username =  self.config.get('MONGODB', 'USERNAME')
                self.password = self.config.get('MONGODB', 'PASSWORD')
                self.database = self.config.get('MONGODB', 'DATABASE')
                self.data_filepath =  f"{parent_dir}/{self.config.get('DATA', 'DBPATH')}"
                # Initialize collection name and sequence collection name
                self.collection = None
                self.set_collection_name(MARKET_DATA)
                self.set_seq_collection_name(TRADING_LOG)

        def _get_collection_info(self, collection_type: str, info_type: str,symbol: str=None, interval: int=None) -> Optional[str]:
                """
                Gets the collection name, sequence collection name, or unique index field name
                based on the collection type.

                Args:
                        collection_type (str): The type of the collection.
                        info_type (str): The type of information to retrieve ('collection', 'seq_collection', 'unique_index').
                        symbol (str, optional): The symbol to use in the collection name. Defaults to None.
                        interval (int, optional): The interval to use in the collection name. Defaults to None.

                Returns:
                        Optional[str]: The collection name, sequence collection name, or unique index field name,
                                                   or None if the collection type is not found.
                """
                if symbol is None:
                        symbol = self.symbol

                if interval is None:
                        interval = self.interval

                info_dict = {
                        'collection': {
                                AIML_TRACING: f"{symbol}_{interval}_aiml_tracing",
                                ROLLING_AI_DATA:f"{symbol}_{interval}_rolling_ai_data",
                                MARKET_DATA: f"{symbol}_{interval}_market_data",
                                MARKET_DATA_TECH: f"{symbol}_{interval}_market_data_tech",
                                MARKET_DATA_ML_UPPER: f"{symbol}_{interval}_market_data_mlts_upper",
                                MARKET_DATA_ML_LOWER: f"{symbol}_{interval}_market_data_mlts_lower",
                                MARKET_DATA_ML: f"{symbol}_{interval}_market_data_ml",
                                TRANSACTION_DATA: "transaction_data",
                                ACCOUNT_DATA: "account_data",
                                TRADE_CONFIG: "trade_config",
                                TRADING_LOG: "trading_log",
                                OPTION_SYMBOL: "option_symbol",
                                OPTION_TICKER: "option_ticker"
                        },
                        'seq_collection': {
                                TRADING_LOG: "trading_log_seq",
                                TRANSACTION_DATA: "transaction_data_seq",
                                ACCOUNT_DATA: "account_data_seq"
                        },
                        'unique_index': {
                                AIML_TRACING: "start_at",
                                ROLLING_AI_DATA: "start_at",
                                MARKET_DATA: "start_at",
                                MARKET_DATA_TECH: "start_at",
                                MARKET_DATA_ML_UPPER: "start_at",
                                MARKET_DATA_ML_LOWER: "start_at",
                                MARKET_DATA_ML: "start_at",
                                TRANSACTION_DATA: "serial",
                                ACCOUNT_DATA: "serial",
                                TRADE_CONFIG: "serial",
                                TRADING_LOG: "serial",
                                OPTION_SYMBOL: "symbol",
                                OPTION_TICKER: "symbol_id"
                        }
                }
                return info_dict[info_type].get(collection_type)

        def set_collection_name(self, collection_name: str, symbol: str=None, interval: int=None) -> None:
                """
                Sets the collection name.

                Args:
                        collection_name (str): The name of the collection to set.
                        symbol (str, optional): The symbol to use in the collection name. Defaults to None.
                        interval (int, optional): The interval to use in the collection name. Defaults to None.
                """
                self.collection = self._get_collection_info(collection_name, 'collection', symbol=symbol, interval=interval)

        def set_direct_collection_name(self, collection_name: str) -> None:
                """
                Sets the collection name directly.
                When using this method, the sequence collection name and
                unique index fields are not automatically set.

                Args:
                collection_name (str): The name of the collection to set.
                """
                self.collection = collection_name

        def set_seq_collection_name(self, collection_name: str) -> None:
                """
                Sets the sequence collection name.

                Args:
                        collection_name (str): The name of the collection to set as the sequence collection.
                """
                self.seq_collection = self._get_collection_info(collection_name, 'seq_collection')

        def set_unique_index(self, field_name: str) -> None:
                """
                Sets the unique index field.

                Args:
                        field_name (str): The name of the field to set as the unique index.
                """
                self.unique_index = self._get_collection_info(field_name, 'unique_index')

        def get_next_serial(self, coll_type: Optional[str] = None) -> int:
                """
                Retrieves the next serial number from the sequence collection.

                Args:
                        coll_type (str, optional): The type of the collection for which to get the next serial number.
                                                                         Defaults to None.

                Returns:
                        int: The next serial number.
                """
                if coll_type is not None:
                        self.set_seq_collection_name(coll_type)
                self.connect(coll_seq=True)
                try:
                        query = {'_id': 'serial'}
                        update = {'$inc': {'seq': 1}}
                        result = self.col.find_one_and_update(
                                query, update, upsert=True, return_document=True
                        )
                        return result['seq']
                except OperationFailure as e:
                        self.logger.log_system_message(f"Failed to get the next sequence number: {str(e)}")
                        raise
                finally:
                        self.close()

        def convert_marketdata(self, df: pd.DataFrame) -> pd.DataFrame:
                """
                Converts data types of a market data DataFrame.

                Args:
                        df (pd.DataFrame): The DataFrame containing market data.

                Returns:
                        pd.DataFrame: The DataFrame with converted data types.
                """
                from datetime import datetime
                df['start_at'] = pd.to_datetime(df['start_at'], unit='s')
                df['date'] = pd.to_datetime(df['date'])
                numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover']
                df[numeric_columns] = df[numeric_columns].astype(float)
                return df

        def connect(self, coll_seq: Optional[bool] = None) -> None:
                """
                Establishes a connection to the MongoDB database.

                Args:
                        coll_seq (bool, optional): Whether to connect to the sequence collection. Defaults to None.
                """
                try:
                        self.client = MongoClient(
                                host=self.host,
                                port=self.port,
                                username=self.username,
                                password=self.password
                        )
                        self.db = self.client[self.database]
                        if coll_seq:
                                self.col = self.db[self.seq_collection]
                        else:
                                self.col = self.db[self.collection]
                except ConnectionFailure as e:
                        self.logger.log_system_message(f"Failed to connect to MongoDB: {str(e)}")
                        raise

        def load_data_from_datetime_period(
                self, start_date: Optional[str] = None, end_date: Optional[str] = None, coll_type: Optional[str] = None,
        symbol: Optional[str] = None,interval: Optional[int]=None) -> Optional[pd.DataFrame]:

                if coll_type is not None:
                        self.set_collection_name(coll_type, symbol=symbol, interval=interval)
                self.connect()

                try:
                        # クエリを作成
                        query = {}
                        if start_date is not None:
                                query['start_at'] = {'$gte': pd.to_datetime(start_date)}
                        if end_date is not None:
                                if 'start_at' in query:
                                        query['start_at']['$lte'] = pd.to_datetime(end_date)  # '$lt' から '$lte' に変更
                        else:
                                query['start_at'] = {'$lte': pd.to_datetime(end_date)}  # '$lt' から '$lte' に変更

                        # データを取得
                        data = list(self.col.find(query))

                        if len(data) > 0:
                                self._df = pd.DataFrame(data)
                                self._df['start_at'] = pd.to_datetime(self._df['start_at'])
                                self.set_df_raw(self._df)
                                self.remove_unuse_colums()
                                return self._df
                        else:
                                self.logger.log_system_message("No data found for the specified datetime range.")
                                return None
                except OperationFailure as e:
                        self.logger.log_system_message(f"Failed to load data: {str(e)}")
                        raise
                finally:
                        self.close()


        def load_data_from_point_date(
                self, point_date: Union[str, datetime], nsteps: int, collection_name: str, symbol: Optional[str] = None,interval: Optional[int]=None
        ) -> Optional[pd.DataFrame]:
                """
                Loads data from MongoDB within a specified past time range from a given date
                and converts it to a DataFrame.

                Args:
                        point_date (Union[str, datetime]): The reference date.
                        Can be specified as a string or datetime object.
                        nsteps (int): Specifies how many steps (intervals) to go back from point_date.
                        collection_name (str): The name of the collection to retrieve data from.

                Returns:
                        Optional[pd.DataFrame]: The DataFrame containing the loaded data,
                        or None if no data is found.
                """
                if interval is None:
                        interval = self.interval
                if interval is None:
                                raise ValueError("Interval must be specified either as an argument or as a class attribute.")

                self.set_collection_name(collection_name, symbol=symbol, interval=interval)
                self.connect()

                try:
                        if interval =="D":
                                interval = 1440

                        nsteps = nsteps - 1
                        # Convert point_date to datetime object
                        if isinstance(point_date, str):
                                point_date = pd.to_datetime(point_date)

                        # Calculate the time interval for nsteps
                        time_delta = timedelta(minutes=int(interval*nsteps))

                        # Calculate start date and end date
                        start_date = point_date - time_delta
                        end_date = point_date + timedelta(minutes=int(interval))

                        # Create query
                        query = {
                                'start_at': {
                                '$gte': start_date,
                                '$lt': end_date
                                }
                         }
                         # Sort by start_at in ascending order
                        data = list(self.col.find(query).sort("start_at", 1))

                        if len(data) > 0:
                                self._df = pd.DataFrame(data)
                                self._df['start_at'] = pd.to_datetime(self._df['start_at'])
                                self.set_df_raw(self._df)
                                self.remove_unuse_colums()
                                return self._df
                        else:
                                print(f"No data found from {start_date} to {end_date}.")
                                self.logger.log_system_message(f"No data found from {start_date} to {end_date}.")
                                return None
                except OperationFailure as e:
                        self.logger.log_system_message(f"Failed to load data: {str(e)}")
                        raise
                finally:
                        self.close()

        def get_latest_n_records(self, coll_type: str, nsteps: int = 100, symbol: Optional[str] = None,
                                     interval: Optional[int] = None) -> Optional[pd.DataFrame]:
                """
                Gets the latest n records from the specified collection, going back in time.

                Args:
                        coll_type (str): The type of the collection to retrieve data from.
                        nsteps (int, optional): The number of records to retrieve. Defaults to 100.
                        symbol (str, optional): The symbol for the collection. Defaults to None.
                        interval (str, optional): The interval for the collection. Defaults to None.

                Returns:
                        Optional[pd.DataFrame]: A DataFrame containing the latest n records, or None if an error occurs.
                """
                self.set_collection_name(coll_type, symbol=symbol, interval=interval)
                self.connect()
                try:
                        # Sort by 'start_at' in descending order (latest to oldest) and limit to nsteps
                        data = list(self.col.find().sort("start_at", -1).limit(nsteps))

                        if len(data) > 0:
                                df = pd.DataFrame(data)
                                df['start_at'] = pd.to_datetime(df['start_at'])
                                # Reorder DataFrame from oldest to latest
                                df_sorted = df.sort_values(by='start_at')
                                self.set_df_raw(df_sorted)
                                return df_sorted
                        else:
                                print("No data found.")
                                return None
                except OperationFailure as e:
                        self.logger.log_system_message(f"Failed to load data: {str(e)}")
                        raise
                finally:
                        self.close()

        def load_data(self, coll_type: str, symbol: Optional[str]=None, interval: Optional[int]=None, check_nan: bool=False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, bool]]:
                """
                Loads data from MongoDB and converts it to a DataFrame.

                Args:
                        coll_type (str): The type of the collection to load data from.
                        symbol (str, optional): The symbol to use in the collection name. Defaults to None.
                        interval (int, optional): The interval to use in the collection name. Defaults to None.
                        check_nan (bool, optional): Whether to check for NaN values in the date column. Defaults to False.

                Returns:
                        Union[pd.DataFrame, Tuple[pd.DataFrame, bool]]:
                        If check_nan is False, returns the loaded DataFrame.
                        If check_nan is True, returns a tuple of (DataFrame, bool) where the bool indicates if NaN check passed.
                """
                self.set_collection_name(coll_type, symbol=symbol, interval=interval)
                self.connect()
                try:
                        data = list(self.col.find())
                        self._df = pd.DataFrame(data)
                        self.set_df_raw(self._df)
                        self.remove_unuse_colums()

                        if check_nan:
                                nan_check_passed = not self._df['date'].isna().any()
                                return self._df, nan_check_passed
                        else:
                                return self._df
                except OperationFailure as e:
                        self.logger.log_system_message(f"Failed to load data: {str(e)}")
                        raise
                finally:
                        self.close()

        '''
        def load_data(self, coll_type: str, symbol: Optional[str]=None, interval: Optional[int]=None) -> None:
                """
                Loads data from MongoDB and converts it to a DataFrame.

                Args:
                        coll_type (str): The type of the collection to load data from.
                        symbol (str, optional): The symbol to use in the collection name. Defaults to None.
                        interval (int, optional): The interval to use in the collection name. Defaults to None.

                """
                self.set_collection_name(coll_type, symbol=symbol, interval=interval)
                self.connect()
                try:
                        data = list(self.col.find())
                        self._df = pd.DataFrame(data)
                        self.set_df_raw(self._df)
                        self.remove_unuse_colums()
                        return self._df
                except OperationFailure as e:
                        self.logger.log_system_message(f"Failed to load data: {str(e)}")
                        raise
                finally:
                        self.close()
        '''

        def close(self) -> None:
                """
                Closes the connection to the MongoDB database.
                """
                try:
                        self.client.close()
                except Exception as e:
                        self.logger.log_system_message(f"Failed to close the connection to MongoDB: {str(e)}")
                        raise

        def create_unique_index(self, coll_type: str) -> None:
                """
                Creates a unique index on the specified field.

                Args:
                        coll_type (str, optional): The type of the collection on which to create the unique index.
                                                                         Defaults to None.
                """
                self.set_unique_index(coll_type)
                field_name = self.unique_index
                self.connect()
                try:
                        self.col.create_index([(field_name, 1)], unique=True)
                except OperationFailure as e:
                        if "already exists" in str(e):
                                pass
                        else:
                                self.logger.log_system_message(f"Failed to create index: {str(e)}")
                                raise
                finally:
                        self.close()

        def insert_data(self, data: pd.DataFrame, coll_type: str, symbol:Optional[str]=None, interval:Optional[int]=None) -> None:
                """
                Inserts data into the MongoDB database.

                Args:
                        data (pd.DataFrame): The DataFrame containing the data to insert.
                        coll_type (str, optional): The type of the collection to insert data into. Defaults to None.
                """
                self.set_collection_name(coll_type, symbol=symbol, interval=interval)
                self.create_unique_index(coll_type)
                self.connect()
                try:
                        docs = data.to_dict(orient='records')
                        for doc in docs:
                                try:
                                        self.col.insert_one(doc)
                                except OperationFailure as e:
                                        pass  # Handle duplicate key errors or other insertion errors as needed
                except Exception as e:
                        self.logger.log_system_message(f"An error occurred while inserting data: {str(e)}")
                        raise
                finally:
                        self.close()

        def update_data(self, query: dict, update: dict, coll_type: str, symbol:Optional[str]=None, interval:Optional[int]=None) -> None:
                """
                Updates data in the MongoDB database.

                Args:
                        query (dict): The query to select the documents to update.
                        update (dict): The update to apply to the selected documents.
                        coll_type (str, optional): The type of the collection to update data in. Defaults to None.
                """

                self.set_collection_name(coll_type, symbol=symbol, interval=interval)
                self.create_unique_index(coll_type)
                self.connect()
                try:
                        self.col.update_many(query, {'$set': update})
                except OperationFailure as e:
                        self.logger.log_system_message(f"Failed to update data: {str(e)}")
                        raise
                finally:
                        self.close()

        def update_data_by_serial(
                self, serial_id: int, new_df: pd.DataFrame, coll_type: str, symbol:Optional[str]=None, interval:Optional[int]=None
                ) -> None:
                """
                Updates a document in the MongoDB database based on its serial ID.

                Args:
                        serial_id (int): The serial ID of the document to update.
                        new_df (pd.DataFrame): The DataFrame containing the updated data.
                        coll_type (str, optional): The type of the collection containing the document to update.
                                                                         Defaults to None.
                """
                self.set_collection_name(coll_type, symbol=symbol, interval=interval)
                self.connect()
                try:
                        query = {'serial': serial_id}
                        update_data = new_df.to_dict(orient='records')[0]
                        result = self.col.update_one(query, {'$set': update_data})
                        if result.modified_count == 0:
                                self.logger.log_system_message(f"No document found with 'serial' {serial_id}.")
                except OperationFailure as e:
                        self.logger.log_system_message(f"Failed to update data: {str(e)}")
                        raise
                finally:
                        self.close()

        def find_data_by_start_at(self, start_at: str, coll_type: str, symbol:Optional[str]=None, interval:Optional[int]=None) -> Optional[str]:
                """
                Finds a document in the MongoDB database based on its 'start_at' field.

                Args:
                        start_at (str): The 'start_at' value of the document to find.
                        coll_type (str, optional): The type of the collection to search for the document.
                                                                         Defaults to None.

                Returns:
                        Optional[str]: The document found, or None if no document is found.
                """
                self.set_collection_name(coll_type, symbol=symbol, interval=interval)
                self.connect()
                try:
                        query = {'start_at': start_at}
                        result = self.col.find_one(query)
                        return result
                except OperationFailure as e:
                        self.logger.log_system_message(f"Failed to find data: {str(e)}")
                        raise
                finally:
                        self.close()

        def update_data_by_start_at(self, start_at: str, new_df: pd.DataFrame, coll_type: str, symbol:Optional[str]=None, interval:Optional[int]=None) -> None:
                """
                Updates a document in the MongoDB database based on its 'start_at' field.

                Args:
                        start_at (str): The 'start_at' value of the document to update.
                        new_df (pd.DataFrame): The DataFrame containing the updated data.
                        coll_type (str, optional): The type of the collection containing the document to update.
                                                                         Defaults to None.
                """
                self.set_collection_name(coll_type, symbol=symbol, interval=interval)
                self.connect()
                try:
                        query = {'start_at': start_at}
                        update_data = new_df.to_dict(orient='records')[0]
                        result = self.col.update_one(query, {'$set': update_data})
                        if result.modified_count == 0:
                                self.logger.log_system_message(f"No document found with 'start_at' {start_at}.")
                except OperationFailure as e:
                        self.logger.log_system_message(f"Failed to update data: {str(e)}")
                        raise
                finally:
                        self.close()


        def delete_data(self, query: dict, coll_type:str) -> None:
                """
                Deletes data from the MongoDB database.

                Args:
                        query (dict): The query to select the documents to delete.
                        coll_type (str, optional): The type of the collection to delete data from. Defaults to None.
                """
                self.set_collection_name(coll_type)
                self.connect()
                try:
                        self.col.delete_many(query)
                except OperationFailure as e:
                        self.logger.log_system_message(f"Failed to delete data: {str(e)}")
                        raise
                finally:
                        self.close()

        def drop_collection_by_colltype(self, coll_type: str, symbol:Optional[ str]=None,interval:Optional[int]=None) -> None:
                """
                Drops a collection from the MongoDB database by its specific symbol.

                Args:
                        coll_type (str): The type of the collection to drop.
                        symbol (str): The symbol to use in the collection name.
                """
                self.set_collection_name(coll_type, symbol=symbol, interval=interval)
                self.connect()
                try:
                        self.col.drop()
                        if self.is_collection_exists(coll_type):
                                self.logger.log_system_message(f"Failed to drop collection '{self.collection}'.")
                        else:
                                self.logger.log_system_message(f"Collection '{self.collection}' dropped successfully.")
                except OperationFailure as e:
                        self.logger.log_system_message(f"Failed to drop collection: {str(e)}")
                        raise
                finally:
                        self.close()

        def drop_collection(self, collection_name: str) -> None:
                """
                Drops a collection from the MongoDB database by its specific name.

                Args:
                        collection_name (str): The name of the collection to drop.
                """
                self.set_direct_collection_name(collection_name)  # Set the collection name directly
                self.connect()  # Establish connection to the database
                try:
                        self.col.drop()  # Drop the collection
                        self.logger.log_system_message(f"Collection '{self.collection}' dropped successfully.")  # Log success message
                except OperationFailure as e:
                        self.logger.log_system_message(f"Failed to drop collection: {str(e)}")  # Log error message if any
                        raise  # Re-raise the exception
                finally:
                        self.close()  # Close the database connection

        def is_collection_exists(self, coll_type: str, symbol:Optional[str]=None,interval:Optional[int]=None) -> bool:
                """
                Checks if a collection with the given name exists in the database.

                Args:
                        collection_name (str): The name of the collection to check.

                Returns:
                        bool: True if the collection exists, False otherwise.
                """
                self.set_collection_name(coll_type, symbol=symbol, interval=interval)
                self.connect()
                try:
                        if self.collection in self.db.list_collection_names():
                                self.logger.log_system_message(f"Collection '{self.collection}' exists.")
                                return True
                        else:
                                self.logger.log_system_message(f"Collection '{self.collection}' does not exist.")
                                return False
                except OperationFailure as e:
                        self.logger.log_system_message(f"Failed to check collection existence: {str(e)}")
                        raise
                finally:
                        self.close()

        def find_duplicate_times(self, coll_type: str) -> List[dict]:
                """
                Finds records with duplicate 'start_at' values in the specified collection type.

                Args:
                        coll_type (str): The type of the collection to check for duplicates.

                Returns:
                        List[dict]: A list of dictionaries, where each dictionary represents a duplicate 'start_at' value
                                                and its count.
                """

                self.set_collection_name(coll_type)
                self.connect()

                try:
                        pipeline = [
                                {"$group": {"_id": "$start_at", "count": {"$sum": 1}}},
                                {"$match": {"count": {"$gt": 1}}},
                                {"$project": {"_id": 0, "start_at": "$_id", "count": 1}}
                        ]
                        duplicates = list(self.col.aggregate(pipeline))
                        return duplicates
                except Exception as e:
                        self.logger.log_system_message(
                                f"An error occurred while finding duplicate times: {str(e)}"
                        )
                        raise
                finally:
                        self.close()

        def print_duplicate_times(self, coll_type: str) -> None:
                """
                Finds and prints records with duplicate 'start_at' values in the specified collection type.

                Args:
                        coll_type (str): The type of the collection to check for duplicates.
                """
                duplicate_times = self.find_duplicate_times(coll_type)

                if duplicate_times:
                        print("Duplicate times:")
                        for doc in duplicate_times:
                                print(f"  start_at: {doc['start_at']}, count: {doc['count']}")
                else:
                        print("No duplicate times found.")

        def export_collection_to_csv(self, coll_type: str, symbol: Optional[str] = None, interval: Optional[int] = None) -> str:
                """
                Exports all data from a specified collection to a CSV file.

                Args:
                        coll_type (str): The type of the collection to export.
                        symbol (str, optional): The symbol to use in the collection name. Defaults to None.
                        interval (int, optional): The interval to use in the collection name. Defaults to None.

                Returns:
                        str: The path of the created CSV file.
                """
                self.set_collection_name(coll_type, symbol=symbol, interval=interval)
                self.connect()

                try:
                        self.logger.log_system_message(f"Attempting to export collection: {self.collection}")

                        # Fetch all documents from the collection
                        data = list(self.col.find())
                        self.logger.log_system_message(f"Found {len(data)} documents in the collection")

                        if not data:
                                self.logger.log_system_message(f"No data found in collection {self.collection}")
                                return ""

                        # Create a filename with current datetime and collection name
                        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{current_time}_{self.collection}.csv"
                        filepath = os.path.join(self.data_filepath, filename)

                        self.logger.log_system_message(f"Attempting to create file: {filepath}")

                        # Ensure the directory exists
                        os.makedirs(os.path.dirname(filepath), exist_ok=True)

                        # Write data to CSV file
                        with open(filepath, 'w', newline='') as csvfile:
                                writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
                                writer.writeheader()
                                for row in data:
                                        writer.writerow(row)

                        self.logger.log_system_message(f"Data successfully exported to {filepath}")
                        return filepath

                except Exception as e:
                        self.logger.log_system_message(f"Failed to export data to CSV: {str(e)}")
                        self.logger.log_system_message(f"Error type: {type(e).__name__}")
                        self.logger.log_system_message(f"Error details: {e.args}")
                        raise
                finally:
                        self.close()

def is_multiple_hour(datetime_str: str, multiple: int) -> bool:
                try:
                        import datetime
                        dt_obj = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
                        return dt_obj.hour % multiple == 0
                except ValueError:
                        print("Invalid datetime format. Please use 'YYYY-MM-DD HH:MM:SS'.")
                return False

def main():
        start_date = "2021-01-01 00:00:00"
        end_date = "2021-01-08 00:00:00"

        db = MongoDataLoader()
        df = db.load_data_from_datetime_period(start_date, end_date, MARKET_DATA)
        print(df)

if __name__ == '__main__':
        main()



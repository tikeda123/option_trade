import pandas as  pd
import os, sys
from typing import Dict, Any

# Set path to parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import AIML_TRACING


class TracingAimlInfo:
        """
        This class manages the database tracing for AIML. It supports loading data from the database, and
        displaying a graph of changes in data over time.
        """

        def __init__(self):
                """
                Initializes an instance of DBTracingAIML.
                """
                self.__data_loader = MongoDataLoader()
                self.__data_loader.set_collection_name(AIML_TRACING)
                #self.__data = self.__data_loader.load_data(coll_type=AIML_TRACING)

        def load_data(self, coll_type: str) -> pd.DataFrame:
                # Load data from the specified collection type
                return self.__data_loader.load_data(coll_type=coll_type)

        def create_dict(self, glist:list[str]) -> Dict[str,Any]:
                """
                Create a dictionary with the given list of AIML versions.

                Args:
                        glist (list[str]): A list of AIML versions.

                Returns:
                        Dict[str,Any]: A dictionary with the given AIML versions.
                """
                # Create a base record dictionary with common fields
                base_record = {
                        'start_at': None,
                        'pred': None,
                        'actual': None,
                        'profit': None,
                }

                # Add version-specific fields to the base record
                for version in glist:
                        base_record.update({
                                f'pred_{version}': None,
                                f'hit_rate_{version}': None,
                                f'avg_error_{version}': None,
                                f'avg_profit_{version}': None
                        })

                return base_record

        def new_record(self, start_at: str, glist:list[str]) -> None:
                """
                Create a new record in the database with the given start time and AIML data.

                Args:
                        start_at (str): The start time of the record.
                        glist (Dict[str,Any]): The AIML data to be stored in the record.
                """

                # Initialize the base record with common fields
                base_record = {
                        'start_at': start_at,
                        'actual': None,
                        'profit': None,
                }

                # Add version-specific fields to the base record
                for version in glist:
                        base_record.update({
                                f'pred_{version}': None,
                                f'hit_rate_{version}': None,
                                f'avg_error_{version}': None,
                                f'avg_profit_{version}': None
                        })

                # Check if a record with the given start time already exists
                record = self.__data_loader.find_data_by_start_at(start_at, AIML_TRACING)

                if record is None:
                        # If no record exists, insert a new one
                        self.__data_loader.insert_data(pd.DataFrame([base_record]), coll_type=AIML_TRACING)
                else:
                        # If a record exists, update it
                        self.__data_loader.update_data_by_start_at(start_at,pd.DataFrame([base_record]), coll_type=AIML_TRACING)

        def find_record_by_start_at(self, start_at: str) -> pd.DataFrame:
                """
                Find a record in the database with the given start time.

                Args:
                        start_at (str): The start time of the record.

                Returns:
                        pd.DataFrame: The record with the given start time.
                """
                return self.__data_loader.find_data_by_start_at(start_at, AIML_TRACING)

        def update_record_by_group(self, start_at: str, aiml_data: Dict[str, Any]) -> None:
                """
                Update an existing record in the database with the given start time, actual trend, profit, and AIML data.

                Args:
                        start_at (str): The start time of the record.
                        actual_trend (int): The actual trend.
                        profit (float): The profit.
                        aiml_data (Dict[str, Any]): The AIML data to be stored in the record.
                """
                # Update an existing record with new AIML data
                self.__data_loader.update_data_by_start_at(start_at,pd.DataFrame([aiml_data]), coll_type=AIML_TRACING)


def main():
        tif = TracingAimlInfo()

        glist = ['rolling_v1','rolling_v2','rolling_v3','rolling_v4']

        dict = tif.create_dict(glist)
        tif.new_record('2021-01-01 00:00:00',glist)

        dict['start_at'] = '2021-01-01 00:00:00'
        dict['actual'] = 1
        dict['profit'] = 0.1
        dict['pred_rolling_v1'] = 0.11
        dict['hit_rate_rolling_v1'] = 0.12
        dict['avg_error_rolling_v1'] = 0.13
        dict['avg_profit_rolling_v1'] = 0.14

        dict['pred_rolling_v2'] = 0.21
        dict['hit_rate_rolling_v2'] = 0.22
        dict['avg_error_rolling_v2'] = 0.23
        dict['avg_profit_rolling_v2'] = 0.24

        dict['pred_rolling_v3'] = 0.31
        dict['hit_rate_rolling_v3'] = 0.32
        dict['avg_error_rolling_v3'] = 0.33
        dict['avg_profit_rolling_v3'] = 0.34

        dict['pred_rolling_v4'] = 0.41
        dict['hit_rate_rolling_v4'] = 0.42
        dict['avg_error_rolling_v4'] = 0.43
        dict['avg_profit_rolling_v4'] =  0.44

        tif.update_record_by_group('2021-01-01 00:00:00',1,0.1,dict)
        print(tif.load_data(coll_type=AIML_TRACING))

        glist = ['rolling_v5','rolling_v6','rolling_v7','rolling_v8']

        dict = tif.create_dict(glist)
        tif.new_record('2021-01-01 00:00:00',glist)

        dict['start_at'] = '2021-01-01 00:00:00'
        dict['actual'] = 1
        dict['profit'] = 0.1

        dict['pred_rolling_v5'] =  0.51
        dict['hit_rate_rolling_v5'] = 0.52
        dict['avg_error_rolling_v5'] = 0.53
        dict['avg_profit_rolling_v5'] = 0.54

        dict['pred_rolling_v6'] = 0.61
        dict['hit_rate_rolling_v6'] = 0.62
        dict['avg_error_rolling_v6'] = 0.63
        dict['avg_profit_rolling_v6'] = 0.64

        dict['pred_rolling_v7'] = 0.71
        dict['hit_rate_rolling_v7'] = 0.72
        dict['avg_error_rolling_v7'] = 0.73
        dict['avg_profit_rolling_v7'] = 0.74

        dict['pred_rolling_v8'] =  0.81
        dict['hit_rate_rolling_v8'] = 0.82
        dict['avg_error_rolling_v8'] = 0.83
        dict['avg_profit_rolling_v8'] = 0.84


        tif.update_record_by_group('2021-01-01 00:00:00',1,0.1,dict)
        print(tif.load_data(coll_type=AIML_TRACING))

        glist = ['rolling_v9','rolling_v10','rolling_v11','rolling_v12']

        dict = tif.create_dict(glist)
        tif.new_record('2021-01-01 00:00:00',glist)

        dict['start_at'] = '2021-01-01 00:00:00'
        dict['actual'] = 1
        dict['profit'] = 0.1

        dict['pred_rolling_v9'] = 0.91
        dict['hit_rate_rolling_v9'] = 0.92
        dict['avg_error_rolling_v9'] = 0.93
        dict['avg_profit_rolling_v9'] = 0.94

        dict['pred_rolling_v10'] = 1.01
        dict['hit_rate_rolling_v10'] = 1.02
        dict['avg_error_rolling_v10'] = 1.03
        dict['avg_profit_rolling_v10'] = 1.04

        dict['pred_rolling_v11'] = 1.11
        dict['hit_rate_rolling_v11'] = 1.12
        dict['avg_error_rolling_v11'] = 1.13
        dict['avg_profit_rolling_v11'] = 1.14

        dict['pred_rolling_v12'] = 1.21
        dict['hit_rate_rolling_v12'] = 1.22
        dict['avg_error_rolling_v12'] = 1.23
        dict['avg_profit_rolling_v12'] = 1.24

        tif.update_record_by_group('2021-01-01 00:00:00',1,0.1,dict)
        print(tif.load_data(coll_type=AIML_TRACING))




if __name__ == '__main__':
        main()
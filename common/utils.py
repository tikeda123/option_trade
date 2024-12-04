import sys
import os
import json
from datetime import datetime, timedelta
from pandas import Timestamp

from common.constants import CONFIG_FILENAME, AIMODLE_CONFIG_FILENAME

def setup_sys_path():
        """
        Add the parent directory to the system path.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        sys.path.append(parent_dir)

def format_dates(start_date, end_date,date_only=False) -> tuple:
        """
        Format and validate the start and end dates.
        """
        if date_only == False:
                formatted_start_date = append_time_if_missing(start_date)
                formatted_end_date = append_time_if_missing(end_date)
        else:
                formatted_start_date = start_date
                formatted_end_date = end_date

        validate_date_format(formatted_start_date,date_only)
        validate_date_format(formatted_end_date,date_only)

        return formatted_start_date, formatted_end_date

def append_time_if_missing(date:str) -> str:
        """
        Append default time and timezone if the date string doesn't include them.
        """
        return date + " 00:00:00+0900" if len(date) == 10 else date

def validate_date_format(date,date_only):
        """
        Validate the date string format. Exit with an error message if invalid.

        Args:
                date (str): The date string to validate
        """
        try:
                if date_only==True:
                        datetime.strptime(date, "%Y-%m-%d")
                else:
                        datetime.strptime(date, "%Y-%m-%d %H:%M:%S%z")
        except ValueError:
                exit_with_message(
                        "Invalid date format. Use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS+ZZZZ")


def exit_with_message(message):
        """
        Print an error message and exit the program with status code 1.
        """
        print(message)
        sys.exit(1)

def get_config_fullpath(filename=None)-> str:
        """
        Get the path to the configuration file from environment variables.

        Args:
                filename (str): The name of the configuration file

        Returns:
                str: The full path to the configuration file
        """
        if filename is None:
                filename = CONFIG_FILENAME
        current_script_path = os.path.dirname(os.path.abspath(__file__))
        config_file_directory = os.path.join(current_script_path, os.pardir)
        config_path = os.path.join(config_file_directory, filename )
        return os.path.abspath(config_path)

def get_config(tag,filename=None)-> dict:
        """
        Open and read the configuration file as JSON.

        Args:
                fillename (str): The name of the configuration file
                tag (str): The tag to retrieve from the configuration file

        Returns:
                dict: The configuration data or a specific tag
        """
        config_path = get_config_fullpath(filename)
        with open(config_path, 'r') as config_file:
         config_data = json.load(config_file)

        if tag is not None:
             return config_data[tag]
        return config_data

def get_config_model(model_group_tag:str,  id:str=None,):
        """
        Get the configuration data for a specific model ID.

        Args:
                model_id (str): The ID of the model
                filename (str): The name of the configuration file

        Returns:
                dict: The configuration data for the model
        """
        config_data = get_config(model_group_tag,AIMODLE_CONFIG_FILENAME )

        if id is not None:
                return config_data[id]
        return config_data

def add_minutes(time_str, minutes)-> str:
        """
        Add minutes to the given datetime string.

        Args:
                time_str (str): The datetime string
                minutes (int): The number of minutes to add

        Returns:
                str: The new datetime string
        """
        return _adjust_minutes(time_str, minutes)

def subtract_minutes(time_str, minutes):
        """
        Subtract minutes from the given datetime string.
        """
        return _adjust_minutes(time_str, -minutes)

def _adjust_minutes(time_str, minutes):
        """
        Internal function to adjust minutes.
        """
        time_obj = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        new_time_obj = time_obj + timedelta(minutes=minutes)
        return new_time_obj.strftime("%Y-%m-%d %H:%M:%S")

def extract_hour_and_minute(time_str):
        """
        Extract hour and minute from the given datetime string.
        """
        time_obj = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        return time_obj.hour, time_obj.minute

# Example usage
"""
original_time = "2024-06-01 00:00:00"
added_time = add_minutes(original_time, 120)  # Add 120 minutes (2 hours)
subtracted_time = subtract_minutes(original_time, 30)  # Subtract 30 minutes

print("Original Time:", original_time)
print("Time After Adding 120 Minutes:", added_time)
print("Time After Subtracting 30 Minutes:", subtracted_time)
"""
from datetime import datetime, timedelta

from datetime import datetime, timedelta

from datetime import datetime, timedelta

def get_higher_timeframe_info(timestamp_str, target_timeframe):
        """
        Convert a given 1-hour timeframe timestamp to the start of a higher timeframe and determine if it's a trigger point.
         :param timestamp_str: A string representing the 1-hour timeframe timestamp (format: "YYYY-MM-DD HH:00:00")
         :param target_timeframe: The target higher timeframe (12, 4, or 2 hours)
        :return: A tuple (is_trigger, result_timestamp)
        is_trigger: Boolean indicating if it's a trigger point
        result_timestamp: String representing the start of the higher timeframe
        """
        # Parse the input timestamp
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

        if target_timeframe == 12:
        # For 12-hour timeframe
                is_trigger = timestamp.hour == 23 or timestamp.hour == 11
                if timestamp.hour >= 12:
                        start_time = timestamp.replace(hour=12, minute=0, second=0)
                else:
                        start_time = timestamp.replace(hour=0, minute=0, second=0)
        elif target_timeframe == 4:
                # For 4-hour timeframe
                is_trigger = timestamp.hour % 4 == 3
                start_hour = (timestamp.hour // 4) * 4
                start_time = timestamp.replace(hour=start_hour, minute=0, second=0)
        elif target_timeframe == 2:
                # For 2-hour timeframe
                is_trigger = timestamp.hour % 2 == 1
                start_hour = (timestamp.hour // 2) * 2
                start_time = timestamp.replace(hour=start_hour, minute=0, second=0)
        else:
                # If the target timeframe is not supported, return False and the original timestamp
                return False, timestamp_str

        result_timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
        return is_trigger, result_timestamp


def calculate_time_difference_in_hours(start_time, end_time):
        """
        Calculates the time difference between two datetime strings or datetime objects in hours.

        Args:
                start_time (str or datetime or Timestamp): The start time
                end_time (str or datetime or Timestamp): The end time

        Returns:
                float: The time difference in hours
        """
        def parse_time(time_input):
                if isinstance(time_input, str):
                        return datetime.strptime(time_input, '%Y-%m-%d %H:%M:%S')
                elif isinstance(time_input, Timestamp):
                        return time_input.to_pydatetime()
                elif isinstance(time_input, datetime):
                        return time_input
                else:
                        raise ValueError(f"Unsupported time input type: {type(time_input)}")

        # Parse the input times
        start_time_dt = parse_time(start_time)
        end_time_dt = parse_time(end_time)

        # Calculate the time difference
        time_difference = end_time_dt - start_time_dt

        # Convert the time difference to hours
        hours_difference = time_difference.total_seconds() / 3600
        return hours_difference

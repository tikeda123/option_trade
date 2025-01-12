import sys
import os
from typing import Tuple

# Add parent directory to sys.path for module resolution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.utils import format_dates, exit_with_message
from common.constants import MARKET_DATA_TECH,AIML_TRACING,TRADING_LOG,TRANSACTION_DATA,ACCOUNT_DATA,ROLLING_AI_DATA
from mongodb.data_loader_mongo import MongoDataLoader
from trading_analysis_kit.simulation_strategy import SimulationStrategy
from trading_analysis_kit.simulation_strategy_context import SimulationStrategyContext

def parse_command_line_arguments(argv: list) -> Tuple[str, str, bool]:
        """Parses command line arguments for start and end dates.

        Args:
                argv (list): List of command line arguments.

        Returns:
                Tuple[str, str, bool]: Tuple containing formatted start_date, end_date and drop_flag.
        """
        if len(argv) < 3 or len(argv) > 4:
                exit_with_message("Usage: python simulation_ai_backtest.py <start_date> <end_date> [--drop]")

        start_date, end_date = format_dates(argv[1], argv[2])
        drop_flag = "--drop" in argv

        return start_date, end_date, drop_flag

def run_backtest(start_date: str, end_date: str, db_drop_flag=True):
        """Loads data, runs the backtest simulation, and prints results.

        Args:
                start_date (str): Start date for the backtest.
                end_date (str): End date for the backtest.
        """
        db = MongoDataLoader()


        if db_drop_flag:
                collections_to_process = [ACCOUNT_DATA, TRANSACTION_DATA, TRADING_LOG, AIML_TRACING, ROLLING_AI_DATA]
                for collection in collections_to_process:
                        if db.is_collection_exists(collection):
                                db.export_collection_to_csv(collection)
                                db.drop_collection_by_colltype(collection)


        db.load_data_from_datetime_period(start_date, end_date, coll_type=MARKET_DATA_TECH)
        dataframe = db.get_df_raw()


        #nan_check_passed = not dataframe['date'].isna().any()
        #print(f'************************************************************************************ flag:{nan_check_passed}')

        strategy_context = SimulationStrategy()
        context = SimulationStrategyContext(strategy_context)
        context.load_data_for_offline(start_date)

        for _, row in dataframe.iterrows():
                context.append_df(row.to_frame().T)
                current_index = context.dm.get_latest_index()
                context.optimize_param(current_index)
                context.event_handle(current_index)

        context.update_roll_ai_data()
        context.print_win_lose()
        context.save_simulation_result(context)

if __name__ == '__main__':
        start_date, end_date,drop_flag = parse_command_line_arguments(sys.argv)
        run_backtest(start_date, end_date,drop_flag)


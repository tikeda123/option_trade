
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


def parse_command_line_arguments(argv) -> tuple:
    from common.utils import format_dates, exit_with_message
    """
    コマンドライン引数から開始日、終了日、およびデータベースフラグを解析して返す。

    Args:
        argv (list): コマンドライン引数のリスト。

    Returns:
        tuple: (開始日, 終了日, データベースフラグ)
    """
    db_flag = '-db' in argv
    dates = [arg for arg in argv[1:] if arg != '-db']

    if len(dates) != 2:
        exit_with_message(
            "使用方法: python script.py <start_date> <end_date> [-db]")

    return *format_dates(*dates), db_flag


def main():
    from common.utils import configure_container
    from common.init_common_module import init_common_module
    start_date, end_date, db_flag = parse_command_line_arguments(sys.argv)
    #start date:(例）2024-02-01 00:00:00+0900
    #end   date:(例）2024-03-01 00:00:00+0900

    container = configure_container(name=__name__)
    init_common_module()

    data_loader_online = container.data_loader_online()
    """
    result = data_loader_online.fetch_historical_data_all(start_date, end_date)
    result = data_loader_online.load_data_from_db()
    result.to_csv('result.csv')
    print(result)


    data_loader_online.update_historical_data_to_now()
    result = data_loader_online.load_data_from_db()
    result.to_csv('result.csv')
    print(result)
    """
    #tech_result = data_loader_online.convert_historical_recent_data_to_tech()
    #tech_result.to_csv('tech_result.csv')
    #print(tech_result)
    tech_result = data_loader_online.convert_all_historical_data_to_tech()
    tech_result.to_csv('tech_result.csv')
    print(tech_result)


if __name__ == "__main__":
    main()
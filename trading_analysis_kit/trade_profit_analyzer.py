import pandas as pd
import os,sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)

from common.config_manager import ConfigManager
from common.data_loader_db import DataLoaderDB
from common.data_loader_tran import DataLoaderTransactionDB
from common.constants import *

class TradeProfitAnalyzer:
    def __init__(self):
        self.config = ConfigManager()
        self.data_loader_db = DataLoaderDB()
        self.data_loader_tran = DataLoaderTransactionDB()
        self.__symbol = self.config.get('SYMBOL')
        self.__interval = self.config.get('INTERVAL')
        self.__table_name_tech = f"{self.__symbol}_{self.__interval}_market_data_tech"
        self.__table_name_acount = f"{self.__symbol}_account"
        self.__table_name_fxtransaction = f"{self.__symbol}_fxtransaction"

    def load_data(self):
        self.df_tech =  self.data_loader_db.load_data_from_db(self.__table_name_tech)
        self.df_account =  self.data_loader_tran.read_db(self.__table_name_acount,num_rows=-1)
        self.df_fxtransaction =  self.data_loader_tran.read_db(self.__table_name_fxtransaction,num_rows=-1)

    def plot_price_and_cash(self, start, end):
        # 開始時刻と終了時刻をdatetime形式に変換
        start_time = pd.to_datetime(start)
        end_time = pd.to_datetime(end)

        # テクニカルデータの時間と価格を取得し、指定された期間でフィルタリング
        tech_time = pd.to_datetime(self.df_tech['start_at'])
        tech_price = self.df_tech.loc[(tech_time >= start_time) & (tech_time <= end_time), 'close']

        # アカウントデータの時間と現金を取得し、指定された期間でフィルタリング
        account_time = pd.to_datetime(self.df_account['date'])
        account_cash = self.df_account.loc[(account_time >= start_time) & (account_time <= end_time), ['cash_out', 'amount']]
        account_cash['total'] = account_cash['cash_out'] + account_cash['amount']

        # グラフの作成
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # 価格の折れ線グラフ
        ax1.plot(tech_time[(tech_time >= start_time) & (tech_time <= end_time)], tech_price, color='blue', label='Price')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price', color='blue')
        ax1.tick_params('y', colors='blue')

        # 現金の折れ線グラフ
        ax2 = ax1.twinx()
        ax2.plot(account_time[(account_time >= start_time) & (account_time <= end_time)], account_cash['total'], color='red', label='Cash')
        ax2.set_ylabel('Cash', color='red')
        ax2.tick_params('y', colors='red')

        # 凡例の表示
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        # x軸の日付フォーマットの設定
        date_format = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
        ax1.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()

        plt.title(f'Price and Cash Over Time ({start} to {end})')
        plt.grid(True)
        plt.show()

def main():
    trade_profit_analyzer = TradeProfitAnalyzer()
    trade_profit_analyzer.load_data()


    trade_profit_analyzer.plot_price_and_cash('2024-04-01 00:00:00', '2024-04-03 00:00:00')

if __name__ == '__main__':
    main()
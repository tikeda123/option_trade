import os
import sys
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit  # ★ 追加
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---- 追加: data_preprocessing.py をインポート ----
from data_preprocessing import clean_option_data

# ユーザー環境に合わせて import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import OPTION_HV,MARKET_DATA,OPTION_TICKER
from option_pricing import simulate_option_prices as calc_option_prices
#from option_pricing_batesmodel import simulate_option_prices as calc_option_prices
from data_preprocessing import clean_option_data

def parse_symbol(symbol: str) -> Tuple[str, str, float, str]:
    """
    Parse an option symbol string into its components.

    Args:
        symbol (str): Option symbol string in format 'TICKER-EXPIRY-STRIKE-TYPE'
                     Example: 'BTC-20240615-50000-C'

    Returns:
        Tuple[str, str, float, str]: A tuple containing:
            - ticker: The underlying asset ticker (e.g., 'BTC')
            - expiry: The option expiry date (e.g., '20240615')
            - strike: The strike price as a float (e.g., 50000.0)
            - option_type: The option type ('C' for Call or 'P' for Put)
    """
    splitted = symbol.split('-')
    ticker = splitted[0]
    expiry = splitted[1]
    strike = float(splitted[2])
    option_type = splitted[3]  # "C" or "P"
    return ticker, expiry, strike, option_type

def process_option_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Process option data by grouping it by symbol and sorting by date.

    Args:
        df (pd.DataFrame): Input DataFrame containing option data with 'symbol' and 'date' columns

    Returns:
        Dict[str, pd.DataFrame]: A dictionary where:
            - keys are option symbols
            - values are DataFrames containing the sorted data for each symbol with date as index
    """
    df['date'] = pd.to_datetime(df['date'])
    symbol_groups = {}
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        symbol_df = symbol_df.sort_values('date')
        symbol_df.set_index('date', inplace=True)
        symbol_groups[symbol] = symbol_df
    return symbol_groups

FEATURE_COLS = [
    'ask1Price',
    'bid1Price',
    'ask1Iv',
    'bid1Iv',
    'markIv',
    'underlyingPrice',
    'delta',
    'gamma',
    'vega',
    'theta',
    'openInterest',
    'markPrice'
]

FEATURE_INDICES = {col: idx for idx, col in enumerate(FEATURE_COLS)}

def simulate_option_prices(start_date: datetime, end_date: datetime, symbol) -> pd.DataFrame:

    db = MongoDataLoader()

    df_iv = db.load_data_from_datetime_period(
        start_date=start_date,
        end_date=end_date,
        coll_type=OPTION_HV,
        symbol="BTC",
        interval=7
    )
    #print(df_iv)

    df_btc = db.load_data_from_datetime_period(
        start_date=datetime(2024, 12, 18),
        end_date=datetime(2025, 1, 12),
        coll_type=MARKET_DATA,
        symbol="BTCUSDT",
        interval=60
    )
    df_btc["hv"] = df_iv["value"]

    ticker, expiry, strike, option_type = parse_symbol(symbol)

    if option_type == "P":
        option_type = "put"
    elif option_type == "C":
        option_type = "call"
    else:
        raise ValueError("Invalid option type")

    date_expiry = datetime.strptime(expiry, "%d%b%y")

    df_op = calc_option_prices(df_btc, strike_prices=[strike], expiry_date=date_expiry, risk_free_rate=0.03, option_type=option_type)
    return df_op

def real_option_prices(start_date: datetime, end_date: datetime, symbol) -> pd.DataFrame:
    db = MongoDataLoader()
    df = db.load_data_from_datetime_period_option(
        start_date=start_date,
        end_date=end_date
    )

    # ------------------------------
    # データ前処理: 0→NaN変換, IQR外れ値処理, 補間, dropna
    # ------------------------------

    df = clean_option_data(
        df,
        group_col='symbol',
        columns_to_clean=['ask1Price', 'ask1Iv', 'bid1Price', 'bid1Iv'],  # 必要に応じて追加
        outlier_factor=1.5,  # IQR factor
        dropna_after=True
    )

    # シンボルごとの時系列データを取得
    symbol_groups = process_option_data(df)
    df_op = symbol_groups[symbol]
    return df_op

def main():
    symbol = "BTC-28MAR25-90000-P"
    df_sim_op = simulate_option_prices(
        start_date=datetime(2024, 12, 17),
        end_date=datetime(2025, 1, 12),
        symbol=symbol
    )
    df_real_op = real_option_prices(
        start_date=datetime(2024, 12, 17),
        end_date=datetime(2025, 1, 12),
        symbol=symbol
    )

    print(df_sim_op)
    print(df_real_op)


    #df_sim_op.to_csv("sim_op.csv")
    #df_real_op.to_csv("real_op.csv")

    # Create time series comparison plot
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(df_sim_op['timestamp']), df_sim_op['price'], label='Simulated Price', marker='o')
    plt.plot(df_real_op.index, df_real_op['bid1Price'], label='Real Bid Price', marker='o')
    plt.plot(df_real_op.index, df_real_op['ask1Price'], label='Real Ask Price', marker='o')
    plt.title(f'DeepSeek:Stochastic Volatility Model + Jump Diffusion Model Option Price Comparison for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('option_price_comparison.png')
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()

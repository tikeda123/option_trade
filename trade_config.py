import pandas as pd

from common.config_manager import ConfigManager
from common.trading_logger_db import TradingLoggerDB
from common.constants import  TRADE_CONFIG
from mongodb.data_loader_mongo import MongoDataLoader


class TradeConfig:
        """
        Manages an FX account, supporting deposits, withdrawals, transaction records,
        and displaying account balance changes over time. Implemented as a Singleton pattern.
        """
        _instance = None
        _initialized = False

        def __new__(cls):
                if cls._instance is None:
                        cls._instance = super(TradeConfig, cls).__new__(cls)
                return cls._instance

        def __init__(self):

                if not TradeConfig._initialized:
                        self.__config_manager = ConfigManager()
                        self.__logger = TradingLoggerDB()
                        self.__data_loader = MongoDataLoader()
                        self.__update = self.__config_manager.get('TRADE_CONFIG', 'UPDATE')

                        if self.__update and self.__data_loader.is_collection_exists(TRADE_CONFIG):
                                self.__data_loader.drop_collection(TRADE_CONFIG)

                        if not self.__data_loader.is_collection_exists(TRADE_CONFIG):
                                self._init_config()
                                self._initialize_db()
                        else:
                                self._load_TRADE_CONFIG()

                        TradeConfig._initialized = True

        def _init_config(self):
                """Initializes account settings from the configuration manager."""
                config = self.__config_manager.get('TRADE_CONFIG')

                self._data = {
                        'serial': 1,
                        'contract': config['CONTRACT'],
                        'init_amount': float(config['INIT_AMOUNT']),
                        'amount': float(config['AMOUNT']),
                        'total_amount': float(config['AMOUNT']),
                        'init_equity': float(config['INIT_EQUITY']),
                        'max_leverage': float(config['MAX_LEVERAGE']),
                        'ptc': float(config['PTC']),
                        'max_losscut': float(config['MAX_LOSSCUT']),
                        'min_losscut': float(config['MIN_LOSSCUT']),
                        'symbol': config['SYMBOL'],
                        'interval': config['INTERVAL'],
                        'entry_enabled': config['ENTRY_ENABLED'],
                        'short_model_ratio': float(config['SHORT_MODEL_RATIO']),
                        'middle_model_ratio': float(config['MIDDLE_MODEL_RATIO']),
                        'long_model_ratio': float(config['LONG_MODEL_RATIO'])
                }

        def _initialize_db(self):
                """Initializes the database with the account information."""
                df = pd.DataFrame([self._data])
                self.__data_loader.insert_data(df, coll_type=TRADE_CONFIG)

        def _load_TRADE_CONFIG(self):
                """Loads account information from the database."""
                df = self.__data_loader.load_data(TRADE_CONFIG)
                self._data = df.iloc[0].to_dict()

        def _update_TRADE_CONFIG(self, updated_fields: dict):
                """Updates the account information in both the internal state and the database.

                Args:
                        updated_fields (dict): Fields to update with their new values.
                """
                self._data.update(updated_fields)
                df = pd.DataFrame([self._data])
                self.__data_loader.update_data_by_serial(1, df, coll_type=TRADE_CONFIG)

        @property
        def short_model_ratio(self) -> float:
                """The ratio of the short model."""
                self._load_TRADE_CONFIG()
                return self._data['short_model_ratio']

        @short_model_ratio.setter
        def short_model_ratio(self, value: float) -> None:
                self._update_TRADE_CONFIG({'short_model_ratio': value})

        @property
        def middle_model_ratio(self) -> float:
                """The ratio of the middle model."""
                self._load_TRADE_CONFIG()
                return self._data['middle_model_ratio']

        @middle_model_ratio.setter
        def middle_model_ratio(self, value: float) -> None:
                self._update_TRADE_CONFIG({'middle_model_ratio': value})

        @property
        def long_model_ratio(self) -> float:
                """The ratio of the long model."""
                self._load_TRADE_CONFIG()
                return self._data['long_model_ratio']

        @long_model_ratio.setter
        def long_model_ratio(self, value: float) -> None:
                self._update_TRADE_CONFIG({'long_model_ratio': value})

        @property
        def entry_enabled(self) -> bool:
                """Whether to enable entry signals."""
                self._load_TRADE_CONFIG()
                return self._data['entry_enabled']

        @entry_enabled.setter
        def entry_enabled(self, value: bool) -> None:
                self._update_TRADE_CONFIG({'entry_enabled': value})

        @property
        def interval(self) -> str:
                """The time interval for trading."""
                self._load_TRADE_CONFIG()
                return self._data['interval']

        @interval.setter
        def interval(self, value: str) -> None:
                self._update_TRADE_CONFIG({'interval': value})

        @property
        def contract(self) -> str:
                """The type of trading contract."""
                self._load_TRADE_CONFIG()
                return self._data['contract']

        @contract.setter
        def contract(self, value: str) -> None:
                self._update_TRADE_CONFIG({'contract': value})

        @property
        def init_amount(self) -> float:
                """The initial account balance."""
                self._load_TRADE_CONFIG()
                return self._data['init_amount']

        @init_amount.setter
        def init_amount(self, value: float) -> None:
                self._update_TRADE_CONFIG({'init_amount': value})

        @property
        def amount(self) -> float:
                """The current account balance."""
                self._load_TRADE_CONFIG()
                return self._data['amount']

        @amount.setter
        def amount(self, value: float) -> None:
                self._update_TRADE_CONFIG({'amount': value})

        @property
        def total_amount(self) -> float:
                """The total amount including unrealized P/L."""
                self._load_TRADE_CONFIG()
                return self._data['total_amount']

        @total_amount.setter
        def total_amount(self, value: float) -> None:
                self._update_TRADE_CONFIG({'total_amount': value})

        @property
        def init_equity(self) -> float:
                """The initial equity for position sizing."""
                self._load_TRADE_CONFIG()
                return self._data['init_equity']

        @init_equity.setter
        def init_equity(self, value: float) -> None:
                self._update_TRADE_CONFIG({'init_equity': value})

        @property
        def max_leverage(self) -> float:
                """The maximum allowed leverage for trades."""
                self._load_TRADE_CONFIG()
                return self._data['max_leverage']

        @max_leverage.setter
        def max_leverage(self, value: float) -> None:
                self._update_TRADE_CONFIG({'max_leverage': value})

        @property
        def ptc(self) -> float:
                """The percentage of transaction costs."""
                self._load_TRADE_CONFIG()
                return self._data['ptc']

        @ptc.setter
        def ptc(self, value: float) -> None:
                self._update_TRADE_CONFIG({'ptc': value})

        @property
        def max_losscut(self) -> float:
                """The maximum loss cut threshold."""
                self._load_TRADE_CONFIG()
                return self._data['max_losscut']

        @max_losscut.setter
        def max_losscut(self, value: float) -> None:
                self._update_TRADE_CONFIG({'max_losscut': value})

        @property
        def min_losscut(self) -> float:
                """The minimum loss cut threshold."""
                self._load_TRADE_CONFIG()
                return self._data['min_losscut']

        @min_losscut.setter
        def min_losscut(self, value: float) -> None:
                self._update_TRADE_CONFIG({'min_losscut': value})

        @property
        def symbol(self) -> str:
                """The trading symbol."""
                self._load_TRADE_CONFIG()
                return self._data['symbol']

        @symbol.setter
        def symbol(self, value: str) -> None:
                self._update_TRADE_CONFIG({'symbol': value})

trade_config = TradeConfig()

def main():
        print(trade_config.contract)
        print(trade_config.init_amount)
        print(trade_config.amount)
        print(trade_config.init_equity)
        print(trade_config.max_leverage)
        print(trade_config.ptc)
        print(trade_config.max_losscut)
        print(trade_config.min_losscut)

        trade_config.init_equity = 400
        print(TRADE_CONFIG.init_equity)


if __name__ == "__main__":
        main()

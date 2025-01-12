import os
import sys
from typing import Tuple, List, Dict, Any


# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the path of the parent directory to sys.path
sys.path.append(parent_dir)

from common.trading_logger import TradingLogger
from mongodb.data_loader_mongo import MongoDataLoader
from common.utils import get_config
from common.constants import *
from trade_config import trade_config

class ModelParam:
    TIME_SERIES_PERIOD = TIME_SERIES_PERIOD  # Length of the time series sequence used for prediction
    LSTM_PARAM_LEARNING_RATE = 0.0005  # Learning rate for the optimizer
    #ROLLING_PARAM_LEARNING_RATE = 0.00005  # Learning rate for the optimizer
    ROLLING_PARAM_LEARNING_RATE = 0.00002  # Learning rate for the optimizer
    #ROLLING_PARAM_LEARNING_RATE = 0.0003  # Learning rate for the optimizer
    #PARAM_LEARNING_RATE = 0.00001 # Learning rate for the optimizer, transfer learning
    PARAM_EPOCHS = 200  # Number of epochs for training
    N_SPLITS = 3  # Number of splits for cross-validation
    BATCH_SIZE = 128 # Batch size for training
    POSITIVE_THRESHOLD = 0  # Threshold for determining a positive prediction


class BaseModel:
    """
    Base class for prediction models.

    Provides common attributes and methods for different prediction models.
    """
    def __init__(
        self,
        id: str,
        config: Dict[str, Any],
        data_loader: MongoDataLoader,
        logger: TradingLogger,
        symbol: str = None,
        interval: str = None,
    ):
        """
        Initializes the BaseModel.

        Args:
            id (str): Model ID.
            data_loader (DataLoader): Data loader instance.
            logger (TradingLogger): Logger instance.
            symbol (str, optional): Symbol name. Defaults to None.
            interval (str, optional): Data interval. Defaults to None.
        """
        self.id = id
        self.symbol = symbol or trade_config.symbol
        self.interval = interval or trade_config.interval
        self.config = config
        self.model_param = ModelParam
        self.data_loader = data_loader
        self.logger = logger


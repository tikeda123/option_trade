import os
import sys
from abc import ABC, abstractmethod
from typing import Any, Tuple
import joblib
import tensorflow as tf

import numpy as np
import pandas as pd

# Set path to the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from aiml.transformerblock import TransformerBlock

class ModelManager:
        _instance = None
        _models = {}  # ロード済みモデルを格納する辞書

        def __new__(cls):
                if cls._instance is None:
                        cls._instance = super().__new__(cls)
                return cls._instance

        @staticmethod
        def load_model(filename, datapath, logger):  # model_id を削除
                model_path = os.path.join(datapath, filename + ".keras")
                if model_path in ModelManager._models:
                            logger.log_system_message(f"Loading model from cache: {model_path}")
                            return ModelManager._models[model_path]

                logger.log_system_message(f"Loading model from file: {model_path}")
                model = tf.keras.models.load_model(
                            model_path, custom_objects={"TransformerBlock": TransformerBlock}
                )

                model_scaler_file = filename + ".scaler"
                model_scaler_path = os.path.join(datapath, model_scaler_file)
                logger.log_system_message(f"Loading scaler from {model_scaler_path}")
                scaler = joblib.load(model_scaler_path)
                model.scaler = scaler

                ModelManager._models[model_path] = model
                return model

class PredictionModel(ABC):
    """
    Abstract class for prediction models.
    """
    @abstractmethod
    def load_and_prepare_data(
        self, start_datetime: str, end_datetime: str, test_size: float = 0.5, random_state: int = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        pass

    @abstractmethod
    def train_with_cross_validation(self, x_data: np.ndarray, y_data: np.ndarray) -> Any:
        pass

    @abstractmethod
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, str, np.ndarray]:
        pass

    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_single(self, data_point: np.ndarray) -> int:
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def get_data_loader(self):
        pass

    @abstractmethod
    def get_feature_columns(self) -> list:
        pass

    @abstractmethod
    def get_data_period(self, date: str, period: int) -> np.ndarray:
        pass
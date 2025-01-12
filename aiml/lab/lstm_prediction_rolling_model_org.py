import os
import sys
from typing import Tuple
import numpy as np
import tensorflow as tf
import pandas as pd
from typing import Dict, Any, Optional

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (
    LSTM,
    Conv1D,
    MaxPooling1D,
    Dense,
    Dropout,
    Input,
    LayerNormalization,
    GlobalAveragePooling1D,
)

import joblib

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

from aiml.prediction_model import PredictionModel
from aiml.model_param import ModelParam, BaseModel
from aiml.aiml_comm import process_timestamp_and_cyclical_features


class LSTMPredictionRollingModel(BaseModel, PredictionModel):
    """
    Model class that performs rolling prediction using CNN and LSTM.

    Args:
        id (str): Model ID.
        data_loader (DataLoader): Data loader instance.
        logger (TradingLogger): Logger instance.
        symbol (str, optional): Symbol name. Defaults to None.
        interval (str, optional): Data interval. Defaults to None.

    Attributes:
        logger (TradingLogger): Instance for handling log information.
        data_loader (MongoDataLoader): Instance for handling data loading.
        config (Dict[str, Any]): Dictionary of model configuration values.
        datapath (str): Data path.
        feature_columns (list): Feature columns.
        symbol (str): Symbol name.
        interval (str): Data interval.
        filename (str): File name.
        target_column (str): Target column.
        scaler (StandardScaler): Instance used for scaling.
        table_name (str): Data table name.
        model (tf.keras.Model): CNN + LSTM model.
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
        Initializes the LSTMPredictionRollingModel.

        Args:
            id (str): Model ID.
            data_loader (DataLoader): Data loader instance.
            logger (TradingLogger): Logger instance.
            symbol (str, optional): Symbol name. Defaults to None.
            interval (str, optional): Data interval. Defaults to None.
        """
        super().__init__(id, config, data_loader, logger, symbol, interval)
        self._initialize_attributes()

    def _initialize_attributes(self):
        """
        Initializes the attributes of the model.
        """
        self.datapath = f"{parent_dir}/{self.config['DATAPATH']}"
        self.feature_columns = self.config["FEATURE_COLUMNS"]
        self.target_column = self.config["TARGET_COLUMN"]
        self.prediction_distance = self.config["PREDICTION_DISTANCE"]
        self.filename = self.config["MODLE_FILENAME"]
        self.scaler = StandardScaler()
        self.table_name = f"{self.symbol}_{self.interval}"
        self.model = None

    def set_parameters(
        self,
        time_series_period: Optional[int] = None,
        param_learning_rate: Optional[float] = None,
        param_epochs: Optional[int] = None,
        n_splits: Optional[int] = None,
        batch_size: Optional[int] = None,
        positive_threshold: Optional[float] = None,
    ) -> None:
        """
        Sets the model parameters.

        Args:
            time_series_period (Optional[int]): Period of time series data.
            param_learning_rate (Optional[float]): Learning rate.
            param_epochs (Optional[int]): Number of epochs.
            n_splits (Optional[int]): Number of splits for cross-validation.
            batch_size (Optional[int]): Batch size.
            positive_threshold (Optional[float]): Positive threshold.

        Raises:
            ValueError: If an invalid parameter value is specified.
        """
        if time_series_period is not None:
            if time_series_period <= 3:
                raise ValueError("time_series_period must be more than 3")
            self.model_param.TIME_SERIES_PERIOD = time_series_period

        if param_learning_rate is not None:
            if param_learning_rate <= 0:
                raise ValueError("param_learning_rate must be positive")
            self.model_param.LSTM_PARAM_LEARNING_RATE = param_learning_rate

        if param_epochs is not None:
            if param_epochs <= 0:
                raise ValueError("param_epochs must be positive")
            self.model_param.PARAM_EPOCHS = param_epochs

        if n_splits is not None:
            if n_splits <= 1:
                raise ValueError("n_splits must be greater than 1")
            self.model_param.N_SPLITS = n_splits

        if batch_size is not None:
            if batch_size <= 0:
                raise ValueError("batch_size must be positive")
            self.model_param.BATCH_SIZE = batch_size

        if positive_threshold is not None:
            self.model_param.POSITIVE_THRESHOLD = positive_threshold

        self.logger.log_system_message("Model parameters updated successfully")

    def get_data_loader(self) -> MongoDataLoader:
        """
        Gets the data loader.

        Returns:
            MongoDataLoader: Data loader instance.
        """
        return self.data_loader

    def get_feature_columns(self) -> list:
        """
        Gets the feature columns used.

        Returns:
            list: List of feature columns.
        """
        return self.feature_columns

    def create_table_name(self) -> str:
        """
        Creates the table name.

        Returns:
            str: Created table name.
        """
        self.table_name = f"{self.symbol}_{self.interval}_market_data_tech"
        return self.table_name

    def load_and_prepare_data(
        self,
        start_datetime,
        end_datetime,
        coll_type,
        test_size=0.5,
        random_state=None,
    ):
        """
        Loads and prepares data from the database for training or evaluation.

        Args:
            start_datetime (str): Start date and time of the data.
            end_datetime (str): End date and time of the data.
            coll_type (str): Collection type to load data from.
            test_size (float, optional): Proportion of data to use for testing. Defaults to 0.5.
            random_state (int, optional): Random seed for data splitting. Defaults to None.

        Returns:
            tuple: A tuple containing the training and testing data.
        """
        data = self.data_loader.load_data_from_datetime_period(
            start_datetime, end_datetime, coll_type
        )
        # Prepare sequences and ensure correct shapes
        scaled_sequences, targets = self._prepare_sequences(data)
        return train_test_split(
            scaled_sequences,
            targets,
            test_size=test_size,
            random_state=random_state,
            shuffle=False,
        )

    def load_and_prepare_data_mix(
        self,
        start_datetime: str,
        end_datetime: str,
        coll_type: str,
        test_size=0.2,
        random_state=None,
    ):
        """
        Loads and prepares data from multiple collections for training or evaluation.

        Args:
            start_datetime (str): Start date and time of the data.
            end_datetime (str): End date and time of the data.
            coll_type (str): List of collection types to load data from.
            test_size (float, optional): Proportion of data to use for testing. Defaults to 0.2.
            random_state (int, optional): Random seed for data splitting. Defaults to None.

        Returns:
            tuple: A tuple containing the training and testing data.
        """
        from aiml.aiml_comm import load_data

        data = load_data(self.data_loader, start_datetime, end_datetime, coll_type)
        x, y = self._prepare_sequences(data)
        return train_test_split(
            x, y, test_size=test_size, random_state=random_state, shuffle=False
        )

    def _prepare_sequences(self, data):
        """
        Prepares the data for training or evaluation.

        Scales the data and creates sequences of time series data with corresponding targets.

        Args:
            data (pd.DataFrame): Raw data to prepare.

        Returns:
            tuple: A tuple containing the scaled sequences and target values.
        """
        time_period = self.model_param.TIME_SERIES_PERIOD  # Get the time series period from model parameters
        forecast_horizon = self.prediction_distance  # Forecast horizon for the target value

        # Convert datetime strings to unix timestamps only if 'date' is in feature columns
        if 'date' in self.feature_columns:
            data = process_timestamp_and_cyclical_features(data)

        def sequence_generator():
            for i in range(len(data) - (time_period + forecast_horizon)):  # Iterate over the data, leaving enough space for sequences
                # Extract a sequence of time series data
                sequence = data.iloc[
                    i: i + time_period, data.columns.get_indexer(self.feature_columns)
                ].values
                # Determine the target value based on the price change
                # in the future time step
                target = int(
                    data.iloc[i + time_period + forecast_horizon - 1][self.target_column[1]]
                    > data.iloc[i + time_period - 1][self.target_column[0]]
                )
                yield sequence, target  # Yield the sequence and target

        sequences = []  # Initialize an empty list to store sequences
        targets = []  # Initialize an empty list to store target values

        # Generate sequences and targets using the sequence generator
        for seq, target in sequence_generator():
            scaled_seq = self.scaler.fit_transform(seq)  # Scale the sequence data
            sequences.append(scaled_seq)  # Append the scaled sequence to the list
            targets.append(target)  # Append the target value to the list

        sequences = np.array(sequences)  # Convert the list of sequences to a NumPy array
        targets = np.array(targets)  # Convert the list of target values to a NumPy array
        return sequences, targets  # Return the sequences and targets

    def create_cnn_lstm_model(
    self,
    input_shape,
    lstm_units=512,  # ユニット数を増加
    dropout_rate=0.3,  # ドロップアウト率を調整
    l2_reg=0.001,  # L2正則化を減少
    num_filters=256,  # フィルター数を増加
    kernel_size=3,
    ):
        inputs = Input(shape=input_shape)

    # Conv1Dレイヤーを増やす
        x = Conv1D(
        filters=num_filters,
        kernel_size=kernel_size,
        activation='relu',
        padding='same',
        name='conv1d_1',
        )(inputs)
        x = Conv1D(
        filters=num_filters,
        kernel_size=kernel_size,
        activation='relu',
        padding='same',
        name='conv1d_2',
        )(x)
        x = MaxPooling1D(pool_size=2, name='maxpool1d')(x)
        x = Dropout(dropout_rate, name='dropout_cnn')(x)

    # LSTMレイヤーを増やす
        x = LSTM(
        lstm_units,
        return_sequences=True,
        kernel_regularizer=l2(l2_reg),
        name='lstm_1',
        )(x)
        x = LSTM(
        lstm_units,
        return_sequences=True,
        kernel_regularizer=l2(l2_reg),
        name='lstm_2',
        )(x)
        x = Dropout(dropout_rate, name='dropout_lstm')(x)
        x = LayerNormalization(epsilon=1e-6, name='layer_norm')(x)
        x = GlobalAveragePooling1D(name='global_avg_pool')(x)
        x = Dropout(dropout_rate, name='dropout_final')(x)
        outputs = Dense(
            1, activation="sigmoid", kernel_regularizer=l2(l2_reg), name='output'
        )(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def train(self, x_train, y_train):
        """
        Trains the model.

        Args:
            x_train (np.array): Training data.
            y_train (np.array): Labels for training data.
        """
        # Model training
        epochs = self.model_param.PARAM_EPOCHS
        batch_size = self.model_param.BATCH_SIZE
        param_learning_rate = self.model_param.PARAM_LEARNING_RATE

        self.model = self.create_cnn_lstm_model(
            (x_train.shape[1], x_train.shape[2])
        )

        self.model.compile(
            optimizer=Adam(learning_rate=param_learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    def train_with_cross_validation(
        self, data: np.ndarray, targets: np.ndarray
    ) -> list:
        """
        Trains the model using K-Fold cross-validation.

        Args:
            data (np.ndarray): Dataset used for training.
            targets (np.ndarray): Target values corresponding to the dataset.

        Returns:
            list: Model performance evaluation results for the test data in each split.
        """
        # Initialize K-Fold cross-validation
        epochs = self.model_param.PARAM_EPOCHS
        batch_size = self.model_param.BATCH_SIZE
        n_splits = self.model_param.N_SPLITS
        param_learning_rate = self.model_param.LSTM_PARAM_LEARNING_RATE

        kfold = KFold(n_splits=n_splits, shuffle=True)

        # List to record the score in each fold
        fold_no = 1
        scores = []

        for train, test in kfold.split(data, targets):
            # Generate the model
            self.model = self.create_cnn_lstm_model(
                (data.shape[1], data.shape[2])
            )

            # Compile the model
            self.model.compile(
                optimizer=Adam(learning_rate=param_learning_rate),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            # Train the model
            self.logger.log_system_message(f"Training for fold {fold_no} ...")
            self.model.fit(
                data[train], targets[train], epochs=epochs, batch_size=batch_size
            )

            # Evaluate the model's performance
            scores.append(self.model.evaluate(data[test], targets[test], verbose=0))

            fold_no += 1

        return scores

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, str, np.ndarray]:
        """
        Evaluates the model on the test dataset.

        Args:
            x_test (np.ndarray): Test dataset.
            y_test (np.ndarray): Correct labels for the test dataset.

        Returns:
            Tuple[float, str, np.ndarray]: Model accuracy, classification report, confusion matrix.
        """
        # Model evaluation
        print(f"evaluate - x_test.shape: {x_test.shape}")  # Debug output
        y_pred = (self.model.predict(x_test) > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        return accuracy, report, conf_matrix

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Makes predictions for the specified data.

        Args:
            data (np.ndarray): Data to predict for.

        Returns:
            np.ndarray: Prediction results.
        """
        return self.model.predict(data)

    def predict_single(self, data_point: np.ndarray) -> int:
        """
        Makes a prediction for a single data point.

        Args:
            data_point (np.ndarray): Data point to predict for.

        Returns:
            int: Predicted class label.
        """
        prediction = self.predict_single_res(data_point)
        prediction = (prediction > 0.5).astype(int)
        return prediction

    def predict_single_res(self, data_point: np.ndarray) -> int:
        """
        Makes a prediction for a single data point.

        Args:
            data_point (np.ndarray): Data point to predict for.

        Returns:
            int: Predicted class label.
        """
        scaled_data_point = self.scaler.transform(data_point)
        # Prediction by the model
        prediction = self.model.predict(
            scaled_data_point.reshape(1, -1, len(self.feature_columns))
        )
        print(prediction)
        return prediction[0][0]

    def save_model(self, filename=None):
        """
        Saves the trained model and scaler to files.
        """
        if filename is not None:
            self.filename = filename

        # Save the model
        model_file_name = self.filename + ".keras"
        model_path = os.path.join(self.datapath, model_file_name)
        self.logger.log_system_message(f"Saving model to {model_path}")
        self.model.save(model_path)

        # Save the scaler
        model_scaler_file = self.filename + ".scaler"
        model_scaler_path = os.path.join(self.datapath, model_scaler_file)
        self.logger.log_system_message(f"Saving scaler to {model_scaler_path}")
        joblib.dump(self.scaler, model_scaler_path)

    def load_model(self, filename=None):
        """
        Loads the saved model and scaler.
        """
        if filename is not None:
            self.filename = filename

        model_file_name = self.filename + ".keras"
        model_path = os.path.join(self.datapath, model_file_name)
        self.logger.log_system_message(f"Loading model from {model_path}")
        self.model = tf.keras.models.load_model(model_path)

        # Load the scaler
        model_scaler_file = self.filename + ".scaler"
        model_scaler_path = os.path.join(self.datapath, model_scaler_file)
        self.logger.log_system_message(f"Loading scaler from {model_scaler_path}")
        self.scaler = joblib.load(model_scaler_path)

    def get_data_period(self, date: str, period: int) -> np.ndarray:
        """
        Gets data for the specified period.

        Args:
            date (str): Start date and time of the period.
            period (int): Length of the period.

        Returns:
            np.ndarray: Data for the specified period.
        """
        data = self.data_loader.load_data_from_datetime_period(
            date, period, self.table_name
        )
        return data.filter(items=self.feature_columns).to_numpy()

    def compute_gradcam(self, data, layer_name="lstm", pred_index=None):
        """
        Computes Grad-CAM heatmap for the given data.

        Args:
            data (np.ndarray): Input data to compute Grad-CAM for.
            layer_name (str, optional): Name of the convolutional layer to use. Defaults to "lstm".
            pred_index (int, optional): Index of the prediction to visualize. Defaults to None (the most likely prediction).

        Returns:
            np.ndarray: Grad-CAM heatmap.
        """
        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [self.model.get_layer(layer_name).output, self.model.output],
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(data)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def display_gradcam(self, data, heatmap, alpha=0.4):
        plt.figure(figsize=(10, 6))
        plt.plot(data.flatten(), label='Input Data')
        plt.imshow(np.expand_dims(heatmap, axis=0), aspect='auto', cmap='viridis', alpha=alpha)
        plt.colorbar(label='Importance')
        plt.xlabel('Time Step')
        plt.ylabel('Feature Index')
        plt.title('Grad-CAM Visualization')
        plt.legend()
        plt.show()


def main():
    from aiml.aiml_comm import COLLECTIONS_TECH
    from common.utils import get_config_model

    data_loader = MongoDataLoader()
    logger = TradingLogger()
    config = get_config_model("MODEL_SHORT_TERM", "lstm_v4")  # Get the model configuration

    model = LSTMPredictionRollingModel(
        id="lstm_v4", config=config, data_loader=data_loader, logger=logger
    )

    """
    # Load and prepare data for training and testing
    x_train, x_test, y_train, y_test = model.load_and_prepare_data_mix(
        "2020-01-01 00:00:00", "2024-04-01 00:00:00", COLLECTIONS_TECH
    )

    # Train the model with cross-validation
    cv_scores = model.train_with_cross_validation(
        np.concatenate((x_train, x_test), axis=0),
        np.concatenate((y_train, y_test), axis=0),
    )

    # Display cross-validation results
    for i, score in enumerate(cv_scores):
        print(f"Fold {i+1}: Accuracy = {score[1]}")

    # Evaluate the model
    accuracy, report, conf_matrix = model.evaluate(x_test, y_test)
    print(f"Accuracy: {accuracy}")
    print(report)
    print(conf_matrix)

    # Save the trained model
    model.save_model()
    """
    model.load_model()

    # Load and prepare data for final evaluation
    x_train, x_test, y_train, y_test = model.load_and_prepare_data(
        "2024-04-01 00:00:00",
        "2024-11-01 00:00:00",
        MARKET_DATA_TECH,
        test_size=0.9,
        random_state=None,
    )

    # Evaluate the model on the final test set
    accuracy, report, conf_matrix = model.evaluate(x_test, y_test)
    print(f"Accuracy: {accuracy}")
    print(report)
    print(conf_matrix)


if __name__ == "__main__":
    main()

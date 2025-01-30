import os
import sys
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import BayesianRidge
import tensorflow as tf  # Unused import

import joblib


# --------------------------------------------------------------------------
# Set up paths and environment
# --------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)  # Add the parent directory to the Python path for local imports

# --------------------------------------------------------------------------
# Local imports (adjust as necessary for your environment)
# --------------------------------------------------------------------------
from mongodb.data_loader_mongo import MongoDataLoader  # Import for MongoDB data loading
from aiml.model_param import BaseModel  # Import for base model class
from common.trading_logger import TradingLogger  # Import for custom logging
from common.constants import MARKET_DATA_TECH  # Import for constants


# --------------------------------------------------------------------------
# BayesianPredictionModel class
# --------------------------------------------------------------------------
class BayesianPredictionModel(BaseModel):
    """
    A price prediction model using BayesianRidge regression.

    This class provides a wrapper for preprocessing, training, and inference,
    allowing flexibility in choosing the target variable (target_col) and
    the prediction horizon (shift_steps).
    """

    def __init__(
        self,
        id: str,  # Model identifier
        config: Dict[str, Any],  # Model configuration dictionary
        data_loader: MongoDataLoader,  # Data loader object
        logger: TradingLogger,  # Logger object
        symbol: str = None,  # Trading symbol (e.g., BTCUSDT)
        interval: str = None,  # Time interval (e.g., 1440 for daily)
        use_gpu: bool = True,  # Whether to use GPU (currently not utilized)
    ):
        super().__init__(id, config, data_loader, logger, symbol, interval)
        self._initialize_attributes()
        self._configure_gpu(use_gpu)  # Configure GPU usage (if applicable)

    def _initialize_attributes(self) -> None:
        """
        Initializes model attributes.
        """
        self.datapath = os.path.join(PARENT_DIR, self.config["DATAPATH"])  # Path to store model data
        self.feature_columns = self.config["FEATURE_COLUMNS"]  # List of feature column names
        self.target_column = self.config["TARGET_COLUMN"][0]  # Target column name for prediction
        self.shift_steps = self.config["PREDICTION_DISTANCE"]  # Number of steps to predict ahead
        self.filename = self.config["MODLE_FILENAME"]  # Filename for saving/loading the model
        self.table_name = f"{self.symbol}_{self.interval}"  # MongoDB collection name
        self.all_data: Optional[pd.DataFrame] = None  # Stores the entire loaded dataset

        # Scaler and model objects
        self.scaler = StandardScaler()  # StandardScaler for feature scaling
        self.model: Optional[BayesianRidge] = None  # BayesianRidge model instance

        # Default model parameters
        default_params = {
            "max_iter": 300,  # Maximum iterations for solver
            "tol": 1e-6,  # Convergence tolerance
            "alpha_1": 1e-6,  # Hyperparameter for the Gamma prior on the precision
            "alpha_2": 1e-6,
            "lambda_1": 1e-6,
            "lambda_2": 1e-6,
        }
        self.model_params = default_params  # Store model parameters

        # Data storage for training and testing
        self.X_train_scaled: Optional[np.ndarray] = None  # Scaled training features
        self.y_train: Optional[pd.Series] = None  # Training target values
        self.X_test_scaled: Optional[np.ndarray] = None  # Scaled test features
        self.y_test: Optional[pd.Series] = None  # Test target values
        self.train_df: Optional[pd.DataFrame] = None  # Training dataframe
        self.test_df: Optional[pd.DataFrame] = None  # Test dataframe

    # ----------------------------------------------------------------------
    # Public Setter / Getter
    # ----------------------------------------------------------------------
    def set_parameters(
        self,
        default_params: Optional[Dict[str, Any]] = None, # Dictionary of model parameters
        param_epochs: Optional[int] = None,  # Not used in this model (remove?)
        n_splits: Optional[int] = None,  # Number of splits for cross-validation
        shift_steps: Optional[int] = None,  # Prediction horizon
    ) -> None:
        """
        Updates model parameters.
        """
        if default_params is not None:
            self.model_params.update(default_params)  # Update model parameters with provided values

        if shift_steps is not None:
            if shift_steps <= 0:
                raise ValueError("shift_steps must be positive.")
            self.shift_steps = shift_steps  # Update prediction horizon

        # The following parameters are not used in this model:
        # if param_epochs is not None: ...
        # if n_splits is not None: ...
        # Consider removing them for clarity

        self.logger.log_system_message("Model parameters updated successfully.")  # Log the update


    def get_data_loader(self) -> MongoDataLoader:
        """Returns the DataLoader instance."""
        return self.data_loader

    def get_feature_columns(self) -> list:
        """Returns the list of feature columns used by the model."""
        return self.feature_columns

    def create_table_name(self) -> str:
        """
        Creates or updates and returns the table name.
        """
        self.table_name = f"{self.symbol}_{self.interval}_market_data_tech" # update table name
        return self.table_name

    # ----------------------------------------------------------------------
    # Data Preprocessing
    # ----------------------------------------------------------------------
    def preprocess_data(
        self,
        train_end_date: str, # End date for the training data
        test_start_date: str, # Start date for the test data
        date_col: str = "date" # Name of the date column
    ) -> None:
        """
        Loads data, splits it into training and test sets, and performs preprocessing.

        1. Converts the date column to datetime and sorts the data.
        2. Splits the data into training and test sets based on the provided dates.
        3. Creates a shifted target column for predictions.
        4. Removes rows with NaN values resulting from the shift.
        5. Fits the StandardScaler on the training data and transforms it.
        6. Transforms the test data using the same fitted scaler.
        """
        df = self._load_and_sort_data(date_col)
        self.all_data = df.copy() # keep original data


        # (2) Split data into training and test sets
        train_df = df[df[date_col] < train_end_date].copy()
        test_df = df[df[date_col] >= test_start_date].copy()

        # (3) Create shifted target column
        label_col = f"{self.target_column}_shifted"
        train_df[label_col] = train_df[self.target_column].shift(-self.shift_steps) # create shifted target data
        test_df[label_col] = test_df[self.target_column].shift(-self.shift_steps) # create shifted target data

        # (4) Remove rows with NaN values in the target
        train_df.dropna(subset=[label_col], inplace=True)
        test_df.dropna(subset=[label_col], inplace=True)

        # (5) Preprocess training data
        if not train_df.empty: # check if the training data is empty
            X_train = train_df[self.feature_columns] # set features
            y_train = train_df[label_col] # set target
            self.X_train_scaled = self.scaler.fit_transform(X_train) # normalize data
            self.y_train = y_train # set target values
            self.train_df = train_df # set dataframe
        else:
            self._reset_train_data() # reset data

        # (6) Preprocess test data
        if not test_df.empty and self.X_train_scaled is not None: # check if the test data is empty and if the training data is set
            X_test = test_df[self.feature_columns] # set features
            y_test = test_df[label_col] # set target
            self.X_test_scaled = self.scaler.transform(X_test) # normalize data using the same scaler
            self.y_test = y_test # set target values
            self.test_df = test_df # set dataframe
        else:
            self._reset_test_data() # reset data

    def _load_and_sort_data(self, date_col: str) -> pd.DataFrame:
        """Loads data from MongoDB, sorts it by date, and returns it as a Pandas DataFrame."""
        df = self.data_loader.load_data( # load data from mongoDB
            coll_type=MARKET_DATA_TECH,
            symbol=self.symbol,
            interval=self.interval,
        )
        df[date_col] = pd.to_datetime(df[date_col]) # convert data type to datetime
        df.sort_values(by=date_col, inplace=True) # sort data
        return df

    def _reset_train_data(self) -> None:
        """Resets training data attributes to None."""
        self.X_train_scaled = None # scaled training data
        self.y_train = None # training labels
        self.train_df = None # raw train dataframe

    def _reset_test_data(self) -> None:
        """Resets test data attributes to None."""
        self.X_test_scaled = None # scaled test data
        self.y_test = None # test labels
        self.test_df = None # raw test dataframe

    # ----------------------------------------------------------------------
    # Training and Validation
    # ----------------------------------------------------------------------
    def cross_validate(
        self,
        n_splits: int = 5, # Number of splits for cross-validation
        plot_last_fold: bool = False  # Option to plot the last fold's predictions (Not implemented)
    ) -> Dict[str, float]:
        """
        Performs time series cross-validation.

        Args:
            n_splits (int): The number of splits for cross-validation. Defaults to 5.
            plot_last_fold (bool): Whether to plot predictions for the last fold.
                                    Not implemented yet. Defaults to False.

        Returns:
            Dict[str, float]: A dictionary containing the average MSE, MAE, and R2 scores
                              across all folds.
        """

        if self.X_train_scaled is None or self.y_train is None: # check if train data is set
            self.logger.log_system_message("[WARN] No training data for cross-validation.")
            return {"mse": np.nan, "mae": np.nan, "r2": np.nan} # if not set then return nan values

        tscv = TimeSeriesSplit(n_splits=n_splits) # generate index for time series cross validation
        mse_list, mae_list, r2_list = [], [], [] # define list

        for fold, (cv_train_idx, cv_val_idx) in enumerate(tscv.split(self.X_train_scaled)):
            X_cv_train = self.X_train_scaled[cv_train_idx]  # Training data for current fold
            X_cv_val = self.X_train_scaled[cv_val_idx]  # Validation data for current fold
            y_cv_train = self.y_train.values[cv_train_idx] # training labels for current fold
            y_cv_val = self.y_train.values[cv_val_idx] # validation labels for current fold

            model_cv = BayesianRidge(**self.model_params) # set model
            model_cv.fit(X_cv_train, y_cv_train) # fit model

            y_cv_pred = model_cv.predict(X_cv_val) # predict using validation data

            # Calculate metrics and store them in lists
            mse_list.append(mean_squared_error(y_cv_val, y_cv_pred))
            mae_list.append(mean_absolute_error(y_cv_val, y_cv_pred))
            r2_list.append(r2_score(y_cv_val, y_cv_pred))


        result = { # set result
            "mse": float(np.mean(mse_list)), # avarage mean square error
            "mae": float(np.mean(mae_list)), # avarage mean absolute error
            "r2": float(np.mean(r2_list)), # avarage r2 score
        }
        return result # return result as dictionary

    def train_final_model(self) -> None:
        """Trains the final model on the entire training dataset."""
        if self.X_train_scaled is None or self.y_train is None:
            self.logger.log_system_message("[WARN] No training data. Cannot train final model.")
            return

        self.model = BayesianRidge(**self.model_params) # initiate model
        self.model.fit(self.X_train_scaled, self.y_train.values) # train model using whole train dataset
        self.logger.log_system_message("Final model trained successfully.")

    # ----------------------------------------------------------------------
    # Inference and Evaluation
    # ----------------------------------------------------------------------
    def predict_test(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Makes predictions on the test data and returns predictions and standard deviations.
        """
        if not self._is_model_ready(): # check if the model is ready
            return None, None
        if self.X_test_scaled is None: # check if the test data is set
            self.logger.log_system_message("[WARN] No test data to predict.")
            return None, None

        y_pred, y_std = self.model.predict(self.X_test_scaled, return_std=True) # predict using test dataset and return standard deviation as well
        return y_pred, y_std

    def evaluate_test(self, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluates the model's performance on the test data using MSE, MAE, and R2.

        Args:
            y_pred (np.ndarray): Model predictions on the test data.

        Returns:
            Dict[str, float]: A dictionary containing the MSE, MAE, and R2 scores.
        """
        if self.y_test is None: # check if the test data is set
            self.logger.log_system_message("[WARN] No test data for evaluation.")
            return {"mse": np.nan, "mae": np.nan, "r2": np.nan}

        return {
            "mse": mean_squared_error(self.y_test, y_pred), # Mean Squared Error
            "mae": mean_absolute_error(self.y_test, y_pred), # Mean Absolute Error
            "r2": r2_score(self.y_test, y_pred), # R-squared score
        }

    def plot_test_result(
        self,
        y_pred: Optional[np.ndarray],  # Predicted values
        y_std: Optional[np.ndarray],  # Standard deviations of predictions
        percentile_threshold: float = 60.0, # percentile threshold for judging uncertainty
        std_multiplier: float = 2.0 # multiplier threshold for judging uncertainty.
    ) -> None:
        """
        Plots the actual vs. predicted values on the test data along with uncertainty.

        Args:
            y_pred (np.ndarray): Predicted values.
            y_std (np.ndarray): Standard deviations of predictions.
            percentile_threshold (float): The percentile of standard deviation above which a prediction is
                                        considered highly uncertain. Default is 60.0
            std_multiplier (float): The multiplier of the average standard deviation above which a prediction is
                                    considered highly uncertain. Defaults to 2.0.
        """
        if not self._can_plot_results(y_pred, y_std):
            return

        dates = self.test_df["date"] # get date data
        self._plot_predictions(dates, self.y_test, y_pred, y_std, percentile_threshold, std_multiplier) # call plot function

    def predict_for_date(
        self,
        date_str: str,  # Date string for prediction (format: 'YYYY-MM-DD HH:MM:SS')
        return_std: bool = True,  # Whether to return standard deviation and uncertainty flag
        percentile_threshold: float = 60.0, # Threshold for uncertainty based on percentile of std
        std_multiplier: float = 2.0 # Threshold for uncertainty based on multiple of average std
    ) -> Tuple[float, Optional[float], Optional[bool]]:
        """
        Predicts the target variable for a single date.

        Args:
            date_str (str): The date for which to make a prediction.
            return_std (bool): Whether to return the standard deviation and uncertainty flag.
            percentile_threshold (float):  Threshold for uncertainty based on percentile of standard deviation.
            std_multiplier (float):  Threshold for uncertainty based on multiple of average standard deviation.

        Returns:
            Tuple[float, Optional[float], Optional[bool]]: A tuple containing the prediction,
                standard deviation (if return_std is True), and uncertainty flag (if return_std is True).
        """


        if not self._is_model_ready(raise_error=True): # raise error if the model is not ready
            return 0.0, None, None # return 0 and None for the rest of values

        if self.all_data is None: # if all_data is None then call load and sort method
            self.all_data = self._load_and_sort_data("date")

        target_date = pd.to_datetime(date_str) # convert to datetime object
        row_data = self.all_data[self.all_data["date"] == target_date] # filter out the target date
        if row_data.empty:
            raise ValueError(f"No data found for date: {date_str}") # raise value error if the target date is not included in the data

        X_scaled = self.scaler.transform(row_data[self.feature_columns]) # scale input data

        if return_std: # if return_std is True
            y_pred, y_std = self.model.predict(X_scaled, return_std=True) # predict the target and return standard deviation
            is_uncertain = self._judge_uncertainty(y_std[0], percentile_threshold, std_multiplier) # decide if the prediction is uncertain
            return y_pred[0], y_std[0], is_uncertain # return prediction, standard deviation and uncertainty boolean
        else:
            y_pred = self.model.predict(X_scaled) # if return_std id False, just return prediction
            return y_pred[0], None, None

    # ----------------------------------------------------------------------
    # Internal Plotting / Utility
    # ----------------------------------------------------------------------
    def _plot_predictions(
        self,
        dates: pd.Series,  # Dates for the x-axis
        y_true: pd.Series,  # True values
        y_pred: np.ndarray,  # Predicted values
        y_std: np.ndarray,  # Standard deviations of predictions
        percentile_threshold: float, # Percentile threshold for uncertainty
        std_multiplier: float # Standard deviation multiplier threshold for uncertainty
    ) -> None:
        """Internal method to plot predictions with uncertainty."""
        plt.figure(figsize=(10, 6))
        plt.plot(dates, y_true, label="Actual", marker="o") # plot actual values
        plt.plot(dates, y_pred, label="Predicted", marker="x") # plot predictions

        # Plot 95% confidence interval
        y_upper = y_pred + 1.96 * y_std  # Calculate upper bound
        y_lower = y_pred - 1.96 * y_std  # Calculate lower bound
        plt.fill_between(dates, y_lower, y_upper, alpha=0.2, color="orange", label="95% CI") # plot confidence interval

        # Highlight regions of high uncertainty based on standard deviation thresholds
        threshold = np.percentile(y_std, percentile_threshold) # Percentile based threshold
        baseline_std = np.mean(y_std) # average standard deviation
        mask_high = (y_std > threshold) | (y_std > baseline_std * std_multiplier) # filter data based on threshold
        plt.fill_between(dates, y_lower, y_upper, where=mask_high, alpha=0.3, color="red", label="High Uncertainty") # plot areas with high uncertainty

        plt.xticks(rotation=45)
        plt.xlabel("Date")
        plt.ylabel(f"Shifted {self.target_column}")
        plt.title("Test Result with Prediction and Uncertainty")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _judge_uncertainty(self, y_std: float, percentile_threshold: float, std_multiplier: float) -> bool:
        """Judges if a prediction is uncertain based on its standard deviation."""

        if self.X_train_scaled is None:
            return False  # Cannot judge uncertainty without training data

        _, all_std = self.model.predict(self.X_train_scaled, return_std=True) # predict standard deviation for train data
        threshold = np.percentile(all_std, percentile_threshold)  # calculate standard deviation threshold based on percentile
        baseline_std = np.mean(all_std) # calculate average standard deviation

        # A prediction is uncertain if its standard deviation exceeds either threshold
        return (y_std > threshold) or (y_std > baseline_std * std_multiplier)

    def _can_plot_results(
        self,
        y_pred: Optional[np.ndarray],
        y_std: Optional[np.ndarray]
    ) -> bool:
        """Checks if all necessary data is available for plotting."""
        if self.test_df is None or self.y_test is None: # check if the test data and target is set
            self.logger.log_system_message("[WARN] No test data to plot.")
            return False
        if y_pred is None or y_std is None: # check if prediction and standard deviation is set
            self.logger.log_system_message("[WARN] Invalid prediction data.")
            return False
        return True

    def _is_model_ready(self, raise_error: bool = False) -> bool:
        """Checks if the model is trained and ready for use."""
        if self.model is None: # check if the model is set
            msg = "[WARN] Model is not trained or loaded."
            if raise_error:
                raise ValueError(msg) # raise error if raise_error is set as True
            else:
                self.logger.log_system_message(msg) # log warning message
            return False
        return True

    # ----------------------------------------------------------------------
    # Model IO
    # ----------------------------------------------------------------------
    def save_model(self, filename: Optional[str] = None) -> None:
        """Saves the trained model and scaler using joblib."""
        if filename is not None: # if filename is set, update it
            self.filename = filename

        if not self._is_model_ready(): # check if the model is ready
            return

        model_file_name = self.filename + ".joblib"  # Create the full filename
        model_path = os.path.join(self.datapath, model_file_name) # define path where the model will be saved
        self.logger.log_system_message(f"Saving model to {model_path}")

        data_to_save = { # define data to save
            "model": self.model, # model itself
            "scaler": self.scaler, # scaler
            "features": self.feature_columns, # feature columns
            "target_col": self.target_column, # target column name
            "shift_steps": self.shift_steps, # number of shift steps
        }
        joblib.dump(data_to_save, model_path)  # Save the model and related data

    def load_model(self, filename: Optional[str] = None) -> None:
        """Loads a pre-trained model and scaler using joblib."""
        if filename is not None: # if filename is set, update it
            self.filename = filename

        model_file_name = self.filename + ".joblib" # Create the full filename
        model_path = os.path.join(self.datapath, model_file_name) # define the path where the model will be loaded

        self.logger.log_system_message(f"Loading model from {model_path}")
        saved_data = joblib.load(model_path) # load the data from model path

        self.model = saved_data["model"] # load model
        self.scaler = saved_data["scaler"] # load scaler
        self.feature_columns = saved_data["features"] # load feature columns
        self.target_column = saved_data["target_col"] # load target column name
        self.shift_steps = saved_data["shift_steps"] # load number of steps to shift

    # ----------------------------------------------------------------------
    # GPU Configuration
    # ----------------------------------------------------------------------
    def _configure_gpu(self, use_gpu: bool) -> None:
        """Configures GPU usage (currently not utilized)."""
        from aiml.aiml_comm import configure_gpu
        configure_gpu(use_gpu=use_gpu,logger=self.logger)


def execute_bayes_prediction_model(current_date: str,symbol: str,interval: int): # define function to execute the bayesian model
    from aiml.aiml_comm import COLLECTIONS_TECH
    from common.utils import get_config_model # import function to get config

    model_id = "bayes_v1" # set model id
    data_loader = MongoDataLoader() # initiate dataloader
    logger = TradingLogger() # initiate logger

    # Get model configuration from config file
    config = get_config_model("MODEL_LONG_BAYES", model_id) # get config
    model = BayesianPredictionModel( # initiate model
        id=model_id,
        config=config,
        data_loader=data_loader,
        logger=logger,
        symbol=symbol,
        interval=interval,
        use_gpu=True
    )

    model.load_model() # load the model
    res = model.predict_for_date(current_date) # predict using the input date
    return res # return prediction


# --------------------------------------------------------------------------
# main function (example usage)
# --------------------------------------------------------------------------
def main():
    """Example usage of the BayesianPredictionModel."""
    from aiml.aiml_comm import COLLECTIONS_TECH
    from common.utils import get_config_model

    model_id = "bayes_v1" # set model id
    data_loader = MongoDataLoader() # initiate data loader
    logger = TradingLogger() # initiate logger

    config = get_config_model("MODEL_LONG_BAYES", model_id) # load config data
    model = BayesianPredictionModel( # initiate the model
        id=model_id,
        config=config,
        data_loader=data_loader,
        logger=logger,
        symbol="BTCUSDT",
        interval=1440,
        use_gpu=True
    )

    model.load_model()  # Load a pre-trained model

    # ----  Sample code for training and evaluation (commented out) ----

    # Example: Predict for a specific date
    res = model.predict_for_date("2025-01-15 00:00:00")  # Example date
    print("Prediction result:", res)



if __name__ == "__main__":
    main()
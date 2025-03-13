import os
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, List
from tqdm import tqdm

# Add parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from aiml.prediction_manager import PredictionManager
from aiml.aiml_comm import load_data, train_and_evaluate
from common.trading_logger import TradingLogger
from common.utils import get_config_model

# Constants for data loading and evaluation
START_DATE = "2020-01-01"
END_DATE = "2024-04-01"
NEW_DATA_START = "2024-04-01 00:00:00"
NEW_DATA_END = "2025-01-05 00:00:00"


def main(
        id: str, modle_gp:str,  epochs: int = 30, n_splits=2,use_gpu: bool = True
):
        """
        Main function for training and evaluating a prediction model.

        This function performs iterative training and evaluation of a prediction model. It uses
        cross-validation for training and evaluates the model on a separate dataset. The training process
        continues until a target accuracy is reached or a specified number of iterations without
        improvement is exceeded.

        Args:
                model_type (str): Type of the prediction model.
                id (str): ID of the prediction model.
                epochs (int, optional): Number of epochs for training. Defaults to 600.
                n_splits (int, optional): Number of splits for cross-validation. Defaults to 3.
                use_gpu (bool, optional): Whether to use GPU for training. Defaults to True.
        """

        target_accuracy = 0.50  # Target accuracy to achieve
        max_iterations = 100  # Maximum number of training iterations
        patience = (
                15  # Number of iterations without improvement before stopping training
        )
        best_accuracy = 0  # Best accuracy achieved so far
        no_improvement = 0  # Number of iterations without improvement
        best_report = ""  # Best classification report achieved so far
        best_conf_matrix = None  # Best confusion matrix achieved so far

        logger = TradingLogger()  # Initialize the logger
        model = PredictionManager()  # Initialize the prediction manager
        config = get_config_model(modle_gp, id)  # Get the model configuration

        logger.log_system_message(f"id: {id}, modle_gp: {modle_gp}, epochs: {epochs}, n_splits: {n_splits}, use_gpu: {use_gpu}")

        # Initialize the prediction model
        model.initialize_model(id, config)
        # Set model parameters
        model.set_parameters(param_epochs=epochs, n_splits=n_splits)

        # Determine the device to use for training (GPU or CPU)
        device = "/GPU:0" if use_gpu and tf.test.is_gpu_available() else "/CPU:0"

        # Iterate through training and evaluation
        for i in tqdm(range(max_iterations), desc="Training Progress"):
                logger.log_system_message(f"\nIteration {i+1}")
                # Train and evaluate the model
                (
                        cv_scores,
                        new_accuracy,
                        new_report,
                        new_conf_matrix,
                ) = train_and_evaluate(
                        model, device, START_DATE, END_DATE, NEW_DATA_START, NEW_DATA_END
                )

                # Log the evaluation results
                logger.log_system_message(f"Cross-validation scores: {cv_scores}")
                logger.log_system_message(f"Mean CV score: {np.mean(cv_scores)}")
                logger.log_system_message(f"New data accuracy: {new_accuracy}")

                # Check if the current model is the best so far
                if new_accuracy > best_accuracy:
                        best_accuracy = new_accuracy
                        best_report = new_report
                        best_conf_matrix = new_conf_matrix
                        no_improvement = 0
                        logger.log_system_message("New best model!")
                        # Save the best model
                        model.save_model()
                else:
                        no_improvement += 1  # Increment the counter for no improvement

                # Stop training if the target accuracy is achieved
                if new_accuracy >= target_accuracy:
                        logger.log_system_message(
                                f"Target accuracy {target_accuracy} achieved. Stopping training."
                        )
                        break

                # Stop training if there is no improvement for a certain number of iterations
                if no_improvement >= patience:
                        logger.log_system_message(
                                f"No improvement for {patience} iterations. Stopping training."
                        )
                        break

        # Log the final training results
        logger.log_system_message("\nTraining completed.")
        logger.log_system_message(f"Best accuracy on new data: {best_accuracy}")
        logger.log_system_message("Final classification report on new data:")
        logger.log_system_message(best_report)
        logger.log_system_message("Final confusion matrix:")
        logger.log_system_message(best_conf_matrix)

        logger.log_system_message("test model load")


if __name__ == "__main__":
        main("rolling_v2", "MODEL_SHORT_TERM", use_gpu=True)

        """
        for i in range(2, 7):
            version = f"rolling_v{i}"
            print(f"Start training: {version}")
            main(version, "MODEL_SHORT_TERM", use_gpu=True)
        """

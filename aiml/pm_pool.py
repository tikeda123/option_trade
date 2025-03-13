import threading
import os
import sys
from typing import Dict
# Add other necessary imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from aiml.prediction_manager import PredictionManager
from common.utils import get_config_model


class PredictionManagerPool:
    """
    Singleton class to manage PredictionManager instances.
    Allows creation and retrieval of managers when needed.
    """
    _instance = None
    _lock = threading.Lock()  # For thread-safe initialization

    def __new__(cls):
        """
        Standard Python singleton implementation.
        Creates an instance only if it doesn't exist, then returns the same instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    # Dictionary to manage instances within the singleton
                    cls._instance._managers = {}
        return cls._instance

    def get_manager(self, manager_id: str):
        """
        Retrieves an existing PredictionManager instance.
        Returns None if not found.

        Args:
            manager_id: Unique identifier for the manager

        Returns:
            PredictionManager instance or None
        """
        return self._managers.get(manager_id, None)

    def create_manager(self, manager_id: str, config_group: str, config_name: str) -> "PredictionManager":
        """
        Creates a new PredictionManager instance and registers it internally.
        Returns existing instance if one with the same ID already exists.

        Args:
            manager_id: Unique identifier for the manager
            config_group: Configuration group name
            config_name: Configuration name

        Returns:
            PredictionManager instance
        """
        existing_manager = self.get_manager(manager_id)
        if existing_manager is not None:
            # Return existing manager if one with the same ID exists
            return existing_manager

        # Create new instance
        pm = PredictionManager()

        # Get config and initialize if needed
        config = get_config_model(config_group, config_name)
        pm.initialize_model(manager_id, config)

        # Register the created instance in the pool
        self._managers[manager_id] = pm
        return pm

    def delete_manager(self, manager_id: str) -> bool:
        """
        Deletes a PredictionManager with the specified ID.
        Returns True if deleted, False if not found.

        Args:
            manager_id: Unique identifier for the manager to delete

        Returns:
            bool: Success status of deletion
        """
        if manager_id in self._managers:
            del self._managers[manager_id]
            return True
        return False

    def list_manager_ids(self) -> Dict[str, "PredictionManager"]:
        """
        Gets a list of all manager IDs currently registered in the pool.

        Returns:
            List of manager IDs
        """
        return list(self._managers.keys())

    def load_model(self, manager_id: str):
        """
        Loads a model with the specified ID and configuration.
        """
        pm = self._managers.get(manager_id, None)
        if pm is not None:
            pm.load_model()

def main():
    # Get singleton instance
    manager_pool = PredictionManagerPool()

    # Create PredictionManager with manager_id = "rolling_m1"
    # (creates new instance first time, returns existing instance thereafter)
    pm1 = manager_pool.create_manager(manager_id="rolling_v2",
                                      config_group="MODEL_SHORT_TERM",
                                      config_name="rolling_v2")

    manager_pool.load_model(manager_id="rolling_v2")



if __name__ == "__main__":
    main()

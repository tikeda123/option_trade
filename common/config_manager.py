import json
import os
from typing import Any, Dict, Optional, Tuple

from common.utils import get_config_fullpath  # Import should be at the top of the file


class ConfigManager:
        """
        Manages configuration data loaded from a JSON file.

        Loads the configuration data on initialization and provides methods to access,
        modify, and save the configuration values.

        Attributes:
                _config (Dict[str, Any]): The loaded configuration data.

        Raises:
                FileNotFoundError: If the configuration file is not found.
                json.JSONDecodeError: If the configuration file is not valid JSON.
                KeyError: If the specified section, key, or subkey is not found.
        """

        def __init__(self) -> None:
                """Initializes the ConfigManager with data from the default configuration file."""

                config_fullpath = get_config_fullpath()
                self._config = self._load_config(config_fullpath)

        def _load_config(self, config_path: str) -> Dict[str, Any]:
                """Loads configuration data from a JSON file.

                Args:
                        config_path (str): The path to the configuration file.

                Returns:
                        Dict[str, Any]: The loaded configuration data as a dictionary.

                Raises:
                        FileNotFoundError: If the configuration file is not found.
                        json.JSONDecodeError: If the configuration file is not valid JSON.
                """

                try:
                        with open(config_path, 'r') as file:
                                return json.load(file)
                except FileNotFoundError:
                        raise FileNotFoundError(f"Configuration file not found: {config_path}")
                except json.JSONDecodeError as e:
                        raise json.JSONDecodeError(
                                f"Failed to load configuration file: {e}", e.doc, e.pos
                        ) from e

        def get(self, section: str, key: Optional[str] = None, subkey: Optional[str] = None) -> Any:
                """Retrieves a value from the configuration.

                Args:
                        section (str): The section containing the key.
                        key (Optional[str]): The key within the section. If None, returns the entire section.
                        subkey (Optional[str]): If the value associated with the key is a dictionary,
                                                                        this argument can be used to access a value within that sub-dictionary.

                Returns:
                        Any: The configuration value, or None if the key or subkey is not found.

                Raises:
                        KeyError: If the specified section, key, or subkey is not found.
                """
                try:
                        data = self._config[section]
                        if key:
                                data = data[key]
                                if subkey:
                                        data = data[subkey]
                        return data
                except KeyError as e:
                        raise KeyError(f"Key not found in configuration: {e}") from e

        def set(self, section: str, value: Any, key: Optional[str] = None, subkey: Optional[str] = None) -> None:
                """Sets a value in the configuration.

                Args:
                        section (str): The section to set the value in.
                        value (Any): The value to set.
                        key (Optional[str]): The key within the section. If None, sets the value for the entire section.
                        subkey (Optional[str]): If the value associated with the key is a dictionary,
                                                                        this argument can be used to set a value within that sub-dictionary.

                Raises:
                        KeyError: If the specified section, key, or subkey is not found.
                """
                try:
                        if key:
                                if subkey:
                                        self._config[section][key][subkey] = value
                                else:
                                        self._config[section][key] = value
                        else:
                                self._config[section] = value
                except KeyError as e:
                        raise KeyError(f"Key not found in configuration: {e}") from e

        def save(self, config_path: Optional[str] = None) -> None:
                """Saves the current configuration to a JSON file.

                Args:
                        config_path (Optional[str]): The path to save the configuration file to.
                                                                                If not specified, saves to the path used during initialization.
                """
                if config_path is None:
                        config_path = get_config_fullpath()

                with open(config_path, 'w') as file:
                        json.dump(self._config, file, indent=4)


def get_directory_path(file: str) -> Tuple[str, str]:
        """Gets the directory paths of the given file.

        Args:
                file (str): The filename.

        Returns:
                Tuple[str, str]: A tuple containing the parent directory and the current directory.
        """

        current_dir = os.path.dirname(os.path.abspath(file))
        parent_dir = os.path.dirname(current_dir)
        return parent_dir, current_dir
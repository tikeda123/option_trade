import json
from typing import Any, Optional


def get_directory_path(file):
    """ファイルが存在するディレクトリのパスを取得します。

    Args:
        file (str): ファイル名。

    Returns:
        tuple: ファイルが存在する親ディレクトリと現在のディレクトリのパス。
    """
    import os
    # b.pyのディレクトリの絶対パスを取得
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得
    return parent_dir,current_dir

class ConfigManager:
    """
    設定ファイルを管理するクラスです。

    設定ファイルからのデータ読み込みと、特定の設定値の取得を行います。

    Attributes:
        __config (dict): 設定ファイルから読み込んだデータ。

    Args:
        config_fullpath (str): 設定ファイルのフルパス。
    """
    def __init__(self):
        from common.utils import get_config_fullpath
        config_fullpath = get_config_fullpath()

        self._load_config(config_fullpath)

    def _load_config(self, config_path):
        """設定ファイルを読み込み、設定データを格納します。

        Args:
            config_path (str): 設定ファイルのパス。
        """
        with open(config_path, 'r') as file:
            self.__config = json.load(file)

    def get(self, section: str, key: Optional[str] = None, key2: Optional[str] = None) -> Any:
        """指定されたセクション、キー、サブキーに対応する設定値を取得します。

        Args:
            section (str): 設定ファイルのセクション名。
            key (Optional[str]): セクション内のキー名。Noneの場合はセクション全体を返します。
            key2 (Optional[str]): キー内のサブキー名。Noneの場合はキー全体を返します。

        Returns:
            Any: 設定値。データ型は設定に依存します。

        Raises:
            ValueError: 指定されたセクション、キー、またはサブキーが設定ファイルに存在しない場合。
        """
        data = self.__config.get(section)
        if data is None:
            raise ValueError(f"Section '{section}' not found in configuration.")

        if key is not None:
            data = data.get(key)
            if data is None:
                raise ValueError(f"Key '{key}' not found in section '{section}'.")

        if key2 is not None:
            data = data.get(key2)
            if data is None:
                raise ValueError(f"Sub-key '{key2}' not found in key '{key}' of section '{section}'.")

        return data

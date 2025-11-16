#!/usr/bin/env python
"""
Configuration Loader for Basketball Prediction System

Loads and validates configuration from YAML file.
Provides type-safe access to configuration values.

Usage:
    from config_loader import Config

    config = Config()
    rolling_window = config.get('data_collection.rolling_window_size')
    bankroll = config.get('betting.bankroll.initial_amount')
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


class Config:
    """
    Configuration loader and accessor.

    Loads configuration from YAML file and provides
    dot-notation access to nested values.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to config.yaml file.
                        If None, looks in default locations.
        """
        if config_path is None:
            config_path = self._find_config_file()

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()

    def _find_config_file(self) -> Path:
        """
        Find config.yaml in standard locations.

        Returns:
            Path to config.yaml

        Raises:
            ConfigError: If config file not found
        """
        # Try multiple locations
        search_paths = [
            Path(__file__).parent.parent / "config.yaml",  # 2026/config.yaml
            Path(__file__).parent / "config.yaml",          # 2026/src/config.yaml
            Path.cwd() / "config.yaml",                     # ./config.yaml
            Path.cwd() / "2026" / "config.yaml",            # ./2026/config.yaml
        ]

        for path in search_paths:
            if path.exists():
                return path

        raise ConfigError(
            f"Config file not found. Searched locations:\n" +
            "\n".join(f"  - {p}" for p in search_paths)
        )

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Returns:
            Configuration dictionary

        Raises:
            ConfigError: If file cannot be loaded
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            if config is None:
                raise ConfigError("Config file is empty")

            return config

        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in config file: {e}")
        except FileNotFoundError:
            raise ConfigError(f"Config file not found: {self.config_path}")
        except Exception as e:
            raise ConfigError(f"Error loading config: {e}")

    def _validate_config(self):
        """
        Validate that required configuration sections exist.

        Raises:
            ConfigError: If required sections are missing
        """
        required_sections = [
            'season',
            'data_collection',
            'machine_learning',
            'betting',
            'paths',
        ]

        for section in required_sections:
            if section not in self.config:
                raise ConfigError(f"Required config section missing: {section}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path to value (e.g., 'betting.kelly.fraction')
            default: Default value if key not found

        Returns:
            Configuration value or default

        Examples:
            >>> config = Config()
            >>> config.get('season.current')
            2026
            >>> config.get('betting.bankroll.initial_amount')
            1000.0
            >>> config.get('nonexistent.key', 'default_value')
            'default_value'
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_required(self, key_path: str) -> Any:
        """
        Get required configuration value.

        Args:
            key_path: Dot-separated path to value

        Returns:
            Configuration value

        Raises:
            ConfigError: If key not found
        """
        value = self.get(key_path)
        if value is None:
            raise ConfigError(f"Required configuration missing: {key_path}")
        return value

    def get_path(self, key_path: str, create_if_missing: bool = False) -> Path:
        """
        Get path from configuration and convert to Path object.

        Args:
            key_path: Dot-separated path to value
            create_if_missing: Create directory if it doesn't exist

        Returns:
            Path object

        Raises:
            ConfigError: If key not found
        """
        path_str = self.get_required(key_path)
        path = Path(path_str)

        # Make absolute if relative
        if not path.is_absolute():
            # Relative to project root (parent of 2026/)
            project_root = Path(__file__).parent.parent.parent
            path = project_root / path

        if create_if_missing:
            path.mkdir(parents=True, exist_ok=True)

        return path

    def get_all(self) -> Dict[str, Any]:
        """
        Get entire configuration dictionary.

        Returns:
            Full configuration
        """
        return self.config.copy()

    def reload(self):
        """Reload configuration from file."""
        self.config = self._load_config()
        self._validate_config()

    def __repr__(self) -> str:
        return f"Config(config_path='{self.config_path}')"

    def __str__(self) -> str:
        return f"Configuration loaded from: {self.config_path}"


# Singleton instance (optional)
_config_instance: Optional[Config] = None


def get_config(reload: bool = False) -> Config:
    """
    Get global configuration instance.

    Args:
        reload: Force reload from file

    Returns:
        Config instance
    """
    global _config_instance

    if _config_instance is None or reload:
        _config_instance = Config()

    return _config_instance


# Convenience functions
def get_value(key_path: str, default: Any = None) -> Any:
    """Get configuration value (convenience function)."""
    return get_config().get(key_path, default)


def get_required_value(key_path: str) -> Any:
    """Get required configuration value (convenience function)."""
    return get_config().get_required(key_path)


if __name__ == "__main__":
    # Test configuration loading
    try:
        config = Config()
        print(f"✅ Configuration loaded successfully: {config}")
        print(f"\nCurrent season: {config.get('season.current')}")
        print(f"Rolling window: {config.get('data_collection.rolling_window_size')}")
        print(f"Initial bankroll: €{config.get('betting.bankroll.initial_amount')}")
        print(f"Kelly fraction: {config.get('betting.kelly.fraction')}")

        print("\n✅ All configuration tests passed!")

    except ConfigError as e:
        print(f"❌ Configuration error: {e}")
        exit(1)

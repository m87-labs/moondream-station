import json

from pathlib import Path
from typing import Any, Dict

SERVICE_PORT = 2020
UPDATE_ENDPOINT = "https://api.github.com/repos/m87/moondream-station/releases/latest"
SERVICE_HOST = "0.0.0.0"
AUTO_START = True
LOG_LEVEL = "INFO"
INFERENCE_WORKERS = 1
INFERENCE_MAX_QUEUE_SIZE = 10
INFERENCE_TIMEOUT = 30.0

# UI Constants
PANEL_WIDTH = 70
PORT_SEARCH_RANGE = 20
HISTORY_DISPLAY_LIMIT = 10

# Network Constants
NETWORK_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 60


class ConfigManager:
    def __init__(self):
        self.config_dir = Path.home() / ".moondream-station"
        self.config_file = self.config_dir / "config.json"
        self.config_dir.mkdir(exist_ok=True)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "current_model": None,
            "service_port": SERVICE_PORT,
            "models_dir": str(self.config_dir / "models"),
            "update_endpoint": UPDATE_ENDPOINT,
            "service_host": SERVICE_HOST,
            "auto_start": AUTO_START,
            "log_level": LOG_LEVEL,
            "inference_workers": INFERENCE_WORKERS,
            "inference_max_queue_size": INFERENCE_MAX_QUEUE_SIZE,
            "inference_timeout": INFERENCE_TIMEOUT,
            "logging": True,
        }

    def _save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, "w") as f:
                json.dump(self._config, f, indent=2)
        except IOError:
            pass

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, default)

    def set(self, key: str, value: Any):
        """Set configuration value"""
        self._config[key] = value
        self._save_config()

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values"""
        return self._config.copy()

    def reset(self):
        """Reset configuration to defaults"""
        self._config = self._get_default_config()
        self._save_config()

    def delete(self, key: str) -> bool:
        """Delete configuration key"""
        if key in self._config:
            del self._config[key]
            self._save_config()
            return True
        return False

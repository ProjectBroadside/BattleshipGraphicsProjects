
import yaml
from pathlib import Path
import os

class AppConfig:
    """Loads and provides access to the application configuration."""

    def __init__(self, config_path='config.yaml'):
        self.config = self._load_config(config_path)
        self._setup_directories()

    def _load_config(self, config_path):
        config_file = Path(__file__).parent / config_path
        if not config_file.is_file():
            raise FileNotFoundError(f"Configuration file not found at: {config_file}")
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def _setup_directories(self):
        base_dir = self.get('output_settings.base_dir', 'generated_images')
        Path(base_dir).mkdir(parents=True, exist_ok=True)

    def get(self, key_path, default=None):
        """Access nested configuration values using dot notation."""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def get_api_key(self):
        """Get API key from environment, falling back to a placeholder."""
        return os.getenv("GOOGLE_API_KEY", "AIzaSyBbz8RJloQMJydYVfkdAJaL6ysq_bUkSe4")

# Global config instance
config = AppConfig()

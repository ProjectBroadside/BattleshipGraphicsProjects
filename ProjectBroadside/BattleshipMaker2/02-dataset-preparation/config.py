
import yaml
from pathlib import Path

class DatasetConfig:
    """Loads and provides access to dataset validation configuration."""

    def __init__(self, config_path='config.yaml'):
        self.config = self._load_config(config_path)

    def _load_config(self, config_path):
        config_file = Path(__file__).parent / config_path
        if not config_file.is_file():
            # Provide a default configuration if the file doesn't exist
            return {
                'validation': {
                    'min_resolution': [1024, 1024],
                    'sharpness_threshold': 100.0,
                    'exposure_range': [50, 200]
                },
                'geometric_validation': {
                    'min_angle_gap': 45.0
                },
                'preprocessing': {
                    'enable_sharpening': True,
                    'enable_denoising': True
                'reconstruction': {
                    'run_reconstruction': False
                }
            }
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def get(self, key_path, default=None):
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

# Global config instance
config = DatasetConfig()

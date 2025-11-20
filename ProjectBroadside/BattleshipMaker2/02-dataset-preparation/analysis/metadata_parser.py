
import json
from pathlib import Path
import logging

class MetadataParser:
    """Parses camera pose and other metadata from JSON files."""

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)

    def extract_poses(self):
        """Extracts camera poses from all JSON files in the dataset directory."""
        poses = []
        for file_path in self.dataset_path.glob('*.json'):
            try:
                with open(file_path, 'r') as f:
                    metadata = json.load(f)
                    if 'camera_settings' in metadata:
                        poses.append(metadata['camera_settings'])
            except (json.JSONDecodeError, IOError) as e:
                logging.warning(f"Could not parse metadata from {file_path.name}: {e}")
        return poses

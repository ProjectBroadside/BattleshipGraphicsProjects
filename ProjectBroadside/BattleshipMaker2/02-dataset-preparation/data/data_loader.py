
from pathlib import Path
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)

class DatasetLoader:
    """Loads images from a directory for validation."""
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_path}")

    def load_images(self):
        """Loads all supported images from the dataset directory."""
        images = {}
        for file_path in self.dataset_path.iterdir():
            if file_path.suffix.lower() in self.SUPPORTED_FORMATS:
                try:
                    img = Image.open(file_path)
                    images[file_path.name] = img
                    logging.info(f"Loaded image: {file_path.name}")
                except Exception as e:
                    logging.warning(f"Could not load image {file_path.name}: {e}")
        return images

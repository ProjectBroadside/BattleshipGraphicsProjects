
import cv2
import numpy as np
from skimage import exposure

class ImageQualityAnalyzer:
    """Analyzes the quality of a single image."""

    def __init__(self, config):
        self.config = config

    def analyze(self, image_name, image):
        """Performs a full quality analysis on an image."""
        np_image = np.array(image.convert('L')) # Convert to grayscale numpy array
        
        resolution = image.size
        sharpness = self._calculate_sharpness(np_image)
        exposure_val = self._calculate_exposure(np_image)

        # Validation checks
        min_res = self.config.get('validation.min_resolution', [1024, 1024])
        sharpness_thresh = self.config.get('validation.sharpness_threshold', 100.0)
        exposure_min, exposure_max = self.config.get('validation.exposure_range', [50, 200])

        is_valid_resolution = resolution[0] >= min_res[0] and resolution[1] >= min_res[1]
        is_sharp = sharpness >= sharpness_thresh
        is_good_exposure = exposure_min <= exposure_val <= exposure_max

        return {
            "image_name": image_name,
            "resolution": resolution,
            "sharpness": sharpness,
            "exposure": exposure_val,
            "checks": {
                "resolution_ok": is_valid_resolution,
                "sharpness_ok": is_sharp,
                "exposure_ok": is_good_exposure
            }
        }

    def _calculate_sharpness(self, image):
        """Calculates image sharpness using the variance of the Laplacian."""
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def _calculate_exposure(self, image):
        """Calculates the mean brightness of an image."""
        return np.mean(image)


import cv2
import numpy as np
from PIL import Image

class Preprocessor:
    """Applies preprocessing steps to images."""

    def __init__(self, config):
        self.config = config

    def correct_exposure(self, image):
        """Corrects the exposure of an image using histogram equalization."""
        np_image = np.array(image)
        # Convert RGB to BGR for OpenCV
        np_image = np_image[:, :, ::-1].copy()

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab = cv2.cvtColor(np_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        return Image.fromarray(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))

    def adaptive_sharpen(self, image):
        """Applies an unsharp mask to sharpen the image."""
        np_image = np.array(image)
        # Apply a Gaussian blur
        blurred = cv2.GaussianBlur(np_image, (0, 0), 3)
        # Subtract the blurred image from the original to create the mask
        sharpened = cv2.addWeighted(np_image, 1.5, blurred, -0.5, 0)
        return Image.fromarray(sharpened)

    def denoise(self, image):
        """Applies non-local means denoising."""
        np_image = np.array(image)
        # Convert RGB to BGR for OpenCV
        np_image = np_image[:, :, ::-1].copy()
        denoised = cv2.fastNlMeansDenoisingColored(np_image, None, 10, 10, 7, 21)
        return Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))

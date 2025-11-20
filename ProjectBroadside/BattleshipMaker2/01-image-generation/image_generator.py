
import os
import json
import time
import logging
from pathlib import Path
from itertools import product
from PIL import Image, ImageDraw, ImageFont
import base64
import io

# Try importing Google AI - fallback to placeholders if not available
try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    logging.warning("Google AI library not available. Using placeholder mode.")

from config import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageGenerator:
    """Generates images based on systematic prompts and camera poses."""

    def __init__(self):
        self.config = config
        self.model_name = self.config.get('generation_model.name')
        self.output_base_dir = Path(self.config.get('output_settings.base_dir') or 'output')
        
        # Initialize Google AI if available
        if GOOGLE_AI_AVAILABLE:
            api_key = self.config.get_api_key()
            if api_key and api_key != "your_api_key_here":
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                logging.info("Google AI configured with live API key")
                self.use_live_api = True
            else:
                logging.warning("No valid API key found, using placeholder mode")
                self.use_live_api = False
        else:
            self.use_live_api = False

    def _generate_with_gemini(self, prompt, output_path):
        """Generate image using Google Gemini API."""
        if not GOOGLE_AI_AVAILABLE or not self.use_live_api:
            return False
            
        try:
            # Note: Google Gemini API currently focuses on text generation
            # For image generation, we would typically use DALL-E, Midjourney, or Stable Diffusion
            # Since the API key provided is for Gemini, we'll enhance the prompt using Gemini
            # then create a placeholder with the enhanced description
            
            # Generate enhanced description using Gemini
            if GOOGLE_AI_AVAILABLE:
                import google.generativeai as genai
                model = genai.GenerativeModel('gemini-1.5-flash')
            else:
                return False
            enhanced_prompt = f"Enhance this image generation prompt for a photorealistic battleship image: {prompt}"
            
            response = model.generate_content(enhanced_prompt)
            enhanced_description = response.text
            
            logging.info(f"Generated enhanced description via Gemini: {enhanced_description[:100]}...")
            
            # For now, create a placeholder with the enhanced description
            # In production, you'd pass this to an actual image generation API
            return self._create_placeholder_image(
                f"Gemini Enhanced:\n{enhanced_description[:200]}...",
                output_path
            )
            
        except Exception as e:
            logging.error(f"Failed to generate with Gemini API: {e}")
            return False

    def _create_placeholder_image(self, text, path):
        """Creates a placeholder image with text, simulating a real API call."""
        try:
            width = self.config.get('image_settings.width') or 512
            height = self.config.get('image_settings.height') or 512
            img = Image.new('RGB', (width, height))
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 15)
            except IOError:
                font = ImageFont.load_default()
            draw.text((10, 10), text, fill=(255, 255, 255), font=font)
            img.save(path)
            logging.info(f"Successfully created placeholder image: {path}")
            return True
        except Exception as e:
            logging.error(f"Failed to create placeholder image at {path}: {e}")
            return False

    def _generate_prompt(self, camera_distance, camera_height, camera_angle):
        """Generates a detailed, structured prompt for image generation."""
        style = self.config.get('prompt_engineering.style')
        negative = self.config.get('prompt_engineering.negative')
        return (
            f"A photorealistic image of the German battleship Bismarck. "
            f"Style: {style}. Negative keywords: {negative}. "
            f"Camera: distance={camera_distance}m, height={camera_height}m, angle={camera_angle}deg."
        )

    def generate_image(self, prompt, metadata, output_path):
        """Generates a single image using Gemini API or placeholder, and saves it with metadata."""
        image_path = output_path.with_suffix('.png')
        metadata_path = output_path.with_suffix('.json')

        # Try Gemini API first, fall back to placeholder
        if self.use_live_api:
            success = self._generate_with_gemini(prompt, image_path)
            if success:
                metadata['generation_method'] = 'gemini_enhanced'
            else:
                success = self._create_placeholder_image(prompt, image_path)
                metadata['generation_method'] = 'placeholder_fallback'
        else:
            success = self._create_placeholder_image(prompt, image_path)
            metadata['generation_method'] = 'placeholder_only'

        if not success:
            return False

        # Save metadata
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            logging.info(f"Successfully saved metadata: {metadata_path}")
            return True
        except IOError as e:
            logging.error(f"Failed to save metadata at {metadata_path}: {e}")
            return False

    def run_batch_generation(self, batch_name, num_images):
        """Generates a batch of images with systematic camera poses."""
        batch_dir = self.output_base_dir / batch_name
        batch_dir.mkdir(parents=True, exist_ok=True)

        distances = self.config.get('camera_poses.distances') or [100, 200, 300]
        heights = self.config.get('camera_poses.heights') or [50, 100, 150]
        angles = self.config.get('camera_poses.angles') or [0, 45, 90, 135, 180]

        # Create all possible camera pose combinations
        all_poses = list(product(distances, heights, angles))
        poses_to_generate = all_poses[:num_images]

        logging.info(f"Starting batch '{batch_name}' to generate {len(poses_to_generate)} images.")

        for i, (dist, h, angle) in enumerate(poses_to_generate):
            prompt = self._generate_prompt(dist, h, angle)
            output_prefix = f"{self.config.get('output_settings.prefix')}_{i:04d}"
            output_path = batch_dir / output_prefix

            metadata = {
                "batch_name": batch_name,
                "image_index": i,
                "model_name": self.model_name,
                "timestamp": time.time(),
                "prompt": prompt,
                "camera_settings": {"distance": dist, "height": h, "angle": angle}
            }

            self.generate_image(prompt, metadata, output_path)

        logging.info(f"Batch '{batch_name}' complete.")

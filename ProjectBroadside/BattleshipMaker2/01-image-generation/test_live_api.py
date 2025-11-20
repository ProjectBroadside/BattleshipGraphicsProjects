#!/usr/bin/env python3
"""
Test script for live Gemini API integration
Generates a small batch of images to verify the implementation
"""

import logging
from image_generator import ImageGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Test the live Gemini API integration with a small batch of images."""
    try:
        # Initialize the image generator
        generator = ImageGenerator()
        
        # Check API status
        if generator.use_live_api:
            logging.info("✅ Live Gemini API is configured and ready")
        else:
            logging.info("⚠️  Using placeholder mode (API not available)")
        
        # Test with a small batch of 3 images
        batch_name = "gemini_api_test"
        num_images = 3
        
        logging.info(f"Starting test batch '{batch_name}' with {num_images} images...")
        
        # Run the batch generation
        generator.run_batch_generation(batch_name, num_images)
        
        logging.info(f"✅ Test batch '{batch_name}' completed successfully!")
        logging.info(f"Check the output directory: {generator.output_base_dir / batch_name}")
        
    except Exception as e:
        logging.error(f"❌ Test failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
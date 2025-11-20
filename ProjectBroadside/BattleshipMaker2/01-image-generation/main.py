
import argparse
from image_generator import ImageGenerator

def main():
    """Main entry point for the image generation CLI."""
    parser = argparse.ArgumentParser(description="Generate a systematic dataset of Bismarck images.")
    parser.add_argument("--batch-name", required=True, help="Name for the generation batch (e.g., 'production_run_01').")
    parser.add_argument("--num-images", type=int, required=True, help="Number of images to generate.")
    
    args = parser.parse_args()
    
    generator = ImageGenerator()
    generator.run_batch_generation(args.batch_name, args.num_images)

if __name__ == "__main__":
    main()

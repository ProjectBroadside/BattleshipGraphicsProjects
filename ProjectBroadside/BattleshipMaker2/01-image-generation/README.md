# Simulated Image Generation Pipeline

This project provides a simulated pipeline for generating a systematic image dataset of the Bismarck battleship, intended for 3D Gaussian Splatting (3DGS) training. It generates placeholder images and corresponding metadata based on a structured set of camera poses and prompts.

**Note:** This is a *simulation*. It creates placeholder images with metadata to mimic a real image generation pipeline but does not make actual API calls to an image generation service.

## Features

-   **Configuration-Driven**: All parameters are managed in `config.yaml` for easy modification.
-   **Systematic Prompt Generation**: Creates detailed prompts for consistent image characteristics.
-   **Structured Metadata**: Saves a JSON file for each image with detailed generation parameters.
-   **Simulated Image Output**: Generates placeholder PNG images with metadata text for each prompt.
-   **CLI Interface**: A simple command-line interface to run batch generation.

## How It Works

The pipeline reads camera poses and prompt details from `config.yaml`. When you run a batch, it iterates through the specified number of camera pose combinations, generates a text prompt, and creates a corresponding placeholder image and a JSON metadata file.

## Setup and Installation

1.  **Clone the Repository** (if you haven't already).

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Review Configuration**:
    Open `config.yaml` to review and adjust parameters like image dimensions, camera poses, and prompt styles.

## Usage

To generate a new batch of simulated images, run `main.py` with a batch name and the number of images to generate.

```bash
python main.py --batch-name <your_batch_name> --num-images <number_of_images>
```

**Example:**

```bash
# Generate 20 placeholder images for a batch named "initial_test_run"
python main.py --batch-name initial_test_run --num-images 20
```

The output will be saved in `generated_images/initial_test_run/`, containing `.png` and `.json` files for each generated item.

## Project Structure

-   `main.py`: The command-line entry point for the application.
-   `image_generator.py`: Contains the core `ImageGenerator` class that handles the generation logic.
-   `config.py`: Loads and provides access to the configuration from `config.yaml`.
-   `config.yaml`: The main configuration file for all parameters.
-   `requirements.txt`: A list of all Python dependencies.
-   `generated_images/`: The default output directory for all generated content.

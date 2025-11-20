# Dataset Preparation and Validation

This module provides a comprehensive pipeline to validate and prepare image datasets for 3D Gaussian Splatting (3DGS) training. It analyzes key quality and geometric metrics, can automatically correct common issues, provides feedback for improving the image generation process, and can even generate a preview 3D reconstruction.

## Features

-   **Configuration-Driven**: All parameters are managed in `config.yaml`.
-   **Image Quality Analysis**: Checks for issues like low resolution, blurriness, and poor exposure.
-   **Geometric Validation**: Analyzes camera pose distribution to identify coverage gaps.
-   **Visual Reporting**: Generates a coverage heatmap to visualize the camera pose distribution.
-   **Automated Preprocessing**: Can automatically correct exposure and sharpness issues.
-   **Feedback Generation**: Provides recommendations for improving the image generation process.
-   **3D Reconstruction Preview**: Can run a sparse 3D reconstruction using COLMAP to provide a direct preview of the dataset's potential.
-   **Batch Processing**: Validates and preprocesses an entire directory of images.
-   **Simple CLI**: Easy to run the entire pipeline from the command line.

## How It Works

The pipeline performs the following steps:

1.  **Loads Data**: Reads images and camera poses.
2.  **Analyzes Quality and Geometry**: Assesses image quality and geometric coverage.
3.  **Generates Visualizations**: Creates a coverage heatmap.
4.  **Generates Report**: Compiles a detailed JSON report with all findings.
5.  **Preprocesses Images**: Corrects quality issues and saves the results.
6.  **Generates Feedback**: Provides actionable recommendations.
7.  **(Optional) 3D Reconstruction**: If enabled, it runs COLMAP to generate a sparse 3D model.

## Setup and Installation

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Install COLMAP**: This module requires the COLMAP command-line tool to be installed and available in your system's PATH. Please follow the official COLMAP installation instructions.

3.  **(Optional) Configure**:
    Create a `config.yaml` file in the `02-dataset-preparation` directory to override the default settings.

## Usage

To run the full validation and preprocessing pipeline, including the 3D reconstruction preview, use the `--run-reconstruction` flag.

```bash
python main.py <path_to_your_dataset> --run-reconstruction
```

**Example:**

```bash
python 02-dataset-preparation/main.py 01-image-generation/generated_images/initial_test_run --run-reconstruction
```

The output will be a JSON report printed to the console. New directories will be created for preprocessed images, visualizations, and the COLMAP reconstruction.

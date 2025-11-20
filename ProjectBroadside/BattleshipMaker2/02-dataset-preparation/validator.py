
from .data.data_loader import DatasetLoader
from .analysis.quality_analyzer import ImageQualityAnalyzer
from .analysis.metadata_parser import MetadataParser
from .analysis.geometric_analyzer import GeometricAnalyzer
from .analysis.reconstruction import ReconstructionAnalyzer
from .preprocessor import Preprocessor
from .feedback_generator import FeedbackGenerator
from .visualizer import Visualizer
from .config import config
import json
from pathlib import Path

class DatasetValidator:
    """Orchestrates the dataset validation, preprocessing, and feedback generation."""

    def __init__(self, dataset_path, run_reconstruction=False):
        self.dataset_path = Path(dataset_path)
        self.run_reconstruction = run_reconstruction
        self.loader = DatasetLoader(self.dataset_path)
        self.quality_analyzer = ImageQualityAnalyzer(config)
        self.metadata_parser = MetadataParser(self.dataset_path)
        self.geometric_analyzer = GeometricAnalyzer(config)
        self.preprocessor = Preprocessor(config)
        self.feedback_generator = FeedbackGenerator(config)
        self.visualizer = Visualizer(config, self.dataset_path.parent / (self.dataset_path.name + "_visuals"))
        if self.run_reconstruction:
            self.reconstruction_analyzer = ReconstructionAnalyzer(config, self.dataset_path)
        self.preprocessed_dir = self.dataset_path.parent / (self.dataset_path.name + "_preprocessed")
        self.preprocessed_dir.mkdir(exist_ok=True)

    def validate(self):
        """Runs the full validation pipeline and returns a report."""
        images = self.loader.load_images()
        if not images:
            print("No images found in the dataset directory.")
            return None

        # Quality Analysis
        quality_report = self._perform_quality_analysis(images)

        # Geometric Analysis
        poses = self.metadata_parser.extract_poses()
        geometric_report = self.geometric_analyzer.analyze_coverage(poses)

        # Visualization
        heatmap_path = self.visualizer.generate_coverage_heatmap(poses)
        if heatmap_path:
            geometric_report['coverage_heatmap'] = heatmap_path

        # Combine reports
        final_report = quality_report
        final_report['geometric_summary'] = geometric_report

        # Preprocessing
        self._preprocess_images(images, quality_report)

        # 3D Reconstruction Preview
        if self.run_reconstruction:
            reconstruction_summary = self.reconstruction_analyzer.run_reconstruction()
            final_report['reconstruction_summary'] = reconstruction_summary

        # Feedback Generation
        feedback = self.feedback_generator.generate_feedback(final_report)
        final_report['feedback'] = feedback

        return final_report

    def _perform_quality_analysis(self, images):
        report = {
            "summary": {
                "total_images": len(images),
                "valid_images": 0,
                "issues_found": 0
            },
            "image_details": []
        }

        for image_name, image in images.items():
            analysis_result = self.quality_analyzer.analyze(image_name, image)
            report["image_details"].append(analysis_result)

            if all(analysis_result["checks"].values()):
                report["summary"]["valid_images"] += 1
            else:
                report["summary"]["issues_found"] += 1
        
        return report

    def _preprocess_images(self, images, report):
        """Applies preprocessing to images that need it and saves them."""
        for image_details in report['image_details']:
            image_name = image_details['image_name']
            image = images[image_name]
            corrected_image = image

            if not image_details['checks']['exposure_ok']:
                corrected_image = self.preprocessor.correct_exposure(corrected_image)

            if not image_details['checks']['sharpness_ok']:
                corrected_image = self.preprocessor.adaptive_sharpen(corrected_image)

            # Save the corrected image if any preprocessing was applied
            if corrected_image != image:
                save_path = self.preprocessed_dir / image_name
                corrected_image.save(save_path)
                print(f"Saved preprocessed image: {save_path}")

    def print_report(self, report):
        """Prints a formatted validation report."""
        if not report:
            return
        print(json.dumps(report, indent=4))


from pathlib import Path
import pycolmap

class ReconstructionAnalyzer:
    """Performs a sparse 3D reconstruction using COLMAP."""

    def __init__(self, config, dataset_path):
        self.config = config
        self.dataset_path = Path(dataset_path)
        self.output_dir = self.dataset_path.parent / (self.dataset_path.name + "_reconstruction")
        self.output_dir.mkdir(exist_ok=True)

    def run_reconstruction(self):
        """Runs the full COLMAP reconstruction pipeline."""
        db_path = self.output_dir / "database.db"
        image_dir = self.dataset_path
        output_model_dir = self.output_dir / "sparse"

        try:
            # Feature extraction
            pycolmap.extract_features(db_path, image_dir)

            # Feature matching
            pycolmap.match_exhaustive(db_path)

            # Incremental reconstruction
            maps = pycolmap.incremental_mapping(db_path, image_dir, output_model_dir)

            # Get reconstruction summary
            if maps:
                summary = {
                    "num_reconstructions": len(maps),
                    "best_reconstruction": self._get_reconstruction_summary(maps[0])
                }
            else:
                summary = {"num_reconstructions": 0}

            return summary

        except Exception as e:
            return {"error": str(e)}

    def _get_reconstruction_summary(self, reconstruction):
        """Extracts key metrics from a COLMAP reconstruction."""
        return {
            "num_reg_images": reconstruction.num_reg_images(),
            "num_points": reconstruction.num_points3D(),
            "mean_track_length": reconstruction.compute_mean_track_length(),
            "mean_observations_per_image": reconstruction.compute_mean_observations_per_reg_image(),
            "mean_reprojection_error": reconstruction.compute_mean_reprojection_error()
        }

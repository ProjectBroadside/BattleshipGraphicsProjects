
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class Visualizer:
    """Generates visualizations for the validation report."""

    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_coverage_heatmap(self, poses):
        """Generates and saves a coverage heatmap from camera poses."""
        if not poses:
            return None

        angles = [p['angle'] for p in poses]
        heights = [p['height'] for p in poses]

        fig, ax = plt.subplots(figsize=(10, 5))
        hist, xedges, yedges = np.histogram2d(angles, heights, bins=[36, 10])

        # Plot the heatmap
        c = ax.pcolormesh(xedges, yedges, hist.T, cmap='viridis')
        fig.colorbar(c, ax=ax, label='Number of Images')

        ax.set_title('Camera Coverage Heatmap')
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Height (meters)')

        heatmap_path = self.output_dir / "coverage_heatmap.png"
        plt.savefig(heatmap_path)
        plt.close(fig)

        return str(heatmap_path)

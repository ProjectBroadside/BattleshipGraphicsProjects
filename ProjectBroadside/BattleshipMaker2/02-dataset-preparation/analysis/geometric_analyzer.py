
import numpy as np

class GeometricAnalyzer:
    """Analyzes the geometric properties of camera poses."""

    def __init__(self, config):
        self.config = config

    def analyze_coverage(self, poses):
        """Analyzes the distribution of camera angles and heights."""
        if not poses:
            return {"angle_distribution": [], "height_distribution": [], "angle_gaps": []}

        angles = sorted([p['angle'] for p in poses])
        heights = [p['height'] for p in poses]

        angle_gaps = self._find_angle_gaps(angles)

        return {
            "angle_distribution": angles,
            "height_distribution": heights,
            "angle_gaps": angle_gaps
        }

    def _find_angle_gaps(self, angles):
        """Finds gaps in the camera angles that exceed the configured threshold."""
        max_gap = self.config.get('geometric_validation.min_angle_gap', 45.0)
        gaps = []
        for i in range(len(angles) - 1):
            gap = angles[i+1] - angles[i]
            if gap > max_gap:
                gaps.append({"start_angle": angles[i], "end_angle": angles[i+1], "gap_size": gap})
        
        # Check the gap between the last and first angle (wraparound)
        if len(angles) > 1:
            wraparound_gap = (360 - angles[-1]) + angles[0]
            if wraparound_gap > max_gap:
                gaps.append({"start_angle": angles[-1], "end_angle": angles[0], "gap_size": wraparound_gap})

        return gaps

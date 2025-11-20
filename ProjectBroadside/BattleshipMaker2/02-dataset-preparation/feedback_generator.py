
class FeedbackGenerator:
    """Generates feedback and recommendations based on the validation report."""

    def __init__(self, config):
        self.config = config

    def generate_feedback(self, report):
        """Analyzes the report and generates a list of recommendations."""
        feedback = []
        
        # Analyze quality issues
        quality_feedback = self._analyze_quality_patterns(report['image_details'])
        if quality_feedback:
            feedback.extend(quality_feedback)

        # Analyze geometric issues
        geometric_feedback = self._analyze_geometric_patterns(report['geometric_summary'])
        if geometric_feedback:
            feedback.extend(geometric_feedback)

        return feedback

    def _analyze_quality_patterns(self, image_details):
        """Identifies common quality issues and suggests improvements."""
        feedback = []
        total_images = len(image_details)
        if total_images == 0: return []

        # Count issues
        blur_count = sum(1 for d in image_details if not d['checks']['sharpness_ok'])
        exposure_count = sum(1 for d in image_details if not d['checks']['exposure_ok'])

        # Generate feedback based on issue prevalence
        if blur_count / total_images > 0.3:
            feedback.append("High prevalence of blurry images. Consider increasing the sharpness in your generation prompts or using a better quality model.")
        
        if exposure_count / total_images > 0.3:
            feedback.append("High prevalence of exposure issues. Check your lighting prompts and ensure they are consistent.")

        return feedback

    def _analyze_geometric_patterns(self, geometric_summary):
        """Identifies geometric issues and suggests improvements."""
        feedback = []
        if geometric_summary['angle_gaps']:
            feedback.append("Significant gaps in camera angles were detected. Consider adding more camera angles to cover these gaps for better 3D reconstruction.")
            for gap in geometric_summary['angle_gaps']:
                feedback.append(f"  - Gap detected between {gap['start_angle']} and {gap['end_angle']} degrees.")
        
        return feedback

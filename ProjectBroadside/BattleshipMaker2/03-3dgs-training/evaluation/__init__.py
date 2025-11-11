"""Evaluation module for 3D Gaussian Splatting."""

from .metrics import (
    evaluate_model,
    compute_perceptual_metrics,
    evaluate_reconstruction_quality,
    MetricTracker,
    benchmark_rendering_speed
)

__all__ = [
    'evaluate_model',
    'compute_perceptual_metrics', 
    'evaluate_reconstruction_quality',
    'MetricTracker',
    'benchmark_rendering_speed'
]
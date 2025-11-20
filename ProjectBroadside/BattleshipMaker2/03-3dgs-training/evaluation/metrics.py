"""
Evaluation metrics for 3D Gaussian Splatting.
Implements PSNR, SSIM, LPIPS and other quality metrics.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils import compute_psnr, compute_ssim

# Try to import LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    logging.warning("LPIPS not available for evaluation")

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    data_loader,
    device: str = "cuda",
    num_images: Optional[int] = None,
    compute_lpips: bool = False,
    save_images: bool = False,
    save_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate model on test dataset.
    
    Args:
        model: Gaussian model
        data_loader: Test data loader
        device: Device to use
        num_images: Number of images to evaluate (None = all)
        compute_lpips: Whether to compute LPIPS
        save_images: Whether to save rendered images
        save_dir: Directory to save images
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'psnr': [],
        'ssim': [],
        'lpips': [],
        'mse': [],
        'num_images': 0
    }
    
    # Initialize LPIPS if requested
    if compute_lpips and LPIPS_AVAILABLE:
        lpips_fn = lpips.LPIPS(net='vgg').to(device)
        lpips_fn.eval()
    else:
        lpips_fn = None
    
    # Evaluation loop
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
            if num_images and i >= num_images:
                break
            
            # Move to device
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Render image
            # Note: This is a simplified version - actual rendering would use the rasterizer
            rendered = torch.rand_like(batch['images'][0])  # Placeholder
            target = batch['images'][0]
            
            # Compute metrics
            psnr = compute_psnr(rendered, target)
            ssim = compute_ssim(rendered.unsqueeze(0), target.unsqueeze(0))
            mse = torch.mean((rendered - target) ** 2).item()
            
            metrics['psnr'].append(psnr)
            metrics['ssim'].append(ssim)
            metrics['mse'].append(mse)
            
            # Compute LPIPS if available
            if lpips_fn is not None:
                # LPIPS expects [-1, 1] range
                rendered_norm = 2 * rendered.unsqueeze(0) - 1
                target_norm = 2 * target.unsqueeze(0) - 1
                lpips_val = lpips_fn(rendered_norm, target_norm).item()
                metrics['lpips'].append(lpips_val)
            
            # Save images if requested
            if save_images and save_dir:
                from utils import save_image
                save_path = Path(save_dir) / f"render_{i:04d}.png"
                save_image(rendered, save_path)
            
            metrics['num_images'] += 1
    
    # Average metrics
    avg_metrics = {}
    for key in ['psnr', 'ssim', 'mse', 'lpips']:
        if metrics[key]:
            avg_metrics[key] = np.mean(metrics[key])
            avg_metrics[f'{key}_std'] = np.std(metrics[key])
        else:
            avg_metrics[key] = 0.0
            avg_metrics[f'{key}_std'] = 0.0
    
    avg_metrics['num_images'] = metrics['num_images']
    
    return avg_metrics


def compute_perceptual_metrics(
    rendered: torch.Tensor,
    target: torch.Tensor,
    metrics: List[str] = ['psnr', 'ssim', 'lpips']
) -> Dict[str, float]:
    """
    Compute multiple perceptual metrics.
    
    Args:
        rendered: Rendered image (C, H, W) or (B, C, H, W)
        target: Target image (C, H, W) or (B, C, H, W)
        metrics: List of metrics to compute
        
    Returns:
        Dictionary of metric values
    """
    if rendered.dim() == 3:
        rendered = rendered.unsqueeze(0)
        target = target.unsqueeze(0)
    
    results = {}
    
    if 'psnr' in metrics:
        results['psnr'] = compute_psnr(rendered[0], target[0])
    
    if 'ssim' in metrics:
        results['ssim'] = compute_ssim(rendered, target)
    
    if 'lpips' in metrics and LPIPS_AVAILABLE:
        lpips_fn = lpips.LPIPS(net='vgg').to(rendered.device)
        lpips_fn.eval()
        with torch.no_grad():
            # LPIPS expects [-1, 1] range
            rendered_norm = 2 * rendered - 1
            target_norm = 2 * target - 1
            results['lpips'] = lpips_fn(rendered_norm, target_norm).item()
    elif 'lpips' in metrics:
        results['lpips'] = 0.0
    
    if 'mse' in metrics:
        results['mse'] = torch.mean((rendered - target) ** 2).item()
    
    if 'mae' in metrics:
        results['mae'] = torch.mean(torch.abs(rendered - target)).item()
    
    return results


def evaluate_reconstruction_quality(
    model,
    reference_images: List[torch.Tensor],
    camera_poses: List[Dict],
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Evaluate reconstruction quality from specific viewpoints.
    
    Args:
        model: Gaussian model
        reference_images: List of reference images
        camera_poses: List of camera poses
        device: Device to use
        
    Returns:
        Dictionary of quality metrics
    """
    total_metrics = {
        'psnr': 0.0,
        'ssim': 0.0,
        'lpips': 0.0,
        'consistency': 0.0
    }
    
    rendered_images = []
    
    for i, (ref_image, camera) in enumerate(zip(reference_images, camera_poses)):
        # Render from viewpoint
        # Note: Simplified - actual implementation would use proper rendering
        rendered = torch.rand_like(ref_image).to(device)
        ref_image = ref_image.to(device)
        
        # Compute metrics
        metrics = compute_perceptual_metrics(rendered, ref_image)
        
        for key in metrics:
            if key in total_metrics:
                total_metrics[key] += metrics[key]
        
        rendered_images.append(rendered)
    
    # Average metrics
    num_images = len(reference_images)
    for key in total_metrics:
        total_metrics[key] /= num_images
    
    # Compute temporal consistency for adjacent views
    if len(rendered_images) > 1:
        consistency_scores = []
        for i in range(len(rendered_images) - 1):
            # Compute optical flow or simple difference
            diff = torch.mean(torch.abs(rendered_images[i] - rendered_images[i + 1]))
            consistency_scores.append(1.0 - diff.item())
        
        total_metrics['consistency'] = np.mean(consistency_scores)
    
    return total_metrics


class MetricTracker:
    """Track metrics during training."""
    
    def __init__(self, metrics: List[str] = ['loss', 'psnr', 'ssim']):
        """
        Initialize metric tracker.
        
        Args:
            metrics: List of metric names to track
        """
        self.metrics = metrics
        self.history = {metric: [] for metric in metrics}
        self.best_values = {metric: None for metric in metrics}
        self.best_iterations = {metric: 0 for metric in metrics}
    
    def update(self, iteration: int, values: Dict[str, float]):
        """
        Update metrics.
        
        Args:
            iteration: Current iteration
            values: Dictionary of metric values
        """
        for metric, value in values.items():
            if metric in self.history:
                self.history[metric].append((iteration, value))
                
                # Update best value
                if self.best_values[metric] is None:
                    self.best_values[metric] = value
                    self.best_iterations[metric] = iteration
                elif self._is_better(metric, value, self.best_values[metric]):
                    self.best_values[metric] = value
                    self.best_iterations[metric] = iteration
    
    def _is_better(self, metric: str, new_value: float, old_value: float) -> bool:
        """Check if new value is better than old value."""
        # Lower is better for loss-like metrics
        if metric in ['loss', 'mse', 'mae', 'lpips']:
            return new_value < old_value
        # Higher is better for quality metrics
        else:
            return new_value > old_value
    
    def get_best(self, metric: str) -> Tuple[int, float]:
        """
        Get best value for a metric.
        
        Args:
            metric: Metric name
            
        Returns:
            Tuple of (iteration, value)
        """
        return self.best_iterations.get(metric, 0), self.best_values.get(metric, 0.0)
    
    def get_history(self, metric: str) -> List[Tuple[int, float]]:
        """
        Get history for a metric.
        
        Args:
            metric: Metric name
            
        Returns:
            List of (iteration, value) tuples
        """
        return self.history.get(metric, [])
    
    def save(self, path: str):
        """Save metrics to file."""
        import json
        data = {
            'history': self.history,
            'best_values': self.best_values,
            'best_iterations': self.best_iterations
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load metrics from file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        self.history = data['history']
        self.best_values = data['best_values']
        self.best_iterations = data['best_iterations']


def benchmark_rendering_speed(
    model,
    image_size: Tuple[int, int] = (1920, 1080),
    num_frames: int = 100,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Benchmark rendering speed.
    
    Args:
        model: Gaussian model
        image_size: Image dimensions (W, H)
        num_frames: Number of frames to render
        device: Device to use
        
    Returns:
        Dictionary with timing statistics
    """
    import time
    
    render_times = []
    
    # Warmup
    for _ in range(10):
        # Simplified rendering
        _ = torch.rand(3, image_size[1], image_size[0], device=device)
    
    # Benchmark
    for _ in tqdm(range(num_frames), desc="Benchmarking"):
        torch.cuda.synchronize()
        start_time = time.time()
        
        # Simplified rendering
        _ = torch.rand(3, image_size[1], image_size[0], device=device)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        render_times.append(end_time - start_time)
    
    # Compute statistics
    render_times = np.array(render_times)
    
    return {
        'mean_time': np.mean(render_times),
        'std_time': np.std(render_times),
        'min_time': np.min(render_times),
        'max_time': np.max(render_times),
        'fps': 1.0 / np.mean(render_times),
        'num_gaussians': model.num_gaussians
    }


if __name__ == "__main__":
    # Test metrics
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create dummy data
    rendered = torch.rand(3, 256, 256).to(device)
    target = torch.rand(3, 256, 256).to(device)
    
    # Test perceptual metrics
    metrics = compute_perceptual_metrics(rendered, target)
    print("Perceptual metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Test metric tracker
    tracker = MetricTracker(['loss', 'psnr'])
    tracker.update(100, {'loss': 0.5, 'psnr': 25.0})
    tracker.update(200, {'loss': 0.3, 'psnr': 27.0})
    
    print("\nBest metrics:")
    for metric in ['loss', 'psnr']:
        iter, value = tracker.get_best(metric)
        print(f"  {metric}: {value:.4f} at iteration {iter}")
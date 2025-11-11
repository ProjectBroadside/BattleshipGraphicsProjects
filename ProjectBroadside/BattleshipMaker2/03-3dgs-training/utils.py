"""
Utility functions for 3D Gaussian Splatting training.
Common helper functions used across the module.
"""

import os
import sys
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import json
import yaml

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def seed_everything(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logging.info(f"Set random seed to {seed}")


def create_output_dir(base_dir: Union[str, Path], experiment_name: str) -> Path:
    """
    Create output directory with timestamp.
    
    Args:
        base_dir: Base output directory
        experiment_name: Name of the experiment
        
    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "renders").mkdir(exist_ok=True)
    (output_dir / "metrics").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    
    logging.info(f"Created output directory: {output_dir}")
    
    return output_dir


def save_image(tensor: torch.Tensor, path: Union[str, Path], normalize: bool = True):
    """
    Save tensor as image.
    
    Args:
        tensor: Image tensor (C, H, W) or (H, W)
        path: Save path
        normalize: Whether to normalize to [0, 1]
    """
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)
    
    if normalize:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
    
    tensor = torch.clamp(tensor * 255, 0, 255).byte()
    
    if tensor.dim() == 2:
        mode = 'L'
    else:
        mode = 'RGB'
    
    image = Image.fromarray(tensor.cpu().numpy(), mode=mode)
    image.save(path)


def load_image(path: Union[str, Path], device: str = "cpu") -> torch.Tensor:
    """
    Load image as tensor.
    
    Args:
        path: Image path
        device: Device to load to
        
    Returns:
        Image tensor (C, H, W) in [0, 1] range
    """
    image = Image.open(path).convert('RGB')
    tensor = torch.from_numpy(np.array(image)).float() / 255.0
    tensor = tensor.permute(2, 0, 1).to(device)
    return tensor


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.
    
    Args:
        pred: Predicted image
        target: Target image
        max_val: Maximum pixel value
        
    Returns:
        PSNR value in dB
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse)).item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Structural Similarity Index.
    
    Args:
        pred: Predicted image (B, C, H, W)
        target: Target image (B, C, H, W)
        
    Returns:
        SSIM value
    """
    from training.loss_functions import SSIM
    ssim = SSIM()
    return ssim(pred, target).item()


def visualize_gaussians(
    positions: torch.Tensor,
    colors: Optional[torch.Tensor] = None,
    opacities: Optional[torch.Tensor] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Gaussian Positions"
):
    """
    Visualize Gaussian positions in 3D.
    
    Args:
        positions: Gaussian positions (N, 3)
        colors: Optional colors (N, 3)
        opacities: Optional opacities (N, 1)
        save_path: Optional save path
        title: Plot title
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert to numpy
    pos = positions.detach().cpu().numpy()
    
    # Prepare colors
    if colors is not None:
        c = colors.detach().cpu().numpy()
    else:
        c = 'blue'
    
    # Prepare sizes based on opacity
    if opacities is not None:
        sizes = opacities.detach().cpu().numpy().squeeze() * 50
    else:
        sizes = 1
    
    # Plot
    scatter = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=c, s=sizes, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Equal aspect ratio
    max_range = np.array([pos[:, 0].max() - pos[:, 0].min(),
                         pos[:, 1].max() - pos[:, 1].min(),
                         pos[:, 2].max() - pos[:, 2].min()]).max() / 2.0
    
    mid_x = (pos[:, 0].max() + pos[:, 0].min()) * 0.5
    mid_y = (pos[:, 1].max() + pos[:, 1].min()) * 0.5
    mid_z = (pos[:, 2].max() + pos[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def create_video_from_images(
    image_dir: Union[str, Path],
    output_path: Union[str, Path],
    fps: int = 30,
    pattern: str = "*.png"
):
    """
    Create video from directory of images.
    
    Args:
        image_dir: Directory containing images
        output_path: Output video path
        fps: Frames per second
        pattern: File pattern to match
    """
    import imageio
    
    image_dir = Path(image_dir)
    images = sorted(image_dir.glob(pattern))
    
    if not images:
        logging.warning(f"No images found in {image_dir} with pattern {pattern}")
        return
    
    writer = imageio.get_writer(output_path, fps=fps)
    
    for img_path in tqdm(images, desc="Creating video"):
        img = imageio.imread(img_path)
        writer.append_data(img)
    
    writer.close()
    logging.info(f"Created video: {output_path}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def save_metrics(metrics: Dict[str, Any], save_path: Union[str, Path]):
    """
    Save metrics to JSON/YAML file.
    
    Args:
        metrics: Dictionary of metrics
        save_path: Save path
    """
    save_path = Path(save_path)
    
    # Convert numpy values to Python types
    def convert_to_python(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python(v) for v in obj]
        return obj
    
    metrics = convert_to_python(metrics)
    
    with open(save_path, 'w') as f:
        if save_path.suffix in ['.yaml', '.yml']:
            yaml.dump(metrics, f, default_flow_style=False)
        else:
            json.dump(metrics, f, indent=2)


def load_metrics(load_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load metrics from JSON/YAML file.
    
    Args:
        load_path: Load path
        
    Returns:
        Dictionary of metrics
    """
    load_path = Path(load_path)
    
    with open(load_path, 'r') as f:
        if load_path.suffix in ['.yaml', '.yml']:
            metrics = yaml.safe_load(f)
        else:
            metrics = json.load(f)
    
    return metrics


def profile_memory(func):
    """
    Decorator to profile GPU memory usage.
    
    Usage:
        @profile_memory
        def my_function():
            ...
    """
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start_memory = torch.cuda.memory_allocated()
        
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            
            logging.info(f"{func.__name__} memory usage:")
            logging.info(f"  Start: {start_memory / 1024**2:.2f} MB")
            logging.info(f"  End: {end_memory / 1024**2:.2f} MB")
            logging.info(f"  Peak: {peak_memory / 1024**2:.2f} MB")
            logging.info(f"  Allocated: {(end_memory - start_memory) / 1024**2:.2f} MB")
        
        return result
    
    return wrapper


class EarlyStopping:
    """Early stopping helper class."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max"
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement
            min_delta: Minimum change to qualify as improvement
            mode: "min" or "max"
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if should stop.
        
        Args:
            score: Current score
            
        Returns:
            True if should stop
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score: float) -> bool:
        """Check if score is an improvement."""
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta


def create_spiral_path(
    n_frames: int = 120,
    n_rounds: int = 2,
    radius: float = 4.0,
    height_variation: float = 0.5,
    center: List[float] = [0, 0, 0]
) -> List[Dict[str, Any]]:
    """
    Create spiral camera path for rendering.
    
    Args:
        n_frames: Number of frames
        n_rounds: Number of rounds around object
        radius: Camera distance from center
        height_variation: Vertical movement range
        center: Center point
        
    Returns:
        List of camera poses
    """
    poses = []
    
    for i in range(n_frames):
        angle = 2 * np.pi * n_rounds * i / n_frames
        height = center[1] + height_variation * np.sin(2 * np.pi * i / n_frames)
        
        # Camera position
        x = center[0] + radius * np.cos(angle)
        y = height
        z = center[2] + radius * np.sin(angle)
        
        # Look at center
        cam_pos = np.array([x, y, z])
        forward = np.array(center) - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        # Compute right and up vectors
        right = np.cross([0, 1, 0], forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)
        
        # Build transformation matrix
        transform = np.eye(4)
        transform[:3, 0] = right
        transform[:3, 1] = up
        transform[:3, 2] = -forward
        transform[:3, 3] = cam_pos
        
        poses.append({
            'transform_matrix': transform,
            'camera_position': cam_pos,
            'frame_id': i
        })
    
    return poses


if __name__ == "__main__":
    # Test utilities
    setup_logging("INFO")
    seed_everything(42)
    
    # Test output directory creation
    output_dir = create_output_dir("./test_outputs", "test_experiment")
    print(f"Created output directory: {output_dir}")
    
    # Test parameter counting
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    params = count_parameters(model)
    print(f"Model parameters: {params}")
    
    # Test metrics saving/loading
    test_metrics = {
        'psnr': 28.5,
        'ssim': 0.92,
        'iteration': 10000
    }
    save_metrics(test_metrics, output_dir / "test_metrics.json")
    loaded_metrics = load_metrics(output_dir / "test_metrics.json")
    print(f"Loaded metrics: {loaded_metrics}")
    
    print("\nUtility tests completed!")
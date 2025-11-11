"""
Main training script for 3D Gaussian Splatting.
Implements the complete training pipeline for battleship reconstruction.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config, load_config, save_config, merge_configs
from data.data_loader import get_data_loader
from models.model import GaussianModel
from training.loss_functions import CombinedLoss, AdaptiveDensityController, psnr
from evaluation.metrics import evaluate_model
from utils import setup_logging, seed_everything, create_output_dir

# Try to import rendering module
try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    RASTERIZER_AVAILABLE = True
except ImportError:
    RASTERIZER_AVAILABLE = False
    logging.warning("Gaussian rasterizer not available. Please install diff-gaussian-rasterization")


class Trainer:
    """Main trainer class for 3D Gaussian Splatting."""
    
    def __init__(self, config: Config):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        self.output_dir = create_output_dir(config.output_dir, config.experiment_name)
        
        # Save configuration
        save_config(config, self.output_dir / "config.yaml")
        
        # Initialize tensorboard
        if config.tensorboard:
            self.writer = SummaryWriter(self.output_dir / "tensorboard")
        else:
            self.writer = None
        
        # Initialize components
        self._setup_data()
        self._setup_model()
        self._setup_optimization()
        self._setup_loss()
        self._setup_densification()
        
        # Training state
        self.iteration = 0
        self.best_psnr = 0.0
        self.patience_counter = 0
    
    def _setup_data(self):
        """Setup data loaders."""
        self.logger.info("Setting up data loaders...")
        
        self.train_loader = get_data_loader(
            self.config.data.data_path,
            split="train",
            batch_size=self.config.training.batch_size,
            num_workers=self.config.data.num_workers,
            resolution=self.config.data.resolution,
            scale_factor=self.config.data.resolution_scale,
            white_background=self.config.data.white_background,
            cache_images=self.config.data.cache_images
        )
        
        self.test_loader = get_data_loader(
            self.config.data.data_path,
            split="test",
            batch_size=1,
            num_workers=2,
            shuffle=False,
            resolution=self.config.data.resolution,
            scale_factor=self.config.data.resolution_scale,
            white_background=self.config.data.white_background,
            cache_images=False
        )
        
        self.logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Test samples: {len(self.test_loader.dataset)}")
    
    def _setup_model(self):
        """Setup Gaussian model."""
        self.logger.info("Setting up model...")
        
        # Initialize with random points or from point cloud
        self.model = GaussianModel(
            sh_degree=self.config.model.sh_degree
        ).to(self.device)
        
        # Initialize from dataset if available
        if hasattr(self.train_loader.dataset, 'point_cloud'):
            points = self.train_loader.dataset.point_cloud
            colors = self.train_loader.dataset.point_colors
            self.model.create_from_pcd({
                'points': points,
                'colors': colors
            })
        else:
            # Random initialization
            num_points = self.config.model.init_points
            points = (torch.rand((num_points, 3)) - 0.5) * 2 * self.config.model.scene_extent
            self.model._init_from_points(points.to(self.device))
        
        self.logger.info(f"Initialized model with {self.model.num_gaussians} Gaussians")
    
    def _setup_optimization(self):
        """Setup optimizer and schedulers."""
        self.logger.info("Setting up optimization...")
        
        # Create parameter groups
        param_groups = [
            {
                'params': [self.model._xyz],
                'lr': self.config.training.position_lr_init,
                'name': 'xyz'
            },
            {
                'params': [self.model._features_dc],
                'lr': self.config.training.feature_lr,
                'name': 'f_dc'
            },
            {
                'params': [self.model._features_rest],
                'lr': self.config.training.feature_lr / 20.0,
                'name': 'f_rest'
            },
            {
                'params': [self.model._opacity],
                'lr': self.config.training.opacity_lr,
                'name': 'opacity'
            },
            {
                'params': [self.model._scaling],
                'lr': self.config.training.scaling_lr,
                'name': 'scaling'
            },
            {
                'params': [self.model._rotation],
                'lr': self.config.training.rotation_lr,
                'name': 'rotation'
            }
        ]
        
        # Create optimizer
        if self.config.optimization.optimizer == "adam":
            self.optimizer = optim.Adam(
                param_groups,
                betas=(self.config.optimization.adam_beta1, self.config.optimization.adam_beta2),
                eps=self.config.optimization.adam_eps,
                weight_decay=self.config.optimization.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimization.optimizer}")
        
        # Store optimizer in model for densification
        self.model.optimizer = self.optimizer
        
        # Setup learning rate scheduler for positions
        self.position_lr_scheduler = self._get_position_lr_scheduler()
    
    def _get_position_lr_scheduler(self):
        """Create position learning rate scheduler."""
        def lr_lambda(iter):
            if iter < 0:
                return 1.0
            
            delay_mult = self.config.training.position_lr_delay_mult
            delay_steps = self.config.training.position_lr_max_steps * delay_mult
            
            if iter < delay_steps:
                return 0.01
            else:
                progress = min((iter - delay_steps) / (self.config.training.position_lr_max_steps - delay_steps), 1.0)
                decay = (1 - progress) * (self.config.training.position_lr_init - self.config.training.position_lr_final)
                return (self.config.training.position_lr_final + decay) / self.config.training.position_lr_init
        
        return lr_lambda
    
    def _setup_loss(self):
        """Setup loss function."""
        self.loss_fn = CombinedLoss(
            l1_weight=self.config.loss.l1_weight,
            ssim_weight=self.config.loss.ssim_weight,
            lpips_weight=self.config.loss.lpips_weight,
            opacity_reg_weight=self.config.loss.opacity_reg_weight,
            scale_reg_weight=self.config.loss.scale_reg_weight,
            device=self.device
        )
    
    def _setup_densification(self):
        """Setup adaptive density controller."""
        self.density_controller = AdaptiveDensityController(
            densify_grad_threshold=self.config.densification.grad_threshold,
            densify_size_threshold=self.config.densification.size_threshold_scale * self.config.model.scene_extent,
            opacity_threshold=self.config.densification.min_opacity,
            scene_extent=self.config.model.scene_extent,
            max_gaussians=self.config.model.max_gaussians
        )
    
    def render(self, viewpoint_camera: Dict) -> torch.Tensor:
        """
        Render image from viewpoint.
        
        Args:
            viewpoint_camera: Camera parameters
            
        Returns:
            Rendered image
        """
        if not RASTERIZER_AVAILABLE:
            # Fallback to simple rendering
            return torch.zeros((3, viewpoint_camera['height'], viewpoint_camera['width']), device=self.device)
        
        # Create rasterization settings
        tanfovx = np.tan(viewpoint_camera['fovx'] * 0.5)
        tanfovy = np.tan(viewpoint_camera['fovy'] * 0.5)
        
        raster_settings = GaussianRasterizationSettings(
            image_height=viewpoint_camera['height'],
            image_width=viewpoint_camera['width'],
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.tensor([1, 1, 1] if self.config.data.white_background else [0, 0, 0], device=self.device),
            scale_modifier=1.0,
            viewmatrix=viewpoint_camera['viewmatrix'],
            projmatrix=viewpoint_camera['projmatrix'],
            sh_degree=self.model.active_sh_degree,
            campos=viewpoint_camera['campos'],
            prefiltered=False,
            debug=False
        )
        
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        # Get Gaussian parameters
        means3D = self.model.get_xyz
        opacity = self.model.get_opacity
        scales = self.model.get_scaling
        rotations = self.model.get_rotation
        shs = self.model.get_features
        
        # Render
        rendered_image, radii = rasterizer(
            means3D=means3D,
            means2D=None,
            shs=shs,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None
        )
        
        # Update radii for densification
        self.model.max_radii2D[radii > 0] = torch.max(self.model.max_radii2D[radii > 0], radii[radii > 0])
        
        return rendered_image
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Dictionary of losses
        """
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Get random camera
        idx = random.randint(0, len(batch['images']) - 1)
        
        # Prepare camera
        viewpoint_camera = {
            'viewmatrix': batch['world_matrices'][idx],
            'projmatrix': self._compute_projection_matrix(batch['camera_matrices'][idx]),
            'campos': self._get_camera_position(batch['world_matrices'][idx]),
            'width': batch['images'][idx].shape[2],
            'height': batch['images'][idx].shape[1],
            'fovx': self._compute_fov(batch['camera_matrices'][idx], batch['images'][idx].shape[2]),
            'fovy': self._compute_fov(batch['camera_matrices'][idx], batch['images'][idx].shape[1])
        }
        
        # Render
        rendered = self.render(viewpoint_camera)
        
        # Get ground truth
        gt_image = batch['images'][idx]
        
        # Compute loss
        loss, losses_dict = self.loss_fn(
            rendered.unsqueeze(0),
            gt_image.unsqueeze(0),
            opacity=self.model.get_opacity if self.config.loss.opacity_reg_weight > 0 else None,
            scales=self.model.get_scaling if self.config.loss.scale_reg_weight > 0 else None
        )
        
        # Backward
        loss.backward()
        
        # Accumulate gradients for densification
        if self.iteration < self.config.densification.end_iter:
            self.model.xyz_gradient_accum += torch.norm(self.model._xyz.grad, dim=1, keepdim=True)
            self.model.denom += 1
        
        # Gradient clipping
        if self.config.training.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.gradient_clip_val,
                norm_type=self.config.training.gradient_clip_norm_type
            )
        
        # Optimizer step
        self.optimizer.step()
        
        # Update learning rate
        self._update_learning_rate()
        
        return losses_dict
    
    def _update_learning_rate(self):
        """Update learning rates based on schedule."""
        # Position learning rate
        lr_scale = self.position_lr_scheduler(self.iteration)
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'xyz':
                param_group['lr'] = self.config.training.position_lr_init * lr_scale
    
    def _compute_projection_matrix(self, intrinsics: torch.Tensor) -> torch.Tensor:
        """Compute projection matrix from intrinsics."""
        # Simplified projection matrix computation
        # This would need proper implementation based on your camera model
        fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
        
        # Create projection matrix
        P = torch.zeros(4, 4, device=intrinsics.device)
        P[0, 0] = 2 * fx / cx
        P[1, 1] = 2 * fy / cy
        P[2, 2] = 1
        P[3, 3] = 0
        P[2, 3] = 1
        
        return P
    
    def _get_camera_position(self, world_matrix: torch.Tensor) -> torch.Tensor:
        """Extract camera position from world matrix."""
        # Camera position is the translation part of the inverse world matrix
        return -world_matrix[:3, :3].T @ world_matrix[:3, 3]
    
    def _compute_fov(self, intrinsics: torch.Tensor, dimension: int) -> float:
        """Compute field of view from intrinsics."""
        focal = intrinsics[0, 0] if dimension == intrinsics.shape[1] else intrinsics[1, 1]
        return 2 * np.arctan(dimension / (2 * focal))
    
    def densify(self):
        """Perform adaptive densification."""
        if (self.iteration >= self.config.densification.start_iter and 
            self.iteration < self.config.densification.end_iter and
            self.iteration % self.config.densification.interval == 0):
            
            stats = self.density_controller.densify_and_prune(
                self.model,
                grad_threshold=self.config.densification.grad_threshold,
                min_opacity=self.config.densification.min_opacity,
                max_screen_size=self.config.densification.max_screen_size
            )
            
            self.logger.info(
                f"Densification at iter {self.iteration}: "
                f"cloned={stats['cloned']}, split={stats['split']}, "
                f"pruned={stats['pruned']}, total={stats['final_count']}"
            )
            
            if self.writer:
                self.writer.add_scalar('densification/num_gaussians', stats['final_count'], self.iteration)
                self.writer.add_scalar('densification/cloned', stats['cloned'], self.iteration)
                self.writer.add_scalar('densification/split', stats['split'], self.iteration)
                self.writer.add_scalar('densification/pruned', stats['pruned'], self.iteration)
    
    def reset_opacity(self):
        """Reset opacity values periodically."""
        if (self.config.densification.opacity_reset_interval > 0 and
            self.iteration % self.config.densification.opacity_reset_interval == 0):
            self.model.reset_opacity()
            self.logger.info(f"Reset opacity at iteration {self.iteration}")
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on test set."""
        self.model.eval()
        metrics = evaluate_model(
            self.model,
            self.test_loader,
            self.device,
            num_images=self.config.evaluation.num_test_images
        )
        self.model.train()
        return metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'iteration': self.iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'best_psnr': self.best_psnr
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_iter_{self.iteration}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            
        # Save PLY file
        ply_path = self.output_dir / f"point_cloud_iter_{self.iteration}.ply"
        self.model.save_ply(str(ply_path))
        
        self.logger.info(f"Saved checkpoint at iteration {self.iteration}")
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        # Training loop
        pbar = tqdm(total=self.config.training.iterations, desc="Training")
        
        while self.iteration < self.config.training.iterations:
            # Get batch
            for batch in self.train_loader:
                if self.iteration >= self.config.training.iterations:
                    break
                
                # Move to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                # Training step
                losses = self.train_step(batch)
                
                # Update iteration
                self.iteration += 1
                pbar.update(1)
                
                # Logging
                if self.iteration % 10 == 0:
                    pbar.set_postfix(**{k: f"{v:.4f}" for k, v in losses.items()})
                
                if self.writer and self.iteration % 100 == 0:
                    for name, value in losses.items():
                        self.writer.add_scalar(f'train/{name}', value, self.iteration)
                    
                    # Log learning rates
                    for param_group in self.optimizer.param_groups:
                        self.writer.add_scalar(f'lr/{param_group["name"]}', param_group['lr'], self.iteration)
                    
                    # Log number of Gaussians
                    self.writer.add_scalar('model/num_gaussians', self.model.num_gaussians, self.iteration)
                
                # Densification
                self.densify()
                
                # Reset opacity
                self.reset_opacity()
                
                # Increase SH degree
                if self.iteration % 1000 == 0:
                    self.model.oneupSHdegree()
                
                # Evaluation
                if self.iteration % self.config.evaluation.interval == 0:
                    metrics = self.evaluate()
                    
                    self.logger.info(
                        f"Evaluation at iter {self.iteration}: "
                        f"PSNR={metrics['psnr']:.2f} dB, "
                        f"SSIM={metrics['ssim']:.4f}"
                    )
                    
                    if self.writer:
                        for name, value in metrics.items():
                            self.writer.add_scalar(f'eval/{name}', value, self.iteration)
                    
                    # Check for improvement
                    if metrics['psnr'] > self.best_psnr:
                        self.best_psnr = metrics['psnr']
                        self.patience_counter = 0
                        is_best = True
                    else:
                        self.patience_counter += 1
                        is_best = False
                    
                    # Early stopping
                    if self.patience_counter >= self.config.training.early_stop_patience:
                        self.logger.info("Early stopping triggered")
                        break
                
                # Save checkpoint
                if self.iteration % self.config.training.checkpoint_interval == 0:
                    self.save_checkpoint(is_best=False)
        
        pbar.close()
        
        # Final save
        self.save_checkpoint(is_best=False)
        self.logger.info("Training completed!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train 3D Gaussian Splatting")
    parser.add_argument("--config", type=str, default="config/default_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment name")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    
    # Override config values
    parser.add_argument("--iterations", type=int, default=None,
                        help="Number of training iterations")
    parser.add_argument("--sh_degree", type=int, default=None,
                        help="Spherical harmonics degree")
    parser.add_argument("--lr_scale", type=float, default=1.0,
                        help="Learning rate scale factor")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply command line overrides
    overrides = {}
    if args.data_path:
        overrides['data.data_path'] = args.data_path
    if args.output_dir:
        overrides['output_dir'] = args.output_dir
    if args.experiment_name:
        overrides['experiment_name'] = args.experiment_name
    if args.seed is not None:
        overrides['seed'] = args.seed
    if args.iterations is not None:
        overrides['training.iterations'] = args.iterations
    if args.sh_degree is not None:
        overrides['model.sh_degree'] = args.sh_degree
    
    # Apply learning rate scaling
    if args.lr_scale != 1.0:
        for key in ['position_lr_init', 'position_lr_final', 'feature_lr', 
                    'opacity_lr', 'scaling_lr', 'rotation_lr']:
            overrides[f'training.{key}'] = getattr(config.training, key) * args.lr_scale
    
    config = merge_configs(config, overrides)
    
    # Setup logging
    setup_logging(config.log_level)
    
    # Set random seed
    seed_everything(config.seed)
    
    # Create trainer
    trainer = Trainer(config)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
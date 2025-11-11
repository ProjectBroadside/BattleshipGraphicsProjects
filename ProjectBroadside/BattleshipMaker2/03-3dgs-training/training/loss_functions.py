"""
Loss functions for 3D Gaussian Splatting training.
Includes L1, SSIM, LPIPS, and regularization losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np
import logging

# Try to import LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    logging.warning("LPIPS not available. Install with: pip install lpips")

logger = logging.getLogger(__name__)


class SSIM(nn.Module):
    """Structural Similarity Index (SSIM) loss."""
    
    def __init__(self, window_size: int = 11, channel: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self._create_window(window_size, channel)
    
    def _gaussian(self, window_size: int, sigma: float) -> torch.Tensor:
        """Create 1D Gaussian kernel."""
        gauss = torch.Tensor([
            np.exp(-(x - window_size//2)**2 / float(2*sigma**2)) 
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        """Create 2D Gaussian window."""
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(
        self, 
        img1: torch.Tensor, 
        img2: torch.Tensor, 
        data_range: float = 1.0,
        size_average: bool = True
    ) -> torch.Tensor:
        """
        Calculate SSIM between two images.
        
        Args:
            img1: First image (B, C, H, W)
            img2: Second image (B, C, H, W)
            data_range: Value range of input images
            size_average: Average over batch
            
        Returns:
            SSIM value(s)
        """
        channel = img1.size(1)
        window = self.window.to(img1.device).type_as(img1)
        
        if channel == self.channel and window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            window = window.to(img1.device).type_as(img1)
            self.window = window
            self.channel = channel
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        
        ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


class CombinedLoss(nn.Module):
    """Combined loss function for Gaussian Splatting."""
    
    def __init__(
        self,
        l1_weight: float = 0.8,
        ssim_weight: float = 0.2,
        lpips_weight: float = 0.0,
        opacity_reg_weight: float = 0.0,
        scale_reg_weight: float = 0.0,
        device: str = "cuda"
    ):
        """
        Initialize combined loss.
        
        Args:
            l1_weight: Weight for L1 loss
            ssim_weight: Weight for SSIM loss  
            lpips_weight: Weight for LPIPS loss
            opacity_reg_weight: Weight for opacity regularization
            scale_reg_weight: Weight for scale regularization
            device: Device to run on
        """
        super().__init__()
        
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.lpips_weight = lpips_weight
        self.opacity_reg_weight = opacity_reg_weight
        self.scale_reg_weight = scale_reg_weight
        
        # Initialize SSIM
        self.ssim = SSIM().to(device)
        
        # Initialize LPIPS if available
        if lpips_weight > 0 and LPIPS_AVAILABLE:
            self.lpips = lpips.LPIPS(net='vgg').to(device)
            for param in self.lpips.parameters():
                param.requires_grad = False
        else:
            self.lpips = None
            if lpips_weight > 0:
                logger.warning("LPIPS weight specified but LPIPS not available")
    
    def forward(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        opacity: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate combined loss.
        
        Args:
            rendered: Rendered image (B, C, H, W)
            target: Target image (B, C, H, W)
            opacity: Gaussian opacities for regularization
            scales: Gaussian scales for regularization
            
        Returns:
            Total loss and dictionary of individual losses
        """
        losses = {}
        total_loss = 0.0
        
        # L1 loss
        if self.l1_weight > 0:
            l1_loss = F.l1_loss(rendered, target)
            losses['l1'] = l1_loss.item()
            total_loss += self.l1_weight * l1_loss
        
        # SSIM loss
        if self.ssim_weight > 0:
            ssim_val = self.ssim(rendered, target)
            ssim_loss = 1.0 - ssim_val
            losses['ssim'] = ssim_loss.item()
            total_loss += self.ssim_weight * ssim_loss
        
        # LPIPS loss
        if self.lpips_weight > 0 and self.lpips is not None:
            # LPIPS expects [-1, 1] range
            rendered_norm = 2 * rendered - 1
            target_norm = 2 * target - 1
            lpips_loss = self.lpips(rendered_norm, target_norm).mean()
            losses['lpips'] = lpips_loss.item()
            total_loss += self.lpips_weight * lpips_loss
        
        # Opacity regularization
        if self.opacity_reg_weight > 0 and opacity is not None:
            opacity_reg = opacity.mean()
            losses['opacity_reg'] = opacity_reg.item()
            total_loss += self.opacity_reg_weight * opacity_reg
        
        # Scale regularization
        if self.scale_reg_weight > 0 and scales is not None:
            scale_reg = (scales ** 2).mean()
            losses['scale_reg'] = scale_reg.item()
            total_loss += self.scale_reg_weight * scale_reg
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses


class AdaptiveDensityController:
    """Controls adaptive density (splitting/cloning/pruning) of Gaussians."""
    
    def __init__(
        self,
        densify_grad_threshold: float = 0.0002,
        densify_size_threshold: float = None,
        opacity_threshold: float = 0.005,
        scene_extent: float = 1.0,
        max_gaussians: int = 5_000_000
    ):
        """
        Initialize density controller.
        
        Args:
            densify_grad_threshold: Gradient threshold for densification
            densify_size_threshold: Size threshold for splitting
            opacity_threshold: Minimum opacity before pruning
            scene_extent: Scene extent for size threshold calculation
            max_gaussians: Maximum number of Gaussians allowed
        """
        self.densify_grad_threshold = densify_grad_threshold
        self.opacity_threshold = opacity_threshold
        self.scene_extent = scene_extent
        self.max_gaussians = max_gaussians
        
        # Calculate size threshold if not provided
        if densify_size_threshold is None:
            self.densify_size_threshold = 0.01 * scene_extent
        else:
            self.densify_size_threshold = densify_size_threshold
    
    def densify_and_prune(
        self,
        gaussians,
        grad_threshold: Optional[float] = None,
        min_opacity: Optional[float] = None,
        max_screen_size: Optional[float] = None
    ) -> Dict[str, int]:
        """
        Perform densification and pruning.
        
        Args:
            gaussians: Gaussian model
            grad_threshold: Override gradient threshold
            min_opacity: Override opacity threshold
            max_screen_size: Maximum screen size for pruning
            
        Returns:
            Statistics about operations performed
        """
        stats = {
            'cloned': 0,
            'split': 0,
            'pruned': 0,
            'initial_count': gaussians.num_gaussians
        }
        
        grad_threshold = grad_threshold or self.densify_grad_threshold
        min_opacity = min_opacity or self.opacity_threshold
        
        # Get gradient statistics
        grads = gaussians.xyz_gradient_accum / gaussians.denom
        grads[grads.isnan()] = 0.0
        
        # Densification
        if gaussians.num_gaussians < self.max_gaussians:
            # Clone small Gaussians with high gradients
            selected_pts_mask = torch.where(
                (grads >= grad_threshold) & 
                (torch.max(gaussians.get_scaling, dim=1).values <= self.densify_size_threshold),
                True, False
            ).squeeze()
            
            stats['cloned'] = selected_pts_mask.sum().item()
            self._clone_gaussians(gaussians, selected_pts_mask)
            
            # Split large Gaussians with high gradients
            selected_pts_mask = torch.where(
                (grads >= grad_threshold) & 
                (torch.max(gaussians.get_scaling, dim=1).values > self.densify_size_threshold),
                True, False
            ).squeeze()
            
            stats['split'] = selected_pts_mask.sum().item()
            self._split_gaussians(gaussians, selected_pts_mask)
        
        # Pruning
        prune_mask = (gaussians.get_opacity < min_opacity).squeeze()
        
        # Prune by screen size if specified
        if max_screen_size is not None:
            big_points_mask = gaussians.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(prune_mask, big_points_mask)
        
        stats['pruned'] = prune_mask.sum().item()
        gaussians.prune_points(prune_mask)
        
        stats['final_count'] = gaussians.num_gaussians
        
        # Reset gradient accumulation
        gaussians.xyz_gradient_accum.zero_()
        gaussians.denom.zero_()
        
        return stats
    
    def _clone_gaussians(self, gaussians, mask: torch.Tensor):
        """Clone selected Gaussians."""
        if mask.sum() == 0:
            return
        
        new_xyz = gaussians._xyz[mask]
        new_features_dc = gaussians._features_dc[mask]
        new_features_rest = gaussians._features_rest[mask]
        new_opacities = gaussians._opacity[mask]
        new_scaling = gaussians._scaling[mask]
        new_rotation = gaussians._rotation[mask]
        
        gaussians.densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_opacities, new_scaling, new_rotation
        )
    
    def _split_gaussians(self, gaussians, mask: torch.Tensor):
        """Split selected Gaussians."""
        if mask.sum() == 0:
            return
        
        n_splits = 2
        
        # Get properties of Gaussians to split
        selected_xyz = gaussians._xyz[mask].repeat(n_splits, 1)
        selected_scaling = gaussians._scaling[mask].repeat(n_splits, 1)
        selected_rotation = gaussians._rotation[mask].repeat(n_splits, 1)
        selected_features_dc = gaussians._features_dc[mask].repeat(n_splits, 1, 1)
        selected_features_rest = gaussians._features_rest[mask].repeat(n_splits, 1, 1)
        selected_opacity = gaussians._opacity[mask].repeat(n_splits, 1)
        
        # Sample new positions
        stds = gaussians.get_scaling[mask]
        samples = torch.randn((stds.shape[0] * n_splits, 3), device=stds.device)
        rots = gaussians._build_rotation_from_quaternion(gaussians._rotation[mask]).repeat(n_splits, 1, 1)
        
        new_xyz = selected_xyz + torch.bmm(rots, (samples * stds.repeat(n_splits, 1)).unsqueeze(-1)).squeeze(-1)
        new_scaling = torch.log(selected_scaling.exp() / (0.8 * n_splits))
        new_opacity = selected_opacity
        
        # Remove original Gaussians
        gaussians.prune_points(mask)
        
        # Add new split Gaussians
        gaussians.densification_postfix(
            new_xyz, selected_features_dc, selected_features_rest,
            new_opacity, new_scaling, selected_rotation
        )


def l1_loss(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Simple L1 loss."""
    return F.l1_loss(rendered, target)


def l2_loss(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Simple L2 loss."""
    return F.mse_loss(rendered, target)


def psnr(rendered: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Calculate PSNR between images."""
    mse = F.mse_loss(rendered, target)
    return 20 * torch.log10(max_val / torch.sqrt(mse))


if __name__ == "__main__":
    # Test losses
    B, C, H, W = 2, 3, 256, 256
    rendered = torch.rand(B, C, H, W).cuda()
    target = torch.rand(B, C, H, W).cuda()
    
    # Test combined loss
    loss_fn = CombinedLoss(l1_weight=0.8, ssim_weight=0.2)
    loss, losses_dict = loss_fn(rendered, target)
    
    print("Loss test:")
    print(f"  Total loss: {loss.item():.4f}")
    for k, v in losses_dict.items():
        print(f"  {k}: {v:.4f}")
    
    # Test PSNR
    psnr_val = psnr(rendered, target)
    print(f"\nPSNR: {psnr_val.item():.2f} dB")
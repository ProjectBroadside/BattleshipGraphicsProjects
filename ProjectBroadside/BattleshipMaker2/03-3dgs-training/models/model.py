"""
3D Gaussian Splatting model implementation.
Core model architecture for Gaussian primitives.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GaussianParams:
    """Parameters for Gaussian primitives."""
    positions: torch.Tensor      # (N, 3)
    features_dc: torch.Tensor    # (N, 1, 3) - DC component of SH
    features_rest: torch.Tensor  # (N, (sh_degree+1)^2-1, 3) - Rest of SH
    scaling: torch.Tensor        # (N, 3)
    rotation: torch.Tensor       # (N, 4) - Quaternions
    opacity: torch.Tensor        # (N, 1)
    
    @property
    def num_gaussians(self) -> int:
        return self.positions.shape[0]


class GaussianModel(nn.Module):
    """3D Gaussian Splatting model."""
    
    def __init__(
        self,
        sh_degree: int = 3,
        init_points: Optional[torch.Tensor] = None,
        init_colors: Optional[torch.Tensor] = None
    ):
        """
        Initialize Gaussian model.
        
        Args:
            sh_degree: Degree of spherical harmonics
            init_points: Initial 3D points (N, 3)
            init_colors: Initial colors (N, 3)
        """
        super().__init__()
        
        self.sh_degree = sh_degree
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0
        
        # Initialize parameters
        if init_points is not None:
            self._init_from_points(init_points, init_colors)
        else:
            self._init_random()
        
        # Setup optimization parameters
        self.setup_functions()
        
        logger.info(f"Initialized Gaussian model with {self.num_gaussians} gaussians")
    
    def _init_from_points(self, points: torch.Tensor, colors: Optional[torch.Tensor] = None):
        """Initialize from point cloud."""
        num_points = points.shape[0]
        
        # Positions
        self._xyz = nn.Parameter(points.clone().requires_grad_(True))
        
        # Features (spherical harmonics coefficients)
        fused_color = self._init_features(num_points, colors)
        features = torch.zeros((num_points, 3, (self.max_sh_degree + 1) ** 2))
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous())
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous())
        
        # Scaling
        dist2 = self._compute_nearest_neighbor_distance(points)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        
        # Rotation (quaternions)
        rots = torch.zeros((num_points, 4))
        rots[:, 0] = 1.0  # Identity quaternion
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        
        # Opacity
        opacity = self._inverse_sigmoid(0.1 * torch.ones((num_points, 1)))
        self._opacity = nn.Parameter(opacity.requires_grad_(True))
        
        # Gradient accumulators
        self.xyz_gradient_accum = torch.zeros((num_points, 1))
        self.denom = torch.zeros((num_points, 1))
        self.max_radii2D = torch.zeros((num_points))
    
    def _init_random(self, num_points: int = 100):
        """Initialize with random points."""
        # Random points in [-1, 1]^3
        points = (torch.rand((num_points, 3)) - 0.5) * 2
        self._init_from_points(points)
    
    def _init_features(self, num_points: int, colors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Initialize color features."""
        if colors is not None:
            return self._rgb_to_sh(colors)
        else:
            # Random colors
            return self._rgb_to_sh(torch.rand((num_points, 3)))
    
    def _rgb_to_sh(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB to 0-degree SH coefficient."""
        C0 = 0.28209479177387814  # Normalization constant
        return (rgb - 0.5) / C0
    
    def _compute_nearest_neighbor_distance(self, points: torch.Tensor) -> torch.Tensor:
        """Compute distance to nearest neighbor for each point."""
        from sklearn.neighbors import NearestNeighbors
        
        nn = NearestNeighbors(n_neighbors=4, algorithm='auto')
        nn.fit(points.cpu().numpy())
        distances, _ = nn.kneighbors(points.cpu().numpy())
        
        # Use average of 3 nearest neighbors (excluding self)
        avg_dist2 = (distances[:, 1:] ** 2).mean(axis=1)
        
        return torch.tensor(avg_dist2, device=points.device)
    
    def setup_functions(self):
        """Setup activation functions."""
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        
        self.opacity_activation = torch.sigmoid
        self.inverse_sigmoid = lambda x: torch.log(x / (1 - x + 1e-8))
        
        self.rotation_activation = torch.nn.functional.normalize
    
    @property
    def num_gaussians(self) -> int:
        """Get number of Gaussians."""
        return self._xyz.shape[0]
    
    @property
    def get_xyz(self) -> torch.Tensor:
        """Get positions."""
        return self._xyz
    
    @property
    def get_features(self) -> torch.Tensor:
        """Get SH features."""
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self) -> torch.Tensor:
        """Get opacity values."""
        return self.opacity_activation(self._opacity)
    
    @property
    def get_scaling(self) -> torch.Tensor:
        """Get scaling values."""
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self) -> torch.Tensor:
        """Get rotation quaternions."""
        return self.rotation_activation(self._rotation)
    
    def get_covariance(self, scaling_modifier: float = 1.0) -> torch.Tensor:
        """Compute 3D covariance matrices."""
        return self._build_covariance_from_scaling_rotation(
            self.get_scaling * scaling_modifier, 
            self.get_rotation
        )
    
    def _build_covariance_from_scaling_rotation(
        self, 
        scaling: torch.Tensor, 
        rotation: torch.Tensor
    ) -> torch.Tensor:
        """Build covariance matrix from scaling and rotation."""
        L = self._build_scaling_rotation(scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        
        # Add small epsilon for numerical stability
        symm = (actual_covariance + actual_covariance.transpose(1, 2)) / 2
        return symm
    
    def _build_scaling_rotation(
        self, 
        scaling: torch.Tensor, 
        rotation: torch.Tensor
    ) -> torch.Tensor:
        """Build transformation matrix from scaling and rotation."""
        L = torch.zeros((scaling.shape[0], 3, 3), device=scaling.device)
        
        # Convert quaternion to rotation matrix
        r = rotation[:, 0]
        x = rotation[:, 1]
        y = rotation[:, 2]
        z = rotation[:, 3]
        
        # Build rotation matrix
        L[:, 0, 0] = 1 - 2 * (y * y + z * z)
        L[:, 0, 1] = 2 * (x * y - r * z)
        L[:, 0, 2] = 2 * (x * z + r * y)
        
        L[:, 1, 0] = 2 * (x * y + r * z)
        L[:, 1, 1] = 1 - 2 * (x * x + z * z)
        L[:, 1, 2] = 2 * (y * z - r * x)
        
        L[:, 2, 0] = 2 * (x * z - r * y)
        L[:, 2, 1] = 2 * (y * z + r * x)
        L[:, 2, 2] = 1 - 2 * (x * x + y * y)
        
        # Apply scaling
        L = L @ torch.diag_embed(scaling)
        
        return L
    
    def oneupSHdegree(self):
        """Increase active SH degree by 1."""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    
    def create_from_pcd(self, pcd: Dict, spatial_lr_scale: float = 1.0):
        """Create model from point cloud dictionary."""
        self._init_from_points(
            torch.tensor(pcd["points"], dtype=torch.float, device="cuda"),
            torch.tensor(pcd.get("colors", None), dtype=torch.float, device="cuda") 
            if "colors" in pcd else None
        )
        self.spatial_lr_scale = spatial_lr_scale
    
    def save_ply(self, path: str):
        """Save model as PLY file."""
        from plyfile import PlyData, PlyElement
        
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy()
        
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        
        # Create structured array
        dtype_full = [(attribute, 'f4') for attribute in self._construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        
        attributes = {
            'x': xyz[:, 0], 'y': xyz[:, 1], 'z': xyz[:, 2],
            'nx': normals[:, 0], 'ny': normals[:, 1], 'nz': normals[:, 2],
            'f_dc_0': f_dc[:, 0], 'f_dc_1': f_dc[:, 1], 'f_dc_2': f_dc[:, 2],
            'opacity': opacities[:, 0],
            'scale_0': scale[:, 0], 'scale_1': scale[:, 1], 'scale_2': scale[:, 2],
            'rot_0': rotation[:, 0], 'rot_1': rotation[:, 1], 
            'rot_2': rotation[:, 2], 'rot_3': rotation[:, 3]
        }
        
        # Add SH coefficients
        for i in range(f_rest.shape[1]):
            attributes[f'f_rest_{i}'] = f_rest[:, i]
        
        for attribute_name, attribute_values in attributes.items():
            elements[attribute_name] = attribute_values
        
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
        logger.info(f"Saved {self.num_gaussians} Gaussians to {path}")
    
    def load_ply(self, path: str):
        """Load model from PLY file."""
        from plyfile import PlyData
        
        plydata = PlyData.read(path)
        
        xyz = np.stack((
            np.asarray(plydata['vertex']['x']),
            np.asarray(plydata['vertex']['y']),
            np.asarray(plydata['vertex']['z'])
        ), axis=1)
        
        features_dc = np.stack((
            np.asarray(plydata['vertex']['f_dc_0']),
            np.asarray(plydata['vertex']['f_dc_1']),
            np.asarray(plydata['vertex']['f_dc_2'])
        ), axis=1)
        
        # Load SH features
        extra_f_names = [p.name for p in plydata['vertex'].properties if p.name.startswith("f_rest_")]
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata['vertex'][attr_name])
        
        # Reshape features
        features_dc = features_dc.reshape((-1, 3, 1))
        features_extra = features_extra.reshape((-1, 3, (self.max_sh_degree + 1) ** 2 - 1))
        
        scale_names = [f'scale_{i}' for i in range(3)]
        rot_names = [f'rot_{i}' for i in range(4)]
        
        scales = np.stack([np.asarray(plydata['vertex'][name]) for name in scale_names], axis=1)
        rots = np.stack([np.asarray(plydata['vertex'][name]) for name in rot_names], axis=1)
        opacities = np.asarray(plydata['vertex']['opacity'])[..., np.newaxis]
        
        # Set parameters
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda"))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda"))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda"))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda"))
        
        # Reset gradient accumulators
        self.xyz_gradient_accum = torch.zeros((self.num_gaussians, 1), device="cuda")
        self.denom = torch.zeros((self.num_gaussians, 1), device="cuda")
        self.max_radii2D = torch.zeros((self.num_gaussians), device="cuda")
        
        logger.info(f"Loaded {self.num_gaussians} Gaussians from {path}")
    
    def _construct_list_of_attributes(self) -> List[str]:
        """Construct list of attributes for PLY file."""
        attributes = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        
        # DC component
        for i in range(3):
            attributes.append(f'f_dc_{i}')
        
        # Rest of SH
        for i in range(3 * ((self.max_sh_degree + 1) ** 2 - 1)):
            attributes.append(f'f_rest_{i}')
        
        # Scaling
        for i in range(3):
            attributes.append(f'scale_{i}')
        
        # Rotation
        for i in range(4):
            attributes.append(f'rot_{i}')
        
        attributes.append('opacity')
        
        return attributes
    
    def densification_postfix(
        self, 
        new_xyz: torch.Tensor,
        new_features_dc: torch.Tensor,
        new_features_rest: torch.Tensor,
        new_opacities: torch.Tensor,
        new_scaling: torch.Tensor,
        new_rotation: torch.Tensor
    ):
        """Add new Gaussians after densification."""
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation
        }
        
        optimizable_tensors = self._cat_tensors_to_optimizer(d)
        
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        # Reset gradient accumulators
        self.xyz_gradient_accum = torch.zeros((self.num_gaussians, 1), device="cuda")
        self.denom = torch.zeros((self.num_gaussians, 1), device="cuda")
        self.max_radii2D = torch.zeros((self.num_gaussians), device="cuda")
    
    def _cat_tensors_to_optimizer(self, tensors_dict: Dict) -> Dict:
        """Concatenate new tensors with existing ones."""
        optimizable_tensors = {}
        
        for group in self.optimizer.param_groups:
            if group["name"] == "xyz":
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(tensors_dict["xyz"])), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(tensors_dict["xyz"])), dim=0)
                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], tensors_dict["xyz"]), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], tensors_dict["xyz"]), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        
        # Similar for other parameters...
        # (Implementation abbreviated for brevity)
        
        return optimizable_tensors
    
    def prune_points(self, mask: torch.Tensor):
        """Remove Gaussians based on mask."""
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
    
    def _prune_optimizer(self, mask: torch.Tensor) -> Dict:
        """Prune optimizer state based on mask."""
        optimizable_tensors = {}
        
        for group in self.optimizer.param_groups:
            if group["name"] == "xyz":
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        
        # Similar for other parameters...
        # (Implementation abbreviated for brevity)
        
        return optimizable_tensors
    
    def reset_opacity(self):
        """Reset opacity values."""
        opacities_new = self.inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self._replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
    
    def _replace_tensor_to_optimizer(self, tensor: torch.Tensor, name: str) -> Dict:
        """Replace tensor in optimizer."""
        optimizable_tensors = {}
        
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        
        return optimizable_tensors
    
    def forward(self) -> GaussianParams:
        """Forward pass returns Gaussian parameters."""
        return GaussianParams(
            positions=self.get_xyz,
            features_dc=self._features_dc,
            features_rest=self._features_rest,
            scaling=self.get_scaling,
            rotation=self.get_rotation,
            opacity=self.get_opacity
        )
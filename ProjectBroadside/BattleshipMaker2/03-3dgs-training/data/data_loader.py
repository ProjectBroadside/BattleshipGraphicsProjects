"""
Data loader for 3D Gaussian Splatting training.
Handles multi-view image datasets with camera parameters.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class GaussianSplattingDataset(Dataset):
    """Dataset for 3D Gaussian Splatting training."""
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        resolution: Optional[Tuple[int, int]] = None,
        scale_factor: float = 1.0,
        white_background: bool = False,
        cache_images: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to dataset directory
            split: Dataset split ("train" or "test")
            resolution: Target resolution (width, height)
            scale_factor: Scale factor for images
            white_background: Use white background instead of black
            cache_images: Cache images in memory
        """
        self.data_path = Path(data_path)
        self.split = split
        self.resolution = resolution
        self.scale_factor = scale_factor
        self.white_background = white_background
        self.cache_images = cache_images
        self.cached_images = {}
        
        # Load camera parameters
        self.cameras = self._load_cameras()
        
        # Load image paths
        self.image_paths = self._load_image_paths()
        
        # Setup transforms
        self.transform = self._create_transform()
        
        logger.info(f"Loaded {len(self)} images for {split} split")
    
    def _load_cameras(self) -> Dict:
        """Load camera parameters from various formats."""
        cameras = {}
        
        # Try loading from cameras.json
        cameras_json = self.data_path / "cameras.json"
        if cameras_json.exists():
            with open(cameras_json, 'r') as f:
                cameras = json.load(f)
                logger.info("Loaded cameras from cameras.json")
                return cameras
        
        # Try loading from transforms.json (NeRF format)
        transforms_json = self.data_path / f"transforms_{self.split}.json"
        if not transforms_json.exists():
            transforms_json = self.data_path / "transforms.json"
        
        if transforms_json.exists():
            with open(transforms_json, 'r') as f:
                data = json.load(f)
                cameras = self._parse_nerf_transforms(data)
                logger.info("Loaded cameras from transforms.json")
                return cameras
        
        # Try loading from COLMAP
        sparse_dir = self.data_path / "sparse" / "0"
        if sparse_dir.exists():
            cameras = self._load_colmap_cameras(sparse_dir)
            logger.info("Loaded cameras from COLMAP")
            return cameras
        
        raise ValueError(f"No camera parameters found in {self.data_path}")
    
    def _parse_nerf_transforms(self, data: Dict) -> Dict:
        """Parse NeRF-style transforms.json."""
        cameras = {
            "camera_model": "PINHOLE",
            "frames": []
        }
        
        # Extract camera intrinsics
        if "camera_angle_x" in data:
            fov_x = data["camera_angle_x"]
            width = data.get("w", 1920)
            height = data.get("h", 1080)
            fx = width / (2 * np.tan(fov_x / 2))
            fy = fx  # Assume square pixels
            cx = width / 2
            cy = height / 2
            cameras["params"] = [fx, fy, cx, cy]
            cameras["width"] = width
            cameras["height"] = height
        
        # Extract frames
        for frame in data["frames"]:
            cameras["frames"].append({
                "file_path": frame["file_path"],
                "transform_matrix": np.array(frame["transform_matrix"]),
                "camera_index": len(cameras["frames"])
            })
        
        return cameras
    
    def _load_colmap_cameras(self, sparse_dir: Path) -> Dict:
        """Load cameras from COLMAP sparse reconstruction."""
        try:
            from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary
        except ImportError:
            logger.warning("COLMAP loader not available, skipping COLMAP format")
            return {}
        
        cameras_bin = sparse_dir / "cameras.bin"
        images_bin = sparse_dir / "images.bin"
        
        cam_intrinsics = read_intrinsics_binary(cameras_bin)
        cam_extrinsics = read_extrinsics_binary(images_bin)
        
        # Convert to our format
        cameras = {
            "camera_model": "PINHOLE",
            "frames": []
        }
        
        for img_id, img_data in cam_extrinsics.items():
            cam_id = img_data.camera_id
            cam = cam_intrinsics[cam_id]
            
            # Extract parameters
            if cameras.get("params") is None:
                cameras["params"] = list(cam.params)
                cameras["width"] = cam.width
                cameras["height"] = cam.height
            
            # Build transform matrix
            R = img_data.qvec2rotmat()
            t = img_data.tvec
            transform = np.eye(4)
            transform[:3, :3] = R.T
            transform[:3, 3] = -R.T @ t
            
            cameras["frames"].append({
                "file_path": f"images/{img_data.name}",
                "transform_matrix": transform,
                "camera_index": len(cameras["frames"])
            })
        
        return cameras
    
    def _load_image_paths(self) -> List[str]:
        """Load and filter image paths based on split."""
        all_frames = self.cameras.get("frames", [])
        
        if self.split == "train":
            # Use 90% for training
            split_idx = int(0.9 * len(all_frames))
            frames = all_frames[:split_idx]
        else:
            # Use 10% for testing
            split_idx = int(0.9 * len(all_frames))
            frames = all_frames[split_idx:]
        
        image_paths = []
        for frame in frames:
            img_path = self.data_path / frame["file_path"]
            if not img_path.suffix:
                # Try common extensions
                for ext in ['.jpg', '.png', '.JPG', '.PNG']:
                    test_path = img_path.with_suffix(ext)
                    if test_path.exists():
                        img_path = test_path
                        break
            
            if img_path.exists():
                image_paths.append(str(img_path))
            else:
                logger.warning(f"Image not found: {img_path}")
        
        return image_paths
    
    def _create_transform(self) -> transforms.Compose:
        """Create image transformation pipeline."""
        transform_list = []
        
        # Resize if resolution specified
        if self.resolution:
            transform_list.append(transforms.Resize(self.resolution[::-1]))  # PIL uses (H, W)
        
        # Scale if needed
        if self.scale_factor != 1.0:
            transform_list.append(transforms.Lambda(
                lambda img: img.resize(
                    (int(img.width * self.scale_factor), 
                     int(img.height * self.scale_factor)),
                    Image.LANCZOS
                )
            ))
        
        # Convert to tensor and normalize
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x if x.shape[0] == 3 else x[:3])  # Handle RGBA
        ])
        
        # Apply white background if needed
        if self.white_background:
            transform_list.append(
                transforms.Lambda(lambda x: x + (1 - x[3:4]) if x.shape[0] == 4 else x)
            )
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Returns:
            Dictionary containing:
                - image: Tensor of shape (3, H, W)
                - camera_matrix: Camera intrinsic matrix (3, 3)
                - world_matrix: Camera extrinsic matrix (4, 4)
                - image_path: Path to the image file
        """
        # Load image
        img_path = self.image_paths[idx]
        
        if self.cache_images and img_path in self.cached_images:
            image = self.cached_images[img_path]
        else:
            image = Image.open(img_path).convert('RGBA')
            image = self.transform(image)
            
            if self.cache_images:
                self.cached_images[img_path] = image
        
        # Get corresponding camera data
        frame_idx = next(
            i for i, f in enumerate(self.cameras["frames"]) 
            if self.data_path / f["file_path"] == Path(img_path) or 
            str(self.data_path / f["file_path"]).replace('\\', '/') == img_path.replace('\\', '/')
        )
        frame = self.cameras["frames"][frame_idx]
        
        # Build camera matrix
        fx, fy, cx, cy = self.cameras["params"]
        camera_matrix = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        # Apply resolution scaling to intrinsics
        if self.resolution:
            scale_x = self.resolution[0] / self.cameras["width"]
            scale_y = self.resolution[1] / self.cameras["height"]
            camera_matrix[0] *= scale_x
            camera_matrix[1] *= scale_y
        
        if self.scale_factor != 1.0:
            camera_matrix[:2] *= self.scale_factor
        
        # Get world matrix
        world_matrix = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
        
        return {
            "image": image,
            "camera_matrix": camera_matrix,
            "world_matrix": world_matrix,
            "image_path": img_path,
            "image_name": os.path.basename(img_path)
        }


def get_data_loader(
    data_path: str,
    split: str = "train",
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
    **dataset_kwargs
) -> DataLoader:
    """
    Create a data loader for Gaussian Splatting training.
    
    Args:
        data_path: Path to dataset directory
        split: Dataset split ("train" or "test")
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle the data
        **dataset_kwargs: Additional arguments for GaussianSplattingDataset
    
    Returns:
        DataLoader instance
    """
    dataset = GaussianSplattingDataset(data_path, split, **dataset_kwargs)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and split == "train",
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )


def collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """Custom collate function for batching."""
    collated = {
        "images": torch.stack([item["image"] for item in batch]),
        "camera_matrices": torch.stack([item["camera_matrix"] for item in batch]),
        "world_matrices": torch.stack([item["world_matrix"] for item in batch]),
        "image_paths": [item["image_path"] for item in batch],
        "image_names": [item["image_name"] for item in batch]
    }
    return collated


if __name__ == "__main__":
    # Test the data loader
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    
    # Create data loader
    train_loader = get_data_loader(
        args.data_path,
        split="train",
        batch_size=args.batch_size,
        resolution=(1920, 1080),
        scale_factor=0.5
    )
    
    # Test loading
    for i, batch in enumerate(train_loader):
        print(f"Batch {i}:")
        print(f"  Images shape: {batch['image'].shape}")
        print(f"  Camera matrices shape: {batch['camera_matrix'].shape}")
        print(f"  World matrices shape: {batch['world_matrix'].shape}")
        print(f"  Image paths: {batch['image_path'][:2]}...")
        
        if i >= 2:
            break
    
    print("\nData loader test completed successfully!")
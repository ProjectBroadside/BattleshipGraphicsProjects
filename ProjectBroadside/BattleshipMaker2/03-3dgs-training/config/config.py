"""
Configuration management for 3D Gaussian Splatting training.
Handles loading, validation, and merging of configuration files.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    sh_degree: int = 3
    init_points: int = 100000
    max_gaussians: int = 5000000
    
    # Feature dimensions
    features_dc_dim: int = 3
    features_rest_dim: int = 15  # (sh_degree + 1)^2 - 1
    
    # Initialization
    init_opacity: float = 0.1
    init_scale_multiplier: float = 1.0
    
    # Scene bounds
    scene_extent: float = 1.0
    
    def __post_init__(self):
        """Validate and compute derived values."""
        self.features_rest_dim = (self.sh_degree + 1) ** 2 - 1


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    iterations: int = 30000
    batch_size: int = 1
    
    # Learning rates
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30000
    
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    
    # Learning rate scheduling
    lr_decay_start: int = 15000
    lr_decay_exp: float = 0.95
    
    # Gradient clipping
    gradient_clip_val: float = 0.0
    gradient_clip_norm_type: float = 2.0
    
    # Checkpoint intervals
    checkpoint_interval: int = 5000
    save_interval: int = 5000
    test_interval: int = 1000
    
    # Early stopping
    early_stop_patience: int = 10
    early_stop_min_delta: float = 0.0001
    
    # Mixed precision
    use_fp16: bool = False
    amp_opt_level: str = "O1"


@dataclass 
class DensificationConfig:
    """Densification configuration parameters."""
    start_iter: int = 500
    end_iter: int = 15000
    interval: int = 100
    
    grad_threshold: float = 0.0002
    size_threshold_scale: float = 0.01  # Relative to scene extent
    
    opacity_reset_interval: int = 3000
    min_opacity: float = 0.005
    
    # Advanced densification
    densify_grad_clip: float = 0.0
    densify_scale_clip: float = 0.0
    
    max_screen_size: Optional[float] = None
    
    # Adaptive thresholds
    adaptive_grad_threshold: bool = False
    grad_threshold_scale_start: float = 1.0
    grad_threshold_scale_end: float = 0.5


@dataclass
class LossConfig:
    """Loss configuration parameters."""
    l1_weight: float = 0.8
    ssim_weight: float = 0.2
    lpips_weight: float = 0.0
    
    # Regularization
    opacity_reg_weight: float = 0.0
    scale_reg_weight: float = 0.0
    
    # Loss scheduling
    use_loss_scheduling: bool = False
    ssim_start_iter: int = 1000
    lpips_start_iter: int = 5000


@dataclass
class EvaluationConfig:
    """Evaluation configuration parameters."""
    interval: int = 1000
    num_test_images: int = 50
    
    # Metrics to compute
    compute_psnr: bool = True
    compute_ssim: bool = True
    compute_lpips: bool = False
    
    # Visualization
    save_test_images: bool = True
    num_visualization_images: int = 5


@dataclass
class DataConfig:
    """Data configuration parameters."""
    data_path: str = ""
    
    # Resolution
    resolution: Optional[tuple] = None
    resolution_scale: float = 1.0
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    cache_images: bool = True
    
    # Augmentation
    use_augmentation: bool = False
    augmentation_prob: float = 0.5
    
    # Background
    white_background: bool = False
    
    # Train/test split
    train_split: float = 0.9
    shuffle_data: bool = True


@dataclass
class OptimizationConfig:
    """Optimization configuration parameters."""
    optimizer: str = "adam"
    
    # Adam parameters
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    
    # Weight decay
    weight_decay: float = 0.0
    
    # Warmup
    use_warmup: bool = False
    warmup_steps: int = 1000
    
    # Scheduler
    scheduler: Optional[str] = None
    scheduler_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    """Complete configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    densification: DensificationConfig = field(default_factory=DensificationConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    
    # Experiment info
    experiment_name: str = "gaussian_splatting"
    output_dir: str = "./outputs"
    seed: int = 42
    
    # Logging
    log_level: str = "INFO"
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "gaussian-splatting"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create from dictionary."""
        config = cls()
        
        # Update model config
        if "model" in config_dict:
            config.model = ModelConfig(**config_dict["model"])
        
        # Update training config
        if "training" in config_dict:
            config.training = TrainingConfig(**config_dict["training"])
        
        # Update densification config
        if "densification" in config_dict:
            config.densification = DensificationConfig(**config_dict["densification"])
        
        # Update loss config
        if "loss" in config_dict:
            config.loss = LossConfig(**config_dict["loss"])
        
        # Update evaluation config
        if "evaluation" in config_dict:
            config.evaluation = EvaluationConfig(**config_dict["evaluation"])
        
        # Update data config
        if "data" in config_dict:
            config.data = DataConfig(**config_dict["data"])
        
        # Update optimization config
        if "optimization" in config_dict:
            config.optimization = OptimizationConfig(**config_dict["optimization"])
        
        # Update top-level attributes
        for key in ["experiment_name", "output_dir", "seed", "log_level", 
                    "tensorboard", "wandb", "wandb_project"]:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        return config


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load file
    with open(config_path, 'r') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            config_dict = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    # Create config object
    config = Config.from_dict(config_dict)
    
    logger.info(f"Loaded config from {config_path}")
    
    return config


def save_config(config: Config, save_path: Union[str, Path]):
    """
    Save configuration to file.
    
    Args:
        config: Config object
        save_path: Path to save configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.to_dict()
    
    with open(save_path, 'w') as f:
        if save_path.suffix in ['.yaml', '.yml']:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        elif save_path.suffix == '.json':
            json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {save_path.suffix}")
    
    logger.info(f"Saved config to {save_path}")


def merge_configs(base_config: Config, override_config: Dict[str, Any]) -> Config:
    """
    Merge override configuration into base configuration.
    
    Args:
        base_config: Base configuration
        override_config: Dictionary of overrides
        
    Returns:
        Merged configuration
    """
    # Deep copy base config
    merged = deepcopy(base_config)
    
    # Apply overrides
    for key, value in override_config.items():
        if '.' in key:
            # Handle nested keys like "model.sh_degree"
            parts = key.split('.')
            obj = merged
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        else:
            # Top-level key
            if hasattr(merged, key):
                setattr(merged, key, value)
            else:
                logger.warning(f"Unknown config key: {key}")
    
    return merged


def create_default_config() -> Config:
    """Create default configuration."""
    return Config()


if __name__ == "__main__":
    # Test configuration
    config = create_default_config()
    
    # Save to YAML
    save_config(config, "test_config.yaml")
    
    # Load from YAML
    loaded_config = load_config("test_config.yaml")
    
    # Test merging
    overrides = {
        "model.sh_degree": 4,
        "training.iterations": 50000,
        "experiment_name": "test_experiment"
    }
    
    merged_config = merge_configs(loaded_config, overrides)
    
    print("Configuration test completed!")
    print(f"SH degree: {merged_config.model.sh_degree}")
    print(f"Iterations: {merged_config.training.iterations}")
    print(f"Experiment: {merged_config.experiment_name}")
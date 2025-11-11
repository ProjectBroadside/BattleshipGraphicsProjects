# 03-3DGS-Training: 3D Gaussian Splatting Training Module

## Module Overview and Learning Objectives

This module implements the core 3D Gaussian Splatting (3DGS) training pipeline for reconstructing historical battleships from synthetic multi-view images. The module is designed to be both educational and production-ready, providing a complete implementation of the state-of-the-art 3DGS algorithm with extensive documentation and visualization tools.

### Learning Objectives
- Understand the theoretical foundations of 3D Gaussian Splatting
- Implement differentiable rendering using Gaussian primitives
- Master the optimization strategies for high-quality 3D reconstruction
- Learn to evaluate and visualize reconstruction quality
- Gain practical experience with GPU-accelerated point cloud processing

### Key Features
- **Modular Architecture**: Clean separation of concerns with reusable components
- **Comprehensive Logging**: Detailed training metrics and visualization
- **GPU Optimization**: Efficient CUDA kernels for real-time rendering
- **Flexible Configuration**: YAML-based hyperparameter management
- **Quality Metrics**: Built-in PSNR, SSIM, and LPIPS evaluation
- **Checkpoint Management**: Automatic saving and recovery of training state

## Prerequisites and Environment Setup

### Hardware Requirements
- **GPU**: NVIDIA RTX 3090 (24GB) minimum, RTX 4090/5090 recommended
- **CPU**: 8+ cores recommended for data preprocessing
- **RAM**: 32GB minimum, 64GB recommended
- **Storage**: 200GB+ for datasets, checkpoints, and results

### Software Requirements
```
Python 3.10+
CUDA 11.8+ with cuDNN
PyTorch 2.0+
COLMAP (optional, for pose refinement)
```

### Environment Setup

1. **Create Conda Environment**
```bash
conda create -n gaussplat python=3.10
conda activate gaussplat
```

2. **Install PyTorch with CUDA**
```bash
# For CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

3. **Install Module Dependencies**
```bash
cd 03-3dgs-training
pip install -r requirements.txt
```

4. **Install Gaussian Splatting CUDA Extensions**
```bash
# Clone the official implementation
git clone https://github.com/graphdeco-inria/gaussian-splatting.git external/gaussian-splatting
cd external/gaussian-splatting

# Install diff-gaussian-rasterization
pip install submodules/diff-gaussian-rasterization

# Install simple-knn
pip install submodules/simple-knn
```

5. **Verify Installation**
```bash
python -m pytest tests/test_environment.py -v
```

## Theoretical Foundations of 3D Gaussian Splatting

### Core Concepts

#### 1. Gaussian Primitives
3D Gaussian Splatting represents a scene as a collection of 3D Gaussian primitives, each defined by:
- **Position** (μ ∈ ℝ³): Center of the Gaussian
- **Covariance** (Σ ∈ ℝ³ˣ³): Shape and orientation
- **Opacity** (α ∈ [0,1]): Transparency
- **Color** (c ∈ ℝ³ or SH coefficients): Appearance

The 3D Gaussian function:
```
G(x) = exp(-½(x-μ)ᵀΣ⁻¹(x-μ))
```

#### 2. Differentiable Rendering
The rendering equation for pixel color C:
```
C = Σᵢ cᵢ αᵢ Tᵢ
```
where:
- cᵢ: Color of Gaussian i
- αᵢ: Opacity after 2D projection
- Tᵢ: Accumulated transmittance = Πⱼ<ᵢ(1-αⱼ)

#### 3. Adaptive Density Control
The algorithm dynamically adjusts the number of Gaussians through:
- **Splitting**: Large Gaussians with high gradients
- **Cloning**: Small Gaussians in under-reconstructed areas
- **Pruning**: Gaussians with low opacity or excessive size

### Mathematical Framework

#### Covariance Decomposition
To ensure positive semi-definiteness:
```
Σ = RSSṞ
```
where:
- R: Rotation matrix (from quaternion q)
- S: Diagonal scaling matrix

#### Spherical Harmonics for View-Dependent Effects
Color representation using SH basis:
```
c(v) = Σₗ₌₀ᴸ Σₘ₌₋ₗˡ cₗₘ Yₗₘ(v)
```

## Dataset Preparation Guidelines

### Expected Input Structure
```
dataset/
├── images/                  # Multi-view images
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
├── sparse/                  # COLMAP output (optional)
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
├── cameras.json            # Camera parameters
└── transforms.json         # NeRF-style format (alternative)
```

### Camera Parameter Format
```json
{
  "camera_model": "PINHOLE",
  "width": 1920,
  "height": 1080,
  "params": [fx, fy, cx, cy],
  "frames": [
    {
      "file_path": "images/000000.jpg",
      "transform_matrix": [[...4x4 matrix...]],
      "camera_index": 0
    }
  ]
}
```

### Data Validation
```python
python scripts/validate_dataset.py --data_path datasets/bismarck
```

## Training Pipeline Architecture

### Pipeline Overview
```
Input Images → Data Loading → Gaussian Initialization → 
Differentiable Rendering → Loss Computation → 
Backpropagation → Adaptive Density Control → 
Evaluation → Checkpoint Save
```

### Core Components

#### 1. Data Loader (`data_loader.py`)
- Efficient batch loading of images
- Camera parameter parsing
- On-the-fly augmentation support
- Multi-resolution training

#### 2. Model Architecture (`model.py`)
- Gaussian parameter storage
- Efficient CUDA rendering
- Adaptive control mechanisms
- Checkpoint serialization

#### 3. Training Loop (`train.py`)
- Progressive training schedule
- Learning rate scheduling
- Memory-efficient gradient accumulation
- Real-time metric tracking

#### 4. Loss Functions (`loss_functions.py`)
- L1/L2 reconstruction loss
- SSIM structural similarity
- Perceptual loss (LPIPS)
- Regularization terms

## Hyperparameter Configuration

### Configuration Structure (`config/default_config.yaml`)
```yaml
# Model parameters
model:
  sh_degree: 3              # Spherical harmonics degree
  init_points: 100000       # Initial point cloud size
  
# Training parameters  
training:
  iterations: 30000         # Total training iterations
  batch_size: 1            # Images per iteration
  
  # Learning rates
  learning_rates:
    position: 0.00016      # 3D position learning rate
    feature: 0.0025        # Feature/color learning rate
    opacity: 0.05          # Opacity learning rate
    scaling: 0.005         # Scale learning rate
    rotation: 0.001        # Rotation learning rate
    
  # Learning rate schedules
  lr_schedules:
    position:
      init: 0.00016
      final: 0.0000016
      max_steps: 30000
      
# Densification parameters
densification:
  start_iter: 500          # When to start densification
  end_iter: 15000         # When to stop densification
  interval: 100           # Densification frequency
  grad_threshold: 0.0002  # Gradient threshold for splitting
  min_opacity: 0.005      # Minimum opacity before pruning
  
# Loss weights
loss:
  l1_weight: 0.8
  ssim_weight: 0.2
  lpips_weight: 0.0       # Optional perceptual loss
  
# Evaluation
evaluation:
  interval: 1000          # Evaluation frequency
  num_test_images: 50     # Test set size
  
# Checkpointing
checkpoint:
  save_interval: 5000     # Checkpoint frequency
  keep_last_n: 5         # Checkpoints to keep
```

### Key Hyperparameters Explained

#### Learning Rates
- **Position LR**: Controls 3D location updates (most critical)
- **Feature LR**: Affects color/appearance convergence
- **Opacity LR**: High value for quick transparency adjustments
- **Scaling LR**: Moderate to prevent instability
- **Rotation LR**: Low to maintain orientation stability

#### Densification Control
- **Gradient Threshold**: Lower = more splitting (higher quality)
- **Opacity Threshold**: Higher = aggressive pruning (efficiency)
- **Start/End Iterations**: Defines densification window

## Loss Functions and Optimization Strategies

### Multi-Component Loss Function
```python
L_total = λ₁L_color + λ₂L_ssim + λ₃L_lpips + λ₄L_reg
```

#### 1. Color Loss (L1/L2)
```python
L_color = ||I_rendered - I_gt||₁
```
- Fast convergence
- Handles outliers well

#### 2. Structural Similarity (SSIM)
```python
L_ssim = 1 - SSIM(I_rendered, I_gt)
```
- Preserves structural details
- Perceptually motivated

#### 3. Perceptual Loss (LPIPS)
```python
L_lpips = LPIPS(I_rendered, I_gt)
```
- Uses VGG features
- Improves texture quality

#### 4. Regularization Terms
```python
L_reg = λ_opacity * mean(opacity) + λ_scale * mean(scale²)
```
- Prevents degenerate solutions
- Improves efficiency

### Optimization Strategy

#### Adam Optimizer Configuration
```python
optimizer = Adam([
    {'params': positions, 'lr': lr_position},
    {'params': features, 'lr': lr_feature},
    {'params': opacities, 'lr': lr_opacity},
    {'params': scales, 'lr': lr_scale},
    {'params': rotations, 'lr': lr_rotation}
])
```

#### Learning Rate Scheduling
- Exponential decay for positions
- Constant rates for other parameters
- Warmup period for stability

## Hardware Configuration for Local Setup

**RTX 3090 + 5090 Configuration**
```python
# Optimal settings for available VRAM
CONFIG = {
    "3090": {
        "max_images": 100,      # Limit concurrent images
        "resolution_scale": 0.5, # Start with half resolution
        "densification_interval": 500,
        "max_splats": 5_000_000
    },
    "5090": {
        "max_images": 200,      # Full dataset
        "resolution_scale": 1.0, # Full resolution
        "densification_interval": 300,
        "max_splats": 8_000_000
    }
}
```

## Evaluation Metrics and Visualization Tools

### Quantitative Metrics

#### 1. Peak Signal-to-Noise Ratio (PSNR)
```python
PSNR = 20 * log10(MAX_I / sqrt(MSE))
```
- Target: > 28 dB (good), > 30 dB (excellent)

#### 2. Structural Similarity Index (SSIM)
```python
SSIM = (2μ_x μ_y + c1)(2σ_xy + c2) / (μ_x² + μ_y² + c1)(σ_x² + σ_y² + c2)
```
- Target: > 0.90 (good), > 0.95 (excellent)

#### 3. Learned Perceptual Image Patch Similarity (LPIPS)
- Target: < 0.1 (good), < 0.05 (excellent)
- More correlated with human perception

### Visualization Tools

#### 1. Training Progress Dashboard
```python
python visualization.py --model_path outputs/experiment_001
```
- Real-time loss curves
- Sample reconstructions
- Gaussian statistics

#### 2. Interactive 3D Viewer
```python
python viewer_3d.py --splat_file outputs/experiment_001/splats/final.ply
```
- Rotate/zoom/pan controls
- Gaussian property visualization
- Export capabilities

#### 3. Quality Report Generator
```python
python generate_report.py --experiment_dir outputs/experiment_001
```
- Comprehensive HTML report
- Metric comparisons
- Visual quality assessment

## Training Configuration and Execution

```python
# train_bismarck.py
import os
import torch
from arguments import ModelParams, PipelineParams, OptimizationParams

def get_bismarck_config():
    parser = ArgumentParser(description="Bismarck training params")
    
    # Model parameters
    model = ModelParams(parser)
    model.sh_degree = 3  # Spherical harmonics degree
    
    # Pipeline parameters  
    pipeline = PipelineParams(parser)
    pipeline.convert_SHs_python = False
    pipeline.compute_cov3D_python = False
    
    # Optimization parameters
    opt = OptimizationParams(parser)
    opt.iterations = 30000  # Can extend to 50000 for quality
    opt.position_lr_init = 0.00016
    opt.position_lr_final = 0.0000016
    opt.position_lr_delay_mult = 0.01
    opt.position_lr_max_steps = 30000
    opt.feature_lr = 0.0025
    opt.opacity_lr = 0.05
    opt.scaling_lr = 0.005
    opt.rotation_lr = 0.001
    
    # Densification
    opt.densify_from_iter = 500
    opt.densify_until_iter = 15000
    opt.densify_grad_threshold = 0.0002
    opt.densification_interval = 100
    opt.opacity_reset_interval = 3000
    
    return model, pipeline, opt
```

### Training Execution

```bash
# Basic training command
python train.py \
    -s data/bismarck_dataset \
    -m output/bismarck_splats \
    --eval \
    --iterations 30000

# With custom config
python train.py \
    -s data/bismarck_dataset \
    -m output/bismarck_splats \
    --config config/default_config.yaml \
    --eval

# Advanced configuration
python train.py \
    -s data/bismarck_dataset \
    -m output/bismarck_splats \
    --eval \
    --sh_degree 3 \
    --images_resolution 2 \
    --resolution 2048 \
    --iterations 50000 \
    --checkpoint_iterations 5000 \
    --save_iterations 5000 10000 20000 30000 50000
```

### Multi-GPU Strategy

For 3090 + 5090 setup:
```python
def distributed_training():
    # Use 5090 for main training
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 5090
    
    # Use 3090 for validation renders
    # Run separate process for evaluation
    
# Alternative: Data parallel training
python train.py --gpus 0,1 --distributed
```

### 6. Training Monitoring

```python
class TrainingMonitor:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        
    def log_metrics(self, iteration, metrics):
        self.writer.add_scalar('Loss/L1', metrics['l1_loss'], iteration)
        self.writer.add_scalar('Loss/SSIM', metrics['ssim_loss'], iteration)
        self.writer.add_scalar('PSNR', metrics['psnr'], iteration)
        self.writer.add_scalar('Num_Splats', metrics['num_gaussians'], iteration)
        
    def log_images(self, iteration, renders, ground_truth):
        self.writer.add_images('Renders', renders, iteration)
        self.writer.add_images('Ground_Truth', ground_truth, iteration)
        self.writer.add_images('Difference', 
                              torch.abs(renders - ground_truth), 
                              iteration)
```

### 7. Quality Checkpoints

```python
def evaluate_checkpoint(model_path, test_views):
    metrics = {
        'psnr': [],
        'ssim': [],
        'lpips': []
    }
    
    for view in test_views:
        rendered = render_gaussian_splats(model_path, view)
        gt = load_ground_truth(view)
        
        metrics['psnr'].append(calculate_psnr(rendered, gt))
        metrics['ssim'].append(calculate_ssim(rendered, gt))
        metrics['lpips'].append(calculate_lpips(rendered, gt))
    
    return {
        'psnr': np.mean(metrics['psnr']),
        'ssim': np.mean(metrics['ssim']),
        'lpips': np.mean(metrics['lpips'])
    }
```

## 🎚️ Hyperparameter Tuning

### Key Parameters to Adjust

1. **Learning Rates**
   - Position LR: Controls splat movement
   - Scaling LR: Controls splat size
   - Opacity LR: Controls transparency

2. **Densification**
   - Gradient threshold: When to split/clone splats
   - Interval: How often to densify
   - Until iteration: When to stop adding splats

3. **Pruning**
   - Opacity threshold: Remove transparent splats
   - Scale threshold: Remove huge splats
   - View-space threshold: Remove off-screen splats

### Bismarck-Specific Settings

```python
# For ship with fine details (rigging, guns)
DETAIL_CONFIG = {
    "densify_grad_threshold": 0.0001,  # Lower = more splats
    "prune_opacity_threshold": 0.005,   # Lower = keep more
    "max_splat_size": 0.1,              # Prevent giant splats
    "position_lr_init": 0.00008         # Slower = more stable
}
```

## 🧪 Validation Process

### During Training
- Monitor loss convergence
- Check splat count growth
- Validate on held-out views
- Watch for overfitting

### Post-Training
1. **Visual Inspection**
   - Render from novel viewpoints
   - Check for floaters/artifacts
   - Verify detail preservation

2. **Quantitative Metrics**
   - PSNR > 28 dB
   - SSIM > 0.90
   - LPIPS < 0.1

3. **Technical Drawing Validation**
   - Measure ship dimensions in renders
   - Compare to blueprint specifications
   - Ensure proportional accuracy

## Common Troubleshooting Scenarios

### 1. Out of Memory (OOM) Errors

**Symptoms**: CUDA out of memory during training

**Solutions**:
```python
# Reduce batch size
python train.py --batch_size 1

# Limit maximum Gaussians
python train.py --max_gaussians 3000000

# Enable gradient checkpointing
python train.py --gradient_checkpoint
```

### 2. Poor Convergence

**Symptoms**: High loss after many iterations

**Solutions**:
```python
# Increase initial learning rate
python train.py --lr_position_init 0.0003

# More aggressive densification
python train.py --densify_grad_threshold 0.0001

# Check camera poses
python scripts/validate_cameras.py --data_path datasets/bismarck
```

### 3. Floating Artifacts

**Symptoms**: Isolated Gaussians in empty space

**Solutions**:
```python
# Stricter opacity pruning
python train.py --opacity_threshold 0.01

# Add regularization
python train.py --opacity_reg_weight 0.001
```

### 4. Blurry Reconstructions

**Symptoms**: Lack of fine details

**Solutions**:
```python
# Higher SH degree
python train.py --sh_degree 4

# Extended training
python train.py --iterations 50000

# Lower densification threshold
python train.py --densify_grad_threshold 0.00005
```

### 5. Training Instability

**Symptoms**: Loss spikes or NaN values

**Solutions**:
```python
# Enable gradient clipping
python train.py --gradient_clip 1.0

# Reduce learning rates
python train.py --lr_scale_factor 0.5

# Check for degenerate inputs
python scripts/check_dataset_health.py --data_path datasets/bismarck
```

## Performance Optimization Tips

### 1. GPU Utilization
```python
# Multi-GPU training
python train.py --gpus 0,1

# Mixed precision training
python train.py --fp16
```

### 2. Data Loading
```python
# Increase workers
python train.py --num_workers 8

# Enable pin memory
python train.py --pin_memory
```

### 3. Memory Efficiency
```python
# Gradient accumulation
python train.py --accumulate_grad_batches 4

# Automatic mixed precision
python train.py --amp_level O1
```

## Common Issues (Legacy)

### Problem: Out of Memory
```python
# Solutions:
# 1. Reduce batch size
opt.batch_size = 1

# 2. Lower resolution initially
opt.resolution_scale = 0.5

# 3. Limit splat count
opt.max_gaussians = 3_000_000
```

### Problem: Poor Convergence
```python
# Solutions:
# 1. Adjust learning rates
opt.position_lr_init *= 0.5

# 2. Extend training
opt.iterations = 50000

# 3. Better initialization
opt.random_init = False
```

### Problem: Floating Artifacts
```python
# Solutions:
# 1. Aggressive pruning
opt.prune_big_splats = True
opt.max_splat_size = 0.05

# 2. Opacity regularization
opt.opacity_reg = 0.01
```

## 📊 Expected Results

### Training Timeline (3090/5090)
- 0-5k iterations: Basic shape emerges
- 5k-15k: Details develop, densification active
- 15k-25k: Refinement, quality improvement
- 25k-30k+: Final polish, convergence

### Quality Targets
- **Splat Count**: 3-8M (memory dependent)
- **Training Time**: 6-12 hours
- **Final PSNR**: >28 dB
- **Final SSIM**: >0.90

## Integration with Project Pipeline

### Input from Stage 2
- Synthetic multi-view images
- Camera parameters (COLMAP or manual)
- Optional: Initial point cloud

### Output to Stage 4
- Trained Gaussian splats (.ply format)
- Refined camera parameters
- Quality metrics report
- Rendered validation images

## Module Structure

```
03-3dgs-training/
├── config/
│   ├── default_config.yaml      # Default hyperparameters
│   └── experiments/             # Experiment configs
├── data/
│   ├── data_loader.py          # Dataset handling
│   └── transforms.py           # Data augmentation
├── models/
│   ├── model.py               # Gaussian model
│   ├── gaussian_utils.py      # Gaussian operations
│   └── rendering.py           # Differentiable renderer
├── training/
│   ├── train.py              # Main training script
│   ├── loss_functions.py     # Loss implementations
│   └── optimizers.py         # Custom optimizers
├── evaluation/
│   ├── metrics.py            # Quality metrics
│   └── visualization.py      # Plotting tools
├── scripts/
│   ├── validate_dataset.py   # Data validation
│   ├── convert_colmap.py     # Format conversion
│   └── batch_experiments.py  # Multi-run manager
├── notebooks/
│   ├── 01_quick_start.ipynb  # Getting started
│   ├── 02_advanced_training.ipynb
│   └── 03_analysis.ipynb
├── tests/
│   ├── test_model.py
│   ├── test_data_loader.py
│   └── test_rendering.py
├── requirements.txt
├── setup.py
└── README.md
```

## References and Resources

### Core Papers
1. **3D Gaussian Splatting for Real-Time Radiance Field Rendering**
   - Kerbl et al., SIGGRAPH 2023
   - [Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
   - [Code](https://github.com/graphdeco-inria/gaussian-splatting)

2. **EfficientGS: Streamlining Gaussian Splatting**
   - [Optimization techniques for faster training]

3. **Mip-Splatting: Alias-free 3D Gaussian Splatting**
   - [Improved rendering quality]

### Related Work
- **NeRF**: Neural Radiance Fields (predecessor technology)
- **Instant-NGP**: Fast neural graphics primitives
- **Plenoxels**: Sparse voxel radiance fields

### Tutorials and Guides
1. [Official 3DGS Tutorial](https://github.com/graphdeco-inria/gaussian-splatting)
2. [Understanding Gaussian Splatting](https://medium.com/@AriaLeeNotAriel/3d-gaussian-splatting-explained)
3. [CUDA Optimization for 3DGS](https://developer.nvidia.com/blog/gaussian-splatting)

### Community Resources
- [Awesome 3D Gaussian Splatting](https://github.com/MrNeRF/awesome-3D-gaussian-splatting)
- [3DGS Discord Community](https://discord.gg/gaussian-splatting)
- [Research Papers Collection](https://paperswithcode.com/method/3d-gaussian-splatting)

## Citation

If you use this implementation in your research, please cite:
```bibtex
@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimkühler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

## License

This module is released under the MIT License. See LICENSE file for details.

## Support

For issues and questions:
- GitHub Issues: [Project Repository]
- Documentation: See `/docs` folder
- Email: [Project Contact]

## Next Steps

After successful training:
1. Save final `.ply` file
2. Generate quality metrics report
3. Create preview renders
4. Move to Stage 4: Splat Refinement
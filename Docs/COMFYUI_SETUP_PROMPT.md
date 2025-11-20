# ComfyUI Setup Prompt for Naval Ship Blueprint Processing

## System Context
I need help setting up ComfyUI on my Linux system (Nobara Linux, Fedora 42-based) for processing naval ship blueprints into 3D models. My hardware includes:
- **GPU**: NVIDIA RTX 3090 (24GB VRAM) 
- **CPU**: Intel i9-14900KF (24 cores, 32 threads)
- **RAM**: 64GB
- **Storage**: Available space on /mnt/superstorage/

## Primary Objective
Set up ComfyUI with specific custom nodes for a 2D-to-3D pipeline that:
1. Processes naval ship blueprint images (technical line drawings)
2. Segments and classifies ship components (turrets, superstructures, hull sections)
3. Generates depth maps and 3D models from the segmented components
4. Achieves 95%+ accuracy in component detection with <5% scale error

## Installation Tasks

### 1. Core ComfyUI Installation
- Install ComfyUI on Linux (preferably using git clone method for latest features)
- Set up Python virtual environment
- Configure for NVIDIA RTX 3090 CUDA support
- Install PyTorch with CUDA support
- Ensure proper GPU memory management settings

### 2. ComfyUI-Manager Installation
- Install ComfyUI-Manager for easy custom node management
- Configure for Linux environment
- Set up proper permissions and paths

### 3. Critical Custom Nodes (Tier 1 Priority)
Install and configure these essential nodes:

#### Background Removal & Preprocessing
- **ComfyUI-RMBG**: Advanced background removal with SAM/SAM2 support
  - Repository: https://github.com/1038lab/ComfyUI-RMBG
  - Includes SAM2Segment node for component segmentation
  - Auto-downloads models to ComfyUI/models/RMBG/

#### 3D Generation
- **ComfyUI-Flowty-TripoSR**: Fast 3D reconstruction from single images
  - Repository: https://github.com/flowtyone/ComfyUI-Flowty-TripoSR
  - For validating component segmentation visually

#### Line Detection (if available)
- **ComfyUI-Anyline** or equivalent line detection node
  - Search in ComfyUI-Manager for line detection alternatives

### 4. Additional Nodes (Tier 2 - Install after basics work)
- **Segment Anything (SAM)** integration nodes
- **YoloWorld-EfficientSAM** for zero-shot detection
- **ComfyUI-Marigold** for depth estimation
- **ComfyUI-DepthAnythingV2** as alternative depth method
- **Jovimetrix** for mathematical validation

### 5. Model Downloads
- Download required checkpoint models for Stable Diffusion
- Download SAM/SAM2 models for segmentation
- Download TripoSR model weights
- Place models in appropriate directories (ComfyUI/models/checkpoints, etc.)

### 6. Workflow Configuration
- Set up custom workflow for blueprint processing pipeline
- Configure nodes for optimal GPU memory usage
- Create template workflow with the following stages:
  1. Background removal (RMBG)
  2. Component detection (SAM/Anyline)
  3. Depth estimation
  4. 3D generation (TripoSR)
  5. Validation steps

### 7. Testing & Validation
- Test with sample blueprint image (gangut_class.jpg if available)
- Verify GPU utilization and memory usage
- Ensure all nodes load without errors
- Test complete pipeline from input to 3D output

## Specific Requirements & Constraints

1. **Linux Compatibility**: Ensure all installation steps work on Nobara Linux (Fedora-based)
2. **Path Management**: Use absolute paths, install location flexible but suggest /mnt/superstorage/_projects/ComfyUI
3. **GPU Optimization**: Configure for RTX 3090 with 24GB VRAM, optimize for batch processing
4. **Error Handling**: Document any dependency issues and provide solutions
5. **Version Control**: Use git for ComfyUI to enable easy updates

## Expected Outputs

1. **Working ComfyUI installation** with web UI accessible
2. **All Tier 1 custom nodes installed** and functional
3. **Test workflow** that processes a blueprint image through the pipeline
4. **Documentation** of:
   - Installation paths and commands used
   - Any Linux-specific modifications needed
   - Model file locations
   - Startup command for ComfyUI
   - Browser URL for accessing the interface

## Additional Context

This is part of a larger project to convert 2D naval ship blueprints into 3D models. The blueprints are technical line drawings with clear geometric shapes. The main challenge is accurate component segmentation and classification (distinguishing turrets from superstructures, identifying gun positions, etc.) rather than simple edge detection.

The final output will be exported to Blender for refinement, so compatibility with standard 3D formats (OBJ, PLY, etc.) is important.

## Questions to Address

1. Should we use a Python virtual environment or conda environment?
2. What's the best way to manage model downloads and storage?
3. How should we configure ComfyUI for headless batch processing later?
4. Are there any Linux-specific performance optimizations we should enable?
5. Should we set up any automatic update mechanisms?

---

Please provide step-by-step instructions with actual commands, handling any errors that may occur during installation. Also document the installation in ~/dependencylog.md as specified in my system configuration.
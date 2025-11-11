# Battleship 3DGS Research Prototype

## 🎯 Overview

This project reconstructs historical battleships using synthetic multi-view images and 3D Gaussian Splatting (3DGS). Our approach solves the fundamental problem: no multi-view photography exists of these historical vessels that sank decades ago.

## 🚀 Core Innovation

**Problem**: Traditional 3DGS requires 50-200 photos from multiple angles—impossible for ships that no longer exist.

**Solution**: A dual-path strategy for synthetic data generation. We primarily use Google's Gemini API for its state-of-the-art prompt understanding, with a fallback to a local Stable Diffusion + ControlNet pipeline for maximum geometric control.

**Key Insight**: Historical technical drawings provide exact dimensional data that photos cannot. By incorporating these as ControlNet conditioning, we ensure our synthetic images maintain precise proportions and details matching the original naval blueprints.

## 📁 Project Structure

```
├── 01-image-generation/         # Synthetic multi-view image generation
├── 02-dataset-preparation/      # Format images for 3DGS training
├── 03-3dgs-training/           # 3D Gaussian Splatting implementation
├── 04-splat-refinement/        # Clean and optimize splat models
├── 05-mesh-conversion/         # Convert splats to traditional meshes
├── 06-final-output/            # Package final deliverables
├── docs/                       # Architecture and implementation docs
└── README.md                   # This file
```

## ⚙️ Technical Stack

### Hardware Requirements
- **Local Development**: RTX 3090 (24GB) + RTX 5090 (32GB)
- **OS**: Windows 11
- **RAM**: 64GB+ recommended
- **Storage**: 2TB+ for datasets and checkpoints

### Software Dependencies
- **Image Generation**: Google Gemini API (Primary), ComfyUI / Stable Diffusion XL (Fallback)
- **ControlNet**: Required for the fallback local pipeline
- **3DGS Framework**: Gaussian Splatting (Inria implementation)
- **Pose Estimation**: COLMAP
- **Development**: Python 3.10+, CUDA 11.8+

## 🔬 Pipeline Stages

### Stage 1: Image Generation
- **Path A (Primary)**: Generate views using the Google Gemini API with detailed, structured prompts for camera control.
- **Path B (Fallback)**: Generate views using a local pipeline with multi-layer ControlNet (depth maps + technical line art) for strong geometric consistency.
- Both paths aim for 200-500 consistent images.

### Stage 2: Dataset Preparation
- Estimate camera poses and validate image consistency
- Create training manifests with 360° coverage verification

### Stage 3: 3DGS Training
- Train high-quality Gaussian splat models (3-8M splats)
- Target metrics: PSNR >28dB, SSIM >0.90
- Training time: 6-12 hours on local GPUs

### Stage 4: Splat Refinement
- Remove floating artifacts and optimize density
- Color correction and opacity threshold adjustment

### Stage 5: Mesh Conversion
- Convert splats to watertight meshes using Poisson reconstruction
- UV unwrapping and texture baking from splats

### Stage 6: Final Output
- Package deliverables including 3D models, turntables, and documentation

## 📊 Quality Metrics

### Synthetic Image Quality
- **Consistency**: <5% scale variation across views
- **Coverage**: No gaps >10° in viewing angles
- **Realism**: Pass manual inspection for historical accuracy

### 3DGS Reconstruction Quality
- **Quantitative**: PSNR >28dB, SSIM >0.90, LPIPS <0.1
- **Visual**: No major artifacts, clear detail preservation
- **Geometric**: Accurate proportions vs. technical drawings

### Final Mesh Quality
- **Topology**: <500k triangles, mostly quads
- **Watertight**: No holes or non-manifold edges
- **Texture**: 4K resolution, minimal seams

## 🎯 Success Criteria

### Minimum Viable Success
- [ ] One complete Bismarck model
- [ ] Visual quality suitable for research publication
- [ ] Basic documentation of methodology

### Target Success
- [ ] Bismarck + USS Iowa models
- [ ] Quantitative comparison with baselines
- [ ] Reproducible pipeline with scripts

### Stretch Goals
- [ ] All three ships (+ Yamato)
- [ ] Real-time viewer application
- [ ] Published paper/blog post

## 🚀 Future Extensions

1. **VLM Integration**: Use archived VLM-editing workflow for intelligent model refinement
2. **Multi-ship Training**: Train on multiple ships simultaneously for better generalization
3. **Interactive Viewer**: WebGL-based 3DGS viewer for results
4. **Automated Pipeline**: End-to-end script from references to final model

## 🔧 Quick Start

See detailed implementation plans in [`docs/01MasterArchitecture.md`](docs/01MasterArchitecture.md)

---

*This architecture provides a clear path from historical references to high-quality 3D models using synthetic multi-view generation and 3DGS.*
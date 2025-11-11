# Complete Plan: 01-Image Generation Pipeline

## 🎯 Mission Statement
Generate 200-500 consistent, high-quality synthetic multi-view images of the Bismarck battleship using AI image generation constrained by historical technical drawings.

## 📋 Prerequisites

### Required Software
- [ ] ComfyUI installed and working
- [ ] SDXL model downloaded (base + refiner)
- [ ] ControlNet models installed:
  - [ ] depth_xl
  - [ ] canny_xl
  - [ ] lineart_xl
- [ ] Python 3.10+ environment
- [ ] CUDA-enabled PyTorch
- [ ] OpenCV for image processing

### Required Hardware
- [ ] RTX 5090 (32GB VRAM) as primary
- [ ] RTX 3090 (24GB VRAM) as secondary
- [ ] 64GB+ system RAM
- [ ] 500GB+ free storage

### Required Assets
- [ ] Bismarck technical drawings (see collection list below)
- [ ] Historical photo references
- [ ] 3D proxy model or ability to create one

## 📚 Phase 1: Reference Collection (Days 1-2)

### Technical Drawing Sources

#### Priority 1: Essential Drawings
1. **"Anatomy of the Ship: The Battleship Bismarck"** by Jack Brower
   - ISBN: 978-1844861200
   - Contains: Complete technical drawings, cross-sections
   - Where: Amazon, naval bookstores

2. **Kagero Super Drawings in 3D #16**
   - Detailed 3D reconstructions
   - Measurement annotations
   - Where: Kagero publishing, specialty stores

3. **Original Blohm & Voss Blueprints**
   - Bundesarchiv-Militärarchiv Freiburg
   - Online: Some available at German Federal Archives

#### Priority 2: Supporting References
1. **Jane's Fighting Ships 1940**
   - Period-accurate specifications
   - Comparison drawings

2. **Warship Profile #18: KM Bismarck**
   - Detail photographs
   - Paint schemes

3. **Model Shipways Plans**
   - 1:350 scale drawings
   - Often more accessible than originals

### Drawing Types Needed

```
bismarck_references/
├── technical_drawings/
│   ├── profile/
│   │   ├── starboard_profile.jpg    # Full side view
│   │   ├── port_profile.jpg         # Opposite side
│   │   └── inboard_profile.jpg      # Internal layout
│   ├── plans/
│   │   ├── main_deck.jpg            # Top-down views
│   │   ├── upper_deck.jpg
│   │   ├── boat_deck.jpg
│   │   └── platform_decks.jpg
│   ├── sections/
│   │   ├── section_frame_40.jpg     # Cross-sections
│   │   ├── section_frame_80.jpg
│   │   ├── section_frame_120.jpg
│   │   └── section_frame_160.jpg
│   └── details/
│       ├── turret_anton.jpg         # Main turrets
│       ├── turret_bruno.jpg
│       ├── bridge_structure.jpg
│       ├── funnel_details.jpg
│       └── secondary_armament.jpg
├── historical_photos/
│   ├── launch_1939/
│   ├── trials_1940/
│   └── operation_1941/
└── reference_measurements.txt
```

### Collection Checklist
- [ ] Scan/photograph drawings at 300+ DPI
- [ ] Align and crop consistently
- [ ] Create measurement reference sheet
- [ ] Note any discrepancies between sources

## 🔧 Phase 2: Technical Setup (Day 3)

### ComfyUI Configuration

#### 1. Install Required Nodes
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Fannovel16/comfyui_controlnet_aux
git clone https://github.com/ltdrdata/ComfyUI-Manager
git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale
```

#### 2. Model Setup
```
ComfyUI/models/
├── checkpoints/
│   ├── sd_xl_base_1.0.safetensors
│   ├── sd_xl_refiner_1.0.safetensors
│   └── juggernautXL_v9.safetensors  # Alternative
├── controlnet/
│   ├── controlnet-depth-sdxl-1.0.safetensors
│   ├── controlnet-canny-sdxl-1.0.safetensors
│   └── controlnet-lineart-sdxl-1.0.safetensors
└── loras/
    └── military_vehicles_xl.safetensors  # If available
```

#### 3. Workflow Templates
Create base workflows for:
- Single ControlNet (testing)
- Dual ControlNet (depth + line art)
- Triple ControlNet (depth + line + reference)
- Batch generation workflow

### Python Environment Setup

```bash
# Create environment
conda create -n bismarck_3dgs python=3.10
conda activate bismarck_3dgs

# Install dependencies
pip install opencv-python pillow numpy
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers diffusers
pip install tqdm matplotlib
```

## 🎨 Phase 3: 3D Proxy Creation (Day 4)

### Option A: Simple Box Model
```python
import numpy as np
import trimesh

def create_bismarck_proxy():
    """Create simplified 3D proxy for depth map generation"""
    
    # Main hull (simplified)
    hull = trimesh.creation.box(
        extents=[251, 36, 40]  # Length, beam, height in meters
    )
    
    # Superstructure blocks
    bridge = trimesh.creation.box(extents=[30, 20, 25])
    bridge.apply_translation([30, 0, 30])
    
    funnel = trimesh.creation.cylinder(radius=4, height=35)
    funnel.apply_translation([0, 0, 35])
    
    # Combine
    proxy = trimesh.util.concatenate([hull, bridge, funnel])
    
    return proxy
```

### Option B: Detailed Proxy
1. Use Blender to create basic shape
2. Model major features:
   - Hull with proper beam curve
   - Superstructure placement
   - Turret positions
   - Funnel location
3. Export as OBJ for rendering

### Depth Map Generation
```python
def render_depth_maps(proxy_model, num_views=72, output_dir="depth_maps"):
    """Render depth maps from multiple angles"""
    
    scene = proxy_model.scene()
    
    for i in range(num_views):
        angle = (i / num_views) * 2 * np.pi
        
        # Set camera position
        camera_transform = trimesh.transformations.rotation_matrix(
            angle, [0, 0, 1], point=[0, 0, 0]
        )
        
        # Render depth
        depth = scene.save_image(resolution=[2048, 2048], 
                                visible=True)
        
        # Save depth map
        save_depth_map(depth, f"{output_dir}/depth_{i:03d}.png")
```

## 🔄 Phase 4: Technical Drawing Processing (Day 5)

### Drawing Preparation Pipeline

```python
import cv2
import numpy as np

class DrawingProcessor:
    def __init__(self, drawing_path):
        self.image = cv2.imread(drawing_path)
        self.processed = None
        
    def extract_lines(self):
        """Extract clean line art from technical drawing"""
        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = enhanced.apply(gray)
        
        # Extract edges
        edges = cv2.Canny(enhanced, 50, 150)
        
        # Clean up lines
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Invert for white background
        self.processed = 255 - cleaned
        
        return self.processed
    
    def create_controlnet_input(self, target_size=(1024, 1024)):
        """Prepare drawing for ControlNet"""
        # Resize maintaining aspect ratio
        resized = self.resize_with_padding(self.processed, target_size)
        
        # Ensure proper format
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            
        return resized
```

### Multi-View Projection
```python
def project_drawing_to_view(profile_drawing, deck_drawing, angle):
    """Project 2D drawings to specific viewing angle"""
    
    if 85 < angle < 95 or 265 < angle < 275:
        # Near side view - use profile
        return profile_drawing
    elif angle < 5 or angle > 355:
        # Near front/back - interpolate
        return blend_drawings(profile_drawing, deck_drawing, 0.7)
    else:
        # Complex angle - use both
        return create_composite_view(profile_drawing, deck_drawing, angle)
```

## 🚀 Phase 5: Test Generation (Day 6)

### Single Image Test
```python
def test_single_generation():
    """Test pipeline with one image"""
    
    # Load inputs
    depth_map = load_depth_map("depth_maps/depth_045.png")
    line_art = load_line_art("drawings/profile_processed.png")
    
    # ComfyUI API call
    prompt = """
    German battleship Bismarck, 1941 configuration,
    aerial photograph, 45 degree angle,
    gray hull, wooden deck, Baltic sea,
    overcast lighting, historical accuracy,
    photorealistic, detailed
    """
    
    result = generate_with_controlnet(
        prompt=prompt,
        depth_control=depth_map,
        line_control=line_art,
        depth_strength=0.7,
        line_strength=0.8
    )
    
    return result
```

### Consistency Validation
```python
def validate_consistency(test_images):
    """Check if generated images maintain consistency"""
    
    checks = {
        'scale': check_ship_scale_consistency(test_images),
        'color': check_color_consistency(test_images),
        'details': check_detail_preservation(test_images),
        'lighting': check_lighting_consistency(test_images)
    }
    
    return all(checks.values()), checks
```

## 🏭 Phase 6: Production Generation (Day 7-8)

### Batch Generation Setup

```python
class BismarckGenerator:
    def __init__(self, config):
        self.config = config
        self.depth_maps = self.load_depth_maps()
        self.line_arts = self.load_line_arts()
        self.camera_poses = self.generate_camera_poses()
        
    def generate_full_dataset(self):
        """Generate complete multi-view dataset"""
        
        results = []
        
        # Progress tracking
        pbar = tqdm(total=self.config['num_views'])
        
        for i, pose in enumerate(self.camera_poses):
            # Select appropriate inputs
            depth = self.select_depth_map(pose)
            lines = self.select_line_art(pose)
            
            # Generate with retries
            for attempt in range(3):
                image = self.generate_view(pose, depth, lines)
                
                if self.validate_image(image):
                    results.append({
                        'image': image,
                        'pose': pose,
                        'metadata': self.create_metadata(i)
                    })
                    break
                    
            pbar.update(1)
            
            # Save progress
            if i % 10 == 0:
                self.save_checkpoint(results)
                
        return results
```

### Camera Distribution
```python
def generate_camera_poses(ship_length=251):
    """Create optimal camera distribution"""
    
    poses = []
    
    # Ring 1: Waterline level (144 views)
    for i in range(144):
        angle = i * 2.5
        poses.append({
            'azimuth': angle,
            'elevation': 0,
            'distance': ship_length * 1.5,
            'type': 'waterline'
        })
    
    # Ring 2: Elevated (72 views)
    for i in range(72):
        angle = i * 5
        poses.append({
            'azimuth': angle,
            'elevation': 20,
            'distance': ship_length * 1.8,
            'type': 'elevated'
        })
    
    # Ring 3: Aerial (36 views)
    for i in range(36):
        angle = i * 10
        poses.append({
            'azimuth': angle,
            'elevation': 45,
            'distance': ship_length * 2.0,
            'type': 'aerial'
        })
    
    # Special views
    special_views = [
        {'azimuth': 0, 'elevation': 90, 'distance': 300, 'type': 'top'},
        {'azimuth': 45, 'elevation': 60, 'distance': 200, 'type': 'detail'},
        # ... more special angles
    ]
    poses.extend(special_views)
    
    return poses
```

### Quality Control
```python
class QualityController:
    def __init__(self, reference_specs):
        self.specs = reference_specs
        self.rejected_count = 0
        
    def validate_image(self, image, pose):
        """Comprehensive image validation"""
        
        checks = {
            'ship_detected': self.detect_ship(image),
            'correct_orientation': self.check_orientation(image, pose),
            'scale_accurate': self.check_scale(image),
            'no_artifacts': self.check_artifacts(image),
            'lighting_consistent': self.check_lighting(image)
        }
        
        if not all(checks.values()):
            self.log_rejection(image, checks)
            self.rejected_count += 1
            return False
            
        return True
```

## 📊 Phase 7: Dataset Validation (Day 9)

### Coverage Analysis
```python
def analyze_dataset_coverage(images, poses):
    """Ensure complete angular coverage"""
    
    # Create coverage heatmap
    coverage_map = np.zeros((36, 18))  # 10° bins
    
    for pose in poses:
        az_bin = int(pose['azimuth'] / 10)
        el_bin = int(pose['elevation'] / 10)
        coverage_map[az_bin, el_bin] += 1
    
    # Find gaps
    gaps = np.where(coverage_map == 0)
    
    if len(gaps[0]) > 0:
        print(f"Coverage gaps found at: {list(zip(gaps[0] * 10, gaps[1] * 10))}")
        
    return coverage_map
```

### Final Quality Report
```python
def generate_quality_report(dataset):
    """Comprehensive dataset quality metrics"""
    
    report = {
        'total_images': len(dataset),
        'coverage': analyze_coverage(dataset),
        'consistency': {
            'scale_variance': calculate_scale_variance(dataset),
            'color_variance': calculate_color_variance(dataset),
            'detail_score': assess_detail_consistency(dataset)
        },
        'technical_accuracy': {
            'length_accuracy': measure_length_accuracy(dataset),
            'proportions': check_proportions(dataset),
            'feature_placement': validate_features(dataset)
        },
        'generation_stats': {
            'time_elapsed': dataset.generation_time,
            'rejection_rate': dataset.rejection_rate,
            'gpu_hours': dataset.gpu_time
        }
    }
    
    return report
```

## 🗓️ Timeline Summary

### Week 1 Schedule
- **Day 1-2**: Collect and process technical drawings
- **Day 3**: Set up ComfyUI and testing environment
- **Day 4**: Create 3D proxy model
- **Day 5**: Process drawings for ControlNet
- **Day 6**: Test generation and validation
- **Day 7-8**: Full production run
- **Day 9**: Validation and quality control

### Milestones
- [ ] All references collected and processed
- [ ] ComfyUI workflow tested and working
- [ ] First successful test image generated
- [ ] 50 consistent images generated
- [ ] 200+ images generated
- [ ] Full dataset validated
- [ ] Ready for 3DGS training

## 🚨 Contingency Plans

### If ControlNet isn't maintaining consistency:
1. Increase control strengths
2. Try different SDXL checkpoints
3. Use img2img with reference image
4. Generate more views and filter heavily

### If generation is too slow:
1. Reduce resolution to 1536x1536
2. Use SDXL-Turbo for initial tests
3. Parallelize on both GPUs
4. Reduce ControlNet layers

### If quality is insufficient:
1. Add more reference images
2. Fine-tune prompts
3. Try different samplers (DPM++ 2M Karras)
4. Increase sampling steps

## ✅ Success Criteria

### Minimum Requirements
- [ ] 200 images generated
- [ ] 360° coverage achieved
- [ ] <5% scale variance
- [ ] Consistent lighting
- [ ] Accurate proportions

### Target Goals
- [ ] 500 images generated
- [ ] <2% scale variance
- [ ] Multiple weather conditions
- [ ] Detail shots included
- [ ] Ready for immediate 3DGS training

## 🔗 Next Steps

Once image generation is complete:
1. Move images to `02-dataset-preparation/`
2. Generate camera pose metadata
3. Create COLMAP-compatible structure
4. Begin 3DGS training preparation
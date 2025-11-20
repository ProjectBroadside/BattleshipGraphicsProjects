# Splat Refinement Pipeline

## 🎯 Purpose
Clean, optimize, and refine the raw 3D Gaussian Splat output to remove artifacts and improve quality before mesh conversion.

## 📥 Inputs
- Trained `.ply` file from Stage 3 (3-8M splats)
- Original training images for reference
- Quality metrics from training

## 📤 Outputs
- Refined `.ply` file with cleaned splats
- Artifact removal report
- Optimized splat count
- Before/after comparison renders

## 🔧 Refinement Pipeline

### 1. Load and Analyze Splats

```python
import numpy as np
from plyfile import PlyData, PlyElement

class SplatRefiner:
    def __init__(self, ply_path):
        self.plydata = PlyData.read(ply_path)
        self.vertices = self.plydata['vertex']
        self.extract_properties()
        
    def extract_properties(self):
        # Position
        self.xyz = np.stack([
            self.vertices['x'],
            self.vertices['y'], 
            self.vertices['z']
        ], axis=1)
        
        # Scale (log scale)
        self.scales = np.stack([
            self.vertices['scale_0'],
            self.vertices['scale_1'],
            self.vertices['scale_2']
        ], axis=1)
        
        # Opacity (sigmoid)
        self.opacity = self.vertices['opacity']
        
        # Color (SH coefficients)
        self.colors = self.extract_sh_coefficients()
```

### 2. Artifact Detection & Removal

#### A. Floating Splats (Floaters)
```python
def remove_floaters(self, distance_threshold=10.0):
    """Remove splats far from main geometry"""
    # Find ship's bounding box
    ship_center = np.median(self.xyz, axis=0)
    distances = np.linalg.norm(self.xyz - ship_center, axis=1)
    
    # Statistical outlier detection
    dist_mean = np.mean(distances)
    dist_std = np.std(distances)
    outlier_threshold = dist_mean + 3 * dist_std
    
    # Remove far splats
    keep_mask = distances < min(distance_threshold, outlier_threshold)
    
    return self.filter_splats(keep_mask)
```

#### B. Giant Splats
```python
def remove_giant_splats(self, scale_threshold=1.0):
    """Remove abnormally large splats"""
    # Convert from log scale
    actual_scales = np.exp(self.scales)
    max_scales = np.max(actual_scales, axis=1)
    
    # Remove splats larger than threshold
    keep_mask = max_scales < scale_threshold
    
    print(f"Removing {np.sum(~keep_mask)} giant splats")
    return self.filter_splats(keep_mask)
```

#### C. Transparent Splats
```python
def remove_transparent_splats(self, opacity_threshold=0.01):
    """Remove nearly invisible splats"""
    # Convert from logit to probability
    actual_opacity = 1 / (1 + np.exp(-self.opacity))
    
    # Keep only visible splats
    keep_mask = actual_opacity > opacity_threshold
    
    return self.filter_splats(keep_mask)
```

### 3. Geometric Optimization

#### A. Splat Merging
```python
def merge_nearby_splats(self, distance_threshold=0.01):
    """Merge very close splats to reduce count"""
    from sklearn.cluster import DBSCAN
    
    # Cluster nearby splats
    clustering = DBSCAN(eps=distance_threshold, min_samples=2)
    labels = clustering.fit_predict(self.xyz)
    
    new_splats = []
    for label in np.unique(labels):
        if label == -1:  # Noise points
            continue
            
        mask = labels == label
        cluster_splats = self.get_splats(mask)
        
        # Merge into single splat
        merged = self.merge_splat_cluster(cluster_splats)
        new_splats.append(merged)
    
    return new_splats
```

#### B. Adaptive Decimation
```python
def adaptive_decimation(self, target_count=3_000_000):
    """Reduce splat count while preserving quality"""
    if len(self.xyz) <= target_count:
        return self
    
    # Calculate importance scores
    importance = self.calculate_importance()
    
    # Keep most important splats
    threshold = np.percentile(importance, 
                            (1 - target_count/len(self.xyz)) * 100)
    keep_mask = importance > threshold
    
    return self.filter_splats(keep_mask)
    
def calculate_importance(self):
    """Score splats by visual contribution"""
    # Factors: opacity, scale, view coverage
    opacity_score = 1 / (1 + np.exp(-self.opacity))
    scale_score = 1 - np.exp(-np.mean(self.scales, axis=1))
    
    importance = opacity_score * scale_score
    return importance
```

### 4. Ship-Specific Refinements

#### A. Waterline Cleanup
```python
def clean_waterline(self, z_water=0.0, tolerance=0.5):
    """Remove splats below waterline"""
    # Bismarck draft ~10m, but splats shouldn't be deep underwater
    keep_mask = self.xyz[:, 2] > (z_water - tolerance)
    
    return self.filter_splats(keep_mask)
```

#### B. Symmetry Enhancement
```python
def enhance_symmetry(self, axis='y', tolerance=0.1):
    """Ships are largely symmetric - enforce this"""
    # Find centerline
    center = np.median(self.xyz[:, 1])
    
    # For each splat, check if symmetric partner exists
    for i, pos in enumerate(self.xyz):
        mirror_pos = pos.copy()
        mirror_pos[1] = 2 * center - pos[1]  # Mirror across Y
        
        # Find nearest splat to mirror position
        distances = np.linalg.norm(self.xyz - mirror_pos, axis=1)
        min_dist = np.min(distances)
        
        if min_dist > tolerance:
            # No symmetric partner - might be artifact
            self.opacity[i] *= 0.5  # Reduce opacity
```

### 5. Quality Enhancement

#### A. Color Consistency
```python
def normalize_colors(self):
    """Ensure consistent ship coloring"""
    # Battleship gray with slight variations
    target_gray = np.array([0.5, 0.5, 0.52])  # Slight blue tint
    
    # Adjust SH coefficients towards target
    dc_colors = self.get_dc_component()  # DC term of SH
    
    # Soft correction towards gray
    alpha = 0.3  # Correction strength
    corrected = (1 - alpha) * dc_colors + alpha * target_gray
    
    self.set_dc_component(corrected)
```

#### B. Detail Preservation
```python
def preserve_details(self, reference_images):
    """Ensure important details aren't lost"""
    detail_regions = [
        {"name": "main_guns", "box": [...]},
        {"name": "bridge", "box": [...]},
        {"name": "secondary_armament", "box": [...]}
    ]
    
    for region in detail_regions:
        mask = self.get_splats_in_box(region["box"])
        
        # Increase importance of detail splats
        self.opacity[mask] = np.clip(self.opacity[mask] * 1.2, 
                                    None, 
                                    10.0)  # Logit scale
```

### 6. Export Refined Splats

```python
def save_refined_ply(self, output_path):
    """Save cleaned splats to new PLY file"""
    # Create vertex data
    vertex_data = []
    for i in range(len(self.xyz)):
        vertex = (
            self.xyz[i, 0], self.xyz[i, 1], self.xyz[i, 2],
            *self.colors[i],
            self.opacity[i],
            self.scales[i, 0], self.scales[i, 1], self.scales[i, 2],
            *self.rotations[i]
        )
        vertex_data.append(vertex)
    
    # Create PLY structure
    vertex = np.array(vertex_data, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        # ... other properties
    ])
    
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(output_path)
```

## 🧪 Validation Pipeline

### Before/After Comparison
```python
def compare_quality(original_ply, refined_ply, test_views):
    metrics = {
        'splat_count': [len(original), len(refined)],
        'file_size': [os.path.getsize(f) for f in [original, refined]],
        'render_quality': []
    }
    
    for view in test_views:
        orig_render = render_splats(original_ply, view)
        refined_render = render_splats(refined_ply, view)
        
        # Should maintain or improve quality
        metrics['render_quality'].append({
            'psnr_change': calculate_psnr(refined_render, orig_render),
            'ssim': calculate_ssim(refined_render, orig_render)
        })
    
    return metrics
```

## ⚙️ Refinement Presets

### Conservative (Minimal Changes)
```python
CONSERVATIVE = {
    'floater_distance': 20.0,
    'opacity_threshold': 0.005,
    'scale_threshold': 2.0,
    'merge_distance': 0.005
}
```

### Aggressive (Maximum Cleanup)
```python
AGGRESSIVE = {
    'floater_distance': 10.0,
    'opacity_threshold': 0.02,
    'scale_threshold': 0.5,
    'merge_distance': 0.01,
    'target_splats': 2_000_000
}
```

### Balanced (Recommended)
```python
BALANCED = {
    'floater_distance': 15.0,
    'opacity_threshold': 0.01,
    'scale_threshold': 1.0,
    'merge_distance': 0.008,
    'preserve_details': True
}
```

## 📊 Quality Metrics

### Expected Improvements
- **Splat Count**: 20-40% reduction
- **File Size**: 15-30% smaller
- **Rendering Speed**: 1.5-2x faster
- **Visual Quality**: Maintained or improved

### Success Criteria
- [ ] No visible floating artifacts
- [ ] Preserved ship details (guns, bridge, etc.)
- [ ] Smooth surfaces where appropriate
- [ ] Consistent coloring
- [ ] Watertight appearance

## ⚠️ Common Issues

### Problem: Over-aggressive cleaning removes details
**Solution**: Use conservative settings, mark detail regions

### Problem: Symmetry enforcement creates artifacts
**Solution**: Reduce correction strength, use soft constraints

### Problem: Color normalization looks unnatural
**Solution**: Reduce correction alpha, preserve original variations

## 🚀 Next Steps

After refinement:
1. Validate quality with test renders
2. Compare metrics with original
3. Export final refined PLY
4. Proceed to mesh conversion stage
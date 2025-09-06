# Naval Ship 2D Pipeline - ComfyUI Implementation Design

## Quick Start Guide

This document provides step-by-step instructions for implementing the 2D image processing pipeline in ComfyUI. Follow these instructions sequentially to build a working naval blueprint analysis workflow.

## 1. Environment Setup

### 1.1 ComfyUI Installation
```bash
# Clone ComfyUI if not already installed
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Install dependencies
pip install -r requirements.txt

# For GPU support (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 1.2 Required Custom Nodes Installation

```bash
cd ComfyUI/custom_nodes

# 1. ComfyUI-RMBG (Background Removal)
git clone https://github.com/Fannovel16/ComfyUI-RMBG.git
cd ComfyUI-RMBG
pip install -r requirements.txt
cd ..

# 2. ComfyUI-Anyline (Technical Line Detection)
git clone https://github.com/TheMistoAI/ComfyUI-Anyline.git
cd ComfyUI-Anyline
pip install -r requirements.txt
cd ..

# 3. Segment Anything (SAM)
git clone https://github.com/storyicon/comfyui_segment_anything.git
cd comfyui_segment_anything
pip install -r requirements.txt
# Download SAM model weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P models/
cd ..

# 4. YoloWorld-EfficientSAM
git clone https://github.com/AIrjen/OneButtonPrompt.git
cd OneButtonPrompt
pip install -r requirements.txt
cd ..

# 5. ComfyUI-Manager (for easier node management)
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
```

### 1.3 Model Downloads

Create model directories and download required models:

```bash
# In ComfyUI root directory
mkdir -p models/sam
mkdir -p models/yolo
mkdir -p models/anyline

# Download SAM model (if not done above)
cd models/sam
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Download YoloWorld model
cd ../yolo
wget https://github.com/AILab-CVC/YOLO-World/releases/download/v2.3/yolow-v8_x_clipv2_frozen_v2.pt

# Return to ComfyUI root
cd ../..
```

## 2. Workflow Construction

### 2.1 Basic Workflow Structure

Create a new workflow file: `naval_blueprint_processor.json`

```json
{
  "workflow_name": "Naval Blueprint 2D Processor",
  "version": "1.0",
  "description": "Extracts components from naval technical drawings"
}
```

### 2.2 Node Connection Diagram

```
┌──────────────┐
│ Image Loader │ → [image]
└──────────────┘
        ↓
┌──────────────┐
│     RMBG     │ → [clean_image, mask]
└──────────────┘
        ↓
    [Split into 3 parallel paths]
        ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│     SAM      │  │  YoloWorld   │  │   Anyline    │
└──────────────┘  └──────────────┘  └──────────────┘
        ↓               ↓                  ↓
   [segments]      [detections]       [contours]
        ↓               ↓                  ↓
┌──────────────────────────────────────────────────┐
│              Consensus Builder                   │
└──────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────┐
│              Naval Validator                     │
└──────────────────────────────────────────────────┘
                        ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ JSON Export  │  │ Preview Save │  │ Mask Export  │
└──────────────┘  └──────────────┘  └──────────────┘
```

### 2.3 Step-by-Step Node Setup

#### Step 1: Image Input
```python
# Node Type: Load Image
# Node ID: image_loader_1
{
  "class_type": "LoadImage",
  "inputs": {
    "image": "gangut_class_top.jpg",
    "upload": "image"
  },
  "outputs": ["IMAGE", "MASK"]
}
```

#### Step 2: Background Removal
```python
# Node Type: RMBG Background Removal
# Node ID: rmbg_1
{
  "class_type": "RMBG",
  "inputs": {
    "image": ["image_loader_1", 0],
    "model": "u2net",
    "threshold": 0.5,
    "edge_smooth": 2
  },
  "outputs": ["IMAGE", "MASK"]
}
```

#### Step 3: SAM Segmentation
```python
# Node Type: SAM Model Loader
# Node ID: sam_loader_1
{
  "class_type": "SAMModelLoader",
  "inputs": {
    "model_name": "sam_vit_h_4b8939.pth",
    "device_mode": "AUTO"
  },
  "outputs": ["SAM_MODEL"]
}

# Node Type: SAM Segment
# Node ID: sam_segment_1
{
  "class_type": "SAMSegment",
  "inputs": {
    "image": ["rmbg_1", 0],
    "sam_model": ["sam_loader_1", 0],
    "points_per_side": 32,
    "pred_iou_thresh": 0.88,
    "stability_score_thresh": 0.95,
    "crop_n_layers": 0,
    "crop_n_points_downscale_factor": 1
  },
  "outputs": ["MASKS", "BOXES"]
}
```

#### Step 4: YoloWorld Detection
```python
# Node Type: YoloWorld Detector
# Node ID: yolo_detector_1
{
  "class_type": "YoloWorldDetector",
  "inputs": {
    "image": ["rmbg_1", 0],
    "model": "yolow-v8_x_clipv2_frozen_v2.pt",
    "classes": "turret, superstructure, funnel, mast, gun mount, bridge",
    "confidence": 0.3,
    "nms_threshold": 0.4
  },
  "outputs": ["DETECTIONS", "LABELS", "SCORES"]
}
```

#### Step 5: Anyline Processing
```python
# Node Type: Anyline Detector
# Node ID: anyline_1
{
  "class_type": "AnylineDetector",
  "inputs": {
    "image": ["rmbg_1", 0],
    "method": "EDSD",
    "threshold1": 50,
    "threshold2": 150,
    "line_threshold": 10,
    "min_line_length": 20,
    "max_line_gap": 5
  },
  "outputs": ["CONTOURS", "LINES"]
}
```

## 3. Custom Node Development

### 3.1 Consensus Builder Node

Create file: `ComfyUI/custom_nodes/naval_consensus/consensus_builder.py`

```python
import numpy as np
import torch
from typing import List, Dict, Tuple
import json

class ConsensusBuilder:
    """
    Combines detections from multiple methods into unified component list.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam_masks": ("MASKS",),
                "yolo_detections": ("DETECTIONS",),
                "anyline_contours": ("CONTOURS",),
                "iou_threshold": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.9}),
                "min_methods": ("INT", {"default": 2, "min": 1, "max": 3}),
            }
        }
    
    RETURN_TYPES = ("COMPONENTS", "CONFIDENCE_MAP", "JSON")
    FUNCTION = "build_consensus"
    CATEGORY = "Naval/Processing"
    
    def build_consensus(self, sam_masks, yolo_detections, anyline_contours, 
                       iou_threshold, min_methods):
        """
        Main consensus building logic.
        """
        components = []
        
        # Convert all inputs to standardized format
        sam_components = self.parse_sam_masks(sam_masks)
        yolo_components = self.parse_yolo_detections(yolo_detections)
        anyline_components = self.parse_anyline_contours(anyline_contours)
        
        # Find overlapping detections
        all_detections = sam_components + yolo_components + anyline_components
        consensus_groups = self.group_by_overlap(all_detections, iou_threshold)
        
        # Build consensus for each group
        for group in consensus_groups:
            if len(group) >= min_methods:
                component = self.merge_detections(group)
                component['confidence'] = len(group) / 3.0
                components.append(component)
        
        # Generate outputs
        confidence_map = self.create_confidence_map(components)
        json_output = json.dumps(components, indent=2)
        
        return (components, confidence_map, json_output)
    
    def parse_sam_masks(self, masks):
        """Convert SAM masks to component format."""
        components = []
        for i, mask in enumerate(masks):
            bbox = self.mask_to_bbox(mask)
            components.append({
                'source': 'sam',
                'type': 'unknown',
                'bbox': bbox,
                'mask': mask,
                'id': f'sam_{i}'
            })
        return components
    
    def parse_yolo_detections(self, detections):
        """Convert YOLO detections to component format."""
        components = []
        for i, det in enumerate(detections):
            components.append({
                'source': 'yolo',
                'type': det['class'],
                'bbox': det['bbox'],
                'confidence': det['score'],
                'id': f'yolo_{i}'
            })
        return components
    
    def parse_anyline_contours(self, contours):
        """Convert Anyline contours to component format."""
        components = []
        for i, contour in enumerate(contours):
            bbox = self.contour_to_bbox(contour)
            components.append({
                'source': 'anyline',
                'type': 'structure',
                'bbox': bbox,
                'contour': contour,
                'id': f'anyline_{i}'
            })
        return components
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union for two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def group_by_overlap(self, detections, threshold):
        """Group detections that overlap above threshold."""
        groups = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
                
            group = [det1]
            used.add(i)
            
            for j, det2 in enumerate(detections[i+1:], i+1):
                if j not in used:
                    iou = self.calculate_iou(det1['bbox'], det2['bbox'])
                    if iou > threshold:
                        group.append(det2)
                        used.add(j)
            
            groups.append(group)
        
        return groups
    
    def merge_detections(self, group):
        """Merge a group of overlapping detections into single component."""
        # Average bounding box
        bboxes = [d['bbox'] for d in group]
        avg_bbox = np.mean(bboxes, axis=0).tolist()
        
        # Vote on type
        types = [d.get('type', 'unknown') for d in group]
        type_counts = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1
        best_type = max(type_counts, key=type_counts.get)
        
        # Collect all source methods
        sources = [d['source'] for d in group]
        
        return {
            'type': best_type,
            'bbox': avg_bbox,
            'sources': sources,
            'detection_count': len(group)
        }
    
    def mask_to_bbox(self, mask):
        """Convert binary mask to bounding box."""
        rows = torch.any(mask, dim=1)
        cols = torch.any(mask, dim=0)
        ymin, ymax = torch.where(rows)[0][[0, -1]]
        xmin, xmax = torch.where(cols)[0][[0, -1]]
        return [xmin.item(), ymin.item(), xmax.item(), ymax.item()]
    
    def contour_to_bbox(self, contour):
        """Convert contour points to bounding box."""
        x_coords = contour[:, 0]
        y_coords = contour[:, 1]
        return [x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max()]
    
    def create_confidence_map(self, components):
        """Create visual confidence map for components."""
        # Implementation for visual confidence overlay
        pass
```

### 3.2 Naval Validator Node

Create file: `ComfyUI/custom_nodes/naval_consensus/naval_validator.py`

```python
class NavalValidator:
    """
    Validates detected components against naval architecture rules.
    """
    
    # Naval architecture constraints
    RULES = {
        'turret_count': {'min': 2, 'max': 6},
        'funnel_count': {'min': 1, 'max': 4},
        'superstructure_count': {'min': 1, 'max': 1},
        'symmetry_tolerance': 0.1,  # 10% deviation allowed
        'turret_spacing_ratio': 0.15,  # Min 15% ship length between turrets
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "components": ("COMPONENTS",),
                "image_width": ("INT",),
                "image_height": ("INT",),
                "ship_class": (["battleship", "cruiser", "destroyer", "carrier"],),
                "strict_mode": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("VALIDATED_COMPONENTS", "CORRECTIONS", "REPORT")
    FUNCTION = "validate_components"
    CATEGORY = "Naval/Validation"
    
    def validate_components(self, components, image_width, image_height, 
                           ship_class, strict_mode):
        """
        Apply naval architecture validation rules.
        """
        validated = []
        corrections = []
        report = {
            'total_components': len(components),
            'validated': 0,
            'corrected': 0,
            'rejected': 0,
            'warnings': []
        }
        
        # Check component counts
        type_counts = self.count_component_types(components)
        count_issues = self.validate_counts(type_counts, ship_class)
        report['warnings'].extend(count_issues)
        
        # Check symmetry
        symmetry_issues = self.check_symmetry(components, image_width)
        report['warnings'].extend(symmetry_issues)
        
        # Validate individual components
        for component in components:
            validation = self.validate_component(component, ship_class, image_width)
            
            if validation['valid']:
                validated.append(component)
                report['validated'] += 1
            elif validation['correctable'] and not strict_mode:
                corrected = self.correct_component(component, validation['corrections'])
                validated.append(corrected)
                corrections.append(validation['corrections'])
                report['corrected'] += 1
            else:
                report['rejected'] += 1
        
        # Generate detailed report
        report_text = self.generate_report(report)
        
        return (validated, corrections, report_text)
    
    def count_component_types(self, components):
        """Count components by type."""
        counts = {}
        for comp in components:
            comp_type = comp.get('type', 'unknown')
            counts[comp_type] = counts.get(comp_type, 0) + 1
        return counts
    
    def validate_counts(self, counts, ship_class):
        """Check if component counts are reasonable."""
        issues = []
        
        # Check turret count
        turret_count = counts.get('turret', 0)
        if turret_count < self.RULES['turret_count']['min']:
            issues.append(f"Too few turrets detected: {turret_count}")
        elif turret_count > self.RULES['turret_count']['max']:
            issues.append(f"Too many turrets detected: {turret_count}")
        
        # Check funnel count
        funnel_count = counts.get('funnel', 0)
        if funnel_count < self.RULES['funnel_count']['min']:
            issues.append(f"No funnels detected")
        
        return issues
    
    def check_symmetry(self, components, image_width):
        """Check bilateral symmetry of components."""
        issues = []
        centerline = image_width / 2
        
        # Group components by type
        by_type = {}
        for comp in components:
            comp_type = comp.get('type', 'unknown')
            if comp_type not in by_type:
                by_type[comp_type] = []
            by_type[comp_type].append(comp)
        
        # Check symmetry for paired components
        for comp_type, comps in by_type.items():
            if comp_type in ['turret', 'secondary_gun']:
                for comp in comps:
                    center_x = (comp['bbox'][0] + comp['bbox'][2]) / 2
                    distance_from_center = abs(center_x - centerline)
                    
                    # Look for symmetric pair
                    mirror_x = centerline - (center_x - centerline)
                    has_pair = self.find_component_near(comps, mirror_x, comp['bbox'][1])
                    
                    if not has_pair and distance_from_center > image_width * 0.1:
                        issues.append(f"No symmetric pair for {comp_type} at x={center_x}")
        
        return issues
    
    def find_component_near(self, components, x, y, tolerance=50):
        """Find component near given position."""
        for comp in components:
            center_x = (comp['bbox'][0] + comp['bbox'][2]) / 2
            center_y = (comp['bbox'][1] + comp['bbox'][3]) / 2
            
            if abs(center_x - x) < tolerance and abs(center_y - y) < tolerance:
                return True
        return False
    
    def validate_component(self, component, ship_class, image_width):
        """Validate individual component."""
        validation = {
            'valid': True,
            'correctable': False,
            'corrections': {}
        }
        
        # Check bounding box sanity
        bbox = component['bbox']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        # Component shouldn't be too small
        if width < 10 or height < 10:
            validation['valid'] = False
            validation['correctable'] = False
        
        # Component shouldn't be larger than ship
        if width > image_width * 0.5:
            validation['valid'] = False
            validation['correctable'] = True
            validation['corrections']['resize'] = 0.5
        
        return validation
    
    def correct_component(self, component, corrections):
        """Apply corrections to component."""
        corrected = component.copy()
        
        if 'resize' in corrections:
            scale = corrections['resize']
            bbox = corrected['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            width = (bbox[2] - bbox[0]) * scale
            height = (bbox[3] - bbox[1]) * scale
            
            corrected['bbox'] = [
                center_x - width/2,
                center_y - height/2,
                center_x + width/2,
                center_y + height/2
            ]
        
        return corrected
    
    def generate_report(self, report):
        """Generate human-readable validation report."""
        lines = [
            "Naval Architecture Validation Report",
            "=" * 40,
            f"Total Components: {report['total_components']}",
            f"Validated: {report['validated']}",
            f"Corrected: {report['corrected']}",
            f"Rejected: {report['rejected']}",
            "",
            "Warnings:",
        ]
        
        for warning in report['warnings']:
            lines.append(f"  - {warning}")
        
        return "\n".join(lines)
```

## 4. Workflow Configuration

### 4.1 Performance Settings

Create `config/naval_pipeline.yaml`:

```yaml
pipeline:
  name: "Naval Blueprint Processor"
  version: "1.0"
  
performance:
  batch_size: 4
  gpu_allocation:
    rtx_5090:
      - sam_segmentation
      - yolo_detection
    rtx_3090:
      - anyline_processing
      - consensus_building
  
  cache:
    enable: true
    max_size_gb: 8
    ttl_seconds: 3600

detection:
  sam:
    model: "sam_vit_h_4b8939.pth"
    points_per_side: 32
    pred_iou_thresh: 0.88
    stability_score_thresh: 0.95
    
  yolo:
    model: "yolow-v8_x_clipv2_frozen_v2.pt"
    confidence: 0.3
    nms_threshold: 0.4
    classes:
      - turret
      - superstructure
      - funnel
      - mast
      - gun_mount
      - bridge
      - secondary_armament
      
  anyline:
    method: "EDSD"
    threshold1: 50
    threshold2: 150
    line_threshold: 10
    
consensus:
  iou_threshold: 0.5
  min_methods: 2
  confidence_weights:
    sam: 0.4
    yolo: 0.35
    anyline: 0.25
    
validation:
  strict_mode: false
  rules:
    turret_count:
      min: 2
      max: 6
    funnel_count:
      min: 1
      max: 4
    symmetry_tolerance: 0.1
```

### 4.2 Workflow JSON Template

Complete workflow template: `workflows/naval_blueprint_complete.json`

```json
{
  "last_node_id": 15,
  "last_link_id": 20,
  "nodes": [
    {
      "id": 1,
      "type": "LoadImage",
      "pos": [50, 100],
      "size": [315, 158],
      "flags": {},
      "order": 0,
      "mode": 0,
      "outputs": [
        {"name": "IMAGE", "type": "IMAGE", "links": [1, 2, 3]},
        {"name": "MASK", "type": "MASK", "links": null}
      ],
      "properties": {},
      "widgets_values": ["gangut_class_top.jpg", "image"]
    },
    {
      "id": 2,
      "type": "RMBG",
      "pos": [400, 100],
      "size": [315, 178],
      "flags": {},
      "order": 1,
      "mode": 0,
      "inputs": [
        {"name": "image", "type": "IMAGE", "link": 1}
      ],
      "outputs": [
        {"name": "IMAGE", "type": "IMAGE", "links": [4, 5, 6]},
        {"name": "MASK", "type": "MASK", "links": [7]}
      ],
      "properties": {},
      "widgets_values": ["u2net", 0.5, 2]
    },
    {
      "id": 3,
      "type": "SAMModelLoader",
      "pos": [750, 50],
      "size": [315, 82],
      "flags": {},
      "order": 0,
      "mode": 0,
      "outputs": [
        {"name": "SAM_MODEL", "type": "SAM_MODEL", "links": [8]}
      ],
      "properties": {},
      "widgets_values": ["sam_vit_h_4b8939.pth", "AUTO"]
    },
    {
      "id": 4,
      "type": "SAMSegment",
      "pos": [750, 200],
      "size": [315, 242],
      "flags": {},
      "order": 2,
      "mode": 0,
      "inputs": [
        {"name": "image", "type": "IMAGE", "link": 4},
        {"name": "sam_model", "type": "SAM_MODEL", "link": 8}
      ],
      "outputs": [
        {"name": "MASKS", "type": "MASKS", "links": [9]},
        {"name": "BOXES", "type": "BOXES", "links": null}
      ],
      "properties": {},
      "widgets_values": [32, 0.88, 0.95, 0, 1]
    },
    {
      "id": 5,
      "type": "YoloWorldDetector",
      "pos": [750, 500],
      "size": [315, 178],
      "flags": {},
      "order": 2,
      "mode": 0,
      "inputs": [
        {"name": "image", "type": "IMAGE", "link": 5}
      ],
      "outputs": [
        {"name": "DETECTIONS", "type": "DETECTIONS", "links": [10]},
        {"name": "LABELS", "type": "LABELS", "links": null},
        {"name": "SCORES", "type": "SCORES", "links": null}
      ],
      "properties": {},
      "widgets_values": [
        "yolow-v8_x_clipv2_frozen_v2.pt",
        "turret, superstructure, funnel, mast, gun mount, bridge",
        0.3,
        0.4
      ]
    },
    {
      "id": 6,
      "type": "AnylineDetector",
      "pos": [750, 750],
      "size": [315, 210],
      "flags": {},
      "order": 2,
      "mode": 0,
      "inputs": [
        {"name": "image", "type": "IMAGE", "link": 6}
      ],
      "outputs": [
        {"name": "CONTOURS", "type": "CONTOURS", "links": [11]},
        {"name": "LINES", "type": "LINES", "links": null}
      ],
      "properties": {},
      "widgets_values": ["EDSD", 50, 150, 10, 20, 5]
    },
    {
      "id": 7,
      "type": "ConsensusBuilder",
      "pos": [1150, 350],
      "size": [315, 158],
      "flags": {},
      "order": 3,
      "mode": 0,
      "inputs": [
        {"name": "sam_masks", "type": "MASKS", "link": 9},
        {"name": "yolo_detections", "type": "DETECTIONS", "link": 10},
        {"name": "anyline_contours", "type": "CONTOURS", "link": 11}
      ],
      "outputs": [
        {"name": "COMPONENTS", "type": "COMPONENTS", "links": [12]},
        {"name": "CONFIDENCE_MAP", "type": "IMAGE", "links": [13]},
        {"name": "JSON", "type": "STRING", "links": [14]}
      ],
      "properties": {},
      "widgets_values": [0.5, 2]
    },
    {
      "id": 8,
      "type": "NavalValidator",
      "pos": [1500, 350],
      "size": [315, 178],
      "flags": {},
      "order": 4,
      "mode": 0,
      "inputs": [
        {"name": "components", "type": "COMPONENTS", "link": 12}
      ],
      "outputs": [
        {"name": "VALIDATED_COMPONENTS", "type": "COMPONENTS", "links": [15]},
        {"name": "CORRECTIONS", "type": "JSON", "links": null},
        {"name": "REPORT", "type": "STRING", "links": [16]}
      ],
      "properties": {},
      "widgets_values": [1920, 1080, "battleship", false]
    },
    {
      "id": 9,
      "type": "SaveJSON",
      "pos": [1850, 300],
      "size": [315, 96],
      "flags": {},
      "order": 5,
      "mode": 0,
      "inputs": [
        {"name": "json_string", "type": "STRING", "link": 14}
      ],
      "properties": {},
      "widgets_values": ["output/components.json"]
    },
    {
      "id": 10,
      "type": "PreviewImage",
      "pos": [1850, 450],
      "size": [315, 246],
      "flags": {},
      "order": 5,
      "mode": 0,
      "inputs": [
        {"name": "images", "type": "IMAGE", "link": 13}
      ],
      "properties": {}
    },
    {
      "id": 11,
      "type": "ShowText",
      "pos": [1850, 750],
      "size": [315, 180],
      "flags": {},
      "order": 5,
      "mode": 0,
      "inputs": [
        {"name": "text", "type": "STRING", "link": 16}
      ],
      "properties": {},
      "widgets_values": [""]
    }
  ],
  "links": [
    [1, 1, 0, 2, 0, "IMAGE"],
    [2, 1, 0, 2, 0, "IMAGE"],
    [3, 1, 0, 2, 0, "IMAGE"],
    [4, 2, 0, 4, 0, "IMAGE"],
    [5, 2, 0, 5, 0, "IMAGE"],
    [6, 2, 0, 6, 0, "IMAGE"],
    [7, 2, 1, null, null, "MASK"],
    [8, 3, 0, 4, 1, "SAM_MODEL"],
    [9, 4, 0, 7, 0, "MASKS"],
    [10, 5, 0, 7, 1, "DETECTIONS"],
    [11, 6, 0, 7, 2, "CONTOURS"],
    [12, 7, 0, 8, 0, "COMPONENTS"],
    [13, 7, 1, 10, 0, "IMAGE"],
    [14, 7, 2, 9, 0, "STRING"],
    [15, 8, 0, null, null, "COMPONENTS"],
    [16, 8, 2, 11, 0, "STRING"]
  ],
  "groups": [],
  "config": {},
  "extra": {}
}
```

## 5. Testing & Validation

### 5.1 Test Script

Create `test_pipeline.py`:

```python
#!/usr/bin/env python3
"""
Test script for naval blueprint processing pipeline.
"""

import os
import json
import time
import requests
from pathlib import Path

class PipelineTester:
    def __init__(self, comfy_url="http://127.0.0.1:8188"):
        self.url = comfy_url
        self.test_images = [
            "TestingImages/gangut_class_top.jpg",
            "TestingImages/gangut_class_side.jpg",
            "TestingImages/iowa_class_top.jpg",
            "TestingImages/iowa_class_side.jpg"
        ]
        
    def run_workflow(self, image_path):
        """Execute workflow for single image."""
        print(f"Processing: {image_path}")
        
        # Load workflow
        with open("workflows/naval_blueprint_complete.json", "r") as f:
            workflow = json.load(f)
        
        # Update image path
        workflow["nodes"][0]["widgets_values"][0] = image_path
        
        # Submit to ComfyUI
        response = requests.post(
            f"{self.url}/api/queue",
            json={"prompt": workflow}
        )
        
        if response.status_code == 200:
            job_id = response.json()["job_id"]
            return self.wait_for_completion(job_id)
        else:
            print(f"Error submitting workflow: {response.status_code}")
            return None
    
    def wait_for_completion(self, job_id, timeout=60):
        """Wait for workflow to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = requests.get(f"{self.url}/api/job/{job_id}")
            
            if response.status_code == 200:
                status = response.json()["status"]
                
                if status == "completed":
                    return response.json()["outputs"]
                elif status == "failed":
                    print(f"Job {job_id} failed")
                    return None
            
            time.sleep(1)
        
        print(f"Job {job_id} timed out")
        return None
    
    def validate_output(self, output_path):
        """Validate the JSON output."""
        with open(output_path, "r") as f:
            data = json.load(f)
        
        # Check structure
        assert "views" in data
        assert "components" in data["views"].get("top", {})
        
        # Check component detection
        components = data["views"]["top"]["components"]
        print(f"  Detected {len(components)} components")
        
        # Check for essential components
        types_found = [c["type"] for c in components]
        essential = ["turret", "superstructure", "funnel"]
        
        for comp_type in essential:
            if comp_type in types_found:
                print(f"  ✓ Found {comp_type}")
            else:
                print(f"  ✗ Missing {comp_type}")
        
        # Calculate metrics
        avg_confidence = sum(c.get("confidence", 0) for c in components) / len(components)
        print(f"  Average confidence: {avg_confidence:.2%}")
        
        return len(components) > 0 and avg_confidence > 0.7
    
    def run_tests(self):
        """Run full test suite."""
        print("Naval Blueprint Pipeline Test Suite")
        print("=" * 50)
        
        results = []
        
        for image in self.test_images:
            print(f"\nTest: {image}")
            print("-" * 30)
            
            # Run workflow
            start = time.time()
            output = self.run_workflow(image)
            duration = time.time() - start
            
            if output:
                print(f"  Processing time: {duration:.1f}s")
                
                # Validate output
                output_file = f"output/{Path(image).stem}_components.json"
                success = self.validate_output(output_file)
                
                results.append({
                    "image": image,
                    "success": success,
                    "duration": duration
                })
            else:
                results.append({
                    "image": image,
                    "success": False,
                    "duration": duration
                })
        
        # Summary
        print("\n" + "=" * 50)
        print("Test Summary")
        print("-" * 50)
        
        successful = sum(1 for r in results if r["success"])
        print(f"Success rate: {successful}/{len(results)}")
        
        avg_time = sum(r["duration"] for r in results) / len(results)
        print(f"Average processing time: {avg_time:.1f}s")
        
        return successful == len(results)

if __name__ == "__main__":
    tester = PipelineTester()
    success = tester.run_tests()
    exit(0 if success else 1)
```

### 5.2 Debugging Checklist

```markdown
## Debugging Checklist

### If detection is failing:
- [ ] Check image preprocessing (RMBG output)
- [ ] Verify model files are downloaded correctly
- [ ] Adjust confidence thresholds
- [ ] Check GPU memory usage
- [ ] Review console logs for errors

### If consensus is poor:
- [ ] Lower IoU threshold for grouping
- [ ] Adjust minimum methods requirement
- [ ] Check individual method outputs
- [ ] Verify bounding box formats match

### If validation is too strict:
- [ ] Disable strict_mode
- [ ] Adjust naval rule thresholds
- [ ] Review warning messages
- [ ] Check ship class selection

### Performance issues:
- [ ] Reduce batch size
- [ ] Check GPU allocation
- [ ] Monitor memory usage
- [ ] Profile individual nodes
- [ ] Consider resolution reduction
```

## 6. Production Deployment

### 6.1 API Wrapper

Create `api/naval_processor.py`:

```python
from flask import Flask, request, jsonify
import subprocess
import json

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_blueprint():
    """API endpoint for blueprint processing."""
    
    # Get image from request
    image = request.files['image']
    view_type = request.form.get('view', 'top')
    ship_class = request.form.get('class', 'battleship')
    
    # Save temporarily
    temp_path = f"/tmp/{image.filename}"
    image.save(temp_path)
    
    # Run ComfyUI workflow
    result = subprocess.run([
        "python", "run_workflow.py",
        "--image", temp_path,
        "--view", view_type,
        "--class", ship_class
    ], capture_output=True)
    
    # Parse output
    output = json.loads(result.stdout)
    
    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 6.2 Batch Processing Script

```bash
#!/bin/bash
# batch_process.sh - Process multiple blueprints

INPUT_DIR="./TestingImages"
OUTPUT_DIR="./output"
WORKFLOW="workflows/naval_blueprint_complete.json"

# Create output directory
mkdir -p $OUTPUT_DIR

# Process each image
for image in $INPUT_DIR/*.jpg; do
    basename=$(basename "$image" .jpg)
    echo "Processing: $basename"
    
    # Run ComfyUI workflow
    python -c "
import json
import requests

# Load and modify workflow
with open('$WORKFLOW', 'r') as f:
    workflow = json.load(f)

workflow['nodes'][0]['widgets_values'][0] = '$image'

# Submit to ComfyUI
response = requests.post('http://127.0.0.1:8188/api/queue', json={'prompt': workflow})
print(f'Submitted: {response.json()}')
"
    
    # Wait for completion
    sleep 30
    
    # Move output
    mv output/components.json "$OUTPUT_DIR/${basename}_components.json"
    
    echo "Completed: $basename"
done

echo "Batch processing complete!"
```

## 7. Troubleshooting Guide

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| SAM not detecting components | Model not loaded | Check model path in SAMModelLoader |
| YOLO missing detections | Classes not defined | Update class list in node settings |
| Anyline producing noise | Threshold too low | Increase threshold1 and threshold2 |
| Consensus conflicts | IoU threshold too high | Lower to 0.3-0.4 |
| Memory errors | Batch size too large | Reduce batch_size in config |
| Slow processing | Single GPU usage | Enable dual GPU in config |

## 8. Next Steps

1. **Fine-tuning**: Adjust detection parameters based on test results
2. **Custom Training**: Train YOLO on naval-specific dataset
3. **Scale Extraction**: Implement automatic scale detection
4. **Multi-View**: Add cross-view correlation logic
5. **3D Integration**: Connect to 3D reconstruction pipeline

---
*Implementation Guide Version: 1.0*
*Last Updated: 2025-01-06*
*Status: Ready for Implementation*
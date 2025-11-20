# Naval Ship 2D Image Processing Pipeline Architecture

## Executive Summary

This document defines the architecture, requirements, and strategic approach for the 2D image processing pipeline that transforms naval technical drawings into structured component data. This pipeline forms the critical first stage of the Naval Ship 3D Model Generator project, establishing the foundation for accurate 3D reconstruction.

## 1. Vision & Objectives

### 1.1 Primary Vision
Create a professional-grade computer vision pipeline that achieves near-human accuracy in interpreting naval technical drawings, extracting component information with the precision required for historical and engineering applications.

### 1.2 Core Objectives
1. **Automated Component Detection**: Eliminate manual identification of ship components
2. **High Accuracy Classification**: Achieve 90%+ accuracy in component type identification
3. **Precise Spatial Mapping**: Extract exact positions and dimensions with <5% error
4. **Multi-View Correlation**: Seamlessly integrate top and side view data
5. **Production Scalability**: Process 50+ ship blueprints per day

## 2. Functional Requirements

### 2.1 Input Requirements
- **Format Support**: JPEG, PNG, TIFF technical drawings
- **Resolution**: Minimum 1920x1080, optimal 4K (3840x2160)
- **View Types**: Top view (plan), Side view (profile)
- **Drawing Types**: Line drawings, blueprints, technical schematics
- **Era Coverage**: WWI through modern vessels

### 2.2 Component Detection Requirements

#### Primary Components (Must Detect)
- **Main Armament**: Primary gun turrets and barbettes
- **Superstructure**: Bridge, command towers, observation posts
- **Propulsion**: Funnels, exhaust stacks
- **Secondary Armament**: Secondary gun mounts, AA positions
- **Navigation**: Masts, radar arrays, communication equipment
- **Hull**: Complete hull outline and waterline

#### Secondary Components (Should Detect)
- Lifeboats and davits
- Searchlights
- Torpedo tubes
- Crane equipment
- Aircraft catapults
- Deck structures

### 2.3 Output Requirements

#### Data Structure
```json
{
  "ship_id": "unique_identifier",
  "metadata": {
    "class": "ship_class_name",
    "processing_date": "ISO_timestamp",
    "confidence_score": 0.95
  },
  "views": {
    "top": {
      "components": [
        {
          "id": "component_uuid",
          "type": "main_turret",
          "subtype": "triple_16_inch",
          "bbox": [x1, y1, x2, y2],
          "mask": "base64_encoded_mask",
          "confidence": 0.92,
          "attributes": {
            "position": "forward",
            "count": 3,
            "estimated_caliber": "406mm"
          }
        }
      ],
      "hull_contour": [[x, y], ...],
      "scale_factors": {
        "pixels_per_meter": 4.5,
        "reference_length": 263.0
      }
    },
    "side": { ... }
  },
  "cross_view_correlations": [
    {
      "top_component_id": "uuid1",
      "side_component_id": "uuid2",
      "correlation_confidence": 0.88
    }
  ]
}
```

### 2.4 Performance Requirements
- **Accuracy**: 90% component detection rate
- **Precision**: 95% classification accuracy for detected components
- **Speed**: <30 seconds per blueprint pair
- **Reliability**: <2% failure rate
- **Scalability**: Batch processing capability for 100+ images

## 3. Technical Architecture

### 3.1 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT LAYER                              │
│  Blueprint Images → Preprocessing → Quality Assessment       │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  DETECTION LAYER                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Method 1   │  │   Method 2   │  │   Method 3   │ ...    │
│  │     SAM      │  │  YoloWorld   │  │   Anyline    │       │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  CONSENSUS LAYER                             │
│  Multi-Method Validation → Conflict Resolution → Scoring     │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  VALIDATION LAYER                            │
│  Naval Architecture Rules → Physics Constraints → QA         │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   OUTPUT LAYER                               │
│  Structured JSON → Confidence Metrics → Export Formats       │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Processing Pipeline Stages

#### Stage 1: Preprocessing
1. **Image Normalization**: Standardize format, resolution, color space
2. **Background Removal**: RMBG for clean extraction
3. **Noise Reduction**: Remove artifacts, text, annotations
4. **Orientation Correction**: Ensure proper alignment
5. **Quality Assessment**: Reject low-quality inputs

#### Stage 2: Multi-Method Detection
1. **SAM (Segment Anything)**
   - Role: Primary component segmentation
   - Strength: Zero-shot object boundaries
   - Output: Pixel-perfect masks

2. **YoloWorld-EfficientSAM**
   - Role: Component classification
   - Strength: Open-vocabulary detection
   - Output: Typed bounding boxes

3. **Anyline**
   - Role: Technical line detection
   - Strength: Engineering drawing expertise
   - Output: Vectorized contours

4. **OpenCV Traditional**
   - Role: Backup and validation
   - Strength: Deterministic geometry
   - Output: Mathematical contours

5. **Florence-2** (Optional)
   - Role: Alternative AI detection
   - Strength: Unified vision model
   - Output: Multi-task predictions

#### Stage 3: Consensus Building
```python
consensus_algorithm:
  for each detected component:
    if detected by >= 3 methods:
      confidence = high (0.9+)
    elif detected by 2 methods:
      confidence = medium (0.7-0.9)
    elif detected by 1 method:
      if method_confidence > threshold:
        confidence = low (0.5-0.7)
      else:
        reject detection
```

#### Stage 4: Naval Validation
- **Symmetry Check**: Warships are bilaterally symmetric
- **Component Count**: Validate expected quantities
- **Proportion Check**: Ensure realistic dimensions
- **Position Logic**: Components in sensible locations
- **Historical Accuracy**: Match known ship classes

## 4. ComfyUI Integration Strategy

### 4.1 Node Architecture

```
[Image Load] → [RMBG Background Removal] → [Clean Image]
                                               ↓
                            ┌──────────────────┼──────────────────┐
                            ↓                  ↓                  ↓
                    [SAM Segmentation]  [YoloWorld]      [Anyline Detection]
                            ↓                  ↓                  ↓
                    [Component Masks]   [Classified BBs]  [Vector Lines]
                            ↓                  ↓                  ↓
                            └──────────────────┼──────────────────┘
                                               ↓
                                    [Consensus Builder Node]
                                               ↓
                                    [Naval Validator Node]
                                               ↓
                                    [JSON Exporter Node]
```

### 4.2 Custom Node Requirements

#### Consensus Builder Node
- **Inputs**: Multiple detection results
- **Processing**: Weighted voting, IoU matching
- **Output**: Unified component list with confidence

#### Naval Validator Node
- **Inputs**: Component detections
- **Processing**: Apply naval architecture rules
- **Output**: Validated and corrected components

#### Scale Calculator Node
- **Inputs**: Image, detected components
- **Processing**: Find scale references, calculate ratios
- **Output**: Pixels-per-meter conversion factor

## 5. Quality Assurance Strategy

### 5.1 Validation Metrics
- **Precision**: Correct classifications / Total classifications
- **Recall**: Detected components / Ground truth components
- **F1 Score**: Harmonic mean of precision and recall
- **IoU Score**: Intersection over Union for boundaries
- **Scale Error**: Measured dimensions vs expected dimensions

### 5.2 Testing Protocol
1. **Unit Testing**: Individual detection methods
2. **Integration Testing**: Consensus mechanism
3. **Regression Testing**: Historical blueprint dataset
4. **Performance Testing**: Throughput and latency
5. **Accuracy Testing**: Manual ground truth comparison

### 5.3 Continuous Improvement
- **Error Analysis**: Categorize and track failure modes
- **Model Tuning**: Adjust thresholds based on performance
- **Dataset Expansion**: Add problematic cases to training
- **Feedback Loop**: Incorporate 3D validation results

## 6. Hardware Optimization

### 6.1 GPU Allocation Strategy
```
RTX 5090 (32GB):
├── SAM Model (6GB)
├── YoloWorld Model (4GB)
├── Florence-2 Backup (4GB)
└── Image Batch Buffer (8GB)

RTX 3090 (24GB):
├── Anyline Processing (3GB)
├── OpenCV Operations (2GB)
├── Consensus Processing (3GB)
└── Validation Pipeline (4GB)
```

### 6.2 Parallel Processing
- **Batch Size**: 4-8 images simultaneously
- **Pipeline Stages**: Concurrent detection methods
- **Memory Management**: Dynamic allocation based on load
- **Cache Strategy**: Reuse model instances across batches

## 7. Risk Management

### 7.1 Technical Risks
| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Poor image quality | Medium | High | Quality gate preprocessing |
| Component occlusion | Low | Medium | Multi-view correlation |
| Unusual ship designs | Low | High | Fallback to manual review |
| Scale extraction failure | Medium | High | Multiple scale detection methods |
| Processing bottlenecks | Low | Medium | Distributed processing option |

### 7.2 Contingency Plans
1. **Detection Failure**: Fallback cascade through methods
2. **Performance Issues**: Reduce batch size, optimize models
3. **Accuracy Problems**: Human-in-the-loop validation
4. **System Failure**: Checkpoint and resume capability

## 8. Success Criteria

### 8.1 Minimum Viable Product (MVP)
- 85% component detection rate
- 90% classification accuracy
- <60 second processing time
- Handles 10 ship classes

### 8.2 Production Target
- 95% component detection rate
- 95% classification accuracy
- <30 second processing time
- Handles 100+ ship classes
- Batch processing capability
- API endpoint availability

### 8.3 Excellence Goals
- 98% accuracy across all metrics
- <10 second processing time
- Real-time preview capability
- Self-improving through feedback
- Industry standard solution

## 9. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
- Set up ComfyUI environment
- Install and configure RMBG, Anyline
- Create basic workflow template
- Test on 5 sample blueprints

### Phase 2: Detection Integration (Week 3-4)
- Integrate SAM for segmentation
- Add YoloWorld classification
- Implement consensus builder
- Achieve 85% accuracy target

### Phase 3: Optimization (Week 5-6)
- Add naval validation rules
- Implement scale extraction
- Optimize for dual GPUs
- Reach 90% accuracy target

### Phase 4: Production (Week 7-8)
- Build batch processing system
- Create monitoring dashboard
- Document API endpoints
- Deploy production pipeline

## 10. Dependencies & Prerequisites

### 10.1 Software Requirements
- ComfyUI (latest version)
- Python 3.10+
- CUDA 12.0+
- PyTorch 2.0+
- OpenCV 4.8+

### 10.2 Model Requirements
- SAM (segment-anything)
- YoloWorld-EfficientSAM
- Florence-2-large (optional)
- Anyline SDK

### 10.3 Data Requirements
- Training dataset: 100+ annotated blueprints
- Validation set: 20+ ground truth examples
- Test set: 10+ unseen ship classes

## Appendix A: Naval Component Taxonomy

### Primary Components
```
main_turret:
  - single_mount
  - twin_mount
  - triple_mount
  - quadruple_mount

superstructure:
  - bridge
  - conning_tower
  - fire_control
  - observation_deck

propulsion:
  - funnel
  - exhaust_stack
  - intake

secondary_armament:
  - casemate_gun
  - deck_mount
  - aa_battery
```

### Component Attributes
```
position: [forward, amidships, aft]
side: [port, starboard, centerline]
deck_level: [main, upper, superstructure]
count: integer
estimated_size: {small, medium, large}
```

## Appendix B: Coordinate Systems

### Image Coordinates
- Origin: Top-left corner
- X-axis: Horizontal (increases right)
- Y-axis: Vertical (increases down)
- Units: Pixels

### Naval Coordinates
- Origin: Ship centerline at bow
- X-axis: Beam (port negative, starboard positive)
- Y-axis: Length (bow to stern)
- Z-axis: Height (keel upward)
- Units: Meters

### Transformation Matrix
```
[Naval] = [Scale] × [Rotation] × [Translation] × [Image]
```

---
*Document Version: 1.0*
*Last Updated: 2025-01-06*
*Status: Active Development*
# ComfyUI Workflow Optimization Prompt for Naval Ship Blueprint Processing

## Context
I have an existing ComfyUI installation at `/home/coldaine/StableDiffusionWorkflow/ComfyUI-stable` that I'll symlink to this project directory. I need help reviewing my current setup and creating/optimizing workflow JSON files specifically for processing naval ship blueprints into 3D models.

### Key Advantages of This Approach
1. **Audit existing setup** - You can see what custom nodes and models I already have installed
2. **Direct workflow JSON editing** - Create and modify workflows programmatically
3. **Build on existing work** - No redundant installations, leverage what's already configured
4. **Programmatic workflow creation** - Generate JSON workflows that can be loaded directly in ComfyUI

### Understanding ComfyUI Workflow JSON Structure
ComfyUI workflows are JSON files that define a node graph for image processing. The structure includes:
- **Nodes**: Individual processing steps with unique IDs (e.g., LoadImage, RMBG, SAM)
- **Links**: Arrays defining connections between node outputs and inputs
- **Parameters**: Widget values and settings for each node
- **Groups**: Logical organization of related nodes
- **Version**: Workflow format version for compatibility

The agent can programmatically create and edit these JSON files to build custom workflows for naval blueprint processing.

## Setup Instructions for Agent

### 1. Create Symlink to Existing ComfyUI
```bash
# Create a symlink from my existing ComfyUI installation
ln -s /home/coldaine/StableDiffusionWorkflow/ComfyUI-stable /mnt/superstorage/_projects/BattleShipGraphicsProjects/ComfyUI
```

### 2. Review Current Installation
Please examine:
- `ComfyUI/custom_nodes/` - List all installed custom nodes
- `ComfyUI/models/` - Check available models (checkpoints, loras, etc.)
- `ComfyUI/workflows/` or any existing workflow JSON files
- Identify which Tier 1 nodes from our strategy are already installed:
  - ComfyUI-RMBG (with SAM/SAM2 support)
  - ComfyUI-Flowty-TripoSR
  - Anyline or equivalent line detection
  - Segment Anything nodes
  - YoloWorld-EfficientSAM

### 3. Workflow JSON Tasks

#### A. Create New Workflow JSON
Create a new workflow file `naval_blueprint_pipeline.json` that implements:

```
Input Blueprint → 
[RMBG] Background Removal →
[SAM/Anyline] Component Detection →
[Consensus] Multi-method validation →
[Depth] Marigold/DepthAnythingV2 →
[3D] TripoSR Generation →
[Export] Save outputs
```

#### B. Workflow JSON Structure
The workflow JSON should include:
- **Nodes**: Each processing step as a node with unique ID
- **Links**: Connections between nodes (inputs/outputs)
- **Parameters**: Configuration for each node
- **Groups**: Logical grouping of related nodes

Example structure:
```json
{
  "last_node_id": 50,
  "last_link_id": 100,
  "nodes": [
    {
      "id": 1,
      "type": "LoadImage",
      "pos": [100, 100],
      "size": [315, 314],
      "outputs": [{"name": "IMAGE", "type": "IMAGE"}],
      "properties": {},
      "widgets_values": ["blueprint.png"]
    },
    {
      "id": 2,
      "type": "ComfyUI-RMBG",
      "inputs": [{"name": "image", "type": "IMAGE", "link": 1}],
      "outputs": [{"name": "IMAGE", "type": "IMAGE"}]
    }
  ],
  "links": [[1, 1, 0, 2, 0, "IMAGE"]],
  "groups": [],
  "config": {},
  "version": 0.4
}
```

#### C. Optimize Existing Workflows
If I have existing workflows, please:
1. Review and identify inefficiencies
2. Add missing nodes for naval blueprint processing
3. Optimize node parameters for technical drawings
4. Add proper error handling and validation steps

### 4. Custom Node Installation (if needed)

For any missing Tier 1 nodes, provide installation commands:
```bash
cd ComfyUI/custom_nodes
git clone [repository_url]
pip install -r [node_name]/requirements.txt
```

### 5. Testing Workflow

Create a test script or instructions to:
1. Load the workflow JSON in ComfyUI
2. Process a sample blueprint image
3. Validate each stage output
4. Measure performance metrics (time, memory usage)
5. Save results for comparison

### 6. Batch Processing Setup

Create additional workflow variations:
- `naval_blueprint_batch.json` - For processing multiple images
- `naval_blueprint_fast.json` - Optimized for speed over quality
- `naval_blueprint_quality.json` - Maximum quality settings

### 7. Integration Points

Document how to:
- Export 3D models in Blender-compatible formats (OBJ, PLY)
- Save intermediate outputs for debugging
- Configure webhook/API endpoints for automation
- Set up command-line execution without GUI

## Expected Deliverables

1. **Workflow Analysis Report**
   - List of installed custom nodes
   - Missing components from Tier 1 strategy
   - Current workflow capabilities assessment

2. **Workflow JSON Files**
   - `naval_blueprint_pipeline.json` - Main processing workflow
   - Additional optimized variants
   - Documentation of each node's purpose and settings

3. **Installation Script** (if needed)
   - Commands to install missing nodes
   - Model download instructions
   - Configuration updates

4. **Testing Documentation**
   - How to run the workflow
   - Expected outputs at each stage
   - Performance benchmarks

## Technical Requirements

- **Input**: Naval ship blueprint images (JPG/PNG, typically 2000x3000px)
- **Output**: 
  - Segmented components (individual PNGs)
  - Depth maps (EXR format)
  - 3D models (OBJ/PLY format)
  - Processing metadata (JSON)

## Node-Specific Configuration

### RMBG Node
- Model: RMBG-2.0 or SAM2
- Remove background completely
- Preserve line quality

### SAM/Detection Nodes
- Prompts: "turret", "superstructure", "gun mount", "hull"
- Confidence threshold: 0.7
- Multi-class output

### TripoSR Node
- Resolution: High
- Mesh quality: Maximum
- Export format: OBJ with textures

## Questions to Address

1. What custom nodes are already installed?
2. Which models are available in the models folder?
3. Are there existing workflows we can build upon?
4. What's the current GPU memory usage pattern?
5. How can we optimize for the RTX 3090 24GB VRAM?

---

Please provide:
- Complete workflow JSON files ready to use
- Step-by-step testing instructions
- Performance optimization recommendations
- Documentation of any issues or limitations found
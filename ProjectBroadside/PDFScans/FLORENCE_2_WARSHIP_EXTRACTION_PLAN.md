# Florence-2 Warship Extraction System - Architectural Plan

## 1. Objective

Create a robust, production-ready system for extracting warship illustrations from historical PDF documents using the Florence-2 vision-language model, with enhanced accuracy, memory efficiency, and comprehensive error handling capabilities.

## 2. Clarifying Questions & Assumptions

### Assumptions Made:
- The project will process Jane's Fighting Ships historical documents (1900, 1919 editions present)
- Output should be high-resolution PNG images suitable for archival purposes
- Processing will be done locally with CUDA support preferred but CPU fallback required
- Multiple detection strategies needed due to varied illustration styles in historical documents
- Batch processing capability required for handling multiple large PDF files
- User requires both extracted images and metadata for cataloging purposes

### Questions for Confirmation:
1. Should the system support real-time processing or is batch processing sufficient?
2. What is the preferred output resolution (300 DPI suggested, but confirm)?
3. Do you need a web interface or CLI-only implementation?
4. Should extracted metadata include OCR text from surrounding contexts?

## 3. High-Level Approach

**Core Strategy:** Implement a multi-stage pipeline using Florence-2-Large model with comprehensive prompt engineering, high-resolution PDF processing, and intelligent post-processing. The system will use a **Factory Pattern** for different detection strategies, **Pipeline Pattern** for processing stages, and **Strategy Pattern** for handling different PDF formats and illustration types.

**Key Architectural Decisions:**
- Use Florence-2-Large over base model for superior accuracy on historical illustrations
- Implement multiple detection passes with specialized prompts for different warship illustration types
- Apply Non-Maximum Suppression (NMS) to eliminate duplicate detections
- Create modular components for easy extension and maintenance
- Include comprehensive error handling and memory optimization
- Generate rich metadata for cataloging and verification

## 4. Component & File Breakdown

| Action | File Path | Description of Changes |
| :--- | :--- | :--- |
| CREATE | [`pyproject.toml`](pyproject.toml) | Poetry configuration with comprehensive dependencies including Florence-2 requirements |
| CREATE | [`src/warship_extractor/__init__.py`](src/warship_extractor/__init__.py) | Package initialization with version and main exports |
| CREATE | [`src/warship_extractor/core/model_manager.py`](src/warship_extractor/core/model_manager.py) | Florence-2 model loading, caching, and device management |
| CREATE | [`src/warship_extractor/core/pdf_processor.py`](src/warship_extractor/core/pdf_processor.py) | High-resolution PDF to image conversion with PyMuPDF optimization |
| CREATE | [`src/warship_extractor/detection/prompt_strategies.py`](src/warship_extractor/detection/prompt_strategies.py) | Comprehensive prompt engineering for different warship illustration types |
| CREATE | [`src/warship_extractor/detection/detector.py`](src/warship_extractor/detection/detector.py) | Main detection engine with multiple strategy support |
| CREATE | [`src/warship_extractor/processing/nms_filter.py`](src/warship_extractor/processing/nms_filter.py) | Non-Maximum Suppression and duplicate removal logic |
| CREATE | [`src/warship_extractor/processing/image_processor.py`](src/warship_extractor/processing/image_processor.py) | Image cropping, enhancement, and format conversion |
| CREATE | [`src/warship_extractor/io/file_manager.py`](src/warship_extractor/io/file_manager.py) | File I/O operations, directory management, and metadata handling |
| CREATE | [`src/warship_extractor/pipeline/extraction_pipeline.py`](src/warship_extractor/pipeline/extraction_pipeline.py) | Main orchestration pipeline combining all components |
| CREATE | [`src/warship_extractor/utils/visualization.py`](src/warship_extractor/utils/visualization.py) | Detection visualization and verification tools |
| CREATE | [`src/warship_extractor/utils/logger.py`](src/warship_extractor/utils/logger.py) | Structured logging with progress tracking |
| CREATE | [`src/warship_extractor/config/settings.py`](src/warship_extractor/config/settings.py) | Configuration management with environment variable support |
| CREATE | [`main.py`](main.py) | CLI entry point with argument parsing and batch processing |
| CREATE | [`tests/test_model_manager.py`](tests/test_model_manager.py) | Unit tests for model loading and device management |
| CREATE | [`tests/test_pdf_processor.py`](tests/test_pdf_processor.py) | Tests for PDF processing with different resolution settings |
| CREATE | [`tests/test_detection.py`](tests/test_detection.py) | Integration tests for detection pipeline |
| CREATE | [`tests/fixtures/sample_data/`](tests/fixtures/sample_data/) | Test PDFs and expected outputs for validation |
| CREATE | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Detailed architecture documentation |
| CREATE | [`docs/API.md`](docs/API.md) | API documentation and usage examples |
| CREATE | [`DEPENDENCIES.md`](DEPENDENCIES.md) | Dependency evaluation and justification document |
| CREATE | [`output/`](output/) | Directory structure for extracted images and metadata |
| CREATE | [`logs/`](logs/) | Directory for processing logs and error reports |

## 5. Step-by-Step Implementation Plan

1. **[Setup]** Initialize Poetry project and install Florence-2 dependencies including torch, transformers, PyMuPDF, and specialized computer vision libraries.

2. **[Core Foundation]** Create the [`model_manager.py`](src/warship_extractor/core/model_manager.py) with device detection, model caching, and memory optimization for Florence-2-Large.

3. **[PDF Processing]** Implement [`pdf_processor.py`](src/warship_extractor/core/pdf_processor.py) with high-resolution conversion (300+ DPI), proper memory management for large documents.

4. **[Prompt Engineering]** Design [`prompt_strategies.py`](src/warship_extractor/detection/prompt_strategies.py) with comprehensive warship detection prompts including technical drawings, schematics, and historical illustration variants.

5. **[Detection Engine]** Build [`detector.py`](src/warship_extractor/detection/detector.py) implementing multiple detection passes, confidence scoring, and batch processing capabilities.

6. **[Post-Processing]** Create [`nms_filter.py`](src/warship_extractor/processing/nms_filter.py) for intelligent duplicate removal and [`image_processor.py`](src/warship_extractor/processing/image_processor.py) for cropping and enhancement.

7. **[I/O Management]** Implement [`file_manager.py`](src/warship_extractor/io/file_manager.py) with structured output organization, metadata generation, and progress tracking.

8. **[Pipeline Integration]** Develop [`extraction_pipeline.py`](src/warship_extractor/pipeline/extraction_pipeline.py) orchestrating all components with error handling and recovery mechanisms.

9. **[Utilities]** Add [`visualization.py`](src/warship_extractor/utils/visualization.py) for detection verification and [`logger.py`](src/warship_extractor/utils/logger.py) for comprehensive logging.

10. **[Configuration]** Create [`settings.py`](src/warship_extractor/config/settings.py) with environment-based configuration management.

11. **[CLI Interface]** Build [`main.py`](main.py) with argparse support for batch processing, resolution settings, and output customization.

12. **[Testing Suite]** Implement comprehensive tests covering model loading, PDF processing, detection accuracy, and integration scenarios.

13. **[Documentation]** Create detailed documentation including architecture overview, API reference, and usage examples.

14. **[Validation]** Process the existing Jane's 1900 and 1919 PDFs to validate the complete pipeline and fine-tune detection parameters.

## 6. Potential Risks & Mitigation

**Risk 1:** Florence-2 model memory requirements may overwhelm available GPU memory  
**Mitigation:** Implement dynamic batch sizing, CPU fallback, and memory monitoring with automatic scaling

**Risk 2:** Historical PDF quality variations may cause inconsistent results  
**Mitigation:** Multi-resolution processing pipeline with adaptive enhancement and multiple detection strategies

**Risk 3:** False positives from decorative elements or non-warship naval content  
**Mitigation:** Implement size filtering, context analysis, and confidence-based rejection with manual review interface

**Risk 4:** Large batch processing may cause system instability or crashes  
**Mitigation:** Implement checkpoint/resume functionality, progress persistence, and graceful error recovery with detailed logging

**Risk 5:** Dependency conflicts between Florence-2 requirements and other packages  
**Mitigation:** Use Poetry for precise dependency management, create isolated virtual environment, and document version compatibility matrix

**Risk 6:** Processing time scalability issues with large document collections  
**Mitigation:** Implement parallel processing where possible, progress monitoring, and estimation algorithms for time-to-completion

## 7. Technical Implementation Details

### Required Dependencies

```toml
[tool.poetry.dependencies]
python = "^3.9"
torch = "^2.0.0"
torchvision = "^0.15.0"
transformers = "^4.35.0"
PyMuPDF = "^1.23.0"
Pillow = "^10.0.0"
einops = "^0.7.0"
timm = "^0.9.2"
accelerate = "^0.24.0"
sentencepiece = "^0.1.99"
protobuf = "^4.24.0"
supervision = "^0.16.0"
numpy = "^1.24.0"
opencv-python = "^4.8.0"
```

### Core Detection Prompts

```python
WARSHIP_PROMPTS = [
    # Primary prompts
    "<OD>warship illustration",
    "<OD>naval vessel diagram", 
    "<OD>battleship drawing",
    "<OD>warship schematic",
    
    # Technical drawings
    "<OD>ship blueprint",
    "<OD>naval architecture drawing",
    "<OD>vessel cross-section",
    "<OD>ship profile view",
    
    # Historical and specific types
    "<OD>military ship",
    "<OD>destroyer illustration",
    "<OD>frigate diagram",
    "<OD>submarine drawing",
    
    # Alternative detection with caption
    "<DENSE_REGION_CAPTION>",
]
```

### Processing Pipeline Architecture

```
PDF Input → High-Res Conversion → Multi-Prompt Detection → NMS Filtering → 
Image Extraction → Metadata Generation → Output Organization
```

### Memory Optimization Strategy

- Dynamic batch sizing based on available GPU memory
- Page-by-page processing to avoid loading entire PDFs
- Automatic model offloading between processing sessions
- Garbage collection and CUDA cache clearing between batches

### Output Structure

```
output/
├── extracted_images/
│   ├── [pdf_name]_page[N]_ship[M]_[label].png
│   └── ...
├── metadata/
│   ├── [pdf_name]_detections.json
│   └── processing_summary.json
├── visualizations/
│   ├── [pdf_name]_page[N]_annotated.png
│   └── ...
└── logs/
    ├── processing.log
    └── errors.log
```

## 8. Success Metrics

- **Accuracy:** >90% detection rate for warship illustrations in test documents
- **Precision:** <10% false positive rate through effective filtering
- **Performance:** Process typical Jane's Fighting Ships page (<5MB) in <30 seconds
- **Memory Efficiency:** Handle documents up to 500MB without system instability
- **Reliability:** 99% successful completion rate for batch processing operations

## 9. Future Enhancements

- **OCR Integration:** Extract surrounding text for enhanced metadata
- **Classification:** Categorize warship types (battleship, destroyer, etc.)
- **Web Interface:** Browser-based upload and processing interface  
- **API Endpoints:** REST API for integration with other systems
- **Database Storage:** PostgreSQL backend for searchable metadata
- **Advanced Filtering:** Machine learning-based false positive reduction

---

*This architectural plan provides a comprehensive foundation for implementing a production-ready warship illustration extraction system using state-of-the-art computer vision technology.*
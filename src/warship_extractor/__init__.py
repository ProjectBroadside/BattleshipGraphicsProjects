"""
Warship Extractor - Florence-2 based warship illustration extraction system.

This package provides comprehensive tools for extracting warship illustrations
from historical PDF documents using the Florence-2 vision-language model.

Key Components:
- ExtractionPipeline: Main orchestration pipeline
- ModelManager: Florence-2 model management
- PDFProcessor: PDF to image conversion
- WarshipDetector: Detection engine with prompt strategies
- ImageProcessor: Image enhancement and cropping
- NMSFilter: Duplicate detection removal
- CLI: Command-line interface
- Utilities: Logging, visualization, and reporting tools
"""

from .config.settings import settings
from .core.model_manager import ModelManager
from .core.pdf_processor import PDFProcessor
from .detection.detector import WarshipDetector
from .detection.prompt_strategies import PromptStrategy
from .pipeline.extraction_pipeline import ExtractionPipeline
from .processing.image_processor import ImageProcessor
from .processing.nms_filter import NMSFilter
from .utils.logger import setup_logging, get_logger, main_logger
from .utils.visualization import draw_bounding_boxes, create_summary_report

__version__ = "1.0.0"
__author__ = "Warship Extractor Team"
__description__ = "Extract warship illustrations from historical PDFs using Florence-2"

__all__ = [
    # Core configuration
    "settings",
    
    # Main pipeline
    "ExtractionPipeline",
    
    # Core components
    "ModelManager", 
    "PDFProcessor",
    "WarshipDetector",
    "PromptStrategy",
    "ImageProcessor",
    "NMSFilter",
    
    # Utilities
    "setup_logging",
    "get_logger", 
    "main_logger",
    "draw_bounding_boxes",
    "create_summary_report",
    
    # Metadata
    "__version__",
    "__author__",
    "__description__"
]


def get_version() -> str:
    """Get the package version."""
    return __version__


def get_system_info() -> dict:
    """Get system information for debugging."""
    import platform
    import sys
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_device = torch.cuda.get_device_name(0) if cuda_available else None
    except ImportError:
        cuda_available = False
        cuda_device = None
    
    return {
        "version": __version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cuda_available": cuda_available,
        "cuda_device": cuda_device
    }
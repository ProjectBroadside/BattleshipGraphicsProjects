"""
Configuration management for the Warship Extractor system.

Handles environment variables, default settings, and configuration validation.
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    """
    
    # Model Configuration
    florence_model_name: str = Field(
        default="microsoft/Florence-2-large",
        description="Florence-2 model name to use"
    )
    device: str = Field(
        default="auto",
        description="Device to use: 'cuda', 'cpu', or 'auto'"
    )
    torch_dtype: str = Field(
        default="auto",
        description="Torch dtype: 'float16', 'float32', or 'auto'"
    )
    
    # Processing Configuration
    default_dpi: int = Field(
        default=300,
        description="Default DPI for PDF to image conversion"
    )
    confidence_threshold: float = Field(
        default=0.5,
        description="Confidence threshold for detections"
    )
    iou_threshold: float = Field(
        default=0.5,
        description="IoU threshold for Non-Maximum Suppression"
    )
    min_detection_area: int = Field(
        default=1000,
        description="Minimum area in pixels for valid detections"
    )
    max_detection_area: Optional[int] = Field(
        default=None,
        description="Maximum area in pixels for valid detections"
    )
    
    # File Management
    output_directory: str = Field(
        default="output",
        description="Default output directory for extracted images"
    )
    log_directory: str = Field(
        default="logs",
        description="Directory for log files"
    )
    temp_directory: str = Field(
        default="temp",
        description="Temporary directory for processing"
    )
    
    # Processing Options
    batch_size: int = Field(
        default=1,
        description="Batch size for processing multiple pages"
    )
    max_workers: int = Field(
        default=1,
        description="Maximum number of worker processes"
    )
    save_visualizations: bool = Field(
        default=True,
        description="Whether to save annotated visualization images"
    )
    save_metadata: bool = Field(
        default=True,
        description="Whether to save detection metadata"
    )
    
    # Image Enhancement
    enhance_images: bool = Field(
        default=False,
        description="Apply image enhancement preprocessing"
    )
    padding_pixels: int = Field(
        default=10,
        description="Padding to add around detected regions"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR"
    )
    enable_file_logging: bool = Field(
        default=True,
        description="Enable logging to file"
    )
    
    class Config:
        env_prefix = "WARSHIP_"
        case_sensitive = False
        
    def get_output_path(self, filename: str = "") -> Path:
        """Get full output path for a given filename."""
        output_path = Path(self.output_directory)
        output_path.mkdir(exist_ok=True)
        return output_path / filename if filename else output_path
    
    def get_log_path(self, filename: str = "") -> Path:
        """Get full log path for a given filename."""
        log_path = Path(self.log_directory)
        log_path.mkdir(exist_ok=True)
        return log_path / filename if filename else log_path
    
    def get_temp_path(self, filename: str = "") -> Path:
        """Get full temp path for a given filename."""
        temp_path = Path(self.temp_directory)
        temp_path.mkdir(exist_ok=True)
        return temp_path / filename if filename else temp_path


# Global settings instance
settings = Settings()
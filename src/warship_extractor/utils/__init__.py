"""
Utility modules for the warship extractor system.

This package provides common utilities including:
- Structured logging with progress tracking
- Visualization tools for detections and results
- Helper functions for common operations
"""

from .logger import (
    setup_logging,
    get_logger,
    log_system_info,
    ProgressLogger,
    PerformanceLogger,
    LoggerMixin,
    main_logger
)

__all__ = [
    'setup_logging',
    'get_logger', 
    'log_system_info',
    'ProgressLogger',
    'PerformanceLogger',
    'LoggerMixin',
    'main_logger'
]
"""
Structured logging utilities for the warship extractor system.

This module provides comprehensive logging functionality with progress tracking,
performance monitoring, and structured output for debugging and analysis.
"""

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..config.settings import settings


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green  
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        """Format log record with colors."""
        if hasattr(record, 'levelname') and record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


class ProgressLogger:
    """Logger for tracking progress through multi-step operations."""
    
    def __init__(self, logger: logging.Logger, total_steps: int, description: str = "Processing"):
        """
        Initialize progress logger.
        
        Args:
            logger: Base logger instance
            total_steps: Total number of steps
            description: Description of the operation
        """
        self.logger = logger
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = datetime.now()
        
    def step(self, message: str = "") -> None:
        """Advance progress by one step."""
        self.current_step += 1
        progress = (self.current_step / self.total_steps) * 100
        
        elapsed = datetime.now() - self.start_time
        if self.current_step > 0:
            estimated_total = elapsed * (self.total_steps / self.current_step)
            remaining = estimated_total - elapsed
            eta_str = f", ETA: {remaining}"
        else:
            eta_str = ""
        
        progress_msg = f"{self.description}: {self.current_step}/{self.total_steps} ({progress:.1f}%){eta_str}"
        if message:
            progress_msg += f" - {message}"
        
        self.logger.info(progress_msg)
    
    def complete(self, message: str = "Completed") -> None:
        """Mark progress as complete."""
        elapsed = datetime.now() - self.start_time
        self.logger.info(f"{self.description} {message} in {elapsed}")


class PerformanceLogger:
    """Logger for performance monitoring and profiling."""
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize performance logger.
        
        Args:
            logger: Base logger instance
        """
        self.logger = logger
        self.timers: Dict[str, datetime] = {}
        self.counters: Dict[str, int] = {}
        self.metrics: Dict[str, float] = {}
    
    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        self.timers[name] = datetime.now()
        self.logger.debug(f"Timer started: {name}")
    
    def end_timer(self, name: str, log_result: bool = True) -> float:
        """
        End a named timer and return elapsed time.
        
        Args:
            name: Timer name
            log_result: Whether to log the result
            
        Returns:
            Elapsed time in seconds
        """
        if name not in self.timers:
            self.logger.warning(f"Timer '{name}' was not started")
            return 0.0
        
        elapsed = (datetime.now() - self.timers[name]).total_seconds()
        del self.timers[name]
        
        if log_result:
            self.logger.info(f"Timer '{name}': {elapsed:.3f}s")
        
        return elapsed
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a named counter."""
        self.counters[name] = self.counters.get(name, 0) + value
        self.logger.debug(f"Counter '{name}': {self.counters[name]}")
    
    def set_metric(self, name: str, value: float) -> None:
        """Set a named metric value."""
        self.metrics[name] = value
        self.logger.debug(f"Metric '{name}': {value}")
    
    def log_summary(self) -> None:
        """Log summary of all performance data."""
        self.logger.info("=== Performance Summary ===")
        
        if self.counters:
            self.logger.info("Counters:")
            for name, value in self.counters.items():
                self.logger.info(f"  {name}: {value}")
        
        if self.metrics:
            self.logger.info("Metrics:")
            for name, value in self.metrics.items():
                self.logger.info(f"  {name}: {value:.3f}")
        
        if self.timers:
            self.logger.info("Active timers:")
            for name in self.timers.keys():
                self.logger.info(f"  {name}: still running")


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None,
    console_output: bool = True,
    enable_colors: bool = True
) -> logging.Logger:
    """
    Set up comprehensive logging for the warship extractor.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (uses settings if None)
        console_output: Whether to output to console
        enable_colors: Whether to use colored console output
        
    Returns:
        Configured logger instance
    """
    log_level = log_level or settings.log_level
    
    # Create main logger
    logger = logging.getLogger('warship_extractor')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        if enable_colors and sys.stdout.isatty():
            console_format = ColoredFormatter(
                '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s',
                datefmt='%H:%M:%S'
            )
        else:
            console_format = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s',
                datefmt='%H:%M:%S'
            )
        
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
    
    # File handler
    if settings.enable_file_logging:
        if log_file is None:
            log_file = settings.get_log_path("warship_extractor.log")
        else:
            log_file = Path(log_file)
        
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Use rotating file handler to prevent huge log files
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)  # Always debug level for files
        
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    # Prevent duplicate logs from propagating to root logger
    logger.propagate = False
    
    logger.info(f"Logging initialized - Level: {log_level}, File: {log_file if settings.enable_file_logging else 'disabled'}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f'warship_extractor.{name}')


def log_system_info(logger: logging.Logger) -> None:
    """Log system information for debugging."""
    import platform
    import psutil
    import torch
    
    logger.info("=== System Information ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"CPU Cores: {psutil.cpu_count()}")
    logger.info(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA Available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    else:
        logger.info("CUDA: Not available")


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if not hasattr(self, '_logger'):
            class_name = self.__class__.__module__ + '.' + self.__class__.__name__
            self._logger = logging.getLogger(class_name)
        return self._logger
    
    def log_method_call(self, method_name: str, **kwargs) -> None:
        """Log a method call with parameters."""
        params = ', '.join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.debug(f"{method_name}({params})")
    
    def log_performance(self, operation: str, duration: float, **metrics) -> None:
        """Log performance metrics for an operation."""
        metric_str = ', '.join(f"{k}={v}" for k, v in metrics.items())
        self.logger.info(f"Performance - {operation}: {duration:.3f}s ({metric_str})")


# Global logger instance
main_logger = setup_logging()
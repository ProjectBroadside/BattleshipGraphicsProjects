"""
Model management for Florence-2 with device optimization and memory management.

Handles model loading, caching, device selection, and memory optimization
for the Florence-2 vision-language model.
"""

import gc
import logging
import warnings
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from ..config.settings import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages Florence-2 model loading, device management, and memory optimization.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None
    ):
        """
        Initialize the model manager.
        
        Args:
            model_name: Florence-2 model name (defaults to settings)
            device: Device to use ('cuda', 'cpu', 'auto')
            torch_dtype: Torch dtype ('float16', 'float32', 'auto')
        """
        self.model_name = model_name or settings.florence_model_name
        self.device = self._determine_device(device or settings.device)
        self.torch_dtype = self._determine_dtype(torch_dtype or settings.torch_dtype)
        
        self.model: Optional[AutoModelForCausalLM] = None
        self.processor: Optional[AutoProcessor] = None
        self._model_loaded = False
        
        logger.info(f"ModelManager initialized with device: {self.device}, dtype: {self.torch_dtype}")
    
    def _determine_device(self, device_setting: str) -> str:
        """Determine the best device to use."""
        if device_setting == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                device = "cpu"
                logger.info("CUDA not available, using CPU")
        else:
            device = device_setting
            
        return device
    
    def _determine_dtype(self, dtype_setting: str) -> torch.dtype:
        """Determine the best dtype to use."""
        if dtype_setting == "auto":
            if self.device == "cuda":
                return torch.float16
            else:
                return torch.float32
        elif dtype_setting == "float16":
            return torch.float16
        else:
            return torch.float32
    
    def load_model(self, force_reload: bool = False) -> Tuple[AutoModelForCausalLM, AutoProcessor]:
        """
        Load the Florence-2 model and processor.
        
        Args:
            force_reload: Force reload even if already loaded
            
        Returns:
            Tuple of (model, processor)
        """
        if self._model_loaded and not force_reload:
            return self.model, self.processor
        
        logger.info(f"Loading Florence-2 model: {self.model_name}")
        
        try:
            # Suppress some warnings during model loading
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                
                # Load processor first
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                
                # Load model with appropriate settings
                model_kwargs = {
                    "torch_dtype": self.torch_dtype,
                    "trust_remote_code": True,
                }
                
                if self.device == "cuda":
                    model_kwargs["device_map"] = "auto"
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
                
                # Move to device if not using device_map
                if self.device != "cuda" or "device_map" not in model_kwargs:
                    if self.model is not None:
                        self.model = self.model.to(self.device)
                
                self._model_loaded = True
                
                logger.info("Model loaded successfully")
                self._log_memory_usage()
                
                return self.model, self.processor
                
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}") from e
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        self._model_loaded = False
        
        # Force garbage collection
        gc.collect()
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        logger.info("Model unloaded and memory cleared")
    
    def _log_memory_usage(self) -> None:
        """Log current memory usage."""
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            logger.info(f"GPU memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
    
    def get_memory_usage(self) -> dict:
        """Get current memory usage statistics."""
        usage = {}
        
        if self.device == "cuda":
            usage["gpu_allocated"] = torch.cuda.memory_allocated() / 1e9
            usage["gpu_cached"] = torch.cuda.memory_reserved() / 1e9
            usage["gpu_total"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        return usage
    
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._model_loaded and self.model is not None
    
    def optimize_memory(self) -> None:
        """Optimize memory usage by clearing caches."""
        gc.collect()
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        logger.debug("Memory optimization completed")
    
    def __enter__(self):
        """Context manager entry."""
        self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload_model()
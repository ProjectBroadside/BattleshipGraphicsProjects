"""
Unit tests for the ModelManager class.

Tests model loading, caching, device management, and memory optimization
without requiring actual model downloads or GPU hardware.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
from pathlib import Path
import tempfile
import shutil

from src.warship_extractor.core.model_manager import ModelManager
from src.warship_extractor.config.settings import Settings


class TestModelManager:
    """Test cases for ModelManager functionality."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = Mock()
        settings.model_name = "microsoft/Florence-2-large"
        settings.model_cache_dir = Path("/tmp/test_cache")
        settings.device = "cpu"
        settings.max_memory_gb = 8.0
        settings.enable_model_caching = True
        settings.model_precision = "float32"
        return settings
    
    @pytest.fixture
    def model_manager(self, mock_settings):
        """Create ModelManager instance for testing."""
        with patch('src.warship_extractor.core.model_manager.AutoProcessor'), \
             patch('src.warship_extractor.core.model_manager.AutoModelForCausalLM'):
            return ModelManager(mock_settings)
    
    def test_initialization(self, mock_settings):
        """Test ModelManager initialization."""
        with patch('src.warship_extractor.core.model_manager.AutoProcessor'), \
             patch('src.warship_extractor.core.model_manager.AutoModelForCausalLM'):
            manager = ModelManager(mock_settings)
            
            assert manager.settings == mock_settings
            assert manager.model is None
            assert manager.processor is None
            assert manager.device == torch.device("cpu")
    
    @patch('torch.cuda.is_available')
    def test_determine_device_gpu_available(self, mock_cuda_available, mock_settings):
        """Test device determination when GPU is available."""
        mock_cuda_available.return_value = True
        mock_settings.device = "auto"
        
        with patch('src.warship_extractor.core.model_manager.AutoProcessor'), \
             patch('src.warship_extractor.core.model_manager.AutoModelForCausalLM'):
            manager = ModelManager(mock_settings)
            
            assert manager.device == torch.device("cuda")
    
    @patch('torch.cuda.is_available')
    def test_determine_device_gpu_not_available(self, mock_cuda_available, mock_settings):
        """Test device determination when GPU is not available."""
        mock_cuda_available.return_value = False
        mock_settings.device = "auto"
        
        with patch('src.warship_extractor.core.model_manager.AutoProcessor'), \
             patch('src.warship_extractor.core.model_manager.AutoModelForCausalLM'):
            manager = ModelManager(mock_settings)
            
            assert manager.device == torch.device("cpu")
    
    def test_determine_device_explicit(self, mock_settings):
        """Test explicit device setting."""
        mock_settings.device = "cuda:1"
        
        with patch('src.warship_extractor.core.model_manager.AutoProcessor'), \
             patch('src.warship_extractor.core.model_manager.AutoModelForCausalLM'):
            manager = ModelManager(mock_settings)
            
            assert manager.device == torch.device("cuda:1")
    
    @patch('src.warship_extractor.core.model_manager.AutoProcessor.from_pretrained')
    @patch('src.warship_extractor.core.model_manager.AutoModelForCausalLM.from_pretrained')
    def test_load_model_success(self, mock_model_load, mock_processor_load, model_manager):
        """Test successful model loading."""
        # Setup mocks
        mock_processor = Mock()
        mock_model = Mock()
        mock_processor_load.return_value = mock_processor
        mock_model_load.return_value = mock_model
        
        # Load model
        model_manager.load_model()
        
        # Verify calls
        mock_processor_load.assert_called_once_with(
            model_manager.settings.model_name,
            cache_dir=str(model_manager.settings.model_cache_dir),
            trust_remote_code=True
        )
        mock_model_load.assert_called_once_with(
            model_manager.settings.model_name,
            cache_dir=str(model_manager.settings.model_cache_dir),
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map=model_manager.device
        )
        
        # Verify model is loaded
        assert model_manager.processor == mock_processor
        assert model_manager.model == mock_model
    
    @patch('src.warship_extractor.core.model_manager.AutoProcessor.from_pretrained')
    @patch('src.warship_extractor.core.model_manager.AutoModelForCausalLM.from_pretrained')
    def test_load_model_with_half_precision(self, mock_model_load, mock_processor_load, mock_settings):
        """Test model loading with half precision."""
        mock_settings.model_precision = "float16"
        
        with patch('src.warship_extractor.core.model_manager.AutoProcessor'), \
             patch('src.warship_extractor.core.model_manager.AutoModelForCausalLM'):
            manager = ModelManager(mock_settings)
        
        # Setup mocks
        mock_processor = Mock()
        mock_model = Mock()
        mock_processor_load.return_value = mock_processor
        mock_model_load.return_value = mock_model
        
        # Load model
        manager.load_model()
        
        # Verify half precision is used
        call_args = mock_model_load.call_args
        assert call_args[1]['torch_dtype'] == torch.float16
    
    @patch('src.warship_extractor.core.model_manager.AutoProcessor.from_pretrained')
    def test_load_model_processor_error(self, mock_processor_load, model_manager):
        """Test handling of processor loading error."""
        mock_processor_load.side_effect = Exception("Processor load failed")
        
        with pytest.raises(RuntimeError, match="Failed to load Florence-2 processor"):
            model_manager.load_model()
    
    @patch('src.warship_extractor.core.model_manager.AutoProcessor.from_pretrained')
    @patch('src.warship_extractor.core.model_manager.AutoModelForCausalLM.from_pretrained')
    def test_load_model_model_error(self, mock_model_load, mock_processor_load, model_manager):
        """Test handling of model loading error."""
        mock_processor_load.return_value = Mock()
        mock_model_load.side_effect = Exception("Model load failed")
        
        with pytest.raises(RuntimeError, match="Failed to load Florence-2 model"):
            model_manager.load_model()
    
    def test_is_loaded_false(self, model_manager):
        """Test is_loaded when model is not loaded."""
        assert not model_manager.is_loaded()
    
    def test_is_loaded_true(self, model_manager):
        """Test is_loaded when model is loaded."""
        model_manager.model = Mock()
        model_manager.processor = Mock()
        assert model_manager.is_loaded()
    
    def test_is_loaded_partial(self, model_manager):
        """Test is_loaded when only one component is loaded."""
        model_manager.model = Mock()
        model_manager.processor = None
        assert not model_manager.is_loaded()
    
    @patch('torch.cuda.empty_cache')
    @patch('gc.collect')
    def test_clear_cache(self, mock_gc, mock_cuda_cache, model_manager):
        """Test cache clearing functionality."""
        model_manager.clear_cache()
        
        mock_gc.assert_called_once()
        mock_cuda_cache.assert_called_once()
    
    def test_unload_model(self, model_manager):
        """Test model unloading."""
        # Setup loaded model
        model_manager.model = Mock()
        model_manager.processor = Mock()
        
        with patch.object(model_manager, 'clear_cache') as mock_clear:
            model_manager.unload_model()
            
            assert model_manager.model is None
            assert model_manager.processor is None
            mock_clear.assert_called_once()
    
    @patch('torch.cuda.get_device_properties')
    def test_get_memory_info_cuda(self, mock_get_props, model_manager):
        """Test memory info retrieval for CUDA device."""
        model_manager.device = torch.device("cuda:0")
        
        mock_props = Mock()
        mock_props.total_memory = 8 * 1024**3  # 8GB
        mock_get_props.return_value = mock_props
        
        with patch('torch.cuda.memory_allocated', return_value=2 * 1024**3), \
             patch('torch.cuda.memory_reserved', return_value=3 * 1024**3):
            
            info = model_manager.get_memory_info()
            
            assert info['device'] == 'cuda:0'
            assert info['total_memory_gb'] == 8.0
            assert info['allocated_memory_gb'] == 2.0
            assert info['reserved_memory_gb'] == 3.0
            assert info['free_memory_gb'] == 5.0
    
    @patch('psutil.virtual_memory')
    def test_get_memory_info_cpu(self, mock_virtual_mem, model_manager):
        """Test memory info retrieval for CPU device."""
        model_manager.device = torch.device("cpu")
        
        mock_mem = Mock()
        mock_mem.total = 16 * 1024**3  # 16GB
        mock_mem.available = 12 * 1024**3  # 12GB available
        mock_virtual_mem.return_value = mock_mem
        
        info = model_manager.get_memory_info()
        
        assert info['device'] == 'cpu'
        assert info['total_memory_gb'] == 16.0
        assert info['available_memory_gb'] == 12.0
        assert info['used_memory_gb'] == 4.0
    
    def test_get_memory_info_error_handling(self, model_manager):
        """Test memory info error handling."""
        with patch('torch.cuda.get_device_properties', side_effect=Exception("Error")):
            info = model_manager.get_memory_info()
            
            assert info['device'] == 'cpu'
            assert 'error' in info
    
    def test_context_manager_success(self, mock_settings):
        """Test context manager successful usage."""
        with patch('src.warship_extractor.core.model_manager.AutoProcessor'), \
             patch('src.warship_extractor.core.model_manager.AutoModelForCausalLM'):
            
            manager = ModelManager(mock_settings)
            
            with patch.object(manager, 'load_model') as mock_load, \
                 patch.object(manager, 'unload_model') as mock_unload:
                
                with manager:
                    mock_load.assert_called_once()
                    pass  # Do something with the manager
                
                mock_unload.assert_called_once()
    
    def test_context_manager_exception(self, mock_settings):
        """Test context manager with exception."""
        with patch('src.warship_extractor.core.model_manager.AutoProcessor'), \
             patch('src.warship_extractor.core.model_manager.AutoModelForCausalLM'):
            
            manager = ModelManager(mock_settings)
            
            with patch.object(manager, 'load_model') as mock_load, \
                 patch.object(manager, 'unload_model') as mock_unload:
                
                with pytest.raises(ValueError):
                    with manager:
                        mock_load.assert_called_once()
                        raise ValueError("Test exception")
                
                mock_unload.assert_called_once()
    
    def test_get_model_info(self, model_manager):
        """Test model information retrieval."""
        info = model_manager.get_model_info()
        
        expected_keys = [
            'model_name', 'device', 'precision', 'cache_dir',
            'is_loaded', 'memory_info'
        ]
        
        for key in expected_keys:
            assert key in info
        
        assert info['model_name'] == model_manager.settings.model_name
        assert info['device'] == str(model_manager.device)
        assert info['is_loaded'] == model_manager.is_loaded()
    
    @patch('src.warship_extractor.core.model_manager.AutoProcessor.from_pretrained')
    @patch('src.warship_extractor.core.model_manager.AutoModelForCausalLM.from_pretrained')
    def test_caching_disabled(self, mock_model_load, mock_processor_load, mock_settings):
        """Test behavior when caching is disabled."""
        mock_settings.enable_model_caching = False
        
        with patch('src.warship_extractor.core.model_manager.AutoProcessor'), \
             patch('src.warship_extractor.core.model_manager.AutoModelForCausalLM'):
            manager = ModelManager(mock_settings)
        
        # Setup mocks
        mock_processor_load.return_value = Mock()
        mock_model_load.return_value = Mock()
        
        # Load model
        manager.load_model()
        
        # Verify cache_dir is not passed when caching is disabled
        processor_call_args = mock_processor_load.call_args
        model_call_args = mock_model_load.call_args
        
        assert 'cache_dir' not in processor_call_args[1]
        assert 'cache_dir' not in model_call_args[1]
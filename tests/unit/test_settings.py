"""
Unit tests for the settings and configuration module.

Tests the Settings class, environment variable handling,
path management, and configuration validation.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest

from warship_extractor.config.settings import Settings


class TestSettings:
    """Test cases for the Settings class."""

    def test_init_default_values(self):
        """Test Settings initialization with default values."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            
            assert settings.model_name == "microsoft/Florence-2-large"
            assert settings.confidence_threshold == 0.3
            assert settings.pdf_dpi == 300
            assert settings.batch_size == 1
            assert settings.enable_gpu is True
            assert settings.log_level == "INFO"

    def test_init_with_env_variables(self):
        """Test Settings initialization with environment variables."""
        env_vars = {
            'WARSHIP_MODEL_NAME': 'microsoft/Florence-2-base',
            'WARSHIP_CONFIDENCE_THRESHOLD': '0.5',
            'WARSHIP_PDF_DPI': '200',
            'WARSHIP_BATCH_SIZE': '2',
            'WARSHIP_ENABLE_GPU': 'false',
            'WARSHIP_LOG_LEVEL': 'DEBUG'
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()
            
            assert settings.model_name == "microsoft/Florence-2-base"
            assert settings.confidence_threshold == 0.5
            assert settings.pdf_dpi == 200
            assert settings.batch_size == 2
            assert settings.enable_gpu is False
            assert settings.log_level == "DEBUG"

    def test_boolean_env_variables(self):
        """Test boolean environment variable parsing."""
        test_cases = [
            ('true', True),
            ('True', True),
            ('TRUE', True),
            ('1', True),
            ('yes', True),
            ('false', False),
            ('False', False),
            ('FALSE', False),
            ('0', False),
            ('no', False),
            ('', False),
            ('invalid', False)
        ]
        
        for env_value, expected in test_cases:
            with patch.dict(os.environ, {'WARSHIP_ENABLE_GPU': env_value}, clear=True):
                settings = Settings()
                assert settings.enable_gpu == expected, f"'{env_value}' should parse to {expected}"

    def test_cache_dir_property(self):
        """Test cache directory property."""
        settings = Settings()
        
        cache_dir = settings.cache_dir
        assert isinstance(cache_dir, Path)
        assert cache_dir.name == "cache"

    def test_output_dir_property(self):
        """Test output directory property."""
        settings = Settings()
        
        output_dir = settings.output_dir
        assert isinstance(output_dir, Path)
        assert output_dir.name == "output"

    def test_logs_dir_property(self):
        """Test logs directory property."""
        settings = Settings()
        
        logs_dir = settings.logs_dir
        assert isinstance(logs_dir, Path)
        assert logs_dir.name == "logs"

    def test_get_cache_path(self):
        """Test cache path generation."""
        settings = Settings()
        
        cache_path = settings.get_cache_path("model.bin")
        
        assert isinstance(cache_path, Path)
        assert cache_path.parent.name == "cache"
        assert cache_path.name == "model.bin"

    def test_get_output_path(self):
        """Test output path generation."""
        settings = Settings()
        
        output_path = settings.get_output_path("results.json")
        
        assert isinstance(output_path, Path)
        assert output_path.parent.name == "output"
        assert output_path.name == "results.json"

    def test_get_log_path(self):
        """Test log path generation."""
        settings = Settings()
        
        log_path = settings.get_log_path("app.log")
        
        assert isinstance(log_path, Path)
        assert log_path.parent.name == "logs"
        assert log_path.name == "app.log"

    def test_custom_base_dir(self):
        """Test settings with custom base directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_base = Path(temp_dir)
            
            with patch.dict(os.environ, {'WARSHIP_BASE_DIR': str(custom_base)}, clear=True):
                settings = Settings()
                
                assert settings.base_dir == custom_base
                assert settings.cache_dir == custom_base / "cache"
                assert settings.output_dir == custom_base / "output"
                assert settings.logs_dir == custom_base / "logs"

    def test_invalid_numeric_env_variables(self):
        """Test handling of invalid numeric environment variables."""
        invalid_values = ['not_a_number', '1.5.2', 'abc', '']
        
        for invalid_value in invalid_values:
            with patch.dict(os.environ, {'WARSHIP_PDF_DPI': invalid_value}, clear=True):
                settings = Settings()
                # Should fall back to default value
                assert settings.pdf_dpi == 300

    def test_confidence_threshold_validation(self):
        """Test confidence threshold validation."""
        test_cases = [
            ('0.0', 0.0),
            ('0.5', 0.5),
            ('1.0', 1.0),
            ('-0.1', 0.3),  # Invalid, should use default
            ('1.1', 0.3),   # Invalid, should use default
            ('invalid', 0.3)  # Invalid, should use default
        ]
        
        for env_value, expected in test_cases:
            with patch.dict(os.environ, {'WARSHIP_CONFIDENCE_THRESHOLD': env_value}, clear=True):
                settings = Settings()
                assert settings.confidence_threshold == expected

    def test_log_level_validation(self):
        """Test log level validation."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        for level in valid_levels:
            with patch.dict(os.environ, {'WARSHIP_LOG_LEVEL': level}, clear=True):
                settings = Settings()
                assert settings.log_level == level
        
        # Test invalid level
        with patch.dict(os.environ, {'WARSHIP_LOG_LEVEL': 'INVALID'}, clear=True):
            settings = Settings()
            assert settings.log_level == 'INFO'  # Should use default

    def test_batch_size_validation(self):
        """Test batch size validation."""
        test_cases = [
            ('1', 1),
            ('4', 4),
            ('8', 8),
            ('0', 1),      # Invalid, should use default
            ('-1', 1),     # Invalid, should use default
            ('invalid', 1) # Invalid, should use default
        ]
        
        for env_value, expected in test_cases:
            with patch.dict(os.environ, {'WARSHIP_BATCH_SIZE': env_value}, clear=True):
                settings = Settings()
                assert settings.batch_size == expected

    def test_pdf_dpi_validation(self):
        """Test PDF DPI validation."""
        test_cases = [
            ('150', 150),
            ('300', 300),
            ('600', 600),
            ('50', 300),    # Too low, should use default
            ('1200', 300),  # Too high, should use default
            ('invalid', 300) # Invalid, should use default
        ]
        
        for env_value, expected in test_cases:
            with patch.dict(os.environ, {'WARSHIP_PDF_DPI': env_value}, clear=True):
                settings = Settings()
                assert settings.pdf_dpi == expected

    def test_enable_file_logging_default(self):
        """Test file logging enabled by default."""
        settings = Settings()
        assert settings.enable_file_logging is True

    def test_enable_file_logging_env_override(self):
        """Test file logging can be disabled via environment."""
        with patch.dict(os.environ, {'WARSHIP_ENABLE_FILE_LOGGING': 'false'}, clear=True):
            settings = Settings()
            assert settings.enable_file_logging is False

    def test_max_concurrent_processes_default(self):
        """Test max concurrent processes default value."""
        settings = Settings()
        # Should be reasonable default, typically CPU count
        assert isinstance(settings.max_concurrent_processes, int)
        assert settings.max_concurrent_processes > 0
        assert settings.max_concurrent_processes <= 16  # Reasonable upper bound

    def test_max_concurrent_processes_env_override(self):
        """Test max concurrent processes environment override."""
        with patch.dict(os.environ, {'WARSHIP_MAX_CONCURRENT_PROCESSES': '2'}, clear=True):
            settings = Settings()
            assert settings.max_concurrent_processes == 2

    def test_memory_limit_default(self):
        """Test memory limit default value."""
        settings = Settings()
        assert isinstance(settings.memory_limit_gb, (int, float))
        assert settings.memory_limit_gb > 0

    def test_memory_limit_env_override(self):
        """Test memory limit environment override."""
        with patch.dict(os.environ, {'WARSHIP_MEMORY_LIMIT_GB': '8'}, clear=True):
            settings = Settings()
            assert settings.memory_limit_gb == 8

    def test_model_cache_size_default(self):
        """Test model cache size default value."""
        settings = Settings()
        assert isinstance(settings.model_cache_size, int)
        assert settings.model_cache_size > 0

    def test_model_cache_size_env_override(self):
        """Test model cache size environment override."""
        with patch.dict(os.environ, {'WARSHIP_MODEL_CACHE_SIZE': '2'}, clear=True):
            settings = Settings()
            assert settings.model_cache_size == 2

    def test_to_dict(self):
        """Test converting settings to dictionary."""
        settings = Settings()
        
        settings_dict = settings.to_dict()
        
        assert isinstance(settings_dict, dict)
        assert 'model_name' in settings_dict
        assert 'confidence_threshold' in settings_dict
        assert 'pdf_dpi' in settings_dict
        assert 'batch_size' in settings_dict
        assert 'enable_gpu' in settings_dict
        assert 'log_level' in settings_dict

    def test_from_dict(self):
        """Test creating settings from dictionary."""
        config_dict = {
            'model_name': 'microsoft/Florence-2-base',
            'confidence_threshold': 0.4,
            'pdf_dpi': 200,
            'batch_size': 2,
            'enable_gpu': False,
            'log_level': 'DEBUG'
        }
        
        settings = Settings.from_dict(config_dict)
        
        assert settings.model_name == 'microsoft/Florence-2-base'
        assert settings.confidence_threshold == 0.4
        assert settings.pdf_dpi == 200
        assert settings.batch_size == 2
        assert settings.enable_gpu is False
        assert settings.log_level == 'DEBUG'

    def test_load_from_file_json(self):
        """Test loading settings from JSON file."""
        config_data = {
            'model_name': 'microsoft/Florence-2-base',
            'confidence_threshold': 0.4,
            'pdf_dpi': 200
        }
        
        json_content = '{"model_name": "microsoft/Florence-2-base", "confidence_threshold": 0.4, "pdf_dpi": 200}'
        
        with patch('builtins.open', mock_open(read_data=json_content)):
            with patch('pathlib.Path.exists', return_value=True):
                settings = Settings.load_from_file(Path('config.json'))
                
                assert settings.model_name == 'microsoft/Florence-2-base'
                assert settings.confidence_threshold == 0.4
                assert settings.pdf_dpi == 200

    def test_load_from_file_yaml(self):
        """Test loading settings from YAML file."""
        yaml_content = """
        model_name: microsoft/Florence-2-base
        confidence_threshold: 0.4
        pdf_dpi: 200
        """
        
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            with patch('pathlib.Path.exists', return_value=True):
                with patch('yaml.safe_load') as mock_yaml:
                    mock_yaml.return_value = {
                        'model_name': 'microsoft/Florence-2-base',
                        'confidence_threshold': 0.4,
                        'pdf_dpi': 200
                    }
                    
                    settings = Settings.load_from_file(Path('config.yaml'))
                    
                    assert settings.model_name == 'microsoft/Florence-2-base'
                    assert settings.confidence_threshold == 0.4
                    assert settings.pdf_dpi == 200

    def test_load_from_file_nonexistent(self):
        """Test loading settings from non-existent file."""
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                Settings.load_from_file(Path('nonexistent.json'))

    def test_save_to_file_json(self):
        """Test saving settings to JSON file."""
        settings = Settings()
        
        with patch('builtins.open', mock_open()) as mock_file:
            settings.save_to_file(Path('config.json'))
            
            mock_file.assert_called_once()
            # Verify JSON content was written
            handle = mock_file()
            handle.write.assert_called()

    def test_save_to_file_yaml(self):
        """Test saving settings to YAML file."""
        settings = Settings()
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('yaml.dump') as mock_yaml_dump:
                settings.save_to_file(Path('config.yaml'))
                
                mock_file.assert_called_once()
                mock_yaml_dump.assert_called_once()

    def test_validate_model_name(self):
        """Test model name validation."""
        valid_names = [
            'microsoft/Florence-2-large',
            'microsoft/Florence-2-base',
            'custom/model-name'
        ]
        
        for name in valid_names:
            with patch.dict(os.environ, {'WARSHIP_MODEL_NAME': name}, clear=True):
                settings = Settings()
                assert settings.model_name == name

    def test_settings_immutability(self):
        """Test that settings can be updated after creation."""
        settings = Settings()
        
        # Settings should be mutable for runtime updates
        original_confidence = settings.confidence_threshold
        settings.confidence_threshold = 0.7
        assert settings.confidence_threshold == 0.7
        assert settings.confidence_threshold != original_confidence

    @pytest.mark.parametrize("dpi,expected", [
        (150, 150),
        (200, 200),
        (300, 300),
        (400, 400),
        (600, 600),
    ])
    def test_valid_dpi_values(self, dpi, expected):
        """Test various valid DPI values."""
        with patch.dict(os.environ, {'WARSHIP_PDF_DPI': str(dpi)}, clear=True):
            settings = Settings()
            assert settings.pdf_dpi == expected

    def test_create_directories(self):
        """Test directory creation functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_base = Path(temp_dir) / "warship_test"
            
            with patch.dict(os.environ, {'WARSHIP_BASE_DIR': str(custom_base)}, clear=True):
                settings = Settings()
                
                # Directories should be created
                settings.create_directories()
                
                assert settings.cache_dir.exists()
                assert settings.output_dir.exists()
                assert settings.logs_dir.exists()

    def test_settings_repr(self):
        """Test settings string representation."""
        settings = Settings()
        
        repr_str = repr(settings)
        
        assert 'Settings' in repr_str
        assert 'model_name' in repr_str
        assert 'confidence_threshold' in repr_str
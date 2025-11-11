"""
Unit tests for the WarshipLogger class.

Tests logging functionality, progress tracking, and performance monitoring
without requiring actual file I/O or external dependencies.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import tempfile
import logging
import json

from src.warship_extractor.utils.logger import WarshipLogger
from src.warship_extractor.config.settings import Settings


class TestWarshipLogger:
    """Test cases for WarshipLogger functionality."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = Mock()
        settings.enable_logging = True
        settings.log_level = "INFO"
        settings.log_file = Path("warship_extraction.log")
        settings.log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        settings.log_max_size_mb = 10
        settings.log_backup_count = 5
        settings.enable_console_logging = True
        settings.enable_file_logging = True
        settings.enable_progress_tracking = True
        settings.enable_performance_monitoring = True
        return settings
    
    @pytest.fixture
    def logger(self, mock_settings):
        """Create WarshipLogger instance for testing."""
        with patch('logging.getLogger'), \
             patch('logging.handlers.RotatingFileHandler'), \
             patch('logging.StreamHandler'):
            return WarshipLogger(mock_settings)
    
    def test_initialization(self, mock_settings):
        """Test WarshipLogger initialization."""
        with patch('logging.getLogger') as mock_get_logger, \
             patch('logging.handlers.RotatingFileHandler') as mock_file_handler, \
             patch('logging.StreamHandler') as mock_console_handler:
            
            mock_logger_instance = Mock()
            mock_get_logger.return_value = mock_logger_instance
            
            logger = WarshipLogger(mock_settings)
            
            assert logger.settings == mock_settings
            mock_get_logger.assert_called_once_with('warship_extractor')
            
            # Should set up handlers based on settings
            if mock_settings.enable_file_logging:
                mock_file_handler.assert_called_once()
            if mock_settings.enable_console_logging:
                mock_console_handler.assert_called_once()
    
    def test_initialization_logging_disabled(self, mock_settings):
        """Test initialization when logging is disabled."""
        mock_settings.enable_logging = False
        
        with patch('logging.getLogger') as mock_get_logger:
            logger = WarshipLogger(mock_settings)
            
            # Should still create logger but not set it up
            mock_get_logger.assert_called_once()
    
    def test_log_info(self, logger):
        """Test info level logging."""
        message = "Test info message"
        
        logger.info(message)
        
        logger.logger.info.assert_called_once_with(message)
    
    def test_log_error(self, logger):
        """Test error level logging."""
        message = "Test error message"
        
        logger.error(message)
        
        logger.logger.error.assert_called_once_with(message)
    
    def test_log_debug(self, logger):
        """Test debug level logging."""
        message = "Test debug message"
        
        logger.debug(message)
        
        logger.logger.debug.assert_called_once_with(message)
    
    def test_log_warning(self, logger):
        """Test warning level logging."""
        message = "Test warning message"
        
        logger.warning(message)
        
        logger.logger.warning.assert_called_once_with(message)
    
    def test_log_with_extra_context(self, logger):
        """Test logging with extra context information."""
        message = "Test message"
        extra_context = {"pdf_path": "test.pdf", "page_number": 5}
        
        logger.info(message, extra=extra_context)
        
        logger.logger.info.assert_called_once_with(message, extra=extra_context)
    
    def test_start_progress_tracking(self, logger):
        """Test starting progress tracking."""
        task_name = "PDF Processing"
        total_items = 100
        
        logger.start_progress(task_name, total_items)
        
        assert logger._current_progress['task_name'] == task_name
        assert logger._current_progress['total_items'] == total_items
        assert logger._current_progress['current_item'] == 0
        assert 'start_time' in logger._current_progress
    
    def test_update_progress(self, logger):
        """Test progress updates."""
        # Start progress first
        logger.start_progress("Test Task", 100)
        
        # Update progress
        logger.update_progress(25, "Processing item 25")
        
        assert logger._current_progress['current_item'] == 25
        assert logger._current_progress['message'] == "Processing item 25"
    
    def test_finish_progress(self, logger):
        """Test finishing progress tracking."""
        # Start progress first
        logger.start_progress("Test Task", 100)
        
        with patch('time.time') as mock_time:
            mock_time.side_effect = [1000, 1010]  # 10 second duration
            
            logger.start_progress("Test Task", 100)  # Reset start time
            logger.finish_progress("Task completed successfully")
            
            assert logger._current_progress['current_item'] == 100
            assert logger._current_progress['message'] == "Task completed successfully"
            assert 'end_time' in logger._current_progress
    
    def test_progress_percentage_calculation(self, logger):
        """Test progress percentage calculation."""
        logger.start_progress("Test Task", 100)
        
        logger.update_progress(25)
        assert logger.get_progress_percentage() == 25.0
        
        logger.update_progress(50)
        assert logger.get_progress_percentage() == 50.0
        
        logger.update_progress(100)
        assert logger.get_progress_percentage() == 100.0
    
    def test_progress_with_zero_total(self, logger):
        """Test progress tracking with zero total items."""
        logger.start_progress("Empty Task", 0)
        
        # Should handle division by zero gracefully
        percentage = logger.get_progress_percentage()
        assert percentage == 0.0
    
    def test_performance_monitoring_start_timer(self, logger):
        """Test starting performance timer."""
        operation_name = "PDF_CONVERSION"
        
        with patch('time.time', return_value=1000):
            logger.start_timer(operation_name)
            
            assert operation_name in logger._timers
            assert logger._timers[operation_name]['start_time'] == 1000
    
    def test_performance_monitoring_end_timer(self, logger):
        """Test ending performance timer."""
        operation_name = "PDF_CONVERSION"
        
        with patch('time.time') as mock_time:
            mock_time.side_effect = [1000, 1005]  # 5 second duration
            
            logger.start_timer(operation_name)
            duration = logger.end_timer(operation_name)
            
            assert duration == 5.0
            assert logger._timers[operation_name]['end_time'] == 1005
            assert logger._timers[operation_name]['duration'] == 5.0
    
    def test_performance_monitoring_end_timer_not_started(self, logger):
        """Test ending timer that was never started."""
        operation_name = "NON_EXISTENT"
        
        duration = logger.end_timer(operation_name)
        
        # Should return None or 0 for non-existent timer
        assert duration in (None, 0)
    
    def test_log_performance_metrics(self, logger):
        """Test logging performance metrics."""
        # Setup some timers
        logger._timers = {
            'PDF_CONVERSION': {
                'start_time': 1000,
                'end_time': 1005,
                'duration': 5.0
            },
            'DETECTION': {
                'start_time': 1005,
                'end_time': 1015,
                'duration': 10.0
            }
        }
        
        logger.log_performance_summary()
        
        # Should log performance information
        logger.logger.info.assert_called()
        
        # Verify the call included performance data
        call_args = logger.logger.info.call_args_list
        assert any('Performance Summary' in str(call) for call in call_args)
    
    def test_context_manager_for_timing(self, logger):
        """Test using logger as context manager for timing."""
        operation_name = "TEST_OPERATION"
        
        with patch('time.time') as mock_time:
            mock_time.side_effect = [1000, 1003]  # 3 second duration
            
            with logger.time_operation(operation_name):
                pass  # Simulate some work
            
            # Should have recorded the timing
            assert operation_name in logger._timers
            assert logger._timers[operation_name]['duration'] == 3.0
    
    def test_context_manager_with_exception(self, logger):
        """Test timing context manager with exception."""
        operation_name = "FAILING_OPERATION"
        
        with patch('time.time') as mock_time:
            mock_time.side_effect = [1000, 1002]
            
            with pytest.raises(ValueError):
                with logger.time_operation(operation_name):
                    raise ValueError("Test exception")
            
            # Should still record timing even with exception
            assert operation_name in logger._timers
            assert logger._timers[operation_name]['duration'] == 2.0
    
    def test_log_structured_data(self, logger):
        """Test logging structured data."""
        data = {
            'pdf_path': 'test.pdf',
            'total_pages': 25,
            'detections_found': 10,
            'processing_time': 15.5
        }
        
        logger.log_structured(data, level='INFO')
        
        # Should log with structured format
        logger.logger.info.assert_called()
    
    def test_log_exception_with_traceback(self, logger):
        """Test logging exceptions with traceback."""
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            logger.log_exception(e, "Error during processing")
        
        # Should log error with exception info
        logger.logger.error.assert_called()
        call_args = logger.logger.error.call_args
        assert 'Error during processing' in str(call_args)
    
    def test_create_log_entry(self, logger):
        """Test creating structured log entries."""
        entry = logger.create_log_entry(
            level='INFO',
            message='Test message',
            component='PDF_PROCESSOR',
            operation='CONVERT',
            metadata={'page': 5}
        )
        
        expected_keys = ['timestamp', 'level', 'message', 'component', 'operation', 'metadata']
        for key in expected_keys:
            assert key in entry
        
        assert entry['level'] == 'INFO'
        assert entry['message'] == 'Test message'
        assert entry['component'] == 'PDF_PROCESSOR'
        assert entry['operation'] == 'CONVERT'
        assert entry['metadata']['page'] == 5
    
    def test_export_logs_to_json(self, logger):
        """Test exporting logs to JSON format."""
        # Setup some log entries
        logger._log_entries = [
            {'timestamp': '2023-01-01T10:00:00', 'level': 'INFO', 'message': 'Test 1'},
            {'timestamp': '2023-01-01T10:01:00', 'level': 'ERROR', 'message': 'Test 2'}
        ]
        
        output_path = Path("test_logs.json")
        
        with patch('builtins.open', mock_open()) as mock_file:
            logger.export_logs(output_path, format='json')
            
            mock_file.assert_called_once_with(output_path, 'w')
            # Should write JSON data
            written_data = ''.join(call.args[0] for call in mock_file().write.call_args_list)
            assert '"timestamp"' in written_data
            assert '"level"' in written_data
    
    def test_export_logs_to_csv(self, logger):
        """Test exporting logs to CSV format."""
        logger._log_entries = [
            {'timestamp': '2023-01-01T10:00:00', 'level': 'INFO', 'message': 'Test 1'},
            {'timestamp': '2023-01-01T10:01:00', 'level': 'ERROR', 'message': 'Test 2'}
        ]
        
        output_path = Path("test_logs.csv")
        
        with patch('builtins.open', mock_open()) as mock_file:
            logger.export_logs(output_path, format='csv')
            
            mock_file.assert_called_once_with(output_path, 'w', newline='')
    
    def test_filter_logs_by_level(self, logger):
        """Test filtering logs by level."""
        logger._log_entries = [
            {'level': 'INFO', 'message': 'Info message'},
            {'level': 'ERROR', 'message': 'Error message'},
            {'level': 'DEBUG', 'message': 'Debug message'},
            {'level': 'ERROR', 'message': 'Another error'}
        ]
        
        error_logs = logger.filter_logs(level='ERROR')
        
        assert len(error_logs) == 2
        assert all(log['level'] == 'ERROR' for log in error_logs)
    
    def test_filter_logs_by_component(self, logger):
        """Test filtering logs by component."""
        logger._log_entries = [
            {'component': 'PDF_PROCESSOR', 'message': 'PDF message'},
            {'component': 'DETECTOR', 'message': 'Detection message'},
            {'component': 'PDF_PROCESSOR', 'message': 'Another PDF message'}
        ]
        
        pdf_logs = logger.filter_logs(component='PDF_PROCESSOR')
        
        assert len(pdf_logs) == 2
        assert all(log['component'] == 'PDF_PROCESSOR' for log in pdf_logs)
    
    def test_filter_logs_by_time_range(self, logger):
        """Test filtering logs by time range."""
        logger._log_entries = [
            {'timestamp': '2023-01-01T09:00:00', 'message': 'Early message'},
            {'timestamp': '2023-01-01T10:30:00', 'message': 'Middle message'},
            {'timestamp': '2023-01-01T12:00:00', 'message': 'Late message'}
        ]
        
        filtered_logs = logger.filter_logs(
            start_time='2023-01-01T10:00:00',
            end_time='2023-01-01T11:00:00'
        )
        
        assert len(filtered_logs) == 1
        assert filtered_logs[0]['message'] == 'Middle message'
    
    def test_get_log_statistics(self, logger):
        """Test getting log statistics."""
        logger._log_entries = [
            {'level': 'INFO', 'component': 'PDF_PROCESSOR'},
            {'level': 'ERROR', 'component': 'DETECTOR'},
            {'level': 'INFO', 'component': 'PDF_PROCESSOR'},
            {'level': 'WARNING', 'component': 'DETECTOR'},
            {'level': 'ERROR', 'component': 'PDF_PROCESSOR'}
        ]
        
        stats = logger.get_log_statistics()
        
        expected_keys = [
            'total_entries', 'levels_count', 'components_count',
            'most_common_level', 'most_active_component'
        ]
        
        for key in expected_keys:
            assert key in stats
        
        assert stats['total_entries'] == 5
        assert stats['levels_count']['INFO'] == 2
        assert stats['levels_count']['ERROR'] == 2
        assert stats['levels_count']['WARNING'] == 1
        assert stats['components_count']['PDF_PROCESSOR'] == 3
        assert stats['components_count']['DETECTOR'] == 2
    
    def test_clear_logs(self, logger):
        """Test clearing log entries."""
        logger._log_entries = [
            {'message': 'Entry 1'},
            {'message': 'Entry 2'}
        ]
        logger._timers = {'TIMER1': {'duration': 5.0}}
        
        logger.clear_logs()
        
        assert len(logger._log_entries) == 0
        assert len(logger._timers) == 0
    
    def test_log_level_validation(self, mock_settings):
        """Test log level validation."""
        mock_settings.log_level = "INVALID"
        
        with pytest.raises(ValueError, match="Invalid log level"):
            with patch('logging.getLogger'), \
                 patch('logging.handlers.RotatingFileHandler'), \
                 patch('logging.StreamHandler'):
                WarshipLogger(mock_settings)
    
    def test_file_handler_rotation(self, mock_settings):
        """Test log file rotation configuration."""
        mock_settings.log_max_size_mb = 5
        mock_settings.log_backup_count = 3
        
        with patch('logging.getLogger'), \
             patch('logging.handlers.RotatingFileHandler') as mock_file_handler, \
             patch('logging.StreamHandler'):
            
            WarshipLogger(mock_settings)
            
            # Should configure rotation with correct parameters
            mock_file_handler.assert_called_once_with(
                filename=str(mock_settings.log_file),
                maxBytes=5 * 1024 * 1024,  # 5MB in bytes
                backupCount=3
            )
    
    def test_console_logging_disabled(self, mock_settings):
        """Test behavior when console logging is disabled."""
        mock_settings.enable_console_logging = False
        
        with patch('logging.getLogger'), \
             patch('logging.handlers.RotatingFileHandler'), \
             patch('logging.StreamHandler') as mock_console_handler:
            
            WarshipLogger(mock_settings)
            
            # Should not create console handler
            mock_console_handler.assert_not_called()
    
    def test_file_logging_disabled(self, mock_settings):
        """Test behavior when file logging is disabled."""
        mock_settings.enable_file_logging = False
        
        with patch('logging.getLogger'), \
             patch('logging.handlers.RotatingFileHandler') as mock_file_handler, \
             patch('logging.StreamHandler'):
            
            WarshipLogger(mock_settings)
            
            # Should not create file handler
            mock_file_handler.assert_not_called()
    
    def test_custom_log_format(self, mock_settings):
        """Test custom log format configuration."""
        custom_format = "%(levelname)s: %(message)s"
        mock_settings.log_format = custom_format
        
        with patch('logging.getLogger'), \
             patch('logging.handlers.RotatingFileHandler'), \
             patch('logging.StreamHandler'), \
             patch('logging.Formatter') as mock_formatter:
            
            WarshipLogger(mock_settings)
            
            # Should use custom format
            mock_formatter.assert_called_with(custom_format)
    
    def test_logger_name_configuration(self, mock_settings):
        """Test logger name configuration."""
        with patch('logging.getLogger') as mock_get_logger:
            WarshipLogger(mock_settings)
            
            # Should create logger with correct name
            mock_get_logger.assert_called_once_with('warship_extractor')
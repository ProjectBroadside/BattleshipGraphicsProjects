"""
Unit tests for the CLI interface.

Tests command-line argument parsing, command execution,
and error handling for the warship extractor CLI.
"""

import argparse
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

import pytest

from warship_extractor.cli import (
    create_parser,
    parse_page_ranges,
    extract_command,
    batch_command,
    info_command,
    main
)


class TestCLIParser:
    """Test cases for CLI argument parsing."""

    def test_create_parser_basic(self):
        """Test basic parser creation."""
        parser = create_parser()
        
        assert isinstance(parser, argparse.ArgumentParser)
        assert parser.prog == 'warship-extractor'

    def test_parser_extract_command(self):
        """Test extract command parsing."""
        parser = create_parser()
        
        args = parser.parse_args(['extract', 'input.pdf'])
        
        assert args.command == 'extract'
        assert args.input_file == Path('input.pdf')
        assert args.confidence_threshold == 0.3  # default
        assert args.dpi == 300  # default

    def test_parser_extract_command_with_options(self):
        """Test extract command with optional arguments."""
        parser = create_parser()
        
        args = parser.parse_args([
            'extract', 'input.pdf',
            '--output-dir', '/output',
            '--confidence-threshold', '0.5',
            '--dpi', '200',
            '--pages', '1-10,15',
            '--max-pages', '20',
            '--no-nms',
            '--save-debug',
            '--generate-report'
        ])
        
        assert args.command == 'extract'
        assert args.input_file == Path('input.pdf')
        assert args.output_dir == Path('/output')
        assert args.confidence_threshold == 0.5
        assert args.dpi == 200
        assert args.pages == '1-10,15'
        assert args.max_pages == 20
        assert args.no_nms is True
        assert args.save_debug is True
        assert args.generate_report is True

    def test_parser_batch_command(self):
        """Test batch command parsing."""
        parser = create_parser()
        
        args = parser.parse_args(['batch', '/input/dir'])
        
        assert args.command == 'batch'
        assert args.input_dir == Path('/input/dir')
        assert args.pattern == '*.pdf'  # default
        assert args.confidence_threshold == 0.3  # default

    def test_parser_batch_command_with_options(self):
        """Test batch command with optional arguments."""
        parser = create_parser()
        
        args = parser.parse_args([
            'batch', '/input/dir',
            '--pattern', '*.PDF',
            '--output-dir', '/output',
            '--confidence-threshold', '0.4',
            '--max-files', '10',
            '--parallel',
            '--generate-reports'
        ])
        
        assert args.command == 'batch'
        assert args.input_dir == Path('/input/dir')
        assert args.pattern == '*.PDF'
        assert args.output_dir == Path('/output')
        assert args.confidence_threshold == 0.4
        assert args.max_files == 10
        assert args.parallel is True
        assert args.generate_reports is True

    def test_parser_report_command(self):
        """Test report command parsing."""
        parser = create_parser()
        
        args = parser.parse_args(['report', 'results.json'])
        
        assert args.command == 'report'
        assert args.results_file == Path('results.json')

    def test_parser_report_command_with_options(self):
        """Test report command with optional arguments."""
        parser = create_parser()
        
        args = parser.parse_args([
            'report', 'results.json',
            '--output-dir', '/reports',
            '--template', 'detailed'
        ])
        
        assert args.command == 'report'
        assert args.results_file == Path('results.json')
        assert args.output_dir == Path('/reports')
        assert args.template == 'detailed'

    def test_parser_info_command(self):
        """Test info command parsing."""
        parser = create_parser()
        
        args = parser.parse_args(['info'])
        
        assert args.command == 'info'

    def test_parser_info_command_with_options(self):
        """Test info command with optional arguments."""
        parser = create_parser()
        
        args = parser.parse_args([
            'info',
            '--show-config',
            '--show-model'
        ])
        
        assert args.command == 'info'
        assert args.show_config is True
        assert args.show_model is True

    def test_parser_global_options(self):
        """Test global options parsing."""
        parser = create_parser()
        
        args = parser.parse_args([
            '--log-level', 'DEBUG',
            '--log-file', '/logs/app.log',
            '--config', '/config/settings.yaml',
            '--no-color',
            'extract', 'input.pdf'
        ])
        
        assert args.log_level == 'DEBUG'
        assert args.log_file == Path('/logs/app.log')
        assert args.config == Path('/config/settings.yaml')
        assert args.no_color is True

    def test_parser_version(self):
        """Test version argument."""
        parser = create_parser()
        
        with pytest.raises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                parser.parse_args(['--version'])


class TestPageRangeParsing:
    """Test cases for page range parsing."""

    def test_parse_single_page(self):
        """Test parsing single page number."""
        pages = parse_page_ranges("5")
        assert pages == [5]

    def test_parse_page_range(self):
        """Test parsing page range."""
        pages = parse_page_ranges("1-5")
        assert pages == [1, 2, 3, 4, 5]

    def test_parse_multiple_ranges(self):
        """Test parsing multiple page ranges."""
        pages = parse_page_ranges("1-3,5,7-9")
        assert pages == [1, 2, 3, 5, 7, 8, 9]

    def test_parse_overlapping_ranges(self):
        """Test parsing overlapping page ranges."""
        pages = parse_page_ranges("1-5,3-7")
        assert pages == [1, 2, 3, 4, 5, 6, 7]

    def test_parse_invalid_range(self):
        """Test parsing invalid page range."""
        with pytest.raises(ValueError):
            parse_page_ranges("invalid")

    def test_parse_empty_string(self):
        """Test parsing empty string."""
        with pytest.raises(ValueError):
            parse_page_ranges("")


class TestExtractCommand:
    """Test cases for extract command execution."""

    @patch('warship_extractor.cli.ExtractionPipeline')
    @patch('warship_extractor.cli.create_summary_report')
    def test_extract_command_success(self, mock_report, mock_pipeline, temp_dir):
        """Test successful extract command execution."""
        # Create test PDF
        test_pdf = temp_dir / "test.pdf"
        test_pdf.touch()
        
        # Mock pipeline
        mock_pipeline_instance = mock_pipeline.return_value
        mock_pipeline_instance.process_pdf.return_value = {
            'detections': [],
            'statistics': {'total_detections': 0, 'processing_time': 1.0},
            'metadata': {}
        }
        
        # Create args mock
        args = Mock()
        args.input_file = test_pdf
        args.output_dir = None
        args.confidence_threshold = 0.3
        args.pages = None
        args.max_pages = None
        args.dpi = 300
        args.no_nms = False
        args.no_enhancement = False
        args.save_debug = False
        args.generate_report = False
        
        result = extract_command(args)
        
        assert result == 0
        mock_pipeline.assert_called_once()
        mock_pipeline_instance.process_pdf.assert_called_once()

    def test_extract_command_file_not_found(self, temp_dir):
        """Test extract command with non-existent file."""
        args = Mock()
        args.input_file = temp_dir / "nonexistent.pdf"
        
        result = extract_command(args)
        
        assert result == 1

    def test_extract_command_invalid_file_type(self, temp_dir):
        """Test extract command with non-PDF file."""
        # Create non-PDF file
        test_file = temp_dir / "test.txt"
        test_file.touch()
        
        args = Mock()
        args.input_file = test_file
        
        result = extract_command(args)
        
        assert result == 1

    @patch('warship_extractor.cli.ExtractionPipeline')
    def test_extract_command_with_pages(self, mock_pipeline, temp_dir):
        """Test extract command with specific pages."""
        test_pdf = temp_dir / "test.pdf"
        test_pdf.touch()
        
        mock_pipeline_instance = mock_pipeline.return_value
        mock_pipeline_instance.process_pdf.return_value = {
            'detections': [],
            'statistics': {'total_detections': 0, 'processing_time': 1.0},
            'metadata': {}
        }
        
        args = Mock()
        args.input_file = test_pdf
        args.output_dir = None
        args.confidence_threshold = 0.3
        args.pages = "1-5,10"
        args.max_pages = None
        args.dpi = 300
        args.no_nms = False
        args.no_enhancement = False
        args.save_debug = False
        args.generate_report = False
        
        result = extract_command(args)
        
        assert result == 0
        # Should have parsed pages correctly
        call_args = mock_pipeline_instance.process_pdf.call_args
        assert 'pages' in call_args.kwargs
        assert call_args.kwargs['pages'] == [1, 2, 3, 4, 5, 10]

    def test_extract_command_invalid_pages(self, temp_dir):
        """Test extract command with invalid page specification."""
        test_pdf = temp_dir / "test.pdf"
        test_pdf.touch()
        
        args = Mock()
        args.input_file = test_pdf
        args.pages = "invalid"
        
        result = extract_command(args)
        
        assert result == 1


class TestBatchCommand:
    """Test cases for batch command execution."""

    @patch('warship_extractor.cli.ExtractionPipeline')
    def test_batch_command_success(self, mock_pipeline, temp_dir):
        """Test successful batch command execution."""
        # Create test PDFs
        pdf1 = temp_dir / "test1.pdf"
        pdf2 = temp_dir / "test2.pdf"
        pdf1.touch()
        pdf2.touch()
        
        mock_pipeline_instance = mock_pipeline.return_value
        mock_pipeline_instance.process_pdf.return_value = {
            'detections': [],
            'statistics': {'total_detections': 0, 'processing_time': 1.0},
            'metadata': {}
        }
        
        args = Mock()
        args.input_dir = temp_dir
        args.pattern = "*.pdf"
        args.output_dir = None
        args.confidence_threshold = 0.3
        args.max_files = None
        args.parallel = False
        args.generate_reports = False
        
        result = batch_command(args)
        
        assert result == 0
        # Should process both PDFs
        assert mock_pipeline_instance.process_pdf.call_count == 2

    def test_batch_command_no_files(self, temp_dir):
        """Test batch command with no matching files."""
        args = Mock()
        args.input_dir = temp_dir
        args.pattern = "*.pdf"
        
        result = batch_command(args)
        
        assert result == 1

    def test_batch_command_directory_not_found(self, temp_dir):
        """Test batch command with non-existent directory."""
        args = Mock()
        args.input_dir = temp_dir / "nonexistent"
        
        result = batch_command(args)
        
        assert result == 1

    @patch('warship_extractor.cli.ExtractionPipeline')
    def test_batch_command_max_files(self, mock_pipeline, temp_dir):
        """Test batch command with max files limit."""
        # Create multiple test PDFs
        for i in range(5):
            (temp_dir / f"test{i}.pdf").touch()
        
        mock_pipeline_instance = mock_pipeline.return_value
        mock_pipeline_instance.process_pdf.return_value = {
            'detections': [],
            'statistics': {'total_detections': 0, 'processing_time': 1.0},
            'metadata': {}
        }
        
        args = Mock()
        args.input_dir = temp_dir
        args.pattern = "*.pdf"
        args.output_dir = None
        args.confidence_threshold = 0.3
        args.max_files = 3
        args.parallel = False
        args.generate_reports = False
        
        result = batch_command(args)
        
        assert result == 0
        # Should process only 3 files
        assert mock_pipeline_instance.process_pdf.call_count == 3


class TestInfoCommand:
    """Test cases for info command execution."""

    @patch('warship_extractor.cli.log_system_info')
    def test_info_command_basic(self, mock_log_system):
        """Test basic info command execution."""
        args = Mock()
        args.show_config = False
        args.show_model = False
        
        result = info_command(args)
        
        assert result == 0
        mock_log_system.assert_called_once()

    @patch('warship_extractor.cli.log_system_info')
    @patch('warship_extractor.cli.settings')
    def test_info_command_show_config(self, mock_settings, mock_log_system):
        """Test info command with config display."""
        args = Mock()
        args.show_config = True
        args.show_model = False
        
        result = info_command(args)
        
        assert result == 0

    @patch('warship_extractor.cli.log_system_info')
    @patch('warship_extractor.core.model_manager.ModelManager')
    def test_info_command_show_model(self, mock_model_manager, mock_log_system):
        """Test info command with model information."""
        mock_model_instance = mock_model_manager.return_value
        mock_model_instance.model_name = 'test-model'
        mock_model_instance.device = 'cpu'
        
        args = Mock()
        args.show_config = False
        args.show_model = True
        
        result = info_command(args)
        
        assert result == 0


class TestMainFunction:
    """Test cases for main CLI function."""

    @patch('warship_extractor.cli.setup_logging')
    @patch('warship_extractor.cli.extract_command')
    def test_main_extract_command(self, mock_extract, mock_logging):
        """Test main function with extract command."""
        mock_extract.return_value = 0
        
        with patch('sys.argv', ['warship-extractor', 'extract', 'test.pdf']):
            result = main()
        
        assert result == 0
        mock_extract.assert_called_once()

    @patch('warship_extractor.cli.setup_logging')
    def test_main_no_command(self, mock_logging):
        """Test main function with no command."""
        with patch('sys.argv', ['warship-extractor']):
            result = main()
        
        assert result == 1

    @patch('warship_extractor.cli.setup_logging')
    def test_main_unknown_command(self, mock_logging):
        """Test main function with unknown command."""
        with patch('sys.argv', ['warship-extractor', 'unknown']):
            with pytest.raises(SystemExit):
                main()

    @patch('warship_extractor.cli.setup_logging')
    def test_main_keyboard_interrupt(self, mock_logging):
        """Test main function with keyboard interrupt."""
        with patch('warship_extractor.cli.extract_command', side_effect=KeyboardInterrupt):
            with patch('sys.argv', ['warship-extractor', 'extract', 'test.pdf']):
                result = main()
        
        assert result == 1

    @patch('warship_extractor.cli.setup_logging')
    def test_main_unexpected_error(self, mock_logging):
        """Test main function with unexpected error."""
        with patch('warship_extractor.cli.extract_command', side_effect=Exception("Test error")):
            with patch('sys.argv', ['warship-extractor', 'extract', 'test.pdf']):
                result = main()
        
        assert result == 1

    def test_main_help_output(self):
        """Test main function help output."""
        with patch('sys.argv', ['warship-extractor', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                main()
        
        assert exc_info.value.code == 0

    @patch('warship_extractor.cli.setup_logging')
    def test_main_logging_setup(self, mock_logging):
        """Test that main function sets up logging correctly."""
        with patch('warship_extractor.cli.extract_command', return_value=0):
            with patch('sys.argv', ['warship-extractor', '--log-level', 'DEBUG', 'extract', 'test.pdf']):
                main()
        
        mock_logging.assert_called_once()
        call_args = mock_logging.call_args
        assert call_args.kwargs['log_level'] == 'DEBUG'

    @patch('warship_extractor.cli.setup_logging')
    @patch('warship_extractor.cli.batch_command')
    def test_main_batch_command(self, mock_batch, mock_logging):
        """Test main function with batch command."""
        mock_batch.return_value = 0
        
        with patch('sys.argv', ['warship-extractor', 'batch', '/input']):
            result = main()
        
        assert result == 0
        mock_batch.assert_called_once()

    @pytest.mark.parametrize("command,expected_func", [
        ('extract', 'extract_command'),
        ('batch', 'batch_command'),
        ('info', 'info_command'),
    ])
    @patch('warship_extractor.cli.setup_logging')
    def test_main_command_routing(self, mock_logging, command, expected_func):
        """Test that main function routes commands correctly."""
        with patch(f'warship_extractor.cli.{expected_func}', return_value=0) as mock_func:
            test_args = ['warship-extractor', command]
            if command == 'extract':
                test_args.append('test.pdf')
            elif command == 'batch':
                test_args.append('/input')
            
            with patch('sys.argv', test_args):
                result = main()
        
        assert result == 0
        mock_func.assert_called_once()
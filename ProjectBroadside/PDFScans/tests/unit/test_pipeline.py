"""
Unit tests for the ExtractionPipeline class.

Tests the main orchestration pipeline, error handling, and workflow coordination
without requiring actual model inference or file I/O.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
from PIL import Image

from src.warship_extractor.pipeline.extraction_pipeline import ExtractionPipeline
from src.warship_extractor.config.settings import Settings


class TestExtractionPipeline:
    """Test cases for ExtractionPipeline functionality."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = Mock()
        settings.output_base_dir = Path("/tmp/test_output")
        settings.enable_logging = True
        settings.log_level = "INFO"
        settings.pdf_max_pages = 100
        settings.detection_batch_size = 4
        settings.pipeline_continue_on_error = True
        settings.pipeline_save_intermediate = True
        settings.pipeline_cleanup_temp_files = True
        return settings
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing."""
        components = {
            'model_manager': Mock(),
            'pdf_processor': Mock(),
            'detector': Mock(),
            'nms_filter': Mock(),
            'image_processor': Mock(),
            'logger': Mock()
        }
        
        # Setup default return values
        components['model_manager'].is_loaded.return_value = True
        components['pdf_processor'].convert_pdf_to_images.return_value = [Mock(spec=Image.Image)]
        components['detector'].detect_warships_batch.return_value = [[]]
        components['nms_filter'].filter_detections.return_value = []
        components['image_processor'].process_detection_images.return_value = []
        
        return components
    
    @pytest.fixture
    def pipeline(self, mock_settings, mock_components):
        """Create ExtractionPipeline instance for testing."""
        with patch('src.warship_extractor.pipeline.extraction_pipeline.ModelManager') as mock_mm, \
             patch('src.warship_extractor.pipeline.extraction_pipeline.PDFProcessor') as mock_pp, \
             patch('src.warship_extractor.pipeline.extraction_pipeline.WarshipDetector') as mock_wd, \
             patch('src.warship_extractor.pipeline.extraction_pipeline.NMSFilter') as mock_nf, \
             patch('src.warship_extractor.pipeline.extraction_pipeline.ImageProcessor') as mock_ip, \
             patch('src.warship_extractor.pipeline.extraction_pipeline.WarshipLogger') as mock_wl:
            
            # Setup mocks to return our test components
            mock_mm.return_value = mock_components['model_manager']
            mock_pp.return_value = mock_components['pdf_processor']
            mock_wd.return_value = mock_components['detector']
            mock_nf.return_value = mock_components['nms_filter']
            mock_ip.return_value = mock_components['image_processor']
            mock_wl.return_value = mock_components['logger']
            
            return ExtractionPipeline(mock_settings)
    
    def test_initialization(self, mock_settings, mock_components):
        """Test ExtractionPipeline initialization."""
        with patch('src.warship_extractor.pipeline.extraction_pipeline.ModelManager') as mock_mm, \
             patch('src.warship_extractor.pipeline.extraction_pipeline.PDFProcessor') as mock_pp, \
             patch('src.warship_extractor.pipeline.extraction_pipeline.WarshipDetector') as mock_wd, \
             patch('src.warship_extractor.pipeline.extraction_pipeline.NMSFilter') as mock_nf, \
             patch('src.warship_extractor.pipeline.extraction_pipeline.ImageProcessor') as mock_ip, \
             patch('src.warship_extractor.pipeline.extraction_pipeline.WarshipLogger') as mock_wl:
            
            pipeline = ExtractionPipeline(mock_settings)
            
            assert pipeline.settings == mock_settings
            # Verify components are initialized
            mock_mm.assert_called_once_with(mock_settings)
            mock_pp.assert_called_once_with(mock_settings)
            mock_nf.assert_called_once_with(mock_settings)
            mock_ip.assert_called_once_with(mock_settings)
            mock_wl.assert_called_once_with(mock_settings)
    
    def test_extract_from_pdf_success(self, pipeline, mock_components):
        """Test successful PDF extraction workflow."""
        pdf_path = Path("test.pdf")
        output_dir = Path("output")
        
        # Setup mock images and detections
        mock_images = [Mock(spec=Image.Image) for _ in range(2)]
        mock_detections = [
            [{'bbox': [100, 100, 200, 200], 'label': 'warship', 'confidence': 0.8}],
            [{'bbox': [300, 150, 400, 250], 'label': 'destroyer', 'confidence': 0.7}]
        ]
        mock_processed = [
            [{'detection': mock_detections[0][0], 'output_path': 'path1'}],
            [{'detection': mock_detections[1][0], 'output_path': 'path2'}]
        ]
        
        # Setup component returns
        mock_components['pdf_processor'].convert_pdf_to_images.return_value = mock_images
        mock_components['detector'].detect_warships_batch.return_value = mock_detections
        mock_components['nms_filter'].filter_detections.side_effect = lambda x: x  # Pass through
        mock_components['image_processor'].process_detection_images.side_effect = [
            mock_processed[0], mock_processed[1]
        ]
        
        result = pipeline.extract_from_pdf(pdf_path, output_dir)
        
        # Verify workflow
        mock_components['pdf_processor'].convert_pdf_to_images.assert_called_once_with(pdf_path)
        mock_components['detector'].detect_warships_batch.assert_called_once_with(mock_images)
        assert mock_components['nms_filter'].filter_detections.call_count == 2
        assert mock_components['image_processor'].process_detection_images.call_count == 2
        
        # Verify result structure
        assert 'pdf_path' in result
        assert 'output_dir' in result
        assert 'total_pages' in result
        assert 'total_detections' in result
        assert 'extracted_images' in result
        assert 'processing_time' in result
        
        assert result['total_pages'] == 2
        assert result['total_detections'] == 2
    
    def test_extract_from_pdf_no_detections(self, pipeline, mock_components):
        """Test PDF extraction when no detections are found."""
        pdf_path = Path("empty.pdf")
        output_dir = Path("output")
        
        # Setup mock with no detections
        mock_images = [Mock(spec=Image.Image)]
        mock_components['pdf_processor'].convert_pdf_to_images.return_value = mock_images
        mock_components['detector'].detect_warships_batch.return_value = [[]]  # No detections
        
        result = pipeline.extract_from_pdf(pdf_path, output_dir)
        
        assert result['total_detections'] == 0
        assert result['extracted_images'] == []
    
    def test_extract_from_pdf_file_not_found(self, pipeline, mock_components):
        """Test handling of non-existent PDF file."""
        pdf_path = Path("nonexistent.pdf")
        output_dir = Path("output")
        
        mock_components['pdf_processor'].convert_pdf_to_images.side_effect = FileNotFoundError("File not found")
        
        with pytest.raises(FileNotFoundError):
            pipeline.extract_from_pdf(pdf_path, output_dir)
    
    def test_extract_from_pdf_with_error_recovery(self, pipeline, mock_components):
        """Test error recovery during PDF extraction."""
        pdf_path = Path("test.pdf")
        output_dir = Path("output")
        
        # Setup partial failure scenario
        mock_images = [Mock(spec=Image.Image) for _ in range(3)]
        mock_detections = [
            [{'bbox': [100, 100, 200, 200], 'label': 'warship', 'confidence': 0.8}],
            [],  # Second page fails
            [{'bbox': [300, 150, 400, 250], 'label': 'destroyer', 'confidence': 0.7}]
        ]
        
        mock_components['pdf_processor'].convert_pdf_to_images.return_value = mock_images
        mock_components['detector'].detect_warships_batch.return_value = mock_detections
        mock_components['nms_filter'].filter_detections.side_effect = lambda x: x
        mock_components['image_processor'].process_detection_images.side_effect = [
            [{'detection': mock_detections[0][0], 'output_path': 'path1'}],
            RuntimeError("Processing failed"),  # Simulate error
            [{'detection': mock_detections[2][0], 'output_path': 'path3'}]
        ]
        
        # Should continue processing despite error
        result = pipeline.extract_from_pdf(pdf_path, output_dir)
        
        # Should have processed 2 out of 3 pages successfully
        assert result['total_detections'] == 2
        mock_components['logger'].error.assert_called()  # Should log the error
    
    def test_extract_single_image_success(self, pipeline, mock_components):
        """Test successful single image extraction."""
        image_path = Path("test.jpg")
        output_dir = Path("output")
        
        mock_image = Mock(spec=Image.Image)
        mock_detections = [
            {'bbox': [100, 100, 200, 200], 'label': 'warship', 'confidence': 0.8}
        ]
        mock_processed = [
            {'detection': mock_detections[0], 'output_path': 'path1'}
        ]
        
        with patch('PIL.Image.open', return_value=mock_image):
            mock_components['detector'].detect_warships.return_value = mock_detections
            mock_components['nms_filter'].filter_detections.return_value = mock_detections
            mock_components['image_processor'].process_detection_images.return_value = mock_processed
            
            result = pipeline.extract_from_image(image_path, output_dir)
            
            assert result['image_path'] == str(image_path)
            assert result['total_detections'] == 1
            assert result['extracted_images'] == mock_processed
    
    def test_extract_single_image_file_not_found(self, pipeline, mock_components):
        """Test handling of non-existent image file."""
        image_path = Path("nonexistent.jpg")
        output_dir = Path("output")
        
        with patch('PIL.Image.open', side_effect=FileNotFoundError("File not found")):
            with pytest.raises(FileNotFoundError):
                pipeline.extract_from_image(image_path, output_dir)
    
    def test_batch_extract_from_pdfs(self, pipeline, mock_components):
        """Test batch extraction from multiple PDFs."""
        pdf_paths = [Path("test1.pdf"), Path("test2.pdf")]
        output_dir = Path("batch_output")
        
        # Mock successful extraction for each PDF
        with patch.object(pipeline, 'extract_from_pdf') as mock_extract:
            mock_extract.side_effect = [
                {'total_detections': 5, 'total_pages': 10},
                {'total_detections': 3, 'total_pages': 8}
            ]
            
            results = pipeline.batch_extract_from_pdfs(pdf_paths, output_dir)
            
            assert len(results) == 2
            assert mock_extract.call_count == 2
            
            # Verify each PDF was processed
            mock_extract.assert_any_call(pdf_paths[0], output_dir / "test1")
            mock_extract.assert_any_call(pdf_paths[1], output_dir / "test2")
    
    def test_batch_extract_with_partial_failure(self, pipeline, mock_components):
        """Test batch extraction with some failures."""
        pdf_paths = [Path("test1.pdf"), Path("bad.pdf"), Path("test3.pdf")]
        output_dir = Path("batch_output")
        
        with patch.object(pipeline, 'extract_from_pdf') as mock_extract:
            mock_extract.side_effect = [
                {'total_detections': 5},  # Success
                RuntimeError("Processing failed"),  # Failure
                {'total_detections': 3}   # Success
            ]
            
            results = pipeline.batch_extract_from_pdfs(pdf_paths, output_dir)
            
            # Should have 2 successful results, 1 error
            successful_results = [r for r in results if not r.get('error')]
            error_results = [r for r in results if r.get('error')]
            
            assert len(successful_results) == 2
            assert len(error_results) == 1
            assert mock_extract.call_count == 3
    
    def test_pipeline_statistics(self, pipeline, mock_components):
        """Test pipeline statistics generation."""
        # Setup pipeline state
        pipeline._stats = {
            'pdfs_processed': 5,
            'pages_processed': 50,
            'images_processed': 45,
            'total_detections': 120,
            'successful_extractions': 115,
            'failed_extractions': 5,
            'processing_time': 300.5
        }
        
        stats = pipeline.get_pipeline_statistics()
        
        expected_keys = [
            'pdfs_processed', 'pages_processed', 'images_processed',
            'total_detections', 'successful_extractions', 'failed_extractions',
            'processing_time', 'success_rate', 'detections_per_page',
            'processing_speed'
        ]
        
        for key in expected_keys:
            assert key in stats
        
        assert stats['success_rate'] == 115 / 120  # successful/total
        assert stats['detections_per_page'] == 120 / 50  # detections/pages
    
    def test_cleanup_resources(self, pipeline, mock_components):
        """Test resource cleanup."""
        # Setup some temporary resources
        pipeline._temp_files = [Path("temp1.png"), Path("temp2.png")]
        pipeline._temp_dirs = [Path("temp_dir1"), Path("temp_dir2")]
        
        with patch('pathlib.Path.unlink') as mock_unlink, \
             patch('pathlib.Path.rmdir') as mock_rmdir, \
             patch('pathlib.Path.exists', return_value=True):
            
            pipeline.cleanup_resources()
            
            # Should clean up temp files and directories
            assert mock_unlink.call_count == 2
            assert mock_rmdir.call_count == 2
            mock_components['model_manager'].unload_model.assert_called_once()
    
    def test_validate_inputs_valid_pdf(self, pipeline):
        """Test input validation for valid PDF."""
        pdf_path = Path("test.pdf")
        output_dir = Path("output")
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.suffix', new_property=lambda self: '.pdf'):
            
            # Should not raise exception
            pipeline._validate_inputs(pdf_path, output_dir)
    
    def test_validate_inputs_invalid_file_type(self, pipeline):
        """Test input validation for invalid file type."""
        invalid_path = Path("test.txt")
        output_dir = Path("output")
        
        with patch('pathlib.Path.exists', return_value=True):
            with pytest.raises(ValueError, match="Unsupported file format"):
                pipeline._validate_inputs(invalid_path, output_dir)
    
    def test_validate_inputs_file_not_exists(self, pipeline):
        """Test input validation for non-existent file."""
        pdf_path = Path("nonexistent.pdf")
        output_dir = Path("output")
        
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(FileNotFoundError, match="Input file does not exist"):
                pipeline._validate_inputs(pdf_path, output_dir)
    
    def test_setup_output_directory(self, pipeline):
        """Test output directory setup."""
        output_dir = Path("test_output")
        
        with patch.object(Path, 'mkdir') as mock_mkdir:
            pipeline._setup_output_directory(output_dir)
            
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    
    def test_context_manager(self, pipeline, mock_components):
        """Test pipeline as context manager."""
        with patch.object(pipeline, 'cleanup_resources') as mock_cleanup:
            with pipeline:
                pass  # Do some work
            
            mock_cleanup.assert_called_once()
    
    def test_context_manager_with_exception(self, pipeline, mock_components):
        """Test context manager with exception."""
        with patch.object(pipeline, 'cleanup_resources') as mock_cleanup:
            with pytest.raises(ValueError):
                with pipeline:
                    raise ValueError("Test exception")
            
            mock_cleanup.assert_called_once()
    
    def test_progress_tracking(self, pipeline, mock_components):
        """Test progress tracking during extraction."""
        pdf_path = Path("test.pdf")
        output_dir = Path("output")
        
        mock_images = [Mock(spec=Image.Image) for _ in range(5)]
        mock_components['pdf_processor'].convert_pdf_to_images.return_value = mock_images
        mock_components['detector'].detect_warships_batch.return_value = [[] for _ in range(5)]
        
        # Mock progress callback
        progress_callback = Mock()
        
        with patch.object(pipeline, '_update_progress') as mock_update:
            pipeline.extract_from_pdf(pdf_path, output_dir, progress_callback=progress_callback)
            
            # Should update progress during processing
            assert mock_update.call_count > 0
    
    def test_memory_optimization(self, pipeline, mock_components):
        """Test memory optimization during processing."""
        pdf_path = Path("large.pdf")
        output_dir = Path("output")
        
        # Simulate large PDF with many pages
        mock_images = [Mock(spec=Image.Image) for _ in range(100)]
        mock_components['pdf_processor'].convert_pdf_to_images.return_value = mock_images
        mock_components['detector'].detect_warships_batch.return_value = [[] for _ in range(100)]
        
        with patch.object(pipeline, '_optimize_memory') as mock_optimize:
            pipeline.extract_from_pdf(pdf_path, output_dir)
            
            # Should call memory optimization
            mock_optimize.assert_called()
    
    def test_error_reporting(self, pipeline, mock_components):
        """Test comprehensive error reporting."""
        pdf_path = Path("test.pdf")
        output_dir = Path("output")
        
        # Simulate various errors
        mock_components['pdf_processor'].convert_pdf_to_images.side_effect = RuntimeError("PDF error")
        
        with pytest.raises(RuntimeError):
            pipeline.extract_from_pdf(pdf_path, output_dir)
        
        # Should log detailed error information
        mock_components['logger'].error.assert_called()
        
        # Should include error context
        error_calls = mock_components['logger'].error.call_args_list
        assert any("PDF error" in str(call) for call in error_calls)
    
    def test_configuration_validation(self, mock_settings):
        """Test pipeline configuration validation."""
        # Test with invalid settings
        mock_settings.detection_batch_size = 0  # Invalid
        
        with pytest.raises(ValueError, match="Invalid configuration"):
            with patch('src.warship_extractor.pipeline.extraction_pipeline.ModelManager'), \
                 patch('src.warship_extractor.pipeline.extraction_pipeline.PDFProcessor'), \
                 patch('src.warship_extractor.pipeline.extraction_pipeline.WarshipDetector'), \
                 patch('src.warship_extractor.pipeline.extraction_pipeline.NMSFilter'), \
                 patch('src.warship_extractor.pipeline.extraction_pipeline.ImageProcessor'), \
                 patch('src.warship_extractor.pipeline.extraction_pipeline.WarshipLogger'):
                ExtractionPipeline(mock_settings)
    
    def test_get_supported_formats(self, pipeline):
        """Test getting supported file formats."""
        formats = pipeline.get_supported_formats()
        
        expected_formats = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        for fmt in expected_formats:
            assert fmt in formats
    
    def test_pipeline_info(self, pipeline):
        """Test pipeline information retrieval."""
        info = pipeline.get_pipeline_info()
        
        expected_keys = [
            'version', 'components', 'configuration', 'statistics',
            'supported_formats', 'memory_usage'
        ]
        
        for key in expected_keys:
            assert key in info
        
        assert 'model_manager' in info['components']
        assert 'pdf_processor' in info['components']
        assert 'detector' in info['components']
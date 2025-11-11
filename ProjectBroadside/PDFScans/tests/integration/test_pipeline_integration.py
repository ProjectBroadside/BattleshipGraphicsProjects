"""
Integration tests for the full extraction pipeline.

Tests the complete workflow from PDF input to final results,
including all component interactions and data flow.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import numpy as np
from PIL import Image

from warship_extractor.pipeline.extraction_pipeline import ExtractionPipeline
from warship_extractor.config.settings import Settings


class TestPipelineIntegration:
    """Integration tests for the complete extraction pipeline."""

    @pytest.fixture
    def mock_components(self):
        """Create mocked components for integration testing."""
        with patch('warship_extractor.core.model_manager.ModelManager') as mock_model_manager, \
             patch('warship_extractor.core.pdf_processor.PDFProcessor') as mock_pdf_processor, \
             patch('warship_extractor.detection.detector.WarshipDetector') as mock_detector, \
             patch('warship_extractor.processing.nms_filter.NMSFilter') as mock_nms_filter, \
             patch('warship_extractor.processing.image_processor.ImageProcessor') as mock_image_processor:
            
            # Configure model manager mock
            mock_model_manager_instance = mock_model_manager.return_value
            mock_model_manager_instance.load_model.return_value = None
            mock_model_manager_instance.is_model_loaded.return_value = True
            
            # Configure PDF processor mock
            mock_pdf_processor_instance = mock_pdf_processor.return_value
            mock_pdf_processor_instance.get_page_count.return_value = 3
            mock_pdf_processor_instance.convert_to_images.return_value = [
                Image.new('RGB', (1200, 800), 'white'),
                Image.new('RGB', (1200, 800), 'white'),
                Image.new('RGB', (1200, 800), 'white')
            ]
            mock_pdf_processor_instance.get_metadata.return_value = {
                'title': 'Test PDF',
                'page_count': 3
            }
            
            # Configure detector mock
            mock_detector_instance = mock_detector.return_value
            mock_detector_instance.detect_warships.return_value = [
                {
                    'bbox': [100, 150, 300, 250],
                    'label': 'warship',
                    'confidence': 0.85,
                    'prompt_used': 'Detect warships'
                },
                {
                    'bbox': [400, 200, 600, 350],
                    'label': 'ship',
                    'confidence': 0.72,
                    'prompt_used': 'Find naval vessels'
                }
            ]
            
            # Configure NMS filter mock
            mock_nms_filter_instance = mock_nms_filter.return_value
            mock_nms_filter_instance.filter_detections.side_effect = lambda x: x  # Pass through
            mock_nms_filter_instance.get_statistics.return_value = {
                'total_input': 2,
                'total_output': 2,
                'removed_count': 0,
                'removal_rate': 0.0
            }
            
            # Configure image processor mock
            mock_image_processor_instance = mock_image_processor.return_value
            mock_image_processor_instance.crop_detections.return_value = [
                {
                    'bbox': [100, 150, 300, 250],
                    'label': 'warship',
                    'confidence': 0.85,
                    'cropped_image': Image.new('RGB', (200, 100), 'blue')
                },
                {
                    'bbox': [400, 200, 600, 350],
                    'label': 'ship',
                    'confidence': 0.72,
                    'cropped_image': Image.new('RGB', (200, 150), 'green')
                }
            ]
            mock_image_processor_instance.enhance_images.side_effect = lambda x: x  # Pass through
            
            yield {
                'model_manager': mock_model_manager_instance,
                'pdf_processor': mock_pdf_processor_instance,
                'detector': mock_detector_instance,
                'nms_filter': mock_nms_filter_instance,
                'image_processor': mock_image_processor_instance
            }

    def test_pipeline_initialization(self):
        """Test pipeline initialization with default settings."""
        pipeline = ExtractionPipeline()
        
        assert pipeline is not None
        assert hasattr(pipeline, 'settings')
        assert hasattr(pipeline, 'logger')

    def test_pipeline_initialization_custom_settings(self):
        """Test pipeline initialization with custom settings."""
        custom_settings = {
            'confidence_threshold': 0.5,
            'pdf_dpi': 200,
            'enable_nms': False
        }
        
        pipeline = ExtractionPipeline(**custom_settings)
        
        assert pipeline is not None

    def test_process_pdf_full_workflow(self, mock_components, temp_dir, test_pdf):
        """Test complete PDF processing workflow."""
        # Create output directory
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        pipeline = ExtractionPipeline()
        
        results = pipeline.process_pdf(
            pdf_path=test_pdf,
            output_dir=output_dir
        )
        
        # Verify results structure
        assert 'detections' in results
        assert 'statistics' in results
        assert 'metadata' in results
        
        # Verify detections
        detections = results['detections']
        assert len(detections) == 2
        assert detections[0]['label'] == 'warship'
        assert detections[1]['label'] == 'ship'
        
        # Verify statistics
        stats = results['statistics']
        assert 'total_detections' in stats
        assert 'pages_processed' in stats
        assert 'processing_time' in stats
        
        # Verify metadata
        metadata = results['metadata']
        assert 'pdf_path' in metadata
        assert 'extraction_date' in metadata

    def test_process_pdf_with_page_limit(self, mock_components, temp_dir, test_pdf):
        """Test PDF processing with page limit."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        pipeline = ExtractionPipeline(max_pages=2)
        
        results = pipeline.process_pdf(
            pdf_path=test_pdf,
            output_dir=output_dir
        )
        
        # Should process only 2 pages
        assert results['statistics']['pages_processed'] <= 2

    def test_process_pdf_specific_pages(self, mock_components, temp_dir, test_pdf):
        """Test PDF processing with specific page numbers."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        pipeline = ExtractionPipeline()
        
        results = pipeline.process_pdf(
            pdf_path=test_pdf,
            output_dir=output_dir,
            pages=[1, 3]  # Process only pages 1 and 3
        )
        
        # Verify that only specified pages were processed
        assert results['statistics']['pages_processed'] == 2

    def test_process_pdf_confidence_filtering(self, mock_components, temp_dir, test_pdf):
        """Test PDF processing with confidence threshold filtering."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # Set high confidence threshold
        pipeline = ExtractionPipeline(confidence_threshold=0.8)
        
        results = pipeline.process_pdf(
            pdf_path=test_pdf,
            output_dir=output_dir
        )
        
        # Should filter out lower confidence detections
        detections = results['detections']
        for detection in detections:
            assert detection['confidence'] >= 0.8

    def test_process_pdf_with_nms_disabled(self, mock_components, temp_dir, test_pdf):
        """Test PDF processing with NMS disabled."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        pipeline = ExtractionPipeline(enable_nms=False)
        
        results = pipeline.process_pdf(
            pdf_path=test_pdf,
            output_dir=output_dir
        )
        
        # NMS filter should not be called
        mock_components['nms_filter'].filter_detections.assert_not_called()

    def test_process_pdf_error_handling(self, temp_dir):
        """Test PDF processing error handling."""
        pipeline = ExtractionPipeline()
        
        # Test with non-existent PDF
        non_existent_pdf = temp_dir / "non_existent.pdf"
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        with pytest.raises(FileNotFoundError):
            pipeline.process_pdf(
                pdf_path=non_existent_pdf,
                output_dir=output_dir
            )

    def test_process_pdf_output_files_created(self, mock_components, temp_dir, test_pdf):
        """Test that output files are created during processing."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        pipeline = ExtractionPipeline(save_debug_images=True)
        
        results = pipeline.process_pdf(
            pdf_path=test_pdf,
            output_dir=output_dir
        )
        
        # Check that results JSON is saved
        results_file = output_dir / "extraction_results.json"
        assert results_file.exists()

    def test_process_multiple_pdfs(self, mock_components, temp_dir):
        """Test processing multiple PDFs."""
        output_dir = temp_dir / "batch_output"
        output_dir.mkdir()
        
        # Create multiple test PDFs
        pdf1 = temp_dir / "test1.pdf"
        pdf2 = temp_dir / "test2.pdf"
        pdf1.touch()
        pdf2.touch()
        
        pipeline = ExtractionPipeline()
        
        all_results = []
        for pdf_path in [pdf1, pdf2]:
            pdf_output_dir = output_dir / pdf_path.stem
            pdf_output_dir.mkdir()
            
            results = pipeline.process_pdf(
                pdf_path=pdf_path,
                output_dir=pdf_output_dir
            )
            all_results.append(results)
        
        assert len(all_results) == 2
        for results in all_results:
            assert 'detections' in results
            assert 'statistics' in results

    def test_memory_management_during_processing(self, mock_components, temp_dir, test_pdf):
        """Test memory management during large processing tasks."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        pipeline = ExtractionPipeline()
        
        # Mock memory monitoring
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 75  # 75% memory usage
            
            results = pipeline.process_pdf(
                pdf_path=test_pdf,
                output_dir=output_dir
            )
            
            # Should complete successfully even with high memory usage
            assert results is not None

    def test_pipeline_statistics_accuracy(self, mock_components, temp_dir, test_pdf):
        """Test accuracy of pipeline statistics."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        pipeline = ExtractionPipeline()
        
        results = pipeline.process_pdf(
            pdf_path=test_pdf,
            output_dir=output_dir
        )
        
        stats = results['statistics']
        detections = results['detections']
        
        # Verify statistics match actual results
        assert stats['total_detections'] == len(detections)
        
        if detections:
            avg_confidence = sum(d['confidence'] for d in detections) / len(detections)
            assert abs(stats['avg_confidence'] - avg_confidence) < 0.01

    def test_component_interaction_order(self, temp_dir, test_pdf):
        """Test that components are called in the correct order."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        call_order = []
        
        with patch('warship_extractor.core.pdf_processor.PDFProcessor') as mock_pdf, \
             patch('warship_extractor.detection.detector.WarshipDetector') as mock_detector, \
             patch('warship_extractor.processing.nms_filter.NMSFilter') as mock_nms, \
             patch('warship_extractor.processing.image_processor.ImageProcessor') as mock_img:
            
            # Track call order
            def track_call(component_name):
                def wrapper(*args, **kwargs):
                    call_order.append(component_name)
                    return MagicMock()
                return wrapper
            
            mock_pdf.return_value.convert_to_images.side_effect = track_call('pdf_convert')
            mock_detector.return_value.detect_warships.side_effect = track_call('detect')
            mock_nms.return_value.filter_detections.side_effect = track_call('nms_filter')
            mock_img.return_value.crop_detections.side_effect = track_call('crop')
            
            # Configure mocks to return appropriate values
            mock_pdf.return_value.get_page_count.return_value = 1
            mock_pdf.return_value.convert_to_images.return_value = [Image.new('RGB', (800, 600))]
            mock_pdf.return_value.get_metadata.return_value = {}
            mock_detector.return_value.detect_warships.return_value = []
            mock_nms.return_value.filter_detections.return_value = []
            mock_img.return_value.crop_detections.return_value = []
            
            pipeline = ExtractionPipeline()
            pipeline.process_pdf(pdf_path=test_pdf, output_dir=output_dir)
            
            # Verify correct order: PDF -> Detect -> NMS -> Crop
            expected_order = ['pdf_convert', 'detect', 'nms_filter', 'crop']
            assert call_order == expected_order

    def test_error_recovery_and_logging(self, mock_components, temp_dir, test_pdf):
        """Test error recovery and logging during processing."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # Make one component fail
        mock_components['detector'].detect_warships.side_effect = Exception("Detection failed")
        
        pipeline = ExtractionPipeline()
        
        # Should handle the error gracefully
        with pytest.raises(Exception):
            pipeline.process_pdf(
                pdf_path=test_pdf,
                output_dir=output_dir
            )

    def test_progress_tracking(self, mock_components, temp_dir, test_pdf):
        """Test progress tracking during processing."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        progress_updates = []
        
        def mock_progress_callback(current, total, message):
            progress_updates.append((current, total, message))
        
        pipeline = ExtractionPipeline()
        
        # Mock progress tracking
        with patch.object(pipeline, '_update_progress', side_effect=mock_progress_callback):
            results = pipeline.process_pdf(
                pdf_path=test_pdf,
                output_dir=output_dir
            )
            
            # Should have received progress updates
            assert len(progress_updates) > 0

    def test_concurrent_processing_safety(self, mock_components, temp_dir):
        """Test that pipeline is safe for concurrent processing."""
        import threading
        import time
        
        results = []
        errors = []
        
        def process_pdf(pdf_name):
            try:
                pdf_path = temp_dir / f"{pdf_name}.pdf"
                pdf_path.touch()
                
                output_dir = temp_dir / f"output_{pdf_name}"
                output_dir.mkdir()
                
                pipeline = ExtractionPipeline()
                result = pipeline.process_pdf(
                    pdf_path=pdf_path,
                    output_dir=output_dir
                )
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=process_pdf, args=[f"test_{i}"])
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 3

    def test_large_pdf_processing(self, mock_components, temp_dir, test_pdf):
        """Test processing of large PDFs with many pages."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # Mock large PDF
        mock_components['pdf_processor'].get_page_count.return_value = 100
        mock_components['pdf_processor'].convert_to_images.return_value = [
            Image.new('RGB', (1200, 800)) for _ in range(100)
        ]
        
        pipeline = ExtractionPipeline(max_pages=10)  # Limit for testing
        
        results = pipeline.process_pdf(
            pdf_path=test_pdf,
            output_dir=output_dir
        )
        
        # Should handle large PDF efficiently
        assert results['statistics']['pages_processed'] <= 10

    def test_settings_propagation(self, mock_components, temp_dir, test_pdf):
        """Test that settings are properly propagated to all components."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        custom_settings = {
            'confidence_threshold': 0.7,
            'pdf_dpi': 200,
            'batch_size': 2
        }
        
        pipeline = ExtractionPipeline(**custom_settings)
        
        results = pipeline.process_pdf(
            pdf_path=test_pdf,
            output_dir=output_dir
        )
        
        # Verify settings were used
        assert results['metadata']['settings']['confidence_threshold'] == 0.7
        assert results['metadata']['settings']['pdf_dpi'] == 200
"""
Unit tests for the visualization utilities.

Tests visualization functionality for detection results, reports, and analysis
without requiring actual image processing or external dependencies.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
from PIL import Image
import numpy as np

from src.warship_extractor.utils.visualization import VisualizationUtils
from src.warship_extractor.config.settings import Settings


class TestVisualizationUtils:
    """Test cases for VisualizationUtils functionality."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = Mock()
        settings.output_dir = Path("test_output")
        settings.visualization_dpi = 150
        settings.bbox_line_width = 2
        settings.bbox_colors = ["red", "blue", "green", "orange"]
        settings.font_size = 12
        settings.save_annotated_images = True
        settings.save_detection_report = True
        return settings
    
    @pytest.fixture
    def visualization_utils(self, mock_settings):
        """Create VisualizationUtils instance for testing."""
        return VisualizationUtils(mock_settings)
    
    @pytest.fixture
    def mock_detection_results(self):
        """Create mock detection results for testing."""
        return [
            {
                'bbox': [100, 150, 300, 250],  # x1, y1, x2, y2
                'confidence': 0.95,
                'label': 'battleship',
                'prompt_used': 'Locate warships and naval vessels in this historical document'
            },
            {
                'bbox': [400, 200, 600, 350],
                'confidence': 0.87,
                'label': 'cruiser',
                'prompt_used': 'Find detailed ship schematics and technical drawings'
            },
            {
                'bbox': [50, 300, 200, 450],
                'confidence': 0.92,
                'label': 'destroyer',
                'prompt_used': 'Identify naval vessel illustrations'
            }
        ]
    
    @pytest.fixture
    def mock_image(self):
        """Create mock PIL Image for testing."""
        mock_img = Mock(spec=Image.Image)
        mock_img.size = (800, 600)
        mock_img.mode = 'RGB'
        return mock_img
    
    def test_initialization(self, mock_settings):
        """Test VisualizationUtils initialization."""
        viz = VisualizationUtils(mock_settings)
        
        assert viz.settings == mock_settings
        assert viz.colors == mock_settings.bbox_colors
    
    def test_draw_bounding_boxes(self, visualization_utils, mock_image, mock_detection_results):
        """Test drawing bounding boxes on image."""
        with patch('PIL.ImageDraw.Draw') as mock_draw_class:
            mock_draw = Mock()
            mock_draw_class.return_value = mock_draw
            
            result_image = visualization_utils.draw_bounding_boxes(
                mock_image, mock_detection_results
            )
            
            # Should create ImageDraw instance
            mock_draw_class.assert_called_once_with(mock_image)
            
            # Should draw rectangles for each detection
            assert mock_draw.rectangle.call_count == len(mock_detection_results)
            
            # Should return the original image
            assert result_image == mock_image
    
    def test_draw_bounding_boxes_with_labels(self, visualization_utils, mock_image, mock_detection_results):
        """Test drawing bounding boxes with labels."""
        with patch('PIL.ImageDraw.Draw') as mock_draw_class, \
             patch('PIL.ImageFont.truetype') as mock_font:
            
            mock_draw = Mock()
            mock_draw_class.return_value = mock_draw
            mock_font.return_value = Mock()
            
            result_image = visualization_utils.draw_bounding_boxes(
                mock_image, mock_detection_results, show_labels=True
            )
            
            # Should draw text labels
            assert mock_draw.text.call_count == len(mock_detection_results)
            
            # Should use font
            mock_font.assert_called()
    
    def test_draw_bounding_boxes_with_confidence(self, visualization_utils, mock_image, mock_detection_results):
        """Test drawing bounding boxes with confidence scores."""
        with patch('PIL.ImageDraw.Draw') as mock_draw_class:
            mock_draw = Mock()
            mock_draw_class.return_value = mock_draw
            
            visualization_utils.draw_bounding_boxes(
                mock_image, mock_detection_results, show_confidence=True
            )
            
            # Should include confidence in text labels
            text_calls = mock_draw.text.call_args_list
            assert len(text_calls) == len(mock_detection_results)
            
            # Check that confidence scores are included in text
            for i, call in enumerate(text_calls):
                text_content = call[0][1]  # Second argument is the text
                expected_confidence = mock_detection_results[i]['confidence']
                assert f"{expected_confidence:.2f}" in text_content
    
    def test_color_cycling(self, visualization_utils, mock_image):
        """Test color cycling for multiple detections."""
        # Create more detections than colors to test cycling
        many_detections = [
            {'bbox': [i*50, i*50, i*50+100, i*50+100], 'confidence': 0.9, 'label': f'ship_{i}'}
            for i in range(6)  # More than the 4 colors in settings
        ]
        
        with patch('PIL.ImageDraw.Draw') as mock_draw_class:
            mock_draw = Mock()
            mock_draw_class.return_value = mock_draw
            
            visualization_utils.draw_bounding_boxes(mock_image, many_detections)
            
            # Should cycle through colors
            rectangle_calls = mock_draw.rectangle.call_args_list
            assert len(rectangle_calls) == 6
            
            # Colors should cycle (red, blue, green, orange, red, blue)
            expected_colors = ["red", "blue", "green", "orange", "red", "blue"]
            for i, call in enumerate(rectangle_calls):
                outline_color = call[1]['outline']  # outline parameter
                assert outline_color == expected_colors[i]
    
    def test_create_detection_report(self, visualization_utils, mock_detection_results):
        """Test creating detection report."""
        pdf_path = "test.pdf"
        page_number = 5
        processing_time = 12.5
        
        report = visualization_utils.create_detection_report(
            pdf_path, page_number, mock_detection_results, processing_time
        )
        
        # Should include all required fields
        expected_keys = [
            'pdf_path', 'page_number', 'total_detections', 'processing_time',
            'detections', 'summary', 'timestamp'
        ]
        
        for key in expected_keys:
            assert key in report
        
        # Check values
        assert report['pdf_path'] == pdf_path
        assert report['page_number'] == page_number
        assert report['total_detections'] == len(mock_detection_results)
        assert report['processing_time'] == processing_time
        assert len(report['detections']) == len(mock_detection_results)
    
    def test_detection_report_summary_statistics(self, visualization_utils, mock_detection_results):
        """Test detection report summary statistics."""
        report = visualization_utils.create_detection_report(
            "test.pdf", 1, mock_detection_results, 10.0
        )
        
        summary = report['summary']
        
        # Should include statistics
        expected_summary_keys = [
            'avg_confidence', 'max_confidence', 'min_confidence',
            'labels_found', 'confidence_distribution'
        ]
        
        for key in expected_summary_keys:
            assert key in summary
        
        # Check calculations
        confidences = [det['confidence'] for det in mock_detection_results]
        assert summary['avg_confidence'] == sum(confidences) / len(confidences)
        assert summary['max_confidence'] == max(confidences)
        assert summary['min_confidence'] == min(confidences)
        
        # Should count unique labels
        unique_labels = set(det['label'] for det in mock_detection_results)
        assert set(summary['labels_found']) == unique_labels
    
    def test_save_annotated_image(self, visualization_utils, mock_image, mock_detection_results):
        """Test saving annotated image."""
        output_path = Path("annotated_output.jpg")
        
        with patch.object(visualization_utils, 'draw_bounding_boxes') as mock_draw, \
             patch.object(mock_image, 'save') as mock_save:
            
            mock_draw.return_value = mock_image
            
            result_path = visualization_utils.save_annotated_image(
                mock_image, mock_detection_results, output_path
            )
            
            # Should draw bounding boxes
            mock_draw.assert_called_once_with(
                mock_image, mock_detection_results, 
                show_labels=True, show_confidence=True
            )
            
            # Should save image
            mock_save.assert_called_once_with(output_path, dpi=(150, 150))
            
            # Should return path
            assert result_path == output_path
    
    def test_save_detection_report_json(self, visualization_utils, mock_detection_results):
        """Test saving detection report as JSON."""
        output_path = Path("report.json")
        
        with patch('builtins.open', create=True) as mock_open, \
             patch('json.dump') as mock_json_dump:
            
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            result_path = visualization_utils.save_detection_report(
                "test.pdf", 1, mock_detection_results, 10.0, output_path
            )
            
            # Should open file for writing
            mock_open.assert_called_once_with(output_path, 'w')
            
            # Should write JSON data
            mock_json_dump.assert_called_once()
            report_data = mock_json_dump.call_args[0][0]
            
            # Verify report structure
            assert 'pdf_path' in report_data
            assert 'detections' in report_data
            assert 'summary' in report_data
            
            assert result_path == output_path
    
    def test_create_batch_report(self, visualization_utils):
        """Test creating batch processing report."""
        batch_results = [
            {
                'pdf_path': 'doc1.pdf',
                'total_pages': 25,
                'total_detections': 15,
                'processing_time': 45.2,
                'pages_with_detections': 8,
                'avg_detections_per_page': 0.6
            },
            {
                'pdf_path': 'doc2.pdf',
                'total_pages': 30,
                'total_detections': 22,
                'processing_time': 52.8,
                'pages_with_detections': 12,
                'avg_detections_per_page': 0.73
            }
        ]
        
        report = visualization_utils.create_batch_report(batch_results)
        
        # Should include summary statistics
        expected_keys = [
            'total_pdfs', 'total_pages', 'total_detections', 'total_processing_time',
            'avg_processing_time_per_pdf', 'avg_detections_per_pdf',
            'pdf_results', 'timestamp'
        ]
        
        for key in expected_keys:
            assert key in report
        
        # Check calculations
        assert report['total_pdfs'] == 2
        assert report['total_pages'] == 55
        assert report['total_detections'] == 37
        assert report['total_processing_time'] == 98.0
    
    def test_create_comparison_visualization(self, visualization_utils, mock_image):
        """Test creating comparison visualization between different methods."""
        method_results = {
            'method_1': [
                {'bbox': [100, 100, 200, 200], 'confidence': 0.9, 'label': 'ship'}
            ],
            'method_2': [
                {'bbox': [105, 105, 205, 205], 'confidence': 0.85, 'label': 'vessel'}
            ]
        }
        
        with patch('PIL.Image.new') as mock_new_image, \
             patch.object(visualization_utils, 'draw_bounding_boxes') as mock_draw:
            
            mock_combined_image = Mock()
            mock_new_image.return_value = mock_combined_image
            mock_draw.return_value = mock_image
            
            result = visualization_utils.create_comparison_visualization(
                mock_image, method_results
            )
            
            # Should create new combined image
            mock_new_image.assert_called_once()
            
            # Should draw bounding boxes for each method
            assert mock_draw.call_count == len(method_results)
            
            assert result == mock_combined_image
    
    def test_generate_confidence_histogram(self, visualization_utils, mock_detection_results):
        """Test generating confidence score histogram."""
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.hist') as mock_hist, \
             patch('matplotlib.pyplot.savefig') as mock_savefig:
            
            mock_fig = Mock()
            mock_figure.return_value = mock_fig
            
            output_path = Path("confidence_hist.png")
            
            result_path = visualization_utils.generate_confidence_histogram(
                mock_detection_results, output_path
            )
            
            # Should create histogram
            mock_hist.assert_called_once()
            hist_data = mock_hist.call_args[0][0]
            
            # Should use confidence scores
            expected_confidences = [det['confidence'] for det in mock_detection_results]
            assert list(hist_data) == expected_confidences
            
            # Should save figure
            mock_savefig.assert_called_once_with(output_path, dpi=150, bbox_inches='tight')
            
            assert result_path == output_path
    
    def test_create_detection_grid(self, visualization_utils):
        """Test creating grid visualization of detected objects."""
        detected_images = [Mock(spec=Image.Image) for _ in range(6)]
        for i, img in enumerate(detected_images):
            img.size = (200, 200)
        
        with patch('PIL.Image.new') as mock_new_image:
            mock_grid_image = Mock()
            mock_new_image.return_value = mock_grid_image
            
            result = visualization_utils.create_detection_grid(
                detected_images, grid_size=(3, 2)
            )
            
            # Should create new grid image
            mock_new_image.assert_called_once()
            
            # Should paste images onto grid
            paste_calls = mock_grid_image.paste.call_args_list
            assert len(paste_calls) == len(detected_images)
            
            assert result == mock_grid_image
    
    def test_create_detection_grid_auto_size(self, visualization_utils):
        """Test automatic grid size calculation."""
        detected_images = [Mock(spec=Image.Image) for _ in range(7)]
        for img in detected_images:
            img.size = (200, 200)
        
        with patch('PIL.Image.new') as mock_new_image:
            mock_grid_image = Mock()
            mock_new_image.return_value = mock_grid_image
            
            # Should automatically calculate grid size
            result = visualization_utils.create_detection_grid(detected_images)
            
            # For 7 images, should create 3x3 grid (9 slots, 7 used)
            expected_width = 3 * 200  # 3 columns * 200px width
            expected_height = 3 * 200  # 3 rows * 200px height
            
            mock_new_image.assert_called_once_with(
                'RGB', (expected_width, expected_height), 'white'
            )
    
    def test_add_watermark(self, visualization_utils, mock_image):
        """Test adding watermark to image."""
        watermark_text = "Florence-2 Warship Extractor"
        
        with patch('PIL.ImageDraw.Draw') as mock_draw_class, \
             patch('PIL.ImageFont.truetype') as mock_font:
            
            mock_draw = Mock()
            mock_draw_class.return_value = mock_draw
            mock_font.return_value = Mock()
            
            result = visualization_utils.add_watermark(mock_image, watermark_text)
            
            # Should draw watermark text
            mock_draw.text.assert_called_once()
            
            # Should use semi-transparent overlay
            text_call = mock_draw.text.call_args
            assert watermark_text in text_call[0][1]  # Text content
            
            assert result == mock_image
    
    def test_export_detection_data_csv(self, visualization_utils, mock_detection_results):
        """Test exporting detection data to CSV."""
        output_path = Path("detections.csv")
        
        with patch('builtins.open', create=True) as mock_open, \
             patch('csv.writer') as mock_csv_writer:
            
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            mock_writer = Mock()
            mock_csv_writer.return_value = mock_writer
            
            result_path = visualization_utils.export_detection_data(
                mock_detection_results, output_path, format='csv'
            )
            
            # Should open file for writing
            mock_open.assert_called_once_with(output_path, 'w', newline='')
            
            # Should write CSV data
            mock_csv_writer.assert_called_once_with(mock_file)
            
            # Should write header and rows
            write_calls = mock_writer.writerow.call_args_list
            assert len(write_calls) == len(mock_detection_results) + 1  # +1 for header
            
            assert result_path == output_path
    
    def test_export_detection_data_json(self, visualization_utils, mock_detection_results):
        """Test exporting detection data to JSON."""
        output_path = Path("detections.json")
        
        with patch('builtins.open', create=True) as mock_open, \
             patch('json.dump') as mock_json_dump:
            
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            result_path = visualization_utils.export_detection_data(
                mock_detection_results, output_path, format='json'
            )
            
            # Should write JSON data
            mock_json_dump.assert_called_once_with(
                mock_detection_results, mock_file, indent=2
            )
            
            assert result_path == output_path
    
    def test_create_performance_chart(self, visualization_utils):
        """Test creating performance analysis chart."""
        performance_data = {
            'pdf_conversion': [2.1, 2.3, 1.9, 2.5],
            'detection': [15.2, 14.8, 16.1, 15.9],
            'post_processing': [1.1, 1.0, 1.2, 1.1]
        }
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.bar') as mock_bar, \
             patch('matplotlib.pyplot.savefig') as mock_savefig:
            
            output_path = Path("performance_chart.png")
            
            result_path = visualization_utils.create_performance_chart(
                performance_data, output_path
            )
            
            # Should create bar chart
            mock_bar.assert_called()
            
            # Should save chart
            mock_savefig.assert_called_once_with(
                output_path, dpi=150, bbox_inches='tight'
            )
            
            assert result_path == output_path
    
    def test_validate_bbox_format(self, visualization_utils):
        """Test bounding box format validation."""
        # Valid bbox
        valid_bbox = [100, 150, 300, 250]
        assert visualization_utils._validate_bbox(valid_bbox) == True
        
        # Invalid bbox (wrong number of coordinates)
        invalid_bbox_1 = [100, 150, 300]
        assert visualization_utils._validate_bbox(invalid_bbox_1) == False
        
        # Invalid bbox (x2 < x1)
        invalid_bbox_2 = [300, 150, 100, 250]
        assert visualization_utils._validate_bbox(invalid_bbox_2) == False
        
        # Invalid bbox (y2 < y1)
        invalid_bbox_3 = [100, 250, 300, 150]
        assert visualization_utils._validate_bbox(invalid_bbox_3) == False
    
    def test_filter_detections_by_confidence(self, visualization_utils, mock_detection_results):
        """Test filtering detections by confidence threshold."""
        min_confidence = 0.9
        
        filtered = visualization_utils.filter_detections(
            mock_detection_results, min_confidence=min_confidence
        )
        
        # Should only include detections above threshold
        assert all(det['confidence'] >= min_confidence for det in filtered)
        assert len(filtered) == 2  # Only 0.95 and 0.92 should remain
    
    def test_filter_detections_by_label(self, visualization_utils, mock_detection_results):
        """Test filtering detections by label."""
        allowed_labels = ['battleship', 'destroyer']
        
        filtered = visualization_utils.filter_detections(
            mock_detection_results, allowed_labels=allowed_labels
        )
        
        # Should only include specified labels
        assert all(det['label'] in allowed_labels for det in filtered)
        assert len(filtered) == 2  # battleship and destroyer
    
    def test_calculate_bbox_area(self, visualization_utils):
        """Test bounding box area calculation."""
        bbox = [100, 150, 300, 250]  # 200x100 = 20000
        
        area = visualization_utils._calculate_bbox_area(bbox)
        
        assert area == 20000
    
    def test_calculate_bbox_overlap(self, visualization_utils):
        """Test bounding box overlap calculation."""
        bbox1 = [100, 100, 200, 200]
        bbox2 = [150, 150, 250, 250]  # Overlapping
        bbox3 = [300, 300, 400, 400]  # Non-overlapping
        
        # Should calculate overlap
        overlap_1_2 = visualization_utils._calculate_bbox_overlap(bbox1, bbox2)
        overlap_1_3 = visualization_utils._calculate_bbox_overlap(bbox1, bbox3)
        
        assert overlap_1_2 > 0  # Should have overlap
        assert overlap_1_3 == 0  # Should have no overlap
    
    def test_merge_overlapping_detections(self, visualization_utils):
        """Test merging overlapping detections."""
        overlapping_detections = [
            {'bbox': [100, 100, 200, 200], 'confidence': 0.9, 'label': 'ship'},
            {'bbox': [150, 150, 250, 250], 'confidence': 0.85, 'label': 'ship'},  # Overlapping
            {'bbox': [300, 300, 400, 400], 'confidence': 0.8, 'label': 'ship'}   # Separate
        ]
        
        merged = visualization_utils.merge_overlapping_detections(
            overlapping_detections, overlap_threshold=0.3
        )
        
        # Should merge overlapping detections
        assert len(merged) == 2  # Two separate groups
        
        # Merged detection should have higher confidence
        merged_confidences = [det['confidence'] for det in merged]
        assert max(merged_confidences) == 0.9  # Should keep highest confidence
    
    def test_error_handling_invalid_image(self, visualization_utils, mock_detection_results):
        """Test error handling with invalid image."""
        invalid_image = None
        
        with pytest.raises(ValueError, match="Invalid image provided"):
            visualization_utils.draw_bounding_boxes(invalid_image, mock_detection_results)
    
    def test_error_handling_empty_detections(self, visualization_utils, mock_image):
        """Test handling of empty detection results."""
        empty_detections = []
        
        # Should handle empty detections gracefully
        result = visualization_utils.draw_bounding_boxes(mock_image, empty_detections)
        
        assert result == mock_image  # Should return original image unchanged
    
    def test_error_handling_invalid_bbox(self, visualization_utils, mock_image):
        """Test handling of invalid bounding box coordinates."""
        invalid_detections = [
            {'bbox': [100, 150, 50, 250], 'confidence': 0.9, 'label': 'ship'}  # x2 < x1
        ]
        
        with patch('PIL.ImageDraw.Draw'):
            # Should skip invalid detections without crashing
            result = visualization_utils.draw_bounding_boxes(mock_image, invalid_detections)
            assert result == mock_image
    
    def test_settings_integration(self, mock_settings):
        """Test integration with settings configuration."""
        viz = VisualizationUtils(mock_settings)
        
        # Should use settings for configuration
        assert viz.dpi == mock_settings.visualization_dpi
        assert viz.line_width == mock_settings.bbox_line_width
        assert viz.colors == mock_settings.bbox_colors
        assert viz.font_size == mock_settings.font_size
    
    def test_custom_color_scheme(self, mock_settings, mock_image, mock_detection_results):
        """Test custom color scheme configuration."""
        custom_colors = ["purple", "cyan", "magenta"]
        mock_settings.bbox_colors = custom_colors
        
        viz = VisualizationUtils(mock_settings)
        
        with patch('PIL.ImageDraw.Draw') as mock_draw_class:
            mock_draw = Mock()
            mock_draw_class.return_value = mock_draw
            
            viz.draw_bounding_boxes(mock_image, mock_detection_results)
            
            # Should use custom colors
            rectangle_calls = mock_draw.rectangle.call_args_list
            for i, call in enumerate(rectangle_calls):
                expected_color = custom_colors[i % len(custom_colors)]
                assert call[1]['outline'] == expected_color
"""
Unit tests for the ImageProcessor class.

Tests image cropping, enhancement, format conversion, and validation
without requiring actual image files.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import io

from src.warship_extractor.processing.image_processor import ImageProcessor
from src.warship_extractor.config.settings import Settings


class TestImageProcessor:
    """Test cases for ImageProcessor functionality."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = Mock()
        settings.image_output_format = "PNG"
        settings.image_quality = 95
        settings.image_enhance_contrast = True
        settings.image_enhance_sharpness = True
        settings.image_resize_max_width = 1024
        settings.image_resize_max_height = 1024
        settings.image_padding_pixels = 10
        settings.image_min_size_pixels = 50
        settings.image_background_color = (255, 255, 255)
        return settings
    
    @pytest.fixture
    def image_processor(self, mock_settings):
        """Create ImageProcessor instance for testing."""
        return ImageProcessor(mock_settings)
    
    @pytest.fixture
    def mock_image(self):
        """Create mock PIL Image for testing."""
        image = Mock(spec=Image.Image)
        image.size = (800, 600)
        image.mode = "RGB"
        image.format = "PNG"
        return image
    
    def test_initialization(self, mock_settings):
        """Test ImageProcessor initialization."""
        processor = ImageProcessor(mock_settings)
        
        assert processor.settings == mock_settings
        assert processor.output_format == mock_settings.image_output_format
        assert processor.quality == mock_settings.image_quality
    
    def test_crop_detections_success(self, image_processor, mock_image):
        """Test successful cropping of detections from image."""
        detections = [
            {'bbox': [100, 100, 300, 250], 'label': 'warship', 'confidence': 0.8},
            {'bbox': [400, 200, 600, 350], 'label': 'destroyer', 'confidence': 0.7}
        ]
        
        # Mock the crop method
        mock_cropped1 = Mock(spec=Image.Image)
        mock_cropped1.size = (200, 150)
        mock_cropped2 = Mock(spec=Image.Image)
        mock_cropped2.size = (200, 150)
        
        mock_image.crop.side_effect = [mock_cropped1, mock_cropped2]
        
        with patch.object(image_processor, '_enhance_image', side_effect=lambda x: x):
            cropped_images = image_processor.crop_detections(mock_image, detections)
            
            assert len(cropped_images) == 2
            assert cropped_images[0]['image'] == mock_cropped1
            assert cropped_images[0]['detection'] == detections[0]
            assert cropped_images[1]['image'] == mock_cropped2
            assert cropped_images[1]['detection'] == detections[1]
            
            # Verify crop calls
            mock_image.crop.assert_any_call((100, 100, 300, 250))
            mock_image.crop.assert_any_call((400, 200, 600, 350))
    
    def test_crop_detections_with_padding(self, mock_settings, mock_image):
        """Test cropping with padding."""
        mock_settings.image_padding_pixels = 20
        processor = ImageProcessor(mock_settings)
        
        detections = [
            {'bbox': [100, 100, 200, 200], 'label': 'warship', 'confidence': 0.8}
        ]
        
        mock_cropped = Mock(spec=Image.Image)
        mock_image.crop.return_value = mock_cropped
        
        with patch.object(processor, '_enhance_image', side_effect=lambda x: x):
            cropped_images = processor.crop_detections(mock_image, detections)
            
            # Should crop with padding (80, 80, 220, 220)
            expected_bbox = (80, 80, 220, 220)
            mock_image.crop.assert_called_once_with(expected_bbox)
    
    def test_crop_detections_boundary_clipping(self, image_processor, mock_image):
        """Test cropping with boundary clipping."""
        # Detection near image boundary
        detections = [
            {'bbox': [750, 550, 850, 650], 'label': 'warship', 'confidence': 0.8}
        ]
        
        mock_cropped = Mock(spec=Image.Image)
        mock_image.crop.return_value = mock_cropped
        
        with patch.object(image_processor, '_enhance_image', side_effect=lambda x: x):
            cropped_images = image_processor.crop_detections(mock_image, detections)
            
            # Should clip to image boundaries (740, 540, 800, 600)
            expected_bbox = (740, 540, 800, 600)
            mock_image.crop.assert_called_once_with(expected_bbox)
    
    def test_crop_detections_too_small(self, mock_settings, mock_image):
        """Test filtering out detections that are too small."""
        mock_settings.image_min_size_pixels = 100
        processor = ImageProcessor(mock_settings)
        
        detections = [
            {'bbox': [100, 100, 110, 110], 'label': 'warship', 'confidence': 0.8},  # 10x10 = too small
            {'bbox': [200, 200, 350, 350], 'label': 'destroyer', 'confidence': 0.7}  # 150x150 = OK
        ]
        
        mock_cropped = Mock(spec=Image.Image)
        mock_image.crop.return_value = mock_cropped
        
        with patch.object(processor, '_enhance_image', side_effect=lambda x: x):
            cropped_images = processor.crop_detections(mock_image, detections)
            
            # Should only process the large enough detection
            assert len(cropped_images) == 1
            assert cropped_images[0]['detection'] == detections[1]
    
    def test_enhance_image_contrast_and_sharpness(self, image_processor, mock_image):
        """Test image enhancement with contrast and sharpness."""
        # Mock ImageEnhance classes
        with patch('PIL.ImageEnhance.Contrast') as mock_contrast_class, \
             patch('PIL.ImageEnhance.Sharpness') as mock_sharpness_class:
            
            mock_contrast_enhancer = Mock()
            mock_sharpness_enhancer = Mock()
            mock_contrast_class.return_value = mock_contrast_enhancer
            mock_sharpness_class.return_value = mock_sharpness_enhancer
            
            enhanced_image = Mock(spec=Image.Image)
            mock_contrast_enhancer.enhance.return_value = enhanced_image
            mock_sharpness_enhancer.enhance.return_value = enhanced_image
            
            result = image_processor._enhance_image(mock_image)
            
            assert result == enhanced_image
            mock_contrast_class.assert_called_once_with(mock_image)
            mock_sharpness_class.assert_called_once_with(enhanced_image)
    
    def test_enhance_image_disabled(self, mock_settings, mock_image):
        """Test image enhancement when disabled."""
        mock_settings.image_enhance_contrast = False
        mock_settings.image_enhance_sharpness = False
        processor = ImageProcessor(mock_settings)
        
        result = processor._enhance_image(mock_image)
        
        # Should return original image unchanged
        assert result == mock_image
    
    def test_resize_image_within_limits(self, image_processor, mock_image):
        """Test image resizing when within size limits."""
        mock_image.size = (500, 400)  # Smaller than max
        
        result = image_processor._resize_image(mock_image)
        
        # Should return original image
        assert result == mock_image
    
    def test_resize_image_exceeds_width(self, image_processor, mock_image):
        """Test image resizing when width exceeds limit."""
        mock_image.size = (1500, 600)  # Width exceeds max_width (1024)
        
        resized_image = Mock(spec=Image.Image)
        mock_image.resize.return_value = resized_image
        
        result = image_processor._resize_image(mock_image)
        
        # Should resize to (1024, 409) maintaining aspect ratio
        expected_size = (1024, 409)
        mock_image.resize.assert_called_once_with(expected_size, Image.Resampling.LANCZOS)
        assert result == resized_image
    
    def test_resize_image_exceeds_height(self, image_processor, mock_image):
        """Test image resizing when height exceeds limit."""
        mock_image.size = (600, 1500)  # Height exceeds max_height (1024)
        
        resized_image = Mock(spec=Image.Image)
        mock_image.resize.return_value = resized_image
        
        result = image_processor._resize_image(mock_image)
        
        # Should resize to (409, 1024) maintaining aspect ratio
        expected_size = (409, 1024)
        mock_image.resize.assert_called_once_with(expected_size, Image.Resampling.LANCZOS)
        assert result == resized_image
    
    def test_convert_format_png_to_jpeg(self, mock_settings, mock_image):
        """Test format conversion from PNG to JPEG."""
        mock_settings.image_output_format = "JPEG"
        processor = ImageProcessor(mock_settings)
        
        # PNG image with transparency
        mock_image.mode = "RGBA"
        
        # Mock the conversion process
        rgb_image = Mock(spec=Image.Image)
        rgb_image.mode = "RGB"
        
        with patch('PIL.Image.new') as mock_new:
            background = Mock(spec=Image.Image)
            mock_new.return_value = background
            background.paste = Mock()
            
            mock_image.convert.return_value = rgb_image
            
            result = processor._convert_format(mock_image)
            
            # Should convert RGBA to RGB
            mock_image.convert.assert_called_with("RGB")
            assert result == rgb_image
    
    def test_convert_format_same_format(self, image_processor, mock_image):
        """Test format conversion when already in target format."""
        mock_image.mode = "RGB"  # Already RGB, target is PNG which supports RGB
        
        result = image_processor._convert_format(mock_image)
        
        # Should return original image
        assert result == mock_image
    
    def test_save_image_success(self, image_processor, mock_image):
        """Test successful image saving."""
        output_path = Path("test_output.png")
        
        with patch.object(mock_image, 'save') as mock_save:
            image_processor.save_image(mock_image, output_path)
            
            mock_save.assert_called_once_with(
                str(output_path),
                format="PNG",
                quality=95,
                optimize=True
            )
    
    def test_save_image_jpeg_quality(self, mock_settings, mock_image):
        """Test JPEG saving with quality setting."""
        mock_settings.image_output_format = "JPEG"
        mock_settings.image_quality = 85
        processor = ImageProcessor(mock_settings)
        
        output_path = Path("test_output.jpg")
        
        with patch.object(mock_image, 'save') as mock_save:
            processor.save_image(mock_image, output_path)
            
            mock_save.assert_called_once_with(
                str(output_path),
                format="JPEG",
                quality=85,
                optimize=True
            )
    
    def test_save_image_with_directory_creation(self, image_processor, mock_image):
        """Test image saving with automatic directory creation."""
        output_path = Path("new_dir/subdir/test_output.png")
        
        with patch.object(mock_image, 'save') as mock_save, \
             patch.object(Path, 'mkdir') as mock_mkdir:
            
            image_processor.save_image(mock_image, output_path)
            
            # Should create parent directories
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
            mock_save.assert_called_once()
    
    def test_save_image_error_handling(self, image_processor, mock_image):
        """Test error handling during image saving."""
        output_path = Path("test_output.png")
        
        with patch.object(mock_image, 'save', side_effect=IOError("Save failed")):
            with pytest.raises(RuntimeError, match="Failed to save image"):
                image_processor.save_image(mock_image, output_path)
    
    def test_process_detection_images_full_pipeline(self, image_processor, mock_image):
        """Test the full detection image processing pipeline."""
        detections = [
            {'bbox': [100, 100, 300, 250], 'label': 'warship', 'confidence': 0.8}
        ]
        output_dir = Path("output")
        
        mock_cropped = Mock(spec=Image.Image)
        mock_enhanced = Mock(spec=Image.Image)
        mock_resized = Mock(spec=Image.Image)
        mock_converted = Mock(spec=Image.Image)
        
        with patch.object(image_processor, 'crop_detections') as mock_crop, \
             patch.object(image_processor, '_enhance_image', return_value=mock_enhanced), \
             patch.object(image_processor, '_resize_image', return_value=mock_resized), \
             patch.object(image_processor, '_convert_format', return_value=mock_converted), \
             patch.object(image_processor, 'save_image') as mock_save:
            
            mock_crop.return_value = [{'image': mock_cropped, 'detection': detections[0]}]
            
            result = image_processor.process_detection_images(mock_image, detections, output_dir)
            
            assert len(result) == 1
            assert result[0]['detection'] == detections[0]
            assert 'output_path' in result[0]
            
            # Verify processing pipeline
            mock_crop.assert_called_once_with(mock_image, detections)
            mock_save.assert_called_once()
    
    def test_get_image_info(self, image_processor, mock_image):
        """Test image information extraction."""
        mock_image.size = (800, 600)
        mock_image.mode = "RGB"
        mock_image.format = "PNG"
        
        # Mock additional image properties
        with patch.object(mock_image, 'getexif', return_value={}):
            info = image_processor.get_image_info(mock_image)
            
            expected_keys = [
                'size', 'mode', 'format', 'width', 'height',
                'aspect_ratio', 'file_size_estimate', 'color_channels'
            ]
            
            for key in expected_keys:
                assert key in info
            
            assert info['width'] == 800
            assert info['height'] == 600
            assert info['aspect_ratio'] == 800 / 600
    
    def test_validate_bbox_format(self, image_processor):
        """Test bounding box format validation."""
        # Valid bounding boxes
        valid_bboxes = [
            [100, 100, 200, 200],
            [0, 0, 800, 600],
            [50.5, 75.3, 150.7, 200.8]  # Float coordinates
        ]
        
        for bbox in valid_bboxes:
            assert image_processor._validate_bbox_format(bbox) == True
        
        # Invalid bounding boxes
        invalid_bboxes = [
            [100, 100, 200],  # Missing coordinate
            [100, 100, 200, 200, 300],  # Too many coordinates
            ["100", "100", "200", "200"],  # String coordinates
            [200, 200, 100, 300],  # x2 < x1
            [100, 300, 200, 200],  # y2 < y1
        ]
        
        for bbox in invalid_bboxes:
            assert image_processor._validate_bbox_format(bbox) == False
    
    def test_calculate_padding_bounds(self, image_processor):
        """Test padding bounds calculation."""
        image_size = (800, 600)
        bbox = [100, 100, 200, 200]
        padding = 20
        
        padded_bbox = image_processor._calculate_padding_bounds(bbox, padding, image_size)
        
        # Should be [80, 80, 220, 220]
        expected = [80, 80, 220, 220]
        assert padded_bbox == expected
    
    def test_calculate_padding_bounds_clipping(self, image_processor):
        """Test padding bounds calculation with boundary clipping."""
        image_size = (800, 600)
        bbox = [10, 10, 50, 50]  # Near boundary
        padding = 20
        
        padded_bbox = image_processor._calculate_padding_bounds(bbox, padding, image_size)
        
        # Should be clipped to [0, 0, 70, 70]
        expected = [0, 0, 70, 70]
        assert padded_bbox == expected
    
    def test_generate_output_filename(self, image_processor):
        """Test output filename generation."""
        detection = {
            'bbox': [100, 100, 200, 200],
            'label': 'warship',
            'confidence': 0.8532
        }
        
        filename = image_processor._generate_output_filename(detection, 0)
        
        # Should include index, label, and confidence
        assert "0_warship_0.85" in filename
        assert filename.endswith(".png")
    
    def test_batch_process_images(self, image_processor):
        """Test batch processing of multiple images."""
        images = [Mock(spec=Image.Image) for _ in range(3)]
        detections_list = [
            [{'bbox': [100, 100, 200, 200], 'label': 'warship', 'confidence': 0.8}],
            [{'bbox': [150, 150, 250, 250], 'label': 'destroyer', 'confidence': 0.7}],
            []  # No detections
        ]
        output_dir = Path("batch_output")
        
        with patch.object(image_processor, 'process_detection_images') as mock_process:
            mock_process.side_effect = [
                [{'detection': detections_list[0][0], 'output_path': 'path1'}],
                [{'detection': detections_list[1][0], 'output_path': 'path2'}],
                []
            ]
            
            results = image_processor.batch_process_images(images, detections_list, output_dir)
            
            assert len(results) == 3
            assert len(results[0]) == 1  # First image: 1 processed detection
            assert len(results[1]) == 1  # Second image: 1 processed detection
            assert len(results[2]) == 0  # Third image: no detections
    
    def test_get_processing_statistics(self, image_processor):
        """Test processing statistics generation."""
        processed_results = [
            [{'detection': {'label': 'warship'}, 'output_path': 'path1'}],
            [{'detection': {'label': 'destroyer'}, 'output_path': 'path2'}],
            []
        ]
        
        stats = image_processor.get_processing_statistics(processed_results)
        
        expected_keys = [
            'total_images_processed', 'total_detections_processed',
            'images_with_detections', 'images_without_detections',
            'label_counts', 'average_detections_per_image'
        ]
        
        for key in expected_keys:
            assert key in stats
        
        assert stats['total_images_processed'] == 3
        assert stats['total_detections_processed'] == 2
        assert stats['images_with_detections'] == 2
        assert stats['images_without_detections'] == 1
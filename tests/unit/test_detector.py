"""
Unit tests for the WarshipDetector class.

Tests detection logic, prompt coordination, and result processing
without requiring actual model inference.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import numpy as np
from PIL import Image

from src.warship_extractor.detection.detector import WarshipDetector
from src.warship_extractor.detection.prompt_strategies import PromptStrategy
from src.warship_extractor.config.settings import Settings


class TestWarshipDetector:
    """Test cases for WarshipDetector functionality."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = Mock()
        settings.detection_confidence_threshold = 0.5
        settings.detection_batch_size = 4
        settings.detection_max_detections = 100
        settings.detection_enable_multi_prompt = True
        settings.detection_prompt_strategies = ["technical_drawing", "schematic", "historical"]
        settings.detection_nms_threshold = 0.5
        settings.detection_adaptive_prompts = True
        return settings
    
    @pytest.fixture
    def mock_model_manager(self):
        """Create mock model manager."""
        manager = Mock()
        manager.model = Mock()
        manager.processor = Mock()
        manager.is_loaded.return_value = True
        return manager
    
    @pytest.fixture
    def mock_prompt_strategy(self):
        """Create mock prompt strategy."""
        strategy = Mock()
        strategy.get_detection_prompts.return_value = [
            "Locate warships in this technical drawing",
            "Find naval vessels and ships",
            "Identify military ships and warships"
        ]
        strategy.should_use_prompt.return_value = True
        strategy.get_adaptive_prompts.return_value = [
            "Enhanced prompt for this image type"
        ]
        return strategy
    
    @pytest.fixture
    def detector(self, mock_settings, mock_model_manager, mock_prompt_strategy):
        """Create WarshipDetector instance for testing."""
        with patch('src.warship_extractor.detection.detector.PromptStrategy') as mock_ps_class:
            mock_ps_class.return_value = mock_prompt_strategy
            return WarshipDetector(mock_settings, mock_model_manager)
    
    def test_initialization(self, mock_settings, mock_model_manager, mock_prompt_strategy):
        """Test WarshipDetector initialization."""
        with patch('src.warship_extractor.detection.detector.PromptStrategy') as mock_ps_class:
            mock_ps_class.return_value = mock_prompt_strategy
            
            detector = WarshipDetector(mock_settings, mock_model_manager)
            
            assert detector.settings == mock_settings
            assert detector.model_manager == mock_model_manager
            assert detector.prompt_strategy == mock_prompt_strategy
            assert detector.confidence_threshold == mock_settings.detection_confidence_threshold
    
    def test_initialization_model_not_loaded(self, mock_settings, mock_model_manager):
        """Test initialization when model is not loaded."""
        mock_model_manager.is_loaded.return_value = False
        
        with pytest.raises(RuntimeError, match="Model must be loaded"):
            WarshipDetector(mock_settings, mock_model_manager)
    
    def test_detect_warships_single_image_success(self, detector):
        """Test successful warship detection on single image."""
        # Create mock image
        mock_image = Mock(spec=Image.Image)
        mock_image.size = (800, 600)
        mock_image.mode = "RGB"
        
        # Setup mock model outputs
        mock_results = {
            'prediction': {
                'bboxes': [[100, 100, 200, 200], [300, 150, 400, 250]],
                'labels': ['warship', 'naval vessel']
            }
        }
        
        # Mock the detection pipeline
        with patch.object(detector, '_run_detection_pipeline', return_value=mock_results):
            detections = detector.detect_warships(mock_image)
            
            assert len(detections) == 2
            assert all('bbox' in det for det in detections)
            assert all('label' in det for det in detections)
            assert all('confidence' in det for det in detections)
    
    def test_detect_warships_batch_processing(self, detector):
        """Test batch processing of multiple images."""
        # Create mock images
        mock_images = [Mock(spec=Image.Image) for _ in range(3)]
        for i, img in enumerate(mock_images):
            img.size = (800, 600)
            img.mode = "RGB"
        
        # Setup mock results for each image
        mock_results = [
            {
                'prediction': {
                    'bboxes': [[100, 100, 200, 200]],
                    'labels': ['warship']
                }
            },
            {
                'prediction': {
                    'bboxes': [[150, 150, 250, 250], [300, 200, 400, 300]],
                    'labels': ['destroyer', 'cruiser']
                }
            },
            {
                'prediction': {
                    'bboxes': [],
                    'labels': []
                }
            }
        ]
        
        with patch.object(detector, '_run_detection_pipeline', side_effect=mock_results):
            all_detections = detector.detect_warships_batch(mock_images)
            
            assert len(all_detections) == 3
            assert len(all_detections[0]) == 1  # First image: 1 detection
            assert len(all_detections[1]) == 2  # Second image: 2 detections
            assert len(all_detections[2]) == 0  # Third image: no detections
    
    def test_run_detection_pipeline_success(self, detector):
        """Test the detection pipeline execution."""
        mock_image = Mock(spec=Image.Image)
        mock_image.size = (800, 600)
        
        # Mock processor and model outputs
        mock_inputs = {'pixel_values': Mock()}
        detector.model_manager.processor.return_value = mock_inputs
        
        mock_generated_ids = Mock()
        detector.model_manager.model.generate.return_value = mock_generated_ids
        
        mock_generated_text = "<OD>warship<loc_100><loc_100><loc_200><loc_200></OD>"
        detector.model_manager.processor.batch_decode.return_value = [mock_generated_text]
        
        # Mock the parsing
        with patch.object(detector, '_parse_florence_output') as mock_parse:
            mock_parse.return_value = {
                'bboxes': [[100, 100, 200, 200]],
                'labels': ['warship']
            }
            
            prompt = "Locate warships"
            result = detector._run_detection_pipeline(mock_image, prompt)
            
            assert 'prediction' in result
            assert result['prediction']['bboxes'] == [[100, 100, 200, 200]]
            assert result['prediction']['labels'] == ['warship']
    
    def test_run_detection_pipeline_with_error_handling(self, detector):
        """Test detection pipeline error handling."""
        mock_image = Mock(spec=Image.Image)
        
        # Make processor raise an exception
        detector.model_manager.processor.side_effect = Exception("Processing failed")
        
        with pytest.raises(RuntimeError, match="Detection failed"):
            detector._run_detection_pipeline(mock_image, "test prompt")
    
    def test_parse_florence_output_valid(self, detector):
        """Test parsing of valid Florence-2 output."""
        # Test various Florence-2 output formats
        test_cases = [
            {
                'text': "<OD>warship<loc_100><loc_100><loc_200><loc_200></OD>",
                'expected_bboxes': [[100, 100, 200, 200]],
                'expected_labels': ['warship']
            },
            {
                'text': "<OD>destroyer<loc_50><loc_75><loc_150><loc_175>battleship<loc_300><loc_400><loc_500><loc_600></OD>",
                'expected_bboxes': [[50, 75, 150, 175], [300, 400, 500, 600]],
                'expected_labels': ['destroyer', 'battleship']
            },
            {
                'text': "<OD></OD>",  # No detections
                'expected_bboxes': [],
                'expected_labels': []
            }
        ]
        
        for case in test_cases:
            result = detector._parse_florence_output(case['text'])
            assert result['bboxes'] == case['expected_bboxes']
            assert result['labels'] == case['expected_labels']
    
    def test_parse_florence_output_invalid(self, detector):
        """Test parsing of invalid Florence-2 output."""
        invalid_outputs = [
            "No OD tags",
            "<OD>incomplete",
            "<OD>warship<loc_invalid></OD>",
            "<OD>warship<loc_100><loc_100><loc_200></OD>",  # Missing coordinate
        ]
        
        for invalid_output in invalid_outputs:
            result = detector._parse_florence_output(invalid_output)
            # Should return empty results for invalid input
            assert result['bboxes'] == []
            assert result['labels'] == []
    
    def test_filter_detections_by_confidence(self, detector):
        """Test filtering detections by confidence threshold."""
        detections = [
            {'bbox': [100, 100, 200, 200], 'label': 'warship', 'confidence': 0.8},
            {'bbox': [300, 150, 400, 250], 'label': 'destroyer', 'confidence': 0.3},
            {'bbox': [500, 200, 600, 300], 'label': 'cruiser', 'confidence': 0.6},
        ]
        
        # Test with threshold 0.5
        filtered = detector._filter_detections_by_confidence(detections, 0.5)
        assert len(filtered) == 2
        assert all(det['confidence'] >= 0.5 for det in filtered)
    
    def test_filter_detections_by_confidence_empty(self, detector):
        """Test confidence filtering with no detections."""
        detections = []
        filtered = detector._filter_detections_by_confidence(detections, 0.5)
        assert filtered == []
    
    def test_normalize_bbox_coordinates(self, detector):
        """Test bounding box coordinate normalization."""
        image_size = (800, 600)  # width, height
        
        test_cases = [
            {
                'bbox': [100, 150, 300, 450],  # x1, y1, x2, y2
                'expected': [100, 150, 300, 450]  # Should remain the same for absolute coords
            },
            {
                'bbox': [0.125, 0.25, 0.375, 0.75],  # Normalized coordinates
                'expected': [100, 150, 300, 450]  # Converted to absolute
            }
        ]
        
        for case in test_cases:
            normalized = detector._normalize_bbox_coordinates(case['bbox'], image_size)
            # Allow small floating point differences
            for i in range(4):
                assert abs(normalized[i] - case['expected'][i]) < 1.0
    
    def test_validate_bbox(self, detector):
        """Test bounding box validation."""
        image_size = (800, 600)
        
        # Valid bounding boxes
        valid_bboxes = [
            [100, 100, 200, 200],
            [0, 0, 799, 599],
            [50, 75, 100, 125]
        ]
        
        for bbox in valid_bboxes:
            assert detector._validate_bbox(bbox, image_size) == True
        
        # Invalid bounding boxes
        invalid_bboxes = [
            [-10, 100, 200, 200],  # Negative coordinate
            [100, 100, 900, 200],  # Beyond image width
            [100, 100, 200, 700],  # Beyond image height
            [200, 200, 100, 300],  # x2 < x1
            [100, 300, 200, 200],  # y2 < y1
        ]
        
        for bbox in invalid_bboxes:
            assert detector._validate_bbox(bbox, image_size) == False
    
    def test_merge_multi_prompt_results(self, detector):
        """Test merging results from multiple prompts."""
        results_list = [
            {
                'prediction': {
                    'bboxes': [[100, 100, 200, 200], [300, 150, 400, 250]],
                    'labels': ['warship', 'destroyer']
                }
            },
            {
                'prediction': {
                    'bboxes': [[150, 125, 250, 225], [500, 300, 600, 400]],
                    'labels': ['cruiser', 'battleship']
                }
            }
        ]
        
        with patch.object(detector, '_apply_nms_filtering') as mock_nms:
            mock_nms.return_value = [
                {'bbox': [100, 100, 200, 200], 'label': 'warship', 'confidence': 0.8},
                {'bbox': [300, 150, 400, 250], 'label': 'destroyer', 'confidence': 0.7},
                {'bbox': [500, 300, 600, 400], 'label': 'battleship', 'confidence': 0.6}
            ]
            
            merged = detector._merge_multi_prompt_results(results_list, (800, 600))
            
            assert len(merged) == 3
            mock_nms.assert_called_once()
    
    def test_apply_nms_filtering(self, detector):
        """Test Non-Maximum Suppression filtering."""
        detections = [
            {'bbox': [100, 100, 200, 200], 'label': 'warship', 'confidence': 0.8},
            {'bbox': [110, 110, 210, 210], 'label': 'warship', 'confidence': 0.6},  # Overlapping
            {'bbox': [300, 150, 400, 250], 'label': 'destroyer', 'confidence': 0.7},
        ]
        
        # Mock the NMS filter
        with patch('src.warship_extractor.detection.detector.NMSFilter') as mock_nms_class:
            mock_nms = Mock()
            mock_nms.filter_detections.return_value = [
                detections[0],  # Keep higher confidence overlapping detection
                detections[2]   # Keep non-overlapping detection
            ]
            mock_nms_class.return_value = mock_nms
            
            filtered = detector._apply_nms_filtering(detections)
            
            assert len(filtered) == 2
            mock_nms.filter_detections.assert_called_once_with(detections)
    
    def test_get_detection_statistics(self, detector):
        """Test detection statistics generation."""
        detections = [
            {'bbox': [100, 100, 200, 200], 'label': 'warship', 'confidence': 0.8},
            {'bbox': [300, 150, 400, 250], 'label': 'destroyer', 'confidence': 0.7},
            {'bbox': [500, 200, 600, 300], 'label': 'warship', 'confidence': 0.6},
        ]
        
        stats = detector.get_detection_statistics(detections)
        
        expected_keys = [
            'total_detections', 'labels_detected', 'confidence_stats',
            'bbox_size_stats', 'label_counts'
        ]
        
        for key in expected_keys:
            assert key in stats
        
        assert stats['total_detections'] == 3
        assert stats['label_counts']['warship'] == 2
        assert stats['label_counts']['destroyer'] == 1
    
    def test_adaptive_prompt_selection(self, detector):
        """Test adaptive prompt selection based on image characteristics."""
        mock_image = Mock(spec=Image.Image)
        mock_image.size = (1200, 800)  # High resolution
        
        # Mock the adaptive prompt selection
        detector.prompt_strategy.get_adaptive_prompts.return_value = [
            "High-resolution technical drawing analysis",
            "Detailed warship schematic detection"
        ]
        
        adaptive_prompts = detector._get_adaptive_prompts(mock_image)
        
        assert len(adaptive_prompts) == 2
        detector.prompt_strategy.get_adaptive_prompts.assert_called_once_with(mock_image)
    
    def test_detection_with_disabled_multi_prompt(self, mock_settings, mock_model_manager, mock_prompt_strategy):
        """Test detection with multi-prompt disabled."""
        mock_settings.detection_enable_multi_prompt = False
        
        with patch('src.warship_extractor.detection.detector.PromptStrategy') as mock_ps_class:
            mock_ps_class.return_value = mock_prompt_strategy
            detector = WarshipDetector(mock_settings, mock_model_manager)
        
        mock_image = Mock(spec=Image.Image)
        mock_image.size = (800, 600)
        
        # Should use only single prompt
        with patch.object(detector, '_run_detection_pipeline') as mock_pipeline:
            mock_pipeline.return_value = {
                'prediction': {'bboxes': [], 'labels': []}
            }
            
            detector.detect_warships(mock_image)
            
            # Should be called only once (single prompt)
            assert mock_pipeline.call_count == 1
    
    def test_detection_max_detections_limit(self, detector):
        """Test maximum detections limit enforcement."""
        mock_image = Mock(spec=Image.Image)
        mock_image.size = (800, 600)
        
        # Create many detections (more than limit)
        many_detections = [
            {'bbox': [i*50, i*50, (i+1)*50, (i+1)*50], 'label': 'warship', 'confidence': 0.8}
            for i in range(150)  # More than max_detections (100)
        ]
        
        with patch.object(detector, '_run_detection_pipeline') as mock_pipeline:
            mock_pipeline.return_value = {
                'prediction': {
                    'bboxes': [det['bbox'] for det in many_detections],
                    'labels': [det['label'] for det in many_detections]
                }
            }
            
            detections = detector.detect_warships(mock_image)
            
            # Should be limited to max_detections
            assert len(detections) <= detector.settings.detection_max_detections
    
    def test_detection_error_recovery(self, detector):
        """Test error recovery during detection."""
        mock_image = Mock(spec=Image.Image)
        mock_image.size = (800, 600)
        
        # Make first prompt fail, second succeed
        with patch.object(detector, '_run_detection_pipeline') as mock_pipeline:
            mock_pipeline.side_effect = [
                RuntimeError("First prompt failed"),
                {
                    'prediction': {
                        'bboxes': [[100, 100, 200, 200]],
                        'labels': ['warship']
                    }
                }
            ]
            
            # Should continue with remaining prompts despite first failure
            detections = detector.detect_warships(mock_image)
            
            # Should still get results from successful prompt
            assert len(detections) == 1
    
    def test_get_detector_info(self, detector):
        """Test detector information retrieval."""
        info = detector.get_detector_info()
        
        expected_keys = [
            'confidence_threshold', 'batch_size', 'max_detections',
            'multi_prompt_enabled', 'prompt_strategies', 'model_loaded'
        ]
        
        for key in expected_keys:
            assert key in info
        
        assert info['confidence_threshold'] == detector.confidence_threshold
        assert info['model_loaded'] == detector.model_manager.is_loaded()
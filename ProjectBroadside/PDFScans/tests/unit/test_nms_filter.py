"""
Unit tests for the NMS (Non-Maximum Suppression) filter.

Tests the NMSFilter class functionality including IoU calculation,
overlap detection, and duplicate removal logic.
"""

import pytest
import numpy as np

from warship_extractor.processing.nms_filter import NMSFilter


class TestNMSFilter:
    """Test cases for the NMSFilter class."""

    def test_init_default_settings(self):
        """Test NMSFilter initialization with default settings."""
        nms_filter = NMSFilter()
        
        assert nms_filter.iou_threshold == 0.5
        assert nms_filter.confidence_threshold == 0.3
        assert nms_filter.enable_label_specific is True

    def test_init_custom_settings(self):
        """Test NMSFilter initialization with custom settings."""
        nms_filter = NMSFilter(
            iou_threshold=0.7,
            confidence_threshold=0.4,
            enable_label_specific=False
        )
        
        assert nms_filter.iou_threshold == 0.7
        assert nms_filter.confidence_threshold == 0.4
        assert nms_filter.enable_label_specific is False

    def test_calculate_iou_no_overlap(self):
        """Test IoU calculation for non-overlapping boxes."""
        nms_filter = NMSFilter()
        
        box1 = [0, 0, 100, 100]
        box2 = [200, 200, 300, 300]
        
        iou = nms_filter.calculate_iou(box1, box2)
        assert iou == 0.0

    def test_calculate_iou_complete_overlap(self):
        """Test IoU calculation for completely overlapping boxes."""
        nms_filter = NMSFilter()
        
        box1 = [100, 100, 200, 200]
        box2 = [100, 100, 200, 200]
        
        iou = nms_filter.calculate_iou(box1, box2)
        assert iou == 1.0

    def test_calculate_iou_partial_overlap(self):
        """Test IoU calculation for partially overlapping boxes."""
        nms_filter = NMSFilter()
        
        box1 = [0, 0, 100, 100]
        box2 = [50, 50, 150, 150]
        
        iou = nms_filter.calculate_iou(box1, box2)
        
        # Intersection area: 50x50 = 2500
        # Union area: 100x100 + 100x100 - 2500 = 17500
        # IoU = 2500 / 17500 = 0.1428...
        expected_iou = 2500 / 17500
        assert abs(iou - expected_iou) < 1e-6

    def test_calculate_iou_invalid_boxes(self):
        """Test IoU calculation with invalid bounding boxes."""
        nms_filter = NMSFilter()
        
        # Invalid box (x2 < x1)
        box1 = [100, 100, 50, 200]
        box2 = [0, 0, 100, 100]
        
        iou = nms_filter.calculate_iou(box1, box2)
        assert iou == 0.0

    def test_has_significant_overlap_true(self):
        """Test overlap detection for significantly overlapping boxes."""
        nms_filter = NMSFilter(iou_threshold=0.5)
        
        detection1 = {
            'bbox': [0, 0, 100, 100],
            'confidence': 0.8,
            'label': 'warship'
        }
        detection2 = {
            'bbox': [30, 30, 130, 130],
            'confidence': 0.7,
            'label': 'warship'
        }
        
        has_overlap = nms_filter.has_significant_overlap(detection1, detection2)
        assert has_overlap is True

    def test_has_significant_overlap_false(self):
        """Test overlap detection for non-overlapping boxes."""
        nms_filter = NMSFilter(iou_threshold=0.5)
        
        detection1 = {
            'bbox': [0, 0, 100, 100],
            'confidence': 0.8,
            'label': 'warship'
        }
        detection2 = {
            'bbox': [200, 200, 300, 300],
            'confidence': 0.7,
            'label': 'warship'
        }
        
        has_overlap = nms_filter.has_significant_overlap(detection1, detection2)
        assert has_overlap is False

    def test_has_significant_overlap_different_labels(self):
        """Test overlap detection with different labels."""
        nms_filter = NMSFilter(enable_label_specific=True)
        
        detection1 = {
            'bbox': [0, 0, 100, 100],
            'confidence': 0.8,
            'label': 'warship'
        }
        detection2 = {
            'bbox': [30, 30, 130, 130],
            'confidence': 0.7,
            'label': 'ship'
        }
        
        has_overlap = nms_filter.has_significant_overlap(detection1, detection2)
        assert has_overlap is False

    def test_get_label_specificity_score(self):
        """Test label specificity scoring."""
        nms_filter = NMSFilter()
        
        assert nms_filter.get_label_specificity_score('warship') == 4
        assert nms_filter.get_label_specificity_score('battleship') == 5
        assert nms_filter.get_label_specificity_score('ship') == 2
        assert nms_filter.get_label_specificity_score('vessel') == 1
        assert nms_filter.get_label_specificity_score('unknown') == 0

    def test_should_keep_detection_confidence(self):
        """Test detection keeping based on confidence."""
        nms_filter = NMSFilter()
        
        better_detection = {
            'bbox': [0, 0, 100, 100],
            'confidence': 0.9,
            'label': 'warship'
        }
        worse_detection = {
            'bbox': [30, 30, 130, 130],
            'confidence': 0.7,
            'label': 'warship'
        }
        
        # Should keep the higher confidence detection
        should_keep = nms_filter.should_keep_detection(better_detection, worse_detection)
        assert should_keep is True
        
        should_keep = nms_filter.should_keep_detection(worse_detection, better_detection)
        assert should_keep is False

    def test_should_keep_detection_specificity(self):
        """Test detection keeping based on label specificity."""
        nms_filter = NMSFilter()
        
        specific_detection = {
            'bbox': [0, 0, 100, 100],
            'confidence': 0.7,
            'label': 'battleship'
        }
        general_detection = {
            'bbox': [30, 30, 130, 130],
            'confidence': 0.8,
            'label': 'ship'
        }
        
        # Should keep more specific label even with lower confidence
        should_keep = nms_filter.should_keep_detection(specific_detection, general_detection)
        assert should_keep is True

    def test_filter_detections_empty_list(self):
        """Test filtering empty detection list."""
        nms_filter = NMSFilter()
        
        filtered = nms_filter.filter_detections([])
        assert filtered == []

    def test_filter_detections_single_detection(self):
        """Test filtering single detection."""
        nms_filter = NMSFilter()
        
        detections = [{
            'bbox': [0, 0, 100, 100],
            'confidence': 0.8,
            'label': 'warship'
        }]
        
        filtered = nms_filter.filter_detections(detections)
        assert len(filtered) == 1
        assert filtered[0] == detections[0]

    def test_filter_detections_remove_duplicates(self, overlapping_detections):
        """Test filtering overlapping detections."""
        nms_filter = NMSFilter(iou_threshold=0.3)
        
        filtered = nms_filter.filter_detections(overlapping_detections)
        
        # Should keep the higher confidence detection and the non-overlapping one
        assert len(filtered) == 2
        
        # Check that highest confidence warship is kept
        warship_detections = [d for d in filtered if d['label'] == 'warship']
        assert len(warship_detections) == 1
        assert warship_detections[0]['confidence'] == 0.9
        
        # Check that ship detection is kept (different label/position)
        ship_detections = [d for d in filtered if d['label'] == 'ship']
        assert len(ship_detections) == 1

    def test_filter_detections_confidence_threshold(self):
        """Test filtering detections below confidence threshold."""
        nms_filter = NMSFilter(confidence_threshold=0.8)
        
        detections = [
            {
                'bbox': [0, 0, 100, 100],
                'confidence': 0.9,
                'label': 'warship'
            },
            {
                'bbox': [200, 200, 300, 300],
                'confidence': 0.6,  # Below threshold
                'label': 'ship'
            }
        ]
        
        filtered = nms_filter.filter_detections(detections)
        assert len(filtered) == 1
        assert filtered[0]['confidence'] == 0.9

    def test_filter_detections_preserve_metadata(self):
        """Test that additional metadata is preserved during filtering."""
        nms_filter = NMSFilter()
        
        detections = [{
            'bbox': [0, 0, 100, 100],
            'confidence': 0.8,
            'label': 'warship',
            'page': 1,
            'prompt_used': 'detailed warship',
            'processing_time': 0.5
        }]
        
        filtered = nms_filter.filter_detections(detections)
        assert len(filtered) == 1
        assert filtered[0]['page'] == 1
        assert filtered[0]['prompt_used'] == 'detailed warship'
        assert filtered[0]['processing_time'] == 0.5

    def test_get_statistics(self, overlapping_detections):
        """Test getting filtering statistics."""
        nms_filter = NMSFilter()
        
        # First filter to get statistics
        filtered = nms_filter.filter_detections(overlapping_detections)
        stats = nms_filter.get_statistics()
        
        assert 'total_input' in stats
        assert 'total_output' in stats
        assert 'removed_count' in stats
        assert 'removal_rate' in stats
        
        assert stats['total_input'] == len(overlapping_detections)
        assert stats['total_output'] == len(filtered)
        assert stats['removed_count'] == len(overlapping_detections) - len(filtered)

    @pytest.mark.parametrize("iou_threshold,expected_count", [
        (0.1, 2),  # Strict threshold, more filtering
        (0.5, 2),  # Medium threshold
        (0.9, 3),  # Loose threshold, less filtering
    ])
    def test_filter_detections_various_thresholds(self, overlapping_detections, iou_threshold, expected_count):
        """Test filtering with various IoU thresholds."""
        nms_filter = NMSFilter(iou_threshold=iou_threshold)
        
        filtered = nms_filter.filter_detections(overlapping_detections)
        assert len(filtered) == expected_count

    def test_filter_detections_invalid_bbox(self):
        """Test filtering detections with invalid bounding boxes."""
        nms_filter = NMSFilter()
        
        detections = [
            {
                'bbox': [0, 0, 100, 100],
                'confidence': 0.8,
                'label': 'warship'
            },
            {
                'bbox': [100, 100, 50, 50],  # Invalid: x2 < x1, y2 < y1
                'confidence': 0.7,
                'label': 'ship'
            },
            {
                'bbox': [],  # Invalid: empty bbox
                'confidence': 0.6,
                'label': 'vessel'
            }
        ]
        
        filtered = nms_filter.filter_detections(detections)
        
        # Should only keep the valid detection
        assert len(filtered) == 1
        assert filtered[0]['label'] == 'warship'

    def test_reset_statistics(self):
        """Test resetting filtering statistics."""
        nms_filter = NMSFilter()
        
        # Process some detections to generate statistics
        detections = [{
            'bbox': [0, 0, 100, 100],
            'confidence': 0.8,
            'label': 'warship'
        }]
        nms_filter.filter_detections(detections)
        
        # Get stats and verify they exist
        stats = nms_filter.get_statistics()
        assert stats['total_input'] > 0
        
        # Reset and verify stats are cleared
        nms_filter.reset_statistics()
        stats = nms_filter.get_statistics()
        assert stats['total_input'] == 0
        assert stats['total_output'] == 0
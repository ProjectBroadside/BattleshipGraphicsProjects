"""
Non-Maximum Suppression (NMS) and duplicate detection filtering.

This module provides intelligent filtering of overlapping detections
and duplicate removal to improve detection quality.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from ..config.settings import settings
from ..detection.detector import Detection

logger = logging.getLogger(__name__)


class NMSFilter:
    """
    Non-Maximum Suppression filter for removing overlapping detections.
    """
    
    def __init__(self, iou_threshold: Optional[float] = None):
        """
        Initialize the NMS filter.
        
        Args:
            iou_threshold: IoU threshold for overlap detection (uses settings if None)
        """
        self.iou_threshold = iou_threshold or settings.iou_threshold
        logger.info(f"NMSFilter initialized with IoU threshold: {self.iou_threshold}")
    
    def calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU value between 0.0 and 1.0
        """
        if len(bbox1) < 4 or len(bbox2) < 4:
            return 0.0
        
        x1_1, y1_1, x2_1, y2_1 = bbox1[:4]
        x1_2, y1_2, x2_2, y2_2 = bbox2[:4]
        
        # Calculate intersection coordinates
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        # Calculate intersection area
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            intersection = 0.0
        else:
            intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate areas of both boxes
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate union area
        union = area1 + area2 - intersection
        
        # Avoid division by zero
        if union <= 0:
            return 0.0
        
        return intersection / union
    
    def apply_nms(
        self, 
        detections: List[Detection],
        iou_threshold: Optional[float] = None
    ) -> List[Detection]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections.
        
        Args:
            detections: List of Detection objects
            iou_threshold: IoU threshold override
            
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
        
        threshold = iou_threshold or self.iou_threshold
        
        # Sort detections by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        filtered_detections = []
        
        for current_detection in sorted_detections:
            # Check if current detection overlaps significantly with any kept detection
            should_keep = True
            
            for kept_detection in filtered_detections:
                iou = self.calculate_iou(current_detection.bbox, kept_detection.bbox)
                
                if iou > threshold:
                    # Decide which detection to keep based on multiple criteria
                    if self._should_replace_detection(current_detection, kept_detection):
                        # Remove the kept detection and add current one
                        filtered_detections.remove(kept_detection)
                        break
                    else:
                        # Don't keep current detection
                        should_keep = False
                        break
            
            if should_keep:
                filtered_detections.append(current_detection)
        
        logger.debug(f"NMS filtered {len(detections)} -> {len(filtered_detections)} detections")
        return filtered_detections
    
    def _should_replace_detection(
        self, 
        new_detection: Detection, 
        existing_detection: Detection
    ) -> bool:
        """
        Determine if a new detection should replace an existing overlapping one.
        
        Args:
            new_detection: New detection candidate
            existing_detection: Existing detection
            
        Returns:
            True if new detection should replace existing one
        """
        # Higher confidence wins
        if new_detection.confidence > existing_detection.confidence:
            return True
        elif new_detection.confidence < existing_detection.confidence:
            return False
        
        # If confidence is equal, prefer more specific labels
        new_specificity = self._calculate_label_specificity(new_detection.label)
        existing_specificity = self._calculate_label_specificity(existing_detection.label)
        
        if new_specificity > existing_specificity:
            return True
        elif new_specificity < existing_specificity:
            return False
        
        # If still equal, prefer larger detection area
        return new_detection.area > existing_detection.area
    
    def _calculate_label_specificity(self, label: str) -> int:
        """
        Calculate specificity score for a detection label.
        
        Args:
            label: Detection label
            
        Returns:
            Specificity score (higher is more specific)
        """
        label_lower = label.lower()
        
        # Specific warship types get higher scores
        specific_types = {
            'battleship': 10,
            'destroyer': 10, 
            'cruiser': 10,
            'frigate': 10,
            'submarine': 10,
            'corvette': 9,
            'gunboat': 9,
            'torpedo boat': 9,
            'dreadnought': 8,
            'ironclad': 8,
            'monitor': 8
        }
        
        for ship_type, score in specific_types.items():
            if ship_type in label_lower:
                return score
        
        # General terms get medium scores
        general_terms = {
            'warship': 7,
            'naval vessel': 6,
            'military ship': 6,
            'ship': 4,
            'vessel': 3,
            'boat': 2
        }
        
        for term, score in general_terms.items():
            if term in label_lower:
                return score
        
        # Default score for unrecognized labels
        return 1
    
    def remove_duplicates_by_similarity(
        self,
        detections: List[Detection],
        similarity_threshold: float = 0.8
    ) -> List[Detection]:
        """
        Remove duplicate detections based on high similarity.
        
        Args:
            detections: List of Detection objects
            similarity_threshold: Similarity threshold for duplicate detection
            
        Returns:
            List with duplicates removed
        """
        if not detections:
            return []
        
        filtered = []
        
        for detection in detections:
            is_duplicate = False
            
            for existing in filtered:
                # Check for high overlap (potential duplicate)
                iou = self.calculate_iou(detection.bbox, existing.bbox)
                
                if iou > similarity_threshold:
                    # Check if labels are similar (same detection from different prompts)
                    if self._are_labels_similar(detection.label, existing.label):
                        is_duplicate = True
                        # Keep the one with better specificity or higher confidence
                        if self._should_replace_detection(detection, existing):
                            filtered.remove(existing)
                            is_duplicate = False
                        break
            
            if not is_duplicate:
                filtered.append(detection)
        
        logger.debug(f"Duplicate removal: {len(detections)} -> {len(filtered)} detections")
        return filtered
    
    def _are_labels_similar(self, label1: str, label2: str) -> bool:
        """
        Check if two labels are similar (likely the same detection).
        
        Args:
            label1: First label
            label2: Second label
            
        Returns:
            True if labels are similar
        """
        label1_lower = label1.lower()
        label2_lower = label2.lower()
        
        # Exact match
        if label1_lower == label2_lower:
            return True
        
        # Check for common keywords
        common_keywords = [
            'warship', 'battleship', 'destroyer', 'cruiser', 'frigate',
            'submarine', 'ship', 'vessel', 'naval', 'military'
        ]
        
        label1_keywords = set(word for word in common_keywords if word in label1_lower)
        label2_keywords = set(word for word in common_keywords if word in label2_lower)
        
        # If both labels share significant keywords, consider them similar
        if label1_keywords and label2_keywords:
            intersection = label1_keywords.intersection(label2_keywords)
            union = label1_keywords.union(label2_keywords)
            similarity = len(intersection) / len(union) if union else 0
            return similarity > 0.5
        
        return False
    
    def filter_by_area(
        self,
        detections: List[Detection],
        min_area: Optional[int] = None,
        max_area: Optional[int] = None
    ) -> List[Detection]:
        """
        Filter detections by bounding box area.
        
        Args:
            detections: List of Detection objects
            min_area: Minimum area threshold
            max_area: Maximum area threshold
            
        Returns:
            Filtered list of detections
        """
        min_area = min_area or settings.min_detection_area
        max_area = max_area or settings.max_detection_area
        
        filtered = []
        
        for detection in detections:
            area = detection.area
            
            # Check minimum area
            if min_area > 0 and area < min_area:
                logger.debug(f"Filtering detection with area {area} < {min_area}")
                continue
            
            # Check maximum area
            if max_area is not None and area > max_area:
                logger.debug(f"Filtering detection with area {area} > {max_area}")
                continue
            
            filtered.append(detection)
        
        logger.debug(f"Area filtering: {len(detections)} -> {len(filtered)} detections")
        return filtered
    
    def apply_comprehensive_filtering(
        self,
        detections: List[Detection],
        **kwargs
    ) -> List[Detection]:
        """
        Apply comprehensive filtering including NMS, duplicate removal, and area filtering.
        
        Args:
            detections: List of Detection objects
            **kwargs: Override parameters for filtering
            
        Returns:
            Comprehensively filtered detections
        """
        if not detections:
            return []
        
        logger.info(f"Starting comprehensive filtering of {len(detections)} detections")
        
        # Step 1: Filter by area
        filtered = self.filter_by_area(
            detections,
            kwargs.get('min_area'),
            kwargs.get('max_area')
        )
        
        # Step 2: Remove similar duplicates
        filtered = self.remove_duplicates_by_similarity(
            filtered,
            kwargs.get('similarity_threshold', 0.8)
        )
        
        # Step 3: Apply NMS for overlapping detections
        filtered = self.apply_nms(
            filtered,
            kwargs.get('iou_threshold')
        )
        
        logger.info(f"Comprehensive filtering complete: {len(detections)} -> {len(filtered)} detections")
        return filtered


# Global instance for easy access
default_nms_filter = NMSFilter()
"""
Main detection engine for warship illustration extraction using Florence-2.

This module orchestrates the Florence-2 model with comprehensive prompt strategies
to detect and extract warship illustrations from PDF page images.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

from ..config.settings import settings
from ..core.model_manager import ModelManager
from .prompt_strategies import WarshipPromptStrategy, AdvancedPromptStrategy

logger = logging.getLogger(__name__)


class Detection:
    """Represents a single warship detection."""
    
    def __init__(
        self,
        bbox: List[float],
        label: str,
        confidence: float = 1.0,
        prompt: str = "",
        source: str = "florence2"
    ):
        """
        Initialize a detection.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            label: Detection label/description
            confidence: Confidence score (0.0 to 1.0)
            prompt: Prompt that generated this detection
            source: Source model/method
        """
        self.bbox = bbox
        self.label = label
        self.confidence = confidence
        self.prompt = prompt
        self.source = source
        self.area = self._calculate_area()
    
    def _calculate_area(self) -> float:
        """Calculate bounding box area."""
        if len(self.bbox) >= 4:
            x1, y1, x2, y2 = self.bbox[:4]
            return abs((x2 - x1) * (y2 - y1))
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary."""
        return {
            "bbox": self.bbox,
            "label": self.label,
            "confidence": self.confidence,
            "prompt": self.prompt,
            "source": self.source,
            "area": self.area
        }
    
    def __repr__(self) -> str:
        return f"Detection(bbox={self.bbox}, label='{self.label}', confidence={self.confidence})"


class WarshipDetector:
    """
    Main detection engine for warship illustrations using Florence-2.
    """
    
    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        prompt_strategy: Optional[WarshipPromptStrategy] = None,
        use_advanced_strategy: bool = False
    ):
        """
        Initialize the warship detector.
        
        Args:
            model_manager: Model manager instance (creates new if None)
            prompt_strategy: Prompt strategy (creates new if None)
            use_advanced_strategy: Whether to use advanced adaptive prompts
        """
        self.model_manager = model_manager or ModelManager()
        self.prompt_strategy = prompt_strategy or WarshipPromptStrategy()
        
        if use_advanced_strategy:
            self.advanced_strategy = AdvancedPromptStrategy(self.prompt_strategy)
        else:
            self.advanced_strategy = None
        
        self.detection_stats = {
            "total_detections": 0,
            "prompt_performance": {},
            "processing_times": []
        }
        
        logger.info("WarshipDetector initialized")
    
    def detect_warships(
        self,
        image: Image.Image,
        prompts: Optional[List[str]] = None,
        confidence_threshold: Optional[float] = None,
        strategy: str = "optimized"
    ) -> List[Detection]:
        """
        Detect warships in an image using multiple prompts.
        
        Args:
            image: PIL Image to process
            prompts: Custom prompts (uses strategy if None)
            confidence_threshold: Minimum confidence (uses settings if None)
            strategy: Prompt strategy to use
            
        Returns:
            List of Detection objects
        """
        start_time = time.time()
        
        # Get prompts to use
        if prompts is None:
            if self.advanced_strategy and strategy == "adaptive":
                prompts = self.advanced_strategy.get_adaptive_prompts()
            elif strategy == "optimized":
                prompts = self.prompt_strategy.get_optimized_prompt_set()
            else:
                prompts = self.prompt_strategy.get_prompts_by_strategy(strategy)
        
        confidence_threshold = confidence_threshold or settings.confidence_threshold
        
        logger.info(f"Starting detection with {len(prompts)} prompts")
        
        # Ensure model is loaded
        if not self.model_manager.is_loaded():
            self.model_manager.load_model()
        
        model, processor = self.model_manager.model, self.model_manager.processor
        
        all_detections = []
        
        for prompt in prompts:
            try:
                detections = self._run_single_prompt_detection(
                    image, prompt, model, processor, confidence_threshold
                )
                all_detections.extend(detections)
                
                # Update performance tracking
                self._update_prompt_performance(prompt, len(detections) > 0)
                
            except Exception as e:
                logger.warning(f"Failed to process prompt '{prompt}': {str(e)}")
                continue
        
        # Update statistics
        processing_time = time.time() - start_time
        self.detection_stats["processing_times"].append(processing_time)
        self.detection_stats["total_detections"] += len(all_detections)
        
        logger.info(f"Completed detection: {len(all_detections)} detections in {processing_time:.2f}s")
        
        return all_detections
    
    def _run_single_prompt_detection(
        self,
        image: Image.Image,
        prompt: str,
        model: Any,
        processor: Any,
        confidence_threshold: float
    ) -> List[Detection]:
        """Run detection with a single prompt."""
        try:
            # Prepare inputs
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.model_manager.device, self.model_manager.torch_dtype)
            
            # Generate predictions
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False
                )
            
            # Decode results
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            
            # Parse the response
            task_type = prompt.split('>')[0] + '>'
            parsed_answer = processor.post_process_generation(
                generated_text,
                task=task_type,
                image_size=(image.width, image.height)
            )
            
            # Extract detections
            detections = self._parse_detections(parsed_answer, prompt, confidence_threshold)
            
            logger.debug(f"Prompt '{prompt}' found {len(detections)} detections")
            return detections
            
        except Exception as e:
            logger.error(f"Error in single prompt detection '{prompt}': {str(e)}")
            return []
    
    def _parse_detections(
        self,
        parsed_answer: Dict[str, Any],
        prompt: str,
        confidence_threshold: float
    ) -> List[Detection]:
        """Parse Florence-2 response into Detection objects."""
        detections = []
        
        if '<OD>' in prompt and 'bboxes' in parsed_answer:
            # Object detection results
            bboxes = parsed_answer.get('bboxes', [])
            labels = parsed_answer.get('labels', [])
            
            for bbox, label in zip(bboxes, labels):
                detection = Detection(
                    bbox=bbox,
                    label=label,
                    confidence=1.0,  # Florence-2 doesn't provide confidence scores
                    prompt=prompt,
                    source="florence2"
                )
                detections.append(detection)
        
        elif '<DENSE_REGION_CAPTION>' in prompt and 'bboxes' in parsed_answer:
            # Dense region caption results - filter for ship-related content
            bboxes = parsed_answer.get('bboxes', [])
            labels = parsed_answer.get('labels', [])
            
            # Filter using prompt strategy
            ship_related_labels = self.prompt_strategy.filter_ship_related_captions(labels)
            
            for bbox, label in zip(bboxes, labels):
                if label in ship_related_labels:
                    detection = Detection(
                        bbox=bbox,
                        label=label,
                        confidence=0.8,  # Lower confidence for caption-based detection
                        prompt=prompt,
                        source="florence2_caption"
                    )
                    detections.append(detection)
        
        # Filter by area if specified in settings
        if settings.min_detection_area > 0:
            detections = [d for d in detections if d.area >= settings.min_detection_area]
        
        if settings.max_detection_area is not None:
            detections = [d for d in detections if d.area <= settings.max_detection_area]
        
        return detections
    
    def _update_prompt_performance(self, prompt: str, had_results: bool) -> None:
        """Update performance tracking for a prompt."""
        if prompt not in self.detection_stats["prompt_performance"]:
            self.detection_stats["prompt_performance"][prompt] = {"hits": 0, "total": 0}
        
        self.detection_stats["prompt_performance"][prompt]["total"] += 1
        if had_results:
            self.detection_stats["prompt_performance"][prompt]["hits"] += 1
        
        # Update advanced strategy if available
        if self.advanced_strategy:
            success_rate = (
                self.detection_stats["prompt_performance"][prompt]["hits"] /
                self.detection_stats["prompt_performance"][prompt]["total"]
            )
            self.advanced_strategy.update_prompt_performance(prompt, success_rate)
    
    def batch_detect(
        self,
        images: List[Image.Image],
        **kwargs
    ) -> List[List[Detection]]:
        """
        Detect warships in multiple images.
        
        Args:
            images: List of PIL Images
            **kwargs: Arguments passed to detect_warships
            
        Returns:
            List of detection lists (one per image)
        """
        results = []
        
        for i, image in enumerate(images):
            logger.debug(f"Processing image {i+1}/{len(images)}")
            try:
                detections = self.detect_warships(image, **kwargs)
                results.append(detections)
            except Exception as e:
                logger.error(f"Failed to process image {i+1}: {str(e)}")
                results.append([])
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detection performance statistics."""
        stats = self.detection_stats.copy()
        
        # Calculate average processing time
        if stats["processing_times"]:
            stats["avg_processing_time"] = sum(stats["processing_times"]) / len(stats["processing_times"])
        else:
            stats["avg_processing_time"] = 0.0
        
        # Calculate prompt success rates
        prompt_success_rates = {}
        for prompt, data in stats["prompt_performance"].items():
            if data["total"] > 0:
                prompt_success_rates[prompt] = data["hits"] / data["total"]
            else:
                prompt_success_rates[prompt] = 0.0
        
        stats["prompt_success_rates"] = prompt_success_rates
        
        return stats
    
    def reset_performance_stats(self) -> None:
        """Reset performance tracking statistics."""
        self.detection_stats = {
            "total_detections": 0,
            "prompt_performance": {},
            "processing_times": []
        }
        logger.info("Performance statistics reset")
    
    def optimize_memory(self) -> None:
        """Optimize memory usage."""
        if self.model_manager:
            self.model_manager.optimize_memory()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.optimize_memory()
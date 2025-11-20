"""
Image processing for warship detection extraction and enhancement.

This module handles cropping detected regions, image enhancement,
and format conversion for extracted warship illustrations.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from ..config.settings import settings
from ..detection.detector import Detection

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Handles image processing for detected warship regions.
    """
    
    def __init__(self, enhance_images: Optional[bool] = None, padding: Optional[int] = None):
        """
        Initialize the image processor.
        
        Args:
            enhance_images: Whether to apply image enhancement (uses settings if None)
            padding: Padding around cropped regions (uses settings if None)
        """
        self.enhance_images = enhance_images if enhance_images is not None else settings.enhance_images
        self.padding = padding if padding is not None else settings.padding_pixels
        
        logger.info(f"ImageProcessor initialized - enhance: {self.enhance_images}, padding: {self.padding}")
    
    def crop_detection(
        self,
        image: Image.Image,
        detection: Detection,
        padding: Optional[int] = None
    ) -> Image.Image:
        """
        Crop a detection region from an image with optional padding.
        
        Args:
            image: Source PIL Image
            detection: Detection object with bounding box
            padding: Padding override (pixels)
            
        Returns:
            Cropped PIL Image
        """
        padding = padding if padding is not None else self.padding
        
        # Get bounding box coordinates
        if len(detection.bbox) < 4:
            raise ValueError(f"Invalid bounding box: {detection.bbox}")
        
        x1, y1, x2, y2 = detection.bbox[:4]
        
        # Apply padding with bounds checking
        x1_padded = max(0, int(x1 - padding))
        y1_padded = max(0, int(y1 - padding))
        x2_padded = min(image.width, int(x2 + padding))
        y2_padded = min(image.height, int(y2 + padding))
        
        # Crop the image
        cropped = image.crop((x1_padded, y1_padded, x2_padded, y2_padded))
        
        logger.debug(f"Cropped region: {(x1_padded, y1_padded, x2_padded, y2_padded)} from {image.size}")
        
        return cropped
    
    def enhance_image(self, image: Image.Image) -> Image.Image:
        """
        Apply image enhancement to improve quality.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Enhanced PIL Image
        """
        if not self.enhance_images:
            return image
        
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply contrast enhancement
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)  # Increase contrast by 20%
            
            # Apply sharpness enhancement
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)  # Increase sharpness by 10%
            
            # Apply slight denoising using PIL filter
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            logger.debug("Applied image enhancement")
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {str(e)}, returning original")
        
        return image
    
    def apply_advanced_enhancement(self, image: Image.Image) -> Image.Image:
        """
        Apply advanced image enhancement using OpenCV.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Enhanced PIL Image
        """
        try:
            # Convert PIL to OpenCV format
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_cv = img_array
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            if len(img_cv.shape) == 3:
                lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                lab[:,:,0] = clahe.apply(lab[:,:,0])
                img_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                img_cv = clahe.apply(img_cv)
            
            # Apply denoising
            if len(img_cv.shape) == 3:
                img_cv = cv2.fastNlMeansDenoisingColored(img_cv, None, 10, 10, 7, 21)
            else:
                img_cv = cv2.fastNlMeansDenoising(img_cv, None, 10, 7, 21)
            
            # Convert back to PIL
            if len(img_cv.shape) == 3:
                img_array = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            else:
                img_array = img_cv
            
            enhanced_image = Image.fromarray(img_array)
            
            logger.debug("Applied advanced OpenCV enhancement")
            return enhanced_image
            
        except Exception as e:
            logger.warning(f"Advanced enhancement failed: {str(e)}, falling back to basic enhancement")
            return self.enhance_image(image)
    
    def resize_image(
        self,
        image: Image.Image,
        target_size: Optional[Tuple[int, int]] = None,
        max_dimension: Optional[int] = None,
        maintain_aspect_ratio: bool = True
    ) -> Image.Image:
        """
        Resize an image with various options.
        
        Args:
            image: Input PIL Image
            target_size: Exact target size (width, height)
            max_dimension: Maximum dimension (resizes to fit)
            maintain_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            Resized PIL Image
        """
        if target_size:
            if maintain_aspect_ratio:
                image.thumbnail(target_size, Image.Resampling.LANCZOS)
            else:
                image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        elif max_dimension:
            if max(image.size) > max_dimension:
                ratio = max_dimension / max(image.size)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    
    def process_detection_batch(
        self,
        image: Image.Image,
        detections: List[Detection],
        output_dir: Optional[Union[str, Path]] = None,
        filename_prefix: str = "detection",
        apply_enhancement: Optional[bool] = None,
        save_format: str = "PNG"
    ) -> List[Tuple[Path, Detection]]:
        """
        Process multiple detections and save cropped regions.
        
        Args:
            image: Source PIL Image
            detections: List of Detection objects
            output_dir: Output directory (uses settings if None)
            filename_prefix: Prefix for output filenames
            apply_enhancement: Whether to enhance images (uses instance setting if None)
            save_format: Output format (PNG, JPEG, etc.)
            
        Returns:
            List of (output_path, detection) tuples
        """
        if not detections:
            return []
        
        if output_dir is None:
            output_dir = settings.get_output_path("extracted_images")
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        apply_enhancement = apply_enhancement if apply_enhancement is not None else self.enhance_images
        
        results = []
        
        for i, detection in enumerate(detections):
            try:
                # Crop the detection
                cropped = self.crop_detection(image, detection)
                
                # Apply enhancement if requested
                if apply_enhancement:
                    cropped = self.enhance_image(cropped)
                
                # Generate filename
                label_clean = self._clean_label_for_filename(detection.label)
                filename = f"{filename_prefix}_{i+1:03d}_{label_clean}.{save_format.lower()}"
                output_path = output_dir / filename
                
                # Save with appropriate settings
                save_kwargs = self._get_save_kwargs(save_format)
                cropped.save(output_path, format=save_format, **save_kwargs)
                
                results.append((output_path, detection))
                
                logger.debug(f"Saved detection {i+1} to {output_path}")
                
            except Exception as e:
                logger.error(f"Failed to process detection {i+1}: {str(e)}")
                continue
        
        logger.info(f"Processed {len(results)}/{len(detections)} detections successfully")
        return results
    
    def _clean_label_for_filename(self, label: str) -> str:
        """Clean a detection label for use in filenames."""
        # Remove or replace invalid filename characters
        invalid_chars = '<>:"/\\|?*'
        cleaned = label
        
        for char in invalid_chars:
            cleaned = cleaned.replace(char, '_')
        
        # Replace spaces with underscores and limit length
        cleaned = cleaned.replace(' ', '_').replace('__', '_').strip('_')
        
        # Limit length
        if len(cleaned) > 50:
            cleaned = cleaned[:50].strip('_')
        
        # Ensure we have at least something
        if not cleaned:
            cleaned = "unknown"
        
        return cleaned
    
    def _get_save_kwargs(self, format: str) -> dict:
        """Get appropriate save kwargs for different image formats."""
        format_upper = format.upper()
        
        if format_upper == "PNG":
            return {
                "optimize": True,
                "compress_level": 6
            }
        elif format_upper == "JPEG":
            return {
                "quality": 95,
                "optimize": True
            }
        elif format_upper == "WEBP":
            return {
                "quality": 95,
                "method": 6
            }
        else:
            return {}
    
    def create_detection_summary_image(
        self,
        image: Image.Image,
        detections: List[Detection],
        output_path: Optional[Union[str, Path]] = None,
        draw_labels: bool = True,
        draw_confidence: bool = True
    ) -> Optional[Path]:
        """
        Create a summary image with all detections annotated.
        
        Args:
            image: Source PIL Image
            detections: List of Detection objects
            output_path: Output path (generates if None)
            draw_labels: Whether to draw detection labels
            draw_confidence: Whether to draw confidence scores
            
        Returns:
            Path to saved summary image or None if failed
        """
        try:
            # Convert to OpenCV format for drawing
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            
            # Draw bounding boxes and labels
            for i, detection in enumerate(detections):
                x1, y1, x2, y2 = [int(coord) for coord in detection.bbox[:4]]
                
                # Choose color based on confidence
                if detection.confidence >= 0.8:
                    color = (0, 255, 0)  # Green for high confidence
                elif detection.confidence >= 0.5:
                    color = (0, 255, 255)  # Yellow for medium confidence
                else:
                    color = (0, 0, 255)  # Red for low confidence
                
                # Draw bounding box
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
                
                # Prepare label text
                label_parts = []
                if draw_labels:
                    label_parts.append(detection.label[:30])  # Truncate long labels
                if draw_confidence:
                    label_parts.append(f"{detection.confidence:.2f}")
                
                if label_parts:
                    label_text = " | ".join(label_parts)
                    
                    # Draw label background
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    cv2.rectangle(
                        img_cv,
                        (x1, y1 - text_height - 5),
                        (x1 + text_width, y1),
                        color,
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        img_cv,
                        label_text,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )
            
            # Convert back to PIL and save
            img_array = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            annotated_image = Image.fromarray(img_array)
            
            if output_path is None:
                output_path = settings.get_output_path("visualizations") / "detection_summary.png"
            else:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            annotated_image.save(output_path, "PNG", optimize=True)
            
            logger.info(f"Saved detection summary to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create detection summary: {str(e)}")
            return None
    
    def validate_image_quality(self, image: Image.Image) -> dict:
        """
        Validate and assess image quality metrics.
        
        Args:
            image: PIL Image to validate
            
        Returns:
            Dictionary with quality metrics
        """
        metrics = {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format,
            "size_mb": 0.0,
            "aspect_ratio": 0.0,
            "is_grayscale": False,
            "estimated_quality": "unknown"
        }
        
        try:
            # Calculate basic metrics
            metrics["aspect_ratio"] = image.width / image.height if image.height > 0 else 0
            metrics["is_grayscale"] = image.mode in ['L', '1']
            
            # Estimate file size
            import io
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            metrics["size_mb"] = buffer.tell() / (1024 * 1024)
            
            # Basic quality assessment
            min_dimension = min(image.width, image.height)
            if min_dimension >= 800:
                metrics["estimated_quality"] = "high"
            elif min_dimension >= 400:
                metrics["estimated_quality"] = "medium"
            else:
                metrics["estimated_quality"] = "low"
            
        except Exception as e:
            logger.warning(f"Quality validation failed: {str(e)}")
        
        return metrics


# Global instance for easy access
default_image_processor = ImageProcessor()